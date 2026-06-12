"""
Phase 2: Correlation Analysis, Scaling, Optimal Sampling Selection, Feature Selection
=====================================================================================

Uses **2024_data.parquet only**. 2025_data is isolated until Phase 5.

Core workflow (redesigned after Phase 1 dome-masking):
  1) Feature group definition (X_base 2 features / X_advanced initial ~70 features)
  2) Categorical encoding (stand/p_throws → binary, pitch_type/alignment → one-hot)
  3) NaN handling — numeric: median imputation (full 2024 — averaging effect across CV)
  4) Robust Scaler applied (numeric only)
  5) Multicollinearity removal |r| > 0.95 (Pearson)
     - Domain-priority drop: derived features (effective_speed, api_break_*,
       wx_wind_gusts_10m, max_wall_height) → variance fallback
  6) Sampling comparison (None / RUS / SMOTE) — XGBoost default, 5-fold CV
     OOF predict_proba → Brier (primary) + LogLoss + F1 + AUC, fold mean±SD
  7) Feature Selection — on best sampling:
     RF RandomizedSearchCV → feature_importances_ (split impurity)
     + MI (30K stratified) → drop if both rank in bottom 30% simultaneously
     (Permutation Importance excluded due to macOS joblib memmap disk limitations)
  8) Save phase2_features.json, scaler, X/y parquet
  9) Auto-generate phase2_report.md

Run:
    /opt/miniconda3/envs/mlb-xba/bin/python pipeline/step2_phase2_correlation_sampling.py
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
REPORT_PATH = PIPELINE_DIR / "phase2_report.md"

INPUT_PARQUET = OUTPUT_DIR / "2024_data.parquet"
X_FULL_PARQUET = OUTPUT_DIR / "phase2_X_full.parquet"
Y_FULL_PARQUET = OUTPUT_DIR / "phase2_y_full.parquet"
FEATURES_JSON = OUTPUT_DIR / "phase2_features.json"
SCALER_PATH = OUTPUT_DIR / "phase2_scaler.joblib"
FS_RANK_CSV = OUTPUT_DIR / "phase2_fs_ranking.csv"

# -----------------------------------------------------------------------------
# Decision constants (user-confirmed — all branches re-verified after Phase 1 re-run)
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS = 5
CORR_THRESHOLD = 0.95
FS_DROP_RANK_THRESHOLD = 0.70  # Drop if both metrics rank in bottom 30% (rank percentile > 0.7)
MI_SUBSAMPLE_SIZE = 30_000

# RF — RandomizedSearchCV for feature importance computation (restored from previous step2 approach)
RF_SEARCH_SPACE = {
    "n_estimators": [100, 200, 500],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 4, 6],
    "criterion": ["gini", "entropy"],
}
RF_SEARCH_N_ITER = 20
RF_SEARCH_CV = 3

# Thermal management
COOLDOWN_SEC = 20
N_JOBS_HEAVY = 2

# -----------------------------------------------------------------------------
# Feature group definitions (Phase 1 data — includes dome-masked weather)
# -----------------------------------------------------------------------------
X_BASE = ["launch_speed", "launch_angle"]

BAT_TRACKING = [
    "bat_speed",
    "swing_length",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
]
BAT_TRACKING_FLAGS = [c + "_is_missing" for c in BAT_TRACKING]

PITCH_NUMERIC = [
    "release_speed",
    "release_pos_x",
    "release_pos_z",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "release_spin_rate",
    "release_extension",
    "spin_axis",
    "effective_speed",
    "api_break_z_with_gravity",
    "api_break_x_arm",
    "api_break_x_batter_in",
    "arm_angle",
]

PA_NUMERIC = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "age_pit",
    "age_bat",
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
]

PARK_STATIC = [
    "left_field",
    "center_field",
    "right_field",
    "min_wall_height",
    "max_wall_height",
    "hr_park_effects",
    "extra_distance",
    "elevation",
    "roof",
    "daytime",
]

WEATHER_DYNAMIC = [
    "wx_temperature_2m",
    "wx_relative_humidity_2m",
    "wx_surface_pressure",
    "wx_wind_speed_10m",
    "wx_wind_direction_10m",
    "wx_precipitation",
    "wx_cloud_cover",
    "wx_wind_gusts_10m",
]

NUMERIC_FEATURES = (
    X_BASE
    + BAT_TRACKING
    + BAT_TRACKING_FLAGS
    + PITCH_NUMERIC
    + PA_NUMERIC
    + PARK_STATIC
    + WEATHER_DYNAMIC
)

BINARY_LR = ["stand", "p_throws"]
CATEGORICAL_OHE = ["pitch_type", "if_fielding_alignment", "of_fielding_alignment"]

# Domain-derived features (drop candidates first when |r| > 0.95)
DERIVED_EXACT = {"effective_speed", "wx_wind_gusts_10m", "max_wall_height"}
DERIVED_PREFIX = ("api_break_",)


def is_derived(col: str) -> bool:
    if col in DERIVED_EXACT:
        return True
    for p in DERIVED_PREFIX:
        if col.startswith(p):
            return True
    return False


# -----------------------------------------------------------------------------
# Feature matrix construction
# -----------------------------------------------------------------------------
def build_raw_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    df["stand_R"] = (df["stand"] == "R").astype("int8")
    df["p_throws_R"] = (df["p_throws"] == "R").astype("int8")

    one_hot_parts = []
    for col in CATEGORICAL_OHE:
        df[col] = df[col].fillna("UNK").astype("string")
        dummies = pd.get_dummies(df[col], prefix=col).astype("int8")
        one_hot_parts.append(dummies)
    one_hot = pd.concat(one_hot_parts, axis=1)

    numeric = df[NUMERIC_FEATURES].copy()
    for col in BAT_TRACKING_FLAGS:
        numeric[col] = numeric[col].fillna(0).astype("int8")

    binary = df[["stand_R", "p_throws_R"]].copy()

    X = pd.concat([numeric, binary, one_hot], axis=1)
    y = df["is_hit"].astype("int8")
    return X, y


def impute_numeric_with_median(
    X: pd.DataFrame, numeric_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    medians: dict[str, float] = {}
    X = X.copy()
    for col in numeric_cols:
        if col not in X.columns:
            continue
        if X[col].isna().any():
            med = float(X[col].median())
            medians[col] = med
            X[col] = X[col].fillna(med)
    return X, medians


# -----------------------------------------------------------------------------
# Multicollinearity removal — domain-priority rules
#   1) Both features in X_BASE: preserve both
#   2) Only one is X_BASE: drop the other
#   3) One is derived, the other is source: drop derived
#   4) Both derived or both source: variance fallback (drop lower-variance one)
#   5) Tied variance: drop alphabetically later (deterministic)
# -----------------------------------------------------------------------------
def correlation_drop_domain_priority(
    X_scaled: pd.DataFrame, threshold: float
) -> tuple[list[str], list[dict]]:
    corr = X_scaled.corr(method="pearson").abs()
    variances = X_scaled.var()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drop_cols: set[str] = set()
    pair_log: list[dict] = []

    for col in upper.columns:
        for row in upper.index:
            val = upper.loc[row, col]
            if pd.notna(val) and val > threshold:
                a, b = row, col
                rule = ""
                if a in X_BASE and b in X_BASE:
                    drop_target = None
                    rule = "X_BASE 둘 다 보존"
                elif a in X_BASE:
                    drop_target = b
                    rule = "X_BASE 보호"
                elif b in X_BASE:
                    drop_target = a
                    rule = "X_BASE 보호"
                elif is_derived(a) and not is_derived(b):
                    drop_target = a
                    rule = "derived drop"
                elif is_derived(b) and not is_derived(a):
                    drop_target = b
                    rule = "derived drop"
                else:
                    va, vb = float(variances[a]), float(variances[b])
                    if va < vb:
                        drop_target = a
                    elif vb < va:
                        drop_target = b
                    else:
                        drop_target = max(a, b)
                    rule = "variance fallback"

                if drop_target is not None:
                    drop_cols.add(drop_target)
                pair_log.append(
                    {
                        "a": a,
                        "b": b,
                        "abs_r": float(val),
                        "var_a": float(variances[a]),
                        "var_b": float(variances[b]),
                        "dropped": drop_target,
                        "rule": rule,
                    }
                )

    drop_cols = drop_cols - set(X_BASE)
    return sorted(drop_cols), pair_log


# -----------------------------------------------------------------------------
# Default model — XGBoost for sampling comparison
# -----------------------------------------------------------------------------
def xgb_default() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
    )


def cooldown(reason: str = "", sec: int = COOLDOWN_SEC) -> None:
    print(f"  [cooldown {sec}s] " + (reason or "발열 관리"), flush=True)
    time.sleep(sec)


# -----------------------------------------------------------------------------
# Sampling comparison (5-fold CV OOF Brier)
# -----------------------------------------------------------------------------
def cv_oof_for_sampling(
    X: pd.DataFrame, y: pd.Series, sampling_name: str, sampler
) -> dict:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(y), dtype=np.float64)
    fold_briers: list[float] = []
    fold_loglosses: list[float] = []
    train_class_counts: list[tuple[int, int]] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr_fold = X.iloc[tr_idx]
        y_tr_fold = y.iloc[tr_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        if sampler is not None:
            X_tr_s, y_tr_s = sampler.fit_resample(X_tr_fold, y_tr_fold)
            X_tr_s = pd.DataFrame(X_tr_s, columns=X.columns)
            y_tr_s = pd.Series(y_tr_s, name="is_hit")
        else:
            X_tr_s, y_tr_s = X_tr_fold, y_tr_fold

        n0 = int((y_tr_s == 0).sum())
        n1 = int((y_tr_s == 1).sum())
        train_class_counts.append((n0, n1))

        model = xgb_default()
        model.fit(X_tr_s, y_tr_s)
        proba = model.predict_proba(X_val_fold)[:, 1]
        oof_proba[val_idx] = proba

        fb = float(brier_score_loss(y_val_fold, proba))
        fl = float(log_loss(y_val_fold, np.clip(proba, 1e-7, 1 - 1e-7)))
        fold_briers.append(fb)
        fold_loglosses.append(fl)
        print(
            f"  [{sampling_name}] fold {fold}/{CV_FOLDS} "
            f"train(0/1)={n0:,}/{n1:,}  Brier={fb:.5f}  LogLoss={fl:.5f}",
            flush=True,
        )

    pred = (oof_proba >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    return {
        "oof_brier": float(brier_score_loss(y, oof_proba)),
        "oof_logloss": float(log_loss(y, np.clip(oof_proba, 1e-7, 1 - 1e-7))),
        "oof_f1": float(f1_score(y, pred)),
        "oof_roc_auc": float(roc_auc_score(y, oof_proba)),
        "oof_precision": float(precision_score(y, pred)),
        "oof_recall": float(recall_score(y, pred)),
        "oof_accuracy": float(accuracy_score(y, pred)),
        "fold_brier_mean": float(np.mean(fold_briers)),
        "fold_brier_std": float(np.std(fold_briers)),
        "fold_logloss_mean": float(np.mean(fold_loglosses)),
        "fold_logloss_std": float(np.std(fold_loglosses)),
        "cm_tn": int(cm[0, 0]),
        "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]),
        "cm_tp": int(cm[1, 1]),
        "train_class_counts_mean_0": int(np.mean([c[0] for c in train_class_counts])),
        "train_class_counts_mean_1": int(np.mean([c[1] for c in train_class_counts])),
    }


def run_sampling_comparison(X: pd.DataFrame, y: pd.Series) -> dict:
    samplers = {
        "None": None,
        "Under": RandomUnderSampler(random_state=RANDOM_STATE),
        "SMOTE": SMOTE(random_state=RANDOM_STATE, k_neighbors=5),
    }
    results: dict = {}
    keys = list(samplers.keys())
    for i, name in enumerate(keys):
        print(f"\n  [sampling] '{name}' 5-fold CV 시작 ...", flush=True)
        results[name] = cv_oof_for_sampling(X, y, name, samplers[name])
        r = results[name]
        print(
            f"  [{name}] OOF Brier={r['oof_brier']:.5f}  LogLoss={r['oof_logloss']:.5f}  "
            f"F1={r['oof_f1']:.4f}  AUC={r['oof_roc_auc']:.4f}  "
            f"(fold Brier {r['fold_brier_mean']:.5f}±{r['fold_brier_std']:.5f})",
            flush=True,
        )
        if i < len(keys) - 1:
            cooldown(f"'{name}' 샘플링 완료, 다음 샘플링 전 대기")

    return results


# -----------------------------------------------------------------------------
# Feature Selection — RF RandomizedSearchCV (feature_importances_) + MI
# -----------------------------------------------------------------------------
def compute_mi(X: pd.DataFrame, y: pd.Series, subsample_size: int) -> pd.Series:
    n_sub = min(subsample_size, len(X))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_sub, random_state=RANDOM_STATE)
    sub_idx, _ = next(sss.split(X, y))
    X_sub = X.iloc[sub_idx]
    y_sub = y.iloc[sub_idx]
    print(
        f"    MI 계산 (stratified n={n_sub:,d}, hit_rate={y_sub.mean():.4f}) ...",
        flush=True,
    )
    mi_arr = mutual_info_classif(
        X_sub.values,
        y_sub.values,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS_HEAVY,
    )
    return pd.Series(mi_arr, index=X.columns).sort_values(ascending=False)


def feature_selection(
    X: pd.DataFrame, y: pd.Series, drop_rank_threshold: float
) -> tuple[dict, list[str], pd.DataFrame, dict, float]:
    """
    Two-criterion approach (restored from previous step2 method):
      (1) RF RandomizedSearchCV → best_estimator_.feature_importances_ (split impurity)
      (2) MI on stratified 30K subsample
    → Drop features where both rank in bottom 30% (percentile rank > drop_rank_threshold)
    Permutation Importance excluded due to macOS joblib memmap disk limitations.
    """
    print(
        f"  RF RandomizedSearchCV 시작 "
        f"(n_iter={RF_SEARCH_N_ITER}, cv={RF_SEARCH_CV}, n_jobs={N_JOBS_HEAVY}) ...",
        flush=True,
    )
    print(f"  search space: {RF_SEARCH_SPACE}", flush=True)
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=RF_SEARCH_SPACE,
        n_iter=RF_SEARCH_N_ITER,
        cv=RF_SEARCH_CV,
        scoring="neg_brier_score",
        n_jobs=N_JOBS_HEAVY,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
    )
    search.fit(X, y)
    best_rf = search.best_estimator_
    rf_imp = pd.Series(
        best_rf.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print(f"  RF best params : {search.best_params_}", flush=True)
    print(
        f"  RF best CV neg_brier_score : {search.best_score_:.5f} "
        f"(Brier = {-search.best_score_:.5f})",
        flush=True,
    )

    cooldown("RF RandomizedSearchCV 완료, MI 전 대기")

    print("  Mutual Information ...", flush=True)
    mi = compute_mi(X, y, MI_SUBSAMPLE_SIZE)

    rank_rf = rf_imp.rank(ascending=False, pct=True)
    rank_mi = mi.rank(ascending=False, pct=True)

    rank_df = pd.DataFrame(
        {
            "rf_importance": rf_imp,
            "mi": mi,
            "rank_rf": rank_rf,
            "rank_mi": rank_mi,
        }
    )
    drop_mask = (rank_rf > drop_rank_threshold) & (rank_mi > drop_rank_threshold)
    drop_cols = [c for c in drop_mask[drop_mask].index.tolist() if c not in X_BASE]
    rank_df["dropped"] = rank_df.index.isin(drop_cols)
    rank_df = rank_df.sort_values("rank_mi", ascending=True)

    artifacts = {
        "rf_importance": rf_imp,
        "mi": mi,
    }
    print(
        f"  → 2개 기준(RF & MI) 모두 하위 {(1-drop_rank_threshold)*100:.0f}% 동시 진입 변수: {len(drop_cols)}개 제거",
        flush=True,
    )
    return (
        artifacts,
        drop_cols,
        rank_df,
        dict(search.best_params_),
        float(search.best_score_),
    )


# -----------------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------------
def write_report(meta: dict, hi_corr_pairs: list[dict], fs_artifacts: dict):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    L: list[str] = []
    L.append("# Phase 2 Report — 상관관계 분석, 스케일링, 최적 샘플링, Feature Selection")
    L.append("")
    L.append(f"_생성: {now}_  ")
    L.append("_실행 스크립트: `pipeline/step2_phase2_correlation_sampling.py`_")
    L.append("")
    L.append("> 본 단계는 **2024_data.parquet 만** 사용한다. 2025_data는 Phase 5까지 격리되어 어떤 통계도 누설되지 않는다.")
    L.append("")

    # Decisions
    L.append("## 1. 결정 사항 (사용자 컨펌, Phase 1 dome-masking 이후 분기 전수 재확인)")
    L.append("")
    L.append("| # | 결정 항목 | 채택안 | 사유 |")
    L.append("|---|---|---|---|")
    L.append("| 1 | X_base 정의 | `launch_speed`, `launch_angle` (2종) | MLB 공식 xBA와 동일 입력으로 통제군 명료화. |")
    L.append("| 2 | 카테고리형 인코딩 | One-Hot (stand/p_throws=L/R binary, pitch_type·alignment=dummy) | 카테고리 적어 차원 폭발 없음. |")
    L.append("| 3 | Train/Test 분할 | **2025는 검증 전용 격리** / 2024 안에서 **StratifiedKFold 5-fold CV (train+val 통합)** | data leakage 0. CV가 hold-out test 대체. |")
    L.append("| 4 | X_advanced 초기 변수 풀 | 광범위 정의(투구·PA·구장·기상 모두) | FS·공선성 단계에서 정리하도록 설계. |")
    L.append(f"| 5 | 다중공선성 임계값 | **|r| > {CORR_THRESHOLD}** (Pearson) | 거의 동일한 변수만 제거하는 보수적 기준. |")
    L.append("| 6 | 다중공선성 drop 규칙 | X_BASE 보존 → **derived 변수 우선 drop** → variance fallback | 도메인 의미 우선(소스 > 파생). derived: `effective_speed`, `api_break_*`, `wx_wind_gusts_10m`, `max_wall_height`. |")
    L.append("| 7 | 처리 순서 | impute → scale → corr_drop → sampling → FS | scale 후 corr 안정성 + variance fallback 적용 위해. |")
    L.append("| 8 | Robust Scaler | numeric만(unique>2) | 이상치 강건. LogReg(Phase 3 통제군) 호환. |")
    L.append("| 9 | NaN imputation (numeric) | **2024 전체 median fill** | CV 격자에 같은 imputation. *_is_missing 플래그는 신호 보존. |")
    L.append("| 10 | NaN 카테고리 처리 | NaN → 'UNK' 더미 | 결측 패턴을 카테고리로 보존. |")
    L.append("| 11 | 상호작용 FE | **Raw 그대로** (트리에 맡김) | dome-masking이 dome-day 시그널을 충분히 전달. 명시적 product feature 추가 없음. |")
    L.append("| 12 | 샘플링 비교 | None / RandomUnderSampler / SMOTE 3종 **모두** | 결과 비교 후 선택. 사전 가정 배제. |")
    L.append(f"| 13 | 샘플링 평가 모델 | XGBoost **default** + StratifiedKFold {CV_FOLDS}-fold CV | 모델 가설 중립. baseline default 공정 비교. |")
    L.append("| 14 | 샘플링 선정 메트릭 | **Brier Score (OOF predict_proba)** 최솟값 | ca-xBA는 확률 값의 정상도가 핵심. F1·AUC는 보조 표시. |")
    L.append("| 15 | EDA 범위 | Pearson 상관, \\|r\\|>0.95 중복 제거, RF importance, MI | 다관점 신호 종합. |")
    L.append("| 16 | FS 계산용 모델 | RF (RandomizedSearchCV 튜닝된 best_estimator) — split impurity 기반 `feature_importances_` | 이전 step2 검증된 방식 복원. 3-model Permutation Importance 는 macOS joblib memmap 디스크 한계로 미채택. |")
    L.append(f"| 17 | FS 제거 규칙 | **2개 기준(RF importance & MI) 모두 하위 {(1-FS_DROP_RANK_THRESHOLD)*100:.0f}% 동시 진입** 시 drop (X_BASE 제외) | 보수적 — 한 지표라도 중요하면 보존. |")
    L.append(f"| 18 | RF 하이퍼파라미터 (FS용) | RandomizedSearchCV: search space `{RF_SEARCH_SPACE}` / n_iter={RF_SEARCH_N_ITER}, cv={RF_SEARCH_CV}, scoring=neg_brier_score | importance 수치 신뢰성 확보. |")
    L.append(f"| 19 | MI subsample | Stratified {MI_SUBSAMPLE_SIZE:,d}행 (is_hit 비율 유지), seed={RANDOM_STATE} | 통계 안정성+연산 효율. |")
    L.append(f"| 20 | M2 발열 관리 | 단계 간 `time.sleep({COOLDOWN_SEC}s)`, `n_jobs={N_JOBS_HEAVY}` | 노트북 thermal throttling 완화. |")
    L.append("")

    # Feature pool construction
    L.append("## 2. 변수 그룹 정의 및 초기 풀 구성")
    L.append("")
    L.append("| 그룹 | 변수 수 | 내용 |")
    L.append("|---|---:|---|")
    L.append(f"| (a) xBA 핵심 | {len(X_BASE)} | launch_speed, launch_angle |")
    L.append(f"| (b) 배트 트래킹 + 결측플래그 | {len(BAT_TRACKING) + len(BAT_TRACKING_FLAGS)} | 7 numeric + 7 *_is_missing |")
    L.append(f"| (c) 타석 정체성 (binary) | 2 | stand_R, p_throws_R |")
    L.append(f"| (d) 투구 물리 (numeric) | {len(PITCH_NUMERIC)} | release_speed, pfx_*, plate_*, spin, break, arm_angle 등 |")
    L.append("| (d2) pitch_type one-hot | (가변) | pitch_type_* 더미 변수 |")
    L.append(f"| (e) PA 상황 (numeric) | {len(PA_NUMERIC)} | balls/strikes/outs/inning/age/order 등 |")
    L.append("| (e2) alignment one-hot | (가변) | if/of_fielding_alignment 더미 |")
    L.append(f"| (f) 구장 정적 | {len(PARK_STATIC)} | 펜스거리·높이·hr_park_effects·extra_distance·고도·roof·daytime |")
    L.append(f"| (g) 기상 동적 (dome-masked) | {len(WEATHER_DYNAMIC)} | 온도·습도·기압·풍속·풍향·강수·운량·돌풍 — closed roof일 시 외부 5종=0, 실내 기온 22°C/습도 50% 대체 (Phase 1 §7b) |")
    L.append(f"| **X_advanced 초기(One-Hot 후)** | **{meta['initial_n_features']}** | |")
    L.append("")

    # NaN imputation
    L.append("## 3. NaN 처리 (전체 2024 median 기반 imputation)")
    L.append("")
    L.append(f"- 결측 imputation 적용 컬럼 수: **{len(meta['imputation_medians'])}**")
    L.append("- *_is_missing 플래그는 결측 패턴 자체를 신호로 보존.")
    if meta["imputation_medians"]:
        L.append("")
        L.append("| 컬럼 | 2024 Median |")
        L.append("|---|---:|")
        for col, med in list(meta["imputation_medians"].items())[:25]:
            L.append(f"| `{col}` | {med:.4f} |")
        if len(meta["imputation_medians"]) > 25:
            L.append(f"| ... 외 {len(meta['imputation_medians']) - 25}개 | |")
    L.append("")

    # CV structure
    L.append(f"## 4. Cross-Validation 구조 ({CV_FOLDS}-fold StratifiedKFold)")
    L.append("")
    L.append(f"- 2024 전체 {meta['n_full']:,}행 → StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, random_state={RANDOM_STATE})")
    L.append(f"- 안타율(전체): {meta['full_hit_rate']:.4f}")
    L.append("- **2025는 Phase 5 외부 검증 전용** — 본 단계에서 어떤 통계도 사용하지 않음")
    L.append("")

    # Multicollinearity
    L.append(f"## 5. 다중공선성 분석 (|r| > {CORR_THRESHOLD}, Pearson)")
    L.append("")
    L.append(f"- 식별된 고상관 쌍: **{len(hi_corr_pairs)}건**")
    L.append(f"- 최종 제거된 변수 수: **{len(meta['dropped_via_correlation'])}개**")
    L.append("")
    if hi_corr_pairs:
        unique_pairs = {(min(p['a'], p['b']), max(p['a'], p['b'])): p for p in hi_corr_pairs}
        sorted_pairs = sorted(unique_pairs.values(), key=lambda x: -x["abs_r"])
        L.append("**고상관 쌍 (|r| 내림차순, 상위 30개) — `var_a/var_b`는 RobustScaler 후 분산**")
        L.append("")
        L.append("| 변수 A | 변수 B | abs(r) | var_a | var_b | 제거 | 규칙 |")
        L.append("|---|---|---:|---:|---:|---|---|")
        for p in sorted_pairs[:30]:
            dropped = p["dropped"] if p["dropped"] else "(보존)"
            va = p.get("var_a", float("nan"))
            vb = p.get("var_b", float("nan"))
            L.append(
                f"| `{p['a']}` | `{p['b']}` | {p['abs_r']:.3f} | "
                f"{va:.3f} | {vb:.3f} | `{dropped}` | {p['rule']} |"
            )
        if len(sorted_pairs) > 30:
            L.append(f"| _(외 {len(sorted_pairs) - 30}건)_ | | | | | | |")
        L.append("")
    L.append("**제거된 변수 전체 목록:**")
    L.append("")
    L.append(", ".join(f"`{c}`" for c in meta["dropped_via_correlation"]) or "_(없음)_")
    L.append("")

    # Scaling
    L.append("## 6. Robust Scaler")
    L.append("")
    L.append(f"- 스케일 적용 컬럼: **{len(meta['scale_cols'])}**개 (이진 0/1 변수는 제외)")
    L.append(f"- 2024 전체 fit, transform → `pipeline/output/phase2_scaler.joblib`")
    L.append("")

    # Sampling comparison
    L.append(f"## 7. 샘플링 비교 (3종 × XGBoost default × {CV_FOLDS}-fold CV)")
    L.append("")
    L.append("**OOF (Out-Of-Fold) predict_proba 기반 메트릭:**")
    L.append("")
    L.append("| 샘플링 | Train mean 0/1 | **OOF Brier** | OOF LogLoss | OOF F1 | OOF AUC | OOF P/R | fold Brier mean±SD |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in meta["sampling_results"].items():
        L.append(
            f"| **{name}** | {r['train_class_counts_mean_0']:,}/{r['train_class_counts_mean_1']:,} | "
            f"**{r['oof_brier']:.5f}** | {r['oof_logloss']:.5f} | "
            f"{r['oof_f1']:.4f} | {r['oof_roc_auc']:.4f} | "
            f"{r['oof_precision']:.4f}/{r['oof_recall']:.4f} | "
            f"{r['fold_brier_mean']:.5f}±{r['fold_brier_std']:.5f} |"
        )
    L.append("")
    L.append(f"- **최종 선정 샘플링: `{meta['best_sampling']}`** (OOF Brier 기준 최소)")
    L.append(f"- 선정 사유: OOF predict_proba 의 Brier(=평균 (y-p)²) 가 가장 낮은 기법. 확률 정상도 우선.")
    L.append("")

    # Feature Selection
    L.append("## 8. Feature Selection (RF importance + MI)")
    L.append("")
    L.append(f"- 학습 데이터: 최적 샘플링(`{meta['best_sampling']}`) 적용 X")
    L.append(
        f"- **RF Tuning**: `RandomizedSearchCV(n_iter={meta['rf_search_n_iter']}, "
        f"cv={meta['rf_search_cv']}, scoring='neg_brier_score', n_jobs={N_JOBS_HEAVY})`"
    )
    L.append(f"  - search space: `{meta['rf_search_space']}`")
    L.append(f"  - **best params**: `{meta['rf_best_params']}`")
    L.append(
        f"  - **best CV neg_brier_score (3-fold avg)**: {meta['rf_best_cv_score']:.5f} "
        f"(Brier = {-meta['rf_best_cv_score']:.5f})"
    )
    L.append(f"- **Mutual Information**: stratified {MI_SUBSAMPLE_SIZE:,d}행 subsample, seed={RANDOM_STATE}")
    L.append(
        f"- 제거 규칙: **RF importance & MI 둘 다 하위 {(1-FS_DROP_RANK_THRESHOLD)*100:.0f}% 동시 진입** "
        f"시 drop (X_BASE 제외)"
    )
    L.append("- 비고: 당초 3-model Permutation Importance 안이 채택되었으나, macOS joblib memmap 디스크 한계(RF default fit-된 모델의 worker 직렬화 시 OSError 28)로 인해 검증된 이전 step2 방식(RF RandomizedSearchCV → `feature_importances_` + MI)으로 복원.")
    L.append("")

    rf_imp = fs_artifacts["rf_importance"]
    mi = fs_artifacts["mi"]

    L.append("### 8.1 RF Importance Top 20")
    L.append("")
    L.append("| Rank | 변수 | Importance |")
    L.append("|---:|---|---:|")
    for i, (col, val) in enumerate(rf_imp.head(20).items(), 1):
        L.append(f"| {i} | `{col}` | {val:.5f} |")
    L.append("")

    L.append("### 8.2 Mutual Information Top 20")
    L.append("")
    L.append("| Rank | 변수 | MI |")
    L.append("|---:|---|---:|")
    for i, (col, val) in enumerate(mi.head(20).items(), 1):
        L.append(f"| {i} | `{col}` | {val:.5f} |")
    L.append("")

    L.append(f"### 8.3 Feature Selection으로 제거된 변수 ({len(meta['dropped_via_feature_selection'])}개)")
    L.append("")
    L.append(", ".join(f"`{c}`" for c in meta["dropped_via_feature_selection"]) or "_(없음)_")
    L.append("")

    # Final X_advanced
    L.append("## 9. 최종 X_advanced 변수 확정")
    L.append("")
    L.append(f"- X_BASE: **{len(X_BASE)}개** — Phase 3 통제군용")
    L.append(f"- X_advanced 초기: **{meta['initial_n_features']}개**")
    L.append(f"- 다중공선성 drop: **{len(meta['dropped_via_correlation'])}개**")
    L.append(f"- Feature Selection drop: **{len(meta['dropped_via_feature_selection'])}개**")
    L.append(f"- **X_advanced 최종: {len(meta['X_advanced_final'])}개**")
    L.append("")
    L.append("**X_advanced 최종 변수 목록:**")
    L.append("")
    L.append(", ".join(f"`{c}`" for c in meta["X_advanced_final"]))
    L.append("")

    # Artifacts
    L.append("## 10. 산출물")
    L.append("")
    L.append(
        "- `pipeline/output/phase2_X_full.parquet` (2024 전체, 스케일·corr-drop·FS-drop 적용된 최종 X)\n"
        "- `pipeline/output/phase2_y_full.parquet` (2024 전체 is_hit)\n"
        "- `pipeline/output/phase2_scaler.joblib` (RobustScaler + scale_cols 메타)\n"
        "- `pipeline/output/phase2_features.json` (X_base / X_advanced / drop 이력 등)\n"
        "- `pipeline/output/phase2_fs_ranking.csv` (RF importance + MI raw 점수 + percentile rank + drop 플래그)\n"
        "- Phase 3 이후는 본 `phase2_X_full.parquet` + 5-fold CV 동일 splits 으로 일관 비교"
    )
    L.append("")

    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] phase2_report.md 작성 완료 → {REPORT_PATH.relative_to(ROOT)}", flush=True)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    print("=" * 80, flush=True)
    print("Phase 2: 상관관계 분석, 스케일링, 샘플링 비교, Feature Selection (재실행)", flush=True)
    print("=" * 80, flush=True)

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"\n[load] 2024_data.parquet: {df.shape}", flush=True)

    print("[build] feature matrix 구성 (encoding) ...", flush=True)
    X, y = build_raw_feature_matrix(df)
    initial_n_features = X.shape[1]
    print(f"  → 초기 X shape: {X.shape}", flush=True)
    print(f"  → is_hit 분포: 0={int((y==0).sum()):,}  1={int((y==1).sum()):,}  hit_rate={y.mean():.4f}", flush=True)

    # (1) NaN imputation
    print("\n[impute] NaN → 전체 2024 median 으로 imputation ...", flush=True)
    numeric_cols_for_impute = [
        c for c in NUMERIC_FEATURES if c in X.columns and not c.endswith("_is_missing")
    ]
    X, medians = impute_numeric_with_median(X, numeric_cols_for_impute)
    print(f"  → imputation 적용 컬럼 수: {len(medians)}", flush=True)

    # (2) Robust Scaler
    print("\n[scale] Robust Scaler (numeric only, unique>2) ...", flush=True)
    scale_cols = [c for c in X.columns if X[c].nunique() > 2]
    print(f"  → 스케일 적용 컬럼: {len(scale_cols)} / 전체 {X.shape[1]}", flush=True)
    scaler = RobustScaler()
    X_s = X.copy()
    X_s[scale_cols] = scaler.fit_transform(X[scale_cols])

    # (3) Correlation drop (|r|>0.95, domain priority)
    print(f"\n[corr] |r| > {CORR_THRESHOLD} (Pearson) — 도메인 우선순위 drop ...", flush=True)
    drop_cols_corr, pair_log = correlation_drop_domain_priority(X_s, CORR_THRESHOLD)
    print(f"  → 고상관 쌍 {len(pair_log)}건, 제거 변수 {len(drop_cols_corr)}개", flush=True)
    if drop_cols_corr:
        print(f"  → 제거: {', '.join(drop_cols_corr[:8])}{', ...' if len(drop_cols_corr) > 8 else ''}", flush=True)
    X_s = X_s.drop(columns=drop_cols_corr)
    final_scale_cols = [c for c in scale_cols if c in X_s.columns]
    print(f"  → 정제 후 X shape: {X_s.shape}", flush=True)

    joblib.dump(
        {
            "scaler": scaler,
            "scale_cols_all": scale_cols,
            "scale_cols_after_corr_drop": final_scale_cols,
            "feature_cols_after_corr_drop": list(X_s.columns),
        },
        SCALER_PATH,
    )
    cooldown("scaling + correlation drop 완료, 샘플링 비교 전 대기")

    # (4) Sampling comparison via 5-fold CV OOF Brier
    print(f"\n[sampling] {CV_FOLDS}-fold CV OOF, XGBoost default, 3가지 샘플링 비교 ...", flush=True)
    sampling_results = run_sampling_comparison(X_s, y)
    best_sampling = min(sampling_results, key=lambda k: sampling_results[k]["oof_brier"])
    print(f"\n[best] 최적 샘플링 = '{best_sampling}' (OOF Brier 기준 최소)", flush=True)
    cooldown("샘플링 비교 완료, Feature Selection 전 대기")

    # (5) Feature Selection on best sampled
    print(f"\n[fs] Feature Selection 시작 (best='{best_sampling}') ...", flush=True)
    if best_sampling == "None":
        X_best, y_best = X_s, y
    elif best_sampling == "Under":
        rus = RandomUnderSampler(random_state=RANDOM_STATE)
        X_best, y_best = rus.fit_resample(X_s, y)
        X_best = pd.DataFrame(X_best, columns=X_s.columns)
        y_best = pd.Series(y_best, name="is_hit")
    elif best_sampling == "SMOTE":
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_best, y_best = sm.fit_resample(X_s, y)
        X_best = pd.DataFrame(X_best, columns=X_s.columns)
        y_best = pd.Series(y_best, name="is_hit")

    fs_artifacts, drop_cols_fs, rank_df, rf_best_params, rf_best_cv_score = feature_selection(
        X_best, y_best, FS_DROP_RANK_THRESHOLD
    )
    rank_df.to_csv(FS_RANK_CSV, index_label="feature")
    print(f"  → FS ranking 저장: {FS_RANK_CSV.relative_to(ROOT)}", flush=True)

    # Finalize X_advanced
    X_advanced_final = [c for c in X_s.columns if c not in drop_cols_fs]
    print(f"\n[final] X_advanced 최종: {len(X_advanced_final)}개 변수", flush=True)

    # Save final X (full 2024, X_advanced_final columns only)
    X_s[X_advanced_final].to_parquet(X_FULL_PARQUET, index=False)
    y.to_frame("is_hit").to_parquet(Y_FULL_PARQUET, index=False)
    print(f"  → 저장: phase2_X_full.parquet, phase2_y_full.parquet", flush=True)

    # Save metadata
    meta = {
        "X_base": X_BASE,
        "initial_features": list(X.columns),
        "initial_n_features": initial_n_features,
        "n_full": int(len(y)),
        "full_hit_rate": float(y.mean()),
        "imputation_medians": medians,
        "dropped_via_correlation": drop_cols_corr,
        "scale_cols": scale_cols,
        "scale_cols_after_corr_drop": final_scale_cols,
        "sampling_results": sampling_results,
        "best_sampling": best_sampling,
        "rf_importance_top20": fs_artifacts["rf_importance"].head(20).to_dict(),
        "mi_top20": fs_artifacts["mi"].head(20).to_dict(),
        "rf_search_space": RF_SEARCH_SPACE,
        "rf_search_n_iter": RF_SEARCH_N_ITER,
        "rf_search_cv": RF_SEARCH_CV,
        "rf_best_params": rf_best_params,
        "rf_best_cv_score": rf_best_cv_score,
        "dropped_via_feature_selection": drop_cols_fs,
        "X_advanced_final": X_advanced_final,
        "settings": {
            "RANDOM_STATE": RANDOM_STATE,
            "CV_FOLDS": CV_FOLDS,
            "CORR_THRESHOLD": CORR_THRESHOLD,
            "FS_DROP_RANK_THRESHOLD": FS_DROP_RANK_THRESHOLD,
            "MI_SUBSAMPLE_SIZE": MI_SUBSAMPLE_SIZE,
            "COOLDOWN_SEC": COOLDOWN_SEC,
            "N_JOBS_HEAVY": N_JOBS_HEAVY,
        },
    }
    FEATURES_JSON.write_text(json.dumps(meta, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"[save] phase2_features.json 저장 → {FEATURES_JSON.relative_to(ROOT)}", flush=True)

    # Report
    write_report(meta, pair_log, fs_artifacts)

    print("\n[done] Phase 2 완료.", flush=True)


if __name__ == "__main__":
    main()
