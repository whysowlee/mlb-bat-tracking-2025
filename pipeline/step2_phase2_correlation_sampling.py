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
                    rule = "preserve both X_BASE"
                elif a in X_BASE:
                    drop_target = b
                    rule = "X_BASE protected"
                elif b in X_BASE:
                    drop_target = a
                    rule = "X_BASE protected"
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
    print(f"  [cooldown {sec}s] " + (reason or "thermal management"), flush=True)
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
        print(f"\n  [sampling] '{name}' 5-fold CV starting ...", flush=True)
        results[name] = cv_oof_for_sampling(X, y, name, samplers[name])
        r = results[name]
        print(
            f"  [{name}] OOF Brier={r['oof_brier']:.5f}  LogLoss={r['oof_logloss']:.5f}  "
            f"F1={r['oof_f1']:.4f}  AUC={r['oof_roc_auc']:.4f}  "
            f"(fold Brier {r['fold_brier_mean']:.5f}±{r['fold_brier_std']:.5f})",
            flush=True,
        )
        if i < len(keys) - 1:
            cooldown(f"'{name}' sampling complete, waiting before next sampling")

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
        f"    Computing MI (stratified n={n_sub:,d}, hit_rate={y_sub.mean():.4f}) ...",
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
        f"  RF RandomizedSearchCV starting "
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

    cooldown("RF RandomizedSearchCV complete, waiting before MI")

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
        f"  → features in bottom {(1-drop_rank_threshold)*100:.0f}% for both criteria (RF & MI): {len(drop_cols)} dropped",
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
    L.append("# Phase 2 Report — Correlation Analysis, Scaling, Optimal Sampling, Feature Selection")
    L.append("")
    L.append(f"_Generated: {now}_  ")
    L.append("_Script: `pipeline/step2_phase2_correlation_sampling.py`_")
    L.append("")
    L.append("> This phase uses **2024_data.parquet only**. 2025_data is isolated until Phase 5 — no statistics are leaked.")
    L.append("")

    # Decisions
    L.append("## 1. Decisions (user-confirmed; all branches re-verified after Phase 1 dome-masking)")
    L.append("")
    L.append("| # | Decision Item | Adopted Approach | Rationale |")
    L.append("|---|---|---|---|")
    L.append("| 1 | X_base definition | `launch_speed`, `launch_angle` (2 features) | Matches MLB official xBA inputs, providing a clean control group. |")
    L.append("| 2 | Categorical encoding | One-Hot (stand/p_throws=L/R binary, pitch_type·alignment=dummy) | Few categories; no dimensionality explosion. |")
    L.append("| 3 | Train/Test split | **2025 isolated as validation only** / within 2024: **StratifiedKFold 5-fold CV (train+val combined)** | Zero data leakage. CV replaces hold-out test. |")
    L.append("| 4 | X_advanced initial feature pool | Broadly defined (pitch, PA, park, weather all included) | Designed to be pruned by multicollinearity and FS stages. |")
    L.append(f"| 5 | Multicollinearity threshold | **|r| > {CORR_THRESHOLD}** (Pearson) | Conservative criterion — removes only near-identical features. |")
    L.append("| 6 | Multicollinearity drop rule | Preserve X_BASE → **drop derived features first** → variance fallback | Domain semantics take priority (source > derived). Derived: `effective_speed`, `api_break_*`, `wx_wind_gusts_10m`, `max_wall_height`. |")
    L.append("| 7 | Processing order | impute → scale → corr_drop → sampling → FS | Ensures correlation stability after scaling and enables variance fallback. |")
    L.append("| 8 | Robust Scaler | numeric only (unique>2) | Robust to outliers; compatible with LogReg (Phase 3 control). |")
    L.append("| 9 | NaN imputation (numeric) | **Full 2024 median fill** | Same imputation applied across all CV folds. *_is_missing flags preserve the missingness signal. |")
    L.append("| 10 | NaN categorical handling | NaN → 'UNK' dummy | Preserves missingness patterns as a category. |")
    L.append("| 11 | Interaction FE | **Raw as-is** (delegated to tree models) | dome-masking already conveys dome-day signals adequately. No explicit product features added. |")
    L.append("| 12 | Sampling comparison | **All three**: None / RandomUnderSampler / SMOTE | Selection made after comparing results. No prior assumptions. |")
    L.append(f"| 13 | Sampling evaluation model | XGBoost **default** + StratifiedKFold {CV_FOLDS}-fold CV | Model-hypothesis neutral. Fair comparison with default baseline. |")
    L.append("| 14 | Sampling selection metric | Minimum **Brier Score (OOF predict_proba)** | ca-xBA hinges on probability calibration quality. F1 and AUC are supplementary. |")
    L.append("| 15 | EDA scope | Pearson correlation, \\|r\\|>0.95 deduplication, RF importance, MI | Multi-perspective signal aggregation. |")
    L.append("| 16 | FS computation model | RF (RandomizedSearchCV-tuned best_estimator) — split-impurity `feature_importances_` | Restores the validated approach from the previous step2. 3-model Permutation Importance excluded due to macOS joblib memmap disk limitations. |")
    L.append(f"| 17 | FS drop rule | **Drop when both criteria (RF importance & MI) simultaneously fall in the bottom {(1-FS_DROP_RANK_THRESHOLD)*100:.0f}%** (X_BASE excluded) | Conservative — retained if important by either metric. |")
    L.append(f"| 18 | RF hyperparameters (for FS) | RandomizedSearchCV: search space `{RF_SEARCH_SPACE}` / n_iter={RF_SEARCH_N_ITER}, cv={RF_SEARCH_CV}, scoring=neg_brier_score | Ensures reliability of importance estimates. |")
    L.append(f"| 19 | MI subsample | Stratified {MI_SUBSAMPLE_SIZE:,d} rows (is_hit ratio preserved), seed={RANDOM_STATE} | Statistical stability + computational efficiency. |")
    L.append(f"| 20 | M2 thermal management | `time.sleep({COOLDOWN_SEC}s)` between stages, `n_jobs={N_JOBS_HEAVY}` | Mitigates laptop thermal throttling. |")
    L.append("")

    # Feature pool construction
    L.append("## 2. Feature Group Definitions and Initial Pool Construction")
    L.append("")
    L.append("| Group | # Features | Description |")
    L.append("|---|---:|---|")
    L.append(f"| (a) xBA core | {len(X_BASE)} | launch_speed, launch_angle |")
    L.append(f"| (b) Bat tracking + missingness flags | {len(BAT_TRACKING) + len(BAT_TRACKING_FLAGS)} | 7 numeric + 7 *_is_missing |")
    L.append(f"| (c) Batter/pitcher handedness (binary) | 2 | stand_R, p_throws_R |")
    L.append(f"| (d) Pitch physics (numeric) | {len(PITCH_NUMERIC)} | release_speed, pfx_*, plate_*, spin, break, arm_angle, etc. |")
    L.append("| (d2) pitch_type one-hot | (variable) | pitch_type_* dummy variables |")
    L.append(f"| (e) PA situation (numeric) | {len(PA_NUMERIC)} | balls/strikes/outs/inning/age/order, etc. |")
    L.append("| (e2) alignment one-hot | (variable) | if/of_fielding_alignment dummies |")
    L.append(f"| (f) Park static | {len(PARK_STATIC)} | fence distances/heights, hr_park_effects, extra_distance, elevation, roof, daytime |")
    L.append(f"| (g) Weather dynamic (dome-masked) | {len(WEATHER_DYNAMIC)} | temp, humidity, pressure, wind speed/direction, precipitation, cloud cover, gusts — when closed roof: 5 outdoor vars=0, indoor temp 22°C/humidity 50% substituted (Phase 1 §7b) |")
    L.append(f"| **X_advanced initial (after One-Hot)** | **{meta['initial_n_features']}** | |")
    L.append("")

    # NaN imputation
    L.append("## 3. NaN Handling (full 2024 median-based imputation)")
    L.append("")
    L.append(f"- Columns with imputation applied: **{len(meta['imputation_medians'])}**")
    L.append("- *_is_missing flags preserve the missingness pattern itself as a signal.")
    if meta["imputation_medians"]:
        L.append("")
        L.append("| Column | 2024 Median |")
        L.append("|---|---:|")
        for col, med in list(meta["imputation_medians"].items())[:25]:
            L.append(f"| `{col}` | {med:.4f} |")
        if len(meta["imputation_medians"]) > 25:
            L.append(f"| ... and {len(meta['imputation_medians']) - 25} more | |")
    L.append("")

    # CV structure
    L.append(f"## 4. Cross-Validation Structure ({CV_FOLDS}-fold StratifiedKFold)")
    L.append("")
    L.append(f"- Full 2024: {meta['n_full']:,} rows → StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, random_state={RANDOM_STATE})")
    L.append(f"- Overall hit rate: {meta['full_hit_rate']:.4f}")
    L.append("- **2025 is reserved for Phase 5 external validation only** — no statistics from it are used in this phase")
    L.append("")

    # Multicollinearity
    L.append(f"## 5. Multicollinearity Analysis (|r| > {CORR_THRESHOLD}, Pearson)")
    L.append("")
    L.append(f"- High-correlation pairs identified: **{len(hi_corr_pairs)}**")
    L.append(f"- Features removed in total: **{len(meta['dropped_via_correlation'])}**")
    L.append("")
    if hi_corr_pairs:
        unique_pairs = {(min(p['a'], p['b']), max(p['a'], p['b'])): p for p in hi_corr_pairs}
        sorted_pairs = sorted(unique_pairs.values(), key=lambda x: -x["abs_r"])
        L.append("**High-correlation pairs (|r| descending, top 30) — `var_a/var_b` are post-RobustScaler variances**")
        L.append("")
        L.append("| Feature A | Feature B | abs(r) | var_a | var_b | Dropped | Rule |")
        L.append("|---|---|---:|---:|---:|---|---|")
        for p in sorted_pairs[:30]:
            dropped = p["dropped"] if p["dropped"] else "(retained)"
            va = p.get("var_a", float("nan"))
            vb = p.get("var_b", float("nan"))
            L.append(
                f"| `{p['a']}` | `{p['b']}` | {p['abs_r']:.3f} | "
                f"{va:.3f} | {vb:.3f} | `{dropped}` | {p['rule']} |"
            )
        if len(sorted_pairs) > 30:
            L.append(f"| _({len(sorted_pairs) - 30} more)_ | | | | | | |")
        L.append("")
    L.append("**Full list of removed features:**")
    L.append("")
    L.append(", ".join(f"`{c}`" for c in meta["dropped_via_correlation"]) or "_(none)_")
    L.append("")

    # Scaling
    L.append("## 6. Robust Scaler")
    L.append("")
    L.append(f"- Columns scaled: **{len(meta['scale_cols'])}** (binary 0/1 variables excluded)")
    L.append(f"- Fit and transformed on full 2024 → `pipeline/output/phase2_scaler.joblib`")
    L.append("")

    # Sampling comparison
    L.append(f"## 7. Sampling Comparison (3 strategies × XGBoost default × {CV_FOLDS}-fold CV)")
    L.append("")
    L.append("**OOF (Out-Of-Fold) predict_proba-based metrics:**")
    L.append("")
    L.append("| Sampling | Train mean 0/1 | **OOF Brier** | OOF LogLoss | OOF F1 | OOF AUC | OOF P/R | fold Brier mean±SD |")
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
    L.append(f"- **Final selected sampling: `{meta['best_sampling']}`** (minimum OOF Brier)")
    L.append(f"- Selection rationale: the technique with the lowest OOF predict_proba Brier score (= mean (y−p)²). Probability calibration is the primary criterion.")
    L.append("")

    # Feature Selection
    L.append("## 8. Feature Selection (RF importance + MI)")
    L.append("")
    L.append(f"- Training data: X with best sampling (`{meta['best_sampling']}`) applied")
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
    L.append(f"- **Mutual Information**: stratified {MI_SUBSAMPLE_SIZE:,d}-row subsample, seed={RANDOM_STATE}")
    L.append(
        f"- Drop rule: **drop when RF importance & MI both simultaneously fall in the bottom {(1-FS_DROP_RANK_THRESHOLD)*100:.0f}%** "
        f"(X_BASE excluded)"
    )
    L.append("- Note: The original 3-model Permutation Importance approach was adopted but subsequently reverted to the validated previous step2 method (RF RandomizedSearchCV → `feature_importances_` + MI) due to macOS joblib memmap disk limitations (OSError 28 during worker serialization of the RF default-fitted model).")
    L.append("")

    rf_imp = fs_artifacts["rf_importance"]
    mi = fs_artifacts["mi"]

    L.append("### 8.1 RF Importance Top 20")
    L.append("")
    L.append("| Rank | Feature | Importance |")
    L.append("|---:|---|---:|")
    for i, (col, val) in enumerate(rf_imp.head(20).items(), 1):
        L.append(f"| {i} | `{col}` | {val:.5f} |")
    L.append("")

    L.append("### 8.2 Mutual Information Top 20")
    L.append("")
    L.append("| Rank | Feature | MI |")
    L.append("|---:|---|---:|")
    for i, (col, val) in enumerate(mi.head(20).items(), 1):
        L.append(f"| {i} | `{col}` | {val:.5f} |")
    L.append("")

    L.append(f"### 8.3 Features Removed by Feature Selection ({len(meta['dropped_via_feature_selection'])})")
    L.append("")
    L.append(", ".join(f"`{c}`" for c in meta["dropped_via_feature_selection"]) or "_(none)_")
    L.append("")

    # Final X_advanced
    L.append("## 9. Final X_advanced Feature Set")
    L.append("")
    L.append(f"- X_BASE: **{len(X_BASE)} features** — Phase 3 control group")
    L.append(f"- X_advanced initial: **{meta['initial_n_features']} features**")
    L.append(f"- Multicollinearity drop: **{len(meta['dropped_via_correlation'])} features**")
    L.append(f"- Feature Selection drop: **{len(meta['dropped_via_feature_selection'])} features**")
    L.append(f"- **X_advanced final: {len(meta['X_advanced_final'])} features**")
    L.append("")
    L.append("**X_advanced final feature list:**")
    L.append("")
    L.append(", ".join(f"`{c}`" for c in meta["X_advanced_final"]))
    L.append("")

    # Artifacts
    L.append("## 10. Artifacts")
    L.append("")
    L.append(
        "- `pipeline/output/phase2_X_full.parquet` (full 2024, final X with scaling/corr-drop/FS-drop applied)\n"
        "- `pipeline/output/phase2_y_full.parquet` (full 2024 is_hit)\n"
        "- `pipeline/output/phase2_scaler.joblib` (RobustScaler + scale_cols metadata)\n"
        "- `pipeline/output/phase2_features.json` (X_base / X_advanced / drop history, etc.)\n"
        "- `pipeline/output/phase2_fs_ranking.csv` (RF importance + MI raw scores + percentile rank + drop flag)\n"
        "- Phase 3 onward uses this `phase2_X_full.parquet` + identical 5-fold CV splits for consistent comparison"
    )
    L.append("")

    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] phase2_report.md written → {REPORT_PATH.relative_to(ROOT)}", flush=True)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    print("=" * 80, flush=True)
    print("Phase 2: Correlation Analysis, Scaling, Sampling Comparison, Feature Selection (re-run)", flush=True)
    print("=" * 80, flush=True)

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"\n[load] 2024_data.parquet: {df.shape}", flush=True)

    print("[build] building feature matrix (encoding) ...", flush=True)
    X, y = build_raw_feature_matrix(df)
    initial_n_features = X.shape[1]
    print(f"  → initial X shape: {X.shape}", flush=True)
    print(f"  → is_hit distribution: 0={int((y==0).sum()):,}  1={int((y==1).sum()):,}  hit_rate={y.mean():.4f}", flush=True)

    # (1) NaN imputation
    print("\n[impute] NaN → imputing with full 2024 median ...", flush=True)
    numeric_cols_for_impute = [
        c for c in NUMERIC_FEATURES if c in X.columns and not c.endswith("_is_missing")
    ]
    X, medians = impute_numeric_with_median(X, numeric_cols_for_impute)
    print(f"  → columns with imputation applied: {len(medians)}", flush=True)

    # (2) Robust Scaler
    print("\n[scale] Robust Scaler (numeric only, unique>2) ...", flush=True)
    scale_cols = [c for c in X.columns if X[c].nunique() > 2]
    print(f"  → columns to scale: {len(scale_cols)} / total {X.shape[1]}", flush=True)
    scaler = RobustScaler()
    X_s = X.copy()
    X_s[scale_cols] = scaler.fit_transform(X[scale_cols])

    # (3) Correlation drop (|r|>0.95, domain priority)
    print(f"\n[corr] |r| > {CORR_THRESHOLD} (Pearson) — domain-priority drop ...", flush=True)
    drop_cols_corr, pair_log = correlation_drop_domain_priority(X_s, CORR_THRESHOLD)
    print(f"  → high-correlation pairs: {len(pair_log)}, dropped features: {len(drop_cols_corr)}", flush=True)
    if drop_cols_corr:
        print(f"  → dropped: {', '.join(drop_cols_corr[:8])}{', ...' if len(drop_cols_corr) > 8 else ''}", flush=True)
    X_s = X_s.drop(columns=drop_cols_corr)
    final_scale_cols = [c for c in scale_cols if c in X_s.columns]
    print(f"  → X shape after drop: {X_s.shape}", flush=True)

    joblib.dump(
        {
            "scaler": scaler,
            "scale_cols_all": scale_cols,
            "scale_cols_after_corr_drop": final_scale_cols,
            "feature_cols_after_corr_drop": list(X_s.columns),
        },
        SCALER_PATH,
    )
    cooldown("scaling + correlation drop complete, waiting before sampling comparison")

    # (4) Sampling comparison via 5-fold CV OOF Brier
    print(f"\n[sampling] {CV_FOLDS}-fold CV OOF, XGBoost default, comparing 3 sampling strategies ...", flush=True)
    sampling_results = run_sampling_comparison(X_s, y)
    best_sampling = min(sampling_results, key=lambda k: sampling_results[k]["oof_brier"])
    print(f"\n[best] best sampling = '{best_sampling}' (minimum OOF Brier)", flush=True)
    cooldown("sampling comparison complete, waiting before Feature Selection")

    # (5) Feature Selection on best sampled
    print(f"\n[fs] Feature Selection starting (best='{best_sampling}') ...", flush=True)
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
    print(f"  → FS ranking saved: {FS_RANK_CSV.relative_to(ROOT)}", flush=True)

    # Finalize X_advanced
    X_advanced_final = [c for c in X_s.columns if c not in drop_cols_fs]
    print(f"\n[final] X_advanced final: {len(X_advanced_final)} features", flush=True)

    # Save final X (full 2024, X_advanced_final columns only)
    X_s[X_advanced_final].to_parquet(X_FULL_PARQUET, index=False)
    y.to_frame("is_hit").to_parquet(Y_FULL_PARQUET, index=False)
    print(f"  → saved: phase2_X_full.parquet, phase2_y_full.parquet", flush=True)

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
    print(f"[save] phase2_features.json saved → {FEATURES_JSON.relative_to(ROOT)}", flush=True)

    # Report
    write_report(meta, pair_log, fs_artifacts)

    print("\n[done] Phase 2 complete.", flush=True)


if __name__ == "__main__":
    main()
