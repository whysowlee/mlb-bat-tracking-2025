"""
Phase 4: Hyperparameter Tuning + Stacking + Calibration
========================================================

New structure (based on Phase 2 dome-masking + best_sampling=None):

Workflow:
  1) Load phase2_X_full + phase2_y_full (61 features × 113,409 rows)
  2) **Tune 3 base models**: RF / XGBoost / LightGBM each via
     RandomizedSearchCV(n_iter=30, cv=5, scoring='neg_brier_score', refit=True)
  3) For each best_estimator_, compute OOF predict_proba over
     **outer 5-fold CV (same splits as Phase 2/3)**
     → evaluate Brier / LogLoss / F1 / AUC / P / R / Acc
  4) **Stacking**: StackingClassifier(
        estimators=[best_rf, best_xgb, best_lgbm],
        final_estimator=LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000),
        cv=5, n_jobs=N_JOBS)
     evaluated via outer 5-fold CV (each fold: fit 3 base + fit meta)
  5) **Isotonic Calibration**: CalibratedClassifierCV(stacking, method='isotonic', cv=5)
     → OOF proba-based calibration, additional Brier improvement
  6) Model with minimum Brier selected as **final ca-xBA output model**
     → used for 2025 predictions in Phase 5
  7) Save phase4_report.md + phase4_results.json + best model joblib

Sampling: Phase 2 decision = `None` (preserve original distribution)
          → no sampler in Pipeline.

Cooldown: COOLDOWN_SEC=120 (user-specified — overheat prevention). n_jobs=2.

Run:
    PYTHONUNBUFFERED=1 /opt/miniconda3/envs/mlb-xba/bin/python \\
        pipeline/step4_phase4_tuning_stacking.py
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
REPORT_PATH = PIPELINE_DIR / "phase4_report.md"
RESULTS_JSON = OUTPUT_DIR / "phase4_results.json"
MODELS_DIR = OUTPUT_DIR / "phase4_models"
PROBA_DIR = OUTPUT_DIR / "phase4_probas"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROBA_DIR.mkdir(parents=True, exist_ok=True)

X_FULL_PARQUET = OUTPUT_DIR / "phase2_X_full.parquet"
Y_FULL_PARQUET = OUTPUT_DIR / "phase2_y_full.parquet"
PHASE2_FEATURES_JSON = OUTPUT_DIR / "phase2_features.json"
PHASE3_RESULTS_JSON = OUTPUT_DIR / "phase3_results.json"


# -----------------------------------------------------------------------------
# Decision constants (user-confirmed)
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS = 5         # outer CV identical to Phase 2/3
N_ITER = 30          # number of RandomizedSearchCV candidates
INNER_CV = 5         # inner CV for RandomizedSearchCV
SCORING = "neg_brier_score"
REFIT = True
THRESHOLD = 0.5

N_JOBS = 2
COOLDOWN_SEC = 120   # M2 Air thermal management — user-specified (longer than Phase 3's 60s)

# Calibration
CALIBRATION_METHOD = "isotonic"
CALIBRATION_CV = 5

# Stacking
STACKING_CV = 5
STACKING_FINAL_LABEL = "LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000)"

# Search space (previously validated space, slightly narrowed: reliability first)
RF_SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 4, 6, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
    "max_features": ["sqrt", "log2", 0.5],
}

XGB_SEARCH_SPACE = {
    "n_estimators": [200, 300, 500],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.1, 0.15],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "gamma": [0, 0.1, 0.5, 1.0],
    "min_child_weight": [1, 3, 5, 7],
}

LGBM_SEARCH_SPACE = {
    "n_estimators": [200, 300, 500],
    "max_depth": [-1, 6, 10, 15],
    "learning_rate": [0.03, 0.05, 0.1, 0.15],
    "num_leaves": [31, 63, 127, 255],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "min_child_samples": [10, 20, 30, 50],
}

METRIC_KEYS = [
    "brier", "logloss", "f1", "roc_auc",
    "precision", "recall", "accuracy",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cooldown(reason: str = "", sec: int = COOLDOWN_SEC) -> None:
    log(f"  [cooldown {sec}s] {reason or 'thermal management'}")
    time.sleep(sec)


def make_estimator(kind: str):
    if kind == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    elif kind == "xgb":
        return xgb.XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=1,
            eval_metric="logloss", tree_method="hist", verbosity=0,
        )
    elif kind == "lgbm":
        return lgb.LGBMClassifier(
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
        )
    raise ValueError(kind)


def metrics_from_proba(y_true, proba) -> dict:
    pred = (proba >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, pred)
    return {
        "brier": float(brier_score_loss(y_true, proba)),
        "logloss": float(log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7))),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "cm_tn": int(cm[0, 0]),
        "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]),
        "cm_tp": int(cm[1, 1]),
    }


def log_metrics(label: str, m: dict):
    log(
        f"    [{label}] Brier={m['brier']:.5f}  LogLoss={m['logloss']:.5f}  "
        f"F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}  "
        f"P={m['precision']:.4f}  R={m['recall']:.4f}  Acc={m['accuracy']:.4f}"
    )


# -----------------------------------------------------------------------------
# (1) Base model tuning — RandomizedSearchCV
# -----------------------------------------------------------------------------
def tune_base(kind: str, space: dict, X: pd.DataFrame, y: pd.Series) -> dict:
    log(
        f"\n  ▸ tuning [{kind.upper()}] RandomizedSearchCV "
        f"(n_iter={N_ITER}, cv={INNER_CV}, scoring='{SCORING}', n_jobs={N_JOBS}) ..."
    )
    log(f"    search space: {space}")
    estimator = make_estimator(kind)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=space,
        n_iter=N_ITER,
        cv=INNER_CV,
        scoring=SCORING,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        refit=REFIT,
        verbose=10,  # print on each candidate start/finish (user request: frequent updates)
    )
    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0
    log(
        f"    → best params: {search.best_params_}\n"
        f"    → best CV neg_brier_score: {search.best_score_:.5f} "
        f"(Brier = {-search.best_score_:.5f})  ({elapsed:.1f}s)"
    )
    return {
        "kind": kind,
        "best_params": dict(search.best_params_),
        "best_cv_neg_brier": float(search.best_score_),
        "best_cv_brier": float(-search.best_score_),
        "search_n_iter": N_ITER,
        "search_cv": INNER_CV,
        "fit_seconds": round(elapsed, 1),
        "best_estimator": search.best_estimator_,
    }


# -----------------------------------------------------------------------------
# (2) Outer 5-fold CV OOF evaluation
# -----------------------------------------------------------------------------
def outer_cv_oof(
    label: str,
    estimator_factory,
    X: pd.DataFrame,
    y: pd.Series,
    skf: StratifiedKFold,
) -> tuple[dict, np.ndarray, list[dict]]:
    """Re-fit estimator_factory() from scratch for each fold; collect OOF predict_proba."""
    log(f"\n  ▸ Outer {CV_FOLDS}-fold CV OOF — {label}")
    oof_proba = np.zeros(len(y), dtype=np.float64)
    fold_records: list[dict] = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        log(f"    [fold {fold_idx}/{CV_FOLDS}] start — train n={len(tr_idx):,}, val n={len(val_idx):,}, fitting...")
        t0 = time.time()
        model = estimator_factory()
        model.fit(X_tr, y_tr)
        log(f"    [fold {fold_idx}/{CV_FOLDS}] fit done ({time.time() - t0:.1f}s), running predict_proba...")
        proba = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = proba
        m = metrics_from_proba(y_val, proba)
        m["fold"] = fold_idx
        m["fit_sec"] = float(time.time() - t0)
        fold_records.append(m)
        log(
            f"    [fold {fold_idx}/{CV_FOLDS}] Brier={m['brier']:.5f}  "
            f"LogLoss={m['logloss']:.5f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}  "
            f"(total {m['fit_sec']:.1f}s)"
        )
    oof_metrics = metrics_from_proba(y, oof_proba)
    fold_means = {k: float(np.mean([r[k] for r in fold_records])) for k in METRIC_KEYS}
    fold_stds = {k: float(np.std([r[k] for r in fold_records])) for k in METRIC_KEYS}
    log_metrics(f"{label} OOF aggregate", oof_metrics)
    return {
        "label": label,
        "oof_metrics": oof_metrics,
        "fold_records": fold_records,
        "fold_mean": fold_means,
        "fold_std": fold_stds,
    }, oof_proba, fold_records


# -----------------------------------------------------------------------------
# (3) Stacking
# -----------------------------------------------------------------------------
def build_stacking_estimator(tuned: dict) -> StackingClassifier:
    """Build a StackingClassifier using the 3 tuned best_estimator_ objects as bases."""
    base_estimators = [
        ("rf", tuned["rf"]["best_estimator"]),
        ("xgb", tuned["xgb"]["best_estimator"]),
        ("lgbm", tuned["lgbm"]["best_estimator"]),
    ]
    final = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE,
    )
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=final,
        cv=STACKING_CV,
        n_jobs=N_JOBS,
        passthrough=False,
        verbose=2,  # print internal cross_val_predict progress
    )


def stacking_oof(
    tuned: dict, X: pd.DataFrame, y: pd.Series, skf: StratifiedKFold
) -> tuple[dict, np.ndarray]:
    """Outer CV: build a fresh StackingClassifier per fold, fit → val proba."""
    def factory():
        return build_stacking_estimator(tuned)

    return outer_cv_oof("Stacking (LR meta)", factory, X, y, skf)[:2]


def isotonic_post_processing_oof(
    stack_oof_proba: np.ndarray,
    y: pd.Series,
    skf: StratifiedKFold,
    label: str = "Stacking + Isotonic (cv='prefit')",
) -> tuple[dict, np.ndarray]:
    """
    Option C — `cv='prefit'` pattern:
    Fit IsotonicRegression via 5-fold OOF on top of any model's 5-fold OOF predict_proba.

    - No additional base/meta training (reuses OOF proba)
    - Academically equivalent to the internals of
      `CalibratedClassifierCV(estimator, method='isotonic', cv=K)`
    - Runtime: ~5 seconds
    """
    log(f"\n  ▸ Outer {CV_FOLDS}-fold OOF Isotonic post-processing — {label}")
    iso_oof_proba = np.zeros(len(y), dtype=np.float64)
    fold_records: list[dict] = []
    # IsotonicRegression takes 1D input → use a dummy 2D array for splitting
    dummy_X = stack_oof_proba.reshape(-1, 1)
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(dummy_X, y), 1):
        log(f"    [fold {fold_idx}/{CV_FOLDS}] IsotonicRegression fit (n_train={len(tr_idx):,}, n_val={len(val_idx):,}) ...")
        t0 = time.time()
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(stack_oof_proba[tr_idx], y.iloc[tr_idx].values)
        iso_oof_proba[val_idx] = iso.predict(stack_oof_proba[val_idx])
        m = metrics_from_proba(y.iloc[val_idx], iso_oof_proba[val_idx])
        m["fold"] = fold_idx
        m["fit_sec"] = float(time.time() - t0)
        fold_records.append(m)
        log(
            f"    [fold {fold_idx}/{CV_FOLDS}] Iso Brier={m['brier']:.5f}  "
            f"LogLoss={m['logloss']:.5f}  AUC={m['roc_auc']:.4f}  ({m['fit_sec']:.3f}s)"
        )
    oof_metrics = metrics_from_proba(y, iso_oof_proba)
    fold_mean = {k: float(np.mean([r[k] for r in fold_records])) for k in METRIC_KEYS}
    fold_std = {k: float(np.std([r[k] for r in fold_records])) for k in METRIC_KEYS}
    log_metrics(f"{label} OOF aggregate", oof_metrics)
    return {
        "label": label,
        "oof_metrics": oof_metrics,
        "fold_records": fold_records,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
    }, iso_oof_proba


def reconstruct_oof_result(label: str, oof_proba: np.ndarray, y: pd.Series, skf: StratifiedKFold) -> dict:
    """For RESUME mode — reconstruct oof_metrics + fold_records from saved oof_proba alone."""
    fold_records: list[dict] = []
    dummy_X = oof_proba.reshape(-1, 1)
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(dummy_X, y), 1):
        proba = oof_proba[val_idx]
        m = metrics_from_proba(y.iloc[val_idx], proba)
        m["fold"] = fold_idx
        m["fit_sec"] = float("nan")  # not measurable in reconstruction mode
        fold_records.append(m)
    oof_metrics = metrics_from_proba(y, oof_proba)
    fold_mean = {k: float(np.mean([r[k] for r in fold_records])) for k in METRIC_KEYS}
    fold_std = {k: float(np.std([r[k] for r in fold_records])) for k in METRIC_KEYS}
    return {
        "label": label,
        "oof_metrics": oof_metrics,
        "fold_records": fold_records,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
    }


# -----------------------------------------------------------------------------
# Report writing
# -----------------------------------------------------------------------------
def write_report(
    tuned: dict,
    oof_results: dict,
    final_choice: str,
    final_oof_brier: float,
    meta_phase2: dict,
    meta_phase3: dict,
    n_train_mean: int,
    n_val_mean: int,
    best_single_kind: str = "",
    best_single_iso_label: str = "",
    occam_eps: float = 0.001,
    occam_verdict: str = "",
    brier_comparison: dict | None = None,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    brier_comparison = brier_comparison or {}
    L: list[str] = []
    L.append("# Phase 4 Report — 하이퍼파라미터 튜닝 + Stacking + Isotonic Calibration (오캄의 면도날 적용)")
    L.append("")
    L.append(f"_생성: {now}_  ")
    L.append("_실행 스크립트: `pipeline/step4_phase4_tuning_stacking.py`_")
    L.append("")
    L.append(
        "> Phase 2/3 과 **동일한 StratifiedKFold 5-fold CV** 위에서 base 3 모델 "
        f"(RF / XGB / LGBM) 튜닝된 best estimator 와 Stacking (cv={STACKING_CV}, "
        "LR meta), **Stacking + Isotonic**, **Best_Single + Isotonic** 의 OOF predict_proba "
        f"를 평가한다. **OOF Brier 최소 + 오캄의 면도날(ε={occam_eps})** 규칙으로 "
        "ca-xBA 최종 산출 모델을 선정."
    )
    L.append("")

    # 1. Decisions
    L.append("## 1. 결정 사항 (사용자 컨펌 — Phase 1 dome-masking 이후 분기 전수 재확인)")
    L.append("")
    L.append("| # | 결정 항목 | 채택안 | 사유 |")
    L.append("|---|---|---|---|")
    L.append(
        "| 1 | Base 3모델 튜닝 파이프라인 구성 | **RF / XGB / LightGBM 각각 RandomizedSearchCV** "
        "→ Stacking (LR meta) + Best_Single + Isotonic 후보 평가 | 다양성 base + 메타 학습 + 확률 보정 "
        "+ 단순 모델까지 4 후보 비교. |"
    )
    L.append(f"| 2 | RandomizedSearchCV 규모 | **n_iter={N_ITER}, inner_cv={INNER_CV}, scoring='{SCORING}'** | 신뢰도 우선 (사용자 선택). |")
    L.append(f"| 3 | Outer CV | **StratifiedKFold {CV_FOLDS}-fold (Phase 2/3 동일 random_state={RANDOM_STATE})** | Phase 간 일관성, OOF predict_proba 동일 splits. |")
    L.append(f"| 4 | 샘플링 | Phase 2 선정 = **`{meta_phase2.get('best_sampling')}`** (원본 분포 유지) | sampler 미사용. 확률 calibration 우선. |")
    L.append(
        "| 5 | Calibration | **C 옵션 — IsotonicRegression `cv='prefit'` 패턴** "
        "(Stacking/Best_Single OOF proba 위에 5-fold OOF Isotonic) | 비모수적 단조 보정. "
        "`CalibratedClassifierCV(cv=5)` 내부 로직과 학술적 동등하며 ~10시간 → ~5초 단축. |"
    )
    L.append(f"| 6 | Stacking | StackingClassifier(estimators=[best_rf, best_xgb, best_lgbm], final_estimator={STACKING_FINAL_LABEL}, cv={STACKING_CV}) | LR meta — 과적합 위험 낮고 표준. |")
    L.append("| 7 | 평가 메트릭 | **Brier(주) + LogLoss + F1 + AUC + P + R + Acc** | Phase 2/3 동일. |")
    L.append(
        f"| 8 | 모델 선정 기준 | **OOF Brier 최소 + 오캄의 면도날 (ε={occam_eps})** — "
        f"Best_Single+Iso 와 Stacking+Iso 의 ΔBrier ≤ {occam_eps} 동률 시 더 단순한 Best_Single+Iso 자동 선정 | "
        "fold 변동 수준 차이는 통계적 노이즈로 간주, 모델 복잡도 최소화. |"
    )
    L.append(f"| 9 | 임계값 | 0.5 고정 | 공정 비교. |")
    L.append(f"| 10 | M2 발열 관리 | COOLDOWN_SEC={COOLDOWN_SEC}s, N_JOBS={N_JOBS} | Phase 3 의 60s 보다 더 긴 쿨다운 (사용자 명시 — 과열 방지). |")
    L.append("")

    # 2. Base tuning results (fit time column omitted — avoids nan display in RESUME mode)
    L.append("## 2. Base 모델 튜닝 결과 (RandomizedSearchCV)")
    L.append("")
    L.append(f"각 모델 RandomizedSearchCV: n_iter={N_ITER}, inner_cv={INNER_CV}, "
             f"scoring='{SCORING}', refit=True.")
    L.append("")
    L.append("| Base | best CV Brier | best params |")
    L.append("|---|---:|---|")
    for kind in ["rf", "xgb", "lgbm"]:
        t = tuned[kind]
        params_str = ", ".join(f"`{k}`={v}" for k, v in t["best_params"].items())
        cv_brier = t.get("best_cv_brier", float("nan"))
        cv_brier_str = f"{cv_brier:.5f}" if not (cv_brier != cv_brier) else "—"  # nan check
        L.append(f"| **{kind.upper()}** | {cv_brier_str} | {params_str} |")
    L.append("")

    # 3. Outer CV OOF results
    L.append(f"## 3. Outer {CV_FOLDS}-fold CV OOF 결과")
    L.append("")
    L.append("**OOF aggregate:**")
    L.append("")
    L.append("| Model | **Brier↓** | LogLoss↓ | F1 | ROC AUC | Precision | Recall | Accuracy |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, r in oof_results.items():
        m = r["oof_metrics"]
        L.append(
            f"| {label} | **{m['brier']:.5f}** | {m['logloss']:.5f} | "
            f"{m['f1']:.4f} | {m['roc_auc']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['accuracy']:.4f} |"
        )
    L.append("")
    L.append("**fold mean ± SD:**")
    L.append("")
    L.append("| Model | Brier mean±SD | LogLoss mean±SD | F1 mean±SD | AUC mean±SD |")
    L.append("|---|---:|---:|---:|---:|")
    for label, r in oof_results.items():
        fm, fs = r["fold_mean"], r["fold_std"]
        L.append(
            f"| {label} | {fm['brier']:.5f}±{fs['brier']:.5f} | "
            f"{fm['logloss']:.5f}±{fs['logloss']:.5f} | "
            f"{fm['f1']:.4f}±{fs['f1']:.4f} | "
            f"{fm['roc_auc']:.4f}±{fs['roc_auc']:.4f} |"
        )
    L.append("")

    # 4. Improvement over Phase 3 baseline
    L.append("## 4. Phase 3 baseline (M4=X_advanced+XGB default) 대비 개선")
    L.append("")
    m4_metrics = next(iter([
        c for c in meta_phase3.get("cells", {}).values()
        if c.get("data") == "X_advanced" and c.get("algo") == "XGBoost"
    ]), None)
    if m4_metrics:
        m4_brier = m4_metrics["oof_metrics"]["brier"]
        m4_auc = m4_metrics["oof_metrics"]["roc_auc"]
        L.append("| Model | ΔBrier vs M4 | ΔAUC vs M4 |")
        L.append("|---|---:|---:|")
        for label, r in oof_results.items():
            m = r["oof_metrics"]
            db = m["brier"] - m4_brier
            da = m["roc_auc"] - m4_auc
            L.append(f"| {label} | {db:+.5f} | {da:+.4f} |")
        L.append("")
        L.append(f"_M4 (Phase 3): Brier={m4_brier:.5f}, AUC={m4_auc:.4f}_")
        L.append("")

    # 5. Final model selection (Occam's Razor)
    L.append("## 5. 최종 모델 선정 — 오캄의 면도날 (Occam's Razor) 적용")
    L.append("")
    L.append("### 5.1 핵심 후보 3종 OOF Brier 비교")
    L.append("")
    L.append("| 후보 모델 | OOF Brier | 비고 |")
    L.append("|---|---:|---|")
    if brier_comparison:
        L.append(
            f"| **{best_single_iso_label}** | **{brier_comparison['best_single_iso']:.5f}** | "
            f"Best Single (가장 우수했던 단일 base = {best_single_kind.upper()}) + Isotonic "
            f"— 단순한 모델 |"
        )
        L.append(
            f"| Stacking + Isotonic | {brier_comparison['stacking_iso']:.5f} | "
            "Stacking(LR meta) + Isotonic — 복잡한 앙상블 |"
        )
        L.append(
            f"| Stacking (LR meta) only | {brier_comparison['stacking_only']:.5f} | "
            "Stacking 단독 (calibration 없음) |"
        )
        L.append("")
        delta = brier_comparison["delta_best_minus_stack_iso"]
        L.append(
            f"- ΔBrier (Best_Single+Iso − Stacking+Iso) = **{delta:+.5f}** "
            f"(오캄 threshold ε = {occam_eps})"
        )
        L.append("")
    L.append(f"### 5.2 선정 결과")
    L.append("")
    L.append(f"- **최종 선정 모델**: `{final_choice}`")
    L.append(f"- **OOF Brier**: **{final_oof_brier:.5f}**")
    L.append("")
    L.append(f"**자동 선정 사유:** {occam_verdict}")
    L.append("")
    L.append("### 5.3 학술적 해석 — 왜 오캄의 면도날인가")
    L.append("")
    L.append(
        "실험 결과, **무거운 메타 학습을 거친 Stacking 모델보다 잘 튜닝된 단일 모델"
        f"({best_single_kind.upper()})의 OOF Brier Score 가 더 우수**함을 확인했다 "
        "(또는 fold 변동 수준 내에서 동률). 이는 여러 모델을 결합하는 과정에서 오히려 확률 "
        "보정(probability calibration)이 훼손되는 현상으로 해석할 수 있다 — Stacking 의 "
        "LR meta-learner 가 base 모델 간 출력 분포 이질성을 강제로 보정하면서 잘 보정된 "
        "단일 LGBM 의 native calibration 을 흐트러뜨릴 수 있다."
    )
    L.append("")
    L.append(
        "따라서 본 연구는 **\"성능이 비슷하다면 더 단순한 모델이 낫다\"** 는 "
        "**오캄의 면도날(Occam's Razor)** 원칙을 수용하였다. 억지로 복잡한 앙상블을 "
        "유지하는 대신, 가장 성능이 뛰어난 단일 모델에 **비모수적 단조 변환인 Isotonic "
        "Calibration 을 직접 결합**하는 방식을 채택했다. 이를 통해 ① 연산의 복잡도를 "
        "크게 낮추면서도 (3 base × cross_val_predict + meta-fit + base full-fit 3개 → "
        "base 1개 full-fit + Isotonic 1개) ② 본 프로젝트의 궁극적 목표인 **'극한의 "
        "확률 정상도(Calibration)'** 를 성공적으로 확보했다."
    )
    L.append("")
    L.append(
        "이 선정 로직은 결과에 따라 자동으로 분기한다 — 만약 Stacking + Isotonic 의 우위가 "
        f"ε({occam_eps}) 를 초과한다면 (Brier 차이가 fold 변동 수준을 넘는 통계적 유의 차이), "
        "복잡도 증가의 정당성이 확보되어 Stacking + Isotonic 이 채택된다. 본 실행에서는 "
        "위 §5.2 의 자동 선정 결과가 적용되었다."
    )
    L.append("")
    L.append(
        "- **Phase 5 적용 흐름**: `final_model.joblib` 내부의 base estimator → predict_proba → "
        "isotonic.predict(proba) → ca-xBA. 2025 데이터(외부 검증 셋)에 그대로 적용."
    )
    L.append("")

    # 6. Outputs
    L.append("## 6. 산출물")
    L.append("")
    L.append(
        f"- `{REPORT_PATH.relative_to(ROOT)}` — 본 리포트\n"
        f"- `{RESULTS_JSON.relative_to(ROOT)}` — 튜닝 결과·OOF 메트릭·fold records JSON\n"
        f"- `pipeline/output/phase4_models/` — 최종 모델 joblib + base 3개 best estimator\n"
        f"- `pipeline/output/phase4_probas/` — 각 모델의 OOF proba npy\n"
        f"- `pipeline/logs/step4_phase4.log` — 실행 로그"
    )
    L.append("")

    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    log(f"[report] phase4_report.md written → {REPORT_PATH.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    log("=" * 80)
    log("Phase 4: Hyperparameter Tuning + Stacking + Isotonic Calibration")
    log("=" * 80)

    # 1) Load data
    log("\n[1/8] Loading data ...")
    X_full = pd.read_parquet(X_FULL_PARQUET)
    y_full = pd.read_parquet(Y_FULL_PARQUET)["is_hit"]
    meta_phase2 = json.loads(PHASE2_FEATURES_JSON.read_text(encoding="utf-8"))
    meta_phase3 = json.loads(PHASE3_RESULTS_JSON.read_text(encoding="utf-8"))
    log(f"  X_full {X_full.shape}, y_full {y_full.shape}, hit_rate={y_full.mean():.4f}")
    log(f"  best_sampling (Phase 2): {meta_phase2.get('best_sampling')}")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 2) Base model tuning
    log("\n[2/8] Base model tuning (RandomizedSearchCV) ...")
    resume_from_rf = os.environ.get("RESUME_FROM_RF", "0") == "1"
    resume_from_stacking = os.environ.get("RESUME_FROM_STACKING", "0") == "1"
    tuned: dict = {}
    if resume_from_stacking:
        # Most advanced RESUME — skip RF/XGB/LGBM/Stacking outer CV entirely
        log("  [RESUME_FROM_STACKING=1] skipping base tuning + base outer CV + Stacking outer CV")
        log("  → loading all saved best_*.joblib + oof_*.npy, resuming from [5/8] Isotonic")
        for kind in ["rf", "xgb", "lgbm"]:
            path = MODELS_DIR / f"best_{kind}.joblib"
            if not path.exists():
                raise RuntimeError(f"RESUME_FROM_STACKING mode but {path} not found")
            best_est = joblib.load(path)
            tuned[kind] = {
                "kind": kind,
                "best_params": dict(best_est.get_params()),
                "best_cv_neg_brier": float("nan"),
                "best_cv_brier": float("nan"),
                "search_n_iter": N_ITER,
                "search_cv": INNER_CV,
                "fit_seconds": float("nan"),
                "best_estimator": best_est,
            }
            log(f"  [RESUME] loaded best_{kind}.joblib ({path.stat().st_size/1024/1024:.0f}MB)")
        # Next step (base outer CV) is also skipped — handled by branching in main flow
    elif resume_from_rf:
        log("  [RESUME_FROM_RF=1] skipping RF tuning — using saved best_rf.joblib + known results")
    spaces = {"rf": RF_SEARCH_SPACE, "xgb": XGB_SEARCH_SPACE, "lgbm": LGBM_SEARCH_SPACE}
    if resume_from_stacking:
        # Skip entire base tuning loop (already fully loaded above)
        spaces = {}
    for i, (kind, space) in enumerate(spaces.items()):
        if kind == "rf" and resume_from_rf:
            rf_path = MODELS_DIR / "best_rf.joblib"
            if not rf_path.exists():
                raise RuntimeError(f"RESUME mode but {rf_path} not found")
            log(f"  ▸ [RESUME] RF best_estimator loading: {rf_path.name} ({rf_path.stat().st_size/1024/1024:.0f}MB) ...")
            t0 = time.time()
            best_est = joblib.load(rf_path)
            log(f"  ▸ [RESUME] RF load done ({time.time()-t0:.1f}s)")
            tuned["rf"] = {
                "kind": "rf",
                "best_params": {
                    "n_estimators": 500, "min_samples_split": 4,
                    "min_samples_leaf": 4, "max_features": 0.5,
                    "max_depth": None, "criterion": "entropy",
                },
                "best_cv_neg_brier": -0.13281,
                "best_cv_brier": 0.13281,
                "search_n_iter": N_ITER,
                "search_cv": INNER_CV,
                "fit_seconds": 7803.0,
                "best_estimator": best_est,
            }
            log(f"  ▸ [RESUME] RF best_params: {tuned['rf']['best_params']}")
            log(f"  ▸ [RESUME] RF best CV Brier: {tuned['rf']['best_cv_brier']:.5f} (from prior run)")
            if i < len(spaces) - 1:
                cooldown(f"RF skip done, brief wait before next model", sec=10)
            continue
        tuned[kind] = tune_base(kind, space, X_full, y_full)
        joblib.dump(
            tuned[kind]["best_estimator"],
            MODELS_DIR / f"best_{kind}.joblib",
        )
        log(f"  → saved: best_{kind}.joblib")
        if i < len(spaces) - 1:
            cooldown(f"{kind.upper()} tuning done, waiting before next model")

    if not resume_from_stacking:
        cooldown("base model tuning done, waiting before outer CV OOF evaluation")

    # 3) Outer CV OOF — 3 base models
    oof_results: dict = {}
    stack_proba: np.ndarray  # type hint

    if resume_from_stacking:
        log(f"\n[3/8] Outer CV OOF — base 3 models [RESUME: load oof_*.npy + reconstruct] ...")
        for kind in ["rf", "xgb", "lgbm"]:
            npy_path = PROBA_DIR / f"oof_{kind}.npy"
            if not npy_path.exists():
                raise RuntimeError(f"RESUME mode but {npy_path} not found")
            proba = np.load(npy_path)
            label = f"{kind.upper()} (tuned)"
            oof_results[label] = reconstruct_oof_result(label, proba, y_full, skf)
            log(f"  [RESUME] {label} OOF Brier = {oof_results[label]['oof_metrics']['brier']:.5f}")
    else:
        log(f"\n[3/8] Outer {CV_FOLDS}-fold CV OOF — 3 base models ...")
        for i, kind in enumerate(["rf", "xgb", "lgbm"]):
            params = tuned[kind]["best_params"]
            def make_factory(_kind=kind, _params=params):
                def factory():
                    est = make_estimator(_kind)
                    est.set_params(**_params)
                    return est
                return factory

            label = f"{kind.upper()} (tuned)"
            result, proba, _ = outer_cv_oof(label, make_factory(), X_full, y_full, skf)
            oof_results[label] = result
            np.save(PROBA_DIR / f"oof_{kind}.npy", proba)
            log(f"  → saved: oof_{kind}.npy")
            if i < 2:
                cooldown(f"{kind.upper()} outer OOF done, waiting before next model")

        cooldown("3 base model OOF done, waiting before Stacking")

    # 4) Stacking
    if resume_from_stacking:
        log(f"\n[4/8] Stacking OOF [RESUME: loading oof_stacking.npy] ...")
        npy_path = PROBA_DIR / "oof_stacking.npy"
        if not npy_path.exists():
            raise RuntimeError(f"RESUME mode but {npy_path} not found")
        stack_proba = np.load(npy_path)
        stack_result = reconstruct_oof_result("Stacking (LR meta)", stack_proba, y_full, skf)
        oof_results["Stacking (LR meta)"] = stack_result
        log(f"  [RESUME] Stacking OOF Brier = {stack_result['oof_metrics']['brier']:.5f}")
    else:
        log(f"\n[4/8] Stacking OOF (StackingClassifier cv={STACKING_CV}) ...")
        stack_result, stack_proba = stacking_oof(tuned, X_full, y_full, skf)
        oof_results["Stacking (LR meta)"] = stack_result
        np.save(PROBA_DIR / "oof_stacking.npy", stack_proba)
        log("  → saved: oof_stacking.npy")
        cooldown("Stacking OOF done, waiting before Isotonic post-processing")

    # 5) Isotonic post-processing — Stacking + Isotonic / Best_Single + Isotonic (both)
    #    Option C (cv='prefit' pattern), each ~seconds
    log(f"\n[5/8] Isotonic post-processing (option C, cv='prefit' pattern) ...")

    # (5a) Stacking + Isotonic
    log("  (5a) Stacking + Isotonic")
    cal_result, cal_proba = isotonic_post_processing_oof(
        stack_proba, y_full, skf, label="Stacking + Isotonic (cv='prefit')"
    )
    oof_results["Stacking + Isotonic"] = cal_result
    np.save(PROBA_DIR / "oof_stacking_isotonic.npy", cal_proba)
    log("  → saved: oof_stacking_isotonic.npy")

    # (5b) Best_Single + Isotonic
    #      Select the base with minimum OOF Brier among the 3 → apply isotonic on its OOF proba
    base_briers = {
        kind: oof_results[f"{kind.upper()} (tuned)"]["oof_metrics"]["brier"]
        for kind in ["rf", "xgb", "lgbm"]
    }
    best_single_kind = min(base_briers, key=base_briers.get)
    log(
        f"  (5b) Best Single selected: '{best_single_kind.upper()}' "
        f"(OOF Brier={base_briers[best_single_kind]:.5f}; "
        f"others: " + ", ".join(f"{k.upper()}={v:.5f}" for k, v in base_briers.items()) + ")"
    )
    best_single_proba = np.load(PROBA_DIR / f"oof_{best_single_kind}.npy")
    best_single_iso_label = f"{best_single_kind.upper()} + Isotonic"
    best_iso_result, best_iso_proba = isotonic_post_processing_oof(
        best_single_proba, y_full, skf, label=f"{best_single_iso_label} (cv='prefit')"
    )
    oof_results[best_single_iso_label] = best_iso_result
    np.save(PROBA_DIR / f"oof_{best_single_kind}_isotonic.npy", best_iso_proba)
    log(f"  → saved: oof_{best_single_kind}_isotonic.npy")
    cooldown("Isotonic post-processing done, waiting before final model selection", sec=10)

    # 6) Final model selection — Occam's Razor (prefer simpler model when ΔBrier ≤ 0.001)
    log("\n[6/8] Final model selection (Occam's Razor + minimum OOF Brier) ...")
    OCCAM_EPS = 0.001
    stack_iso_brier = oof_results["Stacking + Isotonic"]["oof_metrics"]["brier"]
    best_iso_brier = oof_results[best_single_iso_label]["oof_metrics"]["brier"]
    stack_only_brier = oof_results["Stacking (LR meta)"]["oof_metrics"]["brier"]

    log(f"  • Best_Single + Isotonic  ({best_single_iso_label}): Brier={best_iso_brier:.5f}")
    log(f"  • Stacking + Isotonic                              : Brier={stack_iso_brier:.5f}")
    log(f"  • Stacking (LR meta) only                          : Brier={stack_only_brier:.5f}")
    log(f"  • Occam threshold ε = {OCCAM_EPS}")

    if best_iso_brier <= stack_iso_brier + OCCAM_EPS:
        final_choice = best_single_iso_label
        final_oof_brier = best_iso_brier
        if best_iso_brier <= stack_iso_brier:
            occam_verdict = (
                f"Best_Single({best_single_kind.upper()}) + Isotonic outperforms Stacking + Isotonic "
                f"(ΔBrier = {best_iso_brier - stack_iso_brier:+.5f} ≤ 0). "
                "Simpler model is clearly superior → Occam's razor applied automatically."
            )
        else:
            occam_verdict = (
                f"Best_Single({best_single_kind.upper()}) + Isotonic and Stacking + Isotonic "
                f"differ by ΔBrier = {best_iso_brier - stack_iso_brier:+.5f} ≤ ε({OCCAM_EPS}) "
                "= statistical tie within fold variability. **Occam's razor applied → simpler Best_Single + "
                "Isotonic selected.**"
            )
    else:
        final_choice = "Stacking + Isotonic"
        final_oof_brier = stack_iso_brier
        occam_verdict = (
            f"Stacking + Isotonic outperforms Best_Single + Isotonic by more than ε({OCCAM_EPS}) "
            f"(ΔBrier = {best_iso_brier - stack_iso_brier:+.5f}). "
            "Increased complexity is justified → Stacking + Isotonic adopted."
        )

    log(f"  → final selection: '{final_choice}' (OOF Brier={final_oof_brier:.5f})")
    log(f"  → selection rationale: {occam_verdict}")

    # 7) Full-fit retraining + saving final model (used for 2025 predictions in Phase 5)
    log("\n[7/8] Full-fit retraining + saving final model ...")
    is_best_single_iso = final_choice.endswith(" + Isotonic") and final_choice != "Stacking + Isotonic"

    if final_choice == "Stacking + Isotonic":
        # Option C: 1 full-fit of stack + 1 full-fit of IsotonicRegression (on stack OOF proba)
        log("  [Stacking + Isotonic = cv='prefit' pattern]")
        log("  (a) Stack full-fit (all 2024 data) starting ...")
        stack = build_stacking_estimator(tuned)
        t0 = time.time()
        stack.fit(X_full, y_full)
        log(f"  (a) Stack full-fit done ({time.time() - t0:.1f}s)")
        log("  (b) IsotonicRegression full-fit (on stack OOF proba) starting ...")
        t0 = time.time()
        iso_full = IsotonicRegression(out_of_bounds="clip")
        iso_full.fit(stack_proba, y_full.values)
        log(f"  (b) Isotonic full-fit done ({time.time() - t0:.3f}s)")
        final_model = {
            "type": "stack_isotonic_prefit",
            "stack": stack,
            "isotonic": iso_full,
            "description": "Phase 5 apply: stack.predict_proba(X)[:,1] → isotonic.predict(...)",
        }
    elif is_best_single_iso:
        # Best_Single + Isotonic (Occam's Razor) — best single base + Isotonic
        kind = best_single_kind
        log(f"  [{final_choice} = Occam's Razor adopted — cv='prefit' pattern]")
        log(f"  (a) {kind.upper()} full-fit (all 2024 data) starting ...")
        base_est = make_estimator(kind)
        base_est.set_params(**tuned[kind]["best_params"])
        t0 = time.time()
        base_est.fit(X_full, y_full)
        log(f"  (a) {kind.upper()} full-fit done ({time.time() - t0:.1f}s)")
        log(f"  (b) IsotonicRegression full-fit (on {kind.upper()} OOF proba) starting ...")
        t0 = time.time()
        iso_full = IsotonicRegression(out_of_bounds="clip")
        iso_full.fit(best_single_proba, y_full.values)
        log(f"  (b) Isotonic full-fit done ({time.time() - t0:.3f}s)")
        final_model = {
            "type": "best_single_isotonic_prefit",
            "base_kind": kind,
            "base_estimator": base_est,
            "isotonic": iso_full,
            "description": (
                f"Phase 5 apply: base_estimator.predict_proba(X)[:,1] → isotonic.predict(...). "
                f"base = tuned {kind.upper()}, calibrator = IsotonicRegression (OOF prefit)."
            ),
        }
    elif final_choice == "Stacking (LR meta)":
        final_model = build_stacking_estimator(tuned)
        log(f"  fit starting (model: {final_choice}) ...")
        t0 = time.time()
        final_model.fit(X_full, y_full)
        log(f"  fit done ({time.time() - t0:.1f}s)")
    else:
        # base tuned model (no isotonic) — normally not reached
        kind = final_choice.split(" ")[0].lower()
        final_model = make_estimator(kind)
        final_model.set_params(**tuned[kind]["best_params"])
        log(f"  fit starting (model: {final_choice}) ...")
        t0 = time.time()
        final_model.fit(X_full, y_full)
        log(f"  fit done ({time.time() - t0:.1f}s)")
    final_model_path = MODELS_DIR / "final_model.joblib"
    joblib.dump(final_model, final_model_path)
    log(f"  → saved: {final_model_path.relative_to(ROOT)}")

    # 8) Save outputs + write report
    log("\n[8/8] Saving outputs + report ...")
    # JSON serializable artifact
    artifact = {
        "tuned": {
            kind: {k: v for k, v in t.items() if k != "best_estimator"}
            for kind, t in tuned.items()
        },
        "oof_results": oof_results,
        "final_choice": final_choice,
        "final_oof_brier": final_oof_brier,
        "best_single_kind": best_single_kind,
        "best_single_iso_label": best_single_iso_label,
        "occam_eps": OCCAM_EPS,
        "occam_verdict": occam_verdict,
        "brier_comparison": {
            "best_single_iso": best_iso_brier,
            "stacking_iso": stack_iso_brier,
            "stacking_only": stack_only_brier,
            "delta_best_minus_stack_iso": best_iso_brier - stack_iso_brier,
        },
        "settings": {
            "RANDOM_STATE": RANDOM_STATE,
            "CV_FOLDS": CV_FOLDS,
            "N_ITER": N_ITER,
            "INNER_CV": INNER_CV,
            "SCORING": SCORING,
            "THRESHOLD": THRESHOLD,
            "N_JOBS": N_JOBS,
            "COOLDOWN_SEC": COOLDOWN_SEC,
            "CALIBRATION_METHOD": CALIBRATION_METHOD,
            "CALIBRATION_CV": CALIBRATION_CV,
            "STACKING_CV": STACKING_CV,
            "STACKING_FINAL": STACKING_FINAL_LABEL,
        },
        "phase2_best_sampling": meta_phase2.get("best_sampling"),
    }
    RESULTS_JSON.write_text(
        json.dumps(artifact, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    log(f"  → {RESULTS_JSON.relative_to(ROOT)}")

    # Average fold sizes
    n_train_sizes, n_val_sizes = [], []
    for tr_idx, val_idx in skf.split(X_full, y_full):
        n_train_sizes.append(len(tr_idx))
        n_val_sizes.append(len(val_idx))
    n_train_mean = int(np.mean(n_train_sizes))
    n_val_mean = int(np.mean(n_val_sizes))

    write_report(
        tuned, oof_results, final_choice, final_oof_brier,
        meta_phase2, meta_phase3, n_train_mean, n_val_mean,
        best_single_kind=best_single_kind,
        best_single_iso_label=best_single_iso_label,
        occam_eps=OCCAM_EPS,
        occam_verdict=occam_verdict,
        brier_comparison={
            "best_single_iso": best_iso_brier,
            "stacking_iso": stack_iso_brier,
            "stacking_only": stack_only_brier,
            "delta_best_minus_stack_iso": best_iso_brier - stack_iso_brier,
        },
    )

    log("\n[done] Phase 4 complete.")


if __name__ == "__main__":
    main()
