"""
Phase 4 Stacking Post-hoc Calibration — Canonical Re-run (Diagnostic Reinforcement)
=====================================================================================

**Problem discovered (step4 first run):**
  The existing `stack.predict_proba(X_train)` (in-sample) → IsotonicRegression fit approach
  creates a mismatch with the test distribution, worsening Brier 0.1345 → 0.1391
  and sharply degrading LogLoss 0.42 → 0.65.

**Root-cause diagnosis:**
  StackingClassifier(cv=5) uses OOF only for training the final_estimator; during the
  refit stage it retrains base estimators on the full X_train. Therefore
  `stack.predict_proba(X_train)` is in-sample from the bases' perspective → overfit.
  Fitting IsotonicRegression on that overfit distribution produces a calibration function
  that is misaligned with the test distribution.

**Canonical approach:**
  CalibratedClassifierCV(stack, method='isotonic', cv=3) — retrain the stacking itself
  via cv-fold OOF → learn Isotonic from accurate OOF predictions → consistent with
  test distribution.

**Cost:** stacking training ×3 ≈ 66 minutes (22 min/run × cv=3).

Outputs:
  - New proba: pipeline/output/phase4_probas/proba_stack_isotonic_OOF.npy
  - Model: pipeline/output/phase4_models/stacking_isotonic_OOF.joblib
  - Existing in-sample diagnostic data preserved as-is
    (proba_stack_calibrated.npy, stacking_isotonic.joblib)
  - phase4_results.json: stacking.isotonic_OOF section added
  - phase4_report.md §6 Stacking section: diagnostic comparison appended

Run:
    PYTHONUNBUFFERED=1 /opt/miniconda3/envs/mlb-xba/bin/python pipeline/step4_stacking_recalib.py \\
        2>&1 | tee pipeline/logs/step4_recalib.log
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
import lightgbm as lgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Paths & decision constants
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
REPORT_PATH = PIPELINE_DIR / "phase4_report.md"
RESULTS_JSON = OUTPUT_DIR / "phase4_results.json"

X_TRAIN_PARQUET = OUTPUT_DIR / "phase2_X_train.parquet"
X_TEST_PARQUET = OUTPUT_DIR / "phase2_X_test.parquet"
Y_TRAIN_PARQUET = OUTPUT_DIR / "phase2_y_train.parquet"
Y_TEST_PARQUET = OUTPUT_DIR / "phase2_y_test.parquet"

MODELS_DIR = OUTPUT_DIR / "phase4_models"
PROBA_DIR = OUTPUT_DIR / "phase4_probas"

RANDOM_STATE = 42
XGB_THRESHOLD = 0.5
N_JOBS = 2
STACKING_CV = 5
RECALIB_CV = 3
CALIBRATION_METHOD = "isotonic"


def log(msg: str) -> None:
    print(msg, flush=True)


def make_pipeline(clf):
    return ImbPipeline([
        ("sampler", RandomUnderSampler(random_state=RANDOM_STATE)),
        ("clf", clf),
    ])


def evaluate(label: str, y_true, proba, threshold: float = XGB_THRESHOLD) -> dict:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    m = {
        "label": label,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-15, 1 - 1e-15))),
        "cm_tn": int(cm[0, 0]), "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]), "cm_tp": int(cm[1, 1]),
    }
    log(
        f"    → F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}  "
        f"Brier={m['brier']:.4f}  LogLoss={m['log_loss']:.4f}  "
        f"P={m['precision']:.4f}  R={m['recall']:.4f}  Acc={m['accuracy']:.4f}"
    )
    return m


def build_stacking_template(tunes_best_params: dict) -> StackingClassifier:
    """Build a fresh Stacking classifier from best_params in phase4_results.json (no training)."""
    estimators = []
    for kind, params in tunes_best_params.items():
        if kind == "rf":
            clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **params)
        elif kind == "xgb":
            clf = xgb.XGBClassifier(
                random_state=RANDOM_STATE, n_jobs=1,
                eval_metric="logloss", tree_method="hist", verbosity=0, **params,
            )
        elif kind == "lgbm":
            clf = lgb.LGBMClassifier(
                random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **params,
            )
        else:
            continue
        estimators.append((kind, make_pipeline(clf)))

    final_est = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE
    )
    return StackingClassifier(
        estimators=estimators, final_estimator=final_est,
        cv=STACKING_CV, n_jobs=1, passthrough=False, stack_method="predict_proba",
    )


# -----------------------------------------------------------------------------
# Report patch — append diagnostic/OOF results to §6 Stacking section
# -----------------------------------------------------------------------------
def patch_report(insample_m: dict, oof_m: dict, raw_m: dict, elapsed_min: float) -> None:
    md = REPORT_PATH.read_text(encoding="utf-8")

    # Additional section text (inserted after §6, before §7 overall comparison)
    extra = []
    extra.append("### 6.2 Stacking Post-hoc Calibration — 진단 보강 (2차 실행)")
    extra.append("")
    extra.append("**1차 실행 결과 (in-sample Isotonic) 의 이상:**")
    extra.append("")
    extra.append(
        f"- Brier: raw {raw_m['brier']:.4f} → in-sample isotonic {insample_m['brier']:.4f} "
        f"({insample_m['brier'] - raw_m['brier']:+.4f}, **악화**)"
    )
    extra.append(
        f"- LogLoss: raw {raw_m['log_loss']:.4f} → in-sample isotonic {insample_m['log_loss']:.4f} "
        f"({insample_m['log_loss'] - raw_m['log_loss']:+.4f}, **급악화**)"
    )
    extra.append("")
    extra.append("**원인 진단:**")
    extra.append("")
    extra.append(
        "`StackingClassifier(cv=5)` 는 `final_estimator` 학습에만 OOF 를 사용하고, refit 단계에서 "
        "base estimators (RF/XGB/LGBM Pipeline) 를 **전체 X_train 으로 재학습** 한다. 따라서 "
        "`stack.predict_proba(X_train)` 은 base 입장에서 in-sample → overfit. 이 overfit 분포에 "
        "IsotonicRegression 을 학습시키면 calibration 함수가 test 분포와 부적합하게 형성되어 "
        "예측 확률 보정이 깨진다 (특히 LogLoss 가 양 극단 예측의 미세한 오류에 매우 민감하므로 급악화)."
    )
    extra.append("")
    extra.append("**정석 재실행: `CalibratedClassifierCV(stack, method='isotonic', cv=3)`**")
    extra.append("")
    extra.append(
        f"- Stacking 자체를 cv=3 fold 로 재학습 → 각 fold 의 OOF predictions 에 Isotonic 학습 "
        f"→ test 와 분포 일관. 비용: stacking 학습 × 3 ≈ {elapsed_min:.1f}분."
    )
    extra.append("")
    extra.append("| 단계 | F1 | AUC | Brier↓ | LogLoss↓ | P | R | Acc |")
    extra.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    extra.append(
        f"| stack_raw (1차) | {raw_m['f1']:.4f} | {raw_m['roc_auc']:.4f} | "
        f"{raw_m['brier']:.4f} | {raw_m['log_loss']:.4f} | "
        f"{raw_m['precision']:.4f} | {raw_m['recall']:.4f} | {raw_m['accuracy']:.4f} |"
    )
    extra.append(
        f"| stack_isotonic_insample (1차, 진단용) | {insample_m['f1']:.4f} | "
        f"{insample_m['roc_auc']:.4f} | "
        f"⚠️ {insample_m['brier']:.4f} | ⚠️ {insample_m['log_loss']:.4f} | "
        f"{insample_m['precision']:.4f} | {insample_m['recall']:.4f} | {insample_m['accuracy']:.4f} |"
    )
    extra.append(
        f"| **stack_isotonic_OOF (2차, 정석)** | **{oof_m['f1']:.4f}** | **{oof_m['roc_auc']:.4f}** | "
        f"**{oof_m['brier']:.4f}** | **{oof_m['log_loss']:.4f}** | "
        f"{oof_m['precision']:.4f} | {oof_m['recall']:.4f} | {oof_m['accuracy']:.4f} |"
    )
    extra.append("")
    delta_brier = oof_m["brier"] - raw_m["brier"]
    delta_logloss = oof_m["log_loss"] - raw_m["log_loss"]
    extra.append(
        f"**해석:** 정석 OOF Isotonic 적용 후 Brier {delta_brier:+.4f} / LogLoss {delta_logloss:+.4f}. "
        "음수면 calibration 이 실제로 효과 있음 (Phase 5 ca-xBA 산출에 권장). "
        "양수면 stacking 자체가 이미 잘 보정되어 추가 calibration 불요."
    )
    extra.append("")

    # Append at end of §6 — insert just before §7
    marker7 = "\n## 7. 종합 비교"
    if marker7 in md:
        idx = md.index(marker7)
        md = md[:idx].rstrip() + "\n\n" + "\n".join(extra) + "\n" + md[idx:]
    else:
        md = md.rstrip() + "\n\n" + "\n".join(extra) + "\n"

    REPORT_PATH.write_text(md, encoding="utf-8")
    log(f"[report] phase4_report.md patched (§6.2 added)")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    log("=" * 80)
    log("Phase 4 Stacking Calibration canonical re-run (CalibratedClassifierCV cv=3)")
    log("=" * 80)
    t_start = time.time()

    # 1) Load data + existing results
    log("\n[1/5] Loading data + existing phase4_results.json ...")
    X_train = pd.read_parquet(X_TRAIN_PARQUET)
    X_test = pd.read_parquet(X_TEST_PARQUET)
    y_train = pd.read_parquet(Y_TRAIN_PARQUET)["is_hit"]
    y_test = pd.read_parquet(Y_TEST_PARQUET)["is_hit"]
    artifact = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    log(f"  X_train {X_train.shape}, X_test {X_test.shape}")

    # 2) Reconstruct Stacking template (using first-run best_params)
    log("\n[2/5] Reconstructing Stacking template (first-run best_params) ...")
    tunes_best = {k: v["best_params"] for k, v in artifact["tunes"].items()}
    for kind, p in tunes_best.items():
        log(f"  {kind.upper()} best_params: {p}")
    stack_template = build_stacking_template(tunes_best)

    # 3) Canonical calibration via CalibratedClassifierCV(cv=3)
    log(f"\n[3/5] CalibratedClassifierCV(stack, method='{CALIBRATION_METHOD}', cv={RECALIB_CV}) ...")
    log(f"  ⚠️ Cost: stacking training × {RECALIB_CV} ≈ {22*RECALIB_CV} min estimated (M2 Air)")
    t0 = time.time()
    cal = CalibratedClassifierCV(
        estimator=stack_template, method=CALIBRATION_METHOD,
        cv=RECALIB_CV, n_jobs=N_JOBS,
    )
    cal.fit(X_train, y_train)
    elapsed = (time.time() - t0) / 60
    log(f"  Training done ({elapsed:.1f} min).")

    log("\n[4/5] Test evaluation + comparison with first-run results ...")
    proba_oof = cal.predict_proba(X_test)[:, 1]
    m_oof = evaluate("stack_isotonic_OOF", y_test, proba_oof)

    # Existing first-run results
    m_raw = artifact["stacking"]["metrics_raw"]
    m_insample = artifact["stacking"]["metrics_calibrated"]

    log("\n  ▸ Comparison with first-run (in-sample isotonic):")
    log(f"    raw                       : Brier={m_raw['brier']:.4f}  LogLoss={m_raw['log_loss']:.4f}")
    log(f"    isotonic (in-sample, 1차) : Brier={m_insample['brier']:.4f}  LogLoss={m_insample['log_loss']:.4f}  ⚠️")
    log(f"    isotonic (OOF, 2차 정석)   : Brier={m_oof['brier']:.4f}  LogLoss={m_oof['log_loss']:.4f}  ✓")

    # 5) Save outputs + patch report
    log("\n[5/5] Saving outputs + patching report ...")
    np.save(PROBA_DIR / "proba_stack_isotonic_OOF.npy", proba_oof)
    joblib.dump(cal, MODELS_DIR / "stacking_isotonic_OOF.joblib")

    # Update results.json
    artifact["stacking"]["metrics_calibrated_insample"] = m_insample  # preserve first-run result
    artifact["stacking"]["metrics_calibrated_OOF"] = m_oof             # new canonical result
    artifact["stacking"]["calibration_recalib_method"] = (
        f"CalibratedClassifierCV(stack, method='{CALIBRATION_METHOD}', cv={RECALIB_CV})"
    )
    artifact["stacking"]["calibration_recalib_seconds"] = round((time.time() - t0), 1)
    RESULTS_JSON.write_text(json.dumps(artifact, indent=2, default=str, ensure_ascii=False),
                              encoding="utf-8")
    log(f"  → phase4_results.json updated (stacking.metrics_calibrated_OOF added)")

    patch_report(m_insample, m_oof, m_raw, elapsed)

    total = (time.time() - t_start) / 60
    log(f"\n[done] Stacking canonical re-run complete (total {total:.1f} min).")


if __name__ == "__main__":
    main()
