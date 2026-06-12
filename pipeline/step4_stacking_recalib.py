"""
Phase 4 Stacking Post-hoc Calibration — 정석 재실행 (진단 보강)
================================================================

**문제 발견 (step4 1차 실행):**
  기존 `stack.predict_proba(X_train)` (in-sample) → IsotonicRegression fit 방식이
  test 분포와 mismatch 를 만들어 Brier 0.1345 → 0.1391 악화, LogLoss 0.42 → 0.65 급악화.

**원인 진단:**
  StackingClassifier(cv=5) 는 final_estimator 학습에만 OOF 를 사용하고 refit 단계에서
  base estimators 를 전체 X_train 으로 재학습한다. 따라서 `stack.predict_proba(X_train)` 은
  base 입장에서 in-sample → overfit. 그 overfit 분포에 Isotonic 을 학습시키면 보정 함수가
  test 분포에 부적합.

**정석 접근:**
  CalibratedClassifierCV(stack, method='isotonic', cv=3) — stacking 자체를 cv-fold OOF
  로 재학습 → 정확한 OOF predictions 에서 Isotonic 학습 → test 분포와 일관.

**비용:** stacking 한 번 학습 22분 × cv=3 = 약 66분.

산출:
  - 새 proba: pipeline/output/phase4_probas/proba_stack_isotonic_OOF.npy
  - 모델: pipeline/output/phase4_models/stacking_isotonic_OOF.joblib
  - 기존 in-sample 진단용 데이터는 그대로 보존 (proba_stack_calibrated.npy, stacking_isotonic.joblib)
  - phase4_results.json 에 stacking.isotonic_OOF 섹션 추가
  - phase4_report.md 의 §6 Stacking 섹션에 진단·비교 추가

실행:
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
# 경로 & 결정 상수
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
    """기존 phase4_results.json 의 best_params 로 Stacking 새로 구성 (학습 X)."""
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
# 리포트 패치 — § 6 Stacking 섹션에 진단·OOF 결과 추가
# -----------------------------------------------------------------------------
def patch_report(insample_m: dict, oof_m: dict, raw_m: dict, elapsed_min: float) -> None:
    md = REPORT_PATH.read_text(encoding="utf-8")

    # 추가 섹션 텍스트 (§ 6 뒤, § 7 종합 비교 앞에 삽입)
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

    # § 6 끝에 추가 — § 7 직전에 삽입
    marker7 = "\n## 7. 종합 비교"
    if marker7 in md:
        idx = md.index(marker7)
        md = md[:idx].rstrip() + "\n\n" + "\n".join(extra) + "\n" + md[idx:]
    else:
        md = md.rstrip() + "\n\n" + "\n".join(extra) + "\n"

    REPORT_PATH.write_text(md, encoding="utf-8")
    log(f"[report] phase4_report.md 패치 완료 (§ 6.2 추가)")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    log("=" * 80)
    log("Phase 4 Stacking Calibration 정석 재실행 (CalibratedClassifierCV cv=3)")
    log("=" * 80)
    t_start = time.time()

    # 1) 데이터 + 기존 결과 로드
    log("\n[1/5] 데이터 + 기존 phase4_results.json 로드 ...")
    X_train = pd.read_parquet(X_TRAIN_PARQUET)
    X_test = pd.read_parquet(X_TEST_PARQUET)
    y_train = pd.read_parquet(Y_TRAIN_PARQUET)["is_hit"]
    y_test = pd.read_parquet(Y_TEST_PARQUET)["is_hit"]
    artifact = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    log(f"  X_train {X_train.shape}, X_test {X_test.shape}")

    # 2) Stacking template 재구성 (1차 best_params 사용)
    log("\n[2/5] Stacking template 재구성 (1차 best_params) ...")
    tunes_best = {k: v["best_params"] for k, v in artifact["tunes"].items()}
    for kind, p in tunes_best.items():
        log(f"  {kind.upper()} best_params: {p}")
    stack_template = build_stacking_template(tunes_best)

    # 3) CalibratedClassifierCV(cv=3) 으로 정석 calibration
    log(f"\n[3/5] CalibratedClassifierCV(stack, method='{CALIBRATION_METHOD}', cv={RECALIB_CV}) ...")
    log(f"  ⚠️ 비용: stacking 학습 × {RECALIB_CV} ≈ {22*RECALIB_CV}분 예상 (M2 Air 기준)")
    t0 = time.time()
    cal = CalibratedClassifierCV(
        estimator=stack_template, method=CALIBRATION_METHOD,
        cv=RECALIB_CV, n_jobs=N_JOBS,
    )
    cal.fit(X_train, y_train)
    elapsed = (time.time() - t0) / 60
    log(f"  학습 완료 ({elapsed:.1f}분).")

    log("\n[4/5] Test 평가 + 1차 결과와 비교 ...")
    proba_oof = cal.predict_proba(X_test)[:, 1]
    m_oof = evaluate("stack_isotonic_OOF", y_test, proba_oof)

    # 기존 결과
    m_raw = artifact["stacking"]["metrics_raw"]
    m_insample = artifact["stacking"]["metrics_calibrated"]

    log("\n  ▸ 1차 (in-sample isotonic) 와 비교:")
    log(f"    raw                       : Brier={m_raw['brier']:.4f}  LogLoss={m_raw['log_loss']:.4f}")
    log(f"    isotonic (in-sample, 1차) : Brier={m_insample['brier']:.4f}  LogLoss={m_insample['log_loss']:.4f}  ⚠️")
    log(f"    isotonic (OOF, 2차 정석)   : Brier={m_oof['brier']:.4f}  LogLoss={m_oof['log_loss']:.4f}  ✓")

    # 5) 산출물 저장 + 리포트 패치
    log("\n[5/5] 산출물 저장 + 리포트 패치 ...")
    np.save(PROBA_DIR / "proba_stack_isotonic_OOF.npy", proba_oof)
    joblib.dump(cal, MODELS_DIR / "stacking_isotonic_OOF.joblib")

    # results.json 업데이트
    artifact["stacking"]["metrics_calibrated_insample"] = m_insample  # 기존 1차 결과 보존
    artifact["stacking"]["metrics_calibrated_OOF"] = m_oof             # 신규 정석 결과
    artifact["stacking"]["calibration_recalib_method"] = (
        f"CalibratedClassifierCV(stack, method='{CALIBRATION_METHOD}', cv={RECALIB_CV})"
    )
    artifact["stacking"]["calibration_recalib_seconds"] = round((time.time() - t0), 1)
    RESULTS_JSON.write_text(json.dumps(artifact, indent=2, default=str, ensure_ascii=False),
                              encoding="utf-8")
    log(f"  → phase4_results.json 업데이트 (stacking.metrics_calibrated_OOF 추가)")

    patch_report(m_insample, m_oof, m_raw, elapsed)

    total = (time.time() - t_start) / 60
    log(f"\n[done] Stacking 정석 재실행 완료 (총 {total:.1f}분).")


if __name__ == "__main__":
    main()
