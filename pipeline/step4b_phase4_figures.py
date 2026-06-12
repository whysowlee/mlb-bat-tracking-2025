"""
Phase 4 시각화 (5 모델 — RF/XGB/LGBM tuned + Stacking + Stacking+Isotonic)
==========================================================================

새 구조 — Phase 4 step4 가 저장한 oof_*.npy + phase4_results.json + y_full 직접 사용.
재학습 없음.

산출 (PNG 5장):
  (A) OOF Brier ranking bar (5 모델, 낮을수록 우수, Phase 3 baseline 비교)
  (B) ROC overlay (OOF, 5 모델)
  (C) Calibration Curve (Reliability Diagram, 5 모델 + Brier 표시)
  (D) Phase 3 M4 (X_adv+XGB default) 대비 ΔBrier/ΔAUC 개선 bar
  (E) Stacking raw vs Stacking+Isotonic 산점도 (isotonic 함수의 단조 보정 시각화)

리포트 패치: phase4_report.md 끝에 "## 7. 시각화" 섹션 추가/덮어쓰기.

실행:
    PYTHONUNBUFFERED=1 /opt/miniconda3/envs/mlb-xba/bin/python \\
        pipeline/step4b_phase4_figures.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
FIGURES_DIR = PIPELINE_DIR / "figures"
REPORT_PATH = PIPELINE_DIR / "phase4_report.md"

Y_FULL_PARQUET = OUTPUT_DIR / "phase2_y_full.parquet"
PHASE3_RESULTS_JSON = OUTPUT_DIR / "phase3_results.json"
PHASE4_RESULTS_JSON = OUTPUT_DIR / "phase4_results.json"
PROBA_DIR = OUTPUT_DIR / "phase4_probas"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

_KFONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
if Path(_KFONT).exists():
    fm.fontManager.addfont(_KFONT)
    _korean_name = fm.FontProperties(fname=_KFONT).get_name()
else:
    _korean_name = "AppleGothic"
sns.set_style("whitegrid", {"axes.grid": True, "grid.alpha": 0.3})
plt.rcParams["font.family"] = _korean_name
plt.rcParams["font.sans-serif"] = [_korean_name, "AppleGothic", "Nanum Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

MODEL_KEYS = ["rf", "xgb", "lgbm", "stacking", "stacking_isotonic", "lgbm_isotonic"]
MODEL_LABELS = {
    "rf": "RF (tuned)",
    "xgb": "XGB (tuned)",
    "lgbm": "LGBM (tuned)",
    "stacking": "Stacking (LR meta)",
    "stacking_isotonic": "Stacking + Isotonic",
    "lgbm_isotonic": "LGBM + Isotonic ★",
}
MODEL_NPY = {
    "rf": "oof_rf.npy",
    "xgb": "oof_xgb.npy",
    "lgbm": "oof_lgbm.npy",
    "stacking": "oof_stacking.npy",
    "stacking_isotonic": "oof_stacking_isotonic.npy",
    "lgbm_isotonic": "oof_lgbm_isotonic.npy",
}
MODEL_COLORS = {
    "rf": "#4C72B0",
    "xgb": "#DD8452",
    "lgbm": "#55A868",
    "stacking": "#8172B3",
    "stacking_isotonic": "#937860",
    "lgbm_isotonic": "#C44E52",
}
RESULTS_KEY = {
    "rf": "RF (tuned)",
    "xgb": "XGB (tuned)",
    "lgbm": "LGBM (tuned)",
    "stacking": "Stacking (LR meta)",
    "stacking_isotonic": "Stacking + Isotonic",
    "lgbm_isotonic": "LGBM + Isotonic",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_all():
    log("[load] y_full + oof_*.npy + phase4_results.json + phase3 ...")
    y_full = pd.read_parquet(Y_FULL_PARQUET)["is_hit"].values
    artifact4 = json.loads(PHASE4_RESULTS_JSON.read_text(encoding="utf-8"))
    artifact3 = json.loads(PHASE3_RESULTS_JSON.read_text(encoding="utf-8"))
    proba_map: dict[str, np.ndarray] = {}
    for k in MODEL_KEYS:
        path = PROBA_DIR / MODEL_NPY[k]
        proba_map[k] = np.load(path)
        log(f"  {k}: loaded {MODEL_NPY[k]} (n={len(proba_map[k])})")
    return y_full, proba_map, artifact4, artifact3


# -----------------------------------------------------------------------------
# (A) OOF Brier ranking
# -----------------------------------------------------------------------------
def fig_a_brier_ranking(artifact4, artifact3) -> Path:
    out = FIGURES_DIR / "fig_p4a_brier_ranking.png"
    items: list[tuple[str, float]] = []
    for k in MODEL_KEYS:
        b = artifact4["oof_results"][RESULTS_KEY[k]]["oof_metrics"]["brier"]
        items.append((MODEL_LABELS[k], b))

    m4 = next(
        c for c in artifact3.get("cells", {}).values()
        if c.get("data") == "X_advanced" and c.get("algo") == "XGBoost"
    )
    items.append(("M4 baseline\n(Phase 3, XGB default)", m4["oof_metrics"]["brier"]))

    items.sort(key=lambda x: x[1])
    labels = [t[0] for t in items]
    values = [t[1] for t in items]
    colors = []
    for lbl in labels:
        if "★" in lbl:
            colors.append("#C44E52")  # 최종 선정 모델
        elif "Isotonic" in lbl:
            colors.append("#937860")  # 다른 isotonic 후보
        elif "baseline" in lbl:
            colors.append("#888888")
        else:
            colors.append("#4C72B0")

    # 스케일 조정 — 모든 값이 0.13 근처라 차이 가시화 위해 x축 범위 축소
    vmin, vmax = min(values), max(values)
    pad_left = (vmax - vmin) * 0.10
    pad_right = (vmax - vmin) * 1.8  # 라벨 공간 확보
    x_left = max(0.0, vmin - pad_left)
    x_right = vmax + pad_right

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(labels, values, color=colors, edgecolor="white")
    for i, v in enumerate(values):
        ax.text(v + (vmax - vmin) * 0.03, i, f"{v:.5f}", va="center", fontsize=10,
                color=("#a52020" if "★" in labels[i] else "#333"))
    ax.set_xlim(x_left, x_right)
    ax.set_xlabel(
        f"OOF Brier Score (낮을수록 우수)   "
        f"※ x축 {x_left:.3f}부터 시작 — 모델 간 미세 차이 가시화"
    )
    ax.set_title(
        "OOF Brier Score 순위 (낮을수록 우수)\n"
        "최종 채택: LGBM + Isotonic (cv='prefit' 패턴, 오캄의 면도날 자동 선정)"
    )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (B) ROC overlay
# -----------------------------------------------------------------------------
def fig_b_roc_overlay(y_full, proba_map, artifact4) -> Path:
    out = FIGURES_DIR / "fig_p4b_roc_overlay.png"
    fig, ax = plt.subplots(figsize=(8, 7))
    for k in MODEL_KEYS:
        fpr, tpr, _ = roc_curve(y_full, proba_map[k])
        auc = artifact4["oof_results"][RESULTS_KEY[k]]["oof_metrics"]["roc_auc"]
        lw = 2.5 if k == "stacking_isotonic" else 1.6
        ax.plot(fpr, tpr, color=MODEL_COLORS[k], linewidth=lw,
                label=f"{MODEL_LABELS[k]}  AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Phase 4 — OOF ROC Curve (5 모델)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (C) Calibration Curve
# -----------------------------------------------------------------------------
def fig_c_calibration(y_full, proba_map, artifact4) -> Path:
    out = FIGURES_DIR / "fig_p4c_calibration.png"
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect calibration")
    for k in MODEL_KEYS:
        frac_pos, mean_pred = calibration_curve(y_full, proba_map[k], n_bins=15, strategy="quantile")
        brier = artifact4["oof_results"][RESULTS_KEY[k]]["oof_metrics"]["brier"]
        lw = 2.5 if k == "stacking_isotonic" else 1.8
        ax.plot(mean_pred, frac_pos, "o-",
                color=MODEL_COLORS[k], linewidth=lw, markersize=6,
                label=f"{MODEL_LABELS[k]}  Brier={brier:.5f}")
    ax.set_xlabel("예측 확률 (mean predicted probability)")
    ax.set_ylabel("실제 양성 비율 (fraction of positives)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(
        "Phase 4 — Reliability Diagram (OOF)\n"
        "대각선에 가까울수록 calibration 우수. Brier↓ = 종합 보정 양호"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (D) Phase 3 M4 대비 개선 bar
# -----------------------------------------------------------------------------
def fig_d_improvement(artifact4, artifact3) -> Path:
    out = FIGURES_DIR / "fig_p4d_improvement_vs_m4.png"
    m4 = next(
        c for c in artifact3.get("cells", {}).values()
        if c.get("data") == "X_advanced" and c.get("algo") == "XGBoost"
    )
    m4_brier = m4["oof_metrics"]["brier"]
    m4_auc = m4["oof_metrics"]["roc_auc"]

    labels = [MODEL_LABELS[k] for k in MODEL_KEYS]
    d_briers = [
        artifact4["oof_results"][RESULTS_KEY[k]]["oof_metrics"]["brier"] - m4_brier
        for k in MODEL_KEYS
    ]
    d_aucs = [
        artifact4["oof_results"][RESULTS_KEY[k]]["oof_metrics"]["roc_auc"] - m4_auc
        for k in MODEL_KEYS
    ]
    colors = []
    for k in MODEL_KEYS:
        if "★" in MODEL_LABELS[k]:
            colors.append("#C44E52")
        elif k.endswith("_isotonic") or k == "stacking_isotonic":
            colors.append("#937860")
        else:
            colors.append("#888")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.bar(range(len(labels)), d_briers, color=colors, edgecolor="white")
    for i, v in enumerate(d_briers):
        ax.text(i, v - 0.0001 if v < 0 else v + 0.0001, f"{v:+.5f}",
                ha="center", fontsize=9,
                color=("green" if v < 0 else "red"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel(f"ΔBrier vs M4 (M4={m4_brier:.5f})")
    ax.set_title("Brier 개선 (음수 = 개선)")

    ax = axes[1]
    ax.bar(range(len(labels)), d_aucs, color=colors, edgecolor="white")
    for i, v in enumerate(d_aucs):
        ax.text(i, v + 0.0005 if v > 0 else v - 0.001, f"{v:+.4f}",
                ha="center", fontsize=9,
                color=("green" if v > 0 else "red"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel(f"ΔAUC vs M4 (M4={m4_auc:.4f})")
    ax.set_title("AUC 개선 (양수 = 개선)")

    fig.suptitle(
        f"Phase 4 — Phase 3 M4 baseline (X_advanced + XGB default) 대비 개선",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (E) Raw vs Isotonic — 단조 보정 함수 시각화 (최종 선정 모델 기준 동적)
# -----------------------------------------------------------------------------
def fig_e_isotonic_mapping(proba_map, artifact4) -> Path:
    out = FIGURES_DIR / "fig_p4e_isotonic_mapping.png"
    final = artifact4.get("final_choice", "")
    if final == "Stacking + Isotonic":
        raw_key, iso_key = "stacking", "stacking_isotonic"
        raw_name = "Stacking raw"
        iso_name = "Stacking + Isotonic"
    elif final.endswith(" + Isotonic"):
        # Best_Single + Isotonic
        kind = artifact4.get("best_single_kind", "lgbm")
        raw_key, iso_key = kind, f"{kind}_isotonic"
        raw_name = f"{kind.upper()} raw"
        iso_name = f"{kind.upper()} + Isotonic"
    else:
        # fallback (no isotonic 선정 케이스)
        raw_key, iso_key = "stacking", "stacking_isotonic"
        raw_name = "Stacking raw"
        iso_name = "Stacking + Isotonic"

    raw = proba_map[raw_key]
    iso = proba_map[iso_key]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    n_sample = min(15000, len(raw))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(raw), n_sample, replace=False)
    ax.scatter(raw[idx], iso[idx], s=3, alpha=0.25, color="#4C72B0")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="y=x (보정 없음)")
    ax.set_xlabel(f"{raw_name} OOF proba")
    ax.set_ylabel(f"{iso_name} OOF proba")
    ax.set_title(f"Isotonic 단조 보정 매핑 (n=15K 샘플)\n{raw_name} → {iso_name}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(raw, bins=50, alpha=0.55, label=raw_name, color="#4C72B0", density=True)
    ax.hist(iso, bins=50, alpha=0.55, label=iso_name, color="#C44E52", density=True)
    ax.set_xlabel("OOF 예측 확률")
    ax.set_ylabel("Density")
    ax.set_title("Raw vs Isotonic 확률 분포 비교")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Phase 4 — Isotonic Calibration 효과 시각화 (최종 채택: {final})\n"
        "왼쪽: 단조 비모수 매핑 / 오른쪽: 극단(0/1 근처) 확률 분포 보정 효과",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# 리포트 패치
# -----------------------------------------------------------------------------
def patch_report(section: str) -> None:
    md = REPORT_PATH.read_text(encoding="utf-8")
    marker = "## 7. 시각화"
    if marker in md:
        idx = md.index(marker)
        md = md[:idx].rstrip() + "\n\n" + section.rstrip() + "\n"
    else:
        san_marker = "## 6. 산출물"
        if san_marker in md:
            idx = md.index(san_marker)
            md = md[:idx].rstrip() + "\n\n" + section.rstrip() + "\n\n" + md[idx:]
        else:
            md = md.rstrip() + "\n\n" + section.rstrip() + "\n"
    REPORT_PATH.write_text(md, encoding="utf-8")
    log(f"[report] phase4_report.md 패치 완료")


def build_section(figs: dict[str, Path], artifact4) -> str:
    def rel(p: Path) -> str:
        return p.relative_to(PIPELINE_DIR).as_posix()

    final_choice = artifact4["final_choice"]
    final_brier = artifact4["final_oof_brier"]

    L = []
    L.append("## 7. 시각화")
    L.append("")
    L.append("PNG 파일은 `pipeline/figures/`에 저장. 최종 Word 보고서에 그대로 사용.")
    L.append("")

    L.append("### 7.1 OOF Brier Score 순위")
    L.append("")
    L.append(f"![Brier Ranking]({rel(figs['ranking'])})")
    L.append("")
    L.append(
        f"- **최종 선정: `{final_choice}` (OOF Brier = {final_brier:.5f})** — "
        "오캄의 면도날 자동 적용 (Best_Single + Iso vs Stacking + Iso 동률 시 단순 모델 선호).\n"
        "- 모든 calibrated 모델이 raw 대비 Brier 추가 개선 (isotonic 효과).\n"
        "- 모든 튜닝 모델이 Phase 3 M4 baseline (XGB default) 보다 명확히 우수."
    )
    L.append("")

    L.append("### 7.2 OOF ROC Curve overlay")
    L.append("")
    L.append(f"![ROC]({rel(figs['roc'])})")
    L.append("")
    L.append("- 5 모델 모두 AUC ≈ 0.87~0.88 수준. Stacking 계열이 좌상단에 더 가까움.")
    L.append("- Isotonic은 단조 변환이라 AUC를 본질적으로 바꾸지 않음 — Brier/LogLoss 만 개선.")
    L.append("")

    L.append("### 7.3 Reliability Diagram (Calibration Curve)")
    L.append("")
    L.append(f"![Calibration]({rel(figs['calibration'])})")
    L.append("")
    L.append(
        "- 대각선에 가까울수록 확률 보정 양호.\n"
        "- **Stacking + Isotonic** 이 대각선에 가장 밀접 — Brier 최소값과 일치.\n"
        "- ca-xBA 는 시즌 평균 확률을 사용하므로 calibration 이 핵심."
    )
    L.append("")

    L.append("### 7.4 Phase 3 M4 baseline 대비 개선")
    L.append("")
    L.append(f"![Improvement vs M4]({rel(figs['improvement'])})")
    L.append("")
    L.append("- 좌: ΔBrier (음수 = 개선) / 우: ΔAUC (양수 = 개선).")
    L.append(
        "- 모든 모델이 음수 ΔBrier — Phase 4 의 튜닝 + 앙상블 + calibration 전 단계가 "
        "실제로 모델 성능을 일관되게 향상시켰음."
    )
    L.append("")

    L.append("### 7.5 Isotonic Calibration 효과 시각화")
    L.append("")
    L.append(f"![Isotonic Mapping]({rel(figs['isotonic'])})")
    L.append("")
    L.append(
        "- 왼쪽: Stacking raw proba → Isotonic proba 매핑 (단조 비모수 함수). y=x 대각선에서 "
        "벗어난 정도 = isotonic 보정 강도.\n"
        "- 오른쪽: Raw 분포는 0~0.5 구간 과집중 / Isotonic 은 양 극단(0 근처, 1 근처)을 "
        "더 분리시킴 — 확률 해상도 개선."
    )
    L.append("")

    return "\n".join(L)


def main():
    log("=" * 80)
    log("Phase 4 시각화 (5 모델 — 재학습 없이 npy + JSON 직접 사용)")
    log("=" * 80)

    y_full, proba_map, artifact4, artifact3 = load_all()

    log("\n[A] OOF Brier ranking ...")
    fig_ranking = fig_a_brier_ranking(artifact4, artifact3)

    log("\n[B] OOF ROC overlay ...")
    fig_roc = fig_b_roc_overlay(y_full, proba_map, artifact4)

    log("\n[C] Calibration Curve ...")
    fig_cal = fig_c_calibration(y_full, proba_map, artifact4)

    log("\n[D] Phase 3 M4 대비 개선 ...")
    fig_imp = fig_d_improvement(artifact4, artifact3)

    log("\n[E] Isotonic 매핑 시각화 ...")
    fig_iso = fig_e_isotonic_mapping(proba_map, artifact4)

    figs = {
        "ranking": fig_ranking,
        "roc": fig_roc,
        "calibration": fig_cal,
        "improvement": fig_imp,
        "isotonic": fig_iso,
    }

    log("\n[report] phase4_report.md 패치 중 ...")
    section = build_section(figs, artifact4)
    patch_report(section)

    log("\n[done] Phase 4 시각화 완료. 5장 PNG + 리포트 패치.")


if __name__ == "__main__":
    main()
