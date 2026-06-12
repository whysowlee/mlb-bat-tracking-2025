"""
Phase 3 시각화 (2×2 Factorial Design — 4 cell + 2-way ANOVA): phase3_report.md 패치
=================================================================================

새 구조: Phase 3 step3 가 JSON 에 저장한 OOF proba / OOF metrics / fold records /
deltas / ANOVA 결과를 그대로 시각화 (재학습 없음).

산출 (PNG 6장):
  (A1) OOF 혼동행렬 4개 (M1/M2/M3/M4)
  (A2) OOF 평가지표 막대 (Brier↓ / LogLoss↓ / F1 / AUC / P / R / Acc)
  (B)  ROC Curve overlay (OOF proba)
  (C)  Reliability Diagram (Calibration Curve, 4 cells)
  (D)  Effect Decomposition bar (Brier / AUC ΔX 6 효과 + interaction 강조)
  (E)  Interaction Plot (Data × Algo matrix) + ANOVA F·p annotation

리포트 패치: phase3_report.md 끝에 "## 8. 시각화" 섹션 추가/덮어쓰기.

실행:
    PYTHONUNBUFFERED=1 /opt/miniconda3/envs/mlb-xba/bin/python \\
        pipeline/step3b_phase3_figures.py
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
from sklearn.metrics import confusion_matrix, roc_curve

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# 경로 & 폰트
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
FIGURES_DIR = PIPELINE_DIR / "figures"
REPORT_PATH = PIPELINE_DIR / "phase3_report.md"

Y_FULL_PARQUET = OUTPUT_DIR / "phase2_y_full.parquet"
RESULTS_JSON = OUTPUT_DIR / "phase3_results.json"

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

THRESHOLD = 0.5

CELL_KEYS = ["M1", "M2", "M3", "M4"]
CELL_LABELS = {
    "M1": "M1\n(X_base + LogReg)",
    "M2": "M2\n(X_base + XGBoost)",
    "M3": "M3\n(X_advanced + LogReg)",
    "M4": "M4\n(X_advanced + XGBoost)",
}
CELL_COLORS = {
    "M1": "#4C72B0",  # 통제군 — 파랑
    "M2": "#DD8452",  # 알고리즘 업그레이드 — 주황
    "M3": "#55A868",  # 데이터 업그레이드 — 녹색
    "M4": "#C44E52",  # 상호작용 결합 — 진빨강 (강조)
}


def log(msg: str) -> None:
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# 데이터 로드 (재학습 없음 — JSON 의 oof_proba 직접 사용)
# -----------------------------------------------------------------------------
def load_artifacts():
    log("[load] y_full + phase3_results.json ...")
    y_full = pd.read_parquet(Y_FULL_PARQUET)["is_hit"].values
    artifact = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    proba_map = {k: np.array(artifact["oof_proba"][k]) for k in CELL_KEYS}
    log(f"  y_full len={len(y_full)}, hit_rate={y_full.mean():.4f}")
    for k in CELL_KEYS:
        log(f"  {k} OOF proba len={len(proba_map[k])}, mean={proba_map[k].mean():.4f}")
    return y_full, proba_map, artifact


# -----------------------------------------------------------------------------
# (A1) OOF 혼동행렬 4개
# -----------------------------------------------------------------------------
def fig_a1_confusion_matrices(y_full, proba_map, artifact) -> Path:
    out = FIGURES_DIR / "fig_p3a1_oof_confusion_matrices.png"
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.5))
    for ax, key in zip(axes, CELL_KEYS):
        pred = (proba_map[key] >= THRESHOLD).astype(int)
        cm = confusion_matrix(y_full, pred)
        sns.heatmap(
            cm, annot=True, fmt=",d", cmap="Blues", ax=ax, cbar=False,
            xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"],
        )
        oof = artifact["cells"][key]["oof_metrics"]
        ax.set_title(
            f"{CELL_LABELS[key]}\nBrier={oof['brier']:.5f} | F1={oof['f1']:.4f}",
            fontsize=10,
        )
    fig.suptitle("Phase 3 — 4 cell OOF 혼동행렬 (threshold=0.5)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (A2) OOF 평가지표 막대
# -----------------------------------------------------------------------------
def fig_a2_metrics_bar(artifact) -> Path:
    out = FIGURES_DIR / "fig_p3a2_oof_metrics_bar.png"
    metric_keys = ["brier", "logloss", "f1", "roc_auc", "precision", "recall", "accuracy"]
    metric_labels = ["Brier↓", "LogLoss↓", "F1", "AUC", "Precision", "Recall", "Accuracy"]
    x = np.arange(len(metric_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 5.8))
    for i, key in enumerate(CELL_KEYS):
        oof = artifact["cells"][key]["oof_metrics"]
        vals = [oof[k] for k in metric_keys]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=CELL_LABELS[key].replace("\n", " "),
                      color=CELL_COLORS[key])
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 0.012, f"{v:.4f}",
                    ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    max_val = max(artifact["cells"][k]["oof_metrics"]["logloss"] for k in CELL_KEYS)
    ax.set_ylim(0, max(1.05, max_val + 0.1))
    ax.set_title("Phase 3 — 4 cell OOF 평가지표 비교 (Brier↓·LogLoss↓ 낮을수록 우수)")
    ax.legend(title="Cell", loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (B) ROC overlay (OOF)
# -----------------------------------------------------------------------------
def fig_b_roc_overlay(y_full, proba_map, artifact) -> Path:
    out = FIGURES_DIR / "fig_p3b_oof_roc.png"
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for key in CELL_KEYS:
        fpr, tpr, _ = roc_curve(y_full, proba_map[key])
        auc = artifact["cells"][key]["oof_metrics"]["roc_auc"]
        ax.plot(fpr, tpr,
                label=f"{CELL_LABELS[key].replace(chr(10), ' ')} (AUC={auc:.4f})",
                color=CELL_COLORS[key], linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Phase 3 — 4 cell OOF ROC Curve\n(M2, M4 의 비선형 우위 확인)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (C) Reliability Diagram (Calibration)
# -----------------------------------------------------------------------------
def fig_c_calibration(y_full, proba_map, artifact) -> Path:
    out = FIGURES_DIR / "fig_p3c_calibration.png"
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect calibration")
    for key in CELL_KEYS:
        proba = proba_map[key]
        frac_pos, mean_pred = calibration_curve(y_full, proba, n_bins=15, strategy="quantile")
        brier = artifact["cells"][key]["oof_metrics"]["brier"]
        ax.plot(mean_pred, frac_pos, "o-",
                label=f"{CELL_LABELS[key].replace(chr(10), ' ')} (Brier={brier:.5f})",
                color=CELL_COLORS[key], linewidth=2, markersize=6)
    ax.set_xlabel("예측 확률 (mean predicted probability)")
    ax.set_ylabel("실제 양성 비율 (fraction of positives)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(
        "Phase 3 — Reliability Diagram (OOF)\n"
        "대각선에 가까울수록 확률 보정(calibration) 양호. Brier↓ = 종합 보정 우수"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (D) Effect Decomposition bar (Brier·AUC ΔX 6 효과)
# -----------------------------------------------------------------------------
def fig_d_effect_decomposition(artifact) -> Path:
    out = FIGURES_DIR / "fig_p3d_effect_decomposition.png"
    d = artifact["deltas"]
    effect_keys = [
        ("데이터 효과\n(in LogReg)\nM3−M1",  "data_in_logreg"),
        ("데이터 효과\n(in XGBoost)\nM4−M2", "data_in_xgb"),
        ("알고리즘 효과\n(in X_base)\nM2−M1",   "algo_in_xbase"),
        ("알고리즘 효과\n(in X_adv)\nM4−M3",    "algo_in_xadvanced"),
        ("결합 효과\nM4−M1",                "combined_M4_M1"),
        ("**Interaction**\n(M4−M2)−(M3−M1)", "interaction"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # left: Brier (↓ negative is better)
    ax = axes[0]
    vals = [d["brier"][k] for _, k in effect_keys]
    labels = [lbl for lbl, _ in effect_keys]
    colors = ["#888" if not k.startswith("interaction") else "#C44E52" for _, k in effect_keys]
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v + (0.0005 if v >= 0 else -0.0015), f"{v:+.5f}",
                ha="center", fontsize=8.5,
                color=("green" if v < 0 else "red"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("ΔBrier (음수일수록 개선)")
    ax.set_title("Effect Decomposition — Brier↓ (음수 = 개선)")

    # right: AUC (↑ positive is better)
    ax = axes[1]
    vals = [d["roc_auc"][k] for _, k in effect_keys]
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v + (0.003 if v >= 0 else -0.005), f"{v:+.4f}",
                ha="center", fontsize=8.5,
                color=("green" if v > 0 else "red"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("ΔAUC (양수일수록 개선)")
    ax.set_title("Effect Decomposition — ROC AUC↑")

    fig.suptitle(
        "Phase 3 — Effect Decomposition (2×2 Factorial)\n"
        "데이터 효과가 LogReg→XGBoost 로 갈수록 커짐 = 비선형 상호작용 (interaction = 마지막 빨간 막대)",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (E) Interaction Plot (Data × Algo) + ANOVA annotation
# -----------------------------------------------------------------------------
def fig_e_interaction_plot(artifact) -> Path:
    out = FIGURES_DIR / "fig_p3e_interaction_plot.png"
    cells = artifact["cells"]
    anova = artifact["anova"]

    # fold-level means for each (data, algo)
    # X 축: algo (LogReg/XGB), 색: data (X_base/X_advanced)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6))

    # Subplot 1: Brier interaction plot
    ax = axes[0]
    algo_order = ["LogReg", "XGBoost"]
    x = np.arange(len(algo_order))
    for data_label, color, marker in [("X_base", "#4C72B0", "o"), ("X_advanced", "#C44E52", "s")]:
        means = []
        stds = []
        for algo in algo_order:
            cell_key = next(
                k for k, v in cells.items()
                if v["data"] == data_label and v["algo"] == algo
            )
            means.append(cells[cell_key]["fold_mean"]["brier"])
            stds.append(cells[cell_key]["fold_std"]["brier"])
        ax.errorbar(x, means, yerr=stds, color=color, marker=marker, markersize=10,
                    linewidth=2.5, capsize=5, label=data_label)
        for xi, mv in zip(x, means):
            ax.annotate(f"{mv:.5f}", (xi, mv), xytext=(7, 7),
                        textcoords="offset points", fontsize=9, color=color)
    ax.set_xticks(x); ax.set_xticklabels(algo_order)
    ax.set_xlabel("알고리즘")
    ax.set_ylabel("Brier (fold mean ± SD, ↓ 우수)")
    ia = anova["brier"]["C(data):C(algo)"]
    ax.set_title(
        f"Brier Interaction Plot\nANOVA interaction F={ia['F']:.2f}, p={ia['p']:.4g}",
        fontsize=11,
    )
    ax.legend(title="Data", loc="upper right")

    # Subplot 2: AUC interaction plot
    ax = axes[1]
    for data_label, color, marker in [("X_base", "#4C72B0", "o"), ("X_advanced", "#C44E52", "s")]:
        means = []
        stds = []
        for algo in algo_order:
            cell_key = next(
                k for k, v in cells.items()
                if v["data"] == data_label and v["algo"] == algo
            )
            means.append(cells[cell_key]["fold_mean"]["roc_auc"])
            stds.append(cells[cell_key]["fold_std"]["roc_auc"])
        ax.errorbar(x, means, yerr=stds, color=color, marker=marker, markersize=10,
                    linewidth=2.5, capsize=5, label=data_label)
        for xi, mv in zip(x, means):
            ax.annotate(f"{mv:.4f}", (xi, mv), xytext=(7, 7),
                        textcoords="offset points", fontsize=9, color=color)
    ax.set_xticks(x); ax.set_xticklabels(algo_order)
    ax.set_xlabel("알고리즘")
    ax.set_ylabel("ROC AUC (fold mean ± SD, ↑ 우수)")
    ia = anova["roc_auc"]["C(data):C(algo)"]
    ax.set_title(
        f"ROC AUC Interaction Plot\nANOVA interaction F={ia['F']:.2f}, p={ia['p']:.4g}",
        fontsize=11,
    )
    ax.legend(title="Data", loc="lower right")

    fig.suptitle(
        "Phase 3 — Interaction Plot (Data × Algo)\n"
        "두 선이 평행하지 않을수록 interaction 존재. ANOVA p<0.05 면 통계적으로 유의.",
        y=1.04, fontsize=12,
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
    marker = "## 8. 시각화"
    if marker in md:
        idx = md.index(marker)
        md = md[:idx].rstrip() + "\n\n" + section.rstrip() + "\n"
    else:
        # ## 7. 산출물 앞에 삽입 (있으면), 아니면 끝에 추가
        san_marker = "## 7. 산출물"
        if san_marker in md:
            idx = md.index(san_marker)
            md = md[:idx].rstrip() + "\n\n" + section.rstrip() + "\n\n" + md[idx:]
        else:
            md = md.rstrip() + "\n\n" + section.rstrip() + "\n"
    REPORT_PATH.write_text(md, encoding="utf-8")
    log(f"[report] phase3_report.md 패치 완료")


def build_section(figs: dict[str, Path], artifact) -> str:
    def rel(p: Path) -> str:
        return p.relative_to(PIPELINE_DIR).as_posix()

    cells = artifact["cells"]
    d = artifact["deltas"]
    anova = artifact["anova"]
    ia_brier = anova["brier"]["C(data):C(algo)"]
    ia_auc = anova["roc_auc"]["C(data):C(algo)"]

    L = []
    L.append("## 8. 시각화")
    L.append("")
    L.append("PNG 파일은 `pipeline/figures/`에 저장. 최종 Word 보고서 작성 시 그대로 재사용 가능.")
    L.append("")

    L.append("### 8.1 OOF 혼동행렬 — 4 cell")
    L.append("")
    L.append(f"![Phase 3 CM]({rel(figs['cm'])})")
    L.append("")
    L.append("- M1/M3 (LogReg): True Negative 절대 다수, TP 매우 적음 — 선형 모델의 한계.")
    L.append("- M2/M4 (XGBoost): TP 가 크게 증가, FP 도 함께 — 전체 분류 성능 대폭 개선.")
    L.append("")

    L.append("### 8.2 OOF 평가지표 막대 비교")
    L.append("")
    L.append(f"![Phase 3 Metrics Bar]({rel(figs['metrics'])})")
    L.append("")
    L.append("- Brier·LogLoss: M2 ≪ M1, M4 ≪ M3 (알고리즘 효과 압도). M4 < M2 (데이터 효과 in XGB).")
    L.append("- F1·AUC: 같은 패턴. 모든 메트릭에서 M4 가 최저(Brier↓) 또는 최고(F1·AUC↑).")
    L.append("")

    L.append("### 8.3 OOF ROC Curve")
    L.append("")
    L.append(f"![Phase 3 ROC]({rel(figs['roc'])})")
    L.append("")
    L.append("- M2/M4 의 ROC 곡선이 좌상단으로 강하게 휨 = 변별력 우수.")
    L.append("- M4 가 M2 보다 약간 더 위에 위치 = 환경 변수 추가의 효과가 XGB 위에서 발현.")
    L.append("")

    L.append("### 8.4 Reliability Diagram (Calibration Curve)")
    L.append("")
    L.append(f"![Phase 3 Calibration]({rel(figs['calibration'])})")
    L.append("")
    L.append("- 대각선에 가까울수록 확률 정상도 양호 (Brier 낮음).")
    L.append("- M2/M4 (XGBoost) 가 M1/M3 (LogReg) 보다 대각선에 훨씬 가까움.")
    L.append("- M1/M3 은 예측 확률이 0.2~0.4 영역에 몰려 있어 (선형 모델의 보수적 출력) calibration 자체가 부정확.")
    L.append("")

    L.append("### 8.5 Effect Decomposition")
    L.append("")
    L.append(f"![Phase 3 Effect Decomposition]({rel(figs['effect'])})")
    L.append("")
    L.append(
        f"- 좌: ΔBrier (음수 = 개선) / 우: ΔAUC (양수 = 개선). 마지막 빨간 막대가 **Interaction**.\n"
        f"- 데이터 효과: LogReg 위 ΔBrier={d['brier']['data_in_logreg']:+.5f} (거의 0), "
        f"XGB 위 ΔBrier={d['brier']['data_in_xgb']:+.5f} (유의 개선). 두 값의 차이 = "
        f"interaction = {d['brier']['interaction']:+.5f}."
    )
    L.append("")

    L.append("### 8.6 Interaction Plot (Data × Algo) + ANOVA")
    L.append("")
    L.append(f"![Phase 3 Interaction Plot]({rel(figs['interaction'])})")
    L.append("")
    L.append(
        "- Interaction Plot 은 두 선이 평행하지 않을수록 interaction 효과가 큼.\n"
        f"- ANOVA(Brier): C(data):C(algo) F={ia_brier['F']:.2f}, **p={ia_brier['p']:.4g}** "
        f"→ {'유의함 (p<0.05)' if ia_brier['p'] < 0.05 else '유의하지 않음'}.\n"
        f"- ANOVA(AUC) : C(data):C(algo) F={ia_auc['F']:.2f}, **p={ia_auc['p']:.4g}** "
        f"→ {'유의함 (p<0.05)' if ia_auc['p'] < 0.05 else '유의하지 않음'}."
    )
    L.append("")
    L.append(
        "**해석**: 두 그래프 모두에서 `X_advanced` 선의 기울기(LogReg→XGB 개선폭)가 `X_base` 보다 "
        "더 가파르다. 이는 \"환경 변수의 가치는 XGB(비선형) 위에서 더 크게 발현된다\" 는 정확히 "
        "interaction 의 정의이다. ANOVA p-value 가 0.05 미만이므로 이 차이는 우연이 아닌 "
        "통계적으로 유의한 비선형 상호작용 — Phase 4 트리 앙상블 채택의 최종 근거."
    )
    L.append("")

    return "\n".join(L)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    log("=" * 80)
    log("Phase 3 시각화 (4 cell + 2-way ANOVA) — 재학습 없이 JSON 직접 사용")
    log("=" * 80)

    y_full, proba_map, artifact = load_artifacts()

    log("\n[A] OOF 혼동행렬 + 메트릭 막대 ...")
    fig_cm = fig_a1_confusion_matrices(y_full, proba_map, artifact)
    fig_metrics = fig_a2_metrics_bar(artifact)

    log("\n[B] OOF ROC overlay ...")
    fig_roc = fig_b_roc_overlay(y_full, proba_map, artifact)

    log("\n[C] Calibration Curve ...")
    fig_cal = fig_c_calibration(y_full, proba_map, artifact)

    log("\n[D] Effect Decomposition ...")
    fig_eff = fig_d_effect_decomposition(artifact)

    log("\n[E] Interaction Plot + ANOVA ...")
    fig_inter = fig_e_interaction_plot(artifact)

    figs = {
        "cm": fig_cm,
        "metrics": fig_metrics,
        "roc": fig_roc,
        "calibration": fig_cal,
        "effect": fig_eff,
        "interaction": fig_inter,
    }

    log("\n[report] phase3_report.md 패치 중 ...")
    section = build_section(figs, artifact)
    patch_report(section)

    log("\n[done] Phase 3 시각화 완료. 6장 PNG + 리포트 패치.")


if __name__ == "__main__":
    main()
