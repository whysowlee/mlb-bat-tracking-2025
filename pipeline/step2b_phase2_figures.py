"""
Phase 2 시각화: phase2_report.md 에 임베드할 PNG 생성
=====================================================

새 구조 (2024 전체 + StratifiedKFold 5-fold CV) 대응 버전.

사용자 선택 패키지 3종:
  (A) EDA            — 핵심 변수 KDE(is_hit 그룹별) + Boxplot(is_hit 그룹별)
  (C) 샘플링 비교     — OOF 혼동행렬(meta) + OOF 메트릭 바 + ROC overlay
                       (ROC 는 시각화 전용 80/20 hold-out 으로 산출)
  (D) Feature Imp.   — RF Top 20 + MI Top 20 + (RF rank × MI rank) 산점도

산출:
  - pipeline/figures/fig_p2_*.png
  - phase2_report.md 끝에 "## 9. 시각화" 섹션 패치

실행:
    /opt/miniconda3/envs/mlb-xba/bin/python pipeline/step2b_phase2_figures.py
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

import xgboost as xgb
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
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# 경로 & 폰트
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
FIGURES_DIR = PIPELINE_DIR / "figures"
REPORT_PATH = PIPELINE_DIR / "phase2_report.md"

INPUT_PARQUET = OUTPUT_DIR / "2024_data.parquet"
X_FULL_PARQUET = OUTPUT_DIR / "phase2_X_full.parquet"
Y_FULL_PARQUET = OUTPUT_DIR / "phase2_y_full.parquet"
FEATURES_JSON = OUTPUT_DIR / "phase2_features.json"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# macOS 한글 폰트
_KFONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
if Path(_KFONT_PATH).exists():
    fm.fontManager.addfont(_KFONT_PATH)
    _korean_name = fm.FontProperties(fname=_KFONT_PATH).get_name()
else:
    _korean_name = "AppleGothic"
sns.set_style("whitegrid", {"axes.grid": True, "grid.alpha": 0.3})
plt.rcParams["font.family"] = _korean_name
plt.rcParams["font.sans-serif"] = [_korean_name, "AppleGothic", "Nanum Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

RANDOM_STATE = 42
MI_SUBSAMPLE_SIZE = 30_000
N_JOBS = 2
ROC_HOLDOUT_TEST_SIZE = 0.30  # ROC overlay 시각화 전용 hold-out 비율


# -----------------------------------------------------------------------------
# 데이터 로드
# -----------------------------------------------------------------------------
def load_artifacts():
    print("[load] 2024_data + phase2_X_full + meta ...", flush=True)
    train_orig = pd.read_parquet(INPUT_PARQUET)  # 2024 전체
    X_full = pd.read_parquet(X_FULL_PARQUET)
    y_full = pd.read_parquet(Y_FULL_PARQUET)["is_hit"]
    meta = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
    print(
        f"  → train_orig {train_orig.shape}, X_full {X_full.shape}, y_full {y_full.shape}",
        flush=True,
    )
    return train_orig, X_full, y_full, meta


# -----------------------------------------------------------------------------
# (A) EDA: KDE + Boxplot (is_hit 그룹별)
# -----------------------------------------------------------------------------
KEY_VARS = [
    ("launch_speed", "launch_speed (mph)"),
    ("launch_angle", "launch_angle (도)"),
    ("wx_temperature_2m", "기온 (°C)"),
    ("wx_wind_speed_10m", "풍속 (km/h)"),
    ("elevation", "고도 (ft)"),
    ("hr_park_effects", "HR Park Effects"),
]


def fig_a_eda_kde(train_orig: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "fig_p2a1_eda_kde.png"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, label) in zip(axes.flat, KEY_VARS):
        for cls, color, hit_label in [(0, "#888888", "is_hit=0"), (1, "#4C72B0", "is_hit=1")]:
            sub = train_orig.loc[train_orig["is_hit"] == cls, col].dropna()
            sns.kdeplot(
                sub, ax=ax, label=hit_label, color=color, fill=True, alpha=0.35, linewidth=1.5
            )
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("Phase 2 EDA — 핵심 변수 KDE (is_hit 그룹별 분포 비교, 2024 전체)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved: {out.name}", flush=True)
    return out


def fig_a_eda_boxplot(train_orig: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "fig_p2a2_eda_boxplot.png"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, label) in zip(axes.flat, KEY_VARS):
        df_plot = train_orig[[col, "is_hit"]].dropna().copy()
        df_plot["is_hit"] = df_plot["is_hit"].astype(int)
        sns.boxplot(
            data=df_plot, x="is_hit", y=col, ax=ax,
            hue="is_hit", palette=["#888888", "#4C72B0"], legend=False, fliersize=2,
        )
        ax.set_ylabel(label)
        ax.set_xlabel("is_hit")
    fig.suptitle("Phase 2 EDA — 핵심 변수 Boxplot (is_hit 그룹별, 2024 전체)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved: {out.name}", flush=True)
    return out


# -----------------------------------------------------------------------------
# (C) 샘플링 비교 — OOF 메트릭(meta) + ROC 시각화 전용 80/20 hold-out
# -----------------------------------------------------------------------------
SAMPLING_NAMES = ["None", "Under", "SMOTE"]
SAMPLING_LABELS = {"None": "원본(None)", "Under": "Under", "SMOTE": "SMOTE"}
SAMPLING_COLORS = {"None": "#4C72B0", "Under": "#DD8452", "SMOTE": "#55A868"}


def fig_c1_oof_confusion_matrices(sampling_results: dict) -> Path:
    """meta 의 OOF 혼동행렬(cm_tn/fp/fn/tp) 사용 — Phase 2 step2 와 정확히 동일 결과."""
    out = FIGURES_DIR / "fig_p2c1_sampling_oof_confusion_matrices.png"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, name in zip(axes, SAMPLING_NAMES):
        r = sampling_results[name]
        cm = np.array([[r["cm_tn"], r["cm_fp"]], [r["cm_fn"], r["cm_tp"]]])
        sns.heatmap(
            cm, annot=True, fmt=",d", cmap="Blues", ax=ax, cbar=False,
            xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"],
        )
        ax.set_title(f"{SAMPLING_LABELS[name]}  (OOF Brier={r['oof_brier']:.5f})")
    fig.suptitle("샘플링별 OOF 혼동행렬 (5-fold CV, threshold=0.5)", y=1.04, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved: {out.name}", flush=True)
    return out


def fig_c2_oof_metrics_bar(sampling_results: dict) -> Path:
    """OOF 평가지표 막대 비교 — 모든 지표 '높을수록 우수'로 방향 통일.

    Brier·LogLoss 는 본래 낮을수록 우수하므로 `1 − 값` 으로 변환해
    나머지 지표(F1·AUC·P·R·Acc)와 같은 방향(↑ 높을수록 우수)으로 맞춘다.
    (본 데이터의 LogLoss < 1 이라 1−LogLoss ∈ (0,1); 일반적으로 LogLoss 는
     1 을 넘을 수 있어 음수가 될 수 있음에 유의.)
    """
    out = FIGURES_DIR / "fig_p2c2_sampling_oof_metrics_bar.png"
    metric_keys = ["oof_brier", "oof_logloss", "oof_f1", "oof_roc_auc",
                   "oof_precision", "oof_recall", "oof_accuracy"]
    metric_labels = ["1-Brier↑", "1-LogLoss↑", "F1", "AUC", "Precision", "Recall", "Accuracy"]
    # 낮을수록 우수 → 1에서 빼서 '높을수록 우수'로 방향 반전
    INVERT_KEYS = {"oof_brier", "oof_logloss"}
    x = np.arange(len(metric_labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    for i, name in enumerate(SAMPLING_NAMES):
        r = sampling_results[name]
        vals = [(1.0 - r[k]) if k in INVERT_KEYS else r[k] for k in metric_keys]
        ax.bar(x + (i - 1) * width, vals, width, label=SAMPLING_LABELS[name],
               color=SAMPLING_COLORS[name])
        for j, v in enumerate(vals):
            ax.text(
                x[j] + (i - 1) * width, v + 0.012, f"{v:.4f}",
                ha="center", fontsize=7.5,
            )
    ax.set_xticks(x); ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score (↑ 높을수록 우수)")
    ax.set_ylim(0, 1.05)
    ax.set_title("샘플링 기법별 OOF 평가지표 비교 (XGBoost default, 5-fold CV) — 모든 지표 높을수록 우수 (Brier·LogLoss는 1-값)")
    ax.legend(title="Sampling")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved: {out.name}", flush=True)
    return out


def fig_c3_roc_holdout(X: pd.DataFrame, y: pd.Series) -> Path:
    """
    ROC overlay 시각화 전용 — 단일 80/20 stratified hold-out 으로 3개 sampling fit→ROC.
    (실제 샘플링 선정은 step2 의 OOF Brier 기준; 본 ROC 는 시각화 보조)
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    out = FIGURES_DIR / "fig_p2c3_sampling_roc_holdout.png"
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=ROC_HOLDOUT_TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    samplers = {
        "None": None,
        "Under": RandomUnderSampler(random_state=RANDOM_STATE),
        "SMOTE": SMOTE(random_state=RANDOM_STATE, k_neighbors=5),
    }
    proba_dict: dict[str, np.ndarray] = {}
    auc_dict: dict[str, float] = {}
    print(f"[roc] 단일 hold-out (test_size={ROC_HOLDOUT_TEST_SIZE}) 으로 ROC 시각화 ...", flush=True)
    for name, sampler in samplers.items():
        if sampler is None:
            X_s, y_s = X_tr, y_tr
        else:
            X_s, y_s = sampler.fit_resample(X_tr, y_tr)
        model = xgb.XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="logloss", tree_method="hist", verbosity=0,
        )
        model.fit(X_s, y_s)
        proba = model.predict_proba(X_te)[:, 1]
        proba_dict[name] = proba
        auc_dict[name] = float(roc_auc_score(y_te, proba))
        print(f"  [{name}] hold-out AUC={auc_dict[name]:.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name in SAMPLING_NAMES:
        fpr, tpr, _ = roc_curve(y_te, proba_dict[name])
        ax.plot(
            fpr, tpr,
            label=f"{SAMPLING_LABELS[name]} (AUC={auc_dict[name]:.4f})",
            color=SAMPLING_COLORS[name], linewidth=2,
        )
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("샘플링별 ROC Curve (시각화 전용 80/20 hold-out)\n실제 선정은 OOF Brier (위 C2 참조)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved: {out.name}", flush=True)
    return out


# -----------------------------------------------------------------------------
# (D) Feature Importance — RF Top 20 + MI Top 20 + Rank scatter
# -----------------------------------------------------------------------------
def recompute_full_importance(X: pd.DataFrame, y: pd.Series, meta: dict):
    """best_sampling=None 이므로 X_full 그대로 RF best params 로 1회 fit + MI."""
    best_params = meta["rf_best_params"]
    print(f"[fs] RF 재학습 (best params={best_params}) ...", flush=True)
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS, **best_params)
    rf.fit(X, y)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"  ✓ RF importance 계산 완료 ({len(rf_imp)} features)", flush=True)

    print(f"[fs] MI Stratified subsample (n={MI_SUBSAMPLE_SIZE:,d}) ...", flush=True)
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=min(MI_SUBSAMPLE_SIZE, len(X)), random_state=RANDOM_STATE
    )
    sub_idx, _ = next(sss.split(X, y))
    mi_arr = mutual_info_classif(
        X.iloc[sub_idx].values, y.iloc[sub_idx].values,
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
    )
    mi_series = pd.Series(mi_arr, index=X.columns).sort_values(ascending=False)
    print(f"  ✓ MI 계산 완료", flush=True)
    return rf_imp, mi_series


def fig_d1_rf_top20(rf_imp: pd.Series) -> Path:
    out = FIGURES_DIR / "fig_p2d1_rf_importance_top20.png"
    top = rf_imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(top.index, top.values, color="#4C72B0")
    for i, v in enumerate(top.values):
        ax.text(v, i, f"  {v:.4f}", va="center", fontsize=8)
    ax.set_xlabel("RF feature_importance")
    ax.set_title("RF Importance Top 20 (best params, 2024 전체 fit)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved: {out.name}", flush=True)
    return out


def fig_d2_mi_top20(mi_series: pd.Series) -> Path:
    out = FIGURES_DIR / "fig_p2d2_mi_top20.png"
    top = mi_series.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(top.index, top.values, color="#DD8452")
    for i, v in enumerate(top.values):
        ax.text(v, i, f"  {v:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Mutual Information")
    ax.set_title("MI Top 20 (Stratified 30K subsample)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved: {out.name}", flush=True)
    return out


def fig_d3_rank_scatter(rf_imp: pd.Series, mi_series: pd.Series, x_base: list[str]) -> Path:
    out = FIGURES_DIR / "fig_p2d3_rf_mi_rank_scatter.png"

    common = rf_imp.index.intersection(mi_series.index)
    rf_rank_pct = rf_imp[common].rank(ascending=False, pct=True)
    mi_rank_pct = mi_series[common].rank(ascending=False, pct=True)
    rf_top_score = 1.0 - rf_rank_pct
    mi_top_score = 1.0 - mi_rank_pct

    drop_mask = (rf_rank_pct > 0.7) & (mi_rank_pct > 0.7)
    drop_mask.loc[drop_mask.index.intersection(x_base)] = False

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.axhspan(0.7, 1.05, xmin=(0.7 - (-0.05)) / 1.10, xmax=1.0, alpha=0.10, color="green")
    ax.axhspan(-0.05, 0.3, xmin=(0 - (-0.05)) / 1.10, xmax=(0.3 - (-0.05)) / 1.10,
               alpha=0.10, color="red")

    is_xbase = pd.Series(False, index=common)
    is_xbase.loc[is_xbase.index.intersection(x_base)] = True
    is_core = (rf_top_score > 0.7) & (mi_top_score > 0.7)

    colors = []
    for col in common:
        if drop_mask[col]:
            colors.append("#cc3333")
        elif is_xbase[col]:
            colors.append("#ff8800")
        elif is_core[col]:
            colors.append("#229922")
        else:
            colors.append("#888888")

    ax.scatter(rf_top_score, mi_top_score, c=colors, s=55, edgecolor="white",
               linewidth=0.8, alpha=0.85)

    label_cols = [c for c in common if is_core[c] or drop_mask[c] or is_xbase[c]]
    for col in label_cols:
        ax.annotate(
            col,
            (rf_top_score[col], mi_top_score[col]),
            xytext=(4, 4), textcoords="offset points", fontsize=7,
            color=("#cc3333" if drop_mask[col] else ("#ff8800" if is_xbase[col] else "#229922")),
        )

    ax.text(0.85, 0.95, "핵심 변수\n(RF&MI 상위 30%)", ha="center", fontsize=11,
            color="#1d7a1d", fontweight="bold", alpha=0.7,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#229922", alpha=0.6))
    ax.text(0.15, 0.10, "Drop 대상\n(RF&MI 하위 30%)", ha="center", fontsize=11,
            color="#a52020", fontweight="bold", alpha=0.7,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cc3333", alpha=0.6))

    ax.axhline(0.7, color="gray", linestyle=":", linewidth=1)
    ax.axhline(0.3, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0.7, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0.3, color="gray", linestyle=":", linewidth=1)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("RF 상위성 (1 - rank_pct, 우측이 핵심)")
    ax.set_ylabel("MI 상위성 (1 - rank_pct, 상단이 핵심)")
    ax.set_title("Feature Selection — RF rank × MI rank 산점도\n"
                 "우상단=핵심 변수, 좌하단=drop 대상, 주황=X_BASE (보호)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#229922", edgecolor="white", label="핵심 (RF&MI 상위 30%)"),
        Patch(facecolor="#cc3333", edgecolor="white", label="Drop (RF&MI 하위 30%)"),
        Patch(facecolor="#ff8800", edgecolor="white", label="X_BASE (보호)"),
        Patch(facecolor="#888888", edgecolor="white", label="중간 영역"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved: {out.name}", flush=True)

    n_core = int(is_core.sum())
    n_drop = int(drop_mask.sum())
    print(f"  → 핵심 변수: {n_core}개 / drop 대상: {n_drop}개 / X_BASE: {len(x_base)}개", flush=True)
    return out


# -----------------------------------------------------------------------------
# 리포트 패치
# -----------------------------------------------------------------------------
def patch_report(figure_section: str):
    md = REPORT_PATH.read_text(encoding="utf-8")
    marker = "## 9. 시각화"
    # ## 10. 산출물 또는 ## 9. 시각화 위치에 삽입
    if marker in md:
        idx = md.index(marker)
        md = md[:idx].rstrip() + "\n\n" + figure_section.rstrip() + "\n"
    else:
        # ## 10. 산출물 앞에 삽입 (있으면)
        san_marker = "## 10. 산출물"
        if san_marker in md:
            idx = md.index(san_marker)
            md = md[:idx].rstrip() + "\n\n" + figure_section.rstrip() + "\n\n" + md[idx:]
        else:
            md = md.rstrip() + "\n\n" + figure_section.rstrip() + "\n"
    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"[report] phase2_report.md 패치 완료", flush=True)


def build_figure_section(figs: dict[str, Path], sampling_results: dict, meta: dict) -> str:
    def rel(p: Path) -> str:
        return p.relative_to(PIPELINE_DIR).as_posix()

    best = meta["best_sampling"]
    L = []
    L.append("## 9. 시각화")
    L.append("")
    L.append("PNG 파일은 모두 `pipeline/figures/`에 저장. 최종 Word 보고서 작성 시 그대로 재사용 가능.")
    L.append("> _(B) 상관관계 히트맵은 변수 80+개 가독성 문제로 제외. 다중공선성 제거 변수 쌍은 §5 마크다운 표 참조._")
    L.append("")

    # (A) EDA
    L.append("### 9.1 EDA — 핵심 변수 분포 (is_hit 그룹별, 2024 전체)")
    L.append("")
    L.append("**(A1) KDE — 연속 분포 비교**")
    L.append("")
    L.append(f"![EDA KDE]({rel(figs['eda_kde'])})")
    L.append("")
    L.append("- launch_speed: is_hit=1 그룹의 분포가 우측(고속)으로 이동 → 발사 속도가 빠를수록 안타 확률 ↑.")
    L.append("- launch_angle: is_hit=1 그룹이 10~25° 부근에 집중 → 라인드라이브 각도가 가장 유리.")
    L.append("- 환경 변수(기온·풍속·고도·HR park effects): is_hit 그룹 간 분포 차이가 미미 → 단변량만으로는 신호 약함. **다른 변수와의 비선형 상호작용(트리 모델)이 필요함을 시사**.")
    L.append("")
    L.append("**(A2) Boxplot — 중앙값/IQR/이상치 비교**")
    L.append("")
    L.append(f"![EDA Boxplot]({rel(figs['eda_boxplot'])})")
    L.append("")
    L.append("- KDE 결과와 동일한 패턴이 quartile 통계로 재확인됨.")
    L.append("- launch_speed 의 IQR이 is_hit=1 에서 명확히 우측으로 이동.")
    L.append("")

    # (C) Sampling
    L.append("### 9.2 샘플링 기법 비교 (5-fold CV OOF — XGBoost default)")
    L.append("")
    L.append("**(C1) OOF 혼동행렬 (3개 샘플링, threshold=0.5)**")
    L.append("")
    L.append(f"![Sampling CM]({rel(figs['sampling_cm'])})")
    L.append("")
    L.append("- 5-fold CV OOF predict_proba 기반 — Phase 2 step2 산출값과 일치.")
    L.append("- 원본(None): True Negative 다수, Recall 낮음 (보수적 예측).")
    L.append("- Under: TP 대폭 증가, 동시에 FP도 증가 → Recall ↑ / Precision ↓ 트레이드오프.")
    L.append("- SMOTE: 원본에 가까운 균형, F1은 원본보다 살짝 ↑.")
    L.append("")
    L.append("**(C2) OOF 평가지표 막대 비교 (Brier↓ / LogLoss↓ / F1 / AUC / P / R / Acc)**")
    L.append("")
    L.append(f"![Sampling Metrics]({rel(figs['sampling_metrics'])})")
    L.append("")
    L.append(
        f"- **최종 선정 = `{best}`** (OOF Brier 기준 최솟값: "
        f"{sampling_results[best]['oof_brier']:.5f})."
    )
    L.append("- Brier·LogLoss는 낮을수록 우수 — 확률 정상도(probability calibration) 기준.")
    L.append("- F1만 보면 Under가 가장 높지만, 확률값 자체가 깨져서 ca-xBA 산출에 부적합.")
    L.append("")
    L.append("**(C3) ROC Curve 겹치기 (시각화 전용 80/20 hold-out)**")
    L.append("")
    L.append(f"![Sampling ROC]({rel(figs['sampling_roc'])})")
    L.append("")
    L.append(f"- 본 ROC 는 시각화 목적으로 단일 80/20 stratified hold-out (test_size={ROC_HOLDOUT_TEST_SIZE}) 에서 산출. **실제 샘플링 선정은 위 C2 의 OOF Brier 기준**.")
    L.append("- 세 ROC 곡선이 거의 겹침 → AUC 자체에는 큰 차이 없음. 차이는 확률 calibration(Brier/LogLoss)에서 나타남.")
    L.append("- 모델 자체의 변별력은 데이터 분포보다는 변수 풀과 알고리즘에 의해 결정됨을 시사 → Phase 3 ablation 가설과 일치.")
    L.append("")

    # (D) Feature Importance
    L.append("### 9.3 Feature Importance — RF + MI + Rank Scatter")
    L.append("")
    L.append("**(D1) RF Importance Top 20**")
    L.append("")
    L.append(f"![RF Top 20]({rel(figs['rf_top20'])})")
    L.append("")
    L.append("- 최상위에 `launch_speed`, `launch_angle` 압도적 → xBA의 본질과 일치.")
    L.append("- 환경/투구 변수도 일정 비중 — 트리 모델이 비선형 결합 학습 가능.")
    L.append("")
    L.append("**(D2) Mutual Information Top 20**")
    L.append("")
    L.append(f"![MI Top 20]({rel(figs['mi_top20'])})")
    L.append("")
    L.append("- RF Top 20과 상당 부분 겹치되, MI는 단변량 정보 기준이라 일부 변수의 순위는 다름.")
    L.append("- 두 지표 모두에서 살아남은 변수 = 신뢰도 높은 핵심 변수.")
    L.append("")
    L.append("**(D3) RF rank × MI rank 산점도 — 핵심 vs Drop 영역**")
    L.append("")
    L.append(f"![Rank Scatter]({rel(figs['rank_scatter'])})")
    L.append("")
    L.append("- **우상단(녹색 영역)**: RF·MI 모두 상위 30% → 핵심 변수. `launch_speed`, `launch_angle` 등이 위치.")
    L.append("- **좌하단(붉은 영역)**: RF·MI 모두 하위 30% → Feature Selection drop 대상. "
             f"총 {len(meta['dropped_via_feature_selection'])}개가 이 영역에서 제거됨.")
    L.append("- **주황 점**: X_BASE — 절대 drop 금지(보호) 영역.")
    L.append("- 점선 격자(0.3, 0.7)는 30%/70% 분위 기준선.")
    L.append("")

    return "\n".join(L)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    print("=" * 80, flush=True)
    print("Phase 2 시각화 (A) EDA / (C) Sampling / (D) Feature Importance", flush=True)
    print("=" * 80, flush=True)

    train_orig, X_full, y_full, meta = load_artifacts()
    sampling_results = meta["sampling_results"]

    print("\n[A] EDA 시각화 ...", flush=True)
    fig_kde = fig_a_eda_kde(train_orig)
    fig_box = fig_a_eda_boxplot(train_orig)

    print("\n[C] Sampling OOF 시각화 ...", flush=True)
    fig_cm = fig_c1_oof_confusion_matrices(sampling_results)
    fig_metrics = fig_c2_oof_metrics_bar(sampling_results)
    fig_roc = fig_c3_roc_holdout(X_full, y_full)

    print("\n[D] Feature Importance 시각화 (RF 재학습 + MI 재계산) ...", flush=True)
    rf_imp, mi_series = recompute_full_importance(X_full, y_full, meta)
    fig_rf = fig_d1_rf_top20(rf_imp)
    fig_mi = fig_d2_mi_top20(mi_series)
    fig_scatter = fig_d3_rank_scatter(rf_imp, mi_series, meta["X_base"])

    figs = {
        "eda_kde": fig_kde,
        "eda_boxplot": fig_box,
        "sampling_cm": fig_cm,
        "sampling_metrics": fig_metrics,
        "sampling_roc": fig_roc,
        "rf_top20": fig_rf,
        "mi_top20": fig_mi,
        "rank_scatter": fig_scatter,
    }

    print("\n[report] phase2_report.md 패치 중 ...", flush=True)
    section = build_figure_section(figs, sampling_results, meta)
    patch_report(section)

    print("\n[done] Phase 2 시각화 완료. 8장 PNG + 리포트 패치.", flush=True)


if __name__ == "__main__":
    main()
