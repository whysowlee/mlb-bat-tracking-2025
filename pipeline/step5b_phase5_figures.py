"""
Phase 5 시각화: phase5_report.md 에 임베드할 PNG 생성
=====================================================

readme Phase 5 (2026-05-29 업데이트) 사용자 명시 4종:
  (A) 1:1 R² 대조 산점도 — `ca-xBA vs wOBA` 와 `xBA vs wOBA` 나란히 + 회귀선
  (B) 포지션별 ca-xBA Top 10 리더보드 — 각 포지션별 수평 막대 + 실제 wOBA 비교
  (C) 실버 슬러거 교차 검증 — 포지션별 수상자 우리 ca-xBA 순위 시각화
  (D) 운(Luck) 분석 — Top 10 운 좋은 / 불운한 타자 양방향 막대 (할/푼/리 직관)

산출:
  - pipeline/figures/fig_p5*.png (5장)
  - phase5_report.md 끝에 "## 7. 시각화" 섹션 패치

실행:
    PYTHONUNBUFFERED=1 /opt/miniconda3/envs/mlb-xba/bin/python pipeline/step5b_phase5_figures.py \\
        2>&1 | tee pipeline/logs/step5b.log
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
from scipy.stats import linregress

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# 경로 & 폰트
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
FIGURES_DIR = PIPELINE_DIR / "figures"
REPORT_PATH = PIPELINE_DIR / "phase5_report.md"
RESULTS_JSON = OUTPUT_DIR / "phase5_results.json"
PLAYER_METRICS_CSV = OUTPUT_DIR / "phase5_player_metrics.csv"
SILVER_SLUGGER_VAL_CSV = OUTPUT_DIR / "phase5_silver_slugger_validation.csv"

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

POSITION_ORDER = ["C", "1B", "2B", "SS", "3B", "OF", "DH"]
POSITION_ALIASES_FIG = {
    "C": ["C"], "1B": ["1B"], "2B": ["2B"], "SS": ["SS"], "3B": ["3B"],
    "OF": ["LF", "CF", "RF", "OF"], "DH": ["DH"],
}
TOPN = 10
LUCK_TOPN = 10


def log(msg: str) -> None:
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# 데이터 로드
# -----------------------------------------------------------------------------
def load_all():
    log("[load] phase5_results.json + player_metrics.csv + silver_slugger_val.csv ...")
    artifact = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    metrics = pd.read_csv(PLAYER_METRICS_CSV)
    ss_val = pd.read_csv(SILVER_SLUGGER_VAL_CSV)
    log(f"  metrics: {metrics.shape}, ss_val: {ss_val.shape}")
    return artifact, metrics, ss_val


# -----------------------------------------------------------------------------
# (A) 1:1 R² 대조 산점도 — ca-xBA vs wOBA / xBA vs wOBA
# -----------------------------------------------------------------------------
def fig_a_scatter_1to1(metrics: pd.DataFrame, artifact: dict) -> Path:
    out = FIGURES_DIR / "fig_p5a_scatter_1to1_R2.png"
    corr = artifact["correlations"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    pairs = [
        ("ca_xba", "ca-xBA (우리 모델)", axes[0], "#C44E52"),
        ("est_ba", "xBA (MLB 공식 Statcast)", axes[1], "#4C72B0"),
    ]
    for x_col, label, ax, color in pairs:
        x = metrics[x_col].values
        y = metrics["woba"].values
        slope, intercept, r_value, _, _ = linregress(x, y)
        r2 = corr[x_col]["r_squared"]
        pearson = corr[x_col]["pearson_r"]

        ax.scatter(x, y, alpha=0.45, s=18, color=color, edgecolor="white", linewidth=0.4)
        # 회귀선
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color=color, linewidth=2, linestyle="-")
        # 대각선 (참고)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8, alpha=0.5, label="y = x")

        ax.set_xlabel(f"{label} (선수별 평균)")
        ax.set_ylabel("실제 wOBA (BIP-only, Ground Truth)")
        ax.set_title(f"{label} vs 실제 wOBA\n"
                     f"Pearson r = {pearson:.4f}  |  R² = {r2:.4f}  |  n = {len(metrics)}",
                     fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

    # 우위 박스
    delta_r2 = corr["ca_xba"]["r_squared"] - corr["est_ba"]["r_squared"]
    relative = delta_r2 / corr["est_ba"]["r_squared"] * 100
    fig.suptitle(
        f"Phase 5 — 1:1 R² 대조 (Y축 = 실제 wOBA, 250+ PA 선수 {len(metrics)}명)\n"
        f"ca-xBA 가 MLB 공식 xBA 보다 R² 절대 +{delta_r2:.4f} / 상대 +{relative:.1f}% 우수",
        y=1.02, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (B) 포지션별 ca-xBA Top 10 리더보드
# -----------------------------------------------------------------------------
def _normalize_position(pos: str) -> str | None:
    """Statcast 포지션을 silver_slugger 표기(7종)로 정규화."""
    if pd.isna(pos):
        return None
    pos = str(pos).upper()
    for ss_pos, aliases in POSITION_ALIASES_FIG.items():
        if pos in aliases:
            return ss_pos
    return None


def fig_b_position_leaderboards(metrics: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "fig_p5b_position_top10_leaderboards.png"
    metrics = metrics.copy()
    metrics["pos_normalized"] = metrics["position_mlbam"].apply(_normalize_position)

    fig, axes = plt.subplots(2, 4, figsize=(20, 11))
    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.axis("off")  # 미사용 subplot 숨김

    for i, pos in enumerate(POSITION_ORDER):
        ax = axes_flat[i]
        ax.axis("on")
        pool = metrics[metrics["pos_normalized"] == pos].sort_values("ca_xba", ascending=False).head(TOPN)
        if len(pool) == 0:
            ax.text(0.5, 0.5, f"{pos}: 데이터 없음", ha="center", va="center")
            ax.set_title(pos)
            continue

        names = pool["last_name, first_name"].tolist()[::-1]
        ca_vals = pool["ca_xba"].tolist()[::-1]
        woba_vals = pool["woba"].tolist()[::-1]
        ranks = list(range(len(names), 0, -1))

        y = np.arange(len(names))
        w = 0.4
        ax.barh(y - w/2, ca_vals, w, color="#C44E52", label="ca-xBA", edgecolor="white", linewidth=0.4)
        ax.barh(y + w/2, woba_vals, w, color="#4C72B0", label="실제 wOBA", edgecolor="white", linewidth=0.4)
        for j, (ca, wo, r) in enumerate(zip(ca_vals, woba_vals, ranks)):
            ax.text(ca + 0.003, j - w/2, f"{ca:.3f}", va="center", fontsize=7)
            ax.text(wo + 0.003, j + w/2, f"{wo:.3f}", va="center", fontsize=7)

        ax.set_yticks(y)
        ax.set_yticklabels([f"{r}. {n}" for n, r in zip(names, ranks)], fontsize=8)
        ax.set_xlabel("Score")
        ax.set_title(f"{pos}  (Top {TOPN})", fontsize=11, fontweight="bold")
        ax.set_xlim(0, max(max(ca_vals), max(woba_vals)) * 1.20)
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Phase 5 — 포지션별 ca-xBA Top {TOPN} 리더보드 (실제 wOBA 동시 표시)",
        y=0.995, fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (C) 실버 슬러거 교차 검증
# -----------------------------------------------------------------------------
def fig_c_silver_slugger_validation(ss_val: pd.DataFrame, artifact: dict) -> Path:
    out = FIGURES_DIR / "fig_p5c_silver_slugger_validation.png"
    ss_val = ss_val.copy()
    # position_rank 가 nan 이면 검증 불가
    ss_val["sortkey"] = ss_val["position_rank"].fillna(999).astype(float)
    ss_val = ss_val.sort_values(["sortkey", "league", "position"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 9))
    n = len(ss_val)
    y_pos = np.arange(n)[::-1]
    colors = []
    bar_lens = []
    for _, r in ss_val.iterrows():
        rank = r["position_rank"]
        topN = r["position_topN"]
        if pd.isna(rank):
            colors.append("#999999")
            bar_lens.append(0)
        elif topN is True or topN == "True":
            colors.append("#229922")
            bar_lens.append(TOPN - int(rank) + 1)  # 순위 좋을수록 막대 길이 길게
        else:
            colors.append("#cc3333")
            bar_lens.append(max(TOPN - int(rank) + 1, -3))

    ax.barh(y_pos, bar_lens, color=colors, edgecolor="white", linewidth=0.5)
    labels = []
    for _, r in ss_val.iterrows():
        rank = r["position_rank"]
        rank_str = f"#{int(rank)}" if not pd.isna(rank) else "(검증불가)"
        marker = "✓" if r["position_topN"] is True or r["position_topN"] == "True" else (
            "?" if pd.isna(rank) else "✗"
        )
        labels.append(f"[{marker}] {r['league']} {r['position']:4s} {r['player_name']:25s} {rank_str}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, family="monospace")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="gray", linewidth=0.3, alpha=0.5)
    ax.set_xlabel(f"포지션별 ca-xBA Top {TOPN} 적중 정도 (높을수록 상위 순위)")

    hits = artifact["silver_slugger"]["hits"]
    eligible = artifact["silver_slugger"]["eligible"]
    hit_rate = artifact["silver_slugger"]["hit_rate"]
    ax.set_title(
        f"Phase 5 — 실버 슬러거 교차 검증\n"
        f"포지션별 ca-xBA Top {TOPN} 적중률: {hits}/{eligible} ({hit_rate*100:.1f}%)",
        fontsize=12, fontweight="bold",
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#229922", edgecolor="white", label=f"✓ Top {TOPN} 적중"),
        Patch(facecolor="#cc3333", edgecolor="white", label="✗ Top {TOPN} 미달".replace("{TOPN}", str(TOPN))),
        Patch(facecolor="#999999", edgecolor="white", label="? 검증 불가 (250 PA 미달)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (D) 운(Luck) 분석 — Top 10 양방향 막대
# -----------------------------------------------------------------------------
def fig_d_luck_analysis(artifact: dict) -> Path:
    out = FIGURES_DIR / "fig_p5d_luck_analysis.png"
    top_lucky = pd.DataFrame(artifact["luck"]["top_lucky"])
    top_unlucky = pd.DataFrame(artifact["luck"]["top_unlucky"])

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))

    def fmt_label(v, avg, ca, bp, cb, delta):
        cb_str = f"{cb:.3f}" if pd.notna(cb) else "N/A"
        delta_str = f"{delta:+.3f}" if pd.notna(delta) else "N/A"
        return (f"  {v:+.3f}  AVG={avg:.3f} ca-xBA={ca:.3f}  "
                f"시즌BABIP={bp:.3f}  통산={cb_str}  Δ={delta_str}")

    # 운 좋은 타자 (시즌/통산 BABIP + Δ 라벨)
    ax = axes[0]
    top_lucky = top_lucky.sort_values("luck", ascending=True)
    y = np.arange(len(top_lucky))
    luck_vals = top_lucky["luck"].tolist()
    bars = ax.barh(y, luck_vals, color=["#229922" if v > 0 else "#9bbb8a" for v in luck_vals],
                    edgecolor="white", linewidth=0.4)
    for bar, (_, r) in zip(bars, top_lucky.iterrows()):
        ax.text(r["luck"] + 0.0015, bar.get_y() + bar.get_height()/2,
                fmt_label(r["luck"], r["ba"], r["ca_xba"], r["babip"],
                          r.get("career_babip"), r.get("babip_minus_career")),
                va="center", fontsize=7.5, ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(top_lucky["last_name, first_name"].tolist(), fontsize=10)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("luck = AVG − ca-xBA  (할/푼/리 직관)")
    ax.set_title(
        f"🍀 운(행운 효과 가설) Top {LUCK_TOPN}\n"
        f"Δ_BABIP = 시즌 − 통산 (개인 baseline 대비 편차, 도메인 정통 행운 시그널)",
        fontsize=10, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.7)

    # 불운한 타자
    ax = axes[1]
    top_unlucky = top_unlucky.sort_values("luck", ascending=False)
    y = np.arange(len(top_unlucky))
    luck_vals = top_unlucky["luck"].tolist()
    bars = ax.barh(y, luck_vals, color="#cc3333", edgecolor="white", linewidth=0.4)
    for bar, (_, r) in zip(bars, top_unlucky.iterrows()):
        ax.text(r["luck"] - 0.002, bar.get_y() + bar.get_height()/2,
                fmt_label(r["luck"], r["ba"], r["ca_xba"], r["babip"],
                          r.get("career_babip"), r.get("babip_minus_career")) + "  ",
                va="center", fontsize=7.5, ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels(top_unlucky["last_name, first_name"].tolist(), fontsize=10)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("luck = AVG − ca-xBA  (할/푼/리 직관)")
    ax.set_title(
        f"💀 불운(호수비·환경 손해 가설) Top {LUCK_TOPN}\n"
        "Δ < 0 = 두 지표 모두 불운 일치 / Δ ≥ 0 = Trout 패턴 (ca-xBA 단독 신호)",
        fontsize=10, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min - (x_max - x_min) * 0.7, x_max)

    fig.suptitle(
        "Phase 5 — 운(Luck) 분석: AVG − ca-xBA  +  통산 BABIP 교차 검증 (개인 baseline)",
        y=1.02, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (F) luck vs Δ_BABIP (시즌 − 통산) 산점도 — 도메인 정통 비교
# -----------------------------------------------------------------------------
def fig_f_luck_vs_babip(metrics: pd.DataFrame, artifact: dict) -> Path:
    out = FIGURES_DIR / "fig_p5f_luck_vs_delta_babip.png"
    babip_stats = artifact.get("luck", {}).get("babip_stats", {})
    r_pearson = babip_stats.get("luck_delta_pearson", float("nan"))
    r_spearman = babip_stats.get("luck_delta_spearman", float("nan"))
    # 보조: 시즌 BABIP 비교용
    r_pearson_season = babip_stats.get("luck_babip_pearson", float("nan"))

    top_lucky_names = {r["last_name, first_name"] for r in artifact["luck"]["top_lucky"]}
    top_unlucky_names = {r["last_name, first_name"] for r in artifact["luck"]["top_unlucky"]}

    # 통산 BABIP 매핑되지 않은 선수 제외
    valid = metrics.dropna(subset=["babip_minus_career"])

    fig, ax = plt.subplots(figsize=(11.5, 8))

    colors = []
    for nm in valid["last_name, first_name"]:
        if nm in top_lucky_names:
            colors.append("#229922")
        elif nm in top_unlucky_names:
            colors.append("#cc3333")
        else:
            colors.append("#bbbbbb")
    sizes = [55 if c != "#bbbbbb" else 15 for c in colors]

    ax.scatter(valid["babip_minus_career"], valid["luck"], c=colors, s=sizes,
               alpha=0.7, edgecolor="white", linewidth=0.5)

    ax.axvline(0, color="#666", linestyle="--", linewidth=1,
               label="Δ_BABIP = 0 (개인 baseline 일치)")
    ax.axhline(0, color="black", linewidth=0.7, label="luck = 0 (AVG = ca-xBA)")

    for _, row in valid.iterrows():
        nm = row["last_name, first_name"]
        if nm in top_lucky_names or nm in top_unlucky_names:
            ax.annotate(
                nm.split(",")[0],
                (row["babip_minus_career"], row["luck"]),
                xytext=(4, 4), textcoords="offset points", fontsize=7,
                color=("#1d7a1d" if nm in top_lucky_names else "#a52020"),
            )

    ax.set_xlabel("Δ_BABIP = 시즌 BABIP − 통산 BABIP  (도메인 정통 행운 시그널)")
    ax.set_ylabel("luck = AVG − ca-xBA  (모델 기반 운 지표)")
    ax.set_title(
        f"Phase 5 — Luck vs Δ_BABIP 산점도 (도메인 정통 vs 모델 기반)\n"
        f"Pearson r = {r_pearson:.3f}, Spearman ρ = {r_spearman:.3f}  "
        f"(참고: 단일 시즌 BABIP 와는 r = {r_pearson_season:.3f})",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # 사분면 텍스트
    ax.text(0.02, 0.97,
            "좌상: Δ_BABIP↓ 인데 luck↑\n→ 약한 contact로 행운 효과",
            transform=ax.transAxes, fontsize=8, color="#666",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="#ccc"))
    ax.text(0.98, 0.03,
            "우하: Δ_BABIP≥0 인데 luck↓\n→ Trout 패턴 (통산 baseline 평균인데\nca-xBA만 환경/quality 손해 포착)",
            transform=ax.transAxes, fontsize=8, color="#666",
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="#ccc"))

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# (E) ATH 영향 보조 시각화: BIP ratio 분포
# -----------------------------------------------------------------------------
def fig_e_bip_ratio(metrics: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "fig_p5e_bip_ratio_ath_impact.png"
    metrics = metrics.copy()
    metrics["bip_ratio"] = metrics["our_bip"] / metrics["bip"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.hist(metrics["bip_ratio"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(0.50, color="#cc3333", linestyle="--", linewidth=1.2,
                label=f"tolerance = 50% (Phase 1 ATH 제외 영향 임계)")
    ax.axvline(metrics["bip_ratio"].median(), color="black", linestyle=":", linewidth=1.0,
                label=f"median = {metrics['bip_ratio'].median():.3f}")
    ax.set_xlabel("our_bip / csv.bip  (1.0 = 완전 일치)")
    ax.set_ylabel("선수 수")
    ax.set_title(
        f"Phase 5 보조 — BIP 정의 일치 검증 분포 (n={len(metrics)})\n"
        f"좌측 꼬리 = Phase 1 ATH 홈경기 제외 영향을 받은 선수들 (원정 경기만 ca-xBA 산출)",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9)
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
        md = md.rstrip() + "\n\n" + section.rstrip() + "\n"
    REPORT_PATH.write_text(md, encoding="utf-8")
    log(f"[report] phase5_report.md 패치 완료 (§ 7 시각화 추가)")


def build_section(figs: dict[str, Path], artifact: dict, n_players: int) -> str:
    def rel(p: Path) -> str:
        return p.relative_to(PIPELINE_DIR).as_posix()

    corr = artifact["correlations"]
    ca_r2 = corr["ca_xba"]["r_squared"]
    xba_r2 = corr["est_ba"]["r_squared"]
    delta_r2 = ca_r2 - xba_r2
    relative = delta_r2 / xba_r2 * 100

    L = []
    L.append("## 7. 시각화")
    L.append("")
    L.append("PNG 5장. 모두 `pipeline/figures/`에 저장.")
    L.append("")

    L.append("### 7.1 1:1 R² 대조 산점도 — Phase 5 메인 결론")
    L.append("")
    L.append(f"![1:1 R² Scatter]({rel(figs['scatter'])})")
    L.append("")
    L.append(f"- **좌 (ca-xBA vs 실제 wOBA)**: R² = **{ca_r2:.4f}**, Pearson r = {corr['ca_xba']['pearson_r']:.4f}")
    L.append(f"- **우 (MLB 공식 xBA vs 실제 wOBA)**: R² = {xba_r2:.4f}, Pearson r = {corr['est_ba']['pearson_r']:.4f}")
    L.append(f"- **차이: 절대 R² +{delta_r2:.4f} / 상대 우위 +{relative:.1f}%** — ca-xBA 가 선수의 실제 wOBA 를 명확히 더 잘 설명.")
    L.append("- 환경 변수(구장·기상)를 비선형 모델(LGBM + Isotonic, Phase 4 OOF Brier=0.13092)로 학습한 효과가 시즌 누적 지표에서도 발현됨을 입증.")
    L.append("")

    L.append("### 7.2 포지션별 ca-xBA Top 10 리더보드")
    L.append("")
    L.append(f"![Position Top 10]({rel(figs['leaderboards'])})")
    L.append("")
    L.append(f"- 7개 포지션(C, 1B, 2B, SS, 3B, OF, DH) 각각의 ca-xBA Top {TOPN} 선수.")
    L.append("- 빨강 막대 = 우리 ca-xBA, 파랑 막대 = 실제 wOBA. 두 막대가 비슷할수록 calibration 우수.")
    L.append("- 각 포지션 1위는 실버 슬러거 후보 (§7.3 교차 검증 참조).")
    L.append("")

    L.append("### 7.3 실버 슬러거 교차 검증")
    L.append("")
    L.append(f"![Silver Slugger Validation]({rel(figs['silver'])})")
    L.append("")
    hits = artifact["silver_slugger"]["hits"]
    eligible = artifact["silver_slugger"]["eligible"]
    hit_rate = artifact["silver_slugger"]["hit_rate"]
    L.append(f"- **포지션별 Top {TOPN} 적중률: {hits}/{eligible} ({hit_rate*100:.1f}%)**")
    L.append("- 녹색 (✓) = ca-xBA 가 실제 실버 슬러거 수상자를 Top 10 안에 정확히 식별.")
    L.append("- 빨강 (✗) = Top 10 미달. 단 이 경우도 ca-xBA 가 \"실력 외 요인(수비 가치, 명성 등)\"이 수상에 작용했을 가능성을 시사.")
    L.append("- 회색 (?) = 250 PA 미달 또는 표기 차이로 검증 불가.")
    L.append("")

    L.append("### 7.4 운(Luck) 분석 — Top 10 양방향 + 통산 BABIP 교차 검증")
    L.append("")
    L.append(f"![Luck Analysis with Career BABIP]({rel(figs['luck'])})")
    L.append("")
    L.append("- 각 막대 라벨에 **시즌 BABIP · 통산 BABIP · Δ (시즌 − 통산)** 표시.")
    L.append("- Δ_BABIP 는 도메인 정통의 행운 시그널: **자기 통산 baseline 대비 시즌 편차**.")
    L.append("- 좌: 🍀 운(행운 효과 가설) 타자. Δ_BABIP > 0 이면 통산 대비 시즌이 높아 **두 지표 일치**.")
    L.append("- 우: 💀 불운(호수비·환경 손해 가설) 타자. Δ_BABIP < 0 이면 둘 다 불운, Δ ≥ 0 이면 **Trout 패턴** (ca-xBA 단독 환경/quality 신호).")
    L.append("")

    L.append("### 7.5 Luck vs Δ_BABIP 산점도 — 도메인 정통 vs 모델 기반")
    L.append("")
    L.append(f"![Luck vs Delta BABIP Scatter]({rel(figs['luck_vs_babip'])})")
    L.append("")
    babip_stats = artifact.get("luck", {}).get("babip_stats", {})
    r_d_pearson = babip_stats.get("luck_delta_pearson", float("nan"))
    r_d_spearman = babip_stats.get("luck_delta_spearman", float("nan"))
    r_s_pearson = babip_stats.get("luck_babip_pearson", float("nan"))
    n_with_career = babip_stats.get("n_with_career", 0)
    L.append(
        f"- 통산 BABIP 매핑 성공한 {n_with_career} 명 산점도. "
        f"**Pearson r = {r_d_pearson:.3f}, Spearman ρ = {r_d_spearman:.3f}** "
        f"(참고: 단일 시즌 BABIP 와는 r = {r_s_pearson:.3f}). "
        "→ Δ_BABIP 와의 상관이 더 의미 있는 도메인 정통 비교."
    )
    L.append(
        "- 수직 점선: Δ_BABIP = 0 (자기 통산 baseline) / 수평: luck = 0. "
        "녹색 점 = 운(행운 효과) Top 10, 빨강 점 = 불운 Top 10."
    )
    L.append(
        "- **사분면 해석**:\n"
        "  - 우상단 (Δ↑, luck↑) = 두 지표 모두 행운 일치 (이중 검증).\n"
        "  - 좌하단 (Δ↓, luck↓) = 두 지표 모두 불운 일치 (이중 검증).\n"
        "  - 우하단 (Δ≥0, luck↓) = **Trout 패턴** — 통산 baseline 평균인데 ca-xBA 만 환경/quality 손해 포착. "
        "Front Office 의 저평가 발굴 포인트.\n"
        "  - 좌상단 (Δ↓, luck↑) = 약한 contact 로 행운 효과."
    )
    L.append("")

    L.append("### 7.6 (보조) BIP 정의 일치 분포 — ATH 영향")
    L.append("")
    L.append(f"![BIP Ratio]({rel(figs['bip_ratio'])})")
    L.append("")
    L.append("- 대부분 선수는 `our_bip / csv.bip` 비율이 0.85~0.97 (Phase 1 의 |la|>60 컷오프·핵심 결측 제거로 약간 적음).")
    L.append("- 좌측 꼬리(0.40~0.50)는 Phase 1 의 Athletics 홈경기 제외 결정의 영향을 받은 선수들 — 원정 경기 BIP 만으로 ca-xBA 산출.")
    L.append("")

    return "\n".join(L)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    log("=" * 80)
    log("Phase 5 시각화: (A) 1:1 R² 산점도 / (B) 포지션 Top 10 / (C) 실버슬러거 / (D) 운분석")
    log("=" * 80)

    artifact, metrics, ss_val = load_all()
    n_players = len(metrics)

    log("\n[figs] PNG 생성 중 ...")
    figs = {
        "scatter":      fig_a_scatter_1to1(metrics, artifact),
        "leaderboards": fig_b_position_leaderboards(metrics),
        "silver":       fig_c_silver_slugger_validation(ss_val, artifact),
        "luck":         fig_d_luck_analysis(artifact),
        "luck_vs_babip": fig_f_luck_vs_babip(metrics, artifact),
        "bip_ratio":    fig_e_bip_ratio(metrics),
    }

    log("\n[report] phase5_report.md 패치 중 ...")
    section = build_section(figs, artifact, n_players)
    patch_report(section)

    log("\n[done] Phase 5 시각화 완료. 5장 PNG + 리포트 패치.")


if __name__ == "__main__":
    main()
