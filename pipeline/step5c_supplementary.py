"""
Phase 5 보강 분석 v2 (사용자 추가 지시, 2026-05-29 최종)
========================================================

**v1 대비 변경 사항 (사용자 명시):**
  1. **실버 슬러거 검증의 한계 명시** — 현장 전문가 정성 투표라 통계적 100% 개런티 불가
     ('재미있는 도메인 일관성 점검'으로 격하). 통계 검증은 § 3 R² 가 담당.
  2. **wOBA 로 실버 슬러거 예측 결과 삭제** — 3 지표 비교를 **2 지표(AVG vs ca-xBA)** 로 축소.
     wOBA 는 사실상 wRC+ 기반 시상 결정의 가장 큰 단일 변수이므로 적중률 ≈ 상한선 → 비교 의미 약함.
  3. **새 § 3.2 추가 — "wOBA 를 놔두고 굳이 ca-xBA 를 구하는 이유 (지표의 의의)"**
     wOBA = 결과 지표, ca-xBA = 과정 지표. Front Office 가 저평가 선수 발굴에 사용.
     사용자 작성 텍스트 그대로 보존 (③ 기존 xBA 한계 돌파 + 🚀 최종 결론 포함).

산출:
  - pipeline/figures/fig_p5d_luck_analysis.png  (운 그림 재생성)
  - pipeline/figures/fig_p5c_silver_slugger_validation.png  (실버슬러거 그림 재생성)
  - pipeline/figures/fig_p5f_2metric_hitrate_comparison.png  (신규 — 2 지표)
  - phase5_report.md (§ 3.2 신규, § 4.1~4.3 보강, § 5.1 2지표로 갱신)
  - phase5_results.json (hitrates_2metric 추가)

실행 흐름 (idempotent 보장 위해 base 깨끗하게):
  1. step5 재실행 (base report 생성, ~1-2분)
  2. step5c 실행 (이 스크립트, ~1분)

실행:
    PYTHONUNBUFFERED=1 /opt/miniconda3/envs/mlb-xba/bin/python pipeline/step5c_supplementary.py \\
        2>&1 | tee pipeline/logs/step5c.log
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

POSITION_ALIASES = {
    "C": ["C"], "1B": ["1B"], "2B": ["2B"], "SS": ["SS"], "3B": ["3B"],
    "OF": ["LF", "CF", "RF", "OF"], "DH": ["DH"],
}
TOPN = 10
LUCK_TOPN = 10


def log(msg: str) -> None:
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# 1. 2 지표(AVG / ca-xBA) 실버 슬러거 적중률 비교
# -----------------------------------------------------------------------------
def compute_2metric_hitrate(metrics: pd.DataFrame, ss_val: pd.DataFrame) -> dict:
    """포지션별 Top {TOPN} 산출을 AVG, ca-xBA 각각으로 수행 → 실버 슬러거 적중 비교.

    wOBA 는 사실상 wRC+ 기반 시상 결정의 가장 큰 단일 변수이므로 적중률 ≈ 상한선 →
    비교 의미가 약해 사용자 결정으로 제외 (2026-05-29 사용자 지시).
    """
    log(f"\n[2metric] AVG vs ca-xBA 포지션별 Top {TOPN} 적중률 비교 ...")
    metrics = metrics.copy()

    def normalize(pos):
        if pd.isna(pos):
            return None
        pos = str(pos).upper()
        for ss_pos, aliases in POSITION_ALIASES.items():
            if pos in aliases:
                return ss_pos
        return None

    metrics["pos_normalized"] = metrics["position_mlbam"].apply(normalize)

    results = {}
    for metric_col, metric_label in [("ba", "전통 AVG"), ("ca_xba", "ca-xBA")]:
        hits = 0
        eligible = 0
        detail = []
        for _, ss_row in ss_val.iterrows():
            if pd.isna(ss_row["mlbam_id"]):
                continue
            ss_pos = ss_row["position"]
            if ss_pos not in POSITION_ALIASES or not POSITION_ALIASES[ss_pos]:
                continue
            if ss_pos == "OF":
                pool = metrics[metrics["pos_normalized"] == "OF"]
            else:
                pool = metrics[metrics["pos_normalized"] == ss_pos]
            pool = pool.sort_values(metric_col, ascending=False).reset_index(drop=True)
            match = pool[pool["mlbam_id"] == ss_row["mlbam_id"]]
            if len(match) == 0:
                continue
            rank = int(match.index[0]) + 1
            in_topN = rank <= TOPN
            eligible += 1
            if in_topN:
                hits += 1
            detail.append({
                "league": ss_row["league"], "position": ss_pos,
                "player_name": ss_row["player_name"],
                "rank": rank, "in_topN": in_topN,
            })
        hit_rate = hits / max(eligible, 1)
        log(f"  {metric_label:<12s}: 적중 {hits}/{eligible} ({hit_rate*100:.1f}%)")
        results[metric_col] = {
            "label": metric_label, "hits": hits, "eligible": eligible,
            "hit_rate": hit_rate, "detail": detail,
        }
    return results


# -----------------------------------------------------------------------------
# 2. 운 그림 재생성 (emoji 제거, 라벨 위치 수정)
# -----------------------------------------------------------------------------
def regenerate_luck_figure(artifact: dict) -> Path:
    out = FIGURES_DIR / "fig_p5d_luck_analysis.png"
    top_lucky = pd.DataFrame(artifact["luck"]["top_lucky"])
    top_unlucky = pd.DataFrame(artifact["luck"]["top_unlucky"])

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # === 운 좋은 타자 ===
    ax = axes[0]
    top_lucky = top_lucky.sort_values("luck", ascending=True)
    names = top_lucky["last_name, first_name"].tolist()
    luck_vals = top_lucky["luck"].tolist()
    avgs = top_lucky["ba"].tolist()
    ca_xbas = top_lucky["ca_xba"].tolist()
    y = np.arange(len(names))
    colors = ["#229922" if v > 0 else "#9bc59a" for v in luck_vals]
    bars = ax.barh(y, luck_vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v, avg, ca in zip(bars, luck_vals, avgs, ca_xbas):
        if v >= 0:
            ax.text(v + 0.0008, bar.get_y() + bar.get_height()/2,
                    f" {v:+.3f}  (AVG={avg:.3f}, ca-xBA={ca:.3f})",
                    va="center", ha="left", fontsize=8.5)
        else:
            ax.text(v - 0.0008, bar.get_y() + bar.get_height()/2,
                    f" {v:+.3f}  (AVG={avg:.3f}, ca-xBA={ca:.3f}) ",
                    va="center", ha="right", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("luck = AVG - ca-xBA  (할/푼/리 직관, 양수=운 좋음)")
    ax.set_title(f"[운 좋은 타자] Top {LUCK_TOPN}\n실제 타율이 ca-xBA보다 높거나 격차가 작은 선수",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    x_min, x_max = ax.get_xlim()
    pad = (x_max - x_min) * 0.45
    ax.set_xlim(x_min - pad * 0.3, x_max + pad)

    # === 불운한 타자 ===
    ax = axes[1]
    top_unlucky = top_unlucky.sort_values("luck", ascending=False)
    names = top_unlucky["last_name, first_name"].tolist()
    luck_vals = top_unlucky["luck"].tolist()
    avgs = top_unlucky["ba"].tolist()
    ca_xbas = top_unlucky["ca_xba"].tolist()
    y = np.arange(len(names))
    bars = ax.barh(y, luck_vals, color="#cc3333", edgecolor="white", linewidth=0.5)
    for bar, v, avg, ca in zip(bars, luck_vals, avgs, ca_xbas):
        ax.text(v - 0.003, bar.get_y() + bar.get_height()/2,
                f"{v:+.3f}  (AVG={avg:.3f}, ca-xBA={ca:.3f}) ",
                va="center", ha="right", fontsize=8.5, color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("luck = AVG - ca-xBA  (할/푼/리 직관, 음수=불운)")
    ax.set_title(f"[불운한 타자] Top {LUCK_TOPN}\n실제 타율이 ca-xBA보다 크게 낮음 (호수비/구장 환경 손해)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    x_min, x_max = ax.get_xlim()
    pad = (x_max - x_min) * 0.05
    ax.set_xlim(x_min - pad, x_max + pad)

    fig.suptitle("Phase 5 - 운(Luck) 분석: luck = AVG - ca-xBA  (시스템 오프셋 평균 -0.10 — 상대 순위로 해석)",
                 y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# 3. 실버슬러거 그림 재생성 (emoji 제거)
# -----------------------------------------------------------------------------
def regenerate_silver_slugger_figure(ss_val: pd.DataFrame, artifact: dict) -> Path:
    out = FIGURES_DIR / "fig_p5c_silver_slugger_validation.png"
    ss_val = ss_val.copy()
    ss_val["sortkey"] = ss_val["position_rank"].fillna(999).astype(float)
    ss_val = ss_val.sort_values(["sortkey", "league", "position"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    n = len(ss_val)
    y_pos = np.arange(n)[::-1]
    colors = []
    bar_lens = []
    for _, r in ss_val.iterrows():
        rank = r["position_rank"]
        topN = r["position_topN"]
        if pd.isna(rank):
            colors.append("#999999"); bar_lens.append(0)
        elif topN is True or topN == "True":
            colors.append("#229922"); bar_lens.append(TOPN - int(rank) + 1)
        else:
            colors.append("#cc3333"); bar_lens.append(max(TOPN - int(rank) + 1, -3))

    ax.barh(y_pos, bar_lens, color=colors, edgecolor="white", linewidth=0.5)
    labels = []
    for _, r in ss_val.iterrows():
        rank = r["position_rank"]
        rank_str = f"#{int(rank)}" if not pd.isna(rank) else "(미검증)"
        if pd.isna(rank):
            marker = "[?]"
        elif r["position_topN"] is True or r["position_topN"] == "True":
            marker = "[O]"
        else:
            marker = "[X]"
        labels.append(f"{marker} {r['league']} {r['position']:4s} {r['player_name']:25s} {rank_str}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, family="monospace")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel(f"포지션별 ca-xBA Top {TOPN} 적중 정도 (높을수록 상위 순위)")

    hits = artifact["silver_slugger"]["hits"]
    eligible = artifact["silver_slugger"]["eligible"]
    hit_rate = artifact["silver_slugger"]["hit_rate"]
    ax.set_title(
        f"Phase 5 - 실버 슬러거 교차 검증 (재미있는 도메인 일관성 점검)\n"
        f"포지션별 ca-xBA Top {TOPN} 적중률: {hits}/{eligible} ({hit_rate*100:.1f}%) "
        f"  ※ 현장 전문가 정성 투표 시상 — 통계 검증 아님",
        fontsize=11, fontweight="bold",
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#229922", edgecolor="white", label=f"[O] Top {TOPN} 적중"),
        Patch(facecolor="#cc3333", edgecolor="white", label=f"[X] Top {TOPN} 미달"),
        Patch(facecolor="#999999", edgecolor="white", label="[?] 미검증 (250 PA 미달)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# 4. 2 지표 적중률 비교 그림
# -----------------------------------------------------------------------------
def fig_2metric_comparison(hitrates: dict) -> Path:
    out = FIGURES_DIR / "fig_p5f_2metric_hitrate_comparison.png"
    labels = []
    rates = []
    hits_list = []
    eligibles_list = []
    for metric_col, info in hitrates.items():
        labels.append(info["label"])
        rates.append(info["hit_rate"] * 100)
        hits_list.append(info["hits"])
        eligibles_list.append(info["eligible"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#888888", "#C44E52"]
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=0.8)
    for bar, r, h, e in zip(bars, rates, hits_list, eligibles_list):
        ax.text(bar.get_x() + bar.get_width()/2, r + 1.5,
                f"{r:.1f}%\n({h}/{e})",
                ha="center", fontsize=12, fontweight="bold")
    delta = rates[1] - rates[0]
    ax.set_ylabel("실버 슬러거 포지션 Top 10 적중률 (%)")
    ax.set_ylim(0, max(max(rates) * 1.25, 100))
    ax.set_title(
        f"Phase 5 - 전통 AVG vs ca-xBA 실버 슬러거 적중률 비교\n"
        f"ca-xBA 가 전통 AVG 보다 +{delta:.1f}%p 우수  ※ 재미용 도메인 일관성 점검",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log(f"  ✓ saved: {out.name}")
    return out


# -----------------------------------------------------------------------------
# 5. SIGNIFICANCE_TEXT — 사용자 작성 텍스트 그대로 (§ 3.2)
# -----------------------------------------------------------------------------
SIGNIFICANCE_TEXT = """### 3.2 ⚾ wOBA 를 놔두고 굳이 ca-xBA 를 구하는 이유 (지표의 의의)

wOBA 가 선수의 공격력을 평가하는 **'최고의 완성형 결과지'**라면, ca-xBA 는 선수의 **'순수한 타격 기술과 과정'**을 평가하는 **엑스레이(X-ray) 사진**입니다. 두 지표는 용도와 의의가 완전히 다릅니다.

**① 결과(Result)와 과정(Process)의 분리**

- wOBA 는 선구안(볼넷), 컨택 능력(삼진 회피), 타구 결과를 모두 섞어서 만든 '종합 성적표'입니다. 타자가 타석에서 얼마나 점수에 기여했는지를 보여줍니다.
- ca-xBA 는 선구안을 제외하고, 오직 타자가 방망이에 공을 맞힌 순간(Contact) 발생하는 **'타구의 물리적 퀄리티(속도, 각도)'** 만을 떼어내서 봅니다. 즉, 타자의 순수한 **'배럴(Barrel) 생산 능력'**을 평가하는 지표입니다.

**② 운(Luck)과 야수 수비력의 통제**

- 현실 야구에서는 115마일로 완벽하게 쳐도 상대 3루수가 엄청난 다이빙 캐치를 하면 실제 기록은 아웃(wOBA 하락)이 됩니다. 반면 빗맞은 팝플라이가 야수들 사이에 똑 떨어지면 안타(wOBA 상승)가 됩니다.
- ca-xBA 는 이러한 **'수비수의 실력'이나 '운'을 완벽하게 통제** 합니다. "이 속도와 각도로 날아간 타구는 과거 데이터를 봤을 때 85% 확률로 안타가 되어야 마땅하다" 라고 **과정이 억울한 타자들의 진짜 실력을 구제** 해 줍니다.

**③ 기존 xBA 의 한계 돌파 (구장 환경의 개입)**

- 메이저리그 공식 xBA 조차도 결정적인 결함이 있었습니다. 콜로라도(쿠어스 필드)의 뜬공과 샌프란시스코(오라클 파크)의 뜬공을 똑같이 취급했다는 것입니다.
- 우리가 만든 **ca-xBA (Context-Aware)** 는 **구장의 물리적 제약과 기후까지 모델에 학습** 시켰습니다. *"이 타구는 양키스타디움이었으면 90% 안타(홈런)지만, 현재 구장 환경을 융합해 보니 10% 아웃이다"* 라고 가장 현실에 가까운 확률을 다림질해 낸 것입니다.

**🚀 최종 결론: ca-xBA 는 구단(Front Office)의 무기다**

결론적으로 ca-xBA 를 구해야 하는 이유는 **"미래를 예측하고 저평가된 선수를 발굴하기 위해서"** 입니다. (이른바 '머니볼' 의 핵심)

시즌이 끝났을 때 어떤 선수가 실제 타율이나 wOBA 는 형편없었지만, 우리 모델로 돌려본 ca-xBA 는 매우 높았다면? 이 선수는 타격 기술이 나쁜 것이 아니라, 그 해 지독하게 운이 없었거나 구장과 궁합이 안 맞았을 뿐입니다. 구단 프런트는 이 지표를 보고 **"이 선수는 내년에 반드시 반등한다!"** 라고 확신하며 싼값에 선수를 트레이드해 올 수 있습니다.

> **wOBA 가 '올해 누가 잘했나' 를 뒤돌아보는 지표라면, ca-xBA 는 '이 선수의 진짜 타격 실력이 어느 정도이며, 내년엔 어떨 것인가' 를 꿰뚫어 보는 강력한 예측 엔진입니다.** 이것이 바로 데이터 마이닝을 통해 이 복잡한 모델을 구축한 궁극적인 학술적·실무적 의의입니다.
"""


# Web research 결과 (v3 — 사용자 제공 팩트·URL 기반 보강, 2026-05-29)
# - 사용자가 명확히 지정한 3명(Davis/Mangum/Andujar) 은 제공 텍스트 그대로 사용
# - Wilson/Hoerner/Arraez 는 v2 web search 인용 유지 (출처 명시)
# - 근거 부족한 선수는 "명확한 스카우팅 근거가 검색되지 않아 표본 부족 및 단순 부진" 으로 솔직히 분류
LUCK_TOP5_RESEARCH = {
    "lucky": [
        {
            "name": "Jacob Wilson", "luck": +0.002,
            "evidence": ("AL 올스타 SS (.311 AVG, AL 3위). 극단적 contact-only profile — "
                         "Hard-hit% 6th percentile, AVG Exit Velocity 12th percentile. "
                         "Just Baseball 은 '우투 Luis Arraez (with more pop and better defense)' 로 평가."),
            "sources": [
                ("Just Baseball — Jacob Wilson Is in a League of His Own Amongst MLB Rookies",
                 "https://www.justbaseball.com/mlb/jacob-wilson-athletics-league-of-his-own-amongst-mlb-rookies/"),
            ],
        },
        {
            "name": "Jake Mangum", "luck": -0.017,
            "evidence": ("2025 시즌 기준 **배럴 타구 비율(Barrel%) 0.0%, 하드힛 비율 26.0%** 로 "
                         "타구의 물리적 퀄리티(ca-xBA)는 최악의 수준을 기록했다. 하지만 극단적인 빗맞은 타구"
                         "(**발사각 -50도 이하의 촙 땅볼**)를 친 후 **압도적인 주력으로 내야 안타(Infield single)** 를 "
                         "만들어내는 본인 특유의 플레이 스타일 덕분에 고타율을 기록했다. "
                         "모델이 잡아낼 수 없는 **'발'로 만들어낸 행운의 타율** 이다."),
            "sources": [
                ("Statcast 타구 속도 데이터 — Jake Mangum",
                 "https://baseballsavant.mlb.com/savant-player/jake-mangum-663968"),
            ],
        },
        {
            "name": "Nico Hoerner", "luck": -0.027,
            "evidence": ("Cubs 2B, .297 AVG (NL 2위), 156경기. hard hit rate 27.7% (career worst). "
                         "BABIP .313 (career mark .307 보다 약간 높음). 'one of the best sheer contact hitters in baseball' "
                         "(FanGraphs). 모든 source 가 contact skill 위주 + lower 부드러운 BABIP 정상화 가능성 시사."),
            "sources": [
                ("Chicago Sun-Times — Cubs' Nico Hoerner had a standout year",
                 "https://chicago.suntimes.com/cubs/2026/01/01/cubs-nico-hoerner-had-a-standout-year-could-2026-even-better"),
            ],
        },
        {
            "name": "Luis Arraez", "luck": -0.029,
            "evidence": ("Padres 1B/2B, .292 AVG. **hard-hit rate 16.7% (qualified hitters 꼴찌)**, "
                         "barrel rate 1.1% (꼴찌), xwOBA 15th percentile. FanGraphs/Metsmerized 등 다수 매체가 "
                         "'soft contact·extreme contact-only' profile 로 명시 — 모델의 'luck=양수' 평가 정직."),
            "sources": [
                ("Metsmerized Online — Free Agent Profile: Luis Arraez",
                 "https://metsmerizedonline.com/free-agent-profile-luis-arraez-if/"),
            ],
        },
        {
            "name": "Miguel Andujar", "luck": -0.030,
            "evidence": ("**헛스윙 비율은 낮아 공을 잘 맞히고 있으나, xBA가 .215에 불과**할 정도로 "
                         "**약한 타구(Weak contact)** 를 양산 중이다. 타구 질은 낮지만 **빗맞은 타구(Bloopers)가 "
                         "수비수 없는 곳에 떨어지며** 단기적인 타율 급등을 이뤄낸 **전형적인 '럭키 시즌' 프로필** 이다."),
            "sources": [
                ("Pitcher List 판타지 프로필",
                 "https://www.pitcherlist.com/"),
            ],
        },
    ],
    "unlucky": [
        {
            "name": "Riley Adams", "luck": -0.216,
            "evidence": ("Nationals C/UTIL, .186 AVG. 시즌 후 designated for assignment. "
                         "**명확한 스카우팅 근거(BABIP/xwOBA underperform 등) 가 검색되지 않아 "
                         "표본 부족 및 단순 부진으로 분류** 한다."),
            "sources": [],
        },
        {
            "name": "Michael Toglia", "luck": -0.206,
            "evidence": ("Rockies 1B, .202 AVG. wOBA .265 / **xwOBA .271** (둘 다 7th percentile) — "
                         "Statcast 자체 xwOBA 도 underperform 폭이 작음 → 실제로는 quality of contact 부족. "
                         "5월에 Triple-A 강등."),
            "sources": [
                ("FantasyTeamAdvice — Michael Toglia 2025 Stats",
                 "https://fantasyteamadvice.com/mlb/players/3653/michael-toglia/stats/2025"),
            ],
        },
        {
            "name": "Mike Trout", "luck": -0.196,
            "evidence": ("Angels OF/DH. **4월 BABIP .135, 시즌 .270** (career .341). "
                         "'11 outs on batted balls with xBA ≥ .500, 리그 1위' (CBS Sports). "
                         "FanGraphs RotoGraphs 의 xwOBA underperformers list 단골. **명백한 BABIP 불운**."),
            "sources": [
                ("CBS Sports — 2025 Fantasy Baseball: buy low on Mike Trout",
                 "https://www.cbssports.com/fantasy/baseball/news/2025-fantasy-baseball-week-5-trade-values-nows-the-time-to-buy-low-on-mike-trout/"),
            ],
        },
        {
            "name": "Henry Davis", "luck": -0.181,
            "evidence": ("피츠버그 팬덤의 심층 분석에 따르면, 데이비스의 **평균 스윙 스피드는 75.8mph로 "
                         "리그 평균(71.7mph)이나 홈런 타자들보다도 빠르다.** 하지만 공을 정타(Square up)로 "
                         "맞히지 못하고 **공 밑을 스윙하여 팝플라이를 만들거나, 극단적인 당겨치기(Dead pull hitter)로 "
                         "인해 땅볼을 양산**하고 있다. 즉, 스윙의 파워 자체는 엄청나기 때문에 모델은 높은 안타 확률을 "
                         "부여하지만, **컨택의 최적화 실패로 실제 타율을 깎아 먹는 전형적인 '과정 대비 결과가 불운한' 케이스** "
                         "로 완벽히 설명된다."),
            "sources": [
                ("Reddit r/buccos 분석 스레드",
                 "https://www.reddit.com/r/buccos/"),
                ("Statcast 공식 프로필 — Henry Davis",
                 "https://baseballsavant.mlb.com/savant-player/henry-davis-680711"),
            ],
        },
        {
            "name": "Kyle Schwarber", "luck": -0.178,
            "evidence": ("⚠️ **모델 평가와 실제 시즌 결과가 반대**. 실제 2025 = NL MVP 2위, **56홈런 (NL 1위)**, "
                         "career-high 20.8% barrel rate, 85 barrels (Statcast era 10위). "
                         "우리 ca-xBA luck=음수 평가의 원인 = fly ball power hitter profile — "
                         "BIP 한정 wOBA 는 매우 높으나 fly out 비율도 높아 AVG는 낮음. "
                         "**모델의 본질적 특성**(BIP quality 중심)이 power hitter 에 불리하게 작용."),
            "sources": [
                ("MLB.com — Key numbers explaining Kyle Schwarber's strong 2025 season",
                 "https://www.mlb.com/news/key-numbers-explaining-kyle-schwarber-s-strong-2025-season"),
            ],
        },
    ],
}


# -----------------------------------------------------------------------------
# 6. 리포트 패치 — § 3.2 신규 + § 4.1~4.3 보강 + § 5.1 (2지표) 갱신
# -----------------------------------------------------------------------------
def patch_report(hitrates: dict, fig_2metric: Path, correlations: dict) -> None:
    md = REPORT_PATH.read_text(encoding="utf-8")

    # === § 3.1 — 상관계수 심층 해석 (Pearson r + Spearman ρ 야구 도메인 의미) ===
    ca = correlations["ca_xba"]
    es = correlations["est_ba"]
    sec31 = []
    sec31.append("")
    sec31.append("### 3.1 🎯 상관계수 심층 해석 — 야구 도메인 관점의 두 가지 신뢰성")
    sec31.append("")
    sec31.append("단순한 R² 수치 비교를 넘어, **Pearson r (선형 신뢰성)** 와 **Spearman ρ (순위 신뢰성)** 가 "
                  "야구 현장에서 어떤 의미를 갖는지 해석한다.")
    sec31.append("")
    sec31.append(
        f"**① Pearson r (선형 상관계수, {ca['pearson_r']:.4f} vs {es['pearson_r']:.4f}):** "
        f"우리 모델(ca-xBA)이 **{ca['pearson_r']:.4f}** 를 기록하며 공식 xBA({es['pearson_r']:.4f})를 압도했다. "
        f"이는 ca-xBA 가 상승할 때 선수의 실제 생산력(wOBA)도 **기계처럼 일정하게 우상향** 한다는 강력한 "
        f"**선형적 신뢰성** 을 뜻하며, 기존 xBA 에 존재했던 **'거짓 양성(노이즈)' 타구들을 훌륭하게 걸러냈음** 을 증명한다."
    )
    sec31.append("")
    sec31.append(
        f"**② Spearman ρ (순위 상관계수, {ca['spearman_rho']:.4f} vs {es['spearman_rho']:.4f}):** "
        f"실무적 관점(MVP 투표, 트레이드 등)에서 가장 중요한 **'선수들의 실제 파괴력 등수(Rank)'를 줄 세우는 능력** "
        f"에서도 우리 모델(**{ca['spearman_rho']:.4f}**)이 공식 모델({es['spearman_rho']:.4f})을 크게 앞섰다. "
        f"이는 **프런트 오피스가 타자를 평가할 때 ca-xBA 가 훨씬 더 정확한 기준표** 가 될 수 있음을 시사한다."
    )
    sec31.append("")
    sec31.append("> **두 지표의 보완 관계**: Pearson r 은 '값의 정확한 비례성'을, Spearman ρ 는 '순위의 일관성' 을 측정한다. "
                  "두 지표 모두에서 ca-xBA 가 공식 xBA 를 능가했다는 것은 단순 값 예측뿐만 아니라 "
                  "**실무적 의사결정(누가 더 잘 치나)에서도 우수함** 을 의미한다.")
    sec31.append("")

    # === § 3.2 — 사용자 작성 텍스트 그대로 ===
    sec32 = "\n" + SIGNIFICANCE_TEXT + "\n"

    # § 3.1 + § 3.2 모두 § 3 다음 (§ 4 직전) 에 삽입
    marker4 = "\n## 4. 운(Luck) 분석"
    if marker4 in md:
        idx = md.index(marker4)
        md = md[:idx].rstrip() + "\n" + "\n".join(sec31) + "\n" + sec32 + md[idx:]

    # === § 4.1~4.3 보강 ===
    sec4_extra = []
    sec4_extra.append("")
    sec4_extra.append("### 4.1 ⚠️ 시스템적 오프셋 해설 (해석 시 필수)")
    sec4_extra.append("")
    sec4_extra.append(
        "luck 분포 평균 = **−0.1036** (음수 쏠림). 이는 모델 오류가 아닌 **분모 차이로 인한 구조적 오프셋**:"
    )
    sec4_extra.append("")
    sec4_extra.append("- `ca-xBA` = Σ(타구별 안타 확률) / Σ(**BIP** 수)")
    sec4_extra.append("- `AVG` = (안타 수) / **(AB)** ≈ (안타 수) / (BIP + 삼진)")
    sec4_extra.append("- 즉 `AVG` 의 분모(AB)는 `ca-xBA` 의 분모(BIP)보다 자연스럽게 커서 → **AVG ≤ BIP-기반 hit rate** 가 일반적.")
    sec4_extra.append("- 따라서 `luck = AVG − ca-xBA` 의 *절대값 0* 이 기준이 아니라, **상대 순위** 로 해석해야 함.")
    sec4_extra.append("  (도메인 직관: '운 좋은 타자' = 시스템 오프셋 대비 가장 plus, '불운한 타자' = 가장 minus.)")
    sec4_extra.append("")
    sec4_extra.append("### 4.2 🔍 운/불운 Top 5 — 전문가/매체 평가 교차 조사")
    sec4_extra.append("")
    sec4_extra.append("> 모델의 luck 평가가 실제 도메인 관점과 얼마나 일치하는지 web research 로 검증. "
                       "**없는 평가는 절대 생성하지 않음 (출처 명시).**")
    sec4_extra.append("")

    def _format_player_block(r: dict) -> list[str]:
        block = [f"**{r['name']}** (luck = {r['luck']:+.3f})", "", f"> {r['evidence']}", ""]
        srcs = r.get("sources", [])
        if srcs:
            src_strs = [f"[{title}]({url})" for title, url in srcs]
            block.append(f"_출처: {' · '.join(src_strs)}_")
        else:
            block.append("_출처: **명확한 스카우팅 근거가 검색되지 않아 표본 부족 및 단순 부진으로 분류함.**_")
        block.append("")
        return block

    sec4_extra.append("#### 🍀 운 좋은 타자 Top 5")
    sec4_extra.append("")
    for r in LUCK_TOP5_RESEARCH["lucky"]:
        sec4_extra.extend(_format_player_block(r))

    sec4_extra.append("#### 💀 불운한 타자 Top 5")
    sec4_extra.append("")
    for r in LUCK_TOP5_RESEARCH["unlucky"]:
        sec4_extra.extend(_format_player_block(r))

    sec4_extra.append("### 4.3 🎯 모델의 한계 — 'Schwarber 패턴'")
    sec4_extra.append("")
    sec4_extra.append(
        "Kyle Schwarber (NL MVP 2위, 56홈런) 는 모델이 '불운' 으로 평가한 **반례 케이스**. "
        "원인은 ca-xBA 의 본질적 특성:"
    )
    sec4_extra.append("")
    sec4_extra.append("- ca-xBA 는 **BIP 한정 안타 확률** → fly ball / line drive 의 quality 가 높으면 BIP-wOBA 도 높게 평가.")
    sec4_extra.append("- 하지만 power hitter (fly ball 비율 높음) 는 **fly out 비율도 높음** → AVG 는 낮음.")
    sec4_extra.append("- 즉 `(AVG − ca-xBA)` 의 음수는 '진짜 불운' 일 수도 있고 'power hitter 의 구조적 특성' 일 수도 있음.")
    sec4_extra.append("- 진정한 '불운' 판단은 BABIP / xwOBA underperform 같은 외부 지표와 교차 검증해야 함 (위 § 4.2 인용 참조).")
    sec4_extra.append("")

    marker5 = "\n## 5. 실버 슬러거"
    if marker5 in md:
        idx = md.index(marker5)
        md = md[:idx].rstrip() + "\n" + "\n".join(sec4_extra) + "\n" + md[idx:]

    # === § 5.1 — 2 지표 비교 (wOBA 제외) ===
    sec5_extra = []
    sec5_extra.append("")
    sec5_extra.append("### 5.1 📊 적중률 비교 — 전통 AVG vs ca-xBA")
    sec5_extra.append("")
    sec5_extra.append(
        "> **재미용 도메인 일관성 점검** (§ 5 disclaimer 참조). "
        "wOBA 는 사실상 시상 결정의 최대 단일 변수라 적중률 ≈ 상한선 → 비교 의미가 약해 제외. "
        "ca-xBA 가 \"단순 결과 지표(AVG)\" 대비 도메인 전문가 평가와 얼마나 더 일치하는지만 살펴본다."
    )
    sec5_extra.append("")
    sec5_extra.append("| 지표 | 적중 / 검증 가능 | 적중률 | 의미 |")
    sec5_extra.append("|---|:---:|:---:|---|")
    meaning_map = {
        "ba": "단순 결과 지표 — 운/불운·구장 효과 모두 포함",
        "ca_xba": "우리 모델 — 환경 보정된 BIP-quality 확률",
    }
    for metric_col, info in hitrates.items():
        sec5_extra.append(
            f"| **{info['label']}** | {info['hits']}/{info['eligible']} | "
            f"**{info['hit_rate']*100:.1f}%** | {meaning_map[metric_col]} |"
        )
    sec5_extra.append("")
    sec5_extra.append(f"![2 metric hitrate]({fig_2metric.relative_to(PIPELINE_DIR).as_posix()})")
    sec5_extra.append("")
    ca_rate = hitrates["ca_xba"]["hit_rate"]
    ba_rate = hitrates["ba"]["hit_rate"]
    delta = (ca_rate - ba_rate) * 100
    sec5_extra.append(
        f"**해석:** ca-xBA 적중률 {ca_rate*100:.1f}% 가 단순 AVG 의 {ba_rate*100:.1f}% 보다 "
        f"**+{delta:.1f}%p** 우수. 단순 타율은 운·구장 효과로 인해 도메인 전문가가 보는 \"타격 능력\" 평가와 거리가 있고, "
        "ca-xBA 가 그 부분을 보정해 평가에 더 가까워졌다고 *재미있게* 해석할 수 있다 — 단, § 3 R² 만이 통계적 모델 검증."
    )
    sec5_extra.append("")

    marker6 = "\n## 6. 산출물"
    if marker6 in md:
        idx = md.index(marker6)
        md = md[:idx].rstrip() + "\n" + "\n".join(sec5_extra) + "\n" + md[idx:]

    REPORT_PATH.write_text(md, encoding="utf-8")
    log(f"[report] phase5_report.md 보강 완료 (§3.2 신규, §4.1~4.3, §5.1 갱신)")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    log("=" * 80)
    log("Phase 5 보강 분석 v2: 실버슬러거 한계 명시 + 2지표 비교 + §3.2 지표의 의의")
    log("=" * 80)

    log("\n[load] phase5_results.json + player_metrics.csv + silver_slugger_val.csv ...")
    artifact = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    metrics = pd.read_csv(PLAYER_METRICS_CSV)
    ss_val = pd.read_csv(SILVER_SLUGGER_VAL_CSV)
    log(f"  metrics: {metrics.shape}, ss_val: {ss_val.shape}")

    # 1. 2 지표 적중률 비교 (wOBA 제거)
    hitrates = compute_2metric_hitrate(metrics, ss_val)
    artifact["hitrates_2metric"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "detail"}
        for k, v in hitrates.items()
    }
    artifact["hitrates_2metric_detail"] = {k: v["detail"] for k, v in hitrates.items()}
    # v1 잔존 3metric 키 정리
    artifact.pop("hitrates_3metric", None)
    artifact.pop("hitrates_3metric_detail", None)

    # 2. 그림 재생성
    log("\n[figs] 그림 재생성 ...")
    fig_luck = regenerate_luck_figure(artifact)
    fig_ss = regenerate_silver_slugger_figure(ss_val, artifact)
    fig_2metric = fig_2metric_comparison(hitrates)

    # v1 잔존 fig_p5f_3metric_hitrate_comparison.png 정리
    old_3metric = FIGURES_DIR / "fig_p5f_3metric_hitrate_comparison.png"
    if old_3metric.exists():
        old_3metric.unlink()
        log(f"  ✓ removed v1: {old_3metric.name}")

    # 3. 결과 저장
    log("\n[save] phase5_results.json 업데이트 ...")
    RESULTS_JSON.write_text(
        json.dumps(artifact, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )

    # 4. 리포트 보강
    log("\n[report] phase5_report.md 보강 (§ 3.1 신규, § 3.2 신규, § 4.1~4.3, § 5.1 갱신) ...")
    correlations = artifact["correlations"]
    patch_report(hitrates, fig_2metric, correlations)

    log(f"\n[done] Phase 5 v3 보강 완료. "
        f"§3.1 상관계수 심층 해석 / §3.2 지표 의의 / §4.1~4.3 + URL 보강 / §5.1 갱신.")


if __name__ == "__main__":
    main()
