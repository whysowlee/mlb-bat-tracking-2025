"""
Phase 1 시각화: phase1_report.md 에 임베드할 PNG 생성
=====================================================

사용자 선택 시각화 패키지(3종):
  A. 전처리 흐름        — attrition funnel + bb_type 컷오프 전후
  B. 타구 물리 분포      — launch_speed/angle 히스토그램 + 2D 안타율 히트맵
  C. 환경 변수          — 기상 8종 히스토그램 grid + 구장별 환경 히트맵

산출:
  - pipeline/figures/*.png
  - phase1_report.md 끝에 "## 10. 시각화" 섹션을 패치(이미 존재 시 갱신)

실행:
    /opt/miniconda3/envs/mlb-xba/bin/python pipeline/step1b_phase1_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# 경로 & 폰트
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "데이터셋"
PIPELINE_DIR = ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
FIGURES_DIR = PIPELINE_DIR / "figures"
REPORT_PATH = PIPELINE_DIR / "phase1_report.md"
STATCAST_CSV = DATA_DIR / "statcast_bat_tracking_2024_2025.csv"
BALLPARKS_CSV = DATA_DIR / "ballparks.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# macOS의 Apple SD Gothic Neo를 명시적으로 등록 (matplotlib 폰트 캐시 의존성 회피)
_KFONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
if Path(_KFONT_PATH).exists():
    fm.fontManager.addfont(_KFONT_PATH)
    _korean_name = fm.FontProperties(fname=_KFONT_PATH).get_name()
else:
    _korean_name = "AppleGothic"

# 1) seaborn 스타일을 먼저 적용해 폰트 rcParams가 덮이는 것을 방지
sns.set_style("whitegrid", {"axes.grid": True, "grid.alpha": 0.3})
# 2) seaborn 이후에 폰트를 강제 지정
plt.rcParams["font.family"] = _korean_name
plt.rcParams["font.sans-serif"] = [_korean_name, "AppleGothic", "Nanum Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

# Phase 1 결정 상수 (step1과 동일)
BB_TYPES_BIP = {"ground_ball", "fly_ball", "line_drive", "popup"}
EXCLUDE_TEAMS = {"ATH"}
LAUNCH_ANGLE_ABS_MAX = 60.0
WEATHER_VARS = [
    "wx_temperature_2m",
    "wx_relative_humidity_2m",
    "wx_surface_pressure",
    "wx_wind_speed_10m",
    "wx_wind_direction_10m",
    "wx_precipitation",
    "wx_cloud_cover",
    "wx_wind_gusts_10m",
]
WEATHER_KO = {
    "wx_temperature_2m": "기온 (°C)",
    "wx_relative_humidity_2m": "상대습도 (%)",
    "wx_surface_pressure": "지면기압 (hPa)",
    "wx_wind_speed_10m": "풍속 (km/h)",
    "wx_wind_direction_10m": "풍향 (°)",
    "wx_precipitation": "강수량 (mm)",
    "wx_cloud_cover": "운량 (%)",
    "wx_wind_gusts_10m": "돌풍 (km/h)",
}


# -----------------------------------------------------------------------------
# 데이터 로드
# -----------------------------------------------------------------------------
def load_final_df() -> pd.DataFrame:
    df24 = pd.read_parquet(OUTPUT_DIR / "2024_data.parquet")
    df25 = pd.read_parquet(OUTPUT_DIR / "2025_data.parquet")
    return pd.concat([df24, df25], ignore_index=True)


def compute_attrition() -> list[dict]:
    """원본 CSV를 다시 읽어 단계별 행 수를 계산."""
    cols = ["bb_type", "home_team", "launch_speed", "launch_angle"]
    df = pd.read_csv(STATCAST_CSV, usecols=cols, low_memory=False)
    steps = []
    steps.append({"label": "원본 로드", "rows": len(df)})

    df = df[df["bb_type"].isin(BB_TYPES_BIP)]
    steps.append({"label": "BIP 필터", "rows": len(df)})

    df = df[~df["home_team"].isin(EXCLUDE_TEAMS)]
    steps.append({"label": "ATH 제외", "rows": len(df)})

    df = df[df["launch_speed"].notna() & df["launch_angle"].notna()]
    steps.append({"label": "물리 결측 제거", "rows": len(df)})

    df_after_cutoff = df[df["launch_angle"].abs() <= LAUNCH_ANGLE_ABS_MAX]
    steps.append({"label": f"|launch_angle|≤{int(LAUNCH_ANGLE_ABS_MAX)}°", "rows": len(df_after_cutoff)})

    # bb_type before/after 컷오프 (ATH/결측 제거 후)
    bb_before = df["bb_type"].value_counts()
    bb_after = df_after_cutoff["bb_type"].value_counts()
    return steps, bb_before, bb_after


# -----------------------------------------------------------------------------
# A. 전처리 흐름
# -----------------------------------------------------------------------------
def fig_attrition_funnel(steps: list[dict]) -> Path:
    out = FIGURES_DIR / "fig_a1_attrition_funnel.png"
    labels = [s["label"] for s in steps]
    rows = [s["rows"] for s in steps]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels, rows, color="#4C72B0", edgecolor="white")
    ax.invert_yaxis()
    for bar, n in zip(bars, rows):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {n:,d}",
            va="center",
            fontsize=10,
        )
    # 단계 제거량 주석
    for i in range(1, len(rows)):
        removed = rows[i - 1] - rows[i]
        if removed:
            ax.text(
                rows[i] / 2,
                i,
                f"-{removed:,d}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
    ax.set_xlabel("행 수")
    ax.set_title("Phase 1 전처리 단계별 행 수 변화 (Attrition Funnel)")
    ax.set_xlim(0, max(rows) * 1.18)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_bb_type_before_after(bb_before: pd.Series, bb_after: pd.Series) -> Path:
    out = FIGURES_DIR / "fig_a2_bb_type_before_after.png"
    order = ["ground_ball", "line_drive", "fly_ball", "popup"]
    before_vals = [int(bb_before.get(b, 0)) for b in order]
    after_vals = [int(bb_after.get(b, 0)) for b in order]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(order))
    w = 0.4
    ax.bar(x - w / 2, before_vals, w, label="컷오프 전 (BIP+ATH제외+물리결측제거)", color="#888888")
    ax.bar(x + w / 2, after_vals, w, label="|launch_angle|≤60° 적용 후", color="#4C72B0")
    for i, (b, a) in enumerate(zip(before_vals, after_vals)):
        removed = b - a
        if removed:
            ax.text(
                i,
                max(b, a) * 1.02,
                f"-{removed:,d}",
                ha="center",
                fontsize=9,
                color="#c44",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("행 수")
    ax.set_title("bb_type 분포 — |launch_angle|≤60° 컷오프 전후 비교")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# B. 타구 물리 분포
# -----------------------------------------------------------------------------
def fig_launch_speed_angle_hist(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "fig_b1_launch_speed_angle_hist.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Launch speed
    ax = axes[0]
    for yr, color in [(2024, "#4C72B0"), (2025, "#DD8452")]:
        sub = df.loc[df["game_year"] == yr, "launch_speed"]
        ax.hist(sub, bins=60, alpha=0.55, label=str(yr), color=color, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("launch_speed (mph)")
    ax.set_ylabel("BIP 행 수")
    ax.set_title("타구 발사 속도 분포 (BIP 정제 후)")
    ax.legend(title="game_year")

    # Launch angle
    ax = axes[1]
    for yr, color in [(2024, "#4C72B0"), (2025, "#DD8452")]:
        sub = df.loc[df["game_year"] == yr, "launch_angle"]
        ax.hist(sub, bins=60, alpha=0.55, label=str(yr), color=color, edgecolor="white", linewidth=0.3)
    ax.axvline(60, color="#c44", linestyle="--", linewidth=1, label="±60° 컷오프 경계")
    ax.axvline(-60, color="#c44", linestyle="--", linewidth=1)
    ax.set_xlabel("launch_angle (도)")
    ax.set_ylabel("BIP 행 수")
    ax.set_title("타구 발사 각도 분포 (컷오프 적용 후 범위 [-60, +60])")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_launch_speed_angle_heatmap(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "fig_b2_speed_angle_hit_heatmap.png"
    speed_bins = np.arange(40, 121, 2.5)
    angle_bins = np.arange(-60, 61, 3)

    df = df.copy()
    df["sp_bin"] = pd.cut(df["launch_speed"], bins=speed_bins)
    df["la_bin"] = pd.cut(df["launch_angle"], bins=angle_bins)

    grid_rate = (
        df.groupby(["la_bin", "sp_bin"], observed=True)["is_hit"]
        .mean()
        .unstack("sp_bin")
    )
    grid_count = (
        df.groupby(["la_bin", "sp_bin"], observed=True)["is_hit"]
        .size()
        .unstack("sp_bin")
    )
    # 표본 적은 셀(<30)은 마스킹
    grid_rate = grid_rate.where(grid_count >= 30)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.heatmap(
        grid_rate,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "안타율 (is_hit 평균)"},
        ax=ax,
        linewidths=0,
        xticklabels=[f"{int(b.left)}" for b in grid_rate.columns],
        yticklabels=[f"{int(b.left)}" for b in grid_rate.index],
    )
    ax.invert_yaxis()
    ax.set_xlabel("launch_speed bin 시작 (mph)")
    ax.set_ylabel("launch_angle bin 시작 (도)")
    ax.set_title(
        "launch_speed × launch_angle 안타율 히트맵 (셀 표본 ≥ 30)\n"
        "xBA의 본질: 발사속도와 발사각 조합이 안타 확률을 결정"
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# C. 환경 변수
# -----------------------------------------------------------------------------
def fig_weather_grid(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "fig_c1_weather_distributions.png"
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.flat, WEATHER_VARS):
        sub = df[col].dropna()
        ax.hist(sub, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.set_title(WEATHER_KO[col], fontsize=10)
        ax.set_ylabel("행 수")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("기상 변수 분포 (BIP 행 단위, snapshot: daytime≥0.5→13시 / 그 외 19시)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_park_env_heatmap(df: pd.DataFrame) -> Path:
    """29구장 × 환경 변수(고도, 펜스 효과, 평균기온, 평균풍속, 강수, 운량 등) z-score 히트맵."""
    out = FIGURES_DIR / "fig_c2_park_env_heatmap.png"

    park_static = (
        df.groupby("home_team")
        .agg(
            elevation=("elevation", "first"),
            hr_park_effects=("hr_park_effects", "first"),
            extra_distance=("extra_distance", "first"),
            avg_temp_csv=("avg_temp", "first"),
            roof=("roof", "first"),
        )
    )
    park_wx = (
        df.groupby("home_team")
        .agg(
            wx_temp=("wx_temperature_2m", "mean"),
            wx_humidity=("wx_relative_humidity_2m", "mean"),
            wx_pressure=("wx_surface_pressure", "mean"),
            wx_wind=("wx_wind_speed_10m", "mean"),
            wx_precip=("wx_precipitation", "mean"),
            wx_cloud=("wx_cloud_cover", "mean"),
        )
    )
    grid = pd.concat([park_static, park_wx], axis=1)
    grid.columns = [
        "고도(ft)",
        "HR_park_eff",
        "extra_dist",
        "avg_temp(csv)",
        "roof",
        "기온(°C)",
        "습도(%)",
        "기압(hPa)",
        "풍속(km/h)",
        "강수(mm)",
        "운량(%)",
    ]
    # z-score 정규화 (열 단위)로 다변량 동시 가독성
    grid_z = (grid - grid.mean()) / grid.std(ddof=0)

    fig, ax = plt.subplots(figsize=(11.5, 9))
    sns.heatmap(
        grid_z,
        cmap="RdBu_r",
        center=0,
        vmin=-2.5,
        vmax=2.5,
        annot=grid.round(1),
        fmt="",
        annot_kws={"fontsize": 7},
        ax=ax,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "z-score (열별 표준화)"},
    )
    ax.set_title(
        "구장별 환경 특성 히트맵 (29 home_team × 11변수)\n"
        "셀 숫자는 원본값, 색상은 열 z-score (RdBu_r, +가 더 큼)"
    )
    ax.set_xlabel("")
    ax.set_ylabel("home_team")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# 리포트 패치
# -----------------------------------------------------------------------------
def patch_report(figure_section: str):
    md = REPORT_PATH.read_text(encoding="utf-8")
    marker = "## 10. 시각화"
    if marker in md:
        idx = md.index(marker)
        md = md[:idx].rstrip() + "\n\n" + figure_section.rstrip() + "\n"
    else:
        md = md.rstrip() + "\n\n" + figure_section.rstrip() + "\n"
    REPORT_PATH.write_text(md, encoding="utf-8")


def build_figure_section(figs: dict[str, Path]) -> str:
    def rel(p: Path) -> str:
        return p.relative_to(PIPELINE_DIR).as_posix()

    lines = []
    lines.append("## 10. 시각화")
    lines.append("")
    lines.append("아래 PNG는 모두 `pipeline/figures/` 에 저장되어 있으며, 최종 Word 보고서로 옮길 때 그대로 재사용 가능하다.")
    lines.append("")

    lines.append("### 10.1 전처리 흐름")
    lines.append("")
    lines.append("**(A1) 단계별 행 수 변화 — Attrition Funnel**")
    lines.append("")
    lines.append(f"![Attrition Funnel]({rel(figs['attrition'])})")
    lines.append("")
    lines.append("- 원본 1,443,801행 중 BIP 필터 단계에서 **약 82.5%**가 제거됨(타격 외 pitch 단위 행).")
    lines.append("- 도메인 컷오프는 |launch_angle|≤60°가 가장 큰 단일 컷(−16,248행).")
    lines.append("")

    lines.append("**(A2) bb_type 분포 — 컷오프 전후**")
    lines.append("")
    lines.append(f"![bb_type before/after]({rel(figs['bb_type'])})")
    lines.append("")
    lines.append("- popup이 컷오프로 거의 전량 제거됨 → 평균 launch_angle=65.8°, 안타율 1.4%의 사실상 자동 아웃 군집 제거.")
    lines.append("- ground_ball / line_drive / fly_ball 의 페어 영역은 보존.")
    lines.append("")

    lines.append("### 10.2 타구 물리 분포")
    lines.append("")
    lines.append("**(B1) 발사속도/발사각 히스토그램 (연도별 overlay)**")
    lines.append("")
    lines.append(f"![Launch speed/angle hist]({rel(figs['ls_la_hist'])})")
    lines.append("")
    lines.append("- 2024와 2025의 분포가 거의 동일 — Temporal Split 후에도 입력 변수의 분포가 안정적임을 시각적으로 확인.")
    lines.append("- launch_angle은 컷오프 적용으로 [−60, +60] 범위에 갇혀있음(붉은 점선 = 컷오프 경계).")
    lines.append("")

    lines.append("**(B2) launch_speed × launch_angle 안타율 히트맵**")
    lines.append("")
    lines.append(f"![Speed×Angle heatmap]({rel(figs['heatmap'])})")
    lines.append("")
    lines.append("- xBA의 본질 시각화: 발사속도가 빠르고 각도가 약 10~25°일 때 안타율이 가장 높음(녹색 띠).")
    lines.append("- 위 띠 위(40°+, 낮은 EV)는 팝업 영역, 아래 띠(음각·낮은 EV)는 그라운드 아웃 영역으로 안타율이 급락.")
    lines.append("- 환경 변수(바람·기압·온도 등)는 이 *비선형 의존성* 위에 추가 보정을 제공할 것이 Phase 3 가설.")
    lines.append("")

    lines.append("### 10.3 환경 변수")
    lines.append("")
    lines.append("**(C1) 기상 변수 8종 분포 grid**")
    lines.append("")
    lines.append(f"![Weather distributions]({rel(figs['weather'])})")
    lines.append("")
    lines.append("- 기온은 약 23°C에 중심한 정규-유사 분포, 풍속/풍향은 우측 꼬리·균등 분포.")
    lines.append("- 강수는 강한 영(0) 집중 — 변수 변환(예: log1p) 또는 이진화가 Phase 2에서 검토될 수 있음.")
    lines.append("")

    lines.append("**(C2) 구장별 환경 특성 히트맵 (29 home_team × 11 변수)**")
    lines.append("")
    lines.append(f"![Park env heatmap]({rel(figs['park_env'])})")
    lines.append("")
    lines.append("- COL(쿠어스): 고도 5,190ft / 평균기온 정상 / 풍속 정상 — *고도 단일 변수*로 분리되는 극단 구장.")
    lines.append("- MIA·TB·HOU·TEX·AZ·TOR·MIL: roof로 환경 영향이 일부/전면 차단(셀 색이 다른 환경 변수에서 두드러짐).")
    lines.append("- 환경 변수들이 구장 간에 의미 있는 분산을 가지며 — ca-xBA가 추출하려는 *환경 신호*의 원천이 확인됨.")
    lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    print("[load] 최종 데이터 로드 (2024 + 2025 parquet)...")
    df = load_final_df()
    print(f"  → {len(df):,d} rows")

    print("[stat] attrition 단계 재계산 중...")
    steps, bb_before, bb_after = compute_attrition()
    for s in steps:
        print(f"  {s['label']:<30s} {s['rows']:>10,d}")

    print("[figs] PNG 생성 중...")
    figs = {
        "attrition": fig_attrition_funnel(steps),
        "bb_type": fig_bb_type_before_after(bb_before, bb_after),
        "ls_la_hist": fig_launch_speed_angle_hist(df),
        "heatmap": fig_launch_speed_angle_heatmap(df),
        "weather": fig_weather_grid(df),
        "park_env": fig_park_env_heatmap(df),
    }
    for name, path in figs.items():
        print(f"  ✓ {name:12s} → {path.relative_to(ROOT)}")

    print("[report] phase1_report.md 패치 중...")
    section = build_figure_section(figs)
    patch_report(section)
    print(f"  ✓ 패치 완료 → {REPORT_PATH.relative_to(ROOT)}")

    print("\n[done] 시각화 완료.")


if __name__ == "__main__":
    main()
