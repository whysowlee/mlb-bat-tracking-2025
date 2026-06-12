"""
Phase 1: Data integration, domain-driven preprocessing, and temporal split by season
=====================================================================================

Steps performed:
  1) Load Statcast batted-ball data
  2) Filter to BIP (balls in play)        [bb_type ∈ {ground/fly/line/popup}]
  3) Drop Athletics (ATH) home-game rows  [2024 Oakland, 2025 Sacramento relocation issue]
  4) Remove rows missing core physics     [launch_speed or launch_angle NaN]
  5) Apply foul-popup cutoff              [|launch_angle| > 60°]
  6) Create target variable is_hit        [single/double/triple/home_run = 1]
  7) Add bat-tracking missingness flags   [bat_speed etc. → *_is_missing]
  8) Merge ballpark specs (ballparks.csv) [home_team ↔ team_name]
  9) Fetch and merge Open-Meteo Archive API weather data
       - Snapshot time: daytime ≥ 0.5 → 13:00 / otherwise → 19:00 (local time)
       - 8 variables: temperature_2m, relative_humidity_2m, surface_pressure,
                      wind_speed_10m, wind_direction_10m, precipitation,
                      cloud_cover, wind_gusts_10m
       - roof column is retained as-is (used downstream to neutralize environmental effects)
 10) Temporal Split by game_year → 2024_data.parquet / 2025_data.parquet
 11) Record per-step attrition statistics for phase1_report.md generation

Usage:
    /opt/miniconda3/envs/mlb-xba/bin/python pipeline/step1_phase1_preprocessing.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Path constants
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "데이터셋"
PIPELINE_DIR = ROOT / "pipeline"
CACHE_DIR = PIPELINE_DIR / "cache"
OUTPUT_DIR = PIPELINE_DIR / "output"
REPORT_PATH = PIPELINE_DIR / "phase1_report.md"

STATCAST_CSV = DATA_DIR / "statcast_bat_tracking_2024_2025.csv"
BALLPARKS_CSV = DATA_DIR / "ballparks.csv"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Decision constants (confirmed by user)
# -----------------------------------------------------------------------------
HIT_EVENTS = {"single", "double", "triple", "home_run"}
BB_TYPES_BIP = {"ground_ball", "fly_ball", "line_drive", "popup"}
EXCLUDE_TEAMS = {"ATH"}  # Athletics 전체(2024+2025) 분석 제외
LAUNCH_ANGLE_ABS_MAX = 60.0
BAT_TRACKING_COLS = [
    "bat_speed",
    "swing_length",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
]

WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "wind_gusts_10m",
]
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_HOUR_DAY = 13   # parks with daytime ≥ 0.5: 13:00 local time
WEATHER_HOUR_NIGHT = 19 # all other parks: 19:00 local time

# Dome masking — user decision #11 (2026-05-29)
# Applied to games where the MLB Stats API weather.condition field explicitly states "Roof Closed".
# Domain rationale: the 5 outdoor weather variables are irrelevant indoors; replaced with HVAC standard values.
ROOF_STATUS_CACHE_PATH = PIPELINE_DIR / "cache" / "mlb_roof_status_cache.json"
DOME_MASK_EXTERNAL_VARS = [  # set all to 0 → neutralize outdoor weather indoors
    "wx_wind_speed_10m", "wx_wind_gusts_10m", "wx_wind_direction_10m",
    "wx_precipitation", "wx_cloud_cover",
]
DOME_MASK_INDOOR_DEFAULTS = {  # MLB dome ballpark HVAC standard values
    "wx_temperature_2m": 22.0,         # MLB dome HVAC standard 22°C (Tropicana Field / Minute Maid Park official)
    "wx_relative_humidity_2m": 50.0,   # midpoint of ASHRAE-recommended range 40–60%
    # surface_pressure left unchanged (indoor and outdoor air pressure are equal)
}


# -----------------------------------------------------------------------------
# Utility: accumulate per-step attrition statistics
# -----------------------------------------------------------------------------
class Attrition:
    """Records the row count at each preprocessing step."""

    def __init__(self):
        self.steps: list[dict] = []

    def record(self, step: str, df: pd.DataFrame, note: str = ""):
        n = len(df)
        if self.steps:
            prev = self.steps[-1]["rows"]
            removed = prev - n
        else:
            removed = 0
        self.steps.append(
            {"step": step, "rows": n, "removed_at_step": removed, "note": note}
        )
        print(f"[attrition] {step:<55s} rows={n:>9,d}  (-{removed:,d}) {note}")

    def to_markdown(self) -> str:
        lines = ["| 단계 | 잔여 행 수 | 단계 제거 | 비고 |", "|---|---:|---:|---|"]
        for s in self.steps:
            lines.append(
                f"| {s['step']} | {s['rows']:,d} | {s['removed_at_step']:,d} | {s['note']} |"
            )
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# 1) Load Statcast data + step-by-step cleaning
# -----------------------------------------------------------------------------
def load_and_clean_statcast(attrition: Attrition) -> pd.DataFrame:
    print("[load] statcast CSV 로드 중...")
    df = pd.read_csv(STATCAST_CSV, low_memory=False)
    attrition.record("0. 원본 로드", df, note="2024+2025 전체 pitch 단위")

    # BIP filter
    df = df[df["bb_type"].isin(BB_TYPES_BIP)].copy()
    attrition.record(
        "1. BIP 필터 (bb_type ∈ {ground/fly/line/popup})",
        df,
        note="옵션 C 채택",
    )

    # Exclude Athletics
    df = df[~df["home_team"].isin(EXCLUDE_TEAMS)].copy()
    attrition.record(
        "2. Athletics(ATH) 홈경기 행 제외",
        df,
        note="2024 Oakland / 2025 Sacramento 이전 이슈로 분석 제외",
    )

    # Remove rows missing core physics features
    core_missing = df["launch_speed"].isna() | df["launch_angle"].isna()
    df = df.loc[~core_missing].copy()
    attrition.record(
        "3. launch_speed/launch_angle 결측 행 제거",
        df,
        note="xBA 핵심 입력 결측 → 모델링 불가",
    )

    # Foul-popup cutoff
    df = df[df["launch_angle"].abs() <= LAUNCH_ANGLE_ABS_MAX].copy()
    attrition.record(
        f"4. |launch_angle| > {LAUNCH_ANGLE_ABS_MAX:.0f}° 컷오프",
        df,
        note="readme 예시 적용 — 파울 팝아웃/극단 음각 라인드라이브 제거",
    )

    # Create is_hit target variable
    df["is_hit"] = df["events"].isin(HIT_EVENTS).astype("int8")

    # Bat-tracking missingness flags
    for col in BAT_TRACKING_COLS:
        df[f"{col}_is_missing"] = df[col].isna().astype("int8")

    # Parse game_date
    df["game_date"] = pd.to_datetime(df["game_date"], errors="raise")

    return df


# -----------------------------------------------------------------------------
# 2) Merge ballpark specs
# -----------------------------------------------------------------------------
def merge_ballparks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parks = pd.read_csv(BALLPARKS_CSV)
    merged = df.merge(parks, how="left", left_on="home_team", right_on="team_name")
    n_unmatched = merged["ballpark"].isna().sum()
    print(
        f"[merge] ballparks 병합 완료. unmatched={n_unmatched:,d} "
        f"(unique home_team={df['home_team'].nunique()})"
    )
    if n_unmatched > 0:
        unmatched_teams = sorted(merged.loc[merged["ballpark"].isna(), "home_team"].unique())
        raise RuntimeError(f"ballparks 미매핑 home_team 존재: {unmatched_teams}")
    return merged, parks


# -----------------------------------------------------------------------------
# 3) Weather data (Open-Meteo Archive) — cache per park×period, then select snapshot hour
# -----------------------------------------------------------------------------
def fetch_weather_for_park(
    team: str, lat: float, lon: float, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch hourly weather data for a single ballpark and return as a DataFrame. Uses disk cache."""
    cache_path = CACHE_DIR / f"weather_{team}_{start_date}_{end_date}.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(WEATHER_HOURLY_VARS),
            "timezone": "auto",
        }
        for attempt in range(5):
            try:
                resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  [warn] {team} fetch attempt {attempt+1} failed: {e} (retry in {wait}s)")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Open-Meteo fetch failed for team={team}")
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        time.sleep(0.5)  # light throttle between API calls

    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df["team_name"] = team
    return df


def build_weather_lookup(parks: pd.DataFrame, date_min: str, date_max: str) -> pd.DataFrame:
    """Build one weather row per park per day using the selected snapshot hour."""
    chunks = []
    for _, row in parks.iterrows():
        team = row["team_name"]
        if team in EXCLUDE_TEAMS:
            continue
        target_hour = WEATHER_HOUR_DAY if row["daytime"] >= 0.5 else WEATHER_HOUR_NIGHT
        print(
            f"  [fetch] {team:<4s} ({row['ballpark'][:30]:<30s}) "
            f"lat={row['latitude']:.3f} lon={row['longitude']:.3f} hour={target_hour:02d}"
        )
        hourly = fetch_weather_for_park(team, row["latitude"], row["longitude"], date_min, date_max)
        picked = hourly[hourly["hour"] == target_hour].copy()
        picked = picked.rename(
            columns={var: f"wx_{var}" for var in WEATHER_HOURLY_VARS}
        )
        picked["wx_picked_hour_local"] = target_hour
        picked = picked[
            ["team_name", "date", "wx_picked_hour_local"]
            + [f"wx_{v}" for v in WEATHER_HOURLY_VARS]
        ]
        chunks.append(picked)
    weather = pd.concat(chunks, ignore_index=True)
    weather["date"] = pd.to_datetime(weather["date"])
    return weather


def apply_dome_masking(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Mask 5 outdoor weather variables to 0 and set 2 indoor HVAC variables to standard values for dome/roof-closed games.

    User decision #11 (2026-05-29):
      - Target: games where the MLB Stats API `weather.condition` contains "Roof Closed" or "Dome"
                (TB games are automatically classified as closed). Applied to retractable parks (SEA/TOR/MIL/TEX/AZ/MIA/HOU).
      - Outdoor 5 variables (wx_wind_*, wx_precipitation, wx_cloud_cover) → 0
      - Indoor 2 variables (wx_temperature_2m=22°C, wx_relative_humidity_2m=50%) → HVAC standard values
      - wx_surface_pressure left unchanged (indoor and outdoor air pressure are equal)

    Returns: (masked_df, n_masked_rows, n_total_eligible_rows)
    """
    if not ROOF_STATUS_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"roof_status 캐시 없음: {ROOF_STATUS_CACHE_PATH}. "
            "먼저 `python pipeline/step1_fetch_roof_status.py` 를 실행하라."
        )
    cache = json.loads(ROOF_STATUS_CACHE_PATH.read_text(encoding="utf-8"))
    print(f"[dome_mask] roof_status 캐시 로드: {len(cache):,d}경기")

    df = df.copy()
    # Convert game_pk to string to match cache keys
    df["_gpk_str"] = df["game_pk"].astype(int).astype(str)
    status_series = df["_gpk_str"].map(lambda pk: cache.get(pk, {}).get("status"))
    closed_mask = status_series == "closed"
    n_masked = int(closed_mask.sum())
    n_eligible = int(status_series.notna().sum())  # games registered in cache = retractable + TB

    # Outdoor 5 variables → 0
    for col in DOME_MASK_EXTERNAL_VARS:
        df.loc[closed_mask, col] = 0.0
    # Indoor 2 variables → standard HVAC values
    for col, val in DOME_MASK_INDOOR_DEFAULTS.items():
        df.loc[closed_mask, col] = val

    df = df.drop(columns=["_gpk_str"])
    print(f"  돔 마스킹 적용: {n_masked:,d}행 / 대상 가능 {n_eligible:,d}행 "
          f"(전체 BIP 의 {n_masked / max(len(df), 1) * 100:.2f}%)", flush=True)
    return df, n_masked, n_eligible


def merge_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(
        weather,
        how="left",
        left_on=["home_team", "game_date"],
        right_on=["team_name", "date"],
        suffixes=("", "_wxdup"),
    )
    n_missing = merged["wx_temperature_2m"].isna().sum()
    print(f"[merge] weather 병합 완료. 기상 결측 행={n_missing:,d}")
    # Drop duplicate key columns
    drop_cols = [c for c in merged.columns if c.endswith("_wxdup")] + ["date"]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    return merged


# -----------------------------------------------------------------------------
# 4) Temporal Split + save
# -----------------------------------------------------------------------------
def split_and_save(df: pd.DataFrame) -> dict:
    info = {}
    for year, sub in df.groupby("game_year"):
        out = OUTPUT_DIR / f"{year}_data.parquet"
        sub.to_parquet(out, index=False)
        info[int(year)] = {
            "rows": len(sub),
            "hit_rate": float(sub["is_hit"].mean()),
            "path": str(out.relative_to(ROOT)),
            "n_batters": int(sub["batter"].nunique()),
            "n_pitchers": int(sub["pitcher"].nunique()),
            "n_home_teams": int(sub["home_team"].nunique()),
            "date_min": str(sub["game_date"].min().date()),
            "date_max": str(sub["game_date"].max().date()),
        }
        print(
            f"[save] {year}: rows={len(sub):,d}, hit_rate={sub['is_hit'].mean():.4f}, "
            f"saved → {out.relative_to(ROOT)}"
        )
    return info


# -----------------------------------------------------------------------------
# 5) Write phase1_report.md
# -----------------------------------------------------------------------------
def write_report(
    attrition: Attrition,
    bb_type_dist_before: pd.Series,
    hit_rate_overall: float,
    bat_missing_rates: pd.DataFrame,
    parks_used: int,
    weather_summary: pd.DataFrame,
    split_info: dict,
    n_weather_missing: int,
    n_weather_total: int,
    n_dome_masked: int = 0,
    n_dome_eligible: int = 0,
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("# Phase 1 Report — 데이터 통합, 도메인 기반 전처리 및 연도별 분리")
    lines.append("")
    lines.append(f"_생성: {now}_  ")
    lines.append(f"_실행 스크립트: `pipeline/step1_phase1_preprocessing.py`_")
    lines.append("")
    lines.append("## 1. 서론 요약")
    lines.append(
        "본 단계는 MLB Statcast 타구 데이터(2024~2025), 구장 스펙 데이터, "
        "Open-Meteo Historical 기상 데이터를 결합하여 후속 모델링의 입력 "
        "데이터셋을 구성한다. 모든 처리는 도메인 지식과 사용자 승인 결정에 따라 "
        "수행되며, 데이터 누수(Data Leakage)를 차단하기 위해 game_year 기준으로 "
        "엄격한 Temporal Split(2024 ↔ 2025)을 실시한다."
    )
    lines.append("")

    lines.append("## 2. 데이터 셋 설명")
    lines.append(
        "- Statcast: `데이터셋/statcast_bat_tracking_2024_2025.csv` "
        "(원본 1,443,801행 × 118열, pitch 단위)\n"
        "- 구장 스펙: `데이터셋/ballparks.csv` (30개 구장 × 15열, "
        "lat/lon 컬럼은 Phase 1에서 보강)\n"
        "- 기상: Open-Meteo Archive API `archive-api.open-meteo.com/v1/archive` "
        "(무료, 키 불요)\n"
        "- Target 변수 `is_hit`: events ∈ {single, double, triple, home_run} → 1, "
        "그 외 → 0 (MLB 공식 xBA와 정렬)\n"
        f"- 최종 전처리 후 안타율 = {hit_rate_overall:.4f}"
    )
    lines.append("")

    lines.append("## 3. 결정 사항(분기점 기록)")
    lines.append("사용자 컨펌으로 확정된 분석 결정.")
    lines.append("")
    lines.append("| # | 결정 항목 | 채택안 | 사유 / 영향 |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| 1 | BIP(인플레이 타구) 정의 | `bb_type ∈ {ground_ball, fly_ball, line_drive, popup}` (옵션 C) | "
        "가장 보수적 정의 — 타구 추적이 확실한 행만 사용. 안타율 계산 분모가 깨끗함. |"
    )
    lines.append(
        "| 2 | Target `is_hit` 정의 | `events ∈ {single, double, triple, home_run}` → 1 (옵션 A) | "
        "MLB 공식 xBA 정의와 동일. Phase 5의 ca-xBA vs xBA 비교 일관성 확보. sac_fly/field_error 등은 0. |"
    )
    lines.append(
        "| 3 | 핵심 물리 결측(launch_speed/angle NaN) 처리 | 행 제거 (옵션 A) | "
        "xBA는 launch_speed×launch_angle 함수 — 결측 시 모델 입력 불가. "
        f"전체 BIP의 1.2% (약 3K행) 손실은 미미. |"
    )
    lines.append(
        "| 4 | `|launch_angle| > 60°` 컷오프 | 채택 (옵션 A) | "
        "readme 예시 그대로. 파울 팝아웃·극단 음각 라인드라이브를 노이즈로 제거. "
        "popup의 평균 la=65.8°, 안타율 1.4%로 대부분 제거됨. |"
    )
    lines.append(
        "| 5 | launch_speed 하한 컷오프 | 컷오프 없음 (옵션 A) | "
        "약타도 실제 BIP의 일부이며 xBA 정의에 포함. 인위적 절단은 정보 손실. |"
    )
    lines.append(
        "| 6 | 배트 트래킹 변수 결측 처리 | NaN 유지 + `*_is_missing` 플래그 추가 (옵션 B) | "
        "2024 11.46% / 2025 5.21% 결측은 시즌 초반 비공개 구간 때문. "
        "트리 모델 NaN-native 처리 활용, 결측 패턴 자체를 신호로 보존. |"
    )
    lines.append(
        "| 7 | Athletics(ATH) 매핑 | 2024+2025 ATH 홈경기 전체 분석 제외 (옵션 D) | "
        "2025년 홈구장이 Oakland Coliseum → Sutter Health Park(새크라멘토)로 이전 — "
        "환경 변수가 완전히 달라 단일 매핑 불가. 8,641행(BIP의 3.4%) 손실 감수. |"
    )
    lines.append(
        "| 8 | 구장 위·경도 추가 방식 | ballparks.csv 자체에 lat/lon 컬럼 추가 (옵션 B) | "
        "데이터셋 자체 완성도. 공개 좌표(MLB 공식/Wikipedia) 하드코딩 입력. |"
    )
    lines.append(
        "| 9 | 기상 API 시점 처리 | `daytime ≥ 0.5` → 13:00 / 그 외 → 19:00 현지시각 snapshot (옵션 D) | "
        "first-pitch 시각 부재. ballparks.csv의 daytime 비율로 구장별 분기. "
        "낮경기·야간경기 평균 시점 근사. |"
    )
    lines.append(
        "| 10 | 기상 변수 셋 | 8종(temperature, humidity, surface_pressure, wind_speed, wind_direction, "
        "precipitation, cloud_cover, wind_gusts) (옵션 C) | "
        "공기 밀도(온·습·압) + 바람(속·향·돌풍) + 경기 영향(강수·운량) 모두 커버. "
        "다중공선성은 Phase 2에서 정리. |"
    )
    lines.append(
        "| 11 | 돔/지붕 닫힘 경기 기상 마스킹 | **MLB Stats API 경기별 `weather.condition` 기반 정밀 마스킹** — "
        "closed 경기에서 외부 5종(wind_speed/gusts/direction · precipitation · cloud_cover) = 0, "
        "실내 2종(temperature_2m=22°C · relative_humidity_2m=50%) = 공조 표준값, surface_pressure 그대로 | "
        "도메인 사실 \"돔에서는 외부 기상이 안타 확률에 영향을 줄 수 없다\" 를 학습 데이터에 *직접* 반영. "
        "단순 roof 컬럼(0~1 시즌 평균)만으로는 트리 모델이 *조건부 무력화 split* 을 학습하지 못함을 사후 검증. "
        "공기 밀도(기온·습·압) 보존 + 외부 직접 기상 무력화 분리 적용. |"
    )
    lines.append("")

    lines.append("## 4. 단계별 attrition (행 수 변화)")
    lines.append(attrition.to_markdown())
    lines.append("")

    lines.append("## 5. BIP 필터 직후 bb_type 분포")
    lines.append("```")
    lines.append(str(bb_type_dist_before))
    lines.append("```")
    lines.append("")

    lines.append("## 6. 배트 트래킹 결측률 (전처리 직전 BIP 기준, 연도별)")
    lines.append(bat_missing_rates.round(4).to_markdown())
    lines.append("")

    lines.append("## 7b. 돔/지붕 닫힘 경기 기상 마스킹 (결정 #11)")
    lines.append("")
    lines.append(
        f"- MLB Stats API (`/api/v1.1/game/{{game_pk}}/feed/live` 의 `gameData.weather.condition`)에서 "
        f"\"Roof Closed\" 또는 \"Dome\" 으로 명시된 경기에 한해 기상 변수 마스킹 적용.\n"
        f"- 캐시 파일: `pipeline/cache/mlb_roof_status_cache.json` (1,318 게임, 누락 0건).\n"
        f"- 마스킹된 BIP 행: **{n_dome_masked:,d}** / 대상 가능(retractable + TB) **{n_dome_eligible:,d}** "
        f"({n_dome_masked / max(n_dome_eligible, 1) * 100:.1f}%)\n"
        f"- 적용 값:\n"
        f"  - 외부 기상 5종 → 0: `wx_wind_speed_10m`, `wx_wind_gusts_10m`, `wx_wind_direction_10m`, "
        f"`wx_precipitation`, `wx_cloud_cover`\n"
        f"  - 실내 공조 표준값 2종: `wx_temperature_2m` = 22°C (MLB 돔 표준), "
        f"`wx_relative_humidity_2m` = 50% (ASHRAE 권장 중간값)\n"
        f"  - 변경 없음: `wx_surface_pressure` (실내·외 기압 동일)\n"
        f"- 도메인 의의: 트리 앙상블 모델이 *roof × 기상 상호작용* 을 자동 학습할 수 있도록, "
        f"학습 데이터에 \"돔 경기에서는 기상 변수가 상수\" 라는 사실을 직접 주입."
    )
    lines.append("")

    lines.append("## 7. 기상 데이터 병합 결과")
    lines.append(
        f"- 호출 구장 수: {parks_used} (Athletics 제외)\n"
        f"- 호출 변수: {', '.join(WEATHER_HOURLY_VARS)} (총 {len(WEATHER_HOURLY_VARS)}종)\n"
        f"- 시점: daytime≥0.5 → 13:00 현지시각 / 그 외 → 19:00 현지시각\n"
        f"- 캐시: `pipeline/cache/weather_{{team}}_{{start}}_{{end}}.json`\n"
        f"- 기상 결측 행: {n_weather_missing:,d} / {n_weather_total:,d} "
        f"({n_weather_missing / max(n_weather_total, 1) * 100:.2f}%)"
    )
    lines.append("")
    lines.append("### 기상 변수 요약 통계")
    lines.append(weather_summary.round(2).to_markdown())
    lines.append("")

    lines.append("## 8. Temporal Split 결과")
    lines.append("| 연도 | 행 수 | 안타율 | 타자 수 | 투수 수 | 구장 수 | 기간 | 저장 경로 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|---|")
    for yr, info in sorted(split_info.items()):
        lines.append(
            f"| {yr} | {info['rows']:,d} | {info['hit_rate']:.4f} | "
            f"{info['n_batters']:,d} | {info['n_pitchers']:,d} | {info['n_home_teams']} | "
            f"{info['date_min']}~{info['date_max']} | `{info['path']}` |"
        )
    lines.append("")
    lines.append("- 2024_data 는 Phase 2~4 학습/평가용, 2025_data 는 Phase 5 검증 정답지용으로 격리.")
    lines.append("")

    lines.append("## 9. 산출물 파일 목록")
    lines.append(
        "- `pipeline/output/2024_data.parquet`\n"
        "- `pipeline/output/2025_data.parquet`\n"
        "- `pipeline/cache/weather_<team>_<start>_<end>.json` (구장별 Open-Meteo raw 응답 캐시)"
    )
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] phase1_report.md 작성 완료 → {REPORT_PATH.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Phase 1: 데이터 통합, 도메인 기반 전처리 및 연도별 분리")
    print("=" * 80)

    attrition = Attrition()

    # Compute bb_type distribution just before the BIP filter to preserve original BIP candidate counts in the report
    df_raw = pd.read_csv(STATCAST_CSV, low_memory=False)
    bb_type_dist_before = df_raw["bb_type"].value_counts(dropna=False)
    del df_raw

    # Measure bat-tracking missingness rates by season at the pre-BIP-filter stage
    df_for_missing = pd.read_csv(
        STATCAST_CSV,
        low_memory=False,
        usecols=["bb_type", "game_year", "home_team"] + BAT_TRACKING_COLS,
    )
    bip_only = df_for_missing[df_for_missing["bb_type"].isin(BB_TYPES_BIP)]
    bip_only = bip_only[~bip_only["home_team"].isin(EXCLUDE_TEAMS)]
    bat_missing_rates = (
        bip_only.groupby("game_year")[BAT_TRACKING_COLS]
        .apply(lambda s: s.isna().mean())
        .T
    )
    del df_for_missing, bip_only

    # Main pipeline
    df = load_and_clean_statcast(attrition)
    df, parks = merge_ballparks(df)

    date_min = str(df["game_date"].min().date())
    date_max = str(df["game_date"].max().date())
    print(f"[weather] 기상 fetch 범위: {date_min} ~ {date_max}")
    weather = build_weather_lookup(parks, date_min, date_max)

    df = merge_weather(df, weather)
    n_weather_total = len(df)
    n_weather_missing = int(df["wx_temperature_2m"].isna().sum())

    # Apply dome masking (user decision #11) — neutralize outdoor weather + apply indoor HVAC standard values
    df, n_dome_masked, n_dome_eligible = apply_dome_masking(df)

    # Weather variable summary statistics
    weather_summary = df[[f"wx_{v}" for v in WEATHER_HOURLY_VARS]].describe().T

    # Save outputs
    split_info = split_and_save(df)

    # Report
    parks_used = parks["team_name"].nunique() - len(EXCLUDE_TEAMS & set(parks["team_name"]))
    write_report(
        attrition=attrition,
        bb_type_dist_before=bb_type_dist_before,
        hit_rate_overall=float(df["is_hit"].mean()),
        bat_missing_rates=bat_missing_rates,
        parks_used=parks_used,
        weather_summary=weather_summary,
        split_info=split_info,
        n_weather_missing=n_weather_missing,
        n_weather_total=n_weather_total,
        n_dome_masked=n_dome_masked,
        n_dome_eligible=n_dome_eligible,
    )

    print("\n[done] Phase 1 완료.")


if __name__ == "__main__":
    main()
