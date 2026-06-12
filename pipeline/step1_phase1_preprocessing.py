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
EXCLUDE_TEAMS = {"ATH"}  # Exclude all Athletics games (2024+2025) from analysis
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
        lines = ["| Step | Remaining Rows | Removed at Step | Note |", "|---|---:|---:|---|"]
        for s in self.steps:
            lines.append(
                f"| {s['step']} | {s['rows']:,d} | {s['removed_at_step']:,d} | {s['note']} |"
            )
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# 1) Load Statcast data + step-by-step cleaning
# -----------------------------------------------------------------------------
def load_and_clean_statcast(attrition: Attrition) -> pd.DataFrame:
    print("[load] loading statcast CSV...")
    df = pd.read_csv(STATCAST_CSV, low_memory=False)
    attrition.record("0. Raw load", df, note="2024+2025 full dataset, pitch-level")

    # BIP filter
    df = df[df["bb_type"].isin(BB_TYPES_BIP)].copy()
    attrition.record(
        "1. BIP filter (bb_type ∈ {ground/fly/line/popup})",
        df,
        note="Option C adopted",
    )

    # Exclude Athletics
    df = df[~df["home_team"].isin(EXCLUDE_TEAMS)].copy()
    attrition.record(
        "2. Athletics (ATH) home-game rows excluded",
        df,
        note="Excluded from analysis due to 2024 Oakland / 2025 Sacramento relocation issue",
    )

    # Remove rows missing core physics features
    core_missing = df["launch_speed"].isna() | df["launch_angle"].isna()
    df = df.loc[~core_missing].copy()
    attrition.record(
        "3. Rows with missing launch_speed/launch_angle removed",
        df,
        note="Missing core xBA inputs → cannot be used in modeling",
    )

    # Foul-popup cutoff
    df = df[df["launch_angle"].abs() <= LAUNCH_ANGLE_ABS_MAX].copy()
    attrition.record(
        f"4. |launch_angle| > {LAUNCH_ANGLE_ABS_MAX:.0f}° cutoff",
        df,
        note="Per readme example — foul popups and extreme negative-angle line drives removed",
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
        f"[merge] ballparks merge complete. unmatched={n_unmatched:,d} "
        f"(unique home_team={df['home_team'].nunique()})"
    )
    if n_unmatched > 0:
        unmatched_teams = sorted(merged.loc[merged["ballpark"].isna(), "home_team"].unique())
        raise RuntimeError(f"unmapped home_team(s) found in ballparks: {unmatched_teams}")
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
            f"roof_status cache not found: {ROOF_STATUS_CACHE_PATH}. "
            "Run `python pipeline/step1_fetch_roof_status.py` first."
        )
    cache = json.loads(ROOF_STATUS_CACHE_PATH.read_text(encoding="utf-8"))
    print(f"[dome_mask] roof_status cache loaded: {len(cache):,d} games")

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
    print(f"  dome masking applied: {n_masked:,d} rows / eligible {n_eligible:,d} rows "
          f"({n_masked / max(len(df), 1) * 100:.2f}% of total BIP)", flush=True)
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
    print(f"[merge] weather merge complete. missing weather rows={n_missing:,d}")
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
    lines.append("# Phase 1 Report — Data Integration, Domain-Driven Preprocessing, and Temporal Split by Season")
    lines.append("")
    lines.append(f"_Generated: {now}_  ")
    lines.append(f"_Script: `pipeline/step1_phase1_preprocessing.py`_")
    lines.append("")
    lines.append("## 1. Introduction Summary")
    lines.append(
        "This phase combines MLB Statcast batted-ball data (2024–2025), ballpark specification data, "
        "and Open-Meteo Historical weather data to construct the input dataset for subsequent modeling. "
        "All processing is performed in accordance with domain knowledge and user-confirmed decisions. "
        "A strict Temporal Split (2024 ↔ 2025) by game_year is applied to prevent Data Leakage."
    )
    lines.append("")

    lines.append("## 2. Dataset Description")
    lines.append(
        "- Statcast: `데이터셋/statcast_bat_tracking_2024_2025.csv` "
        "(raw 1,443,801 rows × 118 columns, pitch-level)\n"
        "- Ballpark specs: `데이터셋/ballparks.csv` (30 ballparks × 15 columns; "
        "lat/lon columns added in Phase 1)\n"
        "- Weather: Open-Meteo Archive API `archive-api.open-meteo.com/v1/archive` "
        "(free, no API key required)\n"
        "- Target variable `is_hit`: events ∈ {single, double, triple, home_run} → 1, "
        "all others → 0 (aligned with MLB official xBA definition)\n"
        f"- Hit rate after final preprocessing = {hit_rate_overall:.4f}"
    )
    lines.append("")

    lines.append("## 3. Decision Log (Branching Points)")
    lines.append("Analysis decisions confirmed by the user.")
    lines.append("")
    lines.append("| # | Decision Item | Adopted Option | Rationale / Impact |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| 1 | BIP (ball-in-play) definition | `bb_type ∈ {ground_ball, fly_ball, line_drive, popup}` (Option C) | "
        "Most conservative definition — only rows with confirmed batted-ball tracking. Clean denominator for hit-rate calculation. |"
    )
    lines.append(
        "| 2 | Target `is_hit` definition | `events ∈ {single, double, triple, home_run}` → 1 (Option A) | "
        "Identical to MLB official xBA definition. Ensures consistency for ca-xBA vs xBA comparison in Phase 5. sac_fly/field_error etc. are 0. |"
    )
    lines.append(
        "| 3 | Missing core physics (launch_speed/angle NaN) handling | Drop rows (Option A) | "
        "xBA is a function of launch_speed × launch_angle — rows with missing values cannot be used in modeling. "
        f"Loss of ~1.2% of total BIP (~3K rows) is negligible. |"
    )
    lines.append(
        "| 4 | `|launch_angle| > 60°` cutoff | Applied (Option A) | "
        "Follows readme example exactly. Removes foul popups and extreme negative-angle line drives as noise. "
        "Popup mean la=65.8°, hit rate 1.4% — nearly all removed. |"
    )
    lines.append(
        "| 5 | launch_speed lower-bound cutoff | No cutoff (Option A) | "
        "Weak contact is part of real BIP and included in the xBA definition. Artificial truncation loses information. |"
    )
    lines.append(
        "| 6 | Bat-tracking variable missingness handling | Retain NaN + add `*_is_missing` flags (Option B) | "
        "2024 11.46% / 2025 5.21% missingness is due to withheld data at the start of each season. "
        "Exploits NaN-native handling of tree models; preserves missingness pattern itself as a signal. |"
    )
    lines.append(
        "| 7 | Athletics (ATH) mapping | Exclude all 2024+2025 ATH home-game rows from analysis (Option D) | "
        "Home ballpark relocated from Oakland Coliseum → Sutter Health Park (Sacramento) in 2025 — "
        "environmental variables are entirely different; single mapping is not feasible. Accepts 8,641-row loss (3.4% of BIP). |"
    )
    lines.append(
        "| 8 | Ballpark lat/lon addition method | Add lat/lon columns directly to ballparks.csv (Option B) | "
        "Improves dataset self-completeness. Coordinates hard-coded from publicly available sources (MLB official/Wikipedia). |"
    )
    lines.append(
        "| 9 | Weather API snapshot time | `daytime ≥ 0.5` → 13:00 / otherwise → 19:00 local-time snapshot (Option D) | "
        "No first-pitch time available. Park-level branching via the daytime ratio in ballparks.csv. "
        "Approximates average start time for day/night games. |"
    )
    lines.append(
        "| 10 | Weather variable set | 8 variables (temperature, humidity, surface_pressure, wind_speed, wind_direction, "
        "precipitation, cloud_cover, wind_gusts) (Option C) | "
        "Covers air density (temperature/humidity/pressure) + wind (speed/direction/gusts) + in-game effects (precipitation/cloud cover). "
        "Multicollinearity to be addressed in Phase 2. |"
    )
    lines.append(
        "| 11 | Dome / roof-closed game weather masking | **Per-game precision masking via MLB Stats API `weather.condition`** — "
        "for closed games: 5 outdoor variables (wind_speed/gusts/direction · precipitation · cloud_cover) = 0, "
        "2 indoor variables (temperature_2m=22°C · relative_humidity_2m=50%) = HVAC standard values, surface_pressure unchanged | "
        "Directly encodes the domain fact that outdoor weather cannot affect hit probability inside a dome. "
        "Post-hoc validation confirmed that the simple roof column (0–1 seasonal average) alone cannot teach tree models a conditional-nullification split. "
        "Air density (temperature/humidity/pressure) preserved; outdoor weather effects neutralized separately. |"
    )
    lines.append("")

    lines.append("## 4. Per-Step Attrition (Row Count Changes)")
    lines.append(attrition.to_markdown())
    lines.append("")

    lines.append("## 5. bb_type Distribution Immediately After BIP Filter")
    lines.append("```")
    lines.append(str(bb_type_dist_before))
    lines.append("```")
    lines.append("")

    lines.append("## 6. Bat-Tracking Missingness Rates (BIP rows before preprocessing, by season)")
    lines.append(bat_missing_rates.round(4).to_markdown())
    lines.append("")

    lines.append("## 7b. Dome / Roof-Closed Game Weather Masking (Decision #11)")
    lines.append("")
    lines.append(
        f"- Weather masking applied only to games for which the MLB Stats API "
        f"(`/api/v1.1/game/{{game_pk}}/feed/live` → `gameData.weather.condition`) "
        f"explicitly records \"Roof Closed\" or \"Dome\".\n"
        f"- Cache file: `pipeline/cache/mlb_roof_status_cache.json` (1,318 games, 0 missing).\n"
        f"- Masked BIP rows: **{n_dome_masked:,d}** / eligible (retractable + TB) **{n_dome_eligible:,d}** "
        f"({n_dome_masked / max(n_dome_eligible, 1) * 100:.1f}%)\n"
        f"- Values applied:\n"
        f"  - 5 outdoor weather variables → 0: `wx_wind_speed_10m`, `wx_wind_gusts_10m`, `wx_wind_direction_10m`, "
        f"`wx_precipitation`, `wx_cloud_cover`\n"
        f"  - 2 indoor HVAC standard values: `wx_temperature_2m` = 22°C (MLB dome standard), "
        f"`wx_relative_humidity_2m` = 50% (midpoint of ASHRAE-recommended range)\n"
        f"  - Unchanged: `wx_surface_pressure` (indoor and outdoor air pressure are equal)\n"
        f"- Domain significance: directly injects into the training data the fact that "
        f"weather variables are constant in dome games, enabling tree ensemble models to "
        f"automatically learn the *roof × weather interaction*."
    )
    lines.append("")

    lines.append("## 7. Weather Data Merge Results")
    lines.append(
        f"- Ballparks queried: {parks_used} (Athletics excluded)\n"
        f"- Variables fetched: {', '.join(WEATHER_HOURLY_VARS)} (total {len(WEATHER_HOURLY_VARS)})\n"
        f"- Snapshot time: daytime≥0.5 → 13:00 local time / otherwise → 19:00 local time\n"
        f"- Cache: `pipeline/cache/weather_{{team}}_{{start}}_{{end}}.json`\n"
        f"- Rows with missing weather: {n_weather_missing:,d} / {n_weather_total:,d} "
        f"({n_weather_missing / max(n_weather_total, 1) * 100:.2f}%)"
    )
    lines.append("")
    lines.append("### Weather Variable Summary Statistics")
    lines.append(weather_summary.round(2).to_markdown())
    lines.append("")

    lines.append("## 8. Temporal Split Results")
    lines.append("| Season | Rows | Hit Rate | Batters | Pitchers | Ballparks | Period | Output Path |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|---|")
    for yr, info in sorted(split_info.items()):
        lines.append(
            f"| {yr} | {info['rows']:,d} | {info['hit_rate']:.4f} | "
            f"{info['n_batters']:,d} | {info['n_pitchers']:,d} | {info['n_home_teams']} | "
            f"{info['date_min']}~{info['date_max']} | `{info['path']}` |"
        )
    lines.append("")
    lines.append("- 2024_data is reserved for Phase 2–4 training/evaluation; 2025_data is held out as the Phase 5 validation ground truth.")
    lines.append("")

    lines.append("## 9. Output File List")
    lines.append(
        "- `pipeline/output/2024_data.parquet`\n"
        "- `pipeline/output/2025_data.parquet`\n"
        "- `pipeline/cache/weather_<team>_<start>_<end>.json` (per-ballpark Open-Meteo raw response cache)"
    )
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] phase1_report.md written → {REPORT_PATH.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Phase 1: Data integration, domain-driven preprocessing, and temporal split by season")
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
    print(f"[weather] weather fetch range: {date_min} ~ {date_max}")
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

    print("\n[done] Phase 1 complete.")


if __name__ == "__main__":
    main()
