"""
Collect per-game roof_status for retractable/dome ballparks via the MLB Stats API
==================================================================================

**Purpose:** To perform precise per-game dome weather masking in Phase 1,
query and cache the `gameData.weather.condition` field from the MLB Stats API
for all unique games at retractable ballparks (SEA, TOR, MIL, TEX, AZ, MIA, HOU)
and the full dome (TB).

**Rationale (pre-sample inspection):**
  - condition field populated 100% across a 35-game sample (0% missing rate)
  - Games explicitly labeled "Roof Closed" self-validate with wind = "0 mph, None"
  - Season-average roof ratios from ballparks.csv are consistent with sample results

**Processing rules:**
  - TB (Tropicana Field, roof=1.0) → no API call needed; all games assumed closed
  - Remaining 7 parks → fetch via API; classify as closed if condition contains "Roof Closed" or "Dome"

**Cache:** pipeline/cache/mlb_roof_status_cache.json
  - Loaded immediately on subsequent runs after the initial fetch
  - Format: { game_pk: "closed" | "open" | "unknown" }

Usage:
    PYTHONUNBUFFERED=1 /opt/miniconda3/envs/mlb-xba/bin/python pipeline/step1_fetch_roof_status.py \\
        2>&1 | tee pipeline/logs/step1_fetch_roof.log
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "데이터셋"
CACHE_DIR = ROOT / "pipeline" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "mlb_roof_status_cache.json"

STATCAST_CSV = DATA_DIR / "statcast_bat_tracking_2024_2025.csv"

# All unique games at retractable (roof > 0 and < 1) parks + full dome (TB)
RETRACTABLE_TEAMS = ["SEA", "TOR", "MIL", "TEX", "AZ", "MIA", "HOU"]
DOME_TEAMS = ["TB"]

# Keywords used to identify Roof Closed status (MLB API condition field)
CLOSED_KEYWORDS = ["Roof Closed", "Dome"]

# API configuration
STATSAPI_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
REQUEST_DELAY = 0.10  # throttle to at most 10 requests per second
REQUEST_TIMEOUT = 15


def log(msg: str) -> None:
    print(msg, flush=True)


def classify_condition(condition: str | None) -> str:
    """Map an API condition string → 'closed' | 'open' | 'unknown'."""
    if condition is None or condition == "":
        return "unknown"
    if any(k in condition for k in CLOSED_KEYWORDS):
        return "closed"
    return "open"


def fetch_one(game_pk: int) -> dict:
    """Fetch a single game — returns a result dict (status, condition, temp, wind, venue)."""
    try:
        resp = requests.get(STATSAPI_URL.format(game_pk=game_pk), timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return {"status": "unknown", "condition": None, "error": f"HTTP {resp.status_code}"}
        data = resp.json()
        gd = data.get("gameData", {})
        weather = gd.get("weather", {})
        venue = gd.get("venue", {}).get("name", "")
        condition = weather.get("condition")
        return {
            "status": classify_condition(condition),
            "condition": condition,
            "temp": weather.get("temp"),
            "wind": weather.get("wind"),
            "venue": venue,
        }
    except Exception as e:
        return {"status": "unknown", "condition": None, "error": str(e)[:80]}


def main():
    log("=" * 80)
    log("MLB Stats API roof_status fetch — retractable + dome ballparks")
    log("=" * 80)

    # 1. Collect unique target games from our dataset
    log("\n[1/4] Collecting unique games from our statcast data ...")
    df = pd.read_csv(STATCAST_CSV, usecols=["game_pk", "home_team", "game_year"], low_memory=False)

    # Retractable-park games (to be queried via API)
    retract_games = (
        df[df["home_team"].isin(RETRACTABLE_TEAMS)]
        .drop_duplicates(["game_pk", "home_team", "game_year"])
        .reset_index(drop=True)
    )
    log(f"  retractable 7 ballparks unique games: {len(retract_games):,}")
    log(f"  distribution by ballpark x year:")
    log(retract_games.groupby(["home_team", "game_year"]).size().unstack(fill_value=0).to_string())

    # TB (full dome) — no API call needed; automatically classified as closed
    tb_games = (
        df[df["home_team"].isin(DOME_TEAMS)]
        .drop_duplicates(["game_pk", "home_team", "game_year"])
        .reset_index(drop=True)
    )
    log(f"\n  TB (full dome) unique games: {len(tb_games):,} (no API call needed, auto closed)")

    # 2. Load existing cache
    log("\n[2/4] Loading cache ...")
    cache: dict[str, dict] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        log(f"  existing cache: {len(cache):,} entries")
    else:
        log("  no existing cache (full fetch required)")

    # 3. Auto-register TB games
    log("\n[3/4] TB (full dome) auto-registering as closed ...")
    n_tb_added = 0
    for _, r in tb_games.iterrows():
        key = str(int(r["game_pk"]))
        if key not in cache:
            cache[key] = {
                "status": "closed", "condition": "(TB, full dome, auto)",
                "home_team": r["home_team"], "game_year": int(r["game_year"]),
            }
            n_tb_added += 1
    log(f"  TB auto-registered: {n_tb_added} entries")

    # 4. Fetch retractable-park games
    log("\n[4/4] retractable ballpark API fetch ...")
    to_fetch = [
        (int(r["game_pk"]), r["home_team"], int(r["game_year"]))
        for _, r in retract_games.iterrows()
        if str(int(r["game_pk"])) not in cache
    ]
    log(f"  new fetch required: {len(to_fetch):,} games (est. {len(to_fetch) * (REQUEST_DELAY + 0.5) / 60:.1f} min)")

    n_closed = n_open = n_unknown = 0
    last_save = time.time()
    for game_pk, team, year in tqdm(to_fetch, desc="fetch roof_status", ncols=80):
        result = fetch_one(game_pk)
        cache[str(game_pk)] = {**result, "home_team": team, "game_year": year}
        status = result["status"]
        if status == "closed": n_closed += 1
        elif status == "open": n_open += 1
        else: n_unknown += 1
        time.sleep(REQUEST_DELAY)
        # Persist cache every 5 minutes as a safety net against mid-run interruption
        if time.time() - last_save > 300:
            CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
            last_save = time.time()

    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  ✓ new fetch complete: closed={n_closed:,}  open={n_open:,}  unknown={n_unknown:,}")
    log(f"  ✓ cache saved: {CACHE_PATH.relative_to(ROOT)}")

    # 5. Summary statistics
    log("\n[summary] full cache distribution (closed ratio by ballpark) ...")
    df_cache = pd.DataFrame.from_dict(cache, orient="index")
    if "home_team" in df_cache.columns:
        summary = df_cache.groupby("home_team")["status"].value_counts(normalize=True).unstack(fill_value=0)
        summary["n_games"] = df_cache.groupby("home_team").size()
        log(summary.round(3).to_string())

    # Compare against roof ratios from ballparks.csv
    log("\n[validation] comparing roof ratio from ballparks.csv vs. observed closed ratio ...")
    parks = pd.read_csv(DATA_DIR / "ballparks.csv")
    parks_dict = dict(zip(parks["team_name"], parks["roof"]))
    for team in RETRACTABLE_TEAMS + DOME_TEAMS:
        sub = df_cache[df_cache.get("home_team") == team] if "home_team" in df_cache.columns else pd.DataFrame()
        if len(sub) == 0:
            continue
        actual_closed = (sub["status"] == "closed").mean()
        csv_roof = parks_dict.get(team, "N/A")
        log(f"  {team:4s}: csv roof={csv_roof}  observed closed={actual_closed:.3f}  (n={len(sub)})")

    log("\n[done] roof_status fetch complete.")


if __name__ == "__main__":
    main()
