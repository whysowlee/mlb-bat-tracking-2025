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
    log("MLB Stats API roof_status fetch — retractable + dome 구장")
    log("=" * 80)

    # 1. Collect unique target games from our dataset
    log("\n[1/4] 우리 statcast 데이터에서 unique 게임 수집 ...")
    df = pd.read_csv(STATCAST_CSV, usecols=["game_pk", "home_team", "game_year"], low_memory=False)

    # Retractable-park games (to be queried via API)
    retract_games = (
        df[df["home_team"].isin(RETRACTABLE_TEAMS)]
        .drop_duplicates(["game_pk", "home_team", "game_year"])
        .reset_index(drop=True)
    )
    log(f"  retractable 7 구장 unique 게임: {len(retract_games):,}건")
    log(f"  구장×연도별 분포:")
    log(retract_games.groupby(["home_team", "game_year"]).size().unstack(fill_value=0).to_string())

    # TB (full dome) — no API call needed; automatically classified as closed
    tb_games = (
        df[df["home_team"].isin(DOME_TEAMS)]
        .drop_duplicates(["game_pk", "home_team", "game_year"])
        .reset_index(drop=True)
    )
    log(f"\n  TB (완전 돔) unique 게임: {len(tb_games):,}건 (API 불필요, 자동 closed)")

    # 2. Load existing cache
    log("\n[2/4] 캐시 로드 ...")
    cache: dict[str, dict] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        log(f"  기존 캐시: {len(cache):,}건")
    else:
        log("  기존 캐시 없음 (전체 fetch 필요)")

    # 3. Auto-register TB games
    log("\n[3/4] TB (완전 돔) 자동 closed 등록 ...")
    n_tb_added = 0
    for _, r in tb_games.iterrows():
        key = str(int(r["game_pk"]))
        if key not in cache:
            cache[key] = {
                "status": "closed", "condition": "(TB, full dome, auto)",
                "home_team": r["home_team"], "game_year": int(r["game_year"]),
            }
            n_tb_added += 1
    log(f"  TB 자동 등록: {n_tb_added}건")

    # 4. Fetch retractable-park games
    log("\n[4/4] retractable 구장 API fetch ...")
    to_fetch = [
        (int(r["game_pk"]), r["home_team"], int(r["game_year"]))
        for _, r in retract_games.iterrows()
        if str(int(r["game_pk"])) not in cache
    ]
    log(f"  신규 fetch 필요: {len(to_fetch):,}건 (예상 {len(to_fetch) * (REQUEST_DELAY + 0.5) / 60:.1f}분)")

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
    log(f"\n  ✓ 신규 fetch 완료: closed={n_closed:,}  open={n_open:,}  unknown={n_unknown:,}")
    log(f"  ✓ 캐시 저장: {CACHE_PATH.relative_to(ROOT)}")

    # 5. Summary statistics
    log("\n[summary] 전체 캐시 분포 (구장별 closed 비율) ...")
    df_cache = pd.DataFrame.from_dict(cache, orient="index")
    if "home_team" in df_cache.columns:
        summary = df_cache.groupby("home_team")["status"].value_counts(normalize=True).unstack(fill_value=0)
        summary["n_games"] = df_cache.groupby("home_team").size()
        log(summary.round(3).to_string())

    # Compare against roof ratios from ballparks.csv
    log("\n[validation] ballparks.csv 의 roof 비율과 실측 closed 비율 비교 ...")
    parks = pd.read_csv(DATA_DIR / "ballparks.csv")
    parks_dict = dict(zip(parks["team_name"], parks["roof"]))
    for team in RETRACTABLE_TEAMS + DOME_TEAMS:
        sub = df_cache[df_cache.get("home_team") == team] if "home_team" in df_cache.columns else pd.DataFrame()
        if len(sub) == 0:
            continue
        actual_closed = (sub["status"] == "closed").mean()
        csv_roof = parks_dict.get(team, "N/A")
        log(f"  {team:4s}: csv roof={csv_roof}  실측 closed={actual_closed:.3f}  (n={len(sub)})")

    log("\n[done] roof_status fetch 완료.")


if __name__ == "__main__":
    main()
