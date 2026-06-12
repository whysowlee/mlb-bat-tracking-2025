"""
MLB Stats API 로 부터 부분 개폐형/돔 구장의 경기별 roof_status 수집
====================================================================

**목적:** Phase 1 의 돔 구장 기상 마스킹을 *경기 단위로 정밀하게* 수행하기 위해,
retractable 구장 (SEA, TOR, MIL, TEX, AZ, MIA, HOU) + 완전 돔 (TB) 의 모든 unique
게임에 대해 MLB Stats API 의 `gameData.weather.condition` 필드를 조회·캐시한다.

**근거 (사전 표본 점검):**
  - 35 경기 표본에서 condition 필드 100% 채워짐 (누락률 0%)
  - "Roof Closed" 명시 게임은 wind = "0 mph, None" 으로 자체 일관성 검증됨
  - ballparks.csv 의 시즌 평균 roof 비율과 표본 결과가 정합적

**처리 규칙:**
  - TB (Tropicana Field, roof=1.0) → API 호출 불필요, 모두 closed 로 가정
  - 그 외 7 구장 → API fetch 후 condition 에 "Roof Closed" 또는 "Dome" 포함 시 closed

**캐시:** pipeline/cache/mlb_roof_status_cache.json
  - 한 번 fetch 후 재실행 시 즉시 로드
  - 형식: { game_pk: "closed" | "open" | "unknown" }

실행:
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

# 부분 개폐형 + 부분 개폐형 (roof > 0 이고 < 1) 의 모든 unique 게임 + 완전 돔 (TB)
RETRACTABLE_TEAMS = ["SEA", "TOR", "MIL", "TEX", "AZ", "MIA", "HOU"]
DOME_TEAMS = ["TB"]

# Roof Closed 식별 키워드 (MLB API condition 필드)
CLOSED_KEYWORDS = ["Roof Closed", "Dome"]

# API 설정
STATSAPI_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
REQUEST_DELAY = 0.10  # 초당 10회 보호
REQUEST_TIMEOUT = 15


def log(msg: str) -> None:
    print(msg, flush=True)


def classify_condition(condition: str | None) -> str:
    """API condition 문자열 → 'closed' | 'open' | 'unknown'."""
    if condition is None or condition == "":
        return "unknown"
    if any(k in condition for k in CLOSED_KEYWORDS):
        return "closed"
    return "open"


def fetch_one(game_pk: int) -> dict:
    """단일 게임 fetch — 결과 dict (status, condition, temp, wind, venue)."""
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

    # 1. 우리 데이터에서 대상 unique 게임 수집
    log("\n[1/4] 우리 statcast 데이터에서 unique 게임 수집 ...")
    df = pd.read_csv(STATCAST_CSV, usecols=["game_pk", "home_team", "game_year"], low_memory=False)

    # retractable 게임 (API 조회 대상)
    retract_games = (
        df[df["home_team"].isin(RETRACTABLE_TEAMS)]
        .drop_duplicates(["game_pk", "home_team", "game_year"])
        .reset_index(drop=True)
    )
    log(f"  retractable 7 구장 unique 게임: {len(retract_games):,}건")
    log(f"  구장×연도별 분포:")
    log(retract_games.groupby(["home_team", "game_year"]).size().unstack(fill_value=0).to_string())

    # TB (완전 돔) — API 불필요, 자동으로 closed
    tb_games = (
        df[df["home_team"].isin(DOME_TEAMS)]
        .drop_duplicates(["game_pk", "home_team", "game_year"])
        .reset_index(drop=True)
    )
    log(f"\n  TB (완전 돔) unique 게임: {len(tb_games):,}건 (API 불필요, 자동 closed)")

    # 2. 기존 캐시 로드
    log("\n[2/4] 캐시 로드 ...")
    cache: dict[str, dict] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        log(f"  기존 캐시: {len(cache):,}건")
    else:
        log("  기존 캐시 없음 (전체 fetch 필요)")

    # 3. TB 자동 등록
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

    # 4. retractable fetch
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
        # 5분마다 캐시 저장 (중간 종료 안전망)
        if time.time() - last_save > 300:
            CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
            last_save = time.time()

    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n  ✓ 신규 fetch 완료: closed={n_closed:,}  open={n_open:,}  unknown={n_unknown:,}")
    log(f"  ✓ 캐시 저장: {CACHE_PATH.relative_to(ROOT)}")

    # 5. 요약 통계
    log("\n[summary] 전체 캐시 분포 (구장별 closed 비율) ...")
    df_cache = pd.DataFrame.from_dict(cache, orient="index")
    if "home_team" in df_cache.columns:
        summary = df_cache.groupby("home_team")["status"].value_counts(normalize=True).unstack(fill_value=0)
        summary["n_games"] = df_cache.groupby("home_team").size()
        log(summary.round(3).to_string())

    # ballparks.csv 의 roof 비율과 비교
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
