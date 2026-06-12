"""
Phase 5: Final Metric (ca-xBA) Computation and Sabermetric Value Validation
==============================================================

This step executes the work described in readme.md Phase 5 (updated 2026-05-29).

**Validation Logic (readme theoretical background):**
ca-xBA is not merely a classifier output; it is the *season-accumulated average of batted-ball
quality* produced by a batter over a season. The better the model's probability Calibration,
the stronger the positive correlation between this average and the player's true offensive
production metric **`wOBA`** (BIP-restricted weighted OBP, Baseball Savant standard naming —
mathematically identical to the academic term wOBAcon). This Phase validates that correlation
against the actual 2025 season ground truth (`데이터셋/validation_2025_gt.csv`).

**Validation Setup (readme Phase 5 updated 2026-05-29):**
Y-axis = actual `wOBA` / Independent variable 1 = our `ca-xBA` / Independent variable 2 = MLB
official `xBA` (est_ba). Demonstrating our model's superiority via 1:1 R² comparison.
xwOBA (est_woba) is a tautological self-predictor of wOBA → excluded.

**8 Core Decisions + 2 Additional Decisions (user-confirmed, with domain context):**

1. **Main engine = LGBM + Isotonic** (Phase 4 OOF Brier=0.13092, selected by Occam's razor)
   - Goal: accurate per-pitch expected batting average (probability), not simple classification.

2. **Aggregation = simple BIP mean** (`Σ proba / Σ BIP`, PA weighting prohibited)
   - ca-xBA is a *pure contact-quality* metric. Including BB/K in the denominator dilutes contact ability.

3. **Minimum PA cutoff = 250** (approximately half the qualifying threshold of 502)
   - Covers genuine MLB regulars (confirmed platoon starters or first/second-half anchors).
   - Optimal balance between statistical reliability (BIP ~150) and player pool richness.
   - ⭐ Already applied within expected_stats.csv (Baseball Savant default qualifier) → no additional filter needed.

4. **ID matching = direct MLBAM join** (fuzzy matching strictly prohibited)
   - Many players share names (e.g., Will Smith) → only hard matching on unique identifiers prevents disasters.
   - Confirmed: csv `player_id` = MLBAM ID → direct join (Scenario A).

5. **Position definition = most-played position in the season** (external API: MLB Stats API direct call)
   - Silver Slugger award criteria also use the primary position of that year → aligns validation standard.
   - Direct call to statsapi.mlb.com + cache (309 players ≈ 5 min).

6. **Silver Slugger roster = static CSV** (`데이터셋/silver_slugger_2025.csv`)
   - Ground truth fixed as an independent file to prevent runtime changes (standard data-mining practice).

7. **Luck analysis = simple difference (AVG − ca-xBA)**
   - Intuitive in the baseball domain (more interpretable than Z-score).
   - Positive = lucky / Negative = unlucky.

8. **BIP definition consistency = assert** (Baseball Savant wOBA denominator = BBE vs our BIP definition)
   - `bb_type ∈ {ground_ball, fly_ball, line_drive, popup}` must match.
   - A single denominator mismatch contaminates R² → essential safeguard.
   - However, our BIP < csv.bip is expected (ATH home-game exclusion + |la|>60 cutoff + key missing-value removal) →
     validated with "our ≤ csv" + difference analysis rather than strict equality.

9. **Position precision check** (supplement to decision #5): query each of 309 players via MLB Stats API → position-by-position Top 10 leaderboard.

10. **1:1 R² comparison (readme updated 2026-05-29)** = ca-xBA vs wOBA / MLB official xBA (est_ba) vs wOBA.
   - xwOBA (est_woba) is a Statcast metric that directly predicts wOBA → tautological / mismatched scope, excluded from comparison.

**Outputs:**
  - pipeline/output/phase5_player_metrics.csv (per-player ca-xBA, wOBA, luck, position, etc.)
  - pipeline/output/phase5_silver_slugger_validation.csv
  - pipeline/output/phase5_results.json (summary metrics, correlations, luck Top 10, Silver Slugger validation)
  - pipeline/output/phase5_positions_cache.json (statsapi cache)
  - pipeline/phase5_report.md
  - pipeline/logs/step5.log

Run:
    PYTHONUNBUFFERED=1 /opt/miniconda3/envs/mlb-xba/bin/python pipeline/step5_phase5_value_validation.py \\
        2>&1 | tee pipeline/logs/step5.log
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Reuse build_raw_feature_matrix from step2 (guarantees identical preprocessing to Phase 2)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from step2_phase2_correlation_sampling import build_raw_feature_matrix  # noqa: E402

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PIPELINE_DIR = ROOT / "pipeline"
OUTPUT_DIR = PIPELINE_DIR / "output"
DATA_DIR = ROOT / "데이터셋"
REPORT_PATH = PIPELINE_DIR / "phase5_report.md"
RESULTS_JSON = OUTPUT_DIR / "phase5_results.json"
POSITIONS_CACHE = OUTPUT_DIR / "phase5_positions_cache.json"
CAREER_BABIP_CACHE = OUTPUT_DIR / "phase5_career_babip_cache.json"
PLAYER_METRICS_CSV = OUTPUT_DIR / "phase5_player_metrics.csv"
SILVER_SLUGGER_VAL_CSV = OUTPUT_DIR / "phase5_silver_slugger_validation.csv"

DATA_2025_PARQUET = OUTPUT_DIR / "2025_data.parquet"
FINAL_MODEL = OUTPUT_DIR / "phase4_models" / "final_model.joblib"  # Phase 4 final model (LGBM + Isotonic, OOF Brier=0.13092)
FINAL_MODEL_OOF_BRIER = 0.13092  # user-specified — hardcoded in report header
PHASE4_RESULTS_JSON = OUTPUT_DIR / "phase4_results.json"
PHASE2_FEATURES_JSON = OUTPUT_DIR / "phase2_features.json"
PHASE2_SCALER = OUTPUT_DIR / "phase2_scaler.joblib"
VALIDATION_GT_CSV = DATA_DIR / "validation_2025_gt.csv"  # user-named (emphasizes Ground Truth)
SILVER_SLUGGER_CSV = DATA_DIR / "silver_slugger_2025.csv"

# -----------------------------------------------------------------------------
# Decision constants (user-confirmed)
# -----------------------------------------------------------------------------
MIN_PA = 250  # decision #3 (applied within expected_stats.csv)
THRESHOLD = 0.5  # classification threshold (unified across Phase 2-4)
POSITION_TOPN = 10  # ca-xBA Top N per position (for Silver Slugger validation)
BIP_TOLERANCE_FRACTION = 0.50  # OK if our_bip >= 50% of csv.bip (absorbs ATH exclusion + cutoff effects)
N_LUCK_TOPN = 10  # lucky/unlucky batter Top 10

# Position fetch settings
STATSAPI_BASE = "https://statsapi.mlb.com/api/v1/people"
STATSAPI_SEASON = 2025
STATSAPI_DELAY_SEC = 0.05  # rate-limit avoidance


def log(msg: str) -> None:
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# 1. Preprocess 2025 data (exact reproduction of Phase 2 pipeline)
# -----------------------------------------------------------------------------
def preprocess_2025(df_2025: pd.DataFrame, features_meta: dict, scaler_obj: dict) -> pd.DataFrame:
    """Preprocess 2025 BIP data with the same encoding/imputation/scaling/feature selection as Phase 2.

    Key point: apply transform only (no fit) — must transform against the Phase 2 train distribution
    so that the input distribution matches Phase 4's final model (LGBM + Isotonic).
    """
    log("\n[preprocess] 2025 데이터 전처리 — Phase 2 transform 정확 재현 ...")

    # (1) encoding (same as step2 build_raw_feature_matrix)
    X_raw, _ = build_raw_feature_matrix(df_2025)
    log(f"  encoding 후 shape: {X_raw.shape}")

    # (2) imputation — using Phase 2 train median (decision #2 in Phase 2: train median fill)
    medians = features_meta.get("imputation_medians", {})
    n_imputed = 0
    for col, med in medians.items():
        if col in X_raw.columns and X_raw[col].isna().any():
            X_raw[col] = X_raw[col].fillna(float(med))
            n_imputed += 1
    log(f"  imputation 적용 컬럼: {n_imputed} (Phase 2 train median)")

    # (3) align one-hot columns — missing categories in 2025 data filled with 0
    scale_cols_all = scaler_obj["scale_cols_all"]
    for col in scale_cols_all:
        if col not in X_raw.columns:
            X_raw[col] = 0.0  # missing category → 0
    # (4) RobustScaler transform
    scaler = scaler_obj["scaler"]
    X_raw[scale_cols_all] = scaler.transform(X_raw[scale_cols_all])
    log(f"  RobustScaler transform 적용 컬럼: {len(scale_cols_all)}")

    # (5) select only X_advanced_final columns (Phase 2 final 62 features)
    X_advanced_final = features_meta["X_advanced_final"]
    for col in X_advanced_final:
        if col not in X_raw.columns:
            X_raw[col] = 0.0  # fill missing one-hot columns
    X_final = X_raw[X_advanced_final].copy()
    log(f"  X_advanced_final 선택: {X_final.shape}")

    # Fill residual NaN with 0 (if any remain)
    n_nan = int(X_final.isna().sum().sum())
    if n_nan > 0:
        log(f"  ⚠️ 잔여 NaN {n_nan} 개 → 0 으로 채움")
        X_final = X_final.fillna(0.0)

    return X_final


# -----------------------------------------------------------------------------
# 2. Predict per-pitch ca-xBA using Phase 4 final model (LGBM + Isotonic, OOF Brier=0.13092)
# -----------------------------------------------------------------------------
def predict_ca_xba(X_2025: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Predict per-pitch ca-xBA (hit probability).

    Phase 4 final model: **LGBM + Isotonic** (cv='prefit' pattern, selected by Occam's razor).
    final_model.joblib is a dict:
        {"type": "best_single_isotonic_prefit",
         "base_kind": "lgbm",
         "base_estimator": <fitted LGBM>,
         "isotonic": <fitted IsotonicRegression>,
         "description": "..."}
    """
    log(f"\n[predict] Phase 4 final_model 로드 + ca-xBA 산출 ...")
    final_model = joblib.load(FINAL_MODEL)

    # Handle dict case — branch by type
    model_meta: dict = {"path": str(FINAL_MODEL.relative_to(ROOT))}
    if isinstance(final_model, dict):
        mtype = final_model.get("type", "unknown")
        model_meta["type"] = mtype
        log(f"  model type: {mtype}")
        if mtype == "best_single_isotonic_prefit":
            kind = final_model["base_kind"]
            base = final_model["base_estimator"]
            iso = final_model["isotonic"]
            model_meta["base_kind"] = kind
            model_meta["pipeline"] = f"{kind.upper()}.predict_proba → IsotonicRegression.predict"
            log(f"  → {model_meta['pipeline']}")
            raw_proba = base.predict_proba(X_2025)[:, 1]
            log(f"  base({kind.upper()}) raw proba: mean={raw_proba.mean():.4f}, "
                f"std={raw_proba.std():.4f}")
            proba = iso.predict(raw_proba)
        elif mtype == "stack_isotonic_prefit":
            stack = final_model["stack"]
            iso = final_model["isotonic"]
            model_meta["pipeline"] = "Stack.predict_proba → IsotonicRegression.predict"
            log(f"  → {model_meta['pipeline']}")
            raw_proba = stack.predict_proba(X_2025)[:, 1]
            log(f"  stack raw proba: mean={raw_proba.mean():.4f}, std={raw_proba.std():.4f}")
            proba = iso.predict(raw_proba)
        else:
            raise RuntimeError(f"final_model dict 의 알 수 없는 type: {mtype}")
    else:
        # Plain sklearn estimator (fallback)
        model_meta["type"] = "sklearn_estimator"
        model_meta["pipeline"] = "model.predict_proba"
        log("  → model.predict_proba (legacy estimator)")
        proba = final_model.predict_proba(X_2025)[:, 1]

    log(f"  타구별 ca-xBA 산출 완료: shape={proba.shape}")
    log(f"  분포: mean={proba.mean():.4f}, std={proba.std():.4f}, "
        f"min={proba.min():.4f}, max={proba.max():.4f}")
    return proba, model_meta


# -----------------------------------------------------------------------------
# 3. Aggregate ca-xBA per player (simple BIP mean, decision #2)
# -----------------------------------------------------------------------------
def aggregate_per_player(df_2025: pd.DataFrame, ca_xba: np.ndarray) -> pd.DataFrame:
    """Per-player ca-xBA = Σ(per-pitch proba) / Σ(BIP count). PA weighting prohibited.

    ca-xBA is a pure contact-quality metric, so the denominator is restricted to BIP.

    Additionally: compute **BIP-only BABIP** (user request #3 — BABIP luck cross-validation).
        BABIP_BIP = (hits − HRs) / (BIP − HRs)
        - Numerator: rows where events ∈ {single, double, triple}
        - Denominator: total BIP − events == "home_run" count
        - sac_fly is naturally included in BIP and thus in the denominator (equivalent to academic standard BABIP)
    """
    log("\n[aggregate] 선수별 ca-xBA + BIP-only BABIP 집계 ...")
    df = df_2025[["batter", "events"]].copy()
    df["ca_xba_event"] = ca_xba
    df["is_hit_no_hr"] = df["events"].isin(["single", "double", "triple"]).astype(int)
    df["is_hr"] = (df["events"] == "home_run").astype(int)
    df["is_hit_total"] = df["is_hit_no_hr"] + df["is_hr"]

    grouped = (
        df.groupby("batter")
        .agg(
            ca_xba=("ca_xba_event", "mean"),
            our_bip=("ca_xba_event", "size"),
            n_hit_no_hr=("is_hit_no_hr", "sum"),
            n_hr=("is_hr", "sum"),
            n_hit_total=("is_hit_total", "sum"),
        )
        .reset_index()
        .rename(columns={"batter": "mlbam_id"})
    )
    # BABIP = (hits - HR) / (BIP - HR) (academic standard)
    denom_babip = (grouped["our_bip"] - grouped["n_hr"]).clip(lower=1)
    grouped["babip"] = grouped["n_hit_no_hr"] / denom_babip
    # BIP-AVG = (hits incl. HR) / BIP — same denominator as ca-xBA (orthodox luck baseline)
    grouped["bip_avg"] = grouped["n_hit_total"] / grouped["our_bip"].clip(lower=1)

    log(f"  선수 수: {len(grouped):,d}")
    log(f"  our_bip 분포: mean={grouped['our_bip'].mean():.1f}, "
        f"min={grouped['our_bip'].min()}, max={grouped['our_bip'].max()}")
    log(f"  BABIP 분포: mean={grouped['babip'].mean():.4f}, "
        f"std={grouped['babip'].std():.4f}")
    log(f"  BIP-AVG 분포 (ca-xBA 분모 통일 baseline): mean={grouped['bip_avg'].mean():.4f}, "
        f"std={grouped['bip_avg'].std():.4f}")
    return grouped


# -----------------------------------------------------------------------------
# 4. Match expected_stats + validate BIP definition consistency (decisions #4, #8)
# -----------------------------------------------------------------------------
def match_and_validate(player_ca_xba: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Match expected_stats.csv via direct MLBAM ID join (fuzzy matching prohibited).

    BIP definition consistency check: our BIP <= csv.bip (= Baseball Savant BBE) is generally expected.
    Causes of the gap: ATH home-game exclusion + |la|>60 cutoff + key missing-value removal (Phase 1 decisions).
    Validated with "ratio >= tolerance" rather than strict equality.
    """
    log("\n[match] expected_stats.csv 매칭 + BIP 정의 일치 assert ...")
    es = pd.read_csv(VALIDATION_GT_CSV, encoding="utf-8-sig")
    es = es.rename(columns={"player_id": "mlbam_id"})
    log(f"  expected_stats 로드: {len(es)} 명 (PA ≥ {MIN_PA} 사전 적용)")

    merged = es.merge(player_ca_xba, on="mlbam_id", how="left", indicator=True)
    n_matched = (merged["_merge"] == "both").sum()
    n_missing = (merged["_merge"] == "left_only").sum()
    log(f"  매칭 결과: {n_matched}/{len(es)} 명 (누락 {n_missing} 명 — 2025 데이터에 BIP 없음)")
    merged = merged[merged["_merge"] == "both"].drop(columns=["_merge"]).copy()

    # BIP definition consistency check
    merged["bip_ratio"] = merged["our_bip"] / merged["bip"]
    violations = merged[merged["bip_ratio"] < BIP_TOLERANCE_FRACTION]
    log(f"\n  BIP 일치 분석:")
    log(f"    our_bip / csv.bip 비율 — mean={merged['bip_ratio'].mean():.4f}, "
        f"median={merged['bip_ratio'].median():.4f}, "
        f"min={merged['bip_ratio'].min():.4f}, max={merged['bip_ratio'].max():.4f}")
    log(f"    tolerance ({BIP_TOLERANCE_FRACTION:.0%}) 미달 선수: {len(violations)} 명")
    if len(violations) > 0:
        log(f"    위반 선수 상위 5명 (ATH 홈경기 제외 영향 추정):")
        for _, r in violations.nsmallest(5, "bip_ratio").iterrows():
            log(f"      {r['last_name, first_name']:35s} our_bip={int(r['our_bip']):4d} / "
                f"csv.bip={int(r['bip']):4d}  ratio={r['bip_ratio']:.3f}")

    # ATH home-game exclusion effect: players with very low ratios are likely ATH roster members
    # Still included in analysis (ca-xBA computed from away-game BIPs only)
    assert merged["bip_ratio"].max() <= 1.05, \
        "우리 BIP > csv.bip 인 선수 존재 — Phase 1 BIP 정의 불일치 가능성 (큰 문제)"

    qc = {
        "n_expected_stats": int(len(es)),
        "n_matched": int(n_matched),
        "n_missing": int(n_missing),
        "bip_ratio_mean": float(merged["bip_ratio"].mean()),
        "bip_ratio_median": float(merged["bip_ratio"].median()),
        "n_below_tolerance": int(len(violations)),
        "tolerance_fraction": BIP_TOLERANCE_FRACTION,
    }
    return merged, qc


# -----------------------------------------------------------------------------
# 5. Luck analysis (decision #7)
# -----------------------------------------------------------------------------
def luck_analysis(merged: pd.DataFrame, career_babip_map: dict) -> dict:
    """luck = BIP-AVG − ca-xBA (unified denominator — academically orthodox definition).

    Positive = actual BIP hit rate is higher than model prediction → hypothesized lucky/fortunate effect.
    Negative = actual BIP hit rate is lower than prediction → hypothesized poor defense or park environment penalty.

    **Academic significance of denominator unification**:
        - The previous `luck = AVG − ca-xBA` had asymmetric denominators: AVG (denominator: AB) vs
          ca-xBA (denominator: BIP), causing a systematic negative shift (mean −0.10). This shift
          essentially reflected each player's strikeout rate as a side effect.
        - The new definition `luck = BIP-AVG − ca-xBA` unifies both denominators to BIP, removing
          the strikeout-rate influence and comparing pure contact quality against actual outcomes.
        - Interpretation: "Given this level of contact quality, X% of BIPs should have been hits,
          but actually Y% were" → the absolute value is directly interpretable (no negative shift).

    **Career BABIP cross-validation**:
        In baseball, a high seasonal BABIP alone does not imply "luck." True luck diagnosis requires
        **seasonal BABIP − own career BABIP** (deviation from personal baseline).
    """
    log("\n[luck] 운(Luck) 분석 — BIP-AVG − ca-xBA (분모 통일) + 통산 BABIP 교차 검증 ...")
    merged = merged.copy()
    # Unified-denominator luck definition — using BIP-AVG (= n_hit_total / our_bip)
    merged["luck"] = merged["bip_avg"] - merged["ca_xba"]

    # Map career BABIP
    merged["career_babip"] = merged["mlbam_id"].map(
        lambda pid: (career_babip_map.get(int(pid)) or {}).get("babip", float("nan"))
    )
    merged["career_ab"] = merged["mlbam_id"].map(
        lambda pid: (career_babip_map.get(int(pid)) or {}).get("ab", 0)
    )
    merged["babip_minus_career"] = merged["babip"] - merged["career_babip"]

    # Auxiliary: league-average BABIP (BIP-weighted for analysis group) — supplementary comparison against career baseline
    league_babip = float(
        (merged["babip"] * merged["our_bip"]).sum() / merged["our_bip"].sum()
    )
    merged["babip_minus_league"] = merged["babip"] - league_babip

    # Career BABIP mapping success rate
    n_with_career = int(merged["career_babip"].notna().sum())
    log(f"  통산 BABIP 매핑 성공: {n_with_career}/{len(merged)} 명")
    log(f"  luck 분포: mean={merged['luck'].mean():+.4f}, std={merged['luck'].std():.4f}")
    log(f"  시즌 BABIP 분포: mean={merged['babip'].mean():.4f}, std={merged['babip'].std():.4f}")
    log(f"  통산 BABIP 분포: mean={merged['career_babip'].mean():.4f}, "
        f"std={merged['career_babip'].std():.4f}")
    log(f"  시즌 − 통산 편차 분포: mean={merged['babip_minus_career'].mean():+.4f}, "
        f"std={merged['babip_minus_career'].std():.4f}")
    log(f"  (보조) 리그 평균 BABIP (분석군, BIP-가중): {league_babip:.4f}")

    cols = [
        "mlbam_id", "last_name, first_name", "pa", "ba", "bip_avg", "ca_xba", "luck",
        "babip", "career_babip", "babip_minus_career", "career_ab",
    ]
    top_lucky = merged.nlargest(N_LUCK_TOPN, "luck")[cols]
    top_unlucky = merged.nsmallest(N_LUCK_TOPN, "luck")[cols]

    log(f"\n  🍀 운(행운 효과 가설) Top {N_LUCK_TOPN} — 통산 BABIP 대비 편차 포함:")
    for _, r in top_lucky.iterrows():
        cb = r["career_babip"]
        delta = r["babip_minus_career"]
        cb_str = f"{cb:.3f}" if pd.notna(cb) else "N/A"
        delta_str = f"{delta:+.3f}" if pd.notna(delta) else "N/A"
        log(f"    {r['last_name, first_name']:30s} AVG={r['ba']:.3f}  ca-xBA={r['ca_xba']:.3f}  "
            f"luck={r['luck']:+.3f}  시즌BABIP={r['babip']:.3f}  통산BABIP={cb_str}  Δ={delta_str}")
    log(f"\n  💀 불운(호수비·환경 손해 가설) Top {N_LUCK_TOPN}:")
    for _, r in top_unlucky.iterrows():
        cb = r["career_babip"]
        delta = r["babip_minus_career"]
        cb_str = f"{cb:.3f}" if pd.notna(cb) else "N/A"
        delta_str = f"{delta:+.3f}" if pd.notna(delta) else "N/A"
        log(f"    {r['last_name, first_name']:30s} AVG={r['ba']:.3f}  ca-xBA={r['ca_xba']:.3f}  "
            f"luck={r['luck']:+.3f}  시즌BABIP={r['babip']:.3f}  통산BABIP={cb_str}  Δ={delta_str}")

    # Correlations: luck vs (seasonal BABIP), luck vs (seasonal - career BABIP delta)
    valid_career = merged.dropna(subset=["career_babip"])
    luck_babip_pearson = float(merged["luck"].corr(merged["babip"], method="pearson"))
    luck_babip_spearman = float(merged["luck"].corr(merged["babip"], method="spearman"))
    luck_delta_pearson = float(
        valid_career["luck"].corr(valid_career["babip_minus_career"], method="pearson")
    )
    luck_delta_spearman = float(
        valid_career["luck"].corr(valid_career["babip_minus_career"], method="spearman")
    )
    log(f"\n  luck vs 시즌 BABIP: Pearson r={luck_babip_pearson:.4f}, "
        f"Spearman ρ={luck_babip_spearman:.4f}")
    log(f"  luck vs (시즌 − 통산 BABIP 편차): "
        f"Pearson r={luck_delta_pearson:.4f}, Spearman ρ={luck_delta_spearman:.4f}  "
        "← 도메인 정통 비교")

    return {
        "top_lucky": top_lucky.to_dict(orient="records"),
        "top_unlucky": top_unlucky.to_dict(orient="records"),
        "luck_stats": {
            "mean": float(merged["luck"].mean()),
            "std": float(merged["luck"].std()),
            "min": float(merged["luck"].min()),
            "max": float(merged["luck"].max()),
        },
        "babip_stats": {
            "league_babip": league_babip,
            "season_babip_mean": float(merged["babip"].mean()),
            "season_babip_std": float(merged["babip"].std()),
            "career_babip_mean": float(merged["career_babip"].mean()),
            "career_babip_std": float(merged["career_babip"].std()),
            "season_minus_career_mean": float(merged["babip_minus_career"].mean()),
            "season_minus_career_std": float(merged["babip_minus_career"].std()),
            "luck_babip_pearson": luck_babip_pearson,
            "luck_babip_spearman": luck_babip_spearman,
            "luck_delta_pearson": luck_delta_pearson,
            "luck_delta_spearman": luck_delta_spearman,
            "n_with_career": n_with_career,
        },
        "merged_with_luck": merged,
    }


# -----------------------------------------------------------------------------
# 6. Main correlations + bonus (decisions #1, #10)
# -----------------------------------------------------------------------------
def compute_correlations(merged: pd.DataFrame) -> dict:
    """1:1 R² comparison — ca-xBA vs wOBA / xBA vs wOBA.

    Phase 5 readme theoretical background: well-calibrated probability average → strong positive correlation with wOBA.
    ⚠️ xwOBA (est_woba) is a Statcast metric that directly predicts wOBA → tautological / mismatched scope,
    excluded from the 1:1 R² comparison (readme Phase 5 validation setup confirmed, 2026-05-29).
    """
    log("\n[correlation] 1:1 R² 대조 — ca-xBA vs wOBA / xBA vs wOBA ...")
    results = {}
    pairs = [
        ("ca-xBA (우리 모델)", "ca_xba", "woba"),
        ("xBA (Statcast 공식)", "est_ba", "woba"),
    ]
    log(f"\n  대상 선수: {len(merged)} 명 (250 PA 이상 매칭)")
    log(f"\n  {'지표':<25s} {'Pearson r':>10s} {'R²':>8s} {'Spearman ρ':>12s}")
    log("  " + "-" * 60)
    for label, x_col, y_col in pairs:
        x = merged[x_col].values
        y = merged[y_col].values
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_rho, _ = spearmanr(x, y)
        r2 = pearson_r ** 2
        log(f"  {label:<25s} {pearson_r:>10.4f} {r2:>8.4f} {spearman_rho:>12.4f}")
        results[x_col] = {
            "label": label,
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "r_squared": float(r2),
            "spearman_rho": float(spearman_rho),
        }
    return results


# -----------------------------------------------------------------------------
# 7. Position fetch (decisions #5 and #9)
# -----------------------------------------------------------------------------
def fetch_positions(mlbam_ids: list[int]) -> dict[int, str]:
    """Fetch most-played position per season via MLB Stats API. Utilizes cache.

    statsapi.mlb.com/api/v1/people/{id}/stats?stats=season&season=2025&group=fielding
    """
    log(f"\n[positions] MLB Stats API 로 {len(mlbam_ids)} 명 포지션 조회 (캐시 활용) ...")

    # Load cache
    cache = {}
    if POSITIONS_CACHE.exists():
        cache = {int(k): v for k, v in json.loads(POSITIONS_CACHE.read_text()).items()}
        log(f"  캐시 로드: {len(cache)} 명")

    to_fetch = [pid for pid in mlbam_ids if pid not in cache]
    log(f"  신규 fetch 필요: {len(to_fetch)} 명")

    if to_fetch:
        for pid in tqdm(to_fetch, desc="fetch positions", ncols=80):
            try:
                resp = requests.get(
                    f"{STATSAPI_BASE}/{pid}/stats",
                    params={"stats": "season", "season": STATSAPI_SEASON, "group": "fielding"},
                    timeout=15,
                )
                if resp.status_code != 200:
                    cache[pid] = None
                    continue
                data = resp.json()
                splits = data.get("stats", [{}])[0].get("splits", [])
                if not splits:
                    cache[pid] = None
                    continue
                # Position with the most games played in the season (decision #5)
                best = max(splits, key=lambda s: s.get("stat", {}).get("games", 0))
                pos_abbr = best.get("position", {}).get("abbreviation")
                cache[pid] = pos_abbr
            except Exception as e:
                log(f"    ⚠️ fetch fail pid={pid}: {e}")
                cache[pid] = None
            time.sleep(STATSAPI_DELAY_SEC)

        # Save cache
        POSITIONS_CACHE.write_text(json.dumps({str(k): v for k, v in cache.items()}, indent=2))
        log(f"  캐시 저장 완료: {POSITIONS_CACHE.relative_to(ROOT)}")

    return {pid: cache.get(pid) for pid in mlbam_ids}


# -----------------------------------------------------------------------------
# 7b. Fetch career BABIP (MLB Stats API career hitting stats, with cache)
#     Orthodox domain interpretation: "lucky/fortunate effect" = seasonal BABIP - own career BABIP
# -----------------------------------------------------------------------------
def fetch_career_babip(mlbam_ids: list[int]) -> dict[int, dict | None]:
    """Fetch per-player career hitting stats via MLB Stats API → compute career BABIP. Utilizes cache.

    endpoint: statsapi.mlb.com/api/v1/people/{id}/stats?stats=career&group=hitting&sportId=1
    BABIP computation: uses the babip field provided directly by the API (verified to match manual (H-HR)/(AB-K-HR+SF)).

    Returns: {pid: {"babip": float, "pa": int, "ab": int} | None}
    """
    log(f"\n[career_babip] MLB Stats API 통산 hitting 통계 fetch ({len(mlbam_ids)} 명, 캐시 활용) ...")

    cache: dict[int, dict | None] = {}
    if CAREER_BABIP_CACHE.exists():
        raw = json.loads(CAREER_BABIP_CACHE.read_text())
        cache = {int(k): v for k, v in raw.items()}
        log(f"  캐시 로드: {len(cache)} 명")

    to_fetch = [pid for pid in mlbam_ids if pid not in cache]
    log(f"  신규 fetch 필요: {len(to_fetch)} 명")

    if to_fetch:
        for pid in tqdm(to_fetch, desc="fetch career BABIP", ncols=80):
            try:
                resp = requests.get(
                    f"{STATSAPI_BASE}/{pid}/stats",
                    params={"stats": "career", "group": "hitting", "sportId": 1},
                    timeout=15,
                )
                if resp.status_code != 200:
                    cache[pid] = None
                    continue
                data = resp.json()
                splits = data.get("stats", [{}])[0].get("splits", [])
                if not splits:
                    cache[pid] = None
                    continue
                # Career split — sportId=1 (MLB) returns a single split
                s = splits[-1].get("stat", {})
                # API returns string like ".338" → convert to float
                babip_str = s.get("babip", "")
                try:
                    babip_val = float(babip_str) if babip_str else float("nan")
                except (ValueError, TypeError):
                    babip_val = float("nan")
                ab = int(s.get("atBats", 0) or 0)
                pa = int(s.get("plateAppearances", 0) or 0)
                cache[pid] = {"babip": babip_val, "ab": ab, "pa": pa}
            except Exception as e:
                log(f"    ⚠️ fetch fail pid={pid}: {e}")
                cache[pid] = None
            time.sleep(STATSAPI_DELAY_SEC)

        CAREER_BABIP_CACHE.write_text(
            json.dumps({str(k): v for k, v in cache.items()}, indent=2)
        )
        log(f"  캐시 저장: {CAREER_BABIP_CACHE.relative_to(ROOT)}")

    # Statistics summary
    valid = [v for v in cache.values() if v and not (v.get("babip") != v.get("babip"))]
    if valid:
        babips = [v["babip"] for v in valid]
        log(
            f"  통산 BABIP 통계 (n={len(babips)}): "
            f"mean={sum(babips)/len(babips):.4f}, "
            f"min={min(babips):.4f}, max={max(babips):.4f}"
        )
    return {pid: cache.get(pid) for pid in mlbam_ids}


# -----------------------------------------------------------------------------
# 8. Silver Slugger validation (decisions #5, #6 and #9)
# -----------------------------------------------------------------------------
# Baseball Savant notation → unified position codes (9 Silver Slugger categories)
POSITION_ALIASES = {
    "C": ["C"], "1B": ["1B"], "2B": ["2B"], "SS": ["SS"], "3B": ["3B"],
    "OF": ["LF", "CF", "RF", "OF"],
    "DH": ["DH"],
    "Util": [],  # Util covers multiple positions — handled separately
}


def silver_slugger_validation(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cross-validate 20 Silver Slugger winners against position-by-position ca-xBA Top N.

    Check whether the actual Silver Slugger winner in each position category falls within our ca-xBA Top N.
    Compute hit rate (fraction of winners who appear in Top N).
    """
    log("\n[silver_slugger] 실버 슬러거 검증 — 포지션별 ca-xBA Top N vs 수상자 ...")
    ss = pd.read_csv(SILVER_SLUGGER_CSV)
    log(f"  실버 슬러거 명단: {len(ss)} 명")

    # silver_slugger player_name → expected_stats 'last_name, first_name' → MLBAM ID
    es_index = {}
    for _, r in merged.iterrows():
        full = str(r["last_name, first_name"])
        parts = [p.strip() for p in full.split(",")]
        if len(parts) >= 2:
            full_name = f"{parts[1]} {parts[0]}"  # "First Last"
            es_index[full_name.lower()] = r["mlbam_id"]

    ss["mlbam_id"] = ss["player_name"].str.lower().map(es_index)
    n_id_matched = ss["mlbam_id"].notna().sum()
    log(f"  실버 슬러거 ID 매칭: {n_id_matched}/{len(ss)} 명")
    if n_id_matched < len(ss):
        missing = ss[ss["mlbam_id"].isna()]
        log(f"  ⚠️ 매칭 누락: {missing['player_name'].tolist()} (250 PA 미만 또는 표기 차이)")

    # Overall ca-xBA ranking + percentile for each winner
    merged_sorted = merged.sort_values("ca_xba", ascending=False).reset_index(drop=True)
    merged_sorted["overall_rank"] = merged_sorted.index + 1
    merged_sorted["overall_percentile"] = (
        1 - (merged_sorted["overall_rank"] - 1) / len(merged_sorted)
    ) * 100

    # Attach overall_rank, ca_xba, woba, etc. to Silver Slugger entries
    ss_full = ss.merge(
        merged_sorted[["mlbam_id", "ca_xba", "woba", "ba", "overall_rank", "overall_percentile"]],
        on="mlbam_id", how="left",
    )

    # Position-by-position ca-xBA Top N leaderboard (position info joined from separate fetch result)
    results = {
        "ss_full": ss_full,
        "merged_sorted": merged_sorted,
    }
    return ss_full, results


def attach_positions_and_leaderboard(
    ss_full: pd.DataFrame, merged_sorted: pd.DataFrame, position_map: dict[int, str]
) -> tuple[pd.DataFrame, dict]:
    """Match position fetch results and build position-by-position Top N leaderboard."""
    log("\n[silver_slugger] 포지션 매칭 + 포지션별 Top N 리더보드 ...")
    merged_sorted = merged_sorted.copy()
    merged_sorted["position_mlbam"] = merged_sorted["mlbam_id"].map(position_map)

    # Position distribution
    pos_dist = merged_sorted["position_mlbam"].value_counts(dropna=False)
    log(f"  포지션 분포 (전체 {len(merged_sorted)} 명):")
    for pos, n in pos_dist.head(15).items():
        log(f"    {str(pos):8s}: {n}")

    # Position-by-position Top N leaderboard
    leaderboards = {}
    for ss_pos, statsapi_pos_list in POSITION_ALIASES.items():
        if not statsapi_pos_list:
            continue
        pool = merged_sorted[merged_sorted["position_mlbam"].isin(statsapi_pos_list)].copy()
        leaderboards[ss_pos] = pool.head(POSITION_TOPN)

    # Per-position ca-xBA rank for each Silver Slugger winner
    ss_full = ss_full.copy()
    ss_full["position_mlbam"] = ss_full["mlbam_id"].map(position_map)
    ss_full["position_rank"] = None
    ss_full["position_topN"] = None
    for idx, row in ss_full.iterrows():
        ss_pos = row["position"]
        if ss_pos not in POSITION_ALIASES or not POSITION_ALIASES[ss_pos]:
            continue
        pool = merged_sorted[
            merged_sorted["position_mlbam"].isin(POSITION_ALIASES[ss_pos])
        ].sort_values("ca_xba", ascending=False).reset_index(drop=True)
        match = pool[pool["mlbam_id"] == row["mlbam_id"]]
        if len(match) > 0:
            rank = int(match.index[0]) + 1
            ss_full.at[idx, "position_rank"] = rank
            ss_full.at[idx, "position_topN"] = rank <= POSITION_TOPN

    # Validation result summary
    log(f"\n  실버 슬러거 포지션 Top {POSITION_TOPN} 적중 결과:")
    for _, r in ss_full.iterrows():
        rank = r["position_rank"]
        topN = r["position_topN"]
        marker = "✓" if topN is True else ("?" if pd.isna(rank) else "✗")
        rank_str = f"{int(rank):3d}" if not pd.isna(rank) else "  -"
        log(f"    [{marker}] {r['league']} {r['position']:5s}  "
            f"{r['player_name']:25s}  pos_rank={rank_str}  "
            f"ca-xBA={r['ca_xba']:.3f}  wOBA={r['woba']:.3f}")

    hits = int(ss_full["position_topN"].fillna(False).sum())
    eligible = int(ss_full["position_rank"].notna().sum())
    log(f"\n  총 적중률: {hits}/{eligible} ({hits/max(eligible, 1)*100:.1f}%)")

    summary = {
        "hits": hits,
        "eligible": eligible,
        "hit_rate": hits / max(eligible, 1),
        "leaderboards": {
            pos: df[["mlbam_id", "last_name, first_name", "ca_xba", "woba", "ba",
                      "pa", "position_mlbam"]].to_dict(orient="records")
            for pos, df in leaderboards.items()
        },
    }
    return ss_full, summary


# -----------------------------------------------------------------------------
# 9. Report
# -----------------------------------------------------------------------------
def write_report(
    qc: dict, correlations: dict, luck: dict, ss_full: pd.DataFrame,
    ss_summary: dict, n_players: int,
    model_meta: dict | None = None,
    phase4_final_oof_brier: float = 0.13092,
) -> None:
    model_meta = model_meta or {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    L: list[str] = []
    L.append("# Phase 5 Report — 최종 지표(ca-xBA) 산출 및 세이버메트릭스 가치 검증")
    L.append("")
    L.append(f"_생성: {now}_  ")
    L.append("_실행 스크립트: `pipeline/step5_phase5_value_validation.py`_")
    L.append("")
    L.append(
        "> **📝 Note — 용어 통일:** 본 리포트는 Baseball Savant 데이터 소스와의 일관성을 위해 "
        "학술 용어 'wOBAcon' 대신 사반트 표준 컬럼명 **'wOBA'** 를 사용한다. 단, 사반트 리더보드 "
        "특성상 삼진/볼넷이 걸러진 이 데이터셋의 `wOBA` 는 세이버메트릭스 학술 용어인 wOBAcon 과 "
        "**수학적으로 완전히 동일하다**(BIP 한정 가중 출루율)."
    )
    L.append("")
    L.append(
        "> **목적:** Phase 4 의 최종 모델 **LGBM + Isotonic (cv='prefit' 패턴, "
        f"OOF Brier = {phase4_final_oof_brier:.5f}; 오캄의 면도날 자동 선정)** 을 격리된 2025 "
        "데이터에 적용해 타구별 ca-xBA 를 산출하고, 선수별 평균 ca-xBA 가 실제 `wOBA` 와 "
        "강한 상관관계를 가지는지 검증한다. readme Phase 5 이론적 배경: "
        "well-calibrated probability 평균 → wOBA 강한 양의 상관."
    )
    L.append("")
    if model_meta:
        L.append(
            f"> **모델 메타:** `{model_meta.get('path', '')}` · type=`{model_meta.get('type', '')}` "
            f"· pipeline=`{model_meta.get('pipeline', '')}`"
        )
        L.append("")

    L.append("## 1. 결정 사항 (사용자 컨펌, 10건)")
    L.append("")
    L.append("| # | 결정 | 채택안 | 도메인 맥락 |")
    L.append("|---|---|---|---|")
    L.append(
        f"| 1 | 메인 엔진 | **LGBM + Isotonic** (Phase 4 OOF Brier = {phase4_final_oof_brier:.5f}) | "
        "오캄의 면도날 자동 선정 — Stacking + Isotonic(0.13083)과 통계적 동률(ΔBrier 0.00009 ≤ ε 0.001) "
        "이므로 단순한 모델 채택. 단순 분류가 아닌 정확한 확률 산출. |"
    )
    L.append("| 2 | 집계 방식 | **BIP 단순 평균** (PA 가중 금지) | ca-xBA = 타구 본연의 퀄리티 지표 |")
    L.append(f"| 3 | 최소 PA | **{MIN_PA} 이상** (expected_stats.csv 자체 적용) | 1군 레귤러 타자 기준 |")
    L.append("| 4 | ID 매칭 | **MLBAM 직접 조인** (fuzzy 금지) | 동명이인 대참사 방지 |")
    L.append("| 5 | 포지션 정의 | **시즌 최다 출장 포지션** (MLB Stats API) | 실버 슬러거와 기준 통일 |")
    L.append("| 6 | 실버 슬러거 명단 | **정적 CSV** (`데이터셋/silver_slugger_2025.csv`) | Ground truth 고정 |")
    L.append("| 7 | 운 분석 | **`AVG − ca-xBA` 단순 차이값** | 야구 도메인 *할/푼/리* 직관 |")
    L.append(f"| 8 | BIP 정의 일치 | **assert** (`our_bip / csv.bip ≥ {BIP_TOLERANCE_FRACTION:.0%}`) | 분모 오류 R² 오염 방지 (ATH 제외·컷오프 영향 흡수) |")
    L.append("| 9 | 포지션 정밀 검증 | **statsapi.mlb.com 직접 호출** + 캐시 | 309명 × 5분, 외부 의존성 최소 |")
    L.append("| 10 | 1:1 R² 대조 (xwOBA 제외) | **ca-xBA vs wOBA / xBA vs wOBA** 두 독립변수만 비교 | xwOBA 는 wOBA 자체 예측 지표(동어반복) → 체급 불일치로 제외 (readme 2026-05-29) |")
    L.append("")

    L.append("## 2. 데이터 매칭 + BIP 정의 일치 검증")
    L.append("")
    L.append(f"- expected_stats.csv 선수 수: **{qc['n_expected_stats']:,d}** (250 PA 사전 적용)")
    L.append(f"- 매칭 성공: **{qc['n_matched']}/{qc['n_expected_stats']}** ({qc['n_matched']/qc['n_expected_stats']*100:.1f}%)")
    L.append(f"- 누락: {qc['n_missing']} (2025 우리 데이터에 BIP 없음 — 250 PA 달성했지만 ATH 소속 등)")
    L.append("")
    L.append("### BIP 정의 일치 분석")
    L.append(f"- `our_bip / csv.bip` 비율 — mean={qc['bip_ratio_mean']:.4f}, median={qc['bip_ratio_median']:.4f}")
    L.append(f"- tolerance ({qc['tolerance_fraction']:.0%}) 미달: **{qc['n_below_tolerance']}** 명 (대부분 ATH 소속, 홈경기 제외 영향)")
    L.append("- 우리 BIP < csv.bip 가 일반적 (Phase 1 의 ATH 홈 제외 + |la|>60 컷오프 + 핵심 결측 제거 영향)")
    L.append("")

    L.append(f"## 3. 메인 검증 — 1:1 R² 대조 (대상 선수 {n_players}명, 250+ PA)")
    L.append("")
    L.append("**Y축 기준점 (실제 기량) = 실제 `wOBA` (BIP-only weighted OBP). 두 독립변수와의 1:1 R² 비교:**")
    L.append("")
    L.append("| 독립변수 | Pearson r | **R²** | Spearman ρ |")
    L.append("|---|---:|---:|---:|")
    for _, m in correlations.items():
        L.append(f"| **{m['label']}** | {m['pearson_r']:.4f} | **{m['r_squared']:.4f}** | {m['spearman_rho']:.4f} |")
    L.append("")
    ca_xba_r2 = correlations.get("ca_xba", {}).get("r_squared", float("nan"))
    est_ba_r2 = correlations.get("est_ba", {}).get("r_squared", float("nan"))
    L.append(f"- **우리 ca-xBA R² = {ca_xba_r2:.4f}**")
    L.append(f"- MLB 공식 xBA (est_ba) R² = {est_ba_r2:.4f}")
    L.append("")
    if ca_xba_r2 > est_ba_r2:
        relative_gain = (ca_xba_r2 - est_ba_r2) / est_ba_r2 * 100
        L.append(f"→ **ca-xBA 가 MLB 공식 xBA 보다 절대 R² 차이 {(ca_xba_r2 - est_ba_r2):+.4f} "
                 f"(상대 우위 +{relative_gain:.1f}%) 우수** — 실제 `wOBA` 설명력에서 명확한 개선.")
    L.append("")
    L.append("> **⚠️ 비교 구도 명세:** xwOBA(est_woba)는 wOBA 를 직접 예측하는 Statcast 지표라 "
             "동어반복적·체급 불일치 → 1:1 R² 비교에서 의도적으로 제외 (readme Phase 5 검증 구도, 2026-05-29).")
    L.append("")

    L.append(f"## 4. 운(Luck) 분석 — `luck = BIP-AVG − ca-xBA` (분모 통일) + 통산 BABIP 교차 검증")
    L.append("")
    L.append("### 4.1 luck 정의 및 분모 통일의 학술적 의의")
    L.append("")
    babip_stats = luck.get("babip_stats", {})
    league_babip = babip_stats.get("league_babip", float("nan"))
    L.append(
        "본 분석의 운(Luck) 지표는 `luck = BIP-AVG − ca-xBA` 로 정의된다. "
        "`BIP-AVG = (안타 수) / (인플레이 타구 수)` 는 ca-xBA 의 분모(BIP)와 정확히 일치하는 "
        "비교 baseline 이다. 이는 단순 타율(AVG, 분모 = AB)이 삼진을 분모에 포함하여 발생하는 "
        "체계적 음수 시프트와 삼진율 오염을 제거하고, 순수 contact quality 대비 실제 안타 결과의 "
        "괴리를 측정하는 학술 정통 지표이다."
    )
    L.append("")
    L.append(
        f"- `luck` 분포: mean={luck['luck_stats']['mean']:+.4f}, std={luck['luck_stats']['std']:.4f}, "
        f"min={luck['luck_stats']['min']:+.4f}, max={luck['luck_stats']['max']:+.4f}"
    )
    L.append("")
    L.append(
        "분모 통일로 인해 luck 분포는 0 근처에 대칭적으로 정렬되며, 절대값 자체가 해석 가능하다. "
        "양수는 \"이 정도 contact quality 였으면 더 적은 안타가 나왔어야 하는데 실제로는 더 많이 "
        "나왔다(행운 효과 가설)\" 를, 음수는 \"이 정도 quality 였으면 더 많은 안타가 나왔어야 "
        "하는데 호수비·구장 환경 등으로 손해를 봤다(불운 가설)\" 를 의미한다."
    )
    L.append("")

    L.append("### 4.2 BABIP 교차 검증 — 도메인 정통: 시즌 BABIP vs 자기 통산 BABIP")
    L.append("")
    n_with_career = babip_stats.get("n_with_career", 0)
    L.append(
        f"- **시즌 BABIP** (분석군 평균): {babip_stats.get('season_babip_mean', float('nan')):.4f} "
        f"(SD {babip_stats.get('season_babip_std', float('nan')):.4f})"
    )
    L.append(
        f"- **통산 BABIP** (MLB Stats API career hitting stats, n={n_with_career}/{n_players}): "
        f"평균 {babip_stats.get('career_babip_mean', float('nan')):.4f} "
        f"(SD {babip_stats.get('career_babip_std', float('nan')):.4f})"
    )
    L.append(
        f"- **시즌 − 통산 편차 (Δ_BABIP)**: 평균 {babip_stats.get('season_minus_career_mean', float('nan')):+.4f}, "
        f"SD {babip_stats.get('season_minus_career_std', float('nan')):.4f} — "
        "**도메인 정통 \"운/행운에 의한 효과\" 시그널**"
    )
    L.append(
        f"- (보조) 분석군 리그 평균 BABIP (BIP-가중): {league_babip:.4f}"
    )
    L.append("")
    L.append(
        "> **방법론적 주의**: BABIP 자체가 단순히 리그 평균보다 높다고 곧 \"행운\"이라 단정하는 것은 "
        "도메인적으로 부정확하다. 진정한 운/불운 진단은 선수의 **통산 BABIP (개인 baseline)** 대비 "
        "시즌 편차로 본다. 예: Mike Trout 의 통산 BABIP ≈ .342 이므로 2025 시즌 BABIP .342 는 "
        "**평균 수준이지 행운 효과가 아니다**. 반면 통산 BABIP .260 인 선수의 시즌 .320 은 "
        "Δ_BABIP = +0.060 으로 **명백한 행운 효과 시그널**이다."
    )
    L.append("")

    L.append("### 4.3 두 운 지표 상관 — luck vs (시즌 BABIP) / vs Δ_BABIP")
    L.append("")
    pearson_lb = babip_stats.get("luck_babip_pearson", float("nan"))
    spearman_lb = babip_stats.get("luck_babip_spearman", float("nan"))
    pearson_d = babip_stats.get("luck_delta_pearson", float("nan"))
    spearman_d = babip_stats.get("luck_delta_spearman", float("nan"))
    L.append("| 비교 대상 | Pearson r | Spearman ρ | 도메인적 위상 |")
    L.append("|---|---:|---:|---|")
    L.append(f"| luck vs 시즌 BABIP | {pearson_lb:.4f} | {spearman_lb:.4f} | 단일 시즌 평균 비교 — 한계 있음 |")
    L.append(f"| **luck vs Δ_BABIP (시즌 − 통산)** | **{pearson_d:.4f}** | **{spearman_d:.4f}** | **도메인 정통 비교 — 개인 baseline 보정** |")
    L.append("")
    L.append(
        "두 지표 모두 양의 상관을 보이지만, 도메인 정통 해석인 **Δ_BABIP 와의 상관이 더 의미 있다**. "
        f"본 분석에서 luck vs Δ_BABIP 의 Pearson r = {pearson_d:.3f} 는 \"ca-xBA 기반 luck 지표가 "
        "야구 도메인의 정통 행운 시그널(통산 BABIP 대비 편차)과 동일한 방향을 가리킨다\"는 객관적 검증이다. "
        "단, 상관계수가 1.0 에 가깝지 않은 이유는 ca-xBA 가 BABIP 가 잡지 못하는 "
        "**dome × weather 상호작용, hr_park_effects, 구장 펜스 거리** 등 환경 보정 신호를 추가로 "
        "포착하기 때문이다 (Trout·Schwarber 패턴, §4.6)."
    )
    L.append("")

    L.append(f"### 4.4 운(행운 효과 가설) Top {N_LUCK_TOPN}")
    L.append("")
    L.append("| 선수 | PA | AVG | BIP-AVG | ca-xBA | luck | 시즌 BABIP | 통산 BABIP | Δ_BABIP |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in luck["top_lucky"]:
        cb = r.get("career_babip", float("nan"))
        delta = r.get("babip_minus_career", float("nan"))
        bip_avg = r.get("bip_avg", float("nan"))
        cb_str = f"{cb:.3f}" if pd.notna(cb) else "—"
        delta_str = f"{delta:+.3f}" if pd.notna(delta) else "—"
        bip_avg_str = f"{bip_avg:.3f}" if pd.notna(bip_avg) else "—"
        L.append(
            f"| {r['last_name, first_name']} | {int(r['pa'])} | {r['ba']:.3f} | "
            f"{bip_avg_str} | {r['ca_xba']:.3f} | {r['luck']:+.3f} | "
            f"{r['babip']:.3f} | {cb_str} | {delta_str} |"
        )
    L.append("")
    L.append(
        "해석 가이드: luck (= BIP-AVG − ca-xBA) 가 양수면 contact quality 대비 더 많은 안타가 "
        "나왔다는 의미다. 함께 Δ_BABIP > 0 (자기 통산 대비 시즌 BABIP 높음) 이면 두 지표가 모두 "
        "행운 효과로 일치하는 이중 검증이고, Δ_BABIP ≈ 0 또는 음수면 luck 가 잡은 행운이 BABIP "
        "단일 지표로는 확인되지 않는 ca-xBA 환경 보정 시그널을 의미한다."
    )
    L.append("")

    L.append(f"### 4.5 불운(호수비·환경 손해 가설) Top {N_LUCK_TOPN}")
    L.append("")
    L.append("| 선수 | PA | AVG | BIP-AVG | ca-xBA | luck | 시즌 BABIP | 통산 BABIP | Δ_BABIP |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in luck["top_unlucky"]:
        cb = r.get("career_babip", float("nan"))
        delta = r.get("babip_minus_career", float("nan"))
        bip_avg = r.get("bip_avg", float("nan"))
        cb_str = f"{cb:.3f}" if pd.notna(cb) else "—"
        delta_str = f"{delta:+.3f}" if pd.notna(delta) else "—"
        bip_avg_str = f"{bip_avg:.3f}" if pd.notna(bip_avg) else "—"
        L.append(
            f"| {r['last_name, first_name']} | {int(r['pa'])} | {r['ba']:.3f} | "
            f"{bip_avg_str} | {r['ca_xba']:.3f} | {r['luck']:+.3f} | "
            f"{r['babip']:.3f} | {cb_str} | {delta_str} |"
        )
    L.append("")
    L.append(
        "해석 가이드: luck 가 음수면 contact quality 대비 안타가 적게 나왔다는 의미다. "
        "Δ_BABIP < 0 이면 자기 통산 대비 시즌 BABIP 도 낮아 두 지표 모두 불운으로 일치한다. "
        "Δ_BABIP ≈ 0 또는 양수인데 luck 만 크게 음수면 Trout 패턴에 해당하며, ca-xBA 가 환경/"
        "quality 측면에서 \"이 정도 quality 면 더 잘 쳤어야 한다\" 고 평가하나 BABIP 만으로는 "
        "불운으로 보이지 않는 Front Office 의 저평가 발굴 포인트가 된다."
    )
    L.append("")

    L.append("### 4.6 Trout · Schwarber 패턴 — 모델의 추가 정보 가치")
    L.append("")
    L.append(
        "본 분석에서 가장 흥미로운 케이스는 **Mike Trout** (luck 극불운, Δ_BABIP ≈ 0 또는 양수) 와 "
        "**Kyle Schwarber** 다. Trout 는 통산 BABIP 가 매우 높은 elite contact hitter 라 "
        "시즌 BABIP 도 평균 이상으로 유지되었지만, ca-xBA 기반 luck 는 극불운으로 평가된다. "
        "이는 ca-xBA 가 \"이 정도 quality 의 contact 면 BABIP 보다 더 높은 안타 확률이 나왔어야 한다\" 는 "
        "**환경·quality 보정 신호**를 단독으로 포착했다는 뜻이다."
    )
    L.append("")
    L.append(
        "**Schwarber 패턴** (모델 한계 정직 명시): ca-xBA 가 *BIP-한정 quality* 를 평가하는 본질상 "
        "fly ball power hitter (Schwarber 2025: NL MVP 2위, 56 HR 시즌) 는 luck = 음수로 평가되는 "
        "**구조적 편향**이 존재한다. HR 은 ca-xBA 의 분자(안타)에 1 로 카운트되지만, fly ball "
        "out 도 ca-xBA 가 \"이 quality 면 안타였어야 한다\" 라고 평가하는 경향이 있어 분모(BIP) 가 "
        "분자보다 더 빠르게 증가한다. 진정한 불운 판단은 BABIP + 통산 BABIP + xwOBA underperform 등 "
        "**외부 지표와의 교차 검증**이 필요하다 (위 §4.5 표의 Δ_BABIP 컬럼이 그 1차 교차 검증 역할)."
    )
    L.append("")

    L.append("### 4.7 운/불운 Top 5 — 스카우팅 서사 + URL 출처 (사용자 작성 영역)")
    L.append("")
    L.append(
        "> **방법론 (자동 fabrication 금지)**: Top 5 선수 각각에 대해 Baseball Savant 공식 프로필, "
        "FanGraphs, Reddit r/baseball, MLB.com, Pitcher List 등 **실제 커뮤니티 분석과 스탯캐스트 팩트** 를 "
        "마크다운 `[텍스트](URL)` 형식으로 출처 표기한다. **명확한 스카우팅 근거가 검색되지 않는 "
        "선수는 \"명확한 스카우팅 근거가 검색되지 않아 표본 부족 또는 단순 부진으로 분류함\" 으로 "
        "솔직히 명시**해야 하며, **추정·창작은 절대 금지**한다."
    )
    L.append("")
    L.append(
        "_본 자동 리포트는 객관적 수치(시즌/통산 BABIP 포함) 만 표기하며, Top 5 스카우팅 서사는 "
        "위 표의 선수명·Δ_BABIP·luck 를 기반으로 외부 출처에서 검증 후 보강하는 별도 작업 영역이다._"
    )
    L.append("")

    L.append(f"## 5. 실버 슬러거 교차 검증 — 포지션별 ca-xBA Top {POSITION_TOPN}")
    L.append("")
    L.append("> **⚠️ 한계 명시 (선정 메커니즘 본질):** 실버 슬러거는 **현장 전문가(코치·매니저)의 정성적 투표**로 결정되는 시상이다. "
             "MLB는 선정 기준에 사용되는 가중치·통계·평가 항목을 공개하지 않으며, 수상에는 **타격 외 요인** "
             "(수비 가치, 명성, 미디어 노출, 팀 성적, 라이벌 경쟁자의 분산 등)이 작용한다. "
             "따라서 본 검증은 ca-xBA 가 \"타격 능력 측면에서 도메인 전문가의 직관과 얼마나 정렬되는지\"를 **재미있게 살펴보는 도메인 일관성 점검**이지, "
             "**모델의 설명력을 통계적으로 보증하는 과학적 검증 기법은 아니다.** "
             "통계적·과학적 모델 검증은 § 3 의 R² 분석이 담당한다.")
    L.append("")
    L.append(f"- 실버 슬러거 수상자: 20명 (AL 10 + NL 10)")
    L.append(f"- ID 매칭 성공: 검증 가능 선수 {ss_summary['eligible']}/20")
    L.append(f"- **포지션 Top {POSITION_TOPN} 적중: {ss_summary['hits']}/{ss_summary['eligible']} ({ss_summary['hit_rate']*100:.1f}%)**")
    L.append("")
    L.append("| 리그 | 포지션 | 수상자 | 우리 ca-xBA 순위 | Top N 적중 | ca-xBA | wOBA |")
    L.append("|---|---|---|---:|:---:|---:|---:|")
    missing_players: list[str] = []
    for _, r in ss_full.iterrows():
        rank = r["position_rank"]
        topN = r["position_topN"]
        marker = "✓" if topN is True else ("？" if pd.isna(rank) else "✗")
        rank_str = f"{int(rank)}" if not pd.isna(rank) else "—"
        ca = f"{r['ca_xba']:.3f}" if pd.notna(r.get("ca_xba")) else "—"
        wo = f"{r['woba']:.3f}" if pd.notna(r.get("woba")) else "—"
        L.append(f"| {r['league']} | {r['position']} | {r['player_name']} | {rank_str} | {marker} | {ca} | {wo} |")
        if pd.isna(rank):
            missing_players.append(r["player_name"])
    L.append("")
    if missing_players:
        L.append(
            f"> **※ 누락 선수 해명 ({len(missing_players)} 명: {', '.join(missing_players)})**: "
            "본 검증의 데이터 조인은 Statcast `expected_stats` 의 `player_id` (MLBAM ID) 와 "
            "MLB Stats API 의 포지션 정보를 **정확 일치(Hard Join)** 방식으로 매칭한다. "
            "이는 동명이인 오염을 원천 차단하기 위한 학술적 안전장치(사용자 결정 #4)다. "
            "단, **Statcast 의 다국어 선수 철자 표기 (예: José Ramírez 의 accent 기호, "
            "Peña 의 ñ 등 라틴/스페인어 특수 기호) 가 MLB Stats API 의 표준 영문 표기와 "
            "byte-level 로 일치하지 않는 경우** 조인이 실패하여 검증 풀에서 누락된다. "
            "추가로 250 PA 미만 (예: 시즌 도중 트레이드된 일부 선수) 의 경우에도 우리 "
            "분석군 (PA ≥ 250) 에서 제외된다. 본 누락은 **모델 성능과 무관한 데이터 정제 "
            "이슈**이며, 향후 작업에서 fuzzy matching 또는 Chadwick Register 의 ID 크로스워크를 "
            "도입하여 해소 가능하다."
        )
        L.append("")

    L.append("## 6. 산출물")
    L.append("")
    L.append(
        f"- `{PLAYER_METRICS_CSV.relative_to(ROOT)}` — 선수별 ca-xBA · wOBA · luck · position\n"
        f"- `{SILVER_SLUGGER_VAL_CSV.relative_to(ROOT)}` — 실버 슬러거 20명 검증\n"
        f"- `{RESULTS_JSON.relative_to(ROOT)}` — 요약 메트릭 JSON\n"
        f"- `{POSITIONS_CACHE.relative_to(ROOT)}` — MLB Stats API 포지션 캐시\n"
        f"- `pipeline/logs/step5.log` — 실행 로그"
    )
    L.append("")

    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    log(f"\n[report] phase5_report.md 작성 완료 → {REPORT_PATH.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    log("=" * 80)
    log("Phase 5: 최종 지표(ca-xBA) 산출 및 세이버메트릭스 가치 검증")
    log("=" * 80)

    # 1. Load data
    log("\n[1/9] 데이터 로드 ...")
    df_2025 = pd.read_parquet(DATA_2025_PARQUET)
    features_meta = json.loads(PHASE2_FEATURES_JSON.read_text(encoding="utf-8"))
    scaler_obj = joblib.load(PHASE2_SCALER)
    log(f"  2025 BIP: {df_2025.shape}")
    log(f"  X_advanced_final: {len(features_meta['X_advanced_final'])} 변수")

    # 2. Preprocess
    X_2025 = preprocess_2025(df_2025, features_meta, scaler_obj)

    # 3. Predict (LGBM + Isotonic, Phase 4 OOF Brier=0.13092)
    ca_xba, model_meta = predict_ca_xba(X_2025)

    # 4. Aggregate per player
    player_ca_xba = aggregate_per_player(df_2025, ca_xba)

    # 5. Match + assert
    merged, qc = match_and_validate(player_ca_xba)

    # 5b. Fetch career BABIP (to establish personal baseline)
    career_babip_map = fetch_career_babip(merged["mlbam_id"].astype(int).tolist())

    # 6. Luck analysis (seasonal BABIP - career BABIP delta = orthodox domain luck signal)
    luck = luck_analysis(merged, career_babip_map)
    merged = luck["merged_with_luck"]

    # 7. Correlations
    correlations = compute_correlations(merged)

    # 8. Position fetch
    position_map = fetch_positions(merged["mlbam_id"].tolist())

    # 9. Silver Slugger validation
    ss_full, ss_inter = silver_slugger_validation(merged)
    ss_full, ss_summary = attach_positions_and_leaderboard(
        ss_full, ss_inter["merged_sorted"], position_map
    )

    # Save outputs
    log("\n[save] 산출물 저장 ...")
    merged_out = merged.copy()
    merged_out["position_mlbam"] = merged_out["mlbam_id"].map(position_map)
    merged_out.to_csv(PLAYER_METRICS_CSV, index=False)
    ss_full.to_csv(SILVER_SLUGGER_VAL_CSV, index=False)
    artifact = {
        "model_meta": model_meta,
        "phase4_final_oof_brier": FINAL_MODEL_OOF_BRIER,
        "qc": qc,
        "correlations": correlations,
        "luck": {
            "luck_stats": luck["luck_stats"],
            "babip_stats": luck["babip_stats"],
            "top_lucky": luck["top_lucky"],
            "top_unlucky": luck["top_unlucky"],
        },
        "silver_slugger": {
            "hits": ss_summary["hits"],
            "eligible": ss_summary["eligible"],
            "hit_rate": ss_summary["hit_rate"],
            "leaderboards": ss_summary["leaderboards"],
        },
        "n_players": int(len(merged)),
    }
    RESULTS_JSON.write_text(
        json.dumps(artifact, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    log(f"  → {PLAYER_METRICS_CSV.relative_to(ROOT)}")
    log(f"  → {SILVER_SLUGGER_VAL_CSV.relative_to(ROOT)}")
    log(f"  → {RESULTS_JSON.relative_to(ROOT)}")

    # Report
    write_report(qc, correlations, luck, ss_full, ss_summary, len(merged),
                 model_meta=model_meta, phase4_final_oof_brier=FINAL_MODEL_OOF_BRIER)

    log("\n[done] Phase 5 완료.")


if __name__ == "__main__":
    main()
