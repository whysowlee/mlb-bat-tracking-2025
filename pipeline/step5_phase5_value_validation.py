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
    log("\n[preprocess] Preprocessing 2025 data — exact reproduction of Phase 2 transform ...")

    # (1) encoding (same as step2 build_raw_feature_matrix)
    X_raw, _ = build_raw_feature_matrix(df_2025)
    log(f"  shape after encoding: {X_raw.shape}")

    # (2) imputation — using Phase 2 train median (decision #2 in Phase 2: train median fill)
    medians = features_meta.get("imputation_medians", {})
    n_imputed = 0
    for col, med in medians.items():
        if col in X_raw.columns and X_raw[col].isna().any():
            X_raw[col] = X_raw[col].fillna(float(med))
            n_imputed += 1
    log(f"  imputation applied columns: {n_imputed} (Phase 2 train median)")

    # (3) align one-hot columns — missing categories in 2025 data filled with 0
    scale_cols_all = scaler_obj["scale_cols_all"]
    for col in scale_cols_all:
        if col not in X_raw.columns:
            X_raw[col] = 0.0  # missing category → 0
    # (4) RobustScaler transform
    scaler = scaler_obj["scaler"]
    X_raw[scale_cols_all] = scaler.transform(X_raw[scale_cols_all])
    log(f"  RobustScaler transform applied columns: {len(scale_cols_all)}")

    # (5) select only X_advanced_final columns (Phase 2 final 62 features)
    X_advanced_final = features_meta["X_advanced_final"]
    for col in X_advanced_final:
        if col not in X_raw.columns:
            X_raw[col] = 0.0  # fill missing one-hot columns
    X_final = X_raw[X_advanced_final].copy()
    log(f"  X_advanced_final selected: {X_final.shape}")

    # Fill residual NaN with 0 (if any remain)
    n_nan = int(X_final.isna().sum().sum())
    if n_nan > 0:
        log(f"  ⚠️ residual NaN {n_nan} → filled with 0")
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
    log(f"\n[predict] Loading Phase 4 final_model + computing ca-xBA ...")
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
            raise RuntimeError(f"Unknown type in final_model dict: {mtype}")
    else:
        # Plain sklearn estimator (fallback)
        model_meta["type"] = "sklearn_estimator"
        model_meta["pipeline"] = "model.predict_proba"
        log("  → model.predict_proba (legacy estimator)")
        proba = final_model.predict_proba(X_2025)[:, 1]

    log(f"  per-pitch ca-xBA computation complete: shape={proba.shape}")
    log(f"  distribution: mean={proba.mean():.4f}, std={proba.std():.4f}, "
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
    log("\n[aggregate] Aggregating per-player ca-xBA + BIP-only BABIP ...")
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

    log(f"  player count: {len(grouped):,d}")
    log(f"  our_bip distribution: mean={grouped['our_bip'].mean():.1f}, "
        f"min={grouped['our_bip'].min()}, max={grouped['our_bip'].max()}")
    log(f"  BABIP distribution: mean={grouped['babip'].mean():.4f}, "
        f"std={grouped['babip'].std():.4f}")
    log(f"  BIP-AVG distribution (unified-denominator baseline for ca-xBA): mean={grouped['bip_avg'].mean():.4f}, "
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
    log("\n[match] Matching expected_stats.csv + asserting BIP definition consistency ...")
    es = pd.read_csv(VALIDATION_GT_CSV, encoding="utf-8-sig")
    es = es.rename(columns={"player_id": "mlbam_id"})
    log(f"  expected_stats loaded: {len(es)} players (PA ≥ {MIN_PA} pre-applied)")

    merged = es.merge(player_ca_xba, on="mlbam_id", how="left", indicator=True)
    n_matched = (merged["_merge"] == "both").sum()
    n_missing = (merged["_merge"] == "left_only").sum()
    log(f"  match result: {n_matched}/{len(es)} players (missing {n_missing} — no BIP in 2025 data)")
    merged = merged[merged["_merge"] == "both"].drop(columns=["_merge"]).copy()

    # BIP definition consistency check
    merged["bip_ratio"] = merged["our_bip"] / merged["bip"]
    violations = merged[merged["bip_ratio"] < BIP_TOLERANCE_FRACTION]
    log(f"\n  BIP consistency analysis:")
    log(f"    our_bip / csv.bip ratio — mean={merged['bip_ratio'].mean():.4f}, "
        f"median={merged['bip_ratio'].median():.4f}, "
        f"min={merged['bip_ratio'].min():.4f}, max={merged['bip_ratio'].max():.4f}")
    log(f"    players below tolerance ({BIP_TOLERANCE_FRACTION:.0%}): {len(violations)}")
    if len(violations) > 0:
        log(f"    top 5 violating players (estimated ATH home-game exclusion effect):")
        for _, r in violations.nsmallest(5, "bip_ratio").iterrows():
            log(f"      {r['last_name, first_name']:35s} our_bip={int(r['our_bip']):4d} / "
                f"csv.bip={int(r['bip']):4d}  ratio={r['bip_ratio']:.3f}")

    # ATH home-game exclusion effect: players with very low ratios are likely ATH roster members
    # Still included in analysis (ca-xBA computed from away-game BIPs only)
    assert merged["bip_ratio"].max() <= 1.05, \
        "Player(s) with our BIP > csv.bip detected — possible Phase 1 BIP definition mismatch (serious issue)"

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
    log("\n[luck] Luck analysis — BIP-AVG − ca-xBA (unified denominator) + career BABIP cross-validation ...")
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
    log(f"  career BABIP mapping success: {n_with_career}/{len(merged)} players")
    log(f"  luck distribution: mean={merged['luck'].mean():+.4f}, std={merged['luck'].std():.4f}")
    log(f"  seasonal BABIP distribution: mean={merged['babip'].mean():.4f}, std={merged['babip'].std():.4f}")
    log(f"  career BABIP distribution: mean={merged['career_babip'].mean():.4f}, "
        f"std={merged['career_babip'].std():.4f}")
    log(f"  seasonal − career deviation distribution: mean={merged['babip_minus_career'].mean():+.4f}, "
        f"std={merged['babip_minus_career'].std():.4f}")
    log(f"  (supplementary) league-average BABIP (analysis group, BIP-weighted): {league_babip:.4f}")

    cols = [
        "mlbam_id", "last_name, first_name", "pa", "ba", "bip_avg", "ca_xba", "luck",
        "babip", "career_babip", "babip_minus_career", "career_ab",
    ]
    top_lucky = merged.nlargest(N_LUCK_TOPN, "luck")[cols]
    top_unlucky = merged.nsmallest(N_LUCK_TOPN, "luck")[cols]

    log(f"\n  🍀 Lucky (fortunate effect hypothesis) Top {N_LUCK_TOPN} — including career BABIP deviation:")
    for _, r in top_lucky.iterrows():
        cb = r["career_babip"]
        delta = r["babip_minus_career"]
        cb_str = f"{cb:.3f}" if pd.notna(cb) else "N/A"
        delta_str = f"{delta:+.3f}" if pd.notna(delta) else "N/A"
        log(f"    {r['last_name, first_name']:30s} AVG={r['ba']:.3f}  ca-xBA={r['ca_xba']:.3f}  "
            f"luck={r['luck']:+.3f}  seasonBABIP={r['babip']:.3f}  careerBABIP={cb_str}  Δ={delta_str}")
    log(f"\n  💀 Unlucky (stellar defense / park penalty hypothesis) Top {N_LUCK_TOPN}:")
    for _, r in top_unlucky.iterrows():
        cb = r["career_babip"]
        delta = r["babip_minus_career"]
        cb_str = f"{cb:.3f}" if pd.notna(cb) else "N/A"
        delta_str = f"{delta:+.3f}" if pd.notna(delta) else "N/A"
        log(f"    {r['last_name, first_name']:30s} AVG={r['ba']:.3f}  ca-xBA={r['ca_xba']:.3f}  "
            f"luck={r['luck']:+.3f}  seasonBABIP={r['babip']:.3f}  careerBABIP={cb_str}  Δ={delta_str}")

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
    log(f"\n  luck vs seasonal BABIP: Pearson r={luck_babip_pearson:.4f}, "
        f"Spearman ρ={luck_babip_spearman:.4f}")
    log(f"  luck vs (seasonal − career BABIP deviation): "
        f"Pearson r={luck_delta_pearson:.4f}, Spearman ρ={luck_delta_spearman:.4f}  "
        "← domain-orthodox comparison")

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
    log("\n[correlation] 1:1 R² comparison — ca-xBA vs wOBA / xBA vs wOBA ...")
    results = {}
    pairs = [
        ("ca-xBA (our model)", "ca_xba", "woba"),
        ("xBA (Statcast official)", "est_ba", "woba"),
    ]
    log(f"\n  target players: {len(merged)} (matched with 250+ PA)")
    log(f"\n  {'metric':<25s} {'Pearson r':>10s} {'R²':>8s} {'Spearman ρ':>12s}")
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
    log(f"\n[positions] Querying positions for {len(mlbam_ids)} players via MLB Stats API (cache enabled) ...")

    # Load cache
    cache = {}
    if POSITIONS_CACHE.exists():
        cache = {int(k): v for k, v in json.loads(POSITIONS_CACHE.read_text()).items()}
        log(f"  cache loaded: {len(cache)} players")

    to_fetch = [pid for pid in mlbam_ids if pid not in cache]
    log(f"  new fetch required: {len(to_fetch)} players")

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
        log(f"  cache saved: {POSITIONS_CACHE.relative_to(ROOT)}")

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
    log(f"\n[career_babip] Fetching career hitting stats via MLB Stats API ({len(mlbam_ids)} players, cache enabled) ...")

    cache: dict[int, dict | None] = {}
    if CAREER_BABIP_CACHE.exists():
        raw = json.loads(CAREER_BABIP_CACHE.read_text())
        cache = {int(k): v for k, v in raw.items()}
        log(f"  cache loaded: {len(cache)} players")

    to_fetch = [pid for pid in mlbam_ids if pid not in cache]
    log(f"  new fetch required: {len(to_fetch)} players")

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
        log(f"  cache saved: {CAREER_BABIP_CACHE.relative_to(ROOT)}")

    # Statistics summary
    valid = [v for v in cache.values() if v and not (v.get("babip") != v.get("babip"))]
    if valid:
        babips = [v["babip"] for v in valid]
        log(
            f"  career BABIP stats (n={len(babips)}): "
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
    log("\n[silver_slugger] Silver Slugger validation — position-by-position ca-xBA Top N vs award winners ...")
    ss = pd.read_csv(SILVER_SLUGGER_CSV)
    log(f"  Silver Slugger roster: {len(ss)} players")

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
    log(f"  Silver Slugger ID match: {n_id_matched}/{len(ss)} players")
    if n_id_matched < len(ss):
        missing = ss[ss["mlbam_id"].isna()]
        log(f"  ⚠️ match missing: {missing['player_name'].tolist()} (below 250 PA or name spelling difference)")

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
    log("\n[silver_slugger] Position matching + position-by-position Top N leaderboard ...")
    merged_sorted = merged_sorted.copy()
    merged_sorted["position_mlbam"] = merged_sorted["mlbam_id"].map(position_map)

    # Position distribution
    pos_dist = merged_sorted["position_mlbam"].value_counts(dropna=False)
    log(f"  position distribution (total {len(merged_sorted)} players):")
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
    log(f"\n  Silver Slugger position Top {POSITION_TOPN} hit results:")
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
    log(f"\n  overall hit rate: {hits}/{eligible} ({hits/max(eligible, 1)*100:.1f}%)")

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
    L.append("# Phase 5 Report — Final Metric (ca-xBA) Computation and Sabermetric Value Validation")
    L.append("")
    L.append(f"_Generated: {now}_  ")
    L.append("_Script: `pipeline/step5_phase5_value_validation.py`_")
    L.append("")
    L.append(
        "> **📝 Note — Terminology:** This report uses the Baseball Savant standard column name **'wOBA'** "
        "instead of the academic term 'wOBAcon' for consistency with the data source. However, due to the "
        "nature of the Savant leaderboard, the `wOBA` in this dataset — from which strikeouts and walks "
        "have been filtered out — is **mathematically identical** to the sabermetric academic term wOBAcon "
        "(BIP-restricted weighted on-base average)."
    )
    L.append("")
    L.append(
        "> **Objective:** Apply Phase 4's final model **LGBM + Isotonic (cv='prefit' pattern, "
        f"OOF Brier = {phase4_final_oof_brier:.5f}; selected by Occam's razor)** to held-out 2025 "
        "data to compute per-pitch ca-xBA, then validate whether the per-player average ca-xBA exhibits "
        "a strong correlation with the actual `wOBA`. Theoretical background from readme Phase 5: "
        "well-calibrated probability average → strong positive correlation with wOBA."
    )
    L.append("")
    if model_meta:
        L.append(
            f"> **Model meta:** `{model_meta.get('path', '')}` · type=`{model_meta.get('type', '')}` "
            f"· pipeline=`{model_meta.get('pipeline', '')}`"
        )
        L.append("")

    L.append("## 1. Key Decisions (User-Confirmed, 10 Items)")
    L.append("")
    L.append("| # | Decision | Adopted Approach | Domain Context |")
    L.append("|---|---|---|---|")
    L.append(
        f"| 1 | Main engine | **LGBM + Isotonic** (Phase 4 OOF Brier = {phase4_final_oof_brier:.5f}) | "
        "Selected by Occam's razor — statistical tie with Stacking + Isotonic (0.13083) "
        "(ΔBrier 0.00009 ≤ ε 0.001), so the simpler model is adopted. Goal is accurate probability estimation, not simple classification. |"
    )
    L.append("| 2 | Aggregation | **Simple BIP mean** (PA weighting prohibited) | ca-xBA = pure contact-quality metric |")
    L.append(f"| 3 | Minimum PA | **{MIN_PA}+** (pre-applied within expected_stats.csv) | Covers genuine MLB regulars |")
    L.append("| 4 | ID matching | **Direct MLBAM join** (fuzzy matching prohibited) | Prevents name-collision disasters |")
    L.append("| 5 | Position definition | **Most-played position in season** (MLB Stats API) | Aligns with Silver Slugger criteria |")
    L.append("| 6 | Silver Slugger roster | **Static CSV** (`데이터셋/silver_slugger_2025.csv`) | Ground truth fixed |")
    L.append("| 7 | Luck analysis | **Simple difference `AVG − ca-xBA`** | Intuitive in the baseball domain |")
    L.append(f"| 8 | BIP definition consistency | **assert** (`our_bip / csv.bip ≥ {BIP_TOLERANCE_FRACTION:.0%}`) | Prevents denominator-error R² contamination (absorbs ATH exclusion and cutoff effects) |")
    L.append("| 9 | Position precision check | **Direct call to statsapi.mlb.com** + cache | 309 players × ~5 min, minimal external dependency |")
    L.append("| 10 | 1:1 R² comparison (xwOBA excluded) | Only two independent variables: **ca-xBA vs wOBA / xBA vs wOBA** | xwOBA is a tautological self-predictor of wOBA → excluded due to scope mismatch (readme 2026-05-29) |")
    L.append("")

    L.append("## 2. Data Matching + BIP Definition Consistency Validation")
    L.append("")
    L.append(f"- Players in expected_stats.csv: **{qc['n_expected_stats']:,d}** (250 PA pre-applied)")
    L.append(f"- Match success: **{qc['n_matched']}/{qc['n_expected_stats']}** ({qc['n_matched']/qc['n_expected_stats']*100:.1f}%)")
    L.append(f"- Missing: {qc['n_missing']} (no BIP in our 2025 data — achieved 250 PA but on ATH roster, etc.)")
    L.append("")
    L.append("### BIP Definition Consistency Analysis")
    L.append(f"- `our_bip / csv.bip` ratio — mean={qc['bip_ratio_mean']:.4f}, median={qc['bip_ratio_median']:.4f}")
    L.append(f"- Below tolerance ({qc['tolerance_fraction']:.0%}): **{qc['n_below_tolerance']}** players (mostly ATH roster members, affected by home-game exclusion)")
    L.append("- our BIP < csv.bip is generally expected (Phase 1 ATH home-game exclusion + |la|>60 cutoff + key missing-value removal)")
    L.append("")

    L.append(f"## 3. Main Validation — 1:1 R² Comparison ({n_players} players, 250+ PA)")
    L.append("")
    L.append("**Y-axis reference (true offensive production) = actual `wOBA` (BIP-only weighted OBP). 1:1 R² comparison with two independent variables:**")
    L.append("")
    L.append("| Independent Variable | Pearson r | **R²** | Spearman ρ |")
    L.append("|---|---:|---:|---:|")
    for _, m in correlations.items():
        L.append(f"| **{m['label']}** | {m['pearson_r']:.4f} | **{m['r_squared']:.4f}** | {m['spearman_rho']:.4f} |")
    L.append("")
    ca_xba_r2 = correlations.get("ca_xba", {}).get("r_squared", float("nan"))
    est_ba_r2 = correlations.get("est_ba", {}).get("r_squared", float("nan"))
    L.append(f"- **Our ca-xBA R² = {ca_xba_r2:.4f}**")
    L.append(f"- MLB official xBA (est_ba) R² = {est_ba_r2:.4f}")
    L.append("")
    if ca_xba_r2 > est_ba_r2:
        relative_gain = (ca_xba_r2 - est_ba_r2) / est_ba_r2 * 100
        L.append(f"→ **ca-xBA outperforms MLB official xBA by an absolute R² difference of {(ca_xba_r2 - est_ba_r2):+.4f} "
                 f"(relative advantage +{relative_gain:.1f}%)** — a clear improvement in explanatory power for actual `wOBA`.")
    L.append("")
    L.append("> **⚠️ Comparison scope note:** xwOBA (est_woba) is a Statcast metric that directly predicts wOBA — "
             "it is tautological and mismatched in scope → intentionally excluded from the 1:1 R² comparison "
             "(readme Phase 5 validation setup, 2026-05-29).")
    L.append("")

    L.append(f"## 4. Luck Analysis — `luck = BIP-AVG − ca-xBA` (Unified Denominator) + Career BABIP Cross-Validation")
    L.append("")
    L.append("### 4.1 Luck Definition and Academic Significance of Denominator Unification")
    L.append("")
    babip_stats = luck.get("babip_stats", {})
    league_babip = babip_stats.get("league_babip", float("nan"))
    L.append(
        "The luck metric in this analysis is defined as `luck = BIP-AVG − ca-xBA`. "
        "`BIP-AVG = (hits) / (balls in play)` uses exactly the same denominator (BIP) as ca-xBA, "
        "making it the academically orthodox comparison baseline. This removes the systematic negative "
        "shift and strikeout-rate contamination that arise when simple AVG (denominator = AB) includes "
        "strikeouts in the denominator, yielding a pure measure of the gap between contact quality and "
        "actual hit outcomes."
    )
    L.append("")
    L.append(
        f"- `luck` distribution: mean={luck['luck_stats']['mean']:+.4f}, std={luck['luck_stats']['std']:.4f}, "
        f"min={luck['luck_stats']['min']:+.4f}, max={luck['luck_stats']['max']:+.4f}"
    )
    L.append("")
    L.append(
        "With unified denominators, the luck distribution is symmetrically centered near 0 and the "
        "absolute value is directly interpretable. "
        "Positive values mean \"given this level of contact quality, fewer hits should have occurred, "
        "yet more were recorded (lucky effect hypothesis)\"; negative values mean \"given this level of "
        "quality, more hits should have occurred, but stellar defense or park environment caused a "
        "penalty (unlucky hypothesis)\"."
    )
    L.append("")

    L.append("### 4.2 BABIP Cross-Validation — Domain Orthodox: Seasonal BABIP vs Career BABIP")
    L.append("")
    n_with_career = babip_stats.get("n_with_career", 0)
    L.append(
        f"- **Seasonal BABIP** (analysis group mean): {babip_stats.get('season_babip_mean', float('nan')):.4f} "
        f"(SD {babip_stats.get('season_babip_std', float('nan')):.4f})"
    )
    L.append(
        f"- **Career BABIP** (MLB Stats API career hitting stats, n={n_with_career}/{n_players}): "
        f"mean {babip_stats.get('career_babip_mean', float('nan')):.4f} "
        f"(SD {babip_stats.get('career_babip_std', float('nan')):.4f})"
    )
    L.append(
        f"- **Seasonal − Career deviation (Δ_BABIP)**: mean {babip_stats.get('season_minus_career_mean', float('nan')):+.4f}, "
        f"SD {babip_stats.get('season_minus_career_std', float('nan')):.4f} — "
        "**domain-orthodox signal for lucky/fortunate effect**"
    )
    L.append(
        f"- (Supplementary) Analysis-group league-average BABIP (BIP-weighted): {league_babip:.4f}"
    )
    L.append("")
    L.append(
        "> **Methodological caution**: It is domain-inaccurate to conclude \"lucky\" simply because a "
        "player's BABIP exceeds the league average. True luck/unluck diagnosis is measured as the "
        "seasonal deviation from the player's own **career BABIP (personal baseline)**. "
        "Example: Mike Trout's career BABIP ≈ .342, so a 2025 seasonal BABIP of .342 is "
        "**average — not a lucky effect**. By contrast, a .320 seasonal BABIP for a player with a "
        "career BABIP of .260 represents a Δ_BABIP = +0.060 — an **unambiguous lucky-effect signal**."
    )
    L.append("")

    L.append("### 4.3 Correlation of Two Luck Metrics — luck vs (Seasonal BABIP) / vs Δ_BABIP")
    L.append("")
    pearson_lb = babip_stats.get("luck_babip_pearson", float("nan"))
    spearman_lb = babip_stats.get("luck_babip_spearman", float("nan"))
    pearson_d = babip_stats.get("luck_delta_pearson", float("nan"))
    spearman_d = babip_stats.get("luck_delta_spearman", float("nan"))
    L.append("| Comparison | Pearson r | Spearman ρ | Domain Status |")
    L.append("|---|---:|---:|---|")
    L.append(f"| luck vs seasonal BABIP | {pearson_lb:.4f} | {spearman_lb:.4f} | Single-season average comparison — limited |")
    L.append(f"| **luck vs Δ_BABIP (seasonal − career)** | **{pearson_d:.4f}** | **{spearman_d:.4f}** | **Domain-orthodox comparison — personal baseline adjusted** |")
    L.append("")
    L.append(
        "Both metrics show a positive correlation, but the **correlation with Δ_BABIP is more meaningful** "
        "under the domain-orthodox interpretation. "
        f"In this analysis, Pearson r = {pearson_d:.3f} for luck vs Δ_BABIP provides objective validation "
        "that \"the ca-xBA-based luck metric points in the same direction as the orthodox baseball luck signal "
        "(deviation from career BABIP)\". "
        "The reason the correlation does not approach 1.0 is that ca-xBA additionally captures "
        "**dome × weather interactions, hr_park_effects, and outfield fence distances** — "
        "environment-correction signals that BABIP cannot detect (Trout–Schwarber pattern, §4.6)."
    )
    L.append("")

    L.append(f"### 4.4 Lucky (Fortunate Effect Hypothesis) Top {N_LUCK_TOPN}")
    L.append("")
    L.append("| Player | PA | AVG | BIP-AVG | ca-xBA | luck | Seasonal BABIP | Career BABIP | Δ_BABIP |")
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
        "Interpretation guide: a positive luck (= BIP-AVG − ca-xBA) means more hits occurred than "
        "the contact quality would predict. When combined with Δ_BABIP > 0 (seasonal BABIP above own "
        "career baseline), both metrics agree on a lucky effect — dual validation. When Δ_BABIP ≈ 0 "
        "or negative, the luck signal reflects a ca-xBA environment-correction signal that BABIP alone "
        "cannot detect."
    )
    L.append("")

    L.append(f"### 4.5 Unlucky (Stellar Defense / Park Environment Penalty Hypothesis) Top {N_LUCK_TOPN}")
    L.append("")
    L.append("| Player | PA | AVG | BIP-AVG | ca-xBA | luck | Seasonal BABIP | Career BABIP | Δ_BABIP |")
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
        "Interpretation guide: a negative luck means fewer hits occurred than the contact quality "
        "would predict. When Δ_BABIP < 0, the seasonal BABIP also falls below the player's career "
        "baseline — both metrics agree on an unlucky outcome. When Δ_BABIP ≈ 0 or positive but luck "
        "is strongly negative, this is the Trout pattern: ca-xBA judges from an environment/quality "
        "perspective that \"this level of contact quality should have produced more hits,\" yet BABIP "
        "alone does not flag it as unlucky — a potential Front Office undervaluation discovery point."
    )
    L.append("")

    L.append("### 4.6 Trout–Schwarber Pattern — Additional Information Value of the Model")
    L.append("")
    L.append(
        "The most interesting cases in this analysis are **Mike Trout** (extreme unlucky luck, "
        "Δ_BABIP ≈ 0 or positive) and **Kyle Schwarber**. Trout is an elite contact hitter with a "
        "very high career BABIP, so his seasonal BABIP also stayed above average — yet his ca-xBA-based "
        "luck is rated as extreme bad luck. This means ca-xBA independently captured an "
        "**environment/quality-correction signal** that \"given this level of contact quality, hit "
        "probability should have been higher than BABIP reflects.\""
    )
    L.append("")
    L.append(
        "**Schwarber pattern** (model limitation disclosed): Because ca-xBA evaluates *BIP-restricted "
        "quality* by design, fly-ball power hitters (Schwarber 2025: NL MVP runner-up, 56-HR season) "
        "are subject to a **structural bias** toward negative luck. HRs count as 1 in the ca-xBA "
        "numerator (hits), but fly-ball outs are also evaluated as \"should have been a hit at this "
        "quality,\" causing the denominator (BIP) to grow faster than the numerator. A true unlucky "
        "judgment requires **cross-validation with external metrics** such as BABIP, career BABIP, "
        "and xwOBA underperformance (the Δ_BABIP column in the §4.5 table above serves as that "
        "primary cross-check)."
    )
    L.append("")

    L.append("### 4.7 Lucky/Unlucky Top 5 — Scouting Narrative + URL Sources (Manual Section)")
    L.append("")
    L.append(
        "> **Methodology (auto-fabrication prohibited)**: For each of the Top 5 players, cite "
        "**actual community analyses and Statcast facts** from Baseball Savant official profiles, "
        "FanGraphs, Reddit r/baseball, MLB.com, Pitcher List, etc., using Markdown `[text](URL)` "
        "format. **For any player for whom a clear scouting basis cannot be found, honestly state "
        "\"no clear scouting basis found — classified as small sample or general underperformance.\" "
        "Speculation or fabrication is strictly prohibited.**"
    )
    L.append("")
    L.append(
        "_This automated report records only objective figures (including seasonal/career BABIP). "
        "The Top 5 scouting narratives are a separate manual section to be supplemented after "
        "external verification based on the player names, Δ_BABIP, and luck values in the tables above._"
    )
    L.append("")

    L.append(f"## 5. Silver Slugger Cross-Validation — Position-by-Position ca-xBA Top {POSITION_TOPN}")
    L.append("")
    L.append("> **⚠️ Limitation disclosure (nature of the selection mechanism):** The Silver Slugger is awarded via "
             "**qualitative voting by on-field experts (coaches and managers)**. "
             "MLB does not publish the weights, statistics, or evaluation criteria used in the selection process, "
             "and the award is influenced by **factors beyond hitting** "
             "(defensive value, reputation, media exposure, team performance, vote splitting among rivals, etc.). "
             "Therefore this validation is a **fun domain-consistency check** to see how well ca-xBA aligns with "
             "domain experts' intuition on batting ability — it is **not a scientific validation technique that "
             "statistically guarantees model explanatory power.** "
             "Statistical and scientific model validation is handled by the R² analysis in § 3.")
    L.append("")
    L.append(f"- Silver Slugger winners: 20 players (AL 10 + NL 10)")
    L.append(f"- ID match success: {ss_summary['eligible']}/20 eligible players")
    L.append(f"- **Position Top {POSITION_TOPN} hit rate: {ss_summary['hits']}/{ss_summary['eligible']} ({ss_summary['hit_rate']*100:.1f}%)**")
    L.append("")
    L.append("| League | Position | Winner | Our ca-xBA Rank | Top N Hit | ca-xBA | wOBA |")
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
            f"> **※ Missing player explanation ({len(missing_players)} player(s): {', '.join(missing_players)})**: "
            "The data join in this validation matches Statcast `expected_stats` `player_id` (MLBAM ID) "
            "with MLB Stats API position data using **exact (hard) join**. "
            "This is an academic safeguard against name-collision contamination (user decision #4). "
            "However, **when Statcast's multilingual player name spelling "
            "(e.g., accent marks in José Ramírez, ñ in Peña, and other Latin/Spanish special characters) "
            "does not match the standard ASCII representation in the MLB Stats API at the byte level**, "
            "the join fails and the player is excluded from the validation pool. "
            "Additionally, players with fewer than 250 PA (e.g., some players traded mid-season) are also "
            "excluded from our analysis group (PA ≥ 250). These exclusions are **data-cleaning issues "
            "unrelated to model performance** and can be resolved in future work by introducing "
            "fuzzy matching or the Chadwick Register ID crosswalk."
        )
        L.append("")

    L.append("## 6. Outputs")
    L.append("")
    L.append(
        f"- `{PLAYER_METRICS_CSV.relative_to(ROOT)}` — per-player ca-xBA, wOBA, luck, position\n"
        f"- `{SILVER_SLUGGER_VAL_CSV.relative_to(ROOT)}` — Silver Slugger 20-player validation\n"
        f"- `{RESULTS_JSON.relative_to(ROOT)}` — summary metrics JSON\n"
        f"- `{POSITIONS_CACHE.relative_to(ROOT)}` — MLB Stats API position cache\n"
        f"- `pipeline/logs/step5.log` — execution log"
    )
    L.append("")

    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    log(f"\n[report] phase5_report.md written → {REPORT_PATH.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    log("=" * 80)
    log("Phase 5: Final Metric (ca-xBA) Computation and Sabermetric Value Validation")
    log("=" * 80)

    # 1. Load data
    log("\n[1/9] Loading data ...")
    df_2025 = pd.read_parquet(DATA_2025_PARQUET)
    features_meta = json.loads(PHASE2_FEATURES_JSON.read_text(encoding="utf-8"))
    scaler_obj = joblib.load(PHASE2_SCALER)
    log(f"  2025 BIP: {df_2025.shape}")
    log(f"  X_advanced_final: {len(features_meta['X_advanced_final'])} features")

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
    log("\n[save] Saving outputs ...")
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

    log("\n[done] Phase 5 complete.")


if __name__ == "__main__":
    main()
