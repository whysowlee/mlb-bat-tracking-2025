# Context-Aware xBA (ca-xBA): Predictive Modeling & Sabermetric Validation Roadmap

## 🎯 Project Goal
The goal of this project is to overcome the limitations of Major League Baseball's (MLB) official Expected Batting Average (xBA) — namely, that it evaluates only the pure physical quality of batted balls (launch speed, launch angle) while ignoring the real-world environment.

We combine batted-ball physics data with **the ballpark's physical constraints (fence height, distance) and climate/environmental variables (temperature, wind, altitude, etc.)**, and leverage **tree-ensemble models** capable of capturing non-linear interactions to produce a **context-aware expected batting average, `ca-xBA`**.

Ultimately, to prove that the newly constructed `ca-xBA` did not simply overfit noise (luck) but successfully extracted a hitter's **"environment-optimized true talent,"** we mathematically demonstrate that it predicts a hitter's **next-year run production (wOBAcon) more accurately than the official xBA (Year-to-Year Correlation)**.

---

## 📦 Data Preparation & Setup

> This repository tracks **only the core modules that build the ca-xBA binary-classification model**, plus the reference data and the final report. The large raw dataset, the trained models, and the report-writing/figure-generation tooling are **NOT** included (GitHub's 100MB limit and scope; see `.gitignore`).
> Prepare the raw data as described below and run the pipeline in order — all artifacts will be regenerated.

### 1. Directory Structure

```
텀프로젝트/
├── ca-xBA_Final_Report.docx                  ✅ Final report (English)
├── requirements.txt
├── 데이터셋/
│   ├── statcast_bat_tracking_2024_2025.csv   ⚠️ Must be prepared separately (≈807MB, Git-excluded)
│   ├── ballparks.csv                         ✅ Included (specs for 30 ballparks)
│   ├── silver_slugger_2025.csv               ✅ Included (validation)
│   └── validation_2025_gt.csv                ✅ Included (validation)
└── pipeline/
    ├── step1_fetch_roof_status.py            # Fetch per-game roof_status (MLB Stats API)
    ├── step1_phase1_preprocessing.py         # Phase 1: preprocessing · weather/ballpark merge · Temporal Split
    ├── step2_phase2_correlation_sampling.py  # Phase 2: correlation · scaling · sampling · feature selection
    ├── step3_phase3_ablation.py              # Phase 3: ablation (2x2 factorial design)
    ├── step4_phase4_tuning_stacking.py       # Phase 4: tuning · stacking · calibration
    ├── step4_stacking_recalib.py             # Phase 4: stacking recalibration
    ├── step5_phase5_value_validation.py      # Phase 5: ca-xBA computation · sabermetric validation
    ├── cache/    ⤵️ Auto-generated on run (Open-Meteo weather · roof_status cache)
    ├── logs/     ⤵️ Auto-generated on run
    └── output/   ⤵️ Auto-generated on run (parquet intermediates · trained models · result JSON)
```

> Note: figure-generation scripts (`*_figures.py`) and report-writing scripts (`generate_*`, `restruct_*`, etc.) are intentionally excluded from this repository. The dataset folder name (`데이터셋`) and the raw CSV filename are referenced directly by the code, so keep them as-is.

### 2. Preparing the Raw Data (`statcast_bat_tracking_2024_2025.csv`)

Save the 2024 and 2025 Statcast **bat-tracking-inclusive** pitch-level data into the `데이터셋/` folder under the filename above.

**Option A — Baseball Savant**: Export the 2024 and 2025 season data as CSV from [baseballsavant.mlb.com/statcast_search](https://baseballsavant.mlb.com/statcast_search) and concatenate the two seasons.

**Option B — Collect with pybaseball** (recommended):

```python
import pandas as pd
from pybaseball import statcast

df_2024 = statcast(start_dt="2024-03-20", end_dt="2024-09-30")
df_2025 = statcast(start_dt="2025-03-18", end_dt="2025-09-28")

pd.concat([df_2024, df_2025], ignore_index=True) \
  .to_csv("데이터셋/statcast_bat_tracking_2024_2025.csv", index=False)
```

> Required key columns: `launch_speed`, `launch_angle`, `bb_type`, `events`, `bat_speed`, `swing_length`,
> `game_pk`, `game_year`, `game_date`, `home_team`, `batter` (MLBAM ID), etc.
> The Open-Meteo weather data (8 variables) is fetched and cached automatically via API when step1 runs, so no separate preparation is required.

### 3. Environment

- Python 3.10+ (development environment: conda `mlb-xba`)
- Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Running the Pipeline (in order)

Each step depends on artifacts produced by the previous one (e.g. `pipeline/output/*.parquet`), so run them **strictly in order**.

```bash
python pipeline/step1_fetch_roof_status.py           # Fetch per-game roof_status cache (MLB Stats API)
python pipeline/step1_phase1_preprocessing.py        # Phase 1: preprocessing · weather/ballpark merge · Temporal Split
python pipeline/step2_phase2_correlation_sampling.py # Phase 2: correlation · scaling · sampling · feature selection
python pipeline/step3_phase3_ablation.py             # Phase 3: ablation (2x2 factorial design)
python pipeline/step4_phase4_tuning_stacking.py      # Phase 4: tuning · stacking · calibration
python pipeline/step4_stacking_recalib.py            # Phase 4: stacking recalibration
python pipeline/step5_phase5_value_validation.py     # Phase 5: ca-xBA computation · sabermetric validation
```

> On the first run, `step1_fetch_roof_status.py` and `step1_phase1_preprocessing.py` call external APIs (MLB Stats API, Open-Meteo), so network access is required and it may take several minutes (subsequent runs reuse `pipeline/cache/`).

---

## 🗺️ Step-by-Step Roadmap (5 Phases)

This roadmap is built on a **strict year-by-year split (Temporal Split)** to completely prevent data leakage and to reflect baseball sabermetric philosophy. This section provides a macro-level overview of the entire flow; the detailed methodology and statistics for each phase are covered in depth in the body of the report.

### Phase 1: Data Integration, Domain-Based Preprocessing, and Year-by-Year Split
- **Objective:** Integrate Statcast batted-ball, weather, and ballpark data and apply domain-knowledge-based preprocessing to build a leakage-free training dataset.
- **Key tasks:**
  1. Merge Statcast batted-ball data, Open-Meteo API weather data, and ballpark-spec data.
  2. Perform domain-knowledge-based noise removal (foul pop-up ±60° cutoff) and bat-tracking missing-value handling.
  3. For the 8 dome/retractable-roof ballparks, fetch the per-game `roof_status` from the MLB Stats API and mask the external weather for closed-roof games.
  4. Completely separate and isolate the full dataset into `2024_Data` (training/evaluation) and `2025_Data` (final validation) via a Temporal Split.

### Phase 2: Correlation Analysis, Scaling, Optimal Sampling, and Feature Selection
- **Objective:** Resolve multicollinearity and class imbalance, and finalize the model's input feature set (X_advanced) through conservative feature selection.
- **Key tasks:**
  1. Apply NaN imputation (median) → Robust Scaler → multicollinearity removal (|r| > 0.95, Pearson; domain-priority drop rule).
  2. Adopt a StratifiedKFold 5-fold CV structure and evaluate all of 2024 via OOF (2025 stays fully isolated until Phase 5).
  3. Compare three sampling strategies — original (None), undersampling, and SMOTE — under the same CV to select the one with the minimum OOF Brier.
  4. Drop features that simultaneously fall in the bottom 30% on both RF importance and Mutual Information criteria (preserving X_BASE).

### Phase 3: Effect-Decomposition Experiment (Ablation Study)
- **Objective:** Use a 2x2 factorial design to prove the "non-linear interaction" between batted-ball data and the ballpark environment, securing academic justification for introducing a tree-ensemble model.
- **Key tasks:**
  1. Construct four controlled models (M1–M4): physical variables (X_base) / all variables (X_advanced) × linear (Logistic Regression) / non-linear (XGBoost).
  2. Evaluate Brier · LogLoss · F1 · ROC AUC via the same 5-fold CV OOF as Phase 2 to separate the data effect from the algorithm effect.
  3. Statistically test the non-linear interaction with a 2-way ANOVA (factors: dataset × algorithm) using per-fold metrics as the dependent variable, plus an interaction term.

### Phase 4: Advanced Model Tuning + Calibration + Occam's-Razor Auto-Selection
- **Objective:** Maximize the model's probability calibration using all variables (X_advanced), and automatically select the simpler model when performance is statistically tied.
- **Key tasks:**
  1. Tune Random Forest · XGBoost · LightGBM with RandomizedSearchCV (scoring='neg_brier_score').
  2. Evaluate two candidates — Stacking + Isotonic and Best_Single + Isotonic — under the same external 5-fold CV OOF.
  3. Automatically select the final model via Occam's-razor rule (adopt the simpler model when tied, ΔBrier ≤ ε (0.001)).
  4. Apply Isotonic Calibration with the cv='prefit' pattern to drastically cut computation while maintaining academic equivalence.

### Phase 5: Final Metric (ca-xBA) Computation and Sabermetric Value Validation
- **Objective:** Apply Phase 4's final model to the isolated 2025 data to compute ca-xBA, and statistically verify its superiority over the official MLB xBA in explaining actual wOBA.
- **Key tasks:**
  1. Compute per-batted-ball ca-xBA for 2025 with the final model and aggregate per-player season averages (hard join on MLBAM ID, filtering ≥250 PA).
  2. Visualize the model's advantage by placing the 1:1 R² scatter plots of ca-xBA vs wOBA and official xBA vs wOBA side by side.
  3. Confirm domain value via (AVG − ca-xBA)-based luck analysis and career BABIP cross-validation.
  4. Cross-validate the per-position ca-xBA Top 10 against the 2025 MLB Silver Slugger winners (computing a hit rate).
