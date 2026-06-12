# Context-Aware xBA (ca-xBA) 예측 모델링 및 세이버메트릭스 검증 로드맵

## 🎯 프로젝트 목표 (Project Goal)
본 프로젝트의 목적은 기존 메이저리그(MLB)의 공식 기대 타율(xBA)이 지닌 한계—타구의 순수 물리적 질(발사 속도, 발사 각도)만 평가하고 현실의 환경을 무시한다는 점—를 극복하는 것입니다. 

우리는 타구의 물리 데이터에 **구장의 물리적 제약(펜스 높이, 거리) 및 기후 환경 변수(온도, 바람, 고도 등)**를 결합하고, 비선형적 상호작용을 포착할 수 있는 트리 앙상블 모델(Tree Ensembles)을 활용하여 **상황 인지형 기대 타율인 `ca-xBA`**를 산출합니다.

궁극적으로, 새로 구축한 `ca-xBA`가 단순한 운(Noise)을 과적합한 것이 아니라 타자의 **'환경에 최적화된 진짜 실력(True Talent)'**을 성공적으로 추출해 냈음을 증명하기 위해, 기존 xBA보다 타자의 **내년도 득점 생산력(wOBAcon)을 더 정확하게 예측(Year-to-Year Correlation)**한다는 것을 수학적으로 입증합니다.

---

## 📦 데이터 준비 및 실행 (Setup)

> ⚠️ **대용량 원본 데이터와 학습된 모델은 GitHub 100MB 제한으로 저장소에 포함되어 있지 않습니다.** (`.gitignore` 제외)
> 아래 안내대로 원본 데이터를 직접 준비한 뒤 파이프라인을 순서대로 실행하면 모든 산출물이 재생성됩니다.

### 1. 디렉터리 구조

```
텀프로젝트/
├── 데이터셋/
│   ├── statcast_bat_tracking_2024_2025.csv   ⚠️ 별도 준비 필요 (≈807MB, Git 제외)
│   ├── ballparks.csv                         ✅ 저장소에 포함 (30개 구장 스펙)
│   ├── silver_slugger_2025.csv               ✅ 저장소에 포함 (검증용)
│   └── validation_2025_gt.csv                ✅ 저장소에 포함 (검증용)
└── pipeline/
    ├── step1_phase1_preprocessing.py … step5c_supplementary.py   # 실행 코드
    ├── figures/   ✅ 결과 그래프 포함
    ├── cache/     ⤵️ 실행 시 자동 생성 (Open-Meteo 기상 · roof_status 캐시)
    ├── logs/      ⤵️ 실행 시 자동 생성
    └── output/    ⤵️ 실행 시 자동 생성 (parquet 중간 데이터 · 학습 모델 · 결과 JSON)
```

### 2. 원본 데이터 준비 (`statcast_bat_tracking_2024_2025.csv`)

2024·2025 시즌 Statcast **배트 트래킹 포함** 투구 단위(pitch-level) 데이터를 `데이터셋/` 폴더에 위 파일명으로 저장해야 합니다.

**방법 A — Baseball Savant**: [baseballsavant.mlb.com/statcast_search](https://baseballsavant.mlb.com/statcast_search) 에서 2024·2025 시즌 데이터를 CSV로 내보내 두 시즌을 합칩니다.

**방법 B — pybaseball로 수집** (권장):

```python
import pandas as pd
from pybaseball import statcast

df_2024 = statcast(start_dt="2024-03-20", end_dt="2024-09-30")
df_2025 = statcast(start_dt="2025-03-18", end_dt="2025-09-28")

pd.concat([df_2024, df_2025], ignore_index=True) \
  .to_csv("데이터셋/statcast_bat_tracking_2024_2025.csv", index=False)
```

> 필요한 핵심 컬럼: `launch_speed`, `launch_angle`, `bb_type`, `events`, `bat_speed`, `swing_length`,
> `game_pk`, `game_year`, `game_date`, `home_team`, `batter`(MLBAM ID) 등.
> Open-Meteo 기상 데이터(8종 변수)는 step1 실행 시 API로 자동 호출·캐시되므로 별도 준비가 필요 없습니다.

### 3. 실행 환경

- Python 3.10+ (개발 환경: conda `mlb-xba`)
- 패키지 설치:

```bash
pip install -r requirements.txt
```

### 4. 파이프라인 실행 (순서대로)

각 `stepN`은 이전 단계의 산출물(`pipeline/output/*.parquet` 등)에 의존하므로 **반드시 번호순으로** 실행합니다. `stepNb`는 해당 Phase의 그래프 생성 스크립트입니다.

```bash
python pipeline/step1_phase1_preprocessing.py        # Phase 1: 전처리 · 기상/구장 병합 · Temporal Split
python pipeline/step1b_phase1_figures.py
python pipeline/step2_phase2_correlation_sampling.py # Phase 2: 상관/스케일링/샘플링/Feature Selection
python pipeline/step2b_phase2_figures.py
python pipeline/step3_phase3_ablation.py             # Phase 3: Ablation(2x2 요인 설계)
python pipeline/step3b_phase3_figures.py
python pipeline/step4_phase4_tuning_stacking.py      # Phase 4: 튜닝 · Stacking · Calibration
python pipeline/step4_stacking_recalib.py
python pipeline/step4b_phase4_figures.py
python pipeline/step5_phase5_value_validation.py     # Phase 5: ca-xBA 산출 · 세이버메트릭스 검증
python pipeline/step5b_phase5_figures.py
python pipeline/step5c_supplementary.py
```

> 첫 실행 시 step1이 Open-Meteo API를 호출하므로 네트워크가 필요하며 수 분이 소요될 수 있습니다(이후엔 `pipeline/cache/`에서 재사용). 각 Phase의 결과는 `pipeline/phaseN_report.md`에 기록됩니다.

---

## 🗺️ 단계별 수행 로드맵 (5 Phases)

본 로드맵은 데이터 누수(Data Leakage)를 완벽하게 차단하고, 야구 세이버메트릭스 철학을 반영하기 위해 **엄격한 연도별 분리(Temporal Split)**를 기반으로 수행됩니다. 본 절은 전체 흐름을 거시적으로 조망하며, 각 Phase의 세부 방법론과 통계량은 이후 본문에서 상세히 다룬다.

### Phase 1: 데이터 통합, 도메인 기반 전처리 및 연도별 분리
- **실험 목적:** Statcast 타구·기상·구장 데이터를 통합하고 도메인 지식 기반 전처리를 수행하여, 데이터 누수가 없는 학습용 데이터셋을 구축한다.
- **핵심 작업:**
  1. Statcast 타구 데이터, Open-Meteo API 기상 데이터, 구장 스펙 데이터를 병합한다.
  2. 도메인 지식 기반 노이즈 제거(파울 팝아웃 ±60도 컷오프)와 배트 트래킹 결측치 처리를 수행한다.
  3. 돔/개폐형 구장 8종에 대해 MLB Stats API의 게임별 `roof_status`를 fetch하여, closed roof 경기의 외부 기상을 마스킹한다.
  4. 전체 데이터를 학습·평가용 `2024_Data`와 최종 검증용 `2025_Data`로 완전히 분리·격리한다(Temporal Split).

### Phase 2: 상관관계 분석, 스케일링, 최적 샘플링 및 Feature Selection
- **실험 목적:** 다중공선성과 클래스 불균형을 정리하고 보수적 Feature Selection을 통해 모델 입력 변수군(X_advanced)을 확정한다.
- **핵심 작업:**
  1. NaN imputation(median) → Robust Scaler → 다중공선성 제거(|r| > 0.95, Pearson; 도메인 우선순위 drop 규칙)를 적용한다.
  2. StratifiedKFold 5-fold CV 구조를 채택하고 2024 전체를 OOF로 평가한다(2025는 Phase 5까지 완전 격리).
  3. 원본(None)·언더샘플링·SMOTE 3종을 동일 CV로 비교해 OOF Brier 최소 샘플링 기법을 확정한다.
  4. RF importance와 Mutual Information 2개 기준 모두 하위 30% 동시 진입 변수를 drop한다(X_BASE 보존).

### Phase 3: 효과 분리 실험 (Ablation Study)
- **실험 목적:** 2x2 요인 설계(Factorial Design)로 타구 데이터와 구장 환경 간의 '비선형적 상호작용'을 증명하여, 트리 앙상블 모델 도입의 학술적 정당성을 확보한다.
- **핵심 작업:**
  1. 물리 변수(X_base)/전체 변수(X_advanced) × 선형(Logistic Regression)/비선형(XGBoost)의 4개 통제 모델(M1~M4)을 구성한다.
  2. Phase 2와 동일한 5-fold CV OOF로 Brier·LogLoss·F1·ROC AUC를 평가하여 데이터 효과와 알고리즘 효과를 분리한다.
  3. fold별 메트릭을 종속변수로 한 2-way ANOVA(요인: 데이터셋 × 알고리즘)와 interaction term으로 비선형 상호작용을 통계적으로 검정한다.

### Phase 4: Advanced Model 튜닝 + Calibration + 오캄의 면도날 자동 선정
- **실험 목적:** 전체 변수(X_advanced)로 모델의 확률 정상도(calibration)를 극대화하고, 성능이 통계적으로 동률이면 더 단순한 모델을 자동 선정한다.
- **핵심 작업:**
  1. Random Forest·XGBoost·LightGBM을 RandomizedSearchCV(scoring='neg_brier_score')로 튜닝한다.
  2. Stacking + Isotonic과 Best_Single + Isotonic 두 후보를 동일 외부 5-fold CV OOF로 평가한다.
  3. 오캄의 면도날 규칙(ΔBrier ≤ ε(0.001) 동률 시 더 단순한 모델 채택)으로 최종 모델을 자동 선정한다.
  4. Isotonic Calibration을 cv='prefit' 패턴으로 적용해 연산을 대폭 단축하면서 학술적 동등성을 유지한다.

### Phase 5: 최종 지표(ca-xBA) 산출 및 세이버메트릭스 가치 검증
- **실험 목적:** Phase 4의 최종 모델을 격리된 2025 데이터에 적용해 ca-xBA를 산출하고, 실제 wOBA 설명력에서 MLB 공식 xBA 대비 우위를 통계적으로 검증한다.
- **핵심 작업:**
  1. 최종 모델로 2025 타구별 ca-xBA를 산출하고 선수별 시즌 평균을 집계한다(MLBAM ID 하드 조인, 250 PA 이상 필터링).
  2. ca-xBA vs wOBA / 공식 xBA vs wOBA의 1:1 R² 산점도를 나란히 대조하여 모델 우위를 시각화한다.
  3. (AVG − ca-xBA) 기반 운(Luck) 분석과 통산 BABIP 교차 검증으로 도메인 가치를 확인한다.
  4. 포지션별 ca-xBA Top 10과 2025 MLB 실버 슬러거 수상자를 교차 검증한다(적중률 산출).
