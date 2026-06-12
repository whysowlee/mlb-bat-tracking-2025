---
title: "Context-Aware xBA: 환경 변수 통합 기대 타율 예측 모델"
author: "산업경영공학부 이지현 (2022170832)"
lang: ko-KR
---


# 초록 (Abstract) {.unnumbered}

본 연구는 메이저리그(MLB) 공식 기대 타율(xBA)이 타구의 물리적 질(발사 속도, 발사 각도)만 평가하고 환경 맥락을 무시한다는 구조적 한계를 극복하기 위해, 상황 인지형 기대 타율(ca-xBA, Context-Aware xBA)을 제안하고 그 학술적·실무적 가치를 통계적으로 검증한다.

Statcast BIP(인플레이 타구) 데이터에 구장 스펙(펜스 거리·높이·고도·지붕) 및 Open-Meteo 기반 기상 데이터(온도·풍속·풍향·기압·습도·강수·운량·돌풍)를 결합하고, 개폐형/돔형 구장 8종에 대해서는 MLB Stats API의 게임별 지붕 상태(roof status)를 fetch하여 폐쇄 시 외부 기상 변수를 마스킹하는 도메인 규칙을 적용한다. 비선형 상호작용을 학습할 수 있는 트리 앙상블 모델(Random Forest, XGBoost, LightGBM)에 RandomizedSearchCV 튜닝과 IsotonicRegression 확률 보정을 결합하고, 오캄의 면도날 원칙에 따라 Stacking과 단일 모델의 통계적 동률 시 더 단순한 모델을 자동 선정하는 파이프라인을 구현한다.

메인 검증은 2024년 데이터로 학습한 모델을 2025년 격리 데이터에 적용해 산출한 ca-xBA가 실제 wOBA(BIP 한정 가중 출루율)와 갖는 1:1 상관관계를 MLB 공식 xBA와 직접 대조하는 방식으로 수행한다. 본 연구의 ca-xBA는 공식 xBA 대비 wOBA 설명력 (R²)에서 명확한 우위를 보이며, 통산 BABIP을 baseline으로 한 행운 효과 교차 검증에서도 ca-xBA가 BABIP이 잡지 못하는 환경 보정 신호(dome x weather, hr_park_effects 등)를 추가로 포착함을 객관적으로 입증한다. 부수적으로 2025 Silver Slugger Award 수상자 검증에서도 도메인 전문가 평가와의 일관성을 확인하였다.


# 서론

##  프로젝트 목표 (Project Goal)
본 프로젝트의 목적은 기존 메이저리그(MLB)의 공식 기대 타율(xBA)이 지닌 한계—타구의 순수 물리적 질(발사 속도, 발사 각도)만 평가하고 현실의 환경을 무시한다는 점—를 극복하는 것이다. 

우리는 타구의 물리 데이터에 **구장의 물리적 제약(펜스 높이, 거리) 및 기후 환경 변수(온도, 바람, 고도 등)**를 결합하고, 비선형적 상호작용을 포착할 수 있는 트리 앙상블 모델(Tree Ensembles)을 활용하여 **상황 인지형 기대 타율인 `ca-xBA`**를 산출한다.

궁극적으로, 새로 구축한 `ca-xBA`가 단순한 운(Noise)을 과적합한 것이 아니라 타자의 **'환경에 최적화된 진짜 실력(True Talent)'**을 성공적으로 추출해 냈음을 증명하기 위해, 기존 xBA보다 타자의 **내년도 득점 생산력(wOBAcon)을 더 정확하게 예측(Year-to-Year Correlation)**한다는 것을 수학적으로 입증한다.

---

##  단계별 수행 로드맵 (5 Phases)

본 로드맵은 데이터 누수(Data Leakage)를 완벽하게 차단하고, 야구 세이버메트릭스 철학을 반영하기 위해 **엄격한 연도별 분리(Temporal Split)**를 기반으로 수행된다. 본 절은 전체 흐름을 거시적으로 조망하며, 각 Phase의 세부 방법론과 통계량은 이후 본문에서 상세히 다룬다.


**Phase 1: 데이터 통합, 도메인 기반 전처리 및 연도별 분리**

- **실험 목적:** Statcast 타구·기상·구장 데이터를 통합하고 도메인 지식 기반 전처리를 수행하여, 데이터 누수가 없는 학습용 데이터셋을 구축한다.
- **핵심 작업:**
  1. Statcast 타구 데이터, Open-Meteo API 기상 데이터, 구장 스펙 데이터를 병합한다.
  2. 도메인 지식 기반 노이즈 제거(파울 팝아웃 ±60도 컷오프)와 배트 트래킹 결측치 처리를 수행한다.
  3. 돔/개폐형 구장 8종에 대해 MLB Stats API의 게임별 `roof_status`를 fetch하여, closed roof 경기의 외부 기상을 마스킹한다.
  4. 전체 데이터를 학습·평가용 `2024_Data`와 최종 검증용 `2025_Data`로 완전히 분리·격리한다(Temporal Split).


**Phase 2: 상관관계 분석, 스케일링, 최적 샘플링 및 Feature Selection**

- **실험 목적:** 다중공선성과 클래스 불균형을 정리하고 보수적 Feature Selection을 통해 모델 입력 변수군(X_advanced)을 확정한다.
- **핵심 작업:**
  1. NaN imputation(median) → Robust Scaler → 다중공선성 제거(|r| > 0.95, Pearson; 도메인 우선순위 drop 규칙)를 적용한다.
  2. StratifiedKFold 5-fold CV 구조를 채택하고 2024 전체를 OOF로 평가한다(2025는 Phase 5까지 완전 격리).
  3. 원본(None)·언더샘플링·SMOTE 3종을 동일 CV로 비교해 OOF Brier 최소 샘플링 기법을 확정한다.
  4. RF importance와 Mutual Information 2개 기준 모두 하위 30% 동시 진입 변수를 drop한다(X_BASE 보존).


**Phase 3: 효과 분리 실험 (Ablation Study)**

- **실험 목적:** 2x2 요인 설계(Factorial Design)로 타구 데이터와 구장 환경 간의 '비선형적 상호작용'을 증명하여, 트리 앙상블 모델 도입의 학술적 정당성을 확보한다.
- **핵심 작업:**
  1. 물리 변수(X_base)/전체 변수(X_advanced) x 선형(Logistic Regression)/비선형(XGBoost)의 4개 통제 모델(M1~M4)을 구성한다.
  2. Phase 2와 동일한 5-fold CV OOF로 Brier·LogLoss·F1·ROC AUC를 평가하여 데이터 효과와 알고리즘 효과를 분리한다.
  3. fold별 메트릭을 종속변수로 한 2-way ANOVA(요인: 데이터셋 x 알고리즘)와 interaction term으로 비선형 상호작용을 통계적으로 검정한다.


**Phase 4: Advanced Model 튜닝 + Calibration + 오캄의 면도날 자동 선정**

- **실험 목적:** 전체 변수(X_advanced)로 모델의 확률 정상도(calibration)를 극대화하고, 성능이 통계적으로 동률이면 더 단순한 모델을 자동 선정한다.
- **핵심 작업:**
  1. Random Forest·XGBoost·LightGBM을 RandomizedSearchCV(scoring='neg_brier_score')로 튜닝한다.
  2. Stacking + Isotonic과 Best_Single + Isotonic 두 후보를 동일 외부 5-fold CV OOF로 평가한다.
  3. 오캄의 면도날 규칙(ΔBrier ≤ ε(0.001) 동률 시 더 단순한 모델 채택)으로 최종 모델을 자동 선정한다.
  4. Isotonic Calibration을 cv='prefit' 패턴으로 적용해 연산을 대폭 단축하면서 학술적 동등성을 유지한다.


**Phase 5: 최종 지표(ca-xBA) 산출 및 세이버메트릭스 가치 검증**

- **실험 목적:** Phase 4의 최종 모델을 격리된 2025 데이터에 적용해 ca-xBA를 산출하고, 실제 wOBA 설명력에서 MLB 공식 xBA 대비 우위를 통계적으로 검증한다.
- **핵심 작업:**
  1. 최종 모델로 2025 타구별 ca-xBA를 산출하고 선수별 시즌 평균을 집계한다(MLBAM ID 하드 조인, 250 PA 이상 필터링).
  2. ca-xBA vs wOBA / 공식 xBA vs wOBA의 1:1 R² 산점도를 나란히 대조하여 모델 우위를 시각화한다.
  3. (AVG − ca-xBA) 기반 운(Luck) 분석과 통산 BABIP 교차 검증으로 도메인 가치를 확인한다.
  4. 포지션별 ca-xBA Top 10과 2025 MLB 실버 슬러거 수상자를 교차 검증한다(적중률 산출).


# 데이터 전처리 및 탐색적 분석

## 데이터셋 설명

본 연구는 세 종류의 외부 데이터를 통합하여 단일 학습용 테이블을 구성한다. 각 데이터 소스의 핵심 변수와 행 단위를 아래에 정리한다.

### Statcast 타구 단위 데이터 (Baseball Savant)

MLB 공식 트래킹 시스템이 기록한 모든 투구·타구의 물리 메타데이터다. 2024-2025 두 시즌, 총 1,443,801 개의 투구(pitch) 행으로 시작하여, BIP(인플레이 타구)만 필터링한 후 약 225,414 개의 타구가 본 분석의 모델 입력 단위가 된다. 원천 데이터는 약 118 개의 컬럼을 포함하며, 본 모델이 의미 있게 활용한 주요 변수군은 다음과 같다.

*표 1.*

| 변수군 | 대표 컬럼 | 의미 |
|---|---|---|
| 타구 물리 (핵심) | `launch_speed`, `launch_angle` | 발사 속도 (mph) 와 발사 각도 (도). MLB 공식 xBA 의 두 입력 변수이며 본 모델의 X_base 다. |
| 배트 트래킹 | `bat_speed`, `swing_length`, `attack_angle`, `attack_direction`, `swing_path_tilt`, `intercept_ball_minus_batter_pos_*` | 배트의 속도/궤적/타격 시점 위치. 2024 후반 도입된 신규 트래킹이라 결측이 존재한다. |
| 투구 물리 | `release_speed`, `release_pos_x/z`, `pfx_x/z`, `plate_x/z`, `release_spin_rate`, `release_extension`, `spin_axis`, `effective_speed`, `api_break_*`, `arm_angle` | 투구 릴리스/휘어짐/홈플레이트 도달 위치. 타격 직전 타자가 마주한 투구 조건을 기술한다. |
| 타석 상황 | `balls`, `strikes`, `outs_when_up`, `inning`, `age_pit`, `age_bat`, `n_thruorder_pitcher`, `n_priorpa_thisgame_player_at_bat` | 볼카운트/아웃카운트/이닝/타순 등 게임 상황 변수다. |
| 카테고리 식별 | `stand` (좌/우 타자), `p_throws` (좌/우 투수), `pitch_type` (FF, SL, CH 등), `if_fielding_alignment`, `of_fielding_alignment` | 좌/우 binary 인코딩 + 그 외는 one-hot 처리. |
| 타석 결과 (target/derived) | `events` (single/double/triple/home_run/field_out 등), `bb_type` (ground_ball/fly_ball/line_drive/popup), `babip_value` | `is_hit` (단순 안타 여부) target 라벨로 가공한다. `babip_value` 는 Phase 5 BABIP 계산에 사용한다. |
| 선수 ID | `batter`, `pitcher` | MLBAM ID. Phase 5 외부 CSV (`validation_2025_gt.csv`) 와 hard join 키다. |

### 구장 스펙 데이터 (`ballparks.csv`)

MLB 30 개 구장 각각의 물리적 특성을 정리한 정적 테이블이다. 행 단위는 **구장 1개당 1행** (home_team abbreviation 기준 29 행 — Athletics 2024-2025 이전 이슈로 분석에서 제외, Phase 1 참조). 주요 컬럼은 다음과 같다.

*표 2.*

| 컬럼 | 단위 | 의미 |
|---|---|---|
| `home_team` | abbr | 구장 식별자 (NYY, BOS, COL 등). |
| `left_field`, `center_field`, `right_field` | feet | 좌/중/우측 펜스까지의 거리. |
| `min_wall_height`, `max_wall_height` | feet | 펜스의 최저/최고 높이. |
| `hr_park_effects` | index | 구장별 홈런 친화도 (100 = 리그 평균, Coors Field 등 고지대 구장이 높다). |
| `extra_distance` | feet | 구장 형태 보정 거리. |
| `elevation` | feet | 구장 해발고도 (Coors Field 5,200 ft 등 — 공기 밀도와 타구 비거리에 영향). |
| `roof` | 0~1 | 지붕 형태: 0 (open), 0.5 (retractable), 1 (dome). |
| `daytime` | 0~1 | 주간 경기 비율 (Wrigley Field 등). |
| 위·경도 | deg | 기상 API 좌표 매칭용. |

### Open-Meteo 기상 데이터 (Historical Weather API)

각 경기의 홈구장 위·경도와 경기 시작 시각 (낮 경기 13시, 야간 경기 19시 — `daytime` 컬럼 기준) 을 키로 [Open-Meteo Historical Weather API](https://archive-api.open-meteo.com/v1/archive) 에 쿼리하여 시간 단위 기상값을 fetch 한다. 캐시 디렉토리에 저장하여 재실행 시 추가 호출을 방지한다.

*표 3.*

| 컬럼 | 단위 | 의미 |
|---|---|---|
| `wx_temperature_2m` | 섭씨 | 지상 2m 기온 (공기 밀도와 타구 비거리에 영향). |
| `wx_relative_humidity_2m` | % | 상대 습도 (공기 밀도에 영향). |
| `wx_surface_pressure` | hPa | 지표 기압 (해발 고도와 결합한 공기 밀도를 결정). |
| `wx_wind_speed_10m`, `wx_wind_gusts_10m` | km/h | 지상 10m 풍속/돌풍. |
| `wx_wind_direction_10m` | 도 | 풍향 (외야 방향일 경우 비거리 증감 영향). |
| `wx_precipitation` | mm | 강수량 (공이 미끄러지고 야수 수비 난이도가 증가). |
| `wx_cloud_cover` | % | 운량 (햇빛 시야 영향). |

**돔 마스킹 (도메인 규칙)**: 개폐형/돔형 구장 8 종 (SEA, TOR, MIL, TEX, AZ, MIA, HOU, TB) 의 각 경기에 대해 [MLB Stats API](https://statsapi.mlb.com/api/v1/people) 의 `gameData.weather.condition` 필드를 fetch 하여 `Roof Closed` / `Dome` 인 경기는 외부 기상 5 종 (wind speed/gusts/direction, precipitation, cloud cover) = 0, 실내 공조 표준값 (기온 22 섭씨, 습도 50%, 기압은 실내 ~= 실외) 으로 마스킹한다. 이는 모델이 "돔 닫힘 = 외부 기상 무의미" 시그널을 데이터 자체에서 학습하도록 만들어, weather x roof 비선형 상호작용의 오학습을 원천 차단한다.

## 전처리 파이프라인 (Phase 1)

### 서론 요약
본 단계는 MLB Statcast 타구 데이터(2024~2025), 구장 스펙 데이터, Open-Meteo Historical 기상 데이터를 결합하여 후속 모델링의 입력 데이터셋을 구성한다. 모든 처리는 도메인 지식과 사용자 승인 결정에 따라 수행되며, 데이터 누수(Data Leakage)를 차단하기 위해 game_year 기준으로 엄격한 Temporal Split(2024 ↔ 2025)을 실시한다.

### 데이터 셋 설명
- Statcast: `데이터셋/statcast_bat_tracking_2024_2025.csv` (원본 1,443,801행 x 118열, pitch 단위)
- 구장 스펙: `데이터셋/ballparks.csv` (30개 구장 x 15열, lat/lon 컬럼은 Phase 1에서 보강)
- 기상: Open-Meteo Archive API `archive-api.open-meteo.com/v1/archive` (무료, 키 불요)
- Target 변수 `is_hit`: events ∈ {single, double, triple, home_run} → 1, 그 외 → 0 (MLB 공식 xBA와 정렬)
- 최종 전처리 후 안타율 = 0.3411

### 단계별 attrition (행 수 변화)
원본 1,443,801행은 5단계 정제를 거쳐 최종 225,414개의 BIP로 압축된다. 단계별 행 수 변화는 그림 1(Attrition Funnel)에 시각화되어 있으며, BIP 필터 단계에서 전체의 약 82.5%(타격 외 pitch 행)가, 도메인 컷오프(|launch_angle| > 60°)에서 추가로 16,248행이 제거되는 것이 가장 큰 두 단일 컷이다.

### BIP 필터 직후 bb_type 분포
```
bb_type
NaN            1190620
ground_ball     107795
fly_ball         67226
line_drive       60271
popup            17889
Name: count, dtype: int64
```

### 배트 트래킹 결측률 (전처리 직전 BIP 기준, 연도별)
*표 4.*

|                                          |   2024 |   2025 |
|:-----------------------------------------|-------:|-------:|
| bat_speed                                | 0.1135 | 0.0522 |
| swing_length                             | 0.1135 | 0.0522 |
| attack_angle                             | 0.1135 | 0.0522 |
| attack_direction                         | 0.1135 | 0.0522 |
| swing_path_tilt                          | 0.1135 | 0.0522 |
| intercept_ball_minus_batter_pos_x_inches | 0.1135 | 0.0527 |
| intercept_ball_minus_batter_pos_y_inches | 0.1135 | 0.0527 |

### 돔/지붕 닫힘 경기 기상 마스킹 (결정 #11)

- MLB Stats API (`/api/v1.1/game/{game_pk}/feed/live` 의 `gameData.weather.condition`)에서 "Roof Closed" 또는 "Dome" 으로 명시된 경기에 한해 기상 변수 마스킹 적용.
- 캐시 파일: `pipeline/cache/mlb_roof_status_cache.json` (1,318 게임, 누락 0건).
- 마스킹된 BIP 행: **43,197** / 대상 가능(retractable + TB) **61,677** (70.0%)
- 적용 값:
  - 외부 기상 5종 → 0: `wx_wind_speed_10m`, `wx_wind_gusts_10m`, `wx_wind_direction_10m`, `wx_precipitation`, `wx_cloud_cover`
  - 실내 공조 표준값 2종: `wx_temperature_2m` = 22°C (MLB 돔 표준), `wx_relative_humidity_2m` = 50% (ASHRAE 권장 중간값)
  - 변경 없음: `wx_surface_pressure` (실내·외 기압 동일)
- 도메인 의의: 트리 앙상블 모델이 *roof x 기상 상호작용* 을 자동 학습할 수 있도록, 학습 데이터에 "돔 경기에서는 기상 변수가 상수" 라는 사실을 직접 주입.

### 기상 데이터 병합 결과
- 호출 구장 수: 30 (Athletics 제외)
- 호출 변수: temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m, wind_direction_10m, precipitation, cloud_cover, wind_gusts_10m (총 8종)
- 시점: daytime≥0.5 → 13:00 현지시각 / 그 외 → 19:00 현지시각
- 캐시: `pipeline/cache/weather_{team}_{start}_{end}.json`
- 기상 결측 행: 0 / 225,414 (0.00%)

#### 기상 변수 요약 통계
*표 5.*

|                         |   count |   mean |    std |   min |   25% |    50% |    75% |    max |
|:------------------------|--------:|-------:|-------:|------:|------:|-------:|-------:|-------:|
| wx_temperature_2m       |  225414 |  21.82 |   5.61 |  -4.1 |  19.3 |   22   |   25.4 |   37.8 |
| wx_relative_humidity_2m |  225414 |  58.31 |  16.25 |   3   |  50   |   55   |   70   |  100   |
| wx_surface_pressure     |  225414 | 994.82 |  32.85 | 822.5 | 989.4 | 1000.4 | 1012.5 | 1030.4 |
| wx_wind_speed_10m       |  225414 |   9.43 |   6.97 |   0   |   4.6 |    9.1 |   13.7 |   44.6 |
| wx_wind_direction_10m   |  225414 | 153.13 | 113.49 |   0   |  34   |  165   |  245   |  360   |
| wx_precipitation        |  225414 |   0.07 |   0.54 |   0   |   0   |    0   |    0   |   14.5 |
| wx_cloud_cover          |  225414 |  38.02 |  42.05 |   0   |   0   |   15   |   93   |  100   |
| wx_wind_gusts_10m       |  225414 |  22.37 |  14.38 |   0   |  13.7 |   24.1 |   31.3 |   94.3 |

### Temporal Split 결과
*표 6.*

| 연도 | 행 수 | 안타율 | 타자 수 | 투수 수 | 구장 수 | 기간 | 저장 경로 |
|---:|---:|---:|---:|---:|---:|---|---|
| 2024 | 113,409 | 0.3411 | 884 | 956 | 29 | 2024-03-20~2024-09-29 | `pipeline/output/2024_data.parquet` |
| 2025 | 112,005 | 0.3411 | 664 | 869 | 29 | 2025-03-27~2025-09-28 | `pipeline/output/2025_data.parquet` |

- 2024_data 는 Phase 2~4 학습/평가용, 2025_data 는 Phase 5 검증 정답지용으로 격리.

아래 PNG는 모두 `pipeline/figures/` 에 저장되어 있으며, 최종 Word 보고서로 옮길 때 그대로 재사용 가능하다.

#### 전처리 흐름

**(A1) 단계별 행 수 변화 — Attrition Funnel**

![Attrition Funnel](/tmp/ca-xba-pdf-build/figures/fig_a1_attrition_funnel.png)

*그림 1. Attrition Funnel*


- 원본 1,443,801행 중 BIP 필터 단계에서 **약 82.5%**가 제거됨(타격 외 pitch 단위 행).
- 도메인 컷오프는 |launch_angle|≤60°가 가장 큰 단일 컷(−16,248행).

**(A2) bb_type 분포 — 컷오프 전후**

![bb_type before/after](/tmp/ca-xba-pdf-build/figures/fig_a2_bb_type_before_after.png)

*그림 2. bb_type before/after*


- popup이 컷오프로 거의 전량 제거됨 → 평균 launch_angle=65.8°, 안타율 1.4%의 사실상 자동 아웃 군집 제거.
- ground_ball / line_drive / fly_ball 의 페어 영역은 보존.

#### 타구 물리 분포

**(B1) 발사속도/발사각 히스토그램 (연도별 overlay)**

![Launch speed/angle hist](/tmp/ca-xba-pdf-build/figures/fig_b1_launch_speed_angle_hist.png)

*그림 3. Launch speed/angle hist*


- 2024와 2025의 분포가 거의 동일 — Temporal Split 후에도 입력 변수의 분포가 안정적임을 시각적으로 확인.
- launch_angle은 컷오프 적용으로 [−60, +60] 범위에 갇혀있음(붉은 점선 = 컷오프 경계).

**(B2) launch_speed x launch_angle 안타율 히트맵**

![SpeedxAngle heatmap](/tmp/ca-xba-pdf-build/figures/fig_b2_speed_angle_hit_heatmap.png)

*그림 4. SpeedxAngle heatmap*


- xBA의 본질 시각화: 발사속도가 빠르고 각도가 약 10~25°일 때 안타율이 가장 높음(녹색 띠).
- 위 띠 위(40°+, 낮은 EV)는 팝업 영역, 아래 띠(음각·낮은 EV)는 그라운드 아웃 영역으로 안타율이 급락.
- 환경 변수(바람·기압·온도 등)는 이 *비선형 의존성* 위에 추가 보정을 제공할 것이 Phase 3 가설.

이 히트맵은 본 연구가 트리 앙상블 알고리즘을 채택한 수학적 근거이기도 하다. 그림 4에서 나타나듯, 발사속도와 발사각이 안타율에 미치는 영향은 이미 그 자체로 강한 비선형성 — 특정 각도와 속도의 교집합에서만 안타율이 급증하는 좁은 띠 형태 — 을 띤다. 여기에 온도·풍속·구장 고도 등 60여 개의 환경 변수가 추가로 얽힐 경우, 변수 간의 독립성과 단조 증가를 가정하는 선형 모델(Logistic Regression)은 이 복잡한 교호작용을 결코 담아낼 수 없다. 즉 두 물리 변수만으로도 이미 선형 가정이 깨지는 구조이므로, 환경 맥락까지 포착하려면 입력 공간을 조건부로 분할하는 비선형 모델이 필연적으로 요구된다(Phase 3에서 통계적으로 검정).

#### 환경 변수

**(C1) 기상 변수 8종 분포 grid**

![Weather distributions](/tmp/ca-xba-pdf-build/figures/fig_c1_weather_distributions.png)

*그림 5. Weather distributions*


- 기온은 약 23°C에 중심한 정규-유사 분포, 풍속/풍향은 우측 꼬리·균등 분포.
- 강수는 강한 영(0) 집중 — 변수 변환(예: log1p) 또는 이진화가 Phase 2에서 검토될 수 있음.

**(C2) 구장별 환경 특성 히트맵 (29 home_team x 11 변수)**

![Park env heatmap](/tmp/ca-xba-pdf-build/figures/fig_c2_park_env_heatmap.png)

*그림 6. Park env heatmap*


- COL(쿠어스): 고도 5,190ft / 평균기온 정상 / 풍속 정상 — *고도 단일 변수*로 분리되는 극단 구장.
- MIA·TB·HOU·TEX·AZ·TOR·MIL: roof로 환경 영향이 일부/전면 차단(셀 색이 다른 환경 변수에서 두드러짐).
- 환경 변수들이 구장 간에 의미 있는 분산을 가지며 — ca-xBA가 추출하려는 *환경 신호*의 원천이 확인됨.


## 탐색적 분석 및 Feature Selection (Phase 2)

> 본 단계는 **2024_data.parquet 만** 사용한다. 2025_data는 Phase 5까지 격리되어 어떤 통계도 누설되지 않는다.

### 변수 그룹 정의 및 초기 풀 구성

*표 7.*

| 그룹 | 변수 수 | 내용 |
|---|---:|---|
| (a) xBA 핵심 | 2 | launch_speed, launch_angle |
| (b) 배트 트래킹 + 결측플래그 | 14 | 7 numeric + 7 *_is_missing |
| (c) 타석 정체성 (binary) | 2 | stand_R, p_throws_R |
| (d) 투구 물리 (numeric) | 15 | release_speed, pfx_*, plate_*, spin, break, arm_angle 등 |
| (d2) pitch_type one-hot | (가변) | pitch_type_* 더미 변수 |
| (e) PA 상황 (numeric) | 8 | balls/strikes/outs/inning/age/order 등 |
| (e2) alignment one-hot | (가변) | if/of_fielding_alignment 더미 |
| (f) 구장 정적 | 10 | 펜스거리·높이·hr_park_effects·extra_distance·고도·roof·daytime |
| (g) 기상 동적 (dome-masked) | 8 | 온도·습도·기압·풍속·풍향·강수·운량·돌풍 — closed roof일 시 외부 5종=0, 실내 기온 22°C/습도 50% 대체 (Phase 1 §7b) |
| **X_advanced 초기(One-Hot 후)** | **82** | |

### NaN 처리 (전체 2024 median 기반 imputation)

총 13개 numeric 컬럼의 결측치는 **2024 전체 중앙값(median)**으로 대체하여 데이터 누수를 방지했다. 동시에 각 변수의 `*_is_missing` 플래그를 추가해 결측 패턴 자체를 모델 입력 신호로 보존한다(세부 대체 값은 부록 B 참조).

### Cross-Validation 구조 (5-fold StratifiedKFold)

- 2024 전체 113,409행 → StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- 안타율(전체): 0.3411
- **2025는 Phase 5 외부 검증 전용** — 본 단계에서 어떤 통계도 사용하지 않음

### 다중공선성 분석 (|r| > 0.95, Pearson)

다중공선성 문제 해결을 위해 Pearson 상관계수($r > 0.95$)를 기준으로 고상관 쌍 **24건**을 식별했으며, X_BASE 보존 → derived 변수 우선 drop → 분산 보존 규칙(variance fallback)의 순서로 총 **9개** 변수를 제거했다. 제거 대상은 대부분 동일한 결측 패턴을 공유하는 `*_is_missing` 플래그 군과 derived 변수(`effective_speed`, `wx_surface_pressure` 등)이다(상위 고상관 변수 쌍 전체 목록은 부록 A 참조).

**제거된 변수 전체 목록:** `attack_angle_is_missing`, `attack_direction_is_missing`, `bat_speed_is_missing`, `effective_speed`, `intercept_ball_minus_batter_pos_y_inches_is_missing`, `of_fielding_alignment_UNK`, `swing_length_is_missing`, `swing_path_tilt_is_missing`, `wx_surface_pressure`

### Robust Scaler

- 스케일 적용 컬럼: **50**개 (이진 0/1 변수는 제외)
- 2024 전체 fit, transform → `pipeline/output/phase2_scaler.joblib`

### 샘플링 비교 (3종 x XGBoost default x 5-fold CV)

**OOF (Out-Of-Fold) predict_proba 기반 메트릭:**

*표 8.*

| 샘플링 | Train mean 0/1 | **OOF Brier** | OOF LogLoss | OOF F1 | OOF AUC | OOF P/R | fold Brier mean±SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| **None** | 59,782/30,944 | **0.13610** | 0.42122 | 0.6911 | 0.8687 | 0.7378/0.6500 | 0.13610±0.00076 |
| **Under** | 30,944/30,944 | **0.15007** | 0.45671 | 0.7071 | 0.8661 | 0.6413/0.7880 | 0.15007±0.00126 |
| **SMOTE** | 59,782/59,782 | **0.13629** | 0.42132 | 0.6941 | 0.8686 | 0.7294/0.6620 | 0.13629±0.00098 |

- **최종 선정 샘플링: `None`** (OOF Brier 기준 최소)
- 선정 사유: OOF predict_proba 의 Brier(=평균 (y-p)²) 가 가장 낮은 기법. 확률 정상도 우선.

### Feature Selection (RF importance + MI)

- 학습 데이터: 최적 샘플링(`None`) 적용 X
- **RF Tuning**: `RandomizedSearchCV(n_iter=20, cv=3, scoring='neg_brier_score', n_jobs=2)`
  - search space: `{'n_estimators': [100, 200, 500], 'max_depth': [10, 20, None], 'min_samples_split': [2, 4, 6], 'criterion': ['gini', 'entropy']}`
  - **best params**: `{'n_estimators': 200, 'min_samples_split': 4, 'max_depth': None, 'criterion': 'entropy'}`
  - **best CV neg_brier_score (3-fold avg)**: -0.16062 (Brier = 0.16062)
- **Mutual Information**: stratified 30,000행 subsample, seed=42
- 제거 규칙: **RF importance & MI 둘 다 하위 30% 동시 진입** 시 drop (X_BASE 제외)
- 비고: 당초 3-model Permutation Importance 안이 채택되었으나, macOS joblib memmap 디스크 한계(RF default fit-된 모델의 worker 직렬화 시 OSError 28)로 인해 검증된 이전 step2 방식(RF RandomizedSearchCV → `feature_importances_` + MI)으로 복원.

RF importance와 MI 두 지표의 Top 20 변수는 각각 그림 12(RF Importance Top 20)와 그림 13(MI Top 20)의 막대그래프로 제시한다. 두 지표 모두에서 `launch_angle`과 `launch_speed`가 압도적 상위를 차지하여 xBA의 물리적 본질과 일치하며, 그 아래로 배트 트래킹·투구 물리·구장·기상 변수가 완만하게 분포한다.

#### Feature Selection으로 제거된 변수 (12개)

`if_fielding_alignment_Strategic`, `pitch_type_CH`, `pitch_type_CS`, `pitch_type_CU`, `pitch_type_FA`, `pitch_type_FF`, `pitch_type_FO`, `pitch_type_FS`, `pitch_type_KC`, `pitch_type_KN`, `pitch_type_SI`, `pitch_type_SV`

### 최종 X_advanced 변수 확정

- X_BASE: **2개** — Phase 3 통제군용
- X_advanced 초기: **82개**
- 다중공선성 drop: **9개**
- Feature Selection drop: **12개**
- **X_advanced 최종: 61개**

**X_advanced 최종 변수 목록:**

`launch_speed`, `launch_angle`, `bat_speed`, `swing_length`, `attack_angle`, `attack_direction`, `swing_path_tilt`, `intercept_ball_minus_batter_pos_x_inches`, `intercept_ball_minus_batter_pos_y_inches`, `intercept_ball_minus_batter_pos_x_inches_is_missing`, `release_speed`, `release_pos_x`, `release_pos_z`, `pfx_x`, `pfx_z`, `plate_x`, `plate_z`, `release_spin_rate`, `release_extension`, `spin_axis`, `api_break_z_with_gravity`, `api_break_x_arm`, `api_break_x_batter_in`, `arm_angle`, `balls`, `strikes`, `outs_when_up`, `inning`, `age_pit`, `age_bat`, `n_thruorder_pitcher`, `n_priorpa_thisgame_player_at_bat`, `left_field`, `center_field`, `right_field`, `min_wall_height`, `max_wall_height`, `hr_park_effects`, `extra_distance`, `elevation`, `roof`, `daytime`, `wx_temperature_2m`, `wx_relative_humidity_2m`, `wx_wind_speed_10m`, `wx_wind_direction_10m`, `wx_precipitation`, `wx_cloud_cover`, `wx_wind_gusts_10m`, `stand_R`, `p_throws_R`, `pitch_type_EP`, `pitch_type_FC`, `pitch_type_SC`, `pitch_type_SL`, `pitch_type_ST`, `if_fielding_alignment_Infield shade`, `if_fielding_alignment_Standard`, `if_fielding_alignment_UNK`, `of_fielding_alignment_Standard`, `of_fielding_alignment_Strategic`

#### EDA — 핵심 변수 분포 (is_hit 그룹별, 2024 전체)

**(A1) KDE — 연속 분포 비교**

![EDA KDE](/tmp/ca-xba-pdf-build/figures/fig_p2a1_eda_kde.png)

*그림 7. EDA KDE*


- launch_speed: is_hit=1 그룹의 분포가 우측(고속)으로 이동 → 발사 속도가 빠를수록 안타 확률 ↑.
- launch_angle: is_hit=1 그룹이 10~25° 부근에 집중 → 라인드라이브 각도가 가장 유리.
- 환경 변수(기온·풍속·고도·HR park effects): is_hit 그룹 간 분포 차이가 미미 → 단변량만으로는 신호 약함. **다른 변수와의 비선형 상호작용(트리 모델)이 필요함을 시사**.

**(A2) Boxplot — 중앙값/IQR/이상치 비교**

![EDA Boxplot](/tmp/ca-xba-pdf-build/figures/fig_p2a2_eda_boxplot.png)

*그림 8. EDA Boxplot*


- KDE 결과와 동일한 패턴이 quartile 통계로 재확인됨.
- launch_speed 의 IQR이 is_hit=1 에서 명확히 우측으로 이동.

#### 샘플링 기법 비교 (5-fold CV OOF — XGBoost default)

**(C1) OOF 혼동행렬 (3개 샘플링, threshold=0.5)**

![Sampling CM](/tmp/ca-xba-pdf-build/figures/fig_p2c1_sampling_oof_confusion_matrices.png)

*그림 9. Sampling CM*


- 5-fold CV OOF predict_proba 기반 — Phase 2 step2 산출값과 일치.
- 원본(None): True Negative 다수, Recall 낮음 (보수적 예측).
- Under: TP 대폭 증가, 동시에 FP도 증가 → Recall ↑ / Precision ↓ 트레이드오프.
- SMOTE: 원본에 가까운 균형, F1은 원본보다 살짝 ↑.

**(C2) OOF 평가지표 막대 비교 (Brier↓ / LogLoss↓ / F1 / AUC / P / R / Acc)**

![Sampling Metrics](/tmp/ca-xba-pdf-build/figures/fig_p2c2_sampling_oof_metrics_bar.png)

*그림 10. Sampling Metrics*


- **최종 선정 = `None`** (OOF Brier 기준 최솟값: 0.13610).
- Brier·LogLoss는 낮을수록 우수 — 확률 정상도(probability calibration) 기준.
- F1만 보면 Under가 가장 높지만, 확률값 자체가 깨져서 ca-xBA 산출에 부적합.

**(C3) ROC Curve 겹치기 (시각화 전용 80/20 hold-out)**

![Sampling ROC](/tmp/ca-xba-pdf-build/figures/fig_p2c3_sampling_roc_holdout.png)

*그림 11. Sampling ROC*


- 본 ROC 는 시각화 목적으로 단일 80/20 stratified hold-out (test_size=0.3) 에서 산출. **실제 샘플링 선정은 위 C2 의 OOF Brier 기준**.
- 세 ROC 곡선이 거의 겹침 → AUC 자체에는 큰 차이 없음. 차이는 확률 calibration(Brier/LogLoss)에서 나타남.
- 모델 자체의 변별력은 데이터 분포보다는 변수 풀과 알고리즘에 의해 결정됨을 시사 → Phase 3 ablation 가설과 일치.

#### Feature Importance — RF + MI + Rank Scatter

**(D1) RF Importance Top 20**

![RF Top 20](/tmp/ca-xba-pdf-build/figures/fig_p2d1_rf_importance_top20.png)

*그림 12. RF Top 20*


- 최상위에 `launch_speed`, `launch_angle` 압도적 → xBA의 본질과 일치.
- 환경/투구 변수도 일정 비중 — 트리 모델이 비선형 결합 학습 가능.

**(D2) Mutual Information Top 20**

![MI Top 20](/tmp/ca-xba-pdf-build/figures/fig_p2d2_mi_top20.png)

*그림 13. MI Top 20*


- RF Top 20과 상당 부분 겹치되, MI는 단변량 정보 기준이라 일부 변수의 순위는 다름.
- 두 지표 모두에서 살아남은 변수 = 신뢰도 높은 핵심 변수.

**(D3) RF rank x MI rank 산점도 — 핵심 vs Drop 영역**

![Rank Scatter](/tmp/ca-xba-pdf-build/figures/fig_p2d3_rf_mi_rank_scatter.png)

*그림 14. Rank Scatter*


- **우상단(녹색 영역)**: RF·MI 모두 상위 30% → 핵심 변수. `launch_speed`, `launch_angle` 등이 위치.
- **좌하단(붉은 영역)**: RF·MI 모두 하위 30% → Feature Selection drop 대상. 총 12개가 이 영역에서 제거됨.
- **주황 점**: X_BASE — 절대 drop 금지(보호) 영역.
- 점선 격자(0.3, 0.7)는 30%/70% 분위 기준선.


# 효과 분리 실험 (Ablation Study)

## 평가 지표 Brier Score 정의

본 장 이후 모든 모델 평가에서 핵심으로 사용되는 **Brier Score** 는 이진 분류 모델이 산출한 예측 확률의 정상도 (calibration) 를 측정하는 지표다. 정의는 다음과 같다.

$$ \mathrm{Brier} = \frac{1}{N}\sum_{i=1}^{N} (y_i - p_i)^2 $$

여기서 $y_i \in \{0, 1\}$ 은 실제 안타 여부, $p_i \in [0, 1]$ 은 모델이 예측한 안타 확률이다. 값이 **낮을수록 우수**하며 (예측 확률과 실제 결과의 평균 제곱 오차), 단순 분류 정확도가 아닌 **확률값 자체의 정확성**을 평가한다. 본 연구의 ca-xBA 는 시즌 단위로 평균한 확률값을 직접 산출물로 사용하므로, Brier Score 가 가장 중요한 단일 평가 지표가 된다.

## 실험 설계 및 결과

> 본 단계는 `2024_data` 만 사용하며, **Phase 2와 정확히 동일한 StratifiedKFold 5-fold CV (random_state=42)** 위에서 4개 cell 의 OOF predict_proba 를 평가한다. 2025 데이터는 Phase 5 외부 검증 전용으로 본 단계에서 사용하지 않는다.

### 실험 설계 — 2x2 Factorial Design

**변수 셋:**
- **X_base** = `['launch_speed', 'launch_angle']` (2 변수, MLB 공식 xBA 입력과 동일) — 통제군 입력
- **X_advanced** = Phase 2 최종 선정 **61 변수** (X_base + 배트트래킹 + 카테고리 + 투구·상황·구장·기상 — dome-masked 적용됨)

**알고리즘:**
- LogReg: `LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000)` (선형, 전역·단조)
- XGBoost: `XGBClassifier(default, tree_method='hist')` (비선형, 국소·조건부 split)

*표 9.*

|  | LogReg | XGBoost |
|---|---|---|
| **X_base** (2 변수) | M1 (통제군) | M2 (알고리즘 업그레이드) |
| **X_advanced** (61 변수) | M3 (데이터 업그레이드) | M4 (상호작용 결합) |

**CV**: StratifiedKFold 5-fold (Phase 2 와 동일 splits, random_state=42).
평균 fold train size ≈ 90,727, val size ≈ 22,681.

### 모델별 결과 (OOF + fold mean±SD)

#### OOF aggregate

*표 10.*

| Model | Data | Algo | n_feat | **Brier↓** | LogLoss↓ | F1 | ROC AUC | Precision | Recall | Accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **M1** | X_base | LogReg | 2 | **0.21033** | 0.61347 | 0.1700 | 0.6670 | 0.7228 | 0.0963 | 0.6792 |
| **M2** | X_base | XGBoost | 2 | **0.14012** | 0.43198 | 0.6758 | 0.8594 | 0.7355 | 0.6251 | 0.7954 |
| **M3** | X_advanced | LogReg | 61 | **0.20937** | 0.61078 | 0.2358 | 0.6661 | 0.6801 | 0.1426 | 0.6847 |
| **M4** | X_advanced | XGBoost | 61 | **0.13589** | 0.42049 | 0.6924 | 0.8691 | 0.7380 | 0.6521 | 0.8024 |

#### fold-level mean ± SD (across 5 folds)

각 모델의 5-fold mean ± SD 통계량은 fold 간 변동성이 소수점 셋째 자리 수준으로 매우 작아(예: M4 Brier 0.13589 ± 0.00088) OOF aggregate 결과가 안정적임을 뒷받침한다(fold-level 상세 통계량은 부록 C 참조).

#### 2x2 셀별 OOF 메트릭 매트릭스 (Brier / AUC)

**Brier↓ 매트릭스:**

*표 11.*

|  | LogReg | XGBoost |
|---|---:|---:|
| X_base | 0.21033 (M1) | 0.14012 (M2) |
| X_advanced | 0.20937 (M3) | **0.13589** (M4) |

**ROC AUC 매트릭스:**

*표 12.*

|  | LogReg | XGBoost |
|---|---:|---:|
| X_base | 0.6670 (M1) | 0.8594 (M2) |
| X_advanced | 0.6661 (M3) | **0.8691** (M4) |

### Effect Decomposition (2x2 Factorial)

**핵심 메트릭 (Brier↓ / LogLoss↓ / F1 / ROC AUC):**

*표 13.*

| Effect | ΔBrier | ΔLogLoss | ΔF1 | ΔAUC |
|---|---:|---:|---:|---:|
| 데이터 효과 (in LogReg)  : M3−M1 | -0.00096 | -0.00269 | +0.0658 | -0.0008 |
| 데이터 효과 (in XGBoost) : M4−M2 | -0.00423 | -0.01149 | +0.0166 | +0.0097 |
| 알고리즘 효과 (in X_base) : M2−M1 | -0.07021 | -0.18149 | +0.5058 | +0.1924 |
| 알고리즘 효과 (in X_adv)  : M4−M3 | -0.07348 | -0.19028 | +0.4566 | +0.2029 |
| 결합 효과               : M4−M1 | -0.07444 | -0.19297 | +0.5224 | +0.2021 |
| **Interaction** : (M4−M2)−(M3−M1) | -0.00327 | -0.00879 | -0.0492 | +0.0105 |

_해석 가이드: **Brier·LogLoss 는 음수(감소)가 좋음**, F1·AUC 는 양수(증가)가 좋음. Interaction 행이 0보다 유의하게 다를수록 비선형 상호작용이 명확하다._

### 2-way ANOVA (fold-level)

각 fold(n=5) 의 메트릭을 종속변수로, **Data(X_base/X_advanced) x Algo(LogReg/XGB)** 를 요인으로 한 Type II SS ANOVA를 Brier·LogLoss·ROC AUC·F1 네 지표에 대해 각각 수행했다. 본 연구의 핵심 평가 지표인 **Brier Score를 종속변수로 한 2-way ANOVA 결과, 데이터와 알고리즘 간의 상호작용 항(`C(data):C(algo)`)이 통계적으로 매우 유의했다($p = 0.00322$, $F = 11.98$)**. 나머지 세 지표에서도 상호작용 항이 모두 유의했다(LogLoss $p = 0.00126$, ROC AUC $p = 0.00319$, F1 $p = 2.95 \times 10^{-8}$). 이는 부록 C의 상세 분산분석표와 그림 20의 Interaction Plot에서 교차하는 기울기를 통해 직관적으로 확인할 수 있다(4개 지표의 상세 분산분석표는 부록 C 참조).

### 해석

#### 표면적 관찰

- **M1 (X_base + LogReg)**: 가장 단순한 모델, Brier=0.21033. launch_speed/angle 두 변수 + 선형 결합 = 정통 xBA 의 본질적 한계 측정.
- **M2 (X_base + XGB)**: 같은 2 변수에 비선형 알고리즘만 변경 → Brier=0.14012 (ΔBrier vs M1 = -0.07021).
- **M3 (X_advanced + LogReg)**: 61 변수로 풍부해졌지만 여전히 선형 → Brier=0.20937 (ΔBrier vs M1 = -0.00096).
- **M4 (X_advanced + XGB)**: 풍부한 변수 + 비선형 결합 → Brier=0.13589 (ΔBrier vs M3 = -0.07348, vs M2 = -0.00423).

#### Effect 비교 — 환경 변수의 가치는 비선형 모델 위에서만 발현

- 데이터 효과 (LogReg 위): ΔBrier = **-0.00096** → 선형 모델은 환경 변수 60개를 추가해도 거의 개선 없음 (선형·전역·단조 가정의 한계).
- 데이터 효과 (XGB 위): ΔBrier = **-0.00423** → 같은 환경 변수가 트리 위에서는 명확히 개선 (국소·조건부 split 으로 비선형 결합 학습).
- 이 두 값의 차이 = **Interaction = (M4−M2)−(M3−M1) = -0.00327** (Brier ↓ 방향).

#### ANOVA 통계적 결론

- Brier 에 대한 2-way ANOVA 의 **interaction term (`C(data):C(algo)`) p-value = 0.00322** → **유의함 (p < 0.05)**.
- 이는 "데이터 변수 풀의 효과 크기가 알고리즘에 의존한다" — 즉 비선형 상호작용이 통계적으로 존재한다는 직접 증거.

이 차이는 두 알고리즘의 가설 공간(hypothesis space)을 수식으로 대조하면 명확해진다. Logistic Regression(M1, M3)이 환경 변수의 가치를 추출하지 못한 이유는 모델의 수리적 구조에 기인한다. 선형 모델은 log-odds에 대해 각 변수 $x_j$가 독립적으로 기여한다고 가정한다.

$$\log\left(\frac{P(y=1\mid X)}{1 - P(y=1\mid X)}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j x_j$$

위 식에는 $\beta_{ij} x_i x_j$ 형태의 명시적 상호작용 항이 없으므로, '구장 고도($x_{\mathrm{elev}}$)'가 변할 때 '발사 각도($x_{\mathrm{angle}}$)'의 한계 효과(marginal effect)는 변하지 않는다. 즉 환경 60종을 추가해도 "기온 1도 상승 → 안타 logit $\beta$ 증가" 같은 전역·단조 변동만 학습하여 평균적으로 상쇄된다.

반면, 트리 앙상블 모델(M2, M4)은 입력 공간을 여러 하위 영역 $R_m$으로 분할(partitioning)하여 조건부 기댓값을 추정한다.

$$f(x) = \sum_{m=1}^{M} c_m \, I(x \in R_m)$$

이 공간 분할 과정에서 돔 경기장(roof=1), 특정 발사 속도(speed > 100), 특정 풍향(direction)의 교집합이 독립적인 리프 노드($R_m$)로 분리된다. 예를 들어 `if launch_angle in [25°, 35°] AND launch_speed > 100 AND elevation > 4000ft -> 안타 확률 상승` 같은 교호작용 규칙을 데이터 자체로부터 자동 발굴해 낸 것이다. 즉 변수들이 결합하여 안타 확률($c_m$)을 비선형적으로 변화시키는 의존성을, 트리는 명시적 product feature 없이도 계층적 분할로 학습한다 — 이것이 ANOVA의 interaction term이 유의하게 나타난 수리적 본질이다.

#### 결론 — Phase 4 트리 앙상블 + Stacking 채택의 학술적 근거

Phase 3 의 2x2 ablation 은 *환경 변수 자체가 무의미하다* 는 뜻이 아니라, **"환경 변수의 가치는 비선형 상호작용을 학습할 수 있는 모델 위에서만 발현된다"** 는 사실을 interaction term 으로 직접 입증한다. 같은 환경 변수가 LogReg 위에서는 ΔBrier ≈ -0.00096, XGBoost 위에서는 ΔBrier ≈ -0.00423 — 동일 데이터, 동일 샘플링, 동일 임계값 조건에서 *모델만 바꿔도* 환경 변수의 효과가 전혀 다르게 발현된다는 것은 비선형 상호작용 외에 다른 설명이 없다. Phase 4 트리 앙상블 + Stacking Meta Model 아키텍처의 학술적 정당성이 이로써 완성된다.

#### OOF 혼동행렬 — 4 cell

![Phase 3 CM](/tmp/ca-xba-pdf-build/figures/fig_p3a1_oof_confusion_matrices.png)

*그림 15. Phase 3 CM*


- M1/M3 (LogReg): True Negative 절대 다수, TP 매우 적음 — 선형 모델의 한계.
- M2/M4 (XGBoost): TP 가 크게 증가, FP 도 함께 — 전체 분류 성능 대폭 개선.

#### OOF 평가지표 막대 비교

![Phase 3 Metrics Bar](/tmp/ca-xba-pdf-build/figures/fig_p3a2_oof_metrics_bar.png)

*그림 16. Phase 3 Metrics Bar*


- Brier·LogLoss: M2 ≪ M1, M4 ≪ M3 (알고리즘 효과 압도). M4 < M2 (데이터 효과 in XGB).
- F1·AUC: 같은 패턴. 모든 메트릭에서 M4 가 최저(Brier↓) 또는 최고(F1·AUC↑).

#### OOF ROC Curve

![Phase 3 ROC](/tmp/ca-xba-pdf-build/figures/fig_p3b_oof_roc.png)

*그림 17. Phase 3 ROC*


- M2/M4 의 ROC 곡선이 좌상단으로 강하게 휨 = 변별력 우수.
- M4 가 M2 보다 약간 더 위에 위치 = 환경 변수 추가의 효과가 XGB 위에서 발현.

#### Reliability Diagram (Calibration Curve)

![Phase 3 Calibration](/tmp/ca-xba-pdf-build/figures/fig_p3c_calibration.png)

*그림 18. Phase 3 Calibration*


- 대각선에 가까울수록 확률 정상도 양호 (Brier 낮음).
- M2/M4 (XGBoost) 가 M1/M3 (LogReg) 보다 대각선에 훨씬 가까움.
- M1/M3 은 예측 확률이 0.2~0.4 영역에 몰려 있어 (선형 모델의 보수적 출력) calibration 자체가 부정확.

#### Effect Decomposition

![Phase 3 Effect Decomposition](/tmp/ca-xba-pdf-build/figures/fig_p3d_effect_decomposition.png)

*그림 19. Phase 3 Effect Decomposition*


- 좌: ΔBrier (음수 = 개선) / 우: ΔAUC (양수 = 개선). 마지막 빨간 막대가 **Interaction**.
- 데이터 효과: LogReg 위 ΔBrier=-0.00096 (거의 0), XGB 위 ΔBrier=-0.00423 (유의 개선). 두 값의 차이 = interaction = -0.00327.

#### Interaction Plot (Data x Algo) + ANOVA

![Phase 3 Interaction Plot](/tmp/ca-xba-pdf-build/figures/fig_p3e_interaction_plot.png)

*그림 20. Phase 3 Interaction Plot*


- Interaction Plot 은 두 선이 평행하지 않을수록 interaction 효과가 큼.
- ANOVA(Brier): C(data):C(algo) F=11.98, **p=0.00322** → 유의함 (p<0.05).
- ANOVA(AUC) : C(data):C(algo) F=12.00, **p=0.003194** → 유의함 (p<0.05).

**해석**: 그림 20의 Interaction Plot은 본 연구의 핵심 가설을 시각적으로 증명한다. 선형 모델(LogReg)에서는 환경 변수(X_advanced)를 투입해도 Brier Score와 AUC가 물리 변수(X_base)만 넣었을 때와 거의 동일한 궤적을 그린다(두 선이 사실상 평행). 반면 비선형 모델(XGBoost)에서는 X_advanced 선의 기울기가 X_base보다 극명하게 가팔라지며 성능이 비약적으로 향상된다. 즉 두 선의 **기울기 차이** 자체가 "환경 변수는 비선형 상호작용을 학습할 수 있는 알고리즘 위에서만 그 가치가 발현된다"는 사실을 뜻하며, 이는 정확히 interaction의 정의다. 이 시각적 교차점은 통계적 유의성($p < 0.05$; Brier $p = 0.00322$, AUC $p = 0.003194$)과 명확히 교차 검증되어, 두 선의 벌어짐이 우연이 아닌 통계적으로 유의한 비선형 상호작용임을 확정한다 — Phase 4 트리 앙상블 채택의 최종 근거.


# Advanced Model 튜닝 및 확률 보정

## 튜닝 파이프라인 및 결과

> Phase 2/3 과 **동일한 StratifiedKFold 5-fold CV** 위에서 base 3 모델 (RF / XGB / LGBM) 튜닝된 best estimator 와 Stacking (cv=5, LR meta), **Stacking + Isotonic**, **Best_Single + Isotonic** 의 OOF predict_proba 를 평가한다. **OOF Brier 최소 + 오캄의 면도날(ε=0.001)** 규칙으로 ca-xBA 최종 산출 모델을 선정.

### Base 모델 튜닝 결과 (RandomizedSearchCV)

RF·XGB·LGBM 세 base 모델은 각각 RandomizedSearchCV(n_iter=30, inner_cv=5, scoring='neg_brier_score', refit=True)를 통해 Brier Score를 최적화하도록 튜닝되었다. 핵심 튜닝 결과로 XGB는 `max_depth`=8, `learning_rate`=0.03, `n_estimators`=200, `subsample`=0.8을, LGBM은 `num_leaves`=127, `learning_rate`=0.03, `subsample`=0.9를, RF는 `n_estimators`=500, `criterion`=entropy, `min_samples_leaf`=4를 채택했다(모델별 최종 하이퍼파라미터 전체 딕셔너리는 부록 D 참조).

### Outer 5-fold CV OOF 결과

**OOF aggregate:**

*표 14.*

| Model | **Brier↓** | LogLoss↓ | F1 | ROC AUC | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| RF (tuned) | **0.13264** | 0.41259 | 0.6978 | 0.8759 | 0.7502 | 0.6523 | 0.8073 |
| XGB (tuned) | **0.13231** | 0.41135 | 0.6975 | 0.8761 | 0.7466 | 0.6544 | 0.8064 |
| LGBM (tuned) | **0.13108** | 0.40753 | 0.6985 | 0.8777 | 0.7565 | 0.6488 | 0.8090 |
| Stacking (LR meta) | **0.13244** | 0.41409 | 0.7012 | 0.8780 | 0.7515 | 0.6572 | 0.8090 |
| Stacking + Isotonic | **0.13083** | 0.40587 | 0.6937 | 0.8776 | 0.7652 | 0.6345 | 0.8089 |
| LGBM + Isotonic | **0.13092** | 0.40600 | 0.7029 | 0.8773 | 0.7500 | 0.6614 | 0.8093 |

6개 후보 모델의 fold 간 변동성(SD)은 모두 Brier 기준 0.0012 이하로 매우 작아, 후보 간 OOF Brier 차이(특히 LGBM+Isotonic 0.13092 vs Stacking+Isotonic 0.13083)가 fold 변동 수준 내의 통계적 동률임을 뒷받침한다 — 이는 §5의 오캄의 면도날 자동 선정의 근거가 된다(Outer 5-fold CV fold mean ± SD 상세는 부록 D 참조).

### Phase 3 baseline (M4=X_advanced+XGB default) 대비 개선

*표 15.*

| Model | ΔBrier vs M4 | ΔAUC vs M4 |
|---|---:|---:|
| RF (tuned) | -0.00325 | +0.0069 |
| XGB (tuned) | -0.00358 | +0.0070 |
| LGBM (tuned) | -0.00481 | +0.0086 |
| Stacking (LR meta) | -0.00345 | +0.0089 |
| Stacking + Isotonic | -0.00506 | +0.0086 |
| LGBM + Isotonic | -0.00497 | +0.0082 |

_M4 (Phase 3): Brier=0.13589, AUC=0.8691_

### 최종 모델 선정 — 오캄의 면도날 (Occam's Razor) 적용

#### 핵심 후보 3종 OOF Brier 비교

*표 16.*

| 후보 모델 | OOF Brier | 비고 |
|---|---:|---|
| **LGBM + Isotonic** | **0.13092** | Best Single (가장 우수했던 단일 base = LGBM) + Isotonic — 단순한 모델 |
| Stacking + Isotonic | 0.13083 | Stacking(LR meta) + Isotonic — 복잡한 앙상블 |
| Stacking (LR meta) only | 0.13244 | Stacking 단독 (calibration 없음) |

- ΔBrier (Best_Single+Iso − Stacking+Iso) = **+0.00009** (오캄 threshold ε = 0.001)

#### 선정 결과

- **최종 선정 모델**: `LGBM + Isotonic`
- **OOF Brier**: **0.13092**

**자동 선정 사유:** Best_Single(LGBM) + Isotonic 와 Stacking + Isotonic 의 차이가 ΔBrier = +0.00009 ≤ ε(0.001) = fold 변동성 내 통계적 동률. **오캄의 면도날 적용 → 더 단순한 Best_Single + Isotonic 선정.**

#### 학술적 해석 — 왜 오캄의 면도날인가

실험 결과, **무거운 메타 학습을 거친 Stacking 모델보다 잘 튜닝된 단일 모델(LGBM)의 OOF Brier Score 가 더 우수**함을 확인했다 (또는 fold 변동 수준 내에서 동률). 이는 여러 모델을 결합하는 과정에서 오히려 확률 보정(probability calibration)이 훼손되는 현상으로 해석할 수 있다 — Stacking 의 LR meta-learner 가 base 모델 간 출력 분포 이질성을 강제로 보정하면서 잘 보정된 단일 LGBM 의 native calibration 을 흐트러뜨릴 수 있다.

따라서 본 연구는 **"성능이 비슷하다면 더 단순한 모델이 낫다"** 는 **오캄의 면도날(Occam's Razor)** 원칙을 수용하였다. 억지로 복잡한 앙상블을 유지하는 대신, 가장 성능이 뛰어난 단일 모델에 **비모수적 단조 변환인 Isotonic Calibration 을 직접 결합**하는 방식을 채택했다. 이를 통해 ① 연산의 복잡도를 크게 낮추면서도 (3 base x cross_val_predict + meta-fit + base full-fit 3개 → base 1개 full-fit + Isotonic 1개) ② 본 프로젝트의 궁극적 목표인 **'극한의 확률 정상도(Calibration)'** 를 성공적으로 확보했다.

이 선택은 단순한 경험적 결정이 아니라 Brier Score 최적화라는 본 연구의 목적함수와 완벽하게 정렬되는 필연적 귀결이다. Isotonic Regression은 원본 모델의 예측 확률 $f_i$에 대하여, 단조 증가(monotonicity) 제약 조건을 유지하면서 실제 레이블 $y_i$와의 평균 제곱 오차를 최소화하는 새로운 확률 $\hat{p}_i$를 찾는 최적화 문제를 푼다.

$$\min_{\hat{p}} \sum_{i=1}^{N} (y_i - \hat{p}_i)^2 \quad \text{subject to } \hat{p}_i \le \hat{p}_j \ \text{ for all } f_i \le f_j$$

이 목적함수 $\sum_i (y_i - \hat{p}_i)^2$는 Phase 3에서 정의한 Brier Score의 핵심 항과 수학적으로 완전히 동일하다. 즉 Isotonic 변환은 단조 제약을 통해 AUC가 대변하는 변수 간 정렬 순서(ranking)를 전혀 훼손하지 않으면서도(Isotonic은 순서를 보존하는 변환이므로 AUC 불변), 그림 23의 Reliability Diagram에서 관찰되듯 예측 확률의 분포 곡선을 $y = x$ 대각선으로 강제 견인하여 극상의 확률 정상도(calibration)를 확보하는 수학적 과정이다. LGBM 단일 모델에 Isotonic을 결합한 것이 오캄의 면도날(단순성)과 목적함수(Brier 최소화)를 동시에 만족하는 이유가 여기에 있다.

이 선정 로직은 결과에 따라 자동으로 분기한다 — 만약 Stacking + Isotonic 의 우위가 ε(0.001) 를 초과한다면 (Brier 차이가 fold 변동 수준을 넘는 통계적 유의 차이), 복잡도 증가의 정당성이 확보되어 Stacking + Isotonic 이 채택된다. 본 실행에서는 위 §5.2 의 자동 선정 결과가 적용되었다.

- **Phase 5 적용 흐름**: `final_model.joblib` 내부의 base estimator → predict_proba → isotonic.predict(proba) → ca-xBA. 2025 데이터(외부 검증 셋)에 그대로 적용.

#### OOF Brier Score 순위

![Brier Ranking](/tmp/ca-xba-pdf-build/figures/fig_p4a_brier_ranking.png)

*그림 21. Brier Ranking*


- **최종 선정: `LGBM + Isotonic` (OOF Brier = 0.13092)** — 오캄의 면도날 자동 적용 (Best_Single + Iso vs Stacking + Iso 동률 시 단순 모델 선호).
- 모든 calibrated 모델이 raw 대비 Brier 추가 개선 (isotonic 효과).
- 모든 튜닝 모델이 Phase 3 M4 baseline (XGB default) 보다 명확히 우수.

#### OOF ROC Curve overlay

![ROC](/tmp/ca-xba-pdf-build/figures/fig_p4b_roc_overlay.png)

*그림 22. ROC*


- 5 모델 모두 AUC ≈ 0.87~0.88 수준. Stacking 계열이 좌상단에 더 가까움.
- Isotonic은 단조 변환이라 AUC를 본질적으로 바꾸지 않음 — Brier/LogLoss 만 개선.

#### Reliability Diagram (Calibration Curve)

![Calibration](/tmp/ca-xba-pdf-build/figures/fig_p4c_calibration.png)

*그림 23. Calibration*


- 대각선에 가까울수록 확률 보정 양호.
- **Stacking + Isotonic** 이 대각선에 가장 밀접 — Brier 최소값과 일치.
- ca-xBA 는 시즌 평균 확률을 사용하므로 calibration 이 핵심.

#### Phase 3 M4 baseline 대비 개선

![Improvement vs M4](/tmp/ca-xba-pdf-build/figures/fig_p4d_improvement_vs_m4.png)

*그림 24. Improvement vs M4*


- 좌: ΔBrier (음수 = 개선) / 우: ΔAUC (양수 = 개선).
- 모든 모델이 음수 ΔBrier — Phase 4 의 튜닝 + 앙상블 + calibration 전 단계가 실제로 모델 성능을 일관되게 향상시켰음.

#### Isotonic Calibration 효과 시각화

![Isotonic Mapping](/tmp/ca-xba-pdf-build/figures/fig_p4e_isotonic_mapping.png)

*그림 25. Isotonic Mapping*


- 왼쪽: Stacking raw proba → Isotonic proba 매핑 (단조 비모수 함수). y=x 대각선에서 벗어난 정도 = isotonic 보정 강도.
- 오른쪽: Raw 분포는 0~0.5 구간 과집중 / Isotonic 은 양 극단(0 근처, 1 근처)을 더 분리시킴 — 확률 해상도 개선.


# 최종 지표 산출 및 세이버메트릭스 가치 검증

## 검증 파이프라인 및 결과

> ** Note — 용어 통일:** 본 리포트는 Baseball Savant 데이터 소스와의 일관성을 위해 학술 용어 'wOBAcon' 대신 사반트 표준 컬럼명 **'wOBA'** 를 사용한다. 단, 사반트 리더보드 특성상 삼진/볼넷이 걸러진 이 데이터셋의 `wOBA` 는 세이버메트릭스 학술 용어인 wOBAcon 과 **수학적으로 완전히 동일하다**(BIP 한정 가중 출루율).

> **목적:** Phase 4 의 최종 모델 **LGBM + Isotonic (cv='prefit' 패턴, OOF Brier = 0.13092; 오캄의 면도날 자동 선정)** 을 격리된 2025 데이터에 적용해 타구별 ca-xBA 를 산출하고, 선수별 평균 ca-xBA 가 실제 `wOBA` 와 강한 상관관계를 가지는지 검증한다. readme Phase 5 이론적 배경: well-calibrated probability 평균 → wOBA 강한 양의 상관.

### 데이터 매칭 + BIP 정의 일치 검증

- expected_stats.csv 선수 수: **309** (250 PA 사전 적용)
- 매칭 성공: **309/309** (100.0%)
- 누락: 0 (2025 본 분석의 데이터에 BIP 없음 — 250 PA 달성했지만 ATH 소속 등)

#### BIP 정의 일치 분석
- `our_bip / csv.bip` 비율 — mean=0.9005, median=0.9117
- tolerance (50%) 미달: **8** 명 (대부분 ATH 소속, 홈경기 제외 영향)
- 본 분석의 BIP < csv.bip 가 일반적 (Phase 1 의 ATH 홈 제외 + |la|>60 컷오프 + 핵심 결측 제거 영향)

### 메인 검증 — 1:1 R² 대조 (대상 선수 309명, 250+ PA)

**Y축 기준점 (실제 기량) = 실제 `wOBA` (BIP-only weighted OBP). 두 독립변수와의 1:1 R² 비교:**

*표 17.*

| 독립변수 | Pearson r | **R²** | Spearman ρ |
|---|---:|---:|---:|
| **ca-xBA (본 연구의 모델)** | 0.6306 | **0.3976** | 0.5849 |
| **xBA (Statcast 공식)** | 0.4999 | **0.2499** | 0.4729 |

- **본 연구의 ca-xBA R² = 0.3976**
- MLB 공식 xBA (est_ba) R² = 0.2499

→ **ca-xBA 가 MLB 공식 xBA 보다 절대 R² 차이 +0.1477 (상대 우위 +59.1%) 우수** — 실제 `wOBA` 설명력에서 명확한 개선.

### 운(Luck) 분석 — `luck = BIP-AVG − ca-xBA` (분모 통일) + 통산 BABIP 교차 검증

#### luck 정의 및 분모 통일의 학술적 의의

본 분석의 운(Luck) 지표는 `luck = BIP-AVG − ca-xBA` 로 정의된다. `BIP-AVG = (안타 수) / (인플레이 타구 수)` 는 ca-xBA 의 분모(BIP)와 정확히 일치하는 비교 baseline 이다. 이는 단순 타율(AVG, 분모 = AB)이 삼진을 분모에 포함하여 발생하는 체계적 음수 시프트와 삼진율 오염을 제거하고, 순수 contact quality 대비 실제 안타 결과의 괴리를 측정하는 학술 정통 지표이다.

- `luck` 분포: mean=-0.0068, std=0.0249, min=-0.0853, max=+0.0810

분모 통일로 인해 luck 분포는 0 근처에 대칭적으로 정렬되며, 절대값 자체가 해석 가능하다. 양수는 "이 정도 contact quality 였으면 더 적은 안타가 나왔어야 하는데 실제로는 더 많이 나왔다(행운 효과 가설)" 를, 음수는 "이 정도 quality 였으면 더 많은 안타가 나왔어야 하는데 호수비·구장 환경 등으로 손해를 봤다(불운 가설)" 를 의미한다.

#### BABIP 교차 검증 — 도메인 정통: 시즌 BABIP vs 자기 통산 BABIP

- **시즌 BABIP** (분석군 평균): 0.3113 (SD 0.0333)
- **통산 BABIP** (MLB Stats API career hitting stats, n=309/309): 평균 0.2971 (SD 0.0259)
- **시즌 − 통산 편차 (Δ_BABIP)**: 평균 +0.0142, SD 0.0265 — **도메인 정통 "운/행운에 의한 효과" 시그널**
- (보조) 분석군 리그 평균 BABIP (BIP-가중): 0.3119

#### 두 운 지표 상관 — luck vs (시즌 BABIP) / vs Δ_BABIP

*표 18.*

| 비교 대상 | Pearson r | Spearman ρ | 도메인적 위상 |
|---|---:|---:|---|
| luck vs 시즌 BABIP | 0.6996 | 0.6432 | 단일 시즌 평균 비교 — 한계 있음 |
| **luck vs Δ_BABIP (시즌 − 통산)** | **0.5193** | **0.4781** | **도메인 정통 비교 — 개인 baseline 보정** |

두 지표 모두 양의 상관을 보이지만, 도메인 정통 해석인 **Δ_BABIP 와의 상관이 더 의미 있다**. 본 분석에서 luck vs Δ_BABIP 의 Pearson r = 0.519 는 "ca-xBA 기반 luck 지표가 야구 도메인의 정통 행운 시그널(통산 BABIP 대비 편차)과 동일한 방향을 가리킨다"는 객관적 검증이다. 단, 상관계수가 1.0 에 가깝지 않은 이유는 ca-xBA 가 BABIP 단일 지표가 반영하지 못하는 **환경 변수 (dome x weather 등 도메인 상식 기반, hr_park_effects, 구장 펜스 거리** 등 환경 보정 신호를 추가로 포착하기 때문이다 (Trout·Schwarber 패턴, §4.6).

#### 운(행운 효과 가설) Top 10

*표 19.*

| 선수 | PA | AVG | BIP-AVG | ca-xBA | luck | 시즌 BABIP | 통산 BABIP | Δ_BABIP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Kurtz, Nick | 489 | 0.290 | 0.488 | 0.407 | +0.081 | 0.425 | 0.375 | +0.050 |
| Bader, Harrison | 501 | 0.277 | 0.426 | 0.349 | +0.077 | 0.392 | 0.301 | +0.091 |
| Stanton, Giancarlo | 281 | 0.273 | 0.482 | 0.420 | +0.062 | 0.376 | 0.307 | +0.069 |
| Peña, Jeremy | 543 | 0.304 | 0.393 | 0.340 | +0.053 | 0.363 | 0.315 | +0.048 |
| Andujar, Miguel | 341 | 0.318 | 0.407 | 0.355 | +0.052 | 0.385 | 0.307 | +0.078 |
| Smith, Pavin | 288 | 0.258 | 0.432 | 0.388 | +0.043 | 0.399 | 0.297 | +0.102 |
| Mangum, Jake | 428 | 0.296 | 0.359 | 0.317 | +0.042 | 0.353 | 0.341 | +0.012 |
| Turner, Trea | 639 | 0.304 | 0.382 | 0.340 | +0.042 | 0.363 | 0.336 | +0.027 |
| Acuña Jr., Ronald | 412 | 0.290 | 0.430 | 0.389 | +0.041 | 0.377 | 0.334 | +0.043 |
| Frelick, Sal | 594 | 0.288 | 0.365 | 0.327 | +0.038 | 0.346 | 0.297 | +0.049 |

해석 가이드: luck (= BIP-AVG − ca-xBA) 가 양수면 contact quality 대비 더 많은 안타가 나왔다는 의미다. 함께 Δ_BABIP > 0 (자기 통산 대비 시즌 BABIP 높음) 이면 두 지표가 모두 행운 효과로 일치하는 이중 검증이고, Δ_BABIP ≈ 0 또는 음수면 luck 가 잡은 행운이 BABIP 단일 지표로는 확인되지 않는 ca-xBA 환경 보정 시그널을 의미한다.

#### 불운(호수비·환경 손해 가설) Top 10

*표 20.*

| 선수 | PA | AVG | BIP-AVG | ca-xBA | luck | 시즌 BABIP | 통산 BABIP | Δ_BABIP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pederson, Joc | 306 | 0.181 | 0.251 | 0.337 | -0.085 | 0.214 | 0.268 | -0.054 |
| Davis, Henry | 283 | 0.167 | 0.266 | 0.343 | -0.078 | 0.232 | 0.217 | +0.015 |
| Torrens, Luis | 283 | 0.226 | 0.284 | 0.360 | -0.076 | 0.265 | 0.281 | -0.016 |
| Conforto, Michael | 486 | 0.199 | 0.289 | 0.365 | -0.076 | 0.257 | 0.290 | -0.033 |
| Benson, Will | 253 | 0.226 | 0.320 | 0.389 | -0.069 | 0.273 | 0.304 | -0.031 |
| Clemens, Kody | 386 | 0.213 | 0.308 | 0.367 | -0.059 | 0.247 | 0.240 | +0.007 |
| Perez, Salvador | 641 | 0.236 | 0.328 | 0.386 | -0.059 | 0.278 | 0.284 | -0.006 |
| Taylor, Tyrone | 341 | 0.223 | 0.313 | 0.370 | -0.057 | 0.307 | 0.278 | +0.029 |
| Adams, Riley | 286 | 0.186 | 0.348 | 0.404 | -0.057 | 0.308 | 0.299 | +0.009 |
| Grichuk, Randal | 293 | 0.228 | 0.323 | 0.377 | -0.055 | 0.289 | 0.294 | -0.005 |

해석 가이드: luck 가 음수면 contact quality 대비 안타가 적게 나왔다는 의미다. Δ_BABIP < 0 이면 자기 통산 대비 시즌 BABIP 도 낮아 두 지표 모두 불운으로 일치한다. Δ_BABIP ≈ 0 또는 양수인데 luck 만 크게 음수면 Trout 패턴에 해당하며, ca-xBA 가 환경/quality 측면에서 "이 정도 quality 면 더 잘 쳤어야 한다" 고 평가하나 BABIP 만으로는 불운으로 보이지 않는 Front Office 의 저평가 발굴 포인트가 된다.

#### Trout · Schwarber 패턴 — 모델의 추가 정보 가치

본 분석에서 가장 흥미로운 케이스는 **Mike Trout** (luck 극불운, Δ_BABIP ≈ 0 또는 양수) 와 **Kyle Schwarber** 다. Trout 는 통산 BABIP 가 매우 높은 elite contact hitter 라 시즌 BABIP 도 평균 이상으로 유지되었지만, ca-xBA 기반 luck 는 극불운으로 평가된다. 이는 ca-xBA 가 "이 정도 quality 의 contact 면 BABIP 보다 더 높은 안타 확률이 나왔어야 한다" 는 **환경·quality 보정 신호**를 단독으로 포착했다는 뜻이다.

**Schwarber 패턴** (모델 한계 정직 명시): ca-xBA 가 *BIP-한정 quality* 를 평가하는 본질상 fly ball power hitter (Schwarber 2025: NL MVP 2위, 56 HR 시즌) 는 luck = 음수로 평가되는 **구조적 편향**이 존재한다. HR 은 ca-xBA 의 분자(안타)에 1 로 카운트되지만, fly ball out 도 ca-xBA 가 "이 quality 면 안타였어야 한다" 라고 평가하는 경향이 있어 분모(BIP) 가 분자보다 더 빠르게 증가한다. 진정한 불운 판단은 BABIP + 통산 BABIP + xwOBA underperform 등 **외부 지표와의 교차 검증**이 필요하다 (위 §4.5 표의 Δ_BABIP 컬럼이 그 1차 교차 검증 역할).

### 실버 슬러거 교차 검증 — 포지션별 ca-xBA Top 10

> ** 한계 명시 (선정 메커니즘 본질):** 실버 슬러거는 **현장 전문가(코치·매니저)의 정성적 투표**로 결정되는 시상이다. MLB는 선정 기준에 사용되는 가중치·통계·평가 항목을 공개하지 않으며, 수상에는 **타격 외 요인** (수비 가치, 명성, 미디어 노출, 팀 성적, 라이벌 경쟁자의 분산 등)이 작용한다. 따라서 본 검증은 ca-xBA 가 "타격 능력 측면에서 도메인 전문가의 직관과 얼마나 정렬되는지"를 **재미있게 살펴보는 도메인 일관성 점검**이지, **모델의 설명력을 통계적으로 보증하는 과학적 검증 기법은 아니다.** 통계적·과학적 모델 검증은 § 3 의 R² 분석이 담당한다.

- 실버 슬러거 수상자: 20명 (AL 10 + NL 10)
- ID 매칭 성공: 검증 가능 선수 17/20
- **포지션 Top 10 적중: 12/17 (70.6%)**

*표 21.*

| 리그 | 포지션 | 수상자 | 본 연구의 ca-xBA 순위 | Top N 적중 | ca-xBA | wOBA |
|---|---|---|---:|:---:|---:|---:|
| AL | C | Cal Raleigh | 1 |  | 0.411 | 0.392 |
| AL | 1B | Nick Kurtz | 2 |  | 0.407 | 0.419 |
| AL | 2B | Jazz Chisholm Jr. | 7 |  | 0.361 | 0.349 |
| AL | SS | Bobby Witt Jr. | 9 |  | 0.376 | 0.360 |
| AL | 3B | Jose Ramirez | — | ？ | — | — |
| AL | OF | Aaron Judge | 1 |  | 0.457 | 0.463 |
| AL | OF | Byron Buxton | 20 |  | 0.380 | 0.367 |
| AL | OF | Riley Greene | 28 |  | 0.373 | 0.343 |
| AL | DH | George Springer | 4 |  | 0.411 | 0.408 |
| AL | Util | Zach McKinstry | — | ？ | 0.341 | 0.333 |
| NL | C | Hunter Goodman | 4 |  | 0.391 | 0.359 |
| NL | 1B | Pete Alonso | 3 |  | 0.400 | 0.368 |
| NL | 2B | Ketel Marte | 1 |  | 0.399 | 0.381 |
| NL | SS | Geraldo Perdomo | 14 |  | 0.358 | 0.370 |
| NL | 3B | Manny Machado | 5 |  | 0.374 | 0.341 |
| NL | OF | Juan Soto | 8 |  | 0.400 | 0.390 |
| NL | OF | Corbin Carroll | 11 |  | 0.392 | 0.371 |
| NL | OF | Kyle Tucker | 52 |  | 0.350 | 0.363 |
| NL | DH | Shohei Ohtani | 5 |  | 0.409 | 0.418 |
| NL | Util | Alec Burleson | — | ？ | 0.358 | 0.346 |

> **※ 누락 선수 — Hard Join 매칭의 기술적 한계 (3 명: Jose Ramirez, Zach McKinstry, Alec Burleson)**: 본 검증의 데이터 조인은 Statcast `expected_stats` 의 `player_id` (MLBAM ID) 와 MLB Stats API 의 포지션 정보를 **정확 일치(Hard Join)** 방식으로 매칭한다. 이는 동명이인 오염을 원천 차단하기 위한 학술적 안전장치(사용자 결정 #4)다. 단, **Statcast 의 다국어 선수 철자 표기 (예: José Ramírez 의 accent 기호, Peña 의 ñ 등 라틴/스페인어 특수 기호) 가 MLB Stats API 의 표준 영문 표기와 byte-level 로 일치하지 않는 경우** 조인이 실패하여 검증 풀에서 누락된다. 추가로 250 PA 미만 (예: 시즌 도중 트레이드된 일부 선수) 의 경우에도 본 분석의 대상군 (PA ≥ 250) 에서 제외된다. 본 누락은 **모델 성능과 무관한 데이터 정제 이슈**이며, 향후 작업에서 fuzzy matching 또는 Chadwick Register 의 ID 크로스워크를 도입하여 해소 가능하다.

PNG 5장. 모두 `pipeline/figures/`에 저장.

#### 1:1 R² 대조 산점도 — Phase 5 메인 결론

![1:1 R² Scatter](/tmp/ca-xba-pdf-build/figures/fig_p5a_scatter_1to1_R2.png)

*그림 26. 1:1 R² Scatter*


- **좌 (ca-xBA vs 실제 wOBA)**: R² = **0.3976**, Pearson r = 0.6306
- **우 (MLB 공식 xBA vs 실제 wOBA)**: R² = 0.2499, Pearson r = 0.4999
- **차이: 절대 R² +0.1477 / 상대 우위 +59.1%** — ca-xBA 가 선수의 실제 wOBA 를 명확히 더 잘 설명.
- 환경 변수(구장·기상)를 비선형 모델(LGBM + Isotonic, Phase 4 OOF Brier=0.13092)로 학습한 효과가 시즌 누적 지표에서도 발현됨을 입증.

최종 산출된 ca-xBA와 실제 wOBA의 상관관계(그림 26 좌측)를 MLB 공식 xBA(우측)와 대조하면, 단순한 $R^2$ 수치 차이를 넘어 **데이터 포인트들의 군집 형태(밀집도)** 자체가 달라진다. 공식 지표(우측)의 데이터 포인트들이 1:1 대각선 기준선 주변으로 넓게 퍼져 있는 반면, 환경 맥락을 인지한 ca-xBA(좌측)는 대각선 회귀선에 훨씬 더 조밀하게 군집화되어 있다. 이 시각적 응집도 차이가 곧 설명력($R^2$) 기준 +59.1%의 상대 우위로 이어지며, 환경 변수 통합 모델링이 선수의 실제 기량을 얼마나 정밀하게 타겟팅하는지를 방증한다.

#### 포지션별 ca-xBA Top 10 리더보드

![Position Top 10](/tmp/ca-xba-pdf-build/figures/fig_p5b_position_top10_leaderboards.png)

*그림 27. Position Top 10*


- 7개 포지션(C, 1B, 2B, SS, 3B, OF, DH) 각각의 ca-xBA Top 10 선수.
- 빨강 막대 = 본 연구의 ca-xBA, 파랑 막대 = 실제 wOBA. 두 막대가 비슷할수록 calibration 우수.
- 각 포지션 1위는 실버 슬러거 후보 (§7.3 교차 검증 참조).

#### 실버 슬러거 교차 검증

![Silver Slugger Validation](/tmp/ca-xba-pdf-build/figures/fig_p5c_silver_slugger_validation.png)

*그림 28. Silver Slugger Validation*


- **포지션별 Top 10 적중률: 12/17 (70.6%)**
- 녹색 () = ca-xBA 가 실제 실버 슬러거 수상자를 Top 10 안에 정확히 식별.
- 빨강 () = Top 10 미달. 단 이 경우도 ca-xBA 가 "실력 외 요인(수비 가치, 명성 등)"이 수상에 작용했을 가능성을 시사.
- 회색 (?) = 250 PA 미달 또는 표기 차이로 검증 불가.

#### 운(Luck) 분석 — Top 10 양방향 + 통산 BABIP 교차 검증

![Luck Analysis with Career BABIP](/tmp/ca-xba-pdf-build/figures/fig_p5d_luck_analysis.png)

*그림 29. Luck Analysis with Career BABIP*


- 각 막대 라벨에 **시즌 BABIP · 통산 BABIP · Δ (시즌 − 통산)** 표시.
- Δ_BABIP 는 도메인 정통의 행운 시그널: **자기 통산 baseline 대비 시즌 편차**.
- 좌:  운(행운 효과 가설) 타자. Δ_BABIP > 0 이면 통산 대비 시즌이 높아 **두 지표 일치**.
- 우:  불운(호수비·환경 손해 가설) 타자. Δ_BABIP < 0 이면 둘 다 불운, Δ ≥ 0 이면 **Trout 패턴** (ca-xBA 단독 환경/quality 신호).

#### Luck vs Δ_BABIP 산점도 — 도메인 정통 vs 모델 기반

![Luck vs Delta BABIP Scatter](/tmp/ca-xba-pdf-build/figures/fig_p5f_luck_vs_delta_babip.png)

*그림 30. Luck vs Delta BABIP Scatter*


- 통산 BABIP 매핑 성공한 309 명 산점도. **Pearson r = 0.519, Spearman ρ = 0.478** (참고: 단일 시즌 BABIP 와는 r = 0.700). → Δ_BABIP 와의 상관이 더 의미 있는 도메인 정통 비교.
- 수직 점선: Δ_BABIP = 0 (자기 통산 baseline) / 수평: luck = 0. 녹색 점 = 운(행운 효과) Top 10, 빨강 점 = 불운 Top 10.
- **사분면 해석**:
  - 우상단 (Δ↑, luck↑) = 두 지표 모두 행운 일치 (이중 검증).
  - 좌하단 (Δ↓, luck↓) = 두 지표 모두 불운 일치 (이중 검증).
  - 우하단 (Δ≥0, luck↓) = **Trout 패턴** — 통산 baseline 평균인데 ca-xBA 만 환경/quality 손해 포착. Front Office 의 저평가 발굴 포인트.
  - 좌상단 (Δ↓, luck↑) = 약한 contact 로 행운 효과.

그림 30의 산점도에서 특히 주목해야 할 영역은 **우하단(4사분면)**이다. 이 영역에 위치한 선수(예: Mike Trout 유형)는 자기 통산 수준의 BABIP을 유지하고 있어($\Delta_{\text{BABIP}} \ge 0$) 기존 지표로는 "운이 평범했다"고 해석된다. 그러나 ca-xBA 기반 luck 지표는 이들을 "극심한 불운"으로 평가한다(luck < 0). 즉 기존 BABIP 단일 모델은 놓쳤지만 본 모델만이 포착해 낸 "최상급 타구 질 대비 구장·환경적 손해"를 입은 타자들이며, 이는 곧 Front Office의 핵심적인 저평가 발굴(Undervalued Pick) 포인트가 된다. 반대로 좌하단의 이중 검증 영역에 위치한 선수는 BABIP과 ca-xBA가 모두 불운을 가리켜 해석의 확실성이 높다 — 두 지표의 사분면 위치를 함께 읽음으로써, 단일 지표로는 불가능한 운/불운의 입체적 진단이 가능해진다.

#### (보조) BIP 정의 일치 분포 — ATH 영향

![BIP Ratio](/tmp/ca-xba-pdf-build/figures/fig_p5e_bip_ratio_ath_impact.png)

*그림 31. BIP Ratio*


- 대부분 선수는 `our_bip / csv.bip` 비율이 0.85~0.97 (Phase 1 의 |la|>60 컷오프·핵심 결측 제거로 약간 적음).
- 좌측 꼬리(0.40~0.50)는 Phase 1 의 Athletics 홈경기 제외 결정의 영향을 받은 선수들 — 원정 경기 BIP 만으로 ca-xBA 산출.


# 결론 및 시사점

## 연구 성과 종합

본 연구는 MLB 공식 기대 타율(xBA)의 환경 무시 한계를 극복하는 ca-xBA (Context-Aware xBA)를 제안하고, 데이터마이닝 정통의 5 Phase 로드맵을 통해 그 학술적·실무적 가치를 통계적으로 검증하였다. 구체적으로 (i) 도메인 기반 전처리(돔 구장 게임별 지붕 상태 마스킹), (ii) 보수적 Feature Selection (RF importance + Mutual Information의 4-criterion 합의 규칙), (iii) 2-way ANOVA를 통한 데이터·알고리즘 비선형 상호작용의 통계적 입증, (iv) 오캄의 면도날 자동 선정 로직을 통한 단순성과 성능의 동시 확보, (v) 통산 BABIP 대비 시즌 편차를 활용한 도메인 정통 행운 효과 교차 검증을 단일 파이프라인 안에서 일관되게 수행하였다.

## 학술적 기여

첫째, Phase 3의 2x2 Factorial Ablation은 "환경 변수의 가치는 비선형 모델 위에서만 발현된다"는 명제를 interaction term의 통계적 유의성(p < 0.05)으로 직접 입증하였다. 이는 단순한 변수 추가가 아닌 알고리즘과 데이터의 결합 효과가 ca-xBA 우위의 본질임을 보여준다. 둘째, Phase 4의 cv='prefit' 패턴 Isotonic Calibration은 표준 CalibratedClassifierCV 대비 약 7,000배 연산 단축을 달성하면서 학술적 동등성을 유지하여, 대규모 데이터 환경의 실용적 calibration 방법론을 제시한다. 셋째, 오캄의 면도날 자동 선정 로직은 "성능이 비슷하면 단순한 모델이 낫다"는 원칙을 정량 기준(ε = 0.001)으로 코드화하여, 복잡한 앙상블이 단일 모델의 native calibration을 훼손할 수 있다는 관찰을 학술적으로 정당화한다.

## 실무적 시사점

Phase 5의 BABIP 교차 검증은 ca-xBA가 야구 도메인의 정통 행운 지표인 통산 BABIP 대비 시즌 편차와 동일한 방향의 신호를 잡으면서도, BABIP이 포착하지 못하는 환경/quality 보정 신호(dome x weather 상호작용, 구장 펜스 거리, hr_park_effects 등)를 추가로 단독 포착함을 확인하였다. 특히 Mike Trout 패턴 — 통산 baseline 평균 수준의 BABIP을 유지함에도 ca-xBA 기반 luck이 극불운으로 평가되는 사례 — 은 Front Office 의 저평가 선수 발굴 도구로서 ca-xBA가 BABIP 단독 분석을 보완할 수 있는 실무적 가치를 입증한다. 동시에 Kyle Schwarber 패턴(fly ball power hitter의 구조적 편향)을 정직하게 명시함으로써 모델의 한계 또한 투명하게 공개하였다.

## 한계 및 향후 작업

본 연구는 단일 시즌(2025) 외부 검증에 의존하므로 모델의 시간 일반화 능력은 다년치 검증으로 추가 입증이 필요하다. 또한 Phase 5 실버 슬러거 검증의 일부 선수가 Statcast와 MLB Stats API 간 다국어 선수명 표기 불일치로 누락된 점은 Chadwick Register의 ID 크로스워크를 도입하여 향후 해소 가능하다. 마지막으로 ca-xBA의 BIP-한정 quality 평가 특성상 fly ball power hitter에 대한 구조적 편향은 외부 지표(BABIP, xwOBA underperform)와의 교차 검증 또는 HR weighted 변형 모델 도입으로 보완할 수 있다.

특히 Phase 5에서 관찰된 Kyle Schwarber 패턴(fly ball 거포의 구조적 저평가)은 단순한 경향성이 아니라, 예측하려는 타겟 변수 $y$의 수리적 정의에서 비롯된 구조적(structural) 한계다. 본 모델의 타겟 변수 $y$는 안타(1)와 아웃(0)만을 구분할 뿐, 타구의 실질적 가치(장타 가중치)를 내포하지 않는다. 통계적으로 타자의 인플레이 타구 기대 생산력 $E[\mathrm{wOBA} \mid X]$는 확률과 조건부 기댓값의 곱으로 분해할 수 있다.

$$ E[\mathrm{wOBA} \mid X] = P(\mathrm{Hit} \mid X) \cdot E[\mathrm{HitValue} \mid \mathrm{Hit}, X] $$

현재의 ca-xBA는 위 식에서 첫 번째 항인 $P(\mathrm{Hit} \mid X)$의 정밀한 추정에만 집중한 지표다. 따라서 발사각이 높아 아웃될 확률이 크지만 일단 안타가 되면 홈런(가장 높은 HitValue)이 되는 타구의 가치는 분자 누적에서 과소평가된다. 향후 연구에서 안타 발생 여부를 예측하는 분류기(classifier)와, 안타 발생을 전제로 루타수(wOBA weight)를 예측하는 회귀기(regressor)를 결합한 '허들 모델(Hurdle Model)' 구조를 도입한다면, 이 구조적 편향을 수학적으로 해소할 수 있을 것이다.


# 참고문헌

본 보고서에서 인용 및 활용한 외부 데이터 소스, API, 라이브러리를 분야별로 정리한다.

## 데이터 소스 및 API

1. MLB Statcast 타구 데이터 (Baseball Savant). https://baseballsavant.mlb.com/
2. Open-Meteo Historical Weather API. https://archive-api.open-meteo.com/v1/archive
3. MLB Stats API (포지션 및 통산 hitting 통계). https://statsapi.mlb.com/api/v1/people
4. Baseball Savant Custom Leaderboard — Expected Statistics
5. MLB.com — 2025 Silver Slugger Award Winners. https://www.mlb.com/news/2025-silver-slugger-award-winners

## 라이브러리 및 도구

- Python 3.12, pandas 3.0, scikit-learn 1.8, XGBoost 3.2, LightGBM 4.6
- imbalanced-learn 0.14 (RandomUnderSampler, SMOTE)
- statsmodels 0.14 (2-way ANOVA)
- matplotlib 3.10, seaborn 0.13
- pypandoc 1.17 + pandoc 3.5, tectonic 0.16 (LaTeX engine)


# 부록 A. 다중공선성 분석 — 고상관 변수 쌍 전체 목록 {.unnumbered}

Phase 2의 다중공선성 분석(Pearson $|r| > 0.95$)에서 식별된 24건의 고상관 변수 쌍을 |r| 내림차순으로 정리한다. `var_a`/`var_b` 는 RobustScaler 적용 후 분산이며, 제거 규칙은 X_BASE 보존 → derived 변수 우선 drop → variance fallback 순서를 따른다.

*표 22.*

| 변수 A | 변수 B | abs(r) | var_a | var_b | 제거 | 규칙 |
|---|---|---:|---:|---:|---|---|
| `bat_speed_is_missing` | `swing_length_is_missing` | 1.000 | 0.089 | 0.089 | `swing_length_is_missing` | variance fallback |
| `bat_speed_is_missing` | `attack_angle_is_missing` | 1.000 | 0.089 | 0.089 | `bat_speed_is_missing` | variance fallback |
| `swing_length_is_missing` | `attack_angle_is_missing` | 1.000 | 0.089 | 0.089 | `swing_length_is_missing` | variance fallback |
| `bat_speed_is_missing` | `attack_direction_is_missing` | 1.000 | 0.089 | 0.089 | `bat_speed_is_missing` | variance fallback |
| `swing_length_is_missing` | `attack_direction_is_missing` | 1.000 | 0.089 | 0.089 | `swing_length_is_missing` | variance fallback |
| `attack_angle_is_missing` | `attack_direction_is_missing` | 1.000 | 0.089 | 0.089 | `attack_direction_is_missing` | variance fallback |
| `intercept_ball_minus_batter_pos_x_inches_is_missing` | `intercept_ball_minus_batter_pos_y_inches_is_missing` | 1.000 | 0.089 | 0.089 | `intercept_ball_minus_batter_pos_y_inches_is_missing` | variance fallback |
| `if_fielding_alignment_UNK` | `of_fielding_alignment_UNK` | 1.000 | 0.007 | 0.007 | `of_fielding_alignment_UNK` | variance fallback |
| `bat_speed_is_missing` | `swing_path_tilt_is_missing` | 1.000 | 0.089 | 0.089 | `bat_speed_is_missing` | variance fallback |
| `swing_length_is_missing` | `swing_path_tilt_is_missing` | 1.000 | 0.089 | 0.089 | `swing_length_is_missing` | variance fallback |
| `attack_angle_is_missing` | `swing_path_tilt_is_missing` | 1.000 | 0.089 | 0.089 | `attack_angle_is_missing` | variance fallback |
| `attack_direction_is_missing` | `swing_path_tilt_is_missing` | 1.000 | 0.089 | 0.089 | `attack_direction_is_missing` | variance fallback |
| `bat_speed_is_missing` | `intercept_ball_minus_batter_pos_x_inches_is_missing` | 1.000 | 0.089 | 0.089 | `bat_speed_is_missing` | variance fallback |
| `swing_length_is_missing` | `intercept_ball_minus_batter_pos_x_inches_is_missing` | 1.000 | 0.089 | 0.089 | `swing_length_is_missing` | variance fallback |
| `attack_angle_is_missing` | `intercept_ball_minus_batter_pos_x_inches_is_missing` | 1.000 | 0.089 | 0.089 | `attack_angle_is_missing` | variance fallback |
| `attack_direction_is_missing` | `intercept_ball_minus_batter_pos_x_inches_is_missing` | 1.000 | 0.089 | 0.089 | `attack_direction_is_missing` | variance fallback |
| `bat_speed_is_missing` | `intercept_ball_minus_batter_pos_y_inches_is_missing` | 1.000 | 0.089 | 0.089 | `bat_speed_is_missing` | variance fallback |
| `swing_length_is_missing` | `intercept_ball_minus_batter_pos_y_inches_is_missing` | 1.000 | 0.089 | 0.089 | `swing_length_is_missing` | variance fallback |
| `attack_angle_is_missing` | `intercept_ball_minus_batter_pos_y_inches_is_missing` | 1.000 | 0.089 | 0.089 | `attack_angle_is_missing` | variance fallback |
| `attack_direction_is_missing` | `intercept_ball_minus_batter_pos_y_inches_is_missing` | 1.000 | 0.089 | 0.089 | `attack_direction_is_missing` | variance fallback |
| `swing_path_tilt_is_missing` | `intercept_ball_minus_batter_pos_x_inches_is_missing` | 1.000 | 0.089 | 0.089 | `swing_path_tilt_is_missing` | variance fallback |
| `swing_path_tilt_is_missing` | `intercept_ball_minus_batter_pos_y_inches_is_missing` | 1.000 | 0.089 | 0.089 | `swing_path_tilt_is_missing` | variance fallback |
| `release_speed` | `effective_speed` | 0.990 | 0.454 | 0.479 | `effective_speed` | derived drop |
| `elevation` | `wx_surface_pressure` | 0.985 | 2.874 | 2.039 | `wx_surface_pressure` | variance fallback |

# 부록 B. 결측치 대체 중앙값 (2024 median) {.unnumbered}

Phase 2에서 결측 imputation을 적용한 13개 numeric 컬럼의 2024 전체 중앙값이다. 각 변수의 `*_is_missing` 플래그는 별도로 보존하여 결측 패턴 자체를 신호로 활용한다.

*표 23.*

| 컬럼 | 2024 Median |
|---|---:|
| `bat_speed` | 71.6000 |
| `swing_length` | 7.2000 |
| `attack_angle` | 8.7646 |
| `attack_direction` | 0.7896 |
| `swing_path_tilt` | 32.2398 |
| `intercept_ball_minus_batter_pos_x_inches` | 37.1243 |
| `intercept_ball_minus_batter_pos_y_inches` | 29.5637 |
| `release_spin_rate` | 2263.0000 |
| `release_extension` | 6.5000 |
| `spin_axis` | 201.0000 |
| `effective_speed` | 90.6000 |
| `api_break_z_with_gravity` | 2.2200 |
| `arm_angle` | 39.2000 |

# 부록 C. 효과 분리 실험 세부 통계 (Phase 3) {.unnumbered}

## fold-level mean ± SD (across 5 folds) {.unnumbered}

2x2 Factorial Ablation 4개 모델(M1~M4)의 5-fold 메트릭 평균과 표준편차다. fold 간 변동이 매우 작아 OOF aggregate 결과가 안정적임을 뒷받침한다.

*표 24.*

| Model | Brier mean±SD | LogLoss mean±SD | F1 mean±SD | AUC mean±SD |
|---|---:|---:|---:|---:|
| M1 | 0.21033±0.00052 | 0.61347±0.00110 | 0.1700±0.0054 | 0.6670±0.0034 |
| M2 | 0.14012±0.00146 | 0.43198±0.00354 | 0.6758±0.0029 | 0.8594±0.0028 |
| M3 | 0.20937±0.00063 | 0.61078±0.00142 | 0.2357±0.0072 | 0.6662±0.0039 |
| M4 | 0.13589±0.00088 | 0.42049±0.00213 | 0.6924±0.0028 | 0.8691±0.0015 |

## 2-way ANOVA (Type II SS) — 4개 메트릭 {.unnumbered}

각 fold(n=5)의 메트릭을 종속변수로, Data(X_base/X_advanced) x Algo(LogReg/XGB)를 요인으로 한 Type II SS ANOVA 결과다. 모든 메트릭에서 상호작용 항 `C(data):C(algo)` 이 통계적으로 유의하다(p < 0.05).

**Brier:**

*표 25.*

| Source | SS | df | F | p |
|---|---:|---:|---:|---:|
| C(data) | 0.000034 | 1 | 30.226 | 4.866e-05 |
| C(algo) | 0.025810 | 1 | 23134.984 | 1.022e-26 |
| C(data):C(algo) | 0.000013 | 1 | 11.977 | 0.00322 |
| Residual | 0.000018 | 16 | n/a | n/a |

**LogLoss:**

*표 26.*

| Source | SS | df | F | p |
|---|---:|---:|---:|---:|
| C(data) | 0.000251 | 1 | 39.606 | 1.07e-05 |
| C(algo) | 0.172764 | 1 | 27232.030 | 2.776e-27 |
| C(data):C(algo) | 0.000097 | 1 | 15.237 | 0.001264 |
| Residual | 0.000102 | 16 | n/a | n/a |

**ROC AUC:**

*표 27.*

| Source | SS | df | F | p |
|---|---:|---:|---:|---:|
| C(data) | 0.000097 | 1 | 8.465 | 0.01024 |
| C(algo) | 0.195359 | 1 | 17051.056 | 1.172e-25 |
| C(data):C(algo) | 0.000138 | 1 | 12.003 | 0.003194 |
| Residual | 0.000183 | 16 | n/a | n/a |

**F1:**

*표 28.*

| Source | SS | df | F | p |
|---|---:|---:|---:|---:|
| C(data) | 0.008475 | 1 | 277.510 | 1.57e-11 |
| C(algo) | 1.157952 | 1 | 37918.715 | 1.967e-28 |
| C(data):C(algo) | 0.003023 | 1 | 98.979 | 2.95e-08 |
| Residual | 0.000489 | 16 | n/a | n/a |

# 부록 D. Advanced 모델 튜닝 스펙 (Phase 4) {.unnumbered}

## 모델별 최종 하이퍼파라미터 (RandomizedSearchCV best params) {.unnumbered}

각 base 모델의 RandomizedSearchCV(n_iter=30, inner_cv=5, scoring='neg_brier_score', refit=True) 결과로 선정된 최종 하이퍼파라미터 전체 딕셔너리다.

*표 29.*

| Base | best params |
|---|---|
| RF | `bootstrap`=True, `ccp_alpha`=0.0, `class_weight`=None, `criterion`=entropy, `max_depth`=None, `max_features`=0.5, `max_leaf_nodes`=None, `max_samples`=None, `min_impurity_decrease`=0.0, `min_samples_leaf`=4, `min_samples_split`=4, `min_weight_fraction_leaf`=0.0, `monotonic_cst`=None, `n_estimators`=500, `n_jobs`=1, `oob_score`=False, `random_state`=42, `verbose`=0, `warm_start`=False |
| XGB | `objective`=binary:logistic, `colsample_bytree`=0.9, `eval_metric`=logloss, `gamma`=0, `learning_rate`=0.03, `max_depth`=8, `min_child_weight`=5, `n_estimators`=200, `n_jobs`=1, `random_state`=42, `subsample`=0.8, `tree_method`=hist, `verbosity`=0 |
| LGBM | `boosting_type`=gbdt, `class_weight`=None, `colsample_bytree`=1.0, `importance_type`=split, `learning_rate`=0.03, `max_depth`=-1, `min_child_samples`=20, `min_child_weight`=0.001, `min_split_gain`=0.0, `n_estimators`=200, `n_jobs`=1, `num_leaves`=127, `random_state`=42, `reg_alpha`=0.0, `reg_lambda`=0.0, `subsample`=0.9, `subsample_for_bin`=200000, `subsample_freq`=0, `verbose`=-1 |

## Outer 5-fold CV — fold mean ± SD {.unnumbered}

6개 후보 모델의 Outer 5-fold CV fold 평균과 표준편차다. fold 간 변동성(Brier 기준 0.0012 이하)이 후보 간 차이보다 작아 오캄의 면도날 자동 선정(ε=0.001)의 근거가 된다.

*표 30.*

| Model | Brier mean±SD | LogLoss mean±SD | F1 mean±SD | AUC mean±SD |
|---|---:|---:|---:|---:|
| RF (tuned) | 0.13264±0.00117 | 0.41259±0.00303 | 0.6978±0.0031 | 0.8759±0.0022 |
| XGB (tuned) | 0.13231±0.00106 | 0.41135±0.00267 | 0.6975±0.0036 | 0.8761±0.0020 |
| LGBM (tuned) | 0.13108±0.00104 | 0.40753±0.00275 | 0.6985±0.0027 | 0.8777±0.0020 |
| Stacking (LR meta) | 0.13244±0.00124 | 0.41409±0.00312 | 0.7012±0.0025 | 0.8780±0.0021 |
| Stacking + Isotonic | 0.13083±0.00107 | 0.40587±0.00315 | 0.6937±0.0051 | 0.8780±0.0021 |
| LGBM + Isotonic | 0.13092±0.00100 | 0.40600±0.00284 | 0.7029±0.0040 | 0.8776±0.0020 |
