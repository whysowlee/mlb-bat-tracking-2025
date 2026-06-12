# Context-Aware xBA (ca-xBA) 리포트 작성 상세 가이드 (`report_detail.md`)

이 문서는 5-Phase 로드맵의 각 단계별 마크다운 리포트 생성 시, 최종 Word 보고서 작성을 위해 반드시 포함되어야 할 상세 정보와 시각화 요소를 정의합니다. 참고 프로젝트의 목차 구성을 바탕으로 데이터 마이닝의 정석적인 흐름을 따릅니다.

---

## 📄 Phase 1 Report: 데이터 통합 및 연도별 분리
**[참고 보고서 매핑 목차]** 1. 서론 (1.1, 1.2), 2. 데이터 (2.1, 2.2, 2.4)[cite: 3]

이 리포트는 최종 보고서의 '서론' 및 '데이터 전처리' 파트를 구성하기 위한 원천 자료입니다.

*   **1. 서론 (Introduction) 요약:**
    *   연구의 필요성: 기존 xBA의 한계(물리 변수 의존)와 상황 인지형 모델(ca-xBA)의 필요성 서술[cite: 3].
    *   연구 방법론: 타구 데이터와 환경 데이터 결합, 연도별 분리(Temporal Split) 검증 방식 명시[cite: 3].
*   **2. 데이터 셋 설명 (Data Description):**
    *   출처(Statcast, 기상청, 구장 스펙 API 등) 및 총 데이터 건수/변수 개수[cite: 3].
    *   Target 변수(`is_hit`) 정의 및 데이터 불균형 비율 초기 확인[cite: 3].
*   **3. 도메인 기반 결측치 및 이상치 처리 (Handling Missing/Outlier Data):**
    *   타구 데이터 특성을 반영한 필터링 기준 (예: 파울 팝아웃 ±60도 컷오프 등) 및 제거된 행(Row)의 수.
    *   배트 트래킹 변수 결측치 처리 논리.
*   **4. 데이터 분할 (Temporal Split):**
    *   `2024_Data`(훈련/테스트용)와 `2025_Data`(최종 정답지용)로 분리된 후의 각 데이터셋 shape 및 타겟 변수 분포 수치.

---

## 📄 Phase 2 Report: 데이터 탐색, 스케일링 및 샘플링
**[참고 보고서 매핑 목차]** 2. 데이터 (2.2, 2.3, 2.5, 2.6, 2.7)[cite: 3]

이 리포트는 데이터를 시각적으로 이해하고, 모델 학습을 위한 최적의 상태를 증명하는 핵심 파트입니다.

*   **1. 기초 통계량 및 데이터 시각화 (EDA):**
    *   주요 환경 변수(온도, 고도 등) 및 물리 변수의 기초 통계량(평균, min, max, 분위수) 표[cite: 3].
    *   *Word 변환 시 삽입할 시각화 목록 명시:* 연속형 변수 분포(KDE/Boxplot), 범주형 변수 분포(Bar plot), Target 변수와의 이변량 분석 결과 요약[cite: 3].
*   **2. 스케일링 및 상관관계 (Scaling & Correlation):**
    *   Robust Scaler 선택 사유 서술[cite: 3].
    *   다중공선성(Correlation > 0.8)이 확인되어 제거된 변수 쌍 목록[cite: 3].
*   **3. 특성 선택 (Feature Selection):**
    *   Tree, Random Forest, Mutual Information을 활용한 변수 중요도(Feature Importance) Top 20 리스트 및 도메인 관점의 해석 (예: "lead_time 대신 launch_speed와 구장 고도가 높은 중요도를 가짐")[cite: 3].
*   **4. 샘플링 기법 비교 (Sampling Strategy):**
    *   원본, Under Sampling, SMOTE 적용 시의 Target 클래스 분포 비율 변화[cite: 3].
    *   기본 알고리즘(예: XGBoost) 적용 시 각 샘플링 기법별 혼동행렬(Confusion Matrix) 주요 수치 및 성능 지표(Precision, Recall, F1-Score) 표[cite: 3].
    *   최종 샘플링 기법 선정 사유 (95% 신뢰구간 또는 F1-Score 기준)[cite: 3].

---

## 📄 Phase 3 Report: 효과 분리 실험 (Ablation Study)
**[참고 보고서 매핑 목차]** 3. 분류 모델 생성 (3.3 로지스틱, Base 모델 비교)[cite: 3]

모델의 알고리즘과 데이터 추가 효과를 통제군 단위로 분리하여 증명하는 단계입니다.

*   **1. 실험군 통제 설계 명시:**
    *   모델 1 (`X_base` + 선형 알고리즘), 모델 2 (`X_base` + 트리 앙상블), 모델 3 (`X_advanced` + 선형 알고리즘) 구조 설명.
*   **2. 모델별 평가 지표 비교표:**
    *   각 모델의 Accuracy, Precision, Recall, F1-Score, ROC AUC 수치 정리[cite: 3].
*   **3. 비선형 상호작용 심층 분석 (가장 중요):**
    *   데이터 효과 vs 알고리즘 효과의 AUC 증분 비교.
    *   "선형 모델이 환경 변수를 추가해도 성능이 오르지 않는 이유"를 발사각/속도 조건의 비선형적 상호작용 관점에서 서술. (Phase 4의 당위성 부여).

---

## 📄 Phase 4 Report: 하이퍼파라미터 튜닝 및 Meta Model 구축
**[참고 보고서 매핑 목차]** 3. 분류 모델 생성 (3.6~3.10 단일/앙상블/Stacking 비교)[cite: 3]

본격적인 머신러닝 성능 극대화 단계입니다.

*   **1. 단일 트리 앙상블 모델 튜닝 결과:**
    *   RF, XGBoost, LightGBM 모델의 RandomizedSearchCV 탐색 공간 및 최종 선정된 하이퍼파라미터 조합 표[cite: 3].
    *   각 단일 모델의 Test 셋 대상 성능 지표 및 혼동행렬 주요 수치[cite: 3].
*   **2. Stacking Meta Model 구축:**
    *   기반 모델(Base models)과 최종 메타 모델(Meta Learner)의 구성 아키텍처 설명[cite: 3].
*   **3. 모델 종합 비교 (Model Comparison):**
    *   성능 상위 단일 모델들과 Stacking 모델의 F1-Score 및 ROC AUC 비교표[cite: 3].
    *   통계적 우위 판단 결과 명시[cite: 3].
    *   *Word 변환 시 삽입할 시각화 목록 명시:* 4개 모델의 ROC Curve 겹침 플롯 기획[cite: 3].

---

## 📄 Phase 5 Report: 세이버메트릭스 기반 최종 가치 검증
**[참고 보고서 매핑 목차]** 5. 최종 모델 선정 (5.3 비교), 6. 연구의 시사점 및 결론[cite: 3]

단순 분류 성능을 넘어, 비즈니스(야구단) 관점에서의 지표 가치를 증명하는 결론부입니다.

*   **1. 최종 분류 성능 점검 (Event-Level):**
    *   최종 선정된 Meta Model(ca-xBA)의 최적 임계값(Threshold) 설정 결과[cite: 3].
    *   ca-xBA와 기존 xBA의 타구 분류 기준 ROC AUC 및 Brier Score 비교 수치.
*   **2. 세이버메트릭스 검증 (Player-Level Year-to-Year Correlation):**
    *   API에서 로드한 `2025_wOBAcon` 정답 데이터의 출처 및 집계 기준.
    *   논리 전제 명시: "운(Noise)은 독립적이며, 내년 성적을 예측하는 핵심은 환경(Signal)의 추출임."
    *   상관계수 산출 결과 표 ($r_1$: 기존 xBA vs wOBAcon / $r_2$: ca-xBA vs wOBAcon).
    *   *Word 변환 시 삽입할 시각화 목록 명시:* $x$축(예측 타율 지표), $y$축(2025 wOBAcon) 산점도(Scatter plot) 및 회귀선 비교 플롯 기획.
*   **3. 시사점 및 결론 (Conclusion):**
    *   본 프로젝트의 최종 결론: "환경 변수의 통합이 선수의 진짜 타격 생산력(True Talent) 예측 정확도를 어떻게 향상시켰는가."
    *   비즈니스 활용 방안: 구장 특성에 맞춘 FA 선수 영입 및 데이터 기반 트레이드 가치 산정 등[cite: 3].
    *   연구 한계 및 향후 과제 서술[cite: 3].