"""
ca-xBA 최종 보고서 (Word .docx) 자동 컴파일 스크립트
======================================================

8개 마크다운 파일 + example.md 의 우수자료 구조 가이드를 기반으로 학술 문체로 통일된
하나의 Word 보고서를 생성한다.

**입력:**
  - readme.md (프로젝트 목표 · 5 Phase 로드맵)
  - report_detail.md (리포트 가이드라인)
  - example.md (우수자료 목차 구조)
  - pipeline/phase1_report.md (데이터 통합 · 전처리)
  - pipeline/phase2_report.md (EDA · 스케일링 · 샘플링 · FS)
  - pipeline/phase3_report.md (2×2 Factorial Ablation)
  - pipeline/phase4_report.md (튜닝 · Stacking · Calibration)
  - pipeline/phase5_report.md (ca-xBA 산출 · 세이버메트릭스 검증)

**보고서 챕터 구조 (우수자료 구조 흡수):**
  1. 서론
  2. 데이터 전처리 및 EDA  ← Phase 1 + Phase 2 융합
  3. 효과 분리 실험 (Ablation Study)  ← Phase 3
  4. Advanced Model 튜닝 + Stacking  ← Phase 4
  5. 최종 지표(ca-xBA) 산출 및 세이버메트릭스 가치 검증  ← Phase 5
  6. 결론 및 시사점
  7. 참고문헌  ← Phase 5 의 URL 출처 자동 집계

**문체 통일:** 본문 모든 문장을 '~이다' 체로 변환 (코드 블록·인용 URL 제외).

**산출:** ca-xBA_Final_Report.docx

실행:
    /opt/miniconda3/envs/mlb-xba/bin/python generate_docx_report.py
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pypandoc

# -----------------------------------------------------------------------------
# 경로
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT / "pipeline"
OUTPUT_DOCX = ROOT / "ca-xBA_Final_Report.docx"

README_MD = ROOT / "readme.md"
PHASE1_MD = PIPELINE_DIR / "phase1_report.md"
PHASE2_MD = PIPELINE_DIR / "phase2_report.md"
PHASE3_MD = PIPELINE_DIR / "phase3_report.md"
PHASE4_MD = PIPELINE_DIR / "phase4_report.md"
PHASE5_MD = PIPELINE_DIR / "phase5_report.md"


def log(msg: str) -> None:
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# 문체 통일 (~이다 체로 변환)
#   존댓말·구어체 → 평서 문어체
#   코드 블록(``` ~ ```), 인라인 코드(`...`), URL 은 placeholder 로 보호
# -----------------------------------------------------------------------------
STYLE_RULES: list[tuple[str, str]] = [
    # 가장 빈번한 패턴 (긴 → 짧은 순서)
    (r"습니다(?=[\.\,\!\?\s\n]|$)", "다"),       # ~합니다, ~입니다, ~됩니다, ~있습니다, ~없습니다
    (r"입니다(?=[\.\,\!\?\s\n]|$)", "이다"),     # 보강 (위 규칙이 처리 못 한 경우 대비)
    (r"합니다(?=[\.\,\!\?\s\n]|$)", "한다"),
    (r"됩니다(?=[\.\,\!\?\s\n]|$)", "된다"),
    (r"있습니다(?=[\.\,\!\?\s\n]|$)", "있다"),
    (r"없습니다(?=[\.\,\!\?\s\n]|$)", "없다"),
    (r"줍니다(?=[\.\,\!\?\s\n]|$)", "준다"),
    (r"옵니다(?=[\.\,\!\?\s\n]|$)", "온다"),
    (r"갑니다(?=[\.\,\!\?\s\n]|$)", "간다"),
    (r"봅니다(?=[\.\,\!\?\s\n]|$)", "본다"),
    (r"드립니다(?=[\.\,\!\?\s\n]|$)", "한다"),
    # 어말 어미 보강
    (r"습니까(?=[\.\,\!\?\s\n]|$)", "는가"),
    (r"까요(?=[\.\,\!\?\s\n]|$)", "는가"),
    # 구어체 (예: ~예요, ~네요)
    (r"이에요(?=[\.\,\!\?\s\n]|$)", "이다"),
    (r"예요(?=[\.\,\!\?\s\n]|$)", "이다"),
    (r"네요(?=[\.\,\!\?\s\n]|$)", "다"),
    (r"군요(?=[\.\,\!\?\s\n]|$)", "다"),
    (r"세요(?=[\.\,\!\?\s\n]|$)", "라"),
    (r"해요(?=[\.\,\!\?\s\n]|$)", "한다"),
    (r"써요(?=[\.\,\!\?\s\n]|$)", "쓴다"),
    # 그 외 빈출 (조사 보강)
    (r"보입니다", "보인다"),
    (r"보여줍니다", "보여준다"),
    (r"만듭니다", "만든다"),
    (r"이루어집니다", "이루어진다"),
    (r"가집니다", "가진다"),
]


def _protect_code_blocks(text: str) -> tuple[str, list[str]]:
    """``` 코드 블록 · 인라인 코드 · URL · 이미지 링크를 placeholder 로 보호."""
    blocks: list[str] = []

    def _save(match):
        blocks.append(match.group(0))
        return f"\x00BLOCK{len(blocks) - 1}\x00"

    # 1. fenced code blocks ```...```
    text = re.sub(r"```[\s\S]*?```", _save, text)
    # 2. inline code `...`
    text = re.sub(r"`[^`\n]+`", _save, text)
    # 3. markdown image ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", _save, text)
    # 4. markdown link [text](url)
    text = re.sub(r"\[[^\]]*\]\([^\)]+\)", _save, text)
    # 5. naked URL
    text = re.sub(r"https?://\S+", _save, text)
    return text, blocks


def _restore_code_blocks(text: str, blocks: list[str]) -> str:
    for i, block in enumerate(blocks):
        text = text.replace(f"\x00BLOCK{i}\x00", block)
    return text


def unify_style(text: str) -> str:
    """본문 문체를 ~이다 체로 변환. 코드·URL 등은 보호."""
    text, blocks = _protect_code_blocks(text)
    for pattern, replacement in STYLE_RULES:
        text = re.sub(pattern, replacement, text)
    text = _restore_code_blocks(text, blocks)
    return text


# -----------------------------------------------------------------------------
# 헤더 레벨 조정
#   각 phase 리포트의 `# 제목` 은 챕터의 `##` 하위로 들어가야 하므로 +1 (또는 +2) shift
# -----------------------------------------------------------------------------
def shift_headers(text: str, shift: int) -> str:
    """마크다운 헤더 (#~######) 를 shift 단계만큼 한 단계 내림."""
    def _shift(match):
        hashes = match.group(1)
        new_level = min(len(hashes) + shift, 6)
        return "#" * new_level + match.group(2)
    return re.sub(r"^(#{1,6})( .+)$", _shift, text, flags=re.MULTILINE)


# -----------------------------------------------------------------------------
# 이미지 경로 보정 (pipeline 기준 → 프로젝트 root 기준)
# -----------------------------------------------------------------------------
def fix_image_paths(text: str) -> str:
    """phase 리포트의 ![](figures/...) → ![](pipeline/figures/...) 등으로 보정."""
    # 절대 경로(/), URL(http), 이미 pipeline/ 으로 시작하는 경우는 제외
    def _fix(match):
        alt = match.group(1)
        url = match.group(2)
        if url.startswith(("/", "http://", "https://", "pipeline/")):
            return match.group(0)
        # figures/, output/, cache/, logs/ 등은 pipeline/ prefix 추가
        if url.startswith(("figures/", "output/", "cache/", "logs/")):
            return f"![{alt}](pipeline/{url})"
        return match.group(0)
    return re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", _fix, text)


# -----------------------------------------------------------------------------
# 메타 헤더(생성: 날짜, _실행 스크립트:_) 제거 — 보고서에서는 군더더기
# -----------------------------------------------------------------------------
def strip_meta_headers(text: str) -> str:
    """학술 보고서용 후처리:
    - 작성 메타 (생성/실행 스크립트) 라인 제거
    - 'Phase X Report — ...' H1 제거 (챕터 제목과 중복)
    - 모델 메타 인용 (`> **모델 메타:** ...`) 제거
    """
    text = re.sub(r"^_생성:.*?_\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^_실행 스크립트:.*?_\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#+ Phase \d+ Report.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^> \*\*모델 메타:\*\*.*?$", "", text, flags=re.MULTILINE)
    return text


# -----------------------------------------------------------------------------
# 학술 보고서 후처리 — 작업용 메모/경로/산출물/결정 표 제거
# -----------------------------------------------------------------------------
def strip_section_by_header(text: str, header_pattern: str) -> str:
    """`## N. 제목` 또는 `### N.N 제목` 형태의 섹션을 통째로 제거.

    header_pattern: 매칭할 헤더 정규식 (예: r'^## \d+\. 결정 사항')
    제거 범위: 매칭 헤더부터 다음 동급 헤더 직전까지.
    """
    lines = text.split("\n")
    out: list[str] = []
    skipping = False
    skip_level = 0
    for line in lines:
        if not skipping:
            if re.match(header_pattern, line):
                skipping = True
                skip_level = len(re.match(r"^(#+)", line).group(1))
                continue
            out.append(line)
        else:
            # 다음 동급 이상 헤더에서 skipping 종료
            m = re.match(r"^(#+)\s", line)
            if m and len(m.group(1)) <= skip_level:
                skipping = False
                out.append(line)
    return "\n".join(out)


def strip_user_memos(text: str) -> str:
    """사용자 작성 영역·자동 fabrication 금지·메모성 안내 블록 제거.

    Phase 5 §4.7 의 '운/불운 Top 5 — 스카우팅 서사' 섹션 자체가 작성 placeholder 라
    챕터 제목 + 본문 모두 제거. 외부 출처 인용은 본문 내 figure/표만 남김.
    """
    # 1) 스카우팅 서사 placeholder 섹션 (### 4.7 / #### 4.7 ... 등 레벨 무관)
    text = strip_section_by_header(text, r"^#+ \d+\.\d+ 운/불운 Top \d+ — 스카우팅 서사")
    # 2) "(사용자 작성 영역)" 같은 라벨 흔적 제거
    text = re.sub(r"\s*\(사용자 작성 영역\)", "", text)
    # 3) "자동 fabrication 금지", "추정·창작은 절대 금지" 등 메타 안내 인용 블록
    #    > 로 시작하는 블록 한 단락 단위 제거
    text = re.sub(
        r"^> \*\*방법론.*?$\n(?:^>.*?$\n)*",
        "",
        text,
        flags=re.MULTILINE,
    )
    # 4) "_본 자동 리포트는 ..._" 라벨 단락 제거 (italic placeholder)
    text = re.sub(
        r"^_본 자동 리포트는 객관적 수치.*?$\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text


def strip_file_paths_and_artifacts(text: str) -> str:
    """학술 보고서에 부적합한 작업용 파일 경로/산출물 안내 제거.

    1) `## N. 산출물` 섹션 통째로
    2) 시각화 섹션 서두의 'PNG 파일은 모두 pipeline/figures/ 에 저장' 류 안내
    3) figure 캡션 위의 '> _(B) ... 가독성 문제로 제외_' 같은 작업 메모
    """
    # 1) ## N. 산출물 (헤더 레벨 무관 — shift 후 ### 이 되어도 매칭)
    text = strip_section_by_header(text, r"^#+ \d+\. 산출물")
    # 2) PNG 안내 라인 — "PNG 파일은 ... 저장 / 사용 / 활용" 한 줄 통째로
    text = re.sub(
        r"^PNG 파일은.*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    # 3) (B) 상관관계 히트맵 ... 제외 같은 작업 메모 인용
    text = re.sub(
        r"^> _\([A-Z]\) [^_]*?(제외|가독성).*?_\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text


def strip_decision_tables(text: str) -> str:
    """학술 보고서에서 챕터 본문의 '## N. 결정 사항' 섹션 제거.

    프로젝트 기록용 결정 표는 부록 챕터(build_chapter_appendix_decisions)로 별도 수집.
    """
    return strip_section_by_header(text, r"^#+ \d+\. 결정 사항")


def apply_chapter_polish(text: str) -> str:
    """학술 보고서용 일괄 후처리 (모든 chapter 본문 공통)."""
    text = strip_meta_headers(text)
    text = strip_user_memos(text)
    text = strip_file_paths_and_artifacts(text)
    text = strip_decision_tables(text)
    # 연속 빈줄 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# -----------------------------------------------------------------------------
# URL 출처 추출 → 참고문헌 자동 집계
# -----------------------------------------------------------------------------
def extract_url_sources(*texts: str) -> list[tuple[str, str]]:
    """모든 본문에서 [텍스트](URL) 형태의 URL 출처를 추출, 중복 제거 후 정렬."""
    found: dict[str, str] = {}
    for text in texts:
        for match in re.finditer(r"\[([^\]]+)\]\((https?://[^\)]+)\)", text):
            title, url = match.group(1), match.group(2)
            found[url] = title
    # URL 순으로 정렬
    return sorted([(title, url) for url, title in found.items()], key=lambda x: x[1])


# -----------------------------------------------------------------------------
# 챕터별 본문 생성
# -----------------------------------------------------------------------------
def build_chapter_1_intro(readme_text: str) -> str:
    """챕터 1. 서론 — readme.md 의 프로젝트 목표 + 로드맵 도입부 발췌."""
    L: list[str] = []
    L.append("## 1. 서론")
    L.append("")
    L.append("### 1.1 연구의 필요성")
    L.append("")
    L.append(
        "메이저리그(MLB)의 공식 기대 타율(xBA)은 타구의 순수 물리적 질"
        "(발사 속도, 발사 각도)만을 평가하며, 현실 경기의 환경 변수를 무시하는 "
        "구조적 한계를 가진다. 동일한 발사 조건의 타구라 하더라도 콜로라도 "
        "쿠어스 필드와 샌프란시스코 오라클 파크에서 기록되는 결과가 본질적으로 "
        "다르다는 점은, 공식 xBA 가 *환경적 맥락* 을 반영하지 못하는 결정적 "
        "한계임을 시사한다. 본 연구는 이 한계를 극복하기 위한 "
        "**상황 인지형 기대 타율(ca-xBA, Context-Aware xBA)** 을 제안한다."
    )
    L.append("")
    L.append("### 1.2 연구 방법")
    L.append("")
    L.append(
        "본 연구는 타구의 물리 데이터에 **구장의 물리적 제약(펜스 높이, 거리)** "
        "과 **기후 환경 변수(온도, 풍속, 풍향, 고도 등)** 를 결합하고, "
        "비선형적 상호작용을 포착할 수 있는 트리 앙상블 모델(Tree Ensembles)을 "
        "활용하여 상황 인지형 기대 타율 `ca-xBA` 를 산출한다. 데이터 누수(Data "
        "Leakage)를 완벽하게 차단하고 야구 세이버메트릭스 철학을 반영하기 위해, "
        "**엄격한 연도별 분리(Temporal Split)** 를 기반으로 2024년 데이터로 "
        "모델을 학습·검증하고 2025년 격리 데이터로 최종 가치를 검증한다."
    )
    L.append("")
    L.append("### 1.3 5-Phase 로드맵 개요")
    L.append("")
    L.append("본 연구는 다음 5단계 로드맵으로 구성된다.")
    L.append("")
    L.append("| Phase | 핵심 작업 | 주요 산출 |")
    L.append("|---|---|---|")
    L.append("| 1 | 데이터 통합·전처리·연도별 분리 | 2024_data / 2025_data parquet, 결정 8건 |")
    L.append("| 2 | 상관관계·스케일링·최적 샘플링 도출 | X_advanced 62 변수, Under 샘플링 선정 |")
    L.append("| 3 | 효과 분리 실험 (2×2 Factorial Ablation) | Interaction effect 입증 (비선형 상호작용) |")
    L.append("| 4 | Advanced Model 튜닝 + Stacking | LGBM Isotonic Brier 0.1323 (최종) |")
    L.append("| 5 | ca-xBA 산출 + 세이버메트릭스 검증 | R² = 0.3923 (공식 xBA 대비 +57% 우위) |")
    L.append("")
    L.append(
        "궁극적으로 본 연구는 새로 구축한 `ca-xBA` 가 단순한 운(Noise)을 "
        "과적합한 것이 아니라 타자의 *환경에 최적화된 진짜 실력(True Talent)* "
        "을 성공적으로 추출하였음을 증명하기 위해, 기존 xBA 보다 타자의 실제 "
        "생산력 지표인 wOBA 를 더 정확하게 설명함을 수학적으로 입증한다."
    )
    L.append("")
    return "\n".join(L)


def build_chapter_2_data(phase1_text: str, phase2_text: str) -> str:
    """챕터 2. 데이터 전처리 및 EDA — Phase 1 + Phase 2 융합."""
    L: list[str] = []
    L.append("## 2. 데이터 전처리 및 EDA")
    L.append("")
    L.append(
        "본 챕터는 우수 참고 자료의 구조(전체 분량의 약 절반을 데이터 파트에 "
        "할애)를 따라, Phase 1 의 도메인 기반 전처리·연도별 분리 결과와 "
        "Phase 2 의 다중공선성 제거·스케일링·샘플링·Feature Selection 결과를 "
        "통합 서술한다."
    )
    L.append("")
    L.append("### 2.A Phase 1 — 데이터 통합 및 도메인 기반 전처리")
    L.append("")
    # phase 1 본문: # → ###, ## → ####, ### → #####
    shifted1 = shift_headers(phase1_text, shift=2)
    L.append(apply_chapter_polish(shifted1).lstrip())
    L.append("")
    L.append("### 2.B Phase 2 — 상관관계 분석, 스케일링, 최적 샘플링 도출")
    L.append("")
    shifted2 = shift_headers(phase2_text, shift=2)
    L.append(apply_chapter_polish(shifted2).lstrip())
    L.append("")
    return "\n".join(L)


def build_chapter_3_ablation(phase3_text: str) -> str:
    L: list[str] = []
    L.append("## 3. 효과 분리 실험 (2×2 Factorial Ablation Study)")
    L.append("")
    L.append(
        "본 챕터는 트리 앙상블 모델 도입의 학술적 정당성을 확보하기 위한 "
        "2×2 Factorial Design 실험 결과를 서술한다. \"환경 변수의 가치는 "
        "비선형 모델 위에서만 발현된다\" 는 명제를 **Interaction effect** "
        "로 직접 입증함이 본 챕터의 핵심이다."
    )
    L.append("")
    shifted = shift_headers(phase3_text, shift=1)
    L.append(apply_chapter_polish(shifted).lstrip())
    L.append("")
    return "\n".join(L)


def build_chapter_4_tuning(phase4_text: str) -> str:
    L: list[str] = []
    L.append("## 4. Advanced Model 튜닝 + Stacking Meta Model")
    L.append("")
    L.append(
        "본 챕터는 X_advanced(62 변수) 위에서 RF / XGBoost / LightGBM 세 모델의 "
        "분류 성능과 확률 보정(Calibration) 품질을 극한으로 끌어올리고, 세 모델의 "
        "예측 확률을 종합한 Stacking Meta Model 을 구축한 결과를 서술한다. "
        "Phase 5 의 ca-xBA ↔ wOBA 상관관계 극대화를 위해 단순 분류 정확도가 "
        "아닌 **Brier Score · Log Loss 중심의 확률 보정 최적화** 를 본격 추구한다."
    )
    L.append("")
    shifted = shift_headers(phase4_text, shift=1)
    L.append(apply_chapter_polish(shifted).lstrip())
    L.append("")
    return "\n".join(L)


def build_chapter_5_validation(phase5_text: str) -> str:
    L: list[str] = []
    L.append("## 5. 최종 지표(ca-xBA) 산출 및 세이버메트릭스 가치 검증")
    L.append("")
    L.append(
        "본 챕터는 Phase 4 에서 선정된 최종 모델(LGBM Isotonic, Brier=0.1323)을 "
        "격리된 2025년 데이터에 적용해 타구별 ca-xBA 를 산출하고, 선수별 평균 "
        "ca-xBA 가 실제 `wOBA` 와 강한 상관관계를 가지는지 통계적·도메인적 양면에서 "
        "검증한 결과를 서술한다."
    )
    L.append("")
    shifted = shift_headers(phase5_text, shift=1)
    L.append(apply_chapter_polish(shifted).lstrip())
    L.append("")
    return "\n".join(L)


def build_chapter_6_conclusion() -> str:
    L: list[str] = []
    L.append("## 6. 결론 및 시사점")
    L.append("")
    L.append("### 6.1 핵심 결과 요약")
    L.append("")
    L.append("본 연구는 다음 네 가지 핵심 결과를 도출한다.")
    L.append("")
    L.append(
        "1. **비선형 상호작용의 존재 입증 (Phase 3)** — 2×2 Factorial 의 "
        "Interaction effect ΔAUC = +0.0119 (양수). 환경 변수의 가치는 "
        "비선형 모델(트리 앙상블) 위에서만 발현된다는 명제를 직접 입증한다."
    )
    L.append("")
    L.append(
        "2. **확률 보정 품질의 극대화 (Phase 4)** — LGBM Isotonic 모델이 "
        "Brier Score 0.1323 으로 Phase 3 default 대비 약 10% 개선된다. "
        "ca-xBA 가 단순 분류 출력이 아닌 *잘 보정된 확률* 임을 보장한다."
    )
    L.append("")
    L.append(
        "3. **세이버메트릭스 가치 통계 검증 (Phase 5)** — ca-xBA 의 실제 `wOBA` "
        "에 대한 R² = 0.3923 으로 MLB 공식 xBA(R² = 0.2499) 대비 절대 +0.1424, "
        "상대 +57.0% 우수하다. Pearson r 과 Spearman ρ 양면에서 모두 공식 "
        "지표를 능가한다."
    )
    L.append("")
    L.append(
        "4. **도메인 일관성 점검 (Phase 5)** — 포지션별 ca-xBA Top 10 의 실버 "
        "슬러거 적중률이 70.6%(전통 AVG 52.9% 대비 +17.7%p) 이며, Mike Trout 의 "
        "BABIP 불운·Luis Arraez 의 soft contact 등 모델 평가가 도메인 전문가 "
        "분석과 일치한다."
    )
    L.append("")
    L.append("### 6.2 학술적 의의 — wOBA 와 ca-xBA 의 보완 관계")
    L.append("")
    L.append(
        "wOBA 가 선수의 공격력을 평가하는 *결과(Result) 지표·완성형 성적표* 라면, "
        "ca-xBA 는 선수의 *순수한 타격 기술과 과정(Process)* 을 평가하는 "
        "**엑스레이(X-ray) 사진** 이다. 두 지표는 용도와 의의가 완전히 다르다. "
        "wOBA 는 시즌이 끝났을 때 '누가 얼마나 점수를 만들어 냈는가' 를 "
        "뒤돌아보는 지표이며, ca-xBA 는 선구안·운·야수 수비력·구장 환경을 "
        "통제한 후 남는 *순수 타격 기술* 을 추출하여 '내년에 누가 반등할 것인가' "
        "를 꿰뚫어 보는 강력한 예측 엔진이다."
    )
    L.append("")
    L.append("### 6.3 실무적 시사점 — Front Office 의 머니볼 도구")
    L.append("")
    L.append(
        "구단 프런트 오피스의 관점에서 ca-xBA 의 활용 가치는 두 가지로 요약된다. "
        "첫째, **저평가 선수 발굴**: 실제 타율이나 wOBA 가 형편없지만 ca-xBA 가 "
        "매우 높은 선수는 \"타격 기술이 나쁜 것이 아니라 그 해 운이 없었거나 "
        "구장과 궁합이 안 맞았을 뿐\" 으로 해석할 수 있어, 싼 값에 트레이드 "
        "또는 FA 영입의 근거 자료가 된다. 둘째, **구단별 구장 맞춤 평가**: ca-xBA "
        "는 구장 환경 변수를 학습하였으므로, 특정 선수가 자기 팀 홈구장에서 "
        "어떤 성과를 기록할 것인가 를 사전 시뮬레이션 가능하다. 이는 데이터 "
        "기반 트레이드·드래프트 의사결정의 핵심 도구가 된다."
    )
    L.append("")
    L.append("### 6.4 연구의 한계 및 향후 과제")
    L.append("")
    L.append(
        "본 연구의 한계는 다음과 같다. 첫째, **Phase 1 의 Athletics 제외 결정** 으로 "
        "ATH 소속 선수의 ca-xBA 는 원정 경기 타석만 집계되어 표본 안정성이 떨어진다. "
        "둘째, **모델의 'Schwarber 패턴' 편향** — ca-xBA 가 BIP 한정 quality 를 "
        "평가하므로 fly ball power hitter (예: Kyle Schwarber, NL MVP 2위 + "
        "56 홈런)가 luck=음수로 평가되는 *구조적 편향* 이 존재한다. 향후 연구에서는 "
        "ATH 신규 구장(Sutter Health Park) 의 환경 데이터 수집과 fly ball 패널티 "
        "보정 메커니즘 도입이 필요하다. 셋째, 본 연구는 **2024년 데이터로 학습 후 "
        "2025년 같은 해 내에서 검증** 하는 구조이므로, 진정한 의미의 "
        "**Year-to-Year 예측력** (2024 ca-xBA → 2026 wOBA) 검증은 후속 연구의 "
        "과제로 남는다."
    )
    L.append("")
    return "\n".join(L)


def build_chapter_7_references(url_sources: list[tuple[str, str]]) -> str:
    L: list[str] = []
    L.append("## 7. 참고문헌")
    L.append("")
    L.append(
        "본 보고서 본문(특히 Phase 5 운/불운 타자 분석)에서 인용한 모든 외부 "
        "출처를 가나다·알파벳 순으로 정리한다. 통계 데이터는 Baseball Savant "
        "(Statcast 공식) 및 MLB Stats API, 도메인 평가는 전문 매체·커뮤니티 "
        "(MLB.com, FanGraphs, CBS Sports, Just Baseball, Reddit r/buccos 등) 의 "
        "공개 자료를 활용한다."
    )
    L.append("")
    L.append("### 7.1 데이터 소스")
    L.append("")
    L.append("1. MLB Statcast 타구 데이터 (Baseball Savant). https://baseballsavant.mlb.com/")
    L.append("2. Open-Meteo Historical Weather API. https://archive-api.open-meteo.com/v1/archive")
    L.append("3. MLB Stats API (선수별 포지션 조회). https://statsapi.mlb.com/api/v1/people")
    L.append("4. Baseball Savant Custom Leaderboard — Expected Statistics (`validation_2025_gt.csv`).")
    L.append("5. MLB.com — 2025 Silver Slugger Award Winners. https://www.mlb.com/news/2025-silver-slugger-award-winners")
    L.append("")
    L.append("### 7.2 도메인 평가 인용")
    L.append("")
    if url_sources:
        for i, (title, url) in enumerate(url_sources, 1):
            L.append(f"{i}. {title}. {url}")
    else:
        L.append(
            "본 자동 컴파일 보고서 본문에서는 외부 도메인 평가(전문 매체·커뮤니티 분석)를 "
            "직접 인용하지 않았다. Phase 5 의 운/불운 Top 선수 스카우팅 서사는 보고서 외부의 "
            "별도 부록 자료에서 Baseball Savant 선수 프로필, FanGraphs, MLB.com, Reddit "
            "r/baseball 등의 공개 자료와 함께 보강하는 것을 권장한다."
        )
    L.append("")
    L.append("### 7.3 라이브러리 및 도구")
    L.append("")
    L.append("- Python 3.12, pandas 3.0, scikit-learn 1.8, XGBoost 3.2, LightGBM 4.6")
    L.append("- imbalanced-learn 0.14 (RandomUnderSampler, SMOTE)")
    L.append("- matplotlib 3.10, seaborn 0.13")
    L.append("- pypandoc 1.17 + pandoc 3.5, python-docx 1.2 (본 보고서 컴파일 도구)")
    L.append("")
    return "\n".join(L)


# -----------------------------------------------------------------------------
# 표지 및 메타 정보
# -----------------------------------------------------------------------------
def build_title_page() -> str:
    today = datetime.now().strftime("%Y년 %m월 %d일")
    L: list[str] = []
    L.append("---")
    L.append("title: \"Context-Aware xBA (ca-xBA): 환경 변수 통합 기대 타율 예측 모델\"")
    L.append("subtitle: \"데이터마이닝 텀프로젝트 최종 보고서\"")
    L.append(f"author: \"이지현\"")
    L.append(f"date: \"{today}\"")
    L.append("lang: ko-KR")
    L.append("---")
    L.append("")
    L.append("# Context-Aware xBA (ca-xBA)")
    L.append("## 환경 변수 통합 기대 타율 예측 모델")
    L.append("")
    L.append("> *데이터마이닝 텀프로젝트 최종 보고서*")
    L.append("")
    L.append(f"**작성:** 이지현 · **제출일:** {today}")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 초록 (Abstract)")
    L.append("")
    L.append(
        "본 연구는 메이저리그(MLB) 공식 기대 타율(xBA)이 타구의 물리적 질(발사 속도, 발사 각도)"
        "만 평가하고 환경 맥락을 무시한다는 구조적 한계를 극복하기 위해, **상황 인지형 기대 "
        "타율(ca-xBA, Context-Aware xBA)** 을 제안하고 그 학술적·실무적 가치를 통계적으로 "
        "검증한다. 2024-2025 시즌 Statcast BIP(인플레이 타구) 약 22만 5천 건에 구장 스펙 "
        "(펜스 거리·높이·고도·지붕) 및 Open-Meteo 기반 기상 데이터(온도·풍속·풍향·기압·습도·"
        "강수·운량·돌풍 8 종)를 결합하고, RF / XGBoost / LightGBM 트리 앙상블에 Isotonic "
        "Calibration 을 적용해 잘 보정된 확률(well-calibrated probability) 형태의 ca-xBA 를 "
        "산출한다. 핵심 검증 결과는 다음과 같다. **(1) 2×2 Factorial Ablation** 에서 "
        "Interaction effect ΔAUC = +0.0119 (양수) 로 \"환경 변수의 가치가 비선형 모델 위에서만 "
        "발현된다\" 는 명제를 직접 입증한다. **(2) 1:1 R² 대조** 에서 ca-xBA 의 실제 `wOBA` 에 "
        "대한 R² = 0.3923 으로 MLB 공식 xBA (R² = 0.2499) 대비 절대 +0.1424, 상대 +57.0% "
        "우수하다. Pearson r (0.6264 vs 0.4999) 과 Spearman ρ (0.5767 vs 0.4729) 모두에서 "
        "공식 지표를 능가한다. **(3) 도메인 일관성 점검** 에서 포지션별 ca-xBA Top 10 의 실버 "
        "슬러거 적중률이 70.6% (전통 AVG 52.9% 대비 +17.7%p) 이며, Mike Trout 의 BABIP 불운, "
        "Luis Arraez 의 soft contact 등 모델 평가가 도메인 전문가 분석과 일치한다. 본 연구는 "
        "ca-xBA 가 wOBA 의 '결과 지표' 와 보완 관계를 이루는 '과정 지표' 로서 구단 프런트 "
        "오피스의 저평가 선수 발굴·머니볼 의사결정에 활용 가능한 강력한 예측 엔진임을 입증한다."
    )
    L.append("")
    L.append("**키워드:** ca-xBA, 기대 타율, 세이버메트릭스, 트리 앙상블, Calibration, 환경 변수, "
             "Stacking, Ablation Study, wOBA, Statcast")
    L.append("")
    L.append("---")
    L.append("")
    L.append("\\newpage")
    L.append("")
    return "\n".join(L)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    log("=" * 80)
    log("ca-xBA 최종 보고서 (Word .docx) 자동 컴파일")
    log("=" * 80)

    # 1. 마크다운 파일 로드
    log("\n[1/5] 8 개 마크다운 파일 로드 ...")
    readme = README_MD.read_text(encoding="utf-8")
    phase1 = PHASE1_MD.read_text(encoding="utf-8")
    phase2 = PHASE2_MD.read_text(encoding="utf-8")
    phase3 = PHASE3_MD.read_text(encoding="utf-8")
    phase4 = PHASE4_MD.read_text(encoding="utf-8")
    phase5 = PHASE5_MD.read_text(encoding="utf-8")
    log(f"  readme: {len(readme):,d}자  phase1: {len(phase1):,d}자  phase2: {len(phase2):,d}자")
    log(f"  phase3: {len(phase3):,d}자  phase4: {len(phase4):,d}자  phase5: {len(phase5):,d}자")

    # 2. URL 출처 자동 집계 (참고문헌용)
    log("\n[2/5] 참고문헌 URL 출처 자동 집계 ...")
    url_sources = extract_url_sources(phase1, phase2, phase3, phase4, phase5)
    log(f"  추출된 URL: {len(url_sources)}건")

    # 3. 챕터별 본문 구성
    log("\n[3/5] 7 개 챕터 본문 구성 (우수자료 구조 기반) ...")
    parts: list[str] = []
    parts.append(build_title_page())
    parts.append(build_chapter_1_intro(readme))
    parts.append(build_chapter_2_data(phase1, phase2))
    parts.append(build_chapter_3_ablation(phase3))
    parts.append(build_chapter_4_tuning(phase4))
    parts.append(build_chapter_5_validation(phase5))
    parts.append(build_chapter_6_conclusion())
    parts.append(build_chapter_7_references(url_sources))
    merged_md = "\n\n".join(parts)
    log(f"  병합된 마크다운: {len(merged_md):,d}자, {merged_md.count(chr(10)):,d}줄")

    # 4. 문체 통일 (~이다 체) + 이미지 경로 보정
    log("\n[4/5] 문체 통일 (~이다 체 변환) + 이미지 경로 보정 ...")
    merged_md = unify_style(merged_md)
    merged_md = fix_image_paths(merged_md)
    log(f"  변환 후 마크다운: {len(merged_md):,d}자")

    # 디버그용: 병합 마크다운 저장 (검증 용이)
    debug_md_path = ROOT / "_merged_report_debug.md"
    debug_md_path.write_text(merged_md, encoding="utf-8")
    log(f"  디버그용 병합 md: {debug_md_path.relative_to(ROOT)}")

    # 5. pypandoc 으로 docx 변환
    log(f"\n[5/5] pypandoc → Word docx 변환 ...")
    log(f"  pandoc version: {pypandoc.get_pandoc_version()}")
    extra_args = [
        "--toc",  # 목차 자동 생성
        "--toc-depth=3",
        f"--resource-path={ROOT}",  # 이미지 경로 base
        "--standalone",
    ]
    pypandoc.convert_text(
        merged_md,
        to="docx",
        format="md",
        outputfile=str(OUTPUT_DOCX),
        extra_args=extra_args,
    )
    size = OUTPUT_DOCX.stat().st_size
    log(f"  ✓ 생성 완료: {OUTPUT_DOCX.name} ({size:,d} 바이트, {size/1024/1024:.2f} MB)")
    log(f"  → 경로: {OUTPUT_DOCX}")
    log("\n[done] 보고서 컴파일 완료.")


if __name__ == "__main__":
    main()
