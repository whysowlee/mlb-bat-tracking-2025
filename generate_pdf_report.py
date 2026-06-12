"""
ca-xBA 최종 보고서 (PDF, LaTeX engine) 자동 컴파일 스크립트
===========================================================

generate_docx_report.py 의 PDF 버전. 학술 보고서 가독성 + 표/figure 자동 캡션
넘버링을 위해 LaTeX (tectonic engine) 사용.

**사용자 수정사항 (총 13건) 모두 반영:**
  1. 문서 전체 제출일 삭제
  2. 우수자료 참고 언급 삭제
  3. 모든 표/figure 자동 캡션 넘버링 (LaTeX native)
  4. PDF 포맷 (Word 대체)
  5. "Table of Contents" → "목차"
  6. 목차 위 두 줄 (제목/부제) 제거 → 초록부터
  7. 목차 끝나고 바로 초록
  8. 초록에서 구체적 수치 스포일러 제거
  9. 표 테두리 (LaTeX booktabs/longtable + 모든 셀)
  10. 1.3 로드맵 표에서 "주요 산출" 컬럼 삭제
  11. 2.A Phase 1 다음 서론 요약 문단 삭제
  12. 2.A 시각화 중복 서술 통합 (※ 자동 처리 한계 — 본문 정리)
  13. 이모지 제거 (🍀💀✓✗📝⚠️🏆★→ 등)

실행:
    /opt/miniconda3/envs/mlb-xba/bin/python generate_pdf_report.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pypandoc

# LaTeX/tectonic 이 한글 경로를 깨트리는 문제 회피용 영문 임시 디렉토리
# (모든 figure 와 출력 파일을 여기에 두고, 컴파일 후 결과만 한글 경로로 복사)
TMP_ROOT = Path("/tmp/ca-xba-pdf-build")
TMP_FIGURES = TMP_ROOT / "figures"

ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT / "pipeline"
OUTPUT_PDF = ROOT / "ca-xBA_Final_Report.pdf"
DEBUG_MD = ROOT / "_merged_report_debug.md"

README_MD = ROOT / "readme.md"
PHASE1_MD = PIPELINE_DIR / "phase1_report.md"
PHASE2_MD = PIPELINE_DIR / "phase2_report.md"
PHASE3_MD = PIPELINE_DIR / "phase3_report.md"
PHASE4_MD = PIPELINE_DIR / "phase4_report.md"
PHASE5_MD = PIPELINE_DIR / "phase5_report.md"


def log(msg: str) -> None:
    print(msg, flush=True)


# ============================================================================
# 1. 문체 통일 (~이다 체) — generate_docx_report.py 와 동일
# ============================================================================
STYLE_RULES: list[tuple[str, str]] = [
    (r"습니다(?=[\.\,\!\?\s\n]|$)", "다"),
    (r"입니다(?=[\.\,\!\?\s\n]|$)", "이다"),
    (r"합니다(?=[\.\,\!\?\s\n]|$)", "한다"),
    (r"됩니다(?=[\.\,\!\?\s\n]|$)", "된다"),
    (r"있습니다(?=[\.\,\!\?\s\n]|$)", "있다"),
    (r"없습니다(?=[\.\,\!\?\s\n]|$)", "없다"),
    (r"줍니다(?=[\.\,\!\?\s\n]|$)", "준다"),
    (r"옵니다(?=[\.\,\!\?\s\n]|$)", "온다"),
    (r"갑니다(?=[\.\,\!\?\s\n]|$)", "간다"),
    (r"봅니다(?=[\.\,\!\?\s\n]|$)", "본다"),
    (r"드립니다(?=[\.\,\!\?\s\n]|$)", "한다"),
    (r"습니까(?=[\.\,\!\?\s\n]|$)", "는가"),
    (r"까요(?=[\.\,\!\?\s\n]|$)", "는가"),
    (r"이에요(?=[\.\,\!\?\s\n]|$)", "이다"),
    (r"예요(?=[\.\,\!\?\s\n]|$)", "이다"),
    (r"네요(?=[\.\,\!\?\s\n]|$)", "다"),
    (r"군요(?=[\.\,\!\?\s\n]|$)", "다"),
    (r"세요(?=[\.\,\!\?\s\n]|$)", "라"),
    (r"해요(?=[\.\,\!\?\s\n]|$)", "한다"),
    (r"써요(?=[\.\,\!\?\s\n]|$)", "쓴다"),
    (r"보입니다", "보인다"),
    (r"보여줍니다", "보여준다"),
    (r"만듭니다", "만든다"),
    (r"이루어집니다", "이루어진다"),
    (r"가집니다", "가진다"),
]


def _protect_code_blocks(text: str) -> tuple[str, list[str]]:
    blocks: list[str] = []

    def _save(match):
        blocks.append(match.group(0))
        return f"\x00BLOCK{len(blocks) - 1}\x00"

    text = re.sub(r"```[\s\S]*?```", _save, text)
    text = re.sub(r"`[^`\n]+`", _save, text)
    text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", _save, text)
    text = re.sub(r"\[[^\]]*\]\([^\)]+\)", _save, text)
    text = re.sub(r"https?://\S+", _save, text)
    return text, blocks


def _restore_code_blocks(text: str, blocks: list[str]) -> str:
    for i, block in enumerate(blocks):
        text = text.replace(f"\x00BLOCK{i}\x00", block)
    return text


def unify_style(text: str) -> str:
    text, blocks = _protect_code_blocks(text)
    for pattern, replacement in STYLE_RULES:
        text = re.sub(pattern, replacement, text)
    text = _restore_code_blocks(text, blocks)
    return text


# ============================================================================
# 2. 이모지 제거 (수정사항 #13)
# ============================================================================
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F700-\U0001F77F"  # Alchemical
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U00002600-\U000026FF"  # Misc Symbols
    "\U0001F0A0-\U0001F0FF"  # Playing Cards
    # 주의: \U000024C2-\U0001F251 범위는 한글 음절(U+AC00-D7A3)을 포함하므로 사용 금지
    "]+",
    flags=re.UNICODE,
)

# 일부 단독 심볼 (이모지 범위 밖이지만 학술 보고서 부적합)
EXTRA_SYMBOLS_TO_REMOVE = ["★", "✨", "→", "←", "↑", "↓"]
# 단, 특정 위치에는 의미가 있을 수 있으므로 ↑↓는 BABIP 컬럼 라벨에서만 사용됨
# 모두 일괄 제거하면 표가 손상되므로 → 만 화살표 풀어쓰기


def strip_emojis(text: str) -> str:
    text = EMOJI_PATTERN.sub("", text)
    text = text.replace("★", "")
    # Variation selector (U+FE0F) — emoji 의 색 변경자, emoji 본체 없이 잔재 가능
    text = re.sub(r"[︀-️]", "", text)
    # Zero-width joiner (emoji combo)
    text = text.replace("‍", "")
    return text


# ============================================================================
# 2b. LaTeX 호환 문자 변환 — Latin Modern 폰트 없는 수학 기호를 math mode 로
# ============================================================================
UNICODE_MATH_MAP = [
    # 본문에서도 ASCII 또는 newunicodechar (header.tex) 처리. math mode 변환은
    # 일부 케이스(bold/heading 안)에서 깨지므로 ASCII 대체가 가장 안전.
    ("×", "x"),    # 2×2 → 2x2 (학술 문맥에서도 일반적)
    ("·", "·"),   # newunicodechar 가 처리
    ("→", "→"),    # newunicodechar
    ("←", "←"),
    ("↑", "↑"),
    ("↓", "↓"),
    ("²", "²"),   # newunicodechar
    ("³", "³"),
    ("°", "°"),
    # 수학 기호는 newunicodechar 가 \ensuremath 자동 적용. 여기서는 변환 X
]


UNICODE_ASCII_FALLBACK = [
    # 인라인 코드 등 mono 폰트에서도 깨지면 안 되는 ASCII 대체
    ("≈", "~="),
    ("≥", ">="),
    ("≤", "<="),
    ("≠", "!="),
    ("ρ", "rho"),
    ("ε", "eps"),
    ("μ", "mu"),
    ("σ", "sigma"),
    ("β", "beta"),
    ("α", "alpha"),
    ("γ", "gamma"),
    ("Δ", "Delta"),
    ("∈", "in"),
    ("∉", "not in"),
    ("∞", "inf"),
    ("±", "+/-"),
    ("→", "->"),
    ("←", "<-"),
    ("↑", "up"),
    ("↓", "down"),
    ("·", "*"),
    ("×", "x"),
    ("÷", "/"),
]


def convert_unicode_math(text: str) -> str:
    """LaTeX 컴파일 안전성을 위한 unicode/특수문자 처리.

    - **본문 일반 텍스트**: × → x 같은 ASCII 대체. 나머지 수학 기호 (≥ ≤ ρ ε 등)
      는 header.tex 의 `\\newunicodechar` 가 자동으로 `\\ensuremath{...}` 처리.
    - **인라인 코드 ``...``**: ASCII fallback (mono 폰트는 unicode 미지원).
    - **fenced code block**: 원문 보존.
    """
    blocks: list[str] = []
    def _save(m):
        blocks.append(m.group(0))
        return f"\x00FENCED{len(blocks) - 1}\x00"
    text = re.sub(r"```[\s\S]*?```", _save, text)

    # 인라인 코드 안 unicode → ASCII
    def _inline_fallback(m):
        code = m.group(0)
        for ch, ascii_rep in UNICODE_ASCII_FALLBACK:
            code = code.replace(ch, ascii_rep)
        return code
    text = re.sub(r"`[^`\n]+`", _inline_fallback, text)

    # 일반 텍스트 — 안전 ASCII 대체 (math mode 변환 금지: bold/heading 깨짐 방지)
    for ch, ascii_rep in UNICODE_MATH_MAP:
        text = text.replace(ch, ascii_rep)

    # fenced code 복원
    for i, block in enumerate(blocks):
        text = text.replace(f"\x00FENCED{i}\x00", block)
    return text


# ============================================================================
# 3. 학술 보고서 후처리 — 메타/메모/산출물/결정표 strip
# ============================================================================
def strip_section_by_header(text: str, header_pattern: str) -> str:
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
            m = re.match(r"^(#+)\s", line)
            if m and len(m.group(1)) <= skip_level:
                skipping = False
                out.append(line)
    return "\n".join(out)


def strip_meta_headers(text: str) -> str:
    text = re.sub(r"^_생성:.*?_\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^_실행 스크립트:.*?_\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#+ Phase \d+ Report.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^> \*\*모델 메타:\*\*.*?$", "", text, flags=re.MULTILINE)
    return text


def strip_user_memos(text: str) -> str:
    text = strip_section_by_header(text, r"^#+ \d+\.\d+ 운/불운 Top \d+ — 스카우팅 서사")
    text = re.sub(r"\s*\(사용자 작성 영역\)", "", text)
    text = re.sub(
        r"^> \*\*방법론.*?$\n(?:^>.*?$\n)*",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^_본 자동 리포트는 객관적 수치.*?$\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text


def strip_file_paths_and_artifacts(text: str) -> str:
    text = strip_section_by_header(text, r"^#+ \d+\. 산출물")
    text = re.sub(r"^PNG 파일은.*$", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"^> _\([A-Z]\) [^_]*?(제외|가독성).*?_\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text


def strip_decision_tables(text: str) -> str:
    return strip_section_by_header(text, r"^#+ \d+\. 결정 사항")


def replace_first_person_to_academic(text: str) -> str:
    """`우리 BIP`, `우리 ca-xBA`, `우리 모델` 같은 1인칭 단축형을 학술 톤으로 변경.

    - "우리 BIP" → "본 분석의 BIP"
    - "우리 ca-xBA" → "본 연구의 ca-xBA"
    - "우리 모델" → "본 연구의 모델"
    - "우리 데이터" → "본 분석의 데이터"
    - 단독 "우리" 도 가능한 경우 변환 (단 본문 흐름상 학술적이면 유지)
    """
    # 코드/링크는 보호
    blocks: list[str] = []
    def _save(m):
        blocks.append(m.group(0))
        return f"\x00ACAD{len(blocks) - 1}\x00"
    text = re.sub(r"```[\s\S]*?```", _save, text)
    text = re.sub(r"`[^`\n]+`", _save, text)
    text = re.sub(r"\[[^\]]*\]\([^\)]+\)", _save, text)

    rules = [
        (r"우리 모델", "본 연구의 모델"),
        (r"우리 ca-xBA", "본 연구의 ca-xBA"),
        (r"우리 BIP", "본 분석의 BIP"),
        (r"우리 데이터", "본 분석의 데이터"),
        (r"우리 분석군", "본 분석의 대상군"),
        (r"우리 분석", "본 분석"),
        (r"우리의 BIP", "본 분석의 BIP"),
        (r"우리의 ca-xBA", "본 연구의 ca-xBA"),
        (r"우리의 모델", "본 연구의 모델"),
        (r"우리의 분석", "본 분석"),
    ]
    for pat, rep in rules:
        text = re.sub(pat, rep, text)

    for i, b in enumerate(blocks):
        text = text.replace(f"\x00ACAD{i}\x00", b)
    return text


def strip_2x2_cell_label(text: str) -> str:
    """§3 표 11 위 '**2x2 셀:**' 또는 '**2×2 셀:**' 같은 군더더기 label 제거."""
    text = re.sub(r"^\s*\*\*2[x×]2\s*셀\s*[:：]?\*\*\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\*\*2[x×]2\s*Factorial\s*셀\s*[:：]?\*\*\s*$", "", text, flags=re.MULTILINE)
    return text


def strip_comparison_spec(text: str) -> str:
    """phase5 §5.2 '비교 구도 명세' 인용 블록 제거."""
    # `> **⚠️ 비교 구도 명세:** ...` 한 단락
    text = re.sub(
        r"^>\s*\*\*[^*]*비교 구도 명세[^*]*\*\*[^\n]*(?:\n>[^\n]*)*",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text


def replace_explanation_tone(text: str) -> str:
    """톤 조정 — '해명' / 'dome × weather 우리가 확인' 같은 비학술적 표현 보정."""
    rules = [
        (r"※\s*누락\s*선수\s*해명", "※ 누락 선수 — Hard Join 매칭의 기술적 한계"),
        # phase report 에서 dome × weather 가 "우리가 단독 포착" 같은 톤이면 도메인 상식 명시
        (
            r"dome\s*[×x]\s*weather\s*상호작용을?\s*ca-xBA가?\s*단독으로?\s*포착",
            "dome × weather 상호작용 (도메인 상식) 을 ca-xBA 가 모델 입력으로 반영",
        ),
        (
            r"BABIP\s*가?\s*잡지\s*못하는\s*\*\*dome\s*[×x]\s*weather\s*상호작용",
            "BABIP 단일 지표가 반영하지 못하는 **환경 변수 (dome × weather 등 도메인 상식 기반",
        ),
    ]
    for pat, rep in rules:
        text = re.sub(pat, rep, text)
    return text


def strip_manual_section_numbers(text: str) -> str:
    """phase report 본문의 모든 헤더에서 수동 번호 부분 제거.

    예: `## 2. 데이터 매칭` → `## 데이터 매칭`
         `### 4.4 운(행운 효과) Top 10` → `### 운(행운 효과) Top 10`
         `#### 8.1 RF Importance Top 20` → `#### RF Importance Top 20`

    LaTeX `numbersections=true` 가 자동으로 N.M 번호를 부여하므로 수동 번호 중복 방지.
    """
    # 1. "## 4.4 운...", "## 2. 데이터" 형태 (숫자/점)
    text = re.sub(
        r"^(#+)\s+(\d+(?:\.\d+)*)\s*\.?\s+",
        r"\1 ",
        text,
        flags=re.MULTILINE,
    )
    # 2. "## 7b. 돔..." 형태 (숫자+소문자)
    text = re.sub(
        r"^(#+)\s+(\d+[a-z])\s*\.?\s+",
        r"\1 ",
        text,
        flags=re.MULTILINE,
    )
    return text


def strip_md_report_guidance(text: str) -> str:
    """`MD 리포트 포함 요소:` 로 시작하는 작업용 지시문 불렛 제거.

    한 줄로 끝나는 경우도 있고 여러 줄에 걸쳐 길게 이어지는 경우도 있어
    헤더/표/다음 H3 같은 명확한 종결자까지 통째 잘라낸다.
    """
    # "MD 리포트 포함 요소:" 라인부터 그 sub-bullet 들 모두 제거 (들여쓰기 무관)
    # 한 줄씩 처리해서 안전하게 잘라낸다.
    lines = text.split("\n")
    out: list[str] = []
    skipping = False
    for line in lines:
        if not skipping:
            # 헤더/불렛 들여쓰기에 무관하게 "MD 리포트 포함 요소" 시작 라인 감지
            if re.match(r"^\s*[-*]?\s*\*?\*?MD 리포트 포함 요소", line):
                skipping = True
                continue
            out.append(line)
        else:
            # 빈 줄 또는 새 H1/H2/H3 헤더 또는 들여쓰기 없는 평문 단락이 나오면 종료
            if line.strip() == "":
                skipping = False
                out.append(line)
            elif re.match(r"^#{1,6}\s", line):
                skipping = False
                out.append(line)
            elif re.match(r"^[a-zA-Z가-힣]", line):
                # 들여쓰기 없는 평문 (다음 일반 단락 시작)
                skipping = False
                out.append(line)
            # 그 외 (들여쓰기된 sub-bullet 등) 은 계속 skip
    return "\n".join(out)


def strip_guardrail_phrases(text: str) -> str:
    """AI 통제용 가드레일 문구 제거 — 학술 보고서에 부적합.

    예: "(필수)", "(재미용 점검)", "절대 fabricate 금지", "추정·창작 금지",
        "자동 fabrication 금지", "(사용자 작성 영역)"
    """
    patterns = [
        r"\s*\(필수\)",
        r"\s*\(재미용\s*점검\)",
        r"\s*절대\s*fabricate\s*금지",
        r"\s*절대\s*[fF]abrication\s*금지",
        r"\s*추정[·.]\s*창작[은\s]*절대\s*금지(?:한다|함)?",
        r"\s*자동\s*[fF]abrication\s*금지(?:\s*원칙)?",
        r"\s*\(사용자\s*작성\s*영역\)",
        # "방법론 (자동 fabrication 금지)" 형태 인용블록 한 단락
        r"^>\s*\*\*방법론[^\n]*\n(?:^>.*?\n)*",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE)
    return text


def strip_visualization_redundancy(text: str) -> str:
    """`## N. 시각화` 또는 `### N. 시각화` 섹션 내부의 figure 별 해석 텍스트 압축.

    각 figure (`![...](...)`) 앞뒤 1개의 캡션 라인은 유지하고, 그 다음에 이어지는
    불렛/평문 해석 텍스트(다음 ### 헤더 또는 다음 figure 직전까지) 를 제거한다.
    시각화 섹션 본문 자체의 서두 안내는 figure 만 남기고 정리.

    구현 단순화: `## N. 시각화` 섹션 전체를 찾고, 그 안에서 figure markdown 과
    figure 캡션(`*그림 N. ...*`) 만 추출 + 다른 텍스트 제거.
    """
    # 시각화 섹션 헤더 (## N. 시각화 또는 ### N. 시각화)
    section_header_re = re.compile(r"^(#+)\s+\d+(?:\.\d+)?\.\s*시각화\b.*$", re.MULTILINE)
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = section_header_re.match(line)
        if not m:
            out.append(line)
            i += 1
            continue
        # 시각화 섹션 시작
        section_level = len(m.group(1))
        out.append(line)  # 헤더 자체는 유지
        i += 1
        # 다음 동급 헤더 전까지 본문 — figure + 캡션만 유지
        while i < len(lines):
            nxt = lines[i]
            # 다음 동급 이상 헤더 → 시각화 섹션 종료
            mm = re.match(r"^(#+)\s", nxt)
            if mm and len(mm.group(1)) <= section_level:
                break
            # figure markdown 유지
            if re.match(r"^!\[", nxt):
                out.append(nxt)
                i += 1
                # figure 다음 빈 줄 + 캡션 한 줄 ("*그림 N. ...*" 형태) 유지
                while i < len(lines) and lines[i].strip() == "":
                    out.append(lines[i])
                    i += 1
                if i < len(lines) and re.match(r"^\*그림\s\d+\.", lines[i]):
                    out.append(lines[i])
                    i += 1
                continue
            # 하위 헤더 (### N.M ...) 유지 — 그림 그룹 구분
            if mm and len(mm.group(1)) > section_level:
                out.append(nxt)
                i += 1
                continue
            # 그 외 (불렛/평문 해석) → 제거
            i += 1
        # 시각화 섹션 끝
    return "\n".join(out)


# 수정사항 #2: 우수자료 참고 언급 삭제
UWUJARO_PATTERNS = [
    r"\(우수자료\s*[^)]*\)",
    r"\(참고\s*프로젝트[^)]*\)",
    r"우수자료\s*\([^)]*\)",
    r"우수자료\s*벤치마킹\s*",
    r"우수자료\s*구조\s*기반\s*",
    r"우수자료의?\s*훌륭한\s*구조를?\s*",
    r"참고\s*프로젝트\s*벤치마킹\s*",
    r"\(김강현[^)]*\)",
    r"김강현\s*우수자료\s*",
    r"※\s*우수자료[^\n]*",
    r"※\s*[Cc]ite[^\n]*",
    r"\[cite_start\]",
    r"\[cite:\s*\d+(?:,\s*\d+)*\]",
]


def strip_uwujaro_refs(text: str) -> str:
    for pat in UWUJARO_PATTERNS:
        text = re.sub(pat, "", text)
    return text


# 수정사항 #11: 2.A Phase 1 다음 서론 요약 문단 삭제 — phase1 본문 시작부의 인용문 제거
def strip_phase1_intro_blockquote(text: str) -> str:
    """phase1_report.md 의 시작부 '> 2024-2025 ...' 인용문이 챕터 2.A 서두에
    중복으로 나타나면 제거 (실제로는 generate_pdf 의 chapter 2 builder 에서
    이미 '서론' 텍스트로 들어가니 phase1 본문에서는 한 번만 등장하도록 정리)."""
    # phase report 의 H1 (이미 strip 됨) 직후의 첫 인용 블록 한 단락 제거
    text = re.sub(
        r"\A\s*(?:^>.*?$\n)+",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text


def apply_chapter_polish(text: str) -> str:
    text = strip_meta_headers(text)
    text = strip_user_memos(text)
    text = strip_file_paths_and_artifacts(text)
    text = strip_decision_tables(text)
    text = strip_uwujaro_refs(text)
    text = strip_md_report_guidance(text)
    text = strip_guardrail_phrases(text)
    text = strip_2x2_cell_label(text)
    text = strip_comparison_spec(text)
    text = replace_first_person_to_academic(text)
    text = replace_explanation_tone(text)
    # 시각화 § 헤더만 제거 — figure/캡션/해석은 본문 inline 으로 흘러감
    text = strip_visualization_section_headers(text)
    text = strip_manual_section_numbers(text)
    text = strip_emojis(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ============================================================================
# 4. 헤더 시프트 + 이미지 경로 보정
# ============================================================================
def postprocess_longtable_columns(tex: str) -> str:
    """pandoc longtable column spec 을 자동 줄바꿈 + 컬럼별 가중 폭으로 치환.

    - 단순 컬럼: 균등 분배
    - 좁아도 되는 컬럼 (숫자, r-aligned 의 짧은 값): 좁게
    - 넓어야 하는 컬럼 (변수명·설명): 넓게
    """
    pattern = re.compile(r"\\begin\{longtable\}(\[[^\]]*\])?\{(@\{\}[^@}]+@\{\}|[^}]+)\}")

    def _replace(m):
        opts = m.group(1) or ""
        spec = m.group(2)
        clean = spec.replace("@{}", "")
        if not re.match(r"^[lcr]+$", clean):
            return m.group(0)
        n_cols = len(clean)

        # 컬럼별 가중 폭 결정:
        #   l = left align (보통 텍스트/이름) → 가중치 1.4
        #   r = right align (보통 숫자) → 가중치 0.7
        #   c = center → 가중치 1.0
        weights = []
        for c in clean:
            if c == "l":
                weights.append(1.4)
            elif c == "r":
                weights.append(0.7)
            else:
                weights.append(1.0)
        total = sum(weights)
        widths = [w / total * 0.98 for w in weights]  # 0.98 = 안전 margin

        parts = []
        for c, w in zip(clean, widths):
            align = "raggedright" if c == "l" else ("raggedleft" if c == "r" else "centering")
            parts.append(f">{{\\{align}\\arraybackslash}}p{{{w:.4f}\\linewidth}}")
        new_spec = "@{}" + "".join(parts) + "@{}"
        return f"\\begin{{longtable}}{opts}{{{new_spec}}}"

    return pattern.sub(_replace, tex)


def shift_headers(text: str, shift: int) -> str:
    def _shift(match):
        hashes = match.group(1)
        new_level = min(len(hashes) + shift, 6)
        return "#" * new_level + match.group(2)
    return re.sub(r"^(#{1,6})( .+)$", _shift, text, flags=re.MULTILINE)


def fix_image_paths(text: str) -> str:
    """이미지 경로를 영문 임시 디렉토리(TMP_FIGURES) 경로로 변환.

    한글 경로(2026 1학기 등)가 LaTeX/tectonic 에서 깨지므로 모든 figure 를
    /tmp/ca-xba-pdf-build/figures/ 로 복사 후 그 경로를 사용한다.
    """
    TMP_FIGURES.mkdir(parents=True, exist_ok=True)

    def _fix(match):
        alt = match.group(1)
        url = match.group(2)
        if url.startswith(("http://", "https://")):
            return match.group(0)
        # 절대 경로 또는 상대 경로 → 원본 파일 위치 결정
        if url.startswith("/"):
            src = Path(url)
        elif url.startswith(("figures/", "output/", "cache/", "logs/")):
            src = ROOT / "pipeline" / url
        elif url.startswith("pipeline/"):
            src = ROOT / url
        else:
            src = ROOT / url
        if not src.exists():
            return match.group(0)
        # 영문 파일명 보장 (원본 파일명만 사용 — 영문임)
        dst = TMP_FIGURES / src.name
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dst)
        return f"![{alt}]({dst.as_posix()})"
    return re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", _fix, text)


# ============================================================================
# 5. 1.3 로드맵 표에서 "주요 산출" 컬럼 삭제 (수정사항 #10)
# ============================================================================
def strip_roadmap_column(text: str) -> str:
    """readme/intro 의 로드맵 표에서 '주요 산출' / '산출물' 등 컬럼 제거.

    마크다운 표는 | col1 | col2 | col3 | 형태. 마지막 컬럼 또는 '산출' 키워드
    포함 컬럼을 제거.
    """
    lines = text.split("\n")
    out: list[str] = []
    in_table = False
    drop_idx: int | None = None

    for line in lines:
        # 표 시작 감지 — 헤더 라인에 '주요 산출' 또는 '산출물' 키워드
        if re.match(r"^\s*\|.+\|\s*$", line) and not in_table:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            for i, c in enumerate(cells):
                if "주요 산출" in c or "산출물" in c:
                    drop_idx = i
                    in_table = True
                    new_cells = [c for j, c in enumerate(cells) if j != drop_idx]
                    out.append("| " + " | ".join(new_cells) + " |")
                    break
            else:
                out.append(line)
            continue
        # 표 구분선 (|---|---|...) 또는 데이터 행
        if in_table and re.match(r"^\s*\|.+\|\s*$", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if drop_idx is not None and len(cells) > drop_idx:
                new_cells = [c for j, c in enumerate(cells) if j != drop_idx]
                out.append("| " + " | ".join(new_cells) + " |")
            else:
                out.append(line)
        else:
            # 빈 줄 또는 표 외부 — 표 종료
            if in_table and not re.match(r"^\s*\|", line):
                in_table = False
                drop_idx = None
            out.append(line)
    return "\n".join(out)


# ============================================================================
# 6. 표/figure 캡션 자동 넘버링 (LaTeX 처리 위해 markdown 에 미리 표시)
# ============================================================================
def add_caption_numbers(text: str) -> str:
    """모든 figure 와 표에 'Figure N:' / 'Table N:' 자동 캡션 추가.

    Figure: ![alt](path) 직후에 *Figure N: alt* 라인 삽입.
    Table: ## N. 헤더 등 자체 캡션이 있는 경우는 그대로 두고, 익명 표는 위에
           *Table N: 자동 캡션* 삽입.
    """
    fig_counter = [0]
    table_counter = [0]

    def _fig_repl(match):
        alt = match.group(1).strip()
        path = match.group(2)
        fig_counter[0] += 1
        caption = alt if alt else f"Figure"
        return f"![{caption}]({path})\n\n*그림 {fig_counter[0]}. {caption}*\n"

    text = re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", _fig_repl, text)

    # Table: 마크다운 표 헤더 라인 (|...|) 바로 위에 캡션 삽입
    # 단, 직전 라인이 이미 *Table N:* 형태면 skip
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 표 헤더 후보: |...| 행 + 다음 라인이 |---|---|...| 형태
        if (
            re.match(r"^\s*\|.+\|\s*$", line)
            and i + 1 < len(lines)
            and re.match(r"^\s*\|[\s\-:|]+\|\s*$", lines[i + 1])
        ):
            # 직전 라인이 이미 *표 N* 또는 *Table N* 이면 skip
            prev_line = out[-1] if out else ""
            already_captioned = re.search(r"^\*표\s*\d+\.", prev_line.strip())
            if not already_captioned:
                table_counter[0] += 1
                out.append(f"*표 {table_counter[0]}.*")
                out.append("")
            out.append(line)
            i += 1
            continue
        out.append(line)
        i += 1

    return "\n".join(out)


# ============================================================================
# 7. 챕터 빌더 (보고서 본문 구성)
# ============================================================================
def build_title_and_abstract() -> str:
    """수정사항 #6, #7: 표지/제출일 제거 → 목차 위 두 줄 (title page) 삭제.
    YAML frontmatter 의 author/date 도 제거.
    header-includes 는 별도 _header.tex 파일로 분리 (pandoc -H 옵션)."""
    L: list[str] = []
    L.append("---")
    L.append("title: \"Context-Aware xBA: 환경 변수 통합 기대 타율 예측 모델\"")
    L.append("author: \"산업경영공학부 이지현 (2022170832)\"")
    L.append("lang: ko-KR")
    L.append("---")
    L.append("")
    return "\n".join(L)


HEADER_TEX_CONTENT = r"""\usepackage{kotex}
% 한글 폰트 — Apple SD Gothic Neo (macOS 표준, 받침/jongseong 안정)
\setmainhangulfont{Apple SD Gothic Neo}[BoldFont={* Bold}]
\setsanshangulfont{Apple SD Gothic Neo}[BoldFont={* Bold}]
\setmonohangulfont{Apple SD Gothic Neo}
% 영문 폰트 — 시스템 기본 사용
% underscore 자유 사용 (변수명 깨짐 방지)
\usepackage[strings]{underscore}
% 본문 줄간격 확대 — 가독성
\usepackage{setspace}
\onehalfspacing
% 단락 사이 여백 + 들여쓰기 0
\setlength{\parskip}{6pt}
\setlength{\parindent}{0pt}
% 목차 깊이 — N.M 까지만 (사용자 요청: 본문은 더 깊어도 됨)
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{4}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{tabularx}
\usepackage{colortbl}
\usepackage{xcolor}
\usepackage{adjustbox}
% 모든 표에 셀 테두리 강제 (수정사항 #9 — 표 테두리)
\setlength{\arrayrulewidth}{0.4pt}
\renewcommand{\arraystretch}{1.18}
% Figure/Table 캡션 스타일
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=period}
% 표 폭 자동 축소 — Overfull hbox 방지
\usepackage{ragged2e}
% 표 전체 폰트 작게 + 셀 padding 축소
\AtBeginEnvironment{longtable}{\scriptsize\setlength{\tabcolsep}{3pt}}
\AtBeginEnvironment{tabular}{\scriptsize\setlength{\tabcolsep}{3pt}}
% longtable 의 모든 줄을 자동 ragged-right 처리 — 셀 내용이 길어도 줄바꿈
\setlength{\LTpre}{6pt}
\setlength{\LTpost}{6pt}
% Default column type 강화 — 모든 좌측 정렬 컬럼이 자동 줄바꿈하도록
% (pandoc longtable 의 'l' 컬럼은 wrap 안 됨, 'p{...}' 필요)
% → 컴파일 후 manual override 불가능하나, scriptsize 로 대부분 page width 안에 들어감
% 이미지 자동 크기 조정 — figure 페이지당 2-3개 배치되도록 폭 축소
\usepackage{graphicx}
\setkeys{Gin}{width=0.72\linewidth,keepaspectratio}
% figure float 옵션 — 본문 인접 배치 우선, page float 회피
\usepackage{float}
\makeatletter
\renewcommand{\fps@figure}{!htbp}
\makeatother
% 수학 기호 unicode → LaTeX 명령 매핑 (pandoc 이 math mode 를 unicode 로 재변환해도 안전)
\usepackage{newunicodechar}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≠}{\ensuremath{\neq}}
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{ε}{\ensuremath{\varepsilon}}
\newunicodechar{μ}{\ensuremath{\mu}}
\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{γ}{\ensuremath{\gamma}}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{∈}{\ensuremath{\in}}
\newunicodechar{∉}{\ensuremath{\notin}}
\newunicodechar{∞}{\ensuremath{\infty}}
\newunicodechar{±}{\ensuremath{\pm}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{←}{\ensuremath{\leftarrow}}
\newunicodechar{↑}{\ensuremath{\uparrow}}
\newunicodechar{↓}{\ensuremath{\downarrow}}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{÷}{\ensuremath{\div}}
\newunicodechar{²}{\ensuremath{{}^2}}
\newunicodechar{³}{\ensuremath{{}^3}}
\newunicodechar{°}{\ensuremath{^\circ}}
"""


def build_abstract() -> str:
    """수정사항 #8: 초록에서 구체적 수치 스포일러 제거.

    이전: R²=0.3976, Brier=0.13092 등 수치 명시
    수정: 정성적 서술 — "MLB 공식 xBA 대비 명확한 우위", "확률 보정 강화" 등
    """
    L: list[str] = []
    L.append("# 초록 (Abstract) {.unnumbered}")
    L.append("")
    L.append(
        "본 연구는 메이저리그(MLB) 공식 기대 타율(xBA)이 타구의 물리적 질(발사 속도, "
        "발사 각도)만 평가하고 환경 맥락을 무시한다는 구조적 한계를 극복하기 위해, "
        "상황 인지형 기대 타율(ca-xBA, Context-Aware xBA)을 제안하고 그 학술적·실무적 "
        "가치를 통계적으로 검증한다."
    )
    L.append("")
    L.append(
        "Statcast BIP(인플레이 타구) 데이터에 구장 스펙(펜스 거리·높이·고도·지붕) 및 "
        "Open-Meteo 기반 기상 데이터(온도·풍속·풍향·기압·습도·강수·운량·돌풍)를 결합하고, "
        "개폐형/돔형 구장 8종에 대해서는 MLB Stats API의 게임별 지붕 상태(roof status)를 "
        "fetch하여 폐쇄 시 외부 기상 변수를 마스킹하는 도메인 규칙을 적용한다. "
        "비선형 상호작용을 학습할 수 있는 트리 앙상블 모델(Random Forest, XGBoost, "
        "LightGBM)에 RandomizedSearchCV 튜닝과 IsotonicRegression 확률 보정을 결합하고, "
        "오캄의 면도날 원칙에 따라 Stacking과 단일 모델의 통계적 동률 시 더 단순한 "
        "모델을 자동 선정하는 파이프라인을 구현한다."
    )
    L.append("")
    L.append(
        "메인 검증은 2024년 데이터로 학습한 모델을 2025년 격리 데이터에 적용해 산출한 "
        "ca-xBA가 실제 wOBA(BIP 한정 가중 출루율)와 갖는 1:1 상관관계를 MLB 공식 xBA와 "
        "직접 대조하는 방식으로 수행한다. 본 연구의 ca-xBA는 공식 xBA 대비 wOBA 설명력 "
        "(R²)에서 명확한 우위를 보이며, 통산 BABIP을 baseline으로 한 행운 효과 교차 "
        "검증에서도 ca-xBA가 BABIP이 잡지 못하는 환경 보정 신호(dome × weather, "
        "hr_park_effects 등)를 추가로 포착함을 객관적으로 입증한다. 부수적으로 2025 "
        "Silver Slugger Award 수상자 검증에서도 도메인 전문가 평가와의 일관성을 확인하였다."
    )
    L.append("")
    return "\n".join(L)


def build_chapter_1_intro(readme_text: str) -> str:
    """1장 서론 — readme.md 기반.
    수정사항: 1.2 로드맵의 Phase 1~5 H3 헤더를 굵은 단락 헤더로 다운그레이드
    (목차 세분화 회피 — 사용자 요청).
    """
    polished = apply_chapter_polish(readme_text)
    polished = strip_roadmap_column(polished)
    polished = re.sub(r"^# Context-Aware.*$", "", polished, count=1, flags=re.MULTILINE)
    # 1.2 로드맵 Phase 1~5 H3 (### Phase X: ...) → 굵은 단락 텍스트
    # 굵은 단락 뒤에 빈 줄을 넣어, 이어지는 불릿/번호 리스트가 한 단락으로
    # 병합되지 않고 각 항목이 줄바꿈되도록 보장 (사용자 요청: 넘버마다 줄바꿈).
    polished = re.sub(
        r"^### (Phase \d+: [^\n]+)$",
        r"\n**\1**\n",
        polished,
        flags=re.MULTILINE,
    )
    shifted = shift_headers(polished, 0)
    L: list[str] = []
    L.append("# 서론")
    L.append("")
    L.append(shifted.lstrip())
    return "\n".join(L)


def build_chapter_2_data(phase1_text: str, phase2_text: str) -> str:
    """2장 데이터 — Phase 1 + Phase 2 통합. 데이터셋 설명 신규 § 포함."""
    L: list[str] = []
    L.append("# 데이터 전처리 및 탐색적 분석")
    L.append("")
    # §2.1 데이터셋 설명 — 신규 작성 (사용자 요청)
    L.append("## 데이터셋 설명")
    L.append("")
    L.append(
        "본 연구는 세 종류의 외부 데이터를 통합하여 단일 학습용 테이블을 구성한다. "
        "각 데이터 소스의 핵심 변수와 행 단위를 아래에 정리한다."
    )
    L.append("")
    L.append("### Statcast 타구 단위 데이터 (Baseball Savant)")
    L.append("")
    L.append(
        "MLB 공식 트래킹 시스템이 기록한 모든 투구·타구의 물리 메타데이터다. 2024-2025 두 시즌, "
        "총 1,443,801 개의 투구(pitch) 행으로 시작하여, BIP(인플레이 타구)만 필터링한 후 "
        "약 225,414 개의 타구가 본 분석의 모델 입력 단위가 된다. 원천 데이터는 약 118 개의 컬럼을 "
        "포함하며, 본 모델이 의미 있게 활용한 주요 변수군은 다음과 같다."
    )
    L.append("")
    L.append("| 변수군 | 대표 컬럼 | 의미 |")
    L.append("|---|---|---|")
    L.append("| 타구 물리 (핵심) | `launch_speed`, `launch_angle` | 발사 속도 (mph) 와 발사 각도 (도). MLB 공식 xBA 의 두 입력 변수이며 본 모델의 X_base 다. |")
    L.append("| 배트 트래킹 | `bat_speed`, `swing_length`, `attack_angle`, `attack_direction`, `swing_path_tilt`, `intercept_ball_minus_batter_pos_*` | 배트의 속도/궤적/타격 시점 위치. 2024 후반 도입된 신규 트래킹이라 결측이 존재한다. |")
    L.append("| 투구 물리 | `release_speed`, `release_pos_x/z`, `pfx_x/z`, `plate_x/z`, `release_spin_rate`, `release_extension`, `spin_axis`, `effective_speed`, `api_break_*`, `arm_angle` | 투구 릴리스/휘어짐/홈플레이트 도달 위치. 타격 직전 타자가 마주한 투구 조건을 기술한다. |")
    L.append("| 타석 상황 | `balls`, `strikes`, `outs_when_up`, `inning`, `age_pit`, `age_bat`, `n_thruorder_pitcher`, `n_priorpa_thisgame_player_at_bat` | 볼카운트/아웃카운트/이닝/타순 등 게임 상황 변수다. |")
    L.append("| 카테고리 식별 | `stand` (좌/우 타자), `p_throws` (좌/우 투수), `pitch_type` (FF, SL, CH 등), `if_fielding_alignment`, `of_fielding_alignment` | 좌/우 binary 인코딩 + 그 외는 one-hot 처리. |")
    L.append("| 타석 결과 (target/derived) | `events` (single/double/triple/home_run/field_out 등), `bb_type` (ground_ball/fly_ball/line_drive/popup), `babip_value` | `is_hit` (단순 안타 여부) target 라벨로 가공한다. `babip_value` 는 Phase 5 BABIP 계산에 사용한다. |")
    L.append("| 선수 ID | `batter`, `pitcher` | MLBAM ID. Phase 5 외부 CSV (`validation_2025_gt.csv`) 와 hard join 키다. |")
    L.append("")
    L.append("### 구장 스펙 데이터 (`ballparks.csv`)")
    L.append("")
    L.append(
        "MLB 30 개 구장 각각의 물리적 특성을 정리한 정적 테이블이다. 행 단위는 **구장 1개당 1행** "
        "(home_team abbreviation 기준 29 행 — Athletics 2024-2025 이전 이슈로 분석에서 제외, Phase 1 참조). "
        "주요 컬럼은 다음과 같다."
    )
    L.append("")
    L.append("| 컬럼 | 단위 | 의미 |")
    L.append("|---|---|---|")
    L.append("| `home_team` | abbr | 구장 식별자 (NYY, BOS, COL 등). |")
    L.append("| `left_field`, `center_field`, `right_field` | feet | 좌/중/우측 펜스까지의 거리. |")
    L.append("| `min_wall_height`, `max_wall_height` | feet | 펜스의 최저/최고 높이. |")
    L.append("| `hr_park_effects` | index | 구장별 홈런 친화도 (100 = 리그 평균, Coors Field 등 고지대 구장이 높다). |")
    L.append("| `extra_distance` | feet | 구장 형태 보정 거리. |")
    L.append("| `elevation` | feet | 구장 해발고도 (Coors Field 5,200 ft 등 — 공기 밀도와 타구 비거리에 영향). |")
    L.append("| `roof` | 0~1 | 지붕 형태: 0 (open), 0.5 (retractable), 1 (dome). |")
    L.append("| `daytime` | 0~1 | 주간 경기 비율 (Wrigley Field 등). |")
    L.append("| 위·경도 | deg | 기상 API 좌표 매칭용. |")
    L.append("")
    L.append("### Open-Meteo 기상 데이터 (Historical Weather API)")
    L.append("")
    L.append(
        "각 경기의 홈구장 위·경도와 경기 시작 시각 (낮 경기 13시, 야간 경기 19시 — `daytime` 컬럼 기준) "
        "을 키로 [Open-Meteo Historical Weather API](https://archive-api.open-meteo.com/v1/archive) 에 "
        "쿼리하여 시간 단위 기상값을 fetch 한다. 캐시 디렉토리에 저장하여 재실행 시 추가 호출을 방지한다."
    )
    L.append("")
    L.append("| 컬럼 | 단위 | 의미 |")
    L.append("|---|---|---|")
    L.append("| `wx_temperature_2m` | 섭씨 | 지상 2m 기온 (공기 밀도와 타구 비거리에 영향). |")
    L.append("| `wx_relative_humidity_2m` | % | 상대 습도 (공기 밀도에 영향). |")
    L.append("| `wx_surface_pressure` | hPa | 지표 기압 (해발 고도와 결합한 공기 밀도를 결정). |")
    L.append("| `wx_wind_speed_10m`, `wx_wind_gusts_10m` | km/h | 지상 10m 풍속/돌풍. |")
    L.append("| `wx_wind_direction_10m` | 도 | 풍향 (외야 방향일 경우 비거리 증감 영향). |")
    L.append("| `wx_precipitation` | mm | 강수량 (공이 미끄러지고 야수 수비 난이도가 증가). |")
    L.append("| `wx_cloud_cover` | % | 운량 (햇빛 시야 영향). |")
    L.append("")
    L.append(
        "**돔 마스킹 (도메인 규칙)**: 개폐형/돔형 구장 8 종 (SEA, TOR, MIL, TEX, AZ, MIA, HOU, TB) 의 "
        "각 경기에 대해 [MLB Stats API](https://statsapi.mlb.com/api/v1/people) 의 `gameData.weather.condition` "
        "필드를 fetch 하여 `Roof Closed` / `Dome` 인 경기는 외부 기상 5 종 (wind speed/gusts/direction, "
        "precipitation, cloud cover) = 0, 실내 공조 표준값 (기온 22 섭씨, 습도 50%, 기압은 실내 ~= 실외) "
        "으로 마스킹한다. 이는 모델이 \"돔 닫힘 = 외부 기상 무의미\" 시그널을 데이터 자체에서 "
        "학습하도록 만들어, weather x roof 비선형 상호작용의 오학습을 원천 차단한다."
    )
    L.append("")
    L.append("## 전처리 파이프라인 (Phase 1)")
    L.append("")
    polished1 = apply_chapter_polish(phase1_text)
    polished1 = strip_phase1_intro_blockquote(polished1)
    shifted1 = shift_headers(polished1, 1)  # phase1 의 ## 를 ### subsubsection 으로
    L.append(shifted1.lstrip())
    L.append("")
    L.append("## 탐색적 분석 및 Feature Selection (Phase 2)")
    L.append("")
    polished2 = apply_chapter_polish(phase2_text)
    shifted2 = shift_headers(polished2, 1)
    L.append(shifted2.lstrip())
    return "\n".join(L)


def build_chapter_3_ablation(phase3_text: str) -> str:
    L: list[str] = []
    L.append("# 효과 분리 실험 (Ablation Study)")
    L.append("")
    L.append("## 평가 지표 Brier Score 정의")
    L.append("")
    L.append(
        "본 장 이후 모든 모델 평가에서 핵심으로 사용되는 **Brier Score** 는 이진 분류 모델이 "
        "산출한 예측 확률의 정상도 (calibration) 를 측정하는 지표다. 정의는 다음과 같다."
    )
    L.append("")
    L.append("$$ \\mathrm{Brier} = \\frac{1}{N}\\sum_{i=1}^{N} (y_i - p_i)^2 $$")
    L.append("")
    L.append(
        "여기서 $y_i \\in \\{0, 1\\}$ 은 실제 안타 여부, $p_i \\in [0, 1]$ 은 모델이 예측한 안타 확률이다. "
        "값이 **낮을수록 우수**하며 (예측 확률과 실제 결과의 평균 제곱 오차), 단순 분류 정확도가 "
        "아닌 **확률값 자체의 정확성**을 평가한다. 본 연구의 ca-xBA 는 시즌 단위로 평균한 확률값을 "
        "직접 산출물로 사용하므로, Brier Score 가 가장 중요한 단일 평가 지표가 된다."
    )
    L.append("")
    L.append("## 실험 설계 및 결과")
    L.append("")
    polished = apply_chapter_polish(phase3_text)
    shifted = shift_headers(polished, 1)  # phase3 의 ## 를 ### 로 다운 (목차에 안 보임)
    L.append(shifted.lstrip())
    return "\n".join(L)


def strip_visualization_section_headers(text: str) -> str:
    """시각화 § 헤더만 제거 (내부 figure/캡션/해석은 본문 inline 으로 보존).

    사용자 요청: "시각화 라는 목차를 따로 두지 말고 시각 자료를 본문 중간중간
    적재적소에 배치할 것"

    제거 대상: `^#+ \d*\.?\s*시각화`, `^#+ \d+\.\d+\s*시각화`
    내부 figure (`![...](...)`), 캡션 (`*그림 N. ...*`), 해석 텍스트는 그대로 둠.
    """
    text = re.sub(
        r"^#+\s+(?:\d+(?:\.\d+)*\.?\s*)?시각화\b.*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text


def build_chapter_4_tuning(phase4_text: str) -> str:
    L: list[str] = []
    L.append("# Advanced Model 튜닝 및 확률 보정")
    L.append("")
    L.append("## 튜닝 파이프라인 및 결과")
    L.append("")
    polished = apply_chapter_polish(phase4_text)
    shifted = shift_headers(polished, 1)  # phase4 의 ## 를 ### 로 다운 (목차에 안 보임)
    L.append(shifted.lstrip())
    return "\n".join(L)


def build_chapter_5_validation(phase5_text: str) -> str:
    L: list[str] = []
    L.append("# 최종 지표 산출 및 세이버메트릭스 가치 검증")
    L.append("")
    L.append("## 검증 파이프라인 및 결과")
    L.append("")
    polished = apply_chapter_polish(phase5_text)
    shifted = shift_headers(polished, 1)  # phase5 의 ## 를 ### 로 다운 (목차에 안 보임)
    L.append(shifted.lstrip())
    return "\n".join(L)


def build_chapter_6_conclusion() -> str:
    L: list[str] = []
    L.append("# 결론 및 시사점")
    L.append("")
    L.append("## 연구 성과 종합")
    L.append("")
    L.append(
        "본 연구는 MLB 공식 기대 타율(xBA)의 환경 무시 한계를 극복하는 ca-xBA "
        "(Context-Aware xBA)를 제안하고, 데이터마이닝 정통의 5 Phase 로드맵을 통해 "
        "그 학술적·실무적 가치를 통계적으로 검증하였다. 구체적으로 (i) 도메인 기반 "
        "전처리(돔 구장 게임별 지붕 상태 마스킹), (ii) 보수적 Feature Selection "
        "(RF importance + Mutual Information의 4-criterion 합의 규칙), "
        "(iii) 2-way ANOVA를 통한 데이터·알고리즘 비선형 상호작용의 통계적 입증, "
        "(iv) 오캄의 면도날 자동 선정 로직을 통한 단순성과 성능의 동시 확보, "
        "(v) 통산 BABIP 대비 시즌 편차를 활용한 도메인 정통 행운 효과 교차 검증을 "
        "단일 파이프라인 안에서 일관되게 수행하였다."
    )
    L.append("")
    L.append("## 학술적 기여")
    L.append("")
    L.append(
        "첫째, Phase 3의 2x2 Factorial Ablation은 \"환경 변수의 가치는 비선형 모델 "
        "위에서만 발현된다\"는 명제를 interaction term의 통계적 유의성(p < 0.05)으로 "
        "직접 입증하였다. 이는 단순한 변수 추가가 아닌 알고리즘과 데이터의 결합 효과가 "
        "ca-xBA 우위의 본질임을 보여준다. 둘째, Phase 4의 cv='prefit' 패턴 Isotonic "
        "Calibration은 표준 CalibratedClassifierCV 대비 약 7,000배 연산 단축을 달성하면서 "
        "학술적 동등성을 유지하여, 대규모 데이터 환경의 실용적 calibration 방법론을 "
        "제시한다. 셋째, 오캄의 면도날 자동 선정 로직은 \"성능이 비슷하면 단순한 모델이 "
        "낫다\"는 원칙을 정량 기준(ε = 0.001)으로 코드화하여, 복잡한 앙상블이 단일 모델의 "
        "native calibration을 훼손할 수 있다는 관찰을 학술적으로 정당화한다."
    )
    L.append("")
    L.append("## 실무적 시사점")
    L.append("")
    L.append(
        "Phase 5의 BABIP 교차 검증은 ca-xBA가 야구 도메인의 정통 행운 지표인 통산 BABIP "
        "대비 시즌 편차와 동일한 방향의 신호를 잡으면서도, BABIP이 포착하지 못하는 환경/"
        "quality 보정 신호(dome × weather 상호작용, 구장 펜스 거리, hr_park_effects 등)를 "
        "추가로 단독 포착함을 확인하였다. 특히 Mike Trout 패턴 — 통산 baseline 평균 수준의 "
        "BABIP을 유지함에도 ca-xBA 기반 luck이 극불운으로 평가되는 사례 — 은 Front Office "
        "의 저평가 선수 발굴 도구로서 ca-xBA가 BABIP 단독 분석을 보완할 수 있는 "
        "실무적 가치를 입증한다. 동시에 Kyle Schwarber 패턴(fly ball power hitter의 "
        "구조적 편향)을 정직하게 명시함으로써 모델의 한계 또한 투명하게 공개하였다."
    )
    L.append("")
    L.append("## 한계 및 향후 작업")
    L.append("")
    L.append(
        "본 연구는 단일 시즌(2025) 외부 검증에 의존하므로 모델의 시간 일반화 능력은 "
        "다년치 검증으로 추가 입증이 필요하다. 또한 Phase 5 실버 슬러거 검증의 일부 "
        "선수가 Statcast와 MLB Stats API 간 다국어 선수명 표기 불일치로 누락된 점은 "
        "Chadwick Register의 ID 크로스워크를 도입하여 향후 해소 가능하다. 마지막으로 "
        "ca-xBA의 BIP-한정 quality 평가 특성상 fly ball power hitter에 대한 구조적 편향은 "
        "외부 지표(BABIP, xwOBA underperform)와의 교차 검증 또는 HR weighted 변형 모델 "
        "도입으로 보완할 수 있다."
    )
    L.append("")
    L.append(
        "특히 Phase 5에서 관찰된 Kyle Schwarber 패턴(fly ball 거포의 구조적 저평가)은 "
        "단순한 경향성이 아니라, 예측하려는 타겟 변수 $y$의 수리적 정의에서 비롯된 "
        "구조적(structural) 한계다. 본 모델의 타겟 변수 $y$는 안타(1)와 아웃(0)만을 "
        "구분할 뿐, 타구의 실질적 가치(장타 가중치)를 내포하지 않는다. 통계적으로 "
        "타자의 인플레이 타구 기대 생산력 $E[\\mathrm{wOBA} \\mid X]$는 확률과 조건부 "
        "기댓값의 곱으로 분해할 수 있다."
    )
    L.append("")
    L.append(
        "$$ E[\\mathrm{wOBA} \\mid X] = P(\\mathrm{Hit} \\mid X) \\cdot "
        "E[\\mathrm{HitValue} \\mid \\mathrm{Hit}, X] $$"
    )
    L.append("")
    L.append(
        "현재의 ca-xBA는 위 식에서 첫 번째 항인 $P(\\mathrm{Hit} \\mid X)$의 정밀한 "
        "추정에만 집중한 지표다. 따라서 발사각이 높아 아웃될 확률이 크지만 일단 안타가 "
        "되면 홈런(가장 높은 HitValue)이 되는 타구의 가치는 분자 누적에서 과소평가된다. "
        "향후 연구에서 안타 발생 여부를 예측하는 분류기(classifier)와, 안타 발생을 전제로 "
        "루타수(wOBA weight)를 예측하는 회귀기(regressor)를 결합한 '허들 모델(Hurdle "
        "Model)' 구조를 도입한다면, 이 구조적 편향을 수학적으로 해소할 수 있을 것이다."
    )
    L.append("")
    return "\n".join(L)


def build_appendix() -> str:
    """부록 — 본문에서 부록 A~D 로 이동시킨 재현성/세부 통계 표 모음.

    수정사항(보고서 가독성): 흐름을 깨는 방대한 표를 본문에서 부록으로 이동.
      - 부록 A: 다중공선성 고상관 변수 쌍 전체 목록 (phase2 §5, 구 표 10)
      - 부록 B: 결측치 대체 중앙값 (phase2 §3, 구 표 9)
      - 부록 C: 효과 분리 실험 세부 통계 — fold-level mean±SD (phase3 §3.2, 구 표 16)
                + 4개 메트릭 2-way ANOVA (phase3 §5, 구 표 20~23)
      - 부록 D: Advanced 모델별 최종 하이퍼파라미터 (phase4 §2, 구 표 24)
                + Outer 5-fold CV fold mean±SD (phase4 §3, 구 표 26)
    """
    L: list[str] = []
    # ---- 부록 A : 다중공선성 고상관 쌍 -------------------------------------
    L.append("# 부록 A. 다중공선성 분석 — 고상관 변수 쌍 전체 목록 {.unnumbered}")
    L.append("")
    L.append(
        "Phase 2의 다중공선성 분석(Pearson $|r| > 0.95$)에서 식별된 24건의 고상관 "
        "변수 쌍을 |r| 내림차순으로 정리한다. `var_a`/`var_b` 는 RobustScaler 적용 후 "
        "분산이며, 제거 규칙은 X_BASE 보존 → derived 변수 우선 drop → variance fallback "
        "순서를 따른다."
    )
    L.append("")
    L.append("| 변수 A | 변수 B | abs(r) | var_a | var_b | 제거 | 규칙 |")
    L.append("|---|---|---:|---:|---:|---|---|")
    L.append("| `bat_speed_is_missing` | `swing_length_is_missing` | 1.000 | 0.089 | 0.089 | `swing_length_is_missing` | variance fallback |")
    L.append("| `bat_speed_is_missing` | `attack_angle_is_missing` | 1.000 | 0.089 | 0.089 | `bat_speed_is_missing` | variance fallback |")
    L.append("| `swing_length_is_missing` | `attack_angle_is_missing` | 1.000 | 0.089 | 0.089 | `swing_length_is_missing` | variance fallback |")
    L.append("| `bat_speed_is_missing` | `attack_direction_is_missing` | 1.000 | 0.089 | 0.089 | `bat_speed_is_missing` | variance fallback |")
    L.append("| `swing_length_is_missing` | `attack_direction_is_missing` | 1.000 | 0.089 | 0.089 | `swing_length_is_missing` | variance fallback |")
    L.append("| `attack_angle_is_missing` | `attack_direction_is_missing` | 1.000 | 0.089 | 0.089 | `attack_direction_is_missing` | variance fallback |")
    L.append("| `intercept_ball_minus_batter_pos_x_inches_is_missing` | `intercept_ball_minus_batter_pos_y_inches_is_missing` | 1.000 | 0.089 | 0.089 | `intercept_ball_minus_batter_pos_y_inches_is_missing` | variance fallback |")
    L.append("| `if_fielding_alignment_UNK` | `of_fielding_alignment_UNK` | 1.000 | 0.007 | 0.007 | `of_fielding_alignment_UNK` | variance fallback |")
    L.append("| `bat_speed_is_missing` | `swing_path_tilt_is_missing` | 1.000 | 0.089 | 0.089 | `bat_speed_is_missing` | variance fallback |")
    L.append("| `swing_length_is_missing` | `swing_path_tilt_is_missing` | 1.000 | 0.089 | 0.089 | `swing_length_is_missing` | variance fallback |")
    L.append("| `attack_angle_is_missing` | `swing_path_tilt_is_missing` | 1.000 | 0.089 | 0.089 | `attack_angle_is_missing` | variance fallback |")
    L.append("| `attack_direction_is_missing` | `swing_path_tilt_is_missing` | 1.000 | 0.089 | 0.089 | `attack_direction_is_missing` | variance fallback |")
    L.append("| `bat_speed_is_missing` | `intercept_ball_minus_batter_pos_x_inches_is_missing` | 1.000 | 0.089 | 0.089 | `bat_speed_is_missing` | variance fallback |")
    L.append("| `swing_length_is_missing` | `intercept_ball_minus_batter_pos_x_inches_is_missing` | 1.000 | 0.089 | 0.089 | `swing_length_is_missing` | variance fallback |")
    L.append("| `attack_angle_is_missing` | `intercept_ball_minus_batter_pos_x_inches_is_missing` | 1.000 | 0.089 | 0.089 | `attack_angle_is_missing` | variance fallback |")
    L.append("| `attack_direction_is_missing` | `intercept_ball_minus_batter_pos_x_inches_is_missing` | 1.000 | 0.089 | 0.089 | `attack_direction_is_missing` | variance fallback |")
    L.append("| `bat_speed_is_missing` | `intercept_ball_minus_batter_pos_y_inches_is_missing` | 1.000 | 0.089 | 0.089 | `bat_speed_is_missing` | variance fallback |")
    L.append("| `swing_length_is_missing` | `intercept_ball_minus_batter_pos_y_inches_is_missing` | 1.000 | 0.089 | 0.089 | `swing_length_is_missing` | variance fallback |")
    L.append("| `attack_angle_is_missing` | `intercept_ball_minus_batter_pos_y_inches_is_missing` | 1.000 | 0.089 | 0.089 | `attack_angle_is_missing` | variance fallback |")
    L.append("| `attack_direction_is_missing` | `intercept_ball_minus_batter_pos_y_inches_is_missing` | 1.000 | 0.089 | 0.089 | `attack_direction_is_missing` | variance fallback |")
    L.append("| `swing_path_tilt_is_missing` | `intercept_ball_minus_batter_pos_x_inches_is_missing` | 1.000 | 0.089 | 0.089 | `swing_path_tilt_is_missing` | variance fallback |")
    L.append("| `swing_path_tilt_is_missing` | `intercept_ball_minus_batter_pos_y_inches_is_missing` | 1.000 | 0.089 | 0.089 | `swing_path_tilt_is_missing` | variance fallback |")
    L.append("| `release_speed` | `effective_speed` | 0.990 | 0.454 | 0.479 | `effective_speed` | derived drop |")
    L.append("| `elevation` | `wx_surface_pressure` | 0.985 | 2.874 | 2.039 | `wx_surface_pressure` | variance fallback |")
    L.append("")
    # ---- 부록 B : 결측치 대체 중앙값 ---------------------------------------
    L.append("# 부록 B. 결측치 대체 중앙값 (2024 median) {.unnumbered}")
    L.append("")
    L.append(
        "Phase 2에서 결측 imputation을 적용한 13개 numeric 컬럼의 2024 전체 중앙값이다. "
        "각 변수의 `*_is_missing` 플래그는 별도로 보존하여 결측 패턴 자체를 신호로 활용한다."
    )
    L.append("")
    L.append("| 컬럼 | 2024 Median |")
    L.append("|---|---:|")
    L.append("| `bat_speed` | 71.6000 |")
    L.append("| `swing_length` | 7.2000 |")
    L.append("| `attack_angle` | 8.7646 |")
    L.append("| `attack_direction` | 0.7896 |")
    L.append("| `swing_path_tilt` | 32.2398 |")
    L.append("| `intercept_ball_minus_batter_pos_x_inches` | 37.1243 |")
    L.append("| `intercept_ball_minus_batter_pos_y_inches` | 29.5637 |")
    L.append("| `release_spin_rate` | 2263.0000 |")
    L.append("| `release_extension` | 6.5000 |")
    L.append("| `spin_axis` | 201.0000 |")
    L.append("| `effective_speed` | 90.6000 |")
    L.append("| `api_break_z_with_gravity` | 2.2200 |")
    L.append("| `arm_angle` | 39.2000 |")
    L.append("")
    # ---- 부록 C : 효과 분리 실험 세부 통계 ---------------------------------
    L.append("# 부록 C. 효과 분리 실험 세부 통계 (Phase 3) {.unnumbered}")
    L.append("")
    L.append("## fold-level mean ± SD (across 5 folds) {.unnumbered}")
    L.append("")
    L.append(
        "2x2 Factorial Ablation 4개 모델(M1~M4)의 5-fold 메트릭 평균과 표준편차다. "
        "fold 간 변동이 매우 작아 OOF aggregate 결과가 안정적임을 뒷받침한다."
    )
    L.append("")
    L.append("| Model | Brier mean±SD | LogLoss mean±SD | F1 mean±SD | AUC mean±SD |")
    L.append("|---|---:|---:|---:|---:|")
    L.append("| M1 | 0.21033±0.00052 | 0.61347±0.00110 | 0.1700±0.0054 | 0.6670±0.0034 |")
    L.append("| M2 | 0.14012±0.00146 | 0.43198±0.00354 | 0.6758±0.0029 | 0.8594±0.0028 |")
    L.append("| M3 | 0.20937±0.00063 | 0.61078±0.00142 | 0.2357±0.0072 | 0.6662±0.0039 |")
    L.append("| M4 | 0.13589±0.00088 | 0.42049±0.00213 | 0.6924±0.0028 | 0.8691±0.0015 |")
    L.append("")
    L.append("## 2-way ANOVA (Type II SS) — 4개 메트릭 {.unnumbered}")
    L.append("")
    L.append(
        "각 fold(n=5)의 메트릭을 종속변수로, Data(X_base/X_advanced) × Algo(LogReg/XGB)를 "
        "요인으로 한 Type II SS ANOVA 결과다. 모든 메트릭에서 상호작용 항 `C(data):C(algo)` "
        "이 통계적으로 유의하다(p < 0.05)."
    )
    L.append("")
    L.append("**Brier:**")
    L.append("")
    L.append("| Source | SS | df | F | p |")
    L.append("|---|---:|---:|---:|---:|")
    L.append("| C(data) | 0.000034 | 1 | 30.226 | 4.866e-05 |")
    L.append("| C(algo) | 0.025810 | 1 | 23134.984 | 1.022e-26 |")
    L.append("| C(data):C(algo) | 0.000013 | 1 | 11.977 | 0.00322 |")
    L.append("| Residual | 0.000018 | 16 | n/a | n/a |")
    L.append("")
    L.append("**LogLoss:**")
    L.append("")
    L.append("| Source | SS | df | F | p |")
    L.append("|---|---:|---:|---:|---:|")
    L.append("| C(data) | 0.000251 | 1 | 39.606 | 1.07e-05 |")
    L.append("| C(algo) | 0.172764 | 1 | 27232.030 | 2.776e-27 |")
    L.append("| C(data):C(algo) | 0.000097 | 1 | 15.237 | 0.001264 |")
    L.append("| Residual | 0.000102 | 16 | n/a | n/a |")
    L.append("")
    L.append("**ROC AUC:**")
    L.append("")
    L.append("| Source | SS | df | F | p |")
    L.append("|---|---:|---:|---:|---:|")
    L.append("| C(data) | 0.000097 | 1 | 8.465 | 0.01024 |")
    L.append("| C(algo) | 0.195359 | 1 | 17051.056 | 1.172e-25 |")
    L.append("| C(data):C(algo) | 0.000138 | 1 | 12.003 | 0.003194 |")
    L.append("| Residual | 0.000183 | 16 | n/a | n/a |")
    L.append("")
    L.append("**F1:**")
    L.append("")
    L.append("| Source | SS | df | F | p |")
    L.append("|---|---:|---:|---:|---:|")
    L.append("| C(data) | 0.008475 | 1 | 277.510 | 1.57e-11 |")
    L.append("| C(algo) | 1.157952 | 1 | 37918.715 | 1.967e-28 |")
    L.append("| C(data):C(algo) | 0.003023 | 1 | 98.979 | 2.95e-08 |")
    L.append("| Residual | 0.000489 | 16 | n/a | n/a |")
    L.append("")
    # ---- 부록 D : Advanced 모델 튜닝 스펙 ----------------------------------
    L.append("# 부록 D. Advanced 모델 튜닝 스펙 (Phase 4) {.unnumbered}")
    L.append("")
    L.append("## 모델별 최종 하이퍼파라미터 (RandomizedSearchCV best params) {.unnumbered}")
    L.append("")
    L.append(
        "각 base 모델의 RandomizedSearchCV(n_iter=30, inner_cv=5, "
        "scoring='neg_brier_score', refit=True) 결과로 선정된 최종 하이퍼파라미터 전체 "
        "딕셔너리다."
    )
    L.append("")
    L.append("| Base | best params |")
    L.append("|---|---|")
    L.append("| RF | `bootstrap`=True, `ccp_alpha`=0.0, `class_weight`=None, `criterion`=entropy, `max_depth`=None, `max_features`=0.5, `max_leaf_nodes`=None, `max_samples`=None, `min_impurity_decrease`=0.0, `min_samples_leaf`=4, `min_samples_split`=4, `min_weight_fraction_leaf`=0.0, `monotonic_cst`=None, `n_estimators`=500, `n_jobs`=1, `oob_score`=False, `random_state`=42, `verbose`=0, `warm_start`=False |")
    L.append("| XGB | `objective`=binary:logistic, `colsample_bytree`=0.9, `eval_metric`=logloss, `gamma`=0, `learning_rate`=0.03, `max_depth`=8, `min_child_weight`=5, `n_estimators`=200, `n_jobs`=1, `random_state`=42, `subsample`=0.8, `tree_method`=hist, `verbosity`=0 |")
    L.append("| LGBM | `boosting_type`=gbdt, `class_weight`=None, `colsample_bytree`=1.0, `importance_type`=split, `learning_rate`=0.03, `max_depth`=-1, `min_child_samples`=20, `min_child_weight`=0.001, `min_split_gain`=0.0, `n_estimators`=200, `n_jobs`=1, `num_leaves`=127, `random_state`=42, `reg_alpha`=0.0, `reg_lambda`=0.0, `subsample`=0.9, `subsample_for_bin`=200000, `subsample_freq`=0, `verbose`=-1 |")
    L.append("")
    L.append("## Outer 5-fold CV — fold mean ± SD {.unnumbered}")
    L.append("")
    L.append(
        "6개 후보 모델의 Outer 5-fold CV fold 평균과 표준편차다. fold 간 변동성(Brier "
        "기준 0.0012 이하)이 후보 간 차이보다 작아 오캄의 면도날 자동 선정(ε=0.001)의 "
        "근거가 된다."
    )
    L.append("")
    L.append("| Model | Brier mean±SD | LogLoss mean±SD | F1 mean±SD | AUC mean±SD |")
    L.append("|---|---:|---:|---:|---:|")
    L.append("| RF (tuned) | 0.13264±0.00117 | 0.41259±0.00303 | 0.6978±0.0031 | 0.8759±0.0022 |")
    L.append("| XGB (tuned) | 0.13231±0.00106 | 0.41135±0.00267 | 0.6975±0.0036 | 0.8761±0.0020 |")
    L.append("| LGBM (tuned) | 0.13108±0.00104 | 0.40753±0.00275 | 0.6985±0.0027 | 0.8777±0.0020 |")
    L.append("| Stacking (LR meta) | 0.13244±0.00124 | 0.41409±0.00312 | 0.7012±0.0025 | 0.8780±0.0021 |")
    L.append("| Stacking + Isotonic | 0.13083±0.00107 | 0.40587±0.00315 | 0.6937±0.0051 | 0.8780±0.0021 |")
    L.append("| LGBM + Isotonic | 0.13092±0.00100 | 0.40600±0.00284 | 0.7029±0.0040 | 0.8776±0.0020 |")
    L.append("")
    return "\n".join(L)


def build_chapter_7_references() -> str:
    L: list[str] = []
    L.append("# 참고문헌")
    L.append("")
    L.append(
        "본 보고서에서 인용 및 활용한 외부 데이터 소스, API, 라이브러리를 분야별로 "
        "정리한다."
    )
    L.append("")
    L.append("## 데이터 소스 및 API")
    L.append("")
    L.append("1. MLB Statcast 타구 데이터 (Baseball Savant). https://baseballsavant.mlb.com/")
    L.append("2. Open-Meteo Historical Weather API. https://archive-api.open-meteo.com/v1/archive")
    L.append("3. MLB Stats API (포지션 및 통산 hitting 통계). https://statsapi.mlb.com/api/v1/people")
    L.append("4. Baseball Savant Custom Leaderboard — Expected Statistics")
    L.append("5. MLB.com — 2025 Silver Slugger Award Winners. https://www.mlb.com/news/2025-silver-slugger-award-winners")
    L.append("")
    L.append("## 라이브러리 및 도구")
    L.append("")
    L.append("- Python 3.12, pandas 3.0, scikit-learn 1.8, XGBoost 3.2, LightGBM 4.6")
    L.append("- imbalanced-learn 0.14 (RandomUnderSampler, SMOTE)")
    L.append("- statsmodels 0.14 (2-way ANOVA)")
    L.append("- matplotlib 3.10, seaborn 0.13")
    L.append("- pypandoc 1.17 + pandoc 3.5, tectonic 0.16 (LaTeX engine)")
    L.append("")
    return "\n".join(L)


# ============================================================================
# 8. main
# ============================================================================
def main():
    log("=" * 80)
    log("ca-xBA 최종 보고서 (PDF, LaTeX engine) 자동 컴파일")
    log("=" * 80)

    log("\n[1/6] 마크다운 파일 로드 ...")
    readme = README_MD.read_text(encoding="utf-8")
    phase1 = PHASE1_MD.read_text(encoding="utf-8")
    phase2 = PHASE2_MD.read_text(encoding="utf-8")
    phase3 = PHASE3_MD.read_text(encoding="utf-8")
    phase4 = PHASE4_MD.read_text(encoding="utf-8")
    phase5 = PHASE5_MD.read_text(encoding="utf-8")
    log(f"  readme: {len(readme):,d}자  phase1: {len(phase1):,d}자  phase2: {len(phase2):,d}자")
    log(f"  phase3: {len(phase3):,d}자  phase4: {len(phase4):,d}자  phase5: {len(phase5):,d}자")

    log("\n[2/6] 챕터 본문 구성 ...")
    parts: list[str] = []
    parts.append(build_title_and_abstract())
    parts.append(build_abstract())
    parts.append(build_chapter_1_intro(readme))
    parts.append(build_chapter_2_data(phase1, phase2))
    parts.append(build_chapter_3_ablation(phase3))
    parts.append(build_chapter_4_tuning(phase4))
    parts.append(build_chapter_5_validation(phase5))
    parts.append(build_chapter_6_conclusion())
    parts.append(build_chapter_7_references())
    parts.append(build_appendix())
    merged_md = "\n\n".join(parts)
    log(f"  병합된 마크다운: {len(merged_md):,d}자, {merged_md.count(chr(10)):,d}줄")

    log("\n[3/6] 후처리 — 문체 통일 + 이미지 경로 + 캡션 넘버링 + 수학 기호 ...")
    merged_md = unify_style(merged_md)
    merged_md = fix_image_paths(merged_md)
    merged_md = add_caption_numbers(merged_md)
    merged_md = strip_emojis(merged_md)
    merged_md = convert_unicode_math(merged_md)  # LaTeX 호환 수학 기호 변환
    log(f"  변환 후 마크다운: {len(merged_md):,d}자")

    DEBUG_MD.write_text(merged_md, encoding="utf-8")
    log(f"  디버그용 병합 md: {DEBUG_MD.name}")

    log("\n[4/6] pandoc → LaTeX → tectonic 컴파일 ...")
    # header.tex 작성 (LaTeX 패키지 + 표 테두리 + 캡션 스타일)
    header_tex = ROOT / "_header.tex"
    header_tex.write_text(HEADER_TEX_CONTENT, encoding="utf-8")
    log(f"  LaTeX header: {header_tex.name}")

    common_args = [
        "--toc",
        "--number-sections",
        "--standalone",
        "-H", str(header_tex),
        "-V", "geometry:margin=2.3cm",
        "-V", "fontsize=11.5pt",
        "-V", "lang=ko-KR",
        "-V", "toc-title=목차",
        "-V", "documentclass=article",
        "-V", "linkcolor=black",
        "-V", "urlcolor=blue",
    ]
    # 항상 영문 임시 디렉토리에서 컴파일 (한글 경로 회피)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tex_path = TMP_ROOT / "merged_report.tex"
    # header.tex 도 임시 디렉토리로 복사 (-H 가 절대 경로 받음)
    pypandoc.convert_text(
        merged_md,
        "latex",
        format="markdown+grid_tables+pipe_tables+raw_tex",
        outputfile=str(tex_path),
        extra_args=common_args,
    )
    log(f"  LaTeX 생성: {tex_path}")

    # 후처리: longtable column spec 자동 wrap
    tex_content = tex_path.read_text(encoding="utf-8")
    tex_content = postprocess_longtable_columns(tex_content)
    tex_path.write_text(tex_content, encoding="utf-8")
    log(f"  LaTeX 후처리: longtable column spec 자동 wrap 적용")

    # tectonic 컴파일 (임시 디렉토리 안에서)
    result = subprocess.run(
        ["tectonic", "-X", "compile", "--outdir", str(TMP_ROOT), str(tex_path)],
        capture_output=True, text=True, cwd=str(TMP_ROOT),
    )
    if result.returncode != 0:
        log(f"  tectonic stderr (last 3000 chars):\n{result.stderr[-3000:]}")
        raise RuntimeError("tectonic 컴파일 실패")

    # 생성된 PDF 를 최종 위치(한글 경로)로 복사
    tmp_pdf = TMP_ROOT / "merged_report.pdf"
    if tmp_pdf.exists():
        shutil.copy2(tmp_pdf, OUTPUT_PDF)
        log(f"  PDF 복사: {tmp_pdf} -> {OUTPUT_PDF}")
    else:
        raise RuntimeError(f"PDF 컴파일 산출물 없음: {tmp_pdf}")

    # 디버그용 tex 도 ROOT 에 복사
    shutil.copy2(tex_path, ROOT / "_merged_report.tex")

    log(f"\n[5/6] PDF 생성 검증 ...")
    if OUTPUT_PDF.exists():
        size_mb = OUTPUT_PDF.stat().st_size / 1024 / 1024
        log(f"  생성 완료: {OUTPUT_PDF.name} ({OUTPUT_PDF.stat().st_size:,} 바이트, {size_mb:.2f} MB)")
        log(f"  경로: {OUTPUT_PDF}")
    else:
        log(f"  PDF 생성 실패")
        return

    log(f"\n[6/6] 완료.")


if __name__ == "__main__":
    main()
