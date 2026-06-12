"""구조 이동 3건:
 [3] 초록 삭제 + 핵심(para2,3)을 서론(프로젝트 목표 끝)으로 흡수
 [36][42] Brier Score 정의를 3장→2장 샘플링 비교 앞으로 (H2→H3)
 [57] 4개 ANOVA 표를 부록 C→3장 ANOVA 섹션으로, 부록은 fold-level만 유지
"""
import zipfile, copy
from lxml import etree

PATH = "_w2.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
def q(t): return f"{{{W}}}{t}"

zin = zipfile.ZipFile(PATH); items = {n: zin.read(n) for n in zin.namelist()}; zin.close()
root = etree.fromstring(items["word/document.xml"]); body = root.find(q("body"))

def ptext(el): return "".join(t.text or "" for t in el.iter(q("t"))) if el.tag == q("p") else ""
def pstyle(el):
    pPr = el.find(q("pPr"))
    return pPr.find(q("pStyle")).get(q("val")) if (el.tag == q("p") and pPr is not None and pPr.find(q("pStyle")) is not None) else ""
TOC = {"10","20","30","40"}
def find(prefix, style=None, exact=False, nth=1):
    c = 0
    for el in body.iter(q("p")):
        if pstyle(el) in TOC: continue
        if style is not None and pstyle(el) != style: continue
        t = ptext(el).strip()
        if (t == prefix) if exact else t.startswith(prefix):
            c += 1
            if c == nth: return el
    raise LookupError(prefix)
def children_between(start_el, stop_pred):
    """start_el(포함)부터 stop_pred(el)True 직전까지 형제 수집."""
    out = [start_el]; sib = start_el.getnext()
    while sib is not None and not stop_pred(sib):
        out.append(sib); sib = sib.getnext()
    return out
def set_level(el, lvl):
    pPr = el.find(q("pPr"))
    if pPr is None: pPr = etree.Element(q("pPr")); el.insert(0, pPr)
    ps = pPr.find(q("pStyle"))
    if ps is None: ps = etree.SubElement(pPr, q("pStyle"))
    ps.set(q("val"), lvl)
def set_text(el, text):
    from lxml import etree as _e
    runs = el.findall(q("r")); rpr = runs[0].find(q("rPr")) if runs else None
    for r in runs: el.remove(r)
    nr = _e.SubElement(el, q("r"))
    if rpr is not None: nr.append(copy.deepcopy(rpr))
    t = _e.SubElement(nr, q("t")); t.set("{http://www.w3.org/XML/1998/namespace}space","preserve"); t.text = text

# ===== [3] 초록 =====
abs_h1 = find("초록 (Abstract)", style="1")
abs_p1 = find("본 연구는 메이저리그(MLB) 공식 기대 타율(xBA)이 타구의 물리적 질")
abs_p2 = find("Statcast BIP(인플레이 타구) 데이터에 구장 스펙")
abs_p3 = find("메인 검증은 2024년 데이터로 학습한 모델")
roadmap = find("단계별 수행 로드맵", style="2")
# para2,3 을 로드맵 앞으로 이동, 초록 헤딩+para1 삭제
roadmap.addprevious(abs_p2)
roadmap.addprevious(abs_p3)
abs_h1.getparent().remove(abs_h1)
abs_p1.getparent().remove(abs_p1)
print("[3] 초록 삭제 + para2,3 서론 흡수 완료")

# ===== [36][42] Brier 정의 이동 =====
brier_h2 = find("평가 지표 Brier Score 정의", style="2")
brier_block = children_between(brier_h2, lambda el: ptext(el).strip().startswith("실험 설계 및 결과"))
samp = find("샘플링 비교", style="3")
set_level(brier_h2, "3")          # H2→H3 (2장 Phase2 하위)
for el in brier_block:
    samp.addprevious(el)
print(f"[36][42] Brier 정의 블록 {len(brier_block)}개 → 2장 샘플링 앞 이동")

# ===== [57] 4개 ANOVA 표 본문 이동 =====
anova_h2 = find("2-way ANOVA (Type II SS)", style="2")   # 부록 C 하위
anova_block = children_between(anova_h2, lambda el: ptext(el).strip().startswith("부록 D"))
# anova_block = [H2, intro, (Brier:,표,tbl), (LogLoss:..), (ROC AUC:..), (F1:..)]
h2_el, intro_el = anova_block[0], anova_block[1]
move_els = anova_block[2:]        # 라벨+캡션+표 12개
body_anova_intro = find("각 fold(n=5)")     # 3장 본문 (첫 등장)
ip_head = find("Interaction Plot (Data x Algo) + ANOVA")  # H4
for el in move_els:
    ip_head.addprevious(el)
h2_el.getparent().remove(h2_el)
intro_el.getparent().remove(intro_el)
# 부록 C 제목 정리
apx_c = find("부록 C", style="1")
set_text(apx_c, "부록 C. 효과 분리 실험 fold-level 통계 (Phase 3)")
print(f"[57] ANOVA 표 그룹 {len(move_els)}개 → 3장 본문 이동, 부록 C는 fold-level만")

items["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for n, d in items.items(): z.writestr(n, d)
print("S1 완료")
