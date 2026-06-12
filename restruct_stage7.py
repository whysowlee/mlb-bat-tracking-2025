"""Stage 7: Phase 기준 재배치 — 2.2 전처리(Phase1) / 2.3 EDA / 2.4 Phase2(변수정제·CV·FS·샘플링).
 - Phase2 전처리 블록(변수그룹·NaN·Scaler·다중공선성·CV)을 EDA 뒤 Phase2 섹션으로 이동
 - 2.2 H2 → '데이터 전처리 (Phase 1)', FS H2 → 'Phase 2' 통합 제목
 - 샘플링 H2 → H3 (Phase2 하위)
"""
import zipfile, copy
from lxml import etree

PATH = "_w2.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
def q(t): return f"{{{W}}}{t}"

zin = zipfile.ZipFile(PATH); items = {n: zin.read(n) for n in zin.namelist()}; zin.close()
root = etree.fromstring(items["word/document.xml"]); body = root.find(q("body"))
children = list(body)

def ptext(el): return "".join(t.text or "" for t in el.iter(q("t"))) if el.tag == q("p") else ""
def pstyle(el):
    pPr = el.find(q("pPr"))
    return pPr.find(q("pStyle")).get(q("val")) if (pPr is not None and pPr.find(q("pStyle")) is not None) else ""
def idx(prefix, exact=False, style=None, start=0):
    for i in range(start, len(children)):
        if style is not None and pstyle(children[i]) != style: continue
        t = ptext(children[i]).strip()
        if (t == prefix) if exact else t.startswith(prefix):
            return i
    raise LookupError(prefix)
def set_text(el, text):
    runs = el.findall(q("r")); rpr = runs[0].find(q("rPr")) if runs else None
    for r in runs: el.remove(r)
    nr = etree.SubElement(el, q("r"))
    if rpr is not None: nr.append(copy.deepcopy(rpr))
    t = etree.SubElement(nr, q("t")); t.set(f"{{{XML}}}space", "preserve"); t.text = text
def set_level(el, lvl):
    pPr = el.find(q("pPr"))
    if pPr is None: pPr = etree.Element(q("pPr")); el.insert(0, pPr)
    ps = pPr.find(q("pStyle"))
    if ps is None: ps = etree.SubElement(pPr, q("pStyle"))
    ps.set(q("val"), lvl)

i_prep = idx("데이터 전처리", exact=True, style="2")
i_pre2 = idx("변수 그룹 정의 및 초기 풀 구성", style="3", start=i_prep+1)
i_eda  = idx("탐색적 분석 (EDA)", style="2", start=i_pre2+1)
i_fsh2 = idx("Feature Selection", exact=True, style="2", start=i_eda+1)
i_fsrf = idx("Feature Selection (RF importance", style="3", start=i_fsh2+1)
i_samp = idx("샘플링 비교", style="2", start=i_fsrf+1)

pre2_block = children[i_pre2:i_eda]   # 변수그룹·NaN·Scaler·다중공선성·CV (+표/캡션)

# 1) 헤딩 제목 변경
set_text(children[i_prep], "데이터 전처리 (Phase 1)")
set_text(children[i_fsh2], "변수 정제, 교차검증 및 Feature Selection (Phase 2)")
# 2) Phase2 전처리 블록을 'Feature Selection (RF importance...)' H3 앞으로 이동
fsrf_head = children[i_fsrf]
for el in pre2_block:
    fsrf_head.addprevious(el)
# 3) 샘플링 H2 → H3 (Phase2 하위로 격하)
set_level(children[i_samp], "3")

items["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for n, d in items.items(): z.writestr(n, d)
print("Stage 7 완료")
