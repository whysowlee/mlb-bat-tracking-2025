"""Stage 5: 본문 2장 순서 재배치 — 전처리(NaN→Scaler→다중공선성→CV) → EDA → FS → 샘플링.
 (1) Robust Scaler 블록을 다중공선성 앞으로
 (2) 샘플링 비교 블록을 FS 뒤(챕터2 끝)로 이동 + 헤딩 H3→H2 승격
"""
import zipfile, copy
from lxml import etree

PATH = "_w2.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
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
def set_level(el, lvl):
    pPr = el.find(q("pPr"))
    if pPr is None: pPr = etree.Element(q("pPr")); el.insert(0, pPr)
    ps = pPr.find(q("pStyle"))
    if ps is None: ps = etree.SubElement(pPr, q("pStyle"))
    ps.set(q("val"), lvl)

i_corr = idx("다중공선성 분석", style="3")
i_scal = idx("Robust Scaler", exact=True, style="3", start=i_corr+1)
i_cv   = idx("Cross-Validation 구조", style="3", start=i_scal+1)
i_samp = idx("샘플링 비교", style="3", start=i_cv+1)
i_eda  = idx("탐색적 분석 (EDA)", style="2", start=i_samp+1)
i_ch3  = idx("효과 분리 실험 (Ablation Study)", style="1", start=i_eda+1)

corr_block = children[i_corr:i_scal]
scal_block = children[i_scal:i_cv]
samp_block = children[i_samp:i_eda]

# (1) Robust Scaler 블록을 다중공선성 헤딩 앞으로
corr_head = children[i_corr]
for el in scal_block:
    corr_head.addprevious(el)   # 순서 유지하며 이동

# (2) 샘플링 블록: 헤딩 H3→H2 승격 후 챕터2 끝(효과분리 H1 앞)으로 이동
set_level(samp_block[0], "2")
ch3 = children[i_ch3]
for el in samp_block:
    ch3.addprevious(el)

items["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for n, d in items.items(): z.writestr(n, d)
print("Stage 5 완료")
