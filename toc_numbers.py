"""목차(TOC) 항목 텍스트에도 섹션 번호 삽입 (본문 H1/H2 번호와 동일). 부록 항목 제외."""
import zipfile
from lxml import etree

PATH = "_w2.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
def q(t): return f"{{{W}}}{t}"

zin = zipfile.ZipFile(PATH); items = {n: zin.read(n) for n in zin.namelist()}; zin.close()
doc = etree.fromstring(items["word/document.xml"])

def pstyle(el):
    pPr = el.find(q("pPr"))
    return pPr.find(q("pStyle")).get(q("val")) if (pPr is not None and pPr.find(q("pStyle")) is not None) else ""
def title_text(p):
    out = []
    for t in p.iter(q("t")):
        if t.text: out.append(t.text)
    return "".join(out)
def prepend_first(p, prefix):
    for t in p.iter(q("t")):
        if t.text is not None and t.text.strip() != "":
            t.text = prefix + t.text
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            return True
    return False

c1 = c2 = 0; n = 0
for p in doc.iter(q("p")):
    s = pstyle(p)
    if s not in ("10", "20"): continue
    txt = title_text(p).strip()
    if txt.startswith("부록"):       # 부록은 번호 없음
        continue
    if s == "10":
        c1 += 1; c2 = 0; num = str(c1)
    else:
        c2 += 1; num = f"{c1}.{c2}"
    if prepend_first(p, num + " "): n += 1

items["word/document.xml"] = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for nm, d in items.items(): z.writestr(nm, d)
print("목차 항목 번호 적용:", n)
