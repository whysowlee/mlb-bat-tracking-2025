"""제목·목차 섹션 번호 재정렬: 기존 번호 strip 후 재적용. 부록(및 하위)은 번호 제외."""
import zipfile, re
from lxml import etree

PATH = "_w2.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
def q(t): return f"{{{W}}}{t}"
NUMRE = re.compile(r"^\d+(\.\d+)*\s+")

zin = zipfile.ZipFile(PATH); items = {n: zin.read(n) for n in zin.namelist()}; zin.close()
doc = etree.fromstring(items["word/document.xml"]); body = doc.find(q("body"))

def pstyle(el):
    pPr = el.find(q("pPr"))
    return pPr.find(q("pStyle")).get(q("val")) if (pPr is not None and pPr.find(q("pStyle")) is not None) else ""
def first_t(el):
    for t in el.iter(q("t")):
        if t.text is not None and t.text.strip() != "": return t
    return None
def full_text(el): return "".join(t.text or "" for t in el.iter(q("t")))
def strip_num(el):
    t = first_t(el)
    if t is not None and NUMRE.match(t.text):
        t.text = NUMRE.sub("", t.text)
def prepend(el, prefix):
    t = first_t(el)
    if t is not None:
        t.text = prefix + t.text
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

# ── 1) 본문 헤딩(style 1~4): strip 후 재적용 (부록 이후 제외) ──────────────
c = [0, 0, 0, 0]; in_apx = False
for el in body.iter(q("p")):
    s = pstyle(el)
    if s not in ("1", "2", "3", "4"): continue
    strip_num(el)
    if s == "1" and full_text(el).strip().startswith("부록 "): in_apx = True
    if in_apx: continue
    lv = int(s) - 1
    c[lv] += 1
    for k in range(lv + 1, 4): c[k] = 0
    prepend(el, ".".join(str(c[k]) for k in range(lv + 1)) + " ")

# ── 2) 목차(style 10/20): strip 후 재적용 (부록 이후 제외) ──────────────────
c1 = c2 = 0; apx = False
for el in doc.iter(q("p")):
    s = pstyle(el)
    if s not in ("10", "20", "30", "40"): continue
    strip_num(el)
    txt = full_text(el).strip()
    if s == "10" and txt.startswith("부록"): apx = True
    if apx: continue
    if s == "10":
        c1 += 1; c2 = 0; prepend(el, f"{c1} ")
    elif s == "20":
        c2 += 1; prepend(el, f"{c1}.{c2} ")

items["word/document.xml"] = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for n, d in items.items(): z.writestr(n, d)
print("재정렬 완료")
