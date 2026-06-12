"""스타일 자동번호 제거 + 제목 텍스트에 번호 직접 삽입(본문·목차 모두 표시).
   목차(TOC)·부록(부록 A~D 및 하위)은 번호 제외."""
import zipfile
from lxml import etree

PATH = "_w2.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
def q(t): return f"{{{W}}}{t}"

zin = zipfile.ZipFile(PATH); items = {n: zin.read(n) for n in zin.namelist()}; zin.close()

# 1) styles.xml: Heading 1~4 의 numPr 제거(자동번호 끔)
st = etree.fromstring(items["word/styles.xml"])
for stl in st.findall(q("style")):
    if stl.get(q("styleId")) in ("1", "2", "3", "4"):
        pPr = stl.find(q("pPr"))
        if pPr is not None:
            for ex in pPr.findall(q("numPr")): pPr.remove(ex)
items["word/styles.xml"] = etree.tostring(st, xml_declaration=True, encoding="UTF-8", standalone=True)

# 2) document.xml: 본문 제목에 번호 prepend (목차·부록 제외)
doc = etree.fromstring(items["word/document.xml"]); body = doc.find(q("body"))
def ptext(el): return "".join(t.text or "" for t in el.iter(q("t"))) if el.tag == q("p") else ""
def pstyle(el):
    pPr = el.find(q("pPr"))
    return pPr.find(q("pStyle")).get(q("val")) if (el.tag==q("p") and pPr is not None and pPr.find(q("pStyle")) is not None) else ""
def prepend(el, prefix):
    for r in el.findall(q("r")):
        ts = r.findall(q("t"))
        if ts:
            ts[0].text = prefix + (ts[0].text or "")
            ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            return True
    return False

c = [0, 0, 0, 0]; in_apx = False; applied = []
for el in list(body):
    if el.tag != q("p"): continue
    s = pstyle(el)
    if s == "1" and ptext(el).strip().startswith("부록 "): in_apx = True
    if s in ("1", "2", "3", "4") and not in_apx:
        lv = int(s) - 1
        c[lv] += 1
        for k in range(lv + 1, 4): c[k] = 0
        num = ".".join(str(c[k]) for k in range(lv + 1))
        if prepend(el, num + " "):
            applied.append((num, ptext(el)[:30]))
items["word/document.xml"] = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)

with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for n, d in items.items(): z.writestr(n, d)
print("번호 적용 헤딩:", len(applied))
for num, t in applied[:14]: print(f"  {t}")
