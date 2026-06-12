"""제목/소제목 자동 넘버링 — Heading 1~4 스타일에 multilevel(1, 1.1, 1.1.1, 1.1.1.1) 연결.
   부록(부록 A~D 및 하위)은 numId=0 으로 넘버링 제외('부록 A' 라벨 유지)."""
import zipfile, copy
from lxml import etree

PATH = "_w2.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
def q(t): return f"{{{W}}}{t}"
NID = "100"; AID = "100"

zin = zipfile.ZipFile(PATH); items = {n: zin.read(n) for n in zin.namelist()}; zin.close()

# ── 1) numbering.xml: abstractNum(100) + num(100) 추가 ──────────────────────
num_root = etree.fromstring(items["word/numbering.xml"])
def lvl(i):
    txt = ".".join(f"%{k+1}" for k in range(i+1))
    s = (f'<w:lvl xmlns:w="{W}" w:ilvl="{i}"><w:start w:val="1"/>'
         f'<w:numFmt w:val="decimal"/><w:lvlText w:val="{txt}"/><w:lvlJc w:val="left"/>'
         f'<w:suff w:val="space"/><w:pPr><w:ind w:left="0" w:hanging="0"/></w:pPr></w:lvl>')
    return etree.fromstring(s)
abs_el = etree.fromstring(f'<w:abstractNum xmlns:w="{W}" w:abstractNumId="{AID}"><w:multiLevelType w:val="multilevel"/></w:abstractNum>')
for i in range(9): abs_el.append(lvl(i))
# abstractNum 들 뒤(마지막 abstractNum 다음), num 들 앞에 삽입
last_abs = num_root.findall(q("abstractNum"))[-1]
last_abs.addnext(abs_el)
num_el = etree.fromstring(f'<w:num xmlns:w="{W}" w:numId="{NID}"><w:abstractNumId w:val="{AID}"/></w:num>')
num_root.append(num_el)   # num 은 맨 끝
items["word/numbering.xml"] = etree.tostring(num_root, xml_declaration=True, encoding="UTF-8", standalone=True)

# ── 2) styles.xml: Heading 1~4 에 numPr 연결 ────────────────────────────────
st_root = etree.fromstring(items["word/styles.xml"])
BEFORE = {q("keepNext"), q("keepLines"), q("pageBreakBefore"), q("framePr"), q("widowControl")}
for sid, ilvl in (("1","0"),("2","1"),("3","2"),("4","3")):
    for stl in st_root.findall(q("style")):
        if stl.get(q("styleId")) == sid:
            pPr = stl.find(q("pPr"))
            if pPr is None:
                pPr = etree.SubElement(stl, q("pPr"))
            for ex in pPr.findall(q("numPr")): pPr.remove(ex)
            numpr = etree.fromstring(f'<w:numPr xmlns:w="{W}"><w:ilvl w:val="{ilvl}"/><w:numId w:val="{NID}"/></w:numPr>')
            # 올바른 스키마 위치: BEFORE 집합의 마지막 다음
            pos = 0
            for k, ch in enumerate(list(pPr)):
                if ch.tag in BEFORE: pos = k + 1
            pPr.insert(pos, numpr)
            break
items["word/styles.xml"] = etree.tostring(st_root, xml_declaration=True, encoding="UTF-8", standalone=True)

# ── 3) document.xml: 부록 헤딩들 numId=0 (넘버링 제외) ───────────────────────
doc_root = etree.fromstring(items["word/document.xml"]); body = doc_root.find(q("body"))
def ptext(el): return "".join(t.text or "" for t in el.iter(q("t"))) if el.tag == q("p") else ""
def pstyle(el):
    pPr = el.find(q("pPr"))
    return pPr.find(q("pStyle")).get(q("val")) if (el.tag==q("p") and pPr is not None and pPr.find(q("pStyle")) is not None) else ""
in_apx = False; cnt = 0
for el in list(body):
    if el.tag != q("p"): continue
    if pstyle(el) == "1" and ptext(el).strip().startswith("부록 "): in_apx = True
    if in_apx and pstyle(el) in ("1","2","3","4"):
        pPr = el.find(q("pPr"))
        for ex in pPr.findall(q("numPr")): pPr.remove(ex)
        numpr = etree.fromstring(f'<w:numPr xmlns:w="{W}"><w:numId w:val="0"/></w:numPr>')
        # pStyle 다음에 삽입
        ps = pPr.find(q("pStyle"))
        (ps.addnext(numpr) if ps is not None else pPr.insert(0, numpr))
        cnt += 1
items["word/document.xml"] = etree.tostring(doc_root, xml_declaration=True, encoding="UTF-8", standalone=True)
print("부록 넘버링 제외 헤딩:", cnt)

with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for n, d in items.items(): z.writestr(n, d)
print("넘버링 적용 완료")
