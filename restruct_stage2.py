"""Stage 2 (수정): 고상관 쌍 표 본문 복원 + 부록 A 삭제 + 부록 재라벨(B→A,C→B,D→C).
- TOC(스타일 10) 필드는 건드리지 않음(PDF 추출 시 필드 업데이트로 갱신).
- 런이 단어별로 분할되므로 글자 위치 기반 in-place 치환으로 서식 보존.
"""
import zipfile, copy, re
from lxml import etree

PATH = "_w2.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
def q(t): return f"{{{W}}}{t}"

zin = zipfile.ZipFile(PATH); items = {n: zin.read(n) for n in zin.namelist()}; zin.close()
root = etree.fromstring(items["word/document.xml"]); body = root.find(q("body"))

def ptext(el): return "".join(t.text or "" for t in el.iter(q("t"))) if el.tag == q("p") else ""
def pstyle(el):
    pPr = el.find(q("pPr"))
    if pPr is not None and pPr.find(q("pStyle")) is not None:
        return pPr.find(q("pStyle")).get(q("val"))
    return ""
def set_text(el, text):
    runs = el.findall(q("r")); rpr = runs[0].find(q("rPr")) if runs else None
    for r in runs: el.remove(r)
    nr = etree.SubElement(el, q("r"))
    if rpr is not None: nr.append(copy.deepcopy(rpr))
    t = etree.SubElement(nr, q("t")); t.set(f"{{{XML}}}space", "preserve"); t.text = text
def find_p(prefix, exact=False, style=None):
    for el in body.iter(q("p")):
        if style is not None and pstyle(el) != style: continue
        ts = ptext(el).strip()
        if (ts == prefix) if exact else ts.startswith(prefix):
            return el
    raise LookupError(prefix)
def find_table(first_cell):
    for tbl in body.iter(q("tbl")):
        tc = tbl.find(f".//{q('tc')}")
        if tc is not None and "".join(x.text or "" for x in tc.iter(q("t"))).strip() == first_cell:
            return tbl
    return None

# --- 1) 고상관 쌍 표 본문 이동 + 다중공선성 줄글 정리 ----------------------
corr_tbl = find_table("변수 A")
intro = find_p("다중공선성 문제 해결을 위해 Pearson")
set_text(intro,
    "다중공선성 문제 해결을 위해 Pearson 상관계수(|r| > 0.95)를 기준으로 고상관 쌍 24건을 식별하고, "
    "X_BASE 보존 → derived 변수 우선 drop → 분산 보존(variance fallback) 규칙으로 총 9개 변수를 제거했다. "
    "식별된 24개 고상관 쌍과 각 쌍에서 제거된 변수는 아래 표와 같다.")
for el in list(body.iter(q("p"))):
    if ptext(el).strip().startswith("고상관 쌍에서 제거한 변수는"):
        el.getparent().remove(el); break
cap_tmpl = next(el for el in body.iter(q("p")) if re.match(r"^표\s*\d+\.$", ptext(el).strip()))
new_cap = copy.deepcopy(cap_tmpl); set_text(new_cap, "표 0.")
corr_tbl.getparent().remove(corr_tbl)
intro.addnext(corr_tbl)
intro.addnext(new_cap)

# --- 2) 부록 A(다중공선성) 실제 섹션 삭제 (스타일 1 헤딩 기준) -------------
apA = find_p("부록 A", style="1")
to_del = [apA]; sib = apA.getnext()
while sib is not None:
    if sib.tag == q("p") and pstyle(sib) == "1" and ptext(sib).strip().startswith("부록 "):
        break
    to_del.append(sib); sib = sib.getnext()
for el in to_del:
    el.getparent().remove(el)

# --- 3) 부록 재라벨 B→A,C→B,D→C (TOC=스타일10 제외, 글자 위치 in-place) ---
mp = {"B": "A", "C": "B", "D": "C"}
RX = re.compile(r"부록 ([BCD])")
def relabel(el):
    tnodes = list(el.iter(q("t")))
    texts = [t.text or "" for t in tnodes]
    full = "".join(texts)
    if not RX.search(full): return
    edits = [(m.start(1), mp[m.group(1)]) for m in RX.finditer(full)]
    bounds = []; s = 0
    for i, tx in enumerate(texts):
        bounds.append((s, s + len(tx), i)); s += len(tx)
    for pos, nc in sorted(edits, reverse=True):
        for a, b, i in bounds:
            if a <= pos < b:
                texts[i] = texts[i][:pos-a] + nc + texts[i][pos-a+1:]; break
    for t, tx in zip(tnodes, texts): t.text = tx
for el in body.iter(q("p")):
    if pstyle(el) == "10":  # TOC 필드는 건너뜀
        continue
    relabel(el)

items["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for n, d in items.items(): z.writestr(n, d)
print("Stage 2 완료")
