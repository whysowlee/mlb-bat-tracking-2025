"""Stage 3: 그림/표 캡션 전역 재번호(문서 순서) + 본문 내 '그림 N'/'표 N' 참조 갱신."""
import zipfile, re, copy
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
    return pPr.find(q("pStyle")).get(q("val")) if (pPr is not None and pPr.find(q("pStyle")) is not None) else ""
def set_text(el, text):
    runs = el.findall(q("r")); rpr = runs[0].find(q("rPr")) if runs else None
    for r in runs: el.remove(r)
    nr = etree.SubElement(el, q("r"))
    if rpr is not None: nr.append(copy.deepcopy(rpr))
    t = etree.SubElement(nr, q("t")); t.set(f"{{{XML}}}space", "preserve"); t.text = text

FIG_CAP = re.compile(r"^그림\s*(\d+)\.\s*(.*)$")
TBL_CAP = re.compile(r"^표\s*(\d+)\.\s*(.*)$")

# 본문 순서 단락 리스트 (TOC=style 10 제외)
paras = [el for el in body.iter(q("p"))]

# --- Pass 1: 캡션 재번호 + old->new 매핑 ---
fig_map = {}; tbl_map = {}
fig_n = 0; tbl_n = 0
for el in paras:
    if pstyle(el) == "10":  # TOC
        continue
    t = ptext(el).strip()
    m = FIG_CAP.match(t)
    if m:
        fig_n += 1; old = int(m.group(1)); alt = m.group(2)
        fig_map[old] = fig_n
        set_text(el, f"그림 {fig_n}. {alt}".rstrip())
        continue
    m = TBL_CAP.match(t)
    if m and t == f"표 {m.group(1)}.":  # 표 캡션은 '표 N.' 단독
        tbl_n += 1; old = int(m.group(1))
        tbl_map[old] = tbl_n
        set_text(el, f"표 {tbl_n}.")
print("그림 캡션:", fig_n, "| 표 캡션:", tbl_n)
print("fig_map:", fig_map)

# --- Pass 2: 본문 내 참조 갱신 (캡션/TOC 제외) ---
def span_replace(el, rx, mapping):
    tnodes = list(el.iter(q("t"))); texts = [t.text or "" for t in tnodes]
    full = "".join(texts)
    matches = [(m.start(1), m.end(1), int(m.group(1))) for m in rx.finditer(full)]
    matches = [mm for mm in matches if mm[2] in mapping and str(mapping[mm[2]]) != full[mm[0]:mm[1]]]
    if not matches: return 0
    bounds = []; s = 0
    for i, tx in enumerate(texts): bounds.append((s, s + len(tx), i)); s += len(tx)
    for st, en, old in sorted(matches, key=lambda x: -x[0]):
        newtext = str(mapping[old])
        pieces = []
        for a, b, i in bounds:
            ss = max(st, a); ee = min(en, b)
            if ss < ee: pieces.append((i, ss - a, ee - a))
        first = True
        for i, ls, le in pieces:
            if first:
                texts[i] = texts[i][:ls] + newtext + texts[i][le:]; first = False
            else:
                texts[i] = texts[i][:ls] + texts[i][le:]
    for t, tx in zip(tnodes, texts): t.text = tx
    return len(matches)

FIG_REF = re.compile(r"그림\s*(\d+)")
TBL_REF = re.compile(r"표\s*(\d+)")
nfig = ntbl = 0
for el in paras:
    if pstyle(el) == "10": continue
    t = ptext(el).strip()
    if FIG_CAP.match(t) or (TBL_CAP.match(t) and t == f"표 {TBL_CAP.match(t).group(1)}."):
        continue  # 캡션은 Pass1에서 처리
    nfig += span_replace(el, FIG_REF, fig_map)
    ntbl += span_replace(el, TBL_REF, tbl_map)
print("본문 그림참조 갱신:", nfig, "| 표참조 갱신:", ntbl)

items["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for n, d in items.items(): z.writestr(n, d)
print("Stage 3 완료")
