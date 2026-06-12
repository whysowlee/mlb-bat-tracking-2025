"""복구: S2의 잘못된 Brier 이동(TOC 매칭 버그) 되돌리기 + 올바른 Brier 재배치."""
import zipfile
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
def find_body_h3(prefix):
    for el in body:                       # 직계 자식만(=본문), TOC(sdt 내부) 배제
        if pstyle(el) == "3" and ptext(el).strip().startswith(prefix):
            return el
    raise LookupError(prefix)

brier_h = find_body_h3("평가 지표 Brier Score 정의")
fs_h = find_body_h3("Feature Selection (RF importance")

# 1) 잘못 이동된 blk(brier_h..fs_h 직전)을 body 끝으로 되돌림 → 원래 순서 복원
blk = []; node = brier_h
while node is not None and node is not fs_h:
    blk.append(node); node = node.getnext()
print("되돌릴 blk 요소 수:", len(blk))
for node in blk:
    body.append(node)     # 끝으로 이동(원래 위치: FS·최종X·CV 뒤)

# 2) 이제 순서: ...다중공선성, fs_h, FS, 최종X, CV, brier_h, samp, ...
#    Brier 블록(brier_h..samp_h 직전)을 fs_h 앞으로 올바르게 이동
samp_h = find_body_h3("샘플링 비교")
bblk = []; node = brier_h
while node is not None and node is not samp_h:
    bblk.append(node); node = node.getnext()
print("Brier 블록 요소 수:", len(bblk))
for node in bblk:
    fs_h.addprevious(node)

items["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for n, d in items.items(): z.writestr(n, d)
print("복구 완료")
