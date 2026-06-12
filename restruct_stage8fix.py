"""Stage 8 fix: 잘못 삽입된 부록 A(다중공선성) 요소를 TOC에서 빼내 본문 부록에 올바르게 재배치."""
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
    return pPr.find(q("pStyle")).get(q("val")) if (el.tag == q("p") and pPr is not None and pPr.find(q("pStyle")) is not None) else ""
def set_text(el, text):
    runs = el.findall(q("r")); rpr = runs[0].find(q("rPr")) if runs else None
    for r in runs: el.remove(r)
    nr = etree.SubElement(el, q("r"))
    if rpr is not None: nr.append(copy.deepcopy(rpr))
    t = etree.SubElement(nr, q("t")); t.set(f"{{{XML}}}space", "preserve"); t.text = text

# 1) 잘못 들어간 corr_tbl + 앞 3개(new_cap,new_intro,new_head) 회수
corr_tbl = None
for tbl in body.iter(q("tbl")):
    tc = tbl.find(f".//{q('tc')}")
    if tc is not None and "".join(x.text or "" for x in tc.iter(q("t"))).strip() == "변수 A":
        corr_tbl = tbl; break
prev3 = []
sib = corr_tbl.getprevious()
for _ in range(3):
    prev3.append(sib); sib = sib.getprevious()
corr_tbl.getparent().remove(corr_tbl)             # 표 회수(보관)
for el in prev3:                                   # 잘못된 head/intro/cap 제거
    if el is not None: el.getparent().remove(el)
print("잘못 삽입된 부록A 요소 제거 완료")

# 2) 본문 부록 헤딩(style=1, body 직계)에서 '부록 B(결측치)' 찾기
b_head = None
for el in list(body):
    if pstyle(el) == "1" and ptext(el).strip().startswith("부록 B"):
        b_head = el; break
b_intro = b_head.getnext()                         # 결측치 intro (body 단락)
# 캡션 템플릿: body 직계 '표 N.'
cap_tmpl = None
for el in body.iter(q("p")):
    if re.match(r"^표\s*\d+\.$", ptext(el).strip()):
        cap_tmpl = el; break

new_head = copy.deepcopy(b_head); set_text(new_head, "부록 A. 다중공선성 분석 — 고상관 변수 쌍 전체 목록")
new_intro = copy.deepcopy(b_intro)
set_text(new_intro, "Phase 2의 다중공선성 분석(Pearson |r| > 0.95)에서 식별된 24건의 고상관 변수 쌍을 |r| 내림차순으로 "
                    "정리한다. 변수 A/변수 B는 RobustScaler 적용 후 분산이며, 제거 규칙은 X_BASE 보존 → derived 변수 우선 "
                    "drop → variance fallback이다.")
new_cap = copy.deepcopy(cap_tmpl); set_text(new_cap, "표 0.")
b_head.addprevious(new_head)
b_head.addprevious(new_intro)
b_head.addprevious(new_cap)
b_head.addprevious(corr_tbl)
print("본문 부록 A 올바르게 재삽입 완료")

items["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for n, d in items.items(): z.writestr(n, d)
print("Stage 8 fix 완료")
