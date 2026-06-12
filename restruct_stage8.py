"""Stage 8: 표5 82행 삭제 + [24][25] 다중공선성 본문 축약 + 고상관쌍 표 부록 복귀
   (부록 A→B,B→C,C→D 재라벨 후 새 부록 A=다중공선성 추가) + FS 절번호 참조 보정."""
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
    return pPr.find(q("pStyle")).get(q("val")) if (pPr is not None and pPr.find(q("pStyle")) is not None) else ""
def set_text(el, text):
    runs = el.findall(q("r")); rpr = runs[0].find(q("rPr")) if runs else None
    for r in runs: el.remove(r)
    nr = etree.SubElement(el, q("r"))
    if rpr is not None: nr.append(copy.deepcopy(rpr))
    t = etree.SubElement(nr, q("t")); t.set(f"{{{XML}}}space", "preserve"); t.text = text
def find_p(prefix, exact=False, style=None):
    for el in body.iter(q("p")):
        if style is not None and pstyle(el) != style: continue
        t = ptext(el).strip()
        if (t == prefix) if exact else t.startswith(prefix):
            return el
    return None
def find_table(first_cell):
    for tbl in body.iter(q("tbl")):
        tc = tbl.find(f".//{q('tc')}")
        if tc is not None and "".join(x.text or "" for x in tc.iter(q("t"))).strip() == first_cell:
            return tbl
    return None

# --- 1) 표5 'X_advanced 초기' 행 삭제 ----------------------------------------
grp_tbl = find_table("그룹")
if grp_tbl is not None:
    for tr in grp_tbl.findall(q("tr")):
        if "X_advanced 초기" in "".join(x.text or "" for x in tr.iter(q("t"))):
            tr.getparent().remove(tr); print("표5 'X_advanced 초기' 행 삭제"); break

# --- 2) 부록 A→B,B→C,C→D 재라벨 (TOC 제외, 글자 위치 in-place) --------------
mp = {"A": "B", "B": "C", "C": "D"}
RX = re.compile(r"부록 ([ABC])")
def relabel(el):
    tnodes = list(el.iter(q("t"))); texts = [t.text or "" for t in tnodes]
    full = "".join(texts)
    if not RX.search(full): return
    edits = [(m.start(1), mp[m.group(1)]) for m in RX.finditer(full)]
    bnd = []; s = 0
    for i, tx in enumerate(texts): bnd.append((s, s+len(tx), i)); s += len(tx)
    for pos, nc in sorted(edits, reverse=True):
        for a, b, i in bnd:
            if a <= pos < b: texts[i] = texts[i][:pos-a] + nc + texts[i][pos-a+1:]; break
    for t, tx in zip(tnodes, texts): t.text = tx
for el in body.iter(q("p")):
    if pstyle(el) != "10": relabel(el)
print("부록 재라벨 A→B,B→C,C→D 완료")

# --- 3) 고상관쌍 표 부록 A로 이동 + 새 부록 A 섹션 생성 ----------------------
corr_tbl = find_table("변수 A")
body_cap = corr_tbl.getprevious()                 # 본문 캡션 '표 N.'
if body_cap is not None and re.match(r"^표\s*\d+\.$", ptext(body_cap).strip()):
    body_cap.getparent().remove(body_cap)
corr_tbl.getparent().remove(corr_tbl)             # 본문에서 분리

# 클론 템플릿: 결측치(현재 부록 B) 헤딩 + 그 intro 단락
b_head = find_p("결측치 대체 중앙값", style="1") or find_p("부록 B")
b_intro = b_head.getnext()                        # 결측치 intro 단락
cap_tmpl = next(el for el in body.iter(q("p")) if re.match(r"^표\s*\d+\.$", ptext(el).strip()))

new_head = copy.deepcopy(b_head); set_text(new_head, "부록 A. 다중공선성 분석 — 고상관 변수 쌍 전체 목록")
new_intro = copy.deepcopy(b_intro)
set_text(new_intro, "Phase 2의 다중공선성 분석(Pearson |r| > 0.95)에서 식별된 24건의 고상관 변수 쌍을 |r| 내림차순으로 "
                    "정리한다. 변수 A/변수 B는 RobustScaler 적용 후 분산이며, 제거 규칙은 X_BASE 보존 → derived 변수 우선 "
                    "drop → variance fallback이다.")
new_cap = copy.deepcopy(cap_tmpl); set_text(new_cap, "표 0.")
# 결측치(부록 B) 헤딩 앞에 삽입: head, intro, cap, table 순
b_head.addprevious(new_head)
b_head.addprevious(new_intro)
b_head.addprevious(new_cap)
b_head.addprevious(corr_tbl)
print("새 부록 A(다중공선성) 생성 + 표 이동 완료")

# --- 4) 본문 다중공선성 단락 축약 ([24][25]) --------------------------------
corr_para = find_p("다중공선성 문제 해결을 위해 Pearson")
set_text(corr_para,
    "다중공선성 문제 해결을 위해 Pearson 상관계수(|r| > 0.95)를 기준으로 고상관 쌍 24건을 식별하고, 그중 9개 변수를 "
    "제거했다. 제거 규칙은 (1) X_BASE(launch_speed·launch_angle)는 항상 보존, (2) 그 외에는 의미가 파생된 변수(derived)를 "
    "우선 제거, (3) 동률이면 분산이 더 작은 쪽을 제거(variance fallback)다. 쌍은 24건이지만 실제 제거 변수가 9개인 이유는, "
    "24개 쌍의 상당수가 동일한 변수들(특히 7종의 *_is_missing 결측 플래그)끼리 서로 중복 상관된 것이어서 쌍은 많아도 제거되는 "
    "고유 변수 수는 적기 때문이다. 식별된 24개 쌍과 각 쌍의 제거 변수 전체 목록은 부록 A에 정리했다.")
# 기존 '제거된 변수 전체 목록' 줄이 남아있으면 제거
for el in list(body.iter(q("p"))):
    if ptext(el).strip().startswith("제거된 변수 전체 목록"):
        el.getparent().remove(el)

# --- 5) FS 절번호 참조 보정 (샘플링이 같은 절로 이동) ------------------------
fs = find_p("학습 데이터: 원본 분포")
if fs is not None:
    set_text(fs, "학습 데이터: 원본 분포(별도 샘플링 미적용) 위에서 RF importance와 MI를 계산했다. "
                 "이는 뒤의 샘플링 비교에서 원본(None)이 최적으로 확정되는 결과와 일치한다.")

items["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for n, d in items.items(): z.writestr(n, d)
print("Stage 8 완료")
