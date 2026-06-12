"""Stage 1: 챕터2 구조 재배치 (전처리 / EDA / Feature Selection 4분할)."""
import zipfile, copy
from lxml import etree

PATH = "_w2.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
def q(t): return f"{{{W}}}{t}"

zin = zipfile.ZipFile(PATH); items = {n: zin.read(n) for n in zin.namelist()}; zin.close()
root = etree.fromstring(items["word/document.xml"]); body = root.find(q("body"))
children = list(body)

def ptext(el):
    if el.tag != q("p"): return None
    return "".join(t.text or "" for t in el.iter(q("t")))

def fidx(prefix, exact=False, start=0):
    for i in range(start, len(children)):
        t = ptext(children[i])
        if t is None: continue
        ts = t.strip()
        if (ts == prefix) if exact else ts.startswith(prefix):
            return i
    raise LookupError(prefix)

def set_text(el, text):
    runs = el.findall(q("r")); rpr = runs[0].find(q("rPr")) if runs else None
    for r in runs: el.remove(r)
    nr = etree.SubElement(el, q("r"))
    if rpr is not None: nr.append(copy.deepcopy(rpr))
    t = etree.SubElement(nr, q("t")); t.set(f"{{{XML}}}space", "preserve"); t.text = text

def set_level(el, lvl):
    pPr = el.find(q("pPr"))
    if pPr is None:
        pPr = etree.Element(q("pPr")); el.insert(0, pPr)
    ps = pPr.find(q("pStyle"))
    if ps is None:
        ps = etree.SubElement(pPr, q("pStyle"))
    ps.set(q("val"), lvl)

iCH2 = fidx("데이터 전처리 및 탐색적 분석", exact=True)
iA = fidx("데이터셋 설명", exact=True, start=iCH2+1)
iB = fidx("전처리 파이프라인", start=iA+1)
iC = fidx("단계별 attrition", start=iB+1)
iD = fidx("돔/지붕 닫힘 경기 기상 마스킹", start=iC+1)
iE = fidx("기상 데이터 병합 결과", start=iD+1)
iF = fidx("Temporal Split 결과", start=iE+1)
iG = fidx("타구 물리 분포", exact=True, start=iF+1)
iH = fidx("환경 변수", exact=True, start=iG+1)
iI = fidx("탐색적 분석 및 Feature Selection", start=iH+1)
iJ = fidx("변수 그룹 정의 및 초기 풀 구성", start=iI+1)
iK = fidx("NaN 처리", start=iJ+1)
iL = fidx("Cross-Validation 구조", start=iK+1)
iM = fidx("다중공선성 분석", start=iL+1)
iN = fidx("Robust Scaler", exact=True, start=iM+1)
iO = fidx("샘플링 비교", start=iN+1)
iP = fidx("Feature Selection (RF importance", start=iO+1)
iQ = fidx("최종 X_advanced 변수 확정", start=iP+1)
iR = fidx("EDA — 핵심 변수 분포", start=iQ+1)
iS = fidx("(C2) OOF 평가지표 막대", start=iR+1)
iCH3 = fidx("효과 분리 실험 (Ablation Study)", start=iS+1)

A = children[iA:iB]; B = children[iB:iC]; C = children[iC:iD]; D = children[iD:iE]
E = children[iE:iF]; Fb = children[iF:iG]; G = children[iG:iH]; Hb = children[iH:iI]
Ihead = children[iI:iJ]; J = children[iJ:iK]; K = children[iK:iL]; L = children[iL:iM]
M = children[iM:iN]; N = children[iN:iO]; O = children[iO:iP]; P = children[iP:iQ]
Qb = children[iQ:iR]; R = children[iR:iS]; S = children[iS:iCH3]

h2_phase2 = Ihead[0]
intro_para = Ihead[1:]   # "본 단계는 2024_data..." 등

# 헤딩 텍스트/레벨 조정
set_text(B[0], "데이터 전처리")              # H2
set_text(h2_phase2, "탐색적 분석 (EDA)")     # H2 (재활용)
set_level(G[0], "3")                          # 타구 물리 분포 H4→H3
set_level(Hb[0], "3")                         # 환경 변수 H4→H3
set_level(R[0], "3"); set_text(R[0], "핵심 변수 분포")  # H4→H3 + 개명
fs_h2 = copy.deepcopy(h2_phase2); set_text(fs_h2, "Feature Selection")

# S(고아 캡션) 삭제
for el in S: body.remove(el)

# 새 순서 구성
new_seq = (A + B + intro_para + C + D + E + Fb
           + J + K + M + N + L + O
           + [h2_phase2] + G + Hb + R
           + [fs_h2] + P + Qb)

ref = children[iCH3]
for el in new_seq:
    ref.addprevious(el)   # 기존 요소는 이동, fs_h2(신규)는 삽입

items["word/document.xml"] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
with zipfile.ZipFile(PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for n, d in items.items(): z.writestr(n, d)
print("Stage 1 완료")
