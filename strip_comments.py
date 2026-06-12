"""docx에서 모든 코멘트(주석) 제거 — 본문 마커 + 파트 파일 + 관계/콘텐츠타입 정리."""
import sys, zipfile, shutil, re
from lxml import etree

SRC = sys.argv[1] if len(sys.argv) > 1 else "_work.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
nsW = {"w": W}

zin = zipfile.ZipFile(SRC, "r")
items = {n: zin.read(n) for n in zin.namelist()}
zin.close()

COMMENT_PARTS = [
    "word/comments.xml", "word/commentsExtended.xml", "word/commentsIds.xml",
    "word/commentsExtensible.xml", "word/people.xml",
]

# 1) document.xml 에서 코멘트 마커 제거
doc = etree.fromstring(items["word/document.xml"])
def q(tag): return f"{{{W}}}{tag}"
removed = 0
# commentRangeStart / End
for tag in ("commentRangeStart", "commentRangeEnd"):
    for el in doc.findall(f".//{q(tag)}"):
        el.getparent().remove(el); removed += 1
# commentReference 를 포함한 run(w:r) 제거
for ref in doc.findall(f".//{q('commentReference')}"):
    run = ref.getparent()              # w:r
    if run is not None and run.tag == q("r"):
        run.getparent().remove(run); removed += 1
    else:
        ref.getparent().remove(ref); removed += 1
items["word/document.xml"] = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
print("본문 코멘트 마커 제거:", removed)

# 2) 코멘트 파트 파일 삭제
for part in COMMENT_PARTS:
    if part in items:
        del items[part]; print("파트 삭제:", part)

# 3) document.xml.rels 에서 코멘트 관계 제거
rels_name = "word/_rels/document.xml.rels"
if rels_name in items:
    rels = etree.fromstring(items[rels_name])
    R = "http://schemas.openxmlformats.org/package/2006/relationships"
    for rel in list(rels):
        typ = rel.get("Type", "")
        if any(k in typ for k in ("comments", "commentsExtended", "commentsIds",
                                  "commentsExtensible", "/people")):
            rels.remove(rel)
    items[rels_name] = etree.tostring(rels, xml_declaration=True, encoding="UTF-8", standalone=True)

# 4) [Content_Types].xml 에서 코멘트 Override 제거
ct_name = "[Content_Types].xml"
if ct_name in items:
    ct = etree.fromstring(items[ct_name])
    CT = "http://schemas.openxmlformats.org/package/2006/content-types"
    for ov in list(ct):
        pn = ov.get("PartName", "")
        if any(c.replace("word", "/word") == pn or ("/"+c) == pn for c in COMMENT_PARTS):
            ct.remove(ov)
    items[ct_name] = etree.tostring(ct, xml_declaration=True, encoding="UTF-8", standalone=True)

# 5) 다시 zip 작성
with zipfile.ZipFile(SRC, "w", zipfile.ZIP_DEFLATED) as zout:
    for name, data in items.items():
        zout.writestr(name, data)
print("저장 완료:", SRC)
