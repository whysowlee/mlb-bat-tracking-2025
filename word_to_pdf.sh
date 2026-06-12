#!/bin/bash
# Word(.docx) → PDF 변환 (Microsoft Word 엔진 사용, 서식 100% 보존)
# 사용법: ./word_to_pdf.sh "입력.docx" "출력.pdf"
#   인자 없으면 기본값 사용.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
IN="${1:-$DIR/ca-xBA_Final_Report_편집용.docx}"
OUT="${2:-$DIR/ca-xBA_Final_Report.pdf}"

# 절대 경로로 정규화
IN="$(cd "$(dirname "$IN")" && pwd)/$(basename "$IN")"
OUTDIR="$(cd "$(dirname "$OUT")" && pwd)"
OUT="$OUTDIR/$(basename "$OUT")"

echo "변환: $IN"
echo "  → $OUT"

osascript <<EOF
set inFile to POSIX file "$IN"
set outPath to "$OUT"
tell application "Microsoft Word"
    set wasRunning to running
    activate
    open inFile
    set theDoc to active document
    -- 목차(TOC) 등 모든 필드 업데이트 후 PDF 저장
    try
        update field (every field of theDoc)
    end try
    try
        repeat with toc in (get table of contents of theDoc)
            update toc
        end repeat
    end try
    save as theDoc file name outPath file format format PDF
    close theDoc saving no
    if not wasRunning then quit
end tell
EOF

echo "완료: $OUT"
