#!/usr/bin/env bash
# Tải lại nguyên bản bộ dữ liệu BTC cấp và verify SHA-256.
# Chạy từ thư mục provided/:  ./fetch.sh
set -euo pipefail

REPO="TechJam2026/techjam-conversational-search"
TAG="participant-kit"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$HERE/release"
gh release download "$TAG" -R "$REPO" -D "$HERE/release" --clobber

cp -f "$HERE/release/SHA256SUMS" "$HERE/SHA256SUMS"
(cd "$HERE/release" && shasum -a 256 -c SHA256SUMS)

# catalog.jsonl (60MB) không commit lên git — giải nén tại chỗ từ file .gz
gzip -dc "$HERE/release/catalog.jsonl.gz" > "$HERE/techjam-conversational-search/data/catalog.jsonl"

lines=$(wc -l < "$HERE/techjam-conversational-search/data/catalog.jsonl")
[ "$lines" -eq 50000 ] || { echo "catalog.jsonl có $lines dòng, cần đúng 50000" >&2; exit 1; }
echo "OK — catalog.jsonl: 50000 dòng, checksum khớp."
