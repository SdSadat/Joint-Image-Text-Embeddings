#!/bin/sh
# Upload src/ as the private Kaggle dataset that every kernel imports from.
set -e
tmp=$(mktemp -d)
cp src/*.py "$tmp"
cat > "$tmp/dataset-metadata.json" <<META
{"title": "jite-src", "id": "sadat0/jite-src", "licenses": [{"name": "CC0-1.0"}]}
META
if kaggle datasets status sadat0/jite-src >/dev/null 2>&1; then
    kaggle datasets version -p "$tmp" -m "${1:-update src}"
else
    kaggle datasets create -p "$tmp"
fi
rm -rf "$tmp"
