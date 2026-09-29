#!/usr/bin/env bash
# Fetch the upstream model inputs (not redistributed with this repo).
# See ../UPSTREAM.md for what comes from where.
set -euo pipefail

UPSTREAM_URL="https://github.com/ssaiveena/Continuous-Reoptimization"
DEST="${BPA_DATA_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Cloning $UPSTREAM_URL ..."
git clone --depth 1 "$UPSTREAM_URL" "$TMP/upstream"

mkdir -p "$DEST"/{cmip5,lulc,reference}

# NOTE: adjust these source paths if the upstream layout changes.
echo "Copying reference data ..."
for f in nodes.json params.json; do
  find "$TMP/upstream" -name "$f" -print -quit | xargs -I{} cp {} "$DEST/reference/" || \
    echo "  WARN: $f not found upstream"
done
find "$TMP/upstream" -name "historical_medians.csv" -print -quit | \
  xargs -I{} cp {} "$DEST/reference/" || echo "  WARN: historical_medians.csv not found"

echo "Copying scenario data (this is the bulk) ..."
find "$TMP/upstream" -path "*cmip5*" -name "*.csv.zip" -exec cp {} "$DEST/cmip5/" \; 2>/dev/null || true
find "$TMP/upstream" -path "*lulc*"  -name "*.csv"     -exec cp {} "$DEST/lulc/"  \; 2>/dev/null || true

echo
echo "Fetched into: $DEST"
echo "  cmip5:     $(ls "$DEST/cmip5" 2>/dev/null | wc -l | tr -d ' ') files"
echo "  lulc:      $(ls "$DEST/lulc" 2>/dev/null | wc -l | tr -d ' ') files"
echo "  reference: $(ls "$DEST/reference" 2>/dev/null | wc -l | tr -d ' ') files"
echo
echo "Verify with:  python -c \"import sys;sys.path.insert(0,'bdps');import paths;paths.require_data();print('data OK')\""
