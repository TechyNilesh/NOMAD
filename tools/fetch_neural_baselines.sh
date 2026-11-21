#!/usr/bin/env bash
# Fetch the upstream ARCUS and METER code (no licence files upstream, so it is not
# redistributed here) into src/ARCUS and src/METER at the commits we used.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone -q https://github.com/kaist-dmlab/ARCUS "$TMP/ARCUS"
git -C "$TMP/ARCUS" checkout -q 0a6aa263c2c93498836e2f26f1fc4ca3a412b5cd
cp "$TMP/ARCUS/ARCUS.py" "$TMP/ARCUS/CKA.py" "$TMP/ARCUS/utils.py" "$ROOT/src/ARCUS/"
mkdir -p "$ROOT/src/ARCUS/model"
cp "$TMP"/ARCUS/model/*.py "$ROOT/src/ARCUS/model/"

git clone -q https://github.com/zjiaqi725/METER "$TMP/METER"
git -C "$TMP/METER" checkout -q 9c0434b7d7526a7eb6f7974b6cf9c3faa678826a
cp "$TMP/METER/METER.py" "$ROOT/src/METER/meter.py"
rm -rf "$ROOT/src/METER/edl_pytorch"
cp -R "$TMP/METER/edl_pytorch" "$ROOT/src/METER/edl_pytorch"

echo "ARCUS and METER fetched into src/."
