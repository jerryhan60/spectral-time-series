#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Smoke test ==="

# 1. Run unit tests
echo "--- Unit tests ---"
cd "$REPO_ROOT"
pytest -q tests/ -m "not gpu and not slow" || { echo "FAIL: unit tests"; exit 1; }

# 2. Verify tables from CSVs
echo "--- Table verification ---"
python scripts/verify_tables.py --results-dir results --strict || { echo "FAIL: table verification"; exit 1; }

echo ""
echo "=== All smoke tests passed ==="
