#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_ROOT/external"
cd "$REPO_ROOT/external"

echo "=== Installing Uni2TS ==="
if [ ! -d uni2ts ]; then
    git clone https://github.com/SalesforceAIResearch/uni2ts.git
    cd uni2ts
    # Pin to a known-good commit
    # git checkout <COMMIT>
    pip install -e ".[notebook]"
    cd ..
else
    echo "uni2ts already exists, skipping"
fi

echo "=== Installing poly-precond ==="
cd "$REPO_ROOT"
pip install -e ".[dev]"

echo "=== Done ==="
