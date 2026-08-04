#!/usr/bin/env bash
#
# app.sh — launch the educator risk-screening web app.
#
#   ./app.sh
#
# Opens in your browser. Requires the environment and a trained model bundle,
# both produced by ./run.sh. If either is missing, run ./run.sh first.
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="ml_env/bin/python"
# Results are split by dataset source (results/real/ once real data has been
# run, results/synthetic/ until then) — same preference order as
# prediction.resolve_bundle_path(), which is what the app itself uses.
BUNDLE_REAL="results/real/inference_bundle.pkl"
BUNDLE_SYNTHETIC="results/synthetic/inference_bundle.pkl"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Environment not set up yet. Run ./run.sh first."
    exit 1
fi
if [ ! -f "$BUNDLE_REAL" ] && [ ! -f "$BUNDLE_SYNTHETIC" ]; then
    echo "No trained model found ($BUNDLE_REAL or $BUNDLE_SYNTHETIC). Run ./run.sh first."
    exit 1
fi

echo "Starting the screening app - it will open in your browser..."
"$PYTHON_BIN" -m streamlit run app/app.py
