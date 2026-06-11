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
BUNDLE="results/inference_bundle.pkl"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Environment not set up yet. Run ./run.sh first."
    exit 1
fi
if [ ! -f "$BUNDLE" ]; then
    echo "No trained model found ($BUNDLE). Run ./run.sh first."
    exit 1
fi

echo "Starting the screening app — it will open in your browser..."
"$PYTHON_BIN" -m streamlit run app/app.py
