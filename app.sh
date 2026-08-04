#!/usr/bin/env bash
#
# app.sh — launch the licensure risk-screening web app.
#
#   ./app.sh                 # default port 8501
#   PORT=8502 ./app.sh       # if 8501 is taken
#
# Requires the environment and a trained model bundle, both produced by
# ./run.sh. If either is missing, run ./run.sh first.
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="ml_env/bin/python"
PORT="${PORT:-8501}"
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
    echo "No trained model found ($BUNDLE_REAL or $BUNDLE_SYNTHETIC)."
    echo "Run ./run.sh first."
    exit 1
fi

# A bundle exported before a schema change fails at upload time with a confusing
# column error rather than at startup, so check the match up front.
if ! "$PYTHON_BIN" - <<'PY'
import sys
sys.path.insert(0, ".")
import prediction, schema
try:
    bundle = prediction.load_bundle(prediction.resolve_bundle_path())
except Exception as exc:
    print(f"Could not read the model bundle: {exc}")
    sys.exit(1)
expected = set(schema.NUMERIC_COLS) | set(schema.CATEGORICAL_COLS)
actual = set(prediction.required_raw_columns(bundle))
if expected != actual:
    print("The trained model does not match the current feature schema.")
    print(f"  missing from bundle : {sorted(expected - actual) or 'none'}")
    print(f"  unexpected in bundle: {sorted(actual - expected) or 'none'}")
    print("Re-run ./run.sh to export a matching bundle.")
    sys.exit(1)
PY
then
    exit 1
fi

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port ${PORT} is already in use. Either stop the other app, or run:"
    echo "    PORT=8502 ./app.sh"
    exit 1
fi

# Streamlit's startup is slow on macOS (its dependency tree takes 10-30s to
# import on a cold cache) and it prints nothing until ready. Without this
# warning the silence reads as a hang and people interrupt a working launch.
cat <<EOF

Starting the screening app on http://localhost:${PORT}

  First start takes 10 to 30 seconds while Streamlit loads.
  Your browser should open on its own. If it does not, open the link above.
  Press Ctrl+C in this window to stop the app.

EOF

# A clean message instead of a Python traceback when the user stops the app or
# interrupts it mid-startup.
trap 'echo; echo "App stopped."; exit 0' INT TERM

exec "$PYTHON_BIN" -m streamlit run app/app.py --server.port "$PORT"
