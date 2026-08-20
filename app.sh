#!/usr/bin/env bash
#
# app.sh — launch the Release Risk Advisor.
#
#   ./app.sh              open on the default port
#   ./app.sh 8600         open on a specific port
#
# Sets up the Python environment the first time, the same way run.sh does, then
# starts Streamlit. Stop it with Ctrl-C.
#
set -euo pipefail
cd "$(dirname "$0")"

VENV="ml_env"
PYTHON_BIN="$VENV/bin/python"
STREAMLIT_BIN="$VENV/bin/streamlit"
REQUIRED_SERIES="3.11"
PORT="${1:-8501}"

find_python() {
    for candidate in "python$REQUIRED_SERIES" \
                     "/opt/homebrew/bin/python$REQUIRED_SERIES" \
                     "/usr/local/bin/python$REQUIRED_SERIES"; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if [ ! -x "$PYTHON_BIN" ]; then
    echo "First run: setting up the Python environment (one time only)..."
    if ! BASE_PYTHON="$(find_python)"; then
        echo "ERROR: Python $REQUIRED_SERIES is required but was not found."
        echo "On macOS, install it with:  brew install python@$REQUIRED_SERIES"
        exit 1
    fi
    "$BASE_PYTHON" -m venv "$VENV"
    "$PYTHON_BIN" -m pip install --quiet --upgrade pip
    echo "Installing dependencies (this can take a few minutes)..."
    "$PYTHON_BIN" -m pip install --quiet -r requirements.txt
    echo "Environment ready."
fi

# The Study findings tab reads results/disclosure/. Say so here rather than
# letting the tab explain an empty screen after the browser has already opened.
if [ ! -f "results/disclosure/risk_utility_frontier.csv" ]; then
    echo
    echo "Note: no analysis results found in results/disclosure/."
    echo "      The 'Assess a release' tab still works; 'Study findings' will be"
    echo "      empty until you run ./run.sh."
    echo
fi

echo "Starting the Release Risk Advisor on http://localhost:$PORT"
echo "Press Ctrl-C to stop."
echo
exec "$STREAMLIT_BIN" run app/app.py --server.port "$PORT"
