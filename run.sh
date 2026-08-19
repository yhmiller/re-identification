#!/usr/bin/env bash
#
# run.sh — one command to run the whole analysis.
#
#   ./run.sh
#
# Sets up the Python environment the first time (downloads dependencies),
# then runs the five analysis stages and rebuilds every manuscript figure and
# table. Outputs are written to results/disclosure/. Safe to re-run: setup is
# skipped once the environment exists.
#
set -euo pipefail
cd "$(dirname "$0")"

VENV="ml_env"
PYTHON_BIN="$VENV/bin/python"
# sdv 1.11 requires Python <3.12; 3.11 satisfies every pinned dependency.
REQUIRED_SERIES="3.11"

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
        echo "Then run ./run.sh again."
        exit 1
    fi
    echo "Using $BASE_PYTHON to create $VENV ..."
    "$BASE_PYTHON" -m venv "$VENV"
    "$PYTHON_BIN" -m pip install --quiet --upgrade pip
    echo "Installing dependencies (this can take a few minutes)..."
    "$PYTHON_BIN" -m pip install --quiet -r requirements.txt
    echo "Environment ready."
fi

echo "Running the full analysis — outputs will appear in results/ ..."
"$PYTHON_BIN" run_all.py "$@"
