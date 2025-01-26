#!/usr/bin/env bash
# Exit immediately if a command returns a non-zero status.
set -e

echo "===== Creating and Activating Python 3.11 Virtual Environment ====="

# 1. Check that python3.11 exists and create venv
if ! command -v python3 &> /dev/null
then
    echo "python3.11 not found. Make sure Python 3.11 is installed or adjust the command."
    exit 1
fi

python3 -m venv matchmaking

# 2. Activate the venv in the current shell
# Note: This only persists if you 'source' this script.
source matchmaking/bin/activate

# 3. Upgrade pip
pip install --upgrade pip
pip install -r ~/matchMaking/requirements.txt

# 4. Install dependencies
if [[ -f "requirements.txt" ]]; then
    pip install -r ~/matchMaking/requirements.txt
else
    echo "No requirements.txt found in the current directory."
fi

echo "===== Environment Setup Complete ====="
echo "Virtual environment: $(which python)"
echo "Python version: $(python --version)"
echo "To deactivate, run: deactivate"
