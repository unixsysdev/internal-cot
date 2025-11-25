#!/bin/bash
# Setup script for Coconut Qwen3 environment
# This script creates a Python virtual environment and installs all dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="coconut_env"
ENV_PATH="${SCRIPT_DIR}/${ENV_NAME}"

echo "=============================================="
echo "Coconut Qwen3 Environment Setup"
echo "=============================================="

# Check for Python 3
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python 3 is required but not found."
    echo "Please install Python 3.8 or later."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment at: ${ENV_PATH}"
$PYTHON_CMD -m venv "${ENV_PATH}"

# Activate the environment
echo "Activating virtual environment..."
source "${ENV_PATH}/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# Install additional requirements for Qwen3
echo ""
echo "Installing additional packages for Qwen3 support..."
pip install pyyaml accelerate

echo ""
echo "=============================================="
echo "Environment setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  source ${ENV_PATH}/bin/activate"
echo ""
echo "To run the tests:"
echo "  python test_qwen_coconut.py"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
