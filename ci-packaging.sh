#!/bin/bash
set -e

echo "=========================================="
echo "Building and Publishing Package"
echo "=========================================="
echo "${ENVIRONMENT}"

# Check environment variable to determine interactive mode
if [ "${ENVIRONMENT}" = "dev" ]; then
    echo "Running in dev mode (interactive)"
    INTERACTIVE_FLAG=""
else
    echo "Running in non-dev mode (non-interactive)"
    INTERACTIVE_FLAG="--non-interactive"
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build package
echo "Building package..."
python -m build

# Publish to PyPI (uses PYPI_TOKEN from environment)
echo "Publishing to PyPI..."
python -m twine upload dist/* ${INTERACTIVE_FLAG}

echo ""
echo "✓ Package published successfully!"
