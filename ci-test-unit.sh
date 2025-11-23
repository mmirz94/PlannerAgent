#!/bin/bash
set -e

echo "=========================================="
echo "Running Unit Tests"
echo "=========================================="

# Run tests
python -m unittest tests

echo ""
echo "All tests passed!"
