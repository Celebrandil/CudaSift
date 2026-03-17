#!/bin/bash
# Run all benchmarks and tests

set -e

# Find executable directory
if [ -d "build/Release" ]; then
  BIN_DIR="build/Release"
elif [ -d "build" ]; then
  BIN_DIR="build"
else
  echo "Error: Build directory not found. Run scripts/build.sh first."
  exit 1
fi

echo "============================================="
echo "  CudaSift Test & Benchmark Suite"
echo "============================================="

# Run tests
echo ""
echo ">>> Running extraction tests..."
echo ""
"${BIN_DIR}/test_extract" 0
TEST1=$?

echo ""
echo ">>> Running matching tests..."
echo ""
"${BIN_DIR}/test_match" 0
TEST2=$?

echo ""
echo ">>> Running homography tests..."
echo ""
"${BIN_DIR}/test_homography" 0
TEST3=$?

echo ""
echo ">>> Running performance benchmark..."
echo ""
"${BIN_DIR}/benchmark" 0 200

echo ""
echo "============================================="
echo "  All Tests & Benchmarks Complete"
echo "============================================="

TOTAL_FAIL=$((TEST1 + TEST2 + TEST3))
if [ $TOTAL_FAIL -eq 0 ]; then
  echo "  All tests PASSED"
else
  echo "  Some tests FAILED (exit codes: $TEST1, $TEST2, $TEST3)"
fi
