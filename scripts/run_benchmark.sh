#!/bin/bash
# Run all tests, then the benchmark
# Usage, from the repository root (the tests read data/):
#   bash scripts/run_benchmark.sh [build directory, default: build]

set -e

BUILD_DIR="${1:-build}"

if [ ! -f "data/img1.png" ]; then
  echo "Error: data/img1.png not found. Run this script from the repository root."
  exit 1
fi

# Find executable directory
if [ -d "${BUILD_DIR}/Release" ]; then
  BIN_DIR="${BUILD_DIR}/Release"
elif [ -d "${BUILD_DIR}" ]; then
  BIN_DIR="${BUILD_DIR}"
else
  echo "Error: Build directory ${BUILD_DIR} not found. Run scripts/build.sh first."
  exit 1
fi

echo "============================================="
echo "  CudaSift Test & Benchmark Suite"
echo "============================================="

# Run the tests that were built (not every configuration builds all of them)
FAILED=""
for TEST in test_extract test_match test_homography test_homography_gpu test_batch; do
  if [ -x "${BIN_DIR}/${TEST}" ]; then
    echo ""
    echo ">>> Running ${TEST}..."
    echo ""
    "${BIN_DIR}/${TEST}" 0 || FAILED="${FAILED} ${TEST}"
  fi
done

echo ""
echo ">>> Running performance benchmark..."
echo ""
"${BIN_DIR}/benchmark" 0 200

echo ""
echo "============================================="
echo "  All Tests & Benchmarks Complete"
echo "============================================="

if [ -z "${FAILED}" ]; then
  echo "  All tests PASSED"
else
  echo "  Some tests FAILED:${FAILED}"
  exit 1
fi
