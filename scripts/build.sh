#!/bin/bash
# Build script for CudaSift (Ada Lovelace / RTX 4060 Ti)

set -e

BUILD_DIR="build"
BUILD_TYPE="${1:-Release}"

echo "============================================="
echo "  CudaSift Build Script"
echo "  Target: Ada Lovelace (sm_89)"
echo "  Build Type: ${BUILD_TYPE}"
echo "============================================="

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
cmake .. \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DBUILD_TESTS=ON \
  -DBUILD_EXAMPLES=ON \
  -DVERBOSE_OUTPUT=ON

# Build
cmake --build . --config "${BUILD_TYPE}" -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "Build complete!"
echo "Executables in: ${BUILD_DIR}/"
echo ""
echo "Available targets:"
echo "  cudasift        - Main demo"
echo "  demo_extract    - Feature extraction demo"
echo "  demo_match      - Feature matching demo"
echo "  demo_video      - Real-time video demo"
echo "  benchmark       - Performance benchmark"
echo "  test_extract    - Extraction tests"
echo "  test_match      - Matching tests"
echo "  test_homography - Homography tests"
