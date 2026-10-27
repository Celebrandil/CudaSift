#!/bin/bash
# Check the build options that the default build doesn't exercise. Run from the
# repository root:
#   scripts/check_builds.sh
# - CMAKE_CUDA_ARCHITECTURES and CUDAARCHS select the architecture of the local GPU,
#   and test_extract runs on it.
# - USE_MANAGED_MEM=ON builds the library, cudasift, benchmark and test_homography
#   (test_extract, test_match and the examples use SiftData::h_data, which managed
#   memory replaces with m_data), and test_homography passes.
# Skipped, with exit code 0, without nvcc or an NVIDIA GPU. Builds in build-check-*.

NVCC=$(command -v nvcc)
for dir in "${CUDA_HOME}" /usr/local/cuda; do
  if [ -z "${NVCC}" ] && [ -n "${dir}" ] && [ -x "${dir}/bin/nvcc" ]; then
    NVCC="${dir}/bin/nvcc"
  fi
done
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [ -z "${NVCC}" ] || [ -z "${ARCH}" ]; then
  echo "nvcc or an NVIDIA GPU not found: build checks skipped."
  exit 0
fi

FAILED=""
fail() { FAILED="${FAILED} $1"; echo "    FAILED: $1"; }

configure() {   # configure <build dir> [cmake options]
  local dir=$1
  shift
  cmake -S . -B "${dir}" -DCMAKE_BUILD_TYPE=Release -DVERBOSE_OUTPUT=OFF -DCMAKE_CUDA_COMPILER="${NVCC}" "$@" > "${dir}.log" 2>&1
}

for mode in option env; do
  echo ""
  echo ">>> Architecture sm_${ARCH} with the ${mode}"
  dir=build-check-arch-${mode}
  if [ ${mode} = option ]; then
    configure ${dir} -DCMAKE_CUDA_ARCHITECTURES=${ARCH}
  else
    CUDAARCHS=${ARCH} configure ${dir}
  fi || { fail "configure-arch-${mode}"; continue; }
  grep -q "sm_${ARCH}" ${dir}/CMakeFiles/cudasift_lib.dir/flags.make || fail "arch-${mode}: not sm_${ARCH}"
  cmake --build ${dir} -j4 --target test_extract >> ${dir}.log 2>&1 || { fail "build-arch-${mode}"; continue; }
  ${dir}/test_extract 0 > ${dir}/test_extract.log 2>&1 || fail "test_extract-arch-${mode}"
  grep "Results" ${dir}/test_extract.log
done

echo ""
echo ">>> USE_MANAGED_MEM=ON"
dir=build-check-managed
if configure ${dir} -DCMAKE_CUDA_ARCHITECTURES=${ARCH} -DUSE_MANAGED_MEM=ON -DBUILD_EXAMPLES=OFF; then
  for target in cudasift_lib cudasift benchmark test_homography; do
    cmake --build ${dir} -j4 --target ${target} >> ${dir}.log 2>&1 || fail "managed-build-${target}"
  done
  if [ -x ${dir}/test_homography ]; then
    ${dir}/test_homography 0 > ${dir}/test_homography.log 2>&1 || fail "managed-test_homography"
    grep "Results" ${dir}/test_homography.log
  fi
else
  fail "configure-managed"
fi

echo ""
if [ -n "${FAILED}" ]; then
  echo "Build checks failed:${FAILED} (logs in build-check-*.log)"
  exit 1
fi
echo "Build checks passed."
