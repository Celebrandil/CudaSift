#!/bin/bash
# Run the tests under compute-sanitizer: memcheck (out-of-bounds and invalid accesses,
# leaks) and racecheck (shared memory races). Run from the repository root:
#   scripts/run_sanitizers.sh [build directory]
# Skipped, with exit code 0, if compute-sanitizer or an NVIDIA GPU is missing. Fails on
# any error, leak or hazard that the sanitizers report. A test that fails under the
# sanitizers, e.g. a timing check, is reported but does not fail the script. Racecheck
# takes several minutes.

BIN_DIR=${1:-build}
if [ ! -x "${BIN_DIR}/test_extract" ]; then
  echo "Error: ${BIN_DIR}/test_extract not found. Build first, and run from the repository root."
  exit 1
fi

SANITIZER=$(command -v compute-sanitizer)
for dir in "${CUDA_HOME}" /usr/local/cuda; do
  if [ -z "${SANITIZER}" ] && [ -n "${dir}" ] && [ -x "${dir}/bin/compute-sanitizer" ]; then
    SANITIZER="${dir}/bin/compute-sanitizer"
  fi
done
if [ -z "${SANITIZER}" ]; then
  echo "compute-sanitizer not found (CUDA toolkit): sanitizer checks skipped."
  exit 0
fi
if ! nvidia-smi -L > /dev/null 2>&1; then
  echo "No NVIDIA GPU found: sanitizer checks skipped."
  exit 0
fi

FAILED=""
run() {   # run <tool> <test> <expected summary> [options]; skipped if the test was not built
  local tool=$1 test=$2 clean=$3
  shift 3
  [ -x "${BIN_DIR}/${test}" ] || return
  local log="${BIN_DIR}/${tool}_${test}.log"
  echo ""
  echo ">>> ${tool}: ${test}"
  "${SANITIZER}" --tool "${tool}" "$@" "${BIN_DIR}/${test}" 0 > "${log}" 2>&1
  grep "SUMMARY" "${log}"
  grep -E "\[FAIL\]" "${log}" | sed "s/^/  (test check, ignored)/"
  if ! grep -qE "${clean}" "${log}" || grep -qE "LEAK SUMMARY: [1-9]" "${log}"; then
    FAILED="${FAILED} ${tool}:${test}"
  fi
}

for test in test_extract test_match test_homography; do
  run memcheck ${test} "ERROR SUMMARY: 0 errors" --leak-check full
done
for test in test_extract test_match; do
  run racecheck ${test} "RACECHECK SUMMARY: 0 hazards"
done

echo ""
if [ -n "${FAILED}" ]; then
  echo "Sanitizer errors in:${FAILED} (logs in ${BIN_DIR}/<tool>_<test>.log)"
  exit 1
fi
echo "No sanitizer errors."
