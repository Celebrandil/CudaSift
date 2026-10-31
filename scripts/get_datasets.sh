#!/bin/bash
# Download HPatches (hpatches-sequences-release, 116 sequences; 1.3 GB download, 1.8 GB
# unpacked) for eval_hpatches into <dir>/hpatches-sequences-release:
#   scripts/get_datasets.sh <dir>
# From Hugging Face (the original server, icvl.ee.ic.ac.uk, no longer answers). Skipped if
# already complete. Needs curl and unzip.
set -euo pipefail

dir=${1:-}
if [ -z "$dir" ]; then
  echo "Usage: $0 <dir>"
  exit 1
fi
mkdir -p "$dir"
out="$dir/hpatches-sequences-release"

# Number of sequences with images 1.ppm..6.ppm and homographies H_1_2..H_1_6
complete() {
  local n=0 seq k ok
  for seq in "$out"/*/; do
    [ -d "$seq" ] || continue
    ok=1
    for k in 1 2 3 4 5 6; do
      [ -f "$seq$k.ppm" ] || ok=0
    done
    for k in 2 3 4 5 6; do
      [ -f "${seq}H_1_$k" ] || ok=0
    done
    n=$((n + ok))
  done
  echo $n
}

if [ -d "$out" ] && [ "$(complete)" = 116 ]; then
  echo "HPatches: already in $out"
  exit 0
fi
zip="$dir/hpatches-sequences-release.zip"
url1=https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip
url2=https://huggingface.co/datasets/honggyuAn/hpatches-sequences-release/resolve/main/hpatches-sequences-release.zip
echo "HPatches: downloading (1.3 GB)"
curl -fL --retry 3 -o "$zip" "$url1" || curl -fL --retry 3 -o "$zip" "$url2"
unzip -q -o "$zip" -d "$dir"
rm "$zip"
n=$(complete)
echo "HPatches: $n of 116 sequences in $out"
[ "$n" = 116 ]
