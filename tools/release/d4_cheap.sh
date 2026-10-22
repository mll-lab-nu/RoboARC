#!/bin/bash
# One base's appearance-seed cheap checks (tools/release/d4_cheap.py) in one container of the release image:
#   tools/release/d4_cheap.sh <image> <render tree> <base> <out dir> [gpu index]
#   <render tree>: the release render tree (tasks/roboarc-<base>-privileged/tests, scripts/roboarc/oracles/<base>)
set -u
img=$1 tree=$2 base=$3 out=$4 gpu=${5:-}
H=$(cd "$(dirname "$0")" && pwd)
t=$tree/tasks/roboarc-$base-privileged/tests
fam=$(sed -n 's/.*--family \([a-z0-9_]*\).*/\1/p' "$t/test.sh" | head -1)
mkdir -p "$out"
dev=(-e CUDA_VISIBLE_DEVICES=); [ -n "$gpu" ] && dev=(--gpus "device=$gpu" -e NVIDIA_DRIVER_CAPABILITIES=all)
orc=$tree/scripts/roboarc/oracles/$base; mkdir -p "$out/empty"; [ -d "$orc" ] || orc=$out/empty
docker run --rm --name "s1d4-$base" --label qn-cleanup=S1 --memory 16g --cpus 2 --network none "${dev[@]}" \
  -e NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR -e OPENBLAS_CORETYPE=Haswell \
  -e RCB_RENDER=off -e PYTHONPATH=/t:/opt/topogym -v "$t:/t:ro" -v "$orc:/orc:ro" -v "$H:/r:ro" "$img" \
  python /r/d4_cheap.py "$base" "$fam" --tests /t --oracles /orc > "$out/$base.log" 2>&1
echo "rc=$?" >> "$out/$base.log"
