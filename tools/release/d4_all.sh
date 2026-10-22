#!/bin/bash
# tools/release/d4_cheap.sh over all bases of a render tree (a checkout whose tasks/ also holds the --experimental dirs): CPU bases LANES at a time, blanket (GPU physics) on GPU GPU one at a time; MemAvailable
# >= 25 GiB re-read before each base. d4_all.sh <image> <render tree> <out dir>  (env LANES=10 GPU=0 BASES="...")
set -u
H=$(cd "$(dirname "$0")" && pwd); img=$1 tree=$2 out=$3; LANES=${LANES:-10}; GPU=${GPU:-0}
waitmem() { while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 25 ]; do sleep 30; done; }
export -f waitmem
all=${BASES:-$(ls $tree/tasks | sed -n 's/^roboarc-\(.*\)-privileged$/\1/p' | grep -v '^ikea-')}
cpu=$(printf "%s\n" $all | grep -v '^blanket-'); gpu=$(printf "%s\n" $all | grep '^blanket-')
( for b in $gpu; do waitmem; $H/d4_cheap.sh $img $tree $b $out $GPU; done ) &
printf "%s\n" $cpu | xargs -P $LANES -I{} bash -c "waitmem; $H/d4_cheap.sh $img $tree {} $out"
wait
python3 $H/d4_table.py $out > $out/TABLE.md
echo D4_ALLDONE
