#!/bin/bash
# Record one -i01 reference of a slotted_post_ring / orbits chain setting on MuJoCo (run from the repository root, in the
# RoboTangle image's Python):
#   1. make-instance (seed 0; the settling is frozen with it: mujoco_settled_state) in the gripper space the oracle
#      commands (slotted very_easy / easier and #108: native; orbits: width);
#   2. the family's oracle through rcb_topogym run-oracle (scripts/oracle_runners/*_session_oracle.py), which records
#      privileged.npz;
#   3. if it succeeded, two replays, each in a fresh process (rcb_topogym replay --repeat 1), which must end at the
#      recording's hash; then OUT/standard.npz = OUT/privileged.npz.
# usage: record_reference.sh <task> <variation> <width|native> <oracle code> <out dir> <work dir>
set -u
task=$1; var=$2; space=$3; code=$4; out=$5; work=$6
mkdir -p "$out" "$work"
if [ ! -f "$work/instance.json" ]; then
  python scripts/robotangle/shared/rcb_topogym.py make-instance --task "$task" --variation "$var" --scene-seed 0 \
    --gripper-space "$space" --out "$work/instance.json" > "$work/make.log" 2>&1 || { tail -20 "$work/make.log"; exit 1; }
fi
TOPOGYM_ROOT=$PWD SLOTTED_ORACLE_OUTPUT="$work/oracle_out" ORBITS_ORACLE_OUTPUT="$work/oracle_out" RCB_RENDER=off \
  python scripts/robotangle/shared/rcb_topogym.py run-oracle --task "$task" --instance "$work/instance.json" --code "$code" \
  --out "$out/privileged.npz" > "$work/oracle.log" 2>&1
grep -h '^{"traj' "$work/oracle.log" | tail -1 > "$work/oracle.json"
grep -q '"success": true' "$work/oracle.json" || { echo "oracle failed: $(cut -c1-300 "$work/oracle.json")"; exit 2; }
for k in 1 2; do
  RCB_RENDER=off python scripts/robotangle/shared/rcb_topogym.py replay --task "$task" --instance "$work/instance.json" \
    --traj "$out/privileged.npz" --repeat 1 > "$work/replay$k.log" 2>&1
  grep -h '^{' "$work/replay$k.log" | tail -1 > "$work/replay$k.json"
done
cp "$out/privileged.npz" "$out/standard.npz"
python - "$work" <<'PY'
import json, sys
w = sys.argv[1]
o = json.load(open(f"{w}/oracle.json"))
r = [json.load(open(f"{w}/replay{k}.json")) for k in (1, 2)]
shas = [x["replays"][0]["state_sha"] for x in r]
print(json.dumps(dict(n_actions=o["n_actions"], recorded_sha=o["state_sha"], replay_shas=shas,
                      replays_success=[x["all_success"] for x in r], same=all(s == o["state_sha"] for s in shas))))
PY
