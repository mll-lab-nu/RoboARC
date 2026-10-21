#!/bin/bash
# Reference solution (privileged): the trajectory the author's oracle produced (README.md, Reference solution),
# one file per appearance seed: the same rows, stamped with that seed's birth (instance_sha). This copies the file
# whose stamp is the birth (state_sha) of the instance staged in /app (the runner's seed; no seed file: seed 0).
# Uses nothing but the recorded files and that birth.
set -euo pipefail
mkdir -p /app/output
birth=$(python -c 'import json; print(json.load(open("/app/instance.json"))["state_sha"])')
case "$birth" in
  14291ab24d81241e) f=trajectory.npz ;;
  2dd400d8c2dd1f96) f=trajectory.s1.npz ;;
  26d3a2c2f5827c5c) f=trajectory.s2.npz ;;
  *) echo "solve.sh: no reference for the staged instance (birth $birth)" >&2; exit 1 ;;
esac
cp "/solution/$f" /app/output/trajectory.npz
