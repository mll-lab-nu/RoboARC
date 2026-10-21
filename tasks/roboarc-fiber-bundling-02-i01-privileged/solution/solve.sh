#!/bin/bash
# Reference solution (privileged): the trajectory the author's oracle produced (README.md, Reference solution),
# one file per appearance seed: the same rows, stamped with that seed's birth (instance_sha). This copies the file
# whose stamp is the birth (state_sha) of the instance staged in /app (the runner's seed; no seed file: seed 0).
# Uses nothing but the recorded files and that birth.
set -euo pipefail
mkdir -p /app/output
birth=$(python -c 'import json; print(json.load(open("/app/instance.json"))["state_sha"])')
case "$birth" in
  44971f84830ab741) f=trajectory.npz ;;
  8d0b227be4b22ab8) f=trajectory.s1.npz ;;
  6af8c2cc268e2712) f=trajectory.s2.npz ;;
  *) echo "solve.sh: no reference for the staged instance (birth $birth)" >&2; exit 1 ;;
esac
cp "/solution/$f" /app/output/trajectory.npz
