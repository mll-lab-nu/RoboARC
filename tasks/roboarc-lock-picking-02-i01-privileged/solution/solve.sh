#!/bin/bash
# Reference solution (privileged): the trajectory the author's oracle produced (README.md, Reference solution),
# one file per appearance seed: the same rows, stamped with that seed's birth (instance_sha). This copies the file
# whose stamp is the birth (state_sha) of the instance staged in /app (the runner's seed; no seed file: seed 0).
# Uses nothing but the recorded files and that birth.
set -euo pipefail
mkdir -p /app/output
birth=$(python -c 'import json; print(json.load(open("/app/instance.json"))["state_sha"])')
case "$birth" in
  62f546171ad1a5a5) f=trajectory.npz ;;
  4f7e2745e7ef0626) f=trajectory.s1.npz ;;
  61550b97c4021c5c) f=trajectory.s2.npz ;;
  *) echo "solve.sh: no reference for the staged instance (birth $birth)" >&2; exit 1 ;;
esac
cp "/solution/$f" /app/output/trajectory.npz
