#!/bin/bash
# Reference solution (privileged): the trajectory the author's oracle produced and the verifier scored 1
# on the evidence run named in README.md (Reference solution); for an appearance seed S, the same rows
# re-stamped with that seed's birth state (solution/trajectory.s<S>.npz). Uses nothing but the recorded files.
set -euo pipefail
mkdir -p /app/output
S=$(cat /opt/robotangle/seed 2>/dev/null || echo 0)
f=/solution/trajectory.npz; [ "$S" = 0 ] || f=/solution/trajectory.s$S.npz
test -f "$f" || { echo "no reference for seed $S" >&2; exit 1; }
cp "$f" /app/output/trajectory.npz
