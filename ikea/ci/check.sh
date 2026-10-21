#!/usr/bin/env bash
# The IKEA MuJoCo checks (ikea/README.md): both generators' check modes (no simulator needed), and with --image-tests
# the unit tests in the base image (network off, the checkout read-only). Run from anywhere.
#   ikea/ci/check.sh [--image-tests]
set -uo pipefail
I="$(cd "$(dirname "$0")/.." && pwd)"; rc=0
cd "$I" && umask 022
for g in "scripts/ikea_mujoco/mk_tasks.py --check" "scripts/ikea_mujoco/mk_standard_tasks.py --check"; do
  if out=$(python3 $g 2>&1); then echo "ok $g: ${out##*$'\n'}"; else echo "FAIL $g:"; echo "$out" | tail -20; rc=1; fi
done
if [ "${1:-}" = "--image-tests" ]; then
  IMG="ghcr.io/mll-lab-nu/roboarc-ikea:$(cat images/ikea-mujoco/VERSION)"
  for t in scripts/ikea_mujoco/tests/test_grade.py scripts/ikea_mujoco/tests/test_nyx_startup.py scripts/ikea_mujoco/tests/test_standard_grade.py; do
    if out=$(timeout 1800 docker run --rm --network none --cpus 4 --memory 8g -v "$I:/repo:ro" -w /repo -e PYTHONDONTWRITEBYTECODE=1 "$IMG" python "$t" 2>&1); then
      echo "ok $t: ${out##*$'\n'}"; else echo "FAIL $t:"; echo "$out" | tail -20; rc=1; fi
  done
fi
exit $rc
