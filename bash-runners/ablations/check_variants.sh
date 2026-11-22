#!/usr/bin/env bash
# Step 1 of docs/ABLATIONS.md, before any agent trial: build an experiment's task variants (scripts/ablations/
# make_variants.py: tasks/<base>-<condition>-<mode>, each checked against the DIGEST we validated) and check every one
# of them through Harbor on this GPU machine: the reference solution (oracle) must score 1 and doing nothing (nop) 0,
# with the task's determinism check (scripts/ablations/check_variants.py). The report, jobs/ablations/checks/<E>.json
# (and <E>.md), is what bash-runners/ablations/run_<E>.sh asks before it starts: an experiment runs only once every one
# of its variants passed. Send the reports to us before the agent runs (bash-runners/ablations/send_checks.sh).
#   bash-runners/ablations/check_variants.sh all            # E1 E2 E3 E4 E5, in that order
#   bash-runners/ablations/check_variants.sh E1 [E2 ...]    # some of them
# Running it again redoes only what has not passed. Env (defaults): GPUS (every GPU nvidia-smi lists), PER_GPU (4:
# rigid-task trials per GPU; a deformable task's trial has a GPU alone), HARBOR (harbor).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
[ $# -ge 1 ] || { sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d'; exit 2; }
exps=("$@"); [ "$1" = all ] && exps=(E1 E2 E3 E4 E5)
cd "$root"
mkdir -p jobs
rc=0
for e in "${exps[@]}"; do
  case "$e" in E1|E2|E3|E4|E5) ;; *) echo "unknown experiment $e (E1 .. E5, or all)" >&2; exit 2 ;; esac
  echo "== $e: build the variants"
  python3 scripts/ablations/make_variants.py build --experiment "$e" > "jobs/ablation-build-$e.txt" 2>&1 \
    || { mkdir -p jobs; cat "jobs/ablation-build-$e.txt" >&2; echo "$e: the variants could not be built as validated (above); send us jobs/ablation-build-$e.txt" >&2; rc=1; continue; }
  echo "   $(grep -c ' ok$' "jobs/ablation-build-$e.txt") variant(s) built with the DIGEST we validated, $(grep -c ' new$' "jobs/ablation-build-$e.txt") without a recorded one"
  echo "== $e: oracle and nop on every variant"
  python3 scripts/ablations/check_variants.py run "$e" ${GPUS:+--gpus "$GPUS"} --per-gpu "${PER_GPU:-4}" --harbor "${HARBOR:-harbor}" || rc=1
done
for e in "${exps[@]}"; do
  if python3 scripts/ablations/check_variants.py gate "$e" 2>/dev/null; then echo "$e: ALL PASS (jobs/ablations/checks/$e.json)"
  else echo "$e: NOT PASSED (jobs/ablations/checks/$e.md lists what failed)"; rc=1; fi
done
exit $rc
