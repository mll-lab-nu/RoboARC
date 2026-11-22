#!/usr/bin/env bash
# Send a finished ablation experiment back (docs/ABLATIONS.md): scripts/send_results.sh for each of its
# batches (one per repeat), the same way as every other batch (docs/COLLABORATOR_SETUP.md, section 6), but to the
# ablations' own public dataset RoboARC/RoboARC-ablation-outputs (scripts/ablations/experiments.py RESULTS_DATASET),
# into <name>/<batch>/<bundle id>/. Each bundle carries <RESULTS>/<batch>/ablation.json (the experiment, the repeat and
# every planned trial with its condition) and ablation_checks.json (the variant check the experiment ran after),
# written when the experiment started; MANIFEST.md's notes line names the experiment and the repeat, before your NOTES.
#   JOBS=$PWD/jobs/agents RESULTS=/path/outside/the/repo ENV_FILE=$PWD/.env CREATE_PR=1 NOTES="<reruns, outages; or none>" \
#     bash-runners/ablations/send_results.sh E1 <your name, lowercase>
# SMOKE=1: the experiment's smoke batch instead (ablation-smoke-<e>, JOBS default <repo>/jobs/smoke).
# Every setting of scripts/send_results.sh passes through (HF_TOKEN_FILE, CODEX_FORCE_AUTH_JSON, DRY_RUN, ...); HF_DATASET
# overrides the dataset.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
[ $# -eq 2 ] || { sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d'; exit 2; }
E="$1"; name="$2"
case "$E" in E1|E2|E3|E4|E5) ;; *) echo "unknown experiment $E (E1 .. E5)" >&2; exit 2 ;; esac
RESULTS="${RESULTS:-$(dirname "$root")/rcb_results}"
X="python3 $root/scripts/ablations/experiments.py"
HF_DATASET="${HF_DATASET:-$($X dataset "$E")}"
if [ "${SMOKE:-0}" = 1 ]; then
  batches=("ablation-smoke-${E,,}"); export JOBS="${JOBS:-$root/jobs/smoke}"; what="smoke"
else
  mapfile -t batches < <($X batches "$E"); what="repeat"
fi
n=${#batches[@]}; k=0; rc=0
for b in "${batches[@]}"; do
  k=$((k + 1))
  [ -f "$RESULTS/$b/ablation.json" ] || { echo "$RESULTS/$b/ablation.json is missing: was $b run with bash-runners/ablations/run_$E.sh and this RESULTS?" >&2; rc=1; continue; }
  echo "=== $E, $what $k of $n: batch $b, to $HF_DATASET"
  HF_DATASET="$HF_DATASET" NOTES="ablation $E ($($X field "$E" purpose)), $what $k of $n; ${NOTES:-none}" \
    RESULTS="$RESULTS" "$root/scripts/send_results.sh" "$b" "$name" || rc=1
  [ $k -lt $n ] && sleep 61    # send_results.sh names a bundle by the minute: one batch per minute
done
exit $rc
