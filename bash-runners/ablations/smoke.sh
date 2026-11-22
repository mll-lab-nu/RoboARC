#!/usr/bin/env bash
# The smoke of one ablation experiment (docs/ABLATIONS.md, step 2): one trial of each variant of the
# experiment's first task (every condition, and for E4 both modes), all at once, in the batch ablation-smoke-<e> under
# jobs/smoke, with exactly the settings of bash-runners/ablations/run_<E>.sh. It checks the credentials, the network and
# the conditions' new interfaces with the real agent; its trials are not part of the experiment. Ten minutes in:
# python3 scripts/ablations/smoke_check.py ablation-smoke-<e>.
#   MODEL=<as for run_<E>.sh> RESULTS=/path/outside/the/repo bash-runners/ablations/smoke.sh E1
set -euo pipefail
[ $# -eq 1 ] || { sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d'; exit 2; }
SMOKE=1 STAGGER="${STAGGER:-0}" exec "$(dirname "$0")/run_experiment.sh" "$1"
