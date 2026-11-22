#!/usr/bin/env bash
# One lane of an ablation experiment (bash-runners/ablations/run_experiment.sh starts them): scripts/harbor/run_agent_batch.sh
# over the experiment's batches in order, with the environment it was given. A lane goes on to the next batch once no
# trial of this one is left to start (the runner ends then); a lane the runner stopped (STOP lane <pid> in the batch's
# log: a model-account failure, a usage gateway that cannot be restarted) stops here.
#   lane.sh "<batch> [<batch> ...]" <base> [<base> ...]
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
[ $# -ge 2 ] || { sed -n '2,/^set -uo pipefail$/p' "$0" | sed '$d'; exit 2; }
read -ra batches <<< "$1"; shift
for b in "${batches[@]}"; do
  "$root/scripts/harbor/run_agent_batch.sh" "$b" "$@" &
  p=$!
  wait "$p"
  if grep -q "STOP lane $p:" "${JOBS:-$root/jobs/agents}/$b.log" 2>/dev/null; then
    echo "lane: the runner $p stopped this lane in batch $b (see ${JOBS:-$root/jobs/agents}/$b.log); the later batches are not started"
    exit 1
  fi
done
echo "lane: batches ${batches[*]} have no trial left to start"
