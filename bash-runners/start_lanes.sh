#!/usr/bin/env bash
# Start the evaluation run's lanes (docs/COLLABORATOR_SETUP.md, section 5) for the seeds in SEEDS, with that section's
# command: K=3, the usage gateway on with GATEWAY_RAW=0, 4 lanes per GPU, each lane in the background (nohup).
# run_seed0.sh, run_seed1.sh and run_seed2.sh call it with SEEDS=0, 1 and 2:
#   MODEL=openai/<model> EFFORT=high RESULTS=/path/outside/the/repo bash-runners/run_seed0.sh <batch name> [<base> ...]
# Start the next seed's script, with the same settings and batch name, once every lane of the previous one has ended
# (each lane's output file then ends with "runner <pid> of batch <batch> finished"; seed 0:
# tail -qn1 jobs/<batch>.seeds-0.lane*.out): the three runs make one batch of three trials per task and mode. Lanes of
# the next seed started earlier run beside the earlier ones, without waiting for that seed's trials. To restart lanes
# (after a reboot, or a lane stopped by a model-account failure), run the same script again: finished trials are skipped.
# Env (defaults):
#   MODEL     the agent's model, e.g. openai/<model> (required)
#   EFFORT    the reasoning level agreed with us, e.g. high (required; never `default`)
#   RESULTS   a directory outside the repository for the per-trial results (required)
#   AGENT     codex | claude-code | dsh (codex)
#   ENV_FILE  the credentials file (<repo>/.env, a copy of .env.example)
#   JOBS      <repo>/jobs/agents; each lane's output: <JOBS>/../<batch>.seeds-<seed>.lane<k>.out
#   LANES     4 per GPU: of GPUS when it is set (a host shared with other work), else of every GPU nvidia-smi lists
# Every other setting of scripts/harbor/run_agent_batch.sh passes through as it is (MODES, GPUS, MEM_GUARD,
# AGENT_CONFIG, CODEX_FORCE_AUTH_JSON, GATEWAY_EXTRA_HOSTS, ...). The <base>s: the task bases to run (default: every
# base in tasks/, 90).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
[ $# -ge 1 ] || { sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d'; exit 2; }
batch="$1"; shift
AGENT="${AGENT:-codex}"; MODEL="${MODEL:-}"; EFFORT="${EFFORT:-}"; RESULTS="${RESULTS:-}"; SEEDS="${SEEDS:-}"
ENV_FILE="${ENV_FILE:-$root/.env}"; JOBS="${JOBS:-$root/jobs/agents}"
[ -n "$SEEDS" ] || { echo "SEEDS must be set: run bash-runners/run_seed0.sh, run_seed1.sh or run_seed2.sh" >&2; exit 2; }
[ -n "$MODEL" ] || { echo "MODEL must be set, e.g. MODEL=openai/<model>" >&2; exit 2; }
[ -n "$EFFORT" ] && [ "$EFFORT" != default ] || { echo "EFFORT must be set to the reasoning level agreed with us, e.g. EFFORT=high (never default)" >&2; exit 2; }
[ -n "$RESULTS" ] || { echo "RESULTS must be set: a directory outside the repository" >&2; exit 2; }
[ -f "$ENV_FILE" ] || { echo "ENV_FILE=$ENV_FILE: no such file (copy .env.example and fill in your credentials)" >&2; exit 2; }
if [ -z "${LANES:-}" ]; then
  if [ -n "${GPUS:-}" ]; then gpus=$(tr , '\n' <<< "$GPUS" | wc -l); else gpus=$(nvidia-smi -L | wc -l); fi
  LANES=$((4 * gpus))
fi
[[ "$LANES" =~ ^[1-9][0-9]*$ ]] || { echo "LANES must be a positive integer" >&2; exit 2; }
if [ $# -gt 0 ]; then bases=("$@"); else mapfile -t bases < <(ls "$root/tasks" | sed -nE 's/-privileged$//p'); fi
out="$(dirname "$JOBS")"
mkdir -p "$out"
pids=()
for k in $(seq 1 "$LANES"); do
  AGENT="$AGENT" MODEL="$MODEL" EFFORT="$EFFORT" K=3 SEEDS="$SEEDS" ENV_FILE="$ENV_FILE" \
    USAGE_GATEWAY="$root/scripts/usage_gateway" GATEWAY_RAW=0 JOBS="$JOBS" RESULTS="$RESULTS" \
    nohup "$root/scripts/harbor/run_agent_batch.sh" "$batch" "${bases[@]}" > "$out/$batch.seeds-${SEEDS// /-}.lane$k.out" 2>&1 &
  pids+=($!)
done
echo "batch $batch, SEEDS=\"$SEEDS\": $LANES lanes started over ${#bases[@]} task bases (PIDs ${pids[*]})"
echo "progress: $JOBS/$batch.log; lane output: $out/$batch.seeds-${SEEDS// /-}.lane<k>.out"
