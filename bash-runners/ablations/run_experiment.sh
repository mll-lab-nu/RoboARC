#!/usr/bin/env bash
# Start one ablation experiment's agent trials (docs/ABLATIONS.md); bash-runners/ablations/run_E1.sh ..
# run_E5.sh call it. It refuses to start unless the experiment's variant check passed on this machine
# (bash-runners/ablations/check_variants.sh <E>: jobs/ablations/checks/<E>.json) for the variants exactly as they are
# now.
#   MODEL=<as in your main runs> RESULTS=/path/outside/the/repo bash-runners/ablations/run_E1.sh
# Fixed by the experiment (scripts/ablations/experiments.py; not settable): the agent (Codex; E2 Claude Code), EFFORT=high,
# the tasks and conditions, MODES, the batch names (one per repeat: ablation-e<n>-r<k>), seed 0 (K=3 SEEDS=0, as
# bash-runners/run_seed0.sh), the usage gateway with GATEWAY_RAW=0 and its standard token budget, web search off, the
# tasks' own time limits. LANES lanes start in the background (nohup), STAGGER seconds apart; each lane runs
# scripts/harbor/run_agent_batch.sh over the experiment's batches in order (repeat 1, then 2, then 3), so every
# repeat-1 trial starts before any repeat-2 trial and the GPUs stay busy across the repeats. A lane that the runner
# stops (a model-account failure) does not go on to the next batch. Run the same command again to restart lanes (after
# a reboot or a stopped lane): finished trials are skipped.
# Env (defaults):
#   MODEL     the agent's model as your main-evaluation runs named it (E1, E3-E5: gpt-6-astra; E2: claude-opus-5.5); any
#             spelling works that gives the experiment's run id (codex-gpt6_astra-high; claude_code-opus_5_5-high)
#   RESULTS   a directory outside the repository for the per-trial results (required)
#   ENV_FILE  the credentials file (<repo>/.env)     JOBS  <repo>/jobs/agents     LANES  4 per GPU (of GPUS when set)
#   STAGGER   seconds between lane starts (60)
# Every other setting of scripts/harbor/run_agent_batch.sh passes through (GPUS, MEM_GUARD, CODEX_FORCE_AUTH_JSON,
# CODEX_AUTH_JSON_PATH, CLAUDE_FORCE_OAUTH, AGENT_CONFIG, GATEWAY_EXTRA_HOSTS, ...). Each lane's output:
# <JOBS>/../ablation-<e>.lane<k>.out; progress: <JOBS>/<batch>.log, <JOBS>/<batch>.summary.md.
# SMOKE=1 (bash-runners/ablations/smoke.sh): the experiment's smoke instead, the batch ablation-smoke-<e>: one trial of
# every variant of the experiment's first task (one per condition and mode), all at once, JOBS <repo>/jobs/smoke.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
[ $# -ge 1 ] || { sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d'; exit 2; }
E="$1"; shift
case "$E" in E1|E2|E3|E4|E5) ;; *) echo "unknown experiment $E (E1 .. E5)" >&2; exit 2 ;; esac
[ $# -eq 0 ] || { echo "run_$E.sh takes no arguments: the experiment fixes its tasks and batches" >&2; exit 2; }
X="python3 $root/scripts/ablations/experiments.py"
AGENT="$($X field "$E" agent)"; MODEL="${MODEL:-$($X field "$E" model)}"; EFFORT="$($X field "$E" effort)"
MODES="$($X field "$E" modes)"; want_id="$($X field "$E" run_id)"
SMOKE="${SMOKE:-0}"; [ "$SMOKE" = 1 ] && dj="$root/jobs/smoke" || dj="$root/jobs/agents"
RESULTS="${RESULTS:-}"; ENV_FILE="${ENV_FILE:-$root/.env}"; JOBS="${JOBS:-$dj}"; STAGGER="${STAGGER:-60}"
[ -n "$RESULTS" ] || { echo "RESULTS must be set: a directory outside the repository" >&2; exit 2; }
[ -f "$ENV_FILE" ] || { echo "ENV_FILE=$ENV_FILE: no such file (copy .env.example and fill in your credentials)" >&2; exit 2; }
[[ "$STAGGER" =~ ^[0-9]+$ ]] || { echo "STAGGER must be whole seconds" >&2; exit 2; }
# the run id scripts/harbor/run_agent_batch.sh derives (PROTOCOL D4): the same rule, so a MODEL spelled otherwise is
# caught here and not after hours of trials under another id
f() { tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g'; }
got_id="$(printf '%s' "$AGENT" | f)-$(printf '%s' "${MODEL##*/}" | tr 'A-Z' 'a-z' | sed -E 's/^claude-//; s/^gpt-([0-9]+)/gpt\1/' | f)-$(printf '%s' "$EFFORT" | f)"
[ "$got_id" = "$want_id" ] || { echo "MODEL=$MODEL gives the run id $got_id; $E runs as $want_id (MODEL: the model your main-evaluation runs used, e.g. $($X field "$E" model))" >&2; exit 2; }
python3 "$root/scripts/ablations/check_variants.py" gate "$E" || { echo "$E: run bash-runners/ablations/check_variants.sh $E first (docs/ABLATIONS.md, step 1)" >&2; exit 2; }
mapfile -t batches < <($X batches "$E")
mapfile -t bases < <($X bases "$E")
if [ "$SMOKE" = 1 ]; then
  batches=("ablation-smoke-${E,,}")
  first="$($X first "$E")"                  # the experiment's first task: its variants, one per condition
  mapfile -t bases < <(printf '%s\n' "${bases[@]}" | grep -F -- "$first-")
  LANES="${LANES:-$(( ${#bases[@]} * $(wc -w <<< "$MODES") ))}"
fi
if [ -z "${LANES:-}" ]; then
  if [ -n "${GPUS:-}" ]; then gpus=$(tr , '\n' <<< "$GPUS" | wc -l); else gpus=$(nvidia-smi -L | wc -l); fi
  LANES=$((4 * gpus))
fi
[[ "$LANES" =~ ^[1-9][0-9]*$ ]] || { echo "LANES must be a positive integer" >&2; exit 2; }
# what each result belongs to: <RESULTS>/<batch>/ablation.json (the experiment, the repeat, every planned trial with its
# condition) and the variant check it ran after (ablation_checks.json); scripts/send_results.sh packs <RESULTS>/<batch>/
python3 - "$root" "$E" "$RESULTS" "$([ "$SMOKE" = 1 ] && echo "${batches[0]}")" <<'PY'
import json, pathlib, shutil, sys
root, eid, results = sys.argv[1:4]
sys.path.insert(0, f"{root}/scripts/ablations")
import experiments, make_variants as mv
e = experiments.EXPERIMENTS[eid]
smoke = sys.argv[4] if len(sys.argv) > 4 else ""
for k, batch in enumerate([smoke] if smoke else e.batches(), 1):
    d = pathlib.Path(results) / batch
    d.mkdir(parents=True, exist_ok=True)
    rows = [r for r in experiments.plan(eid) if r["batch"] == e.batches()[0]]
    if smoke:                              # the first task's variants, once each
        rows = [dict(r, batch=batch) for r in rows if r["base"] == rows[0]["base"]]
    else:
        rows = [r for r in experiments.plan(eid) if r["batch"] == batch]
    (d / "ablation.json").write_text(json.dumps({
        "experiment": eid, "purpose": e.purpose, "repeat": k, "repeats": e.repeats, "batch": batch, "run_id": e.run_id,
        "agent": e.agent, "effort": e.effort, "seed": 0,
        "results_dataset": f"https://huggingface.co/datasets/{experiments.RESULTS_DATASET}",
        "conditions": {t: {"label": mv.REGISTRY[t].label, "note": mv.REGISTRY[t].note} for t in e.conditions},
        "trials": rows}, indent=1) + "\n")
    shutil.copyfile(pathlib.Path(root) / "jobs" / "ablations" / "checks" / f"{eid}.json", d / "ablation_checks.json")
PY
out="$(dirname "$JOBS")"; mkdir -p "$out" "$JOBS"
pids=()
for k in $(seq 1 "$LANES"); do
  [ "$k" -gt 1 ] && [ "$STAGGER" -gt 0 ] && sleep "$STAGGER"
  AGENT="$AGENT" MODEL="$MODEL" EFFORT="$EFFORT" MODES="$MODES" K=3 SEEDS=0 ENV_FILE="$ENV_FILE" \
    USAGE_GATEWAY="$root/scripts/usage_gateway" GATEWAY_RAW=0 JOBS="$JOBS" RESULTS="$RESULTS" \
    nohup "$root/bash-runners/ablations/lane.sh" "${batches[*]}" "${bases[@]}" > "$out/${batches[0]%-r1}.lane$k.out" 2>&1 &
  pids+=($!)
  echo "lane $k started (PID $!)"
done
echo "$E: $LANES lanes over ${#bases[@]} task variants x ${#batches[@]} batch(es) (${batches[*]}), modes $MODES, run id $want_id"
echo "progress: $JOBS/<batch>.log and <batch>.summary.md; lane output: $out/${batches[0]%-r1}.lane<k>.out"
