#!/usr/bin/env bash
# Run one Harbor-native agent over tasks in both modes (<base>-privileged and <base>-standard), unattended; meant to be
# started in the background (inside `sg docker -c` if docker needs the group):
#   MODEL=<provider/model> scripts/harbor/run_agent_batch.sh <batch-name> <base> [<base> ...]
#   (<base> = a task name without its -privileged / -standard; docs/RUNNING.md)
# Official runs use Harbor's native agents only (PROTOCOL D11): no adapter. Env (defaults):
#   AGENT        codex | claude-code | dsh | antigravity (codex)  MODEL   the -m model (required, e.g. openai/<model>)
#                dsh = DSH, the DeepSeek Harness, through RoboARC's own Harbor agent scripts/harbor/agents/dsh_acp.py
#                (Harbor's ACP agent with dsh installed; not a Harbor built-in: whether its results are official is the
#                benchmark owner's call, PROTOCOL D11), e.g. AGENT=dsh MODEL=deepseek/deepseek-v4.1-flash EFFORT=high
#                antigravity = the Antigravity SDK (google-antigravity 0.1.23) through RoboARC's own Harbor agent
#                scripts/harbor/agents/antigravity_sdk.py, its web tools off, the model served by an OpenAI-compatible
#                Chat Completions server (ANTIGRAVITY_BASE_URL, a plain-HTTP address with a port: a local proxy) and
#                reached through the usage gateway's Gemini translation (USAGE_GATEWAY required; gemini_chat.py);
#                official for Antigravity + Gemini (decided 2026-11-01), e.g. AGENT=antigravity
#                MODEL=google/gemini-4-argon EFFORT=high (EFFORT: minimal, low, medium, high, xhigh or default)
#   EFFORT       reasoning effort, --ak reasoning_effort=<EFFORT> (xhigh); `default` passes none
#   AK           more agent options, e.g. "--ak reasoning_summary=detailed"
#   WEB_SEARCH   disabled (default) = the harness offers the model no web search: Codex --ak web_search=disabled,
#                Claude Code --ak disallowed_tools=WebSearch,WebFetch (PROTOCOL D11; the search runs at the model
#                provider, past any egress allowlist), and Codex none of the other tools that run there (a ChatGPT
#                login's connectors and apps, plugins, image generation: those features = false in a copy of
#                AGENT_CONFIG, scripts/harbor/codex_tools_off.py); "allowed" = leave the harness default (not for official runs)
#   AGENT_CONFIG a native config file, --ak config=<file>: Codex's config.toml (an Azure provider; retries with
#                request_max_retries / stream_max_retries) or Claude Code's settings
#   ENV_FILE     a .env file Harbor loads (--env-file); without it the credentials come from this environment
#   Credentials, as Harbor's agents read them (docs/RUNNING.md):
#     OpenRouter   codex: OPENAI_API_KEY=<openrouter key> OPENAI_BASE_URL=https://openrouter.ai/api/v1
#                  claude-code: ANTHROPIC_API_KEY=<openrouter key> ANTHROPIC_BASE_URL=https://openrouter.ai/api
#     ChatGPT login (codex)          CODEX_FORCE_AUTH_JSON=1 (Harbor uploads ~/.codex/auth.json into each trial)
#     Claude subscription            CLAUDE_FORCE_OAUTH=1 and CLAUDE_CODE_OAUTH_TOKEN (`claude setup-token`)
#     Azure OpenAI (codex)           AGENT_CONFIG=<config.toml with the Azure provider> and its key variable
#     DeepSeek (dsh)                 DEEPSEEK_API_KEY (DeepSeek's API, api.deepseek.com, added to the job's egress)
#     Chat Completions (antigravity) ANTIGRAVITY_BASE_URL=http://<host>:<port>/v1 (the server's base: <base>/chat/completions)
#                                    and GEMINI_API_KEY (the key it expects, sent as a Bearer token; any value if none)
#   MODES        "privileged standard"; add `standard-realtime` for <base>-standard-realtime where it exists
#   K            trials per task and mode (1). Each trial is a Harbor job of its own (`-k 1`) and runs from its own seed:
#                the K trials of a task get distinct seeds of its `[metadata.seeds] allowed`, without replacement, in an
#                order fixed by the batch name and the task's base name (scripts/task_seed.py assign; the same seeds in
#                every mode; docs/RUNNING.md "Seeds"). A task that allows one seed runs it in every trial. Each job's copy
#                of a task is made by scripts/task_seed.py materialize with the trial's seed, and the job records the
#                trial and its seeds in <job>/seed.json (also $JOBS/.seeds/<job>.json, written before the job starts)
#   SEEDS        "" (default) = every trial, in the SEQ=1 order below. Seeds separated by spaces, e.g. "0 1 2" or "1 2"
#                (SEQ=1 only): of the K trials only those of these seeds run, every task's trial of the lowest seed
#                before any trial of the next (docs/RUNNING.md "Running the seeds in turn"). The trials keep their
#                numbers and seeds, so SEEDS="0" now and SEEDS="1 2" later, with the same batch name and JOBS, run the
#                batch's K=3 trials between them. A lane goes back to the top of the plan after each trial it starts; a
#                later seed's trial starts only in room that no waiting trial of an earlier seed fits, and while one
#                waits a later seed's deformable trial is not tried (it would have a GPU kept for it). A task allowed a
#                seed that its K trials do not run stops the batch before it starts; a task not allowed one runs no
#                trial at it. Lanes of the same SEEDS keep that order between them; a lane of other seeds (or none) on
#                the same batch may run beside them, each lane its own trials, with no order between them
#   SEQ          1 = one trial at a time per runner (default): task by task, trial by trial, each mode in MODES order, one
#                Harbor job per task, trial and mode ($JOBS/<batch>-<mode>-<task>/; with K > 1
#                $JOBS/<batch>-<mode>-<task>-t<k>/, k = 1..K). Start the same command again to add a lane: every job is
#                claimed (mkdir $JOBS/.claims/<job>) by exactly one runner, so N runners run N trials at a time. A job that
#                already has its result.json (Harbor writes it when the job starts) is skipped; to rerun an interrupted
#                job (or a trial voided for infrastructure: the retry gets the same seed), move its directory and its
#                claim away first.
#                0 = one job per mode and trial ($JOBS/<batch>-<mode>/; with K > 1 $JOBS/<batch>-<mode>-t<k>/), the
#                modes side by side, a mode's trials one after another, N_P / N_S concurrent trials (2 / 2; N_S also for
#                standard-realtime)
#   WAIT_FOR     a file that must exist before a mode starts, e.g. "standard:<path>" (default: none)
#   JOBS         <repo>/jobs/agents (ignored by git)     LABEL  the run's title for the results README
#   RUN_ID       the run id (PROTOCOL D4, docs/TAXONOMY.md): <harness>-<model>-<effort>, derived from AGENT / MODEL /
#                EFFORT, e.g. codex-gpt5-xhigh. Provider, service tier and settings go in metadata, never in the id.
#   RESULTS      a directory OUTSIDE the repository: after each job scripts/harbor/write_task_results.py writes the per-trial
#                results there, <RESULTS>/<batch>/<task>/<RUN_ID>/t<k>/, with an index of the task's trials in
#                <RESULTS>/<batch>/<task>/<RUN_ID>/ (README.md, trials.json) (PROTOCOL D3: no results in the repository);
#                default <repo>/../rcb_results; "" = skip
#   HOST_CPUS    the host's CPU count (`nproc`): a task's CPU cap above it ([environment] / [verifier.environment] cpus,
#                a compose service's cpus) is clamped to it in the job's own copy of the task (below), as PROTOCOL §2.10
#                requires of a runner (docker refuses a larger cap); tasks/ keeps its values (scripts/harbor/clamp_cpus.py)
#   Every job runs from its own copy of its tasks ($JOBS/.tasks/<job>/, removed afterwards), every file's mtime derived
#   from its content (scripts/stamp_mtimes.py): Docker's BuildKit can otherwise build one task's image with another
#   task's equal-size, equal-mtime file (docs/RUNNING.md "Building task images")
#   LOG_MIRROR   0 (default); 1 = start the append-only agent-record mirror below (a root container) if none runs
#   MEM_GUARD    0 (default); 1 = start the memory guard below if none runs: while host memory is short it kills the
#                running RoboARC trial container that uses the most memory (only containers named roboarc-*)
#   GPU_SCHED    1 (default) = every trial (SEQ=1) is placed on one GPU and admitted by CPUs and memory, through the
#                host's slot ledger (scripts/harbor/gpu_slots.py, docs/RUNNING.md "GPUs and lanes"): a GPU takes up to
#                GPU_SHARE trials of rigid tasks, a trial of a deformable task (DEFORMABLE_FAMILIES) has a GPU of its own,
#                a task whose compose reserves no GPU takes none; a trial starts when the CPUs and memory its task
#                declares ([metadata.resources] min_* and verifier_min_*, times RESERVE_CPU_SCALE / RESERVE_MEM_SCALE)
#                fit beside the trials already running (memory: HOST_MEM_GB, default the host's total, less MEM_HEADROOM_GB,
#                default the larger of 4 GB and 10 %; CPUs: HOST_CPUS), else the lane tries the plan's next job and, when none fits, waits
#                30 s; the job's copy has its containers pinned to that GPU (scripts/harbor/pin_gpu.py; Docker maps
#                `count: 1` to GPU 0) and <job>/placement.json records it. Start up to GPU_SHARE x GPUs lanes.
#                0 = as before: no placement, no admission, every container on GPU 0
#   PHASE_SPLIT  1 (default, with GPU_SCHED=1 and SEQ=1) = a trial's agent phase and its validation are scheduled apart:
#                when the agent phase ends (Harbor's agent-end event, marked by the job plugin plugins/phase_marker.py
#                in $JOBS/.phase/<job>.json), the trial validates in the background on its GPU (the replay, video and
#                grading Harbor runs after the agent: a standard trial's in its sim container) and the lane takes the
#                next trial at once. A validating trial keeps no GPU unit, only its task's verifier CPUs and memory
#                ([metadata.resources] verifier_min_*, times the scales); it still counts on its GPU (GPU_RESIDENT), and
#                a deformable trial reasons only on a GPU where no other trial reasons or validates. The lane stops
#                on a model-account failure before it takes the next trial, as before, and ends when its trials have
#                validated. 0 = a lane waits for each trial's validation; a Harbor without job plugins (`--plugin`)
#                runs as 0
#   GPU_RESIDENT the most trials on one GPU, reasoning and validating (2 x GPU_SHARE)
#   GPUS         the GPUs to use, comma-separated (default every GPU nvidia-smi lists; 0 without nvidia-smi)
#   GPU_SHARE    rigid-task trials per GPU (4)
#   DEFORMABLE_FAMILIES  the families whose trials have a GPU of their own (task.toml [metadata] family); empty
#                (default) = the classes with own_gpu in scripts/task_resources.toml (rope / chain and cloth), the one
#                list of the family classes, which the tasks' declared resources come from too
#   SLOTS_DIR    the ledger (/tmp/roboarc-slots-<uid>): every lane and batch on the host that should share GPUs uses one
#   USAGE_GATEWAY  usage recording, optional (PROTOCOL H.6, docs/RUNNING.md "Usage recording"). Empty (default) = off:
#                the agent calls its API directly, as Harbor runs it, and nothing below about the gateway runs; the
#                token counts in the results are the harness's own (Harbor's result.json). <dir> = on: route the agent's
#                model API calls through the transparent usage gateway installed in <dir> (gateway.py, rawlog.py,
#                api_paths.json). This repository ships one: USAGE_GATEWAY=scripts/usage_gateway (another gateway
#                that takes the same options works too); a <dir> without gateway.py stops the runner at start. It forwards
#                them unchanged but for GATEWAY_IMAGE_CAP (the agent keeps its own credentials) and records each trial's tokens per request in
#                <trial>/gateway/ (usage.jsonl, totals.json). Before each job the runner checks that the gateway answers
#                (/_gateway/health) and restarts it on its port, or stops the lane; a job during which it died gets
#                <job>/gateway_down.json, and a trial whose last model calls failed at the gateway (upstream unreachable,
#                stream broken off, provider 5xx) is published as "not a result", never the agent's 0.
#                Either way every job gets <job>/usage_recording.json (whether the gateway ran, its token budget and image
#                cap), and write_task_results.py copies it into each result's config.json (usage_gateway, usage_source,
#                token_budget, image_cap)
#                The GATEWAY_* and TOKEN_BUDGET_* options below take effect only with USAGE_GATEWAY set.
#   GATEWAY_RAW  1 (default) = the gateway also keeps every request and response body
#                (<trial>/gateway/raw.jsonl.xz, deduplicated, see the gateway's rawlog.py; private: it holds the
#                whole conversation); 0 = token usage only
#                The gateway forwards model-API calls only (its api_paths.json) and answers anything
#                else at the provider 403 (account connectors, files, web tools; counted in totals.json refused_paths)
#   GATEWAY_EGRESS  gateway (default) = each job's copy of the task allows as egress only the gateway's address and the package registries (PyPI, apt), so the
#                agent's container reaches its provider through the gateway's model-API paths only. (Harbor's allowlist
#                names an address, not a port: every TCP service listening on that address of the host is reachable
#                from a trial, as it has been since the gateway exists. Bind host services to 127.0.0.1.)
#                (one exception: Codex with a ChatGPT login also keeps chatgpt.com, which its start-up lookup needs);
#                task = run tasks/ as it is (the task's allowlist: the model APIs' hosts stay reachable directly)
#   GATEWAY_BIND the docker bridge address the trial containers reach the host by (172.17.0.1)
#   GATEWAY_EXTRA_HOSTS  more model API host patterns the gateway forwards to, space-separated (e.g. "*.services.ai.azure.com");
#                a plain-HTTP base URL with a port (ANTHROPIC_BASE_URL / OPENAI_BASE_URL=http://<host>:<port>, a self-hosted
#                model server) is forwarded too, to exactly that <host>:<port>
#   GATEWAY_START_WAIT  seconds to wait for the gateway to come up (10); raise it on busy shared nodes
#   TOKEN_BUDGET_IN / TOKEN_BUDGET_OUT   per-trial token budget the gateway enforces (100000000 / 10000000, PROTOCOL
#                §5): past it every further model request is refused and the trial counts as failed; 0 = no limit.
#                Without the gateway no token budget is enforced (the task's cutoff bounds the trial)
#   GATEWAY_IMAGE_CAP  at most this many images per model request (50, the default; 10, the default for MODEL=<provider>/
#                gemini-4-argon; 0 = no cap): a request with more has its oldest images replaced by a short text note,
#                half the cap at a time (GitHub Copilot refuses a request with more than 50, or more than 10 for Gemini 4
#                Argon, and an agent resends every camera image it has looked at with each call; the gateway's module
#                doc). Without the gateway there is no cap
#   GATEWAY_IMAGE_BYTES  a model request still over this many bytes after GATEWAY_IMAGE_CAP has more of its oldest
#                images replaced, in the same steps, until it is not; the newest image stays (0, the default = off;
#                4718592 = 4.5 MiB, the default for MODEL=<provider>/gemini-4-argon: through a GitHub Copilot proxy a
#                Gemini 4 Argon request over 5 MiB is refused). Without the gateway there is no cap
# Runner-side tools (not task code), in this directory: with LOG_MIRROR=1 the container rcb-log-mirror copies every
# trial's agent/ directory append-only into ${JOBS}_mirror/ (log_mirror.py; root-owned: neither the agent nor the host
# user can delete it); with MEM_GUARD=1 mem_guard.py keeps 2 GiB of host memory available. After each job: a key scan
# with redaction (scan_secrets.py), the task results, the summary table $JOBS/<batch>.summary.md, and a lane stop on
# structured model-account failures (account_failure_check.py). Progress: $JOBS/<batch>.log; in a browser, every
# trial's state (pending / running / done) with scripts/harbor/live_board.py --jobs $JOBS. The machine (host_info.py:
# CPU model and count, memory, each GPU's name, driver and memory, docker and harbor) is logged when a lane starts and
# written into every job folder as host.json, which each trial's results config.json carries as `host`.
set -u
H="$(cd "$(dirname "$0")" && pwd)"          # scripts/harbor/: the runner's tools
R="$(cd "$H/../.." && pwd)"                 # the repository: tasks/, scripts/ (the task layer)
[ $# -ge 2 ] || { sed -n '2,/^set -u$/p' "$0" | sed '$d'; exit 2; }
batch="$1"; shift; tasks=("$@")
AGENT="${AGENT:-codex}"; MODEL="${MODEL:-}"; EFFORT="${EFFORT:-xhigh}"; AK="${AK:-}"
WEB_SEARCH="${WEB_SEARCH:-disabled}"
AGENT_CONFIG="${AGENT_CONFIG:-}"; ENV_FILE="${ENV_FILE:-}"
MODES="${MODES:-privileged standard}"; SEQ="${SEQ:-1}"; N_P="${N_P:-2}"; N_S="${N_S:-2}"; K="${K:-1}"; WAIT_FOR="${WAIT_FOR:-}"
JOBS="${JOBS:-$R/jobs/agents}"; RESULTS="${RESULTS-$(dirname "$R")/rcb_results}"; LABEL="${LABEL:-$AGENT + $MODEL ($EFFORT)}"
LOG_MIRROR="${LOG_MIRROR:-0}"; MEM_GUARD="${MEM_GUARD:-0}"
GPU_SCHED="${GPU_SCHED:-1}"; GPU_SHARE="${GPU_SHARE:-4}"; SLOTS_DIR="${SLOTS_DIR:-/tmp/roboarc-slots-$(id -u)}"
PHASE_SPLIT="${PHASE_SPLIT:-1}"; GPU_RESIDENT="${GPU_RESIDENT:-}"
DEFORMABLE_FAMILIES="${DEFORMABLE_FAMILIES:-}"   # empty: scripts/task_resources.toml's own_gpu classes (gpu_slots.py)
RESERVE_CPU_SCALE="${RESERVE_CPU_SCALE:-1}"; RESERVE_MEM_SCALE="${RESERVE_MEM_SCALE:-1}"; MEM_HEADROOM_GB="${MEM_HEADROOM_GB:-}"
HOST_MEM_GB="${HOST_MEM_GB:-}"
if [ -z "${GPUS:-}" ]; then
  GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -)"; GPUS="${GPUS:-0}"
fi
USAGE_GATEWAY="${USAGE_GATEWAY:-}"
GATEWAY_RAW="${GATEWAY_RAW:-1}"; GATEWAY_BIND="${GATEWAY_BIND:-172.17.0.1}"; GATEWAY_START_WAIT="${GATEWAY_START_WAIT:-10}"
GATEWAY_EGRESS="${GATEWAY_EGRESS:-gateway}"
# GitHub Copilot takes at most 10 images per request for Gemini 4 Argon (decided 2026-11-02); every other model 50.
# A Gemini 4 Argon request over 5 MiB is refused too (HTTP 413): 4.5 MiB for it (decided 2026-11-04); no size cap otherwise
case "${MODEL##*/}" in
  gemini-4-argon) GATEWAY_IMAGE_CAP="${GATEWAY_IMAGE_CAP:-10}"; GATEWAY_IMAGE_BYTES="${GATEWAY_IMAGE_BYTES:-4718592}" ;;
  *) GATEWAY_IMAGE_CAP="${GATEWAY_IMAGE_CAP:-50}"; GATEWAY_IMAGE_BYTES="${GATEWAY_IMAGE_BYTES:-0}" ;;
esac
HOST_CPUS="${HOST_CPUS:-$(nproc)}"
[ -n "$MODEL" ] || { echo "MODEL must be set: the -m model of the agent, e.g. MODEL=openai/<model> (docs/RUNNING.md)" >&2; exit 2; }
# the switch of earlier versions of this script: refused rather than ignored, so that nobody expects a usage record silently
[ -n "${GATEWAY:-}${USAGE_GATEWAY_DIR:-}" ] && { echo "GATEWAY / USAGE_GATEWAY_DIR are no longer read: usage recording is USAGE_GATEWAY=<the gateway's directory> (docs/RUNNING.md \"Usage recording\")" >&2; exit 2; }
if [ -n "$USAGE_GATEWAY" ]; then
  [ -f "$USAGE_GATEWAY/gateway.py" ] || { echo "USAGE_GATEWAY=$USAGE_GATEWAY: no gateway.py there. Point USAGE_GATEWAY at the gateway's directory (this repository ships one: USAGE_GATEWAY=scripts/usage_gateway), or leave USAGE_GATEWAY empty to run without usage recording (docs/RUNNING.md \"Usage recording\")" >&2; exit 2; }
  USAGE_GATEWAY="$(cd "$USAGE_GATEWAY" && pwd)"
  case "$GATEWAY_EGRESS" in gateway|task) ;; *) echo "GATEWAY_EGRESS must be gateway or task" >&2; exit 2 ;; esac
  [[ "$GATEWAY_IMAGE_CAP" =~ ^[0-9]+$ ]] || { echo "GATEWAY_IMAGE_CAP must be a whole number (0 = no cap)" >&2; exit 2; }
  [[ "$GATEWAY_IMAGE_BYTES" =~ ^[0-9]+$ ]] || { echo "GATEWAY_IMAGE_BYTES must be a whole number of bytes (0 = no cap)" >&2; exit 2; }
fi
[[ "$HOST_CPUS" =~ ^[1-9][0-9]*$ ]] || { echo "HOST_CPUS must be a positive integer (the host's CPU count)" >&2; exit 2; }
[[ "$K" =~ ^[1-9][0-9]*$ ]] || { echo "K must be a positive integer (trials per task and mode)" >&2; exit 2; }
read -ra seed_list <<< "${SEEDS:-}"; SEEDS="${seed_list[*]}"
if [ -n "$SEEDS" ]; then
  [[ "$SEEDS" =~ ^[0-9]+( [0-9]+)*$ ]] || { echo "SEEDS must be seeds separated by spaces, e.g. SEEDS=\"0 1 2\"" >&2; exit 2; }
  [ "$SEQ" = 1 ] || { echo "SEEDS needs SEQ=1 (with SEQ=0 one job holds every task, each from its own seed)" >&2; exit 2; }
fi
case "$GPU_SCHED" in 0|1) ;; *) echo "GPU_SCHED must be 0 or 1" >&2; exit 2 ;; esac
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "GPUS must be GPU indices, comma-separated (e.g. 0,1,2)" >&2; exit 2; }
[[ "$GPU_SHARE" =~ ^[1-9][0-9]*$ ]] || { echo "GPU_SHARE must be a positive integer (rigid-task trials per GPU)" >&2; exit 2; }
GPU_RESIDENT="${GPU_RESIDENT:-$((2 * GPU_SHARE))}"
[[ "$GPU_RESIDENT" =~ ^[1-9][0-9]*$ ]] || { echo "GPU_RESIDENT must be a positive integer (trials per GPU, reasoning and validating)" >&2; exit 2; }
case "$PHASE_SPLIT" in 0|1) ;; *) echo "PHASE_SPLIT must be 0 or 1" >&2; exit 2 ;; esac
for v in RESERVE_CPU_SCALE RESERVE_MEM_SCALE; do
  [[ "${!v}" =~ ^[0-9]*\.?[0-9]+$ ]] && [[ "${!v}" =~ [1-9] ]] || { echo "$v must be a positive number" >&2; exit 2; }
done
[ -z "$MEM_HEADROOM_GB" ] || [[ "$MEM_HEADROOM_GB" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "MEM_HEADROOM_GB must be a number of GB (0 or more)" >&2; exit 2; }
[ -z "$HOST_MEM_GB" ] || [[ "$HOST_MEM_GB" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "HOST_MEM_GB must be a number of GB" >&2; exit 2; }
# absolute paths: with the gateway on, Harbor is started from the job's own copy of the task (run_job), not from the repository
abspath() { case "$1" in ""|/*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
JOBS="$(abspath "$JOBS")"; ENV_FILE="$(abspath "$ENV_FILE")"; AGENT_CONFIG="$(abspath "$AGENT_CONFIG")"
[ -n "${RESULTS:-}" ] && RESULTS="$(abspath "$RESULTS")"
[ -n "${CODEX_AUTH_JSON_PATH:-}" ] && { CODEX_AUTH_JSON_PATH="$(abspath "$CODEX_AUTH_JSON_PATH")"; export CODEX_AUTH_JSON_PATH; }
TOKEN_BUDGET_IN="${TOKEN_BUDGET_IN:-100000000}"; TOKEN_BUDGET_OUT="${TOKEN_BUDGET_OUT:-10000000}"
[ -z "$USAGE_GATEWAY" ] || [[ "$TOKEN_BUDGET_IN$TOKEN_BUDGET_OUT" =~ ^[0-9]+$ && -n "$TOKEN_BUDGET_IN" && -n "$TOKEN_BUDGET_OUT" ]] \
  || { echo "TOKEN_BUDGET_IN / TOKEN_BUDGET_OUT must be whole numbers (0 = no limit)" >&2; exit 2; }
case "$AGENT" in codex|claude-code|dsh|antigravity) ;; *) echo "AGENT must be a Harbor native agent, codex or claude-code (PROTOCOL D11), dsh (scripts/harbor/agents/dsh_acp.py) or antigravity (scripts/harbor/agents/antigravity_sdk.py)" >&2; exit 2 ;; esac
case "$AGENT" in dsh|antigravity) [ -n "$AGENT_CONFIG" ] && { echo "AGENT_CONFIG is for Codex and Claude Code; $AGENT takes none" >&2; exit 2; } ;; esac
if [ "$AGENT" = antigravity ]; then
  # the model server speaks Chat Completions only: the gateway translates the SDK's Gemini requests (gemini_chat.py)
  [ -n "$USAGE_GATEWAY" ] || { echo "AGENT=antigravity needs the usage gateway (USAGE_GATEWAY=scripts/usage_gateway): it translates the SDK's Gemini requests for the Chat Completions server" >&2; exit 2; }
  case "$EFFORT" in minimal|low|medium|high|xhigh|default) ;; *) echo "EFFORT for AGENT=antigravity: minimal, low, medium, high, xhigh or default (the SDK's thinking levels)" >&2; exit 2 ;; esac
  [ "$WEB_SEARCH" = disabled ] || { echo "AGENT=antigravity has its web tools off (agents/antigravity/run_agent.py); WEB_SEARCH=allowed is not available" >&2; exit 2; }
fi
# Run id (D4): harness = the agent name with - -> _; model = MODEL's last path segment, lower case, gpt-<n> joined
# (gpt-5 -> gpt5), a leading claude- dropped (claude-opus-4-1 -> opus_4_1), every other
# character outside [a-z0-9] -> _; effort as given ("default" when the harness sets none).
if [ -z "${RUN_ID:-}" ]; then
  f() { tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g'; }
  harness=$(printf '%s' "$AGENT" | f)
  model=$(printf '%s' "${MODEL##*/}" | tr 'A-Z' 'a-z' | sed -E 's/^claude-//; s/^gpt-([0-9]+)/gpt\1/' | f)
  RUN_ID="$harness-$model-$(printf '%s' "$EFFORT" | f)"
fi
[[ "$RUN_ID" =~ ^[a-z0-9_]+-[a-z0-9_]+-(none|minimal|low|medium|high|xhigh|max|default)$ ]] || { echo "run id $RUN_ID is not <harness>-<model>-<effort>" >&2; exit 2; }
AGENT_ARGS=(-a "$AGENT" -m "$MODEL")
if [ "$AGENT" = dsh ]; then
  # RoboARC's own Harbor agent: Harbor imports dsh_acp:DshAcp from scripts/harbor/agents (PYTHONPATH, every harbor run)
  AGENT_ARGS=(-a dsh_acp:DshAcp -m "$MODEL")
  export PYTHONPATH="$H/agents${PYTHONPATH:+:$PYTHONPATH}"
  [ -n "$USAGE_GATEWAY" ] || AGENT_ARGS+=(--allow-agent-host api.deepseek.com)   # with the gateway: through it (below)
fi
if [ "$AGENT" = antigravity ]; then   # RoboARC's own Harbor agent, as dsh's
  AGENT_ARGS=(-a antigravity_sdk:AntigravitySdk -m "$MODEL")
  export PYTHONPATH="$H/agents${PYTHONPATH:+:$PYTHONPATH}"
fi
[ "$EFFORT" != default ] && AGENT_ARGS+=(--ak "reasoning_effort=$EFFORT")
# Codex's other provider-side tools go off with the web search (PROTOCOL D11). Under a ChatGPT login Codex hands the
# model the account's connectors and apps (mcp__codex_apps__*: GitHub fetch, a web search of their own, mail, drives),
# plugin installs and an image generator; they run at OpenAI, reached through chatgpt.com, so neither
# web_search=disabled nor an egress allowlist stops them. The run's config is a copy of AGENT_CONFIG (or an empty one)
# with those features = false; a config that turns one of them on is refused, never overridden silently.
# Harbor keeps the last --ak of a key: a config=... in AK would replace the one written here (and the gateway's copy)
case " $AK " in *[[:space:]]config=*) echo "AK must not carry config=...: name the file in AGENT_CONFIG" >&2; exit 2 ;; esac
if [ "$WEB_SEARCH" = disabled ] && [ "$AGENT" = codex ]; then
  mkdir -p "$JOBS/.agent-config" && cfg="$JOBS/.agent-config/config.toml.$$" \
    && python3 "$H/codex_tools_off.py" ${AGENT_CONFIG:+"$AGENT_CONFIG"} > "$cfg" && grep -q '^features\.apps = false$\|^apps = false$' "$cfg" \
    || { echo "no Codex config without provider-side tools could be written from AGENT_CONFIG=${AGENT_CONFIG:-(none)} (see above); WEB_SEARCH=disabled needs it" >&2; exit 2; }
  AGENT_CONFIG="$cfg"
fi
[ -n "$AGENT_CONFIG" ] && AGENT_ARGS+=(--ak "config=$AGENT_CONFIG")
# shellcheck disable=SC2206
AGENT_ARGS+=($AK)
# no harness web search unless asked for (PROTOCOL D11): it runs at the model provider, so no egress allowlist stops it
case "$WEB_SEARCH" in
  disabled) case "$AGENT" in
              codex) AGENT_ARGS+=(--ak web_search=disabled) ;;
              dsh) AGENT_ARGS+=(--ak web=false) ;;   # dsh's web_search and web_fetch tools off (a dsh patch file)
              antigravity) ;;                        # search_web, read_url_content, generate_image off in run_agent.py
              *) AGENT_ARGS+=(--ak disallowed_tools=WebSearch,WebFetch) ;;
            esac ;;
  allowed) [ "$AGENT" = dsh ] && AGENT_ARGS+=(--ak web=true) ;;
  *) echo "WEB_SEARCH must be disabled or allowed" >&2; exit 2 ;;
esac
SCAN=(); [ -n "$ENV_FILE" ] && SCAN+=(--env-file "$ENV_FILE")
case "${CODEX_FORCE_AUTH_JSON:-}" in 1|true|TRUE|yes) SCAN+=(--codex-auth "${CODEX_AUTH_JSON_PATH:-$HOME/.codex/auth.json}") ;; esac
# Append, so a harbor already on PATH wins (a shared account's ~/.local/bin may hold another user's older harbor);
# hosts that keep harbor only in ~/.local/bin still find it.
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$PATH:$HOME/.local/bin" ;; esac
mkdir -p "$JOBS"
LOG="$JOBS/$batch.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
if [ -n "$ENV_FILE" ] && [ ! -f "$ENV_FILE" ]; then log "no env file $ENV_FILE"; exit 2; fi
if [ "$GPU_SCHED" = 1 ]; then
  sched="GPUs $GPUS ($GPU_SHARE rigid-task trials per GPU, a deformable-task trial alone, deformable: ${DEFORMABLE_FAMILIES:-the own_gpu classes of scripts/task_resources.toml}; CPUs and memory as the tasks declare x $RESERVE_CPU_SCALE / x $RESERVE_MEM_SCALE)"
else
  sched="GPU scheduling off"
fi
# PHASE_SPLIT=1 needs the slot ledger, one trial per lane at a time and a Harbor that loads job plugins
if [ "$PHASE_SPLIT" = 1 ]; then
  if [ "$GPU_SCHED" != 1 ] || [ "$SEQ" != 1 ]; then
    PHASE_SPLIT=0; sched="$sched, validation in the lane (PHASE_SPLIT needs GPU_SCHED=1 and SEQ=1)"
  elif ! harbor run --help 2>/dev/null | grep -q -- '--plugin'; then
    PHASE_SPLIT=0; sched="$sched, validation in the lane (this harbor has no --plugin for the agent-end marker)"
  else
    export PYTHONPATH="$H/plugins${PYTHONPATH:+:$PYTHONPATH}"
    sched="$sched, validation in the background beside the next trial (up to $GPU_RESIDENT trials per GPU)"
  fi
fi
log "runner $$: $AGENT $MODEL effort $EFFORT, run id $RUN_ID, modes $MODES, web search $WEB_SEARCH, usage gateway ${USAGE_GATEWAY:-off}, $sched"
# the machine, as MANIFEST.md writes it (host:, GPUs:, software:), and as JSON for every job's host.json
HOST_INFO="$(python3 "$H/host_info.py" 2>/dev/null)" || HOST_INFO=""
python3 "$H/host_info.py" --lines 2>/dev/null | while IFS= read -r l; do log "$l"; done
# on exit (also a signal): the usage gateway stopped, and this runner's GPU slots released (a lane killed with SIGKILL
# leaves them to gpu_slots.py, which drops the slots of runners that have ended)
gw_pid=""
on_exit() {
  [ -n "$gw_pid" ] && kill "$gw_pid" 2>/dev/null
  [ "$GPU_SCHED" = 1 ] && python3 "$H/gpu_slots.py" release --dir "$SLOTS_DIR" --runner-pid $$ >/dev/null 2>&1
  return 0
}
trap on_exit EXIT
# the batch's plan: every job of the batch, with its trial and seeds ("pending" in live_board.py until it starts);
# $JOBS/.plan/<batch>.tsv (job, task, mode, trial, seed; one line per task of a job) is what the loops below run. With
# SEEDS the plan is the same (lanes of other SEEDS may run beside this one, and send_results.sh indexes the batch from
# it), and the lane runs the trials of its seeds in their order, RUN_PLAN = $JOBS/.plan/<batch>.seeds-<s>-<s>.tsv
mkdir -p "$JOBS/.plan"
PLAN="$JOBS/.plan/$batch.tsv"
RUN_PLAN="$PLAN"; [ -n "$SEEDS" ] && RUN_PLAN="$JOBS/.plan/$batch.seeds-${SEEDS// /-}.tsv"
plan_notes="$(python3 - "$R" "$JOBS/.plan/$batch.json" "$PLAN" "$batch" "$RUN_ID" "$SEQ" "$MODES" "$K" "$SEEDS" "$RUN_PLAN" "${tasks[@]}" <<'PY'
import json, os, pathlib, sys
repo, out, tsv, batch, run_id, seq, modes, K, want, run_tsv, *tasks = sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(repo) / "scripts"))
import task_seed, task_conventions as tc  # noqa: E402
K = int(K)
name = lambda m, t: {"standard": f"{t}-standard", "limited": f"{t}-standard", "standard-realtime": f"{t}-standard-realtime"}.get(m, f"{t}-privileged")
suffix = lambda k: f"-t{k}" if K > 1 else ""
seeds = {}                                 # task -> the seeds of trials 1..K (twins: the same, task_seed.assign)
jobs = []
order = ([(t, k, m) for t in tasks for k in range(1, K + 1) for m in modes.split()] if seq == "1"
         else [(t, k, m) for k in range(1, K + 1) for m in modes.split() for t in tasks])
for t, k, m in order:
    d = pathlib.Path(repo) / "tasks" / name(m, t)
    if not d.is_dir():
        continue
    if d.name not in seeds:
        seeds[d.name] = task_seed.assign(d, batch, K)
    job = f"{batch}-{m}-{t}{suffix(k)}" if seq == "1" else f"{batch}-{m}{suffix(k)}"
    jobs.append({"job": job, "task": d.name, "mode": m, "trial": k, "trials": K, "seed": seeds[d.name][k - 1]})
row = lambda j: f"{j['job']}\t{j['task']}\t{j['mode']}\t{j['trial']}\t{j['seed']}\n"
files = [(out, json.dumps({"batch": batch, "run_id": run_id, "trials": K, "jobs": jobs}, indent=1)), (tsv, "".join(map(row, jobs)))]
if want:                                   # SEEDS: the trials of these seeds only, a seed's trials before the next seed's
    want = sorted({int(s) for s in want.split()})
    for task, run in seeds.items():
        sd = task_seed.seeds(pathlib.Path(repo) / "tasks" / task)
        for s in want:
            if s in run:
                continue
            if tc.seed_allowed(sd, s):
                print(f"SEEDS {' '.join(map(str, want))}: {task} allows seed {s}, but its {K} trial(s) in batch {batch} "
                      f"run seeds {run} (K trials run K of a task's allowed seeds: K=3 runs all three of a RoboARC task)")
                sys.exit(1)
            print(f"SEEDS: {task} does not allow seed {s} (allowed: {sd['allowed']}); none of its trials runs at it")
    turn = sorted((j for j in jobs if j["seed"] in want), key=lambda j: j["seed"])   # stable: task, trial, mode
    print(f"SEEDS {' '.join(map(str, want))}, in that order: "
          f"{' / '.join(str(sum(j['seed'] == s for j in turn)) for s in want)} jobs")
    files.append((run_tsv, "".join(map(row, turn))))
for f, text in files:
    tmp = pathlib.Path(f"{f}.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(f)
PY
)" || { log "could not plan batch $batch (scripts/task_seed.py assign)"; [ -n "$plan_notes" ] && log "$plan_notes"; exit 2; }
[ -n "$plan_notes" ] && while IFS= read -r note; do log "$note"; done <<< "$plan_notes"

# append-only mirror of every trial's agent/ directory, out of every trial container's reach
if [ "$LOG_MIRROR" = 1 ] && ! docker ps --format '{{.Names}}' | grep -qx rcb-log-mirror; then
  docker rm -f rcb-log-mirror >/dev/null 2>&1
  docker run -d --name rcb-log-mirror --restart unless-stopped --network none -v "$H:/s:ro" -v "$JOBS:/src:ro" \
    -v "${JOBS}_mirror:/mirror" python:3.12-slim python3 /s/log_mirror.py --src /src --mirror /mirror --interval 5 >/dev/null \
    && log "started rcb-log-mirror ($JOBS -> ${JOBS}_mirror)" || log "WARNING: could not start rcb-log-mirror"
fi

# host memory red line (MEM_GUARD=1): mem_guard.py keeps 2 GiB available (the largest RoboARC trial container is killed first)
pid="$(cat "$JOBS/.mem_guard.pid" 2>/dev/null)"
if [ "$MEM_GUARD" = 1 ] && ! grep -qs mem_guard.py "/proc/${pid:-0}/cmdline"; then
  setsid nohup python3 "$H/mem_guard.py" --log "$JOBS/mem_guard.log" --pidfile "$JOBS/.mem_guard.pid" >/dev/null 2>&1 </dev/null &
  log "started mem_guard (2 GiB red line; log $JOBS/mem_guard.log)"
fi

# job copies a killed runner left behind (run_job removes its own; never here on exit: Harbor may still be tearing a
# trial down from that directory, whose compose file and bind mounts it holds; a copy is at most 11 MB, and no job runs
# for a month)
[ -d "$JOBS/.tasks" ] && find "$JOBS/.tasks" -mindepth 1 -maxdepth 1 -type d -mtime +30 -exec rm -rf {} + 2>/dev/null

# transparent usage gateway (one per runner, stopped with it): the agent's base URL points at it, it forwards unchanged
ENV_ARGS=()
if [ -n "$USAGE_GATEWAY" ]; then
  mkdir -p "$JOBS/.gateway"; ready="$JOBS/.gateway/port.$$"; rm -f "$ready"
  # Harbor's egress sidecar cuts a plain-HTTP call whose response headers take over 15 s (gost's default readTimeout):
  # every harbor run gets Harbor's Docker environment with the sidecar's readTimeout at the gateway's 900 s
  # (scripts/harbor/environments/roboarc_docker.py); the import is checked here, with Harbor's own Python
  export PYTHONPATH="$H/environments${PYTHONPATH:+:$PYTHONPATH}"; ENV_ARGS=(-e roboarc_docker:RoboARCDocker)
  hpy="$(head -1 "$(command -v harbor)" 2>/dev/null | sed -n 's/^#!//p')"
  if [[ "$hpy" == */python* ]] && [ -x "$hpy" ] && "$hpy" -c 'import harbor' 2>/dev/null && ! "$hpy" -c 'import roboarc_docker' >"$JOBS/.gateway/env_import.$$.err" 2>&1; then
    log "scripts/harbor/environments/roboarc_docker.py does not import with this harbor ($JOBS/.gateway/env_import.$$.err); without it the egress sidecar cuts model calls slower than 15 s"; exit 2
  fi
  # the upstream each base URL would have used: this environment, else the env file, else the API's default
  upstream() {  # <VAR> <default>
    local v="${!1:-}"
    [ -z "$v" ] && [ -n "$ENV_FILE" ] && v=$(sed -n "s/^[[:space:]]*\(export[[:space:]]\+\)\?$1=//p" "$ENV_FILE" | tail -1 | tr -d "\"'")
    printf '%s' "${v:-$2}"
  }
  gw_args=(--jobs "$JOBS" --bind "$GATEWAY_BIND" --ready-file "$ready" --budget-input "$TOKEN_BUDGET_IN" --budget-output "$TOKEN_BUDGET_OUT" --image-cap "$GATEWAY_IMAGE_CAP"); for h in ${GATEWAY_EXTRA_HOSTS:-}; do gw_args+=(--extra-host "$h"); done; [ "$GATEWAY_RAW" = 1 ] && gw_args+=(--raw)
  [ "$GATEWAY_IMAGE_BYTES" != 0 ] && gw_args+=(--image-bytes "$GATEWAY_IMAGE_BYTES")
  [ "$WEB_SEARCH" = allowed ] && gw_args+=(--server-tools record)   # not an official run: the provider-run tools are only recorded
  # a plain-HTTP base URL (a self-hosted model server, e.g. sglang): the gateway may forward to exactly that host:port
  case "$AGENT" in
    claude-code) agent_up=$(upstream ANTHROPIC_BASE_URL https://api.anthropic.com) ;;
    dsh) agent_up=$(upstream DEEPSEEK_BASE_URL https://api.deepseek.com/anthropic)
         gw_args+=(--extra-host api.deepseek.com) ;;   # DeepSeek's API is no task's model host (scripts/set_agent_egress.py)
    antigravity) agent_up=$(upstream ANTIGRAVITY_BASE_URL "")
         [[ "$agent_up" =~ ^http://[A-Za-z0-9.-]+:[0-9]+(/|$) ]] || { log "AGENT=antigravity: ANTIGRAVITY_BASE_URL must be the Chat Completions server's plain-HTTP base with a port, e.g. http://127.0.0.1:4141/v1 (env or ENV_FILE); got '${agent_up:-nothing}'"; exit 2; }
         gw_args+=(--gemini-chat) ;;   # the SDK's Gemini requests -> <base>/chat/completions (gemini_chat.py)
    *) agent_up=$(upstream OPENAI_BASE_URL https://api.openai.com/v1) ;;
  esac
  [[ "$agent_up" =~ ^http://([A-Za-z0-9.-]+:[0-9]+)(/|$) ]] && gw_args+=(--http-host "${BASH_REMATCH[1]}")
  gw_start() {  # [port]: start the gateway (a restart: on the port the agent's base URLs already name); 0 = it listens
    rm -f "$ready"
    setsid python3 "$USAGE_GATEWAY/gateway.py" "${gw_args[@]}" ${1:+--port "$1"} >>"$JOBS/.gateway/gateway.$$.err" 2>&1 </dev/null &
    gw_pid=$!
    local _; for _ in $(seq $((GATEWAY_START_WAIT * 5))); do [ -s "$ready" ] && return 0; kill -0 "$gw_pid" 2>/dev/null || return 1; sleep 0.2; done
    [ -s "$ready" ]
  }
  gw_alive() {  # the gateway process runs and answers its health check
    kill -0 "$gw_pid" 2>/dev/null && python3 -c 'import sys, urllib.request as u
u.build_opener(u.ProxyHandler({})).open(sys.argv[1], timeout=15).read()' "$GW/_gateway/health" 2>/dev/null
  }
  gw_start
  [ -s "$ready" ] || { log "usage gateway did not start within ${GATEWAY_START_WAIT} s (see $JOBS/.gateway/gateway.$$.err); leave USAGE_GATEWAY empty to run without it"; exit 2; }
  gw_port="$(cat "$ready")"; GW="http://$GATEWAY_BIND:$gw_port"
  gw_alive || { log "usage gateway $GW does not answer its health check (see $JOBS/.gateway/gateway.$$.err); not starting"; exit 2; }
  # https://host/path -> $GW/u/host/path; http://host:port/path -> $GW/h/host:port/path (`t`: one rewrite only)
  via_gw() { sed -E -e "s#^https://([^/]+)#$GW/u/\1#; t" -e "s#^http://([A-Za-z0-9.-]+:[0-9]+)#$GW/h/\1#" <<< "$1"; }
  if [ "$AGENT" = claude-code ]; then
    AGENT_ARGS+=(--ae "ANTHROPIC_BASE_URL=$(via_gw "$(upstream ANTHROPIC_BASE_URL https://api.anthropic.com)")")
  elif [ "$AGENT" = antigravity ]; then   # the SDK appends /v1beta/models/<model>:streamGenerateContent to it
    AGENT_ARGS+=(--ae "GOOGLE_GEMINI_BASE_URL=$(via_gw "$agent_up")")
  elif [ "$AGENT" = dsh ]; then   # dsh's Messages root (its /v1/messages and /v1/files below it)
    AGENT_ARGS+=(--ae "DEEPSEEK_BASE_URL=$(via_gw "$(upstream DEEPSEEK_BASE_URL https://api.deepseek.com/anthropic)")")
  else
    case "${CODEX_FORCE_AUTH_JSON:-}" in
      1|true|TRUE|yes) AGENT_ARGS+=(--ae "OPENAI_BASE_URL=$(via_gw https://chatgpt.com/backend-api/codex)") ;;
      *) AGENT_ARGS+=(--ae "OPENAI_BASE_URL=$(via_gw "$(upstream OPENAI_BASE_URL https://api.openai.com/v1)")") ;;
    esac
  fi
  # a native config with its own provider URLs (Codex config.toml, Claude Code settings): a copy that points at the gateway
  if [ -n "$AGENT_CONFIG" ]; then
    cfg="$JOBS/.gateway/$(basename "$AGENT_CONFIG").$$"
    sed -E -e "s#^([[:space:]]*base_url[[:space:]]*=[[:space:]]*\")https://([^/\"]+)#\1$GW/u/\2#" \
           -e "s#(\"(ANTHROPIC|OPENAI)_BASE_URL\"[[:space:]]*:[[:space:]]*\")https://([^/\"]+)#\1$GW/u/\3#" "$AGENT_CONFIG" > "$cfg"
    # Codex reaches a provider by WebSocket first; the gateway speaks HTTPS only
    grep -q '^[[:space:]]*base_url[[:space:]]*=' "$cfg" && ! grep -q supports_websockets "$cfg" \
      && sed -i -E "s#^([[:space:]]*)(base_url[[:space:]]*=.*)#\1\2\n\1supports_websockets = false#" "$cfg"
    for i in "${!AGENT_ARGS[@]}"; do [ "${AGENT_ARGS[$i]}" = "config=$AGENT_CONFIG" ] && AGENT_ARGS[$i]="config=$cfg"; done
  fi
  # the egress allowlist must let the agent reach the gateway; with GATEWAY_EGRESS=gateway the list is the gateway's
  # address and the package registries (PyPI and the apt archives: installing packages is allowed, PROTOCOL §2.11), and
  # no model API's own host
  # (run_job: the job's own copy of the task; Harbor's --allow-agent-host can add a host to a task's list, not remove one)
  AGENT_ARGS+=(--allow-agent-host "$GATEWAY_BIND")
  GATEWAY_ONLY="$GATEWAY_BIND"
  # the tool that narrows a job's copy must work on this host before any job is claimed
  if [ "$GATEWAY_EGRESS" = gateway ]; then
    PACKAGE_HOSTS="$(python3 "$R/scripts/set_agent_egress.py" package-hosts 2>/dev/null)"
    GATEWAY_ONLY="$GATEWAY_BIND${PACKAGE_HOSTS:+,$PACKAGE_HOSTS}"      # PACKAGE_HOSTS may be empty: the gateway alone
    probe="$JOBS/.gateway/egress-probe.$$"; rm -rf "$probe"; [[ "$PACKAGE_HOSTS" =~ ^([A-Za-z0-9.-]+(,[A-Za-z0-9.-]+)*)?$ ]] \
      && mkdir -p "$probe/t" && printf '[agent]\ntimeout_sec = 1.0\n' > "$probe/t/task.toml" \
      && python3 "$R/scripts/set_agent_egress.py" only "$GATEWAY_ONLY" "$probe/t" >/dev/null \
      && grep -q "^allowed_hosts = \[\"${GATEWAY_ONLY//,/\", \"}\"\]\$" "$probe/t/task.toml" \
      || { log "scripts/set_agent_egress.py cannot write a gateway-only task copy on this host (python3: $(python3 -V 2>&1)); not starting"; rm -rf "$probe"; exit 2; }
    rm -rf "$probe"
  fi
  # The exception: Codex with a ChatGPT login. Before its first model call Codex 0.160.1 looks its account's workspace up
  # at chatgpt_base_url (GET /backend-api/wham/accounts/check), not at the model's base URL, and accepts only an HTTPS
  # origin there ("workspace backend must use an HTTPS origin"), which this plain-HTTP gateway is not. So chatgpt.com
  # itself stays reachable for such a run (that host alone: not *.chatgpt.com, auth.openai.com or api.openai.com), and
  # with it, directly, every path of that backend: the account's connectors are then off by the harness's configuration
  # only (docs/RUNNING.md). API-key and Claude Code runs have no such exception.
  case "$AGENT:${CODEX_FORCE_AUTH_JSON:-}" in codex:1|codex:true|codex:TRUE|codex:yes)
    GATEWAY_ONLY="$GATEWAY_ONLY,chatgpt.com"
    [ "$GATEWAY_EGRESS" = gateway ] && log "ChatGPT login: chatgpt.com stays directly reachable beside the gateway (Codex's workspace lookup needs it over HTTPS)" ;;
  esac
  log "usage gateway $GW (pid $gw_pid, raw bodies: $GATEWAY_RAW, token budget $TOKEN_BUDGET_IN in / $TOKEN_BUDGET_OUT out, image cap: $GATEWAY_IMAGE_CAP$([ "$GATEWAY_IMAGE_BYTES" != 0 ] && printf ' (and %s bytes)' "$GATEWAY_IMAGE_BYTES"), agent egress: $GATEWAY_EGRESS); per-trial records in <trial>/gateway/"
fi

gateway_check() {  # before a job is claimed (USAGE_GATEWAY set): the gateway answers, else it is restarted on its port; 1 = it is
  # down for good and the lane stops (every trial would end as an infrastructure failure)
  [ -n "$USAGE_GATEWAY" ] || return 0
  gw_alive && return 0
  log "usage gateway $GW (pid $gw_pid) does not answer; restarting it on port $gw_port"
  kill "$gw_pid" 2>/dev/null; sleep 1
  gw_start "$gw_port" && gw_alive && { log "usage gateway restarted (pid $gw_pid)"; return 0; }
  log "STOP lane $$: the usage gateway could not be restarted on port $gw_port (see $JOBS/.gateway/gateway.$$.err)"; return 1
}

gateway_after() {  # <job>: the gateway died during the job: its trials are infrastructure failures, not the agent's 0
  # (<job>/gateway_down.json, read by account_failure_check.py and write_task_results.py)
  [ -n "$USAGE_GATEWAY" ] && [ -d "$JOBS/$1" ] || return 0
  gw_alive && return 0
  printf '{"gateway": "%s", "pid": "%s", "checked": "%s"}\n' "$GW" "$gw_pid" "$(date -u +%FT%TZ)" > "$JOBS/$1/gateway_down.json"
  log "$1: the usage gateway was down when the job ended: its trials are not results (gateway_down.json)"
}

wait_for() {  # <mode>: block until every WAIT_FOR file of this mode exists
  local w
  for w in $WAIT_FOR; do
    if [ "${w%%:*}" = "$1" ]; then
      log "$1 waits for ${w#*:}"
      until [ -e "${w#*:}" ]; do sleep 30; done
    fi
  done
}

usage_record() {  # <job>: <job>/usage_recording.json, whether this job's trials ran through the usage gateway (and its
  # token budget and image cap; "image_bytes" when GATEWAY_IMAGE_BYTES is on; with AGENT=antigravity "api_translation":
  # the gateway's Gemini -> Chat Completions translation); write_task_results.py copies it into every result's config.json
  if [ -n "$USAGE_GATEWAY" ]; then
    printf '{"usage_gateway": true, "raw_bodies": %s, "token_budget": {"input": %s, "output": %s}, "image_cap": %s%s%s}\n' \
      "$([ "$GATEWAY_RAW" = 1 ] && echo true || echo false)" "$TOKEN_BUDGET_IN" "$TOKEN_BUDGET_OUT" "$GATEWAY_IMAGE_CAP" \
      "$([ "$GATEWAY_IMAGE_BYTES" != 0 ] && printf ', "image_bytes": %s' "$GATEWAY_IMAGE_BYTES")" \
      "$([ "$AGENT" = antigravity ] && printf ', "api_translation": "gemini-chat"')"
  else
    printf '{"usage_gateway": false, "token_budget": null, "image_cap": null}\n'
  fi > "$JOBS/$1/usage_recording.json"
}

finish_job() {  # <job>: the usage record; the raw logs compacted; no key in any saved file (+redact); the task results (D3)
  [ -d "$JOBS/$1" ] || return 0
  usage_record "$1"
  [ -n "$USAGE_GATEWAY" ] && [ "$GATEWAY_RAW" = 1 ] && python3 "$USAGE_GATEWAY/rawlog.py" compact "$JOBS/$1" 2>&1 | sed "s/^/[$1] /" | tee -a "$LOG"
  python3 "$H/scan_secrets.py" "$JOBS/$1" "${SCAN[@]}" --redact 2>&1 | tail -3 | sed "s/^/[$1] /" | tee -a "$LOG"
  [ -n "$RESULTS" ] && python3 "$H/write_task_results.py" "$JOBS" "$1" "$RUN_ID" "$LABEL" --out "$RESULTS/$batch" 2>&1 | tail -3 | sed "s/^/[$1] /" | tee -a "$LOG"
}

summary() {  # one table per comparable configuration over every job of this batch, rewritten after each job (atomically:
  # lanes and their validating trials share it; $BASHPID, as a lane's background trials share its $$, read before the
  # redirection: a redirection's words expand in the forked child)
  local tmp="$JOBS/$batch.summary.md.$BASHPID"
  python3 - "$JOBS" "$batch" "$R" > "$tmp" <<'PY'
import json, pathlib, sys, datetime as dt
from collections import defaultdict
jobs, batch, R = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
sys.path[:0] = [str(R / "scripts" / "harbor"), str(R / "scripts")]
import task_conventions as tc, write_task_results as wtr  # noqa: E402
from task_names import canonical  # noqa: E402
# RUNNING.md "Results" (PROTOCOL v2.0): results are compared only within one configuration (task_conventions.config_key:
# protocol version, mode, cutoff, reset budget, S, network); validated and unvalidated tasks (task_conventions.validated_of)
# are ranked together, the unvalidated ones marked; a trial that is not a result (infrastructure, voided) never counts in a
# success rate. Tasks of the easy set (docs/easy_set.toml, task_conventions.set_of) are listed, marked, and left out of
# the headline rate (PROTOCOL §4).
# token columns: the harness's own counts (Harbor's result.json); the usage gateway's column only when a trial of the
# batch ran through it (USAGE_GATEWAY, optional)
HEAD = "| task | trial | seed | success | result | reward fields | actions | agent wall s | input tok | cached | output tok |{gw} cost $ | exception |"
f = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
groups = defaultdict(list)          # config key -> [(started, row, status, validated, easy)]
per_task = defaultdict(lambda: defaultdict(list))   # config key -> task -> [(trial, seed, status)]: R4, a task over its trials
labels = {}
any_gw = False
for trial in jobs.glob(f"{batch}-*/*__*"):
    if not trial.is_dir():
        continue
    r = json.loads((trial / "result.json").read_text()) if (trial / "result.json").exists() else {}
    rw = json.loads((trial / "verifier/reward.json").read_text()) if (trial / "verifier/reward.json").exists() else {}
    ar = r.get("agent_result") or {}; ex = (r.get("exception_info") or {}).get("exception_type") or ""
    ae = r.get("agent_execution") or {}
    gw = json.loads((trial / "gateway/totals.json").read_text()) if (trial / "gateway/totals.json").exists() else {}
    gw_cell = (f"{gw['input']} / {gw['output']}" + (" OVER BUDGET" if gw.get("over_token_budget") else "")) if gw else "-"
    # requests the gateway refused because they were not model-API calls (PROTOCOL §2.11): attempts, for a look; they
    # do not void the trial, and the harness's own start-up probes are not among them
    left = sum(gw.get(k, 0) for k in ("refused_paths", "refused_server_tools", "refused_bodies", "refused_hosts"))
    gw_cell += f" REFUSED {left}" if left else ""
    any_gw = any_gw or bool(gw)
    try:
        wall = round((f(ae["finished_at"]) - f(ae["started_at"])).total_seconds())
    except Exception:
        wall = ""
    task = canonical((r.get("task_name") or trial.name.split("__")[0]).split("/")[-1])
    if r and (R / "tasks" / task / "task.toml").is_file():
        status, why, _ = wtr.classify(trial, r, rw)
        cfg = wtr.trial_config(trial, task, r)
    else:
        status, why, cfg = ("not_a_result", "no result.json", {}) if not r else ("legacy", None, {"mode": task.rsplit("-", 1)[-1]})
    if "trial" not in cfg:
        cfg["trial"] = wtr.trial_index(trial, task, r)[0] if r else "-"
        rec = (wtr.seed_record(trial).get("tasks") or {}).get(task) or {}
        cfg.setdefault("seed", rec.get("seed", "-"))
    key = tc.config_key(cfg)
    labels[key] = tc.config_label(cfg)
    keys = {k: rw[k] for k in ("deterministic", "live_success", "replay_success", "instance_ok", "missing_trajectory", "grader_error",
                               "infra_error", "resumes") if k in rw}
    res = "not a result (voided" + (f": {why}" if why else "") + ")" if status == "not_a_result" else status
    val = cfg.get("validated") is True
    easy = (R / "tasks" / task / "task.toml").is_file() and tc.set_of(R / "tasks" / task) == "easy"
    per_task[key][task].append((cfg.get("trial"), cfg.get("seed"), status, easy))
    groups[key].append((r.get("started_at") or "~", (f"| {task}{'' if val else ' (unvalidated)'}{' (easy set)' if easy else ''} | {cfg.get('trial')} | {cfg.get('seed')} | {rw.get('success', '-')} | {res} | {json.dumps(keys)} | "
                        f"{rw.get('n_actions', '-')} | {wall} | {ar.get('n_input_tokens', '-')} | {ar.get('n_cache_tokens', '-')} | "
                        f"{ar.get('n_output_tokens', '-')} |", f" {gw_cell} |", f" {round(ar['cost_usd'], 4) if ar.get('cost_usd') is not None else '-'} | {ex} |"),
                        status, val, easy))
head = HEAD.format(gw=" usage gateway in / out |" if any_gw else "")
head += "\n" + "|---" * head.count(" | ") + "|---|"
cell = lambda row: row[0] + (row[1] if any_gw else "") + row[2]      # (before, the gateway's cell, after)
print(f"# {batch}\n\nOne table per comparable configuration (RUNNING.md \"Results\"); success rates count results only "
      "(a trial that is not a result is voided and never counted). Validated and unvalidated tasks are ranked together; "
      "tasks without a validated oracle (`oracle` not `full` and no `[metadata] validated = true`), and trials of a task "
      "version this checkout does not have, are marked (unvalidated). Tasks of the easy set (docs/easy_set.toml) are "
      "marked (easy set) and left out of the success rate (PROTOCOL §4).")
for k in sorted(groups, key=lambda k: tuple(map(str, k))):
    rows = groups[k]
    counted = [(s, v) for _, _, s, v, e in rows if s in ("success", "fail") and not e]
    easy_n = sum(1 for _, _, s, _, e in rows if s in ("success", "fail") and e)
    rate = f"{sum(s == 'success' for s, _ in counted)} / {len(counted)}" if counted else "-"
    cv = [s for s, v in counted if v]
    vrate = f" (validated tasks: {sum(s == 'success' for s in cv)} / {len(cv)})" if len(cv) != len(counted) else ""
    print(f"\n## {labels[k]}\n\nSuccess (results only): {rate}{vrate}; "
          f"{sum(s == 'not_a_result' for _, _, s, _, _ in rows)} voided"
          + (f"; {easy_n} result(s) of easy-set tasks not counted" if easy_n else "") + f".\n\n{head}")
    for _, row, _, _, _ in sorted(rows):
        print(cell(row))
    tt = per_task[k]
    if any(len(v) > 1 for v in tt.values()):     # R4: each task over its trials (results only; the easy set marked)
        print("\nPer task over its trials (results only):\n\n| task | successes / results | trials (trial: seed, result) |\n|---|---|---|")
        for task in sorted(tt):
            v = sorted(tt[task], key=lambda x: str(x[0]))
            res = [x for x in v if x[2] in ("success", "fail")]
            print(f"| {task}{' (easy set)' if v[0][3] else ''} | {sum(x[2] == 'success' for x in res)} / {len(res)} | "
                  + ", ".join(f"{x[0]}: s{x[1]} {x[2]}" for x in v) + " |")
PY
  mv -f "$tmp" "$JOBS/$batch.summary.md"
}

stamp_task_dirs() {  # <root>: give every file of the job's task copies a mtime derived from its content (scripts/stamp_mtimes.py)
  # BuildKit keeps one shared snapshot per build-context NAME (Harbor's are always "environment" and "tests") and re-sends a file
  # only when its name, size, mode or mtime changed: a tree whose files share mtimes (git checkout, tar, cp -r, sed loops) can
  # build one task's image with ANOTHER task's equal-size file. A content-derived mtime never matches another
  # content's. Always on the job's own copy: a checkout update during the trial cannot reset it before the verifier is built.
  python3 "$R/scripts/stamp_mtimes.py" --full "$1/tasks" >/dev/null
}

seed_of() {  # <job> <task>: the seed the plan gives this task in this job
  awk -F'\t' -v j="$1" -v t="$2" '$1 == j && $2 == t { print $5; exit }' "$PLAN"
}

seed_copy() {  # <job> <dir> <task>: <dir>/tasks/<task> = the task from the seed the plan gives it (scripts/task_seed.py
  # materialize: that seed's instance or seed file and goal images in the copy; a single-seed task's copy is the task as it is)
  local seed; seed="$(seed_of "$1" "$3")"
  [[ "$seed" =~ ^-?[0-9]+$ ]] || { log "$1: no seed for $3 in the plan $PLAN"; return 1; }
  python3 "$R/scripts/task_seed.py" materialize "$R/tasks/$3" --seed "$seed" --out "$2/tasks" >/dev/null
}

gateway_copy() {  # <job> <dir> <harbor -i args...>: <dir>/tasks/<task> = the task (from its seed) with the gateway and the package registries as its only egress (PROTOCOL §2.11)
  local job="$1" dir="$2" a; shift 2
  rm -rf "$dir" && mkdir -p "$dir/tasks" || return 1
  while [ $# -gt 0 ]; do
    a="$1"; shift
    if [ "$a" = -i ] && [ $# -gt 0 ]; then
      if [ "$(python3 "$R/scripts/set_agent_egress.py" network "$R/tasks/$1")" = open ]; then    # the task declares open egress (`[metadata] network = "open"`, PROTOCOL §2): the copy keeps it
        seed_copy "$job" "$dir" "$1" || return 1; shift; continue
      fi
      seed_copy "$job" "$dir" "$1" && python3 "$R/scripts/set_agent_egress.py" only "$GATEWAY_ONLY" "$dir/tasks/$1" >/dev/null || return 1
      grep -q "^allowed_hosts = \[\"${GATEWAY_ONLY//,/\", \"}\"\]\$" "$dir/tasks/$1/task.toml" || return 1     # fail closed
      shift
    fi
  done
}

plain_copy() {  # <job> <dir> <harbor -i args...>: <dir>/tasks/<task> = the task from its seed
  local job="$1" dir="$2" a; shift 2
  rm -rf "$dir" && mkdir -p "$dir/tasks" || return 1
  while [ $# -gt 0 ]; do
    a="$1"; shift
    if [ "$a" = -i ] && [ $# -gt 0 ]; then seed_copy "$job" "$dir" "$1" || return 1; shift; fi
  done
}

seed_record() {  # <job>: the job's trial and the seed of each of its tasks, $JOBS/.seeds/<job>.json (before the job
  # starts) and <job>/seed.json (after it); write_task_results.py, the summary and regrade.py read it
  mkdir -p "$JOBS/.seeds"
  python3 - "$R" "$PLAN" "$batch" "$1" "$JOBS/.seeds/$1.json" <<'PY'
import json, os, pathlib, sys
repo, plan, batch, job, out = sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(repo) / "scripts"))
import task_seed  # noqa: E402
rows = [l.rstrip("\n").split("\t") for l in open(plan) if l.split("\t")[0] == job]
tasks = {}
for _, task, mode, k, seed in rows:
    d = pathlib.Path(repo) / "tasks" / task
    sd = task_seed.seeds(d)
    dg = (d / "DIGEST").read_text().strip() if (d / "DIGEST").is_file() else None
    tasks[task] = {"seed": int(seed), "default": sd["default"], "allowed": sd["allowed"], "build": sd["build"], "digest": dg}
rec = {"batch": batch, "job": job, "trial": int(rows[0][3]), "trials": None, "seed_source": "runner",
       "assignment": "scripts/task_seed.py assign (batch name, task base name)", "tasks": tasks}
try:
    rec["trials"] = json.load(open(pathlib.Path(plan).with_suffix(".json"))).get("trials")
except (OSError, ValueError):
    pass
tmp = pathlib.Path(f"{out}.{os.getpid()}")
tmp.write_text(json.dumps(rec, indent=1) + "\n")
tmp.replace(out)
PY
}

task_dirs() {  # <root> <harbor -i args...>: the task directories the -i arguments name under <root>/tasks
  local root="$1" a; shift
  while [ $# -gt 0 ]; do
    a="$1"; shift
    if [ "$a" = -i ] && [ $# -gt 0 ]; then printf '%s\n' "$root/tasks/$1"; shift; fi
  done
}

run_job() {  # <job> <concurrency> <harbor -i args...> (each task from the seed the plan gives it in this job): the copy, then Harbor
  prepare_job "$@" && harbor_job "$@"
}

JOB_ROOT=""   # the job's own copy of its tasks (prepare_job)
prepare_job() {  # <job> <concurrency> <harbor -i args...>: JOB_ROOT = the job's copy, ready for Harbor; 1 = not started
  local job="$1" n="$2"; shift 2
  local root="$R" f dirs=()
  mapfile -t dirs < <(task_dirs "$R" "$@")
  # every job runs from its own copy of the task ($JOBS/.tasks/<job>/, removed afterwards), stamped; PROTOCOL §2.10:
  # a CPU cap above the host's CPU count is clamped to it in that copy (docker refuses such a container: "range of CPUs is
  # from 0.01 to 8.00"), never in tasks/
  local clamp=0; python3 "$H/clamp_cpus.py" --cpus "$HOST_CPUS" --needs "${dirs[@]}" >/dev/null && clamp=1
  [[ "$job" =~ ^[A-Za-z0-9._-]+$ ]] || { log "FAILED $job: not a job name (letters, digits, . _ -); not started"; return 1; }
  root="$JOBS/.tasks/$job"
  if [ -n "$USAGE_GATEWAY" ] && [ "$GATEWAY_EGRESS" = gateway ]; then
    gateway_copy "$job" "$root" "$@" || { log "FAILED $job: could not prepare its task copy (the gateway and the package registries as the only egress) ($root); not started"; rm -rf "$root"; return 1; }
  else
    plain_copy "$job" "$root" "$@" || { log "FAILED $job: could not prepare its task copy ($root); not started"; rm -rf "$root"; return 1; }
  fi
  for f in .env .env.local; do [ -e "$R/$f" ] && ln -s "$R/$f" "$root/$f"; done     # what Harbor would load from the repository
  if [ "$clamp" = 1 ]; then
    mapfile -t dirs < <(task_dirs "$root" "$@")
    python3 "$H/clamp_cpus.py" --cpus "$HOST_CPUS" "${dirs[@]}" | sed "s#^$root/tasks/#[$job] clamped #" | tee -a "$LOG" >/dev/null \
      && ! python3 "$H/clamp_cpus.py" --cpus "$HOST_CPUS" --needs "${dirs[@]}" >/dev/null \
      || { log "FAILED $job: could not clamp its CPU caps to the host's $HOST_CPUS ($root); not started"; rm -rf "$root"; return 1; }
  fi
  # GPU_SCHED=1: every NVIDIA device of the copy (main or sim, rcb_monitor, the verifier) pinned to the job's GPU
  if [ -n "$JOB_GPU" ]; then
    mapfile -t dirs < <(task_dirs "$root" "$@")
    python3 "$H/pin_gpu.py" --gpu "$JOB_GPU" "${dirs[@]}" >/dev/null && python3 "$H/pin_gpu.py" --gpu "$JOB_GPU" --check "${dirs[@]}" >/dev/null \
      || { log "FAILED $job: could not pin its containers to GPU $JOB_GPU (scripts/harbor/pin_gpu.py, $root); not started"; rm -rf "$root"; return 1; }
  fi
  stamp_task_dirs "$root" || { log "FAILED $job: could not stamp its task copy (scripts/stamp_mtimes.py, $root); not started"; rm -rf "$root"; return 1; }
  seed_record "$job" || { log "FAILED $job: could not write its seed record ($JOBS/.seeds/$job.json); not started"; rm -rf "$root"; return 1; }
  rm -f "$JOBS/.phase/$job.json" "$JOBS/.phase/$job.handed"     # an earlier attempt's agent-end records (PHASE_SPLIT)
  log "start $job: $* (concurrency $n; trial $(awk -F'\t' -v j="$job" -v K="$K" '$1 == j { print $4 " of " K; exit }' "$PLAN"), seeds $(awk -F'\t' -v j="$job" '$1 == j { printf "%s%s=%s", s, $2, $5; s = " " }' "$PLAN")${JOB_SLOT:+; ${JOB_GPU:+GPU $JOB_GPU, }reserved $(awk '{ print $2 " CPUs, " $3 " GB" }' <<< "$JOB_SLOT")})"
  JOB_ROOT="$root"
}

harbor_job() {  # <job> <concurrency> <harbor -i args...>: Harbor runs the job from its copy (JOB_ROOT), which is removed
  # afterwards; the job's seed, machine and placement records. PHASE_SPLIT=1: the agent-end marker plugin
  local job="$1" n="$2" root="$JOB_ROOT"; shift 2
  local envf=(); [ -n "$ENV_FILE" ] && envf=(--env-file "$ENV_FILE")
  local phase=(); [ "$PHASE_SPLIT" = 1 ] && phase=(--plugin phase_marker:PhaseMarker --pk "out=$JOBS/.phase/$job.json" --pk "job=$job")
  (cd "$root" && harbor run -p tasks "$@" "${AGENT_ARGS[@]}" "${ENV_ARGS[@]}" "${phase[@]}" "${envf[@]}" -n "$n" -k 1 -y -o "$JOBS" --job-name "$job") > "$JOBS/$job.harbor.log" 2>&1
  log "done $job (harbor rc=$?)"
  [ -d "$JOBS/$job" ] && cp "$JOBS/.seeds/$job.json" "$JOBS/$job/seed.json"
  [ -d "$JOBS/$job" ] && [ -n "$HOST_INFO" ] && printf '%s\n' "$HOST_INFO" > "$JOBS/$job/host.json"
  [ -d "$JOBS/$job" ] && [ -n "$JOB_SLOT" ] && placement "$job"
  rm -rf "$root"
  return 0
}

agent_phase_end() {  # <job> <pid of its background run>: 0 = its agent phase has ended and it validates (the marker of
  # phase_marker.py, PHASE_SPLIT=1); 1 = the run has ended without a marker (PHASE_SPLIT=0: waited for). A run whose
  # marker exists waits for the lane's hand-over (run_in_background), so the lane always sees that marker
  [ "$PHASE_SPLIT" = 1 ] || { wait "$2"; return 1; }
  while kill -0 "$2" 2>/dev/null; do
    [ -s "$JOBS/.phase/$1.json" ] && return 0
    sleep 2
  done
  wait "$2"; return 1
}

run_in_background() {  # <job> <harbor -i args...>: Harbor runs the job, then its records, results and summary, and its
  # slot released. The usage gateway serves the agent phase only: when the job's agent phase ended with a marker, the
  # lane checked the gateway then (gateway_after) and hands the job over (<job>.handed, after `gpu_slots.py phase`);
  # the results wait for that (at most 120 s: a lane killed meanwhile), else the gateway is checked here, as before
  local job="$1" i; shift
  harbor_job "$job" 1 "$@"
  if [ -s "$JOBS/.phase/$job.json" ]; then
    for i in $(seq 60); do [ -e "$JOBS/.phase/$job.handed" ] && break; sleep 2; done
  else
    gateway_after "$job"
  fi
  finish_job "$job"; summary
  release_slot "$job"
}

placement() {  # <job>: <job>/placement.json, the GPU and the reservation the trial ran with (GPU_SCHED=1), and what it
  # kept while it validated (PHASE_SPLIT=1)
  local units cpus mem fam
  read -r units cpus mem fam <<< "$JOB_SLOT"
  printf '{"gpu": %s, "units": %s, "gpu_share": %s, "cpus": %s, "memory_gb": %s, "family": "%s", "gpus": "%s", "host_cpus": %s, "phase_split": %s, "gpu_resident": %s, "validate_cpus": %s, "validate_memory_gb": %s}\n' \
    "$([ -n "$JOB_GPU" ] && echo "\"$JOB_GPU\"" || echo null)" "$units" "$GPU_SHARE" "$cpus" "$mem" "$fam" "$GPUS" "$HOST_CPUS" \
    "$([ "$PHASE_SPLIT" = 1 ] && echo true || echo false)" "$GPU_RESIDENT" "${JOB_VALIDATE% *}" "${JOB_VALIDATE#* }" > "$JOBS/$1/placement.json"
}

JOB_GPU=""; JOB_SLOT=""   # the job's GPU ("" = none or GPU_SCHED=0) and "<units> <cpus> <memory GB> <family>" (GPU_SCHED=1)
JOB_VALIDATE=""           # "<cpus> <memory GB>" the job keeps while it validates (GPU_SCHED=1)
SLOT_WHY="$JOBS/.slots.why.$$"
declare -A SLOT_CLASS=()  # task -> "<units> <cpus> <memory GB> <family>" (gpu_slots.py classify, once per task and lane)
declare -A VALIDATE_NEED=()   # task -> "<cpus> <memory GB>" while it validates (gpu_slots.py classify --phase validate)
slot_class() {  # <task>: SLOT_CLASS[<task>] and VALIDATE_NEED[<task>] filled; 2 = the task could not be classified (why in $SLOT_WHY)
  [ -n "${SLOT_CLASS[$1]:-}" ] && return 0
  local c v
  c="$(python3 "$H/gpu_slots.py" classify "$R/tasks/$1" ${DEFORMABLE_FAMILIES:+--deformable "$DEFORMABLE_FAMILIES"} \
        --share "$GPU_SHARE" --cpu-scale "$RESERVE_CPU_SCALE" --mem-scale "$RESERVE_MEM_SCALE" 2>"$SLOT_WHY")" || return 2
  v="$(python3 "$H/gpu_slots.py" classify "$R/tasks/$1" --phase validate --cpu-scale "$RESERVE_CPU_SCALE" \
        --mem-scale "$RESERVE_MEM_SCALE" 2>"$SLOT_WHY")" || return 2
  SLOT_CLASS[$1]="$c"; VALIDATE_NEED[$1]="$v"
}
acquire_slot() {  # <job> <task>: JOB_SLOT, JOB_VALIDATE and JOB_GPU from the host's ledger (call it directly, not in
  # $(...): it sets them); 3 = not now, other = error; why in $SLOT_WHY
  local rc units cpus mem fam
  JOB_GPU=""; JOB_SLOT=""; JOB_VALIDATE=""
  slot_class "$2" || return 2
  read -r units cpus mem fam <<< "${SLOT_CLASS[$2]}"
  JOB_GPU="$(python3 "$H/gpu_slots.py" acquire --dir "$SLOTS_DIR" --job "$1" --runner-pid $$ --units "$units" --cpus "$cpus" \
              --mem-gb "$mem" --gpus "$GPUS" --share "$GPU_SHARE" --resident "$GPU_RESIDENT" --host-cpus "$HOST_CPUS" \
              ${HOST_MEM_GB:+--host-mem-gb "$HOST_MEM_GB"} ${MEM_HEADROOM_GB:+--headroom-gb "$MEM_HEADROOM_GB"} 2>"$SLOT_WHY")"
  rc=$?
  [ $rc -eq 0 ] && { JOB_SLOT="${SLOT_CLASS[$2]}"; JOB_VALIDATE="${VALIDATE_NEED[$2]}"; return 0; }
  JOB_GPU=""; return $rc
}
validate_slot() {  # <job> <pid>: the job's agent phase has ended: its hold validates, owned by <pid> (gpu_slots.py phase)
  python3 "$H/gpu_slots.py" phase --dir "$SLOTS_DIR" --job "$1" --pid "$2" --cpus "${JOB_VALIDATE% *}" \
    --mem-gb "${JOB_VALIDATE#* }" >/dev/null 2>&1 \
    && log "$1: agent phase ended; it validates${JOB_GPU:+ on GPU $JOB_GPU} (keeps ${JOB_VALIDATE% *} CPUs, ${JOB_VALIDATE#* } GB) while lane $$ takes the next trial"
  JOB_GPU=""; JOB_SLOT=""; JOB_VALIDATE=""
  return 0
}
release_slot() {  # <job>
  [ "$GPU_SCHED" = 1 ] && python3 "$H/gpu_slots.py" release --dir "$SLOTS_DIR" --job "$1" >/dev/null 2>&1
  JOB_GPU=""; JOB_SLOT=""; JOB_VALIDATE=""
  return 0
}

name() {  # <mode> <base> -> task dir name: <base>-privileged / <base>-standard / <base>-standard-realtime (PROTOCOL §2.2)
  case "$1" in
    standard|limited) echo "$2-standard" ;;
    standard-realtime) echo "$2-standard-realtime" ;;
    *) echo "$2-privileged" ;;
  esac
}

not_started=0
for t in "${tasks[@]}"; do
  for m in $MODES; do [ -d "$R/tasks/$(name "$m" "$t")" ] || log "skip $batch-$m-$t (no task dir $(name "$m" "$t"))"; done
done
if [ "$SEQ" = 1 ]; then
  mkdir -p "$JOBS/.claims"
  # passes over the plan: a job that does not fit the host now (GPU_SCHED=1) is left for a later pass and the lane tries
  # the next one; a pass that starts nothing while some job waits for room sleeps 30 s; the lane ends when no job is
  # left unclaimed (or it stops on a failure)
  pass=0; stop=0; said=""; said_at=0
  validating=()   # this lane's trials that validate in the background (PHASE_SPLIT=1)
  seed_wait() {  # <seed>: SEEDS: a job of this seed waits for room in this pass. open_seed: the lowest such seed (a later
    # seed's trial starts only in room no job of it fits); open_n: its jobs that wait
    [ -n "$SEEDS" ] || return 0
    [ -z "$open_seed" ] && open_seed="$1"
    [ "$1" = "$open_seed" ] && open_n=$((open_n + 1))
    return 0
  }
  while [ "$stop" = 0 ]; do
    pass=$((pass + 1)); started=0; waiting=0; first_why=""; open_seed=""; open_n=0; again=0
    declare -A nofit=()   # this pass: the needs ("<units> <cpus> <memory GB>") the host had no room for; later jobs with
    # the same needs wait without asking the ledger again (a pass over a long plan stays cheap)
    while IFS=$'\t' read -r job task m k seed <&3; do
      if [ -f "$JOBS/$job/result.json" ]; then [ "$pass" = 1 ] && log "skip $job (already has result.json)"; continue; fi
      if [ "$GPU_SCHED" = 1 ] && slot_class "$task"; then
        need="${SLOT_CLASS[$task]% *}"
        # SEEDS: while a job of an earlier seed waits for room, a later seed's deformable trial is not tried (the ledger
        # would keep a GPU for it, or drain one)
        if [ -n "$open_seed" ] && [ "$seed" -gt "$open_seed" ] && [ "${need%% *}" -ge "$GPU_SHARE" ] && [ ! -d "$JOBS/.claims/$job" ]; then
          waiting=$((waiting + 1)); continue
        fi
        [ -n "${nofit[$need]:-}" ] && [ ! -d "$JOBS/.claims/$job" ] && { waiting=$((waiting + 1)); seed_wait "$seed"; continue; }
      fi
      mkdir "$JOBS/.claims/$job" 2>/dev/null || continue   # another lane of this batch took it
      wait_for "$m"
      gateway_check || { rmdir "$JOBS/.claims/$job" 2>/dev/null; stop=1; break; }
      if [ "$GPU_SCHED" = 1 ]; then
        acquire_slot "$job" "$task"; rc=$?
        why="$(tail -1 "$SLOT_WHY" 2>/dev/null)"
        if [ $rc -eq 3 ]; then
          [ -n "${SLOT_CLASS[$task]:-}" ] && nofit["${SLOT_CLASS[$task]% *}"]=1
          rmdir "$JOBS/.claims/$job" 2>/dev/null; waiting=$((waiting + 1)); seed_wait "$seed"; [ -z "$first_why" ] && first_why="$job: $why"; continue
        elif [ $rc -ne 0 ]; then
          log "FAILED $job: no GPU slot ($why); not started"; rmdir "$JOBS/.claims/$job" 2>/dev/null; not_started=$((not_started + 1)); continue
        fi
      fi
      [ -n "$open_seed" ] && [ "$seed" -gt "$open_seed" ] && log "lane $$: $job (seed $seed) starts while $open_n job(s) of seed $open_seed wait for room"
      started=$((started + 1))
      prepare_job "$job" 1 -i "$task" || { release_slot "$job"; rmdir "$JOBS/.claims/$job" 2>/dev/null; not_started=$((not_started + 1)); continue; }   # another lane or a later run may take it
      # the trial runs in the background; the lane goes on when its agent phase has ended (PHASE_SPLIT=1: it validates
      # beside the next trial, its slot a validating one) or when it has ended
      run_in_background "$job" -i "$task" &
      run_pid=$!
      if agent_phase_end "$job" "$run_pid"; then
        gateway_after "$job"; validate_slot "$job" "$run_pid"; : > "$JOBS/.phase/$job.handed"
        validating+=("$run_pid")
      fi
      JOB_GPU=""; JOB_SLOT=""; JOB_VALIDATE=""
      # the model account is out of credits, rate-limited or logged out: every further job would fail within minutes. Only
      # structured evidence counts (Codex's rate_limits and last event, Claude Code's result event): account_failure_check.py
      why=$(python3 "$H/account_failure_check.py" "$JOBS/$job"/*__*/ 2>&1); rc=$?
      if [ $rc -eq 4 ]; then
        log "STOP lane $$: $job hit a model-account or config failure, or its model API was unreachable through the gateway ($(grep -m1 '"stop": true' <<< "$why" | cut -c1-300)); move its dir and claim aside to rerun it"; stop=1; break
      elif [ $rc -ne 0 ]; then
        log "account check of $job failed (rc $rc): $(tail -1 <<< "$why" | cut -c1-200)"
      fi
      [ -n "$SEEDS" ] && { again=1; break; }   # SEEDS: after each trial RUN_PLAN is read from the top: an earlier seed's job first
    done 3< "$RUN_PLAN"
    [ "$again" = 1 ] && [ "$stop" = 0 ] && continue
    [ "$stop" = 1 ] || [ "$waiting" = 0 ] && break       # every job is claimed or done: this lane is finished
    if [ "$started" = 0 ]; then
      # what the host lacks for the first job that waits: a line when the kind of shortage (GPU, CPUs, memory) changes,
      # else at most every 10 minutes
      kind="$(sed -E 's/^[^:]*: ([A-Za-z]+).*/\1/' <<< "$first_why")"
      if [ "$kind" != "$said" ] || [ $(( $(date +%s) - said_at )) -ge 600 ]; then
        log "lane $$ waits: $waiting job(s) do not fit the host now; first ${first_why:-(none tried this pass)}"
        said="$kind"; said_at=$(date +%s)
      fi
      sleep 30
    fi
  done
  left=0; for p in "${validating[@]}"; do kill -0 "$p" 2>/dev/null && left=$((left + 1)); done
  [ "$left" -gt 0 ] && log "lane $$: no trial left to start; waiting for $left trial(s) of this lane to finish validating"
  [ ${#validating[@]} -gt 0 ] && wait "${validating[@]}"
  rm -f "$SLOT_WHY"
else
  pids=()
  for m in $MODES; do
    mapfile -t mjobs < <(awk -F'\t' -v m="$m" '$3 == m && !seen[$1]++ { print $1 }' "$PLAN")
    [ ${#mjobs[@]} -eq 0 ] && continue
    n="$N_P"; case "$m" in standard|limited|standard-realtime) n="$N_S" ;; esac   # sidecar modes
    (wait_for "$m"
     for job in "${mjobs[@]}"; do                 # the mode's trials one after another, each job all its tasks at once
       inc=(); while IFS= read -r task; do inc+=(-i "$task"); done < <(awk -F'\t' -v j="$job" '$1 == j { print $2 }' "$PLAN")
       run_job "$job" "$n" "${inc[@]}"
     done) &
    pids+=($!)
  done
  wait "${pids[@]}"
  while IFS= read -r job; do gateway_after "$job"; finish_job "$job"; done < <(awk -F'\t' '!seen[$1]++ { print $1 }' "$PLAN")
fi
[ -d "${JOBS}_mirror" ] && python3 "$H/scan_secrets.py" "${JOBS}_mirror" "${SCAN[@]}" 2>&1 | tail -2 | sed "s/^/[mirror] /" | tee -a "$LOG"
summary
log "summary: $JOBS/$batch.summary.md"
cat "$JOBS/$batch.summary.md" | tee -a "$LOG" >/dev/null
[ "$not_started" -gt 0 ] && log "WARNING: $not_started job(s) of batch $batch were NOT started (their task copy could not be prepared, see FAILED above)"
log "runner $$ of batch $batch finished (no unclaimed job left)"
