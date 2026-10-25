# Running RoboARC with Harbor

How to run RoboARC's tasks with [Harbor](https://github.com/laude-institute/harbor), one trial or a batch, and how a run
is configured and recorded. The task contract is [PROTOCOL.md](PROTOCOL.md) (its Appendix H binds every run); the
standard-mode wire protocol is [STANDARD_MODE.md](STANDARD_MODE.md); the benchmark's own rules are in
[benchmarks/roboarc.md](benchmarks/roboarc.md). The sections from "Official runs" on are the runner's rules:
`scripts/harbor/run_agent_batch.sh` applies them by default.

## Requirements

- Linux x86_64 with an NVIDIA GPU that has RT cores (Nyx renders through Vulkan; driver >= 580; see "GPUs without RT
  cores" below), Docker with the NVIDIA
  Container Toolkit, Harbor 0.23 or newer, Python >= 3.11 for the repository tools. arm64 hosts are not supported (the
  images are linux/amd64, Nyx needs NVIDIA on x86_64, the standard client's lock lists x86_64 wheels).
- The images the tasks name (`ghcr.io/mll-lab-nu/roboarc:<images/roboarc/VERSION>`; IKEA:
  `roboarc-ikea` and `roboarc-ikea-sim`), pulled or built (docs/DEVELOPING.md). The image packages are private and inherit this repository's access: a collaborator with read access to the repository logs in with `docker login ghcr.io -u <github user>` and a GitHub token with the `read:packages` scope.
- Every trial needs one GPU. A task's minimum allocation is its `[metadata.resources]` (README "Compute resources";
  2 GB of GPU memory, measured peak about 1.1 GB).

### GPUs without RT cores

Nyx, the graded renderer, path-traces with Vulkan ray queries (`VK_KHR_ray_query`) and shares its frames with CUDA on
the same GPU. NVIDIA offers ray queries only on GPUs with RT cores: GeForce RTX 20-series or newer, RTX workstation GPUs, T4, L4, L40 / L40S, A10 / A10G, A40. Data-centre GPUs without RT cores
(V100, A100, A30, H100, H200) do not offer them, and a CPU Vulkan driver (Mesa lavapipe) cannot share memory with CUDA, so there is no
software fallback. Before it creates the renderer, `nyx_direct.py` (IKEA standard: its render process) reads the
GPU's Vulkan extensions and raises an error naming what is missing (`nyx_direct.UnsupportedDevice`; the check takes
about 0.1 s) instead of letting the native renderer crash the process. On such a GPU:

| mode | result |
|---|---|
| standard | does not run: the observations are Nyx images, and the `sim` service stops at start-up with that error |
| privileged, verifier | graded: the verdict comes from the physics replays (MuJoCo on the CPU; blanket's cloth on CUDA, bit-identical on an A100); the replay video, evidence only, is not made (`video_rendered` 0, `media.json` status "error") |
| privileged, agent | the physics tooling works; every camera call (`render_frame`, `render_grid`, `render_video`, `render_depth`, `VIDEO_CAMERAS`) raises `UnsupportedDevice`, which the agent's script can catch |

Such runs are **unofficial**: the agent never sees the scene, so their results are not comparable with runs on a GPU
with RT cores. IKEA's privileged tasks do not use Nyx and run on any host.

## Building task images: stamp first

Docker's BuildKit keeps one synced copy of a build context per context directory name and client (Harbor's contexts are
always `environment/` and `tests/`) and re-sends a file only when its path, size, mode or mtime changed. A task built
from a tree whose files share mtimes (a git checkout, `git archive`, tar, `cp -r`) can therefore get another task's
equal-size file in its image, silently, with `--no-cache` too. **Never build a task image from a tree without stamping
it:** `scripts/stamp_mtimes.py` gives every file a mtime derived from its content (content, modes and DIGEST unchanged;
git ignores mtimes). The batch runner and `scripts/roboarc/regrade.py --replay` build from stamped copies.

```bash
python3 scripts/stamp_mtimes.py tasks                              # stamp the checkout in place (again after every pull: --check tells)
c=$(python3 scripts/stamp_mtimes.py stage tasks/<task> "$(mktemp -d)")   # or a stamped copy of one task
python3 scripts/check_image_files.py <image> <context dir>         # after building an image yourself: it holds the context's files
```

## A single trial

```bash
python3 scripts/stamp_mtimes.py tasks
harbor run -p tasks -i roboarc-interlocked-claws-08-i01-privileged -a oracle -o "$PWD/jobs" --job-name oracle-claws-08   # reward 1
harbor run -p tasks -i roboarc-interlocked-claws-08-i01-standard -a nop -o "$PWD/jobs" --job-name nop-claws-08          # reward 0
harbor run -p tasks -i roboarc-orbits-02-i01-standard -a codex -m openai/<model> \
    --ak reasoning_effort=high --ak web_search=disabled -o "$PWD/jobs" --job-name codex-orbits-02
```

- Use Harbor's native agents (`-a codex`, `-a claude-code`) with their credentials in the environment or an
  `--env-file`: Codex `OPENAI_API_KEY` (OpenRouter: also `OPENAI_BASE_URL=https://openrouter.ai/api/v1`), a ChatGPT
  login with `CODEX_FORCE_AUTH_JSON=1` (Harbor uploads `~/.codex/auth.json`); Claude Code `ANTHROPIC_API_KEY`, or a
  subscription with `CLAUDE_FORCE_OAUTH=1` and `CLAUDE_CODE_OAUTH_TOKEN` (`claude setup-token`).
- Turn the harness's provider-side tools off ("No tool outside the container" below): Codex `--ak web_search=disabled`
  and `--ak config=<the output of python3 scripts/harbor/codex_tools_off.py [your config.toml]>`; Claude Code
  `--ak disallowed_tools=WebSearch,WebFetch`.
- Use an absolute `-o`. jobs/ is ignored by git; results never go into the repository.
- The trial's record is `<jobs>/<job>/<trial>/`: `agent/` (the harness's log and trajectory), `verifier/`
  (`reward.txt` 1 or 0, `reward.json`, `replay.mp4`, the resource-limits record), `artifacts/` (privileged: the agent's
  `/app/output/`; standard: the sim service's `/data/` files: `final.json`, `episode.jsonl`, `trajectory.npz`,
  `replay.mp4`, the service log), `result.json`.

## Official runs

A result is official only when its run had all of these:
- **Harbor's native agents**, configured through environment variables; no agent adapter, no proxy that holds
  credentials or changes requests.
- **No tool outside the container**: the harness's web search, account connectors, apps, plugins, image generation and
  cloud agents off (below).
- **The task's own cutoff**, no multiplier (`--timeout-multiplier`, `--agent-timeout-multiplier`, an agent
  `override_timeout_sec`), and **nothing added to the instruction** (no prompt template, no extra instruction).
- **The runner's seed, the task's network, at most one infrastructure retry** (below), and the effective configuration
  recorded with the result.

Recording the agent's model-API usage is **optional** (PROTOCOL H.6): an official run does not need the usage gateway.
Each result records whether usage was recorded and by what ("Usage recording" below).

## Batches: `scripts/harbor/run_agent_batch.sh`

```bash
MODEL=openai/<model> EFFORT=high JOBS="$PWD/jobs/agents" RESULTS=/path/outside/the/repo \
  scripts/harbor/run_agent_batch.sh <batch> roboarc-orbits-02-i01 roboarc-cast-coil-06-i01 ...
```

It runs one Harbor-native agent over `<base>-privileged` and `<base>-standard` of every task base given (`MODES` picks
the modes), `K` trials of each (default 1), one Harbor job per task, mode and trial: `$JOBS/<batch>-<mode>-<task>/`, or
`$JOBS/<batch>-<mode>-<task>-t<k>/` (k = 1..K) when K > 1. Each trial runs from its own seed ("Seeds" below). Starting
the same command again adds a lane: every job is claimed by exactly one runner (`$JOBS/.claims/<job>`); a job that
already has its `result.json` is skipped. Each job runs from its own stamped copy of the task, made from the trial's
seed (`$JOBS/.tasks/<job>/`, removed afterwards; a CPU cap above the host's count is clamped in that copy,
`scripts/harbor/clamp_cpus.py`). The batch's plan (every job, its trial and seeds) is `$JOBS/.plan/<batch>.json`. The
header of the script lists every variable; the main ones:

| what | how |
|---|---|
| agent, model, effort | `AGENT=codex` or `claude-code`, `MODEL` (required), `EFFORT` (`--ak reasoning_effort`, default `xhigh`; `default` passes none), more options in `AK` |
| credentials | as Harbor's agents read them (above); `ENV_FILE` = Harbor's `--env-file` (`.env.example` lists the variables). Azure OpenAI and retries: a Codex `config.toml` in `AGENT_CONFIG` (`--ak config=`), with `request_max_retries` / `stream_max_retries`. For long OpenRouter runs declare OpenRouter as a provider of its own in that file (`model_provider = "openrouter"`, `wire_api = "responses"`, `supports_websockets = false`) |
| trials | `K` per task and mode (1); each its own job and seed |
| web search | `WEB_SEARCH=disabled` (default) or `allowed` (not for official runs) |
| run id | `RUN_ID`, derived as `<harness>-<model>-<effort>` ([TAXONOMY.md](TAXONOMY.md)) |
| results | `RESULTS`: a directory outside the repository (default `../rcb_results` next to the checkout; empty = skip) |
| usage recording | `USAGE_GATEWAY=<dir>`: route the agent's model-API calls through the usage gateway installed in `<dir>` (the repository's: `USAGE_GATEWAY=scripts/usage_gateway`; "Usage recording" below). Empty (default) = off |
| optional host tools | `LOG_MIRROR=1`: an append-only mirror of every trial's `agent/` record in a root-owned container (`rcb-log-mirror`, `scripts/harbor/log_mirror.py`), so neither the agent nor the host user can delete it. `MEM_GUARD=1`: `scripts/harbor/mem_guard.py` keeps 2 GiB of host memory available by killing the largest running RoboARC trial container (names `roboarc-*` only). Both are off by default |

After each job the runner scans the job for keys and redacts them (`scripts/harbor/scan_secrets.py`), writes the
per-task results (`scripts/harbor/write_task_results.py`), rewrites `$JOBS/<batch>.summary.md` (one table per
comparable configuration), and stops its lane when a trial shows a model-account failure
(`scripts/harbor/account_failure_check.py`: out of credits, rate-limited, logged out). Progress is in
`$JOBS/<batch>.log`. Use one key per batch with a spending cap.

To watch a batch: `python3 scripts/harbor/live_board.py --jobs <JOBS>` serves a page on 127.0.0.1 with every trial's
state (pending, setup, running against the task's limit, verifying, done), the reward, the videos and the tail of the
agent's output. It has no login and shows unredacted output: keep it on 127.0.0.1 (reach it with
`ssh -N -L 8765:127.0.0.1:<port> <host>`).

## Network: what the task declares

Every task's `[agent]` allowlist is one list, `HOSTS` in `scripts/set_agent_egress.py`: the model APIs and the package
registries (PyPI: pypi.org, files.pythonhosted.org; apt: deb.debian.org, security.debian.org, archive.ubuntu.com,
security.ubuntu.com, snapshot.ubuntu.com). So `pip install` and `apt-get install` work in every agent container, in both
modes; GitHub, Hugging Face and the general web do not answer (PROTOCOL §5, H.6). Whether a task has open network is
its author's decision (`[metadata] network = "open"`; no RoboARC task has it); `python3 scripts/set_agent_egress.py
network <task>` prints the declaration, and `network` is part of the recorded configuration.

- Another host for every task: add it to `HOSTS`, re-render the tasks (docs/DEVELOPING.md).
- A run with no restriction (not official): run a copy of the tasks, since CI keeps the list in `tasks/`:
  `cp -a tasks /some/dir/tasks && python3 scripts/set_agent_egress.py public /some/dir/tasks/*/`, then
  `harbor run -p /some/dir/tasks ...`.

## No tool outside the container

The harness's own web search runs at the model provider, past any egress allowlist, and so do a ChatGPT login's
connectors, apps, plugins and image generator (Codex) and a claude.ai login's connectors (Claude Code). The batch runner
turns them off unless `WEB_SEARCH=allowed`: Codex gets `--ak web_search=disabled` and a copy of `AGENT_CONFIG` (or a
config of its own) in which `scripts/harbor/codex_tools_off.py` sets `features.apps`, `features.plugins`,
`features.image_generation` and `apps._default.enabled` to false; an `AGENT_CONFIG` that turns one on, or an `AK` with
its own `config=`, is refused. Claude Code gets `--ak disallowed_tools=WebSearch,WebFetch`; its settings stay the
caller's (`AGENT_CONFIG`). An API key can still request a search directly: read the transcript ([LEAKAGE.md](LEAKAGE.md)).

## The agent's time (the cutoff)

A trial's agent gets the task's `[agent] timeout_sec` (10800 s for every RoboARC task but wire-splicing-01-i01, which has 21600 s by the owner's
exception, `scripts/task_conventions.py` `CUTOFF_EXCEPTIONS`) and nothing else. Harbor counts
it around the agent's run only (after the environment has been built and started and the harness installed, each under
a timeout of its own); a standard task's `sim` service starts with the environment, and what it has not finished
starting by then is waited out inside the agent's time. At the cutoff Harbor stops the agent (`AgentTimeoutError`),
collects the artifacts and runs the verifier as after any other end: the trial is graded as it stands, is a result,
and is not run again.

The rules on the number (`ci_checks/check-cutoffs.py`, `check-twins.py`): a whole number of seconds, at most 14400
(or the owner's exception for the task's base, `scripts/task_conventions.py` `CUTOFF_EXCEPTIONS`; `cutoff_s` is part of
the configuration results are ranked by, so an excepted task's trials form their own group); the
privileged and standard versions of a task have the same cutoff (H.7); the instruction's closing `## Time limit` section
(rendered by `scripts/task_conventions.py`, the last thing the agent reads), the README's `Agent time budget` row and,
in standard mode, the sim service's `RCB_WALL_CLOCK_S` state the same number, and nothing else in the instruction states
the limit. The standard service's `time_left_s` counts the same limit from the agent's first request to the service.
`/run/rcb-limits/usage.json` shows the agent its clock. Harbor hands `instruction.md` to the agent as it is, and the
runner adds nothing. `--agent-setup-timeout-multiplier` (for slow harness installs) is fine: set-up is not the agent's
time.

## Seeds

The seed is the runner's, never the agent's: one of the task's `[metadata.seeds] allowed` (PROTOCOL §2). A seed names
one frozen instance of the task; `default` is the one the task directory holds as committed.

**Official batches assign the seeds without replacement.** The K trials of a task get distinct seeds, in an order fixed
by the batch name and the task's base name: the allowed seeds sorted by sha256("`<batch>`\n`<base>`\n`<seed>`"), trial
k running the k-th (`python3 scripts/task_seed.py assign tasks/<task> --batch <batch> --trials K`). So:
- both modes of a base (its privileged and standard twins) run the same seeds;
- with K equal to the number of seeds, every seed runs once; with more trials than seeds the order starts over, so every
  seed runs equally often, give or take one; a task that allows one seed runs it in every trial;
- each trial's seed alone is uniform over `allowed` (the trials of one task are not independent draws);
- a trial run again (the infrastructure retry below, or a rerun of an interrupted job) gets its seed again.

The batch runner makes each job's copy of a task with `scripts/task_seed.py materialize` and the trial's seed: that
seed's instance (a committed one under the privileged twin's `tests/seeds/<S>/`), or, for a task that stages its
instance at run time (`build = "runtime"`), the file `seed` in each place the task stages from (privileged
`environment/` and `tests/`, standard `environment/sim/` and `tests/`), and the seed's goal images
(`tests/seeds/<S>/goal/`) in `environment/goal/`, which the agent sees as `/app/goal/`. The agent never chooses or sees
the list: the copy carries only its seed. The runner records the job's trial and the seed of each of its tasks in
`<job>/seed.json` (and `$JOBS/.seeds/<job>.json`, written before the job starts); results are written from it.

To run one seed by hand:

```bash
python3 scripts/task_seed.py list tasks/<task>                                   # default, allowed, oracle, build
python3 scripts/task_seed.py materialize tasks/<task> --seed 1 --out /tmp/run1   # prints the copy and the seed
harbor run -p /tmp/run1 -i <task> -a <agent> ...
```

Every RoboARC puzzle task allows seeds 0, 1 and 2 from 1.2.0 (its appearance seeds: the same scene and physics in
other looks; `[metadata.seeds] oracle` lists the seeds whose Harbor oracle run passed). A task that allows its
default seed only (the IKEA tasks) runs from a copy that is the task as it is.

## Infrastructure failures: one full retry

PROTOCOL Appendix H.5: infrastructure failures allow at most one full retry with the same instance, seed, and
configuration, clearing all agent and environment state. If that retry also fails, follow the default failure-handling
behavior; no further retries. Normal task failures do not qualify.
- An infrastructure failure is what `scripts/harbor/write_task_results.py` publishes as "not a result": the verifier's
  `infra_error` / `grader_error`, no `reward.json`, a Harbor exception other than the agent's cutoff, a model-account
  failure. The agent's own end, its cutoff and its token budget are results and are never retried.
- The voided attempt's directory is moved aside and never counted; the retry starts from a new environment with the
  first attempt's recorded configuration. With the batch runner: move the job's directory and its claim
  (`$JOBS/.claims/<job>`) aside and start the batch's command again; the job runs again from the trial's seed. A launch
  that never started a trial is no attempt.
- The rule is implemented once, `scripts/harbor/retry_policy.py` (`decide`, `check_attempts`), tested by the required CI
  job `invariants`. It applies to each trial: `write_task_results.py` keeps every attempt of a task x run id x trial in
  that trial's `attempts.json` and flags a sequence that breaks the rule (`attempts_over_cap`). The K trials of a task
  are not attempts of one another.
- In-trial crash recovery (a sim service that replays its journal and resumes the episode, STANDARD_MODE.md) is the
  service layer's implementation, not a rule; a crash it cannot recover is an infrastructure failure like any other.

## Results

Results are not committed. `scripts/harbor/write_task_results.py <jobs_dir> <job> <run id> "<label>" --out <dir>`
writes one folder per trial, `<dir>/<task>/<run id>/t<k>/` (k = 1..K, the trial's number in the run):
- `README.md`: the result and its cost;
- `replay.mp4` and `replay.last.png`;
- `config.json`: the trial's effective configuration (`protocol_version`, `mode`, `cutoff_s`, `reset_budget`,
  `slowdown`, `seed`, `network`, `validated`, `set`, `oracle`, `service_protocol`, `digest`); its trial (`trial`,
  `trials`, the Harbor trial directory `trial_dir` and job `job`) and how its seed is known (`seed_source`: `runner`,
  from the job's `seed.json`; `default` for a job without one); the result and whether it counts (`result`, `counted`,
  `attempts`, `attempts_over_cap`); and the usage recording (`usage_gateway`, `usage_source`, `usage_recorded`,
  `token_budget`, see "Usage recording");
- `attempts.json`: every attempt of this trial (the infrastructure retry above).

Next to the trials, `<dir>/<task>/<run id>/trials.json` and `README.md` list the task's trials in the run: number,
seed, result, attempts, and the task's successes over its trials (results only). A job without `seed.json` (a plain
`harbor run`) numbers the trials of each task in it by their start, on the task's default seed. A results directory
written in the layout before trials (`config.json` directly in `<dir>/<task>/<run id>/`) is refused: write into a new
one. The run id is `<harness>-<model>-<effort>` ([TAXONOMY.md](TAXONOMY.md)); provider, service tier and settings go
in the metadata. The batch runner's `$JOBS/<batch>.summary.md` has one row per trial (with its trial number and seed)
and, per configuration, each task's successes over its trials.

Rankings compare trials only within one comparable configuration (`task_conventions.config_key`: protocol version,
mode, cutoff, reset budget, S, network; the seed is not part of it: a task's trials over its seeds are one task's
results), report success together with cost (tokens, money, turns, wall clock,
simulation steps; tokens and money where usage was recorded), and align reasoning effort, `max_turns` and the token
budget (`token_budget`; null when none was enforced) before comparing agents. Validated and
unvalidated tasks are ranked together (every puzzle task has `oracle = "full"`). The easy set
([`easy_set.toml`](easy_set.toml), `task_conventions.set_of`) is kept apart: its tasks are run and their results shown,
marked, but every headline number counts `core` tasks only. `set` in `config.json` is the value when the result was
written; a report derives it from `docs/easy_set.toml` at the commit it publishes.

## Re-grading

`harbor trial regrade` re-runs the verifier of a recorded trial (a grader error, a missing video). The verdict of a
privileged trial recorded before the grading rule of 2026-10-14 is recomputed by `scripts/roboarc/regrade.py`
(docs/benchmarks/roboarc.md, "Grading replays"):

```bash
python3 scripts/roboarc/regrade.py <jobs dir> ... --out regrade.jsonl            # from the recorded reward.json
python3 scripts/roboarc/regrade.py <jobs dir> ... --replay --out regrade.jsonl   # also replays the trajectory (docker, GPU)
```

A trial's seed comes from its job's `seed.json`; `--replay` grades it with that seed's verifier files.

## Resources: limits and their enforcement

Each task declares its minimum compute (`[metadata.resources]`), its limits (`[metadata.resources.limits]`, shown to the
agent in `## Resource limits`) and the docker caps of every compose service (`[metadata.resources.caps]`; PROTOCOL §6;
values from `scripts/task_resources.toml`, checked by `ci_checks/check-resources.py`).
- **Minimum** = the smallest allocation on which one trial, alone on the machine, runs correctly. Limits sit with a
  margin above measured peaks; the oracle passes within them.
- **Scope**: `trial` (privileged: every container the agent drives) or `agent` (standard: the agent's container; the
  `sim` sidecar is recorded, never a failure). Limits: GPU memory (NVML per process, summed), memory (cgroup
  `anon + shmem`), processes, disk growth under `disk_paths`; a violation is a limit exceeded for `sustain_sec` within
  any `window_sec`. CPU is only a cap; a CPU cap above the host's count means the host's count (the batch runner clamps
  it in its copy); memory and pids caps are never clamped.
- **Monitor**: the `rcb_monitor` sidecar samples every few seconds into its own record, sealed in the first collect
  hook, and shows current values and the agent's clock in `/run/rcb-limits/usage.json`; the verifier recomputes
  violations from the raw samples with its own copy of the limits. A missing or mostly empty record is
  `resource_monitor_missing = 1`. It also keeps an append-only copy of the agent's `/app/output/rcb_session_log.jsonl`.
- **The switch** `RCB_LIMITS_ENFORCE` (default 0): `1`: a confirmed violation kills the scope's processes and fails the
  trial (`resource_limit_exceeded = 1`), never retried; `0`: observe and record only (`limits_violation_observed`,
  `/logs/verifier/resource_limits.json`); any other value: `infra_error = 1`, `resource_switch_error = 1`.
- **Out-of-memory kills**: a kill at a container's own cap in the scope is a violation; a host OOM kill never is
  (enforcement on: `infra_error = 1`, `resource_host_oom`).

## Recordings

- **Every graded trial has a video** (PROTOCOL §4): privileged `verifier/replay.mp4` (rendered from the verdict replay
  after the verdict is written); standard the sim service's `replay.mp4` (collected into `artifacts/`). The one expected
  exception is a privileged trial whose agent submitted no trajectory that replays. A video a trial missed is rendered
  afterwards from the recorded trajectory in the task's own image; the grade is never touched,
  and the agent is never rerun for a video.
- **Standard mode**: the sim service journals every executed control step (`journal.f64`), the agent's calls
  (`episode.jsonl`) and the trajectory it grades (`trajectory.npz`); `final.json` is the live verdict and `replay.json`
  the collect hook's fresh-process replay.
- **Agent logs**: Harbor keeps `agent/` per trial (Codex: `codex.txt`, `trajectory.json`, `sessions/`). Run Codex with
  `--ak reasoning_summary=detailed` to keep reasoning summaries. `python3 scripts/harbor/agent_timeline.py <trial_dir>`
  merges the agent's messages, reasoning summaries, tool calls and token use with the robot service's log into
  `analysis/timeline.md`.

## Usage recording (optional)

Recording the agent's model-API usage is optional, and off by default. Without it, the token counts a result shows are
the harness's own (Harbor's `result.json`, `agent_result`), where the harness reports them.

With `USAGE_GATEWAY=<dir>`, the batch runner routes the agent's model-API calls through a transparent usage gateway
installed in `<dir>` (`gateway.py`, `rawlog.py`, `api_paths.json`). The repository ships one in `scripts/usage_gateway/`
(`USAGE_GATEWAY=scripts/usage_gateway`), taken unchanged from the shared benchmark repository (PROVENANCE.md); another
gateway that takes the runner's options works too. A `USAGE_GATEWAY` without `gateway.py` stops the runner at start,
before any job. The gateway forwards model calls only, unchanged (the agent keeps its own credentials), records each
trial's tokens per request
(`<trial>/gateway/usage.jsonl`, `totals.json`; with `GATEWAY_RAW=1`, the default, also the raw request and response
bodies, which hold the whole conversation: keep them private), enforces the token budget (`TOKEN_BUDGET_IN` /
`TOKEN_BUDGET_OUT`, 100M input tokens, cache included, and 10M output tokens, reasoning included, by default; past it
every further request is refused and the trial counts as failed), and becomes, with the package registries, the agent
container's only egress (each job's task copy is narrowed with `scripts/set_agent_egress.py only`; `GATEWAY_EGRESS=task`
keeps the task's own list). The runner checks the gateway before each job and restarts it if needed; a trial whose last
model calls failed at the gateway, or whose job has `gateway_down.json`, is an infrastructure failure, never the
agent's 0. The header of `scripts/harbor/run_agent_batch.sh` lists the other `GATEWAY_*` options.

Every job of the runner gets `<job>/usage_recording.json` (whether the gateway ran, its token budget), and every result's
`config.json` says what it has:

| field | value |
|---|---|
| `usage_gateway` | `true` / `false`: whether the trial ran through the usage gateway; `null` for a job without the runner's record (a plain `harbor run`) |
| `usage_source` | where the token counts come from: `"usage gateway"`, `"harness"` (Harbor's `result.json`), or `null` (none recorded) |
| `usage_recorded` | `usage_source` is not `null` |
| `token_budget` | `{"input": ..., "output": ...}`, the budget the gateway enforced (0 = no limit); `null` without the gateway |

Without the gateway no token budget is enforced: the task's cutoff bounds the trial.
