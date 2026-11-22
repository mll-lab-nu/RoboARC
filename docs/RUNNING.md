# Running RoboARC with Harbor

How to run RoboARC's tasks for coding agents with [Harbor](https://github.com/laude-institute/harbor): one trial, a
check of a set-up, or a batch, and how a run is configured and recorded. The command `roboarc` starts each of them
(`roboarc --help`; [roboarc/README.md](../roboarc/README.md) names the script behind every subcommand, and each script
stays callable by its path). The task contract is [PROTOCOL.md](PROTOCOL.md) (its Appendix H binds every run), the
standard-mode wire protocol is [STANDARD_MODE.md](STANDARD_MODE.md), and the benchmark's own rules are in
[benchmarks/roboarc.md](benchmarks/roboarc.md). The sections from "Official runs" on are the runner's rules:
`roboarc eval` and the batch runner behind it, `scripts/harbor/run_agent_batch.sh`, apply them by default. A team that
runs the full evaluation follows [COLLABORATOR_SETUP.md](COLLABORATOR_SETUP.md), which goes from the machine to sending
the results. Earlier names of commands, modes and paths are listed in [Names, old and new](#names-old-and-new).

## Requirements

- Linux x86_64 with an NVIDIA GPU that has RT cores (Nyx renders through Vulkan; driver >= 580; see "GPUs without RT
  cores" below) and Docker with the NVIDIA Container Toolkit. arm64 hosts are not supported (the
  images are linux/amd64, Nyx needs NVIDIA on x86_64, the standard client's lock lists x86_64 wheels).
- Python 3.11 or newer as `python3`, uv, the command `roboarc` and Harbor 0.23 or newer: "Install" below.
- The four images the tasks name, each by its digest ([images/pins.json](../images/pins.json)). `roboarc images pull
  --accept-licences` pulls them (about 19 GB) and checks each against its pin; `roboarc images check` checks the local
  ones, and `roboarc images build` builds one from the checkout ([images/README.md](../images/README.md)).
- Every trial needs one GPU (IKEA privileged trials none). A task's minimum allocation is its `[metadata.resources]`
  (README "Compute resources"; 2 GB of GPU memory, measured peak about 1.1 GB). The batch runner places each trial on
  one GPU of the host ("GPUs and lanes" below).

### Install

<!-- unverified: the anonymous clone and the anonymous image pull work once the repository and its packages are public -->
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # uv (https://docs.astral.sh/uv/); then open a new shell
git clone https://github.com/mll-lab-nu/RoboARC.git && cd RoboARC
uv venv --python 3.12 && source .venv/bin/activate     # Python 3.12 as `python3`, in .venv/ (git ignores it)
uv pip install -e .                                    # the command `roboarc`
uv tool install harbor                                 # Harbor; `harbor --version` prints 0.23.0 or newer
roboarc doctor                                         # one line per requirement
roboarc images pull --accept-licences                  # the four images; the licences: LICENSES/THIRD_PARTY.md
```

The repository's scripts call `python3` and need Python 3.11 or newer. Ubuntu 22.04 ships Python 3.10, so run them in
the environment above: activate it (`source .venv/bin/activate`) in every shell that runs `roboarc`, the batch runner
or a script of the repository. In a Python 3.11 or newer environment of your own, `pip install -e .` installs the same
command. Kimi Code's trajectory files need Harbor 0.25.

`roboarc doctor` prints one line per check and, under a check that failed, what to do. On a machine that is ready:

```
RoboARC 2.2.1, this host
  ok    python3                   3.12.9
  ok    repository                /home/you/RoboARC: RoboARC 2.2.1, 90 tasks
  ok    GPU                       0: NVIDIA GeForce RTX 4090, 24564 MiB
  ok    NVIDIA driver             580.95.05
  ok    Vulkan ray query          vulkaninfo names VK_KHR_ray_query, VK_EXT_extended_dynamic_state3, VK_KHR_compute_shader_derivatives
  ok    Docker                    server 29.1.3
  ok    Docker Compose            2.40.3
  ok    NVIDIA Container Toolkit  NVIDIA Container Toolkit CLI version 1.20.1
  ok    Harbor                    0.23.0
  ok    pinned images             4 of 4 are local, by their pinned digests
doctor: every check passed
```

It exits 0 when no check failed. A check whose tool is missing is shown as `?` (`vulkaninfo` is in the package
`vulkan-tools`). `roboarc doctor --json > doctor.json` writes the same as one JSON object, to send with a problem
report.

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
equal-size file in its image, silently, with `--no-cache` too. **Always build a task image from a stamped tree:**
`scripts/stamp_mtimes.py` gives every file a mtime derived from its content (content, modes and DIGEST unchanged;
git ignores mtimes). `roboarc selftest`, `roboarc eval` (the batch runner) and `scripts/roboarc/regrade.py --replay`
build from stamped copies they make themselves. A plain `harbor run -p tasks` builds from the checkout, so stamp the
checkout first:

```bash
python3 scripts/stamp_mtimes.py tasks                              # stamp the checkout in place (again after every pull: --check tells)
c=$(python3 scripts/stamp_mtimes.py stage tasks/<task> "$(mktemp -d)")   # or a stamped copy of one task
python3 scripts/check_image_files.py <image> <context dir>         # after building an image yourself: it holds the context's files
```

The first line ends with `stamp_mtimes: <n> files, <n> hashed, <n> restamped, 0 unstamped, 0 errors`.

## Checking a set-up: `roboarc selftest`

<!-- unverified: the trial list was read with --print and the output lines from roboarc/cli/selftest.py; the trials
     were not run in this lane (Docker, a GPU and the images) -->
```bash
roboarc selftest interlocked-claws-08
```

runs four Harbor trials without a model or an API key: in each mode the reference solution (Harbor's agent `oracle`)
must score 1 and doing nothing (Harbor's agent `nop`) must score 0. It prints

```
selftest: 4 trials over 1 task(s), 1 at a time; jobs and logs in <checkout>/jobs/selftest
[1/4] ok    roboarc-interlocked-claws-08-i01-privileged  oracle  seed 0  score 1 (expected 1)  <seconds> s
[2/4] ok    roboarc-interlocked-claws-08-i01-privileged  nop  seed 0  score 0 (expected 0)  <seconds> s
[3/4] ok    roboarc-interlocked-claws-08-i01-standard  oracle  seed 0  score 1 (expected 1)  <seconds> s
[4/4] ok    roboarc-interlocked-claws-08-i01-standard  nop  seed 0  score 0 (expected 0)  <seconds> s
selftest: 4 of 4 trials have the expected score
```

and exits 0. The privileged reference trial of this task takes about 6 minutes on an RTX 4090. Each trial is one
Harbor job `selftest-<task>-<mode>-<agent>-s<seed>` under `jobs/selftest/`, made the way the batch runner makes its
jobs (a stamped copy of the task at its seed, `scripts/task_seed.py materialize`), with its output in
`jobs/selftest/<job>.log`. A trial that has its score is kept: the same command continues after an interruption.

| to check | command |
|---|---|
| one task per family (release 2.2.1: 20 tasks, 80 trials), two at a time on two GPUs | `roboarc selftest --families all --lanes 2 --gpus 0,1` |
| some families | `roboarc selftest --families orbits,ikea` |
| other seeds (a reference is validated at the seeds of `[metadata.seeds] oracle`) | `roboarc selftest interlocked-claws-09 --seeds 0,1 --agents oracle` |
| what would run, without running it | `roboarc selftest --families all --print` |

## A single trial

One trial is a plain Harbor command, from a stamped checkout (above):

```bash
python3 scripts/stamp_mtimes.py tasks
harbor run -p tasks -i roboarc-interlocked-claws-08-i01-privileged -a oracle -o "$PWD/jobs" --job-name oracle-claws-08   # reward 1
harbor run -p tasks -i roboarc-interlocked-claws-08-i01-standard -a nop -o "$PWD/jobs" --job-name nop-claws-08          # reward 0
harbor run -p tasks -i roboarc-orbits-02-i01-standard -a codex -m openai/<model> \
    --ak reasoning_effort=high --ak web_search=disabled -o "$PWD/jobs" --job-name codex-orbits-02
```

`-i` takes the task directory's name with its mode. The score is in `jobs/<job>/<trial>/verifier/reward.txt`: 1 or 0.

- Use Harbor's native agents (`-a codex`, `-a claude-code`) with their credentials in the environment or an
  `--env-file`: Codex `OPENAI_API_KEY` (OpenRouter: also `OPENAI_BASE_URL=https://openrouter.ai/api/v1`), a ChatGPT
  login with `CODEX_FORCE_AUTH_JSON=1` (Harbor uploads `~/.codex/auth.json`); Claude Code `ANTHROPIC_API_KEY`, or a
  subscription with `CLAUDE_FORCE_OAUTH=1` and `CLAUDE_CODE_OAUTH_TOKEN` (`claude setup-token`). Through a GitHub
  Copilot proxy on the run's host: `OPENAI_BASE_URL` (Codex) or `ANTHROPIC_BASE_URL` (Claude Code) is the proxy's
  plain-HTTP address with its port, and the key variable holds what the proxy expects.
- DSH, the DeepSeek Harness, runs through RoboARC's own Harbor agent `scripts/harbor/agents/dsh_acp.py` (Harbor's ACP
  agent with `@deepseek-ai/dsh` 0.2.1-rc.2 installed): `PYTHONPATH=scripts/harbor/agents harbor run ... -a dsh_acp:DshAcp
  -m deepseek/deepseek-v4.1-flash --ak reasoning_effort=high --ak web=false --allow-agent-host api.deepseek.com`, with
  `DEEPSEEK_API_KEY`. It is RoboARC's own agent: the maintainers decide whether its results are official.
- Antigravity runs through RoboARC's own Harbor agent `scripts/harbor/agents/antigravity_sdk.py`: the Antigravity SDK
  (PyPI `google-antigravity` 0.1.23, every dependency locked in `agents/antigravity/run_agent.py.lock`), its evaluation
  preset (`LocalAgentConfig.eval()`) with `search_web`, `read_url_content` and `generate_image` off. The SDK speaks the
  Gemini API only; a Gemini model served through an OpenAI-compatible Chat Completions server (GitHub Copilot's proxy)
  is reached through the usage gateway, which translates each request and answer (`--gemini-chat`,
  `scripts/usage_gateway/gemini_chat.py`): `AGENT=antigravity` in the batch runner below. The SDK's own
  OpenAI-compatible mode is not used: it drops every image a tool returns and never sends a reasoning effort. Runs
  this way are Antigravity + Gemini's official runs (since 2026-11-01). A 500, 502 or 504 of the server (and the
  gateway's own 502) reaches the SDK as 503, which its evaluation preset retries, as it does a 429 (since 2026-11-02:
  passed on as it came, a 500 ended a collaborator's run at 2 h 46 min). The gateway's records keep the server's
  status (`agent_status`: the one given).
- Kimi Code runs through RoboARC's own Harbor agent `scripts/harbor/agents/kimi_code_cli.py`: Kimi Code CLI (npm
  `@moonshot-ai/kimi-code` 2.5.2) run as Harbor's own kimi-code agent runs it (`kimi --prompt ... --output-format
  stream-json`), with `WebSearch` and `FetchURL` off in its `config.toml`, the model, key, base URL and thinking effort
  in its `KIMI_MODEL_*` variables (`EFFORT` low, medium, high, xhigh or max; Kimi K3.1 Pro takes low, high and max, max
  being its default), on Moonshot's API (`KIMI_API_KEY`; `KIMI_BASE_URL` `https://api.moonshot.ai/v1`, the default, or
  `https://api.moonshot.cn/v1`) through the usage gateway (`AGENT=kimi-code` in the batch runner below, e.g.
  `MODEL=moonshot/kimi-k3.1-pro EFFORT=max`). The gateway's `moonshot` provider forwards its chat completions and the
  image and video uploads it makes before it shows the model an image, and records the reasoning setting each request
  sent (`request_fields`: `thinking`). The CLI takes its prompt on its command line only, so its argv holds the
  instruction, as Codex's does. It is RoboARC's own agent: the maintainers decide whether its results are official.
  trajectory.json needs Harbor 0.25 (its converter of the CLI's session journal).
- Turn the harness's provider-side tools off ("No tool outside the container" below): Codex `--ak web_search=disabled`
  and `--ak config=<the output of python3 scripts/harbor/codex_tools_off.py [your config.toml]>`; Claude Code
  `--ak disallowed_tools=WebSearch,WebFetch`; dsh `--ak web=false` (a dsh patch file turns its web_search and web_fetch
  tools off); Antigravity's are off in `agents/antigravity/run_agent.py` (its `search_web` is a second model call that
  declares Google Search, which the provider runs; the gateway also refuses any Gemini request that declares one);
  Kimi Code's in its `config.toml` (`agents/kimi_code_cli.py`; Moonshot's own search is a declared tool of type
  `builtin_function`, which the gateway refuses).
- Use an absolute `-o`. jobs/ is ignored by git; results never go into the repository.
- The trial's record is `<jobs>/<job>/<trial>/`: `agent/` (the harness's log and trajectory), `verifier/`
  (`reward.txt` 1 or 0, `reward.json`, `replay.mp4`, the resource-limits record), `artifacts/` (privileged: the agent's
  `/app/output/`; standard: the sim service's `/data/` files: `final.json`, `episode.jsonl`, `trajectory.npz`,
  `replay.mp4`, the service log), `result.json`.

## Official runs

A result is official only when its run had all of these:
- **Harbor's native agents**, configured through environment variables; no agent adapter, no proxy that holds
  credentials or changes requests. Two exceptions: a GitHub Copilot proxy between Codex or Claude Code and Copilot,
  which holds the Copilot credentials (since 2026-11-02); and `AGENT=antigravity` (RoboARC's own agent for the
  Antigravity SDK, and the usage gateway's translation of its Gemini requests for a Chat Completions server, such as
  Copilot's; since 2026-11-01; above).
- **No tool outside the container**: the harness's web search, account connectors, apps, plugins, image generation and
  cloud agents off (below).
- **The task's own cutoff**, no multiplier (`--timeout-multiplier`, `--agent-timeout-multiplier`, an agent
  `override_timeout_sec`), and **nothing added to the instruction** (no prompt template, no extra instruction).
- **The runner's seed, the task's network, at most one infrastructure retry** (below), and the effective configuration
  recorded with the result.

Recording the agent's model-API usage is **optional** (PROTOCOL H.6): an official run does not need the usage gateway.
Each result records whether usage was recorded and by what ("Usage recording" below).

## Batches: `roboarc eval`

<!-- unverified: checked with --print, which shows the runner's command; no batch was run in this lane -->
```bash
roboarc eval <batch> orbits-02 cast-coil-06 --agent codex --model openai/<model> --effort high \
    --env-file .env --results /path/outside/the/repo
```

runs a coding agent over the tasks named (every task of the checkout when none is named), in both modes, three trials
each. `<batch>` is a name of your choice: it is the prefix of the job names and fixes the seeds of each task's trials.
`roboarc eval` is a wrapper of the batch runner, `scripts/harbor/run_agent_batch.sh`: each option sets the runner's
variable of the same meaning, and every other variable of your environment reaches the runner unchanged. With
`--print` it shows the runner command and starts nothing. The command above starts the runner as

```bash
AGENT=codex MODEL=openai/<model> EFFORT=high K=3 ENV_FILE=.env RESULTS=/path/outside/the/repo \
  scripts/harbor/run_agent_batch.sh <batch> roboarc-orbits-02-i01 roboarc-cast-coil-06-i01
```

which is also how to call the runner directly. `roboarc eval` takes a task by its id or any alias; the runner takes
task bases, the task directories' names without their mode.

| `roboarc eval` | the runner's variable | default of `roboarc eval` | default of the runner alone |
|---|---|---|---|
| `--agent A` | `AGENT` | required | `codex` |
| `--model M` | `MODEL` | required | required |
| `--effort E` | `EFFORT` | required | `xhigh` |
| `--modes privileged,standard` | `MODES="privileged standard"` | both modes | both modes |
| `--trials N` | `K` | 3 | 1 |
| `--seeds 0,1,2` | `SEEDS="0 1 2"` | every trial, task by task | every trial, task by task |
| `--lanes N` | the runner started N times | 1 | one lane per start |
| `--gpus 0,1` | `GPUS=0,1` | every GPU | every GPU |
| `--env-file FILE` | `ENV_FILE` | the credentials of your environment | the credentials of your environment |
| `--jobs DIR` | `JOBS` | `jobs/agents` in the checkout | `jobs/agents` in the checkout |
| `--results DIR` | `RESULTS` | `rcb_results` next to the checkout | `rcb_results` next to the checkout |
| `--usage-gateway` | `USAGE_GATEWAY=scripts/usage_gateway`, and `GATEWAY_RAW=0` unless you set `GATEWAY_RAW` | `USAGE_GATEWAY` of your environment, else off | off |

A variable you already export is used when its option is not given. The command stays in the foreground until its
lanes end: start a long batch under `nohup` or `tmux`. `roboarc results summary <batch>` prints the batch's summary
table.

The runner runs one Harbor-native agent over `<base>-privileged` and `<base>-standard` of every task base given (`MODES`
picks the modes), `K` trials of each, one Harbor job per task, mode and trial: `$JOBS/<batch>-<mode>-<task>/`, or
`$JOBS/<batch>-<mode>-<task>-t<k>/` (k = 1..K) when K > 1. Each trial runs from its own seed ("Seeds" below). Starting
the same command again adds a lane (`roboarc eval --lanes N` starts N at once): every job is claimed by exactly one
runner (`$JOBS/.claims/<job>`) and placed on a GPU ("GPUs and lanes" below); a job that already has its `result.json` is
skipped. Each job runs from its own stamped copy of the task, made from the trial's seed (`$JOBS/.tasks/<job>/`, removed
afterwards; a CPU cap above the host's count is clamped in that copy, `scripts/harbor/clamp_cpus.py`, and its GPU
devices are pinned, `scripts/harbor/pin_gpu.py`). The batch's plan (every job, its trial and seeds) is
`$JOBS/.plan/<batch>.json`. Each lane logs the machine when it starts, in the `host:`, `GPUs:` and `software:` lines of
MANIFEST.md (CPU model and count, memory, each GPU's index, name, driver and memory, the docker and harbor versions;
`scripts/harbor/host_info.py`), and writes it into every job folder as `host.json`, so a trial's record names its
hardware wherever it is copied. The header of the script lists every variable; the main ones:

| what | how |
|---|---|
| agent, model, effort | `AGENT=codex`, `claude-code`, `dsh` (DSH, the DeepSeek Harness, `scripts/harbor/agents/dsh_acp.py`: `MODEL=deepseek/deepseek-v4.1-flash`, `DEEPSEEK_API_KEY`) or `antigravity` (the Antigravity SDK, `scripts/harbor/agents/antigravity_sdk.py`: `MODEL=google/<model>`, `ANTIGRAVITY_BASE_URL=http://<host>:<port>/v1` (a Chat Completions server), `GEMINI_API_KEY`, the usage gateway on; `EFFORT` minimal, low, medium, high, xhigh or default) or `kimi-code` (Kimi Code CLI, `scripts/harbor/agents/kimi_code_cli.py`: `MODEL=moonshot/<model>`, `KIMI_API_KEY`, `KIMI_BASE_URL` Moonshot's API, the usage gateway on; `EFFORT` low, medium, high, xhigh, max or default), `MODEL` (required), `EFFORT` (`--ak reasoning_effort`; the runner alone defaults to `xhigh`, `roboarc eval` and `bash-runners/` require it; `default` passes none), more options in `AK` |
| credentials | as Harbor's agents read them (above); `ENV_FILE` = Harbor's `--env-file` (`.env.example` lists the variables). Azure OpenAI and retries: a Codex `config.toml` in `AGENT_CONFIG` (`--ak config=`), with `request_max_retries` / `stream_max_retries`. For long OpenRouter runs declare OpenRouter as a provider of its own in that file (`model_provider = "openrouter"`, `wire_api = "responses"`, `supports_websockets = false`) |
| trials | `K` per task and mode (the runner alone: 1; `roboarc eval`: 3); each its own job and seed |
| seeds in turn | `SEEDS="0 1 2"`: only the trials of those seeds, every task's trials of the lowest seed first ("Running the seeds in turn" below); unset (default) = every trial, task by task |
| web search | `WEB_SEARCH=disabled` (default) or `allowed` (not for official runs) |
| run id | `RUN_ID`, derived as `<harness>-<model>-<effort>` ([TAXONOMY.md](TAXONOMY.md)) |
| results | `RESULTS`: a directory outside the repository (default `rcb_results` next to the checkout; empty = skip) |
| usage recording | `USAGE_GATEWAY=<dir>`: route the agent's model-API calls through the usage gateway installed in `<dir>` (the repository's: `USAGE_GATEWAY=scripts/usage_gateway`; "Usage recording" below). Empty (default) = off |
| GPUs | `GPU_SCHED=1` (default): each trial placed on one GPU and admitted by CPUs and memory; `GPUS` (default every GPU), `GPU_SHARE` (4), `DEFORMABLE_FAMILIES`, `RESERVE_CPU_SCALE` / `RESERVE_MEM_SCALE` (1), `MEM_HEADROOM_GB`, `SLOTS_DIR` ("GPUs and lanes" below); `GPU_SCHED=0`: every container on GPU 0. `PHASE_SPLIT=1` (default): a trial validates in the background while its lane starts the next one; `GPU_RESIDENT` (2 × `GPU_SHARE`) |
| optional host tools | `LOG_MIRROR=1`: an append-only mirror of every trial's `agent/` record in a root-owned container (`rcb-log-mirror`, `scripts/harbor/log_mirror.py`), so neither the agent nor the host user can delete it. `MEM_GUARD=1`: `scripts/harbor/mem_guard.py` keeps 2 GiB of host memory available by killing the largest running RoboARC trial container (names `roboarc-*` only). Both are off by default |

### GPUs and lanes

Docker gives a container that reserves `count: 1` GPU the host's first GPU, so on a host with several GPUs every trial
would run on GPU 0. With `GPU_SCHED=1` (the default; `SEQ=1` only) each lane places a trial before it starts, through a
ledger that every lane and batch of the host shares (`scripts/harbor/gpu_slots.py`, in `SLOTS_DIR`, default
`/tmp/roboarc-slots-<uid>`):

- **GPU.** A GPU takes up to `GPU_SHARE` (4) trials of rigid tasks. A trial of a deformable task has a GPU of its own:
  its task.toml `[metadata] family` is in a class with `own_gpu` in `scripts/task_resources.toml`, the one list of the
  family classes that the tasks' declared resources come from too (rope / chain: fiber_bundling, knot_untying, orbits,
  rope_threading, slotted_post_ring, wire_splicing; cloth: blanket_flatten, blanket_fold); `DEFORMABLE_FAMILIES`
  replaces that list for a run. A RoboARC task whose family is in no class is not started (normalize refuses such a
  task as well). A task whose compose files reserve no GPU (IKEA privileged) takes none. Rigid trials go to the
  fullest GPU that has room, so that whole GPUs stay free; while a deformable trial waits, the empty GPUs are kept for
  it and one busy GPU takes no new rigid trial until it is empty. On one GPU this alternates: up to four rigid trials,
  then a deformable trial alone.
- **CPUs and memory.** A trial reserves the most its task declares for the agent's or the verifier's phase
  (`[metadata.resources]` `min_cpus` / `verifier_min_cpus` and `min_memory_gb` / `verifier_min_memory_gb`: 4 CPUs and
  7 GB for a rigid puzzle, 4 CPUs and 10 GB for blanket privileged, 3 and 3 for IKEA privileged, for example), times
  `RESERVE_CPU_SCALE` / `RESERVE_MEM_SCALE` (1). It starts when the CPUs reserved by the trials running plus its own
  are at most `HOST_CPUS`, and their memory plus its own at most the host's total (`HOST_MEM_GB`, default the
  machine's) less `MEM_HEADROOM_GB` (default the larger of 4 GB and 10 %). With nothing else running it always starts. These are the tasks' minimums, three to five
  times the measured peaks: on a host with little memory for its GPUs, memory rather than GPUs bounds the trials at a
  time (32 rigid trials on 8 GPUs reserve about 224 GB and 128 CPUs).
- A job that does not fit is left for a later pass and the lane tries the plan's next one; a lane that can start
  nothing waits 30 s and logs, at most every 10 minutes, what the host lacks for the first job that waits.
- The job's copy has every NVIDIA device of its compose files pinned to that GPU (`device_ids`,
  `scripts/harbor/pin_gpu.py`: main or sim, rcb_monitor and the verifier all see the same one GPU), and
  `<job>/placement.json` records the GPU and the reservation (each result's config.json has it as `gpu`). A job's slot
  is released when the job ends, a runner's when it exits, and a killed runner's at the next placement by any lane.
- **Agent phase and validation.** A trial has two phases: the agent's (it works against the task's limit) and its
  validation, which Harbor runs after the agent: the replay, the replay video and the grading (privileged: the verifier
  container replays twice; standard: the sim sidecar replays once, in its collect hook, then a short verifier). With
  `PHASE_SPLIT=1` (the default; it needs `GPU_SCHED=1`, `SEQ=1` and a Harbor with job plugins) the runner passes Harbor
  the job plugin `scripts/harbor/plugins/phase_marker.py`, which writes `$JOBS/.phase/<job>.json` when the agent
  phase ends. The lane then checks the usage gateway and the model account as it did after a job, and takes the next
  trial at once while the first validates in the background; it ends when all its trials have validated. A validating
  trial stays on its GPU (a standard trial's replay runs in its sim container, which cannot move): it frees its GPU
  unit for another trial's agent phase and keeps only its task's verifier CPUs and memory (`verifier_min_cpus` /
  `verifier_min_memory_gb`, times the scales: 2 CPUs and 5 GB for a puzzle privileged, 2 and 6 standard). It still
  counts toward its GPU's trials: a GPU holds at most `GPU_RESIDENT` (2 × `GPU_SHARE`, 8) trials in either phase, and
  a deformable trial's agent phase needs a GPU on which no other trial reasons or validates. The validation uses the
  GPU too: the Nyx renderer draws the replay video (MuJoCo's physics runs on the CPU), and cloth tasks step the cloth
  on CUDA. `<job>/placement.json` records the mode (`phase_split`, `gpu_resident`, `validate_cpus`,
  `validate_memory_gb`). `PHASE_SPLIT=0` keeps a lane on its trial until the trial's results are written, as runners
  before this option did; a Harbor without `--plugin` runs that way too, and the batch log says so.
- Start up to `GPU_SHARE` × GPUs lanes (32 on 8 GPUs); lanes beyond what fits only wait. `GPUS=0,1` limits the runner
  to some GPUs, for example on a host shared with other work, where `MEM_GUARD=1` helps too: the ledger counts only
  the trials placed through it. `GPU_SCHED=0` turns placement off (every container on GPU 0, no admission), as runners
  before this option did; `SEQ=0` does not place trials.

After each job (with `PHASE_SPLIT=1`, in the background once it has validated) the runner scans the job for keys and
redacts them (`scripts/harbor/scan_secrets.py`), writes the per-task results (`scripts/harbor/write_task_results.py`)
and rewrites `$JOBS/<batch>.summary.md` (one table per comparable configuration). Before the lane takes its next trial
(after the job, or with `PHASE_SPLIT=1` when its agent phase has ended) it stops when the trial shows a model-account
failure (`scripts/harbor/account_failure_check.py`: out of credits, rate-limited, logged out). Progress is in
`$JOBS/<batch>.log`. Use one key per batch, with a usage limit set at the provider.

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

A trial's agent gets the task's `[agent] timeout_sec` (11400 s for every RoboARC task but wire-splicing-01-i01, which
has 21600 s by a recorded exception, `scripts/task_conventions.py` `CUTOFF_EXCEPTIONS`) and nothing else. Harbor counts
it around the agent's run only (after the environment has been built and started and the harness installed, each under a
timeout of its own); a standard task's `sim` service starts with the environment, and what it has not finished starting
by then is waited out inside the agent's time. At the cutoff Harbor stops the agent (`AgentTimeoutError`), collects the
artifacts and runs the verifier as after any other end: the trial is graded as it stands, is a result, and is not run
again.

The rules on the number (`ci_checks/check-cutoffs.py`, `check-twins.py`): a whole number of seconds, at most 14400
(or the recorded exception for the task's base, `scripts/task_conventions.py` `CUTOFF_EXCEPTIONS`; `cutoff_s` is part of
the configuration results are ranked by, so an excepted task's trials form their own group); the
privileged and standard versions of a task have the same cutoff (H.7); the instruction's closing `## Time limit` section
(rendered by `scripts/task_conventions.py`, the last thing the agent reads), the README's `Agent time budget` row and,
in standard mode, the sim service's `RCB_WALL_CLOCK_S` state the same number, and nothing else in the instruction states
the limit. The standard service's `time_left_s` counts the same limit from the agent's first request to the service.
`/run/rcb-limits/usage.json` shows the agent its clock, counted from the trial environment's start, so the harness's
install comes out of it (48-401 s measured; agents stop when it reads 0): the 11400 s are 3 h with 10 min for that. The same section tells the agent it works alone: the harness
runs non-interactively, so a reply without a tool call (a question to the user included) ends the agent's run, and the
trial is graded as it stands, a result like a cutoff. Harbor hands `instruction.md` to the agent as it is, and the
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

Every RoboARC task allows seeds 0, 1 and 2: the puzzles since 1.2.0, the IKEA tasks since the appearance seeds v2
(2.2.0). Seed 0 is the released scene. `[metadata.seeds] oracle` lists the seeds whose Harbor oracle
run passed (IKEA: 0).
- A puzzle's seeds 1 and 2 are its appearance seeds: the same scene and physics in another look. The room's lighting
  changes (which light is the key light, the lights' height, a warm or a cool white, a dim or a bright scene), the table
  top takes another material (wood, metal or stone), the table's frame and the floor change, and so do the colours and
  materials of the task's objects. Parts whose colour the texts name keep it, as does the green goal marker
  ([simulator/seeds/colour_table.md](simulator/seeds/colour_table.md)).
- An IKEA task's three seeds share its one frozen scene: seeds 1 and 2 light the standard mode's camera images
  differently (`environment/sim/rig.json` `seeds`). The privileged mode shows the agent no images, so there every seed
  is the same task.

A task that allows its default seed only runs from a copy that is the task as it is.

### Running the seeds in turn

`SEEDS` (`roboarc eval --seeds 0,1,2`; unset by default) runs a batch seed by seed: with `SEEDS="0 1 2"` and `K=3` the
runner starts every task's seed-0 trials first, then those of seed 1, then those of seed 2.
- **Which trials.** Of the K trials only those whose seed is listed run, each with its trial number and seed (the
  assignment above), so the job names and results are those of the whole batch. `SEEDS="0"` now and `SEEDS="1 2"`
  later, with the same batch name and `JOBS`, run the batch's K=3 trials between them; a trial that already has its
  result is skipped, as always. The order the seeds are listed in does not matter: the lowest goes first.
- **K stays 3.** A task allowed a listed seed that none of its K trials runs (K too small, e.g. left at 1) stops the
  runner before anything starts. A task not allowed a listed seed runs no trial at it, with a line in the log.
- **An order of priority, not rounds.** A later seed's trial starts only in room that no waiting trial of an earlier
  seed fits. While seed-0 trials remain and fit, the GPUs go to them; once every seed-0 trial has started, seed-1
  trials take the room that frees up, beside the last seed-0 trials still running. To keep that order a lane reads its
  trials from the top after each trial it starts, and while a trial of an earlier seed waits for room it does not try a
  later seed's deformable trial (a waiting deformable trial has a GPU kept or drained for it, "GPUs and lanes" above).
  The log says `SEEDS 0 1 2, in that order: <n> / <n> / <n> jobs` at the start, and
  `lane <pid>: <job> (seed <s>) starts while <n> job(s) of seed <s0> wait for room` when a later seed takes free room.
- **One limit.** Tasks reserve different CPUs and memory: a later seed's trial that needs less than an earlier seed's
  waiting one may start in room too small for that one, as between the tasks of one seed.
- **Lanes.** Lanes of the same `SEEDS` keep the order between them. A lane of other seeds can run on the same batch at
  the same time (`SEEDS="1 2"` started while `SEEDS="0"` lanes still run), each lane running its own trials, but its
  trials then do not wait for the other lane's: for the order, start the next seeds once the earlier lanes have ended.
  The batch's plan (`$JOBS/.plan/<batch>.json`) lists every trial of the batch whatever `SEEDS` is, so
  `scripts/send_results.sh` indexes every trial of a batch run in parts; a lane takes its trials in the order of
  `$JOBS/.plan/<batch>.seeds-0-1-2.tsv` (for `SEEDS="0 1 2"`).
- `SEQ=1` only (the default).

Without `SEEDS` the runner plans and schedules as it did before the option.

## Infrastructure failures: one full retry

PROTOCOL Appendix H.5: infrastructure failures allow at most one full retry with the same instance, seed, and
configuration, clearing all agent and environment state. If that retry also fails, follow the default failure-handling
behavior; no further retries. Normal task failures do not qualify.
- An infrastructure failure is what `scripts/harbor/write_task_results.py` publishes as "not a result": the verifier's
  `infra_error` / `grader_error`, no `reward.json`, a Harbor exception other than the agent's own end, a model-account
  failure. The agent's own end is a result and is never retried: its deliverable, its cutoff (`AgentTimeoutError`), its
  token budget, a model response over the harness's output-token maximum (`OutputTokenExceededError`, e.g. Claude Code's
  "response exceeded the 32000 output token maximum" after its own recovery turns) or a request over the model's
  context window (`ContextWindowExceededError`), and the kernel killing the agent at its own container's memory cap in
  the limits' scope (`NonZeroAgentExitCodeError` with `resource_oom_kills` and no host OOM or emergency kill; the task's
  instruction gives the agent its memory limit). The output-token maximum is the harness's, part of the run's
  configuration (Claude Code: 32000 for a model it does not know, 128000 for Opus 5.5; `CLAUDE_CODE_MAX_OUTPUT_TOKENS`
  changes it, and with it the configuration).
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
- `README.md`: the result, its timings and its token counts;
- `replay.mp4` and `replay.last.png`;
- `config.json`: the trial's effective configuration (`protocol_version`, `mode`, `cutoff_s`, `reset_budget`,
  `slowdown`, `seed`, `network`, `validated`, `oracle`, `service_protocol`, `digest`); its trial (`trial`,
  `trials`, the Harbor trial directory `trial_dir` and job `job`) and how its seed is known (`seed_source`: `runner`,
  from the job's `seed.json`; `default` for a job without one); the result and whether it counts (`result`, `counted`,
  `attempts`, `attempts_over_cap`); the usage recording (`usage_gateway`, `usage_source`, `usage_recorded`,
  `token_budget`, see "Usage recording"); and the machine it ran on (`host`, from the job's `host.json`: `cpu_model`,
  `cpus`, `memory_gb`, `gpus` with each GPU's `index`, `name`, `driver` and `memory_mib`, `docker`, `harbor`; `gpu`,
  the GPU the runner placed the job on, with its `gpu_name` and `gpu_driver`);
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
results), report success together with what the trial used (turns, wall clock, simulation steps, and tokens where
usage was recorded), and align reasoning effort, `max_turns` and the token
budget (`token_budget`; null when none was enforced) before comparing agents. Validated and
unvalidated tasks are ranked together (every puzzle task has `oracle = "full"`). Every task counts in every number. A
`config.json` written before the easy set was retired (2026-11) may also carry `set` (`core` or `easy`): readers
ignore it.

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
  (enforcement on: `infra_error = 1`, `resource_host_oom`). With enforcement off, a kill at the own cap that stops the
  agent (Harbor: `NonZeroAgentExitCodeError`) is the agent's own end, a result as graded, never retried ("Infrastructure
  failures"); one that only stops a process the agent started changes nothing.

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
(`USAGE_GATEWAY=scripts/usage_gateway`, which `roboarc eval --usage-gateway` sets), taken from the shared benchmark
repository (PROVENANCE.md), with DeepSeek's Messages API added for dsh (`AGENT=dsh` points dsh's `DEEPSEEK_BASE_URL` at
the gateway) and Moonshot's API for Kimi Code (`AGENT=kimi-code`: its chat completions, its image and video uploads, and
the request's `thinking` recorded as `request_fields`; the gateway's module doc); another gateway that takes the
runner's options works too. A `USAGE_GATEWAY` without `gateway.py` stops the runner at start, before any job. The
gateway forwards model calls only, unchanged but for the image cap below (the agent keeps its own credentials), records
each trial's tokens per request (`<trial>/gateway/usage.jsonl`, `totals.json`; with `GATEWAY_RAW=1`, the runner's
default, also the raw request and response bodies, which hold the whole conversation: keep them private), enforces the
token budget (`TOKEN_BUDGET_IN` / `TOKEN_BUDGET_OUT`, 100M input tokens, cache included, and 10M output tokens,
reasoning included, by default; past it every further request is refused and the trial counts as failed), and becomes,
with the package registries, the agent container's only egress (each job's task copy is narrowed with
`scripts/set_agent_egress.py only`; `GATEWAY_EGRESS=task` keeps the task's own list). The runner checks the gateway
before each job and restarts it if needed; a trial whose last model calls failed at the gateway, or whose job has
`gateway_down.json`, is an infrastructure failure, never the agent's 0. The header of
`scripts/harbor/run_agent_batch.sh` lists the other `GATEWAY_*` options.

With the gateway, every `harbor run` of the runner gets `-e roboarc_docker:RoboARCDocker`
(`scripts/harbor/environments/roboarc_docker.py`): Harbor's Docker environment whose egress sidecar waits up to 900 s,
the gateway's own upstream timeout, for a model call's response headers. Harbor's sidecar (gost, 0.23.0 to 0.25.x)
leaves gost's default of 15 s, so a provider that took longer to start its answer (a slow or rate-limiting endpoint) had
the call cut: the agent saw a connection reset and retried, and the provider still completed the call. The
class builds the sidecar from a patched copy of the installed Harbor's sidecar folder (in `~/.cache/roboarc/`); Harbor's
install is not edited, and the allowlist is unchanged. A `harbor run` of your own through the gateway should pass the
same option, with `scripts/harbor/environments` on `PYTHONPATH`.

The gateway also caps the images in a model request: `GATEWAY_IMAGE_CAP`, 50 by default (since 2026-10-31) and 10 for
Gemini 4 Argon (`MODEL=<provider>/gemini-4-argon`, since 2026-11-02: GitHub Copilot answers it 400, "too many images:
maximum allowed for model gemini-4-argon is 10", which voided a standard-mode trial of a collaborator's first
Antigravity run), 0 = no cap. An agent that looks at the cameras resends every image it has seen with each call, and
some providers refuse a request with more than a fixed number: GitHub Copilot answers 400, "Exceeded maximum number of
images (50) allowed in the request.", and Codex then exits, which voided a standard-mode trial of a collaborator's smoke
test (2026-10-31, 51 images after 52 minutes). A request with more images than the cap has its oldest ones replaced by a
short text note ("[earlier image removed: this conversation keeps at most 50 images, the oldest are dropped]"), half the
cap at a time, so that between half the cap and the cap stay; a request within the cap is forwarded as sent. Replacing
in steps keeps the replaced images the same from one call to the next, so the provider's prompt cache is broken once per
step, not once per image. The image files the agent saved are not touched: it can look at one again. `usage.jsonl`
records `images` and `images_omitted` per request and `totals.json` counts `image_cap_requests`. Without the gateway
there is no cap.

For Gemini 4 Argon the gateway also caps the size of a request: `GATEWAY_IMAGE_BYTES`, 4718592 bytes (4.5 MiB) for
`MODEL=<provider>/gemini-4-argon` (since 2026-11-04) and 0 = off for every other model. Through a GitHub Copilot proxy a
Gemini 4 Argon request over 5 MiB is answered 413 "Request Entity Too Large" and the SDK ends the run, which voided six
trials of a collaborator's Gemini 4 Argon run (2026-11-04, 5.0 to 5.4 MiB with 8 to 10 images; the largest request
answered was 4.98 MiB). A request still over the size after the image cap has more of its oldest images replaced, in the
same steps, until it fits; the newest image always stays, and a request still over it with no older image to replace is
forwarded as it is and recorded with `over_image_bytes`. Off, nothing changes: the gateway is not given the option and
the records have no `image_bytes`.

Every job of the runner gets `<job>/usage_recording.json` (whether the gateway ran, its token budget and image cap), and
every result's `config.json` says what it has:

| field | value |
|---|---|
| `usage_gateway` | `true` / `false`: whether the trial ran through the usage gateway; `null` for a job without the runner's record (a plain `harbor run`) |
| `usage_source` | where the token counts come from: `"usage gateway"`, `"harness"` (Harbor's `result.json`), or `null` (none recorded) |
| `usage_recorded` | `usage_source` is not `null` |
| `token_budget` | `{"input": ..., "output": ...}`, the budget the gateway enforced (0 = no limit); `null` without the gateway |
| `image_cap` | the most images the gateway let a model request carry (0 = no cap); `null` without the gateway, or for a job recorded before the cap existed |
| `image_bytes` | only where the gateway also capped the size of a request (`GATEWAY_IMAGE_BYTES`, Gemini 4 Argon): that size in bytes |

Without the gateway no token budget is enforced: the task's cutoff bounds the trial.

## Names, old and new

### Commands

Every earlier command keeps working: the subcommands of `roboarc` start the same scripts.

| earlier | now | the earlier form |
|---|---|---|
| `python -m roboarc list`, `info`, `replay` | `roboarc list`, `info`, `replay` | the same command; inside an image, where `roboarc` is not on `PATH`, `python -m roboarc` |
| `python -m roboarc check` | `roboarc doctor` | `roboarc check` prints the graded-runtime report that `doctor` shows inside an image |
| `bash images/pull_all.sh` | `roboarc images pull`, `roboarc images check` | `bash images/pull_all.sh [--check]` |
| loops of `harbor run ... -a oracle` and `-a nop` | `roboarc selftest` | a single trial stays `harbor run` |
| `scripts/harbor/run_agent_batch.sh` with variables | `roboarc eval` with options | the script, with the same variables |
| `bash-runners/run_seed0.sh`, `run_seed1.sh`, `run_seed2.sh` | `roboarc eval ... --seeds 0`, then `--seeds 1`, then `--seeds 2` | the scripts |
| `scripts/send_results.sh <batch> <name>` | `roboarc results send <batch> --name <name>` | the script |
| `scripts/receive_results.sh` | `roboarc results receive` | the script |
| `python -m roboarc.vla.run`, `python -m roboarc.vla.stub_server` | `roboarc vla run`, `roboarc vla stub` | the modules |
| `scripts/vla_droid.sh` | `roboarc vla droid` | the script |

### A task's name

| where | the task is named |
|---|---|
| `roboarc` (every subcommand), `roboarc.make()` | its id or any alias: `roboarc-interlocked-claws-08-i01`, `interlocked-claws-08-i01`, `interlocked-claws-08` (IKEA: `roboarc-ikea-lack-i00`, `ikea-lack`, `lack`) |
| `harbor run -i`, the job and result folders | the task directory's name, with its mode: `roboarc-interlocked-claws-08-i01-privileged` |
| `task.toml` `[task].name` | the same name after the namespace `roboarc/` |
| `scripts/harbor/run_agent_batch.sh`, `bash-runners/` | the base, which is the directory's name without its mode: `roboarc-interlocked-claws-08-i01` |
| `scripts/roboarc/harbor_check.py`, `scripts/roboarc/release.json` | the base without `roboarc-`: `interlocked-claws-08-i01` |
| `scripts/task_seed.py`, `scripts/stamp_mtimes.py` | the directory's path: `tasks/roboarc-interlocked-claws-08-i01-privileged` |
| the family (`task.toml`, `docs/families/`, the simulator) | with underscores: `interlocked_claws`; `roboarc list --family` takes `interlocked-claws` too |

### Modes and seeds

| what | `roboarc` | the batch runner | elsewhere |
|---|---|---|---|
| the modes | `--modes privileged,standard` | `MODES="privileged standard"` | the paper calls a mode a regime |
| the seeds | `--seeds 0,1,2` | `SEEDS="0 1 2"`, with `K=3` | `scripts/task_seed.py materialize --seed S`, `roboarc vla run --seed 0,1,2`, `roboarc.make(..., instance_seed=S)` |

Privileged mode was called unlimited and standard mode limited until protocol v1.0, and `eai-standard/2.x` was
`rcb-limited/2.x`. The earlier words remain in file names of the pinned task layer (`rcb_limited.py`).

### Names kept from earlier stages of the project

These names are part of the bytes every task is bound to (file hashes, image paths, variables the graders read), so
they stay as they are.

| name | what it is | where it remains |
|---|---|---|
| `topogym` (TopoGym) | the simulator package | `source/topogym/`, `/opt/topogym` in the image, `TOPOGYM_*` variables, `rcb_topogym.py` |
| `robotangle` (RoboTangle) | the benchmark's name before RoboARC | `/opt/robotangle/` in the image (the frozen instances, `stage_instance.py`) |
| `rcb` (Robot Coding Bench) | the task layer RoboARC shares with other benchmarks | `rcb_topogym.py`, `rcb_ikea.py`, `rcb_client.py`, `rcb1.py`, `scripts/rcb_monitor/`, `RCB_*` variables, the default results directory `rcb_results` next to the checkout |
| `eai` | the standard mode's protocol, `eai-standard/2.4` | `scripts/eai_standard/` |
| `ghcr.io/embodied-first-exam/` | the registry name the IKEA and monitor images were built under | their image labels and the IKEA sim recipe; the tasks and `images/pins.json` name `ghcr.io/mll-lab-nu/` |

Paths of the earlier repositories that pinned files still cite (`docs/rcb/RENDER_BRIDGE.md`,
`scripts/robotangle/shared/`, ...) are mapped to their present places in [rcb/README.md](rcb/README.md). The easy set
and the core set are retired: every task counts in every number, and a reader ignores the field `set` in a result's
`config.json` written before 2026-11.
