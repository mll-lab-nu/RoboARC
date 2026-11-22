# Running the full evaluation: from the machine to sending the results

This guide is for a team that runs the RoboARC evaluation of a coding agent on its own machines and sends the results
to the public results dataset. A public clone of the repository is all it needs. It goes through the machine, the
install, a first check, a smoke test of the batch runner, the evaluation run itself and the delivery of its results.
Every step is a subcommand of `roboarc`; the script each one starts is named beside it and stays callable by its path.
[RUNNING.md](RUNNING.md) explains the runner's rules in full, and its last section maps earlier names of commands,
modes and paths to today's.

## 1. Machine

- Linux x86_64 (not WSL, not arm64), Docker with the NVIDIA Container Toolkit, Harbor 0.23 or newer, Python 3.11 or
  newer as `python3` (section 2 installs both).
- **NVIDIA driver 580 or newer** (RoboARC's stated requirement). The renderer (Nyx) runs from about driver 570; the
  blanket families' cloth physics is built for CUDA 13 and needs 580 or newer, or 570 with NVIDIA's `cuda-compat`
  libraries (docs/API.md).
- **A GPU with RT cores.** The renderer (Nyx) needs the Vulkan extension `VK_KHR_ray_query`:
  - **Has RT cores:** T4, A10/A10G, L4, L40/L40S, A40, RTX 20/30/40/50 series, RTX A-series / Ada / PRO Blackwell.
  - **Does not:** V100, A100, A30, H100, H200. There the renderer stops with an error naming the missing extension
    (`nyx_direct.UnsupportedDevice`): standard mode does not run, and privileged tasks are graded without their replay
    video while the agent gets that error instead of camera images. Such runs are unofficial (docs/RUNNING.md, "GPUs
    without RT cores").
- GPU memory: each task declares 2 GB; the measured peak is about 1.1 GB. Every task runs on MuJoCo, so any card on the
  list above runs every family.
- Disk: about 19 GB for the images, plus room for each trial's task image, logs and videos.
- `roboarc doctor` (section 2) checks all of this. The GPU check by hand, which must print all three names
  (`vulkaninfo` is in the package `vulkan-tools`):
  ```bash
  vulkaninfo | grep -E 'VK_KHR_ray_query|VK_EXT_extended_dynamic_state3|VK_KHR_compute_shader_derivatives'
  ```

## 2. Install and images

<!-- unverified: the anonymous clone and the anonymous image pull work once the repository and its packages are public -->
1. Install uv, clone the branch `main`, and install the command `roboarc` and Harbor:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh        # uv (https://docs.astral.sh/uv/); then open a new shell
   git clone -b main https://github.com/mll-lab-nu/RoboARC.git && cd RoboARC
   uv venv --python 3.12 && source .venv/bin/activate     # Python 3.12 as `python3`, in .venv/ (git ignores it)
   uv pip install -e .                                    # the command `roboarc`
   uv tool install harbor                                 # Harbor; `harbor --version` prints 0.23.0 or newer
   ```
   The repository's scripts call `python3` and need Python 3.11 or newer (Ubuntu 22.04 ships 3.10): activate the
   environment (`source .venv/bin/activate`) in every shell that runs a command of this guide. With Kimi Code, use
   Harbor 0.25 or newer.
2. Pull the four images and check the machine:
   ```bash
   roboarc images pull --accept-licences      # about 19 GB on disk; each image is checked against its pinned digest
   roboarc doctor                             # ends with "doctor: every check passed"
   ```
   `--accept-licences` states that you accept the third-party licences listed in
   [LICENSES/THIRD_PARTY.md](../LICENSES/THIRD_PARTY.md); it is needed once per user. The pull needs no registry login
   ([images/README.md](../images/README.md)). Every task names its image by digest (`images/pins.json`):
   - `roboarc:2.2.0`, about 10 GB on disk, used by every puzzle task;
   - `roboarc-monitor:0.1.0`, about 0.13 GB on disk, used by every task;
   - `roboarc-ikea:1.1.0` and `roboarc-ikea-sim:1.1.0`, about 1.1 GB and 7.7 GB on disk, used by the IKEA tasks.

   The download is smaller, because the layers are compressed. `roboarc images pull` is `bash images/pull_all.sh`.
3. In an existing clone, `git fetch origin && git checkout main && git pull`, then `roboarc images pull` again: a pull
   that brings a new release (a new version in `CHANGELOG.md`) can bring a new image, and only the images that changed
   are downloaded. 2.2.0 moved the puzzle tasks from `roboarc:2.1.0` to `roboarc:2.2.0`; `docker images --digests
   ghcr.io/mll-lab-nu/roboarc` lists the local ones with their image ids, and `docker image rm <image id>` of the
   old one frees its 10 GB.

## 3. First check: one task, both modes, no agent

Start with `interlocked-claws-08`. In each mode the reference solution must score 1, and doing nothing must score 0:

<!-- unverified: the trial lists of this section were read with --print; the trials were not run in this lane -->
```bash
roboarc selftest interlocked-claws-08
```

It runs four Harbor trials, needs no model and no API key, and ends with

```
selftest: 4 of 4 trials have the expected score
```

- Each trial is a Harbor job under `jobs/selftest/`, named `selftest-<task>-<mode>-<agent>-s<seed>` (the agents are
  Harbor's `oracle`, which plays the reference solution, and `nop`). Its score is in
  `jobs/selftest/<job>/<trial>/verifier/reward.txt`: 1 or 0. Its output is in `jobs/selftest/<job>.log`.
- The privileged reference trial of this task takes about 6 minutes on an RTX 4090. The verifier replays the trajectory
  twice and renders a video, which lands in `verifier/replay.mp4`.
- A trial that has its score is kept: after an interruption the same command continues.

Then one scene other than the default: `interlocked-claws-09` at appearance seed 1 (its reference solution passed
Harbor's oracle run at seeds 0 and 1: `[metadata.seeds] oracle` in its `task.toml`). The reference must score 1 in both
modes:

```bash
roboarc selftest interlocked-claws-09 --seeds 1 --agents oracle     # ends with "2 of 2 trials have the expected score"
```

`roboarc selftest` makes each trial's copy of the task the way the batch runner does (`python3 scripts/task_seed.py
materialize <task dir> --seed <seed> --out <dir>`), then starts `harbor run -p <dir> -i <task>-<mode> -a <agent>`.

If a trial fails, see section 7.

## 4. Smoke test: the batch runner on one task

Before the full run (section 5: three trials, both modes, every task), run the batch runner once on one task with your
real agent, model and key, one trial per mode. It checks what section 3 cannot: the agent's credentials and network,
the runner's seeds, the summary and the result files.

<!-- unverified: this command and the one of section 5 were checked with --print; no batch was run in this lane -->
```bash
cp .env.example .env        # then put your credentials in it (the table in section 5)
roboarc eval smoke interlocked-claws-08 --agent codex --model openai/<model> --effort high --trials 1 \
    --env-file .env --usage-gateway --jobs jobs/smoke --results /path/outside/the/repo
```

`roboarc eval` starts the batch runner; this command is `AGENT=codex MODEL=openai/<model> EFFORT=high K=1
ENV_FILE=.env USAGE_GATEWAY=scripts/usage_gateway GATEWAY_RAW=0 JOBS=jobs/smoke RESULTS=/path/outside/the/repo
scripts/harbor/run_agent_batch.sh smoke roboarc-interlocked-claws-08-i01` (`--print` shows it).

- Use the agent, model, effort, credentials file and results directory the full run will use, and the usage gateway as
  above (section 5, "Usage gateway").
- **State the reasoning level** with `--effort` (the commands use `high`), never `--effort default`:
  - The runner passes it to the agent as Harbor's `reasoning_effort` (for Codex `model_reasoning_effort`, over the
    value in an `AGENT_CONFIG` file; for Antigravity the SDK's thinking level, `xhigh` being its `extra_high`, which
    the gateway sends on as `reasoning_effort`; for Kimi Code its thinking effort, which it sends as `thinking.effort`).
  - `default` passes no level. The model then runs at whatever level its harness or provider picks, and no
    record says which. Codex, for example, takes its default level from the provider's model list
    (`GET <base URL>/models`); an endpoint without that list (it answers 404) leaves the level to the provider.
  - The level is part of the run id (`<harness>-<model>-<effort>`), so every trial of a batch runs at one level.
- The two modes run one after the other, each with at most the agent's 3 h 10 min: the standard trial starts when the
  privileged trial's agent is done, while that trial is graded in the background. To run them at the same time, add
  `--lanes 2`.
- It passed when:
  - `roboarc results summary smoke --jobs jobs/smoke` (the file `jobs/smoke/smoke.summary.md`) has a row for
    `roboarc-interlocked-claws-08-i01-privileged` and one for `-standard`, each with a success of 0 or 1 and neither
    marked "not a result" (a 0 is fine here: the trial only has to give a result);
  - each job folder `jobs/smoke/smoke-<mode>-roboarc-interlocked-claws-08-i01/` has `seed.json` and
    `<trial>/verifier/reward.txt`;
  - `<results>/smoke/roboarc-interlocked-claws-08-i01-<mode>/<run id>/t1/` exists for both modes;
  - the agent ran at your level. The `runner` line of `jobs/smoke/smoke.log` names it (`effort high`, a run id ending
    in `-high`), and each job's `config.json` has it as `"reasoning_effort"`. With Codex, the trial's session log also
    records the level Codex ran with. This prints it for each mode, and it must print your level, never `None`:
    ```bash
    for f in jobs/smoke/smoke-*-roboarc-interlocked-claws-08-i01/*/agent/sessions/*/*/*/rollout-*.jsonl; do
      python3 -c 'import json,sys; print(next(json.loads(l)["payload"].get("effort") for l in open(sys.argv[1]) if "\"turn_context\"" in l))' "$f"
    done
    ```
    With Antigravity, the first line of each trial's agent log names the level (`thinking level high`) and the tools
    that are off (`generate_image`, `read_url_content`, `search_web`):
    ```bash
    head -qn1 jobs/smoke/smoke-*-roboarc-interlocked-claws-08-i01/*/agent/antigravity.txt
    ```
    With Kimi Code, the gateway records what each model call sent (`request_fields`): every call must print your level
    (`{'type': 'enabled', 'effort': 'max', ...}` with `--effort max`), and the CLI's config must turn `WebSearch` and
    `FetchURL` off:
    ```bash
    python3 -c 'import collections,json,sys; print(collections.Counter(str(json.loads(l).get("request_fields", {}).get("thinking")) for f in sys.argv[1:] for l in open(f) if "/chat/completions" in l))' \
      jobs/smoke/smoke-*-roboarc-interlocked-claws-08-i01/*/gateway/usage.jsonl
    grep -h disabled jobs/smoke/smoke-*-roboarc-interlocked-claws-08-i01/*/agent/.kimi-code/config.toml
    ```
  - `jobs/smoke/smoke.summary.md` has the column `usage gateway in / out`, with two numbers in both rows: the agent's
    model calls went through the usage gateway. With that column absent, or with `-` in a row, the agent's container
    did not reach the gateway. The gateway listens on the Docker bridge address (172.17.0.1), and a host firewall that
    blocks containers from reaching the host stops it: allow that address, or see section 7 with `jobs/smoke/.gateway/`;
  - the gateway was on as in the command. The `runner` line of `jobs/smoke/smoke.log` names it (`usage gateway
    <checkout>/scripts/usage_gateway`), and each job's `usage_recording.json` says
    `"usage_gateway": true, "raw_bodies": false` and `"image_cap": 50` (10 with Gemini 4 Argon, and `"image_bytes":
    4718592`; with Antigravity also `"api_translation": "gemini-chat"`); with Kimi Code each trial has
    `agent/trajectory.json` (Harbor 0.25);
  - your provider answered the model calls. This counts the HTTP statuses of all the smoke's model calls:
    ```bash
    python3 -c 'import collections,json,sys; print(collections.Counter(json.loads(l)["status"] for f in sys.argv[1:] for l in open(f)))' \
      jobs/smoke/smoke-*-roboarc-interlocked-claws-08-i01/*/gateway/usage.jsonl
    ```
    Almost all should be 200. A 429 is your provider's rate limit ("Usage gateway" in section 5). A few 404s for
    `GET .../models` mean your endpoint has no model list: harmless once the effort is stated (above);
  - `jobs/smoke/smoke.log` shows every lane running to its end (the runner stops a lane when the model account fails:
    its quota used up, rate-limited, logged out);
  - `jobs/smoke/smoke.log` has a line `<job>: agent phase ended; it validates ...`: your Harbor reports the end of the
    agent's phase, so a trial is graded beside the next one. If the runner's first line says `this harbor has no
    --plugin`, the run is still valid and each lane waits for every grading: Harbor 0.23 and newer have the option.
- The seeds the full run gives a task's three trials:
  `python3 scripts/task_seed.py assign tasks/roboarc-interlocked-claws-08-i01-privileged --batch <batch name> --trials 3`
  prints three different seeds (0, 1 and 2, in an order fixed by the batch name).
- If it fails, see section 7.
- Then send the smoke batch as section 6 describes, with `--jobs jobs/smoke` and the smoke's results directory. It
  checks your token and the upload before the full run.
- Give the full run a batch name of its own: the smoke trials are not part of it.

## 5. The evaluation run

Start it once the smoke test (section 4) has passed. The run puts a coding agent on every task, in both modes, three
trials per task and mode. Every puzzle task has three scenes (appearance seeds 0, 1 and 2: the same physics in another
look, with other lighting, table top, floor and object colours and materials), and the runner gives the three trials of
a task different seeds, so they see three different scenes; an IKEA task's three seeds are its one scene, lit three
ways in the standard mode's images.

Credentials go in an env file (copy `.env.example`), as Harbor's agents read them:

| account | what to set |
|---|---|
| OpenAI API key (Codex) | `OPENAI_API_KEY` |
| Anthropic API key (Claude Code) | `ANTHROPIC_API_KEY` |
| OpenRouter | Codex: `OPENAI_API_KEY=<OpenRouter key>` and `OPENAI_BASE_URL=https://openrouter.ai/api/v1`; Claude Code: `ANTHROPIC_API_KEY=<OpenRouter key>` and `ANTHROPIC_BASE_URL=https://openrouter.ai/api`. For a long Codex run, declare OpenRouter as a provider of its own in a Codex `config.toml` given as `AGENT_CONFIG=<file>` (`model_provider = "openrouter"`, `wire_api = "responses"`, `supports_websockets = false`) |
| ChatGPT login (Codex) | `CODEX_FORCE_AUTH_JSON=1` on the command line, in front of `roboarc eval` (there the runner's key scan covers the login too; in the env file alone it reaches Harbor only): Harbor copies `~/.codex/auth.json` into each trial |
| Claude subscription (Claude Code) | `CLAUDE_FORCE_OAUTH=1` and `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`) |
| Azure OpenAI (Codex) | `AGENT_CONFIG=<config.toml with the Azure provider>` and the key variable it names; retries in the same file (`request_max_retries`, `stream_max_retries`) |
| GitHub Copilot proxy (Codex, Claude Code) | Codex: `OPENAI_BASE_URL=http://127.0.0.1:<port>/v1`; Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:<port>` (the proxy's plain-HTTP address with its port, on the run's own host); and `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` set to what the proxy expects (at least 16 characters, any value if it needs none: the key scan of section 6 needs one). Official since 2026-11-02 |
| DeepSeek API key (DSH) | `DEEPSEEK_API_KEY`, with `--agent dsh --model deepseek/deepseek-v4.1-flash`: DSH, the DeepSeek Harness, through `scripts/harbor/agents/dsh_acp.py` |
| Chat Completions server (Antigravity) | `ANTIGRAVITY_BASE_URL=http://127.0.0.1:<port>/v1` (the server's base, plain HTTP with a port: a proxy on the run's own host; the gateway calls `<base>/chat/completions`) and `GEMINI_API_KEY` (the key the server expects, sent as a Bearer token; at least 16 characters, any value if the server needs none: the key scan of section 6 needs one), with `--agent antigravity --model google/<model>` and the usage gateway on: the Antigravity SDK through `scripts/harbor/agents/antigravity_sdk.py`, its Gemini requests translated by the gateway (docs/RUNNING.md "A single trial") |
| Moonshot API key (Kimi Code) | `KIMI_API_KEY`, and `KIMI_BASE_URL=https://api.moonshot.cn/v1` for Moonshot's China platform (unset: `https://api.moonshot.ai/v1`), with `--agent kimi-code --model moonshot/<model>` (e.g. `moonshot/kimi-k3.1-pro`) and the usage gateway on: Kimi Code CLI through `scripts/harbor/agents/kimi_code_cli.py` (docs/RUNNING.md "A single trial"). A model whose context window is not 1M tokens: also `KIMI_MODEL_MAX_CONTEXT_SIZE=<its window>` |

The agent's container reaches only these providers (with `--agent dsh` also DeepSeek's API; with `--agent antigravity`
the gateway alone, which reaches your Chat Completions server; with `--agent kimi-code` the gateway alone, which
reaches Moonshot's API), and with the usage gateway on (below) only through it, and the package registries (PyPI,
apt). Another host that serves one of these APIs is named in `GATEWAY_EXTRA_HOSTS`: an Azure AI Foundry endpoint
(`*.services.ai.azure.com`) needs `GATEWAY_EXTRA_HOSTS="*.services.ai.azure.com"` in front of the command.

```bash
mkdir -p jobs
nohup roboarc eval <batch name> --agent codex --model openai/<model> --effort high --seeds 0,1,2 \
    --lanes $((4 * $(nvidia-smi -L | wc -l))) --env-file .env --usage-gateway \
    --results /path/outside/the/repo > "jobs/<batch name>.out" 2>&1 &
```

With no task named it runs every task of the checkout: 90 (78 puzzles, 12 IKEA). The command is the batch runner
started once per lane, with `K=3 SEEDS="0 1 2" USAGE_GATEWAY=scripts/usage_gateway GATEWAY_RAW=0` and
`JOBS=jobs/agents` (`roboarc eval ... --print` shows the runner's command line; docs/RUNNING.md "Batches" lists each
option beside the runner's variable).

- **Which tasks and modes:** every task in both modes, three trials each: 90 × 2 × 3 = 540 trials. To run a part of
  the benchmark, name the tasks after the batch name (`roboarc eval <batch name> orbits-02 cast-coil-06 ...`).
  `--modes privileged` or `--modes standard` runs one mode only.
- **Seed by seed** (`--seeds 0,1,2`, as in the command): every task's seed-0 trials start first, then seed 1, then
  seed 2; once every seed-0 trial has started, seed-1 trials take the GPUs that free up (docs/RUNNING.md "Running the
  seeds in turn"). So the batch has one trial of every task, at seed 0, well before it ends. The order leaves the
  results as they are: each trial keeps its number and seed (without `--seeds` the runner goes task by task). The
  log's first lines say `SEEDS 0 1 2, in that order: 180 / 180 / 180 jobs` (fewer with a part of the tasks or one
  mode).
  - To run seed 0 alone first: `--seeds 0`. Later, the same command with `--seeds 1,2`, the same batch name and the
    same jobs directory: the two runs make one batch of three trials per task, and finished trials are skipped.
  - One seed at a time, a script per seed: `bash-runners/run_seed0.sh <batch name>`, then `run_seed1.sh`, then
    `run_seed2.sh`, with the same batch name. Each starts the run above with its seed (4 lanes per GPU, `nohup`, the
    usage gateway on), from the settings in its environment:
    ```bash
    MODEL=openai/<model> EFFORT=high RESULTS=/path/outside/the/repo bash-runners/run_seed0.sh <batch name>
    ```
    `AGENT` defaults to `codex`, `ENV_FILE` to `.env` and `JOBS` to `jobs/agents` in the checkout; the runner's other
    variables (`GPUS`, `MEM_GUARD`, `MODES`, `CODEX_FORCE_AUTH_JSON`, ...) go on the same line. Task bases after the
    batch name (`roboarc-orbits-02-i01`) run those only. Each lane's output goes to
    `jobs/<batch name>.seeds-<seed>.lane<k>.out`. Start the next script once every lane of the previous one has ended:
    each of its output files then ends with `runner <pid> of batch <batch name> finished` (seed 0:
    `tail -qn1 jobs/<batch name>.seeds-0.lane*.out`).
  - Keep three trials (the default of `roboarc eval`; `K=3` for the runner): the runner stops at start if a task's
    trials do not run a listed seed. Lanes of other seeds started on the same batch while earlier ones still run do
    run, each lane its own trials, and their trials start beside the earlier seed's: for the order, start the later
    seeds once the earlier lanes have ended.
- **Usage gateway** (`--usage-gateway`, as in the commands; keep it on). The runner routes the agent's model calls
  through a usage gateway on your host (docs/RUNNING.md "Usage recording").
  - It forwards the calls unchanged; the agent keeps its own credentials. The one exception: a call that carries more
    than 50 images (`GATEWAY_IMAGE_CAP`) has its oldest ones replaced by a short note, because some providers, GitHub
    Copilot among them, refuse more than 50. With Gemini 4 Argon (`--model google/gemini-4-argon`) the cap is 10, which
    the runner sets itself: Copilot refuses more than 10 images for that model. For that model the runner also keeps a
    call under 4.5 MiB (`GATEWAY_IMAGE_BYTES`) by replacing more of its oldest images: a call over 5 MiB is refused.
  - It records each trial's tokens per request in `<trial>/gateway/` (`usage.jsonl`, `totals.json`). The token counts
    in the results then come from the gateway, the same way for every agent and provider.
  - It enforces each trial's token budget: 100M input tokens (cache included) and 10M output tokens (reasoning
    included). Past it, every further request is refused and the trial counts as failed.
  - `--usage-gateway` sets `GATEWAY_RAW=0`: the gateway keeps token counts only, and the request and response bodies,
    which hold the whole conversation, stay unrecorded. The dataset the results go to is public (section 6).
  - The runner checks the gateway before each job and restarts it if needed. A trial is "not a result", never the
    agent's 0, if the gateway went down during its job or if its last model calls failed at the gateway.
  - **A 429 is your provider's rate limit**, passed on unchanged. The gateway never answers 429 itself. Its own
    refusals are 400 (past the token budget), 403 (not a model API call) and 502 (the provider cannot be reached).
    With `--agent antigravity`, a 500, 502 or 504 reaches the SDK as 503, which it retries (the records keep the
    provider's status). Codex retries a 429. When it gives up, the trial ends with `ApiRateLimitError`: it is not a
    result, and the runner stops the lane. Kimi Code retries a 429 too, except Moonshot's
    `exceeded_current_quota_error` (a 429 that means the account's quota is used up); a Kimi Code trial whose last
    model call was answered 429 is not a result, and the runner stops the lane. Every lane is one more trial calling
    the model at the same time, so your quota (requests and tokens per minute) must carry all the lanes you start.
    Raise the quota or start fewer lanes, and do that before a trial's one rerun.
- **Keep as they are** (a result counts only when its run had them): Harbor's own agent (`--agent codex` or
  `claude-code`), or `--agent dsh` (RoboARC's own agent for DSH; the maintainers decide whether its results count), or
  `--agent antigravity` (RoboARC's own agent for the Antigravity SDK; official since 2026-11-01), or `--agent
  kimi-code` (RoboARC's own agent for Kimi Code CLI; the maintainers decide whether its results count), with no adapter
  and no proxy that holds the credentials or changes the requests, except a GitHub Copilot proxy in front of the agent
  (official since 2026-11-02) (the usage gateway holds no credentials, and its one change to a request, the image cap,
  is the runner's for the model; with `--agent antigravity` it also translates the SDK's Gemini requests for your Chat
  Completions server, and every result records that as `api_translation`); web search off (leave `WEB_SEARCH` unset);
  the runner's image cap, 50, or 10 and 4.5 MiB for Gemini 4 Argon (leave `GATEWAY_IMAGE_CAP` and
  `GATEWAY_IMAGE_BYTES` unset); the task's own time limit (no `--timeout-multiplier` or `--agent-timeout-multiplier`)
  and its instruction as it is (no prompt template, nothing added through `AK`); the task's network; the runner's
  seeds; one reasoning level for the whole batch (section 4), never `default`.
- **Lanes:** one lane runs one trial at a time, and every trial is taken by exactly one lane; `--lanes N` starts N,
  and the same command started again on the same jobs directory adds its lanes. Before a trial starts, the runner
  gives it a GPU and checks the host's CPUs and memory (docs/RUNNING.md "GPUs and lanes"): up to 4 trials of rigid
  tasks share a GPU, a trial of a deformable task (rope, cloth, cable, fiber) has a GPU of its own, and the CPU cores
  and memory its task declares (most: 4 cores and 7 to 10 GB) must fit beside the trials already running. A trial that
  does not fit waits, and its lane takes the next one. So start 4 lanes per GPU, as above: lanes beyond what the host
  holds only wait. A trial may use up to 32 GB of memory (its limit). On a host shared with other work, name your GPUs
  (`--gpus 0,1`) and put `MEM_GUARD=1` in front of the command. When a trial's agent is done, its grading (the replay
  and its video) runs in the background on the same GPU while the lane starts its next trial, so 4 lanes per GPU stay
  right (docs/RUNNING.md "GPUs and lanes"; `PHASE_SPLIT=0` waits for each grading).
- **Duration:** a trial gives the agent up to 3 h 10 min, plus setup and grading. If every trial used its whole time,
  the 540 trials would take about 884 GPU-hours (the 300 rigid trials with a GPU count a quarter of a GPU each, the 36
  IKEA privileged trials none, the 204 deformable trials a whole one): about 37 days on one GPU, 9.2 on four, 4.6 on
  eight. `nohup` keeps the lanes running after you log out.
- **One release per batch:** finish a batch on the commit it started on. A new release changes the tasks: 2.2.0
  changed every task's `DIGEST` and the puzzles' seeds 1 and 2. Lanes started again after a `git pull` that brought
  one skip the finished trials and run the others on the new tasks, so the batch would mix two releases. Finish the
  batch first, or start a new one under a batch name of its own.
- **More than one host:** give each host its own part of the tasks, the same batch name (a trial's seed depends only
  on the batch name, the task and the trial number) and the same commit, and send each host's batch on its own
  (section 6).
- **Progress and summary:** progress is in `jobs/agents/<batch name>.log`, and the per-trial summary in
  `jobs/agents/<batch name>.summary.md` (`roboarc results summary <batch name>` prints it).
- **Restarting** (after a reboot, or a lane stopped by a model-account failure): start the lanes again with the same
  command. Finished trials are skipped. A trial the reboot interrupted is skipped too: to run it again, move its job
  folder and `jobs/agents/.claims/<job>` out of `jobs/agents/` first. A reboot can interrupt more than one trial per
  lane: the one whose agent was running and those still being graded. It gets the same seed, and this is its one
  rerun (next item).
- **"not a result"** in the summary marks an infrastructure failure: the grader's `infra_error` or `grader_error`, no
  `reward.json`, a Harbor exception other than the agent's own end, or a model-account failure. Such a trial may run
  once more, the same way: move its job folder and its claim out of `jobs/agents/`, start the command again; it gets
  the same seed. Once only: a rerun that fails again stays as it is. Name every rerun in the notes of your delivery
  (section 6). A trial the agent ended itself is a result (0 or 1): it stays as it is. That includes its time limit, a
  model response over the harness's output-token maximum (`OutputTokenExceededError`) or over the model's context
  window (`ContextWindowExceededError`), and the agent killed at its own container's memory cap
  (`NonZeroAgentExitCodeError` after an OOM kill at the cap; RUNNING.md "Infrastructure failures").
- **Time limit and key:** each trial gives the agent up to 3 h 10 min; `roboarc-wire-splicing-01-i01` gives 6 h (a
  recorded exception; its results are ranked as a group of their own). Use one API key per batch, with a usage limit
  set at the provider.
- **Live view:** `python3 scripts/harbor/live_board.py --jobs jobs/agents` serves a page on 127.0.0.1. Keep it local:
  it shows unredacted output.
- **When it has finished:** send it (section 6).

## 6. Sending the results

`roboarc results send` packs one batch and uploads it to the Hugging Face dataset `RoboARC/RoboARC-outputs`. With
`--create-pr`, as below, the upload is a pull request on the dataset, which the maintainers merge. Your own Hugging
Face account opens it and needs no access to the dataset.

**The dataset is public: anyone can read what you send.** The secret scan removes the keys of your credentials file.
Everything else in the job folders goes up as it is, including paths and host names in the logs.

1. Make a token of the type **Write** at https://huggingface.co/settings/tokens. It can do only what your account can
   do, and here that is opening the pull request.
2. Once per host:
   ```bash
   pip install -U huggingface_hub      # the hf CLI
   ( umask 077; cat > ~/.roboarc_hf_token )   # paste your token, press Enter, then Ctrl-D
   ```

The token stays in that file, outside `.env` and outside your shell history. The script reads it only for the upload,
so it never reaches a container or the bundle.

When the batch has finished, run this from the checkout, as the user that ran the batch, with the batch's own jobs
directory, results directory and credentials file:

<!-- unverified: the options were checked against `roboarc results send --help` and scripts/send_results.sh; nothing
     was sent in this lane -->
```bash
roboarc results send <batch name> --name <your name, lowercase> --results /path/outside/the/repo --env-file .env \
    --create-pr --notes "<one line: crashes, OOM kills, outages, reruns; or none>"
```

This is `scripts/send_results.sh <batch name> <your name>` with `RESULTS`, `ENV_FILE`, `CREATE_PR=1` and `NOTES` set;
`--jobs DIR` names a jobs directory other than `jobs/agents`.

- **Secret scan first.** The batch's job folders are redacted against the keys in the credentials file, then scanned
  again. If any key is left, nothing is sent. If the batch ran with `CODEX_FORCE_AUTH_JSON=1`, set it and
  `CODEX_AUTH_JSON_PATH` as for the batch, so the scan also looks for that login.
- **What goes up**, to `<your name>/<batch>/<bundle id>/` in the dataset:
  ```
  MANIFEST.md     the checkout's commit, the host, its GPUs, the image digests, the runner's settings, the trial
                  counts (no credentials)
  index.json      per job: its archive, size and sha256, task, mode, trial, seed, and each trial's reward
  SHA256SUMS      every archive's sha256
  INVENTORY.tsv   every file in the archives, with its sha256
  batch.tar.gz    <batch>.log, <batch>.summary.md, the plan, <results>/<batch>/
  jobs/<job>.tar.gz   one per job folder (<batch>-<mode>-<task>-t<k>), with its seed record and each trial's
                      gateway/ (token counts per request; no request bodies with GATEWAY_RAW=0)
  ```
  The bundle id is the UTC time, so a later delivery never overwrites an earlier one.
- **Dry run.** `--dry-run` does every step but the upload and needs no token. The bundle stays in
  `../send/<batch>__<name>__<bundle id>/` for you to look at.
- **Write access.** An account that has write access to the dataset leaves `--create-pr` out: the bundle then goes in
  directly, as one commit.
- **When it ends:** the script prints the batch, the bundle id and the sha256 of `SHA256SUMS`, and `hf` prints the pull
  request's link. Keep all four: they identify the delivery.
- **If the upload fails** (the token is refused, or the dataset is not found), nothing is lost: run the same command
  again once it is fixed. A run within the same minute as the last one stops: wait a minute.
- `roboarc results receive` lists the deliveries in the dataset, and `roboarc results receive <name>/<batch>/<bundle
  id>` downloads one, checks every archive against its checksums and unpacks it.

## 7. Questions and problems

Open an issue at https://github.com/mll-lab-nu/RoboARC/issues with the command you ran, its last lines, and the output
of `roboarc doctor --json`. For a failing trial add its `result.json`, its `exception.txt` if there is one and the
logs under `verifier/`; for the batch runner also `<jobs>/<batch>.log`. An issue is public: the runner redacts the keys
of your credentials file from a job folder after each job, and `roboarc results send <batch> --name <name> --dry-run`
scans a whole batch again without sending anything.
