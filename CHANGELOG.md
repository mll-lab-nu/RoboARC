# Changelog

RoboARC's releases. Versions before 1.0.0 were released inside the shared benchmark repository
`mll-lab-nu/embodied-first-exam` (formerly robot_coding_bench), under the image name `ghcr.io/mll-lab-nu/eai-roboarc`.
1.0.0 is the first release from this repository. Its content is the "V3" wave of that repository.

## Unreleased

### The README's and the guides' commands as a new machine runs them (docs and tests; no task, image or result change)
- **Why:** the release was installed and run on a new machine from the README and the guides alone (2026-11-22,
  Ubuntu 22.04, Harbor 0.27.0), and some commands failed as written.
- **Install:** the README installs uv before it uses it and installs the package with `uv pip install -e .` (a
  `uv venv` holds no pip, so `pip` was the system's), and the tests' packages with `uv pip install`. Harbor: 0.23 or
  newer, tested with 0.23.0 and 0.27.0.
- **A single trial:** the `harbor run` lines pass `-y`. Harbor asks whether the task may read `RCB_LIMITS_ENFORCE`
  from the environment, and without a terminal it aborts and runs nothing.
- **In an image:** docs/API.md installs pytest into a mounted directory for the tests (the images ship without it)
  and selects `-m "sim and not heavy"`, and installs imageio the same way for `examples/replay_reference.py --video`.
  docs/VLA.md exports the reference into the mounted results directory (the repository is mounted read-only).
- **Smaller:** the sample `roboarc doctor` output names `/work/RoboARC` (`check-protocol.py` flagged the earlier
  sample path), RUNNING.md names the task's README for "Compute resources", and `.gitignore` lists the directories
  these commands write (`results/`, `vla_results/`, `.pytest-lib/`, `.video-lib/`).
- **Tests:** `tests/api` passes inside both images: the doctor's host tests answer as a host there too, the DROID
  tests, which make a one-arm puzzle, run in the puzzle image only, and GR00T's `eef_9d` tests are skipped in an image
  without scipy. On a host nothing changes: a missing scipy still fails them.

### `AGENT=dsh` starts with Harbor 0.25 and newer (agent fix; nothing changes for another agent or with Harbor 0.23)
- **Why:** from 0.25 on, Harbor takes the option `reasoning_effort` only from an agent whose class declares that
  capability. `DshAcp` took the option without declaring it, so Harbor refused to make the agent (`ValueError: Agent
  'dsh' does not support reasoning_effort.`) and `roboarc eval --agent dsh --effort high` started no trial. Found on
  2026-11-22 with Harbor 0.27.0, which `uv tool install harbor` installed on a new machine. Harbor 0.23 and 0.24 have
  no such field and were not affected.
- **Fix:** `scripts/harbor/agents/dsh_acp.py` declares `reasoning_effort` where Harbor has the field, as
  `antigravity_sdk.py` and `kimi_code_cli.py` do, and keeps every other capability of Harbor's ACP agent. What the
  agent does with the effort is unchanged.
- **Tests:** `test_dsh_acp.py` checks that Harbor takes the option and that the capability is declared: Harbor 0.23.0,
  0.25.0 and 0.27.0. No DSH trial was run with Harbor 0.25 or newer.

### The easy set is retired (results; no task change)
- **Why:** the benchmark reports all 90 task bases, so no number leaves a task out.
- **What:** `docs/easy_set.toml` is removed. The batch summary (`$JOBS/<batch>.summary.md`) counts every task's results
  in its success rates and no longer marks a task "(easy set)"; a trial's result page no longer names the set.
  `ci_checks/run_all.sh` no longer runs `task_conventions.py easy-set --check`. `scripts/task_conventions.py` loses
  `easy_set`, `set_of`, `check_easy_set` and its `easy-set` command, and `effective_config` no longer writes `set`
  into a result's `config.json`.
- **Earlier results stay readable:** `set` (`core` / `easy`) was a marker, never part of `config_key`: the groups of
  comparable results, run ids, seeds and the layout of jobs and results are unchanged. A `config.json` that carries
  `set` is read as any other, and a batch continued over such results leaves their files as written
  (`scripts/harbor/tests/test_runner_seeds.py`, `test_earlier_results_with_set`). A summary written again now counts
  the 15 bases the list held at the end: blanket-flatten-06, blanket-fold-02, cast-coil-05, cast-keyhole-02 and 06,
  fiber-bundling-02, 05 and 08, hedgehog-in-cage-03 and 08, horseshoe-ring-04, interlocked-claws-08 and 10,
  knot-untying-04, lock-picking-03.
- **Docs:** PROTOCOL §4 (the row), RUNNING.md "Results", docs/benchmarks/roboarc.md (the section "Easy set"),
  DEVELOPING.md.
- **Tasks:** none changed: the list was never a task file.

### The tools for authoring tasks and references leave the repository (no task or image change)
- **What:** `tools/references/` (how the references were made or re-recorded), `tools/redesign/` (the generators of the
  redesigned tiers), `tools/stress/` (the physics acceptance rounds and their corpus), `source/topogym/planning/` (the
  planners that recorded the references) and `ikea/scripts/ikea_mujoco/stress/` (the IKEA stress harness) are removed:
  814 files, 84.9 MB in a checkout. Tools for authoring new tasks and reference solutions are not part of the public
  release.
- **Unchanged:** nothing that runs, builds or checks the benchmark read these paths (no task file, Dockerfile, CI job
  or kept script). Every task directory equals a fresh render (`scripts/roboarc/mk_tasks.py --check`,
  `ikea/ci/check.sh`), and the simulator source hash is the released one (`tools/release/source_hash.py --check`:
  1552f29d, 455 files): the image never held `planning/`. Their outputs stay: the references in
  `scripts/roboarc/oracles/`, the variation resources under `source/topogym/assets/resources/variations/`, the
  acceptance criteria and results in docs/simulator/STRESS_VALIDATION.md.
- **Left as written:** the image inputs that name `topogym/planning` (`source/.dockerignore`,
  `source/images/topogym/prune.py`; the hashed `cast_coil.py`, `cast_coil_official.py`, `frozen_binding.py` and
  `tasks/rope_threading/redesign/task.py`), the release records (`tools/release/mujoco_release.json`,
  `records/freeze_manifests/`), and the development notes under docs/simulator/, which cite the tools as they were.
- **Docs:** `tools/README.md`, `source/README.md`, `source/topogym/README.md`, `ikea/README.md`, docs/DEVELOPING.md and
  docs/simulator/README.md no longer list them; docs/rcb/README.md maps the five paths ("Not in the public release"),
  so `ci_checks/check-doc-paths.py` accepts them where the development notes cite them.

### fiber-bundling-05-i01 gets a reference solution (task change: its two task directories)
- **Why:** hard fiber bundling was the one puzzle base without a reference (release.json `oracle: "none"`): nothing
  was known to solve it, Harbor checked only the nop agent, and the VLA runner gave it the 12000-step fallback.
- **The reference:** re-planned on the frozen instance in `roboarc:2.2.0` (birth f308c4c1a19865c4) with
  `tools/references/fiber_bundling/replan.py hard` (the released Genesis pickup rows 0-556, the comb loop with 9 combs
  for the 16 strands, the gated closed-loop feed of very_easy and easier), recorded with
  `tools/references/interlocked_claws/rerecord.py --tail 40`: 9915 rows (247.9 s simulated), success at the last row,
  end state 17b6af0171a92459. Every arm joint target moves at most 0.0093 rad per control step (the standard rule:
  0.04 rad) and stays inside the joint limits, so `standard.npz` is the same file; `privileged.s1.npz` and
  `privileged.s2.npz` are re-stamped with the seeds' births (`tools/release/restamp.py`). The recipe and the weld force
  (20 ms mean at most 20.2 N; the slip rule: 65 N) are in docs/simulator/families/fiber_bundling.md, References, "Hard".
- **Evidence** (2026-11-14, `scripts/roboarc/harbor_check.py --instances --references`, rendering off, seed 0; image
  `roboarc:2.2.0` = f8fdee211ec5, simulator source 1552f29d, tooling f2320e4c): the privileged oracle 1, deterministic,
  both grader replays at 17b6af0171a92459; the standard oracle 1, live (9915 control steps, no row slowed) and fresh
  replay both at 17b6af0171a92459; nop 0 in both modes. The host had no ghcr login, so the trials ran the resource
  monitor from a local registry by the same manifest digest (`roboarc-monitor@sha256:ea1b0d4b…`). Seeds 1 and 2: one
  strict replay each of the re-stamped file succeeds (no Harbor seed run, so `[metadata.seeds] oracle` is `[0]`).
- **Tasks:** `tasks/roboarc-fiber-bundling-05-i01-{privileged,standard}` rendered again: `oracle = "full"` and
  `[metadata.seeds] oracle = [0]` in `task.toml`, `solution/`, the README's Oracle row, Reference solution, nop record
  and Change Log, `DIGEST`. The instruction, frozen instance, physics and graders are unchanged: results graded before
  stand.
- **Records:** `scripts/roboarc/oracles/fiber-bundling-05-i01/` (privileged, standard, s1, s2), release.json (both
  modes' evidence, the nop record), `records/freeze_manifests/freeze_manifest.v3.json` (reference, sha256, `expect`).
- **Counts:** all 78 puzzle bases ship a reference (docs/API.md, `tests/api/test_catalogue.py`). The VLA runner's
  default step limit for fiber-bundling-05 is now 19900 (2 x 9915 rows, rounded up; it was 12000), so the 18 one-arm
  puzzles to their limits are 410500 control steps per seed (docs/VLA.md, `tests/api/test_vla.py`).

### Results: the output-token maximum, the context window and the agent's own memory cap are the agent's end (results; nothing changes for a trial that ended otherwise)
- **Why:** `write_task_results.py` voided every trial Harbor ended with an exception other than the cutoff. Two of
  those ends are the agent's own, under the run's fixed configuration, and their reruns only drew again until the
  agent happened not to reach them:
  - `OutputTokenExceededError` (and `ContextWindowExceededError`): GLM 5.3 Flash under Claude Code wrote 32000 tokens
    of thinking four times in a row, after Claude Code's own "Output token limit hit. Resume directly" turns, and
    Claude Code stopped (7 attempts, every model call answered 200). Harbor's hosted runs never retry these either.
  - `NonZeroAgentExitCodeError` after the kernel killed the agent at its own container's memory cap
    (`resource_oom_kills`): the agent's own scripts (numpy over point clouds, seconds from 0.2 GB to the 16 GB cap)
    took the harness process with them (13 attempts: Opus 5.5, GLM 5.3 Flash, DeepSeek V4.1 Flash). The instruction
    gives the agent its 8 GiB limit, and 88 counted trials had the same kill at the cap of a process that was not the
    harness: only which process the kernel picked decided whether the trial was voided.
- **Fix:** both are results, as graded (`AGENT_ENDS`, `own_memory_end()`). A host OOM kill, the monitor's emergency
  kill, no `reward.json`, the verifier's `infra_error`, and the gateway's failed last calls stay "not a result".
- **Docs:** RUNNING.md "Infrastructure failures" and "Resources", COLLABORATOR_SETUP section 5.
- **Tests:** `scripts/harbor/tests/test_result_verdict.py` (synthetic trials); by hand, the published trials of
  2026-11-12: the four GLM and two Opus attempts checked are now `fail`, a 504 timeout stays "not a result".

### `AGENT=dsh`: Harbor's ACP runner gets agent-client-protocol 0.12.1 (agent fix; nothing changes for another agent)
- **Why:** Harbor (0.23 to 0.25, and its 0.25.1 development builds up to 2026-11-10) installs the ACP SDK
  `agent-client-protocol` in its runner's venv during agent setup without a version, so a trial gets the latest
  release. 1.0.0, on PyPI since 2026-11-09 00:34 UTC, has no `text_block`, which Harbor's `acp_runner.py` imports: from
  then on every DSH trial's runner stopped at `ImportError: cannot import name 'text_block' from 'acp'`, before dsh
  started and before any model call. Until then the latest release was 0.12.1 (uv does not pick the 1.0.0 release
  candidates).
- **Fix:** `scripts/harbor/agents/dsh_acp.py` installs `agent-client-protocol==0.12.1` for the runner, the release
  every DSH trial before 2026-11-09 ran with; dsh, its launcher and the agent's environment are unchanged. A Harbor
  release whose setup command no longer has that install line stops the agent's setup with an error naming it,
  instead of installing an unpinned SDK.
- **Tests:** `test_dsh_acp.py` checks the runner's install line (Harbor 0.23, 0.24 and 0.25). By hand, 2026-11-10:
  Harbor's `acp_runner.py` of 0.23.0, 0.24.0 and 0.25.0 starts with SDK 0.12.1 and stops at that import with 1.0.0.

### `AGENT=kimi-code`: Kimi Code CLI on Moonshot's API (runner, agent and gateway; nothing changes for another agent or provider)
- **Why:** a collaborator runs the evaluation with Kimi Code CLI and Kimi K3.1 Pro on Moonshot's API. Harbor's own
  kimi-code agent (Harbor 0.23 to 0.25) installs whatever version is latest, leaves the CLI's web tools on (FetchURL
  fetches from the container itself), takes no reasoning effort, and writes trajectory.json from 0.25 on only; and the
  usage gateway knew no Moonshot host. Whether these runs are official is the owner's call.
- **The agent:** `scripts/harbor/agents/kimi_code_cli.py` (on Harbor's BaseInstalledAgent; Harbor 0.23, 0.24 and 0.25):
  Kimi Code CLI 2.5.2 (npm `@moonshot-ai/kimi-code`; Node through Harbor's nvm snippet) installed during agent setup and
  run as Harbor's own agent runs it (`kimi --prompt ... --output-format stream-json`, `agent/kimi-code.txt`,
  `agent/.kimi-code/`); `WebSearch` and `FetchURL` off in its `config.toml`; the model, key (`KIMI_API_KEY`), base URL
  (`KIMI_BASE_URL`), context window (1M unless `KIMI_MODEL_MAX_CONTEXT_SIZE`) and effort in its `KIMI_MODEL_*`
  variables, the key in the process's environment only; `reasoning_effort` -> its thinking effort (low, medium, high,
  xhigh, max; another is refused before the trial). The CLI takes a prompt on its command line only: its argv holds
  the instruction, as Codex's does. trajectory.json and the token counts from Harbor 0.25's converter of the CLI's
  session journal.
- **The gateway:** a `moonshot` provider (`api.moonshot.ai`, `api.moonshot.cn`; pending the owner's approval, as
  DeepSeek's): chat completions, the model list, and `POST /v1/files` as an upload path for images and videos only.
  Kimi Code uploads every image it shows the model and reads a refusal there as a login failure that ends its run
  (checked with the CLI's release binary against a stand-in Moonshot, 2026-11-06), so the upload is forwarded when it is
  one multipart body with one listed purpose, without the JSON checks; another purpose is refused as a provider-run
  tool. `record_fields`: each model request's `thinking` (and `reasoning_effort`) as sent, in usage.jsonl
  (`request_fields`): Kimi Code sends the effort as `thinking.effort`. Moonshot's usage in a stream's last
  `choices[0]` and its `cached_tokens` are read; the tool rule also looks at a message's `tools`.
- **The runner:** `AGENT=kimi-code` needs `USAGE_GATEWAY` (its token budget is the trial's) and a `KIMI_BASE_URL` at
  Moonshot's API (default `https://api.moonshot.ai/v1`; `.cn` for the China platform), given to the gateway as
  `--extra-host` and to the agent as the gateway's route; an `EFFORT` outside the CLI's levels, `WEB_SEARCH=allowed` and
  `AGENT_CONFIG` are refused at start. `scan_secrets.py` looks for `KIMI_API_KEY`; `account_failure_check.py` stops the
  lane when a Kimi Code trial's last model call was answered 401, 402 or 429 (`exceeded_current_quota_error`, a spent
  balance, is a 429 the CLI does not retry). `.env.example` and `bash-runners/start_lanes.sh` name the variables.
- **Tests:** `test_kimi_code.py` (the agent against a stand-in environment, Harbor 0.24 and 0.25),
  `test_usage_gateway_option.py` (arguments, both hosts, refusals), `test_paths.py` (the provider, uploads, the tool rule
  on messages, an end-to-end run with Moonshot-shaped answers), `test_budget_stop.py` (both usage shapes),
  `account_failure_check.py --selftest`. By hand, 2026-11-06: the CLI's 2.5.2 release binary through the gateway to a
  stand-in Moonshot, offline: it sends `thinking: {"type": "enabled", "effort": "max", "keep": "all"}`, offers no web
  tool, uploads the image (multipart, with a Content-Length) and refers to it as `ms://<id>`, and the gateway records
  the usage, the upload and the effort; with the upload refused the run ends at its first image.

### Gemini 4 Argon on GitHub Copilot: requests kept under 4.5 MiB (runner and gateway; nothing changes for another model)
- **Why:** six of the 90 trials of a collaborator's Gemini 4 Argon privileged run (2026-11-04) were voided: a request of
  5.0 to 5.4 MiB, with 8 to 10 images, was answered 413 "Request Entity Too Large" and the SDK ended the run. The largest
  request answered was 4.98 MiB, while the same proxy's other endpoints answered Codex requests of up to 25 MiB and
  Claude Code ones of up to 20 MiB. Whether Copilot or the proxy sets the limit is not known yet. The image cap of 10
  counts images, not bytes.
- **Size cap:** the gateway's new `--image-bytes B` (0 = off, the default): a request still over B bytes after the
  image cap has more of its oldest images replaced, in the image cap's steps, until it fits; the newest image always
  stays; a request still over B with no older image to replace is forwarded as it is and recorded with
  `over_image_bytes`. The runner's `GATEWAY_IMAGE_BYTES` is 4718592 (4.5 MiB) for `MODEL=<provider>/gemini-4-argon` and
  0 for every other model, as set when set by hand. Off, the gateway is not given the option and its records,
  `usage_recording.json` and the results are as before; on, those two and the results record `image_bytes`.

### Official runs: Codex and Claude Code through a GitHub Copilot proxy count (docs only)
- A GitHub Copilot proxy on the run's host, which holds the Copilot credentials and passes the agent's requests on to
  Copilot, no longer keeps a run of Harbor's Codex or Claude Code agent from being official (decided 2026-11-02).
  RUNNING "Official runs" and COLLABORATOR_SETUP ("Keep as they are", and the credentials table's new row) say so.

### Antigravity + Gemini 4 Argon on GitHub Copilot: an image cap of 10 and server errors retried (runner and gateway; nothing changes for another model or agent)
- **Why:** a collaborator's first run (2026-11-02) voided both trials. Standard mode: Copilot refuses a Gemini 4 Argon
  request with more than 10 images ("too many images: maximum allowed for model gemini-4-argon is 10, got 12"), and the
  gateway's cap of 50 never applied. Privileged mode, at 2 h 46 min: one request ended in a 500 of the proxy ("fetch
  failed"), and the SDK, whose evaluation preset retries 429 and 503 only, ended the run.
- **Image cap:** the runner's default `GATEWAY_IMAGE_CAP` is 10 for `MODEL=<provider>/gemini-4-argon` and stays 50 for
  every other model; a `GATEWAY_IMAGE_CAP` set by hand is used as it is. Results record the cap (`image_cap`) as before.
- **Server errors:** with `--gemini-chat` (`AGENT=antigravity`) a 500, 502 or 504 of the model server, and the
  gateway's own 502, reach the SDK as 503 (`UNAVAILABLE`, the server's status named in the message), so the SDK retries
  them with its backoff. usage.jsonl keeps the server's `status` and records `agent_status`; a trial whose last model
  calls failed is still not a result. Without the flag nothing changes.

### `AGENT=antigravity`: Antigravity + Gemini through a Chat Completions server (runner, agent and gateway; no task, image or result changes for other agents)
- **Why:** the Antigravity + Gemini 4 Argon pair is served only through an OpenAI-compatible Chat Completions server
  (GitHub Copilot's proxy). The Antigravity SDK speaks the Gemini API only, and its own OpenAI-compatible mode, meant
  for local servers, drops every image a tool returns and never sends a reasoning effort (checked 2026-11-01 on
  google-antigravity 0.1.9 and 0.1.23, offline, against a recording server); Harbor's own antigravity-sdk agent pins
  0.1.9 (Harbor 0.23, 0.24) and leaves the SDK's web tools on. Runs this way are the pair's official runs (decided
  2026-11-01).
- **The agent:** `scripts/harbor/agents/antigravity_sdk.py` (on Harbor's BaseInstalledAgent; Harbor 0.23, 0.24 and 0.25):
  google-antigravity 0.1.23 with every dependency locked (`agents/antigravity/run_agent.py` and its uv lock), installed
  with uv 0.11.30 during agent setup and run offline; the SDK's evaluation preset (`LocalAgentConfig.eval()`) with
  `search_web` (a second model call with Google Search, run at the provider), `read_url_content` and `generate_image`
  off; the instruction in a file; `reasoning_effort` -> the SDK's thinking level; token counts from the SDK's session
  totals.
- **The gateway:** `--gemini-chat` (`scripts/usage_gateway/gemini_chat.py`) translates the SDK's Gemini requests into
  Chat Completions requests and the answers back (function calls and responses, images returned by tools, tool
  choice, thinking level -> `reasoning_effort`, streaming, errors); every check applies to the translated request
  (path list, provider-run tools, image cap, budget) and a Gemini provider-run tool (`googleSearch` and its kin) is
  refused. usage.jsonl records the agent's request (`translated_from`). Without the flag nothing changes.
- **The runner:** `AGENT=antigravity` needs `USAGE_GATEWAY` and `ANTIGRAVITY_BASE_URL` (the server's plain-HTTP base: a
  proxy on the run's host) and `GEMINI_API_KEY`; an `EFFORT` without an SDK level, `WEB_SEARCH=allowed` and
  `AGENT_CONFIG` are refused at start. The job's usage_recording.json and each result's config.json record
  `"api_translation": "gemini-chat"` (no other result gets the key). `scan_secrets.py` also looks for `GEMINI_API_KEY`.
- **Tests:** `test_gemini_chat.py` (the translation; the gateway end to end with a stub server, with and without the
  flag), `test_antigravity_sdk.py` (the pins and the lock; install and run against a stand-in environment with Harbor
  0.23, 0.24 and 0.25), `test_usage_gateway_option.py` (the runner's arguments and refusals). By hand, 2026-11-01: the
  agent installed in a `roboarc:2.2.0` container (Python 3.11) and run with the network cut, through the gateway to a
  scripted Chat Completions model (the command ran, the image arrived, `reasoning_effort` on the wire, the SDK's token
  totals equal to the gateway's); and a Harbor 0.23 trial of a toy task under an egress allowlist of the gateway and
  the registries (reward 1; Harbor's token counts equal to the gateway's; a direct fetch from the container refused).

### The agent's time: 11400 s (3 h 10 min) instead of 10800 s (owner's decision; 178 task dirs and their DIGEST)
- **Why:** the agent's clock in `/run/rcb-limits/usage.json` (`time_left_s`, `deadline_utc`) counts from the trial
  environment's start, before Harbor installs the harness; Harbor's cutoff counts from the agent's run. The install
  (48-401 s in recent trials) came out of the clock the agent plans by, and agents stop when it reads 0: a Codex trial
  of 2026-10-31 with a 401 s install stopped at `time_left_s` 0 after 10406 s of its 10800.
- **10800 -> 11400:** `scripts/roboarc/mk_tasks.py` `CUTOFF_SEC` and `ikea/scripts/ikea_mujoco/mk_tasks.py` `AGENT_S`,
  rendered into every task's `[agent] timeout_sec`, the instruction's `## Time limit`, the README's `Agent time budget`
  row, the monitor's limits (`environment/rcb_monitor/limits.json`, `tests/rcb_limits.json`) and the standard sim
  service's `RCB_WALL_CLOCK_S`. wire-splicing-01-i01 keeps its 21600 s. The grading timeouts stay: they already cover
  11400 s. The IKEA standard service's default `RCB_WALL_CLOCK_S` is 11400 too, and the IKEA README's reference timing
  no longer says "against the N s budget" (each record names the 10800 s it was measured against).
- **Results:** `cutoff_s` is part of the configuration results are ranked by, so trials at 11400 s form a group apart
  from earlier ones at 10800 s. A batch runs on one release.
- **Docs and tests:** README, docs/API.md, docs/RUNNING.md "The agent's time", docs/benchmarks/{roboarc,ikea_mujoco}.md,
  the 18 family pages, docs/COLLABORATOR_SETUP.md (the run's duration: about 884 GPU-hours if every trial used its
  whole time); `roboarc/registry.py`'s default; `tests/api/test_catalogue.py`.

### `bash-runners/`: the evaluation run one seed at a time; a batch run in parts keeps every trial in its plan (runner fix)
- **The bug** (the `SEEDS` entry below): the batch's plan (`$JOBS/.plan/<batch>.json` and `.tsv`) listed only the trials
  of the last command's seeds. After `SEEDS="0"`, then `SEEDS="1 2"`, it held the seed-1 and seed-2 trials only, so
  `scripts/send_results.sh` indexed the seed-0 jobs with no task, mode, trial or seed, and its MANIFEST gave the seeds
  used as `[1, 2]`. A lane of other seeds started on the same batch rewrote the plan under the lanes already running:
  they then took its seeds' trials instead of their own, and could find no seed in the plan for a trial of theirs.
- **The fix:** the plan lists every trial of the batch whatever `SEEDS` is (the plan without `SEEDS`); a lane with
  `SEEDS` takes its trials, in their order, from `$JOBS/.plan/<batch>.seeds-<s>-<s>.tsv`. Lanes of other seeds can run
  on the same batch at the same time, each its own trials, with no order between them. Without `SEEDS` nothing
  changes: the plan, its order and the files in `.plan/` are as before.
- **`bash-runners/`:** `run_seed0.sh`, `run_seed1.sh` and `run_seed2.sh` start the lanes of docs/COLLABORATOR_SETUP.md
  section 5's command (`K=3`, the usage gateway with `GATEWAY_RAW=0`, 4 lanes per GPU, `nohup`) with `SEEDS` 0, 1 or
  2; `start_lanes.sh` holds the command. Run them one after the other, the next once the previous one's lanes have
  ended, with the same settings and batch name. They refuse to start without `MODEL`, `RESULTS` or an `EFFORT` other
  than `default`, or without the `ENV_FILE`.
- **Tests:** `test_runner_seeds.py`: with `SEEDS` the plan is the plan without it (the order in the seeds file); a
  batch run in two parts has every trial in its plan; a `SEEDS="1 2"` lane started during a `SEEDS="0"` lane's first
  trial leaves that lane its trials (each of the three fails on the runner before the fix); without `SEEDS`, `.plan/`
  holds only `<batch>.json` and `.tsv`. `test_bash_runners.py` (a fake runner): each script's lanes and settings, and
  the refusals.

### The runner can run a batch seed by seed: `SEEDS` (runner only; no task, image or result changes)
- **Why:** with `K=3` the runner went task by task, each task's three trials (seeds 0, 1 and 2 in an order set by the
  batch name) one after the other, so a batch stopped early had all seeds of some tasks and none of others.
- **The change:** `SEEDS="0 1 2"` (or a part of it, e.g. `"1 2"`; `SEQ=1` only) runs, of the K trials, those of the
  listed seeds, every task's trials of the lowest seed first. The trials keep their numbers and seeds, so `SEEDS="0"`
  and later `SEEDS="1 2"` with the same batch name and `JOBS` make one K=3 batch. It is an order of priority: a later
  seed's trial starts only in room no waiting trial of an earlier seed fits, so seed-1 trials fill the GPUs the last
  seed-0 trials leave free. With `SEEDS` a lane reads the plan from the top after each trial it starts, and while a
  trial of an earlier seed waits for room it does not try a later seed's deformable trial (the slot ledger would keep
  or drain a GPU for it). A task allowed a listed seed that its K trials do not run stops the runner at start. Without
  `SEEDS` nothing changes. docs/RUNNING.md "Running the seeds in turn"; docs/COLLABORATOR_SETUP.md section 5 uses
  `SEEDS="0 1 2"`.
- **Tests** (`scripts/harbor/tests/test_runner_seeds.py`, fake harbor): the plan by seed with the same trials and seeds;
  two runs (`"0"`, then `"1 2"`) making the batch; a seed-1 deformable trial waiting for the seed-0 trials of a GPU; a
  seed-1 trial filling free room before seed 0's last trial is done; the lane back at the top of the plan after each
  trial (a seed-1 trial that had no room goes before seed 2); the refusals. Without `SEEDS`: the plan, the start order
  and the log as before, and a waiting deformable trial still taking the GPU before a 1-unit one. The fake harbor
  takes per-job agent and validation times (`FAKE_PHASES`). The file now runs about 6 minutes: its limit in
  `ci_checks/check-unit-tests.sh` is 600 s.

### The usage gateway caps the images in a model request at 50 (runner and gateway; no task or image changes)
- **The problem** (a collaborator's smoke test, 2026-10-31: Codex + GPT-6 Astra through a GitHub Copilot proxy,
  claws-08): an agent that looks at the cameras resends every image it has seen with each call. After 52 minutes the
  standard-mode trial's request carried 51 images (21 MB), Copilot answered 400 "Exceeded maximum number of images (50)
  allowed in the request.", Codex exited, and the trial was voided. Most long standard-mode trials pass 50 images.
- **The change** (owner 2026-10-31): `gateway.py --image-cap N` and the runner's `GATEWAY_IMAGE_CAP` (50 by default,
  0 = no cap). A request with more than N images (the image blocks of the Anthropic, Responses and Chat Completions
  shapes, tool results included) has its oldest replaced by a text note of its API's shape, N/2 at a time, so between N/2
  and N stay. The rule reads the request alone and replaces the same images from one call to the next until the
  following step, so the provider's prompt cache is broken once per step. A request within the cap is forwarded as
  sent; a capped one re-serialized and uncompressed. The image files the agent saved are not touched.
- **Recorded**: `usage.jsonl` `images` / `images_omitted`, `totals.json` `image_cap_requests`, `usage_recording.json` and
  every result's `config.json` `image_cap` (null without the gateway). Without the gateway there is no cap. PROTOCOL
  H.6, docs/RUNNING.md "Usage recording", docs/COLLABORATOR_SETUP.md.
- **Checked** on the smoke test's own conversation (Codex's session log, rebuilt as a Responses request: 21.1 MB, 51
  images): 25 replaced, 26 kept, 0.12 s.
- **Tests**: `scripts/usage_gateway/tests/test_image_cap.py` (new: the step rule, the three shapes and tool results, a
  tool call's arguments not looked into, the replaced images stable between calls, end to end through gateway.py with a
  gzip body capped and one forwarded as sent); `test_usage_gateway_option.py` (the cap in the records and passed to the
  gateway; a bad `GATEWAY_IMAGE_CAP` stops the runner at start).

### Model calls through the usage gateway are no longer cut after 15 s (runner only; no task, image or result changes)
- **The problem** (Jianwen's report, 2026-10-31): with the usage gateway, Harbor's egress sidecar (gost, a transparent
  proxy that enforces the task's allowlist) reads the agent's plain-HTTP calls to the gateway and waits at most
  `readTimeout` for the response headers. Harbor's `gost.yaml` sets none (Harbor 0.23.0 to 0.25.x), and gost's default
  is 15 s. A call whose provider took longer to send its headers was cut: the agent saw `ECONNRESET` and retried, the
  provider still completed the call, and the gateway logged the client as gone. In a collaborator's deliveries 173 of
  2172 calls (Copilot proxy, Claude Code + Opus 5.5) and 4 of 42 (a rate-limiting Azure endpoint, Codex + GPT-6 Astra)
  were cut, every one after 15.0-15.9 s. Grades were not wrong (a trial that ends on failed model calls is an
  infrastructure failure), but it cost tokens, time on the 3 h clock, and stopped lanes.
- **The fix**: `scripts/harbor/environments/roboarc_docker.py`, a subclass of Harbor's `DockerEnvironment` whose egress
  sidecar is built from a copy of the installed Harbor's sidecar folder with `readTimeout: 900s` (the gateway's own
  upstream timeout) in the handler's metadata. The runner passes `-e roboarc_docker:RoboARCDocker` to every `harbor run`
  when `USAGE_GATEWAY` is set, and stops at start if the class does not import with the installed Harbor. Harbor's
  install is not edited; the patched sidecar image is built once per machine and Harbor version.
- **Checked** (Harbor 0.23.0, oracle agent, a task allowlisting the docker bridge and a host server that sends its
  headers after 5 s and 20 s): with Harbor's environment the 20 s call is cut after 15 s (`wget: error getting
  response`, the server's write hits a broken pipe); with `roboarc_docker` both calls are delivered, and example.com and
  pypi.org stay blocked.
- **Tests**: `scripts/harbor/tests/test_roboarc_docker.py` (new; a stand-in Harbor: the patched gost.yaml, the copy
  byte for byte with modes, reuse, an unexpected gost.yaml stops the import); `test_usage_gateway_option.py` (the runner
  passes the environment with the gateway, not without). Docs: docs/RUNNING.md "Usage recording".

## 2.2.1 (2026-10-30)

Standard mode returns mp4 video again, and every task's time limit says that the agent works alone. The same images
(`roboarc:2.2.0`, the IKEA images 1.1.0), physics, instances, references and graders: privileged results graded under
2.2.0 stand, and so do standard results of trials that never asked for mp4 (their task digests differ: the texts and
the sim service changed).

### The standard service's mp4 video (task change: the 78 puzzle standard directories' `environment/sim/rcb_limited.py`)
- **The problem** (found by Jianwen in a trial a collaborator sent back: claws-08, seed 1, Claude Code + Opus 5.5): the
  puzzle standard service's `encode_mp4` (`scripts/eai_standard/v2_4/rcb_limited.py`) imported `imageio`, which the
  `roboarc` image has never shipped (2.2.0, 2.1.0 and 1.3.0 checked). A `submit` without `video` (the client's default
  is `"mp4"`), or a `submit` / `get_obs` with `video="mp4"` or `"both"` (the instructions' example uses `video="mp4"`),
  executed its rows and then failed while it built the reply: a simulator error (PROTOCOL §4.6), so the episode ended
  with `reason: "error"`, `infra_error: 1`, and the trial was not a result. A rerun that asked for mp4 again ended the
  same way. The release smoke never asked for video: the oracle replays use `video="none"` and our own standard smoke
  happened to use `frames` and `none` only. The graders' replay videos were not affected (they run the ffmpeg binary
  through `topogym.recording`).
- **The fix**: `encode_mp4` writes the frames to the image's ffmpeg binary (libx264, yuv420p, faststart, the same
  output as before; odd sizes are padded to even ones). No image is rebuilt. IKEA's standard service
  (`scripts/eai_standard/v2_4_live/rcb_limited.py`) is unchanged: its image ships imageio (and imageio-ffmpeg's own
  binary, no `ffmpeg` on PATH).
- **Tasks**: the 156 puzzle directories rendered again (the 78 standard ones' `rcb_limited.py`; the Release line
  "RoboARC 2.2.1" and a Change Log entry in every README; `DIGEST`). The IKEA directories change only by the time limit
  note below.
- **Tests**: `scripts/roboarc/tests/test_server_requests.py` (5) asks the service for mp4 (a `submit` with the default
  video, an `observe` with `"both"`) and checks the mp4 parts and that the episode stays alive; without `ffmpeg` on
  PATH it is skipped with a note (`RCB_REQUIRE_FFMPEG=1` makes that a failure). In the `roboarc:2.2.0` image it passes
  with the new service (47 checks) and fails with the old one (HTTP 500, `infra_error` 1, as in the trial). A release
  runs it in the image (docs/DEVELOPING.md, Cutting a release, step 4).

### Several environments at once: `roboarc.make_vec` (preview; API only, no task, image or result changes)
- `roboarc.make_vec(task, num_envs, instance_seeds=..., gpus=..., autoreset=..., **make_options)` runs `num_envs`
  environments of one task, one worker process each (`roboarc/vector.py`, `VecEnv`), and returns batched observations,
  rewards and flags. The simulators are one scene per process, so this is the supported way to step several; the
  docs pointed at `gymnasium.vector.AsyncVectorEnv` before, which needs Gymnasium (the images do not ship it).
- Each worker builds its environment with `roboarc.make()`; physics, rules and images are the single environment's.
  `instance_seeds="all"` spreads the workers over the appearance seeds; `gpus` gives each worker one GPU.
- docs/API.md "Several environments (preview)". Tests: tests/api/test_vector.py (the worker protocol with a stand-in
  environment anywhere; marker sim: three workers of one task, seeds 0, 0 and 1, stepped with the reference rows).

### A trial validates in the background while its lane starts the next trial
- A lane ran one trial from start to end: after the agent's phase (up to 3 h) Harbor replays the trial, records its
  replay video and grades it (privileged: in the verifier container; standard: in the sim sidecar's collect hook), and
  the lane, its GPU unit and its CPU and memory reservation waited for all of it. In the Embodied First Exam's records
  the replay is 11.6 % (privileged) and 22.4 % (standard) of a trial's time.
- `PHASE_SPLIT=1` (the default; `GPU_SCHED=1`, `SEQ=1`): Harbor gets the job plugin
  `scripts/harbor/plugins/phase_marker.py` (new), which writes `$JOBS/.phase/<job>.json` on Harbor's agent-end event.
  The lane checks the usage gateway and the model account then, turns the trial's slot into a validating one and takes
  the next trial; the trial validates and writes its results in the background, and the lane ends when its trials
  have. A validating trial stays on its GPU, frees its unit and keeps only its task's verifier CPUs and memory; a GPU
  holds at most `GPU_RESIDENT` (2 x `GPU_SHARE`) trials in either phase, and a deformable trial's agent phase needs a
  GPU on which no other trial reasons or validates. `<job>/placement.json` records the mode. `PHASE_SPLIT=0`, or a
  Harbor without `--plugin`, runs as before.
- `scripts/harbor/gpu_slots.py`: holds have a phase; `phase` (new) turns a hold into a validating one (no units, the
  smaller CPUs and memory, owned by the process that waits for the validation); `acquire --resident`; `classify --phase
  validate`. `scripts/harbor/live_board.py` shows a trial with the marker as verifying.
- Tests: scripts/harbor/tests/test_phase_marker.py (new), scripts/harbor/tests/test_gpu_slots.py,
  scripts/harbor/tests/test_runner_seeds.py.

### The time limit tells the agent it works alone (task change: every task directory's instruction.md and DIGEST)
- Harbor runs each harness non-interactively (Codex's `codex exec`, for example): the agent's first reply without a
  tool call ends its session, and nobody answers it. In a Codex trial of interlocked-claws-08 (gpt-5.3-codex, high) the
  agent closed with an offer to the user ("If you want, I can keep iterating on gen.py ...") after 68 min of its
  10800 s and was graded on a placeholder trajectory, unsolved.
- The note after "You have N seconds ..." (`scripts/task_conventions.py` `CUTOFF_NOTE`, every mode) now says, before
  the clock sentence: "You work alone: nobody reads your replies or answers questions during the session. Ending your
  turn, or stopping to ask a question, ends the session, and the task is graded at once as it stands. While the task
  is not solved and time is left, keep working." Harness-neutral, the same words in every task, no duration in them.
- All 180 task directories are rendered again: the `## Time limit` section of `instruction.md` and `DIGEST`, nothing
  else. Results of earlier task versions no longer match the task digests (`digest_match` false). Generators, physics,
  graders, images and the cutoffs are unchanged.

## 2.2.0 (2026-10-28)

Appearance seeds 1 and 2 of every puzzle base change much more than the objects' colours: the room's lighting, the
table top's material, the table's frame and the floor change too, and the task's objects change colour and material,
family by family. The green dashed goal marker is 5 mm wide (it was 2 mm) and drawn glowing at every seed. The IKEA
tasks get seeds 0, 1 and 2, which light the standard mode's camera images differently. The physics, the graders and the
reference rows are unchanged. A new image, `ghcr.io/mll-lab-nu/roboarc:2.2.0`, because the simulator source, the
30 goal-marker bases' frozen scenes and every seed instance change. The task set is 2.1.0's: 90 bases, 180 task
directories. The release also carries the runner and resource changes listed at the end of this section.

### What an appearance seed changes (`64a5c387`, `446feb71`, `b2dbcdc5`)
- Seed 0 is the released look, except the goal marker (below). Seeds 1 and 2 carry a task parameter `scene_look`
  (`source/topogym/world/look.py`, drawn by `scene_variation.scene_look`) besides the family's own appearance parameters
  (`scene_variation.appearance(cls, S)`). `scene_look` changes the renderer's configuration only:
  - lighting: the four ceiling lights keep their x, y; one of lights 1-3 is the key light (three times as bright as the
    others, the only one casting shadows; seed 0's key light is light 0); all four hang at 1.9-2.4 m (seed 0: 2.40 m);
    their light is white of a warm (3600-4300 K) or a cool (8000-10000 K) colour temperature (seed 0: about 6000 K); a
    brightness factor scales them and the environment map, which takes the lights' colour as its tint: 1.25-1.50 on one
    seed, 0.65-0.78 on the other. Of seeds 1 and 2, one is warm and one cool, one bright and one dim, and the one with
    the darker table top takes the brighter light. A dim seed on a dark table top (terrazzo, brushed concrete, coated
    pine) dims to 0.85 at least (`23a52adb`), bed_bolt's dim seed to 0.9 at least (`5f131070`);
  - the table top: wood (ash veneer, cherry veneer, coated pine), metal (brushed steel, satin aluminium) or stone
    (terrazzo, brushed concrete), never seed 0's grey plastic laminate; seeds 1 and 2 take two different classes
    (hedgehog_in_cage never takes coated pine; granite tiles were dropped as too dark, `2c454edc`);
  - the floor: wood parquet, tiles (beige or terracotta), concrete or carpet, two different classes, never seed 0's
    blue-grey tiles;
  - the table top's rim and apron: light wood, black powder coat, aluminium, white edge band, grey rubber or blue
    painted steel, two different ones, never seed 0's dark wood;
  - Nyx also draws the roughness and metalness that MJCF materials and mesh surfaces declare (`renderer.pbr_materials`,
    the robot excepted); seed 0 is drawn as released, without them.
  Each base draws, once for both seeds, which seed is warm and which bright, the order of the key lights and of the
  table's and the floor's classes, and one finish per class. The draw is keyed by the task class: blanket-flatten 04 and
  07 (one class) have the same looks.
- The task's objects, per family (seed 1, then seed 2; the families' finish tables and `scene_variation.AXES`):
  - battery_replacement: a Shock Blue controller (back, cover, compartment and its floor; matte), a lime tray, a yellow
    recycling box, green mats and holder, a blue nylon pry stick; an Electric Volt controller (glossy), a pink tray, a
    green box, blue mats and holder, a yellow nylon stick. The cover changes colour only;
  - bed_bolt: a charcoal-painted frame and holders with brass screws, an antique-brass bolt, a red powder-coated key (the
    ratcheting key: a red handle, a black-oxide head); a lacquered mahogany frame with black-oxide hardware, a blue
    anodised bolt, a black-oxide key (ratcheting: a black rubber handle, a gold TiN head);
  - blanket_flatten, blanket_fold: a red wool in a buffalo check with charcoal; a green sateen striped mustard (seed 0:
    the blue cotton with a hairline grid). The hard tiers' fence (released in Nyx's default white): yellow plastic;
    walnut;
  - buzz_wire_3d: the deck, its wooden obstacles and the wire's end blocks blue paint, walnut stain (seed 0 pale wood);
    the wire brass, stainless steel (copper); the loop gunmetal, polished copper (stainless steel); the handle walnut,
    ebony (pale wood; the wood texture kept); the sleeves red, blue rubber (black);
  - caliper_sorting: a green vise and rack, black-oxide V-blocks, a black-coated caliper, dark red bins, brass sleeves; a
    red vise and rack, blue anodised V-blocks, a gold (TiN) coated caliper, dark green bins, black-oxide sleeves;
  - cast_coil, cast_keyhole: the piece that is gold at seed 0 turns red and the silver one blue (anodised aluminium);
    then black and orange (powder coat);
  - clock_assembling: a ruthenium plate under rhodium bridges, walnut stands, gilt screws; a rhodium plate under
    ruthenium bridges and wheels, blue plastic stands, steel screws (the arbors, pinions, snails and jewels keep their
    colours and change material only);
  - fiber_bundling: the jacket glossy blue PVC, matte orange PUR; the other strands of the palette take other colours;
    the ring teal, black anodised (seed 0 copper); the rear terminal, cable cradle, ring mount and pegs re-coloured as
    an orange enclosure, then a black one;
  - hedgehog_in_cage: the mount black anodised, blue powder coat; the cage and ball of 02, 03 and 08 crimson enamel and
    cobalt anodised, gunmetal and red lacquer; the Cast Cage (06) cobalt anodised, gold plated, its star gold plated,
    black nickel, its push rod black oxide, cobalt anodised; the owl's cage (07) cobalt paint, black iron, the owl
    wenge, beech (wood at every seed), its rod yellow, cobalt;
  - horseshoe_ring: the ring red painted, dark coated metal (seed 0 copper); the horseshoes and chain links black oxide,
    bronze (steel);
  - interlocked_claws: anodised (the target piece purple, the other red; Cast Trinity's T red, H blue, B black); powder
    coat (yellow, blue; Trinity blue, yellow, red); seed 0 brown and teal (Trinity gold, silver, copper);
  - knot_untying: the rope blue nylon, black polyester (seed 0 orange polypropylene); in 06, 07 and 09 the posts and curb
    oiled walnut, blue powder coat, the clamp black oxide, red powder coat;
  - lock_picking: the base teak, green powder coat; the stop frames and key tray blue painted, plywood; the rotor plates
    red, green, white and blue acrylic with an orange END plate, sky blue, pink, lime and teal anodised with a natural
    aluminium END plate; the key brass, gunmetal (seed 0 silver);
  - orbits: the base and posts walnut stained, black lacquer; the stopper ball and rings dark wood, red paint; the cord
    orange polypropylene, blue nylon; the wooden handle another wood, lacquered ebony, oiled mahogany (seed 0 light
    wood; `102100ff`);
  - rope_threading: the rope blue braided nylon, black polyester (seed 0 hemp); the rings and grommets glossy, matte
    powder coat at 90 % shade, their hues kept; the bulkhead, pedestals, posts and stands olive drab, steel blue;
  - slotted_post_ring: the cord blue, red nylon (seed 0 hemp); the post and base oiled cherry, navy lacquer, the beads
    black, teal, the end discs pink, yellow, the metal ring violet, pink anodised;
  - wire_splicing: the terminal blocks green with black-oxide screws, black with brass screws (seed 0 white); the
    bending post's base plate and post black, green (steel); the bare cores: wire A's tinned and wire B's aged
    (tarnished copper), then the other way round (seed 0 copper). Colours only.
- Parts whose colour the texts need to tell parts apart keep their hue: rope-threading's rings and grommets (cyan,
  green, purple, amber, red), wire-splicing's red and blue insulation and the post's yellow rod, fiber-bundling's red
  fibers (strands 0 and 8; 90 % shade at seed 2), battery-replacement's cells (grey with a tan band, black with a copper
  band) and red "+" marks, caliper-sorting's printed scale, vernier, the knob's white line and the bins' white label
  cards, bed-bolt's drive card (white: matte paper, then coated card; its drawing unchanged), and the green goal marker.
  The parts the texts name by material keep it (the metal wire and loop, the wooden handles, the rubber sleeves, the
  nylon pry stick, slotted-post-ring's metal ring, the cloth). `docs/simulator/seeds/colour_table.md` lists them.

### The goal marker: 5 mm wide and glowing, at every seed (`b2dbcdc5`)
- The green dashed line round the goal square is 5 mm wide instead of 2 mm and drawn glowing (an emissive surface, 2.5
  times its green; `nyx_direct.py` draws a mesh entity's emission), so it stands out on every table top and under every
  light. The dashes (20 mm, gaps 10 mm) and the colour are as before.
- The stripes grew outward: each stripe's inner edge, the containment limit the texts name, is where it was, so the
  goal and the grading are unchanged. The marker is a drawn mesh (no collision); the grader reads the goal region, not
  the marker.
- This is every seed, seed 0 included: the frozen scenes of the 30 bases that draw the marker are made again
  (cast-keyhole 02-06, hedgehog-in-cage 02, 03, 06-08, horseshoe-ring 01-05, interlocked-claws 02, 05, 08-13, orbits
  01-02, slotted-post-ring 01-05). Their birth states are unchanged (cast-keyhole-02's stays `aad47b14b8722c9e`).
- knot-untying 07 and 09 draw no marker in the scene; their goal pictures paint 5 mm dashes (`8b2c00e4`).
- An instance records the marker's outer extent as `objects.goal_region.size`, which grew by 6 mm. The texts'
  goal-square side is now that size minus 10 mm (was minus 4 mm; `tools/release/topogym_rules.py` `_square_mm`), so
  the stated sides are unchanged: cast-keyhole-02 states 252.5 mm before and after.

### Texts (`810302a8` and the families' commits)
- The colour words for parts whose colour now changes are gone; the parts are named by shape or role:
  - cast-keyhole 02-05: "the keyhole plate" and "the fork plate", with a sentence on their shapes (a closed, stepped
    hole; a stepped slot open at one end); 06: "the right-hand plate" and "the left-hand plate" (the hand that holds it
    at the start). Was "the silver keyhole piece" and "the gold piece";
  - cast-coil 04-08: "the two Cast Coil pieces" (was "the gold and silver Cast Coil pieces");
  - interlocked-claws 02: "the cross piece" and "the frame piece", 08-10: "the key piece" and "the lock piece", each
    with a sentence on their shapes (was "the brown claw piece" and "the other claw piece"); the release design string of
    06-10 says "a key and a lock";
  - fiber-bundling 02, 04, 05, 08, 09: "the fixed ring" (was "the fixed copper ring");
  - buzz-wire-3d 06, 07, 10: "Rubber sleeves on the wire are insulated" (was "Black rubber sleeves").
- Object names stay as they were: `keyhole_silver` / `keyhole_gold` and `coil_silver` / `coil_gold` are identifiers,
  and a text that shows them says once "(the names are identifiers, not colours)".
- "the inner edge of the 2 mm wide dashed green line" became "the inner edge of the dashed green line" in the texts of
  the 28 bases that state it (the 30 marker bases but interlocked-claws 05 and 11-13, which only say the square is not
  part of their goal; and knot-untying 07 and 09). The texts state no width: until the release renders the tasks and
  the image again, the committed tasks still draw the 2 mm line (image 2.1.0), and the containment limit, the line's
  inner edge, and the square's size are the same for both lines. The width was "5 mm" in between (`810302a8`).
- The same sentence said "the region's border runs along the middle of the line", true of the 2 mm line only. It now
  says "the whole line lies outside it".
- The same words in the catalog texts (`scripts/roboarc/shared/rcb_topogym.py` `TASKS`) and in the release tooling
  (`tools/release/topogym_rules.py`, `tools/release/mujoco_release.json`).
- The rules, limits and numbers the texts state are unchanged.

### IKEA: seeds 0, 1 and 2 (`791a168c`, `897288c2`)
- The 24 IKEA task directories allow seeds 0, 1 and 2 (`[metadata.seeds] build = "runtime"`, `oracle = [0]`); they
  allowed seed 0 only. Each carries the files `seed` (0) that `scripts/task_seed.py materialize` overwrites.
- A seed changes the standard mode's Nyx lighting only, by the puzzles' rule with IKEA's own bands: one of lights 1-3 is
  the key light; the lights hang at a height drawn in 1.9-2.4 m above RoboARC's floor (0.98-1.48 m above the work
  surface; seed 0: 1.48 m); their light is warm (3600-4300 K) or cool (9000-10500 K); the brightness factor is
  1.35-1.60 on one seed and 0.85-0.92 on the other (vesken 1.00-1.05, svallet 1.05-1.10, uppsta 1.10-1.15, whose
  released frames are already near the exposure floor). `environment/sim/rig.json` holds every seed's lights (`seeds`),
  and the sim service picks them by `environment/sim/seed`. Seed 0 is unchanged.
- The privileged mode shows the agent no images and its replay video keeps the scene's own lights: there every seed is
  the same task. The scenes, physics, instructions, graders, tooling and references are unchanged, and so are the IKEA
  images: the render process and its rig are files of the task directory.
- `roboarc.make(..., instance_seed=S)` passes the seed to an IKEA environment's Nyx renderer.
  `ci_checks/check-seed-files.py` checks an IKEA standard task's rig for every allowed seed's lighting (no per-seed
  bindings or staging).

### New visual assets
- Nine CC0 textures from Poly Haven (https://polyhaven.com), each its colour, OpenGL normal and ARM (ambient occlusion,
  roughness, metalness) maps at 1k, JPEG: `ash_veneer`, `cherry_veneer`, `coated_pine`, `terrazzo_tiles`,
  `brushed_concrete` (table tops; the light wood frame uses the ash veneer's maps), `diagonal_parquet`,
  `interior_tiles`, `terracotta_floor_tiles`, `poly_wool_herringbone` (floors; the concrete floor uses
  `brushed_concrete`). They are visual assets, read from the renderer's visual asset root
  (`topogym.assets.library.visual_asset_path`), not files of this repository.
- An image built from this source needs them in its visual asset root. The image recipe's download of them
  (`source/images/roboarc/prepare_nyx_studio.py`: sha256-checked, with their authors in a CREDITS.txt) and their row in
  `LICENSES/THIRD_PARTY.md`. The image build downloads them from Poly Haven; nothing is uploaded.

### Physics unchanged
- Every appearance parameter re-colours or re-finishes drawn geometry (visual geoms, GLB materials, MJCF materials,
  textures) or sets the renderer's lights; no colliding geom, mass, joint or contact parameter is one of them.
- `tools/release/derive_seed_instance.py` makes each seed from the seed-0 instance as in 1.2.0: `instance_assets/` byte
  for byte, and it stops unless the seed's physical fields equal seed 0's, its task parameters differ only in
  `appearance(cls, S)`'s keys and its birth differs from seed 0's in the task's instance id only. The references are
  reused: the same rows, re-stamped with each seed's new birth.
- `tools/tests/test_appearance_seeds.py`: for nine settings of eight families, seeds 1 and 2 leave the declared scene
  and the physics content of every generated file as seed 0's; for three, a `scene_look` changes the renderer's
  configuration only; for every base, the two looks differ as above.

### What the release redid
- The image, built from this source (the simulator source hash in `SOURCE.json`, so the runtime every puzzle task is
  bound to, changes) with the nine textures in its visual asset root. The IKEA images stay.
- The seed-0 instances of the 30 marker bases (their scenes; the births and seed-0 references stay), made again even
  where the release starts from the released instance tree (`release_rebuild.py --remake`). The seed-0 instances of the other 48 bases stay as they are.
- Every seed instance: seeds 1 and 2 of all 78 bases, derived again (`tools/release/release_rebuild.py
  --appearance-seeds 1,2`), and the seeds' references re-stamped (rope-threading 04-08's unstamped references serve
  every seed as before).
- The goal pictures: blanket-fold 01-05 at seeds 1 and 2, knot-untying 07 and 09 at seeds 0, 1 and 2 (seed 0's:
  `release_rebuild.py --seed0-goal-pictures`).
- Every puzzle task directory, rendered again (`scripts/roboarc/mk_tasks.py`): the new texts (`tasks/` holds the 5 mm
  sentence but not yet the reworded colour sentences of 22 bases), the tooling copies (`rcb_topogym.py`,
  `nyx_direct.py`), the runtime bindings, the seeds' bindings and references, `DIGEST`. The IKEA task directories are
  rendered already.
- The pins (`scripts/roboarc/assets.json`, `images/pins.json`), `[metadata.seeds] oracle` from the new Harbor runs, and
  the release checks (`docs/benchmarks/roboarc_seeds_release.md`).

### Results
- The grading is unchanged everywhere (graders, success rules, physics, reference rows; the containment limit has not
  moved): results graded under 2.1.0 stand as graded.
- Seed 0 of 33 puzzle bases looks and reads as in 2.1.0 (battery-replacement-01, bed-bolt 01-05, blanket-flatten 03,
  04, 06-08, blanket-fold 01-05, buzz-wire-3d 02 and 04, caliper-sorting-01, clock-assembling-01, knot-untying 02 and
  04-06, lock-picking 03-05, rope-threading 04-08, wire-splicing-01; the release checks their frames, Evidence);
  results there are comparable with earlier ones.
- Seed 0 of the other 45 shows the agent the wider glowing marker (30 bases), a reworded text (22 bases) or redrawn
  goal pictures (knot-untying 07 and 09); the scene's physics and the grading are the same.
- Seeds 1 and 2 are other scenes than 1.2.0-2.1.0's seeds 1 and 2: results at those seeds, and aggregates over a
  task's seeds, are not comparable with earlier ones.
- IKEA: the privileged tasks are the same at every seed; the standard tasks' seed 0 is unchanged, and their seeds 1 and
  2 are new, so the trials of a run with K > 1 now see different lightings.

### Evidence
Scope (the owner, 2026-10-28): the image goes out on the seeds' physics evidence for all 78 bases and a Harbor smoke
set; the full seed Harbor check (one base per family and the 30 changed bases, seeds 0-2, both modes, oracle and nop:
516 trials, plus knot-untying 07 and 09) was not run and is an optional follow-up.
- **Seed physics, all 78 bases (156 seed instances):** `tools/release/release_rebuild.py` derived seeds 1 and 2 of every
  base in the image without instances (`derive_seed_instance.py`). For each of the 156: `instance_assets/` byte for byte
  seed 0's, the physical fields equal, the birth's tick record different from seed 0's only in `task.instance_id`, and
  the world part of the state (`world_sha`) equal to seed 0's: 156/156. Re-run independently with `--check` (derived
  again, then each loaded strictly in a fresh process): 156/156 reloaded exact, 156/156 derived
  again byte for byte the same instance.
- **Seed 0:** the 30 goal-marker bases' instances were made again in the image (the marker's new drawn mesh); all 78
  seed-0 births equal 2.1.0's, and every reference that records its birth was recorded on it (rope-threading 04-08's
  references carry no birth stamp, fiber-bundling-05 has no reference). The other 48 bases' seed-0
  instances are 2.1.0's, `instance_assets/` byte for byte.
- **Release checks:** `release_rebuild.py` PASS: `mk_tasks.py --check` (all bases), `--assets --check` (every seed's
  binding against its instance bytes), `ci_checks/run_all.sh`, `task_conventions.py --check`, `check-resources`,
  `check-cutoffs`.
- **Harbor smoke** (rendering off, the 2.2.0 image, `scripts/roboarc/harbor_check.py`'s task copies by
  `scripts/task_seed.py materialize`): 55/55 as expected on 19 bases, one per family.
  - nop 0 in both modes at seed 1 on each of them (38 trials; the blanket bases on a GPU);
  - the privileged oracle at seed 1, reward 1 with two equal end states, on 16 of them (the shortest base of each
    family; caliper-sorting, lock-picking and wire-splicing are left out, their references taking more than about 22
    minutes), blanket-flatten-07 and blanket-fold-01 on a GPU;
  - the standard oracle at seed 2 on slotted-post-ring-01, reward 1.
  `release.json` records `seeds.<S>.harbor = "passed"` only for these 17 seed runs, with their scope, so
  `[metadata.seeds] oracle` is `[0, 1]` (`[0, 2]` for slotted-post-ring-01's seed 2) on those bases and `[0]` elsewhere.
- **Image:** `ghcr.io/mll-lab-nu/roboarc:2.2.0`, image id `sha256:f8fdee211ec5dbe0783e9959612e97554eefa493da8e169863c975c415db2eaf`,
  manifest digest `sha256:96a0a895c0e4934adca39f7a370e67bf82bfb4db7ef049131a3ac0dc9d0f9057` (`images/pins.json`);
  `images/archive.sh` twice gives the same archive (sha256 `01c6af183b81...`, 5782784000 bytes); pushed through a local
  registry first, digest and config preserved.
- **CI gates** on a fresh clone: `mk_tasks.py --check`, `ikea/ci/check.sh`, `check-image-pins.py`, `run_invariants.sh`,
  `check-unit-tests.sh --all` with check-video / resources / cutoffs / seed-files, the API tests (CI lock),
  `run_all.sh`, `source_hash.py --check`, `check-doc-paths.py`, `check-tooling-identity.sh`: all pass.

The smoke trials:

| base | seed | mode | agent | reward | end state |
|---|---|---|---|---|---|
| battery-replacement-01-i01 | 1 | privileged | nop | 0 | - |
| battery-replacement-01-i01 | 1 | privileged | oracle | 1 | d486f64d1edf7243 |
| battery-replacement-01-i01 | 1 | standard | nop | 0 | e6f51957c06da025 |
| bed-bolt-05-i01 | 1 | privileged | nop | 0 | - |
| bed-bolt-05-i01 | 1 | privileged | oracle | 1 | af64456a53afae07 |
| bed-bolt-05-i01 | 1 | standard | nop | 0 | f3372966c05c5e78 |
| blanket-flatten-07-i01 | 1 | privileged | nop | 0 | - |
| blanket-flatten-07-i01 | 1 | privileged | oracle | 1 | 717f8f4363f6f3ae |
| blanket-flatten-07-i01 | 1 | standard | nop | 0 | 253a4fa221ac7a26 |
| blanket-fold-01-i01 | 1 | privileged | nop | 0 | - |
| blanket-fold-01-i01 | 1 | privileged | oracle | 1 | 264ec69965c7da2a |
| blanket-fold-01-i01 | 1 | standard | nop | 0 | 23a26f6068f815ba |
| buzz-wire-3d-06-i01 | 1 | privileged | nop | 0 | - |
| buzz-wire-3d-06-i01 | 1 | privileged | oracle | 1 | daa67b9e165cc439 |
| buzz-wire-3d-06-i01 | 1 | standard | nop | 0 | 5a29bb0e3998e26b |
| caliper-sorting-01-i01 | 1 | privileged | nop | 0 | - |
| caliper-sorting-01-i01 | 1 | standard | nop | 0 | d76f4c85f4af192e |
| cast-coil-06-i01 | 1 | privileged | nop | 0 | - |
| cast-coil-06-i01 | 1 | privileged | oracle | 1 | 7cce32ce01151ca6 |
| cast-coil-06-i01 | 1 | standard | nop | 0 | 808cd303f5fd8935 |
| cast-keyhole-02-i01 | 1 | privileged | nop | 0 | - |
| cast-keyhole-02-i01 | 1 | privileged | oracle | 1 | c96947cd0bf652ef |
| cast-keyhole-02-i01 | 1 | standard | nop | 0 | f0a9e0fbeacc4b26 |
| clock-assembling-01-i01 | 1 | privileged | nop | 0 | - |
| clock-assembling-01-i01 | 1 | privileged | oracle | 1 | 6adfa4f1a6653d17 |
| clock-assembling-01-i01 | 1 | standard | nop | 0 | d87d7727dfe445f5 |
| fiber-bundling-02-i01 | 1 | privileged | nop | 0 | - |
| fiber-bundling-02-i01 | 1 | privileged | oracle | 1 | 810fed65e4338d55 |
| fiber-bundling-02-i01 | 1 | standard | nop | 0 | 7c6f4f5204395aaf |
| hedgehog-in-cage-02-i01 | 1 | privileged | nop | 0 | - |
| hedgehog-in-cage-02-i01 | 1 | privileged | oracle | 1 | f6b85d183aff70b3 |
| hedgehog-in-cage-02-i01 | 1 | standard | nop | 0 | 6b66e681ceffc376 |
| horseshoe-ring-02-i01 | 1 | privileged | nop | 0 | - |
| horseshoe-ring-02-i01 | 1 | privileged | oracle | 1 | 3bca9f158e1cf72c |
| horseshoe-ring-02-i01 | 1 | standard | nop | 0 | 51e85046fae03e99 |
| interlocked-claws-09-i01 | 1 | privileged | nop | 0 | - |
| interlocked-claws-09-i01 | 1 | privileged | oracle | 1 | 85aa11a058697a04 |
| interlocked-claws-09-i01 | 1 | standard | nop | 0 | f786d14ce930ca66 |
| knot-untying-02-i01 | 1 | privileged | nop | 0 | - |
| knot-untying-02-i01 | 1 | privileged | oracle | 1 | 20a45d05de285d47 |
| knot-untying-02-i01 | 1 | standard | nop | 0 | 6549955e78941e2e |
| lock-picking-03-i01 | 1 | privileged | nop | 0 | - |
| lock-picking-03-i01 | 1 | standard | nop | 0 | 279a7e3cf16dbb71 |
| orbits-01-i01 | 1 | privileged | nop | 0 | - |
| orbits-01-i01 | 1 | privileged | oracle | 1 | 1614d1cd230074b3 |
| orbits-01-i01 | 1 | standard | nop | 0 | 862dfdaa4f758201 |
| rope-threading-04-i01 | 1 | privileged | nop | 0 | - |
| rope-threading-04-i01 | 1 | privileged | oracle | 1 | edc5ed7d3d0d6e06 |
| rope-threading-04-i01 | 1 | standard | nop | 0 | 678a1a84a9df1832 |
| slotted-post-ring-01-i01 | 1 | privileged | nop | 0 | - |
| slotted-post-ring-01-i01 | 1 | privileged | oracle | 1 | ff731bd870c3203b |
| slotted-post-ring-01-i01 | 1 | standard | nop | 0 | baecb33f68fa047e |
| slotted-post-ring-01-i01 | 2 | standard | oracle | 1 | 35f2c8fc005bf2ad |
| wire-splicing-01-i01 | 1 | privileged | nop | 0 | - |
| wire-splicing-01-i01 | 1 | standard | nop | 0 | b36763d4db794b31 |

### Each trial's record names the machine it ran on
- A trial's results recorded only the index of its GPU (`gpu` in config.json). The GPU's name and driver, the CPU
  model, the memory and the docker and harbor versions were written only by `scripts/send_results.sh` into the
  bundle's MANIFEST.md, at upload time and of the machine that sent it, so a batch copied or uploaded any other way had
  no hardware record.
- `scripts/harbor/host_info.py` (new) reads the machine: CPU model and count, memory, every GPU nvidia-smi lists
  (index, name, driver, memory), docker and harbor. The batch runner logs it when a lane starts, as MANIFEST.md's
  `host:`, `GPUs:` and `software:` lines (each GPU with its memory), and writes it into every job folder as
  `host.json`; `write_task_results.py` copies it into each trial's config.json as `host`, with the name and driver of
  the job's GPU (`gpu_name`, `gpu_driver`). A missing tool leaves its field empty and never stops a run.
- Tests: scripts/harbor/tests/test_host_info.py (new), scripts/harbor/tests/test_runner_seeds.py.

### One list of the family classes; five families get their class's resources (task change: 18 task directories)
- `scripts/task_resources.toml` gave the rigid, rope / chain and cloth values by regular expressions that named the
  families; a task they did not name kept the generic rules' values without notice. The five families released after
  the classes (2026-10-23 / 24) were not named: wire_splicing, bed_bolt, battery_replacement, caliper_sorting and
  clock_assembling declared 2 CPUs, and their privileged tasks had a main container of 4 CPUs / 20 GB where their class
  has 16 CPUs / 30 GB (rigid) or 40 GB (rope / chain). The batch runner reserved 2 CPUs for each of their trials.
- The file now lists the classes once (`[class.rigid]`, `[class.rope_chain]`, `[class.cloth]`: `families`), and the
  class rules name their class (`class = "rigid"`) instead of the families. bed_bolt, battery_replacement,
  caliper_sorting and clock_assembling are rigid, wire_splicing rope / chain. `scripts/task_conventions.py` refuses a
  task that `classes_required` finds (every RoboARC task but IKEA's) whose family is in no class, or a family in two
  classes, so CI's `--check` stops a new family without a class. It also writes `[environment].cpus` from the main
  cap, as the generator (`mk_tasks.py` `main_cpus`) does.
- The 18 task directories of those nine bases are rendered again: `[metadata.resources]`, its limits and caps,
  `[environment]` cpus / memory_mb, the standard sim's cap (16 → 8 GB, its class's), the README's resources, the
  privileged `instruction.md` resource limits (GPU memory 16 → 12 GiB, memory 16 → 24 GiB, 32 for wire_splicing) and
  `DIGEST`. Their `measured` strings say these are the class's values, not the family's own peaks. Earlier results of
  these bases no longer match the task digests (`digest_match` false). The other 162 task directories are unchanged.
- `scripts/harbor/gpu_slots.py` reads the deformable families (the classes with `own_gpu`: rope / chain and cloth)
  from the same file instead of its own list; `DEFORMABLE_FAMILIES` is empty by default and still replaces them when
  set. A RoboARC task whose family has no class is not classified, so the runner does not start it. The deformable set
  is unchanged.
- Tests: scripts/tests/test_task_resources.py (new), scripts/harbor/tests/test_gpu_slots.py.
- docs/COLLABORATOR_SETUP.md's GPU-hour estimate counted the 36 IKEA privileged trials, which take no GPU, as a
  quarter of a GPU each: 837 GPU-hours, not 864.
- Known issue, not changed here: wire_splicing's `instruction.md` states the rigid contact-penetration rules (a success
  with a task contact deeper than 3 mm, or 2 mm for 0.25 s, held for review; deeper than 10 mm a simulator breakdown),
  from `tools/release/topogym_rules.py` `RIGID`. The grader installs those rules only for a task with
  `rigid_measurements` (`rcb_topogym._install_rules`), and `WireSplicingTask` is a `TaskDefinition`, not a
  `RigidTaskDefinition`: its successes are credited without that review.

### The batch runner places each trial on one GPU (runner change; no task, image or result changes)
- Docker gives a container that reserves `count: 1` GPU the host's first GPU: on a host with several GPUs every trial
  of `scripts/harbor/run_agent_batch.sh` ran on GPU 0 and the other GPUs stayed idle.
- With `GPU_SCHED=1` (the default, `SEQ=1`) each lane places a trial through a ledger the host's lanes and batches
  share (`scripts/harbor/gpu_slots.py`): up to `GPU_SHARE` (4) rigid-task trials per GPU, a deformable-task trial on a
  GPU of its own (`DEFORMABLE_FAMILIES`: the blanket, fiber, knot, orbits, rope, slotted-post-ring and wire families),
  no GPU for a task without one (IKEA privileged). A trial starts when the CPUs and memory its task declares
  (`[metadata.resources]`) fit beside the trials running; else the lane tries the plan's next job.
- The job's copy has its containers pinned to that GPU (`device_ids`, `scripts/harbor/pin_gpu.py`), and
  `<job>/placement.json` (in config.json as `gpu`) records it. The copy is all that changes: task digests,
  `digest_match` and earlier results are untouched, and no image is rebuilt.
- docs/RUNNING.md "GPUs and lanes"; docs/COLLABORATOR_SETUP.md starts 4 lanes per GPU. `GPU_SCHED=0` is the old
  behaviour.

### DSH, the DeepSeek Harness, in the batch runner: `AGENT=dsh` (runner change)
- `scripts/harbor/agents/dsh_acp.py`, a Harbor agent of RoboARC's own: Harbor's ACP agent with `@deepseek-ai/dsh`
  0.2.1-rc.2 installed during setup and started as `dsh --profile acp`. That version is the release line of the Embodied
  First Exam's DSH runs (2026-10-15 to 10-18); it picks models from its built-in catalog, where DeepSeek-V4.1-Flash is
  `deepseek-flash`, so `MODEL=deepseek/deepseek-v4.1-flash` gives the Embodied First Exam's run id,
  `dsh-deepseek_v4_1_flash-high`.
- Its instruction travels in a file (`dsh/acp_prompt_file.py`), never on a command line, so an agent's
  `pkill -f <a word of the task>` cannot kill its harness. The session's `reasoning_effort` is set from `EFFORT`. The
  model is requested as dsh's own option value (`["deepseek-official","deepseek-flash"]`), the only form Harbor's
  runner matches, and the prompt goes out only when the session runs that model (dsh's ACP default is
  deepseek-v4-flash, DeepSeek-V4-Flash); both are recorded in the trial's `agent/dsh-session.json`. Token counts:
  dsh reports none over ACP, so the trial's come from dsh's own session records (`agent/dsh-home/`, kept with the
  trial), every session's provider-reported usage summed in Harbor's convention, per session in `agent/dsh-usage.json`;
  a trial stopped at its cutoff keeps them too, less the request in flight when it was stopped (the usage gateway
  counts that one: a 15-min smoke through it had 39 requests, dsh 38, the difference exactly the 39th's usage).
  `WEB_SEARCH=disabled` turns dsh's web_search and web_fetch off (`dsh/roboarc.patch.yml`).
- Credentials: `DEEPSEEK_API_KEY` (`.env.example`; `scripts/harbor/scan_secrets.py` scans for it). Egress: the job adds
  `api.deepseek.com`, or, with `USAGE_GATEWAY`, dsh's `DEEPSEEK_BASE_URL` points at the gateway, whose `api_paths.json`
  now forwards DeepSeek's Messages API (`POST /anthropic/v1/messages`; pending the owner's approval).
- DshAcp is not a Harbor built-in: whether its results are official is the owner's call (PROTOCOL D11). Tests:
  `scripts/harbor/tests/test_dsh_acp.py`, DSH cases in `test_usage_gateway_option.py`, `test_paths.py` and
  `account_failure_check.py --selftest`.

## 2.1.0 (2026-10-25)

The owner's decision on the five bed-bolt bases (hex, Torx, square, Phillips and slotted keys): change only the
appearance, so that the drives tell apart at a glance. A grading change for the blanket families (their release check
no longer counts the fixed arm bases), a grader fix for wire_splicing from the family's stress round, and text fixes. A
new image, `ghcr.io/mll-lab-nu/roboarc:2.1.0`, because the simulator source and fifteen frozen instances change. The
task set is 2.0.0's: 90 bases, 180 task directories.

### Blanket flatten and blanket fold: the release check counts only the moving robot parts (grading change)
- Before 2.1.0 the release check measured the cloth's bounding box (grown by 8 mm) against every robot collision box,
  the two arm bases' among them. The bases are fixed to the world and cannot be withdrawn, so a fully spread sheet also
  had to fit between them: an x-span of at most 0.7075 m and centred, which a sheet turned more than about 15 degrees
  cannot meet. The texts never said so. blanket-flatten-07's reference passed with 0.9 mm to spare.
- Now the check leaves out the boxes of robot links fixed to the world (policy `active_parts_clearance_v1`, terminal
  revision `release_settling_v2_active_parts`); every arm link, wrist, gripper and finger keeps the 5 mm clearance. This
  is the rule knot_untying already uses for its rope (world-fixed bases may touch). The instructions say so (both modes).
- Blanket successes can only increase: every state that passed before passes now. A state that failed only because the
  cloth came near a base now passes, for example flatten-07's spread sheet turned 20 degrees with the hands away
  (old check: 0 mm to a base, failed; new: 62.8 mm to the nearest moving part, passed).
- Earlier blanket failures whose only failing gate was the release clearance against a base can be re-graded from their
  trajectories with the new grader (`scripts/roboarc/regrade.py` where it applies); nothing has been re-graded.
- The terminal revision is part of the task's identity, so the ten blanket instances are refreshed
  (`tools/release/refresh_instance.py`: their scenes and archives byte for byte) and their references re-stamped
  (`instance_sha` only). Their birth and end state hashes change in the task part (the instance id and the release
  metrics) and in the release geometry's new fixed flags only; the physics and the rows are unchanged.
- The other families with a release check are unchanged: cast_coil and buzz_wire_3d keep `all_parts_clearance`,
  knot_untying its exact policy; their state is byte-identical.

### Buzz wire 3D 06, 07 and 10: the loop opening in the texts (`04c7dd90`)
- Their texts now state their own usable loop opening (8.77, 8.27 and 6.77 mm) instead of the first tiers' 12.67 mm;
  `tools/release/topogym_rules.py` computes it. Text only; the grader is unchanged.

### Collaborator note (Jianwen, `eba9248c`)
- `docs/COLLABORATOR_SETUP.md` is a collaborator's one note (agent logins, the settings of an official run, which trials
  may run again); `docs/RUNNING.md` points to it. It names the 2.1.0 image and 2.0.0's task set (90 bases, 540 trials).

### Bed bolt: the drive card (Jianwen, `6c24f636`)
- A white card lies flat on the table between the key's stand and the bolt's cradle, in front of them, with the bolt's
  head drawn on it 4.5 times its size (zinc) and the head's recess in black: hexagon, six-lobe star, square, cross or
  slot, from the recess's own dimensions. In `cam_high` the drawn head is about 50 px across.
- Visual only: drawn geoms (no collision, no mass) on the bed frame's static body, which the MuJoCo scene drops. The
  physics scene (`scene.xml` and every mesh), the births, the references and the grading are unchanged. The card is the
  same at every appearance seed.
- Clear of the references: replayed row by row, no robot, bolt or key surface comes within 87 mm of the card (the
  nearest is the left gripper's finger as it takes the key). It covers only the table in every camera: the key, the bolt
  and the hole are seen as before.
- Each bed-bolt instruction (both modes) says in one sentence: "A white card lying flat on the table between the key's
  stand and the bolt's cradle shows the bolt's head and its recess, drawn 4.5 times their size."
- No per-type handle colours (the appearance seeds already vary the key's finish).

### Wire splicing: the coil count on a straight tail (Jianwen, `2b765e6a`)
- The grader faulted (an exception inside the task's step, reported as a simulator breakdown) when the tail of the
  first wire was pulled straight out past the eye: its return strand was a single node. It now counts no turn there.
  The reference never takes that branch; its end state is the recorded one. `tools/tests/test_wire_splicing_criteria.py`
  (run by `check-unit-tests.sh`).
- `tools/stress/round.py` (`097a3123`): a reference's wall limit in the stress round follows the base's agent budget
  where the owner excepted it (`CUTOFF_EXCEPTIONS`).

### Instances, hashes and results
- The five bed-bolt instances and their seeds 1 and 2 are re-made in the image (`release_rebuild.py`,
  `derive_seed_instance.py`). They differ from 2.0.0's only in the bed frame's drawn parts: `instance.json`
  `asset_identity` (`bed_frame.xml`) and `topogym_source`, and `scene.json`'s record of the bed frame file (its hash,
  the number of dropped visual geoms) and of the robot file's cache name. The state hash covers no drawn geometry:
  every birth (seeds 0, 1 and 2) is unchanged, so the references keep their bytes (no re-stamp), and every end state is
  the recorded one.
- Results graded under 2.0.0 stand for every task but the blanket ones, whose grading is looser now (above). The
  bed-bolt tasks look different and their instruction has one more sentence; their physics and grading are unchanged.
  The other tasks change only in their README's Release line, their Change Log and their image pin (and the buzz-wire
  texts above).

### Image
- `roboarc:2.1.0` (`sha256:c813b33cff3f...`, manifest `sha256:ccc3a47a5d9e...`, `images/pins.json`): every file is
  1.3.0's but the changed source files (`geometry/bed_bolt.py`, `tasks/wire_splicing/criteria.py`,
  `evaluation/terminal/settling.py`, `world/mujoco/release.py`, `world/mujoco/world.py` and its `.pyc`,
  `simulator/env.py`, `tasks/blanket_flatten/common.py`, `tasks/goal_description.py`), `SOURCE.json`, 15 `.pyc` (their
  4-byte source-mtime field only) and the fifteen re-made or refreshed instances with their seeds. The numeric
  libraries are byte-identical. It still carries the instances of the 33 bases removed in 2.0.0, as 1.3.0 does
  (`assets.json` pins them). The IKEA images are unchanged.

### Evidence (2.1.0 image)
- Harbor, bed-bolt 01-05, both modes, seeds 0, 1 and 2: oracle 1 and nop 0 in all 60 trials, no infrastructure error;
  every oracle end (two replays, or the live episode and its replay) is the recorded one of its seed. The privileged
  oracle at seed 0 ran with rendering on (its replay video shows the card).
- Harbor, wire-splicing-01, seed 0, both modes: oracle 1 and nop 0; ends the recorded 48f312d445aec62b (privileged, two
  replays) and 164567f6effde5b0 (standard, live and replay).
- Harbor, the ten blanket bases, seed 0, both modes, oracle and nop, and seeds 1 and 2 privileged oracle: oracle 1 and
  nop 0 in all 60 trials, no infrastructure error (two runs were repeated: a GPU lost to new processes after a host
  service reload, and a standard oracle that ran out of agent time on an overloaded GPU). Every oracle's two ends are
  equal; at seed 0 the new ends are the refreshed instances' (task part) of the recorded physics: a leaf-by-leaf
  comparison of birth and end records, 1.3.0 image and 2.0.0 instance against 2.1.0, differs only in the instance id,
  the release metrics and the new fixed flags. blanket-flatten-06's privileged reference reaches success on these hosts.
- One base per other family (15 bases on the CPU; the blanket and wire families by their Harbor runs above): strict
  replays in two fresh processes, as the verifier grades, end on the recorded states; the renderer's scene inputs
  equal 1.3.0's for one base of every family but bed_bolt, whose scene adds the card.
- The release image archive is byte-identical when made twice; the CI checks pass on a fresh clone.

## 2.0.0 (2026-10-25)

The owner pruned the task set: 33 puzzle bases leave the release. RoboARC 2.0.0 has 90 task bases, 78 puzzle bases in
19 families (every family keeps at least one base) and the 12 IKEA scenes, in 180 Harbor task directories (156 puzzle,
24 IKEA). The image is unchanged: `ghcr.io/mll-lab-nu/roboarc:1.3.0` (`images/pins.json`, `images/roboarc/VERSION`),
so are the IKEA tasks and images.

### Removed bases
| family | removed (difficulty) | remaining (difficulty) |
|---|---|---|
| blanket_flatten | 01 (very_easy), 02 (very_easy), 05 (medium), 09 (hard) | 5: 03, 04, 06, 07, 08 (easier 1, easy 2, medium 1, hard 1) |
| buzz_wire_3d | 01 (very_easy), 03 (easy), 05 (hard), 08 (easy), 09 (medium) | 5: 02, 04, 06, 07, 10 (very_easy 1, easier 2, medium 1, hard 1) |
| cast_coil | 01 (very_easy), 02 (easier), 03 (easy) | 5: 04-08 (one per tier) |
| cast_keyhole | 01 (very_easy) | 5: 02-06 (easier 1, easy 1, medium 1, hard 2) |
| fiber_bundling | 01 (very_easy), 03 (easy), 06 (very_easy), 07 (easier), 10 (hard) | 5: 02, 04, 05, 08, 09 (easier 1, easy 1, medium 2, hard 1) |
| hedgehog_in_cage | 01 (very_easy), 04 (medium), 05 (hard) | 5: 02, 03, 06, 07, 08 (easier 2, easy 2, hard 1) |
| interlocked_claws | 01 (very_easy), 03 (easy), 04 (medium), 06 (very_easy), 07 (easier) | 8: 02, 05, 08-13 (easier 1, easy 1, medium 2, hard 4) |
| knot_untying | 01 (very_easy), 03 (easy) | 6: 02, 04, 05, 06, 07, 09 (very_easy 1, easier 2, medium 2, hard 1) |
| lock_picking | 01 (very_easy), 02 (easier) | 3: 03, 04, 05 (easy 1, medium 1, hard 1) |
| rope_threading | 01 (easy), 02 (medium), 03 (hard) | 5: 04-08 (one per tier) |

The other nine families are unchanged: battery_replacement 1, bed_bolt 5, blanket_fold 5, caliper_sorting 1,
clock_assembling 1, horseshoe_ring 5, orbits 2, slotted_post_ring 5, wire_splicing 1. By difficulty, 33 removed
(very_easy 12, easier 4, easy 7, medium 5, hard 5); 78 remain (very_easy 9, easier 17, easy 14, medium 17, hard 21).
Base numbers are never reused: a removed base leaves a gap in its family's numbers.

### Results
- The kept tasks are unchanged but for their README's Release line ("RoboARC 2.0.0") and `DIGEST`: their instructions,
  graders, instances, references and image are 1.4.0's, so results graded under 1.x on these tasks stand.
- Aggregate numbers (success rates over the task set, per family or per difficulty) of 1.x and 2.0.0 are not
  comparable: the set is different.
- The removed tasks are kept by the maintainers, together with their packages, references and release records, on
  the branch `archive/removed-in-2.0.0` (= RoboARC 1.4.0, `315c10ca`), and they run there on the same image.

### Files
- `scripts/roboarc/release.json`: the 33 records removed from `bases`; the version is 2.0.0 (also `pyproject.toml`,
  `CITATION.cff`, `roboarc.__version__`). Their task dirs, packages and references are removed from the tree.
- `tools/release/mujoco_release.json` keeps their entries, each family with a `removed` note per base;
  `release_rebuild.py` skips those bases. Their entries in `roboarc_names.json`, `release_evidence.json` and
  `standard_speed_rule.json` are removed. `records/` (the freeze manifests) and `scripts/roboarc/assets.json` (the
  instance pins) are unchanged.
- The `roboarc` API lists 78 puzzle bases and 90 tasks; 77 puzzle bases ship a reference (fiber-bundling-05 does
  not), each pinned by the V3 freeze manifest or its release record; `reference_pins` pins only the bases of
  `release.json`. 10 privileged references are faster than the standard joint-speed rule.
- `docs/easy_set.toml`: the 16 removed bases' entries are gone; 15 remain.
- Examples, quick starts, tests and the collaborator guide use kept bases (interlocked-claws-08 and -09,
  cast-keyhole-02, cast-coil-06; the bit-identity selection and `test_vla` accordingly). `docs/COLLABORATOR_SETUP.md`:
  a full run is 90 x 2 x 3 = 540 trials.
- Family pages, the README table and the API documentation give the new counts.

## 1.4.0 (2026-10-24)

The nine bases prepared as experimental in 1.2.0 are released into `tasks/` in both modes, after the owner's review
(2026-10-24): four new families and a thirteenth interlocked-claws base. 111 bases in 19 families, 222 puzzle task
directories (246 with IKEA's 24). The image is unchanged: `ghcr.io/mll-lab-nu/roboarc:1.3.0` (`images/pins.json` and
`images/roboarc/VERSION` stay 1.3.0), so are the IKEA tasks and images. The 1.3.0 tasks' instructions, graders,
instances and references are unchanged: results graded under 1.3.0 stand.

### New bases (by Jianwen)
| base | family | difficulty | the task |
|---|---|---|---|
| bed-bolt-01-i01 | bed_bolt (new) | medium | drive a bed frame's corner bolt home with an L-shaped hex key |
| bed-bolt-02-i01 | bed_bolt | hard | the same with an L-shaped Torx key |
| bed-bolt-03-i01 | bed_bolt | very_easy | the same with a ratcheting L-shaped square key |
| bed-bolt-04-i01 | bed_bolt | easier | the same with a ratcheting L-shaped Phillips key |
| bed-bolt-05-i01 | bed_bolt | easy | the same with a ratcheting L-shaped slotted key |
| caliper-sorting-01-i01 | caliper_sorting (new) | hard | measure four look-alike sleeves with a vernier caliper that reads in its own unit, sort them into labelled bins |
| clock-assembling-01-i01 | clock_assembling (new) | hard | assemble a watch movement's gear train, screw down its bridge, wind it and let the mainspring run it (`dynamics = "dynamic"`) |
| battery-replacement-01-i01 | battery_replacement (new) | medium | swap a game controller's two AA cells, the old ones pried out with a pry stick, and latch the cover |
| interlocked-claws-13-i01 | interlocked_claws | hard | Hanayama's Cast Trinity: separate three interlocked pieces |

- `release.json`: each record moved from `experimental_bases` to `bases` (released 2026-10-24); `experimental_bases` is
  empty. The packages' difficulty `untiered` became the owner's tier. Agent time 10800 s for each.
- Family pages `docs/families/bed_bolt.md`, `caliper_sorting.md`, `clock_assembling.md`, `battery_replacement.md`;
  `interlocked_claws.md` lists base 13.
- The `roboarc` API lists them (111 puzzle bases, 123 tasks; 109 puzzle bases ship a reference, the nine new ones pinned by their release
  records' privileged evidence).

### Tasks
- New: the 18 directories `tasks/roboarc-<base>-{privileged,standard}` of the nine bases. Each equals the
  `mk_tasks.py --experimental` render the evidence below ran on, but for `task.toml`'s difficulty and `DIGEST`.
- The 204 directories of 1.3.0 re-rendered for the version only: the README's Release line names RoboARC 1.4.0, and
  `DIGEST`.

### Evidence (the nine bases)
- **Harbor on the 1.3.0 image** (Jianwen, 2026-10-24; image id `499d6f69bd81`, simulator source `4d2cba21`, tooling
  `412cbe02`): oracle 1 and nop 0 in both modes at seeds 0, 1 and 2, no infrastructure error (108 / 108). Seed 0's
  oracle ends equal the recorded end states in both modes; at seeds 1 and 2 the two replays agree and both modes reach
  the same end. The references are unchanged (trajectory sha256 as recorded).
- From 1.2.0 (the same instances and references): the reference replays on Intel equal Zen 4 (9 / 9); reference
  timings at most 14.1 % of the 3 h budget; each seed's reference ends with the same world, robot and state as seed 0.
- Texts: the success hold (0.5 s) and the tool-frame distances were checked against the instances before the review.
  The texts state the grasping rules in more detail than the tooling's own grasp sentence (`Session.grasp_rules()`),
  without contradicting it; see the benchmark page, "Grasping". One sentence was less exact than the simulator and is
  corrected: battery-replacement-01's text let a pluck go once the cell "has moved 60 mm from where it was taken"; it
  now states the simulator's rule, one 60 mm budget per old cell and episode, shared by all its plucks and counted from
  where the cell leaves the compartment (the grading is unchanged).

### Collaborator tooling
- `scripts/send_results.sh` (by Jianwen): sends one batch's records to the public Hugging Face dataset
  `RoboARC/RoboARC-outputs`: a secret scan of the batch's jobs (redacted, then scanned again), one archive per job
  with an index and a manifest, and a scan of the whole bundle once more before the upload, since the dataset is
  public; `CREATE_PR=1` uploads as a pull request, `DRY_RUN=1` stops before the upload. A batch without a verdict yet is
  still sent. `scripts/receive_results.sh` is the maintainers' side.
- `docs/COLLABORATOR_SETUP.md`: run from `main`; a first check in both modes and at another seed, a smoke test of the
  batch runner, the full run over every base (123 bases, 738 trials) with lanes, duration, hosts and restarts, and
  sending the results back with the collaborator's own Hugging Face account.

### Docs
- README rewritten around the interfaces: coding agents (Harbor), VLA policies (in development, openpi conventions) and
  the Python API, on the same tasks, physics and graders; results are never pooled across interfaces.
- Counts updated (README, `docs/API.md`, `docs/DEVELOPING.md`, `docs/benchmarks/roboarc.md`, `docs/families/`,
  `docs/COLLABORATOR_SETUP.md`); `docs/TAXONOMY.md` names clock-assembling-01 as the one dynamic task;
  `tools/release/mujoco_release.json` records the ten bases' releases (wire-splicing-01's 1.3.0 evidence included).

### Checks
- `mk_tasks.py --check`, the generator and seed tests, unit tests, invariants, cutoffs, doc paths, CI self-test,
  `source_hash.py --check`, the API tests (catalogue counts, reference pins, joint-speed rule: 13 privileged references
  faster than the standard rule, as before).

### VLA policies (preview)
- `roboarc.vla` and `python -m roboarc.vla.run`: graded episodes driven by a policy server speaking openpi's websocket
  protocol, with openpi's ALOHA layout (two-arm robots) and DROID layout (the Franka + Robotiq families; joint
  positions, or DROID's joint velocities); the grader is the standard mode's (live success and a fresh-process replay);
  results in their own layout with a `vla:` configuration key. A stub policy server (`roboarc.vla.stub_server`: hold,
  or a replayed reference). Documentation: `docs/VLA.md`. Extra `.[vla]`: msgpack 1.1.0, websockets 15.0.1.
- `roboarc.make(..., instance_seed=S)` runs a puzzle's appearance seeds 1 and 2 (staged against their own bindings, as
  the graded tasks stage them); `TaskInfo.seeds`. `env.observe()` and `env.step_images = False` render images only
  when asked. Graded behaviour is unchanged.

## 1.3.0 (2026-10-23)

A new family, wire_splicing, with its first base wire-splicing-01-i01 (hard) in `tasks/` in both modes: 102 bases in 15
families, 204 puzzle task directories. The owner reviewed its demo on 2026-10-23. Image `ghcr.io/mll-lab-nu/roboarc:1.3.0` (manifest `sha256:2bcded00…`,
id `499d6f69bd81…`); the IKEA tasks and images are unchanged. Every released task's instance, physics, instruction,
grader and reference are unchanged: results graded under 1.2.x stand.

### wire_splicing (new family)
- The family (by Jianwen, ported from TopoGym `cb3d05437c`): two stiff wires from their terminals, 160 ball-jointed
  links with elastoplastic joints; the agent bends an eye in the red wire's bare end round the yellow rod of a bending
  post, threads the blue wire through it, winds the blue wire's bare tail at least 1.5 turns round the red wire's
  doubled end (at least one full turn round both strands together), folds the red wire's end back and lets go.
- wire-splicing-01-i01: packages, references (privileged; standard re-timed to the joint-speed rule), its record in
  `release.json` (released 2026-10-23, difficulty hard), its instance and seeds 1 and 2 (other colours of the terminals, the post and one
  bare core; the red and blue insulation and the post's yellow rod keep their colours) in the
  image and pinned in `assets.json`. Agent time 21600 s by the owner's exception (below); one control step takes about
  0.5 s. Task directories `tasks/roboarc-wire-splicing-01-i01-{privileged,standard}`; family page
  `docs/families/wire_splicing.md`.
- The `roboarc` API lists it (102 puzzle bases, 114 tasks). A reference's pin (file sha256, end hash) comes from the V3
  freeze manifest, or for a base released later from its release record's privileged evidence
  (`roboarc.registry.reference_pins`).
- Its texts state the instance's own values where the bed-bolt template's had been kept: the tool-frame distance 6 cm
  (not 15 cm) and the success hold 2 s (not 0.5 s; `stable_duration_s`), also in `tools/release/package_text_wire.py`.

### Shared code: opt-in only
- `MultiRopeGraspAssistConfig` (`topogym/utils/grasp.py`) gains `weld_links_between_pads` (a hand welds every link of a
  touched wire whose centre lies between its pads, one binding) and `hold_between_pads` (two fingers on the wires with
  every contact between the pads hold as a pinch), both default off; only wire_splicing's grasp configuration builds
  this class.
- The MuJoCo grasp backend's weld work guard (`topogym/world/mujoco/grasp.py`): a weld whose net work within its hold
  exceeds the scene profile's `weld_work_guard_j` lets go as a slip. It is read from the frozen scene's profile, and no
  other family's profile names it, so it is off for every released instance; with it off, the grasp state a checkpoint
  carries is as before.
- New entries for the family in the registry, variation records, goal texts, appearance axes, scene profiles and the
  tooling's task table (`rcb_topogym.py`, tooling v22); no existing entry changes. SOURCE.json `4d2cba21…` (454 files).
- `tools/tests/test_mujoco_grasp_opt_in.py`: the WeakHold field test follows the fields battery's rule C added.

### Owner exceptions to the cutoff cap
- The protocol's 14400 s cap now admits the bases listed in `scripts/task_conventions.py` `CUTOFF_EXCEPTIONS`, each with
  the owner's approval; `check-cutoffs.py` and `check-task-fields.sh` read the same list. The first entry is
  wire-splicing-01-i01 (owner, 2026-10-22: 21600 s; 160 ball-jointed links at about 0.5 s per control step). For an
  excepted base `mk_tasks.py` writes the cutoff and raises the grading timeouts with it: the privileged verifier to 2 x
  cutoff + 1800 s (45000 s), the standard collect hook to 1.5 x cutoff (32400 s). `cutoff_s` is part of the
  configuration results are ranked by, so its trials form their own group. The 202 task dirs did not change by it.

### Tasks
- New: `roboarc-wire-splicing-01-i01-privileged` and `-standard` (agent 21600 s, privileged verifier 45000 s, standard
  collect hook 32400 s).
- The 202 earlier puzzle directories re-rendered: the tooling copies (`rcb_topogym.py`), the runtime bindings of seeds 0-2
  (source and tooling), the image reference, `tooling_version` 22, the README's Release line and Change Log, `DIGEST`.
  Instructions, graders, instances and references are unchanged; so are the IKEA directories.

### Evidence
- **Image:** every file outside `/opt/topogym` and the new instance is byte-identical to 1.2.0's image (path, mode, size,
  sha256 of 104,536 entries; the numeric libraries unchanged); the 110 earlier bases' instances and seeds are 1.2.0's
  bytes. The simulator source differs from 1.2.0's in wire_splicing's files and the shared entries above only.
- **Released bases, seed 0:** in a 1.3.0 build that differed from this one only in wire_splicing's own files (the needle
  meter, removed before the release; the file manifests show those files and nothing else), the strict double
  fresh-process replay of all 101 references reached its recorded end state, with births = frozen and the instance,
  asset and runtime-binding checks (Zen 4); render inputs 101 / 101 identical to 1.2.0. In this image: one base per
  family again (14 / 14 ends as recorded; render inputs identical), and Nyx frames at birth and after up to 300
  reference rows within 1.2.0's own run-to-run spread (14 / 14 families; the path tracer is not pixel-deterministic
  between runs of the same image).
- **Seeds 1 and 2:** the 220 seed instances pass the four checks against their 1.3.0 bindings (assets and physical fields
  = seed 0's, strict staging, birth parts = seed 0's but `task`, the re-stamped reference); derived again in this
  image they give the same births and `instance_assets/`, and `instance.json` differs only in `derived_from.source`.
- **Harbor:** one base per family, oracle 1 / nop 0 in both modes at seed 0, every end as recorded (56 / 56); in the
  earlier build also seeds 1 and 2 of the same bases (112 / 112).
- **wire-splicing-01-i01:** Harbor oracle 1 / nop 0 in both modes at seeds 0, 1 and 2 (12 / 12; two equal replays each;
  ends 48f312d445aec62b privileged, 164567f6effde5b0 standard); its seeds pass the four checks; Intel Xeon Gold 6230R
  = Zen 4 on the strict replay (48f312d445aec62b); one privileged reference execution on an RTX 4090 Zen 4 host under
  its usual shared load: 8171 s, 37.8 % of the 21600 s budget (0.55 s per control step); along the reference the task
  succeeds only at its last row, with every condition met (no earlier state is credited).
- **Checks:** generator, IKEA, image pins, invariants, unit tests, API tests, advisories, `run_all.sh`,
  `source_hash.py --check`, doc paths and tooling identity pass on a fresh clone.

## 1.2.1 (2026-10-22)

Nyx on GPUs without RT cores: a clear error instead of a crash. The same images (`roboarc:1.2.0`, the IKEA images
1.1.0), physics, instances, references, instructions and graders: results graded under 1.2.0 stand.

- **The problem** (measured on an A100 and an H100, NVIDIA driver 610.43.02): GPUs without RT cores offer the Vulkan
  ray-tracing pipeline and acceleration structures but not `VK_KHR_ray_query`, which Nyx's path tracer uses. The
  driver's shader compiler rejected Nyx's pipelines and the native renderer then died with a segmentation fault, which
  no Python code can catch. So on such a GPU the privileged verifier crashed in its video pass before it wrote the
  verdict (the trial got no reward at all), an agent's first camera call killed its Python process, and standard mode's
  `sim` service crashed at start-up.
- **The fix**: `nyx_direct.py` checks the GPU before it creates the native renderer. It finds the Vulkan device that is
  the CUDA device (the same UUID) and requires `VK_KHR_ray_query`, `VK_KHR_acceleration_structure` and
  `VK_KHR_external_memory_fd` there (`NYX_VULKAN_EXTENSIONS`). A missing extension, no Vulkan device for the CUDA GPU
  (a CPU driver such as Mesa lavapipe) or no Vulkan driver raises `nyx_direct.UnsupportedDevice` (a `RuntimeError`)
  naming what is missing and the GPUs Nyx runs on, at once and without the start-up retry; later builds in the same
  process raise it without trying again. A check that cannot complete leaves the decision to the build, as before.
  The check takes about 0.1 s and changes nothing on a GPU with RT cores.
- **What runs where now** (docs/RUNNING.md, "GPUs without RT cores"): on GPUs with RT cores (GeForce RTX 20-series
  or newer, RTX workstation GPUs, T4, L4, L40 / L40S, A10 / A10G, A40) everything is as in 1.2.0. On V100, A100, A30,
  H100 or H200, standard mode does not run (its observations are Nyx images); the privileged verifier grades from the
  physics replays as on any host, without the replay video (`video_rendered` 0, `media.json` status "error"); agents'
  camera calls raise `UnsupportedDevice`. Such runs are unofficial: the agent sees no images. There is no software
  fallback.
- **IKEA**: the standard sidecar's Nyx render process (`ikea_nyx.py`, with IKEA's own copy of `nyx_direct.py`) has
  the same check; it reports an unsupported GPU in its ready line and the service stops at once
  (`ikea_nyx.UnsupportedGPU`, no start-up retry) instead of crashing three times. IKEA's privileged tasks do not use Nyx.
- **Tasks**: the 202 puzzle directories re-rendered (`nyx_direct.py` copies, the runtime bindings'
  `nyx_direct_sha256` for seeds 0, 1 and 2, the Release line "RoboARC 1.2.1", a Change Log entry, `DIGEST`), and the 12
  IKEA standard directories (the sim's `nyx_direct.py` and `ikea_nyx.py`, a Change Log entry, `DIGEST`). The IKEA
  privileged directories do not change. No image is rebuilt: the task tooling is not in the images (each task
  directory carries its pinned copy).
- **Tests**: `tools/tests/test_nyx_device_check.py` (mocked Vulkan extension lists and a stub gs-nyx, both
  `nyx_direct.py` copies; in `ci_checks/check-unit-tests.sh`); IKEA's `test_nyx_startup.py` covers the unsupported start.
- **Docs**: README, docs/RUNNING.md, docs/API.md, the benchmark page and the image README state the GPU requirement.
  New `docs/COLLABORATOR_SETUP.md` (by Jianwen): machine, access, a first check and the evaluation run, for a
  collaborator; linked from the README.

## 1.2.0 (2026-10-21)

Three scenes per task base (appearance seeds), the simulator source and tooling they need, and nine new bases prepared
as experimental. Image `ghcr.io/mll-lab-nu/roboarc:1.2.0` (manifest `sha256:a8561b77…`, id `b600616bee6c…`); the IKEA
tasks and images are unchanged. 226 task directories (202 puzzle, 24 IKEA), every puzzle directory re-rendered.

### Appearance seeds
- Every one of the 101 bases has seeds 0, 1 and 2 (`[metadata.seeds] allowed = [0, 1, 2]`, `build = "runtime"`, both
  twins). Seed 0 is the released scene, unchanged. Seeds 1 and 2 are the same scene and physics in other colours,
  materials and textures (`topogym.tasks.scene_variation.appearance(cls, S)`, scene seed 0), derived from the seed-0
  instance (`tools/release/derive_seed_instance.py`: `instance_assets/` byte for byte, the build's fields recomputed).
- The runner picks the seed (RUNNING.md "Seeds": the K trials of a task get distinct seeds); the image holds every
  seed's frozen instance under `/opt/robotangle/instances/<base>/seeds/<S>/` and the task stages the trial's seed
  against its own runtime binding (`runtime_binding.s<S>.json`, `stage_instance.py --seed-file`, fail closed).
- References: the privileged reference is re-stamped per seed (`solution/trajectory.s<S>.npz`: the same rows with the
  seed's birth; `solve.sh` picks the seed's); rope-threading 04-08's references record no birth and serve every seed as
  they are; the standard oracle streams the same rows for every seed. Goal pictures (blanket-fold 01-05, knot-untying
  07/09) are rendered per seed (`tests/seeds/<S>/goal/`, copied over `environment/goal/` by `task_seed.py materialize`).
- `[metadata.seeds] oracle` lists seed 0 and the seeds whose Harbor oracle run passed (below): `[0, 1, 2]` on 30 bases,
  `[0]` on the other 69 with a reference.
- Seed instances: in the image and pinned by sha256 in `scripts/roboarc/assets.json`; the HF dataset revision that
  will hold them (and the new bases' instances) is a separate step, uploaded only with the owner's approval. Until then
  `images/roboarc/build.sh` builds from a local instance tree (`--instances`).

### Simulator source and tooling
- Simulator source synced from TopoGym's development line (`497a9e9d7f`, `622c7801c6`, `738231f5cb`, `3eed5d8b33`,
  `a849170272`) onto 1.1.0's source: SOURCE.json `cacc831d…` (449 files). The appearance axes of every family (their
  defaults are the released looks), the new families' code (bed_bolt, caliper_sorting, clock_assembling,
  battery_replacement, Cast Trinity), opt-in grasp rules (force closure, weak holds: used by battery_replacement only).
  Task tooling v21 (`rcb_topogym.py`: the new families registered; `nyx_direct.py`: a fixed MJCF entity's visuals drawn).
- fiber-bundling 07-10: the `lay` rule takes either red fiber (the scene has two: strands 0 and 8, 16 strands over an
  8-colour palette); the texts say "the root of one of the two red fibers". A trajectory that met the rule with strand
  0 is graded as before; one that failed only because the other red fiber lay at the stated position now succeeds. The
  references reach their released end states under the new rule. Re-grading earlier results on these four bases needs
  a replay of the trajectory with the 1.2.0 grader: `scripts/roboarc/regrade.py` does not cover this rule (it re-grades
  the 2026-10-14 rule from the record, and the record holds only strand 0's angle); replay the trial's trajectory with
  the 1.2.0 task's `tests/grade.py` in its verifier image (as `regrade.py --replay` does for its rule), or rerun the
  trial.

### Texts that stay right in every seed
- interlocked-claws 06-10: the second piece is "a lock" (it was "a teal lock"; its colour changes with the seed).
- interlocked-claws 01/02 and cast-keyhole 01-06: the pieces are also named by their object names (`piece_0` /
  `piece_1`, `keyhole_silver` / `keyhole_gold`), so no text identifies an object by colour alone.
- Every colour word the texts name, per seed: `docs/simulator/seeds/colour_table.md`.

### New bases (experimental, untiered)
bed-bolt-01..05, caliper-sorting-01, clock-assembling-01 (dynamic: the wound mainspring drives the train),
battery-replacement-01 and interlocked-claws-13 (Cast Trinity): packages, records (`release.json`
`experimental_bases`), instances (with their seeds) in the image and references, rendered only by
`scripts/roboarc/mk_tasks.py --experimental --out <dir>`, never into `tasks/`, until the owner has reviewed their demos.
Their texts state the grasping rules the grader applies, more than the tooling's own grasp sentence reports (an open
item for that review). Evidence in 1.2.0's image: Harbor oracle 1 / nop 0, both modes, seeds 0-2 (108 / 108); the
reference replays on Intel equal Zen 4 (9 / 9); reference timings at most 14.1 % of the 3 h budget; the texts' step
times follow measured values where they were more than 25 % off.

### Evidence
- Seed instances (220 = 110 bases x seeds 1, 2), each: `instance_assets/` byte-identical to seed 0's, the physical
  instance fields and the built MuJoCo model's physics equal, staged against its own binding + a strict Session, the
  birth's tick record equal to seed 0's in every part but `task`, the re-stamped reference's birth = the seed's (220 /
  220; rope-threading 04-08 unstamped, buzz-wire-3d-05 / fiber-bundling-05 without a reference). Derived again in this
  release's image: births and `instance_assets/` equal to the first derivation's except where 1.2.0's later source
  changed the looks (battery, caliper, Trinity, rope-threading 01-08) or the text (fiber 07-10).
- End-state identity, 39 bases (one per family, interlocked-claws 06-10, fiber-bundling 07-10, rope-threading 01-08
  and the 9 new bases): each seed's reference, replayed strictly in two fresh processes, succeeds and ends with the
  same world, robot, coverage, targets and state as seed 0 (39 / 39 x seeds 0, 1, 2; only the task's instance id
  differs; Intel CPU, the blanket bases on the GPU).
- Released seed 0, all 101 bases: strict fresh replays in the 1.2.0 image reach 1.1.x's end states from the frozen
  births (101 / 101; also on Intel and the blanket bases on the GPU); the render inputs are identical to 1.1.0's
  (101 / 101: every probe built one Nyx scene from the same bytes); frames, one base per family, within the path
  tracer's own run-to-run spread (14 / 14), state records equal. Blanket (14 bases): births, ends and render inputs
  (cloth mesh, material map, texture) unchanged; the cloth's identity rule unchanged at seed 0.
- Harbor, rendering off, oracle 1 / nop 0, both modes, on materialized seed copies: the required set (one base per
  family + every touched base: interlocked-claws 06-10, fiber-bundling 07-10, rope-threading 01-08, seeds 1 and 2; the
  three re-made new bases, seeds 0-2): 276 / 276; the rest (colour-text bases at seed 0, bed-bolt 01-05 and
  clock-assembling 01 at seeds 0-2): 104 / 104.
- The image: a rebuild of the recipe with the same instances reproduces every file's content (only build-time file
  times in `.pyc` headers differ).

### Repository (since 1.1.0)
- The usage gateway ships with the repository: `scripts/usage_gateway/` (from the shared benchmark repository,
  unchanged; PROVENANCE.md) and `.env.example`. Usage recording stays optional and off by default; turn it on with
  `USAGE_GATEWAY=scripts/usage_gateway` (docs/RUNNING.md "Usage recording"). Its tests and a check that it takes
  every option the runner passes run in `ci_checks/check-unit-tests.sh`.
- `scripts/set_agent_egress.py` runs its command line only as a script, so importing it (to read `HOSTS`) runs
  nothing; as a script it behaves as before.
- `tools/stress/corpus/SHA256SUMS`: the O2 reproducer's line matches its file again.
- Everything the CI workflow uses is pinned: the runner image `ubuntu-24.04`, `actions/checkout` and
  `actions/setup-python` by commit (v4.4.0, v5.6.0), Python 3.12.15, and the jobs' Python packages by version and
  sha256 (`ci_checks/requirements-ci.lock`, installed with `--no-deps --require-hashes`).
- PROTOCOL.md: image references by digest is a rule (§1) and one of RoboARC's changes from the shared protocol (the
  tasks already comply). README / RUNNING: arm64 hosts are not supported.
- Trials and seeds (owner decisions for 1.2.0; PROTOCOL.md "Changes from the shared protocol", RUNNING.md "Seeds",
  "Results"): the batch runner gives the K trials of a task distinct seeds without replacement, in an order fixed by the
  batch name and the task's base name (`scripts/task_seed.py assign`; the same seeds in both modes), runs each trial as a
  Harbor job of its own from a copy made for its seed (`scripts/task_seed.py materialize`, back from 1.0.0 with
  `assign`, the verifier's `tests/` in standard mode and per-seed goal images), and records the trial and its seeds in
  `<job>/seed.json`. Results are one folder per trial, `<task>/<run id>/t<k>/` (`config.json` gains `trial`, `trials`,
  `trial_dir`, `copy_checksum`; `seed_source` is `runner`), with an index (`trials.json`, `README.md`) per task and run
  id. A single-seed task run with K > 1 is no longer over the retry cap: the retry rule (H.5) counts attempts per trial.
  The seed of a trial was recorded as the default before (the runner's task copy was gone when results were written);
  a trial's task version is now the committed DIGEST the runner copied (Harbor's checksum of the copy, which differs
  from it whenever the copy does, is kept as `copy_checksum`). `scripts/roboarc/regrade.py --replay` grades a trial with its seed's
  verifier files. New check `ci_checks/check-seed-files.py`. Every task still allows one seed: no task changes.

### Credit
The appearance seeds, the derive tooling, the new families and their references are Jianwen's work (TopoGym), ported
onto 1.1.x and validated for this release.

## 1.1.1 (2026-10-21)

RoboARC's four images moved to the `mll-lab-nu` organization, with the same bytes and digests:
`ghcr.io/mll-lab-nu/roboarc:1.1.0`, `roboarc-ikea:1.1.0`, `roboarc-ikea-sim:1.1.0` and `roboarc-monitor:0.1.0`
(`images/pins.json`; before: `ghcr.io/embodied-first-exam/<name>`). The packages are private and inherit the
repository's access, so a collaborator with read access to the repository can pull them (`docker login ghcr.io` with a
token that has `read:packages`). Every task directory is re-rendered: its Dockerfiles, compose files and README name the
new references, its Change Log has a 1.1.1 entry ("image references moved to ghcr.io/mll-lab-nu (same images, same
digests)"), and the puzzles' Release line names the task set 1.1.1; each task gets a new `DIGEST`. The runtime bindings
keep the image ids and the names the images were bound under, and the IKEA sim recipe keeps its `FROM` (an input of
the sim image). `ci_checks/check-image-pins.py` allows `ghcr.io/mll-lab-nu/` only (a task naming the former registry
fails). No instruction, grader, tooling, binding, instance, reference, image or archive changed: results graded under
1.1.0 stand. The repository changes listed under 1.2.0's "Repository" above change no task.

## 1.1.0 (2026-10-20)

The puzzle image without Genesis: `ghcr.io/embodied-first-exam/roboarc:1.1.0` (8.7 GB; 1.0.0 was 12.2 GB), and the IKEA
images rebuilt from fully pinned inputs with only what the tasks run: `roboarc-ikea:1.1.0` (1.05 GB; 1.31 GB) and
`roboarc-ikea-sim:1.1.0` (7.67 GB; 9.08 GB). The task set's version is 1.1.0. Every task binds its new image; the
instructions, the graders' logic, the frozen instances and scenes and the references are unchanged (except
blanket-flatten-06-i01's standard reference, under Records). Results graded under 1.0.x stand: every reference replays
in the new images to the end state it reaches in 1.0.0.
Details: [docs/history/GENESIS_REMOVAL.md](docs/history/GENESIS_REMOVAL.md), `ikea/images/ikea-mujoco/README.md`.

### The image
- Built from the Python base image (`source/images/roboarc/Dockerfile`) instead of the RoboTangle 0.2.22 (Genesis)
  image. Everything is pinned: the base image by digest, the Debian packages by version from snapshot.debian.org, the
  wheels by sha256 (`source/images/roboarc/runtime.lock`, `vbd-requirements.txt`).
- The same files as 1.0.0 for everything it keeps: every library the graded runtime loads (the Python interpreter,
  glibc / libm, libstdc++, numpy's and scipy's OpenBLAS, MuJoCo 3.14.0 with the EPA patch, Warp, Newton, torch and its
  CUDA libraries, gs-nyx, the graphics loaders, ffmpeg) is byte-identical, and so is every file of the 49 Python
  packages it keeps.
- Removed as unused (traced on every family, both modes, rendering on, and on the agent's API): genesis-world, libuipc
  (pyuipc, both builds), quadrants, gs-madrona, gs-nyx-plugin, torchvision, triton, opencv, vtk, matplotlib,
  scikit-image, numba and the other packages only these needed; build-essential, git, vulkan-tools; the uv download
  cache the 0.2.22 image kept (9.3 GB); documentation pages and apt/dpkg logs. curl, ca-certificates and ripgrep stay
  for the agent harness.

### The simulator, the tooling and the bindings
- `source/topogym/`: the Genesis world, scene builder, physics backends (FEM, IPC, the Genesis grasp models), visuals
  and renderer are removed (SOURCE.json: 443 -> 426 files); the runtime identity no longer names genesis-world, pyuipc
  or a libuipc build. The ground plane's checker texture, which the renderer read from the genesis package, ships with
  the image; a Panda + Robotiq robot given in genesis-world's layout resolves to the vendored Panda (the same bytes).
- Tooling v20 (`rcb_topogym.py`): no libuipc build choice by GPU, no Genesis kernel replacements; `nyx_direct.py` finds
  the checker texture in the image.
- All 202 puzzle task dirs: new `runtime_binding.json`, tooling copies, image reference, a 1.1.0 Change Log entry and
  DIGEST; the compose files cite the new recipe.

### Records
- `records/freeze_manifests/freeze_manifest.v3.json`: knot-untying-06-i01 and knot-untying-09-i01 expect the end hashes
  `0d25b901f83db04e` and `3dd3136502cb9e13`. The old values came from a pre-release dry run; the released references end on
  the new ones in images 0.2.0 (Harbor oracle runs), 1.0.0 and 1.1.0, with equal state hashes at every 250-row checkpoint.
  The references themselves are unchanged.
- blanket-flatten-06-i01's standard reference (`scripts/roboarc/oracles/blanket-flatten-06-i01/standard.npz`): its route
  re-recorded within the standard joint-speed rule (7577 rows, end `f6f8062d4270bdcb`; Harbor standard oracle 1 / nop
  0), replacing the privileged rows whose 3 faster rows the service stretched (oracle 0); the privileged reference is
  unchanged.
  On Intel (Xeon Gold 6338, A100 GPU) two strict fresh-process replays and the standard service's rules end on
  `f6f8062d4270bdcb` (every row executed as it is) and the privileged reference on `cc8615b8bc4a0efa`, as on AMD Zen 4.
  Stress, as for blanket-fold-05-i01's new reference (criteria v2.13b, the CPU twin of the GPU cloth step): 371
  episodes (random, directed, cem, periodic; 93,385 controls), 0 breakdown, 0 violent, 0 pressed-pile warning; the
  reference under the observer on the GPU step and nop 400 are CLEAN.
- Harbor oracle records (`scripts/roboarc/release.json` evidence, `tools/release/release_evidence.json`): the 24 that
  said `not_run`, no score claimed (privileged on 11 bases, standard on 13: blanket-flatten-04/05/07/08/09-i01,
  knot-untying-06/07/09-i01, orbits-02-i01 and slotted-post-ring-05-i01 in both modes, blanket-flatten-06-i01
  privileged, blanket-fold-02-i01 and slotted-post-ring-03/04-i01 standard) now hold their trials of the 1.1.0
  validation round on the release image: oracle 1 at the recorded end state (privileged: both replays agree;
  standard: live = replay) and nop 0. The READMEs' Reference solution and Change Log say so; no reference, instance
  or grader changed.

### Repository
- Removed (they only served porting from Genesis; RoboARC 1.0.0 at main 153eb8cf01 keeps them): the Genesis-era task
  texts and tools (`tools/release/templates/`, `package_text/`, `package_text_tool.py`, `scripts/roboarc/port_package.py`),
  the Genesis `-i00` references (`tools/references/oracles_genesis/`), the Genesis release generator
  (`records/genesis_release/`), the Genesis base image's recipe and libuipc build files, and the Genesis kinematics
  comparison test. `release_rebuild.py` gives a new base its template's final packages.

### The IKEA images
- `roboarc-ikea:1.1.0` and `roboarc-ikea-sim:1.1.0` (`ikea/images/ikea-mujoco/`): every input is pinned, the Python base
  image by digest (as before), the Debian packages at the 1.0.0 images' versions from the Debian snapshot archive of
  their build day, the wheels by sha256 (`requirements.lock`, `sim/nyx-requirements.txt`; until 1.0.0 the base resolved
  mujoco's dependencies at build time and the sidecar installed its Debian and pip packages unpinned). The sidecar is
  `FROM` the base by digest. `build.sh` builds both reproducibly (pinned BuildKit, SOURCE_DATE_EPOCH, rewritten
  timestamps) as OCI archives with fixed manifest digests: base `sha256:2c56d9226aaf…` (id `9f9cbab8aa74`), sidecar
  `sha256:f22a1e9042c7…` (id `cbd929360257`).
- The same files as 1.0.0 for everything the graded runtime loads (traced: privileged grading with its replay video,
  the standard sim service with Nyx, the agent's tooling): MuJoCo with the EPA patch, numpy, the scenes, OSMesa and
  ffmpeg, Nyx, torch and its CUDA libraries, the graphics loaders (the dynamic linker's cache lists the added and
  removed libraries; every library resolves to the same file); every Debian package keeps its version. The verifier's
  replay videos are byte-identical to 1.0.0's.
- Removed as unused: mujoco's absl-py, etils, fsspec, websockets and zipp (its viewer, studio and test code); in the
  Nyx venv triton, sympy, mpmath, jinja2, markupsafe, filelock, fsspec, cuda-toolkit, cuda-pathfinder, absl-py, etils,
  glfw and pip; curl in the sidecar; bytecode caches; the second copy of the scenes the 1.0.0 base carried (a `chmod`
  layer).
- The verifier's replay video libraries (OSMesa, imageio with ffmpeg, Pillow) are in the base image: the privileged
  tasks' verifier image installs nothing when it is built (until 1.0.x it installed them from the live Debian archive
  and PyPI at that moment). The privileged agent's container has them too. Each IKEA task's runtime binding names the
  new base image id, and its README a 1.1.0 Change Log entry.
- Push the two images from their OCI archives byte for byte (`images/push_all.sh`), so the registry's digests are
  the ones above; a `docker push` re-compresses the layers.

### Every image by digest; the client lock
- Every `FROM` and compose `image:` of the 226 task dirs names its image by manifest digest, from `images/pins.json`:
  `roboarc:1.1.0@sha256:85ea90032a55…` (the validated image `dadc2d710ad2`, exported without a rebuild into a
  deterministic OCI archive by `images/archive.sh`: its config digest is the image id and its layers' diff ids are the
  image's), `roboarc-ikea:1.1.0@sha256:2c56d9226aaf…`, `roboarc-ikea-sim:1.1.0@sha256:f22a1e9042c7…`, and as before
  `roboarc-monitor:0.1.0@sha256:ea1b0d4b8599…` and the client base. `ci_checks/check-image-pins.py` (CI job
  `image-pins`) fails a bare tag, a digest that is not a pin and another registry.
- `images/push_all.sh` pushes the four archives byte for byte (skopeo `--preserve-digests`; the archives checked
  against their pins before login, the registry's manifests and configs after), the IKEA base before its sidecar. It
  replaces `images/roboarc/push.sh` (a `docker push`, which re-compresses the layers) and `images/monitor/push.sh`.
- The standard agent's client (`scripts/eai_standard/client-requirements.txt`, copied into every standard task) is a
  hashed lock of the same versions, installed with `--no-deps --require-hashes`; a built client image has the same
  `pip freeze` and the same site-packages files as before.
- `release_rebuild.py` archives the release image and pins its digest before rendering; `harbor_check.py --tasks` runs
  a pinned task on the local image of the same id before the push.
- Each task's README has a 2026-10-21 Change Log entry; no instruction, grader, tooling, binding, instance or reference
  changed.

### Validation of 1.1.0
- The 99 references, replayed strictly in fresh processes as the privileged verifier does (two replays, rendering
  off): all succeed, both replays agree, every birth is the frozen instance's; 97 end on the freeze manifest's hash, and
  knot-untying-06-i01 and -09-i01 on the hashes 1.0.0 and the released 0.2.0 image reach (the manifest holds an earlier
  dry run's values for these two). The two bases without a reference build their frozen birth.
- Every library the traced runtime loads and every file of the kept Python packages is byte-identical to 1.0.0's; two
  builds of the recipe on different hosts give the same file contents.
- The agent API's Nyx frames, depth and camera data equal 1.0.0's (five families); the standard service with Nyx
  rendering reaches the recorded end states; `tests/api`: 29 simulator tests and 17 bit-identity tests pass in 1.1.0.
- `release_rebuild.py --onto 1.1.0` with its own instances builds the same image id and changes no file; the
  generators' `--check`, the invariants, unit tests and advisories pass.
- IKEA: the 12 references replayed in fresh processes in the 1.1.0 verifier image exactly as the privileged verifier
  does (two replays and the replay video): all succeed, both replays agree, and every end and settled state equals
  1.0.0's (and the recorded hashes). Harbor oracle 1 / nop 0 on all 12 scenes in both modes (48 trials) with the 1.1.0
  images; every oracle ends on its recorded state. Every file the traced runtime loads is byte-identical to 1.0.0's;
  three builds (two from fresh clones) give byte-identical archives. Nyx's RGB frames of a moving scene differ from
  run to run on one host in 1.0.0 as in 1.1.0 (hundreds to tens of thousands of pixels, by up to 25 levels, between two
  runs of the same image; the frames at reset are identical), so they are compared by that spread; depth,
  proprioception and every physics state are identical. `tests/api` in `roboarc-ikea-sim:1.1.0` with a GPU: 29 passed
  (the IKEA reference bit-identity tests for lack and svallet among them), 19 skipped (puzzles).

## 1.0.1 (2026-10-19)

The task set's version 1.0.1 (`scripts/roboarc/release.json` `version`; the README Release line of every puzzle task).
The images stay `ghcr.io/embodied-first-exam/roboarc:1.0.0`, `roboarc-ikea:1.0.0` and `roboarc-ikea-sim:1.0.0` (the same
ids); the resource monitor's image is now RoboARC's own build of the same files (below). The graders' and the sim
services' logic, the pinned tooling (`rcb_topogym.py`, `render_bridge.py`,
`nyx_direct.py`, `rcb_ikea.py`), the runtime bindings, the frozen instances, the IKEA scenes and the references are
unchanged. Every task has a new DIGEST and a Change Log entry in its README.

### The task texts (all 226 tasks)
A re-render from the cleaned generator inputs; text only for the puzzles:
- **Citations of this repository's paths**: `docs/PROTOCOL.md`, `docs/STANDARD_MODE.md`, `docs/TAXONOMY.md`,
  `docs/simulator/TOPOGYM_PROTOCOL.md`, `tools/release/release_rebuild.py`, `images/roboarc/` and the base image's
  recipe, instead of the development tree's `docs/rcb/...`, `images/topogym/`, `scripts/robotangle/...`,
  `scripts/task_dynamics.toml`. No task text cites a missing path any more except inside files whose bytes stay as
  released (the pinned tooling, the shared eai-standard and monitor layers, IKEA's `rcb_ikea.py` / `ikea_nyx.py`);
  [docs/rcb/README.md](docs/rcb/README.md) maps those.
- **No internal notes**: the resources' `measured` strings, the no-oracle bases' unfilled "PR #N", the Change Log
  entries citing internal documents, the graders', sim service's and oracle player's docstrings naming the earlier
  repository's RoboWits files, issue numbers and GPU-physics claims, IKEA's owner / board / pull-request notes, an
  internal runner script and a branch name.
- **`[task].name` = `roboarc/<task dir>`** (Harbor's `org/name`; it was `robot-coding-bench/<task dir>`); the Harbor
  dataset name is `roboarc/roboarc` (`scripts/harbor/dataset_manifest.py`). `harbor run -p tasks -i <dir>` is
  unchanged (it matches the directory name). The Python API accepts both forms.
- For the puzzles, `instruction.md` is unchanged and every changed `.py` file has the same docstring-free AST as in
  1.0.0 (the audit below); puzzle results graded under 1.0.0 stand.
- **IKEA's verification prose** names machine classes ("a Ryzen 9 7950X workstation", "a dedicated Xeon Gold 6230R
  core", "one process at a time") where the clean-up had left host labels and broken phrases ("host B", "a dedicated
  the cluster Xeon ..."): the rendered prose fields of `ikea/scripts/ikea_mujoco/evidence.json` and
  `standard_evidence.json`; every number, hash, image id, date and run name is unchanged (21 IKEA READMEs).

### IKEA MuJoCo on the repository's task layer, under `tasks/`
The 24 IKEA task dirs moved from `ikea/tasks/` to `tasks/` (same directory names) and are rendered with the
repository's task layer (`scripts/task_conventions.py`), like the puzzles: the instruction's shared closing sections
(`## External material`, `## Resource limits`, `## Time limit`), the resource-monitor sidecar, declared resources,
limits, caps (`scripts/task_resources.toml`, rules `^roboarc-ikea-`: privileged CPU only, the standard sim one GPU; the
agent's container 4 CPUs / 10 GB, the standard sim 4 CPUs / 8 GB; the privileged verifier keeps 4 CPUs / 8 GB for its
two replays and the video replay) and seeds, a `DIGEST`, and the one egress allowlist (`scripts/set_agent_egress.py`:
the model APIs and the package registries, which the instruction already named; IKEA's 1.0.0 allowlist had the model
APIs only). `ikea/scripts/task_conventions.py` and `ikea/scripts/set_agent_egress.py` are gone; IKEA's live-grading
2.4 pair is `scripts/eai_standard/v2_4_live/` (bytes unchanged), its own variant in `check-tooling-identity.sh`. Every
`ci_checks/` check covers the IKEA tasks. **IKEA results before and after 1.0.1 are not comparable**: the instruction
and the egress changed (the graders, scenes, references and images did not).

### The resource monitor's image: RoboARC's own
Every task's `rcb_monitor` service pinned `ghcr.io/mll-lab-nu/efe:monitor-0.1.0@sha256:f2c910f5…`, a package of
another group that RoboARC does not control. It now pins
`ghcr.io/embodied-first-exam/roboarc-monitor:0.1.0@sha256:ea1b0d4b859998d6a47e2cb7a72c3a9ae85870b32c75de1bc1448e387da07093`
(`scripts/rcb_monitor/inject.py` `IMAGE`), built from `images/monitor/`:
- **The same image contents**: the base is pinned by digest (`python:3.12-slim-bookworm`, Python 3.12.15), and the
  five layers have the earlier image's diff ids (the filesystems are equal file for file); the environment, working
  directory and command are the same; the config adds three OCI labels (title, version, source).
- **Reproducible**: `images/monitor/build.sh` builds an OCI archive with a pinned BuildKit, no attestations and a fixed
  `SOURCE_DATE_EPOCH`; two builds are byte-identical, so the digest the tasks pin is known before the push.
  `images/monitor/push.sh` copies that archive unchanged (skopeo `--preserve-digests`) and checks the registry's
  digest. [images/monitor/README.md](images/monitor/README.md).
- The tasks change only in that `image:` line, their 1.0.1 Change Log entry and DIGEST. `check-protocol.py` accepts
  base images from `ghcr.io/embodied-first-exam/` only. The Genesis base image (a build-time input of
  `roboarc:1.0.0`, `source/images/robotangle/Dockerfile.mujoco`) stays in the `efe` package: no task runs it.

### Usage recording is optional; licence
Usage recording is optional: an official run no longer needs the usage gateway (PROTOCOL.md 'Changes from the shared
protocol', H.6). The runner's `GATEWAY=1` is replaced by `USAGE_GATEWAY=<dir>`; results record `usage_gateway`,
`usage_source`, `usage_recorded`, `token_budget`. Licence: MIT (LICENSE, LICENSES/THIRD_PARTY.md).

### The raw simulator interface: the Python package `roboarc`
`roboarc.make(task_id)` returns a Gymnasium-compatible environment for every task (the 101 puzzle bases and the 12 IKEA
tasks) over the graded tooling itself (`rcb_topogym.Session`, `rcb_ikea.Session`): the frozen instance at `reset()`,
one control step of the tooling per `step()`, the tasks' own limits (clamp, never fail) and success predicates, Nyx or
MuJoCo-raster camera images, proprioception in the standard mode's robot frame, state snapshots, and the privileged
verifier's replay. Design, conventions and test results: [docs/API.md](docs/API.md); examples in `examples/`, tests in
`tests/api/` (the cheap ones run in CI, job `api-tests`). Each tested reference stepped through it ends on its recorded state
hash (one base per puzzle family and two IKEA scenes; with Nyx rendering on as well). The package changes no task,
image or graded file.

### Repository clean-up
The repository now holds RoboARC only and is laid out for its two audiences: `source/` is the image build context,
development tools are in `tools/`, records in `records/`, the simulator's internals in `docs/simulator/`, the Harbor
runner in `scripts/harbor/`; new `docs/RUNNING.md` (a Harbor guide), `docs/DEVELOPING.md`, one public page per family
in `docs/families/`, and a rewritten README. The generator's packages are the shipped texts (`mk_tasks.py` halved); the
asset tools check every instance source against its pin; CI runs on `dev/**` too, checks the docs' paths and links, and
runs the simulator source hash. Leftovers of the shared repository (other benchmarks, the eai-standard 2.2 layer,
lab-side runner defaults, internal notes) and duplicate copies (references, tooling, image recipes) are gone; the
session oracles of the orbits and slotted-post-ring references are restored. The clean-up itself changed no task byte
(the re-render above did); the images rebuilt from the cleaned tree have the 1.0.0 ids. Path moves: [PROVENANCE.md](PROVENANCE.md#after-100-the-clean-up).

### Validation of 1.0.1
On a fresh clone (2026-10-20, a shared Ryzen 9 7950X / RTX 4090 host, load 25-60):
- Both generators' `--check` (0 differences), `run_invariants.sh` (226 tasks), `check-unit-tests.sh --all`,
  `run_all.sh` ("all checks passed"), `check-tooling-identity.sh` (226 task dirs, 26 shared files, the 2.4 layer in
  113 standard dirs), `check-doc-paths.py`, `tools/release/source_hash.py --check` (SOURCE.json `c9171f79…`), and every
  job of `.github/workflows/checks.yml` run literally in its own `python:3.12` container, `api-tests` included.
- A diff audit of every changed task file against 1.0.0: every changed `.py` file of the 202 puzzle and 24 IKEA task
  dirs (474 files) has the docstring-free AST it had in 1.0.0; the puzzle `task.toml`s differ only in `measured`,
  `oracle_evidence` and `[task].name`, their shell scripts, Dockerfiles and compose files only in comments; no puzzle
  `instruction.md` changed; the IKEA runtime bindings and other JSON data are unchanged apart from text fields.
- Harbor oracle and nop, both modes: one base per puzzle family (14 bases, 56 trials, rendering off; the blanket bases
  with their CUDA cloth on the GPU) and all 12 IKEA scenes (48 trials, rendering on). All 104 scored as expected
  (oracle 1, nop 0, no infrastructure error), and every oracle ended on its recorded state hash (the puzzles' freeze
  manifest; IKEA's privileged end and settled hashes and standard live hashes). The IKEA privileged verifiers used at
  most 26 % of their timeouts under the new caps (stenerik 5036 s of 19440 s, kyrre 4365 s of 25440 s).
- The `roboarc` package's simulator tests against the re-rendered tasks: 29 passed in `roboarc:1.0.0`, 27 passed and 2
  skipped (puzzle-only) in `roboarc-ikea-sim:1.0.0`.
- The monitor image and the prose fix (2026-10-20, the same host): `images/monitor/build.sh` run four times (three times
  on one checkout, once on a fresh clone) gave byte-identical archives, manifest `sha256:ea1b0d4b…`; its filesystem equals
  the earlier image's file for file (paths, modes, owners, sizes, contents, link targets); `push.sh` into a local test
  registry stored exactly that digest. Every CI job in its own `python:3.12` container on a fresh clone passes,
  as do the gates listed above. Harbor oracle and nop, both modes, on interlocked-claws-03, blanket-fold-01 and IKEA
  lack with that image (12 trials): all scored as expected, every oracle ended on the same state hashes as above, and
  the monitor wrote the same record files and summary fields as before, no violation.

## 1.0.0 (2026-10-19)

The topological puzzles: 101 bases (202 task directories) in 14 families, image
`ghcr.io/embodied-first-exam/roboarc:1.0.0`. IKEA MuJoCo: 12 tasks (24 task directories), images
`ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` and `roboarc-ikea-sim:1.0.0`.

The tasks' graded content is byte-identical to the V3 release tree (dev/robotangle acfcec1089, image
`eai-roboarc:0.2.1`). Only the image references and the version labels changed (see "Repository" below). The image's
filesystem is identical to 0.2.1's (image id 41a17481d67a): the same layer digests, and new labels only.

### IKEA MuJoCo joins RoboARC
The IKEA MuJoCo furniture-assembly benchmark is part of RoboARC from 1.0.0: 12 tasks, 24 task directories under
`ikea/tasks/`, renamed into the RoboARC namespace: `ikea-<scene>-i00-<mode>` -> `roboarc-ikea-<scene>-i00-<mode>` (the map
is below). Scenes by Xiaoyu's team (Xiaoyu, Zhen Wu). Images
`ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` and `roboarc-ikea-sim:1.0.0`, which are the V3 images
`eai-ikea-mujoco:0.3.2` (5d5090ea1910) and `eai-ikea-mujoco-sim:0.4.2` (050cd45d70f6) under new names, with the same
ids. Its V3 content (development branch dev/robotangle_mujoco 5cb4114f10):
- **New tasks:** stenerik, burvik, uppsta and kyrre (the review was approved; kyrre ships with the native-cylinder
  exception below).
- **Franka Hand slew:** `GRIPPER_WIDTH_SPEED_M_S` in every copy of `rcb_ikea.py` (one sha256, 3a63a33b6803), so the hand's
  width command moves at its real speed. New references for the eight earlier tasks were built within the slew (lack,
  eket, ikea365, vesken, gladom, variera and stegesund; svallet's ViperX reference is unchanged).
- **Welded-part contact switch-off** in the scene runtime (`ikea_runtime.py`; kyrre's and uppsta's seated parts).
- **kyrre: MuJoCo's native cylinders**, the one exception to the 32-facet prism rule (owner, 2026-10-19). The stress
  A/B showed 1.15 % non-jerk breakdowns with native cylinders against 2.3 % with the prisms, and no breakdown involved
  a screw. The verifier limit is set by the slow-class replay (a Xeon Gold 6230R core, 8283 s): 25,440 s.
- **Nyx start-up retry** in the standard sidecar's render process (`ikea_nyx.py`, the same file in every standard task;
  sha256 036be49c38e6 in V3; 37a773230f41 in 1.0.0, after the rename changed a path in its docstring and
  a host name left a comment).
- **EPA-patched libmujoco.so**, the same 14-byte patch as the puzzles' image.
- Timing is gated on the i9 / Zen 4 host class. Slow-Xeon times are reported only: gladom, variera, stenerik, uppsta
  and kyrre are over the 40 % bar only there.

Task names (the generators' `name()`; physics, references, hashes and grading unchanged):

| until 1.0.0 | 1.0.0 |
|---|---|
| `ikea-burvik-i00-privileged` | `roboarc-ikea-burvik-i00-privileged` |
| `ikea-burvik-i00-standard` | `roboarc-ikea-burvik-i00-standard` |
| `ikea-eket-i00-privileged` | `roboarc-ikea-eket-i00-privileged` |
| `ikea-eket-i00-standard` | `roboarc-ikea-eket-i00-standard` |
| `ikea-gladom-i00-privileged` | `roboarc-ikea-gladom-i00-privileged` |
| `ikea-gladom-i00-standard` | `roboarc-ikea-gladom-i00-standard` |
| `ikea-ikea365-i00-privileged` | `roboarc-ikea-ikea365-i00-privileged` |
| `ikea-ikea365-i00-standard` | `roboarc-ikea-ikea365-i00-standard` |
| `ikea-kyrre-i00-privileged` | `roboarc-ikea-kyrre-i00-privileged` |
| `ikea-kyrre-i00-standard` | `roboarc-ikea-kyrre-i00-standard` |
| `ikea-lack-i00-privileged` | `roboarc-ikea-lack-i00-privileged` |
| `ikea-lack-i00-standard` | `roboarc-ikea-lack-i00-standard` |
| `ikea-stegesund-i00-privileged` | `roboarc-ikea-stegesund-i00-privileged` |
| `ikea-stegesund-i00-standard` | `roboarc-ikea-stegesund-i00-standard` |
| `ikea-stenerik-i00-privileged` | `roboarc-ikea-stenerik-i00-privileged` |
| `ikea-stenerik-i00-standard` | `roboarc-ikea-stenerik-i00-standard` |
| `ikea-svallet-i00-privileged` | `roboarc-ikea-svallet-i00-privileged` |
| `ikea-svallet-i00-standard` | `roboarc-ikea-svallet-i00-standard` |
| `ikea-uppsta-i00-privileged` | `roboarc-ikea-uppsta-i00-privileged` |
| `ikea-uppsta-i00-standard` | `roboarc-ikea-uppsta-i00-standard` |
| `ikea-variera-i00-privileged` | `roboarc-ikea-variera-i00-privileged` |
| `ikea-variera-i00-standard` | `roboarc-ikea-variera-i00-standard` |
| `ikea-vesken-i00-privileged` | `roboarc-ikea-vesken-i00-privileged` |
| `ikea-vesken-i00-standard` | `roboarc-ikea-vesken-i00-standard` |

Its earlier releases were inside the shared repository: 8 tasks (16 dirs) with images `eai-ikea-mujoco:0.3.0` /
`-sim:0.4.0` (2026-10-17).

### New bases (reviewed and approved by the benchmark owner, 2026-10-18)
- **orbits-02-i01** (orbits, easier): runs on a 0.3125 ms MuJoCo step (solver_substeps 8), see
  [docs/simulator/families/orbits.md](docs/simulator/families/orbits.md). Reference end state b9c68d6502105df4.
- **knot-untying-07-i01** (knot_untying rd, easier): a re-timed route. The reference replays in 28.1 % of the 3 h budget
  (3037 s on a Ryzen 9 7950X), under the knot family's 30 % bar. End state 8ae062ea4c733dbc.
- **blanket-flatten-09-i01** (blanket_flatten, hard): the Genesis release's hard tier, ported to the deterministic GPU
  cloth solver (`blanket_vbd_gpu`), route hard5. End state 6a9c4796da46cf47.

### Changed
- **blanket-fold-05-i01: a new reference** in both modes. It is the released route, re-recorded with the oracle rate
  limiter's guard-stop clamp, so every arm joint step stays within the standard service's joint-speed rule (max
  0.0390 rad). It has 10195 rows (was 17189, one of them over the rule), end state 4f1b6724fc0a6512, and standard rows
  = privileged rows. Timing exception (owner, 2026-10-18): <= 30 % of the 3 h budget (2930 s = 27.1 % on a Zen 4
  trial host). Stress: 1 breakdown in 371 episodes, of the pressed-pile class, counted as an agent-caused warning
  (~0.27 %, under the ~1 % ceiling).
- **MuJoCo 3.14.0 + the EPA horizon-overflow fix.** MuJoCo 3.14.0's EPA keeps the polytope horizon in fixed 24-entry
  arrays and overflows them on complex convex contacts (a SIGSEGV in `mj_narrowphase`). Upstream fixed this in 3.15
  (google-deepmind/mujoco 6d662aed). The image carries a 14-byte patch of the wheel's own `libmujoco.so.3.14.0` that
  backports only that fix (`source/images/robotangle/mujoco-3.14.0-epa/`; sha256 5e7623e3… -> 52e2a24e…). A source
  build is not bit-identical to the wheel, so it is not used. Gate: every released reference reaches its recorded end
  state on the patched library (Zen 4, and Intel for the CPU bases; stock = patched), and 1150 / 1150 stress episodes
  were identical. So 0.2.0's results stand.
- **Nyx start-up retry** (`nyx_direct.py` sha256 52bb406f…, in every task's tooling copy). A renderer build that fails,
  for example on a full shared GPU, is built again after 10 s and then 30 s (x 1-1.5), at most 3 attempts, and each
  failed attempt is logged (the standard service's `nyx_ready` event counts them). The half-built renderer is kept,
  never shut down. A crash inside the build still goes through crash recovery.

### Evidence
V3 build (2026-10-18, image 0.2.1):
- Harbor on the image, both modes, rendering off: orbits-02, knot-07, blanket-flatten-09 and blanket-fold-05. Every
  oracle scored 1 at its reference's end state (privileged deterministic and instance_exact; standard live = replay),
  and every nop scored 0.
- Strict fresh-process replays of one base per family plus the hold-only bases (16 / 16), all at their gate end hashes.

From this repository (2026-10-19, a fresh clone, image `roboarc:1.0.0` built by `images/roboarc/build.sh`):
- The image has the same two layer digests as 0.2.1. All 153,404 filesystem entries are equal (path, mode, owner,
  mtime, size, sha256).
- `mk_tasks.py --check` (202 match), `run_invariants.sh`, unit tests, check-video, check-resources and check-cutoffs
  all pass. `run_all.sh` passes too: the invariants hold, and the advisories report nothing beyond warnings.
- Harbor, oracle and nop in both modes, rendering off, on interlocked-claws-03, hedgehog-in-cage-01 and cast-coil-01:
  12 / 12 as expected (oracle 1 at the reference end, deterministic; nop 0).
- Strict fresh-process replays in the new image (both replays of each): one base per family plus the hold-only bases
  (16) and the four V3 items (orbits-02, knot-untying-07, blanket-flatten-09, blanket-fold-05). All 20 end at their
  recorded hashes, 15 + blanket-flatten-08 equal to the 0.2.1 spot replays, and the V3 items at b9c68d6502105df4,
  8ae062ea4c733dbc, 6a9c4796da46cf47 and 4f1b6724fc0a6512.
- IKEA: both generators' check modes give 0 differences. Both images rebuilt from `ikea/` have the V3 images' ids
  (5d5090ea1910, 050cd45d70f6). The unit tests pass in
  the image. Harbor oracle and nop on svallet and lack in both modes, before and after the rename: 8 / 8 each time
  (oracle 1, nop 0). Fresh replays of svallet's and lack's references from the renamed dirs give the 0.3.2 records'
  end and settled hashes.

Not yet in: the Intel replays of the 13 blanket bases on the GPU. They gate the public image push.

### Repository
- A standalone repository: the RoboARC tasks, generator, simulator source, image recipe, CI checks and the shared task
  layer they use (provenance: [PROVENANCE.md](PROVENANCE.md)).
- Image references: `ghcr.io/mll-lab-nu/eai-roboarc:0.2.1` -> `ghcr.io/embodied-first-exam/roboarc:1.0.0`. The
  monitor sidecar and the Genesis base image are referenced by their unchanged digests in the consolidated package
  `ghcr.io/mll-lab-nu/efe` (`monitor-0.1.0`, `robotangle-0.2.22`), because the old `eai-*` packages no longer exist.
- The task README's release line names RoboARC 1.0.0 and the simulator source commit.
- IKEA task dirs: only the image references changed (the Dockerfiles' `FROM`, the compose `image:` lines, the
  bindings' `reference`, the README rows). The host names in their evidence lines were replaced by host classes.

## 0.2.1 (2026-10-18, V3 build; not released before 1.0.0)

The V3 build on the development branches of the shared repository (image `eai-roboarc:0.2.1`, 41a17481d67a, HF assets
revision 9f34f894a454). Its content is released as 1.0.0 above.

## 0.2.0 (2026-10-17, "V2")

98 bases (196 task directories). Image `eai-roboarc:0.2.0`.
- New bases: knot-untying-06-i01 (rd very_easy) and knot-untying-09-i01 (rd medium; 27.4 % of the budget, under the
  knot family's 30 % bar), blanket-flatten-07-i01 (easy) and blanket-flatten-08-i01 (medium), both ported to
  `blanket_vbd_gpu`.
- Standard references within the standard service's joint-speed rule (0.04 rad per control step) for 11 bases.
- buzz-wire-3d-03 / 04: references where the release had none.
- knot-06 / 09 re-frozen without the lab start's provenance in instance.json. The physics is unchanged; the births
  changed.
- Reference meta hygiene: no host paths in the references' metadata.

## 0.1.0 (2026-10-14)

The MuJoCo release. 94 bases (188 task directories) replace the Genesis release of the same puzzles (RoboTangle,
`eai-robotangle` 0.2.22, whose results are not comparable). Image `eai-roboarc:0.1.0`, HF assets revision 88e0136e1b97.
- MuJoCo 3.14.0 on the CPU in float64 (blanket families: a deterministic CUDA cloth step), and numpy / OpenBLAS SIMD
  masking in every container that runs a session, so every CPU class builds the same frozen bytes.
- The 3 h agent budget in both modes. Standard mode on eai-standard/2.4, with no world coordinates in standard texts.
- Privileged grading replays need the strict goal within the agent's rows (the 2 s hold only confirms). Earlier
  privileged results can be re-graded with `scripts/roboarc/regrade.py`.
- Follow-ups on main: the image's own Harbor evidence and the lock rotor discs drawn where the physics has them.
