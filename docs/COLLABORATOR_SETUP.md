# Running RoboARC: setup for a collaborator

Run RoboARC from the branch `main` of `mll-lab-nu/RoboARC`; it has this note and the script that sends your results back (`scripts/send_results.sh`). You get read access to the repository as a collaborator. The four container images on GHCR (`ghcr.io/mll-lab-nu/...`) inherit the repository's access, so the same invitation covers them. This note covers what your machine needs, how to log in, a first check, a smoke test of the batch runner, the evaluation run itself, and how to send its results back. The repository's `README.md` and `docs/RUNNING.md` are the full reference.

## 1. Machine

- Linux x86_64 (not WSL, not arm64), Docker with the NVIDIA Container Toolkit, Harbor 0.23 or newer, Python 3.11 or newer.
- **NVIDIA driver 580 or newer** (RoboARC's stated requirement). The renderer (Nyx) runs from about driver 570; the
  blanket families' cloth physics is built for CUDA 13 and needs 580 or newer, or 570 with NVIDIA's `cuda-compat`
  libraries (docs/API.md).
- **A GPU with RT cores.** The renderer (Nyx) needs the Vulkan extension `VK_KHR_ray_query`:
  - **Has RT cores:** T4, A10/A10G, L4, L40/L40S, A40, RTX 20/30/40/50 series, RTX A-series / Ada / PRO Blackwell.
  - **Does not:** V100, A100, A30, H100, H200. There the renderer stops with an error naming the missing extension
    (`nyx_direct.UnsupportedDevice`): standard mode does not run, and privileged tasks are graded without their replay
    video while the agent gets that error instead of camera images. Such runs are unofficial (docs/RUNNING.md, "GPUs
    without RT cores").
- GPU memory: each task declares 2 GB; the measured peak is about 1.1 GB. Since 1.1.0 every task runs on MuJoCo, with no Genesis or IPC simulator, so any card on the list above runs every family.
- Check the GPU. All three names must be printed:
  ```bash
  vulkaninfo | grep -E 'VK_KHR_ray_query|VK_EXT_extended_dynamic_state3|VK_KHR_compute_shader_derivatives'
  ```
  `vulkaninfo` is in the package `vulkan-tools`.
- Disk: about 19 GB for the images, plus room for each trial's task image, logs and videos.

## 2. Access and login

1. Accept the invitation to `mll-lab-nu/RoboARC` (GitHub email or https://github.com/mll-lab-nu/RoboARC/invitations).
2. Create a GitHub **classic** token with the scope `read:packages`, then log in and check that you can pull:
   ```bash
   echo <token> | docker login ghcr.io -u <your GitHub username> --password-stdin
   docker pull ghcr.io/mll-lab-nu/roboarc:1.3.0      # access check: it must download
   ```
   Docker uses this login to pull the task images. Every task names its image by digest (`images/pins.json`):
   - `roboarc:1.3.0`, about 10 GB on disk, used by every puzzle task;
   - `roboarc-monitor:0.1.0`, about 0.13 GB on disk, used by every task;
   - `roboarc-ikea:1.1.0` and `roboarc-ikea-sim:1.1.0`, about 1.1 GB and 7.7 GB on disk, used only by the IKEA tasks.

   The download is smaller, because the layers are compressed.
3. Clone the branch (in an existing clone: `git fetch origin && git checkout main && git pull`), and pre-pull
   every image (each is checked against its pinned image id):
   ```bash
   git clone -b main git@github.com:mll-lab-nu/RoboARC.git
   cd RoboARC
   bash images/pull_all.sh
   python3 scripts/stamp_mtimes.py tasks   # once per checkout, so Harbor's build cache hits
   ```

## 3. First check: one task, both modes, no agent

Start with `roboarc-interlocked-claws-01-i01`. In each mode the reference solution must score 1, and doing nothing must
score 0:

```bash
for m in privileged standard; do
  harbor run -p tasks -i roboarc-interlocked-claws-01-i01-$m -a oracle -o "$PWD/jobs" --job-name oracle-claws-01-$m
  harbor run -p tasks -i roboarc-interlocked-claws-01-i01-$m -a nop    -o "$PWD/jobs" --job-name nop-claws-01-$m
done
```

- The score is in `jobs/<job>/<trial>/verifier/reward.txt`: 1 or 0.
- The privileged oracle run of this task takes about 6 minutes on an RTX 4090 (the image already pulled). The verifier
  replays the trajectory twice and renders a video, which lands in `verifier/replay.mp4`.
- The first run also pulls the image, so it takes longer.

Then one scene other than the default, made the way the batch runner makes it: `roboarc-interlocked-claws-03-i01` at
appearance seed 2 (its reference solution is validated at seeds 0, 1 and 2). The oracle must score 1 in both modes:

```bash
for m in privileged standard; do
  python3 scripts/task_seed.py materialize ./tasks/roboarc-interlocked-claws-03-i01-$m --seed 2 --out /tmp/claws03-s2-$m
  harbor run -p /tmp/claws03-s2-$m -i roboarc-interlocked-claws-03-i01-$m -a oracle -o "$PWD/jobs" --job-name oracle-claws-03-s2-$m
done
```

- If something fails, send us `jobs/<job>/` (or at least `result.json` and the `verifier/` logs) and the output of the `vulkaninfo` check above.

## 4. Smoke test: the batch runner on one task

Before the full run (section 5: `K=3`, both modes, every task you were given), run the batch runner once on one task
with your real agent, model and key, one trial per mode. It checks what section 3 cannot: the agent's credentials and
network, the runner's seeds, the summary and the result files.

```bash
AGENT=codex MODEL=openai/<model> EFFORT=high K=1 ENV_FILE=$PWD/.env \
  JOBS="$PWD/jobs/smoke" RESULTS=/path/outside/the/repo \
  scripts/harbor/run_agent_batch.sh smoke roboarc-interlocked-claws-01-i01
```

- Use the `AGENT`, `MODEL`, `EFFORT`, `ENV_FILE` and `RESULTS` the full run will use.
- The two modes run one after the other, each with at most the agent's 3 h. To run them at the same time, start the same
  command again in a second terminal (it adds a lane).
- It passed when:
  - `jobs/smoke/smoke.summary.md` has a row for `roboarc-interlocked-claws-01-i01-privileged` and one for `-standard`,
    each with a success of 0 or 1 and neither marked "not a result" (a 0 is fine here: the trial only has to give a
    result);
  - each job folder `jobs/smoke/smoke-<mode>-roboarc-interlocked-claws-01-i01/` has `seed.json` and
    `<trial>/verifier/reward.txt`;
  - `<RESULTS>/smoke/roboarc-interlocked-claws-01-i01-<mode>/<run id>/t1/` exists for both modes;
  - `jobs/smoke/smoke.log` shows no stopped lane (the runner stops a lane when the model account fails: out of credit,
    rate-limited, logged out).
- The seeds the full run gives a task's three trials:
  `python3 scripts/task_seed.py assign tasks/roboarc-interlocked-claws-01-i01-privileged --batch <batch name> --trials 3`
  prints three different seeds (0, 1 and 2, in an order fixed by the batch name).
- If it fails, send us `jobs/smoke/` (the runner redacts keys after each job) and the agent and provider you used.
- Then send the smoke batch back as section 6 describes, with the smoke's `JOBS` and `RESULTS`. It checks your token and
  the upload before the full run.
- Give the full run a batch name of its own: the smoke trials are not part of it.

## 5. The evaluation run

Start it once the smoke test (section 4) has passed. Run a coding agent over the tasks with the batch runner, three trials per task and mode (`K=3`). Every puzzle task has
three scenes (appearance seeds 0, 1 and 2: the same physics in other colours, materials and textures), and the runner
gives the three trials of a task different seeds, so they see three different scenes; an IKEA task has one scene. Credentials go in an env file; `.env.example` lists the variables, e.g. `OPENAI_API_KEY` for Codex or `ANTHROPIC_API_KEY` for Claude Code.

```bash
BASES=$(ls tasks | sed -nE 's/-privileged$//p')   # every task base: 123 (111 puzzle, 12 IKEA)
mkdir -p jobs
for lane in 1 2 3 4; do                             # 4 lanes: 4 trials at a time ("Lanes" below)
  AGENT=codex MODEL=openai/<model> EFFORT=high K=3 ENV_FILE=$PWD/.env \
    JOBS="$PWD/jobs/agents" RESULTS=/path/outside/the/repo \
    nohup scripts/harbor/run_agent_batch.sh <batch name> $BASES > "jobs/lane$lane.out" 2>&1 &
done
```

- **Which tasks and modes:** every task base in both modes (the default `MODES`), three trials each: 123 × 2 × 3 =
  738 trials. If we send you a shorter list, give those bases instead of `$BASES`. `MODES=privileged` or
  `MODES=standard` runs one mode only.
- **Lanes:** one runner runs one trial at a time; each copy of the same command started on the same `JOBS` adds a lane,
  and every trial is taken by exactly one lane. A running trial needs at least 4 CPU cores, 7 GB of memory and 2 GB of
  GPU memory (the tasks' `[metadata.resources]`) and may use up to 24 GB of memory (its limit). Start with no more lanes
  than cores / 4, memory in GB / 7 and GPU memory in GB / 2 allow, and add one only while the host's memory stays below
  about 80%.
- **Duration:** a trial gives the agent up to 3 h, plus setup and grading. If every trial used its whole budget, the
  738 trials would take about 738 × 3 h / lanes: about 23 days on 4 lanes, 12 on 8. `nohup` keeps the lanes running
  after you log out.
- **More than one host:** give each host its own part of the bases and the same batch name (a trial's seed depends
  only on the batch name, the task and the trial number), and send each host's batch back on its own (section 6).
- **Progress and summary:** progress is in `$JOBS/<batch>.log`, and the per-trial summary in `$JOBS/<batch>.summary.md`.
- **Restarting** (after a reboot, or a lane stopped by a model-account failure): start the lanes again with the same
  command. Finished trials are skipped. A trial that was cut off midway is skipped too: to run it again, move its job
  folder and `$JOBS/.claims/<job>` out of `$JOBS` first.
- **Budget:** each trial gives the agent up to 3 h; `roboarc-wire-splicing-01-i01` gives 6 h (an owner's exception;
  its results are ranked as a group of their own). Use one API key per batch, with a spending cap.
- **Live view:** `python3 scripts/harbor/live_board.py --jobs <JOBS>` serves a page on 127.0.0.1. Keep it local: it shows unredacted output.
- **When it has finished:** send it back (section 6).
- **Everything else:** `docs/RUNNING.md` has the agent options, network rules, re-grading and the result files.

## 6. Sending the results back

`scripts/send_results.sh` packs one batch and uploads it to our Hugging Face dataset
`RoboARC/RoboARC-outputs`. With `CREATE_PR=1`, as below, the upload is a pull request on the dataset,
which we merge. Your own Hugging Face account opens it and needs no access to the dataset.

**The dataset is public: anyone can read what you send.** The secret scan removes the keys in `ENV_FILE`. Everything
else in the job folders goes up as it is, including paths and host names in the logs.

1. Make a token of the type **Write** at https://huggingface.co/settings/tokens. It can do only what your account can
   do, and here that is opening the pull request.
2. Once per host:
   ```bash
   pip install -U huggingface_hub      # the hf CLI
   ( umask 077; cat > ~/.roboarc_hf_token )   # paste your token, press Enter, then Ctrl-D
   ```

The token stays in that file, not in `.env` and not in your shell history. The script reads it only for the upload, so
it never reaches a container or the bundle.

When the batch has finished, run this from the checkout, as the user that ran the batch, with the batch's own `JOBS`,
`RESULTS` and `ENV_FILE`:

```bash
JOBS="$PWD/jobs/agents" RESULTS=/path/outside/the/repo ENV_FILE=$PWD/.env CREATE_PR=1 \
  NOTES="<one line: crashes, OOM kills, outages, reruns; or none>" \
  scripts/send_results.sh <batch name> <your name, lowercase>
```

- **Secret scan first.** The batch's job folders are redacted against the keys in `ENV_FILE`, then scanned again. If
  any key is left, nothing is sent. If the batch ran with `CODEX_FORCE_AUTH_JSON=1`, set it and `CODEX_AUTH_JSON_PATH`
  as for the batch, so the scan also looks for that login.
- **What goes up**, to `<your name>/<batch>/<bundle id>/` in the dataset:
  ```
  MANIFEST.md     the checkout's commit, the host, its GPUs, the image digests, the runner's settings, the trial
                  counts (no credentials)
  index.json      per job: its archive, size and sha256, task, mode, trial, seed, and each trial's reward
  SHA256SUMS      every archive's sha256
  INVENTORY.tsv   every file in the archives, with its sha256
  batch.tar.gz    <batch>.log, <batch>.summary.md, the plan, <RESULTS>/<batch>/
  jobs/<job>.tar.gz   one per job folder (<batch>-<mode>-<task>-t<k>), with its seed record
  ```
  The bundle id is the UTC time, so a later delivery never overwrites an earlier one.
- **Dry run.** `DRY_RUN=1` does every step but the upload and needs no token. The bundle stays in
  `../send/<batch>__<name>__<bundle id>/` for you to look at.
- **Write access.** If we have given your account write access to the dataset, leave `CREATE_PR=1` out: the bundle
  then goes in directly, as one commit.
- **When it ends:** the script prints the batch, the bundle id and the sha256 of `SHA256SUMS`, and `hf` prints the pull
  request's link. Send us all four.
- **If the upload fails** (the token is refused, or the dataset is not found), nothing is lost. Tell us, and run the
  same command again once it is fixed. A run within the same minute as the last one stops: wait a minute.

## 7. Questions

Send setup questions and failing job folders to Jianwen Lyu (GitHub LUIGIVAMPER).
