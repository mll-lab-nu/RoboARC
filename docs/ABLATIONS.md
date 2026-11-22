# Ablation experiments: a step-by-step guide for a collaborator

This note is for a collaborator who has already run the RoboARC evaluation with the batch runner
([COLLABORATOR_SETUP.md](COLLABORATOR_SETUP.md)) and now runs five ablation experiments, E1 to E5, on the branch
`dev/qineng-ablations` of `mll-lab-nu/RoboARC`. Everything it needs is on that branch: the task variants (built on
your machine by a script), one runner per experiment, a check that every variant works before any agent runs, and the
hand-in of the results. Commands can be copied as they are; replace the parts in `<...>`.

What stays as in your evaluation run: the machine, the images, Harbor, the `.env` with your credentials, the usage
gateway, the Hugging Face token and the way results are packed and sent (`scripts/send_results.sh`). What is new: the
tasks are variants of the released ones, each experiment has fixed settings that its runner sets for you, the results go
to a dataset of their own,
[`RoboARC/RoboARC-ablation-outputs`](https://huggingface.co/datasets/RoboARC/RoboARC-ablation-outputs), and **step 1
(the variant check) must pass, and its report must reach us, before any agent trial**.

## 0. What to run

**Terms**

| term | meaning |
|---|---|
| base | a released task without its mode, e.g. `roboarc-bed-bolt-01-i01` |
| condition | what a variant changes about a task, e.g. "C10: the object poses are given". Each condition has a short token used in names (`q1c10`) |
| variant | a task directory built from a released one for one condition: `tasks/<base>-<token>-<mode>`, e.g. `tasks/roboarc-bed-bolt-01-i01-q1c10-standard`. Physics, grader, start state and seeds are the released task's unless the condition says otherwise |
| experiment | E1 to E5: a set of bases, conditions, modes and repeats, run with one agent at one setting |
| repeat | one run of every trial of an experiment. Every trial is at appearance seed 0; a repeat is a batch of its own (`ablation-e1-r1`, `ablation-e1-r2`, ...), so three repeats are three trials of each variant at seed 0 |
| check report | `jobs/ablations/checks/<E>.json`: the result of step 1 for one experiment. An experiment's runner starts only when its report says every variant passed |

**The five experiments**

- **E1**: the information-access ablation on more puzzles (the agent is given object poses, part shapes, or both).
- **E2**: the same ablation with a second agent.
- **E3**: the information-access ablation on deformable tasks (rope, cord, fibre, cloth).
- **E4**: relaxed clearances (a puzzle's gap is made wider, nothing else changes).
- **E5**: observation realism (the Standard mode without the contact-safety readings; then also with real
  depth-camera noise and calibration error).

**Conditions**

| condition | token | what the agent gets beyond the Standard task |
|---|---|---|
| C00 Standard rerun | `q1c00` | nothing (the service only logs a few more fields of each call) |
| C10 + object poses | `q1c10` | every response's `meta["objects"]`: the task objects' poses, in the robot frame |
| C0G + part geometry | `q1c0g` | `r.geometry()`: every part's collision shapes, each in its own frame (no poses) |
| C1G + poses + geometry | `q1c1g` | both of the above, the shapes in the robot frame |
| CS + state | `cstate` | every response's `meta["deformables"]`: the rope's, cord's or fibres' centreline points (every link) or the cloth's mesh vertices, in the robot frame; and C1G's poses and shapes of the rigid parts |
| clearance x2, x4 | `clr2x`, `clr4x` | nothing: the puzzle's clearance is 2 or 4 times the released one (both modes; section 7) |
| V1 | `obsv1` | less: no contact-safety readings (`meta["safety"]` is not sent) |
| V2 | `obsv2` | less: V1, plus RealSense-like depth noise and a fixed camera-calibration error (section 8) |

**The trials** (every trial at seed 0, the agent's time limit is each task's own, 3 h 10 min)

| experiment | agent | tasks | conditions | modes | repeats | trials |
|---|---|---|---|---|---|---|
| E1 | GPT-6 Astra in Codex, effort `high` | 13 | C00, C10, C0G, C1G | standard | 3 | 156 |
| E2 | Claude Opus 5.5 in Claude Code, effort `high` | 15 | C00, C10, C0G, C1G | standard | 2 | 120 |
| E3 | GPT-6 Astra in Codex, effort `high` | 18 | C00, CS | standard | 1 | 36 |
| E4 | GPT-6 Astra in Codex, effort `high` | 6 | clearance x2, x4 | privileged, standard | 1 | 24 |
| E5 | GPT-6 Astra in Codex, effort `high` | 15 | V1, V2 | standard | 1 | 30 |
| | | | | | **total** | **366** |

`python3 scripts/ablations/experiments.py table` prints this table from the code, and
`python3 scripts/ablations/experiments.py plan E1` lists every trial of E1 (batch, repeat, task, condition, mode, seed).

**Before the trials:** 202 variant checks (step 1; each variant once with the reference solution and once doing
nothing: 404 Harbor runs, no model calls) and 16 smoke trials (step 2; not part of the results).

**Duration.** A rigid-task trial takes a quarter of a GPU and a deformable-task trial (E3) a whole one, as in your
evaluation run. If every trial used its whole time limit, the 330 rigid trials would take about 261 GPU-hours and the 36
deformable ones about 114: about 375 GPU-hours, about 4 days on four GPUs. Trials that succeed end earlier.

**Order**

1. Get the branch and build the variants; check every variant (step 1). **Send us the check reports and wait for our
   OK.**
2. Smoke: one trial of each condition of each experiment (step 2), with a check after ten minutes. Send it back and
   wait for our OK.
3. Run E1 to E5 (step 3) and send each experiment back when it has finished (step 4).

## 1. Machine, branch and credentials

**Machine.** The same as for your evaluation run (COLLABORATOR_SETUP.md, section 1): Linux x86_64, Docker with the
NVIDIA Container Toolkit, NVIDIA driver 580 or newer, GPUs with RT cores, Python 3.11 or newer, Harbor 0.23 or newer
(`harbor --version`). Disk: the variants take about 0.5 GB; leave room for the trials' logs and videos as before.

**Images.** No new image: every variant runs on the images of `images/pins.json` that your evaluation run already
pulled (the puzzles' `ghcr.io/mll-lab-nu/roboarc:2.2.0` and `ghcr.io/mll-lab-nu/roboarc-monitor:0.1.0`), as the
released task it is made from.

**Branch.** Use a checkout of its own, so that your evaluation checkout stays as it is:

```bash
git clone -b dev/qineng-ablations git@github.com:mll-lab-nu/RoboARC.git RoboARC-ablations
cd RoboARC-ablations
git log --oneline -1                       # the commit we told you
bash images/pull_all.sh                     # every image: already there, each checked against its pinned id
python3 scripts/stamp_mtimes.py tasks       # after the clone and after every pull, as before
```

Expected: `pull_all.sh` ends without an error (the images are present, so it only checks them).

**Credentials.** Copy the `.env` of your evaluation run into this checkout (`cp <your evaluation checkout>/.env .env;
chmod 600 .env`). E1, E3, E4 and E5 run Codex and E2 runs Claude Code, so the file needs the settings of both, as in your
evaluation runs (COLLABORATOR_SETUP.md, section 5, the table of accounts): for example through your GitHub Copilot
proxy, `OPENAI_BASE_URL=http://127.0.0.1:<port>/v1` with `OPENAI_API_KEY` for Codex and
`ANTHROPIC_BASE_URL=http://127.0.0.1:<port>` with `ANTHROPIC_API_KEY` for Claude Code. How the runner reads them: Harbor
passes the file to the agent's container (`--env-file`), the usage gateway takes the provider's base URL from it, and
after every trial the runner removes these values from every saved file. A ChatGPT login for Codex
(`CODEX_FORCE_AUTH_JSON=1`) or a Claude subscription (`CLAUDE_FORCE_OAUTH=1` and `CLAUDE_CODE_OAUTH_TOKEN`) work the same
way; put those switches on the command line of every run, as in your evaluation run.

**Models.** The runners use the model names of your evaluation runs, `gpt-6-astra` (Codex) and `claude-opus-5.5`
(Claude Code), unless you set `MODEL`. A runner refuses a `MODEL` that would give a run id other than the experiment's
(`codex-gpt6_astra-high`, `claude_code-opus_5_5-high`), so `MODEL=openai/gpt-6-astra` is fine and a typo is caught
before any trial.

## 2. Step 1: build and check every variant (no agent, no model calls)

```bash
cd RoboARC-ablations
nohup bash-runners/ablations/check_variants.sh all > jobs/ablation-check.out 2>&1 &
```

What it does, for E1, E2, E3, E4 and E5 in turn:

1. builds the experiment's variants into `tasks/` (`scripts/ablations/make_variants.py`) and checks that each one is
   byte for byte the variant we validated (its `DIGEST` against `scripts/ablations/digests/`); E4's build re-freezes each
   relaxed scene in a container of the pinned `roboarc:2.2.0` image (Docker, CPU, a few minutes);
2. runs every variant through Harbor twice, from its own copy at seed 0 and pinned to one GPU, as the batch runner runs a
   trial: the reference solution (`-a oracle`) must score 1, with both the live and the replayed success and the task's
   own determinism check, and doing nothing (`-a nop`) must score 0 (`roboarc-fiber-bundling-05-i01` has no reference
   solution, so its variants get the nop run only); up to 4 rigid-task runs share a GPU, an E3 run has one alone
   (`GPUS=0,1` and `PER_GPU=<n>` change that);
3. for E3 and E4, also each condition's own check: for CS, `scripts/ablations/check_cstate.py` compares the state the
   service reports with the simulator's own state in the task's image (1 to 3 minutes per task; the cloth tasks need a
   GPU, GPU 0 unless you set `RCB_CHECK_GPU=<n>`); for E4, `check_clr2x.py` / `check_clr4x.py` measure the clearance in
   each variant's frozen scene, and after E4's runs `scripts/ablations/check_clearance_render.py` renders each relaxed
   privileged variant and its released task with the cameras on one GPU (at the start, middle and end of the reference
   solution) into `jobs/ablations/checks/e4_render/` (`report.md` and the pictures, which we look at);
4. writes the report `jobs/ablations/checks/<E>.json` and a table `jobs/ablations/checks/<E>.md`, after every run.

It takes about half a day on four GPUs (404 Harbor runs: a nop run takes a few minutes, a rigid-task oracle run 10 to 40
minutes, four of them sharing a GPU; E3's deformable runs take a GPU each and are the longest). Each experiment's report
is complete when its block ends, so you can send E1's report while the others still run.
Progress: `tail -f jobs/ablation-check.out`. When it has finished, its last lines are:

```
E1: ALL PASS (jobs/ablations/checks/E1.json)
E2: ALL PASS (jobs/ablations/checks/E2.json)
E3: ALL PASS (jobs/ablations/checks/E3.json)
E4: ALL PASS (jobs/ablations/checks/E4.json)
E5: ALL PASS (jobs/ablations/checks/E5.json)
```

`python3 scripts/ablations/check_variants.py show E1` prints a report as a table, one row per variant: the DIGEST as
locked, the oracle's and the nop's reward, the condition's own check, pass.

**Send us the reports, whether they passed or not**, and wait for our OK before step 2:

```bash
ENV_FILE=$PWD/.env CREATE_PR=1 bash-runners/ablations/send_checks.sh <your name, lowercase>
```

It uploads the reports, E4's rendered scenes and the Harbor job folders of any run that did not pass, to the ablations'
results dataset (section 6) under `<your name>/ablation-checks/<UTC time>/`, with your token in `~/.roboarc_hf_token`
after the same secret scan as `scripts/send_results.sh`, and prints the pull request's link: send it to us. `DRY_RUN=1`
builds the bundle in `../send/` without uploading.

**If a check fails.** Run `bash-runners/ablations/check_variants.sh <E>` once more: it redoes only what has not passed
(a failure of the machine, e.g. a GPU that dropped out, then passes). If it fails again, send the reports as above and do
not run that experiment; the others can go on once we have said so. If the build itself fails ("the variants could not
be built as validated"), the checkout is not the commit we named: check `git log --oneline -1` and send us
`jobs/ablation-build-<E>.txt`.

**The rule.** `bash-runners/ablations/run_<E>.sh` asks `scripts/ablations/check_variants.py gate <E>` before it starts and
refuses (exit 2, no trial) unless the report passed for every variant of the experiment, on this machine (with
rendering), for the variant directories exactly as they are now. Do not edit anything under `tasks/` or
`scripts/ablations/`; a variant built again from the same commit is the same and keeps its check.

## 3. Step 2: smoke, one trial of each condition

One trial of every variant of each experiment's first task, all at once, with exactly the settings of the experiment's
runner, in batches of their own under `jobs/smoke` (`ablation-smoke-e1` ... `ablation-smoke-e5`): 4 + 4 + 2 + 4 + 2 = 16
trials. They check what step 1 cannot: your credentials and network with both agents, the effort, and that the agents
reach the conditions' new interfaces. They are not part of the results.

```bash
RESULTS=<a directory outside the repository> bash-runners/ablations/smoke.sh E1
RESULTS=<the same directory> bash-runners/ablations/smoke.sh E2
RESULTS=<the same directory> bash-runners/ablations/smoke.sh E3
RESULTS=<the same directory> bash-runners/ablations/smoke.sh E4
RESULTS=<the same directory> bash-runners/ablations/smoke.sh E5
```

Each starts its lanes in the background and returns at once; a smoke whose step 1 did not pass is refused.

**After about ten minutes**, check each smoke:

```bash
for e in 1 2 3 4 5; do python3 scripts/ablations/smoke_check.py ablation-smoke-e$e; done
```

Expected, for each batch: a `runner:` line with your agent and model, `effort high`, the run id
(`codex-gpt6_astra-high`, for E2 `claude_code-opus_5_5-high`) and `usage gateway <checkout>/scripts/usage_gateway` (not
`off`); then one line per trial with model calls above 0, statuses nearly all `200`, `refused none`, tokens in and out
above 0, `effort given ['high']` (with Codex also `Codex ran with ['high']`); and a last line `OK`. The column "uses the
condition's interface" says `yes` once the agent's own code has read the new field (`meta["objects"]`, `r.geometry()`,
the deformable state); `not yet` is fine at ten minutes.

**If a batch says NOT OK**, stop that smoke: `ps -ef | grep "ablations/lane.sh"` and `kill <PID>` for each of its lanes
(by number), then `docker ps --format '{{.Names}}' | grep '^roboarc-'` and `docker rm -f <full name>` for each of its
trial containers, and send us the output and `jobs/smoke/<batch>.log`.

**When all smoke trials have ended** (every `jobs/ablation-smoke-e<n>.lane*.out` ends with `lane: batches ... have no
trial left to start`):

1. `jobs/smoke/ablation-smoke-e<n>.summary.md` has a row for every trial, each with a success of 0 or 1 and none marked
   "not a result" (0 is fine);
2. `smoke_check.py` says OK again for every batch;
3. `<RESULTS>/ablation-smoke-e<n>/` has `ablation.json` and a result folder per trial.

Then send the smoke back, one batch per experiment, to the ablations' results dataset (section 6):

```bash
for e in 1 2 3 4 5; do
  SMOKE=1 RESULTS=<results directory> ENV_FILE=$PWD/.env CREATE_PR=1 NOTES="none" \
    bash-runners/ablations/send_results.sh E$e <your name, lowercase>
  sleep 61
done
```

Send us the batch names, bundle ids and pull request links it prints, and **wait for our OK before step 3**.

## 4. Step 3: the runs

One command per experiment. Each refuses to start without a passed step 1 for its variants, then starts 4 lanes per GPU
in the background (60 s apart; `LANES=<n>` and `GPUS=<list>` as in your evaluation run) and returns:

```bash
RESULTS=<results directory> bash-runners/ablations/run_E1.sh     # GPT-6 Astra, 156 trials, batches ablation-e1-r1..r3
RESULTS=<results directory> bash-runners/ablations/run_E2.sh     # Claude Opus 5.5, 120 trials, ablation-e2-r1..r2
RESULTS=<results directory> bash-runners/ablations/run_E3.sh     # 36 deformable trials, ablation-e3-r1
RESULTS=<results directory> bash-runners/ablations/run_E4.sh     # 24 trials, both modes, ablation-e4-r1
RESULTS=<results directory> bash-runners/ablations/run_E5.sh     # 30 trials, ablation-e5-r1
```

- Use one `RESULTS` directory, outside the repository, for all of them, and the default `JOBS` (`jobs/agents`).
- A runner prints its lanes' PIDs and the line `E1: 16 lanes over 52 task variants x 3 batch(es) ...`. Each lane runs
  the batches of the experiment in order, so every trial of repeat 1 starts before any of repeat 2; a lane goes on to
  the next batch when no trial of the current one is left to start.
- **Several experiments at once** are fine: all lanes share the host's GPU ledger, so a trial starts only where it fits,
  and lanes beyond what the host holds wait. E2 calls Claude, the others Codex: start E1 and E2 together, then E5, E4
  and E3 when E1's lanes have ended (each lane's output, `jobs/ablation-e1.lane<k>.out`, then ends with `lane: batches
  ... have no trial left to start`). Your model quotas must carry all the lanes that run at once (each lane is one agent
  calling its model); start fewer (`LANES=<n>`) if they do not.
- What is fixed and cannot be set: the agent, `EFFORT=high`, the tasks and conditions, the modes, the batch names, seed
  0 (`K=3 SEEDS=0`, as `bash-runners/run_seed0.sh`), the usage gateway with `GATEWAY_RAW=0` and its standard token budget
  (100M input, 10M output tokens per trial), web search off, each task's own time limit. Other settings of the batch
  runner pass through as in your evaluation run (`GPUS`, `MEM_GUARD=1` on a shared host, `CODEX_FORCE_AUTH_JSON`, ...).

**Monitoring**

```bash
tail -f jobs/agents/ablation-e1-r1.log                  # every trial's start, end, errors
cat jobs/agents/ablation-e1-r1.summary.md               # finished trials, one table per configuration
tail -qn1 jobs/ablation-e1.lane*.out                    # a lane that is done ends with "lane: batches ... have no trial left to start"
python3 scripts/harbor/live_board.py --jobs jobs/agents  # every trial's state in a browser; keep it local
```

The task names in these tables carry the condition (`roboarc-bed-bolt-03-i01-q1c10-standard` is C10).

## 5. Interruptions, voided trials and reruns

The batch runner's rules hold as in your evaluation run (COLLABORATOR_SETUP.md, section 5):

- **Restart** (after a reboot, or a lane stopped by a model-account failure: `STOP lane` in the batch log): run the
  same `run_<E>.sh` again. Finished trials are skipped; a lane that was stopped did not go on to the next repeat.
- **"not a result"** in a summary is an infrastructure failure (the grader's `infra_error`, no `reward.json`, a Harbor
  exception other than the agent's own end, a model-account failure, a gateway that went down). Such a trial may run
  **once** more: move its job folder `jobs/agents/<job>/` and its claim `jobs/agents/.claims/<job>` out of `jobs/agents`
  (do not delete them), then run the same `run_<E>.sh` again; it gets the same seed. Write every rerun in `NOTES` when
  you send the results back.
- A trial the agent ended itself is a result (0 or 1) and never runs again: its time limit, its token budget, a response
  over the output-token maximum or the context window, the agent killed at its own container's memory cap.

## 6. Step 4: send each experiment back

**Where the results go.** Everything of these experiments goes to the public Hugging Face dataset
[`RoboARC/RoboARC-ablation-outputs`](https://huggingface.co/datasets/RoboARC/RoboARC-ablation-outputs) (not the main
evaluation's `RoboARC/RoboARC-outputs`): each batch into `<your name>/<batch>/<bundle id>/`, the check reports into
`<your name>/ablation-checks/<UTC time>/`. Every run's `<RESULTS>/<batch>/ablation.json` names it too
(`results_dataset`). The bash-runners/ablations scripts send there by default; `HF_DATASET` would override it, so leave
it unset.

**What you need**, once per host, as for your evaluation run (COLLABORATOR_SETUP.md, section 6): a Hugging Face token of
the type **Write** of your own account in `~/.roboarc_hf_token` (mode 600), and the `hf` CLI (`pip install -U
huggingface_hub`). With `CREATE_PR=1`, as in the commands here, each upload opens a pull request on the dataset from
your account, which we merge: your account needs no access to the dataset. If we give your account write access to it,
leave `CREATE_PR=1` out and the bundle goes in directly, as one commit.

When an experiment's lanes have all ended, send it, one bundle per repeat (`scripts/send_results.sh` for each batch, with
the experiment and the repeat in the notes):

```bash
JOBS="$PWD/jobs/agents" RESULTS=<results directory> ENV_FILE=$PWD/.env CREATE_PR=1 \
  NOTES="<crashes, outages, reruns; or none>" \
  bash-runners/ablations/send_results.sh E1 <your name, lowercase>
```

Each bundle carries the batch log, summary and plan, every job folder, the per-trial results and, in the results,
`ablation.json` (the experiment, the repeat and every planned trial with its condition) and `ablation_checks.json` (the
step-1 report the experiment ran after). Send us what the script prints for each batch: the batch name, the bundle id,
the `SHA256SUMS` sha256 and the pull request link. The dataset is public: anyone can read what you send. The secret scan
removes the values in your `.env` first; everything else in the job folders goes up as it is, paths and host names in
the logs included.

A partial delivery (an experiment not yet finished) is fine too: each sending is a new bundle with the batch's state at
that moment and never overwrites an earlier one.

## 7. E4: the relaxed clearances

E4 relaxes one clearance per puzzle, x2 and x4, and changes nothing else (scene, start state, robot, grader logic; the
task texts restate the dimensions):

| task | what is relaxed | released | x2 | x4 |
|---|---|---|---|---|
| `roboarc-bed-bolt-01-i01`, `-03-i01` | the blind bore round the 5 mm shank (bore minus shank) | 0.4 mm | 0.8 mm | 1.6 mm |
| `roboarc-lock-picking-04-i01` | every plate's bore round the key shaft | 1.0 mm | 2.0 mm | 4.0 mm |
| `roboarc-lock-picking-05-i01` | every plate's bore round the key shaft | 0.75 mm | 1.5 mm | 3.0 mm |
| `roboarc-cast-coil-06-i01`, `-07-i01` | the gap between the assembled pieces, with the goal's assembled-pose position tolerance (two thirds of the gap, as released) | 1.5 mm (1 mm) | 3 mm (2 mm) | 6 mm (4 mm) |

The variants are rebuilt from the release image's simulator source with that one parameter edited, frozen as the release
freezes a scene, and locked by DIGEST. Every variant passes the released reference solution (a wider gap still admits
the original motion), and its measured clearance is the intended multiple of the released one.

## 8. E5: the sensor model (V2)

V2 changes only what the agent observes: images are rendered by the true cameras, and the robot, physics, grader and
replay are the Standard task's. The values are those of the real robot's cameras, Intel RealSense D405 on the wrists and
D435i on the static mounts (`scripts/ablations/cond_obs.py`, `SENSORS` and `CALIB`):

- **Depth noise.** Per valid pixel and frame, Gaussian noise with the standard deviation of Intel's RMS-error model,
  sigma(z) = z² · subpixel / (f · B), with subpixel 0.08, f = (848 / 2) / tan(90° / 2) = 424 px and B the stereo
  baseline (D405 18 mm, D435i 50 mm): for example 0.94 mm at 0.5 m and 3.8 mm at 1 m (D435i), 0.42 mm at 0.2 m (D405).
  Sources: Grunnet-Jepsen et al., *Best-Known-Methods for Tuning Intel RealSense D400 Depth Cameras for Best
  Performance*, rev. 1.9, section 3e ("Depth RMS error(mm) = Distance(mm)^2 x Subpixel / (focal length(pixels) x
  Baseline(mm))", "focal length(pixels) = ½ Xres(pixels) / tan(HFOV/2)", with the example "D435 with HFOV=90deg,
  Xres=848, and baseline=50mm ... subpixel=0.08"); *RealSense Product Family D400 Series Datasheet*, rev. 023 (March
  2026), Table 3-49 (baselines 50 mm and 18 mm) and Tables 3-13 / 3-15 (imager HFOV 90°).
- **Quantisation.** Depth rounded to the camera's depth unit: 1 mm for the D435i (Best-Known-Methods, section 8a: "By
  default the D4 VPU provides 16bit depth with a depth unit of 1000um (1mm)") and 0.1 mm for the D405 (librealsense
  `src/ds/d400/d400-device.cpp`: the D405's default depth units 0.0001).
- **No reading** closer than the minimum range at 848×480, 195 mm (D435i) and 70 mm (D405) (Datasheet Table 4-12), and
  at a random 1 % (D435i) and 0.5 % (D405) of the other pixels (fill rate ≥ 99 % and ≥ 99.5 %, Datasheet Table 4-15).
- **Calibration error.** Each camera's stated calibration (`spec["cameras"]`, `meta["cameras"]`, the per-frame
  `extrinsic_cv`) is off by a fixed error drawn once from a fixed seed: the stated pose by a rotation of 0.5° about a
  random axis and 2 mm in a random direction (hand-eye calibration of a RealSense D435 on a robot arm: 0.48° and 2.04 mm
  in Wang et al., *PlaneHEC*, arXiv:2507.19851, section IV; 0.60° and 1.1 mm in ViSP's extrinsic-calibration tutorial),
  the stated focal lengths by 0.3 % (Grunnet-Jepsen et al., *Intel RealSense Self-Calibration for D400 Series Depth
  Cameras*, rev. 2.7, Addendum A: focal-length errors "less than ~0.3%"). The principal point is left exact: no source
  gives its error. The error is drawn per camera name, so it is the same in every call, every trial and every task (one
camera rig, calibrated once), and the trials are reproducible. We measured it on the running service: every camera's
stated pose is off by 0.500° and 2.00 mm, its focal length by 0.30 %, consistently in `spec`, `meta` and the per-frame
extrinsics.

The agent's task text describes this model in one paragraph ("Camera model (this variant)"), as a camera's datasheet
would. V1 and V2 still record the contact-safety readings for the grader and the reviewer; only the agent does not get
them.

## 9. Settings that must not change

Results count only when their run had these:

- the commit we named, for every step: do not pull in the middle; nothing edited under `tasks/` or `scripts/ablations/`
  (the runners check the variants' DIGESTs against step 1's report);
- the runners' fixed settings (section 4): the agent and model of each experiment, `EFFORT=high`, seed 0, the usage
  gateway and its token budget, web search off, the tasks' own time limits and texts (no `--timeout-multiplier`, no
  `AK`, no prompt template);
- one model account per agent for all the experiments' batches.

## 10. Final checklist

- [ ] Checkout `dev/qineng-ablations` at the commit we named; `bash images/pull_all.sh` and `python3
      scripts/stamp_mtimes.py tasks` done; `.env` with the Codex and Claude Code credentials, mode 600; a Hugging Face
      Write token in `~/.roboarc_hf_token` and the `hf` CLI.
- [ ] Step 1: `check_variants.sh all` printed `ALL PASS` for E1 to E5; reports sent with `send_checks.sh`; our OK.
- [ ] Step 2: five smoke batches started, `smoke_check.py` OK at ten minutes and at the end; smoke sent
      (`SMOKE=1 bash-runners/ablations/send_results.sh E<n> <name>`); our OK.
- [ ] Step 3: `run_E1.sh` to `run_E5.sh` started; every lane has ended; no trial left "not a result" without its one
      rerun; every rerun noted.
- [ ] Step 4: `bash-runners/ablations/send_results.sh <E> <name>` for E1 to E5; batch names, bundle ids, sha256 and pull
      request links sent to us; the bundles are in `RoboARC/RoboARC-ablation-outputs` (pull requests until we merge them).

## 11. Questions

Send setup questions, the check reports, failing job folders and smoke outputs to us (Qineng Wang); questions about the
batch runner itself also to Jianwen Lyu (GitHub `LUIGIVAMPER`).
