# Running RoboARC: setup for a collaborator

RoboARC 1.2.1 is on `main` of `mll-lab-nu/RoboARC`; run it from `main`. You get read access to the repository as a collaborator. The four container images on GHCR (`ghcr.io/mll-lab-nu/...`) inherit the repository's access, so the same invitation covers them. This note covers what your machine needs, how to log in, a first check, and the evaluation run itself. The repository's `README.md` and `docs/RUNNING.md` are the full reference.

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
   docker pull ghcr.io/mll-lab-nu/roboarc:1.2.0      # access check: it must download
   ```
   Docker uses this login to pull the task images. Every task names its image by digest (`images/pins.json`):
   - `roboarc:1.2.0`, about 10 GB on disk, used by every puzzle task;
   - `roboarc-monitor:0.1.0`, about 0.13 GB on disk, used by every task;
   - `roboarc-ikea:1.1.0` and `roboarc-ikea-sim:1.1.0`, about 1.1 GB and 7.7 GB on disk, used only by the IKEA tasks.

   The download is smaller, because the layers are compressed.
3. Clone the repository, or `git pull` an existing clone, and pre-pull every image (each is checked against its pinned
   image id):
   ```bash
   git clone git@github.com:mll-lab-nu/RoboARC.git
   cd RoboARC
   bash images/pull_all.sh
   python3 scripts/stamp_mtimes.py tasks   # once per checkout, so Harbor's build cache hits
   ```

## 3. First check: one task

Start with `roboarc-interlocked-claws-01-i01`. The reference solution must score 1, and doing nothing must score 0:

```bash
harbor run -p tasks -i roboarc-interlocked-claws-01-i01-privileged -a oracle -o "$PWD/jobs" --job-name oracle-claws-01
harbor run -p tasks -i roboarc-interlocked-claws-01-i01-privileged -a nop    -o "$PWD/jobs" --job-name nop-claws-01
```

- The score is in `jobs/<job>/<trial>/verifier/reward.txt`: 1 or 0.
- The oracle run of this task takes about 6 minutes on an RTX 4090 (the image already pulled). The verifier replays the
  trajectory twice and renders a video, which lands in `verifier/replay.mp4`.
- The first run also pulls the image, so it takes longer.
- If something fails, send us `jobs/<job>/` (or at least `result.json` and the `verifier/` logs) and the output of the `vulkaninfo` check above.

## 4. The evaluation run

Run a coding agent over the tasks with the batch runner, three trials per task and mode (`K=3`). Every puzzle task has
three scenes (appearance seeds 0, 1 and 2: the same physics in other colours, materials and textures), and the runner
gives the three trials of a task different seeds, so they see three different scenes; an IKEA task has one scene. Credentials go in an env file; `.env.example` lists the variables, e.g. `OPENAI_API_KEY` for Codex or `ANTHROPIC_API_KEY` for Claude Code.

```bash
AGENT=codex MODEL=openai/<model> EFFORT=high K=3 ENV_FILE=$PWD/.env \
  JOBS="$PWD/jobs/agents" RESULTS=/path/outside/the/repo \
  scripts/harbor/run_agent_batch.sh <batch name> roboarc-interlocked-claws-01-i01 roboarc-orbits-02-i01 ...
```

- **Which tasks and modes:** list the task bases to run; ask us which ones. `MODES=privileged` or `MODES=standard` runs one mode only.
- **Progress and summary:** progress is in `$JOBS/<batch>.log`, and the per-trial summary in `$JOBS/<batch>.summary.md`.
- **Restarting:** running the same command again resumes, and it also adds a second lane.
- **Budget:** each trial gives the agent up to 3 h. Use one API key per batch, with a spending cap.
- **Live view:** `python3 scripts/harbor/live_board.py --jobs <JOBS>` serves a page on 127.0.0.1. Keep it local: it shows unredacted output.
- **Everything else:** `docs/RUNNING.md` has the agent options, network rules, re-grading and the result files.

## 5. Questions

Send setup questions and failing job folders to Jianwen Lyu (GitHub LUIGIVAMPER). Ask Qineng Wang how to send the evaluation results back.
