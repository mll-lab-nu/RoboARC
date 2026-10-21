# Developing RoboARC

How the task directories are made, how to check a change, how the images are built, and how a release is cut. For
running tasks see [RUNNING.md](RUNNING.md); the task contract is [PROTOCOL.md](PROTOCOL.md).

## Where things are

| path | what |
|---|---|
| `tasks/roboarc-*` | the 226 task directories: 202 puzzle tasks (`scripts/roboarc/mk_tasks.py`) and 24 IKEA tasks, `roboarc-ikea-*` (`ikea/scripts/ikea_mujoco/mk_tasks.py`, `mk_standard_tasks.py`). Generated: never edit them by hand |
| `scripts/roboarc/` | the generator `mk_tasks.py` and its inputs: `release.json` (one record per base), `packages/` (each task's texts as they ship, with `{{FIELD}}`s the generator fills), `oracles/` (reference trajectories), `shared/` (the pinned tooling and graders), `assets.json` (instance pins); `regrade.py`; `tests/` |
| `scripts/` | the task layer the tasks cite and the generator renders with: `task_conventions.py` (`normalize`, the mode registry, the effective configuration), `set_agent_egress.py` (the egress allowlist), `task_resources.toml` (resource rules), `eai_standard/` (the eai-standard/2.4 client and service layer, copied into every standard task), `rcb_monitor/` (the resource-limits sidecar, copied into every task); `stamp_mtimes.py`, `check_image_files.py` (image builds) |
| `scripts/harbor/` | the Harbor runner and its helpers (RUNNING.md) |
| `ci_checks/` | the repository checks |
| `images/roboarc/`, `images/monitor/` | the release image's build and push scripts; the recipe of the monitor sidecar image (tasks pin it by digest) |
| `source/` | the simulator source the image is built from (its README) |
| `tools/` | maintainer tooling: the release rebuild (`tools/release/`), how the references were made (`tools/references/`), the redesigned tiers' generators, the stress rounds, the simulator's tests ([tools/README.md](../tools/README.md)) |
| `records/` | inputs and archives of earlier releases, kept for provenance ([records/README.md](../records/README.md)) |
| `docs/simulator/` | the simulator's documentation and the per-family development notes; `docs/rcb/README.md` maps the paths the frozen task files cite to where they are now |
| `ikea/` | IKEA MuJoCo: its generators (rendering into `tasks/` with the repository's task layer), scenes, images, evidence and checks (`ikea/README.md`) |

Files under `scripts/eai_standard/` and `scripts/rcb_monitor/` that the generator copies into tasks
(`rcb1.py`, `run.sh`, `client-requirements.txt`, `v2_4/rcb_client.py`, `v2_4/rcb_limited.py`, `monitor.py`,
`rcb_limits.py`, `selftest.py`) are part of every task's bytes: changing one changes every task (re-render, new
DIGESTs, re-validation). So do the texts `scripts/task_conventions.py` renders (the closing sections of every
instruction, the README layout), `HOSTS` in `scripts/set_agent_egress.py` and the values in
`scripts/task_resources.toml`. Paths that task files cite (`docs/PROTOCOL.md`, `docs/STANDARD_MODE.md`,
`docs/TAXONOMY.md`, `docs/benchmarks/roboarc.md`, `scripts/set_agent_egress.py`, `scripts/rcb_monitor/*`,
`scripts/eai_standard/*`, `ci_checks/check-monitor.py`, `ci_checks/check-tooling-identity.sh`) stay where they are.

## Regenerating the tasks

```bash
python3 scripts/roboarc/mk_tasks.py            # render every task dir from scripts/roboarc/ (writes tasks/)
python3 scripts/roboarc/mk_tasks.py --check    # compare tasks/ with a fresh render: 0 differences
```

The generator ends every task with `scripts/task_conventions.py` `normalize` (README layout, resources, the closing
`## External material` / `## Resource limits` / `## Time limit` sections, the monitor sidecar, last the `DIGEST`).
`python3 scripts/task_conventions.py --check tasks/<task> ...` checks one task's normal form. Run both with umask 022
(file modes are compared). A change to a task is a change to the generator's inputs, never to `tasks/`; it gets a new
`DIGEST` and a README Change Log entry (PROTOCOL §1, H.8). A task's texts are `scripts/roboarc/packages/<base>-<mode>/`:
final texts with `{{FIELD}}`s for the values from other sources (`mk_tasks.py` `fields()`; the Generator section of
[benchmarks/roboarc.md](benchmarks/roboarc.md)); a new base's packages start as a copy of a sibling base's.

## Checks

```bash
python3 scripts/roboarc/mk_tasks.py --check           # every task dir equals a fresh render
bash ikea/ci/check.sh [--image-tests]                 # IKEA: both generators' check modes (+ unit tests in its image)
bash ci_checks/run_invariants.sh                      # the protocol's invariants (PROTOCOL Appendix H); required
bash ci_checks/check-unit-tests.sh --all              # unit tests (numpy, requests, scipy for the second tier)
bash ci_checks/run_all.sh                             # the invariants plus the advisory checks (exit 3: advisories only)
```

They need Python >= 3.11 (with numpy, requests and scipy for the second tier of unit tests), no GPU and no image.
`.github/workflows/checks.yml` runs them on pushes to `main` and `dev/**` and on pull requests, in the jobs `generator`,
`ikea`, `image-pins` (`check-image-pins.py`: every image a task names is pinned by digest, images/pins.json),
`invariants` (required), `unit-tests` (with `check-video.py`, `check-resources.py`, `check-cutoffs.py`),
`api-tests` (the Python package's fast tests, once `tests/api/` exists) and `advisories` (warnings only).

What the invariants run: `check-agent-pointers.py` (H.2), `check-dataset-manifest.sh` (H.8: every DIGEST recomputed),
`check-twins.py` (H.7), `check-monitor.py --tier invariants` (H.1), `scripts/harbor/tests/test_retry_policy.py` (H.5),
`check-ci-selftest.py --tier invariants` (real tasks broken one way each; every rule must fire), and per task
`check-task.sh --tier invariants` (verifier isolation, instance copies, the standard agent image). The advisory tier adds
the per-task conventions, `check-tooling-identity.sh` (shared files byte-identical; the 2.4 layer equal to
`scripts/eai_standard/`), `check-standard-client.sh`, resources, cutoffs, seeds (`check-seeds.py`; `check-seed-files.py`:
every allowed seed of a task with `build = "runtime"` has its `seed` file, `runtime_binding.s<S>.json`, goal images and
oracle; also a failing step of the CI job `unit-tests`), videos, `check-protocol.py`,
`check-world-frame-texts.sh` (no world coordinates in standard texts), `check-build-stamps.py`, the easy set and
`check-doc-paths.py` (every path and relative link the docs name exists; docs/history/ excepted).

What CI cannot see is checked on a GPU host with the image: strict replays of the references (end hashes equal to the
recorded ones), and Harbor oracle / nop trials:

```bash
python3 scripts/stamp_mtimes.py tasks
harbor run -p tasks -i roboarc-<family>-<NN>-i01-privileged -a oracle -o "$PWD/jobs" --job-name oracle-...   # reward 1
harbor run -p tasks -i roboarc-<family>-<NN>-i01-standard -a nop -o "$PWD/jobs" --job-name nop-...           # reward 0
```

## Images

```bash
images/roboarc/build.sh                      # the release image; frozen instances from the pinned HF dataset (read access needed)
images/roboarc/build.sh --instances <dir>    # or from a local copy (instances/<base>/ per base)
ikea/images/ikea-mujoco/build.sh             # the two IKEA images
```

`images/roboarc/build.sh` builds `source/images/roboarc/Dockerfile` from `source/` with the frozen instances as a
named build context, checks every instance file against `scripts/roboarc/assets.json`, and checks that the image
installs the runtime every task is bound to (`scripts/roboarc/release.json` `runtime`). Every input of the recipe is
pinned (the base image by digest, Debian packages by version from a snapshot, wheels by sha256), so two builds install
the same files. It needs about 25 GB of disk
([images/roboarc/README.md](../images/roboarc/README.md)). A validated build is exported, not rebuilt, into a
deterministic OCI archive by `images/archive.sh`. The monitor sidecar's image and the two IKEA images are built
reproducibly straight into OCI archives (`images/monitor/build.sh`, `ikea/images/ikea-mujoco/build.sh`). Every task
names each image by the manifest digest of its archive (`images/pins.json`), and `images/push_all.sh` pushes the four
archives byte for byte (the registry token on stdin only; [images/README.md](../images/README.md)).

## Cutting a release

1. Change the generator's inputs (`scripts/roboarc/`; new references and instances come from the release tooling,
   `tools/release/release_rebuild.py`), bump `images/roboarc/VERSION` if the image changes and the task set's version
   (`scripts/roboarc/release.json` `version`, the README Release line; with a Change Log entry for every task that
   changes, release.json `changes`), re-render, and add the release to `CHANGELOG.md`.
   `release_rebuild.py` renders on a commit of this repository (`--onto`); with premade instances (`--instances`) and a
   local tag (`--tag`) it re-makes a released version next to its image: with that release's own instances and
   references it changes no task file and builds the same image.
   A new image is archived (`images/archive.sh`, or the archives of the IKEA / monitor builds) and its manifest
   digest and id go into `images/pins.json` before the tasks are rendered (`release_rebuild.py` does this for the
   release image): the tasks name every image by that digest.
2. In a fresh clone (umask 022): every check above green.
3. Build the image from that clone. A release that keeps the simulator must give an image whose installed filesystem is
   file-for-file identical to the previous one (path, mode, size, sha256; the `.pyc` files aside, whose bytes hold
   source times), apart from the files the release changes.
4. On a GPU host with the new image: strict replays of one base per family (and of every new or changed base), end
   hashes equal to the recorded ones; Harbor oracle (reward 1) and nop (reward 0) on a sample of bases in both modes,
   deterministic, without infra or grader errors.
5. Push the images with `images/push_all.sh` (the archives of `images/pins.json`), then tag the commit.

## Reviewing a change

- Generated tasks: read the generator and the shared sources, not hundreds of copies; re-run it and check that
  `git diff tasks/` is empty.
- Read at least two tasks per mode in this order: `instruction.md`, `tests/`, `solution/`, `environment/`, `task.toml`.
- Could the verifier be written from the instruction alone? Is every convention the agent needs stated? Does any text
  state a world coordinate, the table height or the robot base pose (docs/benchmarks/roboarc.md)?
- Can the agent reach a solution, a pointer to one, or the success predicate (images, tooling, docs, network;
  [LEAKAGE.md](LEAKAGE.md))? Does the verifier depend on anything the agent controlled besides the declared artifacts?
  Is the predicate checked at the end of the replay, and can it be true without doing the task?
- A new family or simulator change: a determinism check (two fresh replays, the same hash) on the hardware used for
  evaluation, and one adversarial trial (an agent told to obtain reward without solving the task) that scores 0.
- Twins share the frozen instance, success rule, seeds and cutoff (H.7). Resource minimums are measured, not copied
  from the caps; limits leave the margin RUNNING.md "Resources" names.
