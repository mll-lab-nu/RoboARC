# Developing RoboARC

How the task directories are made, how to check a change, how the images are built, and how a release is cut. For
running tasks see [RUNNING.md](RUNNING.md), which also has the install lines (uv, a Python 3.11 or newer as `python3`,
Harbor, the command `roboarc`) and a table of earlier names; the task contract is [PROTOCOL.md](PROTOCOL.md).

## Where things are

| path | what |
|---|---|
| `tasks/roboarc-*` | the 180 task directories: 156 puzzle tasks (`scripts/roboarc/mk_tasks.py`) and 24 IKEA tasks, `roboarc-ikea-*` (`ikea/scripts/ikea_mujoco/mk_tasks.py`, `mk_standard_tasks.py`). Generated: never edit them by hand |
| `scripts/roboarc/` | the generator `mk_tasks.py` and its inputs: `release.json` (one record per base), `packages/` (each task's texts as they ship, with `{{FIELD}}`s the generator fills), `oracles/` (reference trajectories), `shared/` (the pinned tooling and graders), `assets.json` (instance pins); `regrade.py`; `tests/` |
| `scripts/` | the task layer the tasks cite and the generator renders with: `task_conventions.py` (`normalize`, the mode registry, the effective configuration), `set_agent_egress.py` (the egress allowlist), `task_resources.toml` (resource rules), `eai_standard/` (the eai-standard/2.4 client and service layer, copied into every standard task), `rcb_monitor/` (the resource-limits sidecar, copied into every task); `stamp_mtimes.py`, `check_image_files.py` (image builds) |
| `scripts/harbor/`, `bash-runners/` | the Harbor batch runner and its helpers; the evaluation run one seed at a time (RUNNING.md, COLLABORATOR_SETUP.md) |
| `roboarc/`, `examples/`, `tests/api/` | the Python package with the command `roboarc` ([roboarc/README.md](../roboarc/README.md): what each subcommand starts), its examples and its tests ([API.md](API.md)) |
| `ci_checks/` | the repository checks |
| `images/` | the image pins (`pins.json`) and the scripts that pull, check, archive and push the images ([images/README.md](../images/README.md)); `images/roboarc/`, `images/monitor/`: the release image's build script, the recipe of the monitor sidecar image (tasks pin both by digest) |
| `source/` | the simulator source the image is built from (its README) |
| `tools/` | maintainer tooling: the release rebuild (`tools/release/`) and the simulator's tests (`tools/tests/`) ([tools/README.md](../tools/README.md)). Tools for authoring new tasks and reference solutions are not part of the public release |
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
python3 ci_checks/check-frozen-bytes.py               # no frozen path differs from its baseline (frozen-bytes.json)
bash ci_checks/run_invariants.sh                      # the protocol's invariants (PROTOCOL Appendix H); required
bash ci_checks/check-unit-tests.sh --all              # unit tests (numpy, requests, scipy for the second tier)
bash ci_checks/run_all.sh                             # the invariants, the unit tests, the advisory checks (exit 3: advisories only)
python3 -m pytest tests/api -m "not sim and not heavy"   # the Python package and the command line
```

The first two lines are the generators' checks, which `run_all.sh` leaves to their own CI jobs. The checks need
Python >= 3.11 (with numpy, requests and scipy for the second tier of unit tests: `uv pip install
--no-deps --require-hashes -r ci_checks/requirements-ci.lock`, the CI jobs' hashed lock for Python 3.12), no GPU and no image.
`.github/workflows/checks.yml` runs them on pushes to `main` and `dev/**` and on pull requests, in the jobs `generator`,
`ikea`, `image-pins` (`check-image-pins.py`: every image a task names is pinned by digest, images/pins.json),
`frozen-bytes` (`check-frozen-bytes.py`: the tasks, the files the generators copy or render from and the images'
inputs equal the baseline whose hashes `ci_checks/frozen-bytes.json` records, with the reason each path is frozen),
`invariants` (required), `unit-tests` (with `check-video.py`, `check-resources.py`, `check-cutoffs.py`),
`api-tests` (the Python package's fast tests, `tests/api/`) and `advisories` (warnings only). The workflow
pins its runner image, the actions (by commit), Python (3.12.15) and the Python packages (`ci_checks/requirements-ci.lock`,
installed with `--require-hashes`).

Where the tests are, and what runs them:

| tests | run by |
|---|---|
| `tests/api/`: the Python package and the command `roboarc` | `python3 -m pytest tests/api -m "not sim and not heavy"`; with the markers `sim` and `heavy` in an image ([API.md](API.md), "Tests") |
| `scripts/tests/`, `scripts/roboarc/tests/`, `scripts/harbor/tests/`, `scripts/usage_gateway/tests/`, `images/tests/`, and the tests of `tools/tests/` that need no simulator | `bash ci_checks/check-unit-tests.sh --all`; one of them: `bash ci_checks/check-unit-tests.sh --one 300 <test path>` |
| `scripts/rcb_monitor/test_monitor.py` | `ci_checks/check-monitor.py` |
| `ikea/scripts/ikea_mujoco/tests/` | `bash ikea/ci/check.sh --image-tests`, in the IKEA image |
| the other tests of `tools/tests/` | in the release image ([tools/README.md](../tools/README.md)) |

What the invariants run: `check-agent-pointers.py` (H.2), `check-dataset-manifest.sh` (H.8: every DIGEST recomputed),
`check-twins.py` (H.7), `check-monitor.py --tier invariants` (H.1), `scripts/harbor/tests/test_retry_policy.py` (H.5),
`check-ci-selftest.py --tier invariants` (real tasks broken one way each; every rule must fire), and per task
`check-task.sh --tier invariants` (verifier isolation, instance copies, the standard agent image). The advisory tier adds
the per-task conventions, `check-tooling-identity.sh` (shared files byte-identical; the 2.4 layer equal to
`scripts/eai_standard/`), `check-standard-client.sh`, resources, cutoffs, seeds (`check-seeds.py`; `check-seed-files.py`:
every allowed seed of a task with `build = "runtime"` has its `seed` file, `runtime_binding.s<S>.json`, goal images and
oracle; also a failing step of the CI job `unit-tests`), videos, `check-protocol.py`,
`check-world-frame-texts.sh` (no world coordinates in standard texts), `check-build-stamps.py` and
`check-doc-paths.py` (every path and relative link the docs name exists; docs/history/ excepted).

What CI cannot see is checked on a GPU host with the images: strict replays of the references (end hashes equal to
the recorded ones: `roboarc replay`, [API.md](API.md)), and Harbor oracle / nop trials:

```bash
roboarc selftest --families all            # one task per family, both modes: the reference scores 1, doing nothing 0
roboarc selftest <task> --seeds 0,1        # a new or changed task, at the seeds of its [metadata.seeds] oracle (here 0, 1)
```

`roboarc selftest` runs each trial as `harbor run -p <a stamped copy at the seed> -i <task>-<mode> -a oracle` (or `-a
nop`); a single trial by hand is in [RUNNING.md](RUNNING.md), "A single trial".

## Images

Building is optional: the images of `images/pins.json` are the release, and `roboarc images pull --accept-licences`
(`bash images/pull_all.sh --accept-licences`) gets them ([images/README.md](../images/README.md)). `roboarc images
build roboarc|ikea|monitor [-- ARGS]` starts the build scripts below after the same licence step.

```bash
images/roboarc/build.sh --accept-licences                      # the release image; frozen instances from the pinned Hugging Face dataset
images/roboarc/build.sh --accept-licences --instances <dir>    # or from a local copy (instances/<base>/ per base)
images/monitor/build.sh --accept-licences                      # the monitor sidecar's image
ikea/images/ikea-mujoco/build.sh                               # the two IKEA images
```

A build installs third-party software: `--accept-licences` (or `ROBOARC_ACCEPT_LICENCES=1`) states that you accept the
licences [LICENSES/THIRD_PARTY.md](../LICENSES/THIRD_PARTY.md) lists, and a build script that takes the option stops
without it. The frozen instances and the IKEA meshes are not in git. They come from two Hugging Face datasets, each
pinned by revision and by the sha256 of every file (`scripts/roboarc/assets.json`,
`ikea/scripts/ikea_mujoco/assets.json`). A public dataset is read without a token, a private one with `hf auth login`
or `HF_TOKEN`. `ROBOARC_ASSETS_SOURCE=<repo>@<revision>` (the instances) and
`ROBOARC_IKEA_ASSETS_SOURCE=<repo>@<revision>` (the meshes) download from another published copy of a dataset; every
file is checked against its pinned sha256 whatever the source.

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
   deterministic, without infra or grader errors. The oracles never ask for video, so also run the standard service's
   tests in the new image, with ffmpeg required (`docker run --rm --network none -e RCB_REQUIRE_FFMPEG=1 -v
   $PWD:/r:ro --entrypoint python <image> /r/scripts/roboarc/tests/test_server_requests.py /r`): they ask for mp4.
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
