# RoboARC v1.0.0: plan of the standalone repository

Goal: one self-contained repository, `Embodied-First-Exam/RoboARC` (private), holding the RoboARC benchmark only and
released as 1.0.0 (= the "V3" content of the shared benchmark repository). The task directories keep their graded
behaviour byte for byte. Only the image references and the version labels change.

## Layout

```
README.md  CHANGELOG.md  PROVENANCE.md  CITATION.cff  PLAN.md
LICENSES/                    the upstream MIT notice of the shared layer (the RoboARC licence is an open question)
tasks/roboarc-*/             202 Harbor task dirs (101 bases x privileged / standard), rendered, with a DIGEST each
scripts/roboarc/             the generator (mk_tasks.py), its inputs (release.json, assets.json, packages/, oracles/,
                             shared/ = the pinned tooling), assets.py, fetch_image_context.sh, regrade.py, tests/, ci/
scripts/                     the shared layer the tasks use: task_conventions.py, set_agent_egress.py, eai_standard/
                             (eai-standard/2.4 client + service layer), rcb_monitor/ (resource-limits sidecar),
                             retry/runner helpers and their tests (scripts/tests/), stamp_mtimes.py, ...
ci_checks/                   the CI checks (invariants, advisories, check-video, resources, unit tests), pruned to RoboARC
images/roboarc/              the release image recipe (Dockerfile, VERSION 1.0.0, stage_instance.py, BUILD_CONTEXT,
                             build.sh, push.sh) and the recipe of its Genesis base (Dockerfile.genesis-0.2.22)
images/monitor/              the recipe of the monitor sidecar image
docs/                        PROTOCOL.md, STANDARD_MODE.md, RUNNING.md, ... (the protocol the tasks cite, unchanged),
                             docs/benchmarks/roboarc.md (the benchmark page), docs/families/ (one page per family)
source/                      the simulator and release source: topogym/ (physics, tasks, planning = oracle source),
                             assets/robots/, images/robotangle/ (Dockerfile.mujoco and its helpers),
                             images/topogym/prune.py, images/robotangle/mujoco-3.14.0-epa/ (the EPA byte patch),
                             scripts/robotangle/ (release_rebuild, manifests, references, packages, package_text,
                             redesign, stress, tests), scripts/topogym_rules.py, docs/ (MuJoCo core, render bridge,
                             release notes)
.github/workflows/checks.yml the cheap checks on GitHub-hosted runners (no GPU, no image)
```

## Copied from where

| new path | from | commit |
|---|---|---|
| tasks/roboarc-*, scripts/roboarc/, images/roboarc/, images/monitor/, scripts/ shared layer, ci_checks/, docs/ (protocol) | `dev/robotangle` (the RoboARC release tree) | acfcec1089 |
| source/topogym, source/assets/robots, source/images/{robotangle,topogym/prune.py}, EPA patch, source/scripts/robotangle, source/scripts/topogym_rules.py, docs/families, source/docs | `dev/robotangle_mujoco` (the dev source) | 5cb4114f10 |
| CHANGELOG entries for 0.1.0 / 0.2.0 / 0.2.1 | release records, `docs/rcb/roboarc_next_wave.md`, `main-before-rewrite` | 19dcad58b5 |

## Left behind

- Every other benchmark (tasks, scripts/<bench>/, images/<bench>/), third_party/, the Genesis-era RoboTangle task dirs (tasks/robotangle-*,
  tasks/topogym-*) and the TopoGym repo tooling not used by the release (helpers, difficulty sweeps, VLA, design viewer).
- AI-agent instruction files; the shared repo's .github workflows (self-hosted runners).
- History: one fresh initial commit, author = the user's git identity. PROVENANCE.md lists the sources.

## Changes against the sources (all of them)

1. Image references: `ghcr.io/mll-lab-nu/eai-roboarc:0.2.1` -> `ghcr.io/embodied-first-exam/roboarc:1.0.0` (mk_tasks.py
   IMAGE, images/roboarc/VERSION 1.0.0). The monitor sidecar `ghcr.io/mll-lab-nu/eai-monitor:0.1.0@sha256:f2c910f5...`
   (that package no longer exists) -> `ghcr.io/mll-lab-nu/efe:monitor-0.1.0@sha256:f2c910f5...`: the same digest, in the
   public consolidated package. The Genesis base `eai-robotangle:0.2.22@sha256:eb1a7517...` -> `efe:robotangle-0.2.22@`
   the same digest.
2. Version labels 0.2.1 -> 1.0.0 where the texts state the release.
3. Paths: the source paths the image recipe copies live under source/ (fetch_image_context.sh copies them from the
   checkout instead of `git archive <release_commit>`); the EPA patch script moves from images/ikea-mujoco/ to
   source/images/robotangle/mujoco-3.14.0-epa/ (same bytes).
4. CI scripts: lists of other benchmarks' tests and hooks removed so the checks run on this tree. No protocol change.
5. DIGEST files re-stamped (they hash the task dir, which now names the new image).

## Build and verify (fresh clone)

1. `python3 scripts/roboarc/mk_tasks.py --check` -> 0 differences.
2. `bash ci_checks/run_invariants.sh`, `bash ci_checks/check-unit-tests.sh --only scripts/roboarc/`,
   `python3 ci_checks/check-video.py tasks/roboarc-*`, `python3 ci_checks/check-resources.py tasks/roboarc-*`.
3. Image: `images/roboarc/build.sh` = fetch_image_context.sh (source/ paths + the frozen instances from the HF dataset
   pinned in assets.json, sha256-checked) + `docker build`; then the installed-runtime check against release.json.
   On the build host: file-for-file comparison with the released 0.2.1 image 41a17481d67a (fs manifest: path, mode, size, sha256).
4. Spot strict replays in the new image: one base per family + the 4 V3 items (orbits-02, knot-07, blanket-flatten-09,
   blanket-fold-05), end hashes equal to the recorded ones.
5. Harbor oracle / nop on 2-3 bases, both modes, the new tag (local only).
6. `images/roboarc/push.sh`: credential-free push (token on stdin), gated on the Intel blanket replay gate.

## Scope change (coordinator, 2026-10-19 evening): IKEA MuJoCo is part of RoboARC 1.0.0

- `ikea/`: a subtree laid out as its source tree (dev/robotangle_mujoco 5cb4114f10), so its generators run unchanged
  with the shared layer they were written against (which differs from the puzzles' layer): `ikea/tasks/` (24 dirs,
  renamed roboarc-ikea-*), `ikea/scripts/ikea_mujoco/`, `ikea/images/ikea-mujoco/`, `ikea/scripts/{task_conventions.py,
  set_agent_egress.py, eai_standard/}`, `ikea/docs/benchmarks/ikea_mujoco.md`.
- Images `ghcr.io/embodied-first-exam/roboarc-ikea:1.0.0` and `roboarc-ikea-sim:1.0.0` (= 0.3.2 / 0.4.2, same ids).
- Checks: both generators' `--check`, the unit tests in the image (`ikea/ci/check.sh`), both images rebuilt with the
  same ids, Harbor oracle / nop on svallet and lack in both modes.

## Open questions (for the user)

- (Decided 2026-10-19: the IKEA tasks are renamed `roboarc-ikea-<scene>-i00-<mode>`.) Move IKEA onto the puzzles'
  task layer (monitor sidecar, DIGEST, RoboARC invariants)? That means a re-render and re-validation.

- Licence of RoboARC (the shared layer is MIT, upstream notice kept; TopoGym source has no licence file).
- Public vs private repo (private now).
- Image registry / name: placeholder `ghcr.io/embodied-first-exam/roboarc:1.0.0`; whether to mirror the Genesis base
  (efe:robotangle-0.2.22) and the monitor (efe:monitor-0.1.0) into the org too.
- HF assets dataset `eai-v2-internal/robotangle-assets` stays private (only the image is public).
- Harbor task names keep the `robot-coding-bench/` prefix (byte identity); rename later?
