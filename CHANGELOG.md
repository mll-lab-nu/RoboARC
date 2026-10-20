# Changelog

RoboARC's releases. Versions before 1.0.0 were released inside the shared benchmark repository
`mll-lab-nu/embodied-first-exam` (formerly robot_coding_bench), under the image name `ghcr.io/mll-lab-nu/eai-roboarc`.
1.0.0 is the first release from this repository. Its content is the "V3" wave of that repository.

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
`-sim:0.4.0` (embodied-first-exam #299, 2026-10-17).

### New bases (reviewed and approved by the benchmark owner, 2026-10-18)
- **orbits-02-i01** (orbits, easier): runs on a 0.3125 ms MuJoCo step (solver_substeps 8), see
  [docs/families/orbits.md](docs/families/orbits.md). Reference end state b9c68d6502105df4.
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

## 0.2.0 (2026-10-17, "V2"; embodied-first-exam #301)

98 bases (196 task directories). Image `eai-roboarc:0.2.0`.
- New bases: knot-untying-06-i01 (rd very_easy) and knot-untying-09-i01 (rd medium; 27.4 % of the budget, under the
  knot family's 30 % bar), blanket-flatten-07-i01 (easy) and blanket-flatten-08-i01 (medium), both ported to
  `blanket_vbd_gpu`.
- Standard references within the standard service's joint-speed rule (0.04 rad per control step) for 11 bases.
- buzz-wire-3d-03 / 04: references where the release had none.
- knot-06 / 09 re-frozen without the lab start's provenance in instance.json. The physics is unchanged; the births
  changed.
- Reference meta hygiene: no host paths in the references' metadata.

## 0.1.0 (2026-10-14; embodied-first-exam #186 / #187)

The MuJoCo release. 94 bases (188 task directories) replace the Genesis release of the same puzzles (RoboTangle,
`eai-robotangle` 0.2.22, whose results are not comparable). Image `eai-roboarc:0.1.0`, HF assets revision 88e0136e1b97.
- MuJoCo 3.14.0 on the CPU in float64 (blanket families: a deterministic CUDA cloth step), and numpy / OpenBLAS SIMD
  masking in every container that runs a session, so every CPU class builds the same frozen bytes.
- The 3 h agent budget in both modes. Standard mode on eai-standard/2.4, with no world coordinates in standard texts.
- Privileged grading replays need the strict goal within the agent's rows (the 2 s hold only confirms). Earlier
  privileged results can be re-graded with `scripts/roboarc/regrade.py`.
- Follow-ups on main: the image's own Harbor evidence (#203) and the lock rotor discs drawn where the physics has them
  (#218).
