# IKEA MuJoCo: decision log

The decisions that shaped the IKEA MuJoCo tasks, as integrated into RoboARC from the scene authors' work (Xiaoyu and
Zhen Wu's team: the twelve scenes, their mechanisms, success checks and robot demos). Each entry says what was decided
and why; the measurements behind them are in `evidence.json`, `standard_evidence.json` and `oracles/<scene>/README.md`,
and the benchmark page is `docs/benchmarks/ikea_mujoco.md` at the repository root. (The file keeps its name because
the kyrre task texts cite it.)

## Grading

- **Terminal rules in every task (2026-10-15).** Success = the scene's `success.py` + released (no robot link touches a
  part) + at rest (every part at most 0.01 m/s and 0.05 rad/s) + no breakdown (no non-finite state, no BADQACC /
  BADQPOS / BADQVEL, no part contact deeper than 10 mm), at the end of the rows and again after the last row is held
  for 1.0 s. A momentary or held pose is not an assembly, so burvik's "the top may still be held by a robot" was
  dropped.
- **A replay crash is the agent's 0 (2026-10-15).** A replay that crashes, or in which the scene's code or MuJoCo
  raises on the state the rows drove it into, is a breakdown, not a grader error: grader and infrastructure errors give
  the trial a free retry (PROTOCOL H.5), which the deliverable's own failures must not get.
- **Verifier isolation (2026-10-15, PROTOCOL H.1).** `test.sh` deletes symlinks, FIFOs, sockets and devices under
  `/app/output` before grading; `grade.py` refuses a symlinked trajectory.
- **Standard mode is graded by the live episode (2026-10-17).** The sim service ends an episode at the first settled
  success (`Session.status()`, then `final_check()` on a snapshot with the last row held 1.0 s) and the verifier
  transcribes that verdict; no grading replay runs. The live and replay rules are the same rule, MuJoCo on the CPU is
  bit-deterministic, and the longest IKEA episodes would need hours of replay. This differs from RoboARC's puzzle
  tasks, whose standard verdict also needs a fresh-process replay; both serve `eai-standard/2.4`, so IKEA keeps its own
  service-layer copy (`../eai_standard/v2_4_live/`).
- **One reference per task for both modes (2026-10-16).** The privileged `solution/trajectory.npz` and the standard
  `solution/oracle_actions.npz` are the same rows, all within the speed limits below.

## Physics (one rule for all scenes, clamp, not fail)

- **Joint-speed clamp, both modes (2026-10-16).** Every arm joint target moves at most the robot's real velocity limit
  per control tick (Franka Panda 2.175 / 2.61 rad/s, ViperX 300 s 3.14159 rad/s); rows within the limit + 1e-9 are
  applied unchanged. The author demos command up to 55x these limits, so every reference was rebuilt or retimed within
  them. The speed table also handles the `left/` / `right/` actuator names of stegesund and uppsta, which the first
  lookup missed.
- **Franka Hand slew (2026-10-18).** The hand's command moves at most its real finger speed (50 mm/s per finger, 0.1
  m/s of commanded finger width; 318.75 units/s in lack and stenerik, 182.142857 units/s in the other Panda scenes; the
  ViperX gripper is not limited). It stops the stress test's instant-close finger slam (~2.5 m/s fingers, 11.6 kN pad
  contacts). Every Panda reference was retimed or rebuilt for it; results graded before it are not comparable.
- **Cylinder -> prism (2026-10-16).** RoboARC's rule: every cylinder a robot geom can touch is a 32-facet convex prism
  at scene build. It removed every cylinder-caused breakdown in the stress test.
- **kyrre keeps native cylinders (2026-10-19, image 0.3.2).** The one exception to the prism rule. A stress A/B on the
  same 203 episodes gave non-jerk breakdowns of 2 / 174 (1.15 %) with native cylinders against 4 / 174 (2.3 %) with
  prisms, the same failure classes in both and none on a cylinder geom; and the author's demo assembles the stool only
  without the rule.
- **Welded-part contact switch-off (2026-10-18, image 0.3.1).** Bodies joined by latched retention welds
  (`sw_cutover.json`, kyrre) or seat welds (`seat_weld.py`, opt-in, on in uppsta) form one rigid group whose members stop
  colliding with each other; every other contact is unchanged. uppsta's end blocks kept ~900 contacts with the board
  after welding, which made its reference too slow to grade.
- **eket dowels (2026-10-16).** The soft dowel shells touch only the side panels; a stiff twin meets everything else.
- **MuJoCo 3.14.0 EPA fix (2026-10-15).** 3.14.0's EPA writes past two fixed 24-entry horizon arrays (fixed upstream
  after 3.14.0, 6d662aed); in eket it caused SIGSEGV or silent memory corruption in about 1 of 400 random episodes. The
  image patches 14 bytes of the wheel's own `libmujoco.so`: bit-identical to the stock wheel on every reference and on
  52 stress episodes, no crash on the 9 reproducers. A from-source build is not bit-identical to the wheel, and changing
  the CCD options broke eket's reference, so neither was used.
- **SIMD masking (2026-10-15).** numpy / OpenBLAS SIMD paths are masked in the image (`NPY_DISABLE_CPU_FEATURES`
  without AVX-512, `OPENBLAS_CORETYPE=Haswell`), as in RoboARC's image: unmasked, eket's reference diverged between
  AVX-512 and AVX2 hosts and crashed MuJoCo on Zen 4.
- **A bit-identical speed-up (2026-10-15).** `ikea_runtime.rotate()` uses a plain 3-vector cross product instead of
  `np.cross` (6.5x faster, the same bits; variera's reference replays 2.1x faster).

## Scenes and texts

- **No upstream pointers in what the agent reads (PROTOCOL H.2).** The scene texts' references to the upstream
  repositories, graders, manuals and USD assets moved to `provenance.json`; `stage_context.py` refuses a scene text
  with such a pointer.
- **Frozen scenes.** kyrre's mechanism read two development switches from the environment; `rcb_ikea` drops them on
  import, so nothing outside the scene files changes a scene.
- **Texts (2026-10-16).** Every instruction states the terminal rules, the speed clamp and (Panda) that joint 7 has no
  joint limit in these scenes. The agent budget is 10800 s of wall clock in both modes, as RoboARC's puzzles; a
  privileged trajectory may have at most the task's row cap, derived from the verifier's time limit.
- **Standard cameras rendered by Nyx (2026-10-16).** RGB from Nyx with RoboARC's look (path tracer, 32 spp, denoised,
  studio HDRI, point lights placed relative to each scene's work surface), in its own process and venv in the sim
  sidecar; depth from MuJoCo's raster. The live state hashes are bit-identical to the EGL renderer's over the first 2000
  ticks of every reference. The privileged replay video stays MuJoCo's renderer on CPU OSMesa (evidence only; the
  verifier has no GPU).
- **Reference timing.** One reference execution is measured against the agent budget on a fast-host class (quiet
  i9-14900HX, Zen 4). gladom, variera, stenerik, uppsta and kyrre take about a quarter of the budget there and 64-80 %
  on a slow Xeon Gold 6230R core; they ship, with the slow-CPU times reported on the benchmark page.
- **RoboARC 1.0.0 (2026-10-19).** The tasks were renamed `roboarc-ikea-<scene>-i00-<mode>` (from `ikea-…`) and the two
  images republished under RoboARC's names with the same ids. Evidence records keep the names their runs had.
- **RoboARC 1.0.1 (2026-10-19).** The tasks moved onto the repository's task layer and into `tasks/`: the shared
  closing sections of the instruction, the resource monitor, declared resources and seeds, a DIGEST, and the one
  egress list, which adds the package registries the instruction already named (1.0.0's allowlist had the model APIs
  only). `[task].name` is `roboarc/<dir>`. Graders, scenes, references and images are unchanged.

## Known issues and suggestions to the scene authors

- stegesund: the floor plane does not collide (class `deco`), so a part pushed off the table falls forever. No verdict
  depends on it.
- burvik: the demo releases the top about 25 mm above the jig, so it lands on the guards and leans 6°; the reference
  seats it with a two-hand servo. A demo recorded on the final physics would avoid it.
- kyrre: the hex key's rest pose is a knife edge (one contact) where the key vibrates at 0.099 rad/s from the start, so a
  key returned exactly there never meets the rest rule; the reference sets it down 12 mm along its rest.
- stenerik: welded screws keep their thread contacts, so late ticks cost about 5x the early ones.
- uppsta: a coarser end-block collision model (15-30 convex pieces instead of 114) would cut the cost per tick by
  roughly half; the board's 640 pieces are the next largest cost.
- Residual stress-test findings, none blocking: energy created at robot joint limits under random arm targets (vesken,
  eket), one table penetration (gladom, prefix-jerk), ViperX self-contact under jerk commands (svallet).
- Harbor with BuildKit can swap same-size, same-mtime files between tasks' build contexts; the `instance_exact` check
  caught it once. Give each task's files distinct mtimes before a Harbor run.
