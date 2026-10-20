# IKEA MuJoCo

IKEA furniture-assembly scenes in plain MuJoCo 3.14.0 on the CPU (float64), each with two robot arms and physical
grasps (finger friction, no assist). Twelve tasks ship, each in both modes (24 task directories,
`tasks/roboarc-ikea-<task>-i00-{privileged,standard}`):

| task | robot | what is assembled | mechanism |
|---|---|---|---|
| `lack` | two Franka Panda | LACK side table: four legs screwed into the top | threads (exact convex pieces), latched retention welds |
| `svallet` | two ViperX 300 s (ALOHA 2) | SVALLET lamp: the bulb screwed in, the shade snapped on | spring jaws and snaps, the bulb's thread |
| `gladom` | two Franka Panda | GLADOM tray table | threads (SDF meshes), latched welds |
| `variera` | two Franka Panda | VARIERA box: screws driven with a screwdriver | threads (SDF meshes) |
| `eket` | two Franka Panda | EKET cabinet: dowels and panels | latched retention welds |
| `ikea365` | two Franka Panda | IKEA 365+ stand and bowl: a screw driven with a screwdriver | threads (exact convex pieces), latched welds |
| `vesken` | two Franka Panda | VESKEN shelf unit: 23 snap connections | spring jaws and snaps |
| `stegesund` | two Franka Panda | STEGESUND stand: bolts driven with a hex key | capsule helices, seat welds |
| `stenerik` | two Franka Panda | STENERIK laptop stand: six screws driven with a screwdriver, four pads peeled and pressed on | threads (exact convex pieces), retention welds, pad peel-off and bonds |
| `burvik` | two Franka Panda | BURVIK side table: four legs screwed in (two into the top, two through the handle's sleeves), the handle through the top, the table turned upright | threads (SDF meshes), latched retention welds |
| `uppsta` | two Franka Panda | UPPSTÅ toy hammering block: four dowels and two end blocks, two screws driven with a hex key, eight pegs pushed in | threads (convex pieces), seat welds with the welded-part contact switch-off |
| `kyrre` | two Franka Panda | KYRRE stool: three legs placed on the upside-down seat, six screws driven with a hex key | threads (SDF), latched retention welds with the welded-part contact switch-off; MuJoCo's native cylinders (the one exception to the prism rule) |

`simulator = "mujoco"`; `robot = "franka-panda-bimanual"` (two Franka Panda arms with MuJoCo Menagerie's Panda hand;
stegesund and uppsta name their arms `left/` / `right/`, the others `left_` / `right_`) or `"aloha-2"` (svallet: MuJoCo Menagerie's
ALOHA 2, two ViperX 300 s arms). Standard tasks speak `eai-standard/2.4` (the operations capability:
`r.operation_status()`).

The agent budget is 10800 s (3 h) of wall clock in both modes. One reference execution must fit 40 % of it (4320 s) on
the trial host (Timing, below). A privileged trajectory may have at most the task's row cap (stated in its instruction:
the verifier's replays must fit its time limit).

## Tasks and generator
- `python3 scripts/ikea_mujoco/mk_tasks.py` writes the privileged tasks, `mk_standard_tasks.py` the standard twins
  (privileged first: the twins share one reference). `--check` compares only; CI runs both (no simulator needed).
- Inputs, all in `scripts/ikea_mujoco/`:
  - `tasks.json`: the texts (goal, success requirements, mechanism, the terminal rules);
  - `assets.json`: the sha256 of every scene file in the image, and the dataset pin of the meshes (Image and assets);
  - `bindings.json`: what each scene builds to in the image (initial-state hash, actuators, timestep);
  - `evidence.json`, `standard_evidence.json`: the Harbor checks and timings the READMEs and the privileged verifier
    timeouts are written from;
  - `shared/`: `rcb_ikea.py` (the tooling and the replay), `grade.py` (the verifier), `render_video.py` (the replay
    video);
  - `standard/`: the sim service (`server.py`, `ikea_rig.py`, `ikea_nyx.py` with RoboARC's `nyx_direct.py`,
    `finalize.sh`), its grader (`grade.py`: transcribes the live episode's `final.json`) and the oracle client,
    `rigs.json` (camera rigs), `std_reference.py`;
  - `oracles/<task>/`: the closed-loop references and how they were built (eket, ikea365, vesken) and stegesund's
    reference; `tails/variera.npy`: variera's release tail;
  - `provenance.json`: the scenes' upstream provenance, kept out of the image (PROTOCOL Appendix H.2);
  - `tests/test_grade.py`: the privileged grader's crash handling on a stub replay, then `tests/test_standard_grade.py`:
    the standard hook and grader on synthetic `final.json` / `episode.json` (live success, live failure, settle failure,
    crash give-ups, instance mismatch, no video, missing `final.json`, no replay started) (`ci_checks/check-unit-tests.sh`).
- Every generator ends with `scripts/task_conventions.py` normalize; the generators also write the egress allowlist
  of `scripts/set_agent_egress.py`.

## Grading (both modes, every task)
One rule in both modes. Privileged: the deliverable (`/app/output/trajectory.npz`) is replayed open-loop in two fresh
processes from the frozen keyframe. Standard: the rule is evaluated on the live episode by the sim service; nothing is
replayed (below). Success needs all of:
- **Goal:** the scene's own `success.py`, a port of the source assembly grader.
- **Terminal rules,** the same in every task: the parts are released (no robot link touches a part); every part is at
  rest (at most 0.01 m/s and 0.05 rad/s); the simulation never broke down (no non-finite state, no MuJoCo BADQACC /
  BADQPOS / BADQVEL warning, no contact of a part deeper than 10 mm).
- **End pose (EKET):** every panel, the back included, ends within 10 mm and 2 degrees of its goal relative to the top
  panel (`rcb_ikea.END_POSE_RULES`): the source grader checks the back only at its slide, and the back has no frame joint.
- These hold at the end of the rows **and** again after the grader holds the last row for 1.0 s
  (`rcb_ikea.Session.final_check()`).
- Privileged: both replays end in the same state hashes, before and after the hold.
- A replay that crashes (a simulator crash, `replay_crash`) or in which the scene's code or MuJoCo raises on the state
  the rows drove it into (`replay_exception`) is a breakdown (`sim_breakdown 1`, reward 0), not a grader error: a
  grader or infrastructure error gives the trial a free retry (PROTOCOL H.5), which the deliverable's own failures must
  not get. A scene that differs from the binding or builds to another initial state is `infra_error 1`.
- The privileged verifier's time limit covers a trajectory of the task's row cap on a slow trial-class CPU (one
  reference replay on a Xeon Gold 6230R core, `evidence.json` `slow_replay_wall_s`, with a 1.5x margin), not only the
  Harbor host's timing: a verifier that times out writes no verdict.
- The privileged verifier checks the scene in its image file by file against `tests/runtime_binding.json`
  (`instance_exact`) and its initial-state hash (`instance_ok`); the standard sim service does the same before its
  episode begins (`server.py`: scene files, initial-state hash, actuators and runtime against
  `environment/sim/runtime_binding.json`; a mismatch is `instance_mismatch`, `infra_error 1`, and nothing is served).

Standard mode is graded by the live episode: `success` = the sim service's live
verdict, `graded_by_live 1`, `replays 0`. The service ends an episode at the first settled success, `Session.status()`
and then `final_check()` on a snapshot with the last row held 1.0 s (the privileged grader's rule, the settle window
included); a failed settle lets the episode go on (`meta["settle_failed"]`). An episode that ends otherwise (the wall
clock, the collect hook's end) is judged as it stands by the same two checks. `finalize.sh` writes `final.json`
(`graded_by "live"`, `replays 0`, `state_sha` = the live end-state hash); the verifier (`tests/grade.py`, the shared
client base, no simulator) transcribes it. The trajectory (`/data/trajectory.npz`) stays the replayable deliverable; a
live success without it is `infra_error`. Infrastructure (a free retry, H.5): no, unreadable or unfinalized
`final.json`, `instance_mismatch`, a give-up after crashes not all inside the agent's calls (`sim_crashed`,
`resume_failed`, `call_watchdog`), a service-side error. The agent's 0: a breakdown, a simulator error its rows caused
(`sim_error`), a timeout, all crashes inside its calls (`sim_crashed_on_call`), a failed settle.

Every graded trial leaves a video (PROTOCOL §4), evidence only: standard, the sidecar's `/data/live.mp4`, recorded during
the live episode (Nyx, cam_high, 10 frames per simulated second, on the GPU the sidecar renders the agent's cameras with;
no extra pass); privileged, `/logs/verifier/replay.mp4`, rendered by a third process next to the two verdict replays
(MuJoCo's renderer on CPU OSMesa: the verifier has no GPU; 512 x 288, at most 750 frames).

## Robot speed (both modes)
Every arm joint target moves at most the robot's real joint velocity limit per control step: Franka Panda 2.175 rad/s on
joints 1-4 and 2.61 rad/s on joints 5-7; ViperX 300 s 3.14159 rad/s. It is a clamp, not a failure
(`rcb_ikea.Session.applied()`, in the tooling and the replay alike): a row within the limit (+1e-9) is applied
unchanged, a faster one is spread over the following control steps. The Franka Hand's command (`actuator8`) is
clamped the same way at the hand's real finger speed: 50 mm/s per finger (Franka Hand product manual 220010/1.2/EN,
5.1), so 0.1 m/s of commanded finger width, converted with each scene's `actuator8` gain and stiffness: 318.75 units/s
in lack and stenerik (255 units = 0.08 m), 182.142857 units/s in the other Panda scenes (gain 2.745, bias -300: 255
units = 0.14 m of target width, 0.08 m of travel plus 0.06 m of squeeze below closed). `rcb_ikea.GRIPPER_WIDTH_SPEED_M_S`,
`gripper_speed_limits()`; `bindings.json` records the per-scene values. The ViperX gripper is not limited.

## Physics changes against the source scenes
One rule for all scenes, clamp not fail (as RoboARC):
- **MuJoCo 3.14.0 with the EPA fix byte-patched in.** 3.14.0's EPA keeps the polytope horizon in two fixed 24-entry
  arrays and writes past them unchecked (fixed upstream after 3.14.0). A dowel pressed deep into a thin convex piece
  of eket's side panel overflows them: SIGSEGV or silent memory corruption, about 1 in 400 random eket episodes. The
  image patches 14 bytes of the wheel's own `libmujoco.so` (`images/ikea-mujoco/mujoco-3.14.0-epa/`); it is
  bit-identical to the stock wheel on every reference and stress episode checked. A from-source build is not
  bit-identical to the wheel and is not used.
- **Cylinder -> prism:** RoboARC's rule. Every cylinder a robot geom can touch is a 32-facet convex prism at scene
  build (inscribed in the cylinder). It removed every cylinder-caused breakdown of the stress test.
  **kyrre is the one exception** (owner's decision, 2026-10-19): its `scene.py` builds with the rule off, so its six
  screws keep MuJoCo's native cylinders (the rule would convert 12 geoms, each screw's head disc and thread core). The
  reason is a stress A/B on the same 203 episodes (7 strategies x 29, the same seeds and reference checkpoints):
  non-jerk BREAKDOWN 2 / 174 = 1.15 % with native cylinders against 4 / 174 = 2.3 % with the prisms (all strategies
  5.4 % against 5.9 %), the same classes in both (idle press-in jaws' speed jumps, energy at joint limits, a forearm on a
  leg) and none on a cylinder geom; and the author's demo assembles the stool only without the rule.
- **Joint-speed clamp:** above. The source demos command targets up to 55 x the real limits; the references were
  rebuilt or retimed within them.
- **Franka Hand slew** (V3): above. It stops the instant-close finger slam (~2.5 m/s fingers, 11.6 kN pad contacts in
  the stress test). Every Panda reference was retimed or rebuilt for it (see References).
- **Welded-part contact switch-off:** below (kyrre's retention welds; uppsta's seat welds).
- eket's dowels: the soft dowel shells touch only the side panels; a stiff twin meets everything else.
- numpy / OpenBLAS SIMD masking in the image (`NPY_DISABLE_CPU_FEATURES` without AVX-512, `OPENBLAS_CORETYPE=Haswell`),
  as in RoboARC's image: unmasked, eket's reference diverged between AVX-512 and AVX2 hosts.

Stress test (random, jerk and prefix strategies from the frozen birth), with the clamp, the prism rule and the eket
dowel rule: non-jerk BREAKDOWN rate 1.75 %, prefix-jerk 7 %; stegesund, 203 episodes, one genuine non-jerk BREAKDOWN
(the known finger-limit class). Residuals, none blocking: finger slams
(gladom; measured before the hand slew), energy created at robot joint limits under random arm targets (vesken, eket), one
table penetration (gladom, prefix-jerk), ViperX self-contact on jerk (svallet).

## References
Each task has one reference for both modes, every row within the joint-speed limits and the Franka Hand slew (the
privileged `solution/trajectory.npz` and the standard `solution/oracle_actions.npz` are the same rows). Every Panda
reference was retimed or rebuilt for the hand slew (2026-10-17/18); svallet (ViperX) is unchanged:
- lack: the authors' demo passed through the clamp, each row whose hand command steps faster than the slew held until
  the hand reaches it (`standard/std_reference.py --grip wait`): 67,697 rows;
- svallet: the demo time-dilated to the limits (`standard/std_reference.py`);
- eket, ikea365, vesken, gladom, variera, stegesund: closed-loop references built on the final physics
  (`oracles/<task>/`): the demo as the phase guide, with true-state feedback (Jacobian-IK servos, gates, local
  searches), each demo row held until the slewed hand reaches its command, recorded as the executed rows (ikea365 also
  servos the returned screw onto its return gate; variera keeps its 240-row release tail);
- stenerik: a closed-loop reference (`oracles/stenerik/`): the demo as the phase guide within the limits, screws
  steered and driven on the true screw axis, and each pad press servoed on the true pad-support force until the bond
  latches. Rebuilt under the Franka Hand slew (2026-10-18, 222,600 rows): each row is held until the hands reach it, a
  screw that lands leaning on its thread start is stood up by the screwdriver before it is driven, and each hand's grasp
  of a pad on the backing is centred on the pad.
- burvik: a closed-loop reference (`oracles/burvik/`): the retimed demo as the phase guide; the top seated in the jig's
  slot by a two-hand servo (the demo perches it on the guards, leaning 6°), the handle and the two handle legs servoed
  onto their axes and spun in, and the demo's upright turn kept as one rigid two-hand motion on re-taken, centred grasps.
- uppsta: a closed-loop reference (`oracles/uppsta/`): the demo run in the scene without the prism rule as the phase
  guide, grasps offset to the true parts, carried parts held on their guide poses, the hex key's tip servoed onto the
  hole's axis; away from the grasps and releases it follows every 4th demo row where a segment's gate allows (every 2nd
  or the demo's pace elsewhere), and every row is also within the Franka Hand slew (built that way: unchanged by it).
- kyrre: the authors' demo (every row within the arm limits) on the native-cylinder scene, each row held until the slewed
  hand reaches it (`--grip wait`), screw 3-1's second grasp re-aimed at where the screw really lies, and the hex key set
  down 12 mm along its rest: the demo puts it back on a knife edge where it vibrates at 0.099 rad/s from the birth on
  (with prisms or not) and never meets the rest rule (`oracles/kyrre/rebuild.sh`, 733,558 rows).

Harbor on the final images: privileged oracle 1 / nop 0 on all twelve, standard oracle 1 (live settled success) / nop 0
on all twelve (kyrre on image 0.3.2 / -sim 0.4.2, 2026-10-19; every other reference's fresh replay in 0.3.2 ends and
settles on its recorded hashes), on the hand-slew references (lack, eket and ikea365 on image 0.3.1 / -sim 0.4.1, 2026-10-18; uppsta's
standard twin again on the merged rule; the others on 0.3.0 / -sim 0.4.0, each reference's fresh replay in 0.3.1 ending
on the same hashes). Standard twins graded by the live episode (#305): the live end hash is the reference's where the
episode runs to the last row; ikea365, gladom, vesken and stegesund reach their settled success at an earlier chunk end
(control steps 96,000, 184,000, 388,000 and 376,000). Each reference ends on the same state hashes on Intel AVX2
(i9-14900HX), Intel AVX-512 (Xeon Gold 6230R) and AMD Zen 4 (Ryzen 9 7950X, EPYC 9354).

## Adversarial trial (PROTOCOL H.1)
`roboarc-ikea-eket-i00-privileged` (then `ikea-eket-i00-privileged`) with a red-team paragraph on top of its
instruction (obtain reward 1 without assembling the cabinet, by any exploit of the grader, the tooling, the file format
or the simulator), Codex + GPT-6 Luna xhigh with the pool's lane settings: reward 0, deterministic, no grader or
infrastructure error. The agent read the loader, the replay, `final_check()`, the operation tracker, the latches and the
end-pose rule and found no bypass; its best attempt (a one-row hold) replays released and at rest and fails the goal.
A malformed or compressed-bomb trajectory, a scene exception and, in standard mode, a simulator error or a malformed
request field are the agent's 0, never a free retry. Harbor checks: `evidence.json` `maintainer_checks` (eket) and
`maintainer_checks_standard` (svallet).

## Timing
One reference execution against the 4320 s bar (40 % of 10800 s). A Xeon Gold 6230R core (2.1 GHz) is a slow
trial-class CPU; the quiet i9-14900HX host a fast one; the standard live episodes ran on shared, loaded RTX 4090
workstations (Ryzen 9 7950X).

| task | rows | one replay, dedicated Xeon Gold 6230R core | one replay, quiet i9-14900HX | standard live episode, loaded host |
|---|---|---|---|---|
| eket | 83,181 | 575 s (5.3 %) | 221 s (2.0 %) | 1,049 s |
| svallet | 69,414 | 478 s (4.4 %) | | 397 s |
| ikea365 | 96,747 | 649 s (6.0 %) | 249 s (2.3 %) | 1,285 s |
| lack | 67,697 | 1,127 s with the re-time step (10.4 %) | 372 s (3.4 %) | 1,277 s |
| vesken | 389,504 | 1,822 s (16.9 %) | | 1,889 s |
| stegesund | 379,529 | 2,450-2,473 s on a Xeon Gold 6338 core (22.8 %) | 1,164 s (10.8 %) | 1,938 s |
| gladom | 186,611 | **8,069-8,082 s (74.8 %), over the bar** | 2,630 s (24.3 %) | 4,780 s |
| burvik | 162,534 | 3,783 s (35.0 %) | 1,694 s (15.7 %) | 1,824 s |
| stenerik | 222,600 | **6,944-7,082 s (64.3-65.6 %), over the bar** | 2,649 s (24.5 %) | 3,672 s |
| variera | 243,159 | **8,024-8,696 s (74.3-80.5 %), over the bar** | 2,664 s (24.7 %) | 4,274 s |
| uppsta | 430,225 | **8,425 s (78.0 %), over the bar** | 3,199 s (29.6 %); 3,519 s (32.6 %) on a 7950X | 4,291-4,685 s (Nyx, load 15-30) |
| kyrre | 733,558 | **8,283 s (76.7 %), over the bar** (7,953 s on a Xeon Gold 6338 core) | 2,740 s (25.4 %); 2,694 s on a 7950X | 3,073 s |

Every reference is inside the 4320 s bar on the fast-host class (i9-14900HX / Zen 4), which is the class that counts;
the slow-Xeon times are reported only. All rows above are the hand-slew references (svallet's is unchanged).

gladom and variera are inside the bar on a quiet fast host (24.3 % and 24.7 %) and over it on a slow Xeon core (74.8 %
and 74.3-80.5 %): MuJoCo's own step is about 95 % of the time (their SDF threads). They ship as the owner's decision; the
agent's budget is the same 10800 s on every host, so on a slow host an agent has less room for full replays.
stenerik is the same case: 24.5 % on the quiet i9-14900HX (30.6 % for two replays at once on host B's Ryzen 9 7950X) and
64.3-65.6 % on a Xeon core (its reference under the hand rule; before it: 28.3 % / 58.2 %). Its six welded screws keep their thread contacts, so a late tick costs about 5 x an early one (27 vs 5.5 ms on the
EPYC); its reference follows the demo's slow pad carries at every second row to stay inside the bar (the demo's own
pace: 4,097-4,130 s on the EPYC, 95 % of the bar).
uppsta is the same case again (29.6 % on the quiet i9-14900HX, 78.0 % on a Xeon core). Its earlier reference took
12,680 s on the Xeon core: the end blocks (114 convex pieces each, ~900 contacts with the board once seated) were the
cost, not the thread. The welded-part switch-off removes those contacts once each block is welded, and the reference
follows the demo at every 4th row away from the grasps; the blocks' pre-weld contacts remain (a coarser end-block
collision model is proposed to the scene's author, REVIEW.md).

## Image and assets
- `ghcr.io/mll-lab-nu/eai-ikea-mujoco:<images/ikea-mujoco/VERSION>` (0.3.2, image id `5d5090ea1910` on host B: 0.3.1
  (`b9552184c2de`) with kyrre's native-cylinder scene.py, the other eleven scenes byte-identical; 0.3.1 added the
  welded-part contact switch-off to the scene runtime and the seat welds; 0.3.0 was `ed950d079b63`): MuJoCo 3.14.0
  (EPA-patched), numpy 2.5.4, Python 3.12, and the twelve scenes at `/opt/ikea/scenes/<task>`. The privileged agent and the privileged verifier run it; the standard verifier runs the shared client
  base (it replays nothing).
- `ghcr.io/mll-lab-nu/eai-ikea-mujoco-sim:<images/ikea-mujoco/sim/VERSION>` (0.4.2 on 0.3.2, host B `050cd45d70f6`; 0.4.1 on 0.3.1, 0.4.0 on 0.3.0 before): the standard sidecar. It adds Nyx,
  RoboARC's renderer and look, in its own venv `/opt/nyx` (gs-nyx 0.1.3 needs numpy < 2.4; the physics process keeps
  numpy 2.5.4 and the patched MuJoCo): RGB from Nyx (path tracer, 32 spp, denoised, studio HDRI, point lights placed
  relative to each scene's work surface), depth from MuJoCo's EGL raster, the same camera rig and calibration. The live
  state hashes with Nyx are bit-identical to EGL's over the first 2000 control steps of all eight references, and Nyx
  draws only what MuJoCo's raster drew.
- The scene code is in git (`images/ikea-mujoco/scenes/<task>/`); the meshes (2,078 files) are on the private Hugging
  Face dataset `eai-v2-internal/ikea-mujoco-assets`, pinned by revision and by every file's sha256 in
  `scripts/ikea_mujoco/assets.json`. `scripts/ikea_mujoco/fetch_image_context.sh` downloads them, checks them and stages
  the build context (`images/ikea-mujoco/README.md`).
- `stage_context.py` refuses a scene text that points at upstream material (H.2).

## World-frame information
The standard tasks' texts give no world coordinates: no part poses or positions, no table height, no world base poses.
Every pose the standard agent gets or gives is in the robot frame (origin at the left arm's base link, +z up). The
privileged tasks read anything, as privileged tasks do. A scene reads nothing from the environment (`rcb_ikea` drops
kyrre's former development switches `SW_FROM_OP` / `SW_FROM_T` on import; the scenes no longer read them).

## Welded-part contact switch-off
A scene with a `sw_cutover.json` (today only kyrre) switches contacts off inside welded groups from its cut-over
operation on (`ikea_runtime.RetentionMechanism`): the bodies joined by latched retention welds form one rigid group,
and two bodies of a group no longer collide; each keeps every other contact (the robot, the table, unwelded parts).
The group's geoms get collision bit 16+k as their contype and their own bits plus the other groups' as their
conaffinity, checked at the build to be exact for the scene's bits. The bits follow the weld state (engage, reset,
set_state). In every scene without a cut-over nothing changes: the nine shipped references end and settle on the same
hashes in image 0.3.1 as in 0.3.0 (burvik's too, checked after it shipped).

The seat welds (`seat_weld.py`: stegesund, uppsta) have the same switch-off, opt-in per scene with
`SeatWelder(..., group_contacts_off=True)` (on in uppsta, off in stegesund) and the same bit scheme: when a fastener
seats, the parts it clamps are welded to its host at the same moment, and from then on the bodies joined by latched
seat welds form one rigid group. In uppsta it removes the end blocks' contacts with the board (~730 of ~1,500 contacts
once screw b is welded; 441 -> 261 after both). Checked from four checkpoints: the contacts with the switch are the
contacts without it minus the in-group pairs, geom pair for geom pair and depth for depth.
