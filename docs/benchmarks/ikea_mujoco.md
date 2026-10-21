# IKEA MuJoCo

Twelve IKEA furniture-assembly tasks in plain MuJoCo 3.14.0 on the CPU (float64). Two robot arms assemble each piece
with physical grasps: friction from the finger contacts, nothing attaches a part to a gripper, no assist. Every task
ships in both modes, so there are 24 Harbor task directories, `tasks/roboarc-ikea-<scene>-i00-{privileged,standard}`.

**Credits.** The scenes, their mechanisms and success checks, and the robot demos the references start from were
authored by Xiaoyu and Zhen Wu's team. The robot models are MuJoCo Menagerie's (Franka Emika Panda with the Franka
Hand; ALOHA 2 with ViperX 300 s arms). The standard cameras' studio lighting uses Poly Haven's `studio_small_09` HDRI
(CC0). The scenes' upstream provenance is recorded in `ikea/scripts/ikea_mujoco/provenance.json`.

## Scenes

| scene | robot | what is assembled | mechanism |
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
| `kyrre` | two Franka Panda | KYRRE stool: three legs placed on the upside-down seat, six screws driven with a hex key | threads (SDF), latched retention welds with the welded-part contact switch-off; MuJoCo's native cylinders |

`task.toml`: `simulator = "mujoco"`, `family = "ikea_<scene>"`, one frozen instance (`i00`, seed 0), and
`robot = "franka-panda-bimanual"` (two Franka Panda arms with the Franka Hand; stegesund and uppsta name their arms
`left/` / `right/`, the others `left_` / `right_`) or `"aloha-2"` (svallet).

## Modes

- **Privileged** (`-privileged`): the agent drives the scene through `rcb_ikea.Session` in its container: it reads any
  simulator state, resets, saves and restores states, and replays rows (`s.step`, `s.status()`, `s.final_check()`,
  `s.replay()`). The deliverable is one open-loop trajectory, `/app/output/trajectory.npz` (`ctrl` rows of the scene's
  actuators, float64, 4 physics steps per row). No GPU is needed: the agent, the verifier and the replay video run on
  the CPU.
- **Standard** (`-standard`, robot as a service, `eai-standard/2.4`): the simulator runs in a `sim` sidecar the agent
  cannot open. The agent's client (`rcb_client.Robot`) gets RGB from four ALOHA-style cameras (848 x 480, optional
  depth), proprioception, `joint_position` chunks and stateless kinematics, every pose in the robot's frame (origin at
  the left arm's base link). There is no reset and no privileged state: no part poses, table height or world
  coordinates. The sidecar needs one NVIDIA GPU with Vulkan and EGL (Nyx renders the RGB, MuJoCo's raster the depth).

## Grading

One rule in both modes. Success needs all of:
- **Goal:** the scene's own `success.py`, a port of the source assembly grader.
- **Terminal rules,** the same in every task: the parts are released (no robot link touches a part); every part is at
  rest (at most 0.01 m/s and 0.05 rad/s); the simulation never broke down (no non-finite state, no MuJoCo BADQACC /
  BADQPOS / BADQVEL warning, no part contact deeper than 10 mm).
- **End pose (eket):** every panel, the back included, ends within 10 mm and 2 degrees of its goal relative to the top
  panel (`rcb_ikea.END_POSE_RULES`).
- These hold at the end of the rows **and** again after the last row is held for 1.0 s
  (`rcb_ikea.Session.final_check()`).

**Privileged:** `tests/grade.py` replays the trajectory open-loop from the frozen keyframe in two fresh processes at
once; the task passes iff both replays pass and end in the same state hashes (`deterministic`). The verifier first
checks the scene in its image file by file against `tests/runtime_binding.json` (`instance_exact`) and by its
initial-state hash (`instance_ok`).

**Standard:** the task is graded by the **live episode**. After every `submit` and every `get_obs(seconds > 0)` the sim
service evaluates the rule; the episode ends at the first settled success (`final_check()` on a snapshot, the robot not
moved), at a breakdown, or when the wall clock runs out (then it is judged as it stands). `finalize.sh` writes
`final.json` (`graded_by "live"`, `replays 0`) and the verifier (`tests/grade.py`, no simulator) transcribes it. The
recorded trajectory (`/data/trajectory.npz`) stays a replayable record, but **no grading replay runs**. This is a
deliberate difference from RoboARC's puzzle tasks, whose standard verdict needs the live success **and** a fresh-process
replay verdict: both suites speak `eai-standard/2.4`, but IKEA's tasks carry their own variant of the service layer
(`scripts/eai_standard/v2_4_live/`), in which the live verdict stands. MuJoCo on the CPU is bit-deterministic, and
the IKEA references' end states were reproduced by fresh replays during validation (below).

**Agent failures and infrastructure failures.** The agent's 0: a breakdown, a replay that crashes or in which the
scene's code raises on the state the rows drove it into (`replay_crash`, `replay_exception`), a malformed or oversized
trajectory, and in standard mode a simulator error its rows caused (`sim_error`), all crashes inside its own calls
(`sim_crashed_on_call`), a failed settle or the timeout. Infrastructure (`infra_error 1`, a free retry under PROTOCOL
H.5): a scene that does not match its binding (`instance_mismatch`), a missing or unfinalized `final.json`, a give-up
after crashes not all inside the agent's calls, a service-side error.

**Video** (evidence only, never the verdict): privileged, `/logs/verifier/replay.mp4`, rendered by a third replay
process with MuJoCo's renderer on CPU OSMesa (512 x 288, at most 750 frames); standard, the sidecar's `/data/live.mp4`
(Nyx, `cam_high`, 10 frames per simulated second, recorded during the live episode).

## Robot speed (both modes)

Every arm joint target moves at most the robot's real joint velocity limit per control step: Franka Panda 2.175 rad/s
on joints 1-4 and 2.61 rad/s on joints 5-7; ViperX 300 s 3.14159 rad/s. It is a clamp, not a failure
(`rcb_ikea.Session.applied()`, in the tooling, the replay and the standard service alike): a row within the limit
(+1e-9) is applied unchanged, a faster one is spread over the following control steps. The Franka Hand's command
(`actuator8`) is clamped the same way at the hand's real finger speed, 50 mm/s per finger (0.1 m/s of commanded finger
width): 318.75 units/s in lack and stenerik, 182.142857 units/s in the other Panda scenes
(`rcb_ikea.GRIPPER_WIDTH_SPEED_M_S`; `bindings.json` records the per-scene values). The ViperX gripper is not limited.

## Time limits

- **Agent:** 10800 s (3 h) of wall clock in both modes (`[agent] timeout_sec`). In standard mode the service counts the
  same limit from the agent's first request (`meta["time_left_s"]`).
- **Privileged trajectory length:** at most the task's row cap, stated in its instruction: the verifier's two replays
  plus the video replay must fit its time limit.
- **Verifier:** privileged, at least 3600 s, sized so that a trajectory at the row cap replays on a slow trial-class
  CPU (one Xeon Gold 6230R core) with a 1.5x margin; standard, 600 s (it replays nothing).

One reference execution, against the 10800 s agent budget:

| scene | rows | quiet i9-14900HX | dedicated Xeon Gold 6230R core | standard live episode (loaded Ryzen 9 7950X host) |
|---|---|---|---|---|
| eket | 83,181 | 221 s (2.0 %) | 575 s (5.3 %) | 1,049 s |
| svallet | 69,414 | | 478 s (4.4 %) | 397 s |
| ikea365 | 96,747 | 249 s (2.3 %) | 649 s (6.0 %) | 1,285 s |
| lack | 67,697 | 372 s (3.4 %) | 1,127 s (10.4 %) | 1,277 s |
| vesken | 389,504 | | 1,822 s (16.9 %) | 1,889 s |
| stegesund | 379,529 | 1,164 s (10.8 %) | 2,450-2,473 s on a Xeon Gold 6338 core (22.8 %) | 1,938 s |
| burvik | 162,534 | 1,694 s (15.7 %) | 3,783 s (35.0 %) | 1,824 s |
| gladom | 186,611 | 2,630 s (24.3 %) | 8,069-8,082 s (74.8 %) | 4,780 s |
| stenerik | 222,600 | 2,649 s (24.5 %) | 6,944-7,082 s (64.3-65.6 %) | 3,672 s |
| variera | 243,159 | 2,664 s (24.7 %) | 8,024-8,696 s (74.3-80.5 %) | 4,274 s |
| uppsta | 430,225 | 3,199 s (29.6 %) | 8,425 s (78.0 %) | 4,291-4,685 s |
| kyrre | 733,558 | 2,740 s (25.4 %) | 8,283 s (76.7 %) | 3,073 s |

The budget is the same on every host, so on a slow CPU an agent has less room for full replays of the longer scenes.
MuJoCo's own step is about 95 % of the time in the threaded scenes (SDF threads, convex thread pieces).

## Physics against the source scenes

One rule for all scenes, clamp rather than fail:
- **MuJoCo 3.14.0 with the EPA fix byte-patched in.** 3.14.0's EPA writes past two fixed 24-entry horizon arrays (fixed
  upstream after 3.14.0). A dowel pressed deep into a thin convex piece of eket's side panel overflowed them (SIGSEGV
  or silent memory corruption, about 1 in 400 random eket episodes). The image patches 14 bytes of the wheel's own
  `libmujoco.so` (`source/images/roboarc/mujoco-3.14.0-epa/`, shared with RoboARC's image); it is bit-identical to
  the stock wheel on every reference and stress episode checked.
- **Cylinder -> prism:** every cylinder a robot geom can touch is a 32-facet convex prism at scene build (RoboARC's
  rule); it removed every cylinder-caused breakdown in the stress test. **kyrre is the one exception:** its six screws
  keep MuJoCo's native cylinders, which gave fewer breakdowns in a stress A/B on the same 203 episodes (1.15 % against
  2.3 % non-jerk breakdowns), and the author's demo assembles the stool only without the rule.
- **Joint-speed clamp and Franka Hand slew:** above. The source demos command targets up to 55x the real limits; every
  reference was rebuilt or retimed within them.
- **Welded-part contact switch-off:** bodies joined by latched retention welds (kyrre, from its `sw_cutover.json`) or by
  seat welds (uppsta) form one rigid group whose members stop colliding with each other; every other contact is kept.
  Checked contact for contact: with the switch, the contacts are those without it minus the in-group pairs.
- **eket's dowels:** the soft dowel shells touch only the side panels; a stiff twin meets everything else.
- **numpy / OpenBLAS SIMD masking** in the image (`NPY_DISABLE_CPU_FEATURES` without AVX-512,
  `OPENBLAS_CORETYPE=Haswell`), as in RoboARC's image, so every CPU class computes the same bits.
- A scene reads nothing from the environment (`rcb_ikea` drops kyrre's former development switches on import).

Stress test (random, jerk and prefix strategies from the frozen start state; 203 episodes per scene for the later
scenes): non-jerk breakdown rates between about 0.5 % and 2 % per scene. Residual classes, none blocking: energy created at robot
joint limits under random arm targets (vesken, eket), one table penetration (gladom, prefix-jerk), ViperX self-contact
under jerk commands (svallet).

## References

Each task has one reference for both modes, every row within the joint-speed limits and the Franka Hand slew: the
privileged `solution/trajectory.npz` and the standard `solution/oracle_actions.npz` are the same rows.
- lack and svallet: the author's demo brought within the limits by `ikea/scripts/ikea_mujoco/standard/std_reference.py`
  (lack: through the clamp, each over-fast hand row held until the hand reaches it, `--grip wait`; svallet: time-dilated
  by a slew-limited tracker).
- eket, ikea365, vesken, gladom, variera, stegesund, stenerik, burvik, uppsta: closed-loop references built on the final
  physics (`ikea/scripts/ikea_mujoco/oracles/<scene>/`, each with a README): the author's demo as the phase guide, with
  true-state feedback (damped-least-squares IK servos onto the parts' true poses, screwdrivers and hex keys on the true
  screw axes, gates and local searches), recorded as the executed rows.
- kyrre: the author's demo on the native-cylinder scene, each row held until the slewed hand reaches it, one grasp
  re-aimed at where the screw really lies and the hex key set down off its knife-edge rest
  (`oracles/kyrre/rebuild.sh`).

Validation on the final images: Harbor oracle 1 / nop 0 in both modes for all twelve scenes; each reference ends on the
same state hashes on Intel AVX2 (i9-14900HX), Intel AVX-512 (Xeon Gold 6230R) and AMD Zen 4 (Ryzen 9 7950X, EPYC
9354); in standard mode the live end state is the reference's where the episode runs to the last row (ikea365, gladom,
vesken and stegesund reach their settled success at an earlier chunk end).

An adversarial trial (PROTOCOL H.1) on `roboarc-ikea-eket-i00-privileged`, with an instruction to obtain reward 1
without assembling the cabinet by any exploit of the grader, the tooling, the file format or the simulator, scored 0:
the agent found no bypass. A malformed or compressed-bomb trajectory, a scene exception and, in standard mode, a
simulator error or a malformed request field are the agent's 0, never a free retry.

## Images

| image | used by | id (1.1.0) | manifest digest (the tasks' pin) |
|---|---|---|---|
| `ghcr.io/mll-lab-nu/roboarc-ikea:1.1.0` | privileged agent and verifier (+ OSMesa and ffmpeg for the replay video); the standard sidecar's physics | `9f9cbab8aa74` | `sha256:2c56d9226aaf…` |
| `ghcr.io/mll-lab-nu/roboarc-ikea-sim:1.1.0` | the standard `sim` sidecar (+ Nyx in `/opt/nyx`, EGL, Vulkan) | `cbd929360257` | `sha256:f22a1e9042c7…` |

Both are built reproducibly from fully pinned inputs (the Python base image by digest, Debian packages by version from
the Debian snapshot archive, wheels by sha256; the sidecar `FROM` the base by digest) and hold only what the tasks run
(`ikea/images/ikea-mujoco/README.md`). The 1.0.0 images (`5d5090ea1910`, `050cd45d70f6`) hold the same runtime: every
file the graded runtime loads is byte-identical.

The standard agent and verifier containers run the shared standard client base (`scripts/eai_standard/client.env`).
The scene code is in git (`ikea/images/ikea-mujoco/scenes/<scene>/`); the meshes are on a private Hugging Face dataset,
pinned by revision and by every file's sha256 in `ikea/scripts/ikea_mujoco/assets.json`. Running the tasks needs only
the images; rebuilding them needs the meshes (`ikea/images/ikea-mujoco/README.md`).

## Files and checks

| path (under `ikea/`; the task dirs are `tasks/roboarc-ikea-*`) | what |
|---|---|
| `scripts/ikea_mujoco/mk_tasks.py`, `mk_standard_tasks.py` | the generators (privileged first: the standard twins share its reference) |
| `scripts/ikea_mujoco/{tasks,assets,bindings,evidence,standard_evidence}.json` | the generators' inputs: texts, scene pins, what each scene builds to, the Harbor checks and timings |
| `scripts/ikea_mujoco/shared/`, `standard/` | the tooling and privileged grader; the sim service, Nyx render process, standard grader, oracle client |
| `scripts/ikea_mujoco/oracles/` | the references and how they were built |
| `scripts/ikea_mujoco/REVIEW.md` | the decision log |
| `images/ikea-mujoco/` | the scenes and the two image recipes, `build.sh` |

```bash
bash ikea/ci/check.sh [--image-tests]     # both generators' --check (+ the grader unit tests inside roboarc-ikea)
harbor run -p tasks/roboarc-ikea-svallet-i00-standard -a oracle
```

Since 1.0.1 the IKEA tasks are rendered with the repository's task layer (`scripts/task_conventions.py`), like the
puzzles: the shared closing sections of the instruction, the resource-monitor sidecar, declared resources, limits,
caps and seeds (`scripts/task_resources.toml`, the `^roboarc-ikea-` rules: privileged CPU only, the standard sim one
GPU), a `DIGEST`, and the one egress list (`scripts/set_agent_egress.py`: the model APIs and the package registries,
which the instruction's "External material" names; 1.0.0's IKEA allowlist had the model APIs only). Every check in
`ci_checks/` covers them. Their standard layer is the live-grading variant `scripts/eai_standard/v2_4_live/`. Results
before and after 1.0.1 are not comparable (instruction and egress changed; graders, scenes and references did not).

**Paths cited inside the task files.** Since 1.0.1 the generated texts cite this repository's paths. Files copied
unchanged whose bytes stay as released (the tooling `rcb_ikea.py`, the Nyx render process `ikea_nyx.py` and its
`nyx_direct.py`, the live-grading pair) still cite the development tree: `scripts/ikea_mujoco/…` and
`images/ikea-mujoco/…` = `ikea/scripts/ikea_mujoco/…` and `ikea/images/ikea-mujoco/…`; `docs/protocol/standard_mode.md`
= `docs/STANDARD_MODE.md`; `docs/rcb/RENDER_BRIDGE.md` = `docs/simulator/RENDER_BRIDGE.md`;
`scripts/robotangle/shared/` (the Nyx driver's origin) = `scripts/roboarc/shared/` ([docs/rcb/README.md](../rcb/README.md)). The privileged READMEs' Harbor-check lines name the image by its pre-1.0
name `eai-ikea-mujoco` (the same ids).
