# orbits on MuJoCo

Shared mechanisms are in [MUJOCO_CORE.md](../MUJOCO_CORE.md); this note (2026-10-10 / 11) covers what is specific to
orbits and why. The evidence was kept outside this repository.

## The task

A closed cord loop hangs from a wooden handle and is threaded around a fixture of posts (each with a stopper ball) and
rings. The ALOHA frees the whole cord and handle and places them in the goal area. Six settings:

| settings | cord | assisted grasp |
|---|---|---|
| `very_easy` (one ball post), `easier` (one ring, two posts) | rigid chain (`assets/geometry/orbits_chain.py`): capsule links r 1.87 mm, 16.5 mm long, on ball joints (damping 2e-4, armature 1e-5, bend stiffness of E 100 MPa), the handle the free root, two branches tied at the middle by a `<connect>`; easier's ring collides as a loop of 48 capsules (9249ebffc) | contact weld on any link (an 8 mm sphere at each link's middle), `max_translation_error_m` 15 mm |
| `easy`, `medium`, `hard`, `reference85` (FEM on Genesis) | on MuJoCo the cord is the same kind of chain (task parameter `cord_model: chain`, set by `rcb_topogym.simulator_config`) along the setting's authored cord, joints from the FEM material; the rings keep their mesh | none (unassisted, as the FEM contract) |

Rules of the chain settings, as on the development line (`below_easy_v12_tie_3mm_v1`): the middle tie stays within 3 mm, the weld
may drift 15 mm. Both were loosened on the Genesis line (from 2 mm and 10 mm) for easier; the measurements below show
what MuJoCo needs.

## Profile (`profiles._orbits`)

- `_CHAIN = _CLAWS` and 0.625 ms steps, as slotted_post_ring; the cord entity's geoms take the claws contact class
  through `mjcf` (`_CHAIN_GEOMS`).
- `birth_overlap_exclude`: the two links at the middle tie overlap at birth and are excluded from contact (Genesis'
  neutral-overlap filter).
- `mjcf_equality`: the tie nearly hard (1.25 ms, dampratio 1, solimp .99 .99); the authored 5 ms is Genesis' soft
  constraint.
- `static_contact.keep`: posts (8.2 mm), ring tubes and stopper balls keep the task's contact against the robot.
- Rings: the FEM settings' chains collide the ring mesh as its 128 exact convex segments (`orbits_chain.ring_parts`);
  easier's 48 capsule meshes are convex and collide as declared (profile `orbits_chain_v4`).
- `speed_clamp`: extra damping on a cord link above 4 m/s or 150 rad/s (below).

## Speed clamp (orbits_chain_v3 / v4 / fem_v2, 2026-10-11)

The stress validation's `prefix_e1` on very_easy (the reference's first 2654 rows, then 200 rows that fling the cord at up
to 8.0 m/s) was a BREAKDOWN on orbits_chain_v1 (8fa6c8edc, final state 28054e21e425068b). At tick 27340 the handle
lands on the ground (90 N) and is spun up to 120 rad/s (the fastest spin of any cord body the tick before: 67 rad/s);
within that tick twelve links (link_11 .. link_22) gain 0.9-1.7 m/s. Rule 6 of the stress criteria asks that the cord's kinetic energy rise by no
more than the outside work on it; it missed by 0.046 J, so the links' jumps were uncaused. No energy was created over
the run, and nothing ran away.

What the tick's books say (`energy_probe.py`, a development tool not included in this repository: every force class's work on the cord's
own dofs, per MuJoCo step, with the velocity averaged over the step):

| term | J |
|---|---|
| change of the links' kinetic energy (the observer's reading; equal to a fresh mj_forward at the tick's two ends) | -0.0986 |
| ground on handle and links, as the integrator applied it (generalized force x mean generalized velocity) | -0.1159 |
| the same contacts as the observer books them (force x mean of the contact point's velocity in the step's start and end states) | -0.1501 |
| joint springs / joint damping | +0.0039 / -0.0011 |
| kinetic energy the step added beyond every force's work (the mass matrix changes with the pose within a step) | +0.0231, of which gravity 0.0018 |
| kinetic energy that went into the joints' armature (not in the observer's reading) | +0.0085 |
| middle tie, joint limits, link-link contacts | 0 |

So the 0.046 J is 0.034 J between the two ways of booking the ground's work on a handle that turns 0.04-0.07 rad per
0.625 ms step, plus 0.021 J of the step's own energy error in the landing tick (in free flight it is within 0.001 J
per tick), less the 0.0085 J taken up by the armature and 0.001 J of damping.

The clamp (`MujocoWorld._clamp_apply`, `MUJOCO_CORE.md`) keeps a flung cord from landing that fast: a link above
4 m/s gets damping on the excess of its centre-of-mass speed, one above 150 rad/s on the excess of its spin.

| thresholds | `prefix_e1` | clamp |
|---|---|---|
| none (orbits_chain_v1) | BREAKDOWN, 25 uncaused jumps at tick 27340 | - |
| 6 m/s, 300 rad/s | BREAKDOWN, 18 uncaused jumps at tick 27338: the landing whips links from 5 to 8 m/s within one tick | 0.083 J removed |
| 4 m/s, 150 rad/s (set) | VIOLENT (speed 4.2 m/s, cord off the table), no uncaused jump; final state e552d86f4533c6d9 in three fresh processes (on 9b5c7c28b's rebuilt instance: VIOLENT, d8f8bbf38fd2bc97, the stress validation's two fresh processes) | 0.235 J removed over 869 steps from tick 26805 |
| 3 m/s, 120 rad/s | VIOLENT (speed 3.2 m/s), no uncaused jump | 0.317 J removed |

Since 2026-10-13 (7625c2156c; profiles orbits_chain_v5 / v6 / fem_v3) the thresholds are 3 m/s / 120 rad/s, as
slotted_post_ring's: the easier acceptance round's prefix7_e2 (a cord landing at ~124 rad/s) read rule 6 short by ~4 mJ,
traced to the MuJoCo step's own energy error in that landing (+3.9 mJ) plus 0.64 mJ of armature release, no contact
creating energy; at 3 / 120 it is VIOLENT (590bd82d9350237b), O1 too (cd7a7a313b4ffc9e), and the references never
activate the clamp (they replay to the hashes below).

The references' maxima are 2.16 m/s (easier, link_57) and 80 rad/s (easier, the handle), so they never reach the
thresholds: both replay to their recorded hashes with 0 activations. The stream is reproducer
`O1_orbits_very_easy_handle_landing_whip` of the stress corpus. The clamp bounds how fast a cord can land; it does not
remove the landing's bookkeeping gap itself, so a cord flung below 4 m/s whose handle lands spinning can still show a
smaller one.

## easier on a 0.3125 ms step: orbits-02-i01 (V3, 2026-10-17/18)

easier's acceptance round FZ3 (v2.13b, 300 episodes, on the 0.625 ms step) failed on one confirmed BREAKDOWN,
`prefix12_e2` (reproducer `O2_orbits_easier_handle_landing_step_residual`): 8 uncaused jumps at tick 59647. The stream
reproduces from the frozen birth a3659d4f47244929 to 69a97615d604cc65 on eai-roboarc 0.1.0 (ef4df961a2) and 0.2.0
(docker on host B, the 0.2.0 SIF on the cluster); a fresh make-instance on 0.2.0 gives the same birth.

What happens (the integrator's books, `energy_probe.py` on the saved states of ticks 59500-59700): the cord,
flung at up to 3.17 m/s (the speed clamp holds it at 3 m/s), falls; its handle (14.7 g, the chain's free root, spinning
~40 rad/s) lands on the table. In the landing tick the static contacts do -12.83 mJ on the cord while its kinetic energy
(armature included) drops only 10.93 mJ: the implicitfast step's own energy residual is +2.31 mJ (then +0.14, +2.63,
-4.54 mJ; ~0.03 mJ per tick in the free flight before). The links next to the handle (link_02 .. 07, 71, 72) whip
+0.6-0.9 m/s; rule 6's look-back balance of the cord misses by 2.09 mJ, over ENTITY_EPS_J (1 mJ), so the jumps read
uncaused. No contact created energy (X <= 0, energy_created 4.3 mJ).

The whip is physical, the residual numerical. From the same pre-state, with the MuJoCo step halved 1-3 times (the tick
and the control period unchanged): residual 2.31 / 1.05 / 0.54 / 0.27 mJ (first order in dt), while 10-16 links still
jump by more than 0.5 m/s. The stream's own continuation (rows 0-5949 replayed in-process, then the step changed, the
observer attached; the 0.625 ms control of it reads BREAKDOWN x8 as the full run): 0.3125 ms VIOLENT (whip
redistribution), 0.156 / 0.078 ms VIOLENT; the implicit integrator at 0.625 ms (residual 1.06 mJ, as dt/2) VIOLENT too.
Per control step (the cluster, the reference's first 1500 rows): 0.625 ms 0.266 s, 0.3125 ms 0.457 s (1.72x), implicit
0.351 s (1.32x).

The fix: easier's MuJoCo step is 0.3125 ms (solver_substeps 8 in 2.5 ms ticks), frozen with its instance
(`make-instance --simulator-overrides {"solver_substeps": 8}`; mujoco_release.json new_bases.orbits-02-i01
simulator_overrides, which release_rebuild.py now passes for new bases). It needs no image or tooling change: every
other base, orbits-01-i01 included, keeps its physics and its reference (27d5ce82a052442d on 0.2.0). The integrator and
the clamp are profile settings in the image; the implicit integrator also changes the references' paths and was
measured no-go before (2026-10-14, excluded tiers: +13-29 %).

| | value |
|---|---|
| instance | orbits-02-i01 (variation orbits_easier, seed 0, width gripper space), birth d2f6fa837d367f5b, made in eai-roboarc:0.2.0 |
| reference | `oracles/orbits-02-i01/privileged.npz` (= standard.npz): the unchanged easier oracle (`orbits_session_oracle.py`), 7918 rows, 197.9 s, end b9c68d6502105df4, sha256 a87ba3fc4e37f523; largest arm-joint target step 0.039 rad (0 rows over the 0.04 rad joint-speed rule) |
| fresh replays | b9c68d6502105df4, success, strict success, in three fresh processes (host B x2, host A) plus the recording (the cluster) |
| timing | host A (Ryzen 9 7950X, `--cpus 1`, load 11-17): 1754 s = 16.2 % of 3 h (0.222 s per control; texts state 0.23 s); the same window: orbits-01-i01 260 s, the 0.625 ms easier reference 1035 s. On host B at load 50-60: 3802 / 3897 s (orbits-01-i01 551 s there) |
| stress round FZ4 | v2.13b, the cluster (the eai-roboarc 0.2.0 SIF), 1000 episodes (seeds 126001-126046, FZ3's strategy mix twice: 46 workers, equal share per strategy), 493,068 controls: **PASS**, 0 confirmed BREAKDOWN (16 candidates confirmed in fresh processes: degenerate-row signatures and fast whips, VIOLENT or CLEAN); reference success CLEAN deterministic (96.8 min with the observer on a loaded node), nop CLEAN no success; 0 grasp slips; P(miss) at a 0.5 % rate 0.0067 |
| visual | the reference video (oblique + top, 3x) and the fastest stress episode (prefix6_e4: the cord flung at 4.3 m/s, falling back over the fixture) look physical: no kink, no tunnelling, the loop lies flat |
| Harbor | oracle 1 / nop 0 in both modes on the 0.2.0 runtime (a local image = eai-roboarc:0.2.0 + this instance; rendering off; host A): oracle trials 3992 / 3991 s, end b9c68d6502105df4 in every replay |

Shipped in RoboARC 0.2.1 (V3 release, 2026-10-18): the texts and the demo were approved (2026-10-18). The instance
is in the assets dataset (revision 9f34f894a454) and in the image; the release build is
`release_rebuild.py` with `records/freeze_manifests/freeze_manifest.v3.json` (step_s 0.23).

## Settling (frozen with the instance)

As slotted_post_ring: `rcb_topogym.MUJOCO_CHAIN_SETTLE_S`, frozen by `make-instance` (`mujoco_settled_state`). very_easy
and easier need 1 s and keep 3 s; easy / medium / hard / reference85 need 24 / 28 / 28 / 14 s and take 35 s.

## Acceptance of the scenes (seed 0, width gripper space)

nop 400 + hold 40 under the stress observer, verdict v2.8; drift over the first 10 s after frame 0.

| setting | state_sha | verdict | deepest contact | 10 s drift | thinnest geom |
|---|---|---|---|---|---|
| very_easy | ca700d18ffbe2f03 | CLEAN | 0.006 mm | 0.02 mm | 2.0 mm |
| easier | a3659d4f47244929 | CLEAN | 0.007 mm | 0.08 mm | 2.0 mm |
| easy | 7ee0e405a76f8993 | CLEAN | 0.01 mm | 0.61 mm | 3.1 mm |
| medium | 30779543e737a10c | CLEAN | 0.01 mm | 0.23 mm | 3.1 mm |
| hard | c6e5275949eab93f | CLEAN | 0.01 mm | 0.65 mm | 3.1 mm |
| reference85 | e3eba3a76009ea8e | CLEAN | 0.01 mm | 0.41 mm | 3.1 mm |

On every setting: nop is not a success, no energy created, the ring holes are open, no geom under 0.5 mm, links and
handle collide with the ALOHA, and the restored birth hash equals the settling Session's.

## Rendering

nyx_direct (`TASKS` renderer "direct"); the FEM settings are drawn as the FEM setting (golden rope, wooden handle)
through `nyx_direct.chain_cord_host`. That is appearance seed 0's look; seeds 1 and 2 re-finish the cord, the fixture
and the handle (orange polypropylene, walnut-stained, lacquered ebony; blue nylon, black lacquer, oiled mahogany:
`cord_finish`, `orbits.FIXTURE_FINISHES`). Against the bridge's Genesis host: input buffers bitwise but rounding at birth,
RGB within the host's own run-to-run spread (at most ~110 pixels > 8 levels per camera, as host against host).

## References (`scripts/robotangle/oracles/<base>-i01/`, `standard.npz` = `privileged.npz`)

Recorded live on MuJoCo by `planning/orbits/native_session.py` (run_feedback_policy) through `rcb_topogym.py run-oracle`
(`tools/references/orbits/record_reference.sh`, oracle code `tools/references/oracle_runners/orbits_session_oracle.py`, not included in this repository)
on the width-space instances above. Each replays in two fresh processes to the recording's hash. (`orbits-very-easy-i01`
is a development name, not included in this repository under that name.)

| base | instance state_sha | rows | final state_sha | largest tie gap | largest weld drift |
|---|---|---|---|---|---|
| orbits-very-easy-i01 | ca700d18ffbe2f03 | 3330 | 27d5ce82a052442d | 0.018 mm | 0.57 mm |
| orbits-easier-i01 | a3659d4f47244929 | 7557 | 558c9a476e4e5f92 | 0.38 mm | 3.69 mm |

Final states on instances rebuilt at 9b5c7c28b (ccd4ba7fa: the soft class at a constant impedance, kept static pieces
at solmix 2^60), two fresh strict replays each; the recordings are unchanged. The tie and weld columns were measured on
the recording build (final states 70a69e284fd8a0e3 / 4121dc0fe560992f). An instance frozen before ccd4ba7fa still
replays to those old final states on the new head: its kept static pieces (posts, rings, stopper ball) carry solmix 1e9,
and `scene.migrate_contacts` moves only kept pieces built at solmix 0 to 2^60. Against a cord link with the same
contact parameters, 1e9 leaves an ulp-level mix where 2^60 gives the link's parameters exactly, so the two builds part
in the last bits at the first contact of a kept piece with a cord link or a grasp pad (both at solmix 1).

So on MuJoCo both references stay far inside the limits these settings had before the Genesis line loosened them (tie
2 mm, weld 10 mm). The limits themselves are unchanged here (3 mm, 15 mm, as imported); whether to tighten them back is
still to be decided.

(`references/orbits/measure_tie_weld.py`: a fresh-process replay measuring, after every control step, the middle tie's
gap and every held weld's translation error.)

Both references also pass the stress observer as single replays (`stress/replay.py --hold 0`, verdict v2.9): CLEAN
(peak task speed 0.88 / 2.16 m/s, deepest task contact 0.37 / 1.85 mm); on 9b5c7c28b's rebuilt instances CLEAN under
v2.10 (the stress validation's head check).

Oracle changes for MuJoCo (plans only):
- shared with knot: `KnotIK` runs on MuJoCo entity views (FK, collision pairs, geom poses);
- easier: leg pinches at a link's middle (its weld region) instead of its root, and a first pinch that has not welded
  within 1 s is repeated; the carry to the set-down is solved ahead and keeps the arm's IK branch; the 40 deg turn of the
  held loop toward the goal is a rigid turn about the grips' midpoint in 8 steps (one straight step loaded the left weld
  to 14.8 mm and the release whip opened the tie to 3.2 mm).

The FEM settings (easy, medium, hard, reference85) never had a passing oracle on any engine.
