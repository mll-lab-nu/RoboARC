# cast_keyhole on MuJoCo

The port itself (profile keyhole_v1 / v2, MuJoCo IK/FK for the family's oracles, the oracle plan changes) is described
in an internal port note, not included in this repository. Shared mechanisms are in
[MUJOCO_CORE.md](../MUJOCO_CORE.md). This note (2026-10-10) covers the -i01 references on the current physics.

## The task

Two 9 mm cast plates, gold and silver, interlocked through the silver's keyhole channel (ALOHA, assisted grasp
`contact_weld_v1`). Separate the silver and hold it inside the goal area, the gold outside on the puzzle side. Five
released tiers: very_easy and easier (simplified straight channel), easy F1, medium F2 (eccentric window), hard F3.
Goals, rules, geometry and difficulty are the released ones.

## Profile keyhole_v4 (`topogym/world/mujoco/profiles.py`, `_KEYHOLE`, key `("cast_keyhole", None)`)

- keyhole_v2 (the claws v2 contacts, `prism_partition`, solimp .99 / .99, impratio 50 against friction creep) plus
  the ALOHA follower finger on the stiff coupling (`robot.coupling = ALOHA_COUPLING`, porting guide trap 10).
- `prism_partition = {"fatten_m": .0005}`: hard's gold had three interior partition slivers 0.456-0.463 mm wide
  (pieces on both sides of the thin direction, so the plate is not thin there); re-cut exactly with a neighbour, the
  thinnest piece is 1.934 mm. The other tiers have no piece under 0.5 mm (easy's thinnest 0.581 mm) and are unchanged.
- The exact constraint diagonal (b1ec43997) applies, as to every family.
- **keyhole_v4 (2026-10-11): robot-robot contacts on the soft class** (`robot.self_contact = SOFT_CONTACT`, 5 ms
  overdamped, as claws v4). keyhole_v3 kept `_RIGID`'s 20 ms class, and the acceptance round on df00ee35b drove one
  ALOHA arm more than 10 mm into the other with random joint targets: easier random_e57 (a left finger 11.4 mm into
  the right gripper base, 163 N; BREAKDOWN in two fresh processes) and hard random_e47 (a left finger 13.7 mm into a
  right finger; BREAKDOWN in a fresh process). On v4 the same streams read 3.4 and 4.7 mm, CLEAN, each in two fresh
  processes with one final state hash. The references touch no robot-robot pair:
  rebuilt on v4, every tier keeps its birth hash, and every reference below replays to its recorded final state hash.
  The shrunk stream is `source/scripts/robotangle/stress/corpus/reproducers/J1_keyhole_easier_arm_arm_depth`.

Scene acceptance (make-instance, nop 400 + 40 under the stress observer): birth hashes as recorded at the port; no observer flags; the released standing assembly topples at
birth (15-22 mm, medium 3 mm) and rests within 0.45 s. Accepted: on very_easy and easier the toppled gold then leans
with its friction saturated and slides on at about 1 um/s (profile comment). The authored-pose clearance in MuJoCo
equals the declared one (4.807 / 4.807 / 3.900 / 0.630 / 3.900 mm).

Rendering: nyx_direct (`TASKS["cast_keyhole"]["renderer"] = "direct"`); at birth its input buffers equal the Genesis
host's bit for bit.

## References (-i01)

The family's own closed-loop oracles, unchanged, run live in the recording Session on the current physics:
- very_easy, easier: `harbor_bridge.below_easy` -> `cast_keyhole.public_session`, SimplifiedKeyholePolicy;
- easy, medium, hard: `cast_keyhole.harbor_session` -> `runner.worker`, CastKeyholePolicy with its two regrasps, the
  coupled-IK route along the released reference poses, closed loop on the measured pieces.

No plan change was needed. The hard stall reported at the port (pull-through jammed at row ~12600, weld 35-40 N) was
the k4 reference replayed open loop on the exact diagonal; the oracle re-run closed loop
on the new physics (the port note's second suggestion) passes the channel with the welds' 20 ms mean at most 6.7 N.

Recipe (the RoboTangle image's Python; `RCB_RENDER=off` for the oracle and replays):
```
rcb_topogym.py make-instance --task cast_keyhole --variation cast_keyhole_<tier> --scene-seed 0 --gripper-space native \
    --out <dir>/instance.json
KEYHOLE_ORACLE_OUTPUT=<dir>/oracle_out rcb_topogym.py run-oracle --task cast_keyhole --instance <dir>/instance.json \
    --code source/scripts/robotangle/references/cast_keyhole/oracle.py --out <dir>/privileged.npz
rcb_topogym.py replay --task cast_keyhole --instance <dir>/instance.json --traj <dir>/privileged.npz --repeat 1 --strict  # x2, fresh processes
source/scripts/robotangle/stress/replay.py --instance <dir>/instance.json --actions nop:400 --hold 40 --out <nop.json>
source/scripts/robotangle/references/cast_keyhole/weldforce.py <dir>/instance.json <dir>/privileged.npz <out.json> \
    --phases <dir>/oracle_out/private/keyhole_monitor.jsonl
render_reference_video.py --instance <dir>/instance.json --traj <dir>/privileged.npz --out <video dir> \
    --label "cast_keyhole <tier>" --expect-sha <final sha> --renderer direct
```

Recorded on 53a93693f (keyhole_v3 with the re-cut) + the oracle code of bb59e378f, as
`scripts/robotangle/oracles/cast-keyhole-<tier>-i01/` (`privileged.npz`; `standard.npz` is the same file; development
names, the tier-named -i01 directories are not included in this repository):

| tier | birth sha | rows | sim s | final state sha | npz sha256 | weld 20 ms mean max L / R (N) |
|---|---|---|---|---|---|---|
| very_easy | 08c02b189f049644 | 5097 | 127.4 | c42c5ce8471f3cd5 | 221c1acad51e457cea0173ee51bfc92fa5a3264e04419e799325723626816b2e | 10.8 / 14.8 |
| easier | 1540c17baba15f29 | 5886 | 147.1 | 9b71c74185a38a2e | 39f7be03018997fa0286b55101e94a865151bc23d287545875485eda2c74c4f7 | 10.4 / 14.8 |
| easy | 13a2f27bfb843313 | 12202 | 305.1 | 4fa96f4174791565 | d120ab53e8dfa8135b7631e678788480992a13862a082eebfdc4792beca33c84 | 28.3 / 26.9 |
| medium | b0445a423499b6e8 | 10803 | 270.1 | e6fd15efeaf27fef | 507636a41c2ece44a73d93cdbf71e977343574d08f7fd5cf7c5c00e61eb1f318 | 26.5 / 36.9 |
| hard | f614489fc39e64a5 | 14587 | 364.7 | ea285c0401b166f1 | 1876c326668b671a14826921f4792ff03d2298b821eb5a26e042f0f69ac80a30 | 32.8 / 27.5 |

Every tier: the oracle succeeds; two fresh-process replays end on the recorded hash with success (strict, the instance
exact); nop 400 + 40 is not a success and the observer flags nothing; the nyx_direct video replays to the same hash.
The weld peaks are the regrasp releases and the lift, all under the 60 N slip rule (0 slips); through the precise
channel stages at most 5.4-6.7 N. The oracles' own choices: pickup opening .028 / raise 3 mm (very_easy, easier);
the left regrasp by joint transit on easy and hard, and goal drops 6 / 0 / 2 cm (easy / medium / hard).
Evidence (oracle logs, replays, nops, weld scans, videos) was kept outside this repository.
