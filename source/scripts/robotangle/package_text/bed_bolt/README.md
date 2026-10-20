# bed_bolt: task texts of the family's new base

bed_bolt (`docs/rcb/families/bed_bolt.md`) is a new family with no released base. Its first setting, `bed_bolt_crossbar`
(experimental, no tier), gets its own base, `bed-bolt-01-i01` (the family's number 01 under the RoboARC names
`<family>-<NN>-i01`; named `bed-bolt-crossbar-i01` until 2026-10-17), made from the released ALOHA base
`interlocked-claws-easy-i00` as its template (`scripts/robotangle/release_rebuild.py` new_bases). These directories hold
the new base's `task.toml`, `README.md` and `instruction.md` for both modes; every other file of a package (the
Dockerfiles, the compose files, `tests/test.sh`) is the template's, and the rebuild writes `runtime_binding.upstream.json`.
The template is another family's base: the rebuild replaces its family by bed_bolt where the grader and the sidecar start
(the privileged `tests/test.sh`'s `grade.py --family`, the standard compose's `RCB_TASK`) and refuses the base while any
other text still names interlocked_claws.

- Not on the release list until the user says so. A temporary manifest makes the base, for local generation and for
  Harbor under its name, on a test-only commit of a temporary branch that is never merged: a bed_bolt family entry with
  `"tiers": []` and this new base, then `release_rebuild.py <commit> --version <v> --families bed_bolt`:

  ```json
  "bed_bolt": {"tiers": [], "note": "bed_bolt_v2 (claws_v6 + the bolt's and key's own MJCF contacts, the thread stop kept hard): no overrides",
    "new_bases": {"bed-bolt-01-i01": {"template": "interlocked-claws-easy-i00",
      "variation": "bed_bolt_crossbar", "package_text": "scripts/robotangle/package_text/bed_bolt",
      "design": "a bed frame's corner bolt: the long bolt carried to the blind hole and pushed in until the thread catches, then driven home with an L-shaped hex key, its short arm in the head's socket and the long arm swung 60 degrees at a time (86 strokes)"}}}
  ```

- Each file is Cast Trinity's text of the same template (`scripts/robotangle/package_text/interlocked_claws/
  interlocked-claws-cast-trinity-i01-<mode>/`, the current wording of an ALOHA new base) with every task line replaced,
  each value from the bed_bolt instance (birth 5304a7d2b2771443) or from what the tooling generates for it
  (`runs/robotangle_mujoco/bed_bolt/20261016/tools/mk_package_text.py <instance.json> 0.024`, run in the RoboTangle
  image from the worktree root):
  - the names (`topogym-bed-bolt-01-i01-<mode>`, the family, the variation, the Session and replay commands);
  - the task sentence and the success requirements: the instance's public goal (`goal_description.describe_goal`,
    `BedBoltTask.goal_words`), with the user's two rules of 2026-10-16 (a push or pull along the axis does not advance
    the engaged bolt; fingers cannot turn it, the key does);
  - the grader details: `scripts/topogym_rules.py` for bed_bolt (rigid contact review and breakdown depth, the rules'
    tick, the two grasp regions named by part), the goal-square and separator lines dropped (no placement goal);
  - the grasp line: the bolt and the key graspable, the tool frame within 15 cm, the grippers start empty; and the
    mechanism's veto: once the thread has caught the bolt is no assisted-grasp target (a hold on it is lost then);
  - the objects line: `bed_frame` (fixed), `bolt`, `key`; privileged agents read the thread's state from the world state;
  - the implementation reference: `BedBoltTask.evaluate_state` (`topogym.tasks.bed_bolt.common`);
  - the per-step wall clock, in both modes (the standard "Time per call" line keeps its wording): 0.024 s, MuJoCo, from
    Harbor's run of the -i01 reference under this base on the trial host (2026-10-17, a lab RTX 4090 host, AMD Ryzen 9 7950X,
    16 cores / 32 threads, load average 39-40 just after; image `eai-robotangle:0.3.0-dryrun-bed-bolt-1`, test-only
    commit 2741dcc233 under the base's earlier name, rendering off): the standard live episode (`episode.json`
    `elapsed_s`) 607.0 s for 25 325 control steps (0.0240 s). The same reference elsewhere: that run's privileged
    grader, its two replays in 1508.3 s (0.030 s; it ran alongside the standard trial); two fresh strict replays on
    a lab RTX 4090 host, 443 s and 463 s (0.018 s); Harbor's standard live episode under bed-bolt-01-i01 on a lab RTX 4090 host (load
    average about 17), 417.5 s (0.0165 s). These texts, with this value, went through Harbor under bed-bolt-01-i01
    (2026-10-17, test-only commit 9c040f7816 on f45025639c): oracle 1 and nop 0 in both modes.

## The drives' bases (2026-10-18)

The user's variants of bed_bolt that differ only in the tool's tip and the matching recess in the bolt's head
(`runs/robotangle_mujoco/bed_bolt/20261018_variants/DESIGN.md`) each get their own base, numbered in the order they
were made. Their texts come from the same generator, made to take the base, the variation and the class
(`runs/robotangle_mujoco/bed_bolt/20261018_variants/tools/mk_package_text.py`, environment `BB_BASE`, `BB_VARIATION`,
`BB_CLASS`; the template is Cast Trinity's under its RoboARC name, `interlocked-claws-13-i01`); since the key's slip was
removed from the family (2026-10-18) its copy `runs/robotangle_mujoco/bed_bolt/20261018_noslip/tools/mk_package_text.py`,
whose objects line names no key slip, writes every base's texts, 01's included:

| base | variation | tool (the task sentence's words) |
|---|---|---|
| `bed-bolt-02-i01` | `bed_bolt_torx` | an L-shaped Torx key (a six-lobe star tip) |
| `bed-bolt-03-i01` | `bed_bolt_square` | a ratcheting L-shaped square key (a square tip on a ratchet); one more line on how the ratchet behaves |
| `bed-bolt-04-i01` | `bed_bolt_phillips` | a ratcheting L-shaped Phillips key (a cross tip on a ratchet); the ratchet's line as the square's |
| `bed-bolt-05-i01` | `bed_bolt_slotted` | a ratcheting L-shaped slotted key (a flat blade on a ratchet); the ratchet's line, and one more: the slot runs across the whole head and is open at both ends, so the blade can slide out along it |

- Only the task sentence's tool differs from 01's (the ratcheting key's bases add one line on the ratchet), and the
  names, the variation and the implementation class (`BedBoltTorxTask`, `BedBoltSquareTask`, `BedBoltPhillipsTask`, `BedBoltSlottedTask`).
- The robot safety line carries the template's present wording (an arm more than 5 mm deep in a fixed object is
  stopped, `static_guard`), which Cast Trinity's texts gained after 01's first texts were written; 01's texts carry it
  too since they were regenerated without the slip (7eed8ef8e6).
- The per-step wall clock is 01's 0.024 s until the drive's own Harbor run measures it.

## The task sentence (2026-10-19)

The task sentence said the bolt is driven "until its head seats on the block", while the grader takes a head within 1.5 mm
of the block face (the first success requirement says so). The metric review of 2026-10-19 (B1; the user: text fixes
only, no rule changes) made `BedBoltTask.goal_words()`'s summary say "...until its head is within 1.5 mm of the block
face." (the number from `seat_tolerance_m`). The five bases' texts were regenerated with the same generator
(`runs/robotangle_mujoco/bed_bolt/20261018_noslip/tools/mk_package_text.py`, step time 0.024 s) on instances made on the
branch: the births are the references' (01 0bfc2739be27db3a, 02 88e0c83081d83d55, 03 83a09f9d160239c0, 04
76964c8ef3746bfd, 05 b920c4a58cb335e4); the generator on the dev source before the change reproduces every file byte for
byte, and after it the only changed lines are the task sentence's (the instruction's task line, the README's task line
and goal, `task.toml`'s description). Evidence: `<host work dir>/metric_review_20261019/text_fixes/` (steps.txt,
texts_dev/, texts_new/).
