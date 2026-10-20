# knot_untying redesign: scripts

Design note: `docs/simulator/redesign/knot_untying.md`. Routes per instance (flags, rows, hashes): `routes.json`.

- `births/<name>/`: the object-level lab start of each instance (scene_start.json, start_state.npz, lab_nodes.npz), from
  the lab's inverse generation (`topogym/planning/knot_untying/redesign/birth.py`). `<name>_r2`: the same start with
  the user's 10-12 goal regions (`tiers_v8.py`); `<name>_r3`: the same start with the 10-13 back strip (x -0.45..0.42).
- `robot/build_instance.py <births/name> <tier> <out>`: freezes a robot-level MuJoCo instance (variation
  knot_untying_rd_<tier>): the scene as task parameters, the rope birth swept along the lab centre line, contact_weld_v1
  targets on every link, the clamp as an AttachmentSpec.
- `robot/check_instance.py <instance.json> <checks dir> --part goal|nop|grasp|summary`: the instance checks.
- `robot/run_robot_oracle.py <instance.json> <tier> <run> [flags]`: the untying oracle on the ALOHA, recorded as rows
  (privileged.npz, oracle.json, trace.jsonl); `--resume <rows> [--resume-rows N]` continues a run, `--stop-after-decisions
  N` ends a chain's stage, `--checkpoint-untied` saves untied_checkpoint.pkl (first decision with no crossing) and
  final_checkpoint.pkl. For a route search over decisions: `--checkpoint-decisions` saves `ck/dNN.pkl` (+ `dNN.json`:
  row, live crossings, state hash, the rope's centre line, the free tail past each under-crossing, wraps round posts)
  after every decision record, `--restore <pkl>` starts from such a state (its rows included, no replay) and lets the
  oracle decide again, `--max-rows N` ends a branch that grew too long, `--first lift_post,loosen,...` makes the
  oracle's own moves in a given order before its deletion loop (`lift_exit[:offset:clear:out]`: lift_over_post with
  the pinch past the wrap on the free side and the hand higher, for a wrap that hugs its post; it took medium_loose's
  last loop off the eye post). Rows of a restored branch replay exactly in a fresh process (checked 10-13: 4396 rows,
  restored at row 2874). `--turbo F` (with `--fast`; 10-14, route length): the fast moves, descents, lifts, drops,
  presses, tail carries and joint moves x F and the lab's minimum move time, turn rate and close time scaled by F;
  strokes keep `--speed`, and every row stays within the standard joint step limit through the rate limiter (turbo 1 =
  the earlier runs, row for row). A placement try sets the same through its spec's `untie` and `lab` dicts.
  `--converge-scale C`: the lab's converge (a hand held still until it settles) waits at most its cap x C (default 1;
  0.5 with --turbo 2 left the arms blocking each other on very_easy, so it stays opt-in).
- `robot/replay_check.py <run>`: a fresh-process replay of a run's rows; the final state hash, success and the forcing
  record against the live run's.
- `robot/place_lab.py <instance.json> <tier> <lab> --resume <untie rows> --resume-rows N`: placement tries from one
  untied state (`topogym/planning/knot_untying/redesign/robot_place.py`), each from the same rewound checkpoint. The lab
  writes `<lab>/checkpoint.pkl` after its replay; `--restore <checkpoint.pkl> --settle-s 0` starts another lab from it
  without the replay (rows replay at about 1 row/s, 5-9 h for an untied state).
- `robot/ckpt.py`: save / restore of a Session's full state (the env's episode state, the Session's counters, rows and
  targets), for the two scripts above.
- `premade/make_premade.py <source instance dir> <out dir>` and `premade/premade_checks.py birth|nop|speed ...` (release
  noinst image, its shared tooling at /tooling): the rd bases' premade frozen instances and their checks
  (passed to tools/release/release_rebuild.py with --instances or as the manifest's instance).
- `region_images.py <instance.json> <out>` (release image, one GPU): a region tier's region images, the start scene from
  cam_high / cam_low with each goal region's dashed line drawn in (the scene does not draw it); the task's texts name
  them (task_rd.REGION_IMAGES), package_text/<base>-<mode>/environment/goal/ ships them.

Environment: `PYTHONPATH=<repo>:<repo>/scripts/roboarc/shared:<repo>/tools/redesign`; MuJoCo 3.14, CPU
float64.
