# cast_keyhole: the double maze's task texts (new base cast-keyhole-06-i01)

The double maze (`cast_keyhole_double_maze`, an experimental variation without a tier;
`docs/simulator/families/cast_keyhole_double_maze.md`) has no released base. Its own base, `cast-keyhole-06-i01`, is
made from the released `cast-keyhole-easy-i00` as its template (`release_rebuild.py` new_bases). These directories hold
the new base's `task.toml`, `README.md` and `instruction.md` for both modes; every other file of a package (the
Dockerfiles, the compose files, `tests/test.sh`) is the template's, and the rebuild writes `runtime_binding.upstream.json`.

- Not on the release list: `mujoco_release.json` lists no cast_keyhole entry. A temporary manifest makes the base, for
  local generation and for Harbor under its name, on a test-only commit of a temporary branch that is never merged: a
  cast_keyhole entry with `"tiers": []` (the released tiers are not made) and this new base, then
  `release_rebuild.py <commit> --version <v> --families cast_keyhole` (with Harbor):

  ```json
  "cast_keyhole": {"tiers": [], "note": "...",
    "new_bases": {"cast-keyhole-06-i01": {"template": "cast-keyhole-easy-i00",
      "variation": "cast_keyhole_double_maze", "package_text": "tools/release/package_text/cast_keyhole",
      "design": "...", "note": "..."}}}
  ```

- The template is easy: the same goal (the family's placement: the whole silver piece in the fixed goal volume, the gold
  piece behind the separator, terminal revision `cast_keyhole_placement_v4_grasp_contact_2mm`), the same robot and the
  same package files; every tier's goal is the same placement.
- Each file is the template's released -i00 file (`tools/release/templates/cast-keyhole-easy-i00-<mode>/`) with only
  the double maze's lines replaced, each value taken from the double-maze instance or from what the tooling generates for
  it (generator and its check: `runs/cast_keyhole_v3_double_maze_20261011/r4/harbor/keyhole_package_text.py`,
  `check.json` beside the generated texts):
  - the names: `topogym-cast-keyhole-easy-i00-<mode>` becomes `topogym-cast-keyhole-06-i01-<mode>` (the task
    name, the README title, the twin);
  - the variation id in the privileged `task.toml` (description and instance comment) and the privileged README's task
    line;
  - the goal volume's height (342.04 mm, the instance's `public_goal`) and the goal square, 350.0 mm (the instance's goal
    region less the 2 mm stripe on each side);
  - the grasp regions (`keyhole_gold`, `keyhole_silver`, 30 mm each);
  - the scene objects in `s.objects()` order on MuJoCo (`goal_region`, `keyhole_gold`, `keyhole_silver`) and the standard
    "Objects here";
  - the privileged instruction's implementation reference: `CastKeyholeDoubleMazeTask` and its module. Its
    `evaluate_state` is `CastKeyholeTask`'s, so the excerpt is unchanged;
  - the success rule (`task.toml`, README): the instance's own, without easy's qualification record (the double maze has
    none);
  - the grasping rule (privileged "- Grippers:", standard "Grasping:"): `Session.grasp_rules()` for this instance, which
    states the in-hand start ("the grippers start closed on their parts ...") and the 60 N slip rule;
  - `task.toml`: no difficulty line (the variation record has no tier);
  - the per-step wall clock, in both modes: 0.031 s, MuJoCo, the -i01 reference on profile keyhole_double_maze_v2
    replayed once (strict) in the dev image `topogym-coil-v2:20261009-02` on the AMD Ryzen 9 7950X host (load average
    about 29): 8 082 control steps in 249.0 s. (On keyhole_double_maze_v1 it read 0.036 s: 293.9 s at load about 36.)
    The released 0.042 s is easy's Genesis measurement.
  - the contact-penetration line (instruction.md and README.md of both modes): the sustained rule in force, "deeper than
    2 mm for at least 0.25 s" (CONTACT_RULE sustained_m 0.002 since cade53224; the released files say 1 mm; 167f4bbd8).
    The held-piece half (4 mm, or 2 mm for 0.25 s) is the released one.
  - Every other line is the released one (base image, tooling version, protocol name, oracle evidence wording, the
    tool-frame axes), so `mk_tasks.py` rewrites it as for every family.
- The reference: `scripts/roboarc/oracles/cast-keyhole-06-i01/` (profile keyhole_double_maze_v2: 8082 rows,
  birth c15ba78d2bfda87e, final state 5bc8f1e7136b3f59).
  Harbor graded the texts of 167f4bbd8 (0.036 s per step) on profile keyhole_double_maze_v1 (birth a804367eaf3bfdd3, the
  reference then ending on c18098f933020b0f) under the new base's name on 2026-10-12, from the test-only commit
  b38a7f6be (167f4bbd8 + the manifest entry above; branch jianwen/keyhole-dm4-harbor-test, never merged): oracle 1, nop 0
  in both modes, the release's package files equal to those texts byte for byte
  (`docs/simulator/families/cast_keyhole_double_maze.md`, "Harbor under the new base name"). On keyhole_double_maze_v2, after
  its acceptance round passed, Harbor graded these texts (e9e29c39e, 0.031 s per step) under the new base's name on
  2026-10-13, from the test-only commit 0956310dd (e9e29c39e + the manifest entry above; branch
  jianwen/keyhole-dm4-v2-harbor-test, never merged): oracle 1, nop 0 in both modes, the oracle ending on 5bc8f1e7136b3f59,
  the release's package files equal to these texts byte for byte (the same document, "Harbor under the new base name on
  v2").
