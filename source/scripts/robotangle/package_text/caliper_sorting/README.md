# caliper_sorting: task texts of the family's new base

caliper_sorting (`docs/rcb/families/caliper_sorting.md`) is a new MuJoCo family with no released base. Its base,
`caliper-sorting-01-i01` (variation `caliper_sorting_vernier`, an experimental variation without a tier), is made
from the released `interlocked-claws-easy-i00` as its template (`release_rebuild.py` new_bases: the ALOHA, contact_weld_v1
and a rigid MuJoCo family, so the template's package files, the Dockerfiles, compose files and `tests/test.sh`, serve).
These directories hold the new base's `task.toml`, `README.md` and `instruction.md` for both modes. The template's two
files that name its family (the privileged `tests/test.sh`, `grade.py --family`, and the standard
`environment/docker-compose.yaml`, `RCB_TASK`) get the new base's family from release_rebuild itself (branch
jianwen/release-rebuild-family), not from files here.

- The texts are the released Cast Trinity texts (`package_text/interlocked_claws/interlocked-claws-13-i01-*`, its RoboARC
  name; the same tooling version) with every task-specific line replaced, each value taken from the caliper instance (scene seed 0,
  native gripper space) or from what the tooling generates for it (the design record's `port/mk_text_v6.py` asserts every
  replacement; version 3 of the task, the caliper on its stand, 2026-10-18, with the terminal revision
  sleeves_in_labelled_bins_v2 of 2026-10-19):
  - the names: `topogym-caliper-sorting-01-i01-<mode>`, the family `caliper_sorting`, the variation id;
  - the task sentence and the success requirements and failure conditions: the instance's `public_goal`
    (`topogym/tasks/goal_description.py`, the caliper_sorting branch);
  - the grader details: `scripts/topogym_rules.py details_section` for the instance (caliper_sorting is a RIGID family
    there: the contact review rules; no goal square, no separator), with the released rigid texts' clause that it replaces
    the 1 mm / 2 mm wording above; the grasp regions are the instance's (the slider 20 mm, about the left hand's grip
    and clear of the locking knob, which is no target; the beam is fixed on its stand and no target; 40 mm for every
    sleeve: one radius for all, so that no text states a sleeve's size);
  - the grasping rule (privileged "- Grippers:", standard "Grasping:"): `Session.grasp_rules()` for the instance (no
    pre-weld: the grippers start open);
  - the objects in `s.objects()` order; in the privileged text the exact keys of the caliper's four links (`mount`, the
    stand's clamp, `beam`, `slider`, `screw`), its joints (`mount`, `slider`, `screw`;
    `s.env._world.read_state()["articulations"]["caliper"]`) and the lock's readout
    (`s.env._world.read_state()["mechanism"]`), and the implementation excerpt (`CaliperSortingTask.evaluate_state` and
    `evaluate_transition`: every sleeve released, and a sleeve left still in another class's bin for 0.5 s fails);
  - the per-step wall clock (both modes): the standard Harbor trial's fresh-process replay of the reference (0.030 s:
    678.3 s for 22962 control steps, Harbor on d0a35aff89; the family notes).
- The standard texts use only what the standard client gives (cameras, proprioception): the sleeves' classes come from
  measuring them with the caliper and reading the vernier in the images, in the caliper's own unit (the labels give
  readings; the unit and zero differ from scene to scene).
