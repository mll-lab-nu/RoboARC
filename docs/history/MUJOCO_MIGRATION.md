# RoboTangle → MuJoCo: migration status (record of 2026-10-09)

Kept as the record of the port's goal and acceptance bar; every family has since been ported and released
(CHANGELOG.md).

From 2026-10-09 the MuJoCo port was developed on its own branch, started from the Genesis line of the 0.2.22
release at 095761f92.

The goal:
- every family runs on MuJoCo physics;
- after that, Genesis is removed completely;
- the Nyx look stays.

## Acceptance bar (every family, every tier)

1. **References and nop.** The reference succeeds, the nop fails, and a fresh-process replay reaches the recording's state hash.
2. **Speed.** One full reference execution takes at most ~72 min (20 % of the 6 h agent budget) on the trial host.
3. **Stress.** The regression corpus is clean, and a fresh fixed-budget self-play search finds no breakdown. The budget is about 1,000 episodes per tier, about 5k for a 5-tier family (decided 2026-10-09).
   - A breakdown is any of:
     - a non-finite state;
     - energy created by a contact: the pair's net work above 0.25 J, or the global energy excess above 0.1 J;
     - a speed jump with no causal contact or weld partner;
     - a contact deeper than 10 mm.
   - Violent but causal motion is not a breakdown.
4. **Standard mode.** A live success, a fresh-replay grade, and a leak audit with 0 findings.
5. **Frames.** Nyx frames come through the render bridge, with parity within Nyx's own run-to-run noise.

## Status (2026-10-09)

| family | physics | tiers | 1 refs/nop/replay | 2 speed | 3 stress | 4 standard + leak | 5 frames |
|---|---|---|---|---|---|---|---|
| interlocked_claws | MuJoCo | all 5 | ok (contact policy v2, references re-recorded) | ok (hard 15 min) | fix in: contact policy v2 (converged CCD, robot hulls <= 64 vertices, dampratio 2, dmax .99): the validator's 95 reproducers 61 -> 0 breakdowns; validator re-test pending | ok, 0 findings | ok |
| fiber_bundling | MuJoCo, profile fiber_v4: the cable as iron wire (elastoplastic joints, 0.4 deg); family note docs/simulator/families/fiber_bundling.md | very_easy..medium; hard physics only | ok: references re-planned on the wire (the released pickup, the planner's combs, a closed-loop feed), 2 fresh replays equal, nops 0 | ok (medium 21 min live, 23 min with the observer) | criteria v2.5: corpus 21 and the 16 fiber reproducers 0 BREAKDOWN, search 70k controls 0 | ok, 0 / 18 | ok |
| cast_coil | Genesis → MuJoCo | in progress | | | | | |
| rope_threading, knot_untying, slotted_post_ring | Genesis (rigid-chain work) | | | | | | |
| buzz_wire_3d, cast_keyhole, hedgehog_in_cage, horseshoe_ring, lock_picking | Genesis | | | | | | |
| blanket_flatten | Genesis (cloth) | | | | | | |

## Road to no Genesis

1. Port every family (the table above).
2. Run Nyx without Genesis.
   - gs-nyx, the renderer, needs only numpy.
   - Only gs-nyx-plugin (six Python files) requires genesis-world.
   - An adapter fed from the MuJoCo scene replaces the plugin and the render bridge's Genesis host. Parity is checked against today's frames.
3. Delete the Genesis code paths and tools, and drop genesis-world from the image.
   - The Panda is vendored (2026-10-09): `assets/robots/franka_emika_panda/` holds MuJoCo Menagerie's
     `franka_emika_panda` at c5d2e83f (panda.xml, the 67 meshes, README, Apache-2.0 LICENSE; `ROBOTANGLE.md`,
     `source.json`), the same bytes as genesis-world 1.2.1 packages. Frozen MuJoCo scenes keep naming its meshes
     `@genesis-assets@/xml/franka_emika_panda/...`; at load they resolve to the vendored copy, so instance and state
     hashes do not change (14 frozen Panda scenes compile to the same model).
   - Left: point `topogym.assets.robot` at the vendored `panda.xml` (the scene builder and the Genesis families still
     read the genesis package's), keeping the `@genesis-assets@` names in new scenes.
4. Release:
   - build a new image;
   - run Harbor oracle and nop in docker;
   - regenerate the task dirs, with new instance ids per PROTOCOL §2.8, because changed physics makes a new task;
   - open a PR to main listing the changed and re-run tasks.

   One command does the first three, once every family has landed: `tools/release/release_rebuild.sh <head>
   --version <v>`. `tools/release/mujoco_release.json` lists the MuJoCo bases and how their instances are made;
   the instances are made inside the release image. The HF upload of the instances, the image push and the PR stay
   separate steps that need the owner's OK.
