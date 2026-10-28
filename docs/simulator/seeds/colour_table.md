# Named colours and materials under the appearance seeds

Which parts of a scene the task texts name by colour or by material, and what each appearance seed draws them in.
Rewritten for the appearance seeds v2 (CHANGELOG, "Unreleased"; 2026-10-28) from the code:
`source/topogym/tasks/scene_variation.py` (`AXES`, `appearance`, `scene_look`), `source/topogym/world/look.py` and
the families' finish tables named below. The texts are those of `scripts/roboarc/packages/` (both modes'
`instruction.md`, `README.md` and `task.toml`); the task directories under `tasks/` are rendered from them, and until
they are rendered again the colour sentences of 22 bases there are the old ones (CHANGELOG).

This page replaces the colour table of 2026-10-21, made for 1.2.0: one row per colour word of every base, with the RGB
of the part at seeds 0, 1 and 2, and a CSV beside it (`colour_table.csv`). Under the new rules most of those rows no
longer hold: the parts change colour and material, and several texts no longer name a colour. So the per-seed RGB rows
and the CSV are dropped (both are in the git history); this page gives the rule per family and the named parts, and the
colour values are in the code tables it names.

## The rule

- Seed 0 is the released look, but for the goal marker: its green dashed line is 5 mm wide and glowing at every seed
  (`source/topogym/assets/geometry/goal_region.py`).
- Seeds 1 and 2 change the room's lighting, the table top (wood, metal or stone), the table's frame and the floor
  (`scene_look`), and the colours and materials of the task's objects (the family's appearance parameters). The
  CHANGELOG lists the looks family by family.
- A part whose colour a text names keeps its hue at every seed; its shade, roughness or metalness may change.
- A part a text names by its material keeps the material; its colour within that material may change.
- Every other part may change colour and material. No text names such a part by colour: where a text did before, it now
  names the part by shape or role.
- Object names that hold a colour word (`keyhole_silver`, `keyhole_gold`, `coil_silver`, `coil_gold`) are identifiers;
  a text that shows them says once "(the names are identifiers, not colours)".

## Parts named by colour: the hue stays

| bases | part | the texts' words | seed 0 | seed 1 | seed 2 | code |
|---|---|---|---|---|---|---|
| the 30 bases that draw the goal square (cast-keyhole 02-06, hedgehog-in-cage 02, 03, 06-08, horseshoe-ring 01-05, interlocked-claws 02, 05, 08-13, orbits 01-02, slotted-post-ring 01-05); knot-untying 07 and 09 (painted into their goal pictures) | the goal marker | "inside the green dashed goal region", "the inner edge of the dashed green line is the containment limit" (28 bases; no width stated); interlocked-claws 05 and 11-13: "The green dashed square on the table is not part of this goal" | green, 5 mm wide, glowing | the same | the same | `goal_region.py` (`GOAL_GREEN`, `GOAL_GLOW`); `scripts/roboarc/goal_images/region_images.py` |
| rope-threading 04-08 | the rings and grommets | "cyan grommet in the wall (G1)", "green ring (A)", "purple grommet in the wall (G2)", "amber ring (B)" (V in 05, T in 07 and 08), "red grommet in the wall (G3)" (06-08), "cyan ring (C)" (08) | as released | the same hues, glossy | the same hues at 90 % shade, matte powder coat | `source/topogym/tasks/rope_threading/redesign/task.py` (`RING_FINISHES`) |
| wire-splicing-01 | the two wires' insulation; the bending post's rod | "Splice the red and the blue wire", "`wire_a` (red)", "`wire_b` (blue)", "the yellow rod of the bending post" | red, blue; yellow | unchanged | unchanged | `source/topogym/assets/geometry/wire_splicing.py` (`RGBA_ROD`; the insulation is in no finish table) |
| fiber-bundling 08, 09 | the red fibers (strands 0 and 8: 16 strands over an 8-colour palette) | "the root of one of the two red fibers" | red | the same red | red at 90 % shade | `source/topogym/assets/geometry/fiber_bundling.py` (`CABLE_FINISHES`: the palette keeps its first slot red) |
| battery-replacement-01 | the old cells; the new cells | "two old cells (grey body, tan band)", "a new cell (black body, copper band)" | as named | unchanged | unchanged | `source/topogym/assets/geometry/battery_replacement.py` (in no finish table) |
| battery-replacement-01 | the polarity marks | "a red "+" on the compartment floor", "marked with a red '+' on the compartment floor" | red on the grey floor | red on the Shock Blue floor | red on the Electric Volt (yellow-green) floor | `battery_replacement.py` (`CONTROLLER_FINISHES`: the floor takes the controller's colour, the marks keep their red) |
| caliper-sorting-01 | the locking knob's index line | "a white line on its top shows its angle" | white | white | white | `source/topogym/assets/geometry/caliper_sorting.py` (`KNOB_MARK`; the printed scale, the vernier and the bins' white label cards keep their colours too) |
| bed-bolt 01-05 | the drive card | "A white card lying flat on the table between the key's stand and the bolt's cradle" | white | white, matte paper | white, coated card | `source/topogym/assets/geometry/bed_bolt.py` (`WOOD_FINISHES` `card`; the drawn head and recess keep their colours) |

## Parts named by material: the material stays

| bases | part | the texts' words | seed 0 | seed 1 | seed 2 | code |
|---|---|---|---|---|---|---|
| buzz-wire-3d 02, 04, 06, 07, 10 | the wire | "the metal wire" | copper | brass | stainless steel | `source/topogym/tasks/buzz_wire_3d/common.py` (`FINISHES`) |
| buzz-wire-3d 02, 04, 06, 07, 10 | the tool's loop | "the metal loop", "all metal parts of the tool" | stainless steel | gunmetal | polished copper | the same |
| buzz-wire-3d 02, 04, 06, 07, 10 | the tool's handle | "Wooden-handle contact is not a metal-wire failure" | pale wood | walnut | ebony (the wood texture kept) | the same |
| buzz-wire-3d 06, 07, 10 | the sleeves on the wire | "Rubber sleeves on the wire are insulated" | black rubber | red rubber | blue rubber | the same |
| orbits 01, 02 | the handle | "the entire cord and wooden handle" | light wood | lacquered ebony | oiled mahogany | `source/topogym/assets/geometry/orbits.py` (`FIXTURE_FINISHES` `handle`) |
| slotted-post-ring 01-05 | the removable ring | "the entire removable metal ring" | as released | violet anodised | pink anodised | `source/topogym/tasks/slotted_post_ring/slotted_post_ring_hard_chain.py` (`PUZZLE_FINISHES`) |
| battery-replacement-01 | the pry stick | "a nylon spudger" | black nylon | blue nylon | yellow nylon | `battery_replacement.py` (`STICK_FINISHES`) |
| hedgehog-in-cage-07 | the owl | "a wooden owl in a cage" (the task directories' README Change Log) | as released | oiled wenge | raw beech | `source/topogym/tasks/hedgehog_in_cage/owl.py` (`OWL_FINISHES`) |
| blanket-flatten 03, 04, 06-08; blanket-fold 01-05 | the blanket | "cloth" | blue cotton, a hairline grid | red wool, a buffalo check with charcoal | green sateen, mustard stripes | `scene_variation.py` (`AXES`: `color`, `texture`) |

## Parts no longer named by colour

| bases | part | the texts' words now (before) | seed 0 | seed 1 | seed 2 | code |
|---|---|---|---|---|---|---|
| cast-keyhole 02-05 | `keyhole_silver`; `keyhole_gold` | "the keyhole plate"; "the fork plate", with a sentence on their shapes ("the silver keyhole piece"; "the gold piece") | silver; gold | blue; red (anodised aluminium) | orange; black (powder coat) | `source/topogym/assets/scene_visuals.py` (`CAST_PIECE_FINISHES`) |
| cast-keyhole-06 | `keyhole_silver`; `keyhole_gold` | "the right-hand plate"; "the left-hand plate" (as above) | silver; gold | blue; red | orange; black | the same |
| cast-coil 04-08 | `coil_silver`, `coil_gold` | "the two Cast Coil pieces" ("the gold and silver Cast Coil pieces") | silver, gold | blue, red | orange, black | the same |
| interlocked-claws 02 | `piece_0`; `piece_1` | "the cross piece"; "the frame piece", with a sentence on their shapes ("the brown claw piece"; "the other claw piece") | brown; teal | purple; red (anodised) | yellow; blue (powder coat) | `source/topogym/tasks/interlocked_claws/common.py` (`piece_finishes`) |
| interlocked-claws 08-10 | the key; the lock | "the key piece"; "the lock piece", with a sentence on their shapes ("the brown claw piece"; "the other claw piece"); the release design string "a key and a lock" ("a brown key and a lock") | brown; teal | purple; red | yellow; blue | the same |
| fiber-bundling 02, 04, 05, 08, 09 | the bundling ring | "the fixed ring" ("the fixed copper ring") | copper | teal metal | black anodised | `fiber_bundling.py` (`CABLE_FINISHES` `ring`) |
| buzz-wire-3d 06, 07, 10 | the sleeves on the wire | "Rubber sleeves" ("Black rubber sleeves"); see the material table | black | red | blue | `buzz_wire_3d/common.py` |

The other pieces of interlocked-claws (05, 11-13: "the two claw pieces", "the C ring" and "the AB plate", "the two
interlocked pieces", Cast Trinity's three pieces) were never named by colour; they change as the family's pieces do
(Cast Trinity's own finishes: `source/topogym/tasks/interlocked_claws/interlocked_claws_cast_trinity.py`).

## Bases whose texts name no colour or material

clock-assembling-01, fiber-bundling 02, 04 and 05 (their ring is now "the fixed ring"), knot-untying 02 and 04-06,
lock-picking 03-05: 11 bases. cast-coil 04-08 name colours only in the identifiers `coil_gold` / `coil_silver` (in the
grasp regions, `s.objects()` and the privileged grader's code).

## How this was checked

- The parts and their looks: `appearance(cls, 1)` and `appearance(cls, 2)` evaluated for every one of the 78 bases (its
  task and variation from its frozen instance) in the 2.1.0 image with this source on the path, and the finish tables
  read in the code named above. No frame was rendered for this page.
- The texts: a search of `scripts/roboarc/packages/` (both modes' `instruction.md`, `README.md`, `task.toml`) for the
  colour words red, green, blue, yellow, white, black, grey, tan, brown, teal, copper, gold, silver, cyan, purple,
  amber, orange and brass, and for the material words wooden, metal, rubber, nylon and cloth. "gold" and "silver" are
  left only in identifiers and in variable names of cast-coil's grader code.
- The README Change Logs of the task directories keep their history as `scripts/roboarc/release.json` records it, so
  they still name colours as they were when each entry was written (for example "brown and teal" for
  interlocked-claws 08-10, "gold and silver" for cast-keyhole-06).
