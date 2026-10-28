# RoboARC families

RoboARC's topological puzzles come in 19 families with 78 task bases in total. Each base is one frozen
MuJoCo scene instance and ships as two Harbor task directories, `tasks/roboarc-<family>-<NN>-i01-privileged` and
`-standard` (the family key with dashes). In privileged mode the agent writes code against a simulator in its
container and delivers one trajectory; in standard mode it drives the robot through a simulator service and sees only
camera images and proprioception. The difficulty of a base is `[metadata].difficulty` in its `task.toml`.

| Family | Puzzle | Robot | Bases | Difficulty |
|---|---|---|---:|---|
| [Battery replacement](battery_replacement.md) | Swap a game controller's two AA cells: pry the old ones out, seat new ones with the right polarity, latch the cover | ALOHA 2 (two arms) | 1 | medium 1 |
| [Bed bolt](bed_bolt.md) | Drive a bed frame's corner bolt into its threaded hole with an L-shaped key | ALOHA 2 (two arms) | 5 | very_easy 1, easier 1, easy 1, medium 1, hard 1 |
| [Blanket flatten](blanket_flatten.md) | Unfold a folded blanket and spread it flat on the table | ALOHA 2 (two arms) | 5 | easier 1, easy 2, medium 1, hard 1 |
| [Blanket fold](blanket_fold.md) | Fold a blanket into a target shape shown in target images | ALOHA 2 (two arms) | 5 | very_easy 1, easier 1, easy 1, medium 1, hard 1 |
| [3D buzz wire](buzz_wire_3d.md) | Guide a metal loop along a bent wire without touching it | Franka Panda + Robotiq 2F-85 | 5 | very_easy 1, easier 2, medium 1, hard 1 |
| [Caliper sorting](caliper_sorting.md) | Measure look-alike sleeves with a vernier caliper and sort them into labelled bins | ALOHA 2 (two arms) | 1 | hard 1 |
| [Cast coil](cast_coil.md) | Assemble two interlocking cast pieces | ALOHA 2 (two arms) | 5 | very_easy 1, easier 1, easy 1, medium 1, hard 1 |
| [Cast keyhole](cast_keyhole.md) | Free the keyhole plate from the fork plate and place it in a goal square | ALOHA 2 (two arms) | 5 | easier 1, easy 1, medium 1, hard 2 |
| [Clock assembling](clock_assembling.md) | Assemble a watch movement's gear train, screw down its bridge, wind it and let it run | ALOHA 2 (two arms) | 1 | hard 1 |
| [Fiber bundling](fiber_bundling.md) | Pass a bundle of fine fibers and its sheath through a ring | Franka Panda + Robotiq 2F-85 | 5 | easier 1, easy 1, medium 2, hard 1 |
| [Hedgehog in cage](hedgehog_in_cage.md) | Get a spiked object out of a fixed cage into a goal square | Franka Panda + Robotiq 2F-85 | 5 | easier 2, easy 2, hard 1 |
| [Horseshoe ring](horseshoe_ring.md) | Free a ring from two chained horseshoes | ALOHA 2 (two arms) | 5 | very_easy 1, easier 1, easy 1, medium 1, hard 1 |
| [Interlocked claws](interlocked_claws.md) | Separate interlocked pieces, or place one in a goal square | ALOHA 2 (two arms) | 8 | easier 1, easy 1, medium 2, hard 4 |
| [Knot untying](knot_untying.md) | Untie a knotted rope and lay it out without crossings | ALOHA 2 (two arms) | 6 | very_easy 1, easier 2, medium 2, hard 1 |
| [Lock picking](lock_picking.md) | Insert a two-tab key through a stack of rotor plates | Franka Panda + Robotiq 2F-85 | 3 | easy 1, medium 1, hard 1 |
| [Orbits](orbits.md) | Free a cord loop and handle from posts with stopper balls | ALOHA 2 (two arms) | 2 | very_easy 1, easier 1 |
| [Rope threading](rope_threading.md) | Thread a rope through rings and grommets in a given order | ALOHA 2 (two arms) | 5 | very_easy 1, easier 1, easy 1, medium 1, hard 1 |
| [Slotted post ring](slotted_post_ring.md) | Free a ring from a cord running through a post | ALOHA 2 (two arms) | 5 | very_easy 1, easier 1, easy 1, medium 1, hard 1 |
| [Wire splicing](wire_splicing.md) | Splice two stiff wires: an eye round a rod, the other wire threaded and wound round it | ALOHA 2 (two arms) | 1 | hard 1 |

Each page describes the puzzle, the robot and grasping, the success rule in plain words, and every released base. The
rules shared by all families (grading replays, safety readings, the standard-mode service, the time budget) are on the
[RoboARC benchmark page](../benchmarks/roboarc.md).
