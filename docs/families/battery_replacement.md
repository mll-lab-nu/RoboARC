# Battery replacement

Family key `battery_replacement` · 1 task base · robot `aloha-2` · physics MuJoCo 3.14.0

A game controller lies face down in a cradle on the table and does not move. Its battery compartment, closed by a
sliding cover, holds two old AA cells (grey body, tan band); two new cells (black body, copper band) lie in a tray, a
pry stick (a nylon spudger) stands in its holder and a recycling box is on the table. The robot must open the cover,
lever each old cell up with the pry stick and take it out, put both old cells in the recycling box, seat the two new
cells with the right polarity and close and latch the cover. Every mechanism is physical: the spring contacts, the
cover's slide and its detent plunger. The family was designed by Jianwen; its base came to RoboARC as an experimental
base in 1.2.0 and was released in 1.4.0.

The compartment has two slots side by side; each has a + contact plate at one end (a red "+" on the compartment floor
beside it) and a spring contact at the other, and the two slots' + ends are at opposite ends. The cover slides about
5 mm toward the controller's top edge to unlatch, then lifts off; to close it, it is laid flat in its recess and slid
back until it stops (it latches there). The old cells sit 4 mm below the compartment's rim with 1 mm of side clearance;
the pry stick's blade goes into the notch at a cell's + end, beside the + contact, and the stick pivots on the notch's
outer edge.

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position
targets (6 arm joints and one gripper target per arm) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping (`contact_weld_v1`) of the cover, the pry stick (around its handle) and the four cells: a closing
gripper takes hold of a part once both fingers touch it inside its grasp region, with the tool frame within 6 cm of it
and the penetration at most 2 mm, for 0.02 s; the part then follows the gripper rigidly. Any opening command releases
it. A hold is lost if the fingers open more than 15 %, the part drifts 1 cm / 0.15 rad from its held pose, or the hold
has to push or pull its part with more than 60 N (averaged over 20 ms); the gripper must then open before it can take
hold again. Two grippers never hold the same part, and other objects are grasped physically (friction). The grippers
start empty and nearly closed: open a gripper, then close it on a part to take hold.

The cells have pinch rules, stated with a pinch's central angle: the angle around the cell's axis between the two
fingers' contact points (180 degrees on opposite sides of a diameter; exactly, force closure at a friction coefficient:
mu 1 is 90 degrees, mu 0.268 is 150 degrees). The angle is decided when the hold is taken.

- An old cell is taken only by a pinch of at least 90 degrees: fingertips on the top of a cell seated in its slot cannot
  take it; a cell levered up with the pry stick can be plucked.
- A hold below 150 degrees is a pluck: it is let go once the cell has moved 60 mm from where it was taken (as a lost
  grasp: the gripper opens before taking hold again), so an old cell is carried only when held from both sides
  (150 degrees or more). The simulator counts these 60 mm as one budget per cell and episode, shared by all its plucks
  and counted from where the cell leaves the compartment (`WeakHoldGraspTarget`).
- A new cell is taken only by a pinch of at least 150 degrees.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **New cells seated.** Each slot holds a new cell, fully seated: its axis within 5 degrees of the slot's, its centre
  within 1.2 mm sideways, 1 mm vertically and 1.5 mm along the axis of the seated position, and the slot's spring
  contact compressed at least 1 mm.
- **Polarity.** A cell's + end (the small nub, and the "+" printed on its coloured band) faces that slot's + contact
  plate; its - end presses the spring contact.
- **Cover latched.** The cover is within 0.5 mm of its closed position along and across the controller (pushed all the
  way back along its slide, where the detent plunger sits in the cover's groove), between 0.2 mm below and 0.6 mm above
  it, tilted at most 1 degree, and the detent plunger is not held down more than 0.25 mm below its upper stop.
- **Old cells recycled.** Both old cells lie inside the recycling box (their centres within its inner walls, below its
  rim).
- **Still.** The four cells and the cover move at most 5 mm/s and 0.1 rad/s. Holding parts is allowed; the pry stick may
  end anywhere (on the table or in a hand).
- All final conditions must hold together, continuously, for 0.5 s of simulated time.

A task piece falling more than 100 mm below the tabletop ends the episode in failure. Contact penetration does not end
the episode by itself. A success with deep interpenetration (beyond the limits stated in the task text) is held for
human review instead of being credited automatically, and a task contact deeper than 10 mm counts as a simulator
breakdown.

Every task has a wall-clock budget of 10800 s (3 h) in both modes. In privileged mode the delivered trajectory is
replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be reached within the
trajectory's own rows. In standard mode the simulator service grades the live episode and a fresh-process replay
confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading replays").

## Bases

1 base (medium 1). It is one frozen scene instance and ships as two Harbor task directories, `-privileged` and
`-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in each `task.toml`.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-battery-replacement-01-i01` | medium | `battery_replacement_xbox_aa` | a game controller's two AA cells swapped, the old ones pried out with the pry stick | [privileged](../../tasks/roboarc-battery-replacement-01-i01-privileged/) / [standard](../../tasks/roboarc-battery-replacement-01-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
- [battery_replacement.md](../simulator/families/battery_replacement.md): development notes (the controller, the goal, the reference, the grasp rule C, stress, Harbor).
