# Wire splicing

Family key `wire_splicing` · 1 task base · robot `aloha-2` · physics MuJoCo 3.14.0

Two stiff wires, red and blue, run from terminal blocks on the table; each has a bare end. The robot must splice them:
bend the red wire's bare end into an eye round the yellow rod of a fixed bending post, thread the blue wire's bare end
through the eye, wind the blue wire's bare tail round the red wire's doubled end, fold the red wire's leftover end back
over the coil, and let go. Each wire is a chain of 160 ball-jointed links with elastoplastic joints, so a bend stays
where it is put. The family was designed by Jianwen and ported from TopoGym to RoboARC in 1.3.0.

## Robot and grasping

Two 6-DoF ALOHA 2 (ViperX) arms with parallel-jaw grippers on a fixed base (`aloha-2`). Actions are 14-D joint-position
targets (6 arm joints and one gripper target per arm) at 40 Hz; standard mode also accepts end-effector pose targets.

Assisted grasping (`contact_weld_v1`): a closing gripper takes hold of a wire once both fingers touch it inside its
links' grasp regions, with the tool frame within 6 cm of it, for 0.02 s. Of each wire it touches, the touched link
nearest the tool frame and every link whose centre lies between the two pads then follow the gripper rigidly. Two
fingers on the wires with every contact between the pads also hold as a pinch (for example the red wire's two legs side
by side, one pad on each). Any opening command releases. A hold is lost if the fingers open too far, a held link drifts
from its held pose, the hold has to push or pull with more than 60 N (averaged over 20 ms), or a weld has fed more than
0.1 J of work into the wire; the gripper must then open before it can take hold again. The bending post is fixed.

## Success rule

What the grader checks, in plain words (each task's `instruction.md` states the exact rule and its limits):

- **Eye.** The red wire's bare end is turned back on itself by more than 150 degrees into an eye that is closed at its
  neck.
- **Threaded.** The blue wire passes through the closed eye exactly once.
- **Coil.** The blue wire's bare tail winds at least 1.5 turns round the red wire's doubled section, at least one full
  turn of them round both strands together, and the coil is tight (on average within 2 core diameters of the axis).
- **Folded end.** The red wire's leftover end is folded back toward the eye.
- **Let go and still.** No hand holds a wire, and the splice is at rest; the coil may not unwind by more than a quarter
  turn and the threading may not change while the conditions are held.
- All final conditions must hold together, continuously, for 2 s of simulated time.

A wire piece falling more than 100 mm below the tabletop ends the episode in failure.

The agent's wall-clock budget is 21600 s (6 h) in both modes, by the owner's exception to the benchmark's 3 h
(`scripts/task_conventions.py` `CUTOFF_EXCEPTIONS`): one control step takes about 0.5 s. Its trials form their own
group in the results (`cutoff_s` is part of the configuration results are ranked by). In privileged mode the delivered
trajectory is replayed open-loop twice in a fresh process: both replays must succeed, and the strict goal must be
reached within the trajectory's own rows. In standard mode the simulator service grades the live episode and a
fresh-process replay confirms it. The details are on the [benchmark page](../benchmarks/roboarc.md) (section "Grading
replays").

## Bases

1 base (hard 1). It is one frozen scene instance and ships as two Harbor task directories, `-privileged` and
`-standard`, with the same scene, goal and grader. The difficulty is `[metadata].difficulty` in each `task.toml`.

| Base | Difficulty | Variation (task text) | Notes | Task dirs |
|---|---|---|---|---|
| `roboarc-wire-splicing-01-i01` | hard | `wire_splicing_eye_wrap` | eye round the post's rod, 1.5-turn coil, end folded back; appearance seeds 0-2 | [privileged](../../tasks/roboarc-wire-splicing-01-i01-privileged/) / [standard](../../tasks/roboarc-wire-splicing-01-i01-standard/) |

## Links

- [RoboARC benchmark page](../benchmarks/roboarc.md): the rules shared by all families (grading replays, safety readings in standard mode, the standard-mode service, the time budget).
- [Standard mode](../STANDARD_MODE.md) and [protocol](../PROTOCOL.md).
- [All families](README.md).
