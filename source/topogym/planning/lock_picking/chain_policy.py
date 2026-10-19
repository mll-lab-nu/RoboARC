"""Measured-feedback robot oracle for chain locks.

Executes a certified plan of two kinds of steps (see chain_route.py):
  ("go", L)                 move the key to the neighbouring level L through the gap
  ("turn", L, angles_deg)   turn the rotor(s) engaged at level L to these angles
Level L puts the long tip tab in rotor L and the short tab in rotor L-1 (1-based;
rotor N+1 is END). Every key angle is computed from the measured rotor angles,
and rotors move only because a tab pushes a slot wall. Only ordinary EE_ABS
actions are emitted; the key is grasped through the contact-triggered assist.
"""
import numpy as np

from .lock_picking_medium_planning import LockPickingMediumPolicy


def wrap(angle):
    return (angle+180.) % 360.-180.


class ChainLockPolicy(LockPickingMediumPolicy):
    # Measured side-grasp (roll -45 deg) key-angle envelope along the whole axis
    # was [-150, 153] deg or wider; keep 5 deg of margin at both ends.
    KEY_RANGE_DEG = (-145., 148.)

    def __init__(self, *, plan, endpoint_depth, **kwargs):
        super().__init__(**kwargs)
        g = self.geometry
        self.n = len(g["slots"])
        self.deep = [list(row) for row in g["slots"]]
        self.shallow = [list(row) for row in g["shallow_slots"]]
        self.backlash = g["slot_half_deg"]-g["tab_half_deg"]
        # Tab sides and slot walls are radial lines about two axes (key, rotor), so at the bore's edge a tab clears
        # its slot wall by only bore_radius*sin(backlash): 0.52 mm at 4 deg, 0.20 mm in hard, less than the key's
        # sideways wander while it moves. Each level is entered from a stop 1.5 mm before the tabs meet the plates'
        # faces, with the key centred within 30 % of that clearance (0.15 mm at most), then slowly (1 mm/s).
        self.entry_reach = (g["disc_thickness"]+g["tab_thickness"])/2/g["pitch"]
        self.entry_margin = .0015/g["pitch"]
        self.entry_tolerance = min(.00015, .3*g["bore_radius"]*np.sin(np.radians(self.backlash)))
        self.plan = [tuple(step) for step in plan]
        if not self.n+1 < endpoint_depth < self.n+1.15:
            raise ValueError("END insertion depth must leave the leading tab just past END")
        self.endpoint_depth = float(endpoint_depth)
        self.level, self.theta = 0, 0.

    # Measured slot geometry -------------------------------------------------
    def _world_slots(self, disc, kinds):
        a = self._disc_angle(disc)
        rows = (self.deep[disc] if "deep" in kinds else [])+(self.shallow[disc] if "shallow" in kinds else [])
        return [a+s for s in rows]

    def _window(self, level):
        """Centre and offset of the aligned slot pair the tabs use at a level."""
        if level == 1:
            centres = self._world_slots(0, ("deep",))
            best = min(centres, key=lambda w: abs(wrap(w-self.theta)))
            return best, 0.
        lower, upper = level-2, level-1
        uppers = ([self.geometry["endpoint_deg"]] if upper == self.n else
                  self._world_slots(upper, ("deep",)))
        pairs = [(abs(wrap(u-w)), w, u) for w in self._world_slots(lower, ("deep", "shallow")) for u in uppers]
        error, w, u = min(pairs)
        if error > 2*self.backlash-.4:
            raise RuntimeError(f"No aligned slot pair at level {level}: offset {error:.2f} deg")
        return w+wrap(u-w)/2, wrap(u-w)

    def _near(self, world):
        """The reachable representation of a world key angle closest to the current one."""
        low, high = self.KEY_RANGE_DEG
        base = self.theta+wrap(world-self.theta)
        values = [v for v in (base-360., base, base+360.) if low <= v <= high]
        if not values:
            raise RuntimeError(f"Key angle {wrap(world):.1f} deg leaves the arm's roll envelope")
        return min(values, key=lambda v: abs(v-self.theta))

    # Key motions ------------------------------------------------------------
    def _rotate(self, target, phase, max_step=90.):
        target = self._near(target)
        while abs(target-self.theta) > 1e-9:
            step = float(np.clip(target-self.theta, -max_step, max_step))
            angle = self.theta+step
            yield from self._move_key(angle=angle, seconds=max(1.2, abs(step)/12.), phase=phase)
            self.theta = angle

    def _rotate_slowly(self, target, phase):
        target = self._near(target)
        yield from self._move_key(angle=target, seconds=max(2., abs(target-self.theta)/1.5), phase=phase)
        self.theta = target

    def _depth(self, depth, phase):
        # The shaft runs in bores with 0.75-1 mm play; move slowly and re-centre the
        # measured key every quarter pitch so arm tracking error cannot reach a bore.
        start = self._key_depth()
        stop = self._entry_stop(start, depth)
        end = depth if stop is None else stop
        pieces = max(1, int(np.ceil(abs(end-start)/.25-1e-9)))
        for k in range(1, pieces+1):
            target = start+(end-start)*k/pieces
            yield from self._move_key(depth=target, angle=self.theta,
                                      seconds=max(1.5, 7.*abs(end-start)/pieces), phase=phase,
                                      tolerance=self.entry_tolerance if stop is not None and k == pieces else None)
        if stop is not None:
            yield from self._move_key(depth=depth, angle=self.theta, phase=phase,
                                      seconds=max(3., abs(depth-stop)*self.geometry["pitch"]/.001))

    def _entry_stop(self, start, depth):
        """The depth 1.5 mm before the tabs meet the faces of the plates of the level `depth` (from either side), if
        the move from `start` crosses it; None for moves into a gap."""
        level = self.n+1 if depth > self.n+1 else round(depth)
        if abs(depth-level) > .1 or level < 1:
            return None
        side = 1. if start > depth else -1.
        stop = level+side*(self.entry_reach+self.entry_margin)
        return stop if (start-stop)*side > 0 else None

    def _key_depth(self):
        g = self.geometry
        return 1.+(self._key()[0][0]-g["first_x"])/g["pitch"]

    def _go(self, level, index):
        start = self.level
        if abs(level-start) != 1:
            raise ValueError("Plan steps move one level at a time")
        if start >= 1:
            centre, _ = self._window(start)
            yield from self._rotate(centre, f"{index:03d}_centre_L{start}")
        yield from self._depth((start+level)/2, f"{index:03d}_gap_L{start}_L{level}")
        if level >= 1:
            centre, _ = self._window(level)
            yield from self._rotate(centre, f"{index:03d}_align_L{level}")
        depth = self.endpoint_depth if level == self.n+1 else float(level)
        yield from self._depth(depth, f"{index:03d}_enter_L{level}")
        self.level = level

    def _turn(self, level, targets, index):
        if level != self.level or not 1 <= level <= self.n:
            raise ValueError("A turn must happen at the current rotor level")
        discs = [0] if level == 1 else [level-2, level-1]
        if len(targets) != len(discs):
            raise ValueError("One target angle per engaged rotor is required")
        reference, goal = discs[-1], targets[-1]
        for attempt in range(4):
            error = goal-self._disc_angle(reference)
            if abs(error) < .35 and all(abs(t-self._disc_angle(d)) < .6 for d, t in zip(discs, targets)):
                break
            centre, _ = self._window(level)
            side = np.sign(error)
            # The pushing tab sits one backlash beyond the slot centre it drives.
            # Approach fast to 3 deg short of it, then push the rest slowly so the
            # key's tracking overshoot does not carry the rotors past the target.
            push = centre+error+side*self.backlash
            if abs(error) > 3.:
                # While a tab pushes, the reaction shoves the shaft against the bores;
                # re-centre the measured key every 8 degrees so it cannot tilt and bind.
                yield from self._rotate(push-side*3., f"{index:03d}_turn_L{level}_{attempt}", max_step=8.)
            yield from self._rotate_slowly(push, f"{index:03d}_push_L{level}_{attempt}")
            yield from self._hold(.4, require_grasp=True)
        else:
            raise RuntimeError(f"Rotor turn did not converge at step {index}: "
                               f"{[round(self._disc_angle(d), 2) for d in discs]} vs {targets}")
        centre, _ = self._window(level)
        yield from self._rotate(centre, f"{index:03d}_release_L{level}")

    def _actions(self):
        if self.part == "pickup":
            yield from super()._actions()
            return
        self.grip = .8
        self.precise_insertion = True
        for index, step in enumerate(self.plan):
            if step[0] == "go":
                yield from self._go(int(step[1]), index)
            elif step[0] == "turn":
                yield from self._turn(int(step[1]), [float(v) for v in step[2]], index)
            else:
                raise ValueError(f"Unknown plan step {step!r}")
        if self.level != self.n+1:
            raise RuntimeError("The plan did not finish at END")
        self.phase = "confirm_insertion"
        yield from self._hold(1., require_grasp=True)
