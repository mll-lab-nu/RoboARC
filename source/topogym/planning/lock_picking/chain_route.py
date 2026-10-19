"""Exact quasi-static search over chain-lock states, and the oracle plan it certifies.

Physical picture. The key's tip tab is long, its second tab (one pitch behind)
is short. At depth L (level L) the long tab sits in rotor L and the short tab in
rotor L-1, so the key turns rotor 1 alone (level 1) or one adjacent rotor pair.
Rotor N+1 is the fixed END plate. Between levels both tabs sit in the gaps and
the key turns freely. Entering a level needs a slot of each engaged rotor at the
key angle: the long tab fits deep slots only, the short tab deep or shallow ones.

Backlash. Each tab is narrower than its slot by 2*m (m = slot_half - tab_half on
each side). With offset o = w_upper - w_lower between the two engaged slot
centres (|o| <= 2m), turning the key in either direction first moves only the
lagging rotor until o = 0; the pair then turns together. |o| never grows while
the tabs stay engaged. Rotors move only when a tab pushes a slot wall, and keep
their angles when disengaged. Rotor stops bound every angle.

States are rotor angles on a 1-degree grid with the level. Turning costs 0 and
each level change costs 1, so the search returns the minimum number of level
changes; the oracle plan is one shortest solution. Scope: coaxial key, no face
friction, inertia or off-axis wedging; it is a design certificate, not dynamics.
"""
from collections import deque
import numpy as np


def wrap(angle):
    return (np.asarray(angle)+180.) % 360.-180.


def tilt_allowance_deg(geometry, level):
    """Extra pair offset a tilted or shifted key can absorb at a level (degrees).

    At level L the shaft runs through the bores of plates 1..L, so the lateral
    offsets e0 (long tab) and e1 (short tab) obey |e| <= c and |e0-e1| <= 2c/(L-1),
    with c the radial shaft-bore clearance. The tabs then differ in angle by up
    to |e0/r_long - e1/r_short|; this is the bound's maximum, on a fine grid.
    """
    g = geometry
    c = g.bore_radius-g.shaft_radius
    delta = 2*c/(level-1)
    e = np.linspace(-c, c, 201)
    e0, e1 = np.meshgrid(e, e)
    gap = np.abs(e0/g.tab_radius_at(0)-e1/g.tab_radius_at(1))
    return float(np.degrees(gap[np.abs(e0-e1) <= delta+1e-12].max()))


class ChainModel:
    def __init__(self, geometry, rotor_stops, *, tolerance_deg=None, stop_pad_deg=0, extra_deg=None):
        """extra_deg: optional {level: added pair-offset tolerance in degrees}."""
        g = geometry
        self.n = len(g.slots)
        self.deep = [[int(round(s)) for s in row] for row in g.slots]
        self.shallow = [[int(round(s)) for s in g.shallow_of(i)] for i in range(self.n)]
        self.end_slot = int(round(g.endpoint_deg))
        backlash = g.slot_half_deg-g.tab_half_deg
        self.tol = int(round(2*backlash if tolerance_deg is None else tolerance_deg))
        self.level_tol = {L: self.tol+int(np.ceil((extra_deg or {}).get(L, 0.)-1e-9)) for L in range(2, self.n+2)}
        self.stops = [(int(np.floor(s["bounds_deg"][0]))-stop_pad_deg, int(np.ceil(s["bounds_deg"][1]))+stop_pad_deg)
                      for s in rotor_stops]
        self.initial = [int(round(a)) for a in g.initial_disc_deg]
        for a, (lo, hi) in zip(self.initial, self.stops):
            if not lo <= a <= hi:
                raise ValueError("Initial rotor angle outside its stops")
        self.shape = tuple(hi-lo+1 for lo, hi in self.stops)
        self.axes = [np.arange(lo, hi+1) for lo, hi in self.stops]
        self._offsets = {L: self._offset_array(L) for L in range(2, self.n+2)}

    # Alignment -----------------------------------------------------------------
    def _offset_array(self, L):
        """Signed offset of the closest aligned slot pair at level L (nan if none)."""
        lower, upper = L-2, L-1
        tol = self.level_tol[L]
        wl = self.axes[lower][:, None]+np.asarray(self.deep[lower]+self.shallow[lower])[None, :]
        shape = [1]*self.n
        if upper == self.n:
            diff = wrap(self.end_slot-wl)
            diff = np.where(np.abs(diff) <= tol, diff, np.nan)
            best = np.take_along_axis(diff, np.nanargmin(np.where(np.isnan(diff), np.inf, np.abs(diff)), 1)[:, None], 1)[:, 0]
            shape[lower] = len(best)
            return best.reshape(shape)
        wu = self.axes[upper][:, None]+np.asarray(self.deep[upper])[None, :]
        diff = wrap(wu[None, :, None, :]-wl[:, None, :, None])
        diff = np.where(np.abs(diff) <= tol, diff, np.nan)
        flat = diff.reshape(diff.shape[0], diff.shape[1], -1)
        index = np.argmin(np.where(np.isnan(flat), np.inf, np.abs(flat)), -1)
        best = np.take_along_axis(flat, index[..., None], -1)[..., 0]
        shape[lower], shape[upper] = best.shape
        return best.reshape(shape)

    def aligned(self, L):
        if L <= 1:
            return np.ones((1,)*self.n, bool)
        return ~np.isnan(self._offsets[L])

    def offset(self, L, index):
        o = np.broadcast_to(self._offsets[L], self.shape)[index]
        return None if np.isnan(o) else float(o)

    # Zero-cost turning at one level -------------------------------------------
    def _shift(self, x, axis, k):
        out = np.zeros_like(x)
        src, dst = [slice(None)]*self.n, [slice(None)]*self.n
        src[axis], dst[axis] = (slice(None, -k), slice(k, None)) if k > 0 else (slice(-k, None), slice(None, k))
        out[tuple(dst)] = x[tuple(src)]
        return out

    def closure(self, states, L):
        if L == 0 or L == self.n+1:
            return states.copy()
        if L == 1:
            return np.broadcast_to(states.any(axis=0, keepdims=True), states.shape).copy()
        lower, upper = L-2, L-1
        off = np.broadcast_to(self._offsets[L], states.shape)
        valid = ~np.isnan(off)
        neg, pos, zero = valid & (off < -.5), valid & (off > .5), valid & (np.abs(off) <= .5)
        cur = states.copy()
        for _ in range(self.level_tol[L]+1):          # the lagging rotor catches up
            grow = cur | self._shift(cur & neg, upper, 1) | self._shift(cur & neg, lower, -1)
            grow |= self._shift(cur & pos, upper, -1) | self._shift(cur & pos, lower, 1)
            grow &= valid
            if (grow == cur).all():
                break
            cur = grow
        z = np.moveaxis(cur & zero, (lower, upper), (-2, -1))   # the pair turns together
        shape = z.shape
        flat = z.reshape(-1, shape[-2], shape[-1])
        out = np.zeros_like(flat)
        for k in range(-shape[-2]+1, shape[-1]):
            diagonal = np.diagonal(flat, offset=k, axis1=1, axis2=2)
            rows = np.nonzero(diagonal.any(-1))[0]
            if rows.size:
                i = np.arange(max(0, -k), max(0, -k)+diagonal.shape[-1])
                out[rows[:, None], i[None, :], (i+k)[None, :]] = True
        cur |= np.moveaxis(out.reshape(shape), (-2, -1), (lower, upper))
        return cur

    # Search ----------------------------------------------------------------------
    def reachable(self):
        """Every state reachable from the start, by level (the full search)."""
        levels = range(self.n+2)
        start = np.zeros(self.shape, bool)
        start[tuple(a-lo for a, (lo, _) in zip(self.initial, self.stops))] = True
        seen = {L: np.zeros(self.shape, bool) for L in levels}
        seen[0] |= start
        frontier = {0: start}
        while frontier:
            grown = {L: self.closure(entry, L) & ~seen[L] | entry for L, entry in frontier.items()}
            for L, states in grown.items():
                seen[L] |= states
            frontier = {}
            for L, states in grown.items():
                for M in (L-1, L+1):
                    if 0 <= M <= self.n+1:
                        new = states & self.aligned(M) & ~seen[M]
                        if new.any():
                            frontier[M] = frontier.get(M, np.zeros(self.shape, bool)) | new
            for M, new in frontier.items():
                seen[M] |= new
        return seen

    def solve(self, max_cost=400, keep_entries=False):
        """Minimum number of level changes from key-outside to END, or None."""
        levels = range(self.n+2)
        start = np.zeros(self.shape, bool)
        start[tuple(a-lo for a, (lo, _) in zip(self.initial, self.stops))] = True
        seen = {L: np.zeros(self.shape, bool) for L in levels}
        seen[0] |= start
        frontier = {0: start}
        self.entries = [] if keep_entries else None
        for cost in range(max_cost+1):
            if keep_entries:
                self.entries.append({L: np.packbits(a.ravel()) for L, a in frontier.items()})
            if self.n+1 in frontier:
                return cost
            grown = {}
            for L, entry in frontier.items():
                states = self.closure(entry, L) & ~seen[L] | entry
                seen[L] |= states
                grown[L] = states
            frontier = {}
            for L, states in grown.items():
                for M in (L-1, L+1):
                    if 0 <= M <= self.n+1:
                        new = states & self.aligned(M) & ~seen[M]
                        if new.any():
                            frontier[M] = frontier.get(M, np.zeros(self.shape, bool)) | new
            for M, new in frontier.items():
                seen[M] |= new
            if not frontier:
                return None
        return None

    def reverse_closure(self, states, L):
        """States at level L from which turning alone can reach one of the given states."""
        if L == 0 or L == self.n+1:
            return states.copy()
        if L == 1:
            return np.broadcast_to(states.any(axis=0, keepdims=True), states.shape).copy()
        lower, upper = L-2, L-1
        off = np.broadcast_to(self._offsets[L], states.shape)
        valid = ~np.isnan(off)
        neg, pos, zero = valid & (off < -.5), valid & (off > .5), valid & (np.abs(off) <= .5)
        cur = states & valid
        z = np.moveaxis(cur & zero, (lower, upper), (-2, -1))    # rigid turning is reversible
        shape = z.shape
        flat = z.reshape(-1, shape[-2], shape[-1])
        out = np.zeros_like(flat)
        for k in range(-shape[-2]+1, shape[-1]):
            diagonal = np.diagonal(flat, offset=k, axis1=1, axis2=2)
            rows = np.nonzero(diagonal.any(-1))[0]
            if rows.size:
                i = np.arange(max(0, -k), max(0, -k)+diagonal.shape[-1])
                out[rows[:, None], i[None, :], (i+k)[None, :]] = True
        cur |= np.moveaxis(out.reshape(shape), (-2, -1), (lower, upper))
        for _ in range(self.level_tol[L]+1):          # undo the lagging rotor's catch-up
            grow = cur | (self._shift(cur, upper, -1) & neg) | (self._shift(cur, lower, 1) & neg)
            grow |= (self._shift(cur, upper, 1) & pos) | (self._shift(cur, lower, -1) & pos)
            if (grow == cur).all():
                break
            cur = grow
        return cur

    def shortest_entries(self, cost):
        """For every cost, the level entries that lie on at least one shortest solution."""
        on = [dict() for _ in range(cost+1)]
        on[cost][self.n+1] = self._entry(cost, self.n+1)
        for c in range(cost, 0, -1):
            for L, target in on[c].items():
                for M in (L-1, L+1):
                    entry = self._entry(c-1, M) if 0 <= M <= self.n+1 else None
                    if entry is None:
                        continue
                    found = self.reverse_closure(target, M) & entry
                    if found.any():
                        on[c-1][M] = on[c-1].get(M, np.zeros(self.shape, bool)) | found
        return on

    def entry_slots(self, L, states):
        """(lower slot, upper slot) pairs through which the given states enter level L."""
        if L <= 1:
            return {("rotor1", self.deep[0][0])} if states.any() else set()
        lower, upper = L-2, L-1
        tol = self.level_tol[L]
        uppers = [("END", self.end_slot)] if upper == self.n else [(t, None) for t in self.deep[upper]]
        used = set()
        for s in self.deep[lower]+self.shallow[lower]:
            wl = self.axes[lower]+s
            for t, end in uppers:
                shape = [1]*self.n
                if end is not None:
                    ok = np.abs(wrap(end-wl)) <= tol
                    shape[lower] = len(ok)
                else:
                    ok = np.abs(wrap((self.axes[upper]+t)[None, :]-wl[:, None])) <= tol
                    shape[lower], shape[upper] = ok.shape
                if (states & np.broadcast_to(ok.reshape(shape), self.shape)).any():
                    used.add((s, t))
        return used

    def shortest_route_signature(self, cost):
        """Per step, the set of (level, lower slot, upper slot) used by any shortest solution."""
        steps = []
        for c, levels in enumerate(self.shortest_entries(cost)[1:], 1):
            steps.append(sorted((L, s, t) for L, states in levels.items() for s, t in self.entry_slots(L, states)))
        return steps

    # Plan recovery -------------------------------------------------------------
    def _entry(self, cost, L):
        packed = self.entries[cost].get(L)
        if packed is None:
            return None
        return np.unpackbits(packed)[:int(np.prod(self.shape))].reshape(self.shape).astype(bool)

    def _predecessors(self, L, y):
        """States one zero-cost turning move before y at level L."""
        def inside(x):
            return all(0 <= x[i] < self.shape[i] for i in range(self.n))
        result = []
        if L == 1:
            for step in (-1, 1):
                x = list(y); x[0] += step
                if inside(x):
                    result.append(tuple(x))
            return result
        if not 2 <= L <= self.n:
            return result
        lower, upper = L-2, L-1
        for axis, step, sign in ((upper, -1, -1), (lower, 1, -1), (upper, 1, 1), (lower, -1, 1)):
            x = list(y); x[axis] += step
            if inside(x):
                o = self.offset(L, tuple(x))
                if o is not None and (o < -.5 if sign < 0 else o > .5):
                    result.append(tuple(x))
        for step in (-1, 1):
            x = list(y); x[lower] += step; x[upper] += step
            if inside(x):
                o = self.offset(L, tuple(x))
                if o is not None and abs(o) <= .5:
                    result.append(tuple(x))
        return result

    def shortest_path(self, cost):
        """One shortest solution as (level, grid index) states, start to END."""
        y = tuple(int(v) for v in np.argwhere(self._entry(cost, self.n+1))[0])
        level, reverse = self.n+1, [(self.n+1, y)]
        for c in range(cost, 0, -1):
            found = None
            for M in (level-1, level+1):
                entry = self._entry(c-1, M) if 0 <= M <= self.n+1 else None
                if entry is None:
                    continue
                queue, parent = deque([y]), {y: None}
                while queue:
                    z = queue.popleft()
                    if entry[z]:
                        found = (M, z, parent)
                        break
                    for x in self._predecessors(M, z):
                        if x not in parent:
                            parent[x] = z
                            queue.append(x)
                if found:
                    break
            if found is None:
                raise RuntimeError(f"Plan recovery failed at cost {c}")
            M, z, parent = found
            segment = []
            while z is not None:
                segment.append((M, z))
                z = parent[z]
            reverse.extend(reversed(segment))
            level, y = M, segment[0][1]
        return reverse[::-1]

    def plan(self, path):
        """Compress a state path to ('go', L) and ('turn', L, [rotor angles in degrees])."""
        steps = []
        for (L0, a0), (L1, a1) in zip(path, path[1:]):
            angles = [int(self.axes[i][a1[i]]) for i in range(self.n)]
            if L0 != L1:
                steps.append(("go", L1))
            elif a0 != a1:
                engaged = [0] if L1 == 1 else [L1-2, L1-1]
                turn = ("turn", L1, [angles[i] for i in engaged])
                if steps and steps[-1][0] == "turn" and steps[-1][1] == L1:
                    steps[-1] = turn
                else:
                    steps.append(turn)
        return steps


def turn_reachable(model, level, state, goal):
    """Whether turning the key at a level can take the rotors from state to goal.

    Same rule as ChainModel.closure, for one state: the lagging rotor first closes
    the offset, then the pair turns rigidly; no intermediate angle leaves the stops.
    """
    if level == 1:
        return all(state[i] == goal[i] for i in range(1, model.n))
    lower, upper = level-2, level-1
    if any(state[i] != goal[i] for i in range(model.n) if i not in (lower, upper)):
        return False
    o0, o1 = model.offset(level, state), model.offset(level, goal)
    if o0 is None or o1 is None:
        return False
    du, dl = goal[upper]-state[upper], goal[lower]-state[lower]
    if abs(o1-o0-(du-dl)) > .5:
        return False                      # a different slot pair: not one mode
    if abs(o0) <= .5:
        return du == dl
    sign = 1 if o0 < 0 else -1            # moves that shrink |o|: upper +sign or lower -sign
    if abs(o1) > .5:
        if o1*o0 < 0 or abs(o1) > abs(o0):
            return False
        return sign*du >= 0 and sign*dl <= 0
    for k in range(min(dl, du), max(dl, du)+1):
        u, l = du-k, dl-k
        if sign*u < 0 or sign*l > 0:
            continue
        mid_upper, mid_lower = state[upper]+u, state[lower]+l
        if 0 <= mid_upper < model.shape[upper] and 0 <= mid_lower < model.shape[lower]:
            return True
    return False


def run_plan(model, steps):
    """Check a plan step by step in the exact model; return the visited states."""
    level = 0
    state = tuple(a-lo for a, (lo, _) in zip(model.initial, model.stops))
    visited = [(level, state)]
    for index, step in enumerate(steps):
        if step[0] == "go":
            target = int(step[1])
            if abs(target-level) != 1 or not np.broadcast_to(model.aligned(target), model.shape)[state]:
                raise ValueError(f"Step {index} {step}: level {target} is not enterable")
            level = target
        else:
            _, at, angles = step
            engaged = [0] if at == 1 else [at-2, at-1]
            if at != level or len(angles) != len(engaged):
                raise ValueError(f"Step {index} {step}: turn at the wrong level")
            goal = list(state)
            for disc, angle in zip(engaged, angles):
                goal[disc] = int(angle)-model.stops[disc][0]
                if not 0 <= goal[disc] < model.shape[disc]:
                    raise ValueError(f"Step {index} {step}: target outside the stops")
            if not turn_reachable(model, level, state, tuple(goal)):
                raise ValueError(f"Step {index} {step}: target not reachable by turning")
            state = tuple(goal)
        visited.append((level, state))
    if level != model.n+1:
        raise ValueError("The plan does not end at END")
    return visited


def entry_margins(model, visited):
    """Remaining backlash (deg) at every level entry of a plan, smallest first."""
    return sorted(model.level_tol[L1]-abs(model.offset(L1, state))
                  for (L0, _), (L1, state) in zip(visited, visited[1:]) if L1 != L0 and L1 >= 2)


def simplify(model, steps):
    """Drop unneeded turns, then shift turn targets to centre later entries.

    Every candidate plan is re-checked with run_plan in the exact model.
    """
    steps = [tuple(s) for s in steps]
    changed = True
    while changed:
        changed = False
        for i in range(len(steps)):
            if steps[i][0] != "turn":
                continue
            candidate = steps[:i]+steps[i+1:]
            try:
                run_plan(model, candidate)
            except ValueError:
                continue
            steps, changed = candidate, True
            break
    best = entry_margins(model, run_plan(model, steps))
    improved = True
    while improved:
        improved = False
        for i, step in enumerate(steps):
            if step[0] != "turn":
                continue
            for shift in (-1, 1, -2, 2, -3, 3):
                candidate = list(steps)
                candidate[i] = ("turn", step[1], [a+shift for a in step[2]])
                try:
                    margin = entry_margins(model, run_plan(model, candidate))
                except ValueError:
                    continue
                if margin > best:
                    steps, best, improved = candidate, margin, True
                    break
    return steps


def key_route(geometry, rotor_stops, steps, *, endpoint_depth):
    """Object-space key waypoints (depth, key angle, rotor angles) that the oracle follows.

    Before an axial move the key is centred in the current aligned slots; levels
    are changed through the mid-gap, where the key turns to the next centre. A turn
    pushes the slot walls one backlash past the target centre, then re-centres.
    Same dict format as object_route.construct_route, for kinematics.lift_route.
    """
    model = ChainModel(geometry, rotor_stops)
    visited = run_plan(model, steps)
    angles = [[float(model.axes[i][state[i]]) for i in range(model.n)] for _, state in visited]
    return route_waypoints(steps, angles, passages=[row[0] for row in model.deep], end_deg=model.end_slot,
                           backlash_deg=model.tol/2, endpoint_depth=endpoint_depth)


def route_waypoints(steps, rotor_angles, *, passages, end_deg, backlash_deg, endpoint_depth):
    """key_route's waypoints from the rotor angles before and after every plan step.

    At level L the key is centred on rotor L's long-tab slot (passages[L-1]) or on
    END's slot; no shallow slot is needed, so a route can be lifted to IK before
    the rest of the lock exists (planning/instance_optimizer).
    """
    n = len(passages)

    def centre(level, a):
        if level == n+1:
            return float(end_deg)
        return a[level-1]+passages[level-1]

    level, a = 0, rotor_angles[0]
    states = [dict(phase="start", depth=0., key_deg=0., disc_deg=list(a))]

    def add(phase, depth=None, key=None, discs=None):
        s = dict(states[-1], phase=phase)
        if depth is not None:
            s["depth"] = float(depth)
        if key is not None:
            s["key_deg"] = float(states[-1]["key_deg"]+wrap(key-states[-1]["key_deg"]))
        if discs is not None:
            s["disc_deg"] = discs
        states.append(s)

    for index, (step, new_a) in enumerate(zip(steps, rotor_angles[1:])):
        new_level = int(step[1]) if step[0] == "go" else level
        if step[0] == "go":
            if level >= 1:
                add(f"{index:03d}_centre", key=centre(level, a))
            add(f"{index:03d}_gap", depth=(level+new_level)/2)
            add(f"{index:03d}_align", key=centre(new_level, a))
            add(f"{index:03d}_enter_L{new_level}",
                depth=endpoint_depth if new_level == n+1 else new_level)
        else:
            old, new = centre(level, a), centre(level, new_a)
            push = new+np.sign(wrap(new-old))*backlash_deg
            add(f"{index:03d}_turn_L{level}", key=push, discs=list(new_a))
            add(f"{index:03d}_release", key=new)
        level, a = new_level, new_a
    return states


def decoy_slots(geometry, rotor_stops, *, wider_backlash_deg=1, step_deg=4, kinds=("shallow",)):
    """Extra through-slots that open no shortcut, yet the key can reach and use.

    For each rotor and candidate angle, one more slot of the given kind is cut.
    It qualifies when the physical optimum and the tilt-widened optimum stay
    unchanged, and some reachable state enters a level through that slot.
    Returns (rotor index, kind, angle) triples; each was checked alone.
    """
    from dataclasses import replace
    g = geometry
    base = ChainModel(g, rotor_stops)
    cost = base.solve()
    extra = {L: tilt_allowance_deg(g, L)+wider_backlash_deg for L in range(2, base.n+2)}
    wide_cost = ChainModel(g, rotor_stops, extra_deg=extra).solve(max_cost=cost)
    found = []
    for i in range(base.n):
        rows = [list(r) for r in g.slots], [list(g.shallow_of(k)) for k in range(base.n)]
        for kind in kinds:
            for angle in range(-180, 180, step_deg):
                existing = rows[0][i]+rows[1][i]
                if any(abs(float(wrap(angle-s))) <= 2*g.slot_half_deg+1 for s in existing):
                    continue
                deep, shallow = [list(r) for r in rows[0]], [list(r) for r in rows[1]]
                (deep if kind == "deep" else shallow)[i].append(float(angle))
                candidate = replace(g, slots=tuple(map(tuple, deep)), shallow_slots=tuple(map(tuple, shallow)))
                model = ChainModel(candidate, rotor_stops)
                if model.solve(max_cost=cost) != cost:
                    continue
                if ChainModel(candidate, rotor_stops, extra_deg=extra).solve(max_cost=cost) != wide_cost:
                    continue
                live = live_slots(model, model.reachable())[(i, kind, float(angle))]
                if live:
                    found.append((i, kind, float(angle)))
    return found


def live_slots(model, seen):
    """For every rotor slot, whether some reachable state enters a level through it."""
    result = {}
    for i in range(model.n):
        for kind, rows in (("deep", model.deep[i]), ("shallow", model.shallow[i])):
            for s in rows:
                used = False
                for L in ((i+1, i+2) if kind == "deep" else (i+2,)):
                    if not 2 <= L <= model.n+1 or not seen[L].any():
                        continue
                    lower, upper = L-2, L-1
                    tol = model.level_tol[L]
                    shape = [1]*model.n
                    if lower == i:
                        w = model.axes[i]+s
                        if upper == model.n:
                            ok = np.abs(wrap(model.end_slot-w)) <= tol
                            shape[i] = len(ok)
                        else:
                            wu = model.axes[upper][:, None]+np.asarray(model.deep[upper])[None, :]
                            ok = (np.abs(wrap(wu[None, :, :]-w[:, None, None])) <= tol).any(-1)
                            shape[i], shape[upper] = ok.shape
                    else:
                        wl = model.axes[lower][:, None]+np.asarray(model.deep[lower]+model.shallow[lower])[None, :]
                        w = model.axes[i]+s
                        ok = (np.abs(wrap(w[None, :, None]-wl[:, None, :])) <= tol).any(-1)
                        shape[lower], shape[i] = ok.shape
                    used |= bool((seen[L] & np.broadcast_to(ok.reshape(shape), model.shape)).any())
                result[(i, kind, float(s))] = used
    return result


def add_decoys(geometry, rotor_stops, candidates, count, *, wider_backlash_deg=1):
    """Greedily add candidate slots that stay off every shortest solution.

    A candidate is kept when the optimum and the tilt-widened optima are unchanged,
    the set of (level, slot pair) entries used by shortest solutions is unchanged
    (so the decoy lies on no shortest solution), and every slot remains reachable.
    """
    from dataclasses import replace
    g = geometry
    base = ChainModel(g, rotor_stops)
    cost = base.solve(keep_entries=True)
    signature = base.shortest_route_signature(cost)
    widenings = [{L: tilt_allowance_deg(g, L)+margin for L in range(2, base.n+2)}
                 for margin in (0., wider_backlash_deg)]
    reference = [ChainModel(g, rotor_stops, extra_deg=w).solve(max_cost=cost) for w in widenings]
    deep = [list(r) for r in g.slots]
    shallow = [list(g.shallow_of(k)) for k in range(base.n)]
    chosen = []
    for i, kind, angle in candidates:
        if len(chosen) == count:
            break
        rows = deep if kind == "deep" else shallow
        if any(abs(float(wrap(angle-s))) <= 2*g.slot_half_deg+1 for s in deep[i]+shallow[i]):
            continue
        rows[i].append(float(angle))
        candidate = replace(g, slots=tuple(map(tuple, deep)), shallow_slots=tuple(map(tuple, shallow)))
        model = ChainModel(candidate, rotor_stops)
        ok = model.solve(max_cost=cost, keep_entries=True) == cost and all(
            ChainModel(candidate, rotor_stops, extra_deg=w).solve(max_cost=cost) == r
            for w, r in zip(widenings, reference))
        ok = ok and model.shortest_route_signature(cost) == signature
        if ok:
            live = live_slots(model, model.reachable())
            ok = all(live.values())
        if ok:
            chosen.append((i, kind, float(angle)))
            g = candidate
        else:
            rows[i].pop()
    return g, chosen


def certify(geometry, rotor_stops, *, stop_margin_deg=2, wider_backlash_deg=1):
    """Certify the shortest solution of the physical lock and an oracle plan for it.

    The oracle plan is searched with every stop pulled in by stop_margin_deg, so
    no planned turn presses a rotor pin against its stop; it must still be as
    short as the physical optimum. The optimum is re-solved with each level's
    pair tolerance widened by the tilt/shift bound plus wider_backlash_deg, to show
    that the shaft's play in the bores opens no shortcut.
    """
    physical = ChainModel(geometry, rotor_stops)
    cost = physical.solve()
    if cost is None:
        raise ValueError("The lock has no solution in the quasi-static model")
    narrow = ChainModel(geometry, rotor_stops, stop_pad_deg=-stop_margin_deg)
    physical.solve(keep_entries=True)
    signature = physical.shortest_route_signature(cost)
    planned = narrow.solve(keep_entries=True)
    if planned != cost:
        raise ValueError(f"Keeping {stop_margin_deg} deg from the stops lengthens the plan: {planned} vs {cost}")
    steps = simplify(narrow, narrow.plan(narrow.shortest_path(planned)))
    visited = run_plan(physical, steps)
    margin_deg = entry_margins(physical, visited)[0]
    tilt = {L: tilt_allowance_deg(geometry, L) for L in range(2, physical.n+2)}
    bound = ChainModel(geometry, rotor_stops, extra_deg=tilt).solve(max_cost=cost)
    extra = {L: tilt[L]+wider_backlash_deg for L in tilt}
    tilted = ChainModel(geometry, rotor_stops, extra_deg=extra)
    wide = tilted.solve(max_cost=cost)
    levels = [0]+[s[1] for s in steps if s[0] == "go"]
    moves = [b-a for a, b in zip(levels, levels[1:])]
    keys = []
    for level, state in visited:
        if 1 <= level <= physical.n:
            keys.append(float(physical.axes[level-1][state[level-1]]+physical.deep[level-1][0]))
    margin = (physical.tol/2)
    return dict(
        minimum_level_changes=cost, turns=sum(s[0] == "turn" for s in steps),
        retreats=sum(1 for i, d in enumerate(moves) if d < 0 and (i == 0 or moves[i-1] > 0)),
        backward_level_changes=sum(1 for d in moves if d < 0),
        tolerance_deg=physical.tol, tilt_allowance_deg={str(L): v for L, v in tilt.items()},
        minimum_with_tilt_bound=bound, robust_to_tilt_bound=bound in (None, cost),
        wider_tolerance_by_level_deg={str(L): v for L, v in tilted.level_tol.items()},
        minimum_with_wider_backlash=wide, robust=wide in (None, cost),
        stop_margin_deg=stop_margin_deg, minimum_entry_backlash_left_deg=margin_deg,
        unique_shortest_route=all(len(step) == 1 for step in signature),
        shortest_route_choices=[[list(map(float, e)) for e in step] for step in signature if len(step) > 1],
        key_angle_range_deg=[min(keys)-margin, max(keys)+margin], plan=[list(s) for s in steps],
        scope="Coaxial quasi-static tab/slot/stop model on a 1-degree grid; not a dynamics proof")
