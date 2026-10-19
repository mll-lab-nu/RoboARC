"""IK-first chain-lock instances: the oracle route comes first, the lock is derived from it.

Physical picture (tasks/lock_picking/chain.py). N rotors on a 120 mm pitch, then a
fixed END plate; the key has a long tip tab and a short tab one pitch behind. At
level L the long tab sits in rotor L and the short tab in rotor L-1, and turning
the key turns rotor 1 alone (L = 1) or that adjacent pair.

optimize() keeps this order:
  1. Route. The oracle's turns and level changes, and one long-tab slot per rotor
     ("passage"). At level L the key angle is rotor L's angle plus its passage
     (END's slot at level N+1), so the key's whole path is fixed here.
  2. IK. The full robot path (default home, pickup, every move of the route) is
     lifted to Genesis IK and checked for joint limits, continuity and robot/scene
     clearance on a lock that has only the passages. Slots lie within 45 mm of the
     axis, where the robot never reaches, so the check does not depend on them.
  3. Lock. Only after the IK passes are the short tab's slots cut where the route
     needs them, the stops set around the angles the route uses and the decoys
     added. The exact search must return this route, step for step, as the lock's
     unique shortest solution.
"""
from dataclasses import dataclass

import numpy as np

from topogym.assets.geometry.lock_picking_chain import ChainLockGeometry
from topogym.planning.lock_picking.chain_route import add_decoys, certify, route_waypoints, wrap
from topogym.tasks.lock_picking.chain import chain_parameters

GRASP_ROLL_DEG = -45.    # side grasp of the lock oracles (runner --grasp-roll-deg)
END_DEPTH_PAST = 1.08    # the oracle inserts to rotors + 1.08 pitches (runner default)
# Executed paths came within about 1 mm of the IK gap (easier on the original tray:
# 0.19 mm by IK, 1.06 mm executed), so a route needs this much room from the scene.
MIN_CLEARANCE_M = .005


@dataclass(frozen=True)
class ChainSpec:
    """One tier: the oracle route and the tolerances its lock is built with."""
    initial_deg: tuple        # rotor angles at the start
    steps: tuple              # ("go", L) or ("turn", L, target angles of the engaged rotors)
    stop_pad_deg: tuple       # each rotor's stops sit this far beyond the angles the route uses
    slot_half_deg: float = 8.
    shaft_radius: float = .0065
    key_offset_deg: float = 0.    # every passage and END turned by this, to keep the key in the arm's roll range
    decoys: tuple = ()            # (rotor index, "shallow"/"deep", angle) in acceptance order


@dataclass(frozen=True)
class Route:
    steps: tuple
    rotor_angles: tuple       # before the route, then after every step
    passages_deg: tuple       # each rotor's long-tab slot, key offset included
    end_deg: float
    stops_deg: tuple          # (lo, hi) of each rotor
    entry_offsets_deg: tuple  # per rotor i: angle of rotor i+1 (END: 0) relative to rotor i at entries to level i+2


def ratchet_steps(strokes, initial_deg):
    """Route that turns the last rotor from its initial angle to 0 by nested ratchets.

    The pair (k-1, k) advances rotor k by at most the stroke left to rotor k-1. When
    rotor k-1 is at the end of its stroke, the key goes back, resets it to the other
    end (recursively; rotor 1 turns alone at level 1) and returns. strokes[k] is
    (lo, hi) of rotor k for every rotor but the last. Ends by entering END.
    """
    n = len(initial_deg)
    a = [float(x) for x in initial_deg]
    steps, level = [], 0

    def go(target):
        nonlocal level
        while level != target:
            level += 1 if target > level else -1
            steps.append(("go", level))

    def turn(at, delta):
        go(at)
        engaged = [0] if at == 1 else [at-2, at-1]
        for i in engaged:
            a[i] += delta
        steps.append(("turn", at, tuple(a[i] for i in engaged)))

    def move(k, delta):
        sign = 1 if delta > 0 else -1
        while delta:
            if k < len(strokes) and not strokes[k][0] <= a[k]+delta <= strokes[k][1]:
                raise ValueError(f"Rotor {k+1} would leave its stroke")
            if k == 0:
                turn(1, delta)
                return
            lo, hi = strokes[k-1]
            room = (hi if sign > 0 else lo)-a[k-1]
            if room == 0:
                move(k-1, (lo if sign > 0 else hi)-a[k-1])
                continue
            step = sign*min(abs(delta), abs(room))
            turn(k+1, step)
            delta -= step

    move(n-1, -a[n-1])
    go(n+1)
    return tuple(steps)


def plan_route(spec):
    """Step 1: rotor angles along the route, the passages and the stops."""
    n = len(spec.initial_deg)
    a = [float(x) for x in spec.initial_deg]
    angles = [tuple(a)]
    offsets = [set() for _ in range(n)]
    for step in spec.steps:
        if step[0] == "turn":
            engaged = [0] if step[1] == 1 else [step[1]-2, step[1]-1]
            for i, target in zip(engaged, step[2]):
                a[i] = float(target)
        elif step[1] >= 2:
            lower, upper = step[1]-2, step[1]-1
            offsets[lower].add(int(wrap((0. if upper == n else a[upper])-a[lower])))
        angles.append(tuple(a))
    offsets = [sorted(row) for row in offsets]
    separation = int(2*spec.slot_half_deg)+1
    for i, row in enumerate(offsets):
        if any(abs(wrap(x-y)) <= separation for j, x in enumerate(row) for y in row[j+1:]):
            raise ValueError(f"The route needs overlapping slots in rotor {i+1}")
    # The last rotor's passage is where it meets END. An earlier rotor's passage
    # doubles as a short-tab slot when one fits (fewer slots), otherwise it is the
    # free 5-degree angle nearest 0.
    passages = [None]*n
    passages[n-1] = offsets[n-1][0]
    for i in range(n-2, -1, -1):
        needed = [int(wrap(passages[i+1]+r)) for r in offsets[i]]
        best = None
        for c in range(-180, 180, 5):
            if all(abs(wrap(c-s)) > separation or abs(wrap(c-s)) < 1e-9 for s in needed):
                score = (0 if any(abs(wrap(c-s)) < 1e-9 for s in needed) else 1, abs(c))
                if best is None or score < best[0]:
                    best = (score, c)
        if best is None:
            raise ValueError(f"No long-tab slot fits rotor {i+1}")
        passages[i] = best[1]
    used = np.asarray(angles)
    stops = tuple((float(used[:, i].min()-pad), float(used[:, i].max()+pad)) for i, pad in enumerate(spec.stop_pad_deg))
    return Route(steps=tuple(spec.steps), rotor_angles=tuple(angles),
                 passages_deg=tuple(float(wrap(p+spec.key_offset_deg)) for p in passages),
                 end_deg=float(spec.key_offset_deg), stops_deg=stops,
                 entry_offsets_deg=tuple(tuple(row) for row in offsets))


def key_waypoints(spec, route):
    """The key's object-space path of the route, for kinematics.lift_route."""
    g = ik_scene_parameters(spec, route)["geometry"]
    return route_waypoints(list(route.steps), [list(a) for a in route.rotor_angles], passages=list(route.passages_deg),
                           end_deg=route.end_deg, backlash_deg=g["slot_half_deg"]-g["tab_half_deg"],
                           endpoint_depth=len(route.passages_deg)+END_DEPTH_PAST)


def plan_steps(steps):
    """Steps in one comparable form: ["go", L] or ["turn", L, [angles]]."""
    return [["go", int(s[1])] if s[0] == "go" else ["turn", int(s[1]), [float(a) for a in s[2]]] for s in steps]


def _parameters(spec, route, shallow):
    return chain_parameters(deep=[[p] for p in route.passages_deg], shallow=shallow, initial=spec.initial_deg,
                            stops=route.stops_deg, slot_half_deg=spec.slot_half_deg,
                            shaft_radius=spec.shaft_radius, endpoint_deg=route.end_deg)


def ik_scene_parameters(spec, route):
    """The lock the IK runs on: the passages only, no short-tab slot or decoy."""
    return _parameters(spec, route, [[] for _ in route.passages_deg])


def lock_parameters(spec, route, *, decoys=True):
    """Step 3: the short tab's slots where the route needs them, then the spec's decoys."""
    n = len(route.passages_deg)
    base = [p-spec.key_offset_deg for p in route.passages_deg]
    shallow = []
    for i in range(n):
        needed = route.entry_offsets_deg[i] if i == n-1 else [base[i+1]+r for r in route.entry_offsets_deg[i]]
        row = sorted(x for x in {int(wrap(v)) for v in needed} if abs(wrap(x-base[i])) > 1e-9)
        shallow.append([float(wrap(x+spec.key_offset_deg)) for x in row])
    parameters = _parameters(spec, route, shallow)
    if decoys:
        geometry = parameters["geometry"]
        for i, kind, angle in spec.decoys:
            geometry["slots" if kind == "deep" else "shallow_slots"][i].append(float(angle))
    return parameters


def check_ik(spec, route, work_dir, *, min_clearance_m=MIN_CLEARANCE_M):
    """Step 2: lift the route to IK on the passages-only lock and measure clearance (no tick)."""
    import torch
    from topogym.planning.lock_picking.kinematics import lift_route, route_clearance
    from topogym.simulator import Env, SimulatorConfig
    torch.set_num_threads(1)
    config = SimulatorConfig(task_id="lock_picking", variation_id="lock_picking_design",
                             task_parameters=ik_scene_parameters(spec, route), robot_id="panda_robotiq",
                             compute_backend="cpu", precision="64", enable_rgb=False, enable_privileged=True,
                             work_dir=work_dir)
    with Env(config) as env:
        env.reset(seed=0)
        report, samples = lift_route(env, key_waypoints(spec, route), grasp_roll_deg=GRASP_ROLL_DEG)
        clearance = route_clearance(env, samples) if report["passed"] else None
        scene_instance_id = env._task.instance_id
    gap = clearance["minimum_gaps"]["robot_fixture_m"]["gap_m"] if clearance else None
    passed = bool(report["passed"] and clearance["passed"] and gap >= min_clearance_m)
    return dict(passed=passed, scene_instance_id=scene_instance_id, minimum_robot_scene_gap_m=min_clearance_m,
                robot_scene_gap_m=gap, ik=report, clearance=clearance), samples


def optimize(spec, work_dir):
    """Route, then IK (must pass), then the lock and its certificate. Returns (record, IK samples)."""
    route = plan_route(spec)
    ik, samples = check_ik(spec, route, work_dir)
    record = dict(route=dict(steps=plan_steps(route.steps), passages_deg=list(route.passages_deg),
                             end_deg=route.end_deg, stops_deg=[list(s) for s in route.stops_deg]),
                  ik=ik, parameters=None, accepted=False)
    if not ik["passed"]:
        return record, samples
    without = lock_parameters(spec, route, decoys=False)
    plain = ChainLockGeometry(**without["geometry"])
    _, kept = add_decoys(plain, without["rotor_stops"], list(spec.decoys), len(spec.decoys))
    parameters = lock_parameters(spec, route)
    geometry = ChainLockGeometry(**parameters["geometry"])
    geometry.validate()
    certificate = certify(geometry, parameters["rotor_stops"])
    same_route = plan_steps(certificate["plan"]) == plan_steps(route.steps)
    record.update(parameters=parameters, decoys_kept=[list(d) for d in kept], certificate=certificate,
                  certified_plan_is_route=same_route,
                  accepted=bool(same_route and len(kept) == len(spec.decoys) and certificate["unique_shortest_route"]
                                and certificate["robust_to_tilt_bound"]))
    return record, samples


SPECS = {
    # Turn rotor 1 alone onto rotor 2's slot, then the pair onto END.
    "very_easy": ChainSpec(initial_deg=(20., -25.),
                              steps=(("go", 1), ("turn", 1, (-25.,)), ("go", 2), ("turn", 2, (0., 0.)), ("go", 3)),
                              stop_pad_deg=(4., 4.)),
    "easier": ChainSpec(initial_deg=(15., -15., -50.),
                           steps=ratchet_steps(((-15., 15.), (-15., 15.)), (15., -15., -50.)),
                           stop_pad_deg=(2., 2., 7.)),
    "easy": ChainSpec(initial_deg=(-25., -25., -130.),
                         steps=ratchet_steps(((-25., 25.), (-25., 25.)), (-25., -25., -130.)),
                         stop_pad_deg=(2., 2., 7.),
                         decoys=((1, "shallow", -155.), (1, "shallow", 30.), (0, "shallow", 16.))),
    "medium": ChainSpec(initial_deg=(-18., -18., -120.),
                           steps=ratchet_steps(((-18., 18.), (-18., 18.)), (-18., -18., -120.)),
                           stop_pad_deg=(2., 2., 7.), slot_half_deg=6.,
                           decoys=((1, "shallow", -138.), (0, "shallow", -12.), (1, "shallow", 24.))),
    "hard": ChainSpec(initial_deg=(15., -15., 15., -150.),
                         steps=ratchet_steps(((-15., 15.),)*3, (15., -15., 15., -150.)),
                         stop_pad_deg=(2., 2., 2., 2.), slot_half_deg=5.5, shaft_radius=.00675, key_offset_deg=20.,
                         decoys=((2, "shallow", 35.), (2, "shallow", -145.))),
}
