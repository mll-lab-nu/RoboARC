"""ALOHA execution of a virtual-hand Blanket route with pbd_local_patch_v1.

Every phase is planned from the measured state at its start and executed as
native JOINT_ABS actions through SegmentRunner. The policy never writes cloth
or robot state: it sends joint and gripper targets only. The assist attaches a
patch when both real finger surfaces are within its admission distance, and
releases it on an opening command. IK is Genesis' native solver with checked
residuals, seeded along the path.
"""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.robot.grasp_geometry import read_grasp_geometry
from topogym.planning.oracle_ik import OracleIKSolver as GenesisIKSolver
from topogym.utils.arrays import to_numpy
from topogym.utils.grasp import material_patch
from topogym.utils.io import to_jsonable, write_json_atomic
from .object_oracle import segment_goal
from .routes import PATCH, route_for

SIDES = ("left", "right")
# Local +x (approach) points down; local y (closing) along world -y. Half-turn
# variants about world z are searched per arm for joint-limit margin.
FAST_TCP_M_S = .12         # route["fast"]: the descent's TCP speed
DOWN = Rotation.from_quat([.7071067811865476, 0., -.7071067811865476, 0.])
PRESHAPE_M, PINCH_M, OPEN_M = .015, .006, .037   # native finger rail coordinates
FINGERTIP_GAP_M = .0015                          # fingertip clearance above the patch top at the start of contact
MIN_OVERLAP_VOTES = 20                           # fold_stage: fewer flap particles over / under the sheet = an exposed flap


class Sequence:
    """Precomputed actions; optionally stop early when the observation says so."""
    def __init__(self, actions, stop=None):
        self.actions, self.stop, self.index = list(actions), stop, 0

    def act(self, obs):
        if self.index == len(self.actions) or (self.stop is not None and self.stop(obs)):
            return None
        self.index += 1
        return self.actions[self.index-1]


def held(obs):
    hands = obs["privileged"]["world"].get("grasp", {}).get("hands", {})
    return {side: bool(hands.get(side, {}).get("binding")) for side in SIDES}


class Arms:
    def __init__(self, env):
        self.env, self.robot = env, env._robot
        self.ik = GenesisIKSolver(self.robot.entity)
        self.q_index = {s: list(self.robot._q_indices[s][:6]) for s in SIDES}
        self.dof_index = {s: list(self.robot._indices[s][:6]) for s in SIDES}
        self.limits = {s: np.asarray(env.action_spec["groups"][s]["joint_limits_rad"]) for s in SIDES}
        self.rotation = {}

    def qpos(self):
        return to_numpy(self.robot.entity.get_qpos()).reshape(-1).copy()

    def tcp(self):
        measured = self.robot.observe()
        return {s: np.asarray(measured[s]["tcp_position_world_m"]) for s in SIDES}

    def solve(self, side, position, seed, rotation=None):
        group = self.robot.groups[side]
        rotation = self.rotation[side] if rotation is None else rotation
        hand = rotation*Rotation.from_quat(np.roll(group.tcp_orientation_in_hand_wxyz, -1)).inv()
        return self.ik.solve_checked(link_name=group.hand_link,
            position_world_m=np.asarray(position)-hand.apply(group.tcp_position_in_hand_m),
            orientation_wxyz=hand.as_quat()[[3, 0, 1, 2]], initial_qpos=seed,
            joint_dof_indices=self.dof_index[side], max_solver_iterations=300)

    def margin(self, side, q):
        value, limit = q[self.q_index[side]], self.limits[side]
        return float(np.min(np.minimum(value-limit[:, 0], limit[:, 1]-value)))

    def action(self, q, grip):
        return {s: {"joint_position_rad": q[self.q_index[s]].tolist(), "gripper_position_m": float(grip[s])}
                for s in SIDES}

    def fingertip_extension(self, side):
        """Actual finger collision surfaces beyond the TCP along the approach axis."""
        measured = self.robot.observe()[side]
        tcp = np.asarray(measured["tcp_position_world_m"])
        axis = Rotation.from_quat(np.roll(measured["tcp_orientation_wxyz"], -1)).apply([1., 0., -.023])
        axis /= np.linalg.norm(axis)
        return float(max(((v-tcp) @ axis).max() for v, _ in read_grasp_geometry(self.robot, side)["surfaces"]))


    def jaw_axis(self, side):
        """The jaws' closing axis in the TCP frame: the line between the two fingers' surface centroids."""
        measured = self.robot.observe()[side]
        R = Rotation.from_quat(np.roll(measured["tcp_orientation_wxyz"], -1))
        a, b = (v.mean(axis=0) for v, _ in read_grasp_geometry(self.robot, side)["surfaces"][:2])
        axis = R.inv().apply(np.asarray(b)-np.asarray(a))
        return axis/np.linalg.norm(axis)


STACK_LIFT_EXCLUDE_M, STACK_LIFT_RADIUS_M = .05, .012


def stack_lift(x, uv, center, patch_goal, particle_size):
    """How far to raise a held patch's goal so it lands on the cloth under it rather than inside: the stack top under the
    goal (the highest particle within STACK_LIFT_RADIUS_M in xy, leaving out the held material within
    STACK_LIFT_EXCLUDE_M of the patch centre) plus one particle (the held layer resting on it); 0 where no cloth lies
    under the goal or the goal is already above it."""
    away = np.linalg.norm(uv-uv[center], axis=1) > STACK_LIFT_EXCLUDE_M
    near = away & (np.linalg.norm(x[:, :2]-np.asarray(patch_goal)[:2], axis=1) <= STACK_LIFT_RADIUS_M)
    if not near.any():
        return 0.
    floor = float(x[near, 2].max())+float(particle_size)
    return max(0., floor-float(patch_goal[2]))


def footing_mismatch(x, point, yaw, jaw_tcp, *, half_span_m, radius_m, tolerance_m):
    """The height difference between the cloth tops under the two fingertips of a top-down pinch at `point` with this
    yaw: each fingertip half_span_m from the point along the jaw axis; a top is the highest particle within radius_m
    (xy) of it, the table (the lowest particle) where none is."""
    axis = (Rotation.from_euler("z", yaw, degrees=True)*DOWN).apply(jaw_tcp)
    axis = axis[:2]/max(np.linalg.norm(axis[:2]), 1e-9)
    tops = []
    for sign in (-1., 1.):
        f = np.asarray(point[:2])+sign*half_span_m*axis
        near = np.linalg.norm(x[:, :2]-f, axis=1) <= radius_m
        tops.append(float(x[near, 2].max()) if near.any() else float(x[:, 2].min()))
    return abs(tops[0]-tops[1])


def smoothstep(s):
    return s*s*(3.-2.*s)


def export_robot_geometry(env, output):
    """Imported collision surfaces in authored link frames, for the offline audit."""
    arrays, links = {}, []
    for link in env._robot.entity.links:
        p = to_numpy(link.get_pos()).reshape(3)
        r = Rotation.from_quat(to_numpy(link.get_quat()).reshape(4)[[1, 2, 3, 0]])
        keys = []
        for geom in link.geoms:
            key = f"geom_{len(arrays)//2}"
            arrays[key+"_vertices"] = r.inv().apply(to_numpy(geom.get_verts()).reshape(-1, 3)-p)
            arrays[key+"_faces"] = np.asarray(geom.init_faces)
            keys.append(key)
        links.append(dict(name=link.name, index=int(link.idx), local_index=int(link.idx_local), geoms=keys))
    np.savez_compressed(output/"robot_geometry.npz", **arrays)
    write_json_atomic(output/"robot_geometry.json", dict(links=links,
        sha256=hashlib.sha256((output/"robot_geometry.npz").read_bytes()).hexdigest(),
        frame="Genesis relative=True authored link frames", source="imported robot collision surfaces"))


class Contact:
    """Lower each unheld hand 1 mm per step until the assist admits it.

    Every step re-reads the live cloth: the fingertips are commanded to the
    pinched material point's current XY and to `depth` below the hover height
    over its patch, so a patch that sags or slides is followed rather than
    approached at a stale pose. Admitted hands keep their command.

    With `hold_admitted` (2026-10-07): the arms do not hold a pose exactly
    (2-6 mm), and an admitted hand that waits for the other drags and presses
    its held patch into the layer under it. So unheld hands descend together:
    up to 3 mm a step while their measured finger distance exceeds `ready_m`,
    then they wait there until every unheld hand has arrived and step 1 mm
    together; an admitted hand is held at the measured TCP of its admission;
    and a lost hand ends the phase (the assist re-admits it only after the
    jaws open, and lowering it further crushes the stack).
    """
    def __init__(self, arms, commands, q, grip, *, hover, centers, uv, patch, step_m=.001, max_steps=25, dwell=10,
                 sides=SIDES, hold_admitted=False, ready_m=.0085):
        self.arms, self.commands, self.q, self.grip = arms, commands, q, grip
        self.sides = tuple(sides)     # oraclefix 11: a one-hand stage lowers only its own hand
        self.hover, self.centers, self.uv, self.patch = hover, centers, uv, patch
        self.step_m, self.max_steps, self.dwell = step_m, max_steps, dwell
        self.depth = dict.fromkeys(SIDES, 0.)
        self.steps, self.tick, self.log = 0, 0, []
        self.hold_admitted, self.ready_m, self.held_tcp = hold_admitted, ready_m, {}

    def act(self, obs):
        state = held(obs)
        if all(state[s] for s in self.sides):
            return None
        if self.tick % self.dwell == 0:
            if self.steps == self.max_steps:
                return None
            self.steps += 1
            world = obs["privileged"]["world"]
            x = np.asarray(world["particles"]["blanket"]["position_m"])
            self.log.append(dict(step=self.steps, held=state, admission=world.get("grasp", {}).get("admission", {}),
                                 commands={s: c.tolist() for s, c in self.commands.items()}))
            step = dict.fromkeys(self.sides, self.step_m)
            if self.hold_admitted:
                grasp = world.get("grasp", {})
                if any(grasp.get("hands", {}).get(s, {}).get("phase") == "lost" for s in self.sides):
                    return None
                tcp = self.arms.tcp()
                # Finger surface to nearest particle centre; NaN when not measured (strain rejection), which keeps the
                # 1 mm step and is never ready (NaN compares false).
                admission = grasp.get("admission", {})
                gap = {s: min(admission.get(s, {}).get("minimum_finger_distances_m") or [np.nan]) for s in self.sides}
                waiting = [s for s in self.sides if not state[s]]
                ready = all(gap[s] <= self.ready_m for s in waiting)
                for s in waiting:
                    if gap[s] > self.ready_m:
                        step[s] = min(max(gap[s]-self.ready_m, self.step_m), .003)
                    elif gap[s] <= self.ready_m and not ready:
                        step[s] = 0.
            for side in self.sides:
                if not state[side]:
                    self.depth[side] += step[side]
                    center = self.centers[side]
                    top = float(x[material_patch(x, self.uv, center, **self.patch), 2].max())
                    self.commands[side] = np.array([*x[center][:2], top+self.hover[side]-self.depth[side]])
                elif self.hold_admitted:
                    held_tcp = self.held_tcp.setdefault(side, np.asarray(tcp[side], float))
                    self.commands[side] = self.commands[side]+(held_tcp-np.asarray(tcp[side], float))
                self.q = self.arms.solve(side, self.commands[side], self.q)
        self.tick += 1
        return self.arms.action(self.q, self.grip)


def cartesian(arms, q, starts, goals, duration_s, dt, grip):
    """Smoothstep TCP interpolation, checked IK per control tick, seeded along the path."""
    actions, worst = [], np.inf
    count = max(1, int(round(duration_s/dt)))
    for i in range(count):
        u = smoothstep((i+1)/count)
        for side in goals:            # oraclefix 11: both hands, or the one hand of a one-hand stage
            q = arms.solve(side, starts[side]+u*(goals[side]-starts[side]), q)
            worst = min(worst, arms.margin(side, q))
        actions.append(arms.action(q, grip))
    return actions, q, worst


class Quiet:
    """oraclefix 11: a Sequence stop that ends a wait once every cloth particle has stayed at or below the Task's own
    speed limit for its stable window (sampled per control step); the wait's length stays the cap."""
    def __init__(self, limit_m_s, window_s, dt):
        self.limit, self.need, self.count, self.done = float(limit_m_s), max(1, int(round(window_s/dt))), 0, False

    def __call__(self, obs):
        v = np.asarray(obs["privileged"]["world"]["particles"]["blanket"]["velocity_m_s"], float)
        self.count = self.count+1 if float(np.linalg.norm(v, axis=1).max()) <= self.limit else 0
        self.done = self.count >= self.need
        return self.done

def fold_stage(x, uv, table, layout, fold, tcp):
    """oraclefix 11: one stage that lays out the largest folded corner of the measured sheet; (stage or None, diagnosis).

    Flat reference: the rigid 2-D fit (rotation and translation; a face-down sheet fits mirrored) of the rest
    coordinates onto the single-layer core of the sheet. Particles farther than `deviation_m` from that fit are folded;
    the corner with the most folded particles within `corner_radius_m` (material) is laid out if it has at least
    `min_particles`. Over (the flap lies on top of the sheet): grasp the flap's tip and turn it back like a page onto its
    flat position, as route stage A turns the ends. Under (tucked): grasp the raised crest next to the flap, drag it
    outward past the flat position at 10 cm so the flap hangs below the hand, then lower while moving back 3 cm, as the
    medium route lays out its tucked strip. One hand: the one nearer the grasp point. Release 3 cm above the table.
    """
    from scipy.spatial import cKDTree
    x, uv = np.asarray(x, float), np.asarray(uv, float)
    size = float(layout.particle_size_m)
    half = np.array([layout.length_m, layout.width_m])/2
    h = x[:, 2]-table
    core = (np.abs(uv[:, 0]) < .6*half[0]) & (np.abs(uv[:, 1]) < .6*half[1]) & (h < 1.5*size)
    diagnosis = dict(core_particles=int(core.sum()))
    if core.sum() < 50:
        return None, dict(diagnosis, reason="no single-layer core")
    a, b = uv[core], x[core, :2]
    ca, cb = a.mean(0), b.mean(0)
    left, _, right = np.linalg.svd((a-ca).T @ (b-cb))
    rotation = (left @ right).T         # rest (u, v) -> world (x, y); determinant -1: the sheet lies mirrored
    flat = (uv-ca) @ rotation.T+cb
    deviation = np.linalg.norm(x[:, :2]-flat, axis=1)
    folded = deviation > fold["deviation_m"]
    diagnosis.update(fit_rms_m=float(np.sqrt(np.mean(deviation[core]**2))), folded_particles=int(folded.sum()),
                     mirrored=bool(np.linalg.det(rotation) < 0))
    best = None
    for corner in ((-half[0], -half[1]), (-half[0], half[1]), (half[0], -half[1]), (half[0], half[1])):
        near = np.linalg.norm(uv-np.asarray(corner), axis=1) < fold["corner_radius_m"]
        count = int((folded & near).sum())
        if best is None or count > best[1]:
            best = (np.asarray(corner, float), count, near)
    corner, count, near = best
    diagnosis.update(corner_uv=corner.tolist(), corner_folded_particles=count)
    if count < fold["min_particles"]:
        return None, dict(diagnosis, reason="no folded corner")
    flap, base = np.flatnonzero(folded & near), np.flatnonzero(~folded)
    base_tree = cKDTree(x[base, :2])
    distance, nearest = base_tree.query(x[flap, :2])
    overlap = distance < size
    votes = h[flap][overlap]-h[base[nearest[overlap]]]
    # A flap that barely overlaps the sheet is not tucked under it: its tip is exposed, so it is turned back like a page
    # (2026-10-10, MuJoCo: very_easy's corner overlapped by 5 particles at the crease, median +-0.1 mm; the vote sent it
    # into the under-mode drag, which crumpled it, coverage 0.93 -> 0.56).
    over = bool(votes.size < MIN_OVERLAP_VOTES or np.median(votes) > 0)
    tip = int(flap[np.argmin(np.linalg.norm(uv[flap]-corner, axis=1))])
    target = flat[tip]
    outward = (target-cb)/max(float(np.linalg.norm(target-cb)), 1e-9)
    diagnosis.update(over=over, overlap_votes=int(votes.size),
                     median_height_difference_m=float(np.median(votes)) if votes.size else None,
                     tip=tip, tip_xy=x[tip, :2].tolist(), tip_flat_xy=target.tolist())
    if over:
        grasp, p = tip, x[tip, :2]
        mid = (p+target)/2
        path = (("lift", 1.5, (*p, .054)), ("over", 3.5, (*mid, .09)), ("down", 3.5, (*(target+.01*outward), .03)))
    else:
        flap_distance, _ = cKDTree(x[flap, :2]).query(x[base, :2])
        border = base[flap_distance < 2*size]
        raised = border[h[border] > 1.2*size]
        crest_set = raised if len(raised) else border
        if not len(crest_set):
            return None, dict(diagnosis, reason="no crest next to the tucked flap")
        grasp = int(crest_set[np.argmin(np.linalg.norm(uv[crest_set]-corner, axis=1))])
        p = x[grasp, :2]
        reach = float(np.linalg.norm(target-p))
        drag = (target-p)/max(reach, 1e-9)
        path = (("lift", 1.5, (*p, .10)), ("drag", 6., (*(p+(reach+.03)*drag), .10)),
                ("lower", 3., (*(p+reach*drag), .03)))
    side = min(SIDES, key=lambda s: float(np.linalg.norm(np.asarray(tcp[s])[:2]-x[grasp, :2])))
    diagnosis.update(grasp=grasp, grasp_uv=uv[grasp].tolist(), side=side,
                     path=[dict(name=n, duration_s=d, to=[float(v) for v in g]) for n, d, g in path])
    stage = dict(grasp_uv={side: tuple(float(v) for v in uv[grasp])},
                 segments=tuple(dict(name=n, duration_s=d, to={side: tuple(float(v) for v in g)}) for n, d, g in path))
    return stage, diagnosis


def edge_stage(x, uv, table, layout, fold, tcp):
    """One two-hand stage that lays out the largest folded edge strip of the measured sheet; (stage or None, diagnosis).

    (2026-10-10, MuJoCo) A strip folded along a whole edge (medium: the far edge's 5 cm strip tucked under over its full
    length) is not a corner flap: fold_stage lays out one corner of it. The same flat reference as fold_stage (the rigid
    fit of the rest coordinates onto the single-layer core); folded = farther than `deviation_m` from it. The edge whose
    band (material distance < `edge_band_m`) holds the most folded particles is laid out if it holds at least
    `min_particles` per hand, with both hands at a quarter and three quarters of the strip's length: over (on top, or
    barely overlapping: exposed), each hand grasps the strip's free edge and turns it back like a page onto its flat
    position; under (tucked), each grasps the crest next to the strip and drags it outward past the flat position at
    10 cm, then lowers while moving back 3 cm, as fold_stage's under mode and the medium route lay out a tucked strip.
    Release 3 cm above the table.
    """
    from scipy.spatial import cKDTree
    x, uv = np.asarray(x, float), np.asarray(uv, float)
    size = float(layout.particle_size_m)
    half = np.array([layout.length_m, layout.width_m])/2
    h = x[:, 2]-table
    core = (np.abs(uv[:, 0]) < .6*half[0]) & (np.abs(uv[:, 1]) < .6*half[1]) & (h < 1.5*size)
    diagnosis = dict(kind="edge", core_particles=int(core.sum()))
    if core.sum() < 50:
        return None, dict(diagnosis, reason="no single-layer core")
    a, b = uv[core], x[core, :2]
    ca, cb = a.mean(0), b.mean(0)
    left, _, right = np.linalg.svd((a-ca).T @ (b-cb))
    rotation = (left @ right).T
    flat = (uv-ca) @ rotation.T+cb
    folded = np.linalg.norm(x[:, :2]-flat, axis=1) > fold["deviation_m"]
    band_m = float(fold.get("edge_band_m", .06))
    best = None
    for axis in (0, 1):
        for sign in (-1., 1.):
            band = np.abs(uv[:, axis]-sign*half[axis]) < band_m
            count = int((folded & band).sum())
            if best is None or count > best[2]:
                best = (axis, sign, count, band)
    axis, sign, count, band = best
    diagnosis.update(folded_particles=int(folded.sum()), edge_axis=axis, edge_sign=sign, edge_folded_particles=count)
    if count < 2*fold["min_particles"]:
        return None, dict(diagnosis, reason="no folded edge")
    strip, base = np.flatnonzero(folded & band), np.flatnonzero(~folded)
    along = 1-axis                                  # the material coordinate along the edge
    s_lo, s_hi = np.quantile(uv[strip, along], [.25, .75])
    distance, nearest = cKDTree(x[base, :2]).query(x[strip, :2])
    overlap = distance < size
    votes = h[strip][overlap]-h[base[nearest[overlap]]]
    over = bool(votes.size < MIN_OVERLAP_VOTES or np.median(votes) > 0)
    diagnosis.update(over=over, overlap_votes=int(votes.size),
                     median_height_difference_m=float(np.median(votes)) if votes.size else None,
                     strip_span=[float(s_lo), float(s_hi)])
    if not over:
        flap_distance, _ = cKDTree(x[strip, :2]).query(x[base, :2])
        border = base[flap_distance < 2*size]
        raised = border[h[border] > 1.2*size]
        crest_set = raised if len(raised) else border
        if len(crest_set) < 2:
            return None, dict(diagnosis, reason="no crest next to the tucked strip")
    grasps, paths = [], []
    for s_k in (s_lo, s_hi):
        edge_point = np.zeros(2)
        edge_point[along], edge_point[axis] = s_k, sign*half[axis]
        tip = int(strip[np.argmin(np.linalg.norm(uv[strip]-edge_point, axis=1))])
        target = flat[tip]
        outward = (target-cb)/max(float(np.linalg.norm(target-cb)), 1e-9)
        if over:
            grasp, p = tip, x[tip, :2]
            mid = (p+target)/2
            path = (("lift", 1.5, (*p, .054)), ("over", 3.5, (*mid, .09)), ("down", 3.5, (*(target+.01*outward), .03)))
        else:
            grasp = int(crest_set[np.argmin(np.linalg.norm(uv[crest_set]-edge_point, axis=1))])
            p = x[grasp, :2]
            reach = float(np.linalg.norm(target-p))
            drag = (target-p)/max(reach, 1e-9)
            path = (("lift", 1.5, (*p, .10)), ("drag", 6., (*(p+(reach+.03)*drag), .10)),
                    ("lower", 3., (*(p+reach*drag), .03)))
        grasps.append(grasp)
        paths.append(path)
    if grasps[0] == grasps[1]:
        return None, dict(diagnosis, reason="one grasp point for both hands")
    # the hands: the assignment with the shorter total TCP travel
    d = lambda side, g: float(np.linalg.norm(np.asarray(tcp[side])[:2]-x[g, :2]))
    order = (0, 1) if d("left", grasps[0])+d("right", grasps[1]) <= d("left", grasps[1])+d("right", grasps[0]) else (1, 0)
    hands = dict(left=order[0], right=order[1])
    diagnosis.update(grasp_uv={side: uv[grasps[k]].tolist() for side, k in hands.items()},
                     path={side: [dict(name=n, duration_s=t, to=[float(v) for v in g]) for n, t, g in paths[k]]
                           for side, k in hands.items()})
    stage = dict(grasp_uv={side: tuple(float(v) for v in uv[grasps[k]]) for side, k in hands.items()},
                 segments=tuple(dict(name=paths[0][i][0], duration_s=paths[0][i][1],
                                     to={side: tuple(float(v) for v in paths[k][i][2]) for side, k in hands.items()})
                                for i in range(len(paths[0]))))
    return stage, diagnosis


def crease_stage(x, uv, faces, table, layout, fold, tcp):
    """One two-hand stage that lays out the largest strip turned over along an edge, measured locally; (stage or None,
    diagnosis).

    (2026-10-10, MuJoCo) After hard's page turns the MuJoCo sheet keeps both long edges folded (a tri-fold, a third of
    the cloth turned over) and the flat fit of fold_stage / edge_stage misses by 4-5 cm everywhere: no rigid placement
    is known. This stage needs none. Every triangle's material orientation on the table (the sign of its uv -> xy map
    against the sheet's majority) marks the turned-over strips; the edge whose band (material distance < `edge_band_m`)
    holds the most turned-over particles is taken if it holds at least 2 x `min_particles`. Its crease is the material
    line where turned-over and face-up triangles meet; the strip's width is the material distance from the crease to the
    edge, and its flat direction the world image of the crease-to-edge material direction under the local uv -> xy map
    of the face-up sheet next to the crease. Both hands work at a quarter and three quarters of the crease (the stage's
    `span_quantile`, default .25): tucked (the strip under the sheet), each grasps the crease, lifts it 10 cm (`hang_m`),
    drags it outward past the flat position by 3 cm (`beyond_m`) so the strip hangs below, then lowers while moving back
    3 cm (`back_m`), as the medium route lays out its tucked strip; on top (or barely overlapping: exposed), each grasps
    the strip's free edge and turns it back like a page onto the flat position. Release 3 cm above the table.
    `lay="drop"`: a lowering that moves back while the strip still hangs free cannot lay it (on the limp MuJoCo flex the
    strip piled up into a roll); the hand first comes straight down until the strip's free edge is 1 cm above the table,
    then moves back by the strip's width while lowering, so the strip unrolls outward from its free edge.
    """
    from scipy.spatial import cKDTree
    x, uv, faces = np.asarray(x, float), np.asarray(uv, float), np.asarray(faces, int)
    size = float(layout.particle_size_m)
    half = np.array([layout.length_m, layout.width_m])/2
    h = x[:, 2]-table
    a, b, c = (x[faces[:, k], :2] for k in range(3))
    ua, ub, uc = (uv[faces[:, k]] for k in range(3))
    cross = lambda p, q: p[:, 0]*q[:, 1]-p[:, 1]*q[:, 0]
    world, material = cross(b-a, c-a), cross(ub-ua, uc-ua)
    sign = world*material
    reference = 1. if float(np.sum(sign)) >= 0 else -1.
    turned_tri = sign*reference < 0
    up_tri = ~turned_tri
    n = len(x)
    in_turned = np.zeros(n, bool)
    in_up = np.zeros(n, bool)
    in_turned[faces[turned_tri].ravel()] = True
    in_up[faces[up_tri].ravel()] = True
    band_m = float(fold.get("edge_band_m", .12))
    hang, beyond, back = (float(fold.get(key, .10 if key == "hang_m" else .03)) for key in ("hang_m", "beyond_m", "back_m"))
    diagnosis = dict(kind="crease", turned_triangles=float(turned_tri.mean()), majority_face_down=bool(reference < 0))
    best = None
    for axis in (0, 1):
        for edge_sign in (-1., 1.):
            if fold.get("edge") is not None and (axis, edge_sign) != tuple(fold["edge"]):
                continue                    # a stage that names its edge: (material axis, sign)
            band = np.abs(uv[:, axis]-edge_sign*half[axis]) < band_m
            count = int((in_turned & ~in_up & band).sum())
            if best is None or count > best[2]:
                best = (axis, edge_sign, count, band)
    axis, edge_sign, count, band = best
    diagnosis.update(edge_axis=axis, edge_sign=edge_sign, edge_turned_particles=count)
    if count < 2*fold["min_particles"]:
        return None, dict(diagnosis, reason="no turned-over edge strip")
    along = 1-axis
    strip = np.flatnonzero(in_turned & ~in_up & band)
    crease = np.flatnonzero(in_turned & in_up & (np.abs(uv[:, axis]-edge_sign*half[axis]) < band_m+.02))
    crease = crease[np.abs(edge_sign*half[axis]-uv[crease, axis]) >= .02]     # a fold line, not the edge's own rim
    if len(crease) < 10:
        return None, dict(diagnosis, reason="no crease")
    flat_up = np.flatnonzero(in_up & ~in_turned)
    up_tree = cKDTree(uv[flat_up])
    quantile = float(fold.get("span_quantile", .25))
    s_lo, s_hi = np.quantile(uv[crease, along], [quantile, 1-quantile])
    # over / under: the strip's height against the face-up sheet it overlaps
    distance, nearest = cKDTree(x[flat_up, :2]).query(x[strip, :2])
    overlap = distance < size
    votes = h[strip][overlap]-h[flat_up[nearest[overlap]]]
    over = bool(votes.size < MIN_OVERLAP_VOTES or np.median(votes) > 0)
    diagnosis.update(over=over, overlap_votes=int(votes.size),
                     median_height_difference_m=float(np.median(votes)) if votes.size else None,
                     crease_span=[float(s_lo), float(s_hi)])
    grasps, paths = [], []
    for s_k in (s_lo, s_hi):
        k = int(crease[np.argmin(np.abs(uv[crease, along]-s_k))])
        width = abs(edge_sign*half[axis]-uv[k, axis])
        # the local uv -> xy map of the face-up sheet within 6 cm (material) of the crease point
        near = flat_up[up_tree.query_ball_point(uv[k], .06)]
        if len(near) < 12:
            return None, dict(diagnosis, reason="no face-up sheet next to the crease")
        A = np.c_[uv[near]-uv[near].mean(0)]
        B = x[near, :2]-x[near, :2].mean(0)
        J = np.linalg.lstsq(A, B, rcond=None)[0].T          # 2x2: d(xy)/d(uv)
        e = np.zeros(2)
        e[axis] = edge_sign
        out = J @ e
        out = out/max(float(np.linalg.norm(out)), 1e-9)
        p = x[k, :2]
        if over:
            row = (np.abs(uv[strip, along]-uv[k, along]) < .02)
            if not row.any():
                return None, dict(diagnosis, reason="no strip edge next to the crease")
            tip = int(strip[row][np.argmax(np.abs(uv[strip[row], axis]-uv[k, axis]))])
            target = p+width*out
            mid = (x[tip, :2]+target)/2
            grasp = tip
            path = (("lift", 1.5, (*x[tip, :2], .054)), ("over", 3.5, (*mid, .09)), ("down", 3.5, (*(target+.01*out), .03)))
        elif fold.get("lay") == "drop":
            # the strip hangs free below the crease (held at least 3 cm above its own width), is carried out past its flat
            # place, comes down until its free edge is 1 cm above the table, then the hand lowers to 3 cm while moving back
            # by the strip's width less 2 cm, laying the strip down outward from its free edge as a carpet unrolls
            grasp = k
            top = max(hang, width+.03)
            far = p+(width+beyond)*out
            path = (("lift", 1.5, (*p, top)), ("drag", 6., (*far, top)), ("drop", 1.5, (*far, min(top, width+.01))),
                    ("lay", 3., (*(far-(width-.02)*out), .03)))
        else:
            grasp = k
            path = (("lift", 1.5, (*p, hang)), ("drag", 6., (*(p+(width+beyond)*out), hang)),
                    ("lower", 3., (*(p+(width+(beyond-back))*out), .03)))
        grasps.append(grasp)
        paths.append(path)
    if grasps[0] == grasps[1]:
        return None, dict(diagnosis, reason="one grasp point for both hands")
    d = lambda side, g: float(np.linalg.norm(np.asarray(tcp[side])[:2]-x[g, :2]))
    order = (0, 1) if d("left", grasps[0])+d("right", grasps[1]) <= d("left", grasps[1])+d("right", grasps[0]) else (1, 0)
    hands = dict(left=order[0], right=order[1])
    diagnosis.update(grasp_uv={side: uv[grasps[k]].tolist() for side, k in hands.items()},
                     path={side: [dict(name=nm, duration_s=t, to=[float(v) for v in g]) for nm, t, g in paths[k]]
                           for side, k in hands.items()})
    stage = dict(grasp_uv={side: tuple(float(v) for v in uv[grasps[k]]) for side, k in hands.items()},
                 segments=tuple(dict(name=paths[0][i][0], duration_s=paths[0][i][1],
                                     to={side: tuple(float(v) for v in paths[k][i][2]) for side, k in hands.items()})
                                for i in range(len(paths[0]))))
    return stage, diagnosis


def slide_stage(x, uv, table, layout, fold, tcp):
    """One two-hand stage that slides the sheet toward the robot when its far side lies beyond the arms' top-down reach;
    (stage or None, diagnosis).

    (2026-10-10, MuJoCo) The ALOHA reaches top-down only to y 0.36-0.44 over the middle of the table and to y -0.40 on
    the robot's side (work/reach_map.py). A strip left tucked along the far edge at y 0.45-0.5 (medium's route on the
    MuJoCo cloth) cannot be laid out where it lies. Measured: if the sheet's far extent exceeds `far_y_m`, both hands
    grasp the sheet `edge_depth_m` inside its near boundary (a complete patch, no strip ahead of the hand long enough to
    fold under) at a quarter and three quarters of its x extent, lift `lift_m` and slide it toward the robot by the
    excess (at most `max_slide_m`) at `speed_m_s`, then come down to 3 cm; the usual release and retreat follow."""
    x, uv = np.asarray(x, float), np.asarray(uv, float)
    far, near = float(x[:, 1].max()), float(x[:, 1].min())
    diagnosis = dict(kind="slide", far_y_m=far, near_y_m=near, target_far_y_m=float(fold["far_y_m"]))
    if far <= fold["far_y_m"]:
        return None, dict(diagnosis, reason="within reach")
    d = min(far-float(fold["far_y_m"]), float(fold.get("max_slide_m", .15)))
    lo, hi = float(x[:, 0].min()), float(x[:, 0].max())
    depth_lo, depth_hi = fold.get("edge_depth_m", (.02, .035))
    band = np.flatnonzero((x[:, 1]-near >= depth_lo) & (x[:, 1]-near <= depth_hi))
    width = float(np.ptp(x[band, 0])) if len(band) else 0.
    if width < .15:                 # the near boundary is a corner (a sheet lying diagonally), not an edge to pull
        return None, dict(diagnosis, reason="no near edge", near_band_width_m=width)
    grasps = [int(band[np.argmin(np.abs(x[band, 0]-(lo+q*(hi-lo))))]) for q in (.25, .75)]
    h, speed = float(fold.get("lift_m", .05)), float(fold.get("speed_m_s", .03))
    stage = dict(grasp_uv=dict(left=tuple(float(v) for v in uv[grasps[0]]), right=tuple(float(v) for v in uv[grasps[1]])),
                 segments=(dict(name="lift", duration_s=1., by=dict(left=(0., 0., h), right=(0., 0., h))),
                           dict(name="slide", duration_s=max(1.5, d/speed), by=dict(left=(0., -d, h), right=(0., -d, h))),
                           dict(name="lower", duration_s=1., by=dict(left=(0., -d, .03), right=(0., -d, .03)))))
    diagnosis.update(slide_m=d, grasp_xy={s: x[g, :2].tolist() for s, g in zip(SIDES, grasps)},
                     grasp_uv={s: uv[g].tolist() for s, g in zip(SIDES, grasps)})
    return stage, diagnosis


def base_clearance_stage(x, uv, state, task, cfg):
    """oraclefix 11b: one stage that slides the sheet away from an arm base it is too close to; (stage or None, diagnosis).

    The Task's final settling needs every robot collision cover >= clearance_m from the cloth's box padded by the particle
    radius (evaluation/terminal/settling.py). The arm bases are fixed at the table's x ends and their conservative covers
    reach x = 0.231 / -0.502 on lab, a window only ~5 cm wider per side than the sheet: a sheet left that far to one
    side can never pass, whatever the arms do (v17det blanket hard: the laid-out right corner put the sheet's box 1 cm into
    the right base's cover, every cloth gate true, 3600 settle steps without success). Measured on the current state:
    the covers that face the cloth box across x (overlapping it in y and z); if one is closer than clearance_m +
    margin_m, the hand on the far side grasps a complete patch 2.5-5 cm inside its own edge at the sheet's mid-width,
    lifts it lift_m and slides the sheet away by the shortfall + extra_m (bounded by the room on that side), then the
    usual release 3 cm above the table and retreat follow."""
    spec = task.final_settling_spec()
    pad, need = float(spec.particle_radius_m), float(spec.clearance_m)+float(cfg["margin_m"])
    x, uv = np.asarray(x, float), np.asarray(uv, float)
    lo, hi = x.min(0)-pad, x.max(0)+pad
    boxes = np.asarray((state.get("release_geometry") or {}).get("robot_bounds_m", []), float).reshape(-1, 2, 3)
    diagnosis = dict(cloth_box=[lo.tolist(), hi.tolist()], need_m=need, covers=len(boxes))
    facing = boxes[(boxes[:, 0, 1] < hi[1]+need) & (boxes[:, 1, 1] > lo[1]-need)
                   & (boxes[:, 0, 2] < hi[2]+need) & (boxes[:, 1, 2] > lo[2]-need)]
    middle = (lo[0]+hi[0])/2
    right = facing[(facing[:, 0, 0]+facing[:, 1, 0])/2 > middle]
    left = facing[(facing[:, 0, 0]+facing[:, 1, 0])/2 <= middle]
    gap_right = float(right[:, 0, 0].min()-hi[0]) if len(right) else float("inf")
    gap_left = float(lo[0]-left[:, 1, 0].max()) if len(left) else float("inf")
    diagnosis.update(gap_right_m=gap_right, gap_left_m=gap_left)
    if gap_right >= need and gap_left >= need:
        return None, dict(diagnosis, reason="clear")
    if gap_right < need and gap_left < need:
        return None, dict(diagnosis, reason="no room on either side")
    toward_left = gap_right < need
    shift = need-(gap_right if toward_left else gap_left)
    room = (gap_left if toward_left else gap_right)-need
    pull = min(shift+float(cfg["extra_m"]), room)
    diagnosis.update(shift_m=shift, room_m=room, pull_m=pull)
    if pull < shift:
        return None, dict(diagnosis, reason="not enough room on the far side")
    side = "left" if toward_left else "right"
    depth = (x[:, 0]-lo[0]-pad) if toward_left else (hi[0]-pad-x[:, 0])
    band = np.flatnonzero((depth >= cfg["edge_band_m"][0]) & (depth <= cfg["edge_band_m"][1]))
    if not len(band):
        return None, dict(diagnosis, reason="no grasp band")
    grasp = int(band[np.argmin(np.abs(x[band, 1]-(lo[1]+hi[1])/2))])
    h, d = float(cfg["lift_m"]), (-pull if toward_left else pull)
    stage = dict(grasp_uv={side: tuple(float(v) for v in uv[grasp])},
                 segments=(dict(name="lift", duration_s=1., by={side: (0., 0., h)}),
                           dict(name="slide", duration_s=max(1.5, pull/float(cfg["speed_m_s"])), by={side: (d, 0., h)})))
    diagnosis.update(side=side, grasp=grasp, grasp_xy=x[grasp, :2].tolist(), grasp_uv=uv[grasp].tolist())
    return stage, diagnosis

def run_sequence(env, runner, output, route, *, variation):
    """route["sequence"] (blanket hard port, 2026-10-17): parts run back to back, each a route of its own (stages, an
    optional planner and its options; a callable builds a fresh one) executed by run_route_oracle from the state the
    previous part left: the joint targets are re-read from the robot, the jaws start open, and each part ends with its
    own settle. A part inherits the sequence's max_strain_ratio. The sequence ends early once the episode terminates."""
    reports = []
    for k, part in enumerate(route["sequence"]):
        part = part() if callable(part) else dict(part)
        part.setdefault("max_strain_ratio", route["max_strain_ratio"])
        reports.append(run_route_oracle(env, runner, Path(output)/f"part_{k+1}", variation=variation, _part=part))
        if env._task.result.terminated:
            break
    report = dict(status="complete", variation_id=variation, parts=len(reports), sequence=reports,
                  phases=[phase for r in reports for phase in r["phases"]],
                  task_success=bool(env._task.result.task_success), task_result=to_jsonable(env._task.result))
    write_json_atomic(Path(output)/"oracle_report.json", to_jsonable(report))
    return report


def run_route_oracle(env, runner, output, *, variation="easy", _part=None):
    """_part (run_sequence only): one part of a route["sequence"], run from the current state."""
    output = Path(output)
    if _part is None and env._world.physics_step != 0:
        raise ValueError("The complete oracle must begin at the episode's default home state")
    if env.config.grasp_assist.model != "pbd_local_patch_v1" or env.config.control_mode != "JOINT_ABS":
        raise ValueError("The Blanket oracle uses pbd_local_patch_v1 and JOINT_ABS")
    route = route_for(variation) if _part is None else _part
    if _part is None and route.get("sequence"):
        return run_sequence(env, runner, output, route, variation=variation)
    output.mkdir(parents=True, exist_ok=True)
    task, world = env._task, env._world
    table, uv = task.table_height_m, task.material_uv
    dt = env.config.physics_dt_s*env.config.control_decimation
    arms = Arms(env)
    source = b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
    if env.config.grasp_assist.targets[0].max_strain_ratio != route["max_strain_ratio"]:
        raise ValueError("Assist strain guard must match the route's declared value")
    planner = route.get("planner")
    report = dict(status="running", start="default_home", variation_id=variation,
                  route={k: (v if k != "planner" else getattr(v, "name", repr(v))) for k, v in route.items()}, patch=PATCH,
                  assistance="pbd_local_patch_v1", control_mode="JOINT_ABS", config=to_jsonable(env.config),
                  instance_id=task.instance_id, phases=[], task_success=False, accepted_proof=False,
                  grasp_semantics="bilateral finger-surface proximity admission; native kinematic patch attachment; "
                                  "no measured clamping force")
    export_robot_geometry(env, output)
    # Inputs of the independent audit: rest material map, gates and frozen start identity.
    np.savez_compressed(output/"material_map.npz", uv=uv)
    declaration = task.frozen_initial_state()
    from .audit import archive_particles
    initial_particles = world.read_state()["particles"]["blanket"]
    if getattr(env.config, "physics_engine", "genesis") == "mujoco":
        # a MuJoCo cloth is born on its flex (world/mujoco/cloth.py): its particles lie on the coarse flex surface, off
        # the frozen PBD particles by the flex's documented birth deviation; the deviation is recorded, not refused
        off = np.linalg.norm(archive_particles(declaration.manifest_path, field="pos")
                             - np.asarray(initial_particles["position_m"]), axis=1)
        report["initial_particle_position_off_frozen_archive_m"] = dict(max=float(off.max()), mean=float(off.mean()))
        birth = getattr(world, "fem_birth", None) or {}
        if isinstance(birth, dict) and birth.get("redesign_start"):
            report["redesign_start"] = birth["redesign_start"]
    else:
        for saved_field, live_field in (("pos", "position_m"), ("vel", "velocity_m_s")):
            if not np.array_equal(archive_particles(declaration.manifest_path, field=saved_field),
                                  np.asarray(initial_particles[live_field])):
                raise ValueError(f"Oracle initial particle {saved_field} differs from its frozen archive")
        report["initial_particle_position_and_velocity_exact"] = True
    report.update(task_config={k: task.config[k] for k in ("minimum_coverage_ratio", "maximum_height_m",
                               "maximum_stretch_ratio", "maximum_speed_m_s", "stable_duration_s")},
                  layout=dict(length_m=task.layout.length_m, width_m=task.layout.width_m,
                              particle_size_m=task.layout.particle_size_m),
                  table_height_m=table, initial_state_manifest=str(declaration.manifest_path),
                  initial_state_sha256=declaration.sha256,
                  joint_limits_rad={s: arms.limits[s].tolist() for s in SIDES},
                  material_map_sha256=hashlib.sha256((output/"material_map.npz").read_bytes()).hexdigest())

    def cloth():
        return np.asarray(world.read_state()["particles"]["blanket"]["position_m"])

    def moves(segment, side):
        """The segment names a goal for this hand (`to` / `by`); a hand it does not name holds."""
        return any(side in (segment.get(key) or {}) for key in ("to", "by"))

    def run(name, policy, actions=None):
        if task.result.terminated:
            return dict(name=name, skipped="episode already terminated", held=dict.fromkeys(SIDES, False))
        print(f"{name}: {'feedback' if actions is None else len(actions)} control steps", flush=True)
        if actions is not None:
            write_json_atomic(output/f"{name}_actions.json", actions)
        digest = hashlib.sha256(source+name.encode()+json.dumps(actions, sort_keys=True).encode()).hexdigest()
        # Only feedback policies read world/grasp state; open-loop phases take none, keeping the per-tick
        # step records small. No camera is selected, so the runner records no frames or video.
        reads_world = actions is None or getattr(policy, "stop", None) is not None
        result = runner.run_policy(request_id=name, policy=policy, submission_hash=digest, mode="unlimited",
                                   observation_profile={"rgb_camera_ids": []},
                                   state_fields=("world",) if reads_world else ())
        grasp = world.read_state().get("grasp", {})
        phase = dict(name=name, result=result, held={s: bool(grasp.get("hands", {}).get(s, {}).get("binding"))
                                                      for s in SIDES}, health=grasp.get("health"))
        report["phases"].append(phase)
        write_json_atomic(output/"oracle_report.json", to_jsonable(report))
        if not result["evidence_complete"]:
            raise RuntimeError(f"{name}: incomplete execution evidence ({result.get('reason')})")
        return phase

    def hold(seconds, q, grip):
        return [arms.action(q, grip)]*int(np.ceil(seconds/dt))

    write_json_atomic(output/"oracle_report.json", to_jsonable(report))
    try:
        extension = {s: arms.fingertip_extension(s) for s in SIDES}
        jaw = {s: arms.jaw_axis(s) for s in SIDES}
        report.update(fingertip_extension_m=extension, stages=[])
        q = arms.qpos()
        # oraclefix 11: a route may append one measured fold-layout stage after its last stage (see fold_stage).
        stages, fold_layouts, nudges = list(route["stages"]), 0, 0
        grip = {s: OPEN_M for s in SIDES}
        # route["planner"] (the redesigned tiers, redesign_routes.py): planner(cloth, uv, table, target, task=task) ->
        # route-format stages planned on the measured cloth ([] = done). Called from the start when the route declares
        # no stages, and after each last stage once the sheet is quiet, while the task is not yet successful, at most
        # route["planner_rounds"] times. Its stages are planned live: an unreachable or unadmitted one is skipped.
        planner_calls = []
        # route["fast"] (the redesigned tiers, 2026-10-11 main): approach at the standard joint-step limit, descend at
        # FAST_TCP_M_S, 0.1 s holds, a 5-control contact dwell, a 0.3 s release, a 3 cm lift between the stages of one
        # planner call (8 cm after its last); the cloth's own moves (the route segments) and the quiet settles unchanged
        fast = bool(route.get("fast"))
        from topogym.planning.standard_rate import MARGIN, joint_step_limit
        step_limit = joint_step_limit()*MARGIN

        def plan(q, grip, settle_s):
            if planner is None or len(planner_calls) >= int(route.get("planner_rounds", 6)) or task.result.terminated:
                return
            if settle_s:
                actions = hold(settle_s, q, grip)
                quiet = Quiet(task.config["maximum_speed_m_s"], task.config["stable_duration_s"], dt)
                run(f"plan_settle_{len(planner_calls)+1}", Sequence(actions, stop=quiet), actions)
                if task.result.task_success:
                    return
            # the last planned stage's outcome, for a planner that takes it (`last`): admitted per hand, a grasp lost on
            # the way, or skipped (no top-down IK)
            kw = dict(task=task)
            if "last" in inspect.signature(planner).parameters:
                # not_admitted holds the assist's held flags of a stage it did not admit in full
                def outcome(done):
                    return dict(
                        name=stages[done["stage"]].get("name"), skipped=done.get("skipped"),
                        admitted={h: bool(done["not_admitted"][h]) if "not_admitted" in done else "skipped" not in done
                                  for h in done["grasp_uv"]},
                        grasp_lost=done.get("grasp_lost"))
                done = report["stages"][-1] if report["stages"] else None
                kw["last"] = None if done is None else outcome(done)
                if done is not None and planner_calls:
                    # every stage of the previous call (a call may plan several: redesign_routes.run_stages)
                    first = planner_calls[-1]["stage_index"]
                    kw["last"]["stages"] = [outcome(d) for d in report["stages"] if d["stage"] >= first]
            new = list(planner(cloth(), uv, table, getattr(task, "fold_target_particles", None), **kw))
            planner_calls.append(dict(call=len(planner_calls)+1, stage_index=len(stages),
                                      stages=[dict(name=st.get("name"), grasp_uv=st.get("grasp_uv"),
                                                   segments=len(st.get("segments", ()))) for st in new]))
            report["planner_calls"] = planner_calls
            write_json_atomic(output/"oracle_report.json", to_jsonable(report))
            stages.extend(new)
        def after(index, q, grip, log):
            # after a stage (or a measured stage skipped): the fold layouts and the base clearance, on the last stage
            nonlocal fold_layouts, nudges
            # oraclefix 11: after the route's last stage, let the sheet relax, then lay out a folded corner if one is
            # measured (one extra stage, or up to the route's `max_stages`: each lays out the then largest folded
            # corner). A fold left on the table keeps jittering under PBD self-contact.
            fold = route.get("fold_layout")
            if fold and fold_layouts < fold.get("max_stages", 1) and index == len(stages)-1 and not task.result.task_success:
                fold_layouts += 1
                # Adaptive: stop once every particle has stayed within the Task's speed limit for its window (cap settle_s)
                actions = hold(fold["settle_s"], q, grip)
                quiet = Quiet(task.config["maximum_speed_m_s"], task.config["stable_duration_s"], dt)
                run("fold_settle" if fold_layouts == 1 else f"fold_settle_{fold_layouts}", Sequence(actions, stop=quiet),
                    actions)
                log["fold_settle_ended_quiet"] = quiet.done
                if not task.result.task_success:
                    extra, diagnosis = fold_stage(cloth(), uv, table, task.layout, fold, arms.tcp())
                    if fold.get("creases"):
                        # a strip turned over along an edge on a sheet no rigid fit places (crease_stage)
                        c_extra, c_diagnosis = crease_stage(cloth(), uv, task.particle_faces, table, task.layout, fold,
                                                            arms.tcp())
                        if c_extra is not None:
                            extra, diagnosis = c_extra, dict(c_diagnosis, corner=diagnosis)
                    if fold.get("edges"):
                        # a strip folded along a whole edge: the two-hand edge layout when it holds more folded
                        # particles than the best corner (edge_stage)
                        e_extra, e_diagnosis = edge_stage(cloth(), uv, table, task.layout, fold, arms.tcp())
                        if e_extra is not None and (extra is None or e_diagnosis["edge_folded_particles"]
                                                    > diagnosis.get("corner_folded_particles", 0)):
                            extra, diagnosis = e_extra, dict(e_diagnosis, corner=diagnosis)
                    report["fold_layout"] = diagnosis
                    if "max_stages" in fold:
                        report.setdefault("fold_layouts", []).append(diagnosis)
                    write_json_atomic(output/"oracle_report.json", to_jsonable(report))
                    if extra is not None:
                        stages.append(extra)
            if planner is not None and index == len(stages)-1 and not task.result.task_success:
                plan(q, grip, max(float(stages[index].get("settle_s", 0.) or 0.), float(route.get("planner_settle_s", 6.))))
            # oraclefix 11b: once the sheet is quiet, slide it off an arm base it is too close to (base_clearance_stage)
            nudge = route.get("base_clearance")
            if (nudge and nudges < nudge["max_stages"] and index == len(stages)-1
                    and not task.result.task_success):
                nudges += 1
                actions = hold(nudge["settle_s"], q, grip)
                quiet = Quiet(task.config["maximum_speed_m_s"], task.config["stable_duration_s"], dt)
                run(f"clearance_settle_{nudges}", Sequence(actions, stop=quiet), actions)
                if not task.result.task_success:
                    extra, diagnosis = base_clearance_stage(cloth(), uv, world.read_state(), task, nudge)
                    report.setdefault("base_clearance", []).append(diagnosis)
                    write_json_atomic(output/"oracle_report.json", to_jsonable(report))
                    if extra is not None:
                        stages.append(extra)

        if not stages:
            plan(q, grip, 0.)
        for index, stage in enumerate(stages):
            if task.result.terminated:
                break
            measured = stage.get("measured")
            if measured:
                # a route stage planned on the live cloth when it starts (crease_stage / fold_stage with the stage's
                # own parameters), e.g. a tucked strip laid out before the route's own pulls; none found: skipped
                # (its own name: `planner` is the route's planner hook, which after() still calls)
                stage_planner = dict(crease=crease_stage, slide=slide_stage).get(measured, fold_stage)
                args = (cloth(), uv, task.particle_faces) if measured == "crease" else (cloth(), uv)
                planned, measured_diagnosis = stage_planner(*args, table, task.layout, stage, arms.tcp())
                report.setdefault("measured_stages", []).append(dict(measured_diagnosis, stage=index))
                write_json_atomic(output/"oracle_report.json", to_jsonable(report))
                if planned is None:
                    after(index, q, grip, {})
                    continue
                stage = stages[index] = planned
            active = tuple(s for s in SIDES if s in stage["grasp_uv"])
            # Stage 0 keeps the single-stage phase names; later stages are suffixed.
            name = (lambda base: base) if index == 0 else (lambda base, i=index: f"{base}_{i+1}")
            log = dict(stage=index, grasp_uv=stage["grasp_uv"])
            report["stages"].append(log)
            # Grasp sites: material points named by the route, located on the live cloth.
            x = cloth()
            centers = {s: int(np.argmin(np.sum((uv-np.asarray(stage["grasp_uv"][s]))**2, axis=1))) for s in active}
            # Top-down orientation per arm: the half-turn with the larger joint margin along the stage. A route with
            # `footing` (the redesigned tiers' planned stages, 2026-10-11) also tries the quarter turns and prefers the
            # yaws whose two fingertips stand on the same layer (footing_mismatch): a grasp 12 mm inside a top layer's
            # edge with the jaws across the edge puts one finger beyond it, and the assist never admits that pinch.
            choice = {}
            footing = route.get("footing")
            for side in active:
                best = None
                for yaw in ((0., 90., 180., 270.) if footing else (0., 180.)):
                    rotation = Rotation.from_euler("z", yaw, degrees=True)*DOWN
                    try:
                        probe, worst = q.copy(), np.inf
                        points = [x[centers[side]]+[0., 0., .10]]+[
                            segment_goal(seg, side, x[centers[side]], None, table)+[0., 0., .02]
                            for seg in stage["segments"] if moves(seg, side)]
                        if route.get("check_contact_ik"):
                            # Include the low grasp pose before choosing an
                            # orientation or moving away from home.
                            points.insert(1, x[centers[side]]+[0., 0., .005])
                        for point in points:
                            probe = arms.solve(side, point, probe, rotation)
                            worst = min(worst, arms.margin(side, probe))
                    except ValueError:
                        continue
                    if footing:
                        mismatch = footing_mismatch(x, x[centers[side]], yaw, jaw[side], **footing)
                        log.setdefault("footing_mismatch_m", {}).setdefault(side, {})[str(yaw)] = mismatch
                        key = (mismatch > footing["tolerance_m"], round(mismatch, 3) if mismatch > footing["tolerance_m"]
                               else 0., -worst)
                        if best is None or key < best[2]:
                            best = (yaw, worst, key)
                    elif best is None or worst > best[1]:
                        best = (yaw, worst)
                if best is None:
                    if planner is not None and index >= len(route["stages"]):   # a planned stage: skipped, replanned
                        log["skipped"] = f"no top-down IK for {side}"
                        break
                    if index >= len(route["stages"]):     # oraclefix 11: an unreachable fold layout is skipped
                        report.setdefault("fold_layout", {})["skipped"] = f"no top-down IK for {side}"
                        break
                    if measured:                           # an unreachable measured stage is skipped too
                        report["measured_stages"][-1]["skipped"] = f"no top-down IK for {side}"
                        break
                    raise RuntimeError(f"No top-down IK along stage {index} for {side}")
                choice[side] = best
                arms.rotation[side] = Rotation.from_euler("z", best[0], degrees=True)*DOWN
            if len(choice) < len(active):    # oraclefix 11: the fold layout was unreachable; go to the final settle
                if measured or (planner is not None and index >= len(route["stages"])):
                    # an unreachable measured / planned stage: on to the route's next stage (a planner replans)
                    after(index, q, grip, log)
                    continue
                break
            log.update(yaw_deg={s: c[0] for s, c in choice.items()},
                       planned_route_joint_margin_rad={s: c[1] for s, c in choice.items()})
            grip = {s: PRESHAPE_M if s in active else OPEN_M for s in SIDES}

            # 1. Current pose (home, or 8 cm above the last release) -> 10 cm above the sites,
            #    joint-space, jaws pre-shaped narrow.
            goal = q.copy()
            for side in active:
                goal = arms.solve(side, x[centers[side]]+[0., 0., .10], goal)
            if fast:
                # as fast as the standard joint-step limit allows: a smoothstep peaks at 1.5x its mean speed
                move = max(float(np.abs(goal-q)[arms.q_index[s]].max()) for s in SIDES)
                count = max(int(np.ceil(1.6*move/(.9*step_limit))), int(round(.6/dt)))
            else:
                count = int(round(3./dt))
            actions = [arms.action(q+smoothstep((i+1)/count)*(goal-q), grip) for i in range(count)]
            actions += hold(.1 if fast else .3, goal, grip)
            run(name("approach"), Sequence(actions), actions)
            # 2. Descend so fingertips stop above the live patch top: 1.5 mm, or the route's `fingertip_gap_m` where
            #    the arm's tracking error would press the fingers into a stack of layers.
            gap = route.get("fingertip_gap_m", FINGERTIP_GAP_M)
            x, q = cloth(), arms.qpos()
            commands = {}
            for side in active:
                patch = material_patch(x, uv, centers[side], **PATCH)
                top = float(x[patch, 2].max())+task.layout.particle_size_m/2
                commands[side] = np.array([*x[centers[side]][:2], top+gap+extension[side]])
            log["pinch_commands_m"] = {s: c.tolist() for s, c in commands.items()}
            tcp0 = arms.tcp()
            reach = max(float(np.linalg.norm(commands[s]-np.asarray(tcp0[s]))) for s in active)
            actions, q, _ = cartesian(arms, q, tcp0, commands, max(.5, reach/FAST_TCP_M_S) if fast else 1.5, dt, grip)
            actions += hold(.1 if fast else .3, q, grip)
            run(name("descend"), Sequence(actions), actions)
            # 3. Close the jaws in the air, then lower until both assists admit.
            grip = {s: PINCH_M if s in active else OPEN_M for s in SIDES}
            actions = hold(.3 if fast else .6, q, grip)
            run(name("close"), Sequence(actions, stop=lambda obs: all(held(obs)[s] for s in active)), actions)
            hover = {s: task.layout.particle_size_m/2+gap+extension[s] for s in active}
            contact = Contact(arms, commands, q, grip, hover=hover, centers=centers, uv=uv, patch=PATCH, sides=active,
                              dwell=5 if fast else 10,
                              hold_admitted=route.get("hold_admitted_height", False))
            phase = run(name("contact"), contact)
            log["contact_log"] = contact.log
            if task.result.terminated:      # the Task's goal held while the hands came down: nothing left to grasp
                break
            # a stage planned on the live cloth (measured, or a layout after the route) never ends the episode: a pinch
            # the assist does not admit, or a grasp lost on the way, opens the hands and the oracle goes on
            planned_live = bool(measured) or index >= len(route["stages"])
            if not all(phase["held"][s] for s in active):
                if not planned_live:
                    raise RuntimeError(f"Assist did not admit both hands: {phase['held']}")
                log["not_admitted"] = phase["held"]
            # 4. Route segments: the held patches follow the witness route. Commands continue from the
            #    admitted contact command, keeping the arm's steady tracking offset out of the route.
            x, q = cloth(), contact.q
            origin = {s: x[centers[s]].copy() for s in active}
            offset = {s: contact.commands[s]-origin[s] for s in active}
            log["command_minus_patch_center_m"] = {s: o.tolist() for s, o in offset.items()}
            starts = dict(contact.commands)
            for segment in (() if "not_admitted" in log else stage["segments"]):
                goals = {s: (segment_goal(segment, s, origin[s], None, table)+offset[s]
                             if moves(segment, s) else starts[s]) for s in active}
                if fast and planned_live:
                    # a lab hand is a point: a planned target may lie inside the stack under the hand, which the table
                    # guard's cloth floor (finger links 6 mm into the cloth) would stop. Lift it onto the stack instead.
                    x_now = cloth()
                    for s in active:
                        if not moves(segment, s):
                            continue
                        lift = stack_lift(x_now, uv, centers[s], goals[s]-offset[s], task.layout.particle_size_m)
                        if lift > 0.:
                            goals[s] = goals[s]+[0., 0., lift]
                            log.setdefault("stack_lifts", []).append(dict(segment=segment["name"], side=s,
                                                                          lift_mm=round(lift*1000, 1)))
                try:
                    actions, q, worst = cartesian(arms, q, starts, goals, segment["duration_s"], dt, grip)
                except ValueError as error:
                    # a planned stage whose segment leaves the arm's reach: the hands release where they are and the
                    # planner is told (grasp_lost), as for a grasp lost on the way
                    if not planned_live:
                        raise
                    log["grasp_lost"] = dict(segment=segment["name"], unreachable=str(error))
                    q = arms.qpos()
                    starts = {s: np.asarray(arms.tcp()[s], float) for s in active}
                    break
                log.setdefault("route_joint_margin_rad", {})[segment["name"]] = worst
                phase = run(name(segment["name"]), Sequence(actions, stop=lambda obs: not all(held(obs)[s] for s in active)),
                            actions)
                if not all(phase["held"][s] for s in active) and not task.result.task_success:
                    if not planned_live:
                        raise RuntimeError(f"{segment['name']}: assisted grasp lost {phase['held']} {phase['health']}")
                    log["grasp_lost"] = dict(segment=segment["name"], held=phase["held"])
                    q = arms.qpos()
                    starts = {s: np.asarray(arms.tcp()[s], float) for s in active}
                    break
                starts = goals
            # 5. Open and lift the hands 8 cm. fast: the active hands open only to the pre-shape width, enough for the
            #    assist to let go (max_finger_opening_fraction), so the jaws do not spread through the stack they pinched
            #    (a full opening stretched the cloth 2-3.5x at release on the GPU step's fresh collider kinematics)
            grip = {s: PRESHAPE_M if fast and s in active else OPEN_M for s in SIDES}
            actions = hold(.3 if fast else .5, q, grip)
            run(name("release"), Sequence(actions), actions)
            # fast: between the stages of one planner call the next stage's approach starts 3 cm up; after a call's last
            # stage the hands clear 8 cm for the quiet settle
            within_call = fast and index < len(stages)-1
            up = {s: starts[s]+[0., 0., .03 if within_call else .08] for s in active}
            actions, q, _ = cartesian(arms, q, starts, up, (.3 if within_call else .6) if fast else 1.5, dt, grip)
            run(name("retreat"), Sequence(actions), actions)
            after(index, q, grip, log)
        # 6. Wait for the Task's own settled predicate.
        if not task.result.task_success:
            actions = hold(route.get("settle_s", 10.), q, grip)
            run("settle", Sequence(actions), actions)
        report.update(status="complete", task_success=bool(task.result.task_success),
                      task_result=to_jsonable(task.result), sim_time_s=world.sim_time_s,
                      physics_steps=world.physics_step)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      task_result=to_jsonable(task.result))
        raise
    finally:
        write_json_atomic(output/"oracle_report.json", to_jsonable(report))
    return report
