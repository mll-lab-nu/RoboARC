"""rope_threading redesign: five new tiers made by inverse generation (scene geometry only; no physics here).

The checks (`check_design`): the goal centreline clears every fixture by the rope radius + FIXTURE_MARGIN_M, its ordered
signed aperture crossings by the task's own predicate are exactly the required sequence, the start rope crosses no
aperture and clears every fixture.

Coordinates: the task frame of every rope_threading instance (origin = the robot root on the table surface; x towards
the right arm, z up). `layout_dict(world=True)` writes world x, y and z above the table (ThreadingLayout's convention,
before the entity's table offset).

Only the scene is declared here; the success predicate stays RopeThreadingTask.evaluate_state (ordered signed crossings,
thickness clear, unambiguous, quiet). Nothing adds an episode-ending rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

ROPE_RADIUS_M = 0.004
TUBE_RADIUS_M = 0.006            # the ring tube (as the released rings)
LINK_M = 0.0055                  # rope_chain_v1 link length
FEED_STATION_M = 0.14            # feeder grasp: this far behind the tip, on the ring axis
POKE_M = 0.085                   # the tip passes the ring plane by this much before the receiver pinches it
RECEIVE_STATION_M = 0.020        # receiver pinch: this far from the tip (beyond the ring)
FEED_APPROACH_M = 0.06           # the tip starts this far before the ring plane (v1: 6 cm; v0 8 cm)
DRAW_STOP_M = 0.16               # the previous receiver draws the tip to this far before the next ring (its hand keeps
HAND_CLEAR_M = 0.03              # a hand stays this far from a ring plane (finger half width + tube + margin)
LEAD_IN_M = FEED_STATION_M + FEED_APPROACH_M + 0.01   # straight run before a ring (the feeder's push line)
LEAD_OUT_M = POKE_M + 0.01                             # straight run after a ring (the receiver's pinch)
FIXTURE_MARGIN_M = 0.010         # goal centreline clearance to fixtures beyond the rope radius
MIN_BEND_RADIUS_M = 0.035        # goal centreline (the chain's bend stop allows 5 mm; a stiff rope wants more)
TASK_ROOT_WORLD = np.array([-0.1356786, 0.0, 0.0])
TABLE_Z_WORLD = 0.9196429
ROPE_Z_M = ROPE_RADIUS_M + 0.0005

RING_RGBA = [(0.20, 0.70, 0.80, 1.0), (0.35, 0.80, 0.40, 1.0), (0.75, 0.50, 0.85, 1.0), (0.95, 0.75, 0.20, 1.0),
             (0.90, 0.35, 0.35, 1.0)]
WALL_RGBA, POST_RGBA, BAR_RGBA = (0.62, 0.64, 0.68, 1.0), (0.45, 0.48, 0.50, 1.0), (0.55, 0.42, 0.30, 1.0)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _perp_basis(n):
    n = _unit(n)
    u = np.cross([0.0, 0.0, 1.0], n)
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0.0, 0.0])
    u = _unit(u - (u @ n) * n)
    return u, np.cross(n, u)


# ----------------------------------------------------------------------------------------------- fixtures
@dataclass
class Ring:
    name: str
    center: np.ndarray
    travel: np.ndarray               # the tip's travel direction through the ring (the goal tangent)
    axis_sign: int                   # stored normal = axis_sign * travel (the task's sign is then axis_sign)
    inner_radius: float
    mount: str                       # "bulkhead" (held in a wall window) or "post" (a stand from the table)
    mount_dir: tuple = (0.0, 0.0, -1.0)   # direction from the centre to the stand's rim attachment
    rgba: tuple = RING_RGBA[0]
    role: str = ""
    lead_in: float = LEAD_IN_M
    lead_out: float = LEAD_OUT_M
    s_goal: float = 0.0
    lay_gap: float | None = None     # the relay lays the tip this far before the ring (None: relay_plan.LAY_GAP_M)

    @property
    def normal(self):
        return _unit(self.travel) * self.axis_sign

    @property
    def sign(self):
        return int(self.axis_sign)

    @property
    def radius(self):                # tube centreline radius (ThreadingLayout's radius_m)
        return self.inner_radius + TUBE_RADIUS_M

    @property
    def outer_radius(self):
        return self.inner_radius + 2 * TUBE_RADIUS_M

    def basis(self):
        n = self.normal
        d = np.asarray(self.mount_dir, float)
        d = d - (d @ n) * n
        if np.linalg.norm(d) < 1e-6:
            u, v = _perp_basis(n)
        else:
            v = _unit(d)
            u = np.cross(v, n)
        return _unit(u), _unit(v), n

    def stand(self):
        if self.mount != "post":
            return None
        u, v, n = self.basis()
        top = self.center + (self.radius + TUBE_RADIUS_M) * v
        return dict(a=np.array([top[0], top[1], 0.008]), b=top, radius=0.005,
                    foot=dict(center=np.array([top[0], top[1], 0.004]), half=np.array([0.025, 0.025, 0.004])))

    def tube_points(self, n=96):
        u, v, _ = self.basis()
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return self.center + self.radius * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)

    def distance(self, p):
        p = np.atleast_2d(p)
        d = p - self.center
        a = d @ self.normal
        rad = np.sqrt(np.maximum((d * d).sum(1) - a * a, 0))
        return np.sqrt(a * a + (rad - self.radius) ** 2) - TUBE_RADIUS_M

    def as_layout(self, world=False):
        u, v, n = self.basis()
        off = TASK_ROOT_WORLD if world else 0
        out = dict(name=self.name, center_m=(self.center + off).tolist(), normal=n.tolist(), basis_u=u.tolist(),
                   basis_v=v.tolist(), radius_m=self.radius, tube_radius_m=TUBE_RADIUS_M,
                   inner_radius_m=self.inner_radius, segments=64, rgba=list(self.rgba), shape="closed",
                   required_sign=int(self.sign), travel=_unit(self.travel).tolist(), mount=self.mount,
                   goal_arc_length_m=float(self.s_goal), role=self.role)
        st = self.stand()
        if st is not None:
            out["stand"] = dict(fromto=np.r_[st["a"] + off, st["b"] + off].tolist(), radius=st["radius"])
            out["base"] = dict(center=(st["foot"]["center"] + off).tolist(), half=st["foot"]["half"].tolist())
        return out


@dataclass
class Box:
    name: str
    center: np.ndarray
    half: np.ndarray
    yaw: float = 0.0
    rgba: tuple = WALL_RGBA
    role: str = "obstacle"
    roll: float = 0.0                # rotation about x after the yaw (the bulkhead's corner fillers)

    def R(self):
        return Rotation.from_euler("zx", [self.yaw, self.roll]).as_matrix()

    def distance(self, p):
        p = np.atleast_2d(p)
        local = (p - self.center) @ self.R()
        q = np.abs(local) - self.half
        out = np.linalg.norm(np.maximum(q, 0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0)
        return out + inside

    def as_layout(self, world=False):
        return dict(kind="box", name=self.name, center_m=(self.center + (TASK_ROOT_WORLD if world else 0)).tolist(),
                    half_extents_m=self.half.tolist(), yaw_rad=self.yaw, roll_rad=self.roll,
                    quat_wxyz=Rotation.from_matrix(self.R()).as_quat()[[3, 0, 1, 2]].tolist(), rgba=list(self.rgba),
                    role=self.role)


@dataclass
class Capsule:
    name: str
    a: np.ndarray
    b: np.ndarray
    radius: float
    rgba: tuple = POST_RGBA
    role: str = "obstacle"

    def distance(self, p):
        p = np.atleast_2d(p)
        ab = self.b - self.a
        t = np.clip(((p - self.a) @ ab) / (ab @ ab), 0, 1)
        return np.linalg.norm(p - (self.a + t[:, None] * ab), axis=-1) - self.radius

    def as_layout(self, world=False):
        off = TASK_ROOT_WORLD if world else 0
        return dict(kind="capsule", name=self.name, fromto_m=np.r_[self.a + off, self.b + off].tolist(),
                    radius_m=self.radius, rgba=list(self.rgba), role=self.role)


def bulkhead(x_w, y_lo, y_hi, height, windows, thickness=0.012, name="bulkhead"):
    boxes, hx = [], thickness / 2
    windows = sorted(windows, key=lambda w: w[0])
    edges = [y_lo]
    for (y, z, h) in windows:
        edges += [y - h, y + h]
    edges.append(y_hi)
    for k, i in enumerate(range(0, len(edges), 2)):
        a, b = edges[i], edges[i + 1]
        if b - a > 1e-4:
            boxes.append(Box(f"{name}_strip{k}", np.array([x_w, (a + b) / 2, height / 2]),
                             np.array([hx, (b - a) / 2, height / 2]), role="bulkhead"))
    for j, (y, z, h) in enumerate(windows):
        r_rim = h - 0.003 + 0.001
        depth = (h * math.sqrt(2) - r_rim) / 2
        for k, (sy, sz) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            c_r = r_rim + depth
            cy, cz = y + sy * c_r / math.sqrt(2), z + sz * c_r / math.sqrt(2)
            roll = math.atan2(sz, sy)                        # the box's local y along the corner diagonal
            boxes.append(Box(f"{name}_corner{j}{k}", np.array([x_w, cy, cz]), np.array([hx, depth, depth * 2.2]),
                             roll=roll, role="bulkhead"))
        lo, hi = z - h, z + h
        if lo > 1e-4:
            boxes.append(Box(f"{name}_sill{j}", np.array([x_w, y, lo / 2]), np.array([hx, h, lo / 2]), role="bulkhead"))
        if height - hi > 1e-4:
            boxes.append(Box(f"{name}_lintel{j}", np.array([x_w, y, (hi + height) / 2]),
                             np.array([hx, h, (height - hi) / 2]), role="bulkhead"))
    return boxes


# ----------------------------------------------------------------------------------------------- the goal curve
def hermite(p0, t0, p1, t1, scale=1.0, step=0.002):
    chord = float(np.linalg.norm(p1 - p0))
    m0, m1 = _unit(t0) * chord * scale, _unit(t1) * chord * scale
    n = max(8, int(math.ceil(chord * 1.6 / step)))
    u = np.linspace(0, 1, n + 1)[1:, None]
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def curvature_radius(points):
    p = np.asarray(points)
    a, b, c = p[:-2], p[1:-1], p[2:]
    ab, bc, ca = np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = ab * bc * ca / np.maximum(2 * area2, 1e-15)
    return np.where(area2 < 1e-12, np.inf, R)


def resample(points, step):
    p = np.asarray(points)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    n = max(2, int(round(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, p[:, i]) for i in range(3)])


def best_connector(p0, t0, p1, t1, obstacles=(), clearance=0.0):
    best = None
    for scale in (0.6, 0.8, 1.0, 1.2, 1.5, 1.8):
        pts = hermite(p0, t0, p1, t1, scale)
        full = np.vstack([p0, pts])
        R = float(np.min(curvature_radius(resample(full, 0.004)))) if len(full) > 4 else np.inf
        gap = min([float(np.min(o.distance(full))) for o in obstacles] or [np.inf])
        ok = gap >= clearance
        key = (ok, R if ok else gap, -float(np.sum(np.linalg.norm(np.diff(full, axis=0), axis=1))))
        if best is None or key > best[0]:
            best = (key, pts, R, gap, scale)
    return best[1], dict(min_radius=best[2], gap=best[3], scale=best[4])


def goal_curve(tail, tail_dir, rings, tip_overhang, obstacles=(), clearance=0.0):
    pts = [np.asarray(tail, float)]
    t_prev = _unit(tail_dir)
    reports = []
    for r in rings:
        d = _unit(r.travel)
        start = r.center - d * r.lead_in
        con, rep = best_connector(pts[-1], t_prev, start, d, obstacles, clearance)
        reports.append(dict(to=r.name, **rep))
        pts.extend(con)
        n = max(2, int(math.ceil((r.lead_in + r.lead_out) / 0.002)))
        run = start + np.linspace(0, r.lead_in + r.lead_out, n + 1)[1:, None] * d
        pts.extend(run)
        t_prev = d
    last = rings[-1]
    extra = tip_overhang - last.lead_out
    if extra > 0:
        n = max(2, int(math.ceil(extra / 0.002)))
        pts.extend(pts[-1] + np.linspace(0, extra, n + 1)[1:, None] * _unit(last.travel))
    P = resample(np.asarray(pts), 0.002)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    for r in rings:
        r.s_goal = float(s[int(np.argmin(np.linalg.norm(P - r.center, axis=1)))])
    return P, reports


class Turtle:

    def __init__(self, start, heading, step=0.002):
        self.p, self.t, self.step = np.asarray(start, float), _unit(heading), step
        self.pts, self.marks = [self.p.copy()], []

    @property
    def s(self):
        return float(np.sum(np.linalg.norm(np.diff(np.asarray(self.pts), axis=0), axis=1))) if len(self.pts) > 1 else 0.0

    def line(self, length):
        if length <= 1e-9:
            return self
        n = max(1, int(math.ceil(length / self.step)))
        for i in range(1, n + 1):
            self.pts.append(self.p + self.t * length * i / n)
        self.p = self.p + self.t * length
        return self

    def line_to(self, axis, value):
        k = self.t[axis]
        if abs(k) < 1e-6 or (value - self.p[axis]) / k < 0:
            raise ValueError(f"line_to {axis}={value}: heading {self.t} never gets there from {self.p}")
        return self.line((value - self.p[axis]) / k)

    def bend(self, radius, degrees, axis):
        k = np.asarray(axis, float)
        k = _unit(k - (k @ self.t) * self.t)
        a = math.radians(degrees)
        centre = self.p + radius * math.copysign(1.0, a) * np.cross(k, self.t)
        n = max(2, int(math.ceil(abs(a) * radius / self.step)))
        p0, t0 = self.p.copy(), self.t.copy()
        for i in range(1, n + 1):
            R = Rotation.from_rotvec(k * a * i / n).as_matrix()
            self.pts.append(centre + R @ (p0 - centre))
        R = Rotation.from_rotvec(k * a).as_matrix()
        self.p, self.t = centre + R @ (p0 - centre), _unit(R @ t0)
        return self

    def turn(self, radius, degrees):
        return self.bend(radius, degrees, [0.0, 0.0, 1.0])

    def pitch(self, radius, degrees):
        side = np.cross([0.0, 0.0, 1.0], self.t)
        if np.linalg.norm(side) < 1e-6:
            raise ValueError("pitch from a vertical heading: use bend with an explicit axis")
        return self.bend(radius, degrees, side)

    def ring(self, name):
        self.marks.append(dict(name=name, s=self.s, point=self.p.copy(), tangent=self.t.copy()))
        return self

    def curve(self):
        return resample(np.asarray(self.pts), 0.002)


def rings_from_turtle(turtle, specs):
    out = []
    for i, m in enumerate(turtle.marks):
        sp = specs[m["name"]]
        travel = _unit(m["tangent"])
        axis_sign = int(sp.get("axis_sign", 1))
        out.append(Ring(m["name"], m["point"].copy(), travel, axis_sign, sp["inner_radius"], sp["mount"],
                        tuple(sp.get("mount_dir", (0, 0, -1))), sp.get("rgba", RING_RGBA[i % len(RING_RGBA)]),
                        role=sp.get("role", ""), s_goal=m["s"]))
    return out


# ----------------------------------------------------------------------------------------------- the design
@dataclass
class TierDesign:
    tier: str
    title: str
    rings: list
    obstacles: list
    goal: np.ndarray
    start: np.ndarray
    legs: list
    notes: dict = field(default_factory=dict)

    @property
    def rope_length(self):
        return float(np.sum(np.linalg.norm(np.diff(self.goal, axis=0), axis=1)))

    def required_crossings(self):
        return [(r.name, r.sign) for r in sorted(self.rings, key=lambda r: r.s_goal)]

    def fixtures(self):
        out = list(self.obstacles)
        for r in self.rings:
            st = r.stand()
            if st is not None:
                out.append(Capsule(f"{r.name}_stand", st["a"], st["b"], st["radius"], role="stand"))
                out.append(Box(f"{r.name}_base", st["foot"]["center"], st["foot"]["half"], role="stand"))
        return out

    def layout_dict(self, world=False):
        off = TASK_ROOT_WORLD if world else 0
        return dict(
            schema="rope_threading_redesign_layout/1", tier=self.tier, title=self.title,
            frame=("world x, y; z above the table (ThreadingLayout convention)" if world else
                   "task (robot root on the table surface; world = task + (-0.1356786, 0, table_z))"),
            rope=dict(radius_m=ROPE_RADIUS_M, length_m=self.rope_length, link_m=LINK_M,
                      links=int(round(self.rope_length / LINK_M)),
                      material=dict(E=5e6, nu=0.3, rho=650.0, friction_mu=0.3), reference_shape="initial",
                      start_centerline_m=(self.start + off).tolist()),
            rings=[r.as_layout(world) for r in sorted(self.rings, key=lambda r: r.s_goal)],
            obstacles=[o.as_layout(world) for o in self.obstacles],
            required_crossings=[f"{n}{'+' if s > 0 else '-'}" for n, s in self.required_crossings()],
            goal_centerline_m=(self.goal[::2] + off).tolist(), goal_length_m=self.rope_length,
            legs=self.legs, notes=self.notes)


def start_line(tip, direction, length):
    d = _unit([direction[0], direction[1], 0.0])
    tip = np.array([tip[0], tip[1], ROPE_Z_M])
    n = int(round(length / 0.002))
    return tip[None, :] + np.linspace(-length, 0.0, n + 1)[:, None] * d[None, :]


def start_arc(tip, tip_heading, radius, length, left=True):
    h = _unit([tip_heading[0], tip_heading[1], 0.0])
    side = np.array([-h[1], h[0], 0.0]) * (1 if left else -1)
    tip = np.array([tip[0], tip[1], ROPE_Z_M])
    centre = tip + radius * side
    n = int(round(length / 0.002))
    phi = np.linspace(-length / radius, 0.0, n + 1) * (1 if left else -1)
    r0 = tip - centre
    return np.asarray([centre + Rotation.from_rotvec([0, 0, f]).as_matrix() @ r0 for f in phi])


def _windows(rings, margin=0.003):
    return [(r.center[1], r.center[2], r.outer_radius + margin) for r in rings if r.mount == "bulkhead"]


def _with_length(start_fn, length):
    return start_fn(length)


# ----------------------------------------------------------------------------------------------- tiers
BULKHEAD = dict(x_w=0.0, y_lo=-0.62, y_hi=0.46, height=0.32)
FEED = dict(lead_in=FEED_STATION_M + FEED_APPROACH_M + 0.01, lead_out=POKE_M + 0.01)   # straight runs per ring


def start_zigzag(L, tip=(0.05, -0.25), runs=2, gap=0.14, x_far=0.56):
    r = gap / 2
    run = (L - (runs - 1) * np.pi * r) / runs
    tip = np.array([tip[0], tip[1], ROPE_Z_M])
    pts = [tip.copy()]                       # built tip -> tail, reversed at the end
    p, d = tip.copy(), np.array([1.0, 0.0, 0.0])
    for k in range(runs):
        n = int(math.ceil(run / 0.002))
        pts += [p + d * run * (i / n) for i in range(1, n + 1)]
        p = p + d * run
        if k < runs - 1:
            c = p + np.array([0.0, -r, 0.0])
            m = int(math.ceil(np.pi * r / 0.002))
            sgn = 1 if d[0] > 0 else -1
            for i in range(1, m + 1):
                a = np.pi * i / m
                pts.append(c + r * np.array([sgn * np.sin(a), np.cos(a), 0.0]))
            p = c + np.array([0.0, -r, 0.0])
            d = -d
    pts = np.asarray(pts)[::-1]
    return pts


def chord_threadable(a, b, max_angle_deg=60.0):
    d = _unit(b.center - a.center)
    ang = [math.degrees(math.acos(np.clip(d @ _unit(r.travel), -1, 1))) for r in (a, b)]
    return dict(angles_deg=[round(x, 1) for x in ang], threadable=bool(max(ang) < max_angle_deg))


def _post(name, x, y, top=0.30, r=0.010):
    return Capsule(name, np.array([x, y, 0.0]), np.array([x, y, top]), r, role="post")


def _chord_post(name, a, b, frac=0.5, top=0.30, r=0.010, shift=(0.0, 0.0), goal=None, clear=0.03, keepout=None):
    fracs = [frac] if goal is None else list(np.linspace(0.25, 0.75, 21))
    best = None
    for f in fracs:
        p = a.center + f * (b.center - a.center)
        gap = np.inf if goal is None else float(np.min(np.linalg.norm(goal[:, :2] - p[:2], axis=1))) - r - ROPE_RADIUS_M
        if keepout is not None and len(keepout):
            gap = min(gap, float(np.min(np.linalg.norm(keepout[:, :2] - p[:2], axis=1))) - 0.055)
        if best is None or gap > best[0]:
            best = (gap, p)
    if goal is not None and best[0] < clear:
        raise ValueError(f"{name}: no chord station clears the goal by {clear} m (best {best[0]:.3f})")
    p = best[1]
    return _post(name, p[0] + shift[0], p[1] + shift[1], top, r)


def _shelf(x0, x1, y0, y1, z, name="shelf"):
    return [Box(name, np.array([(x0 + x1) / 2, (y0 + y1) / 2, z + 0.005]), np.array([(x1 - x0) / 2, (y1 - y0) / 2, 0.005]),
                rgba=BAR_RGBA, role="shelf"),
            Capsule(f"{name}_leg", np.array([x1 - 0.012, y0 + 0.012, 0.0]), np.array([x1 - 0.012, y0 + 0.012, z]),
                    0.010, rgba=BAR_RGBA, role="shelf")]


def _make(tier, title, t, specs, legs, notes, start_fn, chords=(), shelf=None):
    rings = rings_from_turtle(t, specs)
    goal = t.curve()
    L = float(np.sum(np.linalg.norm(np.diff(goal, axis=0), axis=1)))
    obstacles = bulkhead(BULKHEAD["x_w"], BULKHEAD["y_lo"], BULKHEAD["y_hi"], BULKHEAD["height"], _windows(rings))
    k = 0
    for (i, j) in chords:
        if not chord_threadable(rings[i], rings[j])["threadable"]:
            continue                       # a straight push cannot pass both apertures: no post needed
        k += 1
        top = max(r.center[2] + r.outer_radius for r in (rings[i], rings[j])) + 0.03
        obstacles.append(_chord_post(f"P{k}", rings[i], rings[j], goal=goal, top=top))
    if shelf is not None:
        idx, dxm, dxp, dym, dyp, z = shelf
        c = rings[idx].center
        obstacles += _shelf(c[0] - dxm, c[0] + dxp, c[1] - dym, c[1] + dyp, z)
    start = start_fn(L)
    return TierDesign(tier, title, rings, obstacles, goal, start, legs, dict(notes))


def _tail(t, z_to, run=0.03, radius=0.05, angle=60):
    t.line(run)
    t.pitch(radius, -angle)
    rise = z_to - t.p[2] - radius * (1 - math.cos(math.radians(angle)))
    t.line(max(0.0, rise / math.sin(math.radians(angle))))
    t.pitch(radius, angle)
    return t


def _climb(t, dz, radius=0.04, angle=35):
    if abs(dz) < 1e-6:
        return t
    sgn = 1 if dz > 0 else -1
    t.pitch(radius, -sgn * angle)
    rest = abs(dz) - 2 * radius * (1 - math.cos(math.radians(angle)))
    t.line(max(0.0, rest / math.sin(math.radians(angle))))
    t.pitch(radius, sgn * angle)
    return t


def _down_eyelet(t, top_z, side_axis):
    t.pitch(0.045, -90)                                         # heading up
    t.line(max(0.0, top_z - t.p[2] - 0.045))
    t.bend(0.045, 180, side_axis)                              # over the top, heading down
    return t


def tier_very_easy():
    li, lo = FEED["lead_in"], FEED["lead_out"]
    t = Turtle([0.33, -0.22, ROPE_Z_M], [-1, 0, 0])
    _tail(t, 0.15)
    t.line_to(0, 0.0).ring("G1").line(lo + 0.06)
    t.turn(0.06, -90)                                   # heading +y
    t.line(li).ring("A").line(lo)
    t.turn(0.06, -90)                                   # heading +x
    t.line_to(0, 0.0).ring("G2").line(lo)
    t.turn(0.06, -90)                                   # heading -y (towards the front; +y would leave B beyond the
    t.line(li).ring("B").line(0.12)                     # right arm's top-down reach)
    specs = {"G1": dict(axis_sign=-1, inner_radius=0.050, mount="bulkhead", role="grommet"),
             "A": dict(axis_sign=1, inner_radius=0.050, mount="post", mount_dir=(-1, 0, -1),
                       role="side ring (left half)"),
             "G2": dict(axis_sign=1, inner_radius=0.050, mount="bulkhead", role="grommet"),
             "B": dict(axis_sign=-1, inner_radius=0.050, mount="post", mount_dir=(1, 0, -1),
                       role="side ring (right half)")}
    legs = []      # the hand plan is authoring data (topogym/planning/rope_threading/redesign/legs.py)
    return _make("very_easy", "Bulkhead lacing, x and y rings", t, specs, legs,
                 dict(slack="tail before G1 about 0.4 m (generous)"),
                 lambda L: start_zigzag(L, tip=(0.10, -0.30), runs=3, gap=0.12), chords=[(0, 1), (1, 2), (2, 3)])


def tier_easier():
    li, lo = FEED["lead_in"], FEED["lead_out"]
    t = Turtle([0.31, -0.24, ROPE_Z_M], [-1, 0, 0])
    _tail(t, 0.15)
    t.line_to(0, 0.0).ring("G1").line(lo)
    _lace_left(t, li, lo, rise=0.07)
    t.line_to(0, 0.0).ring("G2").line(lo)
    t.turn(0.06, 90)                                    # heading +y
    t.line(li).ring("B").line(0.12)
    specs = {"G1": dict(axis_sign=-1, inner_radius=0.045, mount="bulkhead", role="grommet"),
             "V": dict(axis_sign=-1, inner_radius=0.045, mount="post", mount_dir=(0, -1, 0),
                       role="vertical eyelet (left half)"),
             "G2": dict(axis_sign=1, inner_radius=0.045, mount="bulkhead", role="grommet"),
             "B": dict(axis_sign=1, inner_radius=0.045, mount="post", mount_dir=(1, 0, -1),
                       role="side ring (right half)")}
    legs = []      # the hand plan is authoring data (topogym/planning/rope_threading/redesign/legs.py)
    return _make("easier", "Bulkhead lacing with a downward eyelet", t, specs, legs,
                 dict(slack="tail before G1 about 0.35 m"),
                 lambda L: start_zigzag(L, tip=(0.10, -0.30), runs=4, gap=0.11), chords=[(0, 1), (1, 2), (2, 3)])


V_HEIGHT = 0.20          # the vertical eyelet's centre: the receiving hand must fit under it (pinch 6.5 cm below)


def _lace_left(t, li, lo, top=None, rise=0.07):
    t.pitch(0.045, -90)                                         # heading up
    t.line(max(0.0, (top if top is not None else V_HEIGHT + li + 0.045) - t.p[2] - 0.045))
    t.bend(0.045, 180, [0, -1, 0])                             # over the top towards -x, heading down
    t.line(li).ring("V").line(lo)
    t.bend(0.04, 90, [1, 0, 0])                                # down -> +y
    _climb(t, rise - (t.p[2] - 0.065))                         # rise = the height gained above 0.065 m
    t.turn(0.06, -90)                                          # +y -> +x
    return t


def tier_easy():
    li, lo = FEED["lead_in"], FEED["lead_out"]
    t = Turtle([0.30, -0.24, ROPE_Z_M], [-1, 0, 0])
    _tail(t, 0.15)
    t.line_to(0, 0.0).ring("G1").line(lo)
    _lace_left(t, li, lo)
    t.line_to(0, 0.0).ring("G2").line(lo)
    _climb(t, -0.015)
    t.turn(0.06, 90)                                    # +x -> +y
    t.line(li).ring("B").line(0.11)
    specs = {"G1": dict(axis_sign=-1, inner_radius=0.045, mount="bulkhead", role="grommet"),
             "V": dict(axis_sign=-1, inner_radius=0.040, mount="post", mount_dir=(0, -1, 0),
                       role="vertical eyelet (left half)"),
             "G2": dict(axis_sign=1, inner_radius=0.045, mount="bulkhead", role="grommet"),
             "B": dict(axis_sign=1, inner_radius=0.040, mount="post", mount_dir=(1, 0, -1),
                       role="ring facing away (right half, under the shelf)")}
    legs = []      # the hand plan is authoring data (topogym/planning/rope_threading/redesign/legs.py)

    return _make("easy", "Bulkhead lacing under a shelf", t, specs, legs, dict(slack="tail before G1 about 0.3 m"),
                 lambda L: start_zigzag(L, tip=(0.10, -0.30), runs=4, gap=0.11), chords=[(0, 1), (1, 2), (2, 3)],
                 shelf=(3, 0.08, 0.08, 0.07, 0.10, 0.29))


def tier_medium():
    li, lo = FEED["lead_in"], FEED["lead_out"]
    t = Turtle([0.27, -0.25, ROPE_Z_M], [-1, 0, 0])
    _tail(t, 0.15)
    t.line_to(0, 0.0).ring("G1").line(lo)
    _lace_left(t, li, lo)
    t.line_to(0, 0.0).ring("G2").line(lo)
    _climb(t, 0.10)
    t.turn(0.06, 70)                                    # +x -> mostly +y
    t.pitch(0.06, 25)                                   # nose down 25 deg
    t.line(li).ring("T").line(0.10)
    specs = {"G1": dict(axis_sign=-1, inner_radius=0.040, mount="bulkhead", role="grommet"),
             "V": dict(axis_sign=-1, inner_radius=0.040, mount="post", mount_dir=(0, -1, 0),
                       role="vertical eyelet (left half)"),
             "G2": dict(axis_sign=1, inner_radius=0.040, mount="bulkhead", role="grommet"),
             "T": dict(axis_sign=1, inner_radius=0.040, mount="post", mount_dir=(1, 0, -1),
                       role="tilted ring facing away (right half, under the shelf)")}
    legs = []      # the hand plan is authoring data (topogym/planning/rope_threading/redesign/legs.py)

    return _make("medium", "Bulkhead lacing, tilted ring under a shelf", t, specs, legs,
                 dict(slack="tail before G1 about 0.25 m"),
                 lambda L: start_zigzag(L, tip=(0.10, -0.30), runs=4, gap=0.11), chords=[(0, 1), (1, 2), (2, 3)],
                 shelf=(3, 0.08, 0.08, 0.07, 0.10, 0.29))


def tier_hard():
    li, lo = FEED["lead_in"], FEED["lead_out"]
    t = Turtle([0.24, -0.27, ROPE_Z_M], [-1, 0, 0])
    _tail(t, 0.15)
    t.line_to(0, 0.0).ring("G1").line(lo)
    _lace_left(t, li, lo)
    t.line_to(0, 0.0).ring("G2").line(lo)
    _climb(t, 0.10)
    t.turn(0.06, 70)
    t.pitch(0.06, 25)
    t.line(li).ring("T").line(lo)
    t.pitch(0.05, -25)                                  # level again
    t.turn(0.07, 110)                                   # -> heading -x
    _climb(t, 0.0)
    t.line_to(0, 0.0).ring("G3").line(0.10)
    specs = {"G1": dict(axis_sign=-1, inner_radius=0.040, mount="bulkhead", role="grommet"),
             "V": dict(axis_sign=-1, inner_radius=0.035, mount="post", mount_dir=(0, -1, 0),
                       role="vertical eyelet (left half)"),
             "G2": dict(axis_sign=1, inner_radius=0.040, mount="bulkhead", role="grommet"),
             "T": dict(axis_sign=1, inner_radius=0.035, mount="post", mount_dir=(1, 0, -1),
                       role="tilted ring facing away (right half, under the shelf)"),
             "G3": dict(axis_sign=-1, inner_radius=0.040, mount="bulkhead", role="grommet")}
    legs = []      # the hand plan is authoring data (topogym/planning/rope_threading/redesign/legs.py)

    return _make("hard", "Bulkhead lacing, five passages", t, specs, legs,
                 dict(slack="tail before G1 about 0.2 m; tip overhang 0.10 m"),
                 lambda L: start_zigzag(L, tip=(0.10, -0.26), runs=5, gap=0.095),
                 chords=[(0, 1), (1, 2), (2, 3), (3, 4)], shelf=(3, 0.07, 0.07, 0.07, 0.05, 0.27))


TIERS_V0 = {"very_easy": tier_very_easy, "easier": tier_easier, "easy": tier_easy, "medium": tier_medium,
            "hard": tier_hard}


# ----------------------------------------------------------------------------------------------- geometry v1
RING_LOW_Z = 0.070               # centre height of a pedestal ring (its aperture bottom 2-3.5 cm over the table)
GROMMET_Z = 0.150                # centre height of a grommet in the bulkhead


def _pedestal(ring, width=0.030):
    u, v, n = ring.basis()
    bottom = ring.center + (ring.radius + TUBE_RADIUS_M) * v          # v points down for mount_dir (0, 0, -1)
    top = float(bottom[2]) + 0.002
    yaw = math.atan2(u[1], u[0])
    return Box(f"{ring.name}_pedestal", np.array([bottom[0], bottom[1], top / 2]),
               np.array([width / 2, TUBE_RADIUS_M, top / 2]), yaw=yaw, rgba=(0.45, 0.48, 0.50, 1.0), role="pedestal")


def _ring(name, center, travel, sign, inner, mount, rgba_i, role="", mount_dir=(0.0, 0.0, -1.0)):
    tr = _unit(travel)
    return Ring(name, np.asarray(center, float), tr, int(sign), float(inner), mount, tuple(mount_dir),
                RING_RGBA[rgba_i % len(RING_RGBA)], role=role)


def goal_curve_v1(tail_start, tail_end, rings, tip_overhang, obstacles=(), clearance=0.0):
    a, b = np.asarray(tail_start, float), np.asarray(tail_end, float)
    n = max(2, int(math.ceil(np.linalg.norm(b - a) / 0.002)))
    flat = a + np.linspace(0, 1, n + 1)[:-1, None] * (b - a)
    P, reports = goal_curve(b, b - a, rings, tip_overhang, obstacles, clearance)
    P = resample(np.vstack([flat, P]), 0.002)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    for r in rings:
        r.s_goal = float(s[int(np.argmin(np.linalg.norm(P - r.center, axis=1)))])
    return P, reports


def _lay_keepout(rings, gap=0.12, back=0.16):
    pts = []
    for r in rings:
        d = _unit(r.travel)
        h = np.array([d[0], d[1], 0.0])
        if np.linalg.norm(h) < 0.45:
            continue
        h = h / np.linalg.norm(h)
        for f in np.linspace(gap, gap + back, 9):
            pts.append(r.center - f * h)
    return np.asarray(pts)


def _make_v1(tier, title, rings, legs, notes, tail, tip_overhang, start_fn, shelf=None, chord_posts=True,
             keep_lay_lines=False, skip_chords=()):
    obstacles = bulkhead(BULKHEAD["x_w"], BULKHEAD["y_lo"], BULKHEAD["y_hi"], BULKHEAD["height"], _windows(rings))
    obstacles += [_pedestal(r) for r in rings if r.mount == "pedestal"]
    goal, reports = goal_curve_v1(tail[0], tail[1], rings, tip_overhang, obstacles, clearance=0.02)
    k = 0
    if chord_posts:
        for a, b in zip(rings[:-1], rings[1:]):
            if not chord_threadable(a, b)["threadable"]:
                continue
            if (a.name, b.name) in skip_chords:
                notes = dict(notes, **{f"no_post_{a.name}_{b.name}": "left open"})
                continue
            top = max(r.center[2] + r.outer_radius for r in (a, b)) + 0.03
            try:
                post = _chord_post(f"P{k + 1}", a, b, goal=goal, top=top,
                                   keepout=_lay_keepout(rings) if keep_lay_lines else None)
            except ValueError:
                notes = dict(notes, **{f"no_post_{a.name}_{b.name}": "no post station clears the goal line by its margin"})
                continue
            k += 1
            obstacles.append(post)
    if shelf is not None:
        x0, x1, y0, y1, z = shelf
        obstacles += _shelf(x0, x1, y0, y1, z)
    L = float(np.sum(np.linalg.norm(np.diff(goal, axis=0), axis=1)))
    d = TierDesign(tier, title, list(rings), obstacles, goal, start_fn(L), legs,
                   dict(notes, connectors=reports, geometry="v1"))
    return d


def _base_rings(inner, g2_y=0.28, g1_y=-0.24):
    G1 = _ring("G1", (0.0, g1_y, GROMMET_Z), (-1, 0, 0), -1, inner["G1"], "bulkhead", 0,
               "grommet (front window)")
    A = _ring("A", (-0.26, 0.03, RING_LOW_Z), (0, 1, 0), 1, inner["A"], "pedestal", 1,
              "low ring (left half), facing +y")
    G2 = _ring("G2", (0.0, g2_y, GROMMET_Z), (1, 0, 0), 1, inner["G2"], "bulkhead", 2,
               "grommet (back window)")
    return G1, A, G2


def _start_cw():
    def f(L):
        P = start_spiral(L)
        cy = -0.40
        P = P.copy()
        P[:, 1] = 2 * cy - P[:, 1]
        return P
    return f


def _start(tip=(0.10, -0.30), runs=4, gap=0.10):
    return lambda L: start_spiral(L)


def start_spiral(L, center=(0.40, -0.40), r_out=0.27, pitch=0.035, tip_angle=math.pi):
    c = np.array([center[0], center[1]])
    dphi = 0.002 / r_out
    pts, phi, total = [], tip_angle, 0.0
    r = r_out
    prev = c + r * np.array([math.cos(phi), math.sin(phi)])
    pts.append(prev)
    while total < L:
        phi -= dphi
        r = r_out - pitch * (tip_angle - phi) / (2 * math.pi)
        if r < 0.10:
            raise ValueError(f"start_spiral: {L:.2f} m does not fit (inner radius below 0.10 m)")
        q = c + r * np.array([math.cos(phi), math.sin(phi)])
        total += float(np.linalg.norm(q - prev))
        pts.append(q)
        prev = q
    pts = np.asarray(pts[::-1])
    out = np.c_[pts, np.full(len(pts), ROPE_Z_M)]
    return resample(out, 0.002)


TAIL = ((0.40, -0.24, ROPE_Z_M), (0.36, -0.24, ROPE_Z_M))     # the tail: flat on the table, then the rise to G1


def v1_very_easy():
    G1, A, G2 = _base_rings(dict(G1=0.050, A=0.050, G2=0.050))
    B = _ring("B", (0.26, 0.01, RING_LOW_Z), (0, -1, 0), -1, 0.050, "pedestal", 3,
              "low ring (right half), facing the front, last")
    rings = [G1, A, G2, B]
    legs = []      # the hand plan is authoring data (topogym/planning/rope_threading/redesign/legs.py)
    return _make_v1("very_easy", "Bulkhead lacing v1: two grommets, two low rings", rings, legs,
                    dict(slack="tail: 4 cm flat + the rise to G1 (~0.47 m before G1's plane)"),
                    tail=TAIL, tip_overhang=0.12, start_fn=_start())


def v1_easier():
    G1, A, G2 = _base_rings(dict(G1=0.045, A=0.045, G2=0.045))
    V = _ring("V", (0.22, 0.0, 0.17), (0, 0, -1), 1, 0.045, "post", 3,
              "vertical eyelet (right half), last", mount_dir=(0, -1, 0))
    rings = [G1, A, G2, V]
    legs = []      # the hand plan is authoring data (topogym/planning/rope_threading/redesign/legs.py)
    return _make_v1("easier", "Bulkhead lacing v1: a downward eyelet last", rings, legs,
                    dict(slack="tail as very_easy"), tail=TAIL, tip_overhang=0.12, start_fn=_start())


def v1_easy():
    G1, A, G2 = _base_rings(dict(G1=0.045, A=0.045, G2=0.045), g2_y=0.34)
    B = _ring("B", (0.30, 0.12, RING_LOW_Z), (0, -1, 0), -1, 0.040, "pedestal", 3,
              "low ring (right half), facing the front")
    G3 = _ring("G3", (0.0, -0.07, GROMMET_Z), (-1, 0, 0), -1, 0.040, "bulkhead", 4,
               "grommet (middle window), last")
    rings = [G1, A, G2, B, G3]
    legs = []      # the hand plan is authoring data (topogym/planning/rope_threading/redesign/legs.py)
    return _make_v1("easy", "Bulkhead lacing v1: three wall passages", rings, legs,
                    dict(slack="tail as very_easy"), tail=TAIL, tip_overhang=0.10,
                    start_fn=_start(tip=(0.10, -0.28), runs=5, gap=0.08),
                    skip_chords={("B", "G3")})


def _yawed(deg):
    a = math.radians(deg)
    return (math.sin(a), -math.cos(a), 0.0)


def _tilted(deg_down):
    a = math.radians(deg_down)
    return (0.0, -math.cos(a), -math.sin(a))


T_VARIANT = {}      # fan-out variants (10-11 afternoon): {"yaw": deg, "ccw": bool} override medium / hard's T and start


def _t_yaw(default):
    return T_VARIANT.get("yaw", default)


def _t_start():
    return _start() if T_VARIANT.get("ccw") else _start_cw()


def v1_medium():
    G1, A, G2 = _base_rings(dict(G1=0.040, A=0.040, G2=0.040), g2_y=0.34, g1_y=-0.28)
    A = _ring("A", (-0.22, 0.03, RING_LOW_Z), (0, 1, 0), 1, 0.040, "pedestal", 1, A.role)
    T = _ring("T", (0.28, 0.12, RING_LOW_Z), _yawed(_t_yaw(10)), -1, 0.040, "pedestal", 3,
              "low ring (right half) turned 10 deg off the front, facing neither arm", mount_dir=(1, 0, 0))
    G3 = _ring("G3", (0.0, -0.12, GROMMET_Z), (-1, 0, 0), -1, 0.040, "bulkhead", 4,
               "grommet (middle window), last")
    rings = [G1, A, G2, T, G3]
    legs = []      # the hand plan is authoring data (topogym/planning/rope_threading/redesign/legs.py)
    return _make_v1("medium", "Bulkhead lacing v1: a ring turned off both arms", rings, legs,
                    dict(slack="tail: the rise to G1 only"), tail=((0.37, -0.28, ROPE_Z_M), (0.36, -0.28, ROPE_Z_M)),
                    tip_overhang=0.10, start_fn=_t_start(),
                    shelf=None, skip_chords={("G1", "A")})   # plan2: a shelf at 0.25 m blocked the right arm everywhere near T (upper arm / forearm)


def v1_medium_alt():
    G1, A, G2 = _base_rings(dict(G1=0.040, A=0.040, G2=0.040), g2_y=0.34)
    T = _ring("T", (0.28, 0.12, RING_LOW_Z), _yawed(25), -1, 0.040, "pedestal", 3,
              "low ring (right half) yawed 25 deg off the front, facing neither arm", mount_dir=(1, 0, 0))
    G3 = _ring("G3", (0.0, -0.12, GROMMET_Z), (-1, 0, 0), -1, 0.040, "bulkhead", 4,
               "grommet (middle window), last")
    rings = [G1, A, G2, T, G3]
    legs = []      # the hand plan is authoring data (topogym/planning/rope_threading/redesign/legs.py)
    return _make_v1("medium_alt", "Bulkhead lacing v1: a ring turned off both arms (alt)", rings, legs,
                    dict(slack="tail: the rise to G1 only"), tail=((0.37, -0.24, ROPE_Z_M), (0.36, -0.24, ROPE_Z_M)),
                    tip_overhang=0.10, start_fn=_start_cw(), shelf=None)


def v1_hard():
    G1, A, G2 = _base_rings(dict(G1=0.040, A=0.035, G2=0.040), g2_y=0.34, g1_y=-0.28)
    A = _ring("A", (-0.22, 0.03, RING_LOW_Z), (0, 1, 0), 1, 0.035, "pedestal", 1, A.role)
    T = _ring("T", (0.28, 0.12, RING_LOW_Z), _yawed(_t_yaw(10)), -1, 0.035, "pedestal", 3,
              "low ring (right half) turned 10 deg off the front, facing neither arm", mount_dir=(1, 0, 0))
    G3 = _ring("G3", (0.0, -0.12, GROMMET_Z), (-1, 0, 0), -1, 0.040, "bulkhead", 4,
               "grommet (middle window)")
    C = _ring("C", (-0.32, -0.12, GROMMET_Z), (-1, 0, 0), -1, 0.035, "post", 0,
              "ring on a post behind G3 (left half), last", mount_dir=(0, 0, -1))
    rings = [G1, A, G2, T, G3, C]
    legs = []      # the hand plan is authoring data (topogym/planning/rope_threading/redesign/legs.py)
    return _make_v1("hard", "Bulkhead lacing v1: six rings, three wall passages", rings, legs,
                    dict(slack="tail: the rise to G1 only"), tail=((0.37, -0.28, ROPE_Z_M), (0.36, -0.28, ROPE_Z_M)),
                    tip_overhang=0.10, start_fn=_t_start(),
                    shelf=None, keep_lay_lines=True)   # plan2: a shelf at 0.25 m blocked the right arm everywhere near T (upper arm / forearm)


TIERS_V1 = {"very_easy": v1_very_easy, "easier": v1_easier, "easy": v1_easy, "medium": v1_medium, "hard": v1_hard}
def _variant(fn, **kw):
    def f():
        old = dict(T_VARIANT)
        T_VARIANT.clear()
        T_VARIANT.update(kw)
        try:
            d = fn()
        finally:
            T_VARIANT.clear()
            T_VARIANT.update(old)
        return d
    return f


TIERS_ALT = {"medium_alt": v1_medium_alt,
             "medium_t0": _variant(v1_medium, yaw=0.0), "medium_t0ccw": _variant(v1_medium, yaw=0.0, ccw=True),
             "medium_tm10": _variant(v1_medium, yaw=-10.0), "medium_tp10": _variant(v1_medium, yaw=10.0),
             "hard_t0": _variant(v1_hard, yaw=0.0), "hard_t0ccw": _variant(v1_hard, yaw=0.0, ccw=True),
             "hard_tm10": _variant(v1_hard, yaw=-10.0), "hard_tp10": _variant(v1_hard, yaw=10.0)}
TIERS = dict(TIERS_V0)
TIERS.update(TIERS_V1)
TIERS.update(TIERS_ALT)


def build(tier):
    return TIERS[tier]()


# ----------------------------------------------------------------------------------------------- checks
def usable_radius(ring, allowance=0.001, margin=0.0001):
    return ring.radius - (TUBE_RADIUS_M + ROPE_RADIUS_M - allowance) - margin


def crossings(points, rings, allowance=0.001):
    from topogym.utils.geometry.passage import aperture_crossings
    hits, ambiguous = [], False
    for r in rings:
        h, amb = aperture_crossings(points, r.center, r.normal, usable_radius(r, allowance))
        ambiguous |= amb
        hits += [dict(x, ring=r.name) for x in h]
    hits.sort(key=lambda h: h["material_coordinate"])
    return [(h["ring"], h["sign"]) for h in hits], ambiguous, hits


def check_design(d: TierDesign):
    rep = {}
    seq, amb, hits = crossings(d.goal, d.rings)
    rep["goal_sequence"] = [f"{n}{'+' if s > 0 else '-'}" for n, s in seq]
    rep["goal_sequence_ok"] = (seq == d.required_crossings()) and not amb
    gaps = {}
    for f in d.fixtures():
        g = float(np.min(f.distance(d.goal))) - ROPE_RADIUS_M
        gaps[f.name] = g
    rep["goal_min_fixture_gap_m"] = min(gaps.values()) if gaps else None
    rep["goal_fixture_gaps_m"] = {k: round(v, 4) for k, v in sorted(gaps.items(), key=lambda kv: kv[1])[:6]}
    ring_gaps = {r.name: float(np.min(r.distance(d.goal))) - ROPE_RADIUS_M for r in d.rings}
    rep["goal_ring_tube_gaps_m"] = {k: round(v, 4) for k, v in ring_gaps.items()}
    R = curvature_radius(resample(d.goal, 0.004))
    rep["goal_min_bend_radius_m"] = float(np.min(R))
    rep["goal_min_height_m"] = float(np.min(d.goal[:, 2]))
    sseq, samb, _ = crossings(d.start, d.rings)
    rep["start_crossings"] = len(sseq)
    sg = {f.name: float(np.min(f.distance(d.start))) - ROPE_RADIUS_M for f in d.fixtures()}
    sg.update({r.name: float(np.min(r.distance(d.start))) - ROPE_RADIUS_M for r in d.rings})
    rep["start_min_fixture_gap_m"] = min(sg.values())
    rep["start_closest"] = min(sg, key=sg.get)
    rep["rope_length_m"] = d.rope_length
    srt = sorted(d.rings, key=lambda r: r.s_goal)
    rep["route_first_to_last_ring_m"] = srt[-1].s_goal - srt[0].s_goal
    rep["tail_before_first_ring_m"] = srt[0].s_goal
    rep["tip_after_last_ring_m"] = d.rope_length - srt[-1].s_goal
    chords = []
    fixtures = [f for f in d.fixtures()]
    for a, b in zip(srt[:-1], srt[1:]):
        seg = a.center + np.linspace(0, 1, 200)[:, None] * (b.center - a.center)
        inner = seg[10:-10]
        blocked = [f.name for f in fixtures if float(np.min(f.distance(inner))) < ROPE_RADIUS_M]
        other_rings = [r.name for r in d.rings if r.name not in (a.name, b.name)
                       and float(np.min(r.distance(inner))) < ROPE_RADIUS_M]
        th = chord_threadable(a, b)
        chords.append(dict(chord=f"{a.name}->{b.name}", blocked_by=blocked + other_rings, **th,
                           shortcut_open=bool(th["threadable"] and not (blocked + other_rings))))
    rep["chords"] = chords
    return rep
