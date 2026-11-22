"""The five redesign tiers of buzz_wire_3d (scene generators). Each factory returns a Design; `TIERS` maps the tier
name to (variation_id, factory). Geometry rationale per tier: docs/rcb/redesign/buzz_wire_3d.md."""
from __future__ import annotations

import numpy as np

from . import motifs as M
from .design import Design

Z0 = 0.153            # first / last control height (toy z; as the released layouts: 10.5 mm above the deck top)
POST = 0.06           # vertical start / finish post height above Z0


def _post_up(x, y, h=POST):
    return M.line([x, y, Z0], [x, y, Z0 + h])


def _post_down(x, y, h=POST):
    return M.line([x, y, Z0 + h], [x, y, Z0])


ROBOT_XY = np.array([-0.57, 0.0])     # the Panda's base in world xy (every buzz scene)


class Local:

    def __init__(self, center_xy):
        d = -(ROBOT_XY - np.asarray(center_xy, float))
        self.phi = float(np.arctan2(d[1], d[0]))
        c, s = np.cos(self.phi), np.sin(self.phi)
        self.R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])

    def to_toy(self, pts):
        return np.asarray(pts, float) @ self.R.T

    def vec(self, v):
        return self.R @ np.asarray(v, float)


def _arc_range(poly, sel):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    idx = np.flatnonzero(sel)
    return float(s[idx[0]]), float(s[idx[-1]])


def _sleeve_at(poly, i, half=0.022):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    return (float(s[i] - half), float(s[i] + half))


def robot_dir(center_xy, at_xy=(0.0, 0.0)):
    r = ROBOT_XY - np.asarray(center_xy, float) - np.asarray(at_xy, float)
    return np.r_[r / np.linalg.norm(r), 0.0]


def vertical_loop(c, axis, r, pitch, turns=1.0):
    w = np.asarray(axis, float) / np.linalg.norm(axis)
    u = np.cross([0, 0, 1.], w)
    return M.helix(np.asarray(c, float) - 0.5 * pitch * turns * w, w, u, r, pitch, turns, a0=-np.pi / 2)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def post_out(start_xy, side, rise=0.07, run=0.03, radius=0.03):
    t = M.Turtle([*start_xy, Z0], [0, 0, 1])
    t.straight(rise).turn_to([*side, 0.0], radius).straight(run)
    return t.polyline(), t.p.copy(), t.h.copy()


def join(a, ta, b, tb, scale=None):
    if scale is not None:
        return M.hermite([a, b], [ta, tb], scale)[1:-1]
    best = None
    for sc in (0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0):
        c = M.hermite([a, b], [ta, tb], sc)
        rb = M.min_bend(c) if len(c) > 8 else 1.0
        if best is None or rb > best[0]:
            best = (rb, c)
    return best[1][1:-1]


def assemble(name, tier, center, S, F, r_s, r_f, features, *, loop_radius, rise=0.07, run=0.10, bend=0.03,
             feature_sleeves=(), join_scale=None, obstacles=(), notes="", base_size=(0.5201, 0.3689, 0.0175)):
    t = M.Turtle([*S, Z0], [0, 0, 1])
    t.straight(rise).turn_to([*r_s, 0.0], bend).straight(run)
    a = t.polyline()
    s1_pt = t.p - 0.5 * run * np.r_[r_s, 0.0]
    tf = M.Turtle([*F, Z0], [0, 0, 1])                 # built backwards from the finish post's foot
    tf.straight(rise).turn_to([*(-np.asarray(r_f)), 0.0], bend).straight(run)
    f = tf.polyline()[::-1]
    s2_pt = tf.p + 0.5 * run * np.r_[r_f, 0.0]
    parts = [a]
    prev_p, prev_h = t.p, t.h
    for feat in features:
        tin = M.tangent_in(feat)
        parts.append(join(prev_p, prev_h, feat[0], tin, join_scale))
        parts.append(feat)
        prev_p, prev_h = feat[-1], M.tangent_out(feat)
    parts.append(join(prev_p, prev_h, f[0], M.tangent_in(f), join_scale))
    parts.append(f)
    poly = M.resample(np.concatenate(parts), 0.002)
    sleeves = []
    for q in [s1_pt, *feature_sleeves, s2_pt]:
        i = int(np.argmin(np.linalg.norm(poly - np.asarray(q), axis=1)))
        sleeves.append(_sleeve_at(poly, i))
    n = len(poly)
    secs = [("start_post", 30), ("route", n - 30), ("finish_post", n - 1)]
    return Design(name, tier, poly, secs, loop_radius_m=loop_radius, sleeves=sleeves, center_xy_m=tuple(center),
                  obstacles=list(obstacles), notes=notes, base_size_m=tuple(base_size))


DECK_Z = 0.1425                      # the deck top (toy z)
DECK = (0.62, 0.45, 0.0175)          # the released deck (0.52 x 0.37) made 10 cm wider and 8 cm deeper: room for the routes
CENTER = (-0.10, -0.30)              # its centre (world xy): back edge 7.5 cm in front of the robot's base line


def posts(center=CENTER, d=0.18):
    rd = robot_dir(center)[:2]
    perp = np.array([-rd[1], rd[0]])
    if perp[1] < 0:
        perp = -perp
    S, F = -d * perp, d * perp
    return S, F, rd, -rd


def finish_run(F, r_f, rise=0.07, run=0.07, bend=0.03):
    tf = M.Turtle([*F, Z0], [0, 0, 1])
    tf.straight(rise).turn_to([*(-np.asarray(r_f)), 0.0], bend).straight(run)
    return tf.polyline()[::-1], tf.p.copy(), rise + 0.5 * np.pi * bend


def finish_join(t, F, r_f, **kw):
    f, f0, ins = finish_run(F, r_f, **kw)
    j = join(t.p, t.h, f0, np.r_[r_f, 0.0])
    return np.concatenate([t.polyline(), j, f]), ins


def vertical_helix(c_xy, z0, r, pitch, turns, a0, ccw=True):
    sgn = 1.0 if ccw else -1.0
    L = turns * np.hypot(2 * np.pi * r, pitch)
    n = int(np.ceil(L / 0.001)) + 1
    t = np.linspace(0, turns * 2 * np.pi, n)
    a = a0 + sgn * t
    return np.c_[c_xy[0] + r * np.cos(a), c_xy[1] + r * np.sin(a), z0 + pitch * t / (2 * np.pi)]


def route(S, F, r_s, r_f, features, *, rise=0.07, run=0.10, run_f=0.07, bend=0.03):
    t = M.Turtle([*S, Z0], [0, 0, 1])
    t.straight(rise).turn_to([*r_s, 0.0], bend).straight(run)
    s1 = t.p - 0.5 * run * np.r_[r_s, 0.0]
    parts = [t.polyline()]
    prev_p, prev_h = t.p, t.h
    for feat in features:
        parts.append(join(prev_p, prev_h, feat[0], M.tangent_in(feat)))
        parts.append(feat)
        prev_p, prev_h = feat[-1], M.tangent_out(feat)
    f, f0, ins = finish_run(F, r_f, rise=rise, run=run_f, bend=bend)
    parts.append(join(prev_p, prev_h, f0, np.r_[r_f, 0.0]))
    parts.append(f)
    return M.resample(np.concatenate(parts), 0.002), s1, ins


def spiral_test(center=CENTER, r=0.075, pitch=0.10, turns=1.25, axis_xy=(-0.09, 0.07)):
    S, F, r_s, r_f = posts(center, 0.15)
    hel = vertical_helix(np.asarray(axis_xy), 0.20, r, pitch, turns, a0=np.deg2rad(-90))
    poly, s1, ins = route(S, F, r_s, r_f, [hel])
    L = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
    sl = [_sleeve_at(poly, int(np.argmin(np.linalg.norm(poly - s1, axis=1)))), (L - ins - 0.005, L + 0.01)]
    n = len(poly)
    post = dict(kind="cylinder", pos=[axis_xy[0], axis_xy[1], 0.1425 + 0.13], radius=0.022, height=0.26)
    return Design("spiral_test", "probe", poly, [("start_post", 30), ("route", n - 30), ("finish_post", n - 1)],
                  loop_radius_m=0.0115, sleeves=sl, center_xy_m=tuple(center), base_size_m=DECK, obstacles=[post],
                  notes="probe: vertical helix around a post")


def hook(t, axis, r, back=0.0):
    h = t.h.copy()
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    t._segment(np.pi * r, a * (1 / r))
    if back:
        t.straight(back)
    return t


def very_easy(center=CENTER):
    return program("dip_v17", "very_easy", [("run", 0.08), ("sleeve", 0.04), ("turn", "right", 0.04), ("run", 0.05),
                                             ("dip", 0.15, 0.05, 0.03), ("run", 0.05)],
                   post_d=0.15, rise=0.18, finish_rise=0.16, loop_radius=0.0115,
                   notes="start run, a U-dip, finish run; insulated finish post")


def program(name, tier, prog, *, center=CENTER, post_d=0.15, rise=0.20, finish_rise=0.14, loop_radius=0.0115,
            obstacles=(), notes=""):
    S, F, r_s, r_f = posts(center, post_d)
    t = M.Turtle([*S, Z0], [0, 0, 1])
    t.straight(rise).turn_to([*r_s, 0.0], 0.03)
    marks = []
    centres = {}
    finished = False
    for cmd in prog:
        op = cmd[0]
        if op == "run":
            t.straight(cmd[1])
        elif op == "turn":
            hz = t.h.copy(); hz[2] = 0; hz /= np.linalg.norm(hz)
            left = np.cross([0, 0, 1.], hz)
            t.turn_to(left if cmd[1] == "left" else -left, cmd[2])
        elif op == "arc":                     # ("arc", "left"|"right", r, degrees, mark_centre_name)
            hz = t.h.copy(); hz[2] = 0; hz /= np.linalg.norm(hz)
            left = np.cross([0, 0, 1.], hz)
            sgn = 1.0 if cmd[1] == "left" else -1.0
            centre = t.p + sgn * cmd[2] * left
            if len(cmd) > 4:
                centres[cmd[4]] = centre
            t._segment(np.deg2rad(cmd[3]) * cmd[2], np.array([0, 0, sgn / cmd[2]]))
        elif op == "hook":
            hz = t.h.copy(); hz[2] = 0; hz /= np.linalg.norm(hz)
            left = np.cross([0, 0, 1.], hz)
            hook(t, left if cmd[1] == "down" else -left, cmd[2], back=cmd[3])
        elif op == "dip":                     # ("dip", depth, width, r): a U-valley in the vertical plane of the heading
            depth, width, rr = cmd[1], cmd[2], cmd[3]
            hz = t.h.copy(); hz[2] = 0; hz /= np.linalg.norm(hz)
            left = np.cross([0, 0, 1.], hz)
            t.turn_to([0, 0, -1.], rr)
            t.straight(max(0.0, depth - rr - width / 2))
            t._segment(np.pi * width / 2, -left * (2 / width))         # bottom half circle: down -> forward -> up
            t.straight(max(0.0, depth - rr - width / 2))
            t.turn_to(hz, rr)
        elif op == "zig":                     # ("zig", "down"|"up", r, mid): two opposite 180-deg vertical hooks
            hz = t.h.copy(); hz[2] = 0; hz /= np.linalg.norm(hz)
            left = np.cross([0, 0, 1.], hz)
            a = left if cmd[1] == "down" else -left
            hook(t, a, cmd[2], back=cmd[3])
            hook(t, -a, cmd[2], back=0.0)
        elif op == "channel":                 # ("channel", L, gap, height): walls both sides of the next L of straight
            hz = t.h.copy(); hz[2] = 0; hz /= np.linalg.norm(hz)
            left = np.cross([0, 0, 1.], hz)
            L, gap, hgt = cmd[1], cmd[2], cmd[3]
            mid = t.p + 0.5 * L * hz
            yaw = np.degrees(np.arctan2(hz[1], hz[0]))
            top = t.p[2] + hgt
            for sgn in (1, -1):
                c = mid + sgn * (gap + 0.003) * left
                obstacles = list(obstacles) + [dict(kind="box", pos=[c[0], c[1], (DECK_Z + top) / 2],
                                                    size=[L, 0.006, top - DECK_Z], euler_deg=[0, 0, yaw])]
            t.straight(L)
        elif op == "climb":
            dz, L = cmd[1], cmd[2]
            h0 = t.h.copy()
            ang = np.arctan2(2 * dz, L)
            up = np.cos(ang) * h0 + np.sin(ang) * np.array([0, 0, 1.])
            rr = L / 4 / max(np.sin(ang), 1e-6) * 0.5
            t.turn_to(up, rr).straight(max(0.0, np.hypot(L / 2, dz) - 2 * rr * np.tan(ang / 2))).turn_to(h0, rr)
        elif op == "sleeve":
            marks.append(t.p - cmd[1] * t.h)
        elif op == "finish":                  # ("finish", bend r): bend straight down and descend to the deck: the
            t.turn_to([0, 0, -1.], cmd[1])    # route ends here (no join); the finish post is where it lands
            t.straight(t.p[2] - Z0)
            ins = (t.p[2] + cmd[1]) * 0 + np.pi / 2 * cmd[1]
            finished = True
        elif op == "post":                    # ("post", centre name, radius, height above the wire)
            c = centres[cmd[1]]
            top = t.p[2] + cmd[3]
            obstacles = list(obstacles) + [dict(kind="cylinder", pos=[c[0], c[1], (DECK_Z + top) / 2], radius=cmd[2],
                                                height=top - DECK_Z)]
        else:
            raise ValueError(op)
    if finished:
        poly = t.polyline()
        post_len = float(poly[-1 - int(round(0.0 / M.STEP))][2])
        ins = np.pi / 2 * prog[-1][1] + (finish_top - Z0 if (finish_top := float(np.max(poly[-200:, 2]))) else 0)
    else:
        poly, ins = finish_join(t, F, r_f, rise=finish_rise)
    poly = M.resample(poly, 0.002)
    L = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
    sl = [_sleeve_at(poly, int(np.argmin(np.linalg.norm(poly - q, axis=1)))) for q in marks]
    sl.append((L - ins - 0.005, L + 0.01))
    n = len(poly)
    return Design(name, tier, poly, [("start_post", 30), ("route", n - 30), ("finish_post", n - 1)],
                  loop_radius_m=loop_radius, sleeves=sl, center_xy_m=tuple(center), base_size_m=DECK,
                  obstacles=list(obstacles), notes=notes)


def orbit_test(center=CENTER):
    return program("orbit_test", "probe", [("run", 0.08), ("sleeve", 0.04), ("turn", "right", 0.04), ("run", 0.03),
                                           ("arc", "left", 0.075, 270, "c1"), ("post", "c1", 0.022, 0.045),
                                           ("run", 0.04)], post_d=0.15, rise=0.15)


def grammar_test(center=CENTER):
    return program("grammar_test", "probe", [("run", 0.08), ("sleeve", 0.04), ("turn", "right", 0.04), ("run", 0.04),
                                             ("zig", "down", 0.04, 0.03), ("run", 0.08), ("sleeve", 0.04),
                                             ("channel", 0.12, 0.04, 0.04), ("run", 0.02)],
                   post_d=0.15, rise=0.26, finish_rise=0.14)


MOUNTS = {"hard": (-0.54, 0.0, 0.0)}


def _mounted(tier, design):
    import os
    m = MOUNTS.get(tier)
    env = os.environ.get("BUZZ_MOUNT_OVERRIDE", "")
    if env.startswith(tier + ":"):
        m = tuple(float(v) for v in env.split(":", 1)[1].split(","))
        design.notes = (design.notes + " " if design.notes else "") + f"[mount override {m}]"
    design.robot_mount = None if m is None else tuple(float(v) for v in m)
    return design


TIERS = {
    "very_easy": ("buzz_wire_3d_rd_very_easy", very_easy),
    "easier": ("buzz_wire_3d_rd_easier", lambda center=CENTER: _mounted("easier", easier(center))),
    "easy": ("buzz_wire_3d_rd_easy", lambda center=CENTER: _mounted("easy", easy(center))),
    "medium": ("buzz_wire_3d_rd_medium", lambda center=CENTER: _mounted("medium", medium(center))),
    "hard": ("buzz_wire_3d_rd_hard", lambda center=CENTER: _mounted("hard", hard(center))),
}


# ------------------------------------------------------------------ design probes (two mid-route parks)
def probe_A(center=CENTER):
    return program("probe_A", "probe", [("run", 0.08), ("sleeve", 0.04), ("turn", "right", 0.04), ("run", 0.04),
                                        ("dip", 0.14, 0.05, 0.03), ("run", 0.07), ("sleeve", 0.035),
                                        ("dip", 0.12, 0.05, 0.03), ("run", 0.04)],
                   post_d=0.15, rise=0.18, finish_rise=0.16, loop_radius=0.0115)


def probe_B(center=CENTER):
    return program("probe_B", "probe", [("run", 0.06), ("sleeve", 0.03), ("turn", "right", 0.04), ("run", 0.03),
                                        ("dip", 0.14, 0.05, 0.03), ("run", 0.07), ("sleeve", 0.035),
                                        ("turn", "right", 0.05), ("turn", "right", 0.05), ("run", 0.03),
                                        ("dip", 0.12, 0.05, 0.03), ("run", 0.04)],
                   post_d=0.15, rise=0.18, finish_rise=0.16, loop_radius=0.0115)


def probe_C(center=CENTER):
    return program("probe_C", "probe", [("run", 0.06), ("sleeve", 0.03), ("turn", "right", 0.04), ("run", 0.03),
                                        ("dip", 0.14, 0.05, 0.03), ("run", 0.07), ("sleeve", 0.035),
                                        ("hook", "down", 0.04, 0.06), ("run", 0.02)],
                   post_d=0.15, rise=0.20, finish_rise=0.12, loop_radius=0.0115)


def probe_D(center=CENTER):
    return program("probe_D", "probe", [("run", 0.06), ("sleeve", 0.03), ("turn", "right", 0.04), ("run", 0.03),
                                        ("dip", 0.14, 0.05, 0.03), ("run", 0.07), ("sleeve", 0.035),
                                        ("turn", "left", 0.05), ("run", 0.03), ("zig", "down", 0.035, 0.03),
                                        ("run", 0.04)],
                   post_d=0.15, rise=0.22, finish_rise=0.12, loop_radius=0.0115)


def probe_E(center=CENTER):
    return program("probe_E", "probe", [("run", 0.06), ("sleeve", 0.03), ("turn", "right", 0.04), ("run", 0.03),
                                        ("dip", 0.14, 0.05, 0.03), ("run", 0.07), ("sleeve", 0.035),
                                        ("turn", "left", 0.05), ("run", 0.03), ("dip", 0.12, 0.05, 0.03),
                                        ("run", 0.04)],
                   post_d=0.15, rise=0.18, finish_rise=0.16, loop_radius=0.0115)


# ------------------------------------------------------------------ the course builder (laid-out legs + features)
def _feature(t, f):
    op = f[0]
    hz = t.h.copy(); hz[2] = 0; hz /= np.linalg.norm(hz)
    left = np.cross([0, 0, 1.], hz)
    if op in ("dip", "hump"):                 # ("dip"|"hump", at, depth, width, r)
        depth, width, rr = f[2], f[3], f[4]
        vz = -1.0 if op == "dip" else 1.0
        t.turn_to([0, 0, vz], rr)
        t.straight(max(0.0, depth - rr - width / 2))
        t._segment(np.pi * width / 2, (-left if op == "dip" else left) * (2 / width))
        t.straight(max(0.0, depth - rr - width / 2))
        t.turn_to(hz, rr)
    else:
        raise ValueError(op)


def _adv(f):
    if f[0] in ("dip", "hump", "fdip"):
        return 2 * f[4] + f[3]
    return f[2] if f[0] == "channel" else 0.0


def course(name, tier, S, F, H, corners, legs, *, center=CENTER, rc=0.04, rb=0.03, loop_radius=0.0115,
           obstacles=(), notes="", deck=DECK):
    S, F = np.asarray(S, float), np.asarray(F, float)
    obstacles = list(obstacles)
    pts = [S] + [np.asarray(c, float) for c in corners] + [F]
    dirs = [(pts[i + 1] - pts[i]) / np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
    t = M.Turtle([*S, Z0], [0, 0, 1])
    t.straight(H - rb - Z0).turn_to([*dirs[0], 0.0], rb)
    marks = []
    for i, d in enumerate(dirs):
        start = pts[i] + (rb * d if i == 0 else 0)
        end = pts[i + 1] - (rb * d if i == len(dirs) - 1 else 0)
        L = float(np.linalg.norm(end - start))
        t_in = 0.0 if i == 0 else rc * np.tan(0.5 * np.arccos(np.clip(dirs[i - 1] @ d, -1, 1)))
        t_out = 0.0 if i == len(dirs) - 1 else rc * np.tan(0.5 * np.arccos(np.clip(d @ dirs[i + 1], -1, 1)))
        feats = legs[i] if i < len(legs) else []
        free = L - t_in - t_out - sum(_adv(f) for f in feats)
        if free < 0:
            raise ValueError(f"leg {i} too short by {-free:.3f} m")
        done = 0.0
        for f in feats:
            gap = f[1] * free - done
            if gap < -1e-9:
                raise ValueError(f"leg {i}: features out of order")
            t.straight(gap); done += gap
            if f[0] == "sleeve":
                marks.append(t.p.copy())
            elif f[0] == "channel":           # ("channel", at, length, half_gap, top_above_wire): walls both sides
                Lc, hg, top = f[2], f[3], f[4]
                hz = t.h.copy(); hz[2] = 0; hz /= np.linalg.norm(hz)
                left = np.cross([0, 0, 1.], hz)
                mid = t.p + 0.5 * Lc * hz
                yaw = float(np.degrees(np.arctan2(hz[1], hz[0])))
                ztop = t.p[2] + top
                for sgn in (1, -1):
                    c = mid + sgn * (hg + 0.004) * left
                    obstacles.append(dict(kind="box", pos=[float(c[0]), float(c[1]), (DECK_Z + ztop) / 2],
                                          size=[Lc, 0.008, ztop - DECK_Z], euler_deg=[0, 0, yaw]))
                t.straight(Lc)
            elif f[0] == "helix":              # ("helix", at, r, pitch, turns, +1 ccw / -1 cw, (sleeve fractions))
                rr, pitch, turns, sgn = f[2], f[3], f[4], f[5]
                hz = t.h.copy(); hz[2] = 0; hz /= np.linalg.norm(hz)
                left = np.cross([0, 0, 1.], hz)
                centre = t.p + sgn * rr * left
                t.turn_to(M.Turtle.helix_heading([0, 0, 1.], hz, rr, pitch), 0.04)
                i0 = len(t.pts)
                t.helix([0, 0, sgn], rr, pitch * sgn, turns)
                hel = t.polyline()[i0:]
                for fr in (f[6] if len(f) > 6 else ()):
                    marks.append(hel[int(round(fr * (len(hel) - 1)))].copy())
                hz2 = t.h.copy(); hz2[2] = 0
                t.turn_to(hz2, 0.04)
                if len(f) > 7 and f[7]:          # a wooden post on the axis
                    top = float(t.p[2]) + 0.03
                    obstacles.append(dict(kind="cylinder", pos=[float(centre[0]), float(centre[1]), (DECK_Z + top) / 2],
                                          radius=f[7], height=top - DECK_Z))
            elif f[0] == "fdip":               # ("fdip", at, depth, width, r, side +1 left / -1 right, offset): a dip
                hz = t.h.copy(); hz[2] = 0; hz /= np.linalg.norm(hz)   # with a wooden fence beside its plane
                left = np.cross([0, 0, 1.], hz)
                adv = 2 * f[4] + f[3]
                mid = t.p + 0.5 * adv * hz + f[5] * (f[6] + 0.004) * left
                yaw = float(np.degrees(np.arctan2(hz[1], hz[0])))
                ztop = t.p[2] + 0.02
                obstacles.append(dict(kind="box", pos=[float(mid[0]), float(mid[1]), (DECK_Z + ztop) / 2],
                                      size=[adv + 0.04, 0.008, ztop - DECK_Z], euler_deg=[0, 0, yaw]))
                _feature(t, ("dip",) + tuple(f[1:5]))
            else:
                _feature(t, f)
        t.straight(free - done)
        if i < len(dirs) - 1:
            t.turn_to([*dirs[i + 1], 0.0], rc)
    z_run = float(t.p[2])
    t.turn_to([0, 0, -1.], rb)
    t.straight(z_run - rb - Z0)
    ins = (z_run - rb - Z0) + 0.5 * np.pi * rb
    poly = M.resample(t.polyline(), 0.002)
    Ltot = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
    sl = [_sleeve_at(poly, int(np.argmin(np.linalg.norm(poly - q, axis=1)))) for q in marks]
    sl.append((Ltot - ins - 0.005, Ltot + 0.01))
    n = len(poly)
    return Design(name, tier, poly, [("start_post", 30), ("route", n - 30), ("finish_post", n - 1)],
                  loop_radius_m=loop_radius, sleeves=sl, center_xy_m=tuple(center), base_size_m=tuple(deck),
                  obstacles=list(obstacles), notes=notes)


def rpt(r, deg, center=CENTER):
    base = ROBOT_XY - np.asarray(center, float)
    d0 = -base / np.linalg.norm(base)
    a = np.deg2rad(deg)
    d = np.array([np.cos(a) * d0[0] - np.sin(a) * d0[1], np.sin(a) * d0[0] + np.cos(a) * d0[1]])
    return base + r * d


def probe_P1(center=CENTER, H=0.32):
    P = lambda r, a: rpt(r, a, center)
    return course("probe_P1", "probe", P(.60, -22), P(.62, 18), H, [P(.42, -22), P(.42, 5), P(.62, 5)],
                  [[("sleeve", .5)], [("dip", .5, .14, .05, .03)], [("sleeve", .2), ("dip", .75, .13, .05, .03)], []],
                  center=center)


def probe_P2(center=CENTER, H=0.32):
    P = lambda r, a: rpt(r, a, center)
    return course("probe_P2", "probe", P(.64, -24), P(.62, 6), H, [P(.40, -24), P(.40, 6)],
                  [[("sleeve", .15), ("dip", .7, .14, .05, .03)], [("sleeve", .15), ("dip", .7, .13, .05, .03)], []],
                  center=center)


def probe_V18(center=CENTER, H=0.36):
    P = lambda r, a: rpt(r, a, center)
    return course("probe_V18", "probe", P(.58, -16), P(.60, 18), H, [P(.41, -16), P(.41, 9), P(.62, 9)],
                  [[("sleeve", .5)], [("dip", .45, .14, .05, .03)], [("sleeve", .12), ("channel", .45, .11, .055, .09)],
                   []], center=center)


# ------------------------------------------------------------------ cells: one section between two rest sleeves
SECTIONS = {
    "dip": [("dip", .5, .13, .05, .03)],
    "deepdip": [("dip", .5, .16, .05, .03)],
    "hump": [("hump", .5, .10, .05, .03)],
    "channel": [("channel", .5, .12, .055, .09)],
    "run": [],
}
ORIENT = {"in": 180.0, "out": 0.0, "ccw": 90.0, "cw": -90.0}   # leg direction relative to the radial outward direction


def cell(kind, orient, r, a, H, *, center=CENTER, approach=0.12, side=1, deck=(0.70, 0.52, 0.0175)):
    base = ROBOT_XY - np.asarray(center, float)
    c = rpt(r, a, center)
    radial = (c - base) / np.linalg.norm(c - base)
    ang = np.deg2rad(ORIENT[orient])
    d = np.array([np.cos(ang) * radial[0] - np.sin(ang) * radial[1], np.sin(ang) * radial[0] + np.cos(ang) * radial[1]])
    n = side * np.array([-d[1], d[0]])
    feats = SECTIONS[kind]
    Ls = sum(_adv(f) for f in feats) + 0.10
    A = c - 0.5 * Ls * d
    B = c + 0.5 * Ls * d
    S = A - (approach + 0.03) * n
    F = B + (approach + 0.03) * n
    return course(f"cell__{kind}__{orient}__{r}__{a}__{H}", "probe", S, F, H, [A, B],
                  [[("sleeve", .5)], feats, [("sleeve", .5)]], center=center, rc=0.035, deck=deck)


def __getattr__(name):
    if name.startswith("cell__"):
        _, kind, orient, r, a, H, *rest = name.split("__")
        side = int(rest[0]) if rest else 1
        return lambda center=CENTER: cell(kind, orient, float(r), float(a), float(H), center=center, side=side)
    raise AttributeError(name)


def _ve(name, extra, **kw):
    base = [("run", 0.08), ("sleeve", 0.04), ("turn", "right", 0.04), ("run", 0.05), ("dip", 0.15, 0.05, 0.03)]
    args = dict(post_d=0.15, rise=0.18, finish_rise=0.16, loop_radius=0.0115)
    args.update(kw)
    return program(name, "probe", base + extra, **args)


def probe_W1(center=CENTER):
    return _ve("probe_W1", [("run", 0.05), ("sleeve", 0.025), ("turn", "left", 0.04), ("run", 0.03),
                            ("dip", 0.12, 0.05, 0.03), ("run", 0.03)])


def probe_W2(center=CENTER):
    return _ve("probe_W2", [("run", 0.05), ("sleeve", 0.025), ("turn", "right", 0.04), ("run", 0.03),
                            ("dip", 0.12, 0.05, 0.03), ("run", 0.03)], post_d=0.17)


def probe_W3(center=CENTER):
    return _ve("probe_W3", [("run", 0.06), ("sleeve", 0.03), ("run", 0.02), ("dip", 0.12, 0.05, 0.03), ("run", 0.03)])


def probe_W4(center=CENTER):
    return _ve("probe_W4", [("run", 0.05), ("sleeve", 0.025), ("turn", "left", 0.04), ("run", 0.02),
                            ("hook", "down", 0.035, 0.03), ("run", 0.02)], finish_rise=0.10)


def probe_V17c(center=CENTER, H=0.363):
    P = lambda r, a: rpt(r, a, center)
    return course("probe_V17c", "probe", P(.577, -15), P(.577, 15), H, [P(.41, -16), P(.41, 14)],
                  [[("sleeve", .45)], [("dip", .45, .15, .05, .03)], []], center=center)


def _x2(name, leg1, leg2=(), *, a0=-15, a1=15, c1=(.41, -16), c2=(.41, 16), H=0.363, center=CENTER, extra=(),
        rf=.577):
    P = lambda r, a: rpt(r, a, center)
    corners = [P(*c1), P(*c2)] + [P(*c) for c in extra]
    legs = [[("sleeve", .45)], list(leg1), list(leg2)] + [[] for _ in extra]
    return course(name, "probe", P(.577, a0), P(rf, a1), H, corners, legs, center=center)


DIP = lambda at, d=.15: ("dip", at, d, .045, .025)
HUMP = lambda at, h=.08: ("hump", at, h, .045, .025)


def probe_X1(center=CENTER):   # dip, S2, dip on the same tangential leg
    return _x2("probe_X1", [DIP(.1, .14), ("sleeve", .5), DIP(.9, .12)], c1=(.41, -25), c2=(.41, 25), a0=-22, a1=22)


def probe_X2(center=CENTER):   # dip, S2, hump on the same leg
    return _x2("probe_X2", [DIP(.1, .14), ("sleeve", .5), HUMP(.9)], c1=(.41, -25), c2=(.41, 25), a0=-22, a1=22)


def probe_X3(center=CENTER):   # dip on the tangential leg; S2 then a dip on the outward finish leg
    return _x2("probe_X3", [DIP(.45, .15)], [("sleeve", .15), DIP(.8, .12)], c2=(.41, 12), a1=13, rf=.67)


def probe_X4(center=CENTER):   # dip on the tangential leg; S2 then a hump on the outward finish leg
    return _x2("probe_X4", [DIP(.45, .15)], [("sleeve", .15), HUMP(.8)], c2=(.41, 12), a1=13, rf=.67)


def probe_X5(center=CENTER):   # dip; S2 on the tangential leg; corner; a dip on an inward-returning extra leg
    return _x2("probe_X5", [DIP(.25, .15), ("sleeve", .85)], [DIP(.5, .12)], c1=(.41, -22), c2=(.41, 14),
               extra=[(.66, 14)], a1=24, rf=.62)


def probe_X6(center=CENTER, H=0.363):
    P = lambda r, a: rpt(r, a, center)
    return course("probe_X6", "probe", P(.577, -17), P(.66, -5), H, [P(.41, -18), P(.41, 15), P(.53, 15), P(.53, -5)],
                  [[("sleeve", .45)], [DIP(.4, .15)], [("sleeve", .5)], [DIP(.55, .14)], []], center=center)


def probe_X7(center=CENTER, H=0.363):
    P = lambda r, a: rpt(r, a, center)
    return course("probe_X7", "probe", P(.577, -17), P(.64, -6), H, [P(.40, -18), P(.40, 15), P(.48, 15), P(.48, -6)],
                  [[("sleeve", .45)], [DIP(.4, .15)], [("sleeve", .5)], [DIP(.55, .13)], []], center=center)


def probe_X3L(center=CENTER):   # X3 with a fence on the left of dip 2's plane
    return _x2("probe_X3L", [DIP(.45, .15)], [("sleeve", .15), ("fdip", .8, .12, .045, .025, 1, .035)], c2=(.41, 12),
               a1=13, rf=.67)


def probe_X3R(center=CENTER):   # X3 with a fence on the right of dip 2's plane
    return _x2("probe_X3R", [DIP(.45, .15)], [("sleeve", .15), ("fdip", .8, .12, .045, .025, -1, .035)], c2=(.41, 12),
               a1=13, rf=.67)


def helix_course(name, tier, *, center=CENTER, axis=(.46, 0.0), r=0.07, pitch=0.07, turns=1.0, z0=0.22, a0=-90.0,
                 fr=(1 / 3, 2 / 3), post=True, ccw=True, loop_radius=0.0115, post_d=0.15, notes=""):
    S, F, r_s, r_f = posts(center, post_d)
    axy = rpt(axis[0], axis[1], center)
    hel = vertical_helix(axy, z0, r, pitch, turns, a0=np.deg2rad(a0), ccw=ccw)
    poly, s1, ins = route(S, F, r_s, r_f, [hel])
    L = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
    sl = [_sleeve_at(poly, int(np.argmin(np.linalg.norm(poly - s1, axis=1))))]
    for f in fr:
        q = hel[int(round(f * (len(hel) - 1)))]
        sl.append(_sleeve_at(poly, int(np.argmin(np.linalg.norm(poly - q, axis=1)))))
    sl.append((L - ins - 0.005, L + 0.01))
    n = len(poly)
    obs = []
    if post:
        top = z0 + turns * pitch + 0.03
        obs.append(dict(kind="cylinder", pos=[float(axy[0]), float(axy[1]), (DECK_Z + top) / 2], radius=0.02,
                        height=top - DECK_Z))
    return Design(name, tier, poly, [("start_post", 30), ("route", n - 30), ("finish_post", n - 1)],
                  loop_radius_m=loop_radius, sleeves=sl, center_xy_m=tuple(center), base_size_m=DECK, obstacles=obs,
                  notes=notes)


def probe_H1(center=CENTER):
    return helix_course("probe_H1", "probe")


def probe_H2(center=CENTER):   # four pieces per turn
    return helix_course("probe_H2", "probe", fr=(0.25, 0.5, 0.75))


def probe_H3(center=CENTER):   # clockwise
    return helix_course("probe_H3", "probe", ccw=False, a0=90.0)


def _goto(t, target_xy, rc, end_offset=0.0):
    import copy
    tgt = np.asarray(target_xy, float)
    d = tgt - t.p[:2]; d /= np.linalg.norm(d)
    for _ in range(8):
        tt = copy.deepcopy(t)
        tt.turn_to([*d, 0.0], rc)
        e = tgt - tt.p[:2]
        d = e / np.linalg.norm(e)
    t.turn_to([*d, 0.0], rc)
    rem = float(np.linalg.norm(tgt - t.p[:2])) - end_offset
    if rem < 0:
        raise ValueError("goto: target too close")
    t.straight(rem)
    return t


def helix2(name, tier, *, center=CENTER, axis=(.46, 0.0), r=0.07, pitch=0.06, turns=1.0, H=0.24, fr=(1 / 3, 2 / 3),
           ccw=True, post=True, loop_radius=0.0115, S=(.577, -15), F=(.60, 18), via=(), rc=0.04, rb=0.03,
           run=0.10, notes=""):
    P = lambda rr, aa: rpt(rr, aa, center)
    Sxy, Fxy, Axy = P(*S), P(*F), P(*axis)
    t = M.Turtle([*Sxy, Z0], [0, 0, 1])
    rd = robot_dir(center, Sxy)
    t.straight(H - rb - Z0).turn_to(rd, rb).straight(run)
    s1 = t.p - 0.5 * run * rd
    import copy
    sgn = 1.0 if ccw else -1.0
    E = Axy
    base = ROBOT_XY - np.asarray(center, float)
    rad = (E - base) / np.linalg.norm(E - base)
    tan = sgn * np.array([-rad[1], rad[0]])
    _goto(t, E - 0.07 * tan, rc)
    _goto(t, E, rc)
    h = t.h[:2] / np.linalg.norm(t.h[:2])
    Axy = t.p[:2] + sgn * r * np.array([-h[1], h[0]])
    hel_start = len(t.pts)
    t.turn_to(M.Turtle.helix_heading([0, 0, 1.], t.h, r, pitch), 0.02)
    t.helix([0, 0, sgn], r, pitch * sgn, turns)
    hel_end = len(t.pts)
    hz = t.h.copy(); hz[2] = 0
    t.turn_to(hz, 0.03)
    for v in via:
        _goto(t, P(*v), rc)
    _goto(t, Fxy, rc, end_offset=rb)
    z_run = float(t.p[2])
    t.turn_to([0, 0, -1.], rb)
    t.straight(t.p[2] - Z0)
    poly_raw = t.polyline()
    hel_pts = poly_raw[hel_start:hel_end]
    poly = M.resample(poly_raw, 0.002)
    L = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
    ins = (z_run - rb - Z0) + 0.5 * np.pi * rb
    sl = [_sleeve_at(poly, int(np.argmin(np.linalg.norm(poly - s1, axis=1))))]
    for f in fr:
        q = hel_pts[int(round(f * (len(hel_pts) - 1)))]
        sl.append(_sleeve_at(poly, int(np.argmin(np.linalg.norm(poly - q, axis=1)))))
    sl.append((L - ins - 0.005, L + 0.01))
    obs = []
    if post:
        top = H + turns * pitch + 0.04
        obs.append(dict(kind="cylinder", pos=[float(Axy[0]), float(Axy[1]), (DECK_Z + top) / 2], radius=0.02,
                        height=top - DECK_Z))
    n = len(poly)
    return Design(name, tier, poly, [("start_post", 30), ("route", n - 30), ("finish_post", n - 1)],
                  loop_radius_m=loop_radius, sleeves=sl, center_xy_m=tuple(center), base_size_m=DECK, obstacles=obs,
                  notes=notes)


def probe_H4(center=CENTER):
    return helix2("probe_H4", "probe", axis=(.42, -6), r=.065)


def probe_H5(center=CENTER):
    return helix2("probe_H5", "probe", axis=(.42, -6), r=.065, fr=(0.25, 0.5, 0.75))


def probe_H6(center=CENTER, H=0.25):
    P = lambda r, a: rpt(r, a, center)
    return course("probe_H6", "probe", P(.577, -15), P(.60, 16), H, [P(.44, -16), P(.44, 14)],
                  [[("sleeve", .45)], [("helix", .35, .065, .07, 1.0, 1, (1 / 3, 2 / 3), .02)], []], center=center)


def probe_H7(center=CENTER, H=0.25):
    P = lambda r, a: rpt(r, a, center)
    return course("probe_H7", "probe", P(.577, -15), P(.60, 16), H, [P(.40, -16), P(.40, 14)],
                  [[("sleeve", .45)], [("helix", .35, .065, .07, 1.0, -1, (1 / 3, 2 / 3), .02)], []], center=center)


def _hc(name, *, H=0.25, r=.065, pitch=.07, turns=1.0, fr=(1 / 3, 2 / 3), pre=.12, at=.40, c1=(.44, -16),
        c2=(.44, 14), F=(.60, 16), post=.02, center=CENTER, tail=(), sgn=1):
    P = lambda rr, aa: rpt(rr, aa, center)
    leg1 = ([("sleeve", pre)] if pre is not None else []) + [("helix", at, r, pitch, turns, sgn, fr, post)] + list(tail)
    return course(name, "probe", P(.577, -15), P(*F), H, [P(*c1), P(*c2)], [[("sleeve", .45)], leg1, []],
                  center=center)


def probe_H8(center=CENTER):
    return _hc("probe_H8")


def probe_H9(center=CENTER):
    return _hc("probe_H9", fr=(.25, .5, .75))


def probe_H10(center=CENTER):
    return _hc("probe_H10", r=.08, pitch=.08)


def probe_H12(center=CENTER):
    return _hc("probe_H12", H=0.28)


def probe_H13(center=CENTER):
    return _hc("probe_H13", turns=1.5, fr=(.2, .4, .6, .8), pitch=.065)


def probe_H14(center=CENTER):   # V17c's height and corner, a descending spiral (one turn, -7 cm)
    return _hc("probe_H14", H=0.363, pitch=-.07, c1=(.41, -16), c2=(.41, 14))


def probe_H15(center=CENTER):   # as H14, the spiral without the pre-sleeve
    return _hc("probe_H15", H=0.363, pitch=-.07, c1=(.41, -16), c2=(.41, 14), pre=None)


def probe_H16(center=CENTER):   # as H14, rests at 1/2 only
    return _hc("probe_H16", H=0.363, pitch=-.07, c1=(.41, -16), c2=(.41, 14), fr=(.5,))


def probe_H17(center=CENTER):   # H15 + a rest right after the spiral's exit
    return _hc("probe_H17", H=0.363, pitch=-.07, c1=(.41, -16), c2=(.41, 14), pre=None, at=.30, tail=[("sleeve", .9)])


def probe_H18(center=CENTER):   # H17 with rests at 1/4, 1/2, 3/4 of the turn
    return _hc("probe_H18", H=0.363, pitch=-.07, c1=(.41, -16), c2=(.41, 14), pre=None, at=.30, fr=(.25, .5, .75),
               tail=[("sleeve", .9)])


def probe_H19(center=CENTER):   # pre-sleeve, a 3/4-turn descending spiral (no turn above another), a rest after it
    return _hc("probe_H19", H=0.363, pitch=-.07, turns=.75, c1=(.41, -16), c2=(.41, 14), pre=.10, at=.35,
               fr=(1 / 3, 2 / 3), tail=[("sleeve", .9)])


def probe_H20(center=CENTER):   # as H19 with 0.9 turn, steeper
    return _hc("probe_H20", H=0.363, pitch=-.10, turns=.9, c1=(.41, -16), c2=(.41, 14), pre=.10, at=.35,
               fr=(1 / 3, 2 / 3), tail=[("sleeve", .9)])


def probe_H21(center=CENTER):   # one full turn, steep (12 cm per turn: the exit passes well below the entry)
    return _hc("probe_H21", H=0.363, pitch=-.12, turns=1.0, c1=(.41, -16), c2=(.41, 14), pre=.10, at=.35,
               fr=(1 / 3, 2 / 3), tail=[("sleeve", .9)])


def probe_H22(center=CENTER):   # as H21, one rest at the half turn
    return _hc("probe_H22", H=0.363, pitch=-.12, turns=1.0, c1=(.41, -16), c2=(.41, 14), pre=.10, at=.35,
               fr=(.5,), tail=[("sleeve", .9)])


def probe_H23(center=CENTER):   # H21 turning clockwise: the spiral's axis outwards of the leg (away from the robot)
    return _hc("probe_H23", H=0.363, pitch=-.12, turns=1.0, c1=(.41, -16), c2=(.41, 14), pre=.10, at=.35,
               fr=(1 / 3, 2 / 3), tail=[("sleeve", .9)], sgn=-1)


def probe_H24(center=CENTER):   # H21 on a leg further out (r 0.48): the axis at ~0.42 from the robot
    return _hc("probe_H24", H=0.363, pitch=-.12, turns=1.0, c1=(.48, -16), c2=(.48, 12), pre=.10, at=.35,
               fr=(1 / 3, 2 / 3), tail=[("sleeve", .9)], F=(.62, 14))


def probe_H25(center=CENTER):   # H23 (clockwise, axis outwards) climbing: 0.33 -> 0.43
    return _hc("probe_H25", H=0.33, pitch=.10, turns=1.0, c1=(.41, -16), c2=(.41, 14), pre=.10, at=.35,
               fr=(1 / 3, 2 / 3), tail=[("sleeve", .9)], sgn=-1)


def probe_H26(center=CENTER):   # H24 (counter-clockwise, leg at 0.48) climbing: 0.33 -> 0.43
    return _hc("probe_H26", H=0.33, pitch=.10, turns=1.0, c1=(.48, -16), c2=(.48, 12), pre=.10, at=.35,
               fr=(1 / 3, 2 / 3), tail=[("sleeve", .9)], F=(.62, 14))


def probe_H27(center=CENTER):   # H24's layout (leg at 0.48, start at 0.363) with a climbing spiral (0.363 -> 0.463)
    return _hc("probe_H27", H=0.363, pitch=.10, turns=1.0, c1=(.48, -16), c2=(.48, 12), pre=.10, at=.35,
               fr=(1 / 3, 2 / 3), tail=[("sleeve", .9)], F=(.62, 14))


def probe_H28(center=CENTER):   # H27 with 1.25 turns (0.363 -> 0.463) and four rests on the spiral
    return _hc("probe_H28", H=0.363, pitch=.08, turns=1.25, c1=(.47, -24), c2=(.47, 0), pre=.10, at=.30,
               fr=(.2, .4, .6, .8), tail=[("sleeve", .9)], F=(.62, 2))


def probe_H29(center=CENTER):   # H27 without the post on the axis
    return _hc("probe_H29", H=0.363, pitch=.10, turns=1.0, c1=(.48, -16), c2=(.48, 12), pre=.10, at=.35,
               fr=(1 / 3, 2 / 3), tail=[("sleeve", .9)], F=(.62, 14), post=0)


def probe_H30(center=CENTER):   # H24 with a 0.6-turn spiral (one rest at its middle), then on to the finish
    return _hc("probe_H30", H=0.363, pitch=-.12, turns=.6, c1=(.48, -16), c2=(.48, 12), pre=.10, at=.30,
               fr=(.5,), F=(.62, 14))


def probe_H31(center=CENTER):   # H24 with a 0.75-turn spiral (rests at 1/3, 2/3)
    return _hc("probe_H31", H=0.363, pitch=-.12, turns=.75, c1=(.48, -16), c2=(.48, 12), pre=.10, at=.30,
               fr=(1 / 3, 2 / 3), F=(.62, 14))


def spiral_tier(name, tier, *, center=CENTER, S=(.577, -15), C1=(.48, -16), H=0.363, L1=0.12, r=.065, pitch=-.12,
                turns=.67, fr=(.5,), L2=0.10, tail=True, rc=0.04, rb=0.03, post=.02, loop_radius=0.0115,
                extra=(), l1_feats=(), l0_feats=(), notes=""):
    P = lambda rr, aa: rpt(rr, aa, center)
    Sxy, C = P(*S), P(*C1)
    t = M.Turtle([*Sxy, Z0], [0, 0, 1])
    d0 = (C - Sxy) / np.linalg.norm(C - Sxy)
    t.straight(H - rb - Z0).turn_to([*d0, 0.0], rb)
    L0 = float(np.linalg.norm(C - Sxy)) - rb - rc
    marks = []
    if not l0_feats:
        t.straight(L0 / 2); marks.append(t.p.copy()); t.straight(L0 / 2)
    else:                                   # features on the start run (sleeves, dips)
        free0 = L0 - sum(_adv(f) for f in l0_feats)
        done0 = 0.0
        for f in l0_feats:
            gap = f[1] * free0 - done0
            t.straight(max(gap, 0)); done0 += max(gap, 0)
            if f[0] == "sleeve":
                marks.append(t.p.copy())
            else:
                _feature(t, f)
        t.straight(max(free0 - done0, 0))
    base = ROBOT_XY - np.asarray(center, float)
    rad = (C - base) / np.linalg.norm(C - base)
    tan = np.array([-rad[1], rad[0]])
    t.turn_to([*tan, 0.0], rc)
    if not l1_feats:
        t.straight(L1 / 2); marks.append(t.p.copy()); t.straight(L1 / 2)
    else:                                   # features on the tangential run before the spiral (sleeves, dips)
        free1 = L1 - sum(_adv(f) for f in l1_feats)
        done1 = 0.0
        for f in l1_feats:
            gap = f[1] * free1 - done1
            t.straight(max(gap, 0)); done1 += max(gap, 0)
            if f[0] == "sleeve":
                marks.append(t.p.copy())
            else:
                _feature(t, f)
        t.straight(max(free1 - done1, 0))
    hz = t.h.copy(); hz[2] = 0; hz /= np.linalg.norm(hz)
    centre = t.p + r * np.cross([0, 0, 1.], hz)
    t.turn_to(M.Turtle.helix_heading([0, 0, 1.], hz, r, pitch), 0.04)
    i0 = len(t.pts)
    t.helix([0, 0, 1.], r, pitch, turns)
    hel = t.polyline()[i0:]
    for f in fr:
        marks.append(hel[int(round(f * (len(hel) - 1)))].copy())
    hz2 = t.h.copy(); hz2[2] = 0
    t.turn_to(hz2, 0.04)
    obstacles = []
    if post:
        top = float(np.max(hel[:, 2])) + 0.02
        obstacles.append(dict(kind="cylinder", pos=[float(centre[0]), float(centre[1]), (DECK_Z + top) / 2],
                              radius=post, height=top - DECK_Z))
    feats = list(extra)
    free = L2 - sum(_adv(f) for f in feats)
    done = 0.0
    if tail:
        feats = [("sleeve", 0.5 if not extra else 0.15)] + feats
    for f in feats:
        gap = f[1] * free - done
        t.straight(max(gap, 0)); done += max(gap, 0)
        if f[0] == "sleeve":
            marks.append(t.p.copy())
        else:
            _feature(t, f)
    t.straight(max(free - done, 0))
    z_run = float(t.p[2])
    t.turn_to([0, 0, -1.], rb)
    t.straight(z_run - rb - Z0)
    ins = (z_run - rb - Z0) + 0.5 * np.pi * rb
    poly = M.resample(t.polyline(), 0.002)
    Ltot = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
    sl = [_sleeve_at(poly, int(np.argmin(np.linalg.norm(poly - q, axis=1)))) for q in marks]
    sl.append((Ltot - ins - 0.005, Ltot + 0.01))
    n = len(poly)
    return Design(name, tier, poly, [("start_post", 30), ("route", n - 30), ("finish_post", n - 1)],
                  loop_radius_m=loop_radius, sleeves=sl, center_xy_m=tuple(center), base_size_m=DECK,
                  obstacles=obstacles, notes=notes)


def probe_P10(center=CENTER):
    return spiral_tier("probe_P10", "probe")


def probe_P11(center=CENTER):
    return spiral_tier("probe_P11", "probe", turns=.6, pitch=-.10, C1=(.50, -18))


def probe_P12(center=CENTER):
    return spiral_tier("probe_P12", "probe", turns=.67, pitch=-.10, L1=.16, C1=(.48, -20))


def probe_P13(center=CENTER):
    return spiral_tier("probe_P13", "probe", turns=.55, pitch=-.12, L1=.10, C1=(.46, -16), fr=(.33, .66))


def probe_Q1(center=CENTER):
    return spiral_tier("probe_Q1", "probe", turns=.75, pitch=-.10, C1=(.44, -18), fr=(.4, .8))


def probe_Q2(center=CENTER):
    return spiral_tier("probe_Q2", "probe", turns=.85, pitch=-.10, C1=(.44, -18), fr=(.4, .8))


def probe_Q3(center=CENTER):
    return spiral_tier("probe_Q3", "probe", turns=.75, pitch=-.10, C1=(.48, -18), fr=(.4, .8))


def probe_Q4(center=CENTER):
    return spiral_tier("probe_Q4", "probe", turns=.85, pitch=-.10, C1=(.48, -18), fr=(.4, .8))


def probe_Q5(center=CENTER):
    return spiral_tier("probe_Q5", "probe", turns=.8, pitch=-.12, C1=(.42, -16), fr=(.4, .8))


def probe_F1(center=CENTER, H=0.363, off=.035, pre=.15):
    P = lambda r, a: rpt(r, a, center)
    return course("probe_F1", "probe", P(.577, -15), P(.577, 15), H, [P(.41, -16), P(.41, 14)],
                  [[("sleeve", .45)], [("sleeve", pre), ("fdip", .6, .15, .05, .03, 1, off)], []], center=center)


def probe_F2(center=CENTER):
    return probe_F1(center, off=.045)


def probe_F3(center=CENTER):   # the fence on the far side (control)
    P = lambda r, a: rpt(r, a, center)
    return course("probe_F3", "probe", P(.577, -15), P(.577, 15), 0.363, [P(.41, -16), P(.41, 14)],
                  [[("sleeve", .45)], [("sleeve", .15), ("fdip", .6, .15, .05, .03, -1, .035)], []], center=center)


def probe_Q6(center=CENTER):   # start post moved right (away from the spiral), 0.75 turn
    return spiral_tier("probe_Q6", "probe", S=(.56, -28), C1=(.46, -29), L1=.16, turns=.75, pitch=-.10, fr=(.4, .8))


def probe_Q7(center=CENTER):   # as Q6, 0.85 turn
    return spiral_tier("probe_Q7", "probe", S=(.56, -28), C1=(.46, -29), L1=.16, turns=.85, pitch=-.10, fr=(.4, .8))


def probe_Q8(center=CENTER):   # as Q6, 0.6 turn (control)
    return spiral_tier("probe_Q8", "probe", S=(.56, -28), C1=(.46, -29), L1=.16, turns=.6, pitch=-.10, fr=(.5,))


def probe_Q9(center=CENTER):   # Q6 + a U-dip on the run after the spiral (a rest before it)
    return spiral_tier("probe_Q9", "probe", S=(.56, -28), C1=(.46, -29), L1=.16, turns=.75, pitch=-.10, fr=(.4, .8),
                       L2=.24, extra=[("dip", .6, .11, .045, .025)])


def probe_Q10(center=CENTER):  # Q6 + a climb-over (hump) on the run after the spiral
    return spiral_tier("probe_Q10", "probe", S=(.56, -28), C1=(.46, -29), L1=.16, turns=.75, pitch=-.10, fr=(.4, .8),
                       L2=.24, extra=[("hump", .6, .08, .045, .025)])


def probe_Q11(center=CENTER):   # V17's dip on the tangential run, a rest, then Q6's spiral
    return spiral_tier("probe_Q11", "probe", S=(.577, -15), C1=(.41, -16), L1=.30, turns=.75, pitch=-.10,
                       fr=(.4, .8), l1_feats=[("dip", .25, .15, .05, .03), ("sleeve", .75)])


def probe_Q12(center=CENTER):   # as Q11 with the start moved right and the corner further out
    return spiral_tier("probe_Q12", "probe", S=(.56, -28), C1=(.44, -29), L1=.30, turns=.75, pitch=-.10,
                       fr=(.4, .8), l1_feats=[("dip", .25, .15, .05, .03), ("sleeve", .75)])


def probe_Q13(center=CENTER):   # a U-dip on the start run (rest before it), then Q6's corner, rest and spiral
    return spiral_tier("probe_Q13", "probe", S=(.62, -18), C1=(.44, -24), L1=.16, turns=.75, pitch=-.10,
                       fr=(.4, .8), l0_feats=[("sleeve", .2), ("dip", .6, .14, .045, .025)])


def probe_Q14(center=CENTER):   # as Q13, the dip on the tangential run before the spiral instead
    return spiral_tier("probe_Q14", "probe", S=(.56, -28), C1=(.44, -29), L1=.26, turns=.75, pitch=-.10,
                       fr=(.4, .8), l1_feats=[("dip", .3, .14, .045, .025), ("sleeve", .85)])


def probe_Q15(center=CENTER):   # Q13 with a flatter spiral and a U-dip on the run after it (a rest before the dip)
    return spiral_tier("probe_Q15", "probe", S=(.62, -18), C1=(.44, -24), L1=.16, turns=.75, pitch=-.05,
                       fr=(.4, .8), l0_feats=[("sleeve", .2), ("dip", .6, .14, .045, .025)],
                       L2=.24, extra=[("dip", .6, .11, .045, .025)])


def easier(center=CENTER):
    import os
    ring, feats, name = 0.0110, [("sleeve", .35)], "spiral_v9"
    if os.environ.get("BUZZ_EASIER"):
        v = [float(x) for x in os.environ["BUZZ_EASIER"].split(",")]
        ring = v[0] / 1000
        if len(v) > 1 and v[1] > 0:
            feats = [("sleeve", .35), ("dip", .75, v[1], .045, .025)]
        name = f"spiral_v9x_{os.environ['BUZZ_EASIER'].replace(',', '_')}"
    return spiral_tier(name, "easier", S=(.68, -16), C1=(.42, -24), L1=.16, turns=.75, pitch=-.10,
                       fr=(.4, .8), loop_radius=ring, center=center, L2=.10, post=.012, l0_feats=feats,
                       notes="start run, 3/4-turn spiral staircase around a post, finish post")


def easy(center=CENTER):
    return _easy_like("dip_spiral_v8", "easy", center=center)


def probe_Q16(center=CENTER):   # Q13 with 0.82 turn and rests at 0.33 / 0.66 of the spiral
    return spiral_tier("probe_Q16", "probe", S=(.62, -18), C1=(.44, -24), L1=.16, turns=.82, pitch=-.10,
                       fr=(.33, .66), l0_feats=[("sleeve", .2), ("dip", .6, .14, .045, .025)], loop_radius=0.0100)


def probe_Q17(center=CENTER):   # Q13 with a hump (climb-over) on the tangential run before the spiral rest
    return spiral_tier("probe_Q17", "probe", S=(.62, -18), C1=(.44, -24), L1=.26, turns=.75, pitch=-.10,
                       fr=(.4, .8), l0_feats=[("sleeve", .2), ("dip", .6, .14, .045, .025)],
                       l1_feats=[("hump", .35, .07, .045, .025), ("sleeve", .9)], loop_radius=0.0100)


def _easy_like(name, tier, *, fr=(.4, .8), dip=.14, ring=0.0105, turns=.75, center=CENTER):
    return spiral_tier(name, tier, S=(.68, -16), C1=(.42, -24), L1=.16, turns=turns, pitch=-.10, fr=fr, L2=.10, post=.012,
                       l0_feats=[("sleeve", .35), ("dip", .75, dip, .045, .025)], loop_radius=ring, center=center,
                       notes="start-run U-dip, spiral staircase around a wooden post")


def probe_M1(center=CENTER):
    return _easy_like("probe_M1", "probe", dip=.16, ring=0.0100)


def probe_H1b(center=CENTER):
    return _easy_like("probe_H1b", "probe", fr=(.3, .6), ring=0.0100)


def probe_H2b(center=CENTER):
    return _easy_like("probe_H2b", "probe", fr=(.25, .5, .75), ring=0.0095)


def medium(center=CENTER):
    return _easy_like("dip_spiral_deep_v8", "medium", dip=.16, ring=0.0100, center=center)


def hard(center=CENTER):
    return _easy_like("dip_spiral_tight_v8", "hard", fr=(.25, .5, .75), ring=0.0095, center=center)


def probe_Hh1(center=CENTER):   # hard candidate: a tighter spiral (r 5.5 cm) round a thicker post
    return spiral_tier("probe_Hh1", "probe", S=(.62, -18), C1=(.44, -24), L1=.16, turns=.75, pitch=-.10, r=.055,
                       post=.025, fr=(.25, .5, .75), l0_feats=[("sleeve", .2), ("dip", .6, .14, .045, .025)],
                       loop_radius=0.0095)


def probe_Hh2(center=CENTER, C1=(.44, -28), L1=.22, S=(.60, -22)):   # hard candidate: easy + a 2nd U-dip on the tangential run
    return spiral_tier("probe_Hh2", "probe", S=S, C1=C1, L1=L1, turns=.75, pitch=-.10, r=.06,
                       fr=(.4, .8), l0_feats=[("sleeve", .2), ("dip", .6, .14, .045, .025)],
                       l1_feats=[("sleeve", .08), ("dip", .5, .12, .04, .025), ("sleeve", .95)], loop_radius=0.0095)


def rest_check(d, *, handle=0.075, local=0.035, finish_r=0.10):
    w = d.wire(); r = w.rail
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))]
    out = []
    for k, (a, b) in enumerate(d.sleeves[:-1]):
        c = 0.5 * (a + b); i = int(np.searchsorted(s, c)); p = r[i]
        far = np.abs(s - c) > local
        seg = np.linspace(p, p - [0, 0, handle], 15)
        dmin = min(float(np.min(np.linalg.norm(r[far] - q, axis=1))) for q in seg)
        for ob in d.obstacles:
            dxy = float(np.linalg.norm(np.asarray(ob["pos"][:2]) - p[:2])) - ob.get("radius", 0.0)
            dmin = min(dmin, dxy)
        out.append((f"rest S{k + 1} at {c:.3f}", round(dmin, 4)))
    F = r[-1]
    near = (s < s[-1] - 0.25) & (r[:, 2] < DECK_Z + 0.12)
    dF = float(np.min(np.linalg.norm(r[near][:, :2] - F[:2], axis=1))) if near.any() else 1.0
    for ob in d.obstacles:
        dF = min(dF, float(np.linalg.norm(np.asarray(ob["pos"][:2]) - F[:2])) - ob.get("radius", 0.0))
    out.append(("finish post surroundings", round(dF, 4)))
    return out


def design(tier):
    return TIERS[tier][1]()
