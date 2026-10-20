"""Game controller battery swap: the MJCF of battery_replacement's entities (coordinates in the ALOHA mount frame).

Physical picture (design: runs/controller_battery_design_20261014, README.md v2 and README_v3.md v3b): an Xbox-style
controller lies face down in a cradle on the table; its battery compartment (two AA slots side by side, a 1.6 mm rib
between them) is closed by a sliding cover (three hooks under the -y edge, a tongue under the +y lip, a spring-loaded
detent plunger). The slots hold two old AA cells, each pushed by its spring contact (a plate on a slide joint at the
- end) against the + contact plate. The cells sit 4 mm below the rim with 1 mm of side clearance, so two fingers cannot
take a seated cell (the old cells are force-closure grasp targets); a pry notch beside each + plate takes the blade of
the pry stick, whose tip carries a small foot that lifts the cell from below when the stick is levered about the notch's
outer edge. Two new cells lie in a tray, the stick stands in a holder behind the controller, a recycling box sits in front.

Frames. Mount frame: the ALOHA's (x from the left arm base toward the right, y away from the observation cameras, z up,
origin on the table top between the arm bases). Controller frame C (all compartment geometry, millimetres): origin at the
compartment floor's centre, x along the cell axes (slot A's + end toward +x, slot B's toward -x), y toward the controller's
top edge, z up out of the compartment. Cell frame: origin at the cell's overall centre, +x toward the + nub. Cover frame:
origin at the closed position's (0, 0, Z_RIM) in C (the plate's underside). Stick frame: origin at the blade tip, +z toward
the handle; the blade is BLADE[0] wide along x (across the slot when used) and the foot sticks out toward -x.

Every collision geom is in group 3 with a coloured visual twin (contype 0, no mass): the renderer draws a colliding geom
of groups 0-2 grey, a visual geom in its colour (the cells: old grey body, tan band; new black body, copper band). Contacts follow the topogym rigid profile (solref 1.25 ms / dampratio 2, solimp .9 .99, friction .3)."""
from __future__ import annotations

import numpy as np

ENTITIES = ("controller", "cover", "old_A", "old_B", "new_1", "new_2", "stick")
CELLS = ("old_A", "old_B", "new_1", "new_2")
MM = 1e-3

# ---------------------------------------------------------------- AA cell (Energizer E91 / IEC LR6), mm
AA_D = 14.5
AA_R = AA_D / 2
AA_L = 50.5                  # overall, the + nub included
NUB_D, NUB_H = 5.5, 1.0
AA_MASS_KG = .023
BODY_L = AA_L - NUB_H
PLUS_BAND = 17.0             # the coloured band at the + end (visual)

# ---------------------------------------------------------------- compartment (controller frame C), mm
C_GAP = 1.0                  # side clearance between a seated cell and the wall / rib
RIB_T = 1.6
D_SINK = 4.0                 # rim height above a seated cell's top
X_IN = 28.0                  # end walls' inner faces
Y_AX = RIB_T / 2 + C_GAP + AA_R                 # cell axes at |y| = 9.05
Y_IN = RIB_T / 2 + 2 * C_GAP + AA_D             # long walls' inner faces, |y| = 17.3
Z_RIM = AA_D + D_SINK                           # 18.5
Y_LEDGE, X_LEDGE = 22.0, 30.0                   # the cover's recess
COVER_T = 2.0
Z_TOP = Z_RIM + COVER_T                         # the shell's back surface, 20.5
PLATE_T, PLATE_Z, PLATE_W = 1.0, (3.0, 11.5), 7.0
X_PLUS = X_IN - PLATE_T                         # slot A's + plate face (B: -27)
SPRING_PRE, SPRING_TRAVEL, SPRING_K, SPRING_T = 2.0, 3.5, 1.2, .8   # mm, mm, N/mm, mm
NOTCH_Y0 = PLATE_W / 2 + .2                     # the notch starts 3.7 mm off the cell axis
NOTCH_INTO_WALL, NOTCH_WALL_CUT, NOTCH_X0 = 2.0, 2.0, 26.0
NOTCH_CHAMFER, NOTCH_RAMP, RIM_CHAMFER = 1.0, 2.0, 1.0
POCKET = (20.0, 1.0)                            # floor pocket under the notch: from |x| = 20, 1 mm deep
HOOK_X, HOOK_W = (-12.0, 0.0, 12.0), 5.0
SLIDE = 5.0
TONGUE_SLOT_Y, LIP_Z = 33.5, 19.7
PLUNGER_X, PLUNGER_Y, PLUNGER_R = 6.0, -19.6, 1.5
PLUNGER_PRE_N, PLUNGER_K = 2.0, .5              # N, N/mm
GROOVE_W = 2.4
RIDGE_H = 3.0                                   # the cover's thumb ridge

# ---------------------------------------------------------------- pry stick (stick frame), mm
BLADE = (2.0, 1.2, 30.0)     # width x thickness x length
FOOT = (2.0, 1.0)            # beyond the blade's -x edge, height along the stick
SHAFT_R, SHAFT_L = 3.0, 40.0
HANDLE = (16.0, 16.0, 50.0)
STICK_MASS_KG = .03
STAND_FUNNEL = 2.0           # the stick holder's square hole: a 2 mm x 45 deg lead-in at its mouth

# ---------------------------------------------------------------- layout (mount frame, m)
LAYOUT_DY = .02              # the design layout moved 20 mm toward +y (the arms' reach band's middle; README_v3 4.1)
CTRL_XY_M = (0., .10 + LAYOUT_DY)
Z_FLOOR_M = .034             # the compartment floor above the table
TRAY_XY_M = (.065, -.045 + LAYOUT_DY)
TRAY_FLOOR = 10.0            # the tray's groove floor above the table, mm
BOX_XY_M = (-.065, -.045 + LAYOUT_DY)
BOX_IN = (70.0, 50.0, 30.0)  # inner x, y, depth, mm
STAND_XY_M = (0., .205 + LAYOUT_DY)
PAD_XY_M = (-.120, .200 + LAYOUT_DY)

CONTACT = 'solref="0.00125 2" solimp="0.9 0.99 0.001 0.5 2" condim="3"'
FRICTION = 'friction="0.3 0.005 0.0001"'
VIS = 'contype="0" conaffinity="0" group="2" mass="0"'


def _f(*v):
    return " ".join(f"{x:.9g}" for x in v)


def subtract(box, cuts):
    """An axis-aligned box minus axis-aligned boxes -> disjoint boxes (x0, x1, y0, y1, z0, z1), merged along x then y."""
    axes = []
    for k in range(3):
        lo, hi = box[2 * k], box[2 * k + 1]
        cs = {lo, hi}
        for c in cuts:
            for v in (c[2 * k], c[2 * k + 1]):
                if lo < v < hi:
                    cs.add(v)
        axes.append(sorted(cs))
    keep = {}
    for i in range(len(axes[0]) - 1):
        for j in range(len(axes[1]) - 1):
            for k in range(len(axes[2]) - 1):
                cx = (axes[0][i] + axes[0][i + 1]) / 2
                cy = (axes[1][j] + axes[1][j + 1]) / 2
                cz = (axes[2][k] + axes[2][k + 1]) / 2
                if not any(c[0] < cx < c[1] and c[2] < cy < c[3] and c[4] < cz < c[5] for c in cuts):
                    keep[i, j, k] = True
    out, used = [], set()
    for (i, j, k) in sorted(keep):
        if (i, j, k) in used:
            continue
        i1 = i
        while (i1 + 1, j, k) in keep and (i1 + 1, j, k) not in used:
            i1 += 1
        j1 = j
        while all((ii, j1 + 1, k) in keep and (ii, j1 + 1, k) not in used for ii in range(i, i1 + 1)):
            j1 += 1
        for ii in range(i, i1 + 1):
            for jj in range(j, j1 + 1):
                used.add((ii, jj, k))
        out.append((axes[0][i], axes[0][i1 + 1], axes[1][j], axes[1][j1 + 1], axes[2][k], axes[2][k + 1]))
    return out


def box_geom(b, name=None, rgba="0.16 0.16 0.17 1", extra="", offset=(0., 0., 0.)):
    """A box geom from mm bounds (x0, x1, y0, y1, z0, z1), offset (m) added to its centre."""
    c = [(b[0] + b[1]) / 2 * MM + offset[0], (b[2] + b[3]) / 2 * MM + offset[1], (b[4] + b[5]) / 2 * MM + offset[2]]
    s = [(b[1] - b[0]) / 2 * MM, (b[3] - b[2]) / 2 * MM, (b[5] - b[4]) / 2 * MM]
    nm = f' name="{name}"' if name else ""
    return f'<geom{nm} type="box" pos="{_f(*c)}" size="{_f(*s)}" rgba="{rgba}" {extra}/>'


# ---------------------------------------------------------------- the compartment
def chamfer_edges():
    """Edges with a 45 deg chamfer: (axis along, (a0, a1) along it, u0 the edge on the other horizontal axis, side of the
    solid (+1: u > u0), z0 the edge height, size), mm."""
    E = []
    yn0, yn1 = Y_AX + NOTCH_Y0, Y_IN + NOTCH_WALL_CUT          # the notch across the slot: 12.75 .. 19.3
    xw = X_IN + NOTCH_INTO_WALL                                  # the notch's back wall, 30
    for sg in (1., -1.):                                         # slot A, slot B (mirrored in x and y)
        c = NOTCH_CHAMFER
        E += [("x", tuple(sorted((sg * NOTCH_X0, sg * xw))), sg * yn1, sg, Z_RIM, c),
              ("x", tuple(sorted((sg * X_IN, sg * xw))), sg * yn0, -sg, Z_RIM, c),
              ("y", tuple(sorted((sg * yn0, sg * yn1))), sg * xw, sg, Z_TOP, c)]
        r = RIM_CHAMFER
        lo, hi = (-X_IN, NOTCH_X0) if sg > 0 else (-NOTCH_X0, X_IN)
        gaps = [] if sg > 0 else sorted([(h - 3., h + 3.) for h in HOOK_X] + [(PLUNGER_X - 2.5, PLUNGER_X + 2.5)])
        a = lo
        for g0, g1 in gaps + [(hi, hi)]:
            if g0 > a:
                E.append(("x", (a, min(g0, hi)), sg * Y_IN, sg, Z_RIM, r))
            a = max(a, g1)
        E.append(("y", tuple(sorted((sg * (RIB_T / 2), sg * yn0))), sg * X_IN, sg, Z_RIM, r))
        E.append(("y", tuple(sorted((sg * (RIB_T / 2), sg * Y_IN))), -sg * X_IN, -sg, Z_RIM, r))
    return E


def chamfer_cut_and_prism(e):
    """The square to cut from the solid and the triangular prism (vertices, mm) to add back."""
    ax, (a0, a1), u0, su, z0, c = e
    u1 = u0 + su * c
    cut_u = tuple(sorted((u0, u1)))
    tri = [(u1, z0), (u1, z0 - c), (u0, z0 - c)]
    if ax == "x":
        cut = (a0, a1, cut_u[0], cut_u[1], z0 - c, z0 + .01)
        verts = [(a, u, z) for a in (a0, a1) for (u, z) in tri]
    else:
        cut = (cut_u[0], cut_u[1], a0, a1, z0 - c, z0 + .01)
        verts = [(u, a, z) for a in (a0, a1) for (u, z) in tri]
    return cut, verts


def ramps():
    """The notch's -x end in the long wall as a 45 deg ramp in plan view: [(cut, prism vertices mm)] for A and B."""
    out, r = [], NOTCH_RAMP
    for sg in (1., -1.):
        x1, x0 = sg * NOTCH_X0, sg * (NOTCH_X0 - r)
        y0, y1 = sg * Y_IN, sg * (Y_IN + r)
        cut = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1), 0., Z_RIM + .01)
        tri = [(x0, y0), (x0, y1), (x1, y1)]
        out.append((cut, [(x, y, z) for z in (0., Z_RIM) for (x, y) in tri]))
    return out


def shell_boxes():
    """The static solid around the compartment (C, mm)."""
    solid = (-45., 45., -40., 40., -4., Z_TOP)
    big = 60.
    cuts = [(-X_IN, X_IN, -Y_IN, Y_IN, 0., big),                                    # cavity
            (-X_LEDGE, X_LEDGE, -Y_LEDGE, Y_LEDGE, Z_RIM, big),                     # the cover's recess
            (-X_LEDGE, X_LEDGE, Y_LEDGE - .01, TONGUE_SLOT_Y, Z_RIM - 3., LIP_Z),   # tongue pocket under the +y lip
            (X_IN - .01, X_IN + NOTCH_INTO_WALL, Y_AX + NOTCH_Y0, Y_IN + NOTCH_WALL_CUT, 0., big),     # pry notch A
            (NOTCH_X0, X_IN + .01, Y_IN - .01, Y_IN + NOTCH_WALL_CUT, 0., big),                         # and into the wall
            (-X_IN - NOTCH_INTO_WALL, -X_IN + .01, -Y_IN - NOTCH_WALL_CUT, -Y_AX - NOTCH_Y0, 0., big),  # pry notch B
            (-X_IN - .01, -NOTCH_X0, -Y_IN - NOTCH_WALL_CUT, -Y_IN + .01, 0., big),
            (POCKET[0], X_IN + NOTCH_INTO_WALL, Y_AX + NOTCH_Y0, Y_IN + NOTCH_WALL_CUT, -POCKET[1], .01),     # pocket A
            (-X_IN - NOTCH_INTO_WALL, -POCKET[0], -Y_IN - NOTCH_WALL_CUT, -Y_AX - NOTCH_Y0, -POCKET[1], .01),  # pocket B
            (PLUNGER_X - 2.5, PLUNGER_X + 2.5, PLUNGER_Y - 1.8, PLUNGER_Y + 1.8, 12., big)]                 # plunger well
    for h in HOOK_X:
        cuts.append((h - 3., h + 3., -Y_LEDGE - .01, -Y_IN + .01, 15.5, big))     # hook slots through the -y wall's top
        cuts.append((h - 3., h + 3., -26., -Y_LEDGE + .01, 15.3, 16.8))           # hook pockets under the -y edge
    cuts += [chamfer_cut_and_prism(e)[0] for e in chamfer_edges()]
    cuts += [r[0] for r in ramps()]
    return subtract(solid, cuts)


def _mesh(name, verts_mm):
    return f'<mesh name="{name}" vertex="{" ".join(f"{v * MM:.9g}" for p in verts_mm for v in p)}"/>'


def _compartment(meshes):
    """The shell body's content (C frame, positions in metres relative to the compartment floor centre)."""
    shell = "0.17 0.17 0.18 1"
    L = [box_geom(b, rgba=shell, extra=f"{CONTACT} {FRICTION}") for b in shell_boxes()]
    rt = RIB_T / 2
    L.append(box_geom((-X_IN, X_IN, -rt, rt, 0., Z_RIM - RIM_CHAMFER), name="rib", rgba=shell, extra=f"{CONTACT} {FRICTION}"))
    meshes.append(_mesh("rib_roof", [(x, y, z) for x in (-X_IN, X_IN) for (y, z) in
                                     ((-rt, Z_RIM - RIM_CHAMFER), (rt, Z_RIM - RIM_CHAMFER), (0., Z_RIM - RIM_CHAMFER + rt))]))
    L.append(f'<geom name="rib_roof" type="mesh" mesh="rib_roof" rgba="{shell}" {CONTACT} {FRICTION}/>')
    for i, e in enumerate(chamfer_edges()):
        meshes.append(_mesh(f"chamfer_{i}", chamfer_cut_and_prism(e)[1]))
        L.append(f'<geom name="chamfer_{i}" type="mesh" mesh="chamfer_{i}" rgba="{shell}" {CONTACT} {FRICTION}/>')
    for i, (_, verts) in enumerate(ramps()):
        meshes.append(_mesh(f"ramp_{i}", verts))
        L.append(f'<geom name="ramp_{i}" type="mesh" mesh="ramp_{i}" rgba="{shell}" {CONTACT} {FRICTION}/>')
    L.append(box_geom((-POCKET[0] + .1, POCKET[0] - .1, -Y_IN + .01, Y_IN - .01, 0., .02), rgba="0.62 0.62 0.60 1", extra=VIS))
    for slot in ("A", "B"):
        s = 1. if slot == "A" else -1.
        y0, y1 = sorted((s * (Y_AX - PLATE_W / 2), s * (Y_AX + PLATE_W / 2)))
        L.append(box_geom(tuple(sorted((s * X_PLUS, s * X_IN))) + (y0, y1, PLATE_Z[0], PLATE_Z[1]),
                          name=f"plus_plate_{slot}", rgba="0.85 0.68 0.30 1", extra=f"{CONTACT} {FRICTION}"))
        # polarity marks: a red "+" on the floor near the + end, a dark "-" near the spring end, and on the rim top
        xp, xm, yc = s * (X_PLUS - 9.), -s * (X_PLUS - 9.), s * Y_AX
        L.append(f'<geom type="box" pos="{_f(xp * MM, yc * MM, .05 * MM)}" size="{_f(3 * MM, .8 * MM, .05 * MM)}" rgba="0.85 0.1 0.1 1" {VIS}/>')
        L.append(f'<geom type="box" pos="{_f(xp * MM, yc * MM, .05 * MM)}" size="{_f(.8 * MM, 3 * MM, .05 * MM)}" rgba="0.85 0.1 0.1 1" {VIS}/>')
        L.append(f'<geom type="box" pos="{_f(xm * MM, yc * MM, .05 * MM)}" size="{_f(3 * MM, .8 * MM, .05 * MM)}" rgba="0.1 0.1 0.12 1" {VIS}/>')
        yr, xr = s * (Y_IN + 2.6), s * (X_PLUS - 14.)
        L.append(f'<geom type="box" pos="{_f(xr * MM, yr * MM, (Z_RIM + .05) * MM)}" size="{_f(1.8 * MM, .45 * MM, .05 * MM)}" rgba="0.85 0.1 0.1 1" {VIS}/>')
        L.append(f'<geom type="box" pos="{_f(xr * MM, yr * MM, (Z_RIM + .05) * MM)}" size="{_f(.45 * MM, 1.8 * MM, .05 * MM)}" rgba="0.85 0.1 0.1 1" {VIS}/>')
        L.append(f'<geom type="box" pos="{_f(-xr * MM, yr * MM, (Z_RIM + .05) * MM)}" size="{_f(1.8 * MM, .45 * MM, .05 * MM)}" rgba="0.92 0.92 0.92 1" {VIS}/>')
        # the spring contact at the - end: a plate on a slide joint along the cell axis (joint 0 = the seated face,
        # the spring's rest 2 mm further toward the cell, a hard limit 1.5 mm beyond seated)
        xs_seat = s * (X_PLUS - AA_L)
        bx = xs_seat - s * SPRING_T / 2
        L.append(f'<body name="spring_{slot}" pos="{_f(bx * MM, yc * MM, 7. * MM)}">')
        L.append(f'<joint name="spring_{slot}" type="slide" axis="{_f(s, 0, 0)}" range="{_f(-(SPRING_TRAVEL - SPRING_PRE) * MM, SPRING_PRE * MM)}" '
                 f'solreflimit="0.002 1" stiffness="{SPRING_K * 1000:.6g}" springref="{SPRING_PRE * MM:.6g}" damping="0.5" armature="0.0005"/>')
        L.append(box_geom((-SPRING_T / 2, SPRING_T / 2, -6.5, 6.5, -5., 5.), name=f"spring_plate_{slot}", rgba="0.78 0.78 0.80 1",
                          extra=f'{CONTACT} {FRICTION} mass="0.002"'))
        L.append("</body>")
    # the detent plunger: a horizontal capsule on a vertical slide joint (joint 0 = extended, top 0.6 mm above the ledge)
    zc = Z_RIM + .6 - PLUNGER_R
    L.append(f'<body name="plunger" pos="{_f(PLUNGER_X * MM, PLUNGER_Y * MM, zc * MM)}">')
    L.append(f'<joint name="plunger" type="slide" axis="0 0 1" range="{_f(-1.2 * MM, 0)}" solreflimit="0.002 1" '
             f'stiffness="{PLUNGER_K * 1000:.6g}" springref="{PLUNGER_PRE_N / PLUNGER_K * MM:.6g}" damping="0.3" armature="0.0005"/>')
    L.append(f'<geom name="plunger" type="capsule" fromto="{_f(-2 * MM, 0, 0, 2 * MM, 0, 0)}" size="{PLUNGER_R * MM:.6g}" '
             f'rgba="0.75 0.75 0.78 1" mass="0.001" {CONTACT} {FRICTION}/>')
    L.append("</body>")
    # the controller's outer shape lying face down (visual), and its face as a flat collision box under the body
    col = "0.16 0.16 0.17 1"
    vis = f'{VIS} rgba="{col}"'
    L += [f'<geom type="box" pos="{_f(0, 0, -.017)}" size="{_f(.050, .040, .013)}" {vis}/>',
          f'<geom type="capsule" fromto="{_f(-.048, -.032, -.016, -.048, .034, -.016)}" size="0.0155" {vis}/>',
          f'<geom type="capsule" fromto="{_f(.048, -.032, -.016, .048, .034, -.016)}" size="0.0155" {vis}/>',
          f'<geom type="capsule" fromto="{_f(-.052, -.034, -.012, -.062, -.082, .006)}" size="0.0195" {vis}/>',
          f'<geom type="capsule" fromto="{_f(.052, -.034, -.012, .062, -.082, .006)}" size="0.0195" {vis}/>',
          f'<geom type="box" pos="{_f(0, .044, -.008)}" size="{_f(.050, .005, .008)}" {vis}/>',
          f'<geom name="ctrl_face" type="box" pos="{_f(0, -.01, -.0275)}" size="{_f(.055, .045, .0025)}" rgba="0.16 0.16 0.17 1" group="3" {CONTACT}/>']
    return L


def _props(meshes):
    """The static props, each a top-level body (positions in the mount frame)."""
    L = []
    cx, cy = CTRL_XY_M
    L.append(f'<body name="cradle" pos="{_f(cx, cy - .01, 0)}">')
    L.append(f'<geom name="cradle" type="box" pos="0 0 0.0075" size="{_f(.070, .060, .0075)}" rgba="0.36 0.38 0.42 1" {CONTACT}/>')
    L.append("</body>")
    tx, ty = TRAY_XY_M
    L.append(f'<body name="tray" pos="{_f(tx, ty, 0)}">')
    for b in subtract((-35., 35., -25., 25., 0., 14.), [(-28., 28., -21., -3., TRAY_FLOOR, 20.), (-28., 28., 3., 21., TRAY_FLOOR, 20.)]):
        L.append(box_geom(b, rgba="0.25 0.55 0.35 1", extra=CONTACT))
    L.append("</body>")
    bx, by = BOX_XY_M
    ix, iy, dz = BOX_IN
    L.append(f'<body name="recycle_box" pos="{_f(bx, by, 0)}">')
    for b in subtract((-ix / 2 - 3, ix / 2 + 3, -iy / 2 - 3, iy / 2 + 3, 0., dz + 3), [(-ix / 2, ix / 2, -iy / 2, iy / 2, 3., dz + 10)]):
        L.append(box_geom(b, rgba="0.80 0.45 0.10 1", extra=CONTACT))
    L.append("</body>")
    rx, ry = STAND_XY_M
    hw = SHAFT_R + .4        # a 6.8 x 6.8 mm square hole 40 mm deep: the 6 mm square shaft stands in it within 0.4 mm,
    c = STAND_FUNNEL         # its mouth chamfered 45 deg (a lead-in: the shaft finds the hole from 2.4 mm off)
    edges = [("x", (-hw - c, hw + c), hw, 1., 45., c), ("x", (-hw - c, hw + c), -hw, -1., 45., c),
             ("y", (-hw - c, hw + c), hw, 1., 45., c), ("y", (-hw - c, hw + c), -hw, -1., 45., c)]
    L.append(f'<body name="stick_stand" pos="{_f(rx, ry, 0)}">')
    for b in subtract((-15., 15., -15., 15., 0., 45.), [(-hw, hw, -hw, hw, 5., 50.)] + [chamfer_cut_and_prism(e)[0] for e in edges]):
        L.append(box_geom(b, rgba="0.45 0.45 0.48 1", extra=CONTACT))
    for i, e in enumerate(edges):
        meshes.append(_mesh(f"funnel_{i}", chamfer_cut_and_prism(e)[1]))
        L.append(f'<geom name="funnel_{i}" type="mesh" mesh="funnel_{i}" rgba="0.45 0.45 0.48 1" {CONTACT}/>')
    L.append("</body>")
    px, py = PAD_XY_M
    L.append(f'<body name="cover_pad" pos="{_f(px, py, 0)}">')
    L.append(f'<geom type="box" pos="0 0 0.0002" size="{_f(.045, .035, .0002)}" rgba="0.55 0.6 0.7 1" contype="0" conaffinity="0" group="2" mass="0"/>')
    L.append("</body>")
    return L


# ---------------------------------------------------------------- movable parts
def cell_geoms(old):
    """A cell in its frame: collision cylinder + nub (group 3), coloured visual twins, a printed '+'."""
    body_col = "0.55 0.56 0.58 1" if old else "0.05 0.05 0.06 1"
    band_col = "0.62 0.50 0.32 1" if old else "0.78 0.42 0.16 1"
    x0, x1 = -AA_L / 2, AA_L / 2 - NUB_H
    xb = x1 - PLUS_BAND
    q90 = "0.7071068 0 0.7071068 0"
    return [
        f'<geom name="col" type="cylinder" pos="{_f((x0 + x1) / 2 * MM, 0, 0)}" quat="{q90}" size="{_f(AA_R * MM, (x1 - x0) / 2 * MM)}" '
        f'mass="{AA_MASS_KG * .99:.6g}" rgba="0 0 0 0" group="3" {CONTACT} {FRICTION}/>',
        f'<geom name="nub" type="cylinder" pos="{_f((x1 + NUB_H / 2) * MM, 0, 0)}" quat="{q90}" size="{_f(NUB_D / 2 * MM, NUB_H / 2 * MM)}" '
        f'mass="{AA_MASS_KG * .01:.6g}" rgba="0.8 0.8 0.82 1" {CONTACT} {FRICTION}/>',
        f'<geom type="cylinder" pos="{_f((x0 + xb) / 2 * MM, 0, 0)}" quat="{q90}" size="{_f(AA_R * MM, (xb - x0) / 2 * MM)}" rgba="{body_col}" {VIS}/>',
        f'<geom type="cylinder" pos="{_f((xb + x1) / 2 * MM, 0, 0)}" quat="{q90}" size="{_f(AA_R * MM, (x1 - xb) / 2 * MM)}" rgba="{band_col}" {VIS}/>',
        f'<geom type="cylinder" pos="{_f((x0 - .05) * MM, 0, 0)}" quat="{q90}" size="{_f(3.6 * MM, .1 * MM)}" rgba="0.75 0.75 0.78 1" {VIS}/>',
        f'<geom type="box" pos="{_f((x1 - 8) * MM, 0, (AA_R + .05) * MM)}" size="{_f(2.5 * MM, .5 * MM, .05 * MM)}" rgba="0.1 0.1 0.1 1" {VIS}/>',
        f'<geom type="box" pos="{_f((x1 - 8) * MM, 0, (AA_R + .05) * MM)}" size="{_f(.5 * MM, 2.5 * MM, .05 * MM)}" rgba="0.1 0.1 0.1 1" {VIS}/>']


def cover_geoms():
    """The cover in its frame: plate (with the detent groove), tongue, three hooks, thumb ridge, an embossed arrow."""
    col = "0.13 0.13 0.14 1"
    y0, y1 = -Y_LEDGE + .2, Y_LEDGE - SLIDE - .2        # full-thickness plate (5 mm short at +y: room to slide)
    gw = GROOVE_W / 2
    g = [box_geom(b, rgba=col, extra=f"{CONTACT} {FRICTION}") for b in subtract(
        (-X_LEDGE + .2, X_LEDGE - .2, y0, y1, 0., COVER_T), [(PLUNGER_X - 3.5, PLUNGER_X + 3.5, PLUNGER_Y - gw, PLUNGER_Y + gw, -1., .8)])]
    g.append(box_geom((-X_LEDGE + .2, X_LEDGE - .2, y1 - .01, y1 + SLIDE + 6., 0., 1.), name="tongue", rgba=col, extra=f"{CONTACT} {FRICTION}"))
    for h in HOOK_X:
        g.append(box_geom((h - HOOK_W / 2, h + HOOK_W / 2, y0, y0 + 1.5, -3., .01), rgba=col, extra=CONTACT))
        g.append(box_geom((h - HOOK_W / 2, h + HOOK_W / 2, y0 - 3.5, y0 + 1.5, -3., -2.), rgba=col, extra=CONTACT))
    g.append(box_geom((-10., 10., -10., -7., COVER_T - .01, COVER_T + RIDGE_H), name="ridge", rgba="0.2 0.2 0.21 1", extra=CONTACT))
    g.append(f'<geom type="box" pos="{_f(0, 2 * MM, (COVER_T + .05) * MM)}" size="{_f(.6 * MM, 4 * MM, .05 * MM)}" rgba="0.35 0.35 0.37 1" {VIS}/>')
    g.append(f'<geom type="box" pos="{_f(-1.6 * MM, 5 * MM, (COVER_T + .05) * MM)}" quat="0.9238795 0 0 0.3826834" size="{_f(.5 * MM, 2.6 * MM, .05 * MM)}" rgba="0.35 0.35 0.37 1" {VIS}/>')
    g.append(f'<geom type="box" pos="{_f(1.6 * MM, 5 * MM, (COVER_T + .05) * MM)}" quat="0.9238795 0 0 -0.3826834" size="{_f(.5 * MM, 2.6 * MM, .05 * MM)}" rgba="0.35 0.35 0.37 1" {VIS}/>')
    return g


def stick_geoms():
    """The pry stick in its frame: blade (yellow), foot, square shaft, square handle (blue)."""
    bx, by, bl = BLADE
    fl, fh = FOOT
    yellow, blue = "0.85 0.75 0.15 1", "0.15 0.35 0.75 1"
    return [box_geom((-bx / 2, bx / 2, -by / 2, by / 2, 0., bl), name="blade", rgba=yellow, extra=f"{CONTACT} {FRICTION}"),
            box_geom((-bx / 2 - fl, -bx / 2 + .01, -by / 2, by / 2, 0., fh), name="foot", rgba=yellow, extra=f"{CONTACT} {FRICTION}"),
            box_geom((-SHAFT_R, SHAFT_R, -SHAFT_R, SHAFT_R, bl - 2, bl + SHAFT_L), name="shaft", rgba=yellow, extra=CONTACT),
            box_geom((-HANDLE[0] / 2, HANDLE[0] / 2, -HANDLE[1] / 2, HANDLE[1] / 2, bl + SHAFT_L, bl + SHAFT_L + HANDLE[2]),
                     name="handle", rgba=blue, extra=CONTACT)]


# ---------------------------------------------------------------- poses (mount frame, 4x4)
def _T(p, yaw=0.):
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = p
    return T


def compartment_T():
    """The controller frame C in the mount frame (metres)."""
    return _T((CTRL_XY_M[0], CTRL_XY_M[1], Z_FLOOR_M))


def seated_pose(slot):
    """A seated cell's frame in C (mm, yaw): slot A's + end toward +x, slot B's toward -x."""
    sgn = 1. if slot == "A" else -1.
    return np.array([sgn * (X_PLUS - AA_L / 2), sgn * Y_AX, AA_R]), (0. if slot == "A" else np.pi)


def seated_T(slot):
    c, yaw = seated_pose(slot)
    return compartment_T() @ _T(c * MM, yaw)


def cover_closed_T():
    return compartment_T() @ _T((0., 0., Z_RIM * MM))


def tray_T(k, plus_x):
    tx, ty = TRAY_XY_M
    return _T((tx, ty - .012 + .024 * k, (TRAY_FLOOR + AA_R) * MM), 0. if plus_x else np.pi)


def stick_start_T():
    rx, ry = STAND_XY_M
    return _T((rx, ry, .005), np.pi / 2)


def start_poses(polarity="+-"):
    """Every movable entity's birth pose; polarity: the new cells' + end toward +x ('+') or -x ('-') in the tray."""
    return dict(cover=cover_closed_T(), old_A=seated_T("A"), old_B=seated_T("B"),
                new_1=tray_T(0, polarity[0] == "+"), new_2=tray_T(1, polarity[1] == "+"), stick=stick_start_T())


def _quat(T):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]


def _visual_twins(lines):
    """Every colliding geom outside group 3 moved to group 3 and followed by its visual twin (contype 0, no mass, its
    colour): the renderer (nyx_direct, as Genesis) draws a colliding geom of groups 0-2 grey and only visual geoms in
    their own colour. The physics never sees the twins (the scene builder drops visual geoms)."""
    import xml.etree.ElementTree as ET
    out = []
    for line in lines:
        if not line.startswith("<geom") or 'contype="0"' in line or 'group="3"' in line:
            out.append(line)
            continue
        g = ET.fromstring(line)
        twin = {k: g.get(k) for k in ("type", "mesh", "pos", "quat", "fromto", "size", "rgba") if g.get(k) is not None}
        out.append(line[:-2].rstrip() + ' group="3"/>')
        out.append("<geom " + " ".join(f'{k}="{v}"' for k, v in twin.items()) + f" {VIS}/>")
    return out


def entity_xml(name, polarity="+-") -> str:
    """The MJCF text of one entity at the task's birth (ENTITIES), in the mount frame."""
    meshes, body = [], []
    if name == "controller":
        C = compartment_T()
        body.append(f'<body name="shell" pos="{_f(*C[:3, 3])}" quat="1 0 0 0">')
        body += _compartment(meshes)
        body.append("</body>")
        body += _props(meshes)
    elif name == "cover":
        T = start_poses(polarity)["cover"]
        body.append(f'<body name="cover" pos="{_f(*T[:3, 3])}" quat="{_f(*_quat(T))}"><freejoint name="root"/>')
        body += cover_geoms()
        body.append("</body>")
    elif name in CELLS:
        T = start_poses(polarity)[name]
        body.append(f'<body name="cell" pos="{_f(*T[:3, 3])}" quat="{_f(*_quat(T))}"><freejoint name="root"/>')
        body += cell_geoms(old=name.startswith("old"))
        body.append("</body>")
    elif name == "stick":
        T = start_poses(polarity)["stick"]
        body.append(f'<body name="stick" pos="{_f(*T[:3, 3])}" quat="{_f(*_quat(T))}"><freejoint name="root"/>')
        body.append(f'<inertial pos="0 0 0.11" mass="{STICK_MASS_KG}" diaginertia="1.2e-4 1.2e-4 2e-6"/>')
        body += stick_geoms()
        body.append("</body>")
    else:
        raise ValueError(f"battery_replacement has no entity {name!r}")
    body = _visual_twins(body)
    lines = [f'<mujoco model="{name}">', '<compiler angle="radian" autolimits="true"/>']
    if meshes:
        lines += ["<asset>", *meshes, "</asset>"]
    lines += ["<worldbody>", *body, "</worldbody>", "</mujoco>"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- points (grasp regions, the fall check)
def cell_points():
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    ring = lambda x: np.c_[np.full(len(ang), x * MM), AA_R * MM * np.cos(ang), AA_R * MM * np.sin(ang)]
    return np.vstack([ring(-AA_L / 2), ring(AA_L / 2 - NUB_H), [[AA_L / 2 * MM, 0, 0]]])


def cover_points():
    x, y0, y1 = X_LEDGE - .2, -Y_LEDGE + .2, Y_LEDGE - SLIDE - .2 + SLIDE + 6.
    return np.array([[sx * x, y, z] for sx in (-1, 1) for y in (y0 - 3.5, y1) for z in (-3., COVER_T + RIDGE_H)]) * MM


def ridge_points():
    return np.array([[x, y, z] for x in (-10., 10.) for y in (-10., -7.) for z in (COVER_T, COVER_T + RIDGE_H)]) * MM


def stick_points():
    h0 = BLADE[2] + SHAFT_L
    pts = [[sx * HANDLE[0] / 2, sy * HANDLE[1] / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (h0, h0 + HANDLE[2])]
    pts += [[sx * BLADE[0] / 2, sy * BLADE[1] / 2, 0.] for sx in (-1, 1) for sy in (-1, 1)]
    return np.array(pts) * MM


def handle_points():
    h0 = BLADE[2] + SHAFT_L
    return np.array([[sx * HANDLE[0] / 2, sy * HANDLE[1] / 2, z] for sx in (-1, 1) for sy in (-1, 1)
                     for z in (h0, h0 + HANDLE[2])]) * MM
