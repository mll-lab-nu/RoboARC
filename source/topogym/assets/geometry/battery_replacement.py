"""Game controller battery swap: the MJCF of battery_replacement's entities (coordinates in the ALOHA mount frame).

Physical picture (design: runs/controller_battery_design_20261014, README.md v2 and README_v3.md v3b): an Xbox-style
controller lies face down in a cradle on the table; its battery compartment (two AA slots side by side, a 1.6 mm rib
between them) is closed by a sliding cover (three hooks under the -y edge, a tongue under the +y lip, a spring-loaded
detent plunger). The slots hold two old AA cells, each pushed by its spring contact (a plate on a slide joint at the
- end) against the + contact plate. The cells sit 4 mm below the rim with 1 mm of side clearance, so two fingers cannot
take a seated cell (the old cells are force-closure grasp targets); a pry notch beside each + plate takes the blade of
the pry stick, whose tip carries a small foot that lifts the cell from below when the stick is levered about the notch's
outer edge. Two new cells lie in a tray, the stick stands in a holder behind the controller, a recycling box sits in front.
The controller's outer body is an Xbox Wireless Controller at 1:1 (BODY_PIECES, option C of 2026-10-19): static convex
collision pieces around the compartment's shell (grips, shoulders, the top and bottom edges, bumpers, triggers, the fill
under the shell down to the cradle), each drawn by its visual twin, so a hand or a cell meets what it sees.

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

# ---------------------------------------------------------------- the controller's outer body (controller frame C), mm
# An Xbox Wireless Controller at 1:1 (152.9 x 101.9 x 61 mm, 280 g: dimensions.com "Xbox One Controller"; retailers list the
# Series model at 153 x 102 x 61 mm), face down: its face on the cradle's top (FACE_Z, 0.1 mm above it), the back of its
# centre the shell's back (Z_TOP), the grips toward -y (the cameras), the top edge with the bumpers toward +y, the triggers
# standing up from the back at the top edge. Static collision, each piece a convex hull (_rounded_hull) of rounded discs
# (centre x, centre y, radius, top z) in plan; the user's choice of 2026-10-19 (design xbox_visual/: option C), so that
# what is seen is what collides. Overall x +-76.5, y -55 .. +47, z from the stick caps (-33, hidden in the cradle) to the
# triggers' tops (+27).
FACE_Z = -18.9
BODY_PIECES = {   # name: (discs, top rounding, bottom rounding[, bottom z: FACE_Z]); BODY_SPLIT ones mirrored in x (_L, _R)
    "side": ([(46., -36., 3., Z_TOP), (46., 36., 3., Z_TOP)], 3., 0.),
    "top": ([(-46., 40., 5., Z_TOP), (46., 40., 5., Z_TOP)], 5., 0.),
    "bottom": ([(-40., -40., 3., Z_TOP), (40., -40., 3., Z_TOP), (-32., -44., 4., 4.), (32., -44., 4., 4.)], 3., 3.),
    "shoulder": ([(47., 40., 6., Z_TOP), (62., 34., 9., 19.), (64., 12., 12.5, 18.), (47., 12., 2., Z_TOP)], 5., 4.),
    "grip": ([(64., 12., 12.5, 18.), (60., -20., 14., 16.), (52., -41., 14., 11.), (46., 5., 2., Z_TOP), (46., -30., 2., 19.)], 6., 4.),
    "bumper": ([(22., 42., 4.5, -3.), (50., 40., 6.5, -3.)], 3., 2.),
    "trigger": ([(38., 41., 5., 27.), (49., 41., 5., 25.5), (43., 36., 3., 23.)], 3., 0., 17.),
}
BODY_SPLIT = ("side", "shoulder", "grip", "bumper", "trigger")      # one piece each side
BODY_RGBA = "0.17 0.17 0.18 1"            # Carbon Black, the shell's colour
TRIM_RGBA = "0.11 0.11 0.12 1"            # bumpers and triggers

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


def _rounded_hull(discs, z0, r_top, r_bot, n=20, k=3):
    """Vertices (mm) whose convex hull is a slab over the plan hull of the discs (cx, cy, radius, top z), from z0 up to
    each disc's top, its top edge rounded by r_top and its bottom edge by r_bot (quarter circles in k steps)."""
    out = []
    ang = np.linspace(0., 2 * np.pi, n, endpoint=False)
    for cx, cy, R, zt in discs:
        for i in range(k + 1):
            th = np.pi / 2 * i / k
            for rad, z in ((R - r_top * (1 - np.cos(th)), zt - r_top + r_top * np.sin(th)),
                           (R - r_bot * (1 - np.cos(th)), z0 + r_bot - r_bot * np.sin(th))):
                rad = max(rad, .1)
                out += [(cx + rad * np.cos(a), cy + rad * np.sin(a), z) for a in ang]
    return out


def body_pieces():
    """The outer body's convex pieces: name -> (vertices mm, rgba), and the fill under the shell (a box, mm bounds)."""
    pieces = {}
    for name, (discs, r_top, r_bot, *z0) in BODY_PIECES.items():
        sides = (("L", 1.), ("R", -1.)) if name in BODY_SPLIT else (("", 1.),)
        for tag, sx in sides:
            d = [(sx * cx, cy, R, zt) for cx, cy, R, zt in discs]
            pieces[f"xbox_{name}{'_' + tag if tag else ''}"] = (
                _rounded_hull(d, z0[0] if z0 else FACE_Z, r_top, r_bot), TRIM_RGBA if name in ("bumper", "trigger") else BODY_RGBA)
    return pieces, (-45., 45., -40., 40., FACE_Z, -4.)


def _face_details():
    """The controller's face, visual only (VIS): face down it lies inside the cradle's top (the sticks, the D-pad and the
    buttons stand below FACE_Z). The user's left hand is at +x face down."""
    z, L = FACE_Z * MM, []
    def disc(x, y, r, h, rgba):       # a button: a short cylinder below the face
        return f'<geom type="cylinder" pos="{_f(x * MM, y * MM, z - h / 2 * MM)}" size="{_f(r * MM, h / 2 * MM)}" rgba="{rgba}" {VIS}/>'
    for x, y in ((37., 16.), (-20., -6.)):      # the left stick (upper) and the right stick (lower): shaft, cap
        L.append(disc(x, y, 4., 10.1, "0.12 0.12 0.13 1"))
        L.append(f'<geom type="cylinder" pos="{_f(x * MM, y * MM, z - 12. * MM)}" size="{_f(9.5 * MM, 2. * MM)}" rgba="0.10 0.10 0.11 1" {VIS}/>')
    for sx, sy in ((9.5, 3.), (3., 9.5)):        # the D-pad (lower left: +x face down)
        L.append(f'<geom type="box" pos="{_f(20. * MM, -6. * MM, z - 1. * MM)}" size="{_f(sx * MM, sy * MM, 1. * MM)}" rgba="0.12 0.12 0.13 1" {VIS}/>')
    for dx, dy, rgba in ((0., -9.5, "0.20 0.62 0.25 1"), (-9.5, 0., "0.80 0.16 0.16 1"),       # A, B (face down: B at -x)
                         (9.5, 0., "0.16 0.36 0.80 1"), (0., 9.5, "0.92 0.76 0.12 1")):         # X, Y
        L.append(disc(-37. + dx, 16. + dy, 4.6, 2.5, rgba))
    L.append(disc(0., 28., 7., 1.5, "0.86 0.86 0.86 1"))                                       # the Xbox button
    for x, y in ((10., 11.), (-10., 11.), (0., 2.)):                                           # view, menu, share
        L.append(disc(x, y, 2.6, 1.5, "0.12 0.12 0.13 1"))
    return L


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
        # on the rim beside the slot (x 21 mm off centre: until 2026-10-19 at 13 mm, where slot B's marks lay over the
        # cover's hook slots, 3 mm above their floor)
        yr, xr = s * (Y_IN + 2.6), s * (X_PLUS - 6.)
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
    # the controller's outer body (option C, 2026-10-19): static collision pieces around the shell and the fill under it,
    # each drawn by its visual twin; the face's sticks and buttons (visual, inside the cradle); the headset jack on the
    # bottom edge, facing the cameras
    pieces, fill = body_pieces()
    L.append(box_geom(fill, name="xbox_fill", rgba=BODY_RGBA, extra=f"{CONTACT} {FRICTION}"))
    for name, (verts, rgba) in pieces.items():
        meshes.append(_mesh(name, verts))
        L.append(f'<geom name="{name}" type="mesh" mesh="{name}" rgba="{rgba}" {CONTACT} {FRICTION}/>')
    L += _face_details()
    L.append(f'<geom type="cylinder" pos="{_f(0, -48.05 * MM, -8. * MM)}" quat="0.7071068 0.7071068 0 0" '
             f'size="{_f(1.9 * MM, .1 * MM)}" rgba="0.04 0.04 0.04 1" {VIS}/>')
    return L


# The props' look (2026-10-20, the user's choices from the cam_high realism list; visual only, every colliding geom as before)
PROP_ROUND = 1.0                          # mm: the cradle's edges, drawn rounded inside its box
STICK_ROUND = 1.2                         # mm: the stick's shaft and handle, likewise (at most 0.88 mm off a corner)
CRADLE_RGBA, CRADLE_RIM_RGBA = "0.26 0.29 0.33 1", "0.42 0.45 0.50 1"      # slate, a lighter rim band on its top
TRAY_RGBA = "0.23 0.25 0.28 1"            # dark plastic
BIN_RGBA = "0.20 0.32 0.48 1"             # a blue collection bin, a recycling label on its front
STAND_RGBA = "0.20 0.20 0.22 1"           # the stick's holder: dark grey
PAD_RGBA = "0.27 0.27 0.29 1"             # the cover's pad: a dark grey rubber mat
STICK_RGBA, HANDLE_RGBA, GRIP_RGBA = "0.07 0.07 0.08 1", "0.11 0.11 0.12 1", "0.20 0.20 0.21 1"   # black nylon


def _rounded_box_visual(name, b, r, rgba, meshes, offset=(0., 0., 0.)):
    """A visual (VIS) drawing the box b (mm bounds) with every edge rounded by r inside it: a convex mesh named name."""
    x0, x1, y0, y1, z0, z1 = b
    discs = [(x, y, r, z1) for x in (x0 + r, x1 - r) for y in (y0 + r, y1 - r)]
    meshes.append(_mesh(name, _rounded_hull(discs, z0, r, r)))
    return [f'<geom type="mesh" mesh="{name}" pos="{_f(*offset)}" rgba="{rgba}" {VIS}/>']


def _recycling_label(y_face, z_mid):
    """A recycling label on the recycling box's front face (y_face, mm, facing -y): a white panel 30 x 18 mm and a green
    triangle of three bars (the recycling mark's outline), each 0.1 mm proud of the one behind it."""
    L = [box_geom((-15., 15., y_face - .1, y_face, z_mid - 9., z_mid + 9.), rgba="0.92 0.92 0.90 1", extra=VIS)]
    side, cz = 11., z_mid + .5
    for phi in (0., 120., 240.):         # the bars' midpoints sit side / (2 sqrt 3) from the centre, square to it
        a = np.radians(phi - 90.)
        mx, mz = side / (2 * np.sqrt(3)) * np.cos(a), side / (2 * np.sqrt(3)) * np.sin(a)
        th = -np.radians(phi)            # the bar's long axis (cos phi, 0, sin phi): a turn of -phi about y
        q = (np.cos(th / 2), 0., np.sin(th / 2), 0.)
        L.append(f'<geom type="box" pos="{_f(mx * MM, (y_face - .15) * MM, (cz + mz) * MM)}" quat="{_f(*q)}" '
                 f'size="{_f(side / 2 * MM, .05 * MM, .9 * MM)}" rgba="0.16 0.58 0.28 1" {VIS}/>')
    return L


def _props(meshes):
    """The static props, each a top-level body (positions in the mount frame)."""
    L = []
    cx, cy = CTRL_XY_M
    L.append(f'<body name="cradle" pos="{_f(cx, cy - .01, 0)}">')
    L.append(f'<geom name="cradle" type="box" pos="0 0 0.0075" size="{_f(.070, .060, .0075)}" rgba="0.36 0.38 0.42 1" group="3" {CONTACT}/>')
    L += _rounded_box_visual("cradle_vis", (-70., 70., -60., 60., 0., 15.), PROP_ROUND, CRADLE_RGBA, meshes)
    for b in ((-69., 69., -59., -54.), (-69., 69., 54., 59.), (-69., -64., -54., 54.), (64., 69., -54., 54.)):   # its rim
        L.append(box_geom(b + (15., 15.1), rgba=CRADLE_RIM_RGBA, extra=VIS))
    L.append("</body>")
    tx, ty = TRAY_XY_M
    L.append(f'<body name="tray" pos="{_f(tx, ty, 0)}">')
    for b in subtract((-35., 35., -25., 25., 0., 14.), [(-28., 28., -21., -3., TRAY_FLOOR, 20.), (-28., 28., 3., 21., TRAY_FLOOR, 20.)]):
        L.append(box_geom(b, rgba=TRAY_RGBA, extra=CONTACT))
    L.append("</body>")
    bx, by = BOX_XY_M
    ix, iy, dz = BOX_IN
    L.append(f'<body name="recycle_box" pos="{_f(bx, by, 0)}">')
    for b in subtract((-ix / 2 - 3, ix / 2 + 3, -iy / 2 - 3, iy / 2 + 3, 0., dz + 3), [(-ix / 2, ix / 2, -iy / 2, iy / 2, 3., dz + 10)]):
        L.append(box_geom(b, rgba=BIN_RGBA, extra=CONTACT))
    L += _recycling_label(-iy / 2 - 3., (dz + 3.) / 2)
    L.append("</body>")
    rx, ry = STAND_XY_M
    hw = SHAFT_R + .4        # a 6.8 x 6.8 mm square hole 40 mm deep: the 6 mm square shaft stands in it within 0.4 mm,
    c = STAND_FUNNEL         # its mouth chamfered 45 deg (a lead-in: the shaft finds the hole from 2.4 mm off)
    edges = [("x", (-hw - c, hw + c), hw, 1., 45., c), ("x", (-hw - c, hw + c), -hw, -1., 45., c),
             ("y", (-hw - c, hw + c), hw, 1., 45., c), ("y", (-hw - c, hw + c), -hw, -1., 45., c)]
    L.append(f'<body name="stick_stand" pos="{_f(rx, ry, 0)}">')
    for b in subtract((-15., 15., -15., 15., 0., 45.), [(-hw, hw, -hw, hw, 5., 50.)] + [chamfer_cut_and_prism(e)[0] for e in edges]):
        L.append(box_geom(b, rgba=STAND_RGBA, extra=CONTACT))
    for i, e in enumerate(edges):
        meshes.append(_mesh(f"funnel_{i}", chamfer_cut_and_prism(e)[1]))
        L.append(f'<geom name="funnel_{i}" type="mesh" mesh="funnel_{i}" rgba="{STAND_RGBA}" {CONTACT}/>')
    L.append("</body>")
    px, py = PAD_XY_M
    L.append(f'<body name="cover_pad" pos="{_f(px, py, 0)}">')
    L.append(f'<geom type="box" pos="0 0 0.0002" size="{_f(.045, .035, .0002)}" rgba="{PAD_RGBA}" contype="0" conaffinity="0" group="2" mass="0"/>')
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


def stick_geoms(meshes):
    """The pry stick in its frame, drawn as a black nylon spudger (2026-10-20): the blade and its foot, the square shaft
    and the square handle collide as before (group 3); each is drawn by a visual inside it, the shaft's and the handle's
    edges rounded by STICK_ROUND, five grip bands around the handle 0.15 mm proud."""
    bx, by, bl = BLADE
    fl, fh = FOOT
    yellow, blue = "0.85 0.75 0.15 1", "0.15 0.35 0.75 1"
    blade = (-bx / 2, bx / 2, -by / 2, by / 2, 0., bl)
    foot = (-bx / 2 - fl, -bx / 2 + .01, -by / 2, by / 2, 0., fh)
    shaft = (-SHAFT_R, SHAFT_R, -SHAFT_R, SHAFT_R, bl - 2, bl + SHAFT_L)
    hx, hy, h0 = HANDLE[0] / 2, HANDLE[1] / 2, bl + SHAFT_L
    handle = (-hx, hx, -hy, hy, h0, h0 + HANDLE[2])
    L = [box_geom(blade, name="blade", rgba=yellow, extra=f'group="3" {CONTACT} {FRICTION}'),
         box_geom(foot, name="foot", rgba=yellow, extra=f'group="3" {CONTACT} {FRICTION}'),
         box_geom(shaft, name="shaft", rgba=yellow, extra=f'group="3" {CONTACT}'),
         box_geom(handle, name="handle", rgba=blue, extra=f'group="3" {CONTACT}'),
         box_geom(blade, rgba=STICK_RGBA, extra=VIS), box_geom(foot, rgba=STICK_RGBA, extra=VIS)]
    L += _rounded_box_visual("stick_shaft_vis", shaft, STICK_ROUND, STICK_RGBA, meshes)
    L += _rounded_box_visual("stick_handle_vis", handle, STICK_ROUND, HANDLE_RGBA, meshes)
    band = (-hx - .15, hx + .15, -hy - .15, hy + .15, 0., 1.5)
    meshes.append(_mesh("stick_grip_band", _rounded_hull([(x, y, STICK_ROUND + .15, band[5]) for x in (band[0] + STICK_ROUND + .15,
                        band[1] - STICK_ROUND - .15) for y in (band[2] + STICK_ROUND + .15, band[3] - STICK_ROUND - .15)], 0., .6, .6)))
    for k in range(5):
        L.append(f'<geom type="mesh" mesh="stick_grip_band" pos="{_f(0, 0, (h0 + 14. + 5. * k) * MM)}" rgba="{GRIP_RGBA}" {VIS}/>')
    return L


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


def _repaint(line, look):
    """A visual geom's line with its colour replaced as look ({colour: colour}) says."""
    for a, b in look.items():
        if f'rgba="{a}"' in line:
            return line.replace(f'rgba="{a}"', f'rgba="{b}"')
    return line


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


# The appearance parameters (BatteryReplacementTask defaults controller_finish, tray_finish, box_finish, props_finish; the
# user, 2026-10-20: a seed varies appearance only): each replaces the colours of the visual geoms drawn in the first
# entry's colours (as written above) and no colliding geom's. Controller: the Xbox outer body (BODY_RGBA, which the
# compartment's walls share, as on the real controller, whose battery bay is the body's colour), its bumpers and triggers
# (TRIM_RGBA), the cover, its ridge, its arrow (the compartment's floor, its marks and contacts, the sticks and the
# buttons keep theirs); tray: the tray; box: the recycling bin (its label keeps its colours); props: the cradle and its rim
# band, the stick's holder, the cover's pad and, in the stick entity alone (STICK_FINISHES, the same names), the pry
# stick's blade, shaft, handle and grip bands. Since 2026-10-21 the defaults are the props' realistic look of 2026-10-20
# (the tray dark, the bin navy, the props grey with a black nylon stick); the other finishes differ from it in every part.
# The cells and the red "+" marks keep their colours: the texts name them (the stick's are no longer named).
CONTROLLER_FINISHES = dict(
    black=(BODY_RGBA, TRIM_RGBA, "0.13 0.13 0.14 1", "0.2 0.2 0.21 1", "0.35 0.35 0.37 1"),
    white=("0.86 0.87 0.88 1", "0.80 0.81 0.82 1", "0.84 0.85 0.86 1", "0.76 0.77 0.78 1", "0.68 0.69 0.70 1"),
    red=("0.62 0.08 0.10 1", "0.50 0.06 0.08 1", "0.58 0.07 0.09 1", "0.48 0.06 0.08 1", "0.75 0.32 0.32 1"),
    blue=("0.14 0.28 0.62 1", "0.11 0.23 0.52 1", "0.13 0.26 0.58 1", "0.11 0.22 0.48 1", "0.40 0.50 0.75 1"))
TRAY_FINISHES = dict(dark=(TRAY_RGBA,), white=("0.85 0.85 0.83 1",), grey=("0.60 0.62 0.64 1",))
BOX_FINISHES = dict(navy=(BIN_RGBA,), blue=("0.30 0.55 0.85 1",), green=("0.20 0.50 0.25 1",))
PROPS_FINISHES = dict(      # the cradle, its rim band, the stick's holder, the cover's pad
    grey=(CRADLE_RGBA, CRADLE_RIM_RGBA, STAND_RGBA, PAD_RGBA),
    wood=("0.55 0.40 0.26 1", "0.68 0.52 0.35 1", "0.60 0.45 0.30 1", "0.74 0.62 0.46 1"),       # a cork pad
    black=("0.10 0.10 0.11 1", "0.32 0.32 0.34 1", "0.09 0.09 0.10 1", "0.40 0.40 0.42 1"))      # a grey pad
STICK_FINISHES = dict(      # the stick's blade, foot and shaft, its handle, its grip bands (props_finish's names)
    grey=(STICK_RGBA, HANDLE_RGBA, GRIP_RGBA),                                                    # black nylon
    wood=("0.80 0.62 0.38 1", "0.62 0.44 0.26 1", "0.50 0.34 0.20 1"),                             # orange-wood
    black=("0.14 0.24 0.48 1", "0.10 0.17 0.36 1", "0.22 0.33 0.58 1"))                            # blue nylon


def finish_colours(controller="black", tray="dark", box="navy", props="grey", entity=None) -> dict:
    """{default look colour: its colour in these finishes} (entity_xml's look) for the entity named (None: as for every
    entity but the stick): the stick takes STICK_FINISHES alone (its handle's colour is the bumpers' too), every other
    entity the controller's, tray's, box's and props' tables."""
    tables = ((STICK_FINISHES, props),) if entity == "stick" else (
        (CONTROLLER_FINISHES, controller), (TRAY_FINISHES, tray), (BOX_FINISHES, box), (PROPS_FINISHES, props))
    out = {}
    for table, name in tables:
        out.update(zip(next(iter(table.values())), table[name]))
    return out


def entity_xml(name, polarity="+-", look=None) -> str:
    """The MJCF text of one entity at the task's birth (ENTITIES), in the mount frame; look: finish_colours() (None: the
    look as drawn)."""
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
        body += stick_geoms(meshes)
        body.append("</body>")
    else:
        raise ValueError(f"battery_replacement has no entity {name!r}")
    body = _visual_twins(body)
    if look:                                    # visual geoms only: no colliding geom changes
        body = [_repaint(line, look) if 'contype="0"' in line else line for line in body]
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
