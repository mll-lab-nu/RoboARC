"""Bed-frame corner bolt: the MJCF of bed_bolt's three entities (coordinates in the ALOHA mount frame).

Physical picture (runs/bed_bolt_design_20261012, README sections 7-9, v2 with the cross board): a small bed frame on the
table (60 x 44 cm, four 2.2 cm boards 12 cm high on four round legs, the board bottoms 9 cm above the table, slats on
ledges). In the far corner of the right end board sits a corner block; a blind bore (radius 5.4 mm, 86 mm deep) runs
from the block's face along +x into the end board, its mouth 13.5 cm above the table. A cross board 16 cm in front of
the block face closes a bay around the hole, so the L-shaped hex key fits only with its short arm in the bolt's socket
and its long arm standing up as a lever. The bolt (M10-like: shank 10 mm x 85 mm, head 16 mm x 10 mm, hex socket 8.15 mm
across flats, 6 mm deep, a 1 mm countersink at its mouth) lies in a V cradle in front of the frame; the key (8 mm hex,
long arm 200 mm, short arm 45 mm, both ends 8.0 mm hex prisms with 1.5 mm chamfers) stands short arm down in its stand.

Thread (the task's mechanism, topogym.tasks.bed_bolt.thread): a carriage at the bore mouth, aligned with the bore: two
tilt hinges (pivot at the mouth) -> a slide along the bore -> a spin hinge about it. A joint equality couples the slide to
the spin (slide = lead / (2 pi) x spin), the slide stops where the head meets the block face (BOLT L), and a weld from the
carriage's spin body to the bolt (inactive at birth) carries the bolt once the thread catches.

Entities: bed_frame (static: the frame with its bore, the bolt's cradle and the key's stand), bolt (the bolt, a free
body, and the thread carriage with the lead coupling and the weld) and key (a free body).
Frames: the mount frame is the ALOHA's (x toward the right arm, z up, origin on the table top between the arm bases).
Bolt frame: origin at the tip end of the shank axis, +x toward the head (the head spans x in [L, L + head_h]). Key frame:
origin at the tip of the long arm, +x along the long arm toward the bend, the short arm along +z.
Collision geoms are in group 3 with coloured visual twins (contype 0, no mass), as slotted_chain: the renderer copies a
collision geom of groups 0-2 into a grey visual."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ENTITIES = ("bed_frame", "bolt", "key")

# ---------------------------------------------------------------- dimensions (metres)
BOLT = dict(d=.010, L=.085, head_d=.016, head_h=.010, hex_s=.008, hex_t=.006, lead=.0057, tip=.004)
KEY = dict(s=.008, long=.200, short=.045)
FRAME = dict(L=.60, W=.44, t=.022, zb=.09, h=.12, leg_r=.025, leg_top=.23, ledge=.018, slat_w=.05, slat_t=.012,
             slat_xc=(.25, -.025, -.105, -.185, -.265), blk_d=.07, blk_w=.15, blk_h=.095, hole_cy=.10, hole_cz=.045,
             mb_t=.022)
D_BAY = .16                     # block face to the cross board's near face
FRAME["mb_x"] = FRAME["L"] / 2 - FRAME["t"] - FRAME["blk_d"] - D_BAY
PLACE = (-.04, .13, 0.)         # bed frame centre (x, y) in the mount frame and its yaw
HOLD = dict(bolt_xy=(.15, -.21), key_xy=(-.32, -.20), cradle_h=.022, stand_h=.05, stand_w=.04, key_depth=.035)
THETA0 = np.radians(30.)        # the bolt's spin in its cradle (socket flats at 30 deg)

RB = .0054                      # bore radius (shank radius 5.0 mm)
WALL = .004
DEPTH = .086                    # blind bore depth from the mouth
SOCKET_AF = .00815              # socket across flats (key 8.0 mm)
SOCKET_CS = .001                # 45 deg countersink at the socket mouth (as ISO 4762 sockets have)
KEY_CHAMFER, KEY_TIP_AF = .0015, .0074

# The drives (user 2026-10-18; runs/robotangle_mujoco/bed_bolt/20261018_variants/DESIGN.md): the recess in the bolt's head
# and the matching tip on both ends of the L-shaped tool. Everything else is the same bolt and the same tool: the 8 mm hex
# bars, their mass and grasp region. "hex" is bed_bolt_crossbar's, built by the code below as it was (_bolt, _key); the
# others by their own functions. Each tip is 0.075 mm smaller than its recess on every side, as the hex key in its socket.
#   period_deg  the recess's symmetry: it matches itself turned by this (the lever's strokes)
#   depth, cs   the recess's depth below the head face and its mouth's 45 deg chamfer
#   in_socket   key_end_in_socket: a key end's tip centre this deep below the head face (m, [min, max]) and within this
#               of the bolt axis (hex: the values of the check thread.py keeps written out as it was)
#   tip_len     the tip's length at each end of the tool (the hex bar ends there), its end chamfer: 0.3 mm smaller over
#               tip_chamfer, as the hex key's 1.5 mm chamfer (8.0 -> 7.4 mm)
# torx: a six-lobe (hexalobular, ISO 10664) recess of about T50's size (an M10 Torx socket head screw's): lobe tips 8.95
# mm across, inner diameter 6.4 mm. The recess is a ring of 12 blocks whose flats are A / 2 from the axis (the outer
# lobes' reach) with 6 round bumps between the lobes (radius bump_r, centred bump_c from the axis, reaching in to B / 2);
# the tip is a core cylinder (tip_B / 2) with 6 round lobes (radius lobe_r, centred lobe_c, reaching out to tip_A / 2).
# The tip's lobes sit where the hex key's corners are and the recess's bumps where the hex socket's flats are, so the
# tip fits the recess at every angle the hex key fits the hex socket.
DRIVES = dict(
    hex=dict(period_deg=60., in_socket=dict(depth=(-.0005, .0075), radial=.0025)),
    torx=dict(period_deg=60., in_socket=dict(depth=(-.0005, .0075), radial=.0025), depth=.006, cs=.001,
              A=.00895, B=.0064, bump_r=.0012, bump_c=.0044, tip_A=.0088, tip_B=.00625, lobe_r=.00095, lobe_c=.00345,
              tip_len=.008, tip_chamfer=.0015, n_round=16),
    # square (Robertson-like): a 6.50 mm square recess, the tip a 1/4 in (6.35 mm) square on the bit of the ratcheting
    # driver (RATCHET; the user 2026-10-18: the square, Phillips and slotted drives use one); the recess's walls are the
    # head's disk cut by each side's line (4 convex pieces, the round head kept), turned by `rotate` so that the bit's
    # flats meet them at the reference's first insertion (its tip flats at the bit's 0 / 90 deg: the hex key's flats there
    # meet the bolt's -120 deg, -120 = 60 mod 90)
    square=dict(period_deg=90., ratchet=True, in_socket=dict(depth=(-.0005, .0075), radial=.002), depth=.006, cs=.001,
                AF=.0065, tip_AF=.00635, tip_len=.008, tip_chamfer=.0015, rotate_deg=60., n_arc=24),
    # phillips: a cross recess of about PH4's size: four slots w wide whose ends slope in from R_face at the face to
    # R_bottom at the recess's depth (the cross's cone), their edges bevelled cs at the face; the tip four wings tip_w
    # thick (0.075 mm under the slot on each side) whose ends follow the slope 0.075 mm inside it with the tip's end at
    # tip_depth (pushed in, the wings' ends seat on the slots' ends), the wings thinning to tip_end_w at the very end.
    # Turned like the square (the wings at the bit's 0 / 90 deg). The driving faces are parallel to the axis (the real
    # cross's are tapered and push the tip out under torque; the user removed that cam-out from the family, 2026-10-18).
    phillips=dict(period_deg=90., ratchet=True, in_socket=dict(depth=(-.0005, .0075), radial=.002), depth=.006, cs=.0006,
                  w=.0022, R_face=.0046, R_bottom=.0024, tip_w=.00205, tip_end_w=.0012, tip_depth=.0055, tip_len=.008,
                  tip_chamfer=.0015, rotate_deg=60.),
    # slotted: a straight slot w wide and depth deep across the whole head (open at both ends: a blade can slide out
    # along it), its edges bevelled cs; the tip a parallel-ground blade tip_t thick and tip_W wide, thinning to tip_end_t
    # over tip_chamfer. The slot runs along rotate_deg (the blade's width along the bit's 0 deg). in_socket: the blade's
    # end centre within `across` of the slot's mid-plane and `along` of the axis. Below 2 turns the open slot gives the
    # loose bolt no support along it, and the blade walked off the head (runs/robotangle_mujoco/bed_bolt/
    # 20261018_variants/slotted/dev/RUNS.md); the user's choices (2026-10-18/19):
    #   cradle_deg  the bolt's spin in its cradle, 40 deg less than THETA0's 30: its thread lines up with the hole's near the
    #               end of the first thread (about 7.1 mm in, 1.25 turns, 10 deg inside the catch's 30 deg tolerance)
    #               instead of at its start (2.5 mm, 0.44 turns), which shortens that stretch; the slot is turned in the
    #               head (rotate_deg 20 against the square's 60) so that, caught there, it lies where the reference's first
    #               insertion meets it (150 deg in the mouth frame's y-z plane);
    #   tip_W       the blade 15 mm wide, nearly the head's 16 mm, as a real M10 slotted driver's: the few millimetres it
    #               walks along the open slot in a stroke leave it on the head; the key's stand has a hole 2 stand_hole_y
    #               long along the blade (its width along y at birth) for it;
    #   depth       7.5 mm, the smallest that held the blade in (4.5 and 6.5 mm did not), 2.5 mm of the 10 mm head below it.
    slotted=dict(period_deg=180., ratchet=True, in_socket=dict(depth=(-.0005, .009), across=.001, along=.0075), depth=.0075,
                 cs=.0005, w=.0025, tip_t=.00235, tip_W=.015, tip_end_t=.0016, tip_len=.008, tip_chamfer=.001,
                 rotate_deg=20., cradle_deg=-10., stand_hole_y=.0085),
)
# The ratcheting L-shaped offset driver of the square, Phillips and slotted drives (user 2026-10-18): the L's long arm and a
# ratchet housing at the bend are one body ("key"); the short arm, the bit with the drive's tip, is a second body
# ("key_bit") on a hinge about the short arm's axis ("key_ratchet", +z of the key frame: from the bend toward the tip, the
# tightening direction's axis once the tip is in the recess). The task mechanism lets the hinge turn one way only: the bit
# may turn forward against the L (the L swung back while the bit stays in the recess), never back (the L swung forward
# carries the bit and the bolt): the hinge's lower limit follows its angle up (thread.BoltThread). drag_nm is the
# free-wheel's drag (the pawl over the teeth), small but not zero; the housing (radius, extent along the short arm from
# the bend, away from the tip so it clears the key's stand at birth) holds the ratchet; the bit's bar starts below it.
RATCHET = dict(drag_nm=.03, house_r=.007, house_z=(-.012, .002), switch=(.004, .003, .0015))

# ---------------------------------------------------------------- thread (the mechanism's parameters)
THREAD = dict(
    hold_turns=2.,              # engaged turns the bolt needs before the thread carries it (user 2026-10-12)
    fall_tilt_deg=3.,           # below hold_turns, an unsupported bolt drooping past this lets go
    catch_depth_m=.0025,        # the thread catches with the tip this deep in the bore ...
    catch_depth_max_m=.0082,    # ... and at most this deep: the first thread, one lead past catch_depth_m (user 2026-10-17;
                                # the design's 20 mm let a push buy up to 2.9 of the 14.8 turns)
    catch_radial_m=.001,        # ... its tip this close to the bore axis ...
    catch_angle_deg=2.,         # ... and its axis this close to the bore axis
    recatch_wait_s=.5,          # after letting go, no catch for this long
    # the hole is threaded (user 2026-10-17): the thread catches only lined up with the hole's thread, the bolt's roll about
    # the bore axis (its y axis, from the mouth frame's y toward z) within phase_tol_deg of phase_rad + 2 pi depth / lead;
    # phase_rad is the -i01 reference's own at its catch (row 489: roll - spin, constant while caught), so it catches on
    # the same tick
    phase_rad=.90238,
    phase_tol_deg=30.,
    stop_margin_m=.0003,        # a loose bolt's tip stops this far past the deepest it may catch (the stop plate's face)
    backed_out_m=.0015,         # caught, a slide below this lets go (the bolt backed out)
    rate_window_s=.025,         # the turning rate is the turn over this window
    k_weak=2e-4,                # tilt hinge stiffness (N m / rad), constant
    tilt_ease_deg_s=2.,         # from hold_turns the tilt is held (thread_tilt_y / _z) and its target eases to 0 at this
    tilt_gap_deg=.2,            # ... rate, while the actual tilt is within this of the target
    relock_rad_s=.05,           # key in the socket: a broken-away spin locks again below this rate (over rate_window_s)
)
# The thread's friction on the spin hinge (N m). The user (2026-10-16): pushing the bolt in must meet a large resistance,
# as a real thread locks against an axial push, and fingers turning the bolt directly must slip. 1 N m takes about
# 1.1 kN along the axis to back-drive the 5.7 mm lead (the design's 0.03 N m: 33 N, a 40 N push screwed the bolt home
# in 1.8 s) and exceeds a two-finger pinch on the head (2 x 35 N, mu 1, 8 mm: about 0.56 N m); the key at 17 cm turns it
# with about 6 N at the hand. A friction row creeps below its bound in proportion to the load (here about 9e-5 turn per
# second per newton of push: 60 N, 1.04 turns in 180 s; the key left in the socket turned the bolt 3.9 deg in 5 s by its
# own weight, 0.035 N m), so the mechanism holds the spin with a lock (thread_lock) instead: always while no key end is in
# the socket, and with a key end in it as static friction, broken away above SPIN_FRICTION (then the friction row acts)
# and locked again once the spin slows below relock_rad_s.
SPIN_FRICTION = 1.

STEEL = 7850.
SOLREF_HARD = (.003, 1.)
# The thread's friction holds (topogym): MuJoCo's friction rows are soft and creep under a load below their limit; on the
# exact constraint diagonal (MujocoWorld.build) at the default impedance (.9 .95) and 20 ms the spin hinge crept about
# 1 rad/s under a 20 N push on the bolt's head (an estimate from the row's regulariser; topogym's first reference runs:
# the bolt drove itself in during failed push-ins and ran 2.6 turns ahead of the plan). At 2 dt and impedance .99 the
# creep is a thousandth of that.
SOLIMP_HARD = (.99, .999, .001, .5, 2.)
# The key in the socket drives the thread's 1 N m: with the steel parts' 3 ms contacts at MuJoCo's default impedance the
# key's hex corners sank 0.6 mm into the socket's flats (more than the 0.55 mm a corner stands proud of a flat) and the
# key turned through the socket without turning the bolt (topogym, 2026-10-16). The key and the socket's blocks meet at
# 2 dt, overdamped, and near-rigid impedance, as the families' rigid contacts are timed.
SOCKET_CONTACT = dict(solref=(.00125, 2.), solimp=SOLIMP_HARD)
# The thread's equalities (lead, carriage weld, spin lock, tilt holds) hold an axial push on the bolt: they act on the
# almost massless carriage, so on the exact diagonal their regulariser is large, and at SOLIMP_HARD a 60 N push moved the
# bolt 0.7 mm (0.12 turns of lead; 200 N: 1.2 mm) without the spin turning (topogym, 2026-10-16). At .9999 a 200 N push
# or pull moves it 30 um, steady (no creep).
SOLIMP_RIGID = (.9999, .9999, .001, .5, 2.)
MU_STEEL = (.1, .005, .0001)    # key and socket: oiled steel on steel

C_WOOD, C_WOOD2, C_BLOCK = (.80, .62, .42, 1), (.86, .70, .52, 1), (.74, .55, .36, 1)
C_STEEL, C_KEY, C_HOLE, C_HOLDER = (.55, .57, .60, 1), (.22, .23, .25, 1), (.12, .10, .08, 1), (.35, .45, .60, 1)
C_THREAD_HOLE = (.42, .31, .20, 1)                   # the hole's thread: the corner block's wood, darker
# the visual threads (user 2026-10-17; visual only, the contacts are the plain shank and bore): a single-start right-hand
# thread of the lead's pitch, the bolt's crest at the shank's radius, the hole's root at the bore's
THREAD_LOOK = dict(bolt_root=.0043, hole_crest=.0044, hole_outer=.0056, n_around=48, n_per_pitch=16)


def _v(x) -> str:
    return " ".join(format(float(a), ".17g") for a in np.asarray(x, dtype=float).reshape(-1))


def _quat(R) -> str:
    q = Rotation.from_matrix(np.asarray(R, dtype=float)).as_quat()
    return _v([q[3], q[0], q[1], q[2]])


def place_T(place=PLACE) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", place[2]).as_matrix()
    T[:3, 3] = (place[0], place[1], 0.)
    return T


def hole_frame_point(f=FRAME):
    """The bore mouth's centre in the bed frame's own coordinates and the bore axis (into the bore)."""
    return (np.array([f["L"] / 2 - f["t"] - f["blk_d"], f["W"] / 2 - f["t"] - f["hole_cy"], f["zb"] + f["hole_cz"]]),
            np.array([1., 0., 0.]))


def hole_T(place=PLACE) -> np.ndarray:
    """The bore mouth frame in the mount frame: x into the bore, z as near to up as possible."""
    p, a = hole_frame_point()
    T = place_T(place)
    x = T[:3, :3] @ a
    z = np.array([0., 0., 1.]) - x[2] * x
    z /= np.linalg.norm(z)
    H = np.eye(4)
    H[:3, :3] = np.c_[x, np.cross(z, x), z]
    H[:3, 3] = T[:3, :3] @ p + T[:3, 3]
    return H


def bolt_axis_z(h=HOLD) -> float:
    """The bolt axis' height resting in its cradle."""
    return h["cradle_h"] + BOLT["d"] / 2 - .002


def bolt_start_T(h=HOLD, theta0=THETA0) -> np.ndarray:
    """The bolt in its cradle along +x (tip toward +x), turned theta0 about its axis, 0.3 mm above its rest."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([theta0, 0, 0]).as_matrix() @ Rotation.from_euler("z", np.pi).as_matrix()
    x, y = h["bolt_xy"]
    T[:3, 3] = (x + BOLT["L"] / 2, y, bolt_axis_z(h) + .0003)
    return T


def key_start_T(h=HOLD, k=KEY) -> np.ndarray:
    """The key with its short arm down in the stand, the long arm level toward +x."""
    T = np.eye(4)
    T[:3, :3] = np.c_[[-1., 0, 0], [0, 1., 0], [0, 0, -1.]]
    x, y = h["key_xy"]
    bend = np.array([x, y, h["stand_h"] - h["key_depth"]]) + np.array([0, 0, k["short"] - k["s"] / 2])
    T[:3, 3] = bend + np.array([k["long"] - k["s"] / 2, 0, 0])
    return T


def key_end_points(k=KEY) -> dict:
    """The two ends' tip centres in the key frame: the long arm's (the origin) and the short arm's."""
    return dict(long=np.zeros(3), short=np.array([k["long"] - k["s"] / 2, 0, k["short"] - k["s"] / 2]))


def frame_boxes(f=FRAME):
    """The bed frame's static pieces (name, kind, centre, half sizes or (r, half height), rgba) in its own coordinates,
    without the corner block and the right end board (built around the bore by scene_xml)."""
    L, W, t, zb, h = f["L"], f["W"], f["t"], f["zb"], f["h"]
    out, zc = [], zb + h / 2
    for name, sy in (("rail_near", -1), ("rail_far", 1)):
        out.append((name, "box", (0, sy * (W / 2 - t / 2), zc), (L / 2 - f["leg_r"], t / 2, h / 2), C_WOOD))
    out.append(("end_left", "box", (-(L / 2 - t / 2), 0, zc), (t / 2, W / 2 - f["leg_r"], h / 2), C_WOOD))
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append((f"leg_{'LR'[sx > 0]}{'NF'[sy > 0]}", "cylinder",
                        (sx * (L / 2 - f["leg_r"]), sy * (W / 2 - f["leg_r"]), f["leg_top"] / 2),
                        (f["leg_r"], f["leg_top"] / 2), C_WOOD2))
    lz = zb + h - .022 - f["ledge"] / 2
    for name, sy in (("ledge_near", -1), ("ledge_far", 1)):
        out.append((name, "box", (0, sy * (W / 2 - t - f["ledge"] / 2), lz), (L / 2 - t, f["ledge"] / 2, f["ledge"] / 2),
                    C_WOOD2))
    sz = zb + h - .022 + f["slat_t"] / 2
    for i, x in enumerate(f["slat_xc"]):
        out.append((f"slat_{i}", "box", (x, 0, sz), (f["slat_w"] / 2, W / 2 - t, f["slat_t"] / 2), C_WOOD2))
    top = zb + h - .022
    out.append(("middle_board", "box", (f["mb_x"] - f["mb_t"] / 2, 0, (zb + top) / 2),
                (f["mb_t"] / 2, W / 2 - t, (top - zb) / 2), C_WOOD))
    return out


class _Body:
    """An MJCF body with collision geoms in group 3, each with a coloured visual twin."""

    def __init__(self, parent, name, pos=(0, 0, 0), R=None):
        self.el = ET.SubElement(parent, "body", name=name, pos=_v(pos), quat=_quat(np.eye(3) if R is None else R))

    def geom(self, name, kind, pos, rgba, R=None, visual=True, **attrs):
        """kind box: attrs size (half sizes); cylinder: size (r, half length) along local z; sphere; mesh."""
        a = dict(type=kind, pos=_v(pos), **{k: (v if isinstance(v, str) else _v(v)) for k, v in attrs.items()})
        if R is not None:
            a["quat"] = _quat(R)
        ET.SubElement(self.el, "geom", name=name, group="3", rgba=_v(rgba), **a)
        if visual:
            twin = {k: v for k, v in a.items() if k in ("type", "pos", "quat", "size", "mesh")}
            ET.SubElement(self.el, "geom", name=name + "_look", contype="0", conaffinity="0", mass="0", group="1",
                          rgba=_v(rgba), **twin)

    def look(self, name, kind, pos, rgba, R=None, **attrs):
        """A visual-only geom."""
        a = dict(type=kind, pos=_v(pos), **{k: (v if isinstance(v, str) else _v(v)) for k, v in attrs.items()})
        if R is not None:
            a["quat"] = _quat(R)
        ET.SubElement(self.el, "geom", name=name, contype="0", conaffinity="0", mass="0", group="1", rgba=_v(rgba), **a)


def _bed_frame(world, asset, place=PLACE, f=FRAME):
    """The static bed frame with the corner block and the right end board built around a square channel that holds the
    bore tube (16 wall boxes of radius RB, DEPTH long, closed at the far end by the board)."""
    T = place_T(place)
    body = _Body(world, "bedframe", T[:3, 3], T[:3, :3])
    for name, kind, c, s, rgba in frame_boxes(f):
        body.geom(name, kind, c, rgba, size=s)
    p, _ = hole_frame_point(f)
    L, W, t, zb, h = f["L"], f["W"], f["t"], f["zb"], f["h"]
    hc = RB + WALL                                   # half side of the square channel
    y0, y1 = W / 2 - t - f["blk_w"], W / 2 - t       # the block's y range
    x0, x1 = p[0], L / 2 - t                         # the block's x range (face .. end board)
    cy, cz = p[1], p[2]

    def slab(name, xs, ys, zs, rgba):
        body.geom(name, "box", ((xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2, (zs[0] + zs[1]) / 2), rgba,
                  size=((xs[1] - xs[0]) / 2, (ys[1] - ys[0]) / 2, (zs[1] - zs[0]) / 2))
    slab("block_lo", (x0, x1), (y0, y1), (zb, cz - hc), C_BLOCK)
    slab("block_hi", (x0, x1), (y0, y1), (cz + hc, zb + f["blk_h"]), C_BLOCK)
    slab("block_n", (x0, x1), (y0, cy - hc), (cz - hc, cz + hc), C_BLOCK)
    slab("block_f", (x0, x1), (cy + hc, y1), (cz - hc, cz + hc), C_BLOCK)
    xe0, xe1, ye, xb = x1, L / 2, W / 2 - f["leg_r"], p[0] + DEPTH
    slab("end_r_n", (xe0, xe1), (-ye, cy - hc), (zb, zb + h), C_WOOD)
    slab("end_r_f", (xe0, xe1), (cy + hc, ye), (zb, zb + h), C_WOOD)
    slab("end_r_lo", (xe0, xe1), (cy - hc, cy + hc), (zb, cz - hc), C_WOOD)
    slab("end_r_hi", (xe0, xe1), (cy - hc, cy + hc), (cz + hc, zb + h), C_WOOD)
    slab("end_r_cap", (xb, xe1), (cy - hc, cy + hc), (cz - hc, cz + hc), C_WOOD)
    n = 16
    for i in range(n):
        th = 2 * np.pi * i / n
        body.geom(f"bore_{i}", "box", (p[0] + DEPTH / 2, cy + (RB + WALL / 2) * np.cos(th), cz + (RB + WALL / 2) * np.sin(th)),
                  C_BLOCK, R=Rotation.from_euler("x", th).as_matrix(),
                  size=(DEPTH / 2, WALL / 2, (RB + WALL) * np.tan(np.pi / n) + .0003), solref=SOLREF_HARD)
    # the hole's thread, drawn (the contacts are the plain bore walls and the thread mechanism): its crest's middle half a
    # turn from the bolt's root when lined up (THREAD phase_rad), so the two interleave once caught; the dark mark at the
    # bore's bottom (it closed the mouth before the hole was drawn threaded)
    _thread_mesh(asset, "hole_thread_m", 0., DEPTH, THREAD["phase_rad"] + np.pi, THREAD_LOOK["hole_crest"], RB,
                 r_outer=THREAD_LOOK["hole_outer"])
    body.look("hole_thread", "mesh", p, C_THREAD_HOLE, mesh="hole_thread_m")
    body.look("hole_mark", "cylinder", p + np.array([DEPTH - .0004, 0, 0]), C_HOLE,
              R=Rotation.from_euler("y", np.pi / 2).as_matrix(), size=(RB + .0004, .0004))


def _holders(world, h=HOLD, hole_y=.005):
    """The bolt's V cradle and the key's stand (a 10 mm square hole the short arm sits in, 2 hole_y long along y) with a
    rest under the key's long arm."""
    body = _Body(world, "holders")
    x, y = h["bolt_xy"]
    za, r, s45 = bolt_axis_z(h), BOLT["d"] / 2 + .002, np.sqrt(.5)
    for dx in (-.020, .025):                         # two V grooves under the shank (the head stays clear)
        for sgn in (-1, 1):                          # a wall on the +y (sgn +1) or -y side, 45 deg from vertical
            nrm = np.array([0, -sgn * s45, s45])     # the wall normal, toward the bolt axis
            R = Rotation.from_euler("x", np.arctan2(nrm[2], nrm[1])).as_matrix()
            body.geom(f"cradle_{int(round(dx * 1e3)):+d}_{sgn:+d}", "box", (x + dx, y - nrm[1] * r, za - nrm[2] * r),
                      C_HOLDER, R=R, size=(.010, .002, .012))
    x, y = h["key_xy"]
    w, hw, H, dep = h["stand_w"] / 2, .005, h["stand_h"], h["key_depth"]
    hy = hole_y
    body.geom("stand_base", "box", (x, y, (H - dep) / 2), C_HOLDER, size=(w, w, (H - dep) / 2))
    zc = H - dep / 2
    body.geom("stand_n", "box", (x, y - (w + hy) / 2, zc), C_HOLDER, size=(w, (w - hy) / 2, dep / 2))
    body.geom("stand_f", "box", (x, y + (w + hy) / 2, zc), C_HOLDER, size=(w, (w - hy) / 2, dep / 2))
    body.geom("stand_l", "box", (x - (w + hw) / 2, y, zc), C_HOLDER, size=((w - hw) / 2, hy, dep / 2))
    body.geom("stand_r", "box", (x + (w + hw) / 2, y, zc), C_HOLDER, size=((w - hw) / 2, hy, dep / 2))
    zt = key_start_T(h)[2, 3] - KEY["s"] / np.sqrt(3)   # under the long arm's lower hex corner: no overlap at birth
    body.geom("key_rest", "box", (x + .10, y, zt / 2), C_HOLDER, size=(.008, .012, zt / 2))


# The carriage's range limits as hard as MuJoCo allows (2 dt, overdamped, nearly rigid): a key turned against the seated
# head presses the slide's limit through the lead with kN (MuJoCo's default 20 ms limit let it sink 1.3 mm past the seat
# in topogym's first reference run, storing the energy a soft limit gives back)
HARD_LIMIT = dict(solreflimit="0.00125 2", solimplimit="0.99 0.999 0.001 0.5 2")


def _thread(world, Th):
    """The thread carriage at the bore mouth (its frames aligned with the mouth frame)."""
    c = THREAD
    tilt = ET.SubElement(world, "body", name="thr_tilt", pos=_v(Th[:3, 3]), quat=_quat(Th[:3, :3]))
    # tilt armature about the bolt's own inertia about the mouth (2e-4 kg m^2): with an almost massless carriage the weld
    # to the bolt was soft, and the key in the socket bent the bolt 1-2 deg off the carriage axis (design README 8)
    for name, axis in (("thr_ty", "0 1 0"), ("thr_tz", "0 0 1")):
        ET.SubElement(tilt, "joint", name=name, type="hinge", axis=axis, range="-0.2 0.2", damping="0.003",
                      stiffness=_v(c["k_weak"]), armature="0.0002", **HARD_LIMIT)
    point = dict(pos="0 0 0", mass="0.001", diaginertia="4e-10 4e-10 4e-10")
    ET.SubElement(tilt, "inertial", **point)
    slide = ET.SubElement(tilt, "body", name="thr_slide")
    ET.SubElement(slide, "joint", name="thr_s", type="slide", axis="1 0 0", range=_v([-.01, BOLT["L"]]), damping="0.5",
                  armature="0.0001", **HARD_LIMIT)
    ET.SubElement(slide, "inertial", **point)
    spin = ET.SubElement(slide, "body", name="thr_spin")
    ET.SubElement(spin, "joint", name="thr_h", type="hinge", axis="1 0 0", damping="0.0001",
                  frictionloss=_v(SPIN_FRICTION), armature="0.00001", solreffriction="0.00125 1",
                  solimpfriction=_v(SOLIMP_HARD))
    ET.SubElement(spin, "inertial", **point)
    # the thread's start for a loose bolt (user 2026-10-17: a bolt not turned in cannot be pushed past the first thread):
    # a collision-only plate across the bore that the tip's end face meets, a body of its own after the carriage (every
    # earlier body and geom keeps its index). The mechanism places it at the deepest the bolt may catch (stop_x) and
    # switches it off while the thread is caught.
    stop = _Body(world, "thr_stop", Th[:3, 3], Th[:3, :3])
    # as hard as the socket, winning its contacts (the profile's static_contact.keep for the bolt entity's one static
    # body): at the static bodies' soft class a 200 N push on a loose bolt tilted 3 deg pressed the tip 3-5 mm in and,
    # into a 1 mm plate, popped it (1.6 J created); 10 mm thick, so no push reaches its far side
    stop.geom("thr_stop_plate", "box", (stop_x(0.), 0, 0), C_HOLE, visual=False,
              size=(STOP_T / 2, RB + WALL / 2, RB + WALL / 2), **SOCKET_CONTACT)


STOP_T = .01                    # the stop plate's thickness


def stop_x(s_max: float, c=THREAD) -> float:
    """The stop plate's centre along the bore (from the mouth): its face stop_margin_m past the deepest a loose bolt's tip
    may go, the first thread (catch_depth_max_m) or, once the thread has caught, the deepest it has been turned in."""
    return max(c["catch_depth_max_m"], s_max) + c["stop_margin_m"] + STOP_T / 2


def _profile(u, r_crest, r_root, crest=.25, root=.25):
    """A thread's radius at u, the fraction of a turn from a crest's middle (0 and 1: crest; .5: root): flat crest and
    root, straight flanks between."""
    v = 2 * np.minimum(u % 1., 1. - u % 1.)          # 0 at the crest's middle, 1 at the root's
    f = np.clip((v - crest) / (1. - crest - root), 0., 1.)
    return r_crest + (r_root - r_crest) * f


def _thread_mesh(asset, name, x0, x1, phase, r_crest, r_root, r_outer=None):
    """A closed mesh of a thread along local x from x0 to x1, the crest at the angle (about +x, from +y toward +z)
    phase + 2 pi x / lead (right-handed). External (r_outer None): the threaded shank, capped at both ends. Internal: the
    ring between the threaded bore (r_crest inside, r_root out) and a plain cylinder of r_outer, faces pointing out of
    the solid. Visual only (the MJCF compiler needs it closed for its volume)."""
    L = THREAD_LOOK
    lead, na = BOLT["lead"], L["n_around"]
    nx = max(2, int(np.ceil((x1 - x0) / lead * L["n_per_pitch"])))
    xs, th = np.linspace(x0, x1, nx + 1), 2 * np.pi * np.arange(na) / na
    X, TH = np.meshgrid(xs, th, indexing="ij")
    R = _profile((TH - phase - 2 * np.pi * X / lead) / (2 * np.pi), r_crest, r_root)
    V = [np.c_[X.ravel(), (R * np.cos(TH)).ravel(), (R * np.sin(TH)).ravel()]]
    idx = lambda i, j: i * na + j % na                # noqa: E731
    F = []
    out = 1 if r_outer is None else -1                # the threaded surface faces out (shank) or in (bore)
    for i in range(nx):
        for j in range(na):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)
            F += [(a, b, c), (b, d, c)] if out > 0 else [(a, c, b), (b, c, d)]
    n = (nx + 1) * na
    if r_outer is None:                               # end caps: fans round a centre point at each end
        V.append(np.array([[x0, 0, 0], [x1, 0, 0]]))
        c0, c1 = n, n + 1
        for j in range(na):
            F += [(c0, idx(0, j + 1), idx(0, j)), (c1, idx(nx, j), idx(nx, j + 1))]
    else:                                             # the outer cylinder (one ring at each end) and the end rings
        V.append(np.c_[np.repeat([x0, x1], na), np.tile(r_outer * np.cos(th), 2), np.tile(r_outer * np.sin(th), 2)])
        o = lambda k, j: n + k * na + j % na          # noqa: E731
        for j in range(na):
            F += [(o(0, j), o(0, j + 1), o(1, j)), (o(0, j + 1), o(1, j + 1), o(1, j))]          # outer: faces out
            F += [(idx(0, j), idx(0, j + 1), o(0, j)), (o(0, j), idx(0, j + 1), o(0, j + 1))]    # x0 ring: faces -x
            F += [(idx(nx, j), o(1, j), idx(nx, j + 1)), (o(1, j), o(1, j + 1), idx(nx, j + 1))]  # x1 ring: faces +x
    ET.SubElement(asset, "mesh", name=name, vertex=_v(np.vstack(V)), face=" ".join(str(int(k)) for f in F for k in f))


def _hex_prism(asset, name, x0, x1, af):
    rc = af / np.sqrt(3)
    ang = np.radians(np.arange(6) * 60 + 30)
    ET.SubElement(asset, "mesh", name=name,
                  vertex=_v([[x, rc * np.cos(a), rc * np.sin(a)] for x in (x0, x1) for a in ang]))


def _bolt(world, asset, T0, drive="hex"):
    if drive != "hex":
        return _bolt_drive(world, asset, T0, drive)
    b = BOLT
    body = _Body(world, "bolt", T0[:3, 3], T0[:3, :3])
    ET.SubElement(body.el, "freejoint", name="bolt")
    along_x = Rotation.from_euler("y", np.pi / 2).as_matrix()        # a cylinder's z onto the bolt's x
    hard = dict(density=STEEL, solref=SOLREF_HARD)
    body.geom("bolt_tip", "cylinder", (.002, 0, 0), C_STEEL, R=along_x, size=(.004, .002), **hard)
    body.geom("bolt_shank", "cylinder", (.004 + (b["L"] - .004) / 2, 0, 0), C_STEEL, R=along_x, visual=False,
              size=(b["d"] / 2, (b["L"] - .004) / 2), **hard)
    # drawn threaded: the crest at the shank's radius (its contacts), the crest's middle at 2 pi x / lead about +x
    _thread_mesh(asset, "bolt_thread_m", .004, b["L"], 0., b["d"] / 2, THREAD_LOOK["bolt_root"])
    body.look("bolt_thread", "mesh", (0, 0, 0), C_STEEL, mesh="bolt_thread_m")
    hb = b["head_h"] - b["hex_t"]
    body.geom("bolt_head", "cylinder", (b["L"] + hb / 2, 0, 0), C_STEEL, R=along_x, size=(b["head_d"] / 2, hb / 2),
              friction=MU_STEEL, **hard)
    # the socket: 6 boxes round a hex hole (flats normal to +-y at 0, 60, ... deg) from the head face back hex_t, the
    # mouth's countersink SOCKET_CS deep (6 bevel boxes; the flats start below it)
    t = b["head_d"] / 2 - SOCKET_AF / 2
    face, cs = b["L"] + b["head_h"], SOCKET_CS
    half_w = b["head_d"] / 2 * np.tan(np.radians(30)) + .0004
    for i in range(6):
        a = np.radians(60 * i)
        c = (SOCKET_AF / 2 + t / 2) * np.array([np.cos(a), np.sin(a)])
        depth = b["hex_t"] - cs
        body.geom(f"sock_{i}", "box", (face - cs - depth / 2, c[0], c[1]), C_STEEL, R=Rotation.from_euler("x", a).as_matrix(),
                  size=(depth / 2, t / 2, half_w), friction=MU_STEEL, density=STEEL, **SOCKET_CONTACT)
        r_hat, x_hat = np.array([0., np.cos(a), np.sin(a)]), np.array([1., 0., 0.])
        u, v = (x_hat + r_hat) / np.sqrt(2), (-x_hat + r_hat) / np.sqrt(2)      # up the slope; into the head
        mid = (face - cs / 2) * x_hat + (SOCKET_AF / 2 + cs / 2) * r_hat
        th = .001
        # exactly the bevel's length along the slope: any more and its lower end would stick into the hex hole
        body.geom(f"sock_cs_{i}", "box", mid + v * th, C_STEEL, R=np.c_[u, v, np.cross(u, v)],
                  size=(cs * np.sqrt(2) / 2, th, half_w + cs), friction=MU_STEEL, density=STEEL, **SOCKET_CONTACT)
    body.look("notch", "box", (face - .0006, 0, b["head_d"] / 2 - .0015), (.95, .45, .10, 1), size=(.0006, .0012, .0015))


def _key(world, asset, T0, k=KEY, drive="hex"):
    if drive != "hex":
        return _key_drive(world, asset, T0, drive, k)
    s = k["s"]
    body = _Body(world, "key", T0[:3, 3], T0[:3, :3])
    ET.SubElement(body.el, "freejoint", name="key")
    # each arm ends in a 1.5 mm chamfer (a 7.4 mm hex) that finds the socket from ~0.3 mm off
    _hex_prism(asset, "key_long_m", KEY_CHAMFER, k["long"] - s / 2, s)
    _hex_prism(asset, "key_long_tip_m", 0., KEY_CHAMFER, KEY_TIP_AF)
    _hex_prism(asset, "key_short_m", 0., k["short"] - s / 2 - KEY_CHAMFER, s)
    _hex_prism(asset, "key_short_tip_m", k["short"] - s / 2 - KEY_CHAMFER, k["short"] - s / 2, KEY_TIP_AF)
    kw = dict(density=STEEL, friction=MU_STEEL, **SOCKET_CONTACT)
    bend, R_short = (k["long"] - s / 2, 0, 0), Rotation.from_euler("y", -np.pi / 2).as_matrix()
    body.geom("key_long", "mesh", (0, 0, 0), C_KEY, mesh="key_long_m", **kw)
    body.geom("key_long_tip", "mesh", (0, 0, 0), C_KEY, mesh="key_long_tip_m", **kw)
    body.geom("key_short", "mesh", bend, C_KEY, R=R_short, mesh="key_short_m", **kw)
    body.geom("key_short_tip", "mesh", bend, C_KEY, R=R_short, mesh="key_short_tip_m", **kw)
    body.geom("key_bend", "sphere", bend, C_KEY, size=s * .55, **kw)


def _round(ctr, r, x, n):
    """n points round (ctr in the y-z plane, radius r) at x."""
    t = 2 * np.pi * np.arange(n) / n
    return np.c_[np.full(n, x), ctr[0] + r * np.cos(t), ctr[1] + r * np.sin(t)]


def _bolt_drive(world, asset, T0, drive):
    """The bolt with another drive's recess (DRIVES): the shank, its drawn thread, the head below the recess and the head's
    mark as _bolt builds them."""
    b, c = BOLT, DRIVES[drive]
    body = _Body(world, "bolt", T0[:3, 3], T0[:3, :3])
    ET.SubElement(body.el, "freejoint", name="bolt")
    along_x = Rotation.from_euler("y", np.pi / 2).as_matrix()        # a cylinder's z onto the bolt's x
    hard = dict(density=STEEL, solref=SOLREF_HARD)
    body.geom("bolt_tip", "cylinder", (.002, 0, 0), C_STEEL, R=along_x, size=(.004, .002), **hard)
    body.geom("bolt_shank", "cylinder", (.004 + (b["L"] - .004) / 2, 0, 0), C_STEEL, R=along_x, visual=False,
              size=(b["d"] / 2, (b["L"] - .004) / 2), **hard)
    _thread_mesh(asset, "bolt_thread_m", .004, b["L"], 0., b["d"] / 2, THREAD_LOOK["bolt_root"])
    body.look("bolt_thread", "mesh", (0, 0, 0), C_STEEL, mesh="bolt_thread_m")
    hb = b["head_h"] - c["depth"]
    body.geom("bolt_head", "cylinder", (b["L"] + hb / 2, 0, 0), C_STEEL, R=along_x, size=(b["head_d"] / 2, hb / 2),
              friction=MU_STEEL, **hard)
    dict(torx=_recess_torx, square=_recess_square, phillips=_recess_phillips, slotted=_recess_slotted)[drive](body, asset, c)
    face = b["L"] + b["head_h"]
    body.look("notch", "box", (face - .0006, 0, b["head_d"] / 2 - .0015), (.95, .45, .10, 1), size=(.0006, .0012, .0015))


def _ring_with_bevel(body, n, r_in, depth, cs):
    """The head's rim round a convex n-gon hole whose flats lie r_in from the axis (normals at 2 pi i / n), from
    the head face back `depth`, its mouth bevelled cs at 45 deg: n blocks and n bevel blocks, as the hex socket's."""
    b = BOLT
    face, rh = b["L"] + b["head_h"], b["head_d"] / 2
    t, half_w = rh - r_in, rh * np.tan(np.pi / n) + .0004
    kw = dict(friction=MU_STEEL, density=STEEL, **SOCKET_CONTACT)
    for i in range(n):
        a = 2 * np.pi * i / n
        c = (r_in + t / 2) * np.array([np.cos(a), np.sin(a)])
        dz = depth - cs
        body.geom(f"sock_ring_{i}", "box", (face - cs - dz / 2, c[0], c[1]), C_STEEL,
                  R=Rotation.from_euler("x", a).as_matrix(), size=(dz / 2, t / 2, half_w), **kw)
        r_hat, x_hat = np.array([0., np.cos(a), np.sin(a)]), np.array([1., 0., 0.])
        u, v = (x_hat + r_hat) / np.sqrt(2), (-x_hat + r_hat) / np.sqrt(2)      # up the slope; into the head
        mid = (face - cs / 2) * x_hat + (r_in + cs / 2) * r_hat
        body.geom(f"sock_cs_{i}", "box", mid + v * .001, C_STEEL, R=np.c_[u, v, np.cross(u, v)],
                  size=(cs * np.sqrt(2) / 2, .001, half_w + cs), **kw)


def _recess_torx(body, asset, c):
    """The six-lobe recess: the rim round a 12-gon whose flats are A / 2 from the axis (the outer lobes' reach; a flat at
    each outer lobe, 30 + 60 k deg) and 6 round bumps between the lobes (at 60 k deg, the hex socket's flats), each a
    prism of bump_r round bump_c from the axis reaching in to B / 2, its top end bevelled 45 deg."""
    b = BOLT
    face, depth, cs = b["L"] + b["head_h"], c["depth"], c["cs"]
    _ring_with_bevel(body, 12, c["A"] / 2, depth, cs)
    kw = dict(friction=MU_STEEL, density=STEEL, **SOCKET_CONTACT)
    r = c["bump_r"]
    for i in range(6):
        a = np.radians(60 * i)
        ctr = c["bump_c"] * np.array([np.cos(a), np.sin(a)])
        V = np.vstack([_round(ctr, r, face - depth, c["n_round"]), _round(ctr, r, face - cs, c["n_round"]),
                       _round(ctr, r - cs, face, c["n_round"])])
        ET.SubElement(asset, "mesh", name=f"sock_bump_m{i}", vertex=_v(V))
        body.geom(f"sock_bump_{i}", "mesh", (0, 0, 0), C_STEEL, mesh=f"sock_bump_m{i}", **kw)


def _recess_walls(body, asset, n, r_in, depth, cs, rotate, name="sock_wall"):
    """The head's rim round a convex n-gon hole whose flats lie r_in from the axis (normals at rotate + 2 pi i / n), from
    the head face back `depth`, its mouth bevelled cs at 45 deg: each side's wall is the head's disk beyond the side's line
    (a convex piece: the round head stays round), the bevel its edge at the mouth."""
    b = BOLT
    face, rh = b["L"] + b["head_h"], b["head_d"] / 2
    kw = dict(friction=MU_STEEL, density=STEEL, **SOCKET_CONTACT)
    for i in range(n):
        a = rotate + 2 * np.pi * i / n
        e, t = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        h0, h1 = np.arccos(r_in / rh), np.arccos((r_in + cs) / rh)
        arc = lambda h: [rh * np.array([np.cos(a + u), np.sin(a + u)]) for u in np.linspace(-h, h, 24)]   # noqa: E731
        chord = lambda r, h: [r * e - rh * np.sin(h) * t, r * e + rh * np.sin(h) * t]          # noqa: E731
        # below the bevel the side's line at r_in; at the face it has moved out to r_in + cs (the arc shorter there)
        V = [[x, *p] for x in (face - depth, face - cs) for p in arc(h0) + chord(r_in, h0)]
        V += [[face, *p] for p in arc(h1) + chord(r_in + cs, h1)]
        ET.SubElement(asset, "mesh", name=f"{name}_m{i}", vertex=_v(V))
        body.geom(f"{name}_{i}", "mesh", (0, 0, 0), C_STEEL, mesh=f"{name}_m{i}", **kw)


def _recess_square(body, asset, c):
    """The square recess: four walls (_recess_walls) round a square of AF, turned rotate_deg."""
    _recess_walls(body, asset, 4, c["AF"] / 2, c["depth"], c["cs"], np.radians(c["rotate_deg"]))


def _uv(a):
    """The unit vectors at angle a and a + 90 deg in the bolt's y-z plane."""
    return np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])


def _recess_phillips(body, asset, c):
    """The cross recess: four quadrant blocks between the slots (each the head's disk beyond both slots' edges, the edges
    bevelled cs at the face) and four slot-end walls (the disk beyond the slot's sloping end, R_bottom at the recess's
    depth to R_face at the face, a little wider than the slot so they meet the quadrants)."""
    b = BOLT
    face, rh, D, w, cs = b["L"] + b["head_h"], b["head_d"] / 2, c["depth"], c["w"], c["cs"]
    kw = dict(friction=MU_STEEL, density=STEEL, **SOCKET_CONTACT)
    rot = np.radians(c["rotate_deg"])
    for k in range(4):
        u0, _ = _uv(rot + k * np.pi / 2)
        u1, _ = _uv(rot + (k + 1) * np.pi / 2)
        V = []
        for x, o in ((face - D, w / 2), (face - cs, w / 2), (face, w / 2 + cs)):
            t0 = np.arcsin(o / rh)
            V += [[x, *(o * u0 + o * u1)]]
            V += [[x, *(rh * (np.cos(t) * u0 + np.sin(t) * u1))] for t in np.linspace(t0, np.pi / 2 - t0, 16)]
        ET.SubElement(asset, "mesh", name=f"sock_quad_m{k}", vertex=_v(V))
        body.geom(f"sock_quad_{k}", "mesh", (0, 0, 0), C_STEEL, mesh=f"sock_quad_m{k}", **kw)
    hw = w / 2 + cs + .0003
    for k in range(4):
        u, n = _uv(rot + k * np.pi / 2)
        V = []
        for x, R in ((face - D, c["R_bottom"]), (face, c["R_face"])):
            e = np.sqrt(rh ** 2 - hw ** 2)
            V += [[x, *(R * u + sg * hw * n)] for sg in (-1, 1)] + [[x, *(e * u + sg * hw * n)] for sg in (-1, 1)]
            t1 = np.arcsin(hw / rh)
            V += [[x, *(rh * (np.cos(t) * u + np.sin(t) * n))] for t in np.linspace(-t1, t1, 5)]
        ET.SubElement(asset, "mesh", name=f"sock_end_m{k}", vertex=_v(V))
        body.geom(f"sock_end_{k}", "mesh", (0, 0, 0), C_STEEL, mesh=f"sock_end_m{k}", **kw)


def _recess_slotted(body, asset, c):
    """The straight slot: the head's two halves beyond the slot's edges (_recess_walls with two sides), along
    rotate_deg."""
    _recess_walls(body, asset, 2, c["w"] / 2, c["depth"], c["cs"], np.radians(c["rotate_deg"] + 90.))


def _key_drive(world, asset, T0, drive, k=KEY):
    """The L-shaped tool with another drive's tips (DRIVES): the hex bars, as _key's, end tip_len short of each end; the tip
    there runs to the end."""
    c = DRIVES[drive]
    if c.get("ratchet"):
        return _key_ratchet(world, asset, T0, drive, k)
    s, L = k["s"], c["tip_len"]
    body = _Body(world, "key", T0[:3, 3], T0[:3, :3])
    ET.SubElement(body.el, "freejoint", name="key")
    _hex_prism(asset, "key_long_m", L, k["long"] - s / 2, s)
    _hex_prism(asset, "key_short_m", 0., k["short"] - s / 2 - L, s)
    kw = dict(density=STEEL, friction=MU_STEEL, **SOCKET_CONTACT)
    bend, R_short = (k["long"] - s / 2, 0, 0), Rotation.from_euler("y", -np.pi / 2).as_matrix()
    body.geom("key_long", "mesh", (0, 0, 0), C_KEY, mesh="key_long_m", **kw)
    body.geom("key_short", "mesh", bend, C_KEY, R=R_short, mesh="key_short_m", **kw)
    body.geom("key_bend", "sphere", bend, C_KEY, size=s * .55, **kw)
    tip = dict(torx=_tip_torx)[drive]
    tip(body, asset, c, "long", L, 0., (0, 0, 0), None)                                 # the long arm's tip: the origin
    tip(body, asset, c, "short", k["short"] - s / 2 - L, k["short"] - s / 2, bend, R_short)


def _key_ratchet(world, asset, T0, drive, k=KEY):
    """The ratcheting L-shaped driver (RATCHET): the L body "key" (the long arm as the hex key's, its end chamfered, the
    bend, the ratchet housing with its reversing switch) and the bit body "key_bit" at the bend (the short arm's hex bar
    below the housing and the drive's tip at its end), on the hinge key_ratchet about the short arm; the key frame as the
    hex key's, so the short arm's end is where the hex key's is."""
    c, r = DRIVES[drive], RATCHET
    s, L = k["s"], c["tip_len"]
    body = _Body(world, "key", T0[:3, 3], T0[:3, :3])
    ET.SubElement(body.el, "freejoint", name="key")
    _hex_prism(asset, "key_long_m", KEY_CHAMFER, k["long"] - s / 2, s)
    _hex_prism(asset, "key_long_tip_m", 0., KEY_CHAMFER, KEY_TIP_AF)
    kw = dict(density=STEEL, friction=MU_STEEL, **SOCKET_CONTACT)
    bend = np.array([k["long"] - s / 2, 0, 0])
    R_short = Rotation.from_euler("y", -np.pi / 2).as_matrix()       # a mesh's x onto the key's z (the short arm)
    body.geom("key_long", "mesh", (0, 0, 0), C_KEY, mesh="key_long_m", **kw)
    body.geom("key_long_tip", "mesh", (0, 0, 0), C_KEY, mesh="key_long_tip_m", **kw)
    body.geom("key_bend", "sphere", bend, C_KEY, size=s * .55, **kw)
    z0, z1 = r["house_z"]
    body.geom("key_ratchet_house", "cylinder", bend + (0, 0, (z0 + z1) / 2), C_KEY, size=(r["house_r"], (z1 - z0) / 2), **kw)
    sw = r["switch"]                                  # the reversing switch on the housing's top, toward the long arm
    body.look("key_ratchet_switch", "box", bend + (-r["house_r"] / 2, 0, z0 - sw[2]), C_KEY, size=sw)
    bit = _Body(body.el, "key_bit", bend)
    ET.SubElement(bit.el, "joint", name="key_ratchet", type="hinge", axis="0 0 1", range="0 10000",
                  frictionloss=_v(r["drag_nm"]), damping="0.0001", armature="0.000001", **HARD_LIMIT)
    _hex_prism(asset, "key_short_m", z1, k["short"] - s / 2 - L, s)
    bit.geom("key_short", "mesh", (0, 0, 0), C_KEY, R=R_short, mesh="key_short_m", **kw)
    bit.look("key_bit_collar", "cylinder", (0, 0, z1 + .001), C_KEY, size=(r["house_r"] - .0015, .001))
    dict(square=_tip_square, phillips=_tip_phillips, slotted=_tip_slotted)[drive](
        bit, asset, c, "short", k["short"] - s / 2 - L, k["short"] - s / 2, (0, 0, 0), R_short)


def _tip_square(body, asset, c, end, x_bar, x_end, pos, R):
    """A square tip of tip_AF along its mesh x from x_bar to x_end, its flats' normals at 0 and 90 deg, the last
    tip_chamfer 0.3 mm smaller (each side) at the end."""
    x_ch = x_end - np.sign(x_end - x_bar) * c["tip_chamfer"]
    sq = lambda x, af: [[x, af / 2 * u, af / 2 * v] for u in (-1, 1) for v in (-1, 1)]   # noqa: E731
    mesh = f"key_{end}_tip_m"
    ET.SubElement(asset, "mesh", name=mesh, vertex=_v(sq(x_bar, c["tip_AF"]) + sq(x_ch, c["tip_AF"]) + sq(x_end, c["tip_AF"] - .0006)))
    body.geom(f"key_{end}_tip", "mesh", pos, C_KEY, R=R, mesh=mesh, density=STEEL, friction=MU_STEEL, **SOCKET_CONTACT)


def _tip_phillips(body, asset, c, end, x_bar, x_end, pos, R):
    """A cross tip: four wings (mesh angles 0, 90, 180, 270 deg) from the axis out to the slots' sloping ends less 0.075 mm
    (the tip's end at tip_depth when pushed in), tip_w thick, thinning to tip_end_w over tip_chamfer at the end."""
    sg = np.sign(x_end - x_bar)
    D, Dt = c["depth"], c["tip_depth"]
    wall = lambda depth: c["R_face"] - (c["R_face"] - c["R_bottom"]) * depth / D   # noqa: E731
    prof = [(0., wall(Dt) - .000075, c["tip_end_w"] / 2), (c["tip_chamfer"], wall(Dt - c["tip_chamfer"]) - .000075, c["tip_w"] / 2),
            (Dt, c["R_face"] - .000075, c["tip_w"] / 2), (abs(x_end - x_bar), c["R_face"] - .000075, c["tip_w"] / 2)]
    for j in range(4):
        u, n = _uv(j * np.pi / 2)
        V = [[x_end - sg * si, *(r * u + h * n)] for si, Rw, hw in prof for r in (0., Rw) for h in (-hw, hw)]
        mesh = f"key_{end}_tip_wing_m{j}"
        ET.SubElement(asset, "mesh", name=mesh, vertex=_v(V))
        body.geom(f"key_{end}_tip_wing_{j}", "mesh", pos, C_KEY, R=R, mesh=mesh, density=STEEL, friction=MU_STEEL, **SOCKET_CONTACT)


def _tip_slotted(body, asset, c, end, x_bar, x_end, pos, R):
    """A flat blade tip_W wide along the mesh's y and tip_t thick, thinning to tip_end_t over tip_chamfer at the end."""
    sg = np.sign(x_end - x_bar)
    W = c["tip_W"] / 2
    V = [[x, y, z] for x, t in ((x_bar, c["tip_t"]), (x_end - sg * c["tip_chamfer"], c["tip_t"]), (x_end, c["tip_end_t"]))
         for y in (-W, W) for z in (-t / 2, t / 2)]
    mesh = f"key_{end}_tip_m"
    ET.SubElement(asset, "mesh", name=mesh, vertex=_v(V))
    body.geom(f"key_{end}_tip", "mesh", pos, C_KEY, R=R, mesh=mesh, density=STEEL, friction=MU_STEEL, **SOCKET_CONTACT)


def key_geom_names(drive="hex"):
    """The tool's collision geoms (thread.BoltThread: the stop's clear check)."""
    if drive == "hex":
        return ["key_long", "key_long_tip", "key_short", "key_short_tip", "key_bend"]
    if DRIVES[drive].get("ratchet"):
        tips = [f"key_short_tip_wing_{j}" for j in range(4)] if drive == "phillips" else ["key_short_tip"]
        return ["key_long", "key_long_tip", "key_bend", "key_ratchet_house", "key_short"] + tips
    parts = dict(torx=["core"] + [f"lobe_{j}" for j in range(6)])[drive]
    return ["key_long", "key_short", "key_bend"] + [f"key_{e}_tip_{p}" for e in ("long", "short") for p in parts]


def key_ends(drive="hex"):
    """The tool's ends that carry the drive's tip (key_end_points): the ratcheting driver's long arm has none."""
    return ("short",) if DRIVES[drive].get("ratchet") else ("long", "short")


def socket_names(drive="hex"):
    """The recess's collision geoms (the head below the recess not counted)."""
    if drive == "hex":
        return [f"sock_{i}" for i in range(6)] + [f"sock_cs_{i}" for i in range(6)]
    if drive == "torx":
        return [f"sock_ring_{i}" for i in range(12)] + [f"sock_cs_{i}" for i in range(12)] + [f"sock_bump_{i}" for i in range(6)]
    if drive == "square":
        return [f"sock_wall_{i}" for i in range(4)]
    if drive == "slotted":
        return [f"sock_wall_{i}" for i in range(2)]
    if drive == "phillips":
        return [f"sock_quad_{i}" for i in range(4)] + [f"sock_end_{i}" for i in range(4)]
    raise ValueError(f"bed_bolt has no drive {drive!r}")


def _tip_torx(body, asset, c, end, x_bar, x_end, pos, R):
    """A six-lobe tip along its mesh x from x_bar (the hex bar's end) to x_end (the tool's end): a core of tip_B / 2 and 6
    lobes of lobe_r round lobe_c, at the hex bar's corners (30 + 60 k deg); the last tip_chamfer 0.3 mm smaller at the end."""
    n = c["n_round"]
    x_ch = x_end - np.sign(x_end - x_bar) * c["tip_chamfer"]
    f = (c["tip_A"] / 2 - .0003) / (c["tip_A"] / 2)
    kw = dict(density=STEEL, friction=MU_STEEL, **SOCKET_CONTACT)
    pieces = [("core", (0., 0.), c["tip_B"] / 2)]
    for j in range(6):
        a = np.radians(30 + 60 * j)
        pieces.append((f"lobe_{j}", (c["lobe_c"] * np.cos(a), c["lobe_c"] * np.sin(a)), c["lobe_r"]))
    for part, ctr, r in pieces:
        ctr = np.asarray(ctr)
        V = np.vstack([_round(ctr, r, x_bar, n), _round(ctr, r, x_ch, n), _round(ctr * f, r * f, x_end, n)])
        mesh = f"key_{end}_tip_{part}_m"
        ET.SubElement(asset, "mesh", name=mesh, vertex=_v(V))
        body.geom(f"key_{end}_tip_{part}", "mesh", pos, C_KEY, R=R, mesh=mesh, **kw)


def entity_xml(name, place=PLACE, drive="hex") -> str:
    """The MJCF text of one entity at the task's birth (ENTITIES), in the mount frame, the bolt's recess and the tool's
    tips of `drive` (DRIVES)."""
    root = ET.Element("mujoco", model=name)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    if name == "bed_frame":
        _bed_frame(world, asset, place)
        _holders(world) if "stand_hole_y" not in DRIVES[drive] else _holders(world, hole_y=DRIVES[drive]["stand_hole_y"])
    elif name == "bolt":
        cradle = DRIVES[drive].get("cradle_deg")
        _bolt(world, asset, bolt_start_T() if cradle is None else bolt_start_T(theta0=np.radians(cradle)), drive)  # first: the entity's base link
        _thread(world, hole_T(place))
        eq = ET.SubElement(root, "equality")
        # the thread's lead: slide = lead / (2 pi) x spin, as hard as MuJoCo allows (2 dt); the design's per-step
        # projection (runs/bed_bolt_design_20261012/scripts/phys3_replay.py) wrote qpos outside the solver
        ET.SubElement(eq, "joint", name="thread_lead", joint1="thr_s", joint2="thr_h",
                      polycoef=_v([0, BOLT["lead"] / (2 * np.pi), 0, 0, 0]), solref="0.00125 1",
                      solimp=_v(SOLIMP_RIGID))
        # the spin's lock: while the thread is engaged and no key end sits in the socket the mechanism holds the spin at
        # its angle (polycoef[0]), so a push, a pull or a finger's twist does not turn the bolt (the user, 2026-10-16)
        ET.SubElement(eq, "joint", name="thread_lock", joint1="thr_h", polycoef="0 0 0 0 0", active="false",
                      solref="0.00125 1", solimp=_v(SOLIMP_RIGID))
        # the tilt's hold from hold_turns: each tilt hinge held at polycoef[0], set at its angle when the thread reaches
        # hold_turns and eased to 0 (THREAD tilt_ease_deg_s); before topogym the sandbox switched the tilt stiffness to
        # 3 N m / rad there, which stored 0.5 k tilt^2 at the switch and let a 100 N push buckle the bolt to 16 deg
        for j in ("y", "z"):
            ET.SubElement(eq, "joint", name=f"thread_tilt_{j}", joint1=f"thr_t{j}", polycoef="0 0 0 0 0", active="false",
                          solref="0.00125 1", solimp=_v(SOLIMP_RIGID))
        # the carriage carries the caught bolt (the mechanism writes the relative pose and switches it)
        # near-rigid impedance: at the default (.9 .95) the weld twisted about 8 deg per N m of the thread's friction
        ET.SubElement(eq, "weld", name="thread_weld", body1="thr_spin", body2="bolt", active="false", solref="0.00125 1",
                      solimp=_v(SOLIMP_RIGID))
    elif name == "key":
        _key(world, asset, key_start_T(), drive=drive)
    else:
        raise ValueError(f"bed_bolt has no entity {name!r}")
    _dress(world, asset, name, drive)
    if not len(asset):
        root.remove(asset)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------- the look (visual only)
# The user (2026-10-19; runs/robotangle_mujoco/bed_bolt/20261018_variants/VISUAL_AUDIT_zh.md, items 1-8): the scene's
# look made more like the real things. _dress changes only geoms that neither collide nor have mass (the visual twins
# "<name>_look" and the looks) and adds only such geoms, with their own meshes; the MuJoCo scene drops all of them
# (topogym.world.mujoco.mjcf_entity.merge_mjcf_entity: visual geoms and the meshes only they use), so the physics scene
# is unchanged and only the renderer's entity files differ.
# - Sharp edges: the renderer (nyx_direct) draws a box or cylinder geom as a trimesh whose faces share vertices and
#   average their normals, so every edge looks rounded (a box's corner normal is (-.577, -.577, -.577)); a vertex-only
#   mesh is drawn with MuJoCo's own normals (the key's hex bars: sharp). Box and cylinder looks become such meshes, the
#   same solid or a chamfered one inside it (LOOK_BEVEL; a 48-sided prism for a cylinder, its corners on the circle).
# - Colours (LOOK_RGBA): pine boards each a shade apart, birch slats, a darker hardwood corner block and holders, a
#   black-oxide bolt, a satin chrome key (02's too) and the ratcheting driver's dark handle with a bright head and bit.
# - Additions: a zinc threaded insert's flange in the block face (the bore's look 0.3 mm short of the face there, the
#   hole's drawn thread in the insert's colour), two zinc screw heads at each slat end, the bolt's cradle walls as solid
#   V-blocks down to the table, the hex and Torx keys' bend as a bent bar (BEND_R on the centre line; the straight arms'
#   looks stop where it starts) instead of a ball, and a ring on the ratchet housing's top.
# - The bolt's head round: the recess's rim pieces (the hex and Torx rings of boxes stand out of the head's circle at
#   their corners) drawn cut to the head's 48-sided prism, its top rim chamfered HEAD_CHAMFER.
BEND_R = .008
HEAD_CHAMFER = .0008
LOOK_BEVEL = (("rail_", .0015), ("end_left", .0015), ("middle_board", .0015), ("slat_", .0015), ("ledge_", .001),
              ("key_rest", .001), ("cradle_", .0004))          # first match; anything else: the same solid, sharp
PINE, BIRCH, HARDWOOD = (.70, .50, .30, 1), (.85, .74, .55, 1), (.47, .32, .19, 1)
ZINC, BLACK_OXIDE, CHROME = (.66, .67, .69, 1), (.16, .16, .17, 1), (.66, .68, .71, 1)
SHADE = dict(rail_near=1., rail_far=.95, end_left=1.04, end_r=.97, middle_board=1.02, leg=.94, ledge=1.03)


def _shade(rgba, f):
    return tuple(min(1., c * f) for c in rgba[:3]) + (rgba[3],)


def _look_rgba(entity, name, drive):
    """The look's colour (None: keep it), by its geom name (a twin's without "_look")."""
    if entity == "bed_frame":
        if name.startswith(("cradle_", "stand_", "key_rest", "vblock_")):
            return HARDWOOD
        if name.startswith("slat_"):
            return _shade(BIRCH, (.97, 1.03, 1., .98, 1.02)[int(name.split("_")[1]) % 5])
        if name.startswith(("block_", "bore_")):
            return _shade(HARDWOOD, 1.25)
        if name.startswith(("hole_thread", "insert_", "screw_")):
            return ZINC
        for k, f in SHADE.items():
            if name.startswith(k):
                return _shade(PINE, f)
        return None
    if entity == "bolt":
        return None if name == "notch" else BLACK_OXIDE
    if DRIVES[drive].get("ratchet"):
        dark = name.startswith(("key_long", "key_bend"))
        return _shade(CHROME, .3) if dark else _shade(CHROME, 1.1)
    return CHROME


def _box_corners(h, c=0.):
    """A box of half sizes h, each edge chamfered c at 45 deg: the hull of the box shrunk by c across each pair of axes."""
    h = np.asarray(h, dtype=float)
    shrunk = [h] if c <= 0 else [np.where(np.arange(3) == k, h, h - c) for k in range(3)]
    return np.array([s * (sx, sy, sz) for s in shrunk for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def _prism_corners(r, hz, c_lo=0., c_hi=0., n=48):
    """A 48-sided prism along z (corners on the circle of radius r, from -hz to hz), its ends' rims chamfered c_lo, c_hi."""
    t = 2 * np.pi * np.arange(n) / n
    ring = lambda rr, z: np.c_[rr * np.cos(t), rr * np.sin(t), np.full(n, z)]   # noqa: E731
    out = [ring(r - c_lo, -hz)] + ([ring(r, -hz + c_lo)] if c_lo > 0 else [])
    out += ([ring(r, hz - c_hi)] if c_hi > 0 else []) + [ring(r - c_hi, hz)]
    return np.vstack(out)


def _annulus(asset, mesh, r_in, r_out, z0, z1, n=48):
    """A closed ring along local z from z0 to z1 between radii r_in and r_out, its faces pointing out of the solid."""
    t = 2 * np.pi * np.arange(n) / n
    V = np.vstack([np.c_[r * np.cos(t), r * np.sin(t), np.full(n, z)] for z in (z0, z1) for r in (r_in, r_out)])
    q = lambda k, j: k * n + j % n                    # noqa: E731   k: 0 inner z0, 1 outer z0, 2 inner z1, 3 outer z1
    F = []
    for j in range(n):
        F += [(q(0, j), q(1, j + 1), q(1, j)), (q(0, j), q(0, j + 1), q(1, j + 1))]          # z0 end: faces -z
        F += [(q(2, j), q(3, j), q(3, j + 1)), (q(2, j), q(3, j + 1), q(2, j + 1))]          # z1 end: faces +z
        F += [(q(1, j), q(1, j + 1), q(3, j + 1)), (q(1, j), q(3, j + 1), q(3, j))]          # outer: faces out
        F += [(q(0, j), q(2, j + 1), q(0, j + 1)), (q(0, j), q(2, j), q(2, j + 1))]          # inner: faces in
    ET.SubElement(asset, "mesh", name=mesh, vertex=_v(V), face=" ".join(str(int(k)) for f in F for k in f))


def _look_geom(parent, name, mesh, rgba, pos=(0, 0, 0), quat=None):
    a = dict(name=name, type="mesh", mesh=mesh, pos=_v(pos), contype="0", conaffinity="0", mass="0", group="1",
             rgba=_v(rgba))
    if quat is not None:
        a["quat"] = quat
    return ET.SubElement(parent, "geom", a)


def _dress(world, asset, entity, drive):
    """The entity's look (the section above): its visual geoms changed and added, its colliding geoms untouched."""
    def visual(g):
        return g.get("contype") == "0" and g.get("conaffinity") == "0" and g.get("mass") == "0"

    for body in list(world.iter("body")):
        for g in [g for g in body.findall("geom") if visual(g)]:
            name = g.get("name")
            base = name[:-len("_look")] if name.endswith("_look") else name
            kind, size = g.get("type"), np.fromstring(g.get("size", ""), sep=" ")
            V = None
            if kind == "box":
                c = next((c for p, c in LOOK_BEVEL if base.startswith(p)), 0.)
                if base.startswith("bore_"):           # the insert's flange takes the bore's first 0.3 mm
                    size[0] -= INSERT_T / 2
                    g.set("pos", _v(np.fromstring(g.get("pos"), sep=" ") + (INSERT_T / 2, 0, 0)))
                V = _box_corners(size, c)
            elif kind == "cylinder":
                c_lo, c_hi = dict(leg=(0., .002), bolt_head=(.0005, 0.), bolt_tip=(.0008, 0.),
                                  key_ratchet_house=(.0008, .0008)).get(base.split("_")[0] if base.startswith("leg_") else base, (0., 0.))
                V = _prism_corners(size[0], size[1], c_lo, c_hi)
            if V is not None:
                ET.SubElement(asset, "mesh", name=f"{name}_m", vertex=_v(V))
                for k in ("size", "type"):
                    del g.attrib[k]
                g.set("type", "mesh")
                g.set("mesh", f"{name}_m")
            rgba = _look_rgba(entity, base, drive)
            if rgba is not None:
                g.set("rgba", _v(rgba))
    if entity == "bed_frame":
        _dress_frame(world, asset, DRIVES[drive].get("stand_hole_y", .005))
    elif entity == "bolt":
        _dress_head(world, asset)
    elif entity == "key" and not DRIVES[drive].get("ratchet"):
        _dress_bend(world, asset, drive)
    elif entity == "key":
        _dress_ratchet(world, asset)


INSERT_T = .0003                # the insert's flange: in the block face, this thick


def _dress_frame(world, asset, hole_y, h=HOLD):
    """The insert's flange, the slats' screw heads, the bolt's V-blocks and the key's stand as one block (its five
    pieces' looks shaded apart) with its hole's mouth drawn dark on top."""
    frame, holders = world.find("body[@name='bedframe']"), world.find("body[@name='holders']")
    p, _ = hole_frame_point()
    _annulus(asset, "insert_flange_m", THREAD_LOOK["hole_outer"], RB + WALL, 0., INSERT_T)
    along_x = _quat(Rotation.from_euler("y", np.pi / 2).as_matrix())
    _look_geom(frame, "insert_flange", "insert_flange_m", ZINC, pos=p, quat=along_x)
    f = FRAME
    ET.SubElement(asset, "mesh", name="screw_head_m", vertex=_v(_prism_corners(.003, .00015, 0., .0001, n=24)))
    top = f["zb"] + f["h"] - .022 + f["slat_t"]
    for i, x in enumerate(f["slat_xc"]):
        for sy in (-1, 1):
            for dx in (-1, 1):
                y = sy * (f["W"] / 2 - f["t"] - f["ledge"] / 2)
                _look_geom(frame, f"screw_{i}_{'nf'[sy > 0]}{'ab'[dx > 0]}", "screw_head_m", ZINC,
                           pos=(x + dx * f["slat_w"] / 4, y, top + .00005))
    # each cradle wall swept straight down to the table: below its V face, so the bolt's room above it stays clear
    for g in [g for g in holders.findall("geom") if g.get("name", "").startswith("cradle_") and g.get("contype") is None]:
        q = np.fromstring(g.get("quat"), sep=" ")
        R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        C = np.fromstring(g.get("pos"), sep=" ") + _box_corners(np.fromstring(g.get("size"), sep=" ")) @ R.T
        base = C.copy()
        base[:, 2] = 0.
        mesh = f"vblock_{g.get('name')}_m"
        ET.SubElement(asset, "mesh", name=mesh, vertex=_v(np.vstack([C, base])))
        _look_geom(holders, f"vblock_{g.get('name')}", mesh, HARDWOOD)
    for part in ("base", "n", "f", "l", "r"):
        holders.remove(holders.find(f"geom[@name='stand_{part}_look']"))
        asset.remove(asset.find(f"mesh[@name='stand_{part}_look_m']"))
    (x, y), w = h["key_xy"], h["stand_w"] / 2
    ET.SubElement(asset, "mesh", name="stand_block_m", vertex=_v(_box_corners((w, w, h["stand_h"] / 2), .001)))
    _look_geom(holders, "stand_block", "stand_block_m", HARDWOOD, pos=(x, y, h["stand_h"] / 2))
    ET.SubElement(asset, "mesh", name="stand_hole_m", vertex=_v(_box_corners((.005, hole_y, .0001))))
    _look_geom(holders, "stand_hole", "stand_hole_m", C_HOLE, pos=(x, y, h["stand_h"] + .0001))


def _dress_head(world, asset, n=48):
    """Each look of the recess's rim (sock_*) cut to the head: the solid it draws (a box or a vertex-only mesh's hull)
    intersected with the head's n-sided prism (its corners on the head's circle, as the head's look) and the 45 deg
    chamfer HEAD_CHAMFER round its face; the cut piece's corners in the bolt frame."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    body, b = world.find("body[@name='bolt']"), BOLT
    face, rh = b["L"] + b["head_h"], b["head_d"] / 2
    ap = rh * np.cos(np.pi / n)
    th = 2 * np.pi * (np.arange(n) + .5) / n
    nrm = np.c_[np.zeros(n), np.cos(th), np.sin(th)]
    cut = np.vstack([np.c_[nrm, np.full(n, -ap)],                                     # n . p <= ap
                     np.c_[nrm + (1, 0, 0), np.full(n, -(face + ap - HEAD_CHAMFER))]])  # x + n . p <= face + ap - c
    meshes = {m.get("name"): m for m in asset.findall("mesh")}
    for g in body.findall("geom"):
        if not (g.get("name", "").startswith("sock_") and g.get("name").endswith("_look")):
            continue
        q = np.fromstring(g.get("quat", "1 0 0 0"), sep=" ")
        V = np.fromstring(meshes[g.get("mesh")].get("vertex"), sep=" ").reshape(-1, 3)
        V = V @ Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix().T + np.fromstring(g.get("pos"), sep=" ")
        H = np.vstack([ConvexHull(V).equations, cut])
        lp = linprog(np.r_[np.zeros(3), -1.], A_ub=np.c_[H[:, :3], np.linalg.norm(H[:, :3], axis=1)], b_ub=-H[:, 3],
                     bounds=[(None, None)] * 3 + [(0, None)])
        P = HalfspaceIntersection(H, lp.x[:3]).intersections
        ET.SubElement(asset, "mesh", name=g.get("name") + "_cut_m", vertex=_v(P))
        if g.get("mesh") == g.get("name") + "_m":       # the box's own look mesh (_dress): no other geom uses it
            asset.remove(meshes[g.get("mesh")])
        g.set("mesh", g.get("name") + "_cut_m")
        g.set("pos", "0 0 0")
        g.attrib.pop("quat", None)


def _dress_bend(world, asset, drive, k=KEY, n=24):
    """The hex or Torx key bent round BEND_R (centre line) at the corner: the arms' looks stop where the bend starts, the
    ball's look goes, and one mesh of the hex bar swept through the quarter turn in n steps (its faces sharing corners
    along the sweep, so each side shades smoothly round the bend and the hex edges stay sharp) takes its place."""
    body, s = world.find("body[@name='key']"), k["s"]
    x_b, rc = k["long"] - s / 2, s / np.sqrt(3)
    start = KEY_CHAMFER if drive == "hex" else DRIVES[drive]["tip_len"]
    end = k["short"] - s / 2 - (KEY_CHAMFER if drive == "hex" else DRIVES[drive]["tip_len"])
    _hex_prism(asset, "key_long_look_m", start, x_b - BEND_R, s)
    _hex_prism(asset, "key_short_look_m", BEND_R, end, s)
    for g in body.findall("geom"):
        if g.get("name") in ("key_long_look", "key_short_look"):
            g.set("mesh", g.get("name") + "_m")
    body.remove(body.find("geom[@name='key_bend_look']"))
    _look_geom(body, "key_bend_arc", "key_bend_arc_m", CHROME)
    ET.SubElement(asset, "mesh", name="key_bend_arc_m", **bend_arc(k, n))


def bend_arc(k=KEY, n=24) -> dict:
    """The bend's mesh (_dress_bend) in the key frame: vertex (n + 1 cross-sections of the 6 hex corners, the long arm's
    at the start, the short arm's at the end; ordered so) and face (outward)."""
    s = k["s"]
    x_b, rc = k["long"] - s / 2, s / np.sqrt(3)
    C = np.array([x_b - BEND_R, 0., BEND_R])
    ang = np.radians(np.arange(6) * 60 + 30)
    V = []
    for phi in np.linspace(0., np.pi / 2, n + 1):     # (y, z(phi), the sweep's tangent) right-handed; corners counter-clockwise
        z_hat = np.array([-np.sin(phi), 0., np.cos(phi)])
        P = C + BEND_R * np.array([np.sin(phi), 0., -np.cos(phi)])
        V += [P + rc * (np.cos(a) * np.array([0., 1., 0.]) + np.sin(a) * z_hat) for a in ang]
    q = lambda i, j: 6 * i + j % 6                    # noqa: E731
    F = [f for i in range(n) for j in range(6) for f in ((q(i, j), q(i, j + 1), q(i + 1, j + 1)), (q(i, j), q(i + 1, j + 1), q(i + 1, j)))]
    F += [(q(0, 0), q(0, j + 1), q(0, j)) for j in range(1, 5)] + [(q(n, 0), q(n, j), q(n, j + 1)) for j in range(1, 5)]
    return dict(vertex=_v(np.array(V)), face=" ".join(str(int(v)) for f in F for v in f))


def _dress_ratchet(world, asset, r=RATCHET):
    """The ratchet housing's top: a ring proud of it by 0.15 mm."""
    body = world.find("body[@name='key']")
    bend = np.array([KEY["long"] - KEY["s"] / 2, 0, 0])
    _annulus(asset, "key_ratchet_ring_m", r["house_r"] - .0025, r["house_r"] - .0012, -.00015, 0.)
    _look_geom(body, "key_ratchet_ring", "key_ratchet_ring_m", _shade(CHROME, .75), pos=bend + (0, 0, r["house_z"][0]))


def bolt_points() -> np.ndarray:
    """Points spanning the bolt's solid in its frame (shank, head), for its grasp region."""
    b = BOLT
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    ring = lambda x, r: np.c_[np.full(len(ang), x), r * np.cos(ang), r * np.sin(ang)]
    return np.vstack([ring(0., .004), ring(b["L"], b["d"] / 2), ring(b["L"], b["head_d"] / 2),
                      ring(b["L"] + b["head_h"], b["head_d"] / 2)])


def key_points(k=KEY) -> np.ndarray:
    """The key's hex prism corners in its frame, for its grasp region."""
    s = k["s"]
    rc = s / np.sqrt(3)
    ang = np.radians(np.arange(6) * 60 + 30)
    hexagon = np.c_[rc * np.cos(ang), rc * np.sin(ang)]
    long = [[x, py, pz] for x in (0., k["long"] - s / 2) for py, pz in hexagon]
    short = [[k["long"] - s / 2 - pz, py, z] for z in (0., k["short"] - s / 2) for py, pz in hexagon]
    return np.array(long + short)
