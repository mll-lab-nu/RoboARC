"""Caliper sorting: a 150 mm vernier caliper fixed face up on a stand, its jaws out over the table, four sleeves that
differ only in their inside and outside diameters, a rack and four labelled bins.

Design record: runs/caliper_sorting_design_20261014 (README_v3.md: the hand-held caliper; README.md: the vernier reading
test, the bins; 2026-10-17, the user's decisions 4.1 / 4.2: the locking screw on the slider's face, the caliper's own
unit and zero, four sleeves of two classes; 2026-10-18, the user: the caliper on a stand, STAND_FEASIBILITY.md - a
pinch on the beam's tail could not carry the hand-held caliper). Every number here is the design's.

Frames. Mount frame: the ALOHA mount (table top z = 0, x from the left arm base toward the right one, y away from the
observation cameras); the task moves everything into the world (InterlockedClawsTask.finalize_spec's rule). Caliper frame
C: origin on the closed outside measuring faces, on the beam's centre line and mid-thickness; x along the beam toward its
tail, y toward the inside jaws, z out of the scale face. The slider link is C moved by the jaw opening s along x (the
prismatic joint `slider`, metres; s = the distance between the outside measuring faces, also the inside edges' span).

Caliper (Mitutoyo 530 series 0-150 mm, catalogue D-10): beam 229 x 16 x 3.5 mm with the main scale; outside jaws 40 mm
deep; knife-edge inside jaws 21 mm above the beam, edges round 0.3 mm; slider 72 x 30.2 mm with the long vernier (10
divisions over 19 units), thumb roller (does not turn); depth bar drawn only. The slider moves on its joint with 1.5 N
friction loss and 2 N s/m damping. Masses 130 g (beam) and 50 g (slider), assumed.

The caliper's unit (the user, 2026-10-17): its scale is not in millimetres. Per scene seed one main-scale division is k mm
(0.95-1.25) and the closed jaws read b units (10-60): at jaw opening s mm the vernier zero points at the reading b + s / k.
The vernier gives tenths of a unit. The bin labels name readings on this caliper ("ID 38.8 / OD 96.2"), no unit.

Locking screw (the user, 2026-10-17: on the slider's face, its axis normal to the scale face, a knurled knob 7 mm across and
8 mm high with a white index line on top, turned by a pinch from above; left of the vernier window, on the jaws' side): link `screw` on
the revolute joint `screw` (axis -C z: clockwise seen from the scale side tightens) between stops 0 (loose) and 45 deg,
friction loss 0.02 N m. The task's mechanism (topogym.tasks.caliper_sorting.lock) locks the slider where it stands by the
joint equality `slider_lock` once the screw is 20 deg or more from the loose stop and frees it at 10 deg or less; the
caliper starts locked with the screw on its tight stop and the jaws closed.

Sleeves: four per scene, of two classes (two sleeves each) that share one diameter's class; 20 mm long, 200 g; outside
66-74 mm, bore 7.5-9.4 mm; each sleeve's true size is its class's +-0.02 mm (uniform, inside and outside independent).
Collision: a ring of 192 boxes (outer faces on the outside diameter, inner faces on the inside one: polygon flats, 0.009 mm
short of the circle at the corners); drawn as a turned tube with 0.5 mm chamfers.

Textures: drawn here with numpy (lines, rectangles and a stroke font of the few glyphs needed), so the bytes depend on no
system font. Nyx addresses a URDF's visuals by link name, so every link has ONE textured visual mesh (a GLB whose texture
atlas holds the scale or label images and a row of colour swatches; as topogym.assets.geometry.lock_picking).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MM = 1e-3

# ---------------------------------------------------------------- caliper (frame C, mm)
BEAM = dict(x0=-20.0, x1=209.0, half_w=8.0, half_t=1.75)
OUT_JAW = dict(depth=40.0, root_w=18.0, tip_w=8.0, tip_len=10.0)
IN_JAW = dict(root_w=5.0, root_y1=16.0, tip_w=2.0, tip_y1=29.0, gap=0.1)    # tip 21 mm above the beam's edge
EDGE_R = 0.3                                    # inside measuring edge radius
SLIDER = dict(x0=3.0, x1=75.0, y0=-17.7, y1=12.5, z0=-4.5, z_top=2.05)       # vernier plate 0.3 mm above the beam face
ROLLER = dict(x=64.0, y=-29.7, r=6.0, half_h=4.0)
# the locking screw's knob on the slider's face (slider frame): axis through (x, y), from the vernier plate's top up h.
# Left of the vernier window, on the jaws' side (2026-10-17: right of it, slider x 70, the right hand pinching it from above
# met the left hand on the beam's tail at the outside-diameter opening, the gripper housings 38 mm into each other)
KNOB = dict(x=12.0, y=-8.85, r=3.5, h=8.0, mass_kg=0.002)
KNOB_TRAVEL_DEG, KNOB_FRICTION_NM = 45.0, 0.02          # between its stops; friction loss of its joint
KNOB_DAMPING = 1e-4                                     # N m s / rad
LOCK_DEG, RELEASE_DEG = 20.0, 10.0                      # from the loose stop: locks at >= 20, frees at <= 10 (hysteresis)
Z0 = 26.0                                       # where the vernier zero sits at closed jaws (C x)
LINE_W = 0.15                                   # graduation line width
VERNIER = dict(least_count=0.1, divisions=10, length_units=19.0)       # 10 divisions over 19 units: tenths of a unit
SLIDER_RANGE_M = (0.0, 0.152)
SLIDER_FRICTION_N, SLIDER_DAMPING_N_S_M = 1.5, 2.0
BEAM_MASS_KG, SLIDER_MASS_KG = 0.13, 0.05
# the slider's grip (slider frame, mm): where the hand pinching the slider's two side faces holds it (the caliper on its
# stand: the left hand, 2026-10-18; its grasp region is a sphere about this point)
GRIP_SLIDER = (44.0, -2.6)
# the slider's grasp region: a sphere about that grip, clear of the knob (a pinch on the knob never takes hold of the
# slider; the user, 2026-10-17)
SLIDER_GRASP_RADIUS_M = 0.020
# the caliper's unit (per scene seed): k mm per unit, b units at closed jaws
UNIT_MM_RANGE, ZERO_READING_RANGE = (0.95, 1.25), (10.0, 60.0)

# ---------------------------------------------------------------- sleeves (mm)
# classes by (inside, outside) class level, 0 the smaller reading: A (0, 0), B (1, 0), C (0, 1), D (1, 1)
CLASS_LEVELS = {"A": (0, 0), "B": (1, 0), "C": (0, 1), "D": (1, 1)}
CLASS_STEP = 0.3                                # units between a diameter's two class readings (3 vernier divisions)
OD_MM_RANGE, ID_MM_RANGE = (66.0, 74.0), (7.5, 9.4)     # every class size: wider than the gripper opens, no finger fits
N_SLEEVES = 4
SLEEVE_L, SLEEVE_MASS_KG, SLEEVE_N_BOX, CHAMFER, SIZE_DEV = 20.0, 0.200, 192, 0.5, 0.02
# every sleeve's grasp region: one radius for all (a radius per size would state the size in the task texts); the
# farthest point of a 74 mm sleeve is 38.3 mm from its centre
SLEEVE_GRASP_RADIUS_M = 0.040

# ---------------------------------------------------------------- fixtures (mm; mount frame relative to the work origin)
WORK_ORIGIN_M = (0.16, 0.02)                    # the design's base placement (the hand-held design's measuring plate)
# the caliper's stand (2026-10-18, STAND_FEASIBILITY.md): C origin at the work origin + offset (m), height above the table
# (mm); a block under the beam's tail, C x x0..x1 (the slider reaches C x 155 at 80 mm open), half_w either side of C y 0,
# from the table up to the beam's underside
STAND = dict(offset_m=(-0.18, 0.07), height=90.0, x0=165.0, x1=209.0, half_w=12.0)
MOUNT_PLAY_M, MOUNT_DAMPING = 1e-6, 10.0          # the beam's joint on the stand's clamp (build_caliper): its limits, damping
RACK = dict(pitch=44.0, n=6, floor=5.0, div_h=22.0, curb_x=42.0, curb_h=4.0)
BINS = dict(in_x=80.0, in_y=80.0, wall=3.0, h=40.0)
# the bin label card (drawn only): w x h mm tilted tilt_deg up from the horizontal toward cam_high; a front-row bin's
# hangs from its front wall (its top edge just in front of the wall's top), a back-row bin's stands behind its back wall
# on a post (its bottom edge back_gap mm behind the wall and back_z mm up: cam_high sees it over the bin's own sleeves,
# and it is over no other bin); text of cap height cap mm, strokes stroke x cap wide, track x cap between glyphs, black
# on white, ppm texture pixels per mm.
# 2026-10-19 (the user: readable in cam_high, 848 x 480, about 1.5 mm per pixel at the bins): 84 x 40 mm, 25 deg (facing
# cam_high within 18-26 deg instead of 35-44), cap 14 mm, strokes 0.17 cap with 0.15 cap more space between glyphs;
# before: 76 x 40 mm, 50 deg, cap 10 mm, strokes 0.13, no extra space.
LABEL = dict(w=84.0, h=40.0, tilt_deg=25.0, cap=14.0, stroke=0.17, track=0.15, ppm=24.0, back_gap=2.0, back_z=50.0)

# ---------------------------------------------------------------- colours (linear-ish RGB 0..1)
STEEL, DARK, KNURL, ROLLER_RGB = (0.80, 0.81, 0.83), (0.14, 0.14, 0.16), (0.35, 0.35, 0.37), (0.55, 0.56, 0.58)
KNOB_MARK = (0.97, 0.97, 0.97)                  # the knob's white index line
FIXTURE_RGB, BIN_RGB, SLEEVE_RGB = (0.25, 0.42, 0.55), (0.86, 0.87, 0.88), (0.80, 0.81, 0.83)
SCALE_BG, VERNIER_BG, INK = (205, 208, 212), (222, 224, 228), (12, 12, 12)
# The appearance parameters (CaliperSortingTask defaults fixture_finish, bin_finish, sleeve_finish; the user, 2026-10-20:
# a seed varies appearance only): the rack's and the stand's colour, the bins' and all four sleeves' (they stay alike);
# the first entry of each is the colour above. The caliper, its scale and the label cards keep theirs.
FIXTURE_FINISHES = dict(slate_blue=FIXTURE_RGB, forest_green=(0.20, 0.40, 0.28), safety_orange=(0.85, 0.45, 0.12),
                        wood=(0.62, 0.47, 0.32))
BIN_FINISHES = dict(light_grey=BIN_RGB, pale_blue=(0.62, 0.75, 0.88), pale_yellow=(0.92, 0.86, 0.55))
SLEEVE_FINISHES = dict(steel=SLEEVE_RGB, brass=(0.80, 0.64, 0.32), black_oxide=(0.18, 0.18, 0.20))


# ================================================================ scenario
def _class_size_mm(reading, k, b):
    return (reading - b) * k


def _class_readings(rng, k, b, mm_range):
    """A diameter's two class readings (one decimal, CLASS_STEP apart) whose sizes both lie in mm_range (mm)."""
    lo, hi = mm_range
    for _ in range(1000):
        r0 = round(b + rng.uniform(lo, hi) / k, 1)
        r1 = round(r0 + CLASS_STEP, 1)
        if lo <= _class_size_mm(r0, k, b) and _class_size_mm(r1, k, b) <= hi:
            return r0, r1
    raise RuntimeError("caliper_sorting: no class readings in range")


def scenario(seed: int) -> dict:
    """The scene of a scene seed. unit_mm k (mm per unit) and zero_reading b (the reading at closed jaws); per diameter
    two class readings 0.3 apart ("ID" the inside jaws' readings, "OD" the outside jaws'); the classes A-D (CLASS_LEVELS)
    as (ID, OD) readings; the four sleeves (rack slot, class, true inside and outside diameter in mm); the four bins'
    labels (bin k at the work origin + (0.06 + 0.09 (k % 2), -0.14 - 0.09 (k // 2)) m); the caliper's (and its stand's)
    shift from its nominal place (m, the key rest_dxy_m kept from the hand-held design's rest).
    The sleeves are two classes, two sleeves each, that share one diameter's class (the user, 2026-10-17: no rule makes
    every class appear, so sorting by relative size alone does not place them: a sorter that never reads the scale is
    right half the time). Seed 0 is the reference's scene: k 1.08, b 31.4; ID 38.8 / 39.1, OD 96.2 / 96.5 (7.99 / 8.32
    mm, 69.98 / 70.31 mm); classes A and B (both OD 96.2) in rack slots 0-3 as A B B A; labels C A D B; no shift. Other
    seeds draw k (0.95-1.25), b (10-60), the class readings (sizes 66-74 mm outside, 7.5-9.4 mm inside), which diameter
    the two classes share and its class, the four rack slots of six and the sleeves' order in them, the label order and a
    +-15 mm shift of the caliper. Sizes: class size +-0.02 mm."""
    if seed == 0:
        rng = np.random.default_rng(0)
        k, b = 1.08, 31.4
        readings = {"ID": (38.8, 39.1), "OD": (96.2, 96.5)}
        rack = [(0, "A"), (1, "B"), (2, "B"), (3, "A")]
        bins = ["C", "A", "D", "B"]
        rest = (0.0, 0.0)
    else:
        rng = np.random.default_rng(1000 + seed)
        k = float(rng.uniform(*UNIT_MM_RANGE))
        b = float(rng.uniform(*ZERO_READING_RANGE))
        readings = {"ID": _class_readings(rng, k, b, ID_MM_RANGE), "OD": _class_readings(rng, k, b, OD_MM_RANGE)}
        shared = int(rng.integers(2))                  # 0: the classes share the inside class, 1: the outside one
        level = int(rng.integers(2))
        pair = [c for c, lv in CLASS_LEVELS.items() if lv[shared] == level]
        classes = [str(c) for c in rng.permutation(pair + pair)]
        slots = sorted(int(x) for x in rng.choice(RACK["n"], N_SLEEVES, replace=False))
        rack = list(zip(slots, classes))
        bins = [str(c) for c in rng.permutation(list(CLASS_LEVELS))]
        rest = tuple(float(v) for v in rng.uniform(-0.015, 0.015, 2))
    table = {c: (readings["ID"][li], readings["OD"][lo]) for c, (li, lo) in CLASS_LEVELS.items()}
    sleeves = []
    for slot, c in rack:
        di, do = rng.uniform(-SIZE_DEV, SIZE_DEV, 2)
        sleeves.append(dict(cls=c, slot=int(slot), id_mm=float(_class_size_mm(table[c][0], k, b) + di),
                            od_mm=float(_class_size_mm(table[c][1], k, b) + do)))
    return dict(seed=int(seed), unit_mm=float(k), zero_reading=float(b), readings=readings,
                classes={c: list(v) for c, v in table.items()}, sleeves=sleeves, bins=bins, rest_dxy_m=rest)


def label_lines(scen: dict, cls: str) -> list[str]:
    i, o = scen["classes"][cls]
    return [f"ID {i:.1f}", f"OD {o:.1f}"]


def layout(scen: dict) -> dict:
    """Mount-frame positions (m) of the scene's parts for a scenario."""
    wx, wy = WORK_ORIGIN_M
    dx, dy = scen["rest_dxy_m"]
    rack_x = wx + 0.17
    ox, oy = STAND["offset_m"]
    return dict(
        caliper=np.array([wx + ox + dx, wy + oy + dy, STAND["height"] * MM]),   # C origin on its stand; beam toward -x
        rack=[np.array([rack_x, wy + 0.04 + k * RACK["pitch"] * MM]) for k in range(RACK["n"])],
        bins=[np.array([wx + 0.06 + (k % 2) * 0.09, wy - 0.14 - (k // 2) * 0.09]) for k in range(4)])


CALIPER_QUAT_WXYZ = (0.0, 0.0, 0.0, 1.0)        # C in the mount frame at the start: half a turn about z
SLEEVE_QUAT_WXYZ = (math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0)          # sleeve axis (local z) along y


def reading_units(s_mm: float, k: float, b: float) -> float:
    """The caliper's exact reading (units) at jaw opening s (mm)."""
    return b + s_mm / k


def vernier_reading(s_mm: float, k: float, b: float) -> float:
    """What a reader who never misreads sees on the vernier at jaw opening s (mm), in the caliper's units: the main
    scale's whole units left of the vernier zero plus the coinciding vernier line times the least count (0.1 unit)."""
    lc = VERNIER["least_count"]
    r = reading_units(s_mm, k, b)
    whole = math.floor(r)
    return round(whole + int(round((r - whole) / lc)) * lc, 1)


# ================================================================ drawing (numpy; image rows downward)
def _arc(cx, cy, rx, ry, a0, a1, n=12):
    a = np.radians(np.linspace(a0, a1, n + 1))
    return [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in a]


def _glyphs():
    """Stroke glyphs: polylines in units of the cap height (x right, y up), and the advance width."""
    g = {
        "0": ([_arc(.3, .5, .27, .5, 0, 360, 32)], .7),
        "1": ([[(.12, .78), (.33, 1.), (.33, 0.)]], .7),
        "2": ([_arc(.3, .72, .26, .27, 160, -30, 12) + [(.03, 0.), (.58, 0.)]], .7),
        "3": ([_arc(.29, .76, .24, .24, 150, -90, 12), _arc(.29, .26, .27, .26, 90, -150, 12)], .7),
        "4": ([[(.44, 0.), (.44, 1.), (.02, .3), (.6, .3)]], .7),
        "5": ([[(.54, 1.), (.1, 1.), (.06, .56)] + _arc(.3, .32, .27, .32, 140, -150, 14)], .7),
        "6": ([_arc(.3, .3, .27, .3, 0, 360, 24), _arc(.56, .32, .53, .66, 180, 105, 10)], .7),
        "7": ([[(.03, 1.), (.58, 1.), (.2, 0.)]], .7),
        "8": ([_arc(.3, .76, .23, .24, 0, 360, 24), _arc(.3, .26, .27, .26, 0, 360, 24)], .7),
        "I": ([[(0., 1.), (.24, 1.)], [(.12, 1.), (.12, 0.)], [(0., 0.), (.24, 0.)]], .46),
        "D": ([[(.05, 0.), (.05, 1.), (.28, 1.)] + _arc(.28, .5, .3, .5, 90, -90, 14) + [(.28, 0.), (.05, 0.)]], .72),
        "O": ([_arc(.34, .5, .32, .5, 0, 360, 32)], .8),
        ".": ([[(.06, .03), (.06, .05)]], .22),
        "/": ([[(0., 0.), (.35, 1.)]], .47),
        "m": ([[(.04, 0.), (.04, .6)], [(.04, .45)] + _arc(.2, .45, .16, .15, 180, 0, 8) + [(.36, 0.)],
               [(.36, .45)] + _arc(.52, .45, .16, .15, 180, 0, 8) + [(.68, 0.)]], .82),
        " ": ([], .35),
    }
    g["9"] = ([[(.6 - x, 1. - y) for x, y in line] for line in g["6"][0]], .7)
    return g


GLYPHS = _glyphs()
STROKE = 0.13                                   # stroke width, units of the cap height


class Canvas:
    """An RGB image in float, drawn in millimetres: (u, v) = (mm right of the left edge, mm down from the top edge)."""

    def __init__(self, w_mm, h_mm, ppm, rgb):
        self.ppm = float(ppm)
        self.w, self.h = int(round(w_mm * ppm)), int(round(h_mm * ppm))
        self.img = np.empty((self.h, self.w, 3), np.float64)
        self.img[:] = np.asarray(rgb, np.float64)

    def rect(self, u0, v0, u1, v1, rgb):
        """Axis-aligned rectangle, antialiased by its exact pixel coverage."""
        p = self.ppm
        x0, x1, y0, y1 = u0 * p, u1 * p, v0 * p, v1 * p
        c0, c1 = max(int(math.floor(x0)), 0), min(int(math.ceil(x1)), self.w)
        r0, r1 = max(int(math.floor(y0)), 0), min(int(math.ceil(y1)), self.h)
        if c1 <= c0 or r1 <= r0:
            return
        cols, rows = np.arange(c0, c1), np.arange(r0, r1)
        cx = np.clip(np.minimum(cols + 1, x1) - np.maximum(cols, x0), 0, 1)
        cy = np.clip(np.minimum(rows + 1, y1) - np.maximum(rows, y0), 0, 1)
        a = cy[:, None] * cx[None, :]
        patch = self.img[r0:r1, c0:c1]
        patch += a[..., None] * (np.asarray(rgb, np.float64) - patch)

    def stroke(self, points, width, rgb):
        """Polyline with round joins and caps, antialiased over one pixel."""
        p, r = self.ppm, width * self.ppm / 2
        pts = np.asarray(points, np.float64) * p
        for a, b in zip(pts[:-1], pts[1:]) if len(pts) > 1 else [(pts[0], pts[0])]:
            lo = np.floor(np.minimum(a, b) - r - 1).astype(int)
            hi = np.ceil(np.maximum(a, b) + r + 1).astype(int)
            lo = np.maximum(lo, 0)
            hi = np.minimum(hi, [self.w, self.h])
            if np.any(hi <= lo):
                continue
            X, Y = np.meshgrid(np.arange(lo[0], hi[0]) + .5, np.arange(lo[1], hi[1]) + .5)
            d = b - a
            t = np.zeros_like(X) if not d.any() else np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / (d @ d), 0, 1)
            dist = np.hypot(X - a[0] - t * d[0], Y - a[1] - t * d[1])
            cov = np.clip(r - dist + .5, 0, 1)
            patch = self.img[lo[1]:hi[1], lo[0]:hi[0]]
            patch += cov[..., None] * (np.asarray(rgb, np.float64) - patch)

    def text(self, s, u, v, cap_mm, rgb, anchor="left", stroke=STROKE, track=0.0):
        """Text with its baseline at v (mm down), starting at u (anchor left) or centred on u (anchor centre); strokes
        stroke x cap_mm wide, track x cap_mm extra space between glyphs."""
        width = (sum(GLYPHS[c][1] for c in s) + track * (len(s) - 1)) * cap_mm
        x = u - width / 2 if anchor == "centre" else u
        for c in s:
            lines, adv = GLYPHS[c]
            for line in lines:
                self.stroke([(x + px * cap_mm, v - py * cap_mm) for px, py in line], stroke * cap_mm, rgb)
            x += (adv + track) * cap_mm

    def pil(self):
        from PIL import Image
        return Image.fromarray(np.clip(np.rint(self.img), 0, 255).astype(np.uint8), "RGB")


def main_scale_canvas(k, b, ppm=30.0):
    """The beam face, x from BEAM x0 to x1 (u = x - x0), y from +8 (top row) to -8, in the caliper's unit (k mm per unit,
    b units at closed jaws): the line of the whole reading n at C x = Z0 + (n - b) k, from the centre line toward +y,
    2.5 / 3.5 / 4.5 mm long (n a multiple of 1 / 5 / 10), the multiples of 10 numbered near the +y edge."""
    B = BEAM
    c = Canvas(B["x1"] - B["x0"], 2 * B["half_w"], ppm, SCALE_BG)
    centre = B["half_w"]
    for n in range(math.ceil(b), math.floor(b + (B["x1"] - 3.0 - Z0) / k) + 1):
        u = Z0 + (n - b) * k - B["x0"]
        ln = 4.5 if n % 10 == 0 else 3.5 if n % 5 == 0 else 2.5
        c.rect(u - LINE_W / 2, centre - ln, u + LINE_W / 2, centre, INK)
        if n % 10 == 0 and u > 4.0:
            c.text(str(n), u, 2.6, 2.0, INK, anchor="centre")
    return c


def vernier_canvas(k, ppm=40.0):
    """The vernier plate, x (slider frame) from Z0 - 6 to Z0 + 6 + 19 k (u = x - (Z0 - 6)), y from 0 (top row) to -10:
    the 11 lines (19 k / 10 mm apart) hang from the top edge, long (3.5 mm) at 0, 5, 10 with numbers, 2.3 mm otherwise;
    the least count '0.1' (a tenth of the caliper's unit) at the right."""
    n, length = VERNIER["divisions"], VERNIER["length_units"] * k
    x0 = Z0 - 6.0
    c = Canvas(12.0 + length, 10.0, ppm, VERNIER_BG)
    for j in range(n + 1):
        u = Z0 + length / n * j - x0
        long_ = j % 5 == 0
        c.rect(u - LINE_W / 2, 0.0, u + LINE_W / 2, 3.5 if long_ else 2.3, INK)
        if long_:
            c.text(str(j), u, 5.9, 1.8, INK, anchor="centre")
    c.text(f"{VERNIER['least_count']:g}", 12.0 + length - 6.0, 9.0, 1.4, (50, 50, 50))
    return c


def label_canvas(lines, ppm=None):
    """A bin label: white card, black border, the lines (cap height LABEL cap, bold strokes) black, centred, filling the
    card's height."""
    Lb = LABEL
    w, h, cap = Lb["w"], Lb["h"], Lb["cap"]
    c = Canvas(w, h, Lb["ppm"] if ppm is None else ppm, (255, 255, 255))
    t = 1.0
    for u0, v0, u1, v1 in ((0.3, 0.3, w - 0.3, 0.3 + t), (0.3, h - 0.3 - t, w - 0.3, h - 0.3),
                           (0.3, 0.3, 0.3 + t, h - 0.3), (w - 0.3 - t, 0.3, w - 0.3, h - 0.3)):
        c.rect(u0, v0, u1, v1, (0, 0, 0))
    gap = (h - len(lines) * cap) / (len(lines) + 1)
    for i, s in enumerate(lines):
        c.text(s, w / 2, gap + (i + 1) * cap + i * gap, cap, (0, 0, 0), anchor="centre", stroke=Lb["stroke"],
               track=Lb["track"])
    return c


# ================================================================ meshes with a texture atlas
class Atlas:
    """One texture for one visual mesh: a row of colour swatches at the top, then the images stacked below it.
    uv() maps a point of an image (u, v in mm, as Canvas) to texture coordinates (trimesh's convention: v up)."""
    SW = 16

    def __init__(self, colors, canvases=()):
        self.colors = [tuple(float(c) for c in rgb) for rgb in colors]
        self.canvases = list(canvases)
        self.W = max([self.SW * len(self.colors)] + [c.w for c in self.canvases])
        self.rows, y = [], self.SW
        for c in self.canvases:
            self.rows.append(y)
            y += c.h
        self.H = y
        img = np.zeros((self.H, self.W, 3), np.uint8)
        img[:self.SW] = 128
        for k, rgb in enumerate(self.colors):
            img[:self.SW, k * self.SW:(k + 1) * self.SW] = np.clip(np.rint(np.asarray(rgb) * 255), 0, 255).astype(np.uint8)
        for c, y0 in zip(self.canvases, self.rows):
            img[y0:y0 + c.h, :c.w] = np.clip(np.rint(c.img), 0, 255).astype(np.uint8)
        from PIL import Image
        self.image = Image.fromarray(img, "RGB")

    def swatch(self, rgb):
        k = self.colors.index(tuple(float(c) for c in rgb))
        return ((k + .5) * self.SW / self.W, 1 - .5 * self.SW / self.H)

    def uv(self, index, u_mm, v_mm):
        c, y0 = self.canvases[index], self.rows[index]
        return (u_mm * c.ppm / self.W, 1 - (y0 + v_mm * c.ppm) / self.H)


class VisualMesh:
    """Triangles with per-vertex UVs into one Atlas, collected part by part (mm in, metres out)."""

    def __init__(self, atlas):
        self.atlas, self.v, self.f, self.uv = atlas, [], [], []

    def _add(self, verts, faces, uvs):
        n = len(self.v)
        self.v.extend(map(tuple, verts))
        self.f.extend([(a + n, b + n, c + n) for a, b, c in faces])
        self.uv.extend(uvs)

    def box(self, lo, hi, rgb):
        """An axis-aligned box (mm), each face its own four vertices (flat shading)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        uv = self.atlas.swatch(rgb)
        for axis in range(3):
            for side in (0, 1):
                a, b = [i for i in range(3) if i != axis]
                quad = []
                for ca, cb in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    p = np.empty(3)
                    p[axis] = hi[axis] if side else lo[axis]
                    p[a] = hi[a] if ca else lo[a]
                    p[b] = hi[b] if cb else lo[b]
                    quad.append(p)
                normal = np.zeros(3)
                normal[axis] = 1 if side else -1
                faces = [(0, 1, 2), (0, 2, 3)]
                if np.cross(quad[1] - quad[0], quad[2] - quad[0]) @ normal < 0:
                    faces = [(0, 2, 1), (0, 3, 2)]
                self._add(quad, faces, [uv] * 4)

    def quad_up(self, x0, x1, y0, y1, z, index, image_u0, image_v_top):
        """A face looking up +z (mm) carrying image `index`: image u = x - x0 + image_u0, image v = image_v_top - y
        (mm; image rows run toward -y)."""
        quad = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        uvs = [self.atlas.uv(index, x - x0 + image_u0, image_v_top - y) for x, y, _ in quad]
        self._add(quad, [(0, 1, 2), (0, 2, 3)], uvs)

    def mesh(self, m, rgb, transform=None):
        """A trimesh solid (mm), flat-shaded, in one colour."""
        m = m.copy()
        if transform is not None:
            m.apply_transform(transform)
        tri = np.asarray(m.vertices)[np.asarray(m.faces)].reshape(-1, 3)
        self._add(tri, [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(m.faces))], [self.atlas.swatch(rgb)] * len(tri))

    def export(self, path, metallic, roughness):
        import trimesh
        from trimesh.visual.material import PBRMaterial
        from trimesh.visual.texture import TextureVisuals
        tm = trimesh.Trimesh(vertices=np.asarray(self.v) * MM, faces=np.asarray(self.f), process=False)
        tm.visual = TextureVisuals(uv=np.asarray(self.uv), material=PBRMaterial(
            baseColorTexture=self.atlas.image, metallicFactor=float(metallic), roughnessFactor=float(roughness)))
        tm.export(path, include_normals=True)
        _name_glb(path, Path(path).stem)
        return tm


def _name_glb(path, stem):
    """Give the GLB's mesh, node, material, texture and image names of its own (trimesh leaves the material, texture and
    image unnamed). Nyx shares unnamed materials between the GLB files of a scene: a second file's texture was drawn
    with the first file's image."""
    import json
    import struct
    data = Path(path).read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    n, kind = struct.unpack_from("<II", data, 12)
    if magic != 0x46546C67 or kind != 0x4E4F534A:
        raise ValueError(f"{path}: not a GLB with a leading JSON chunk")
    doc = json.loads(data[20:20 + n])
    for key, suffix in (("meshes", "mesh"), ("nodes", "node"), ("materials", "material"), ("textures", "texture"),
                        ("images", "image")):
        for k, item in enumerate(doc.get(key, [])):
            item["name"] = f"{stem}_{suffix}" + (f"_{k}" if k else "")
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    rest = data[20 + n:]
    Path(path).write_bytes(struct.pack("<III", magic, version, 12 + 8 + len(text) + len(rest))
                           + struct.pack("<II", len(text), kind) + text + rest)


def _cylinder(r, h, sections, axis="z", center=(0, 0, 0)):
    import trimesh
    T = np.eye(4)
    if axis == "y":
        T[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    elif axis == "x":
        T[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    T[:3, 3] = center
    m = trimesh.creation.cylinder(radius=r, height=h, sections=sections)
    m.apply_transform(T)
    return m


def tube_mesh(ro, ri, length=SLEEVE_L, ch=CHAMFER, n=64):
    """The drawn sleeve (mm): a lathe profile with 45-degree chamfers on its four edges, axis local z, centred."""
    import trimesh
    prof = [(ri + ch, -length / 2), (ro - ch, -length / 2), (ro, -length / 2 + ch), (ro, length / 2 - ch),
            (ro - ch, length / 2), (ri + ch, length / 2), (ri, length / 2 - ch), (ri, -length / 2 + ch)]
    m = len(prof)
    verts, faces = [], []
    for k in range(n):
        a = 2 * math.pi * k / n
        verts += [(r * math.cos(a), r * math.sin(a), z) for r, z in prof]
    for k in range(n):
        k2 = (k + 1) % n
        for j in range(m):
            j2 = (j + 1) % m
            a, b, c, d = k * m + j, k * m + j2, k2 * m + j2, k2 * m + j
            faces += [(a, c, b), (a, d, c)]          # outward normals (the design's winding drew the tube inside out)
    return trimesh.Trimesh(np.asarray(verts, float), np.asarray(faces), process=False)


# ================================================================ URDF writing
def _fmt(values):
    return " ".join(repr(float(v)) for v in values)


class Link:
    """A URDF link: box / cylinder / sphere collisions (mm in), an inertial, one visual mesh file."""

    def __init__(self, name):
        self.name, self.collisions, self.inertial, self.visual = name, [], None, None

    def box(self, lo, hi, name=None):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        self.collisions.append(("box", (hi + lo) / 2 * MM, (0, 0, 0), dict(size=_fmt((hi - lo) * MM)), name))

    def cylinder(self, centre, r, length, rpy=(0, 0, 0), name=None):
        self.collisions.append(("cylinder", np.asarray(centre, float) * MM, rpy,
                                dict(radius=repr(r * MM), length=repr(length * MM)), name))

    def capsule(self, a, b, r, name=None):
        """A capsule from a to b (mm) as the cylinder and two end spheres (profile option capsules merges them back into
        one exact capsule)."""
        from scipy.spatial.transform import Rotation
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        rot, _ = Rotation.align_vectors([(b - a) / L], [[0, 0, 1]])        # +z onto the axis (URDF rpy: extrinsic xyz)
        self.cylinder((a + b) / 2, r, L, rot.as_euler("xyz"), name)
        for p in (a, b):
            self.collisions.append(("sphere", p * MM, (0, 0, 0), dict(radius=repr(r * MM)), None))

    def xml(self, root):
        link = ET.SubElement(root, "link", name=self.name)
        if self.inertial is not None:
            mass, com, diag = self.inertial
            node = ET.SubElement(link, "inertial")
            ET.SubElement(node, "origin", xyz=_fmt(com), rpy="0 0 0")
            ET.SubElement(node, "mass", value=repr(float(mass)))
            ET.SubElement(node, "inertia", ixx=repr(float(diag[0])), iyy=repr(float(diag[1])), izz=repr(float(diag[2])),
                          ixy="0", ixz="0", iyz="0")
        if self.visual is not None:
            node = ET.SubElement(link, "visual")
            ET.SubElement(ET.SubElement(node, "geometry"), "mesh", filename=str(Path(self.visual).resolve()))
        for kind, xyz, rpy, attrs, name in self.collisions:
            node = ET.SubElement(link, "collision", **({"name": name} if name else {}))
            ET.SubElement(node, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
            ET.SubElement(ET.SubElement(node, "geometry"), kind, **attrs)
        return link


def write_urdf(path, name, links, joints=()):
    root = ET.Element("robot", name=name)
    for link in links:
        link.xml(root)
    for j in joints:
        node = ET.SubElement(root, "joint", name=j["name"], type=j["type"])
        ET.SubElement(node, "parent", link=j["parent"])
        ET.SubElement(node, "child", link=j["child"])
        ET.SubElement(node, "origin", xyz=_fmt(np.asarray(j.get("xyz_mm", (0, 0, 0)), float) * MM), rpy="0 0 0")
        ET.SubElement(node, "axis", xyz=_fmt(j["axis"]))
        ET.SubElement(node, "limit", lower=repr(j["lower"]), upper=repr(j["upper"]), effort="100", velocity="1")
        ET.SubElement(node, "dynamics", damping=repr(j["damping"]), friction=repr(j["friction"]))
    ET.indent(root)
    Path(path).write_text(ET.tostring(root, encoding="unicode") + "\n")


# ================================================================ the parts
def _box_corners(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([[(hi if i & 1 else lo)[0], (hi if i & 2 else lo)[1], (hi if i & 4 else lo)[2]] for i in range(8)])


def build_caliper(folder: Path, k: float, b: float):
    """caliper.urdf: link beam (frame C), link slider on the prismatic joint slider, link screw (the locking screw's
    knob) on the revolute joint screw; the scale and the vernier in the caliper's unit (k mm per unit, b units at closed
    jaws). Returns (path, beam vertices in C (m), grasp regions {link: (centre (m), radius (m))})."""
    B, J, I, S, Rr, Kn = BEAM, OUT_JAW, IN_JAW, SLIDER, ROLLER, KNOB
    hw, ht = B["half_w"], B["half_t"]
    jaw_root_y = -hw - (J["depth"] - J["tip_len"])
    beam_boxes = [((B["x0"], -hw, -ht), (B["x1"], hw, ht)),                                     # beam
                  ((B["x0"], -hw, -ht), (0.0, 12.5, ht)),                                       # fixed head
                  ((-J["root_w"], jaw_root_y, -ht), (0.0, -hw, ht)),                            # fixed outside jaw root
                  ((-J["tip_w"], -hw - J["depth"], -ht), (0.0, jaw_root_y, ht)),                # its tip
                  ((0.0, hw, -ht), (I["root_w"], I["root_y1"], -I["gap"])),                     # fixed inside jaw root
                  ((EDGE_R, I["root_y1"], -0.25), (I["tip_w"], I["tip_y1"], 0.25))]             # its tip
    beam = Link("beam")
    for lo, hi in beam_boxes:
        beam.box(lo, hi)
    beam.capsule((EDGE_R, I["root_y1"] + EDGE_R, 0), (EDGE_R, I["tip_y1"] - EDGE_R, 0), EDGE_R)
    beam.inertial = (BEAM_MASS_KG, (0.09, 0.0, 0.0), (2e-6, 6e-4, 6e-4))
    slider_boxes = [((S["x0"], S["y0"], S["z0"]), (S["x1"], 0.0, S["z_top"])),                 # plate (vernier on top)
                    ((S["x0"], hw, S["z0"]), (S["x1"], S["y1"], ht)),                           # flange
                    ((S["x0"], 0.0, S["z0"]), (S["x1"], hw, -ht)),                              # bottom
                    ((0.0, jaw_root_y, -ht), (J["root_w"], -hw, ht)),                           # moving outside jaw root
                    ((0.0, -hw - J["depth"], -ht), (J["tip_w"], jaw_root_y, ht)),               # its tip
                    ((-I["root_w"], hw, I["gap"]), (6.0, I["root_y1"], ht)),                    # moving inside jaw root
                    ((-I["tip_w"], I["root_y1"], -0.25), (-EDGE_R, I["tip_y1"], 0.25)),         # its tip
                    ((Rr["x"] - 3, Rr["y"], -3.0), (Rr["x"] + 3, S["y0"] + 0.5, 3.0))]          # roller neck
    slider = Link("slider")
    for lo, hi in slider_boxes:
        slider.box(lo, hi)
    slider.capsule((-EDGE_R, I["root_y1"] + EDGE_R, 0), (-EDGE_R, I["tip_y1"] - EDGE_R, 0), EDGE_R)
    slider.cylinder((Rr["x"], Rr["y"], 0.0), Rr["r"], 2 * Rr["half_h"])
    slider.inertial = (SLIDER_MASS_KG, (0.04, -0.004, 0.0), (2e-5, 2e-5, 3e-5))
    # the knob: its link frame on the knob's axis at the vernier plate's top face, z out of the scale face
    knob = Link("screw")
    knob.cylinder((0.0, 0.0, Kn["h"] / 2), Kn["r"], Kn["h"])
    r_m, h_m = Kn["r"] * MM, Kn["h"] * MM
    knob.inertial = (Kn["mass_kg"], (0.0, 0.0, h_m / 2), (Kn["mass_kg"] * (3 * r_m ** 2 + h_m ** 2) / 12,
                                                            Kn["mass_kg"] * (3 * r_m ** 2 + h_m ** 2) / 12,
                                                            Kn["mass_kg"] * r_m ** 2 / 2))

    # visuals: beam with the main scale (three image rows of 76.33 mm), slider with the vernier, the knob
    scale = main_scale_canvas(k, b)
    third = (B["x1"] - B["x0"]) / 3
    pieces = []
    for k in range(3):
        c = Canvas(third, 2 * hw, scale.ppm, SCALE_BG)
        c.img[:] = scale.img[:, int(round(k * third * scale.ppm)):int(round(k * third * scale.ppm)) + c.w]
        pieces.append(c)
    atlas = Atlas([STEEL], pieces)
    vm = VisualMesh(atlas)
    for lo, hi in beam_boxes:
        vm.box(lo, hi, STEEL)
    vm.mesh(_cylinder(EDGE_R, I["tip_y1"] - I["root_y1"], 12, "y", (EDGE_R, (I["root_y1"] + I["tip_y1"]) / 2, 0)), STEEL)
    for k in range(3):
        vm.quad_up(B["x0"] + k * third, B["x0"] + (k + 1) * third, -hw, hw, ht + 0.01, k, 0.0, hw)
    vm.export(folder / "caliper_beam.glb", metallic=0.3, roughness=0.45)
    beam.visual = folder / "caliper_beam.glb"

    vern = vernier_canvas(k)
    atlas = Atlas([STEEL, DARK, KNURL, ROLLER_RGB], [vern])
    vm = VisualMesh(atlas)
    for lo, hi in slider_boxes[:-1]:
        vm.box(lo, hi, STEEL)
    vm.box(*slider_boxes[-1], DARK)
    vm.mesh(_cylinder(EDGE_R, I["tip_y1"] - I["root_y1"], 12, "y", (-EDGE_R, (I["root_y1"] + I["tip_y1"]) / 2, 0)), STEEL)
    vm.mesh(_cylinder(Rr["r"], 2 * Rr["half_h"], 40, "z", (Rr["x"], Rr["y"], 0.0)), ROLLER_RGB)
    import trimesh
    for k in range(20):                                         # knurls on the roller (drawn only)
        a = 2 * math.pi * k / 20
        knurl = trimesh.creation.box(extents=(0.7, 1.0, 2 * Rr["half_h"]))
        T = np.eye(4)
        T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        T[:3, 3] = (Rr["x"] + Rr["r"] * math.cos(a), Rr["y"] + Rr["r"] * math.sin(a), 0.0)
        vm.mesh(knurl, KNURL, T)
    vm.box((S["x1"], -1.0, -ht - 1.0), (B["x1"], 1.0, -ht - 0.05), STEEL)          # depth bar (drawn only)
    vx0 = Z0 - 6.0
    vm.quad_up(vx0, vx0 + vern.w / vern.ppm, -10.0, 0.0, S["z_top"] + 0.01, 0, 0.0, 0.0)
    vm.export(folder / "caliper_slider.glb", metallic=0.3, roughness=0.45)
    slider.visual = folder / "caliper_slider.glb"

    # the knob: a knurled cylinder (24 ribs) with a white index line on its top from the axis to the rim (link +x)
    atlas = Atlas([ROLLER_RGB, KNURL, KNOB_MARK])
    vm = VisualMesh(atlas)
    vm.mesh(_cylinder(Kn["r"] - 0.15, Kn["h"], 40, "z", (0.0, 0.0, Kn["h"] / 2)), ROLLER_RGB)
    for j in range(24):
        a = 2 * math.pi * j / 24
        rib = trimesh.creation.box(extents=(0.45, 0.6, Kn["h"] - 1.0))
        T = np.eye(4)
        T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        T[:3, 3] = ((Kn["r"] - 0.2) * math.cos(a), (Kn["r"] - 0.2) * math.sin(a), Kn["h"] / 2)
        vm.mesh(rib, KNURL, T)
    vm.box((0.0, -0.45, Kn["h"]), (Kn["r"] - 0.3, 0.45, Kn["h"] + 0.06), KNOB_MARK)
    vm.export(folder / "caliper_screw.glb", metallic=0.3, roughness=0.5)
    knob.visual = folder / "caliper_screw.glb"

    path = folder / "caliper.urdf"
    # the stand's clamp (2026-10-18): the caliper's root link, fixed to the world, and the beam on a joint its limits hold
    # within MOUNT_PLAY_M: MuJoCo filters a parent's contacts with its child only when the parent is not welded to the
    # world, and the slider wraps the beam (welded to the world, the beam collided with it 2-4 mm deep)
    mount = Link("mount")
    write_urdf(path, "caliper", [mount, beam, slider, knob], [
        dict(name="mount", type="prismatic", parent="mount", child="beam", axis=(0, 0, 1), lower=-MOUNT_PLAY_M,
             upper=MOUNT_PLAY_M, damping=MOUNT_DAMPING, friction=0.0),
        dict(name="slider", type="prismatic", parent="beam", child="slider", axis=(1, 0, 0), lower=SLIDER_RANGE_M[0],
             upper=SLIDER_RANGE_M[1], damping=SLIDER_DAMPING_N_S_M, friction=SLIDER_FRICTION_N),
        dict(name="screw", type="revolute", parent="slider", child="screw", xyz_mm=(Kn["x"], Kn["y"], S["z_top"]),
             axis=(0, 0, -1), lower=0.0, upper=math.radians(KNOB_TRAVEL_DEG), damping=KNOB_DAMPING, friction=KNOB_FRICTION_NM)])
    verts = np.concatenate([_box_corners(lo, hi) for lo, hi in beam_boxes]) * MM
    # grasp regions (contact_weld_v1 targets): each link whole, centred on its collision bounding box
    def region(boxes):
        pts = np.concatenate([_box_corners(lo, hi) for lo, hi in boxes]) * MM
        centre = (pts.min(0) + pts.max(0)) / 2
        return centre, float(np.linalg.norm(pts - centre, axis=1).max())
    regions = {"slider": (np.array([GRIP_SLIDER[0], GRIP_SLIDER[1], 0.0]) * MM, SLIDER_GRASP_RADIUS_M)}   # the beam is fixed
    return path, verts, regions


def build_sleeve(folder: Path, index: int, id_mm: float, od_mm: float, rgb=SLEEVE_RGB):
    """sleeve_<index>.urdf: one link `sleeve` (axis local z), 192-box ring collision, drawn in rgb. Returns (path, drawn
    vertices (m))."""
    ro, ri = od_mm / 2, id_mm / 2
    half_t = math.tan(math.pi / SLEEVE_N_BOX) * ro
    rc = (ro + ri) / 2
    link = Link("sleeve")
    for k in range(SLEEVE_N_BOX):
        a = 2 * math.pi * k / SLEEVE_N_BOX
        link.collisions.append(("box", np.array([rc * math.cos(a), rc * math.sin(a), 0.0]) * MM, (0.0, 0.0, a),
                                dict(size=_fmt(np.array([ro - ri, 2 * half_t, SLEEVE_L]) * MM)), None))
    m, r2 = SLEEVE_MASS_KG, ((ro * MM) ** 2 + (ri * MM) ** 2)
    ixx = m * (3 * r2 + (SLEEVE_L * MM) ** 2) / 12
    link.inertial = (m, (0.0, 0.0, 0.0), (ixx, ixx, m * r2 / 2))
    tube = tube_mesh(ro, ri)
    atlas = Atlas([rgb])
    vm = VisualMesh(atlas)
    vm.mesh(tube, rgb)
    vm.export(folder / f"sleeve_{index}.glb", metallic=0.2, roughness=0.5)
    link.visual = folder / f"sleeve_{index}.glb"
    path = folder / f"sleeve_{index}.urdf"
    write_urdf(path, f"sleeve_{index}", [link])
    return path, np.asarray(tube.vertices) * MM


def build_fixture(folder: Path, name: str, boxes, rgb, extra=None):
    """A fixed one-link URDF of boxes (mm, in the fixture's own frame), drawn in one colour; extra(vm) adds drawn-only
    parts (and the atlas images they need: extra = (canvases, callable))."""
    link = Link(name)
    for lo, hi in boxes:
        link.box(lo, hi)
    canvases, draw = extra if extra is not None else ((), None)
    atlas = Atlas([rgb, (0.2, 0.2, 0.2)], canvases)
    vm = VisualMesh(atlas)
    for lo, hi in boxes:
        vm.box(lo, hi, rgb)
    if draw is not None:
        draw(vm)
    vm.export(folder / f"{name}.glb", metallic=0.0, roughness=0.6)
    link.visual = folder / f"{name}.glb"
    path = folder / f"{name}.urdf"
    write_urdf(path, name, [link])
    return path


def rack_boxes():
    """In the rack's frame: x through the slots' centres, y = 0 at slot 0, slots at y = k * pitch."""
    R = RACK
    y_last = (R["n"] - 1) * R["pitch"]
    out = [((-48.0, -22.0, 0.0), (48.0, y_last + 22.0, R["floor"]))]
    for k in range(R["n"] + 1):
        y = -R["pitch"] / 2 + k * R["pitch"]
        out.append(((-30.0, y - 1.0, R["floor"]), (30.0, y + 1.0, R["floor"] + R["div_h"])))
    for sx in (-1, 1):
        out.append(((sx * R["curb_x"] - 2, -22.0, R["floor"]), (sx * R["curb_x"] + 2, y_last + 22.0, R["floor"] + R["curb_h"])))
    return out


def stand_boxes():
    """In the caliper's frame C: the stand's block under the beam's tail, from the table to the beam's underside."""
    S = STAND
    return [((S["x0"], -S["half_w"], -S["height"]), (S["x1"], S["half_w"], -BEAM["half_t"]))]


def bin_boxes():
    """In the bin's frame (centre of its floor's underside)."""
    Bn = BINS
    ix, iy, wt, h = Bn["in_x"] / 2, Bn["in_y"] / 2, Bn["wall"], Bn["h"]
    return [((-ix - wt, -iy - wt, 0.0), (ix + wt, iy + wt, wt)), ((-ix - wt, -iy - wt, wt), (-ix, iy + wt, h)),
            ((ix, -iy - wt, wt), (ix + wt, iy + wt, h)), ((-ix, -iy - wt, wt), (ix, -iy, h)), ((-ix, iy, wt), (ix, iy + wt, h))]


def _label_frame(back=False):
    """The label card's centre and rotation in its bin's frame (mm), the card tilted LABEL tilt_deg up from the
    horizontal (its +y edge the top). Front row: its top edge 1 mm in front of the bin's front wall and 2 mm above its
    top. Back row (back): its bottom edge back_gap mm behind the back wall and back_z mm above the floor's underside."""
    Bn, Lb = BINS, LABEL
    tilt = math.radians(Lb["tilt_deg"])
    up = np.array([0.0, math.cos(tilt), math.sin(tilt)])
    R = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]])
    if back:
        bottom = np.array([0.0, Bn["in_y"] / 2 + Bn["wall"] + Lb["back_gap"], Lb["back_z"]])
        return bottom + up * Lb["h"] / 2, R
    top = np.array([0.0, -Bn["in_y"] / 2 - Bn["wall"] - 1.0, Bn["h"] + 2.0])
    return top - up * Lb["h"] / 2, R


def label_card_corners(back=False):
    """The label card's corners in its bin's frame (mm): the texture's top left, top right, bottom right, bottom left
    as the card is read."""
    c, R = _label_frame(back)
    hw, hh = LABEL["w"] / 2, LABEL["h"] / 2
    return [c + R @ np.array([x, y, 0.0]) for x, y in ((-hw, hh), (hw, hh), (hw, -hh), (-hw, -hh))]


def _label_drawer(back=False):
    """The tilted label card of a bin (drawn only, as in the design), on a thin post: in front of a front-row bin, behind
    a back-row bin (back; LABEL, _label_frame)."""
    Lb = LABEL

    def draw(vm):
        c, R = _label_frame(back)
        hw, hh = Lb["w"] / 2, Lb["h"] / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        top = [c + R @ np.array([x, y, 0.0]) for x, y in corners]
        bottom = [c + R @ np.array([x, y, -0.8]) for x, y in corners]
        uv_top = [vm.atlas.uv(0, x + hw, hh - y) for x, y in corners]
        vm._add(top, [(0, 1, 2), (0, 2, 3)], uv_top)
        grey = vm.atlas.swatch((0.2, 0.2, 0.2))
        vm._add(bottom, [(0, 2, 1), (0, 3, 2)], [grey] * 4)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            vm._add([top[a], top[b], bottom[b], bottom[a]], [(0, 2, 1), (0, 3, 2)], [grey] * 4)
        low = c + R @ np.array([0.0, -hh, -0.8])
        vm.box((-1.5, low[1] - 1.5, 0.0), (1.5, low[1] + 1.5, low[2] + 0.5), (0.2, 0.2, 0.2))
    return draw


def caliper_lock_equality() -> dict:
    """The slider lock (EntitySpec morph_options joint_equalities): the slider held at polycoef[0] while active; born
    active at the closed jaws (the caliper starts locked); the task's mechanism writes polycoef[0] and switches it.
    Nearly hard, as bed_bolt's thread lock (docs/rcb/MUJOCO_CORE.md, task mechanisms: constraints that carry loads)."""
    return dict(name="slider_lock", joint1="slider", polycoef=[0.0, 0.0, 0.0, 0.0, 0.0], active=True,
                solref=[0.00125, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2.0])


def build_caliper_sorting(folder, seed: int, fixture_finish="slate_blue", bin_finish="light_grey", sleeve_finish="steel"):
    """Write every part's URDF and GLB into folder/caliper_sorting; return the scene description: entities (name, URDF,
    mount-frame position and quaternion, fixed, initial joint positions, morph options), the scenario, local vertices of
    the movable parts, grasp regions and the bins' centres (mount frame, m). The finishes (FIXTURE_FINISHES,
    BIN_FINISHES, SLEEVE_FINISHES) change only the GLB textures."""
    folder = Path(folder) / "caliper_sorting"
    folder.mkdir(parents=True, exist_ok=False)
    scen = scenario(seed)
    L = layout(scen)
    ents, local, regions = [], {}, {}
    cal_path, cal_verts, cal_regions = build_caliper(folder, scen["unit_mm"], scen["zero_reading"])
    ents.append(SimpleNamespace(name="caliper", path=cal_path, pos=tuple(float(v) for v in L["caliper"]), quat=CALIPER_QUAT_WXYZ,
                                fixed=True, joints={"mount": 0.0, "slider": 0.0, "screw": math.radians(KNOB_TRAVEL_DEG)},
                                options={"joint_equalities": [caliper_lock_equality()]}))
    local["caliper"] = cal_verts
    regions.update({("caliper", k): v for k, v in cal_regions.items()})
    for k, s in enumerate(scen["sleeves"]):
        path, verts = build_sleeve(folder, k, s["id_mm"], s["od_mm"], SLEEVE_FINISHES[sleeve_finish])
        rx, ry = L["rack"][s["slot"]]
        ents.append(SimpleNamespace(name=f"sleeve_{k}", path=path, pos=(float(rx), float(ry), (RACK["floor"] + s["od_mm"] / 2) * MM),
                                    quat=SLEEVE_QUAT_WXYZ, fixed=False, joints={}, options={}))
        local[f"sleeve_{k}"] = verts
        regions[(f"sleeve_{k}", "sleeve")] = (np.zeros(3), SLEEVE_GRASP_RADIUS_M)
    rx, ry = L["rack"][0]
    fixture_rgb = FIXTURE_FINISHES[fixture_finish]
    ents.append(SimpleNamespace(name="sleeve_rack", path=build_fixture(folder, "sleeve_rack", rack_boxes(), fixture_rgb),
                                pos=(float(rx), float(ry), 0.0), quat=(1.0, 0.0, 0.0, 0.0), fixed=True, joints={}, options={}))
    ents.append(SimpleNamespace(name="caliper_stand", path=build_fixture(folder, "caliper_stand", stand_boxes(), fixture_rgb),
                                pos=tuple(float(v) for v in L["caliper"]), quat=CALIPER_QUAT_WXYZ, fixed=True, joints={},
                                options={}))
    for k, cls in enumerate(scen["bins"]):
        bx, by = L["bins"][k]
        path = build_fixture(folder, f"bin_{k}", bin_boxes(), BIN_FINISHES[bin_finish],
                             extra=([label_canvas(label_lines(scen, cls))], _label_drawer(back=k // 2 == 0)))
        ents.append(SimpleNamespace(name=f"bin_{k}", path=path, pos=(float(bx), float(by), 0.0), quat=(1.0, 0.0, 0.0, 0.0),
                                    fixed=True, joints={}, options={}))
    return SimpleNamespace(entities=ents, scenario=scen, layout=L, local_vertices=local, grasp_regions=regions,
                           bins=[np.asarray(b, float) for b in L["bins"]])
