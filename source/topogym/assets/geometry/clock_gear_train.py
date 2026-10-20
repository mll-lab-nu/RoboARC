"""A watch movement's gear train enlarged 8 times: the MJCF of clock_assembling's one entity (coordinates in the ALOHA
mount frame).

Physical picture (runs/watch_assembly_design_20261014, README_v3.md; the user's reference video BV14bAjzEEcA 0:31-1:19, an
IWC calibre 81925): a brass main plate 304 mm across (38 mm real) lies 10 cm above the table on a holder. Four wheels stand
on a wooden stand: the barrel (a drum with 90 teeth and a square arbor), the center wheel (80 teeth, a 12-leaf pinion), the
third wheel (75, 10) and the fourth wheel (70, 10). The train bridge (six 20 mm bars, 12 mm thick) lies on four posts,
two screws stand in a tray, a thin probe (9 cm) lies on its stand. The wheels go onto their lower pivot holes in the plate
in the train's order (each pinion meshes the previous wheel's teeth; each wheel's disc lies above the discs it overlaps in
plan, so the order is forced), the bridge over them on two steady pins and three pillars, every upper pivot into its
bridge hole, two screws through the bridge into the pillars.

The rules (the task's mechanism, topogym.tasks.clock_assembling.mechanism) act on what this file declares:
  carriers   a body at each lower pivot hole, not colliding: two lean hinges (+-TILT_MAX, the pivot's play in its jewel)
             and a spin hinge; a seated wheel is welded to its carrier (weld seat_<wheel>, inactive at birth);
  meshing    joint equalities mesh_<wheel> between the spin hinges (polycoef[1] = the ratio; inactive at birth);
  gates      a short cylinder at each hole mouth (center, third, fourth), colliding with the wheels only while its
             wheel's teeth meet the previous wheel's tooth on tooth;
  fixing     welds screw<k>_fix (screw to bridge) and bridge_fix (bridge to the world), and joint equalities
             lock_<axis>_<wheel> on the lean hinges (all inactive at birth);
  run test   the barrel is a drum (its teeth, the lower pivot) and an arbor (its child body: the upper pivot, the ratchet,
             the square) joined by the mainspring hinge spring_barrel (its stiffness is the mainspring); the joint equality
             spring_lock holds it rigid until the bridge is fixed (active at birth); then the weld arbor_fix joins the
             arbor to its own carrier carr_barrel_a, whose hinge click_barrel is the click: the rules move its lower limit
             up to the largest angle reached less one ratchet tooth, so the arbor winds one way only and the mainspring
             drives the drum and the train.

Entity: one MJCF entity, "clock": the static movement, stands and holder (bodies without joints), the carriers, and the
free bodies barrel (the entity's base link), center, third, fourth, bridge, screw1, screw2, probe. One entity, because its
collision classes (contype / conaffinity, below) act only inside an entity (Genesis' MJCF semantics, mjcf_entity.py) and
its equalities connect bodies of one entity. Against the robot and the table every geom collides.
Collision classes (bits; a geom collides with the classes in its conaffinity, MuJoCo's OR rule; ten, so the remapped
entity leaves the robot's finger pads the free bits they need, robot.separate_finger_pads):
  TOOL 1      stands, holder, probe              WHEEL 2   discs, drums, pinion cores, arbors, the barrel's square
  PLATE 4     main plate, pillars, screw tray    PIVOT 8   lower and upper pivots, the barrel's bridge arbor
  BRIDGE 16   bridge bars and bosses             HOLE 32   pivot holes (plate and bridge), end stones, entry cones
  HEAD 64     screw heads                        PEG 128   steady pins, screw shanks
  PEGHOLE 256 the bridge's pin and screw holes, the pillars' and the tray's screw holes
  GATE 512    mesh gates
Collision geoms are in group 3 with coloured visual twins (contype 0, no mass), as bed_bolt: the renderer copies a
collision geom of groups 0-2 into a grey visual. Wheel discs and drums are their root circles and pinions their cores, so
two meshing wheels never touch; the teeth are visual only.
Frames: W, the movement frame: origin on the main plate's top face at the movement's centre, z up, axes as the mount
frame's, at P_W. A wheel's frame: origin on its axis at the plate face when seated, z up. The bridge's frame: W when the
bridge is seated (its underside at H_BRIDGE). A screw's: origin at the head's underside, the shank along -z. The probe's:
origin at the back end of the rod, the rod along +x to the tip."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ENTITY = "clock"
WHEELS = ("barrel", "center", "third", "fourth")
PARTS = WHEELS + ("bridge", "screw1", "screw2", "probe")
GRASP_LINKS = PARTS + ("barrel_arbor",)         # the barrel is carried and wound by its arbor's square
SCALE = 8.

# ---------------------------------------------------------------- the movement (W, metres)
# The design's layout in the mount frame, moved as the design's base-placement search chose (the whole scene by
# (-0.03, -0.06) m, the one placement of 25 where both arms reach the whole episode; README_v3.md 6.3).
OFFSET = np.array([-.03, -.06])
P_W = np.array([0., .12, .10]) + np.r_[OFFSET, 0.]

H_BRIDGE = .048           # bridge underside above the plate, seated (6 mm real)
T_BRIDGE = .012           # bridge thickness (1.5 mm)
PIVOT_R = .0010           # pivots 2.0 mm (0.25 mm)
PIVOT_HOLE_R = .0014      # jewel holes 2.8 mm
CHAMFER = .0012           # 45 deg entry cone under each bridge hole (a tip within 2.6 mm of the hole axis slides in)
LOW_PIVOT_L = .004        # lower pivot below the plate face
UP_PIVOT_L = .006         # upper pivot above the arbor's shoulder
SHOULDER_Z = H_BRIDGE - .0004    # the arbor's shoulder 0.4 mm under the bridge (end shake)
ARBOR_R = .0025           # graspable arbor 5 mm
TILT_MAX = .0262          # lean of a wheel standing on its lower pivot alone, per hinge (rad, 1.5 deg: the center
                          # wheel's disc edge, 34 mm out, then stays 0.1 mm clear of the barrel drum 1 mm under it)
COL_SHRINK = .0008        # collision discs / pinion cores this much inside the root circles

GEARS = {
    # teeth, module, wheel band z (W); pinion leaves, module, band z (it meshes the previous wheel's teeth). Each wheel's
    # disc lies 1 mm over the disc (the barrel: the drum) it overlaps in plan (0.125 mm real): with the 4 mm lower pivots
    # no wheel can be slid in under a wheel already seated, so the train goes together in one order only, barrel, center,
    # third, fourth (runs/watch_assembly_design_20261014/RUN_TEST_PROPOSAL.md, the order check)
    "barrel": dict(teeth=90, m=.0011, band=(.016, .020), drum=(.002, .025)),
    "center": dict(teeth=80, m=.0009, band=(.026, .028), pin=12, pm=.0011, pband=(.016, .020)),
    "third": dict(teeth=75, m=.0006, band=(.029, .031), pin=10, pm=.0009, pband=(.026, .028)),
    "fourth": dict(teeth=70, m=.0006, band=(.032, .034), pin=10, pm=.0006, pband=(.029, .031)),
}
PARENT = {"center": "barrel", "third": "center", "fourth": "third"}
MASS = {"barrel": .050, "center": .015, "third": .008, "fourth": .008}
COLORS = {"barrel": (.72, .72, .76, 1), "center": (.86, .68, .22, 1), "third": (.90, .72, .25, 1),
          "fourth": (.70, .72, .76, 1)}


def pitch_r(name):
    g = GEARS[name]
    return g["m"] * g["teeth"] / 2


def pin_pitch_r(name):
    g = GEARS[name]
    return g["pm"] * g["pin"] / 2


def _layout():
    pos = {"barrel": np.array([-.055, -.035])}
    for name, ang in (("center", 60.), ("third", 0.), ("fourth", 60.)):
        par = PARENT[name]
        a = math.radians(ang)
        pos[name] = pos[par] + (pitch_r(par) + pin_pitch_r(name)) * np.array([math.cos(a), math.sin(a)])
    return pos


POS = _layout()                                   # wheel axes in W
RATIO = {n: -GEARS[PARENT[n]]["teeth"] / GEARS[n]["pin"] for n in PARENT}     # child spin = ratio x parent spin
TOTAL_RATIO = RATIO["center"] * RATIO["third"] * RATIO["fourth"]               # fourth / barrel = -450

PILLARS = {"P1": np.array([-.065, .045]), "P2": np.array([.060, 0.]), "P3": np.array([-.110, -.070])}
SCREWED = ("P1", "P2")    # P3 carries only a steady pin: three feet make the unscrewed bridge sit flat
PILLAR_R = .011
PIN_OFF, SCREW_OFF = np.array([-.0045, 0.]), np.array([.0045, 0.])
PIN_R, PIN_TOP = .0020, H_BRIDGE + .008
SHANK_R, SHANK_L = .0025, .018
HEAD_R, HEAD_H = .0065, .011
SCREW_TURNS = 2.           # clockwise turns that tighten a screw seated on the bridge (no thread simulated)
BARREL_ARBOR_R = .0040
# the run test (the arbor above the bridge): a ratchet wheel 22 mm across on the arbor just over the bridge, the square
# above it; the click (a lever on the bridge, drawn) lets the arbor turn one way only, one tooth of 18 deg at a time
RATCHET_R, RATCHET_TEETH = .0095, 20        # the ratchet's collision disc (root circle); teeth drawn out to 11.2 mm
RATCHET_Z = (H_BRIDGE + T_BRIDGE + .0005, H_BRIDGE + T_BRIDGE + .0035)
CLICK_PITCH = 2 * math.pi / RATCHET_TEETH
SQUARE_BOTTOM = RATCHET_Z[1]
BARREL_SQUARE_TOP = SQUARE_BOTTOM + .012
BOSS_HOLE_R = .0125       # the bridge's barrel boss drawn as a ring around the ratchet (its collision cylinder stays)
SPRING_K = .002           # N m / rad: the mainspring (the arbor's hinge in the drum)
SPRING_DAMPING = 1e-5     # N m s / rad
CLICK_DAMPING = 1e-3      # N m s / rad on the arbor's turn (the click's spring and the ratchet's friction): a released
                          # arbor falls back onto the click at a few rad/s, not the 140 rad/s of a bare light arbor
ARBOR_MASS = .006
PLATE_R = .152

BRIDGE_PTS = {"B": POS["barrel"], "C": POS["center"], "T": POS["third"], "F": POS["fourth"],
              "P1": PILLARS["P1"], "P2": PILLARS["P2"], "P3": PILLARS["P3"]}
BRIDGE_BARS = (("P1", "C"), ("C", "B"), ("C", "T"), ("T", "F"), ("F", "P2"), ("B", "P3"))
BOSS_R = {"B": .022, "P1": PILLAR_R + .002, "P2": PILLAR_R + .002, "P3": PILLAR_R + .002}
BAR_W = .020
BRIDGE_MASS = .06
BRIDGE_GRASP = BRIDGE_PTS["F"] + .65 * (BRIDGE_PTS["P2"] - BRIDGE_PTS["F"])   # the bridge point over its stand's centre (W)

# ---------------------------------------------------------------- stands (mount frame)
LAYOUT = {k: tuple(float(c) for c in np.add(v, OFFSET)) for k, v in dict(
    wheel_stand=(-.25, -.02), bridge_stand=(.30, .20), screw_tray=(0., -.115), probe_stand=(-.24, .24)).items()}
STAND_TOP = .04
CUP_LIFT = LOW_PIVOT_L + .002     # a wheel rests in its cup with its lower pivot 2 mm above the stand top
CUP = {"barrel": (.020, .040, .002), "center": (.008, .014, .026), "third": (.006, .012, .029),
       "fourth": (.004, .010, .032)}         # cup inner / outer radius, height of the resting face in the wheel frame
WHEEL_STAND_POS = {"barrel": (-.055, 0.), "center": (.050, .030), "third": (.030, -.045), "fourth": (.088, -.035)}
BASE_SPIN = {"barrel": .3, "center": 1.1, "third": -.7, "fourth": 2.}      # plus the seed's turn (task)
BRIDGE_STAND_TOP = .06
SCREW_TRAY_TOP = .03
SCREW_TRAY_X = (-.022, .022)
PROBE_STAND_Z = .11
PROBE_L, PROBE_R, PROBE_TIP, PROBE_TIP_R = .09, .003, .012, .0012

# ---------------------------------------------------------------- collision classes and contact parameters
TOOL, WHEEL, PLATE, PIVOT, BRIDGE, HOLE, HEAD, PEG, PEGHOLE, GATE = (1 << k for k in range(10))
AFF = {TOOL: TOOL | WHEEL | PLATE | PIVOT | BRIDGE | HEAD | PEG,
       WHEEL: TOOL | WHEEL | PLATE,
       PLATE: TOOL | WHEEL | BRIDGE | HEAD,
       PIVOT: TOOL | HOLE,
       BRIDGE: TOOL | PLATE | HEAD,
       HOLE: PIVOT,
       HEAD: TOOL | PLATE | BRIDGE,
       PEG: TOOL | PEGHOLE,
       PEGHOLE: PEG,
       GATE: WHEEL}
# claws' task contacts (1.25 ms, dampratio 2, .9 / .99), the design's friction
CONTACT = dict(solref="0.00125 2", solimp="0.9 0.99 0.001 0.5 2", friction="0.3 0.005 0.0001", condim="3")
# hard joint limits and static friction (the shared lessons, docs/rcb/MUJOCO_CORE.md "Task mechanisms"): 2 dt, dampratio 2
HARD_LIMIT = dict(solreflimit="0.00125 2", solimplimit="0.99 0.999 0.001 0.5 2")
HARD_FRICTION = dict(solreffriction="0.00125 2", solimpfriction="0.99 0.999 0.001 0.5 2")
WELD_SOLREF = "0.002 1"   # the task's welds (seats, screws, the bridge's fixing), as in the design's runs

C = dict(plate=(.78, .55, .45, 1), brass=(.80, .66, .30, 1), steel=(.70, .72, .76, 1), ruby=(.75, .10, .35, 1),
         bridge=(.82, .58, .48, 1), dark=(.18, .18, .20, 1), wood=(.62, .48, .34, 1), tool=(.20, .45, .70, 1),
         red=(.90, .10, .10, 1), screw=(.35, .40, .70, 1))


def _v(x) -> str:
    return " ".join("%.9g" % float(c) for c in np.atleast_1d(x))


def quat_z(a):
    return (math.cos(a / 2), 0., 0., math.sin(a / 2))


def _quat_zy(a, b):
    """wxyz of a turn about z by a, then about the new y by b."""
    cz, sz, cy, sy = math.cos(a / 2), math.sin(a / 2), math.cos(b / 2), math.sin(b / 2)
    return (cz * cy, -sz * sy, cz * sy, sz * cy)


class _Body:
    """Geoms of one MJCF body: geom() a collision geom (group 3) of a class with its coloured visual twin unless hidden,
    look() a visual-only geom."""

    def __init__(self, el):
        self.el = el

    @staticmethod
    def _shape(kind, pos, quat, attrs):
        shape = {k: (_v(v) if not isinstance(v, str) else v) for k, v in attrs.items()}
        if "fromto" in shape:              # a capsule's ends set its pose
            return dict(type=kind, **shape)
        return dict(type=kind, pos=_v(pos), quat=_v(quat), **shape)

    def geom(self, name, kind, pos, cls, rgba, *, quat=(1, 0, 0, 0), hidden=False, **attrs):
        ET.SubElement(self.el, "geom", name=name, contype=str(cls), conaffinity=str(AFF[cls]), group="3", mass="0",
                      rgba="1 1 1 0", **CONTACT, **self._shape(kind, pos, quat, attrs))
        if not hidden:
            self.look(f"{name}_v", kind, pos, rgba, quat=quat, **attrs)

    def look(self, name, kind, pos, rgba, *, quat=(1, 0, 0, 0), **attrs):
        ET.SubElement(self.el, "geom", name=name, rgba=_v(rgba), contype="0", conaffinity="0", group="2", mass="0",
                      **self._shape(kind, pos, quat, attrs))

    def ring(self, prefix, c, r_in, r_out, z0, z1, n, cls, rgba, hidden=False):
        rm, w = (r_in + r_out) / 2, (r_out - r_in) / 2
        ht = rm * math.tan(math.pi / n) * 1.02
        for i in range(n):
            a = 2 * math.pi * (i + .5) / n
            self.geom(f"{prefix}{i}", "box", (c[0] + rm * math.cos(a), c[1] + rm * math.sin(a), (z0 + z1) / 2), cls, rgba,
                      quat=quat_z(a), size=(w, ht, (z1 - z0) / 2), hidden=hidden)

    def cone(self, prefix, c, r_in, depth, n, cls):
        """A 45 deg entry cone under a hole: n thin boxes whose inner faces run from radius r_in + depth at z = 0 up to
        r_in at z = depth (thin, so their outer lower corners stay inside the pad face: no lip under the bridge)."""
        t = .0003
        L = depth * math.sqrt(2)
        rm, zm = r_in + depth / 2 + t / 2 / math.sqrt(2), depth / 2 - t / 2 / math.sqrt(2)
        ht = (r_in + depth) * math.tan(math.pi / n) * 1.05
        for i in range(n):
            a = 2 * math.pi * (i + .5) / n
            self.geom(f"{prefix}{i}", "box", (c[0] + rm * math.cos(a), c[1] + rm * math.sin(a), zm), cls, (0, 0, 0, 0),
                      quat=_quat_zy(a, -math.pi / 4), size=(t / 2, ht, L / 2), hidden=True)

    def teeth(self, prefix, n, r_pitch, m, z0, z1, rgba):
        """Visual teeth: n boxes standing out of the root circle (2.25 m high, about half the pitch wide)."""
        h, w = 2.25 * m, math.pi * r_pitch / n * .55
        for i in range(n):
            a = 2 * math.pi * i / n
            rm = r_pitch - .25 * m
            self.look(f"{prefix}{i}", "box", (rm * math.cos(a), rm * math.sin(a), (z0 + z1) / 2), rgba, quat=quat_z(a),
                      size=(h / 2, w, (z1 - z0) / 2))


def _body(parent, name, pos, quat=(1, 0, 0, 0)):
    return ET.SubElement(parent, "body", name=name, pos=_v(pos), quat=_v(quat))


# ---------------------------------------------------------------- the static movement and stands
def _movement(world):
    o = P_W
    b = _Body(_body(world, "movement", o))         # W frame
    # the plate collides as a box (a 30 cm flat cylinder against thin wheel discs gave false deep contacts), drawn round
    b.look("plate_vis", "cylinder", (0, 0, -.006), C["plate"], size=(PLATE_R, .006))
    b.geom("plate", "box", (0, 0, -.006), PLATE, C["plate"], size=(PLATE_R * .75, PLATE_R * .75, .006), hidden=True)
    b.geom("holder", "box", (0, 0, -(o[2] - .012) / 2 - .012), TOOL, C["brass"], size=(.05, .05, (o[2] - .012) / 2))
    for name in WHEELS:
        p = POS[name]
        b.ring(f"hole_{name}_", p, PIVOT_HOLE_R, PIVOT_HOLE_R + .002, -LOW_PIVOT_L - .001, 0., 10, HOLE, C["ruby"],
               hidden=True)
        b.geom(f"endstone_{name}", "cylinder", (p[0], p[1], -LOW_PIVOT_L - .0005), HOLE, C["ruby"],
               size=(PIVOT_HOLE_R + .0005, .0005), hidden=True)
        if name != "barrel":       # mesh gate: the arbor's flat bottom lands on it while the teeth meet tip on tip
            b.geom(f"gate_{name}", "cylinder", (p[0], p[1], LOW_PIVOT_L / 2), GATE, (1, 0, 0, 0),
                   size=(.0035, LOW_PIVOT_L / 2), hidden=True)
        b.look(f"jewel_{name}", "cylinder", (p[0], p[1], .00005), C["ruby"], size=(PIVOT_HOLE_R + .0012, .00005))
    for k, p in PILLARS.items():
        b.geom(f"pillar_{k}", "cylinder", (p[0], p[1], H_BRIDGE / 2), PLATE, C["plate"], size=(PILLAR_R, H_BRIDGE / 2))
        pp = p + PIN_OFF
        b.geom(f"pin_{k}", "capsule", (0, 0, 0), PEG, C["steel"], size=PIN_R,
               fromto=(pp[0], pp[1], H_BRIDGE, pp[0], pp[1], PIN_TOP - PIN_R))
        if k in SCREWED:
            sp = p + SCREW_OFF
            b.ring(f"phole_{k}_", sp, SHANK_R + .0003, SHANK_R + .0025, H_BRIDGE - .012, H_BRIDGE, 10, PEGHOLE, C["dark"],
                   hidden=True)
            b.look(f"phole_{k}_top", "cylinder", (sp[0], sp[1], H_BRIDGE + .00005), C["dark"], size=(SHANK_R + .0005, .00005))


def _carriers(world):
    """A carrier at each lower pivot hole (no collision): lean hinges tx, ty about the pivot, then the spin hinge."""
    point = dict(pos="0 0 0", mass="0.001", diaginertia="1e-8 1e-8 1e-8")
    for name in WHEELS:
        p = P_W + np.r_[POS[name], -.002]
        # armature about the wheel's own inertia about the pivot: a nearly massless carrier makes its limits and the seat
        # weld soft (the bed_bolt lesson)
        a_tilt = MASS[name] * (.02 ** 2 + .3 * pitch_r(name) ** 2)
        a_spin = .5 * MASS[name] * pitch_r(name) ** 2
        body = _body(world, f"carr_{name}", p)
        ET.SubElement(body, "inertial", **point)
        lean = dict(type="hinge", range=_v([-TILT_MAX, TILT_MAX]), damping="0.002", frictionloss="0.00002",
                    armature=_v(a_tilt), **HARD_LIMIT, **HARD_FRICTION)
        ET.SubElement(body, "joint", name=f"tx_{name}", axis="1 0 0", **lean)
        body_y = ET.SubElement(body, "body", name=f"carr_{name}_y")
        ET.SubElement(body_y, "inertial", **point)
        ET.SubElement(body_y, "joint", name=f"ty_{name}", axis="0 1 0", **lean)
        body_s = ET.SubElement(body_y, "body", name=f"carr_{name}_s")
        ET.SubElement(body_s, "inertial", **point)
        # the barrel's drag: the free train must not creep
        ET.SubElement(body_s, "joint", name=f"spin_{name}", type="hinge", axis="0 0 1", damping="1e-6",
                      frictionloss=_v(2e-5 if name == "barrel" else 1e-7), armature=_v(a_spin), **HARD_FRICTION)
        if name == "barrel":   # the arbor's carrier: its hinge is the click (the rules move its lower limit)
            body_a = ET.SubElement(body_y, "body", name="carr_barrel_a")
            ET.SubElement(body_a, "inertial", **point)
            ET.SubElement(body_a, "joint", name="click_barrel", type="hinge", axis="0 0 1", limited="true",
                          range="-100 100", damping=_v(CLICK_DAMPING), armature=_v(ARBOR_MASS * BARREL_ARBOR_R ** 2),
                          **HARD_LIMIT)


def _stands(world):
    b = _Body(_body(world, "stands", (0, 0, 0)))
    x, y = LAYOUT["wheel_stand"]
    b.geom("wheel_stand", "box", (x, y, STAND_TOP / 2), TOOL, C["wood"], size=(.11, .065, STAND_TOP / 2))
    for name, (dx, dy) in WHEEL_STAND_POS.items():          # a cup under each wheel's disc (the barrel's drum)
        r_in, r_out, z_rest = CUP[name]
        b.ring(f"cup_{name}_", (x + dx, y + dy), r_in, r_out, STAND_TOP, STAND_TOP + z_rest + CUP_LIFT, 12, TOOL, C["wood"])
    bp = opening_poses()["bridge"][0]
    for k in ("B", "P1", "P2", "P3"):
        q = BRIDGE_PTS[k]
        b.geom(f"bridge_post_{k}", "cylinder", (bp[0] + q[0], bp[1] + q[1], BRIDGE_STAND_TOP / 2), TOOL, C["wood"],
               size=(.008, BRIDGE_STAND_TOP / 2))
    x, y = LAYOUT["screw_tray"]
    # PLATE class: the screw heads rest on it, the shanks pass into its holes
    b.geom("screw_tray", "box", (x, y, SCREW_TRAY_TOP / 2), PLATE, C["wood"], size=(.040, .025, SCREW_TRAY_TOP / 2))
    for k, dx in zip(("s1", "s2"), SCREW_TRAY_X):
        b.ring(f"thole_{k}_", (x + dx, y), SHANK_R + .0004, SHANK_R + .003, SCREW_TRAY_TOP - SHANK_L - .001,
               SCREW_TRAY_TOP + .0001, 10, PEGHOLE, C["dark"], hidden=True)
    x, y = LAYOUT["probe_stand"]
    for dx in (.035, PROBE_L - .01):                 # two supports under the rod, clear of the grasp at its back end
        b.geom(f"probe_v{int(round(dx * 1000))}", "box", (x - .09 + dx, y, (PROBE_STAND_Z - PROBE_R) / 2), TOOL, C["wood"],
               size=(.008, .012, (PROBE_STAND_Z - PROBE_R) / 2))


# ---------------------------------------------------------------- free bodies
def _barrel_arbor(barrel, z_drum):
    """The barrel's arbor: a child body of the drum on the mainspring hinge (stiffness SPRING_K, held rigid by the joint
    equality spring_lock until the bridge is fixed): the arbor between drum and bridge, the upper pivot, the ratchet just
    above the bridge and the square."""
    el = ET.SubElement(barrel, "body", name="barrel_arbor", pos="0 0 0")
    zc = .5 * (z_drum + BARREL_SQUARE_TOP)
    ET.SubElement(el, "inertial", pos=_v([0, 0, zc]), mass=_v(ARBOR_MASS),
                  diaginertia=_v([ARBOR_MASS * (BARREL_SQUARE_TOP - z_drum) ** 2 / 12] * 2 + [ARBOR_MASS * RATCHET_R ** 2 / 4]))
    ET.SubElement(el, "joint", name="spring_barrel", type="hinge", axis="0 0 1", stiffness=_v(SPRING_K), springref="0",
                  damping=_v(SPRING_DAMPING), armature=_v(ARBOR_MASS * BARREL_ARBOR_R ** 2))
    b = _Body(el)
    b.geom("barrel_arbor", "cylinder", (0, 0, (z_drum + H_BRIDGE) / 2), WHEEL, C["steel"],
           size=(BARREL_ARBOR_R, (H_BRIDGE - z_drum) / 2))
    b.geom("barrel_upper", "capsule", (0, 0, 0), PIVOT, C["steel"], size=BARREL_ARBOR_R,
           fromto=(0, 0, H_BRIDGE - .002, 0, 0, H_BRIDGE + T_BRIDGE - BARREL_ARBOR_R))
    r0, r1 = RATCHET_Z
    b.geom("barrel_ratchet", "cylinder", (0, 0, (r0 + r1) / 2), WHEEL, C["steel"], size=(RATCHET_R, (r1 - r0) / 2))
    b.teeth("ratchet_t", RATCHET_TEETH, RATCHET_R + .001, .0008, r0, r1, C["steel"])
    b.look("ratchet_mark", "cylinder", (.6 * RATCHET_R, 0, r1 + .0002), C["red"], size=(.0012, .0002))
    b.geom("barrel_square", "box", (0, 0, (SQUARE_BOTTOM + BARREL_SQUARE_TOP) / 2), WHEEL, C["steel"],
           size=(.0032, .0032, (BARREL_SQUARE_TOP - SQUARE_BOTTOM) / 2))


def _wheel(world, name, pos, quat):
    g, rgba = GEARS[name], COLORS[name]
    el = _body(world, name, pos, quat)
    ET.SubElement(el, "freejoint", name=f"{name}_free")
    m, r = MASS[name], pitch_r(name)
    ET.SubElement(el, "inertial", pos="0 0 0.02", mass=_v(m),
                  diaginertia=_v([max(m * 3e-4, m * r ** 2 * .3)] * 2 + [m * r ** 2 / 2]))
    b = _Body(el)
    b.geom(f"{name}_lowpivot", "capsule", (0, 0, 0), PIVOT, C["steel"], size=PIVOT_R,
           fromto=(0, 0, -LOW_PIVOT_L + PIVOT_R, 0, 0, .0005))
    if name == "barrel":
        z0, z1 = g["drum"]
        b.geom("barrel_drum", "cylinder", (0, 0, (z0 + z1) / 2), WHEEL, rgba, size=(r - 1.25 * g["m"] - COL_SHRINK, (z1 - z0) / 2))
        b.look("barrel_cover", "cylinder", (0, 0, z1 + .00025), C["steel"], size=(r * .7, .00025))
        b.teeth("barrel_t", g["teeth"], r, g["m"], g["band"][0], g["band"][1], rgba)
        mark_x, mark_z = .5 * r, z1 + .0006
        _barrel_arbor(el, z1)
    else:
        z0, z1 = g["band"]
        # the arbor starts 0.8 mm above the plate face (end shake): only the lower pivot touches the jewel, so a wheel
        # without its bridge leans freely on that pivot
        b.geom(f"{name}_arbor", "cylinder", (0, 0, (SHOULDER_Z + .0008) / 2), WHEEL, C["steel"],
               size=(ARBOR_R, (SHOULDER_Z - .0008) / 2))
        b.geom(f"{name}_uppivot", "cylinder", (0, 0, SHOULDER_Z + (UP_PIVOT_L - PIVOT_R) / 2), PIVOT, C["steel"],
               size=(PIVOT_R, (UP_PIVOT_L - PIVOT_R) / 2))
        b.geom(f"{name}_tip", "sphere", (0, 0, SHOULDER_Z + UP_PIVOT_L - PIVOT_R), PIVOT, C["steel"], size=PIVOT_R)
        rr = r - 1.25 * g["m"]
        b.geom(f"{name}_disc", "cylinder", (0, 0, (z0 + z1) / 2), WHEEL, rgba, size=(rr - COL_SHRINK, (z1 - z0) / 2))
        b.look(f"{name}_hub", "cylinder", (0, 0, (z0 + z1) / 2), C["steel"], size=(ARBOR_R * 2.2, (z1 - z0) / 2 + .001))
        k_win = 5 if name != "fourth" else 4                 # the spokes' cut-outs drawn as dark windows
        for k in range(k_win):
            a = 2 * math.pi * (k + .5) / k_win
            b.look(f"{name}_win{k}", "box", (.58 * rr * math.cos(a), .58 * rr * math.sin(a), z1 + .00005), (.15, .12, .08, 1),
                   quat=quat_z(a), size=(.22 * rr, .16 * rr, .00005))
        b.teeth(f"{name}_t", g["teeth"], r, g["m"], z0, z1, rgba)
        p0, p1 = g["pband"]
        pr = pin_pitch_r(name) - 1.25 * g["pm"]
        b.geom(f"{name}_pinion", "cylinder", (0, 0, (p0 + p1) / 2), WHEEL, C["steel"], size=(pr - COL_SHRINK / 2, (p1 - p0) / 2))
        b.teeth(f"{name}_p", g["pin"], pin_pitch_r(name), g["pm"], p0, p1, C["steel"])
        mark_x, mark_z = .6 * r, z1 + .0002
    b.look(f"{name}_mark", "cylinder", (mark_x, 0, mark_z), C["red"], size=(.0015, .0002))   # shows the spin


def _bridge(world, pos, quat):
    el = _body(world, "bridge", pos, quat)
    ET.SubElement(el, "freejoint", name="bridge_free")
    ET.SubElement(el, "inertial", pos="-0.01 0.01 0.006", mass=_v(BRIDGE_MASS), diaginertia="2e-4 2e-4 4e-4")
    b = _Body(el)
    for i, (pa, pb) in enumerate(BRIDGE_BARS):
        a, c = BRIDGE_PTS[pa], BRIDGE_PTS[pb]
        mid, d = (a + c) / 2, c - a
        yaw = math.atan2(d[1], d[0])
        b.geom(f"bridge_bar{i}", "box", (mid[0], mid[1], T_BRIDGE / 2), BRIDGE, C["bridge"], quat=quat_z(yaw),
               size=(np.linalg.norm(d) / 2, BAR_W / 2, T_BRIDGE / 2), hidden="B" in (pa, pb))
        if "B" in (pa, pb):        # drawn from the barrel boss's opening on (the ratchet rises through it)
            u = d / np.linalg.norm(d)
            a2, c2 = (a + u * BOSS_HOLE_R, c) if pa == "B" else (a, c - u * BOSS_HOLE_R)
            m2 = (a2 + c2) / 2
            b.look(f"bridge_bar{i}_v", "box", (m2[0], m2[1], T_BRIDGE / 2), C["bridge"], quat=quat_z(yaw),
                   size=(np.linalg.norm(c2 - a2) / 2, BAR_W / 2, T_BRIDGE / 2))
    for k, r in BOSS_R.items():
        p = BRIDGE_PTS[k]
        b.geom(f"bridge_boss_{k}", "cylinder", (p[0], p[1], T_BRIDGE / 2), BRIDGE, C["bridge"], size=(r, T_BRIDGE / 2),
               hidden=k == "B")
    p = BRIDGE_PTS["B"]            # the barrel boss drawn as a ring (its opening is wider than the ratchet) ...
    n_seg = 28
    rm, w = .5 * (BOSS_HOLE_R + BOSS_R["B"]), .5 * (BOSS_R["B"] - BOSS_HOLE_R)
    for i in range(n_seg):
        a = 2 * math.pi * (i + .5) / n_seg
        b.look(f"bridge_bossring_{i}", "box", (p[0] + rm * math.cos(a), p[1] + rm * math.sin(a), T_BRIDGE / 2), C["bridge"],
               quat=quat_z(a), size=(w, rm * math.tan(math.pi / n_seg) * 1.04, T_BRIDGE / 2))
    # ... and the click on it: a lever whose tip rests on the ratchet's teeth (drawn only; the click is a joint limit)
    a = math.radians(-35)
    tip = p + (RATCHET_R + .0012) * np.array([math.cos(a), math.sin(a)])
    piv = tip + np.array([.006, -.010])
    mid, d = (tip + piv) / 2, piv - tip
    b.look("click", "box", (mid[0], mid[1], T_BRIDGE + .0015), C["brass"], quat=quat_z(math.atan2(d[1], d[0])),
           size=(np.linalg.norm(d) / 2, .0018, .0015))
    b.look("click_screw", "cylinder", (piv[0], piv[1], T_BRIDGE + .0015), C["steel"], size=(.0022, .0016))
    for name in ("center", "third", "fourth"):
        p = POS[name]
        b.ring(f"pad_{name}_", p, PIVOT_HOLE_R, .010, CHAMFER, .008, 12, HOLE, C["bridge"], hidden=True)
        b.ring(f"padlow_{name}_", p, PIVOT_HOLE_R + CHAMFER, .010, -.0003, CHAMFER, 12, HOLE, C["bridge"], hidden=True)
        b.cone(f"cham_{name}_", p, PIVOT_HOLE_R, CHAMFER, 12, HOLE)
        b.look(f"bjewel_{name}", "cylinder", (p[0], p[1], T_BRIDGE + .00005), C["ruby"], size=(PIVOT_HOLE_R + .0015, .00005))
    p = POS["barrel"]
    b.ring("pad_barrel_", p, BARREL_ARBOR_R + .0005, .014, -.0005, T_BRIDGE, 14, HOLE, C["bridge"], hidden=True)
    for k, q in PILLARS.items():
        pp = q + PIN_OFF
        b.ring(f"bpin_{k}_", pp, PIN_R + .0003, PIN_R + .0025, -.0005, .009, 10, PEGHOLE, C["bridge"], hidden=True)
        if k in SCREWED:
            sp = q + SCREW_OFF
            b.ring(f"bshole_{k}_", sp, SHANK_R + .0003, SHANK_R + .0025, -.0005, T_BRIDGE + .0005, 10, PEGHOLE, C["bridge"],
                   hidden=True)
            b.look(f"bshole_{k}_top", "cylinder", (sp[0], sp[1], T_BRIDGE + .00005), C["dark"], size=(SHANK_R + .0005, .00005))


def _screw(world, name, pos, quat):
    el = _body(world, name, pos, quat)
    ET.SubElement(el, "freejoint", name=f"{name}_free")
    ET.SubElement(el, "inertial", pos="0 0 0", mass="0.004", diaginertia="2e-7 2e-7 1e-7")
    b = _Body(el)
    b.geom(f"{name}_shank", "capsule", (0, 0, 0), PEG, C["steel"], size=SHANK_R, fromto=(0, 0, -SHANK_L + SHANK_R, 0, 0, 0))
    b.geom(f"{name}_head", "cylinder", (0, 0, HEAD_H / 2), HEAD, C["screw"], size=(HEAD_R, HEAD_H / 2))
    b.look(f"{name}_slot", "box", (0, 0, HEAD_H + .0002), C["dark"], size=(HEAD_R * .9, .0006, .0002))


def _probe(world, pos, quat):
    el = _body(world, "probe", pos, quat)
    ET.SubElement(el, "freejoint", name="probe_free")
    ET.SubElement(el, "inertial", pos=_v([.45 * PROBE_L, 0, 0]), mass="0.02", diaginertia="1e-6 2e-5 2e-5")
    b = _Body(el)
    b.geom("probe_rod", "cylinder", (PROBE_L / 2, 0, 0), TOOL, C["tool"], quat=(.7071068, 0, .7071068, 0),
           size=(PROBE_R, PROBE_L / 2))
    b.geom("probe_tip", "capsule", (0, 0, 0), TOOL, C["dark"], size=PROBE_TIP_R,
           fromto=(PROBE_L, 0, 0, PROBE_L + PROBE_TIP, 0, 0))
    # The probe's own contacts win (priority 1) and are hard (the HARD_LIMIT impedance): a 20 g rod squeezed between an
    # arm link and its stand otherwise pumped energy through the soft robot x probe contact (stress round clock_r2,
    # cem_e155: 1.74 J; runs/watch_assembly_design_20261014/out/stress_clock_r2/). It also changes the probe's rest on its
    # stand, so the birth changes with it.
    for g in el.findall("geom"):
        if g.get("group") == "3":
            g.set("priority", "1")
            g.set("solimp", HARD_LIMIT["solimplimit"])


def opening_poses(spins=None):
    """{part: (position, wxyz)} at birth (mount frame): wheels in their cups (axis vertical, turned by spins), the bridge
    on its posts (the grasp bar over the stand), screws in the tray, the probe on its stand."""
    spins = dict(BASE_SPIN) if spins is None else spins
    P = {}
    x, y = LAYOUT["wheel_stand"]
    for name, (dx, dy) in WHEEL_STAND_POS.items():
        P[name] = (np.array([x + dx, y + dy, STAND_TOP + CUP_LIFT]), quat_z(spins[name]))
    x, y = LAYOUT["bridge_stand"]
    P["bridge"] = (np.array([x - BRIDGE_GRASP[0], y - BRIDGE_GRASP[1], BRIDGE_STAND_TOP]), (1., 0., 0., 0.))
    x, y = LAYOUT["screw_tray"]
    P["screw1"] = (np.array([x + SCREW_TRAY_X[0], y, SCREW_TRAY_TOP]), (1., 0., 0., 0.))
    P["screw2"] = (np.array([x + SCREW_TRAY_X[1], y, SCREW_TRAY_TOP]), (1., 0., 0., 0.))
    x, y = LAYOUT["probe_stand"]
    P["probe"] = (np.array([x - .09, y, PROBE_STAND_Z]), (1., 0., 0., 0.))
    return P


def entity_xml(spins=None) -> str:
    """The MJCF text of the entity at birth, in the mount frame; spins: {wheel: turn about its axis (rad)}."""
    P = opening_poses(spins)
    root = ET.Element("mujoco", model=ENTITY)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    world = ET.SubElement(root, "worldbody")
    for name in WHEELS:                       # the barrel first: the entity's base link
        _wheel(world, name, *P[name])
    _bridge(world, *P["bridge"])
    for s in ("screw1", "screw2"):
        _screw(world, s, *P[s])
    _probe(world, *P["probe"])
    _movement(world)
    _carriers(world)
    _stands(world)
    eq = ET.SubElement(root, "equality")
    for n in WHEELS:
        ET.SubElement(eq, "weld", name=f"seat_{n}", body1=f"carr_{n}_s", body2=n, active="false", solref=WELD_SOLREF)
    for n, par in PARENT.items():
        ET.SubElement(eq, "joint", name=f"mesh_{n}", joint1=f"spin_{n}", joint2=f"spin_{par}",
                      polycoef=_v([0, RATIO[n], 0, 0, 0]), active="false", solref=WELD_SOLREF)
    for n in WHEELS:
        for ax in ("tx", "ty"):
            ET.SubElement(eq, "joint", name=f"lock_{ax}_{n}", joint1=f"{ax}_{n}", polycoef="0 0 0 0 0", active="false")
    ET.SubElement(eq, "weld", name="bridge_fix", body1="bridge", active="false", solref=WELD_SOLREF)   # to the world
    # the run test: the mainspring held rigid until the bridge is fixed; then the arbor welded to its click carrier
    ET.SubElement(eq, "joint", name="spring_lock", joint1="spring_barrel", polycoef="0 0 0 0 0", active="true",
                  solref=WELD_SOLREF)
    ET.SubElement(eq, "weld", name="arbor_fix", body1="carr_barrel_a", body2="barrel_arbor", active="false",
                  solref=WELD_SOLREF)
    for s in ("screw1", "screw2"):
        ET.SubElement(eq, "weld", name=f"{s}_fix", body1="bridge", body2=s, active="false", solref=WELD_SOLREF)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------- points spanning each part (grasp regions, checks)
def _ring_pts(r, z, n=16):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z)]


def part_points(name) -> np.ndarray:
    """Points spanning a part's solid in its own frame."""
    if name == "barrel_arbor":
        return np.vstack([_ring_pts(RATCHET_R + .0012, RATCHET_Z[0]), _ring_pts(RATCHET_R + .0012, RATCHET_Z[1]),
                          _ring_pts(.0045, BARREL_SQUARE_TOP), [[0, 0, GEARS["barrel"]["drum"][1]]]])
    if name in WHEELS:
        g = GEARS[name]
        r = pitch_r(name) + g["m"]
        top = BARREL_SQUARE_TOP if name == "barrel" else SHOULDER_Z + UP_PIVOT_L
        z0, z1 = g["drum"] if name == "barrel" else g["band"]
        return np.vstack([_ring_pts(r, z0), _ring_pts(r, z1), [[0, 0, -LOW_PIVOT_L], [0, 0, top]]])
    if name == "bridge":
        pts = []
        for k, p in BRIDGE_PTS.items():
            rr = BOSS_R.get(k, BAR_W / 2)
            for z in (0., T_BRIDGE):
                pts.append(_ring_pts(rr, z, 8) + np.r_[p, 0.])
        return np.vstack(pts)
    if name in ("screw1", "screw2"):
        return np.vstack([_ring_pts(HEAD_R, HEAD_H), _ring_pts(HEAD_R, 0.), [[0, 0, -SHANK_L]]])
    if name == "probe":
        return np.array([[0, 0, 0], [PROBE_L + PROBE_TIP, 0, 0], [0, PROBE_R, 0], [0, -PROBE_R, 0], [0, 0, PROBE_R],
                         [0, 0, -PROBE_R]])
    raise ValueError(f"clock_assembling has no part {name!r}")
