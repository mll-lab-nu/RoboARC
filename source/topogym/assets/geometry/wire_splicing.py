"""wire_splicing: the MJCF of the family's four entities (coordinates in the ALOHA mount frame).

Physical picture (runs/wire_splice_design_20261014, README_v5 sections 1-5, README_v6): a table with two terminal blocks
at its back. Two stiff insulated wires lie naturally on the table; each one's far end is fixed at its terminal (the
simulated power end), its stripped end lies free on the table in front. Wire A (red): 700 mm insulated + 117 mm bare;
wire B (blue): 550 mm insulated + 290 mm bare; copper core 3.56 mm, insulation 7.0 mm (twice a BV 2.5 mm2 wire). A
bending post stands between them: a vertical post with a horizontal rod (7.1 mm across) along -y, the mandrel for A's
eye. Two ALOHA arms bend an eye into A's bare end round the rod, thread B through it, wind B's bare tail one and a half
turns round A's doubled end, at least one of them round both of its strands together, and fold A's end back.

Wires (wire_model.py, v5_model.py): ball-joint chains of capsule links (the last insulated link a cylinder, the strip's
flat end), elastoplastic joints (the profile's plastic rule, MujocoWorld._plastic_return): a stiff spring, elastic up to
0.4 deg, past its yield torque (plastic moment 0.130 N m bare, 0.156 N m insulated: copper's yield stress / 4) the rest
orientation follows. The first link of each wire hangs on a ball joint at its terminal's front face. Links 4 or more
apart on the same wire collide. The lay: a planar curve integrated from the terminal (heading and curvature along the
insulated part; the bare part straight), 1.5 mm above the insulation's resting height; the task settles it under
gravity before the birth (reset_physics_steps).

Entities: wire_a (A's terminal, a static body, and A's chain), wire_b (the same for B), jig (static: the bending post).
No splice meter (removed 2026-10-22, user decision: a visual-only meter by the terminals, which the red wire passed
through when lifted).
Frames: the mount frame is the ALOHA's (table top z = 0, x toward the right arm, y away from the cameras, z up). The
design's coordinates (its "scene frame") are the mount frame minus PLACE (v6_armsim.PLACE: the scene sits at
(-0.05, +0.10) m on the mount).
Collision geoms are in group 3 with coloured visual twins (contype 0, no mass), as bed_bolt: the renderer copies a
collision geom of groups 0-2 into a grey visual."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

ENTITIES = ("wire_a", "wire_b", "jig")
WIRE_OF = {"wire_a": "A", "wire_b": "B"}            # entity -> the design's wire letter (body names A_000, B_000, ...)
PLACE = np.array([-.05, .10, 0.])                   # design scene frame -> mount frame

# ---------------------------------------------------------------- cross-section and material (wire_model.py)
CORE_D = 1.78e-3 * 2.               # BV 2.5 mm2 solid core 1.78 mm, x2
INS_D = 3.5e-3 * 2.                 # its insulation's outer diameter, x2
R_CU, R_INS = CORE_D / 2, INS_D / 2
RHO_CU, RHO_PVC = 8960., 1400.      # kg/m3
SIGMA_SIM = 69e6 / 4                # Pa: annealed copper's 0.2 % yield stress / 4 (keeps the twisting torque ~0.3 N m)
M_P_CORE = 4. / 3. * SIGMA_SIM * R_CU ** 3       # fully plastic bending moment of the bare core, N m
M_P_INS = 1.2 * M_P_CORE            # insulated: core + PVC (an estimate)
YIELD_DEG = .4                      # elastic give per joint (fiber_bundling v4's)
ARMATURE = 2e-6                     # joint armature, kg m2
FRICTION_WW = .5                    # copper on copper
CONTACT = dict(solref=(.00125, 2.), solimp=(.9, .99, .001, .5, 2.))   # the design's geom default (wire_model.header)
# Contact margin of the wire geoms: 0, not the design's 0.2 mm (wire_model.py). With a margin a contact acts from that
# distance, so a pinch rests inside it at a positive distance, and contact_weld_v1 counts only contacts at zero or more
# penetration: the ALOHA's fully closed fingers (about 18 N) left all seven finger contacts on an insulated link 0.01-0.13
# mm short and never welded (out/l3, 2026-10-18). Without it the wires rest 0.22 mm lower and a pinch welds in 0.125 s.
MARGIN = 0.
DROP = .0015                        # the lay starts this far above the insulation's resting height

# ---------------------------------------------------------------- layout (the design's scene frame), m
TERM = dict(A=(-.24, .40), B=(.24, .40))             # terminal front faces (the wire roots)
JIG = dict(x=0., y=.24, z=.15, r=CORE_D, length=.035)   # rod radius 1 d (7.12 mm across); axis along y, free end -y
# insulated / bare length and links; the lay: heading at the terminal (deg, from +x) and the insulated part's
# curvature kappa(s) = amplitude sin(2 pi s / period) + offset (1/m); the bare part lies straight
WIRES = dict(
    A=dict(l_ins=.700, n_ins=17, l_bare=.117, n_bare=46, heading0_deg=-65., kappa=(-10., .70, 4.),
           rgba=(.80, .12, .10, 1.)),
    B=dict(l_ins=.550, n_ins=14, l_bare=.290, n_bare=83, heading0_deg=-75., kappa=(8., .55, -5.5),
           rgba=(.13, .30, .78, 1.)),
)
RGBA_CU, RGBA_SEAM = (.85, .52, .28, 1.), (.45, .25, .12, 1.)
# The bending post's rod: yellow at every seed (user decision 2026-10-22). In cam_high the rod points at the camera in
# front of the post; grey like the post, the eye sliding off its free end read as a wire passing through the post.
RGBA_ROD = (.95, .76, .10, 1.)


def _v(values) -> str:
    return " ".join(format(float(x), ".17g") for x in np.atleast_1d(np.asarray(values, dtype=float)))


def n_links(wire: str) -> int:
    return WIRES[wire]["n_ins"] + WIRES[wire]["n_bare"]


def segments(wire: str) -> np.ndarray:
    """Link lengths of a wire, from its terminal."""
    w = WIRES[wire]
    return np.array([w["l_ins"] / w["n_ins"]] * w["n_ins"] + [w["l_bare"] / w["n_bare"]] * w["n_bare"])


def lay_curve(wire: str, z0: float) -> np.ndarray:
    """Nodes (n + 1, 3) of the wire lying on the table, mount frame: start at the terminal, integrate the heading
    (midpoint rule on the insulated part, v5_model.lay_curve)."""
    w = WIRES[wire]
    amplitude, period, offset = w["kappa"]
    kappa = lambda s: amplitude * np.sin(2 * np.pi * s / period) + offset       # noqa: E731
    p = [np.r_[np.asarray(TERM[wire]) + PLACE[:2], z0]]
    s, th = 0., np.radians(w["heading0_deg"])
    for k, length in enumerate(segments(wire)):
        if k < w["n_ins"]:
            th_mid = th + .5 * length * kappa(s)
            th += length * kappa(s + .5 * length)
        else:
            th_mid = th
        p.append(p[-1] + length * np.array([np.cos(th_mid), np.sin(th_mid), 0.]))
        s += length
    return np.array(p)


def _link_frames(P):
    """Rotation of every link: z along the link, x horizontal-ish (parallel transport from the first link)."""
    Rs = []
    for k in range(len(P) - 1):
        z = P[k + 1] - P[k]
        z /= np.linalg.norm(z)
        if k == 0:
            x = np.cross([0, 0, 1.], z)
            x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-9 else np.array([1., 0, 0])
        else:
            x = Rs[-1][:, 0] - np.dot(Rs[-1][:, 0], z) * z
            x /= np.linalg.norm(x)
        Rs.append(np.c_[x, np.cross(z, x), z])
    return Rs


def _wxyz(R) -> np.ndarray:
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


def joint_stiffness(plastic_moment: float) -> float:
    """The elastoplastic joint's spring: its plastic moment reached at the yield angle."""
    return plastic_moment / np.radians(YIELD_DEG)


def _header(model: str) -> list[str]:
    return [f'<mujoco model="{model}">', '  <compiler angle="radian" autolimits="true"/>',
            f'  <default><geom solref="{_v(CONTACT["solref"])}" solimp="{_v(CONTACT["solimp"])}"/></default>',
            '  <worldbody>']


def _box(pos, size, rgba, *, collide, friction=None):
    """A collision box (group 3) and its coloured visual twin, or a visual box alone."""
    out = []
    fr = f' friction="{_v([friction, .005, .0001])}"' if friction is not None else ""
    if collide:
        out.append(f'<geom type="box" pos="{_v(pos)}" size="{_v(size)}" group="3"{fr}/>')
    out.append(f'<geom type="box" pos="{_v(pos)}" size="{_v(size)}" rgba="{_v(rgba)}" contype="0" conaffinity="0" '
               f'mass="0" group="1"/>')
    return out


def _terminal(wire: str) -> list[str]:
    """The terminal block (v5_model.terminal_xml): a white box on the table, two screw heads and the wire's mouth."""
    x, y = np.asarray(TERM[wire]) + PLACE[:2]
    out = [f'<body name="terminal" pos="{_v([x, y + .016, 0.])}">']
    out += _box((0, 0, .011), (.022, .016, .011), (.93, .93, .90, 1.), collide=True)
    for sx in (-1., 1.):
        out.append(f'<geom type="cylinder" pos="{_v([sx * .009, .004, .0235])}" size="{_v([.0055, .0015])}" '
                   f'rgba="{_v([.70, .70, .72, 1.])}" contype="0" conaffinity="0" mass="0" group="1"/>')
    out += _box((0, -.017, .0035), (.006, .001, .0035), (.3, .3, .3, 1.), collide=False)
    out.append('</body>')
    return out


def wire_xml(entity: str) -> str:
    """One wire entity: its terminal (static) and its chain, laid on the table (lay_curve, DROP above rest)."""
    wire = WIRE_OF[entity]
    w = WIRES[wire]
    P = lay_curve(wire, R_INS + DROP)
    seg = segments(wire)
    Rs = _link_frames(P)
    n = len(seg)
    out, close = _header(entity) + _terminal(wire), []
    for k in range(n):
        insulated = k < w["n_ins"]
        length = seg[k]
        r = R_INS if insulated else R_CU
        if k == 0:
            pos, q = P[0], _wxyz(Rs[0])
        else:
            pos, q = np.array([0, 0, seg[k - 1]]), _wxyz(Rs[k - 1].T @ Rs[k])
        if insulated:
            mass = RHO_CU * np.pi * R_CU ** 2 * length + RHO_PVC * np.pi * (R_INS ** 2 - R_CU ** 2) * length
        else:
            mass = RHO_CU * np.pi * R_CU ** 2 * length
        Ip, Ia = mass * (3 * r * r + length * length) / 12, mass * r * r / 2
        k_j = joint_stiffness(M_P_INS if insulated else M_P_CORE)
        kind = "cylinder" if (insulated and k == w["n_ins"] - 1) else "capsule"
        rgba = w["rgba"] if insulated else RGBA_CU
        shape = f'type="{kind}" fromto="0 0 0 0 0 {_v(length)}" size="{_v(r)}"'
        out.append(f'<body name="{wire}_{k:03d}" pos="{_v(pos)}" quat="{_v(q)}">')
        out.append(f'  <joint name="{wire}_j{k:03d}" type="ball" stiffness="{_v(k_j)}" '
                   f'damping="{_v(2 * np.sqrt(k_j * ARMATURE))}" armature="{_v(ARMATURE)}"/>')
        out.append(f'  <inertial pos="0 0 {_v(length / 2)}" mass="{_v(mass)}" diaginertia="{_v([Ip, Ip, Ia])}"/>')
        out.append(f'  <geom name="{wire}_g{k:03d}" {shape} friction="{_v([FRICTION_WW, .005, .0001])}" '
                   f'margin="{_v(MARGIN)}" group="3"/>')
        out.append(f'  <geom {shape} rgba="{_v(rgba)}" contype="0" conaffinity="0" mass="0" group="1"/>')
        if not insulated:   # a thin dark seam along the bare core, so its twist shows in renders (visual only)
            out.append(f'  <geom type="box" pos="{_v([R_CU * .92, 0, length / 2])}" '
                       f'size="{_v([R_CU * .12, R_CU * .25, length / 2])}" rgba="{_v(RGBA_SEAM)}" contype="0" '
                       f'conaffinity="0" mass="0" group="1"/>')
        close.append('</body>')
    out += close[::-1] + ['  </worldbody>', '  <contact>']
    out += [f'    <exclude body1="{wire}_{i:03d}" body2="{wire}_{j:03d}"/>'
            for i in range(n) for j in range(i + 2, min(n, i + 4))]
    out += [f'    <exclude body1="terminal" body2="{wire}_{k:03d}"/>' for k in range(2)]   # the root sits in the terminal
    out += ['  </contact>', '</mujoco>', '']
    return "\n".join(out)


def jig_xml() -> str:
    """The bending post (v5_model.jig_xml): a base plate, a vertical post and the yellow rod along -y (friction 0.1). The
    rod has contact priority 1, so its friction 0.1 is the one its contacts with the wires use (MuJoCo otherwise takes the
    larger of the two geoms' frictions, the wires' 0.5; both carry the same solref / solimp). At 0.5 the bight closed
    round the rod gripped it and the eye could not be slid off its free end (the design's runs, README_v6)."""
    j = JIG
    x = j["x"] + PLACE[0]
    y0, y1 = j["y"] + .020 + PLACE[1], j["y"] - j["length"] + .020 + PLACE[1]
    ys = y0 + .006
    out = _header("jig") + ['<body name="jig" pos="0 0 0">']
    out += _box((x, ys, .004), (.020, .015, .004), (.35, .35, .38, 1.), collide=True)
    for name, a, b, r, rgba, friction in (
            ("post", (x, ys, 0.), (x, ys, j["z"] + .006), .005, (.45, .45, .48, 1.), None),
            ("rod", (x, ys, j["z"]), (x, y1, j["z"]), j["r"], RGBA_ROD, .1)):
        shape = f'type="cylinder" fromto="{_v([*a, *b])}" size="{_v(r)}"'
        fr = f' friction="{_v([friction, .005, .0001])}" priority="1"' if friction is not None else ""
        out.append(f'<geom name="{name}" {shape} group="3"{fr}/>')
        out.append(f'<geom {shape} rgba="{_v(rgba)}" contype="0" conaffinity="0" mass="0" group="1"/>')
    out += ['</body>', '  </worldbody>', '</mujoco>', '']
    return "\n".join(out)


# Appearance seeds (user decision 2026-10-22: a colour axis, scene_variation AXES["wire_splicing"]; the task's
# terminal_finish, post_finish, core_finish): visual geoms only, re-coloured by look_colours(). The texts name the wires
# red (A) and blue (B) and the post's yellow rod: the insulation and the rod keep their colours. The first entry of each
# table is the recorded look (seed 0).
TERMINAL_FINISHES = dict(white=((.93, .93, .90, 1.),), black=((.14, .14, .15, 1.),), grey=((.55, .57, .60, 1.),))
POST_FINISHES = dict(       # the base plate, the post
    steel=((.35, .35, .38, 1.), (.45, .45, .48, 1.)),
    green=((.14, .30, .18, 1.), (.18, .38, .23, 1.)),
    black=((.09, .09, .10, 1.), (.12, .12, .13, 1.)))
CORE_FINISHES = dict(copper=(RGBA_CU, RGBA_SEAM), tinned=((.74, .75, .77, 1.), (.42, .43, .46, 1.)))
CORE_CHOICES = dict(copper=("copper", "copper"), b_tinned=("copper", "tinned"), a_tinned=("tinned", "copper"))


def look_colours(name, terminal="white", post="steel", core="copper") -> dict:
    """{a default colour as the entity's MJCF writes it: its colour in these finishes} for the entity named (entity_xml's
    look). core: which wire's bare core is tinned, if any (CORE_CHOICES), so the two cores also tell apart up close."""
    def pairs(table, finish):
        return {_v(a): _v(b) for a, b in zip(next(iter(table.values())), table[finish])}
    if name == "jig":
        return pairs(POST_FINISHES, post)
    wire = WIRE_OF[name]
    return {**pairs(TERMINAL_FINISHES, terminal), **pairs(CORE_FINISHES, CORE_CHOICES[core]["AB".index(wire)])}


def entity_xml(name: str, look=None) -> str:
    """The entity's MJCF; look: look_colours() (None: the recorded look), applied to visual geoms (contype 0) only."""
    xml = jig_xml() if name == "jig" else wire_xml(name)
    if not look:
        return xml

    def repaint(line):
        if 'contype="0"' not in line:
            return line
        for old, new in look.items():
            line = line.replace(f'rgba="{old}"', f'rgba="{new}"')
        return line
    return "\n".join(repaint(line) for line in xml.split("\n"))


def link_name(entity: str, k: int) -> str:
    return f"{WIRE_OF[entity]}_{k:03d}"
