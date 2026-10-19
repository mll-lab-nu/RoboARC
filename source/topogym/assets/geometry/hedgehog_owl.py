"""Bird in Cage (an owl in a cage) as a hedgehog_in_cage setting: a wooden cage on the family's mount, a wooden owl inside
it whose two wing plates turn together on one hinge, and a thin rod lying on the table.

Source: the design session's model v6 with a rung across the widest gap at 37 mm (runs/bird_in_cage_design_20261010,
scripts/owl2.py, `preset("v6", extras=door_rung(preset("v6"), 37.))`; the rod from scripts/v6_grav.py). The parameters
below are REAL millimetres, measured on the product photos; the model is built at SCALE times the
real size (the robot's gripper is far larger than fingertips).

Frames. Cage: origin at the centre of the floor (the top face of the bottom disc), z up along the cage axis, the widest
gap between bars centred on +x. Owl: origin at the body centre, +x forward (beak), +y left, +z toward the head. The two
wing plates are one rigid part turning about the owl's +y axis through (0, 0, shoulder_z); wing angle 0 = the plates
pointing backward (-x), 90 deg = toward the head, 270 deg = the straight edge along the body toward the tail. Rod: origin
at the handle's centre, +x along the shaft toward the tip, +z up when the rod lies flat.

Collision: exact MuJoCo primitives and convex pieces only. The cage's bars and rung are capsules (cage_parts), each disc
is a ring of 32 convex wedges (its central hole stays open); the owl's body is one convex mesh with every vertex on its
ellipsoid (BODY_SUBDIVISIONS), its beak and legs capsules, its tail one convex hexahedron and each wing plate one convex
half-disc prism; the rod is a capsule and a box. Every mesh is given by its vertices only, so the solid is their convex
hull, which is what MuJoCo collides.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

MM = 1e-3
SCALE = 2.5

# Cage (real mm): five bars on a 22.5 mm circle; the widest gap 24.0 mm clear, the bars beside it 72 deg further round.
BAR_CIRCLE_R, BAR_R, HEIGHT = 22.5, 2.75, 64.     # HEIGHT: clear height between the two discs
DISC_R, DISC_T, HOLE_R = 27.5, 10., 11.
WIDE_CLEAR, SIDE_GAP_DEG = 24., 72.
RUNG_Z, RUNG_R = 37., 2.0                          # a level rung across the widest gap
DISC_WEDGES = 32
# Owl (real mm).
BODY_R, BODY_HALF = 11.2, 10.3                     # ellipsoid semi-axes BODY_R, BODY_R, BODY_HALF + BODY_R
# The body collides and renders as one convex mesh: an icosphere of this many subdivisions with every vertex on the
# ellipsoid (10242 vertices; its faces lie at most 0.0143 mm inside the ellipsoid at model scale, level 4: 0.057 mm).
BODY_SUBDIVISIONS = 5
BEAK = ((8.2, 0., 8.5), (12.2, 0., 6.2), 1.6)      # capsule from, to, radius
LEG_ROOT, LEG_FORWARD_DEG, LEG_OUTWARD_DEG, LEG_LEN, LEG_R = (5., 6., -15.), 25., 20., 16., 2.0   # left leg (+y)
TAIL_ROOT, TAIL_LEN, TAIL_W, TAIL_T = (-7., 0., -13.), 34., (10., 24.), 5.   # width at the root, at the end
SHOULDER_Z, WING_INNER, WING_T, WING_LEN, WING_DEPTH = 7., 11.8, 5., 36., 17.
WING_PIVOT = (17.2, 6.0)          # the hinge: distance from the front end of the straight edge, depth into the plate
WING_OUTLINE_POINTS = 14
# Rod (real mm; the handle is real robot size, metres): a shaft and a handle box at its end.
ROD_R, ROD_LEN = 1.5, 80.         # shaft radius; length from the tip centre to the start of the handle
HANDLE_HALF_M = (.030, .012, .012)
# Masses (kg) and inertias (kg m^2): the design session's.
BODY_MASS, BODY_COM_M, BODY_INERTIA = .22, (0., 0., -.004), (2.2e-4, 2.2e-4, 1.1e-4)
WINGS_MASS, WINGS_INERTIA = .04, (4e-5, 4e-5, 4e-5)
ROD_MASS = .03
# The wing hinge: dry friction (N m) that holds the wings against gravity, and a little damping (N m s / rad).
WING_FRICTIONLOSS, WING_DAMPING = .008, .002

WOOD_L, WOOD_D, WOOD_W, PEG = (.86, .72, .50, 1.), (.36, .22, .12, 1.), (.47, .30, .17, 1.), (.90, .78, .56, 1.)
ROD_RGBA, HANDLE_RGBA = (.75, .15, .15, 1.), (.2, .2, .22, 1.)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def gap_deg(clear, bar_circle_r=BAR_CIRCLE_R, bar_r=BAR_R):
    """Angle between two bars whose gap is `clear` mm wide."""
    return float(2 * np.degrees(np.arcsin((clear + 2 * bar_r) / (2 * bar_circle_r))))


def bar_az():
    """Bar azimuths (deg, cage frame): the widest gap is the one across 0 deg."""
    a = gap_deg(WIDE_CLEAR) / 2
    return (a, a + SIDE_GAP_DEG, 180., 360. - a - SIDE_GAP_DEG, 360. - a)


def leg_dir():
    """Left leg direction (not normalised): 25 deg forward and 20 deg outward of straight down."""
    return (float(np.tan(np.radians(LEG_FORWARD_DEG))), float(np.tan(np.radians(LEG_OUTWARD_DEG))), -1.)


def _leg_low():
    return -(LEG_ROOT[2] + _unit(leg_dir())[2] * LEG_LEN) + LEG_R


def tail_dir():
    """The tail's downward slope at which its lowest corner is level with the leg tips (the owl stands on three points)."""
    f = lambda e: TAIL_ROOT[2] - TAIL_LEN * np.sin(e) - TAIL_T / 2 * np.cos(e) + _leg_low()
    e = brentq(f, 0., np.pi / 2)
    return (-float(np.cos(e)), 0., -float(np.sin(e)))


def tail_vertices():
    """The tail board's 8 corners, owl frame, real mm."""
    d = _unit(tail_dir())
    r = np.asarray(TAIL_ROOT, float)
    n = np.cross(d, [0, 1, 0])
    n /= np.linalg.norm(n)
    return np.array([r + d * s + np.array([0, y, 0]) + n * h for s, w in ((0., TAIL_W[0]), (TAIL_LEN, TAIL_W[1]))
                     for y in (-w / 2, w / 2) for h in (-TAIL_T / 2, TAIL_T / 2)])


def wing_vertices(side):
    """One wing plate at wing angle 0, relative to the hinge point (0, 0, SHOULDER_Z) of the owl frame, real mm: a half
    disc (straight edge along -x, bulge toward -z) of WING_OUTLINE_POINTS outline points, WING_T thick."""
    t = np.linspace(0, np.pi, WING_OUTLINE_POINTS)
    rad = WING_LEN / 2
    a, b = rad - rad * np.cos(t), np.sin(t) * WING_DEPTH
    x, z = -(a - WING_PIVOT[0]), -(b - WING_PIVOT[1])
    y0, y1 = side * WING_INNER, side * (WING_INNER + WING_T)
    n = len(t)
    return np.r_[np.c_[x, np.full(n, y0), z], np.c_[x, np.full(n, y1), z]]


def body_semi_axes():
    return np.array([BODY_R, BODY_R, BODY_HALF + BODY_R])


def body_vertices():
    """The body's vertices, owl frame, real mm: a unit icosphere's (trimesh, BODY_SUBDIVISIONS) scaled onto the ellipsoid."""
    import trimesh
    return np.asarray(trimesh.creation.icosphere(subdivisions=BODY_SUBDIVISIONS, radius=1.).vertices) * body_semi_axes()


def ring_wedges(r_in, r_out, z0, z1, n=DISC_WEDGES):
    """A flat ring as n convex wedges (8 corners each), real mm."""
    out = []
    for i in range(n):
        a0, a1 = 2 * np.pi * i / n, 2 * np.pi * (i + 1) / n
        out.append(np.array([(r * np.cos(a), r * np.sin(a), z) for a in (a0, a1) for r in (r_in, r_out) for z in (z0, z1)]))
    return out


def _m(v):
    """Real mm -> model metres."""
    return np.asarray(v, float) * SCALE * MM


def _f(values):
    return " ".join(format(float(x), ".17g") for x in np.asarray(values, float).reshape(-1))


def _rgba(c):
    return " ".join(format(x, "g") for x in c)


# -- the parts, as MJCF geoms in their link frames (metres)
def cage_parts():
    """[(name, kind, data)] of the cage, cage frame: ("capsule", (from, to, radius)) or ("mesh", vertices).

    The bars and the rung are capsules of the same solid as the prototype's cylinders: each bar's cylinder runs from the
    middle of the floor disc to the middle of the roof disc, and its capsule ends a radius short of those points, so
    both round ends lie inside the discs; the rung's ends lie on the bars' axes, inside the thicker bars. On MuJoCo 3.14
    (multi-point convex collision) a mesh against a cylinder returned a few normals pointing into the mesh (the owl's
    body: 8 of 15,526 contacts up to 3 mm deep, some 0.3 mm), and a capsule against a cylinder 11.6 mm deep crashed
    the process; against capsules neither happened (0 of 13,860)."""
    parts = []
    for k, a in enumerate(bar_az()):
        x, y = BAR_CIRCLE_R * np.cos(np.radians(a)), BAR_CIRCLE_R * np.sin(np.radians(a))
        parts.append((f"bar{k}", "capsule", (_m([x, y, -DISC_T / 2 + BAR_R]), _m([x, y, HEIGHT + DISC_T / 2 - BAR_R]),
                                             _m(BAR_R))))
    a = np.radians(gap_deg(WIDE_CLEAR) / 2)
    xb, yb = BAR_CIRCLE_R * np.cos(a), BAR_CIRCLE_R * np.sin(a)
    parts.append(("rung", "capsule", (_m([xb, -yb, RUNG_Z]), _m([xb, yb, RUNG_Z]), _m(RUNG_R))))
    for tag, z0, z1 in (("floor", -DISC_T, 0.), ("roof", HEIGHT, HEIGHT + DISC_T)):
        for i, V in enumerate(ring_wedges(HOLE_R, DISC_R, z0, z1)):
            parts.append((f"{tag}_{i:02d}", "mesh", _m(V)))
    return parts


def owl_body_parts():
    """[(name, kind, data, colliding)] of the owl's body link, owl frame."""
    parts = [("body", "mesh", _m(body_vertices()), True)]
    a, b, r = BEAK
    parts.append(("beak", "capsule", (_m(a), _m(b), _m(r)), True))
    for side, tag in ((1, "l"), (-1, "r")):
        r0 = np.array(LEG_ROOT, float) * [1, side, 1]
        r1 = r0 + _unit(np.array(leg_dir(), float) * [1, side, 1]) * LEG_LEN
        parts.append((f"leg_{tag}", "capsule", (_m(r0), _m(r1), _m(LEG_R)), True))
    parts.append(("tail", "mesh", _m(tail_vertices()), True))
    for side, tag in ((1, "l"), (-1, "r")):          # eyes: appearance only
        e = np.array([BODY_R * .80, side * 5.2, 12.5])
        parts.append((f"eye_{tag}", "cylinder", (_m(e * [.9, 1, 1]), _m(e * [1.06, 1.08, 1.02]), _m(3.6)), False))
        parts.append((f"pupil_{tag}", "cylinder", (_m(e * [.95, 1, 1]), _m(e * [1.085, 1.09, 1.022]), _m(1.6)), False))
    return parts


def owl_wing_parts():
    """[(name, kind, data, colliding)] of the wings link, its frame at the hinge point, wing angle 0."""
    span = WING_INNER + WING_T
    return [("wing_l", "mesh", _m(wing_vertices(+1)), True), ("wing_r", "mesh", _m(wing_vertices(-1)), True),
            ("pin", "cylinder", (_m([0, -span, 0]), _m([0, span, 0]), _m(1.5)), False)]


def rod_parts():
    """[(name, kind, data)] of the rod, rod frame (origin at the handle's centre, +x toward the tip)."""
    hx = HANDLE_HALF_M[0]
    return [("handle", "box", np.asarray(HANDLE_HALF_M, float)),
            ("shaft", "capsule", (np.array([hx, 0., 0.]), np.array([hx + ROD_LEN * SCALE * MM, 0., 0.]), ROD_R * SCALE * MM))]


def rod_tip_m():
    """The shaft tip's centre in the rod frame."""
    return np.array([HANDLE_HALF_M[0] + ROD_LEN * SCALE * MM, 0., 0.])


def rod_inertial():
    """(mass, centre of mass, full inertia [ixx iyy izz ixy ixz iyz] about it) of a rod of uniform density and mass
    ROD_MASS, rod frame: the handle box, the shaft's cylinder from the handle's face to the tip and the tip's
    hemisphere (the shaft's other end cap lies inside the handle)."""
    hx, hy, hz = HANDLE_HALF_M
    r, length = ROD_R * SCALE * MM, ROD_LEN * SCALE * MM
    vb, vc, vh = 8 * hx * hy * hz, np.pi * r * r * length, 2 / 3 * np.pi * r ** 3
    pieces = []   # (volume, centre, inertia about the centre per unit volume, principal axes along the rod frame)
    pieces.append((vb, np.zeros(3), np.array([hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy]) / 3))
    pieces.append((vc, np.array([hx + length / 2, 0, 0]), np.array([r * r / 2, (3 * r * r + length * length) / 12,
                                                                   (3 * r * r + length * length) / 12])))
    perp = (83 / 320) * r * r      # a solid hemisphere about its centre of mass, perpendicular to its axis
    pieces.append((vh, np.array([hx + length + 3 * r / 8, 0, 0]), np.array([2 / 5 * r * r, perp, perp])))
    density = ROD_MASS / sum(v for v, _, _ in pieces)
    com = sum(v * c for v, c, _ in pieces) / sum(v for v, _, _ in pieces)
    inertia = np.zeros((3, 3))
    for v, c, i in pieces:
        m, d = density * v, c - com
        inertia += np.diag(m * i) + m * (d @ d * np.eye(3) - np.outer(d, d))
    return ROD_MASS, com, np.array([inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]])


# -- MJCF
def _geom_xml(name, kind, data, rgba, colliding=True, assets=None, twin=False):
    """One MJCF geom. twin: a colliding geom goes to group 3 with a visual-only copy in its colour (Genesis and Nyx
    draw a colliding geom of groups 0-2 grey; the MuJoCo scene drops the massless copy), as fiber_bundling does."""
    if colliding and twin:
        return (_geom_xml(name, kind, data, rgba, True, assets).replace("/>", ' group="3"/>')
                + _geom_xml(f"{name}_visual", kind, (name, data) if kind == "mesh" else data, rgba, False, assets))
    extra = "" if colliding else ' contype="0" conaffinity="0" mass="0"'
    if kind == "mesh":
        mesh, data = data if isinstance(data, tuple) else (name, data)      # a visual copy reuses its geom's mesh
        if mesh == name:
            assets.append(f'<mesh name="{name}" vertex="{_f(data)}"/>')
        return f'<geom name="{name}" type="mesh" mesh="{mesh}" rgba="{_rgba(rgba)}"{extra}/>'
    if kind in ("cylinder", "capsule"):
        a, b, r = data
        return f'<geom name="{name}" type="{kind}" fromto="{_f(np.r_[a, b])}" size="{_f([r])}" rgba="{_rgba(rgba)}"{extra}/>'
    return f'<geom name="{name}" type="{kind}" size="{_f(data)}" rgba="{_rgba(rgba)}"{extra}/>'


def _document(model, assets, body):
    return (f'<mujoco model="{model}">\n  <compiler angle="radian"/>\n  <asset>\n    ' + "\n    ".join(assets)
            + f'\n  </asset>\n  <worldbody>\n{body}\n  </worldbody>\n</mujoco>\n')


def cage_mjcf():
    """The cage as one static body (no joint): the start pose's clearance check (start_clearance)."""
    assets = []
    geoms = [_geom_xml(n, k, d, WOOD_L, assets=assets) for n, k, d in cage_parts()]
    return _document("owl_cage", assets, '    <body name="cage">\n      ' + "\n      ".join(geoms) + "\n    </body>")


def cage_urdf(folder):
    """owl_cage.urdf and owl_cage_meshes/ in folder: the scene's cage, a fixed URDF (Nyx renders a fixed entity from a
    URDF, not from an MJCF). One link "cage"; each capsule of cage_parts as a cylinder with a sphere of its radius on each
    end cap (profile `capsules` collides the three as that one capsule), each wedge as its convex mesh; one visual mesh of
    the whole cage."""
    import trimesh
    folder = Path(folder)
    (folder / "owl_cage_meshes").mkdir(parents=True, exist_ok=True)
    collisions, visual = [], []
    for name, kind, data in cage_parts():
        if kind == "capsule":
            a, b, r = data
            T = np.eye(4)
            T[:3, :3] = Rotation.align_vectors([_unit(b - a)], [[0., 0., 1.]])[0].as_matrix()
            T[:3, 3] = (a + b) / 2
            rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
            collisions.append(f'<collision name="{name}"><origin xyz="{_f(T[:3, 3])}" rpy="{_f(rpy)}"/><geometry>'
                              f'<cylinder radius="{_f([r])}" length="{_f([np.linalg.norm(b - a)])}"/></geometry></collision>')
            collisions += [f'<collision><origin xyz="{_f(end)}"/><geometry><sphere radius="{_f([r])}"/></geometry></collision>'
                           for end in (a, b)]
            visual.append(trimesh.creation.capsule(height=float(np.linalg.norm(b - a)), radius=float(r), transform=T))
        else:
            hull = trimesh.convex.convex_hull(data)
            hull.export(folder / "owl_cage_meshes" / f"{name}.obj", digits=17)
            collisions.append(f'<collision name="{name}"><geometry><mesh filename="owl_cage_meshes/{name}.obj"/>'
                              f'</geometry></collision>')
            visual.append(hull)
    # The family's light wood finish (hedgehog_in_cage.write_body, finish "wood"): a box-projected wood texture in a GLB.
    from topogym.assets.library import resource_path
    from topogym.assets.surfaces import wood_visual
    mesh = trimesh.util.concatenate(visual)
    mesh = wood_visual(mesh, resource_path("hedgehog_in_cage/materials/light_wood.png"), roughness=.62,
                       tile_size=float(mesh.extents.max()) * .7)
    mesh.export(folder / "owl_cage_meshes" / "visual.glb", include_normals=True)
    (folder / "owl_cage.urdf").write_text(
        '<robot name="owl_cage">\n  <link name="cage">\n    <visual><geometry><mesh filename="owl_cage_meshes/visual.glb"/>'
        '</geometry></visual>\n    ' + "\n    ".join(collisions) + "\n  </link>\n</robot>\n")


def owl_mjcf():
    """The owl: the body link on a free joint, the wings link on a hinge about the owl's +y axis (joint value = wing
    angle; 0 = the authored plates pointing backward)."""
    assets = []
    body = [_geom_xml(n, k, d, PEG if n.startswith(("beak", "leg", "eye")) else (0.05, .04, .03, 1.) if n.startswith("pupil")
                      else WOOD_D, c, assets, twin=True) for n, k, d, c in owl_body_parts()]
    wings = [_geom_xml(n, k, d, PEG if n == "pin" else WOOD_W, c, assets, twin=True) for n, k, d, c in owl_wing_parts()]
    xml = (f'    <body name="body">\n      <freejoint name="root"/>\n'
           f'      <inertial pos="{_f(BODY_COM_M)}" mass="{_f([BODY_MASS])}" diaginertia="{_f(BODY_INERTIA)}"/>\n      '
           + "\n      ".join(body)
           + f'\n      <body name="wings" pos="{_f(_m([0, 0, SHOULDER_Z]))}">\n'
           f'        <joint name="wing" type="hinge" axis="0 1 0" damping="{_f([WING_DAMPING])}" '
           f'frictionloss="{_f([WING_FRICTIONLOSS])}"/>\n'
           f'        <inertial pos="0 0 0" mass="{_f([WINGS_MASS])}" diaginertia="{_f(WINGS_INERTIA)}"/>\n        '
           + "\n        ".join(wings) + "\n      </body>\n    </body>")
    return _document("owl", assets, xml)


def rod_mjcf():
    """The rod on a free joint."""
    mass, com, inertia = rod_inertial()
    geoms = [_geom_xml(n, k, d, HANDLE_RGBA if n == "handle" else ROD_RGBA, twin=True) for n, k, d in rod_parts()]
    xml = (f'    <body name="rod">\n      <freejoint name="root"/>\n'
           f'      <inertial pos="{_f(com)}" mass="{_f([mass])}" fullinertia="{_f(inertia)}"/>\n      '
           + "\n      ".join(geoms) + "\n    </body>")
    return _document("rod", [], xml)


# -- points for the placement rule: every geom, colliding or not, conservatively (topogym.evaluation.task_metrics
#    rigid_asset_vertices: a mesh by its vertices, an analytic surface by a box around it)
def _points(kind, data):
    corners = np.array([(x, y, z) for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    if kind == "mesh":
        return np.asarray(data, float)
    if kind in ("cylinder", "capsule"):
        a, b, r = data
        lo, hi = np.minimum(a, b) - r, np.maximum(a, b) + r
        return (lo + hi) / 2 + corners * (hi - lo) / 2
    return corners * np.asarray(data, float)      # box half sizes


def cage_points():
    return np.vstack([_points(k, d) for _, k, d in cage_parts()])


def owl_link_points():
    """{"body": points in the owl frame, "wings": points in the wings link's frame (at the hinge point)}. The body mesh
    (10242 vertices, read every tick) by the 8 corners of the box around its ellipsoid, which contains it."""
    box = ("body", "box", _m(body_semi_axes()), True)
    return {"body": np.vstack([_points(k, d) for _, k, d, _ in [box] + owl_body_parts()[1:]]),
            "wings": np.vstack([_points(k, d) for _, k, d, _ in owl_wing_parts()])}


def wings_in_owl_frame(points, wing_rad):
    """Wings-link points in the owl frame at a wing angle."""
    return Rotation.from_euler("y", float(wing_rad)).apply(points) + _m([0, 0, SHOULDER_Z])


def rod_points():
    return np.vstack([_points(k, d) for _, k, d in rod_parts()])


# -- the start pose's clearance, measured by MuJoCo on the same geoms
def start_clearance(owl_position_m, owl_quaternion_wxyz, wing_rad, distmax=.01):
    """(smallest owl-to-cage distance in metres, negative = overlap; (owl geom, cage geom)) at the owl's pose in the cage
    frame, by mj_geomDistance between every colliding pair (exact for these primitives and convex meshes)."""
    import mujoco
    import xml.etree.ElementTree as ET
    cage, owl = ET.fromstring(cage_mjcf()), ET.fromstring(owl_mjcf())
    root = ET.Element("mujoco", model="owl_check")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", gravity="0 0 0")
    asset, world = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
    for src, prefix in ((cage, "cage/"), (owl, "owl/")):
        for m in src.find("asset"):
            m.set("name", prefix + m.get("name"))
            asset.append(m)
        for g in src.iter("geom"):
            g.set("name", prefix + g.get("name"))
            if g.get("mesh"):
                g.set("mesh", prefix + g.get("mesh"))
        world.extend(list(src.find("worldbody")))
    body = world.find("body[@name='body']")
    body.set("pos", _f(owl_position_m))
    body.set("quat", _f(owl_quaternion_wxyz))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    data.qpos[:7] = np.r_[owl_position_m, owl_quaternion_wxyz]
    data.qpos[7] = wing_rad
    mujoco.mj_kinematics(model, data)
    collide = lambda g: model.geom_contype[g] or model.geom_conaffinity[g]
    names = [model.geom(g).name for g in range(model.ngeom)]
    owl_geoms = [g for g in range(model.ngeom) if names[g].startswith("owl/") and collide(g)]
    cage_geoms = [g for g in range(model.ngeom) if names[g].startswith("cage/") and collide(g)]
    best, pair, fromto = distmax, None, np.zeros(6)
    for a in owl_geoms:
        for b in cage_geoms:
            d = mujoco.mj_geomDistance(model, data, a, b, distmax, fromto)
            if d < best:
                best, pair = float(d), (names[a][4:], names[b][5:])
    return best, pair


def write_assets(folder, owl_pose):
    """owl_cage.urdf (and its meshes), owl.xml, rod.xml and asset_report.json in folder; returns the report."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cage_urdf(folder)
    for name, text in (("owl.xml", owl_mjcf()), ("rod.xml", rod_mjcf())):
        (folder / name).write_text(text)
    clearance, pair = start_clearance(owl_pose["position_m"], owl_pose["quaternion_wxyz"], owl_pose["wing_rad"])
    az = bar_az()
    report = dict(geometry_status="Bird in Cage model v6 + rung at 37 mm (runs/bird_in_cage_design_20261010)",
                  scale=SCALE, bar_azimuths_deg=list(az), widest_gap_clear_mm=WIDE_CLEAR * SCALE,
                  tail_dir=list(tail_dir()), rod_mass_kg=ROD_MASS,
                  initial_pose=dict(owl_pose, clearance_to_cage_m=clearance, closest_pair=pair),
                  reference_route_verified=False, physics_qualified=False)
    (folder / "asset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
