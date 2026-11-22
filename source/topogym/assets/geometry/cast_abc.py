"""Hanayama Cast ABC as the two rigid pieces of an interlocked_claws setting: the AB plate (the letters A and B cast as one
flat plate) and the C ring (a long closed loop with a gap in one long side, both ends of the gap pointed).

The puzzle (Hanayama HZ 1-03, level 1; "design supervised by Nob. Yoshigahara", after a puzzle from England, 1911):
the plate's bars carry eleven cross grooves cut into both faces. The C starts round the A's left leg.

Units and frames. The shapes are authored in bar widths w (the width of the plate's bars) and built at SCALE_M metres per
w. Plate frame: x to the right and y up when the letters read "AB", z out of the front face, origin at the lower-left
corner of the plate's bounding box (eyelet excluded), mid-plane at z = 0. Ring frame: centre at the origin, long axis
along y, gap in the middle of the +x long side, z across the ring's depth.

MEASURED comes from a top-down photograph of both pieces lying flat (20.5 px per w). ASSUMED holds what no photograph
resolves.

Collision: each piece collides as its own exact convex decomposition (profile "split": plate_parts, ring_parts), so the
windows, grooves and the C's gap stay open in MuJoCo."""
from pathlib import Path
from types import SimpleNamespace
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

SCALE_M = .010                 # metres per bar width w
DENSITY = 600.                 # kg/m^3, as every interlocked_claws piece (interlocked_claws.DENSITY)
COLORS = ((.74, .40, .24, 1.), (.20, .53, .61, 1.))   # the family's: the C ring (target) brown, the plate teal

PX_W, X0, Y0 = 20.5, 33., 343.  # px per w in the reference frame; px of the plate frame's origin (y down)
MEASURED = dict(
    y_top=72., y_base=332., y_mid=174., y_cross=250., x_stem=279.5,       # bar centre lines, px
    top_left_end=127., base_left_end=43.,                                  # centres of the round caps, px
    apex=(168.6, 88.8), left_leg_slope=-.333, right_leg_slope=.42,         # the A's legs: dx / dy (y down)
    top_bowl=(378., 123., 51.), bottom_bowl=(397., 253., 79.),             # centre x, y and centre-line radius, px
    eyelet=(500., 274., 10., 4.5),                                         # centre x, y, outer and hole radius, px
    c_outer_w=7., c_outer_l=12.7, c_ring=1.5, c_depth=1.5, c_shoulder_gap=2.3,   # the C ring, in w
)
# name -> (px x, px y, bar direction in degrees in the plate frame (None: along the A's leg), web flag)
GROOVES = {
    "G1_top": (230., 72., 0., False),
    "G2_arc": (414.1, 86.9, -45., False),
    "G3_stem_up": (279.5, 139., 90., True),
    "G4_mid": (351., 174., 0., True),
    "G5_rleg_up": (195.1, 152., None, True),       # frame: y 156 +- 3
    "G6_rleg_low": (227., 228., None, True),
    "G7_cross": (214., 250., 0., True),            # frame: x 206 +- 4
    "G8_lleg_low": (106.2, 276., None, False),
    "G9_base_left": (238., 332., 0., True),        # frame: x 243 +- 4
    "G10_base_right": (315., 332., 0., False),
    "G11_stem_low": (279.5, 238., 90., False),
}
ASSUMED = dict(
    T=1.9,              # plate thickness
    web_real=.35,       # metal left between the two notches of a groove (thin web)
    web_dummy=1.3,      # ... (thick web)
    notch_bottom=.4,    # length of a notch's flat bottom along the bar
    notch_top=1.7,      # length of a notch at the plate face
    tip_gap=.85,        # distance between the C's two tip points (less than a bar width)
    tip_base_r=.65,     # radius of a tip cone where it meets the ring's full section
    eps=.2,             # the notch cutters reach this far above the face
    half_width=.8,      # ... and this far to each side of the bar's centre line
)
GRID_M = 1e-7           # outlines are snap-rounded to this grid (metres), as the URDF meshes' own rounding


def _px(x, y):
    return np.array([(x - X0) / PX_W, (Y0 - y) / PX_W])


def _leg_x(which, y):
    ax, ay = MEASURED["apex"]
    return ax + MEASURED["left_leg_slope" if which == "L" else "right_leg_slope"] * (y - ay)


def plate_outline():
    """(plate polygon with its five windows, eyelet annulus), shapely, in w."""
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import unary_union
    M, h = MEASURED, .5
    yt, yb, ym, yc, xs = M["y_top"], M["y_base"], M["y_mid"], M["y_cross"], M["x_stem"]
    (cxt, cyt, rt), (cxb, cyb, rb) = M["top_bowl"], M["bottom_bowl"]

    def arc(cx, cy, r, a0, a1, n=40):
        return [_px(cx + r * np.cos(t), cy - r * np.sin(t)) for t in np.radians(np.linspace(a0, a1, n))]
    strokes = [
        LineString([_px(M["top_left_end"], yt), _px(cxt, yt)] + arc(cxt, cyt, rt, 90, -90) + [_px(xs, ym)]).buffer(h, cap_style=1, join_style=1),
        LineString([_px(xs, ym), _px(cxb, ym)] + arc(cxb, cyb, rb, 90, -90) + [_px(M["base_left_end"], yb)]).buffer(h, cap_style=1, join_style=1),
        LineString([_px(xs, yt), _px(xs, yb)]).buffer(h, cap_style=2),
        LineString([_px(_leg_x("L", yt), yt), _px(_leg_x("L", yb), yb)]).buffer(h, cap_style=2),
        LineString([_px(_leg_x("R", yt), yt), _px(_leg_x("R", yb), yb)]).buffer(h, cap_style=2),
        LineString([_px(_leg_x("L", yc), yc), _px(_leg_x("R", yc), yc)]).buffer(h, cap_style=2),
        # the solid foot where the A's right leg runs into the stem
        Polygon([_px(_leg_x("R", 236.), 236.), _px(xs, 254.), _px(xs, yb), _px(_leg_x("R", yb), yb)]),
    ]
    # the eyelet (a key-ring hole, no part of the puzzle) as 12-gons: finer circles split it into sub-millimetre sectors
    ex, ey, er, eh = M["eyelet"]
    eye = Point(*_px(ex, ey)).buffer(er / PX_W, resolution=3).difference(Point(*_px(ex, ey)).buffer(eh / PX_W, resolution=3))
    return unary_union(strokes).buffer(0), eye


def grooves():
    """name -> dict(centre (2,), along (2,) unit vector along the bar, across (2,), real, web) in w."""
    out = {}
    for name, (gx, gy, deg, real) in GROOVES.items():
        if deg is None:
            s = MEASURED["left_leg_slope" if "lleg" in name else "right_leg_slope"]
            d = np.array([s, -1.]) / np.hypot(s, 1.)
        else:
            d = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        out[name] = dict(centre=_px(gx, gy), along=d, across=np.array([-d[1], d[0]]), real=real,
                         web=ASSUMED["web_real" if real else "web_dummy"])
    return out


def groove_cutters():
    """name -> (top cutter, bottom cutter): the notches as convex vertex sets (16, 3) in w. A notch is a trapezoid in
    the plane along the bar and through the plate (flat bottom notch_bottom long at the depth of a real notch, walls
    reaching notch_top at the face), carried half_width to each side of the bar's centre line; a thick-web notch is the
    same cut stopped at a shallower depth."""
    A, T = ASSUMED, ASSUMED["T"]
    deep = (T - A["web_real"]) / 2
    slope = (A["notch_top"] - A["notch_bottom"]) / 2 / deep          # wall run per unit depth
    out = {}
    for name, g in grooves().items():
        depth = (T - g["web"]) / 2
        lb, lt = A["notch_top"] / 2 - depth * slope, A["notch_top"] / 2 + A["eps"] * slope
        pair = []
        for sgn in (1., -1.):
            pts = [[*(g["centre"] + s * g["along"] + t * g["across"]), sgn * z]
                   for s, z in ((-lb, T / 2 - depth), (lb, T / 2 - depth), (lt, T / 2 + A["eps"]), (-lt, T / 2 + A["eps"]))
                   for t in (-A["half_width"], A["half_width"])]
            pair.append(np.array(pts))
        out[name] = tuple(pair)
    return out


def groove_footprints():
    """name -> shapely rectangle (w): where a notch cuts the plate (notch_top long, 2 half_width wide)."""
    from shapely.geometry import Polygon
    out = {}
    for name, g in grooves().items():
        a, t = ASSUMED["notch_top"] / 2, ASSUMED["half_width"]
        c, d, n = g["centre"], g["along"], g["across"]
        out[name] = Polygon([c - a * d - t * n, c + a * d - t * n, c + a * d + t * n, c - a * d + t * n])
    return out


def c_outline():
    """The C ring's outline (shapely, w): a stadium ring with the gap cut from the middle of its +x long side."""
    from shapely.geometry import LineString, Polygon
    M = MEASURED
    W, L, r, gap = M["c_outer_w"], M["c_outer_l"], M["c_ring"], M["c_shoulder_gap"]
    spine = LineString([(0., -(L - W) / 2), (0., (L - W) / 2)])
    ring = spine.buffer(W / 2, resolution=24).difference(spine.buffer(W / 2 - r, resolution=24))
    return ring.difference(Polygon([(.5, -gap / 2), (W, -gap / 2), (W, gap / 2), (.5, gap / 2)]))


def tip_cones():
    """The C's two pointed tips (17, 3) each, in w: a 16-sided cone from the end face of the gap side (y = +-gap / 2)
    to the tip point; the cone's base lies on that face, so cone and ring only touch."""
    M, A = MEASURED, ASSUMED
    xm, sg, h = M["c_outer_w"] / 2 - M["c_ring"] / 2, M["c_shoulder_gap"], M["c_depth"]
    ang = np.linspace(0., 2 * np.pi, 16, endpoint=False)
    rad = A["tip_base_r"]
    if rad >= min(M["c_ring"], h) / 2:
        raise ValueError("A tip cone must stay inside the ring's end face")
    return [np.vstack([np.c_[xm + rad * np.cos(ang), np.full(16, sgn * sg / 2), rad * np.sin(ang)], [[xm, sgn * A["tip_gap"] / 2, 0.]]])
            for sgn in (1., -1.)]


def c_features():
    """Points of the C in its own frame (w): the centres of its gap side and plain side."""
    xm = MEASURED["c_outer_w"] / 2 - MEASURED["c_ring"] / 2
    return dict(gap_centre=np.array([xm, 0., 0.]), plain_centre=np.array([-xm, 0., 0.]), side_sep=2 * xm)


# ---------------------------------------------------------------- solids (metres)

def _snap(geom):
    import shapely
    from shapely import affinity
    return shapely.set_precision(affinity.scale(geom, SCALE_M, SCALE_M, origin=(0., 0.)), GRID_M)


def _manifold(mesh):
    import manifold3d
    return manifold3d.Manifold(manifold3d.Mesh64(vert_properties=np.asarray(mesh.vertices, np.float64),
                                                 tri_verts=np.asarray(mesh.faces, np.uint64)))


def _trimesh(solid):
    m = solid.to_mesh64()
    return trimesh.Trimesh(np.asarray(m.vert_properties, np.float64)[:, :3], np.asarray(m.tri_verts, np.int64))


def _extrude(poly, z0, z1):
    m = trimesh.creation.extrude_polygon(poly, z1 - z0)
    m.apply_translation([0., 0., z0])
    return m


def _hull(V):
    return trimesh.convex.convex_hull(np.asarray(V, np.float64))


def plate_solid():
    """The AB plate (trimesh, metres, plate frame): outline extruded, every notch cut from both faces, the eyelet
    (half the plate's thickness) added. Booleans in double precision."""
    T = ASSUMED["T"] * SCALE_M
    plate, eye = (_snap(g) for g in plate_outline())
    body = _manifold(_extrude(plate, -T / 2, T / 2))
    for pair in groove_cutters().values():
        for K in pair:
            body = body - _manifold(_hull(K * SCALE_M))
    body = body + _manifold(_extrude(eye, -T / 4, T / 4))
    mesh = _trimesh(body)
    if not mesh.is_volume:
        raise ValueError("The AB plate is not one closed solid")
    return mesh


def ring_solid():
    """The C ring (trimesh, metres, ring frame)."""
    h = MEASURED["c_depth"] * SCALE_M
    body = _manifold(_extrude(_snap(c_outline()), -h / 2, h / 2))
    for K in tip_cones():
        body = body + _manifold(_hull(K * SCALE_M))
    mesh = _trimesh(body)
    if not mesh.is_volume:
        raise ValueError("The C ring is not one closed solid")
    return mesh


# ---------------------------------------------------------------- exact convex decomposition (profile "split")

def _clip(V, n, d):
    """conv(V) with n.x <= d, as vertices, or None when (nearly) empty (topogym.world.mujoco.prisms._clip)."""
    s = V @ n - d
    if (s <= 1e-12).all():
        return V
    if (s >= -1e-12).all():
        return None
    h = ConvexHull(V)
    edges = {tuple(sorted((a, b))) for t in h.simplices for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    out = [V[s <= 1e-12]] + [(V[i] + s[i] / (s[i] - s[j]) * (V[j] - V[i]))[None] for i, j in edges
                             if min(s[i], s[j]) < -1e-12 and max(s[i], s[j]) > 1e-12]
    Q = np.unique(np.vstack(out), axis=0)
    try:
        h = ConvexHull(Q)
    except Exception:
        return None
    return Q[h.vertices] if h.volume > 1e-18 else None


def _planes(K):
    """The facet planes (n, d) of conv(K), n.x <= d inside, one per distinct plane."""
    eq = ConvexHull(K).equations
    eq = eq / np.linalg.norm(eq[:, :3], axis=1, keepdims=True)
    return [(e[:3], -e[3]) for e in np.unique(np.round(eq, 12), axis=0)]


def _minus(Q, K):
    """conv(Q) minus conv(K) as disjoint convex vertex sets: peel off, plane by plane, the part of Q outside each of K's
    facets, the largest part first."""
    planes, rest, out = _planes(K), Q, []
    while planes and rest is not None:
        best = None
        for k, (n, d) in enumerate(planes):
            piece = _clip(rest, -n, -d)
            v = ConvexHull(piece).volume if piece is not None else 0.
            if best is None or v > best[0]:
                best = (v, k, piece)
        v, k, piece = best
        if piece is None:
            break                    # rest lies inside every remaining facet: it is Q's part inside K
        out.append(piece)
        n, d = planes.pop(k)
        rest = _clip(rest, n, d)
    return out


def _cells(poly, footprints):
    """poly cut along every footprint's outline: the faces of that arrangement inside poly (shapely, metres)."""
    from shapely.ops import polygonize, unary_union
    edges = unary_union([poly.boundary] + [f.boundary for f in footprints])
    faces = [f for f in polygonize(edges) if poly.contains(f.representative_point())]
    area = sum(f.area for f in faces)
    if abs(area - poly.area) > 1e-9 * poly.area:
        raise ValueError(f"cast_abc: the arrangement's faces cover {area} of {poly.area}")
    return faces


def _prisms(face, z0, z1, fatten_m=None):
    """A planar face extruded from z0 to z1 as fat convex prisms (topogym.world.mujoco.prisms.convex_partition)."""
    import shapely
    from topogym.world.mujoco.prisms import convex_partition
    tris = [np.asarray(t.exterior.coords)[:3] for t in shapely.constrained_delaunay_triangles(face).geoms]
    pieces, _ = convex_partition([np.vstack([np.c_[t, np.full(3, z0)], np.c_[t, np.full(3, z1)]]) for t in tris],
                                 fatten_m=fatten_m)
    return [V for V, _ in pieces]


def _width3(V):
    h = ConvexHull(V)
    p = V @ h.equations[:, :3].T
    return float(np.min(p.max(0) - p.min(0)))


def _finish(pieces, fatten_m):
    from topogym.world.mujoco.prisms import fatten
    out = [(V, np.asarray(trimesh.convex.convex_hull(V).faces)) for V in pieces]
    if fatten_m and min(_width3(V) for V in pieces) < fatten_m:
        out, _ = fatten(out, fatten_m)
    return out


def plate_parts(vertices=None, faces=None, fatten_m=.001):
    """Exact convex pieces of the AB plate (metres, plate frame), for MuJoCo (profile "split"), which collides a mesh as
    its convex hull: the hull would seal the windows and fill the notches. The outline is cut along every notch's
    footprint, each face partitioned into fat convex prisms, the prisms inside a footprint cut by that notch's two
    cutters plane by plane, the eyelet's part outside the outline added. (vertices, faces) is the declared collision
    mesh, which the caller checks the pieces against (topogym.world.mujoco.scene); it is not needed to build them."""
    T = ASSUMED["T"] * SCALE_M
    plate, eye = (_snap(g) for g in plate_outline())
    cutters = {k: tuple(K * SCALE_M for K in pair) for k, pair in groove_cutters().items()}
    prints = {k: _snap(f) for k, f in groove_footprints().items()}
    pieces = []
    for face in _cells(plate, list(prints.values())):
        inside = [k for k, f in prints.items() if f.buffer(GRID_M).contains(face)]
        cells = _prisms(face, -T / 2, T / 2)
        for k in inside:
            cut = []
            for Q in cells:
                parts = [Q]
                for K in cutters[k]:
                    parts = [p for P in parts for p in _minus(P, K)]
                cut += parts
            cells = cut
        pieces += cells
    rim = eye.difference(plate)
    for face in (rim.geoms if rim.geom_type == "MultiPolygon" else [rim]):
        if face.area > 1e-12:
            pieces += _prisms(face, -T / 4, T / 4)
    return _finish(pieces, fatten_m)


def ring_parts(vertices=None, faces=None, fatten_m=.001):
    """Exact convex pieces of the C ring (metres, ring frame): the outline's fat convex prisms and the two tip cones."""
    h = MEASURED["c_depth"] * SCALE_M
    ring = _snap(c_outline())
    pieces = []
    for face in (ring.geoms if ring.geom_type == "MultiPolygon" else [ring]):
        pieces += _prisms(face, -h / 2, h / 2)
    pieces += [K * SCALE_M for K in tip_cones()]
    return _finish(pieces, fatten_m)


# ---------------------------------------------------------------- the opening and the assets

# The opening (the family's pregrasp birth): each piece held in its hand, the C round the A's left leg. Poses in the
# ALOHA model frame (assets/robots/aloha/aloha.xml).
OPENING = dict(
    plate_pose=[[0.8660254037844387, 0.49999999999999994, 0.0, 0.09409137996071748],
                [-0.49999999999999994, 0.8660254037844387, 0.0, -0.05308012701892219],
                [0.0, 0.0, 1.0, 0.19999999999999998], [0.0, 0.0, 0.0, 1.0]],
    ring_pose=[[0.798635510047293, 0.0, -0.601815023152048, 0.164598421515879],
               [-0.601815023152048, 0.0, -0.798635510047293, -0.012958348754011681],
               [0.0, 1.0, 0.0, 0.19999999999999998], [0.0, 0.0, 0.0, 1.0]],
    # the TCP frame (aloha.xml site <side>/gripper: x approach, y the fingers' closing direction) in the held piece
    grasp_left=[[1.0, 0.0, 0.0, 0.013], [0.0, 0.0, -1.0, 0.0054], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    grasp_right=[[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.056], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    seed_q=[0.15128297714525224, 0.4299359595374724, 0.2773263161747356, 0.8888054156806583, -0.9353888217181712,
            0.9395961216857286, 0., 0., -0.014025989483866498, -0.18925383475225252, 0.17976160573391314, 0.,
            1.580288555813236, -2.509846819835758, 0., 0.],
    # finger joints with both pads on the piece's faces (the plate is 19 mm thick, the C 15 mm deep)
    finger_q=(.0174, .0150),
)


def _write_urdf(path, name, mesh, color):
    root = ET.Element("robot", name=name)
    link = ET.SubElement(root, "link", name=name)
    props = mesh.mass_properties
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(map(repr, map(float, props.center_mass))), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=repr(float(props.mass)))
    ET.SubElement(inertial, "inertia", **{f"i{a}{b}": repr(float(props.inertia[j, k])) for a, b, j, k in
        (("x", "x", 0, 0), ("y", "y", 1, 1), ("z", "z", 2, 2), ("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2))})
    for kind in ("visual", "collision"):
        node = ET.SubElement(link, kind)
        ET.SubElement(ET.SubElement(node, "geometry"), "mesh", filename=f"{name}.obj")
        if kind == "visual":
            ET.SubElement(ET.SubElement(node, "material", name=name), "color", rgba=" ".join(map(str, color)))
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")


def build_cast_abc(folder):
    """Write both pieces (OBJ + URDF, metres) into folder/family and return (assembly, report, initial 16 ALOHA joints).
    The C ring comes first: it is the target piece."""
    from topogym.assets.geometry.interlocked_claws import load_reference, retarget_qpos, tcp_frames
    folder = Path(folder)
    root = folder / "family"
    root.mkdir(parents=True, exist_ok=False)
    poses = dict(c_ring=np.asarray(OPENING["ring_pose"]), ab_plate=np.asarray(OPENING["plate_pose"]))
    grasp = dict(ab_plate=np.asarray(OPENING["grasp_left"]), c_ring=np.asarray(OPENING["grasp_right"]))
    targets = [poses["ab_plate"] @ grasp["ab_plate"], poses["c_ring"] @ grasp["c_ring"]]
    reference = load_reference()
    q = retarget_qpos(reference, np.asarray(OPENING["seed_q"], float), targets)
    q[[6, 7]], q[[14, 15]] = OPENING["finger_q"]
    bodies, parts = [], []
    for (name, mesh), color in zip((("c_ring", ring_solid()), ("ab_plate", plate_solid())), COLORS):
        mesh.density = DENSITY
        mesh.export(root / f"{name}.obj", digits=12, include_texture=False)
        _write_urdf(root / f"{name}.urdf", name, mesh, color)
        bodies.append(SimpleNamespace(name=name, mesh=mesh, pose=poses[name], color=color, asset_path=root / f"{name}.urdf"))
        parts.append(dict(name=name, mass_kg=float(mesh.mass), volume_m3=float(mesh.volume),
                          bounds_m=mesh.bounds.tolist(), triangles=int(len(mesh.faces)), watertight=bool(mesh.is_watertight)))
    tcp = tcp_frames(reference, q)
    report = dict(family="hanayama_cast_abc", version="cast_abc_v4", scale_m_per_bar_width=SCALE_M, density_kg_m3=DENSITY,
                  measured=MEASURED, assumed=ASSUMED,
                  parts=parts, initial_poses={k: v.tolist() for k, v in poses.items()}, initial_robot_qpos=q.tolist(),
                  tcp_position_error_m=max(float(np.linalg.norm(a[:3, 3] - b[:3, 3])) for a, b in zip(tcp, targets)),
                  collision="each piece's exact convex decomposition (profile split: cast_abc.plate_parts, ring_parts)",
                  robot_geometry_modified=False, initialization="both pieces held in their start configuration")
    (folder / "geometry_report.json").write_text(json.dumps(report, indent=2) + "\n")
    scene = trimesh.Scene()
    for body in bodies:
        scene.add_geometry(body.mesh, node_name=body.name, transform=body.pose)
    scene.export(folder / "assembly.glb")
    return SimpleNamespace(bodies=bodies, reference={"density_kg_m3": DENSITY}), report, q
