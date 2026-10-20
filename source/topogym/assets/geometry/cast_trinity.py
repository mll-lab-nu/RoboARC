"""Hanayama Cast Trinity (level 6, design Kyoo Wong): three Lambda-shaped pieces, rebuilt from photos as model v12.

No 3D model of the puzzle is published; the pieces are built from rods, flat bands and blocks with the official
mechanism (runs/cast_trinity_design_20261011: README.md, PROGRESS.md, ref/official_sequence.md; the user accepted on
2026-10-12 a model adjusted from the photos as long as the solution stays as involved as the original, feasible and
without shortcut). Scale: 150 mm long (the real piece about 55 mm), density 600 kg/m^3 as every interlocked_claws piece.

Piece frame: x from the tip (x = 0) to the head end, y across (the round leg at +y, the band leg at -y; H is T's
mirror image), z normal to the piece's plane.
  tip hook   a thin rod bent into a half circle of radius RHO = SLOT_HALF + R_TIP round (0, 0).
  round leg  a rod thickening from R_TIP to R_LEG, its inner side on the window's edge y = window(x).
  band leg   a thin rod to X_BAND0, then a flat band BAND_W wide: BAND_T thick, NECK_T thick for the last 48 mm before the
             heads (the neck: the only part thinner than the gap between two heads).
  window     half width window(x): the tip slot SLOT_HALF up to X_SLOT_END, then a round opening closed by the heads.
  heads      x in [X_HEAD, LEN]. T, H: two blocks side by side across y with a SLIT-wide vertical gap; B: two L-shaped
             blocks interlocking along z, the gap between them horizontal (turned 90 degrees).
Each piece is built from exact convex cells (cells(kind)) that touch along shared faces only and add up to the piece;
MuJoCo collides those cells (profile "split": t_parts, h_parts, b_parts), so the windows and the gaps stay open.

The opening (the family's pregrasp birth): the three tips hooked into one another's tip slots, every pair linked; T in
the left hand (pads on the band's neck, the hand outside the band), B in the right hand (pads on its upper head plate),
H hanging. Poses in the ALOHA model frame (arm bases at z = 0.02, the family's source robot-mount coordinates) from the
reference plan's first state at the placement chosen by a search over placements in which both arms follow the whole
plan by inverse kinematics (runs/cast_trinity_design_20261011/v2: plan_hands_v3, iks_hands_v3_full)."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

MM = 1e-3
DENSITY = 600.
KINDS = ("T", "H", "B")
NAMES = {"T": "trinity_t", "H": "trinity_h", "B": "trinity_b"}
COLORS = {"T": (.85, .68, .25, 1.), "H": (.55, .60, .68, 1.), "B": (.72, .45, .30, 1.)}

LEN = 150 * MM
X_HEAD = 118 * MM
SLOT_HALF = 8.0 * MM
X_SLOT_END = 62 * MM
X_ROUND_MID = 96 * MM
ROUND_HALF = 20 * MM
R_TIP = 3.0 * MM
RHO = SLOT_HALF + R_TIP
R_LEG = 7.0 * MM
X_THICK = 40 * MM
X_BAND0 = 22 * MM
BAND_W = 18 * MM
BAND_W0 = 8 * MM
NECK_T = 4.0 * MM
BAND_T = 5.5 * MM
X_NECK = 66 * MM
X_FLAT = 88 * MM
SLIT = 5.0 * MM
HEAD_W = 28 * MM
HEAD_T = 20 * MM
B_PLATE = 9.0 * MM
Y_TAB = 18.0 * MM
N_LEG = 16
N_ARC = 10


# ---------------------------------------------------------------- outline

def window(x):
    if x <= X_SLOT_END:
        return SLOT_HALF
    if x <= X_ROUND_MID:
        u = (x - X_SLOT_END) / (X_ROUND_MID - X_SLOT_END)
        return SLOT_HALF + (ROUND_HALF - SLOT_HALF) * np.sin(u * np.pi / 2)
    u = (x - X_ROUND_MID) / (X_HEAD - X_ROUND_MID)
    return ROUND_HALF * (1 - .15 * u * u)


def r_round(x):
    u = np.clip((x - RHO) / (X_THICK - RHO), 0, 1)
    return R_TIP + (R_LEG - R_TIP) * (3 * u * u - 2 * u ** 3)


def band_t(x):
    u = np.clip((x - X_NECK) / (6 * MM), 0, 1)
    return BAND_T + (NECK_T - BAND_T) * u


def band_w(x):
    u = np.clip((x - X_BAND0) / (X_FLAT - X_BAND0), 0, 1)
    return BAND_W0 + (BAND_W - BAND_W0) * (3 * u * u - 2 * u ** 3)


def _xs(x0, x1, n):
    return np.linspace(x0, x1, n + 1)


def chain_rod():
    """Round leg (head end -> hook), hook, and the band leg's thin rod (-> X_BAND0): points and radii."""
    pts, rad = [], []
    for x in _xs(X_HEAD, RHO, N_LEG)[:-1]:
        pts.append(np.array([x, window(x) + r_round(x), 0.]))
        rad.append(r_round(x))
    for k in range(N_ARC + 1):
        f = np.pi * k / N_ARC
        pts.append(np.array([RHO - RHO * np.sin(f), RHO * np.cos(f), 0.]))
        rad.append(R_TIP)
    for x in _xs(RHO, X_BAND0, 3)[1:]:
        u = (x - RHO) / (X_BAND0 - RHO)
        r = R_TIP + (BAND_T / 2 - R_TIP) * u
        pts.append(np.array([x, -(window(x) + r), 0.]))
        rad.append(r)
    return pts, rad


def chain_band():
    return [np.array([x, -(window(x) + band_w(x) / 2), 0.]) for x in _xs(X_BAND0, X_HEAD, N_LEG)]


def _box(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def heads(kind):
    g = SLIT / 2
    if kind in ("T", "H"):
        return [_box([X_HEAD, g, -HEAD_T / 2], [LEN, g + HEAD_W, HEAD_T / 2]),
                _box([X_HEAD, -g - HEAD_W, -HEAD_T / 2], [LEN, -g, HEAD_T / 2])]
    # B: two L-shaped heads. Upper head = upper plate + the band side's tab reaching down; lower head = lower plate + the
    # round side's tab reaching up; the gap between them is SLIT wide everywhere.
    yb = window(X_HEAD) + band_w(X_HEAD) + 4 * MM
    yr = window(X_HEAD) + 2 * R_LEG + 4 * MM
    P = B_PLATE
    return [_box([X_HEAD, -yb, g], [LEN, Y_TAB - 2 * g, g + P]),
            _box([X_HEAD, -yb, -g - P], [LEN, -Y_TAB, g]),
            _box([X_HEAD, -(Y_TAB - 2 * g), -g - P], [LEN - 6 * MM, yr, -g]),
            _box([X_HEAD, Y_TAB, -g], [LEN - 6 * MM, yr, g + P])]


# ---------------------------------------------------------------- exact convex cells

def _frame(d, up=(0., 0., 1.)):
    d = d / np.linalg.norm(d)
    e1 = np.cross(np.asarray(up, float), d)
    if np.linalg.norm(e1) < 1e-9:
        e1 = np.cross([1., 0, 0], d)
    e1 /= np.linalg.norm(e1)
    return d, e1, np.cross(d, e1)


def _cut_polygon(c, d, e1, e2, r, m, n, p):
    """The regular m-gon prism of axis (c, d) and circumradius r cut by the plane through p with normal n."""
    out = []
    for a in 2 * np.pi * (np.arange(m) + .5) / m:
        q = c + r * (np.cos(a) * e1 + np.sin(a) * e2)
        out.append(q + np.dot(n, p - q) / np.dot(n, d) * d)
    return np.array(out)


def tube(pts, r, m=12, start=None, end=None):
    """A polygonal rod along the polyline pts (radius r per point; the polygon's sides tangent to radius r), mitred at the
    bisecting planes; start / end: (n, p) cut planes. One polygon per joint plane, shared by both neighbours."""
    pts = [np.asarray(p, float) for p in pts]
    R = np.broadcast_to(np.asarray(r, float), (len(pts),)) / np.cos(np.pi / m)
    nseg = len(pts) - 1
    dirs = [(pts[i + 1] - pts[i]) / np.linalg.norm(pts[i + 1] - pts[i]) for i in range(nseg)]
    planes = []
    for i in range(nseg + 1):
        if i == 0:
            planes.append(start if start is not None else (dirs[0], pts[0]))
        elif i == nseg:
            planes.append(end if end is not None else (dirs[-1], pts[-1]))
        else:
            nb = dirs[i - 1] + dirs[i]
            planes.append((nb / np.linalg.norm(nb), pts[i]))
    polys = []
    for j in range(nseg + 1):
        d, e1, e2 = _frame(dirs[max(j - 1, 0)])
        polys.append(_cut_polygon(pts[max(j - 1, 0)], d, e1, e2, R[j], m, *planes[j]))
    return [np.vstack([polys[i], polys[i + 1]]) for i in range(nseg)]


def _rect(c, d, w, t):
    d, e1, e2 = _frame(d)
    return np.array([c + su * w / 2 * e1 + sv * t / 2 * np.cross(d, e1) for su in (-1, 1) for sv in (-1, 1)])


def _on_plane(R, d, n, p):
    return np.array([q + np.dot(n, p - q) / np.dot(n, d) * d for q in R])


def band(pts, widths, ts, start=None, end=None):
    """A flat band through pts (width and thickness per point): one rectangle per point, each cell the hull of two."""
    pts = [np.asarray(p, float) for p in pts]
    n = len(pts)
    secs = []
    for i in range(n):
        d = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
        d = d / np.linalg.norm(d)
        R = _rect(pts[i], d, widths[i], ts[i])
        if i == 0 and start is not None:
            R = _on_plane(R, d, *start)
        if i == n - 1 and end is not None:
            R = _on_plane(R, d, *end)
        secs.append(R)
    return [np.vstack([secs[i], secs[i + 1]]) for i in range(n - 1)]


def cells(kind):
    """The piece's exact convex cells (vertex arrays, metres, piece frame)."""
    if kind == "H":
        return [V * np.array([1., -1., 1.]) for V in cells("T")]
    hcut = (np.array([-1., 0, 0]), np.array([X_HEAD, 0, 0]))
    pts, rad = chain_rod()
    bp = chain_band()
    d = bp[1] - bp[0]
    joint = (d / np.linalg.norm(d), bp[0])
    out = tube(pts, rad, start=hcut, end=joint)
    out += band(bp, [band_w(p[0]) for p in bp], [band_t(p[0]) for p in bp], start=joint, end=(np.array([1., 0, 0]), np.array([X_HEAD, 0, 0])))
    out += heads(kind)
    return out


def centerline(kind):
    """Closed centre line (round leg + hook + band leg + a segment across the gap between the heads), for the linking
    number of two pieces (Gauss integral)."""
    if kind == "H":
        return centerline("T") * np.array([1., -1., 1.])
    pts, _ = chain_rod()
    L = pts + chain_band()[1:]
    xc = X_HEAD + 12 * MM
    a, b = L[-1], L[0]
    return np.array(L + [np.array([xc, a[1], a[2]]), np.array([xc, 0., .5 * (a[2] + b[2])]), np.array([xc, b[1], b[2]])])


def solid(kind):
    """The piece as one watertight mesh: the union of its cells (manifold3d, double precision)."""
    import manifold3d
    body = None
    for V in cells(kind):
        h = trimesh.convex.convex_hull(np.asarray(V, np.float64))
        m = manifold3d.Manifold(manifold3d.Mesh64(vert_properties=np.asarray(h.vertices, np.float64),
                                                  tri_verts=np.asarray(h.faces, np.uint64)))
        body = m if body is None else body + m
    out = body.to_mesh64()
    return trimesh.Trimesh(np.array(out.vert_properties, np.float64)[:, :3].copy(), np.array(out.tri_verts, np.int64), process=False)


def _parts(kind):
    out = []
    for V in cells(kind):
        h = trimesh.convex.convex_hull(np.asarray(V, np.float64))
        out.append((np.asarray(h.vertices, np.float64), np.asarray(h.faces, np.int64)))
    return out


def t_parts(vertices=None, faces=None):
    """Exact convex pieces of T for MuJoCo (profile split); (vertices, faces) is the declared mesh the caller checks against."""
    return _parts("T")


def h_parts(vertices=None, faces=None):
    return _parts("H")


def b_parts(vertices=None, faces=None):
    return _parts("B")


# ---------------------------------------------------------------- the opening and the assets

OPENING = dict(
    # piece poses (4x4, metres, ALOHA model frame): the reference plan's first state at the chosen placement (the plan
    # turned 95 deg about the vertical and shifted to (-0.04, 0.05, 0.48) m from the arms' mounting point)
    poses=dict(
        T=[[0.7672558119947084, -0.1417831433493789, 0.6254726686453295, -0.04],
           [0.27925827763381933, -0.8040921632055845, -0.5248338855717649, 0.05],
           [0.5773502691896257, 0.5773502691896256, -0.5773502691896258, 0.5],
           [0.0, 0.0, 0.0, 1.0]],
        H=[[-0.5950174228042772, 0.0798783506718448, 0.7997335279037024, -0.01884887328849402],
           [0.8036128185347062, 0.043434591093546984, 0.5935653916656882, 0.036155609647838795],
           [0.012676925733845728, 0.9958578640501273, -0.09003560497634475, 0.4969872045102647],
           [0.0, 0.0, 0.0, 1.0]],
        B=[[-0.8210316177731227, 0.4329296942412682, 0.37212761582689763, -0.019243688396069106],
           [-0.4865286658984572, -0.1896154287383265, -0.8528398715136668, 0.05702296265264338],
           [-0.2986585673907859, -0.881259251882522, 0.3663129687788037, 0.5134877005171375],
           [0.0, 0.0, 0.0, 1.0]]),
    # the TCP frame (aloha.xml site <side>/gripper: x approach, y the fingers' closing direction) in the held piece:
    # left on T's band neck at x = 76 mm (hand outside the band, pads across its 4 mm); right pinching B's whole head
    # block (both plates, 23 mm) on its round-leg side, approaching from +y
    grasp_left=[[0.4535389230917378, -0.0, -0.8912364698781052, 0.07555555555555556],
           [0.8912364698781052, -0.0, 0.4535389230917378, -0.023567229061059417],
           [-0.0, -1.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]],
    grasp_right=[[0.0, 0.0, -1.0, 0.131],
           [-1.0, 0.0, 0.0, 0.028],
           [0.0, 1.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]],
    width_left=0.004, width_right=0.023,
    seed_q=[0.46931688522611104, 0.5067525008414355, -0.48584844586641035, 0.6799069760977693, -1.656293274799944, 0.3402168771085542, 0.0, 0.0, 0.05942755959688172, 0.5642809247228623, -0.2451043394706207, -2.9232229344269194, 1.447363273562814, -2.7050890769425386, 0.0, 0.0],
    source="runs/cast_trinity_design_20261011/v2/out: plan_hands_v5r19b, iks_hands_v5r19b_full rank 0 (full_v5)",
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


def build_cast_trinity(folder):
    """Write the three pieces (OBJ + URDF, metres) into folder/family and return (assembly, report, initial 16 ALOHA
    joints). T comes first."""
    from topogym.assets.geometry.interlocked_claws import load_reference, retarget_qpos, tcp_frames
    folder = Path(folder)
    root = folder / "family"
    root.mkdir(parents=True, exist_ok=False)
    poses = {k: np.asarray(OPENING["poses"][k]) for k in KINDS}
    targets = [poses["T"] @ np.asarray(OPENING["grasp_left"]), poses["B"] @ np.asarray(OPENING["grasp_right"])]
    reference = load_reference()
    q = retarget_qpos(reference, np.asarray(OPENING["seed_q"], float), targets)
    q[[6, 7]] = (OPENING["width_left"] + .0157) / 2           # both pads on the piece (pad gap = 2 q - 0.0157 m)
    q[[14, 15]] = (OPENING["width_right"] + .0157) / 2
    bodies, parts = [], []
    for k in KINDS:
        name, mesh = NAMES[k], solid(k)
        mesh.density = DENSITY
        mesh.export(root / f"{name}.obj", digits=12, include_texture=False)
        _write_urdf(root / f"{name}.urdf", name, mesh, COLORS[k])
        bodies.append(SimpleNamespace(name=name, mesh=mesh, pose=poses[k], color=COLORS[k], asset_path=root / f"{name}.urdf"))
        parts.append(dict(name=name, mass_kg=float(mesh.mass), volume_m3=float(mesh.volume), bounds_m=mesh.bounds.tolist(),
                          triangles=int(len(mesh.faces)), watertight=bool(mesh.is_watertight), cells=len(cells(k))))
    tcp = tcp_frames(reference, q)
    report = dict(family="hanayama_cast_trinity", version="cast_trinity_v12", density_kg_m3=DENSITY, parts=parts,
                  initial_poses={NAMES[k]: v.tolist() for k, v in poses.items()}, initial_robot_qpos=q.tolist(),
                  tcp_position_error_m=max(float(np.linalg.norm(a[:3, 3] - b[:3, 3])) for a, b in zip(tcp, targets)),
                  collision="each piece's exact convex decomposition (profile split: cast_trinity.t_parts, h_parts, b_parts)",
                  robot_geometry_modified=False, opening=OPENING.get("source"),
                  initialization="T in the left hand, B in the right, H hanging; the reference plan's first state")
    (folder / "geometry_report.json").write_text(json.dumps(report, indent=2) + "\n")
    scene = trimesh.Scene()
    for body in bodies:
        scene.add_geometry(body.mesh, node_name=body.name, transform=body.pose)
    scene.export(folder / "assembly.glb")
    return SimpleNamespace(bodies=bodies, reference={"density_kg_m3": DENSITY}), report, q
