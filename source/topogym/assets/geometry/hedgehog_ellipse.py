"""Hedgehog medium cage: waisted windows, narrow slots with one wide band.

Replaced the six 58 mm elliptical windows on 2026-09-29.
"""
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .hedgehog_in_cage import write_body
from topogym.assets.variation_resources import unpack_assets


BALL_QUAT_XYZW = (.7341127330265252, -.3109226781652186,
                 -.09424941943357217, -.5962571847343461)


def window_parts(wide_width, narrow_width, band_center, band_half_height, ramp, slot_half_height):
    """Convex (tangent, z) outlines: lower slot, wide band with ramps, upper slot."""
    n, w = narrow_width/2, wide_width/2
    low, high = band_center-band_half_height, band_center+band_half_height
    overlap = .001  # Slots enter the band so the union has no coplanar seams.
    return [[(-n, -slot_half_height), (n, -slot_half_height), (n, low-ramp+overlap), (-n, low-ramp+overlap)],
            [(-n, low-ramp), (n, low-ramp), (w, low), (w, high), (n, high+ramp), (-n, high+ramp), (-w, high), (-w, low)],
            [(-n, high+ramp-overlap), (n, high+ramp-overlap), (n, slot_half_height), (-n, slot_half_height)]]


# The shell: a closed 192-gon tube (outer radius, total height) with a 192-gon cavity of CAVITY_HEIGHT, i.e. 8 mm ends.
HEIGHT, INNER, OUTER, CAVITY_HEIGHT, SECTIONS = .240, .082, .090, .224, 192
SLOT_HALF_HEIGHT = .104


def window_cutters(window_count, wide_width, narrow_width, band_center, band_half_height, ramp):
    """Window centres (deg) and the convex cutters: per window its lower slot, wide band and upper slot, each a prism
    along the window's radial direction through the whole wall."""
    parts = window_parts(wide_width, narrow_width, band_center, band_half_height, ramp, SLOT_HALF_HEIGHT)
    centers = -90+np.arange(window_count)*360/window_count
    cutters = []
    for angle in np.deg2rad(centers):
        radial = np.array([np.cos(angle), np.sin(angle), 0.])
        tangent = np.array([-np.sin(angle), np.cos(angle), 0.])
        for outline in parts:
            points = [t*tangent+z*np.array([0., 0., 1.])+r*radial
                      for t, z in outline for r in (INNER-.02, OUTER+.02)]
            cutters.append(trimesh.convex.convex_hull(np.array(points)))
    return centers, cutters


def build_waisted_cage(folder, ball_resource, window_count=4, wide_width=.105, narrow_width=.048,
                       band_center=.020, band_half_height=.055, ramp=.015):
    folder = Path(folder)
    unpack_assets(ball_resource, folder)
    height, inner, outer = HEIGHT, INNER, OUTER
    slot_half_height = SLOT_HALF_HEIGHT
    shell = trimesh.boolean.difference([
        trimesh.creation.cylinder(radius=outer, height=height, sections=SECTIONS),
        trimesh.creation.cylinder(radius=inner, height=CAVITY_HEIGHT, sections=SECTIONS),
    ], engine="manifold")
    centers, cutters = window_cutters(window_count, wide_width, narrow_width, band_center, band_half_height, ramp)
    cage = trimesh.boolean.difference([shell, *cutters], engine="manifold")
    if not cage.is_watertight or len(cage.split()) != 1:
        raise ValueError("Waisted cage must be one closed connected solid")
    # Keep the concave collision surface: a global convex hull seals the windows.
    write_body(folder, "cage", cage, [cage], (1., 1., 1., 1.), 650., finish="wood")
    ball = trimesh.load(folder/"hedgehog.obj", force="mesh", process=False)
    z = -.112 - Rotation.from_quat(BALL_QUAT_XYZW).apply(ball.vertices)[:, 2].min() + .0002
    q = np.asarray(BALL_QUAT_XYZW)[[3, 0, 1, 2]].tolist()
    post = 2*inner*np.sin(np.pi/window_count)-wide_width  # chord left between wide bands at the inner wall
    report = dict(family="waisted_windows", window_count=window_count, window_pitch_deg=360/window_count,
        window_centers_deg=centers.tolist(), cage_height_m=height, cage_outer_radius_m=outer,
        cage_inner_radius_m=inner, window_slot_half_height_m=slot_half_height,
        window_wide_width_m=wide_width, window_narrow_width_m=narrow_width,
        window_band_center_m=band_center, window_band_half_height_m=band_half_height, window_ramp_m=ramp,
        minimum_post_chord_at_inner_wall_m=float(post), closed_end_thickness_m=.008, finish="dark_wood_texture",
        initial_pose=dict(position_m=[0., 0., float(z)], quaternion_wxyz=q),
        core_diameter_m=.054, spike_count=10, ball_source=ball_resource,
        ball_minimum_projected_width_m=.097, geometry_connected=True,
        historical_certificate_transferred=False, full_exit_path_verified=False, robot_solution_verified=False)
    (folder/"asset_report.json").write_text(json.dumps(report, indent=2)+"\n")
    return cage, ball, -height/2, report


def waisted_cage_hulls(vertices=None, faces=None, window_count=4, wide_width=.105, narrow_width=.048,
                       band_center=.020, band_half_height=.055, ramp=.015):
    """The waisted cage as exact convex pieces, for an engine that collides convex shapes only (MuJoCo: a mesh collides
    as its convex hull, which would seal the windows). Built from the same primitives as build_waisted_cage: the two
    closed ends (192-gon prisms), and the wall between them cut into its 192 wedges by the radial planes through the
    polygon vertices; each wedge minus the window cutters is split along the cutters' face planes (horizontal ones
    first), and pieces of one wedge whose union is convex are merged again. (vertices, faces) is the declared collision mesh, which the caller
    checks the pieces against (topogym.world.mujoco.scene); it is not needed to build them.
    Returns [(vertices (n, 3), faces (m, 3))] in the cage frame."""
    from scipy.spatial import ConvexHull
    outer = trimesh.creation.cylinder(radius=OUTER, height=HEIGHT, sections=SECTIONS).vertices
    cavity = trimesh.creation.cylinder(radius=INNER, height=CAVITY_HEIGHT, sections=SECTIONS).vertices

    def ring(v, r):                     # the polygon's vertices in angle order (the same coordinates as the mesh's)
        v = v[np.isclose(np.linalg.norm(v[:, :2], axis=1), r) & (v[:, 2] > 0)][:, :2]
        return v[np.argsort(np.arctan2(v[:, 1], v[:, 0]))]

    po, pi = ring(outer, OUTER), ring(cavity, INNER)
    if len(po) != SECTIONS or len(pi) != SECTIONS:
        raise ValueError("Unexpected cylinder tessellation")
    if not np.allclose(np.arctan2(po[:, 1], po[:, 0]), np.arctan2(pi[:, 1], pi[:, 0]), atol=1e-12):
        raise ValueError("The cage's outer and inner polygons must share their vertex angles")
    top, mid = HEIGHT/2, CAVITY_HEIGHT/2
    pieces = [np.r_[np.c_[po, np.full(len(po), z0)], np.c_[po, np.full(len(po), z1)]]
              for z0, z1 in ((mid, top), (-top, -mid))]
    _, cutters = window_cutters(window_count, wide_width, narrow_width, band_center, band_half_height, ramp)
    planes = []
    for hull in cutters:              # each cutter as unique half-spaces n.x <= d
        rows = {}
        for n, x in zip(hull.face_normals, hull.triangles[:, 0]):
            key = tuple(np.round(np.r_[n, n @ x], 12))
            rows.setdefault(key, (n, float(n @ x)))
        # Horizontal faces first, then the ramps, then the vertical sides: a wedge is cut above and below a window
        # before beside it, so no split leaves a thin column at a window's side (0.2-0.5 mm otherwise).
        planes.append(sorted(rows.values(), key=lambda row: -abs(row[0][2])))
    tol, vol_min = 1e-12, 1e-15

    def volume(p):
        try:
            return ConvexHull(p).volume
        except Exception:
            return 0.

    def clip(p, n, d):                # conv(p) with n.x <= d, as its vertices, or None if (nearly) empty
        s = p @ n - d
        if (s <= tol).all():
            return p
        if (s >= -tol).all():
            return None
        hull = ConvexHull(p)
        edges = {tuple(sorted((a, b))) for t in hull.simplices for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
        out = [p[s <= tol]] + [(p[i] + s[i]/(s[i]-s[j])*(p[j]-p[i]))[None] for i, j in edges
                               if (s[i] < -tol < tol < s[j]) or (s[j] < -tol < tol < s[i])]
        q = np.unique(np.vstack(out), axis=0)
        if len(q) < 4 or volume(q) < vol_min:
            return None
        return q[ConvexHull(q).vertices]

    def minus(p, halfspaces):         # conv(p) minus a convex cutter: convex pieces split along the cutter's planes
        inside = p
        for n, d in halfspaces:
            inside = clip(inside, n, d)
            if inside is None:
                return [p]
        out, rest = [], p
        for n, d in halfspaces:
            part = clip(rest, -n, -d)
            if part is not None:
                out.append(part)
            rest = clip(rest, n, d)
            if rest is None:
                break
        return out

    for k in range(SECTIONS):
        j = (k + 1) % SECTIONS
        quad = np.array([pi[k], pi[j], po[j], po[k]])
        wedge = [np.r_[np.c_[quad, np.full(4, -mid)], np.c_[quad, np.full(4, mid)]]]
        for halfspaces in planes:
            wedge = [q for p in wedge for q in minus(p, halfspaces)]
        merged = True                 # merge pieces of this wedge whose union is exactly convex
        while merged and len(wedge) > 1:
            merged = False
            for a in range(len(wedge)):
                for b in range(a + 1, len(wedge)):
                    u = np.r_[wedge[a], wedge[b]]
                    if volume(u) <= (volume(wedge[a]) + volume(wedge[b])) * (1 + 1e-9):
                        hull = ConvexHull(u)
                        wedge[a] = u[hull.vertices]
                        del wedge[b]
                        merged = True
                        break
                if merged:
                    break
        pieces += wedge
    out = []
    for p in pieces:
        m = trimesh.convex.convex_hull(p)
        out.append((np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)))
    return out
