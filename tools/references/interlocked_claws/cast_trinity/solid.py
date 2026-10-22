"""凸块并成一整件（manifold3d，双精度）和 MuJoCo 场景构建器的精确分解检查（照 topogym/world/mujoco/scene.py 的 _family_parts）。"""
import numpy as np
import trimesh
from scipy.spatial import ConvexHull


def union(cells):
    import manifold3d
    body = None
    for V in cells:
        h = trimesh.convex.convex_hull(np.asarray(V, np.float64))
        m = manifold3d.Manifold(manifold3d.Mesh64(vert_properties=np.asarray(h.vertices, np.float64),
                                                  tri_verts=np.asarray(h.faces, np.uint64)))
        body = m if body is None else body + m
    out = body.to_mesh64()
    return trimesh.Trimesh(np.asarray(out.vert_properties, np.float64)[:, :3], np.asarray(out.tri_verts, np.int64), process=False)


def check(cells, mesh):
    """返回 dict(volume_rel_error, max_part_outside_m, max_uncovered_m)，和场景构建器的三项一样。"""
    hulls = [ConvexHull(np.asarray(v, np.float64)) for v in cells]
    volume = float(sum(h.volume for h in hulls))
    points = np.vstack([np.asarray(v, np.float64) for v in cells])
    signed = trimesh.proximity.signed_distance(mesh, points)
    outside = float(max(0., -signed.min()))
    probe = np.r_[mesh.vertices, mesh.triangles_center]
    depth = np.full(len(probe), np.inf)
    for h in hulls:
        depth = np.minimum(depth, np.max(probe @ h.equations[:, :3].T + h.equations[:, 3], axis=1))
    return dict(volume_rel_error=float(abs(volume - mesh.volume) / mesh.volume), max_part_outside_m=outside,
                max_uncovered_m=float(max(0., depth.max())), parts=len(cells), volume_m3=volume)
