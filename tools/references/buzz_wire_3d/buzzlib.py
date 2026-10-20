"""Small helpers for the buzz MuJoCo work: exact wand / wire clearance from the scene's capsules."""
import numpy as np
import mujoco


def role_geoms(world, role):
    col = world._contact_collector
    code = col.names.index(role)
    return np.flatnonzero(col.role == code)


def capsules(world, ids):
    """(a, b, r): end points and radii of capsule geoms `ids` now (world frame)."""
    m, d = world.model, world.data
    ids = np.asarray([g for g in ids if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_CAPSULE])
    c = np.asarray(d.geom_xpos)[ids]
    z = np.asarray(d.geom_xmat)[ids].reshape(-1, 3, 3)[:, :, 2]
    h = np.asarray(m.geom_size)[ids, 1][:, None]
    return c - h * z, c + h * z, np.asarray(m.geom_size)[ids, 0], ids


def segment_distance(p1, q1, p2, q2):
    """Closest distance between segments p1q1 (n) and p2q2 (m): an (n, m) array (Ericson, Real-Time Collision Detection)."""
    d1 = (q1 - p1)[:, None, :]
    d2 = (q2 - p2)[None, :, :]
    r = p1[:, None, :] - p2[None, :, :]
    a = np.einsum("ijk,ijk->ij", d1, d1) + 0 * r[..., 0]
    e = np.einsum("ijk,ijk->ij", d2, d2) + 0 * r[..., 0]
    f = np.einsum("ijk,ijk->ij", d2, r)
    c = np.einsum("ijk,ijk->ij", d1, r)
    b = np.einsum("ijk,ijk->ij", d1, d2)
    denom = a * e - b * b
    s = np.where(denom > 1e-30, np.clip((b * f - c * e) / np.where(denom > 1e-30, denom, 1), 0, 1), 0.)
    t = (b * s + f) / np.where(e > 1e-30, e, 1)
    tn = np.clip(t, 0, 1)
    s = np.where(t < 0, np.clip(-c / np.where(a > 1e-30, a, 1), 0, 1), np.where(t > 1, np.clip((b - c) / np.where(a > 1e-30, a, 1), 0, 1), s))
    t = tn
    c1 = p1[:, None, :] + d1 * s[..., None]
    c2 = p2[None, :, :] + d2 * t[..., None]
    return np.linalg.norm(c1 - c2, axis=-1)


def clearance(world, cache={}):
    """Smallest surface distance (m) between the wand's metal capsules and the wire's metal capsules, and the pair."""
    key = id(world)
    if key not in cache:
        cache.clear()
        cache[key] = (role_geoms(world, "wand_metal"), role_geoms(world, "wire_metal"))
    wand, wire = cache[key]
    a1, b1, r1, i1 = capsules(world, wand)
    a2, b2, r2, i2 = capsules(world, wire)
    dist = segment_distance(a1, b1, a2, b2) - r1[:, None] - r2[None, :]
    k = np.unravel_index(int(np.argmin(dist)), dist.shape)
    return float(dist[k]), int(i1[k[0]]), int(i2[k[1]])
