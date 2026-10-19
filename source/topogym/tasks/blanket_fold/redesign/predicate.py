"""blanket_fold: the goal predicate (v3.1: v3 with the 1 mm layer gap, 18:35), measured against a target state from inverse generation, particle by
particle (material IDs).; argued in docs/rcb/redesign/blanket_fold.md.

A run meets its goal when the sheet is settled and intact (the family's gates, as in blanket_flatten: speed, stretch
<= 1.15, above the table) and all five terms hold:
  named points    the tier's graspable named points (12 mm inside the corners: the patch rule's complete patch)
                  within `named_mm` of their target positions;
  stacking order  per particle, the PANEL (material region between the target's creases) of the material directly
                  under it, or none; agreement with the target >= `stack_all` overall and >= `stack_covered` over the
                  particles that have something under them in the target;
  outline         IoU of the occupied 2 cm cells after the best rigid 2-D fit >= `iou`;
  flat            |dz| against the target heights <= `dz_mm`, nothing more than `over_mm` above the target's top;
  placement       the rigid fit's offset <= `offset_mm` and its angle <= `angle_deg`.
"""
from __future__ import annotations


import numpy as np
from scipy.spatial import cKDTree

CELL = .02
DEFAULT = dict(named_mm=30., stack_all=.9, stack_covered=.8, iou=.85, dz_mm=20., over_mm=10., offset_mm=40., angle_deg=5.)


def kabsch2(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1., d]) @ U.T
    return R, cb - R @ ca


def layer_map(P, n_all, x0, y0, nx, ny):
    per_layer = n_all / (.60 * .48) * CELL * CELL
    ix = np.clip(((P[:, 0] - x0) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((P[:, 1] - y0) / CELL).astype(int), 0, ny - 1)
    cnt = np.zeros((ny, nx))
    np.add.at(cnt, (iy, ix), 1.)
    return np.rint(cnt / per_layer)


def panels(uv, creases):
    lab = np.zeros(len(uv), int)
    for k, (a, b) in enumerate(creases or []):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        lab += (((uv - a) @ n) > 0).astype(int) << k
    return lab


# v3.1 (18:35): the layer gap for "below" is 1 mm, not 3 mm. The coarse flex is a 12 mm grid of r 4.5 mm spheres, and
# a layer resting in the holes of the layer below (grids in registry) sits sqrt((2r)^2 - (12 mm / sqrt 2)^2) = 3.0 mm
# above it, exactly at the old limit: such nested layers counted as "nothing below" (letter_flip's inside panel at
# 2.8-3.1 mm: stacking 0.80 -> 0.95; the triangle flap 0.88 / 0.17 -> 0.96 / 0.97). The same layer is excluded by the
# material distance (`far`), not by the gap. The breakers' scores barely move (pile 0.43 -> 0.35, half_mid_edge
# 0.898 -> 0.903, mirrored 0.0); every breaker still fails on its other terms.
def below_material(uv, P, r=.008, dz_lo=.001, dz_hi=.02, far=.03):
    """Per particle, the material coordinates of the nearest-below particle within r in xy (dz in (dz_lo, dz_hi), more
    than `far` away in material), else NaN. All neighbour pairs at once (cKDTree.query_pairs, both directions: the
    pairs query_ball_point lists, without its per-particle Python lists; a particle with no other neighbour has none),
    the same per-pair arithmetic, and per particle the smallest dz, ties to the smaller neighbour index (the sorted
    order query_ball_point gave), as the per-particle loop it replaces (speed lane, 2026-10-13)."""
    tree = cKDTree(P[:, :2])
    out = np.full((len(P), 2), np.nan)
    pairs = tree.query_pairs(r, output_type="ndarray")
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    nb = np.concatenate([pairs[:, 1], pairs[:, 0]])
    dz = P[i, 2] - P[nb, 2]
    ok = (dz > dz_lo) & (dz < dz_hi) & (np.linalg.norm(uv[nb] - uv[i], axis=1) > far)
    i, nb, dz = i[ok], nb[ok], dz[ok]
    if len(i):
        order = np.lexsort((nb, dz, i))                     # per particle: smallest dz, then the smaller neighbour
        first = order[np.r_[True, i[order][1:] != i[order][:-1]]]
        out[i[first]] = uv[nb[first]]
    return out


_TARGET = {}      # (uv, target, creases) bytes -> the target's panel per particle (constant for an episode)


def stacking(uv, T, A, creases):
    key = (uv.tobytes(), T.tobytes(), repr(creases))
    cached = _TARGET.get("key") == key
    if not cached:
        lab = panels(uv, creases)
        tree = cKDTree(uv)
    else:
        lab, tree, pt = _TARGET["lab"], _TARGET["tree"], _TARGET["pt"]

    def to_panel(B):
        out = np.full(len(B), -1)
        ok = ~np.isnan(B[:, 0])
        if ok.any():
            out[ok] = lab[tree.query(B[ok])[1]]
        return out
    if not cached:
        pt = to_panel(below_material(uv, T))
        _TARGET.clear()
        _TARGET.update(key=key, lab=lab, tree=tree, pt=pt)
    pa = to_panel(below_material(uv, A))
    same = pt == pa
    cov = pt >= 0
    return float(same.mean()), (float(same[cov].mean()) if cov.any() else 1.)


def named_point_uv(names=("LN", "LF", "RN", "RF"), inset=.012):
    pts = {}
    for nm in names:
        cu = -.3 if nm[0] == "L" else .3
        cv = .24 if nm[1] == "F" else -.24
        pts[nm] = (cu - np.sign(cu) * inset, cv - np.sign(cv) * inset)
    return pts


def evaluate(uv, T, A, creases, *, named=None, th=None):
    """T, A: target / achieved particle positions (N x 3, same material order); uv: material coordinates (N x 2)."""
    th = dict(DEFAULT, **(th or {}))
    R, t = kabsch2(A[:, :2], T[:, :2])
    Af = A[:, :2] @ R.T + t
    ang = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    off = float(np.linalg.norm(A[:, :2].mean(0) - T[:, :2].mean(0)))
    nerr = {}
    for nm, g in (named or named_point_uv()).items():
        k = int(np.argmin(np.sum((uv - np.asarray(g)) ** 2, axis=1)))
        nerr[nm] = round(1e3 * float(np.linalg.norm(A[k, :2] - T[k, :2])), 1)
    s_all, s_cov = stacking(uv, T, A, creases)
    x0 = min(Af[:, 0].min(), T[:, 0].min()) - CELL
    y0 = min(Af[:, 1].min(), T[:, 1].min()) - CELL
    nx = int((max(Af[:, 0].max(), T[:, 0].max()) - x0) / CELL) + 2
    ny = int((max(Af[:, 1].max(), T[:, 1].max()) - y0) / CELL) + 2
    La, Lt = layer_map(Af, len(A), x0, y0, nx, ny), layer_map(T[:, :2], len(A), x0, y0, nx, ny)
    occ = (La > 0) | (Lt > 0)
    iou = float(((La > 0) & (Lt > 0)).sum() / max(occ.sum(), 1))
    dz = float(np.abs(A[:, 2] - T[:, 2]).max())
    over = float(A[:, 2].max() - T[:, 2].max())
    m = dict(named_points_mm=nerr, stacking_all=round(s_all, 3), stacking_covered=round(s_cov, 3),
             outline_iou_fitted=round(iou, 3), dz_max_mm=round(1e3 * dz, 1), over_target_mm=round(1e3 * over, 1),
             placement=dict(offset_mm=round(1e3 * off, 1), angle_deg=round(ang, 2)))
    terms = dict(named_points=max(nerr.values()) <= th["named_mm"],
                 stacking_order=s_all >= th["stack_all"] and s_cov >= th["stack_covered"],
                 outline=iou >= th["iou"],
                 flat=1e3 * dz <= th["dz_mm"] and 1e3 * over <= th["over_mm"],
                 placement=1e3 * off <= th["offset_mm"] and abs(ang) <= th["angle_deg"])
    m.update(terms=terms, ok=all(terms.values()), thresholds=th)
    return m
