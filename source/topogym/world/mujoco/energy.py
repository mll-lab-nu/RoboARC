"""Elastic potential energy of MuJoCo 2D flexes, which mujoco 3.14's own energy (mjENBL_ENERGY, mj_energyPos) leaves out.

MuJoCo books gravity and joint / tendon springs in data.energy[0], and the kinetic energy of every dof (flex vertices
included) in data.energy[1]; a deformed 2D flex adds nothing to data.energy[0] although its elastic forces
(qfrc_spring) store energy. flex_energy computes that potential from the model's own precomputed elasticity data, so an
energy balance (the stress observer's E = kinetic + potential) can book it:

  membrane (elastic2d "stretch" / "both", Saint Venant-Kirchhoff on squared edge lengths): per element, with its three
  edges' e_i = l_i^2 - l0_i^2 and the symmetric 3x3 K from flex_stiffness (21 numbers per element, the upper triangle
  first, in the order 00 01 02 11 12 22):  E_m = 1/4 sum_elements e^T K e;
  bending (elastic2d "bend" / "both", the constant-Hessian discrete shell): per interior edge, over its four vertices
  (the edge's two, then the two flaps of flex_edgeflap) and the 4x4 B of flex_bending (17 numbers per edge, the first 16
  row-major):  E_b = 1/2 sum_edges sum_ab B_ab x_a . x_b (computed on positions relative to the edge's first vertex,
  the same value since B's rows sum to zero).

Both are exact against the path-integrated work of MuJoCo's elastic forces (scripts/robotangle/test_flex_energy.py:
membrane to ~3e-7 relative, the trapezoid's own error; bending to ~1e-13). A flat rest configuration is assumed; a flex
with a curved rest (flex_bending's 17th number, its rest-curvature term) is refused, and so is a 1D or 3D flex with
elasticity (not booked here; one without contributes nothing).
"""
from __future__ import annotations

import numpy as np

_UPPER3 = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def flex_energy(model, data) -> tuple[float, float]:
    """(membrane, bending) elastic potential [J] of every 2D flex at data's flexvert_xpos (call after mj_flex /
    mj_fwdPosition / mj_forward, or after a step: MuJoCo refreshes flexvert_xpos in the position stage)."""
    em = eb = 0.
    if not model.nflex:
        return em, eb
    xall = np.asarray(data.flexvert_xpos)
    stiff = np.asarray(model.flex_stiffness).reshape(-1, 21)
    for f in range(int(model.nflex)):
        vadr, vnum = int(model.flex_vertadr[f]), int(model.flex_vertnum[f])
        eadr, enum = int(model.flex_edgeadr[f]), int(model.flex_edgenum[f])
        eladr, elnum = int(model.flex_elemadr[f]), int(model.flex_elemnum[f])
        if int(model.flex_dim[f]) != 2:
            if np.any(stiff[eladr:eladr + elnum]) or np.any(np.asarray(model.flex_bending).reshape(-1, 17)[eadr:eadr + enum]):
                raise NotImplementedError(f"flex_energy: the elasticity of a {int(model.flex_dim[f])}D flex is not booked")
            continue
        x = xall[vadr:vadr + vnum]
        edges = np.asarray(model.flex_edge[eadr:eadr + enum]).reshape(-1, 2)
        # membrane
        k = stiff[eladr:eladr + elnum, :6]
        if np.any(k):
            l0 = np.asarray(model.flexedge_length0[eadr:eadr + enum])
            e = np.sum((x[edges[:, 1]] - x[edges[:, 0]]) ** 2, axis=1) - l0 ** 2
            ee = np.asarray(model.flex_elemedge[int(model.flex_elemedgeadr[f]):int(model.flex_elemedgeadr[f]) + 3 * elnum])
            s = e[ee.reshape(-1, 3)]
            K = np.zeros((elnum, 3, 3))
            for c, (i, j) in enumerate(_UPPER3):
                K[:, i, j] = K[:, j, i] = k[:, c]
            em += .25 * float(np.einsum("ni,nij,nj->", s, K, s))
        # bending
        bend = np.asarray(model.flex_bending).reshape(-1, 17)[eadr:eadr + enum]
        if np.any(bend):
            if np.any(bend[:, 16]):
                raise NotImplementedError("flex_energy: a curved bending rest configuration is not booked")
            flap = np.asarray(model.flex_edgeflap[eadr:eadr + enum]).reshape(-1, 2)
            inner = np.flatnonzero((flap[:, 0] >= 0) & (flap[:, 1] >= 0))
            v = np.c_[edges[inner], flap[inner]]
            X = x[v] - x[v[:, :1]]                          # (n, 4, 3), relative to the edge's first vertex: B's rows
            gram = np.einsum("nai,nbi->nab", X, X)          # sum to zero, so this is exact and avoids cancellation
            eb += .5 * float(np.einsum("nab,nab->", bend[inner, :16].reshape(-1, 4, 4), gram))
    return em, eb
