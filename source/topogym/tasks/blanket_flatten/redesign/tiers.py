"""The redesigned blanket_flatten tiers (v0): scene options, the inverse-generation fold program applied to the flat
sheet (start state).

Material coordinates (u, v): u along the 0.60 m length (+u = +x, toward the right arm), v along the 0.48 m width (+v =
+y, away from the arms' base line); the sheet's centre lies at world (-0.1357, 0.15). Corners: NL (-.3, -.24), NR (.3,
-.24), FL (-.3, .24), FR (.3, .24). """
from __future__ import annotations

import numpy as np


CX, CY = -.1356786, .05
CENTRE_SHIFT = (0., -.10)       # FlexSim: the instance scene's sheet lies at y 0.15
GRIP_IN = .012           # grasp points one grid spacing inside the edges (a complete patch)


def W(u, v):
    return (CX + u, CY + v)


def corner(cu, cv, inset=GRIP_IN):
    return (cu - np.sign(cu) * inset, cv - np.sign(cv) * inset)



# ----------------------------------------------------------------------------------------------- tier: very_easy
def _mat_side(a, b, side):
    a, b, side = (np.asarray(x, float) for x in (a, b, side))
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]])
    if (side - a) @ n < 0:
        n = -n
    return lambda uv: (uv - a) @ n >= -1e-9


def fold(name, a_uv, b_uv, side_uv, *, material=None, duration_s=3., settle_s=1.5, d0=.015):
    f = dict(a=W(*a_uv), b=W(*b_uv), side=W(*side_uv), duration_s=duration_s, settle_s=settle_s, d0=d0)
    if material is not None:
        f.update(select="material", material_region=material)
    return dict(name=name, fold=f)


def fold_world(name, a_xy, b_xy, side_xy, *, material=None, duration_s=3., settle_s=1.5, d0=.015):
    f = dict(a=a_xy, b=b_xy, side=side_xy, duration_s=duration_s, settle_s=settle_s, d0=d0)
    if material is not None:
        f.update(select="material", material_region=material)
    return dict(name=name, fold=f)


FENCE = dict(name="fence", centre_m=(CX, CY + .24 + .012 + .02), half_m=(.24, .02, .03))
