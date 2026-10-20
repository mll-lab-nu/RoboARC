"""The assist's exposure rule on folded cloth (topogym/utils/grasp.py exposed_patch), 2026-10-11.

Synthetic sheets (material grid every 6 mm, 0.60 x 0.48 m) folded analytically with a layer spacing `gap` (VBD 7 mm,
the flex 3 mm; one layer 3 mm above the table, bends semicircles of radius gap / 2), patches as the assist builds them
(particles within r = 18 mm of the centre in material and in space), clearance 6 mm unless said:
  (a) a bend grip (centre on a half fold's crease apex) passes the VBD rule (remote_from_patch) and fails the flex rule;
  (b) bottom-layer centres under a half fold fail the VBD rule, 1.5-15 cm from the crease (at clearance 8 mm they would
      pass under either rule: the covering layer is 7 mm higher; the clearance stays 6 mm);
  (c) the bottom layer under a dog-ear's flap, a corner tucked under the sheet, and the middle and bottom layers of a
      letter fold fail the VBD rule; the top of each passes;
  (d) the flex rule is unchanged: remote_from_patch=False equals the previous code on every probe here and on random
      sheets.
  python tools/tests/test_exposure_rule.py"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))
from topogym.utils.grasp import exposed_patch  # noqa: E402

R, CLEAR, TABLE = .018, .006, .003


def reference(positions, material, indices, *, radius, clearance):
    """The previous exposed_patch (the flex rule), verbatim in behaviour."""
    x, uv, indices = np.asarray(positions), np.asarray(material), np.asarray(indices)
    for i in indices:
        local = np.linalg.norm(x[:, :2]-x[i, :2], axis=1) <= radius
        remote = np.linalg.norm(uv-uv[i], axis=1) > 2*radius
        if np.any(local & remote & (x[:, 2] > x[i, 2]+clearance)):
            return False
    return True


gu, gv = np.meshgrid(np.arange(-.30, .3001, .006), np.arange(-.24, .2401, .006))
UV = np.c_[gu.ravel(), gv.ravel()]


def half_fold(gap, crease=0.):
    X = np.c_[UV, np.full(len(UV), TABLE)]
    s = UV[:, 0] - crease
    m = s > 0
    r = gap / 2
    th = np.minimum(s[m] / r, math.pi)
    flat = np.maximum(s[m] - math.pi * r, 0.)
    X[m, 0] = crease + r * np.sin(th) - flat
    X[m, 2] = TABLE + r - r * np.cos(th)
    return X


def letter_fold(gap):
    """Far third (u > 0.1) over the middle, then the near third (u < -0.1) over both (three layers)."""
    X = np.c_[UV, np.full(len(UV), TABLE)]
    r = gap / 2
    s = UV[:, 0] - .1
    m = s > 0
    th = np.minimum(s[m] / r, math.pi)
    flat = np.maximum(s[m] - math.pi * r, 0.)
    X[m, 0] = .1 + r * np.sin(th) - flat
    X[m, 2] = TABLE + r - r * np.cos(th)
    s2 = -.1 - UV[:, 0]
    n = s2 > 0
    R2 = gap                     # the second bend wraps two layers
    th = np.minimum(s2[n] / R2, math.pi)
    flat = np.maximum(s2[n] - math.pi * R2, 0.)
    X[n, 0] = -.1 - R2 * np.sin(th) + flat
    X[n, 2] = TABLE + R2 - R2 * np.cos(th)
    return X


def dog_ear(gap, size=.06):
    """The corner (.3, .24) folded over the diagonal crease u + v = .54 - size (a flap of legs `size`)."""
    X = np.c_[UV, np.full(len(UV), TABLE)]
    d = np.array([1., 1.]) / math.sqrt(2)
    c0 = np.array([.3 - size, .24])
    s = (UV - c0) @ d
    m = s > 0
    r = gap / 2
    th = np.minimum(s[m] / r, math.pi)
    flat = np.maximum(s[m] - math.pi * r, 0.)
    base = UV[m] - np.outer(s[m], d)
    X[m, :2] = base + np.outer(r * np.sin(th) - flat, d)
    X[m, 2] = TABLE + r - r * np.cos(th)
    return X


def tucked_corner(gap, size=.06):
    """The corner (.3, .24) folded UNDER the sheet along the same diagonal: the flap on the table, the sheet over it."""
    X = dog_ear(gap, size)
    d = np.array([1., 1.]) / math.sqrt(2)
    c0 = np.array([.3 - size, .24])
    s = (UV - c0) @ d
    flap = s > 0
    out = np.c_[UV, np.full(len(UV), TABLE)]
    out[flap, :2] = X[flap, :2]
    out[flap, 2] = TABLE + gap - X[flap, 2] + TABLE          # mirrored in height: the flap below
    reach = size / math.sqrt(2)                  # the flap's extent beyond the crease
    over = (~flap) & (s > -reach - .01)
    out[over, 2] = TABLE + gap                   # the sheet lies one layer up over the tucked flap
    return out


def patch_of(X, c):
    return np.flatnonzero((np.linalg.norm(UV - UV[c], axis=1) <= R) & (np.linalg.norm(X - X[c], axis=1) <= R))


def at(u, v):
    return int(np.argmin(np.sum((UV - np.array([u, v])) ** 2, axis=1)))


def rules(X, c, clearance=CLEAR):
    p = patch_of(X, c)
    new = exposed_patch(X, UV, p, radius=R, clearance=clearance, remote_from_patch=True)
    old = exposed_patch(X, UV, p, radius=R, clearance=clearance)
    ref = reference(X, UV, p, radius=R, clearance=clearance)
    assert old == ref, ("flex rule changed", c)
    return new, old


fails = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


GAP = .007
H = half_fold(GAP)
apex = at(math.pi * GAP / 4, 0.)
new, old = rules(H, apex)
check(f"(a) bend grip at the apex passes the VBD rule (new {new}, flex rule {old})", new and not old)
for du in (.003, .006, -.006):
    c = at(math.pi * GAP / 4 + du, 0.)
    check(f"(a) bend grip {du * 1e3:+.0f} mm off the apex passes the VBD rule", rules(H, c)[0])
for u in (-.015, -.03, -.05, -.08, -.15):
    c = at(u, 0.)
    new6, old6 = rules(H, c)
    new8, _ = rules(H, c, .008)
    check(f"(b) bottom layer u {u:+.3f} under the half fold fails the VBD rule at 6 mm", not new6 and not old6)
    print(f"     (at 8 mm clearance the VBD rule says {'exposed' if new8 else 'occluded'})")
for u in (.03, .08):
    check(f"top layer of the half fold (material u {u:+.2f}) passes", rules(H, at(u, 0.))[0])
D = dog_ear(GAP, .10)              # the corner (.30, .24) lands at (.20, .14)
check("(c) the sheet under the dog-ear's tip fails", not rules(D, at(.205, .145))[0])
check("(c) the dog-ear's tip (the turned corner, on top) passes", rules(D, at(.288, .228))[0])
T = tucked_corner(GAP, .10)
check("(c) the corner tucked under the sheet fails", not rules(T, at(.288, .228))[0])
check("(c) the sheet over the tucked corner passes", rules(T, at(.205, .145))[0])
L = letter_fold(GAP)
check("(c) letter fold: the bottom (middle third) fails", not rules(L, at(0., 0.))[0])
check("(c) letter fold: the middle layer (far third) fails", not rules(L, at(.2, 0.))[0])
check("(c) letter fold: the top (near third) passes", rules(L, at(-.2, 0.))[0])
# (d) the flex rule = the previous code: the same probes on 3 mm layers, and random sheets
for X in (half_fold(.003), dog_ear(.003), tucked_corner(.003), letter_fold(.003), H, D, T, L):
    for c in range(0, len(UV), 97):
        rules(X, c)
rng = np.random.default_rng(0)
for _ in range(60):
    X = np.c_[UV + rng.normal(0, .01, UV.shape), TABLE + np.abs(rng.normal(0, .01, len(UV)))]
    c = int(rng.integers(len(UV)))
    rules(X, c, float(rng.uniform(.002, .01)))
check("(d) the flex rule (remote_from_patch=False) equals the previous code on every probe", True)
print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
