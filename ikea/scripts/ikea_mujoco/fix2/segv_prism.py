#!/usr/bin/env python3
"""The stress agent's standalone eket SIGSEGV pairs (scripts/ikea_mujoco/stress/segv/*.xml: a dowel cylinder pressed into
a thin convex side-panel piece) with the dowel as the rule's 32-facet prism, on the stock 3.14.0 wheel.

usage: python segv_prism.py <xml> cyl|prism [jitter_seed]   -> mj_forward once; prints ncon (a crash is rc -11)
  jitter_seed > 0: body B's position moved by up to 0.3 mm and its orientation by up to 1 deg (seeded), to sample the
  neighbourhood of the crashing pose.
"""
import sys
import numpy as np
import mujoco

xml, kind = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
spec = mujoco.MjSpec.from_file(xml)
if seed:
    rng = np.random.default_rng(seed)
    b = spec.body("B")
    b.pos = np.asarray(b.pos) + rng.uniform(-3e-4, 3e-4, 3)
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    ang = np.radians(rng.uniform(-1, 1))
    dq = np.r_[np.cos(ang / 2), np.sin(ang / 2) * ax]
    q = np.zeros(4); mujoco.mju_mulQuat(q, dq, np.asarray(b.quat, float)); b.quat = q
if kind == "prism":
    g = next(g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER)
    r, h, n = float(g.size[0]), float(g.size[1]), 32          # ikea_runtime.cylinder_prisms' construction
    ang = 2 * np.pi * np.arange(n) / n
    v = np.array([[r * np.cos(t), r * np.sin(t), z] for z in (-h, h) for t in ang])
    f = [[k, (k + 1) % n, n + k] for k in range(n)] + [[(k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)]
    f += [[0, k + 1, k] for k in range(1, n - 1)] + [[n, n + k, n + k + 1] for k in range(1, n - 1)]
    mesh = spec.add_mesh(); mesh.name = "prism"; mesh.uservert = v.reshape(-1); mesh.userface = np.asarray(f).reshape(-1)
    g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = "prism"; g.size = np.zeros(3)
m = spec.compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
print(kind, seed, "ncon", d.ncon, "maxdepth_mm", round(-1000 * min([c.dist for c in d.contact[:d.ncon]] or [0]), 3))
