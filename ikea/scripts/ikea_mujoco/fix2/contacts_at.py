#!/usr/bin/env python3
"""The contacts between two bodies at one MuJoCo substep of a stress reproducer (from the stress run's checkpoint state),
to attribute an observer finding (a body pair at tick T, substep S) to geoms.

usage: python contacts_at.py --task T --scene-dir D --rows X.npz --ckpt CKDIR --substep S --bodies A,B [--window 4]
  --substep : the observer's global substep (tick t covers substeps 4(t-1)+1 .. 4t)
  --window  : also print the substeps this many before S
Prints, per substep, every contact between the two bodies' trees: geom names and types, penetration (mm), normal force.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stress"))
import ikea_stress as IS  # noqa: E402
import mujoco  # noqa: E402
R = IS.R

ap = argparse.ArgumentParser()
for k in ("--task", "--scene-dir", "--rows", "--ckpt", "--bodies"):
    ap.add_argument(k, required=True)
ap.add_argument("--substep", type=int, required=True)
ap.add_argument("--window", type=int, default=4)
a = ap.parse_args()
with np.load(a.rows, allow_pickle=False) as z:
    rows = np.ascontiguousarray(z["ctrl"], np.float64)
    r0 = int(z["ckpt_row"]) if "ckpt_row" in z.files else 0
s = R.Session(a.task, a.scene_dir)
if r0:
    s.set_state(IS.load_ckpt(a.ckpt, next(c for c in IS.ckpt_index(a.ckpt)["checkpoints"] if c["row"] == r0)))
m, d = s.model, s.data
want = []
for name in a.bodies.split(","):
    b = 0 if name == "world" else int(m.body_rootid[m.body(name).id])
    want.append(b)
root = lambda g: int(m.body_rootid[m.geom_bodyid[g]]) if m.geom_bodyid[g] else 0
prisms = {p["geom"] for p in (getattr(s.S, "cylinder_prisms", []) or [])}
gt = {int(v): k for k, v in mujoco.mjtGeom.__members__.items()}
sub, out = 0, []
f6 = np.zeros(6)
for row in rows:
    d.ctrl[:] = row
    for _ in range(R.SUBSTEPS):
        s.S.step(1)
        sub += 1
        if sub < a.substep - a.window:
            continue
        cons = []
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {root(c.geom1), root(c.geom2)}
            if pair != set(want) and not (len(set(want)) == 1 and pair == set(want)):
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            n1, n2 = m.geom(c.geom1).name, m.geom(c.geom2).name
            cons.append(dict(g1=n1, t1=gt.get(int(m.geom_type[c.geom1]), "?"), g2=n2,
                             t2=gt.get(int(m.geom_type[c.geom2]), "?"), depth_mm=round(-1000 * c.dist, 3),
                             fn=round(float(f6[0]), 2), prism=bool({n1, n2} & prisms)))
        cons.sort(key=lambda x: -x["depth_mm"])
        out.append(dict(substep=sub, tick=(sub - 1) // R.SUBSTEPS + 1, n=len(cons), contacts=cons[:12]))
        if sub >= a.substep:
            break
    if sub >= a.substep:
        break
for o in out:
    print(json.dumps(o))
