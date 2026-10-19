#!/usr/bin/env python3
"""Re-run a stress breakdown reproducer (ikea_stress.py shrink / episode rows file) with the stress observer, and attribute
contacts on the cylinder geoms the prism rule converts.

usage: python repro.py --task T --scene-dir D --rows X.npz --ckpt CKDIR [--ref R.npz --birth] --out OUT.json [--geoms a,b]
  default : start from the stress run's own checkpoint state at the rows' ckpt_row (CKDIR: the reference checkpoints the
            reproducer was cut from, saved with the scene as it was), then the rows, observed. The same pre-breakdown
            state for the scene with and without the rule.
  --birth : as ikea_stress.py confirm: from the birth, the reference's first ckpt_row rows unobserved, then the rows.
  --geoms : the geom names to attribute (default: the scene's cylinder_prisms; give the same list to a base run).
Per tick it records the active contacts on those geoms (count, deepest penetration, summed normal force, the partner
bodies), and at the end the observer's verdict (ikea_stress.classify, criteria v2.13b).
Environment: RT_STRESS_DIR (the RoboTangle observer + verdict), as ikea_stress.py.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stress"))
import ikea_stress as IS  # noqa: E402  (puts shared/ and the observer on sys.path)
import mujoco  # noqa: E402
R = IS.R

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--scene-dir", required=True)
ap.add_argument("--rows", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ref")
ap.add_argument("--birth", action="store_true")
ap.add_argument("--geoms")
ap.add_argument("--out", required=True)
a = ap.parse_args()

with np.load(a.rows, allow_pickle=False) as z:
    rows = np.ascontiguousarray(z["ctrl"], np.float64)
    r0 = int(z["ckpt_row"]) if "ckpt_row" in z.files else 0
t0 = time.time()
sess = R.Session(a.task, a.scene_dir)
idx = IS.ckpt_index(a.ckpt)
c = next((c for c in idx["checkpoints"] if c["row"] == r0), None)
if a.birth:
    if r0:
        sess.run(IS.load_ref(a.ref)[:r0])
    start = "birth"
else:
    if r0:
        sess.set_state(IS.load_ckpt(a.ckpt, c))
    start = "ckpt"
start_sha = sess.state_sha()
start_matches_ckpt = (c["sha"] == start_sha) if (c is not None and r0) else None
sess.validity.reset()
sess.rows = []
m, d = sess.model, sess.data
prisms = list(getattr(sess.S, "cylinder_prisms", []) or [])
names = a.geoms.split(",") if a.geoms else [p["geom"] for p in prisms]
ids = np.array([m.geom(n).id for n in names], dtype=np.int64)
mark = np.zeros(m.ngeom, bool)
mark[ids] = True
trace = []                                    # (tick, n, depth_m, force_n, partners)


def record(tick):
    n = d.ncon
    if not n or not ids.size:
        return
    cc = d.contact
    g1, g2 = cc.geom1[:n], cc.geom2[:n]
    sel = (mark[g1] | mark[g2]) & (cc.efc_address[:n] >= 0)
    if not sel.any():
        return
    k = np.flatnonzero(sel)
    pairs = sorted({f"{m.geom(int(g1[i])).name}|{m.geom(int(g2[i])).name}" for i in k})
    trace.append((tick, int(sel.sum()), float(max(0., -cc.dist[k].min())),
                  float(np.abs(d.efc_force[cc.efc_address[k]]).sum()), pairs))


orig = IS.Observed.step


def step(self, row):
    orig(self, row)
    record(len(self.s.rows))


IS.Observed.step = step
rec = IS.run_rows(sess, rows, task=a.task, kind="fix2_repro",
                  extra=dict(id=Path(a.rows).stem, start=start, ckpt_row=r0, start_sha=start_sha,
                             start_matches_ckpt=start_matches_ckpt))
deep = max(trace, key=lambda t: t[2]) if trace else None
strong = max(trace, key=lambda t: t[3]) if trace else None
rec.update(traced_geoms=names, cylinder_prisms=prisms, traced_ticks=len(trace),
           traced_first_tick=trace[0][0] if trace else None,
           traced_deepest=dict(tick=deep[0], depth_mm=round(1000 * deep[2], 3), pairs=deep[4]) if deep else None,
           traced_strongest=dict(tick=strong[0], force_n=round(strong[3], 1), pairs=strong[4]) if strong else None,
           traced_pairs=sorted({p for t in trace for p in t[4]}), total_wall_s=round(time.time() - t0, 1))
np.savez_compressed(os.path.splitext(a.out)[0] + ".npz",
                    tick=np.array([t[0] for t in trace], np.int64), n=np.array([t[1] for t in trace], np.int64),
                    depth=np.array([t[2] for t in trace]), force=np.array([t[3] for t in trace]))
IS._write(a.out, rec)
v = rec.get("verdict") or {}
print(json.dumps(dict(id=rec["id"], start=start, start_ok=start_matches_ckpt, verdict=v.get("verdict"),
                      fails=v.get("fails"), sha=(rec.get("state_sha") or "")[:16], traced_deepest=rec["traced_deepest"],
                      traced_strongest=rec["traced_strongest"])), flush=True)
