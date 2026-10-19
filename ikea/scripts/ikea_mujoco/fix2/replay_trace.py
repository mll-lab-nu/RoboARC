#!/usr/bin/env python3
"""Replay one IKEA reference like the grader (rcb_ikea.Session: fresh build, rows open-loop, final_check()) and trace the
contacts of the cylinder geoms the prism rule converts, so a run with the rule and one without can be compared.

usage: python replay_trace.py --task T --scene-dir D --traj F --out OUT.json [--geoms a,b,...] [--every 1000]
  --geoms   geom names to trace (default: the scene's own cylinder_prisms; give the same list to a base run)
Writes OUT.json (verdict, hashes) and OUT.npz: per row the number of active contacts on the traced geoms, their deepest
penetration (m), their summed normal force (N) and, every --every rows, the integration-state sha256 (first 16 hex).
"""
import argparse, json, os, sys, time
import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import rcb_ikea  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--scene-dir", required=True)
ap.add_argument("--traj", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--geoms", default=None)
ap.add_argument("--every", type=int, default=1000)
a = ap.parse_args()

t0 = time.time()
s = rcb_ikea.Session(a.task, a.scene_dir)
m, d = s.model, s.data
prisms = list(getattr(s.S, "cylinder_prisms", []) or [])
names = a.geoms.split(",") if a.geoms else [p["geom"] for p in prisms]
ids = np.array([m.geom(n).id for n in names], dtype=np.int64)
mark = np.zeros(m.ngeom, bool)
mark[ids] = True
ctrl = rcb_ikea.load_trajectory(a.traj, s.nu, s.timestep)
N = len(ctrl)
ncon = np.zeros(N, np.int32)
depth = np.zeros(N)
force = np.zeros(N)
shas = []
init = s.state_sha()
for i, row in enumerate(ctrl):
    s.step(row)
    n = d.ncon
    if n and ids.size:
        c = d.contact
        sel = mark[c.geom1[:n]] | mark[c.geom2[:n]]
        sel &= c.efc_address[:n] >= 0
        if sel.any():
            ncon[i] = int(sel.sum())
            depth[i] = float(max(0.0, -c.dist[:n][sel].min()))
            force[i] = float(np.abs(d.efc_force[c.efc_address[:n][sel]]).sum())
    if (i + 1) % a.every == 0:
        shas.append((i + 1, s.state_sha()[:16]))
fc = s.final_check()
first = int(np.argmax(ncon > 0)) if (ncon > 0).any() else None
out = dict(task=a.task, scene_dir=a.scene_dir, traj=a.traj, rows=N, initial_state_sha=init, success=fc["success"],
           goal_end=fc["end"]["goal"], released_end=fc["end"]["released"], at_rest_end=fc["end"]["at_rest"],
           breakdown_end=fc["end"]["breakdown"], goal_settled=fc["settled"]["goal"],
           released_settled=fc["settled"]["released"], at_rest_settled=fc["settled"]["at_rest"],
           breakdown_settled=fc["settled"]["breakdown"], end_sha=fc["state_sha"], settled_sha=fc["settled_sha"],
           max_part_depth_mm=fc["settled"]["terminal"]["max_part_depth_mm"], report_end=fc["end"]["report"],
           traced_geoms=names, cylinder_prisms=prisms, first_traced_contact_row=first,
           traced_contact_rows=int((ncon > 0).sum()), traced_max_depth_mm=round(1000 * float(depth.max()), 4),
           traced_max_force_n=float(force.max()), wall_s=round(time.time() - t0, 1),
           mujoco=mujoco.__version__, numpy=np.__version__)
np.savez_compressed(os.path.splitext(a.out)[0] + ".npz", ncon=ncon, depth=depth, force=force,
                    sha_rows=np.array([r for r, _ in shas]), sha=np.array([h for _, h in shas]))
tmp = a.out + ".tmp"
with open(tmp, "w") as f:
    json.dump(out, f, default=rcb_ikea._plain)
os.replace(tmp, a.out)
print(json.dumps({k: out[k] for k in ("task", "success", "end_sha", "settled_sha", "first_traced_contact_row",
                                      "traced_max_depth_mm", "wall_s")}), flush=True)
