"""Kinematic plan of one held segment of a redesign tier (planning model; a workstation or a cloud host): between two cut points
(start / rest-sleeve centres / end), for one grasp, find a continuous (psi, lag, joints) path that starts at the given
handle direction and ends hanging (handle down) at a sleeve, or at the finish heading.
usage: segplan.py <tier> <seg> <grasp spec> --frozen <frozen scene dir> --assets <asset root> --out <json>"""
import faulthandler
faulthandler.enable()
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from topogym.tasks.buzz_wire_3d.redesign import tiers
from topogym.planning.buzz_wire_3d.redesign.scene import planning_model
from topogym.planning.buzz_wire_3d.redesign.kin import PandaKin, HOME
from topogym.planning.buzz_wire_3d.redesign.wandpath import WirePath, parse_grasp
from topogym.planning.buzz_wire_3d.redesign.feasibility import explore, extract
from topogym.assets.geometry.buzz_wire_3d import endpoint_pose

ap = argparse.ArgumentParser()
ap.add_argument("tier"); ap.add_argument("seg", type=int); ap.add_argument("grasp")
ap.add_argument("--frozen", required=True); ap.add_argument("--assets", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--psi-n", type=int, default=24); ap.add_argument("--ds", type=float, default=0.003)
ap.add_argument("--dq-max", type=float, default=0.35)
ap.add_argument("--lags", default="0"); ap.add_argument("--clear", type=float, default=0.002)
ap.add_argument("--seeds", type=int, default=24); ap.add_argument("--per-bin", type=int, default=4)
ap.add_argument("--s0", type=float, default=None); ap.add_argument("--s1", type=float, default=None)
ap.add_argument("--free-start", action="store_true", help="any start psi (anti-hack: merged segments)")
ap.add_argument("--free-end", action="store_true")
ap.add_argument("--psi-tol", type=int, default=1, help="end psi tolerance in bins (hanging pose)")
ap.add_argument("--prior", type=float, default=0.0, help="null-space pull towards mid-range joints")
ap.add_argument("--mount", default=None, help="x,y,yaw_deg: the Panda's table-top mount (default: the design's)")
a = ap.parse_args()
d = tiers.design(a.tier)
w = d.wire(); o = d.origin(); L = d.layout()
m = planning_model(a.frozen, a.assets, d, margin=0.006,
                   mount=None if a.mount is None else tuple(float(v) for v in a.mount.split(",")))
k = PandaKin(m)
path = WirePath(w.rail + o)
psis = np.arange(a.psi_n) * 2 * np.pi / a.psi_n
lags = np.deg2rad([float(x) for x in a.lags.split(",")])
rng = np.random.default_rng(1)
seeds = [HOME] + [rng.uniform(k.lo, k.hi) for _ in range(a.seeds)]
cuts = [0.0] + [0.5 * (x + y) for x, y in d.sleeves] + [path.length]
sA = cuts[a.seg] if a.s0 is None else a.s0
sB = cuts[a.seg + 1] if a.s1 is None else a.s1
p0, r0 = endpoint_pose(w, 0); p1, r1 = endpoint_pose(w, 1)
def psi_index(s, direction):
    return int(np.round(path.psi_toward(s, direction) / (2 * np.pi / a.psi_n))) % a.psi_n
first, last = abs(sA) < 1e-9, abs(sB - path.length) < 1e-9
jA = psi_index(sA, r0.apply([1, 0, 0]) if first else [0, 0, -1])
jB = psi_index(sB, r1.apply([1, 0, 0]) if last else [0, 0, -1])
kind, roll = a.grasp.split(":")[0], float(a.grasp.split(":")[1])
k.set_gap(0.0070 if (kind.startswith("side") or (kind == "end" and int(round(roll)) % 180 == 90)) else 0.0098)
g = parse_grasp(a.grasp, L)
t = time.time()
r = explore(k, path, g, s0=sA, s1=sB, ds=a.ds, psi_n=a.psi_n, seeds=seeds, start_psis=None if a.free_start else [jA],
            lags=lags, start_lags=[int(np.argmin(np.abs(lags)))], keep=True, clear_robot=a.clear, per_bin=a.per_bin,
            dq_max=a.dq_max,
            prior_w=a.prior)
pth = None
if r["complete"]:
    pth = extract(r, end_psi=None if a.free_end else jB, end_lag=None if a.free_end else int(np.argmin(np.abs(lags))),
                  psi_tol=a.psi_tol)
res = dict(tier=a.tier, seg=a.seg, grasp=a.grasp, sA=sA, sB=sB, jA=jA, jB=jB, reached=r["reached_s"],
           complete=r["complete"], end_ok=pth is not None, wall=round(time.time() - t, 1),
           feas=r["feas"].tolist(), s=r["s"].tolist(), free_start=a.free_start, free_end=a.free_end, mount=a.mount,
           path=None if pth is None else [(s, p, l, q.tolist()) for s, p, l, q in pth])
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(a.out, "w"))
print(json.dumps({k_: v for k_, v in res.items() if k_ not in ("feas", "s", "path")}), flush=True)
