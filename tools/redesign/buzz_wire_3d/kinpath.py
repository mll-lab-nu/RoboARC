"""A tier's kinematic oracle path for previews (planning model): the held segments' planned joint paths (segplan JSON
outputs), forward kinematics -> TCP path, ring path, held flags; segments are joined by straight hand moves (the parks
and regrasps). usage: kinpath.py <tier> <out.npz> <seg0.json> [<seg1.json> ...] --frozen <dir> --assets <dir>"""
import argparse, json
import numpy as np
from topogym.tasks.buzz_wire_3d.redesign import tiers
from topogym.planning.buzz_wire_3d.redesign.scene import planning_model
from topogym.planning.buzz_wire_3d.redesign.kin import PandaKin, HOME
from topogym.planning.buzz_wire_3d.redesign.wandpath import WirePath

ap = argparse.ArgumentParser()
ap.add_argument("tier"); ap.add_argument("out"); ap.add_argument("segs", nargs="+")
ap.add_argument("--frozen", required=True); ap.add_argument("--assets", required=True)
a = ap.parse_args()
d = tiers.design(a.tier)
m = planning_model(a.frozen, a.assets, d, margin=0.0)
k = PandaKin(m)
path = WirePath(d.wire().rail + d.origin())
tcp, ring, held, seg_id = [], [], [], []
q_prev = HOME
for i, f in enumerate(a.segs):
    js = json.load(open(f))
    if not js.get("path"):
        raise SystemExit(f"{f}: no path")
    qs = [np.asarray(x[3]) for x in js["path"]]
    # transit: home / previous end -> this segment's start (not held), joint blend
    for t in np.linspace(0, 1, 30)[1:-1]:
        p, _ = k.fk(q_prev + (qs[0] - q_prev) * t)
        tcp.append(p); ring.append(path.p[path.idx(js["path"][0][0])]); held.append(False); seg_id.append(-1)
    for (s, psi, lag, _), q in zip(js["path"], qs):
        p, _ = k.fk(q)
        tcp.append(p); ring.append(path.p[path.idx(s)]); held.append(True); seg_id.append(i)
    q_prev = qs[-1]
np.savez_compressed(a.out, tcp_p=np.asarray(tcp), wand_p=np.asarray(ring), held=np.asarray(held), seg=np.asarray(seg_id))
print(json.dumps(dict(points=len(tcp), segments=len(a.segs),
                      tcp_path_m=round(float(np.sum(np.linalg.norm(np.diff(np.asarray(tcp), axis=0), axis=1))), 3))))
