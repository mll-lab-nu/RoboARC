"""The minimal oracle chain of a tier from its all-pairs analysis (segplan JSONs p<i>_<j>_<grasp>.json): the shortest
path start -> drop over feasible pairs (fewest held segments; ties: the shortest planned joint path), each segment's
grasp and planned path with forward kinematics (TCP, ring centre), the anti-hack facts (which pairs no grasp can do), and
the ORACLE_PLANS entry. usage: chain.py <tier> <analysis dir> <out.json> --frozen <dir> --assets <dir>"""
import argparse, glob, heapq, json, os, re
import numpy as np
from topogym.tasks.buzz_wire_3d.redesign import tiers
from topogym.planning.buzz_wire_3d.redesign.scene import planning_model
from topogym.planning.buzz_wire_3d.redesign.kin import PandaKin

ap = argparse.ArgumentParser()
ap.add_argument("tier"); ap.add_argument("dir"); ap.add_argument("out")
ap.add_argument("--frozen", required=True); ap.add_argument("--assets", required=True)
a = ap.parse_args()
d = tiers.design(a.tier)
res = {}
n = 0
for f in glob.glob(os.path.join(a.dir, "p*.json")):
    i, j = map(int, re.match(r"p(\d+)_(\d+)_", os.path.basename(f)).groups())
    js = json.load(open(f))
    n = max(n, j)
    res.setdefault((i, j), []).append(js)


def cost(js):
    q = np.array([x[3] for x in js["path"]])
    return float(np.abs(np.diff(q, axis=0)).sum())


best = {}
for (i, j), lst in res.items():
    ok = [x for x in lst if x["end_ok"] and x.get("path")]
    if ok:
        best[i, j] = min(ok, key=cost)
dist = {0: (0, 0.0)}; prev = {}; pq = [(0, 0.0, 0)]
while pq:
    k, c, u = heapq.heappop(pq)
    if (k, c) != dist.get(u):
        continue
    for v in range(u + 1, n + 1):
        if (u, v) in best:
            nk, nc = k + 1, c + cost(best[u, v])
            if (nk, nc) < dist.get(v, (1e9, 1e9)):
                dist[v] = (nk, nc); prev[v] = u; heapq.heappush(pq, (nk, nc, v))
out = dict(tier=a.tier, design=d.name, rests=n + 1, complete=n in dist)
centres = [0.5 * (x + y) for x, y in d.sleeves[:-1]]
rest_s = [0.0] + centres + [round(d.sleeves[-1][0] + 0.03, 4)]
graph = {}
for (i, j), js in best.items():
    oks = sorted([x for x in res[i, j] if x["end_ok"]], key=lambda x: (x["grasp"] != js["grasp"], x["grasp"]))
    for x in oks:
        graph.setdefault(str(i), []).append([j, x["grasp"]])
out["graph_plan"] = dict(pick="top:180:0.9:-10:-2", rests=rest_s, graph=graph)
infeasible = sorted([f"{i}->{j}" for (i, j) in res if (i, j) not in best])
out["infeasible_pairs"] = infeasible
out["pairs_tested"] = len(res)
out["grasps_tested"] = sorted({x["grasp"] for lst in res.values() for x in lst})
if n in dist:
    m = planning_model(a.frozen, a.assets, d, margin=0.0)
    k = PandaKin(m)
    path = [n]
    while path[-1] != 0:
        path.append(prev[path[-1]])
    path = path[::-1]
    segs = []
    for u, v in zip(path[:-1], path[1:]):
        js = best[u, v]
        tcp = [k.fk(np.asarray(x[3]))[0].round(4).tolist() for x in js["path"]]
        segs.append(dict(i=u, j=v, s0=js["sA"], s1=js["sB"], grasp=js["grasp"], nodes=len(js["path"]),
                         joint_path_rad=round(cost(js), 3), tcp=tcp))
    out["segments"] = segs
    out["held_segments"] = len(segs)
    out["parks"] = len(segs) - 1
    # every rest the chain skips must be skippable; every rest it uses is forced if no pair jumps over it
    used = path[1:-1]
    forced = []
    for r in used:
        jumps = [(i, j) for (i, j) in best if i < r < j]
        forced.append(dict(rest=r, forced=not jumps, skips=[f"{i}->{j}" for i, j in jumps]))
    out["rests_used"] = forced
    out["oracle_plan"] = dict(pick=segs[0]["grasp"] + ("" if segs[0]["grasp"].count(":") >= 4 else ""),
                              segments=[dict(grasp=sg["grasp"], s1=sg["s1"], end="drop" if sg["j"] == n else "park")
                                        for sg in segs])
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k_: v for k_, v in out.items() if k_ not in ("segments",)}, indent=1))
