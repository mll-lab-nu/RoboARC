"""Base search, stage A (planning model only): for candidate table-top Panda mounts (x, y, yaw_deg), check that the
tier's fixed grasp chain stays feasible and score the deck pick and the finish grasps.

Per mount, in order (a failed check ends that mount):
  1. legal: link0 footprint inside the table (edge margin) and DECK_CLEARANCE from the deck (design.mount_problems);
  2. reset: the home configuration collision-free with margin >= 5 mm (robot to wire, deck, posts);
  3. chain: every held segment of the chain (chain.py JSON) explored with its grasp options (the chain's grasp first,
     then the other grasps the default-mount analysis found for that pair), as segplan.py does (psi 24, ds 3 mm);
  4. pick: deck-pick grasp configurations at the start pose (collision-checked, checked approach line);
  5. finish: the flipped wand resting at the finish pose turned about the finish post by each of OFFSETS: topb grasp
     configurations (count, best robot margin); for |offset| <= 40 deg also a grasp-turn-lower plan onto the target.
Writes one JSON line per mount to --out (appending). usage:
  basescreen.py <tier> --chain <chain.json> --pairs <analysis dir> --frozen <dir> --assets <dir> --out <jsonl>
                --mounts x,y,yaw[;x,y,yaw...]"""
import faulthandler
faulthandler.enable()
import argparse, glob, json, os, re, sys, time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from topogym.tasks.buzz_wire_3d.redesign import tiers
from topogym.tasks.buzz_wire_3d.redesign.design import mount_problems
from topogym.planning.buzz_wire_3d.redesign.scene import planning_model
from topogym.planning.buzz_wire_3d.redesign.kin import HOME
from topogym.planning.buzz_wire_3d.redesign.wandpath import parse_grasp
from topogym.planning.buzz_wire_3d.redesign.feasibility import explore, extract
from topogym.planning.buzz_wire_3d.redesign.oracle import TierOracle, gap_for
from topogym.assets.geometry.buzz_wire_3d import endpoint_pose

OFFSETS = (0, 20, -20, 40, -40, 60, -60, 80, -80, 100, -100, 120, -120, 140, -140, 160, -160, 180)
PICK_SPECS = ("top:180:0.9:-10:2", "top:0:0.9:10:2", "top:180:0.9:0:2", "top:0:0.9:0:2", "top:180:0.8:-20:2",
              "top:0:0.8:20:2", "top:180:0.7:-10:2", "top:0:0.7:10:2")
FINISH_SPECS = ("topb:180:0.9:-10:2", "topb:0:0.9:10:2", "topb:180:0.9:0:2", "topb:0:0.9:0:2", "topb:180:0.8:10:1",
                "topb:0:0.8:-10:1", "topb:180:0.85:-20:2", "topb:0:0.85:20:2")


class StubIO:
    def __init__(self, model):
        self.model = model

    def q(self):
        return HOME.copy()


def pair_grasps(pairs_dir, i, j):
    out = []
    for f in sorted(glob.glob(os.path.join(pairs_dir, f"p{i}_{j}_*.json"))):
        js = json.load(open(f))
        if js.get("end_ok"):
            out.append(js["grasp"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tier"); ap.add_argument("--chain", required=True); ap.add_argument("--pairs", required=True)
    ap.add_argument("--frozen", required=True); ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--mounts", required=True)
    ap.add_argument("--max-options", type=int, default=4)
    ap.add_argument("--no-chain", action="store_true", help="skip step 3 (pick / finish only)")
    a = ap.parse_args()
    d = tiers.design(a.tier)
    chain = json.load(open(a.chain))
    segs = chain["segments"]
    pick = chain.get("graph_plan", {}).get("pick") or segs[0]["grasp"]
    w = d.wire(); o = d.origin()
    for ms in a.mounts.split(";"):
        mount = tuple(float(v) for v in ms.split(","))
        t0 = time.time()
        rec = dict(tier=a.tier, design=d.name, mount=mount)
        bad = mount_problems(mount, d.center_xy_m, d.base_size_m)
        if bad:
            rec.update(stage="legal", ok=False, why=bad)
            emit(a.out, rec, t0); continue
        m = planning_model(a.frozen, a.assets, d, margin=0.006, mount=mount)
        orc = TierOracle(StubIO(m), d, [], log=lambda **kv: None)
        orc.seeds = orc.seeds[:12]          # grasp checks: home + 11 random seeds (the oracle uses 25)
        kin = orc.kin
        kin.set_gap(0.085)
        p0, r0 = endpoint_pose(w, 0)
        home_margin = orc.robot_margin(HOME, wand=(p0 + o, r0), deck_ok=False)
        rec["home_margin_mm"] = round(1e3 * home_margin, 1)
        if home_margin < 0.005:
            rec.update(stage="reset", ok=False)
            emit(a.out, rec, t0); continue
        # 3. the chain
        path = orc.path
        psi_n = 24
        rng = np.random.default_rng(1)
        seeds = [HOME] + [rng.uniform(kin.lo, kin.hi) for _ in range(24)]

        def psi_index(s, direction):
            return int(np.round(path.psi_toward(s, direction) / (2 * np.pi / psi_n))) % psi_n
        seg_out = []
        chain_ok = True
        if not a.no_chain:
            for k, sg in enumerate(segs):
                i, j, sA, sB = sg["i"], sg["j"], sg["s0"], sg["s1"]
                last = k == len(segs) - 1
                opts = [sg["grasp"]] + [g for g in pair_grasps(a.pairs, i, j) if g != sg["grasp"]]
                opts = opts[:a.max_options]
                jA = psi_index(sA, r0.apply([1, 0, 0]) if sA < 1e-9 else [0, 0, -1])
                jB = psi_index(sB, [0, 0, -1])
                found = None; reached = 0.0
                for spec in opts:
                    kind, roll = spec.split(":")[0], float(spec.split(":")[1])
                    kin.set_gap(0.0070 if (kind.startswith("side") or (kind == "end" and int(round(roll)) % 180 == 90))
                                else 0.0098)
                    g = parse_grasp(spec, d.layout())
                    r = explore(kin, path, g, s0=sA, s1=sB, ds=0.003, psi_n=psi_n, seeds=seeds, start_psis=[jA],
                                lags=np.deg2rad([0.0]), start_lags=[0], keep=True, clear_robot=0.002, per_bin=4,
                                dq_max=0.35)
                    reached = max(reached, float(r["reached_s"] or 0.0))
                    if r["complete"]:
                        pth = extract(r, end_psi=None if last else jB, end_lag=None if last else 0, psi_tol=6)
                        if pth is not None:
                            found = spec; break
                drop = None
                if last and found is not None:
                    # the oracle's drop: the handle at the finish heading at release (psi tolerance in 15-degree bins,
                    # smallest that plans), at the drop point and carried down the post (as far as it goes)
                    fh = endpoint_pose(w, 1)[1].apply([1, 0, 0])
                    drop = {}
                    for s_end in (path.length - 0.05, path.length - 0.09, path.length - 0.13, sB):
                        if s_end < sB:
                            continue
                        r2 = r if s_end == sB else explore(
                            kin, path, g, s0=sA, s1=s_end, ds=0.003, psi_n=psi_n, seeds=seeds, start_psis=[jA],
                            lags=np.deg2rad([0.0]), start_lags=[0], keep=True, clear_robot=0.002, per_bin=4,
                            dq_max=0.35)
                        if not r2["complete"]:
                            continue
                        jF = psi_index(s_end, fh)
                        tol = next((t for t in (0, 1, 2, 3, 4, 6, 8, 12)
                                    if extract(r2, end_psi=jF, end_lag=0, psi_tol=t) is not None), None)
                        drop = dict(s_end=round(s_end, 4), below_drop_m=round(s_end - sB, 3), heading_tol_bins=tol)
                        break
                seg_out.append(dict(i=i, j=j, grasp=found, reached=round(reached, 4), s1=round(sB, 4), drop=drop))
                if found is None:
                    chain_ok = False; break
        rec["segments"] = seg_out
        if not chain_ok:
            rec.update(stage="chain", ok=False)
            emit(a.out, rec, t0); continue
        # 3b. the regrasps at the chain's rests: the oracle's variants (regrasp()) at the ideal hanging pose
        rg = []
        for k, sg in enumerate(seg_out[1:] if not a.no_chain else [], start=1):
            f = sg["grasp"].split(":")
            other = "0" if f[1] == "180" else "180"
            variants = list(dict.fromkeys([sg["grasp"]] + [":".join([f[0], r, fr] + f[3:]) for r in (f[1], other)
                                                           for fr in ("0.55", "0.7", "0.8")]))
            pw, Rw = orc.hang_pose(segs[k]["s0"])
            ok = [v for v in variants if orc.grasp_candidates(v, pw, Rw, n_keep=1)]
            rg.append(dict(rest=sg["i"], n=len(ok), specs=ok))
        rec["regrasp"] = rg
        dr = seg_out[-1].get("drop") if seg_out else None
        rec["drop_heading_tol_bins"] = None if not dr else dr["heading_tol_bins"]
        rec["drop_below_m"] = None if not dr else dr["below_drop_m"]
        if any(r["n"] == 0 for r in rg):
            rec.update(stage="regrasp", ok=False)
            emit(a.out, rec, t0); continue
        # 4. the deck pick at the start pose
        sp = p0 + o
        specs = [pick.rsplit(":", 1)[0] + ":2" if pick.count(":") == 4 else pick]
        specs += [x for x in PICK_SPECS if x not in specs]
        per = {}
        for spec in specs:
            c = orc.grasp_candidates(spec, sp, r0, deck_ok=True, n_keep=2)
            if c:
                per[spec] = round(1e3 * max(x[0] for x in c), 1)
        cands = per
        rec["pick"] = dict(spec=pick, n=len(per), specs=per,
                           margin_mm=max(per.values(), default=-1000.0))
        # 5. the finish: the flipped wand about the finish post
        tp, tr = endpoint_pose(w, 1)
        target_p = tp + o
        flip = tr * R.from_euler("x", np.pi)
        post = target_p.copy()
        fin = []
        for off in OFFSETS:
            Rz = R.from_euler("z", np.deg2rad(off))
            pw = post + Rz.apply(target_p - post)
            Rw = Rz * flip
            n, best, plan = 0, -1.0, None
            for spec in FINISH_SPECS:     # stops at the first planned spec (|off| <= 40) or the second grasp
                c = orc.grasp_candidates(spec, pw, Rw, deck_ok=True, n_keep=2)
                if c:
                    n += 1; best = max(best, max(x[0] for x in c))
                    if plan is None and abs(off) <= 40:
                        if orc._rot_plan(spec, pw, Rw, post, -np.deg2rad(off), target_p, sweep=False) is not None:
                            plan = spec
                    if (plan is not None or abs(off) > 40) and n >= 1:
                        break
            fin.append(dict(off=off, n=n, margin_mm=round(1e3 * best, 1), plan=plan))
        rec["finish"] = fin
        rec["finish_score"] = sum(1 for f in fin if f["n"] > 0) + sum(1 for f in fin if f["plan"])
        rec["finish_cover_deg"] = 20 * sum(1 for f in fin if f["n"] > 0)
        rec["finish_margin_mm"] = fin[0]["margin_mm"]
        rec.update(stage="done", ok=True)
        emit(a.out, rec, t0)


def emit(out, rec, t0):
    rec["wall"] = round(time.time() - t0, 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("finish",)}), flush=True)


if __name__ == "__main__":
    main()
