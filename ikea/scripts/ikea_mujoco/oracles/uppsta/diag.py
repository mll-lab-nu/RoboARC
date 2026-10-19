#!/usr/bin/env python3
"""Replay rows with a phase log (study tool): every --every rows one JSON line with the scene's success report reduced to
numbers (screws, end blocks, dowels, pegs, board tilt), the mechanism events, robot-part contacts and the clamp count;
saved states (Session.get_state, rows dropped) every --save rows. --noclamp switches the joint-speed clamp off (a study
of the demo as sent, not a rule).
  python diag.py --traj X.npz --out DIR [--noclamp] [--every 4000] [--save 40000] [--start-state P --start-row K]"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def summary(s):
    rep = s.S.success()
    a = rep["assembled"]
    out = dict(goal=bool(rep["success"]), tilt=a["on_its_feet_tilt_deg"])
    for k in ("screw_a", "screw_b"):
        out[k] = [a[k]["welded"], a[k]["axial_mm"], a[k]["lateral_mm"], a[k]["tilt_deg"]]
    for k in ("end_block_a", "end_block_b"):
        out[k] = [a[k]["pos_mm"], a[k]["rot_deg"]]
    out["dowels"] = {v["part"]: [v["pos_mm"], v["axis_deg"]] for v in a["dowels"].values()}
    out["pegs"] = {v["part"]: [v["lateral_mm"], v["above_up_mm"], v["axis_deg"]] for v in a["pegs"].values()}
    out["contacts"] = [list(x) for x in s.validity.robot_part_contacts()]
    out["depth_mm"] = round(1000 * s.validity.max_part_depth_m, 3)
    out["breakdown"] = s.validity.breakdown
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--noclamp", action="store_true"); ap.add_argument("--every", type=int, default=4000)
    ap.add_argument("--save", type=int, default=40000); ap.add_argument("--rcb-ikea", default=os.path.join(HERE, "..", "..", "shared"))
    ap.add_argument("--start-state"); ap.add_argument("--start-row", type=int, default=0)
    ap.add_argument("--stop-row", type=int, default=0)
    ap.add_argument("--noprism", action="store_true", help="a study: the scene without the cylinder->prism rule")
    ap.add_argument("--scene-dir", default=None)
    ap.add_argument("--track", type=int, default=0, help="every N rows: robot qpos and every free body's pose -> track.npz")
    ap.add_argument("--contacts", action="store_true", help="track: also per free body a contact mask (1 left hand, 2 "
                    "right hand, 4 anything else that is not the robot)")
    ap.add_argument("--flush", type=int, default=0, help="also write the track so far every N rows (a guide to follow)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.rcb_ikea))
    import rcb_ikea as R
    if a.noprism:
        sd = a.scene_dir or os.path.join(R.SCENES_ROOT, "uppsta")
        sys.path.insert(0, sd)
        import ikea_runtime
        ikea_runtime.cylinder_prisms = lambda spec: []
    os.makedirs(a.out, exist_ok=True)
    s = R.Session("uppsta", a.scene_dir)
    if a.noclamp:
        s._limited[:] = False
    c = np.load(a.traj)["ctrl"]
    r0 = a.start_row
    if a.start_state:
        with open(a.start_state, "rb") as f:
            st = pickle.load(f)
        st["rows"] = []
        s.set_state(st)
    stop = a.stop_row or len(c)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time(); nev = len(s.S.mech.events)
    m = s.model
    free = [b for b in range(m.nbody) if m.body_jntnum[b] and m.jnt_type[m.body_jntadr[b]] == 0]
    arm = [m.jnt_qposadr[j] for j in range(m.njnt) if m.body(m.jnt_bodyid[j]).name.startswith(("left/", "right/"))]
    tr_rows, tr_q, tr_p, tr_ctrl, tr_c = [], [], [], [], []
    root_of = np.asarray(m.body_rootid)
    free_idx = {b: k for k, b in enumerate(free)}
    owner = np.full(m.nbody, -1)                       # body -> index of its free root body (ribs belong to their part)
    for b in range(m.nbody):
        r = b
        while r > 0 and r not in free_idx:
            r = int(m.body_parentid[r])
        owner[b] = free_idx.get(r, -1)
    side = np.zeros(m.nbody, int)
    for b in range(m.nbody):
        n = m.body(b).name
        side[b] = 1 if n.startswith("left/") else 2 if n.startswith("right/") else 0

    def contact_mask():
        out = np.zeros(len(free), np.uint8); d = s.data
        for k in range(int(d.ncon)):
            b1, b2 = m.geom_bodyid[d.contact.geom1[k]], m.geom_bodyid[d.contact.geom2[k]]
            for x, y in ((b1, b2), (b2, b1)):
                if owner[x] >= 0:
                    out[owner[x]] |= (1 if side[y] == 1 else 2 if side[y] == 2 else 4)
        return out

    def dump_track():
        tmp = os.path.join(a.out, "track.tmp.npz")
        np.savez(tmp, rows=np.array(tr_rows), arm_qpos=np.array(tr_q), applied=np.array(tr_ctrl),
                 body_pose=np.array(tr_p), bodies=np.array([m.body(b).name for b in free]),
                 **({"contact": np.array(tr_c)} if a.contacts else {}))
        os.replace(tmp, os.path.join(a.out, "track.npz"))
    for i in range(r0, stop):
        s.step(c[i])
        k = i + 1
        if k % a.every == 0 or k == stop:
            ev = s.S.mech.events[nev:]; nev = len(s.S.mech.events)
            log.write(json.dumps(dict(row=k, t=round(float(s.data.time), 4), wall=round(time.time() - t0, 1),
                                      slewed=s.slewed_ticks, ncon=int(s.data.ncon), events=ev, **summary(s)), default=str) + "\n")
            log.flush()
        if a.track and k % a.track == 0:
            tr_rows.append(k); tr_q.append(s.data.qpos[arm].copy()); tr_ctrl.append(s.data.ctrl.copy())
            tr_p.append(np.concatenate([s.data.xpos[free], s.data.xquat[free]], 1).copy())
            if a.contacts:
                tr_c.append(contact_mask())
        if a.track and a.flush and k % a.flush == 0:
            dump_track()
        if k % a.save == 0:
            st = s.get_state(); st["rows"] = []
            with open(os.path.join(a.out, f"state_{k:07d}.pkl"), "wb") as f:
                pickle.dump(st, f, protocol=4)
    if a.track:
        dump_track()
    if stop == len(c):
        fc = s.final_check()
        json.dump(dict(success=fc["success"], state_sha=fc["state_sha"], settled_sha=fc["settled_sha"],
                       end=fc["end"]["report"], wall=round(time.time() - t0, 1)), open(os.path.join(a.out, "final.json"), "w"),
                  default=str)


if __name__ == "__main__":
    main()
