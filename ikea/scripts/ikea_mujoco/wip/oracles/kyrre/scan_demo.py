"""Phase scan of the kyrre demo: every EVERY rows, the grippers, what each hand touches, the welds, the hex key's pose and
each screw's pose against its hole. --noprism: the author's scene geometry (the prism rule off; a study only).
--ck-every N: pickle checkpoints (rcb_ikea get_state + rows) every N rows.
usage: scan.py REF.npz OUT.jsonl [--every 100] [--noprism] [--ck-every N --ck-dir D]"""
import json, os, sys, time, pickle
import numpy as np, mujoco
sys.path.insert(0, os.path.join(os.environ.get("IKEA_SCENES", "/opt/ikea/scenes"), "kyrre"))
import ikea_runtime
if "--noprism" in sys.argv:
    ikea_runtime.cylinder_prisms = lambda spec: []
import rcb_ikea as R
from ikea_runtime import rotate
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
ref, out = sys.argv[1:3]; EVERY = arg("--every", 100); CKE = arg("--ck-every", 0); CKD = arg("--ck-dir", ".")
s = R.Session("kyrre"); m, d = s.model, s.data; M = s.S.mech
c = R.load_trajectory(ref, s.nu, s.timestep)
hand = {a: set(i for i in range(m.nbody) if m.body(i).name.startswith(a + "_") and ("finger" in m.body(i).name or "hand" in m.body(i).name)) for a in ("left", "right")}
parts = ["seat", "leg_1", "leg_2", "leg_3", "hex_key"] + [o["part"] for o in M.ops if o["mode"] == "screw"]
pid = {p: m.body(p).id for p in parts}
f = open(out, "w"); t0 = time.time()
for i, row in enumerate(c):
    s.step(row)
    if (i + 1) % EVERY == 0:
        n = d.ncon; gb = m.geom_bodyid; tch = {"left": set(), "right": set()}
        for k in range(n):
            x, y = int(gb[d.contact.geom1[k]]), int(gb[d.contact.geom2[k]])
            for a in ("left", "right"):
                if x in hand[a] and y not in hand[a]: tch[a].add(m.body(y).name)
                if y in hand[a] and x not in hand[a]: tch[a].add(m.body(x).name)
        gap, lat, ang = M.geometry_all()[:3]
        sc = {o["key"][3:]: [round(float(gap[j]) * 1e3, 2), round(float(lat[j]) * 1e3, 2), round(float(ang[j]), 1)] for j, o in enumerate(M.ops)}
        rec = dict(r=i + 1, gL=round(float(d.ctrl[7]), 1), gR=round(float(d.ctrl[15]), 1), L=sorted(tch["left"]), R=sorted(tch["right"]),
                   w=int(d.eq_active[M.eq_ids].sum()), key=[round(float(x) * 1e3, 1) for x in d.xpos[pid["hex_key"]]],
                   ops=sc, ncon=int(n), wall=round(time.time() - t0, 1))
        f.write(json.dumps(rec) + "\n"); f.flush()
    if CKE and (i + 1) % CKE == 0:
        st = s.get_state(); pickle.dump(st, open(os.path.join(CKD, "demo_%07d.pkl" % (i + 1)), "wb"))
fc = s.final_check()
f.write(json.dumps(dict(done=True, success=fc["success"], sha=fc["state_sha"], end={k: fc["end"][k] for k in ("goal", "released", "at_rest")})) + "\n")
