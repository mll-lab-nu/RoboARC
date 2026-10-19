#!/usr/bin/env python3
"""Extract a two-geom narrowphase crash from a scene into a standalone MJCF (diagnosis of MuJoCo SIGSEGVs).

From a saved pre-crash state (crashrow.py: {state, row}) the scene is stepped to the crashing substep; the two geoms'
world poses and, for a mesh, its COMPILED vertices and faces (mjModel.mesh_vert / mesh_face, the mesh frame) are
written into a minimal MJCF: two free bodies, each holding one geom at its body origin, with the scene's <option>
(timestep, cone, ccd_iterations, ccd_tolerance, multiccd / nativeccd flags). The poses are then perturbed by tiny
amounts (the recompiled model's frames differ from the scene's by round-off) and tried one by one in a child process
that logs each index before mj_forward: the first crashing pose is written as <out>.xml and checked in a fresh process.

usage: segv_extract.py --task T --state pre.pkl --substeps K --geoms G1 G2 --out DIR [--tries 4000]
       (run with the scene's Python: rcb_ikea / ikea_stress on sys.path via --code)
"""
import argparse, json, os, pickle, subprocess, sys
from pathlib import Path

import numpy as np


def mjcf(opt, flags, a, b, pa, qa, pb, qb):
    def f(v):
        return " ".join(repr(float(x)) for x in np.ravel(v))
    asset = ""
    geoms = []
    for g, tag in ((a, "a"), (b, "b")):
        if g["type"] == "mesh":
            asset += f'<mesh name="m{tag}" vertex="{f(g["vert"])}" face="{" ".join(str(int(i)) for i in np.ravel(g["face"]))}"/>'
            geoms.append(f'<geom type="mesh" mesh="m{tag}" margin="{g["margin"]!r}" gap="{g["gap"]!r}" condim="{g["condim"]}"/>')
        else:
            geoms.append(f'<geom type="{g["type"]}" size="{f(g["size"])}" margin="{g["margin"]!r}" gap="{g["gap"]!r}" '
                         f'condim="{g["condim"]}"/>')
    fl = " ".join(f'{k}="{v}"' for k, v in flags.items())
    return (f'<mujoco model="segv">\n<compiler angle="radian"/>\n'
            f'<option timestep="{opt["timestep"]!r}" cone="{opt["cone"]}" integrator="{opt["integrator"]}" '
            f'ccd_iterations="{opt["ccd_iterations"]}" ccd_tolerance="{opt["ccd_tolerance"]!r}"><flag {fl}/></option>\n'
            f'<size memory="64M"/>\n<asset>{asset}</asset>\n<worldbody>\n'
            f' <body name="A" pos="{f(pa)}" quat="{f(qa)}"><freejoint/>{geoms[0]}</body>\n'
            f' <body name="B" pos="{f(pb)}" quat="{f(qb)}"><freejoint/>{geoms[1]}</body>\n'
            f'</worldbody>\n</mujoco>\n')


CHILD = r'''
import sys, json, mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(sys.argv[1]); d = mujoco.MjData(m)
poses = json.load(open(sys.argv[2])); log = open(sys.argv[3], "w")
q0 = d.qpos.copy()
for i, p in enumerate(poses):
    log.write(f"{i}\n"); log.flush()
    d.qpos[:] = q0; d.qpos[0:3] = p[0]; d.qpos[3:7] = p[1]; d.qpos[7:10] = p[2]; d.qpos[10:14] = p[3]
    mujoco.mj_forward(m, d)
print("no crash in", len(poses))
'''

CHECK = "import mujoco,sys; m=mujoco.MjModel.from_xml_path(sys.argv[1]); d=mujoco.MjData(m); mujoco.mj_forward(m,d); print('ncon', d.ncon, 'version', mujoco.__version__)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True); ap.add_argument("--task", required=True); ap.add_argument("--state", required=True)
    ap.add_argument("--substeps", type=int, required=True); ap.add_argument("--geoms", nargs=2, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--tries", type=int, default=4000)
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    import mujoco
    import ikea_stress as I, rcb_ikea as R  # noqa: F401
    p = pickle.load(open(a.state, "rb"))
    s = R.Session(a.task); m, d = s.model, s.data
    if a.substeps:
        s.set_state(p["state"]); d.ctrl[:] = p["row"]; s.S.step(a.substeps)
    else:       # the crash is at this very state: only kinematics (set_state's mj_forward would collide and crash)
        mujoco.mj_setState(m, d, p["state"]["physics"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_kinematics(m, d)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    G = []
    for name in a.geoms:
        g = m.geom(name).id
        t = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}[int(m.geom_type[g])]
        q = np.zeros(4); mujoco.mju_mat2Quat(q, d.geom_xmat[g])
        rec = dict(name=name, type=t, size=np.array(m.geom_size[g]).tolist(), margin=float(m.geom_margin[g]),
                   gap=float(m.geom_gap[g]), condim=int(m.geom_condim[g]), pos=np.array(d.geom_xpos[g]).tolist(),
                   quat=q.tolist(), xmat=np.array(d.geom_xmat[g]).tolist(), body=m.body(int(m.geom_bodyid[g])).name)
        if t == "mesh":
            mid = int(m.geom_dataid[g])
            va, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            fa, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            rec.update(mesh=m.mesh(mid).name, vert=np.array(m.mesh_vert[va:va + nv]).tolist(),
                       face=np.array(m.mesh_face[fa:fa + nf]).tolist(), nvert=nv, nface=nf)
        G.append(rec)
    o = m.opt
    E = mujoco.mjtEnableBit; D = mujoco.mjtDisableBit
    flags = {}
    for nm, bit, en in (("multiccd", getattr(E, "mjENBL_MULTICCD", None), True), ("multiccd", getattr(D, "mjDSBL_MULTICCD", None), False),
                        ("nativeccd", getattr(D, "mjDSBL_NATIVECCD", None), False), ("nativeccd", getattr(E, "mjENBL_NATIVECCD", None), True)):
        if bit is None:
            continue
        on = bool(o.enableflags & int(bit)) if en else not bool(o.disableflags & int(bit))
        flags[nm] = "enable" if on else "disable"
    opt = dict(timestep=float(o.timestep), cone={0: "pyramidal", 1: "elliptic"}[int(o.cone)],
               integrator={0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}[int(o.integrator)],
               ccd_iterations=int(o.ccd_iterations), ccd_tolerance=float(o.ccd_tolerance))
    json.dump(dict(task=a.task, geoms=G, opt=opt, flags=flags, mujoco=mujoco.__version__), open(out / "pair.json", "w"), indent=1)
    print("opt", opt, "flags", flags, "geoms", [(g["name"], g["type"], g.get("nvert")) for g in G], flush=True)
    xml = mjcf(opt, flags, G[0], G[1], G[0]["pos"], G[0]["quat"], G[1]["pos"], G[1]["quat"])
    (out / "base.xml").write_text(xml)
    rng = np.random.default_rng(0)
    poses = []
    for i in range(a.tries):
        sig = 0. if i == 0 else [1e-13, 1e-11, 1e-9, 1e-7, 1e-6][i % 5]
        def pert(g):
            pos = np.array(g["pos"]) + rng.normal(0, sig, 3)
            q = np.array(g["quat"]) + rng.normal(0, sig, 4); q /= np.linalg.norm(q)
            return pos.tolist(), q.tolist()
        pa, qa = pert(G[0]); pb, qb = pert(G[1])
        poses.append([pa, qa, pb, qb])
    json.dump(poses, open(out / "poses.json", "w"))
    (out / "child.py").write_text(CHILD)
    r = subprocess.run([sys.executable, str(out / "child.py"), str(out / "base.xml"), str(out / "poses.json"), str(out / "poses.log")],
                       capture_output=True, text=True)
    print("sweep rc", r.returncode, r.stdout.strip()[-200:], flush=True)
    if r.returncode == 0:
        return 1
    k = int((out / "poses.log").read_text().split()[-1])
    pa, qa, pb, qb = poses[k]
    xml = mjcf(opt, flags, G[0], G[1], pa, qa, pb, qb)
    (out / "segv_min.xml").write_text(xml)
    for t in range(2):
        r = subprocess.run([sys.executable, "-c", CHECK, str(out / "segv_min.xml")], capture_output=True, text=True)
        print(f"crashing pose index {k}; fresh process check {t}: rc {r.returncode} {r.stdout.strip()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
