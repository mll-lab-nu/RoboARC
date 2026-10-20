"""cast_coil legacy tiers on MuJoCo: collision geometry and rest checks of one frozen instance (evidence for
docs/rcb/families/cast_coil.md).

usage (in the image; PYTHONPATH also has scripts/roboarc/shared): python geom_check.py <instance.json> <out.json>

1. Collision geometry of the built model: every task collision geom's type and thinnest extent (a box: its smallest
   edge), how many are thinner than 0.5 mm.
2. The assembled clearances in MuJoCo: the silver piece is put at the goal pose relative to the gold one (the task's own
   goal: identical rotation, the URDF body origins in the authored frame coincide), lifted 0.2 m above the table; then
   mj_forward's contacts between the two pieces (must be none) and mj_geomDistance between every gold / silver geom pair
   (the smallest = the declared clearance: 0.3 mm comb / baffle, 0.75 mm layers).
3. Rest: a fresh Session holds the birth targets 440 control steps (nop); the two pieces' positions are read after every
   control step: total drift from the birth pose, drift after the 0.5 mm birth drop has settled (from step 40), drift in
   the last 1 s, the deepest task contact and the final success flag.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, "/work/topogym/tools/stress")
import _env  # noqa: F401
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from rcb_topogym import Session

PIECES = ("coil_gold", "coil_silver")


def body(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{name}::{name}")


def geometry(s):
    w = s.env._world
    m = w.model
    out, kinds = {}, {}
    for name in PIECES:
        b = body(m, name)
        rows = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            kind = mujoco.mjtGeom(m.geom_type[g]).name
            if kind == "mjGEOM_BOX":
                thin = 2 * float(m.geom_size[g].min())
            elif kind == "mjGEOM_MESH":
                v = m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:][:m.mesh_vertnum[m.geom_dataid[g]]]
                thin = float(np.ptp(v, axis=0).min())
            else:
                thin = 2 * float(m.geom_size[g][0])
            rows.append((g, kind, thin))
            kinds[kind] = kinds.get(kind, 0) + 1
        th = np.array([r[2] for r in rows]) * 1000
        out[name] = dict(collision_geoms=len(rows), thinnest_mm=round(float(th.min()), 4),
                         under_0p5mm=int((th < .5).sum()), thinnest5_mm=np.round(np.sort(th)[:5], 4).tolist())
    return dict(by_kind=kinds, pieces=out, profile=s.env._world.profile.get("name"),
                solref=m.geom_solref[0].tolist(), timestep=float(m.opt.timestep),
                ccd_iterations=int(m.opt.ccd_iterations))


def assembled_gap(s):
    task, w = s.env._task, s.env._world
    m = w.model
    d = mujoco.MjData(m)
    d.qpos[:] = w.data.qpos
    joint = {n: m.body_jntadr[body(m, n)] for n in PIECES}
    adr = {n: m.jnt_qposadr[j] for n, j in joint.items()}
    qg = d.qpos[adr["coil_gold"]:adr["coil_gold"] + 7].copy()
    qg[2] += .2
    rg = Rotation.from_quat(qg[[4, 5, 6, 3]])
    # body frame = the piece's centre of mass in the authored frame (task.body_origins); assembled: the authored frames coincide
    authored = qg[:3] - rg.apply(task.body_origins["coil_gold"])
    qs = qg.copy()
    qs[:3] = authored + rg.apply(task.body_origins["coil_silver"])
    d.qpos[adr["coil_gold"]:adr["coil_gold"] + 7] = qg
    d.qpos[adr["coil_silver"]:adr["coil_silver"] + 7] = qs
    mujoco.mj_forward(m, d)
    bg, bs = body(m, "coil_gold"), body(m, "coil_silver")
    touching = [i for i in range(d.ncon) if {m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]} == {bg, bs}]
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bg and m.geom_contype[g]]
    ss = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bs and m.geom_contype[g]]
    fromto = np.zeros(6)
    dist = np.array([[mujoco.mj_geomDistance(m, d, a, b, .01, fromto) for b in ss] for a in gs])
    # the goal predicate on this pose (must read as aligned)
    pg, ps = d.xpos[bg], d.xpos[bs]
    err = np.linalg.norm((ps - Rotation.from_quat(d.xquat[bs][[1, 2, 3, 0]]).apply(task.body_origins["coil_silver"]))
                         - (pg - Rotation.from_quat(d.xquat[bg][[1, 2, 3, 0]]).apply(task.body_origins["coil_gold"])))
    return dict(contacts_between_pieces=len(touching), min_geom_distance_mm=round(1000 * float(dist.min()), 4),
                pairs_under_1mm=int((dist < .001).sum()), pairs=int(dist.size),
                goal_relative_position_error_mm=round(1000 * float(err), 6))


def rest(instance):
    s = Session(json.loads(Path(instance).read_text())["task"], instance=str(instance), strict=True)
    w = s.env._world
    m = w.model
    ids = [body(m, n) for n in PIECES]
    trace = [w.data.xpos[ids].copy()]
    depth = 0.
    t0 = time.time()
    for _ in range(440):
        s.step(s.hold_action())
        trace.append(w.data.xpos[ids].copy())
        for i in range(w.data.ncon):
            c = w.data.contact[i]
            if m.geom_bodyid[c.geom1] in ids or m.geom_bodyid[c.geom2] in ids:
                depth = max(depth, -float(c.dist))
    wall = time.time() - t0
    trace = np.asarray(trace)
    hz = s.control_hz
    last = int(round(hz))

    def drift(a, b):
        return {n: round(1000 * float(np.linalg.norm(trace[b, k] - trace[a, k])), 4) for k, n in enumerate(PIECES)}

    st = s.status()
    out = dict(control_hz=hz, steps=440, wall_s=round(wall, 1), drift_from_birth_mm=drift(0, -1),
               drift_after_settle_mm=drift(40, -1), drift_last_1s_mm=drift(-1 - last, -1),
               deepest_task_contact_mm=round(1000 * depth, 4), success=st.get("success"),
               episode_end=st.get("episode_end"), state_sha=st.get("state_sha"))
    s.close()
    return out


def main():
    instance, out = sys.argv[1], sys.argv[2]
    s = Session(json.loads(Path(instance).read_text())["task"], instance=str(instance), strict=True)
    rec = dict(instance=instance, instance_sha=s.instance_sha, geometry=geometry(s), assembled=assembled_gap(s))
    s.close()
    rec["rest"] = rest(instance)
    Path(out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
