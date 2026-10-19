"""Which hex-key rest poses come to rest (both step parities)? From a checkpoint, teleport the key (zero velocity) to
candidate poses: the cradle and the table, the key rotated about its shaft; hold the last row 3 s; report.
usage: keyrest.py TRAJ CK.pkl"""
import json, pickle, sys
import numpy as np, mujoco
import rcb_ikea as R
traj, ck = sys.argv[1:3]
s = R.Session('kyrre'); c = R.load_trajectory(traj, s.nu, s.timestep); st0 = pickle.load(open(ck, 'rb'))
m, d = s.model, s.data; k = m.body('hex_key').id; j = m.joint('hex_key_free'); a, v = j.qposadr[0], j.dofadr[0]
s.set_state(st0); p0 = d.qpos[a:a+3].copy(); q0 = d.qpos[a+3:a+7].copy()
Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, q0); Rm = Rm.reshape(3, 3); axis = Rm[:, 2]
print('shaft axis world', axis.round(3).tolist(), 'arm dir', Rm[:, 0].round(3).tolist())
def lowest(q, p):
    mujoco.mj_kinematics(m, d); zs = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == k and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]; vs = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            zs.append((d.geom_xpos[g][None] + vs @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
    return min(zs)
res = []
for place in ('cradle', 'table'):
    for deg in range(0, 360, 30):
        s.set_state(st0)
        qr = np.zeros(4); mujoco.mju_axisAngle2Quat(qr, axis, np.radians(deg)); q = np.zeros(4); mujoco.mju_mulQuat(q, qr, q0)
        p = p0.copy()
        if place == 'table': p[1] -= 0.10; p[2] = 0.2
        d.qpos[a:a+3] = p; d.qpos[a+3:a+7] = q; d.qvel[v:v+6] = 0
        lz = lowest(q, p); base = 0.15 if place == 'cradle' else 0.10
        if place == 'table' or lz < base + 0.0005: d.qpos[a+2] += (base + 0.001) - lz
        mujoco.mj_forward(m, d)
        row = c[-1]; ws = []
        for t in range(int(3.0 / s.control_dt)):
            s.step(row)
        for t in range(20):
            s.step(row); tm = s.validity.terminal(); ws.append((tm['at_rest'], round(tm['max_part_angular_rad_s'], 4), tm['fastest_part']))
        res.append(dict(place=place, deg=deg, at_rest_all=all(w[0] for w in ws), max_ang=max(w[1] for w in ws), fastest=ws[-1][2],
                        key_pos=(d.xpos[k] * 1e3).round(1).tolist()))
        print(json.dumps(res[-1]))
