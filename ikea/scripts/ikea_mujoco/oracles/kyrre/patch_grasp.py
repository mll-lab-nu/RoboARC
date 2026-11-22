"""Re-aim one arm's grasp of a part that lies off its demo pose: over a row window the arm's joint targets are moved so
its TCP keeps the demo's pose RELATIVE TO THE PART (rigid correction T = T_part_actual * T_part_demo^-1, blended in over
[r0, r1], held over [r1, r2], blended out over [r2, r3]); 6-D damped IK on the target configuration; then every arm
joint is clamped to the robot's per-tick speed limit (rows after the window follow until they catch up).
usage: patch_grasp.py IN.npz OUT.npz ARM 'px,py,pz,qw,qx,qy,qz demo(mm)' 'same actual' r0 r1 r2 r3"""
import sys, json
import numpy as np, mujoco
import rcb_ikea as R
src, out, arm, pd, pa = sys.argv[1:6]; r0, r1, r2, r3 = map(int, sys.argv[6:10])
s = R.Session('kyrre'); m = s.model; d = mujoco.MjData(m)
c = R.load_trajectory(src, s.nu, s.timestep).copy()
cols = [i for i, n in enumerate(s.actuators) if n.startswith(arm + '_') and not n.endswith('actuator8')]
jids = [int(m.actuator_trnid[i, 0]) for i in cols]; qadr = [int(m.jnt_qposadr[j]) for j in jids]; dadr = [int(m.jnt_dofadr[j]) for j in jids]
site = m.site(arm + '_tcp').id
import os
MATH = os.environ.get('PG_MATH', 'v2')      # v1 / v2: the two IK step forms the reference's two steps were made with
NULL = [m.body(arm + '_' + b).id for b in os.environ.get('NULL_BODIES', '').split(',') if b]
def pose(v):
    v = [float(x) for x in v.split(',')]; p = np.array(v[:3]) / 1e3; q = np.array(v[3:]); Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, q); return p, Rm.reshape(3, 3)
(pD, RD), (pA, RA) = pose(pd), pose(pa)
Rc = RA @ RD.T; tc = pA - Rc @ pD                     # x_actual = Rc x_demo + tc
rv = np.zeros(3); qc = np.zeros(4); mujoco.mju_mat2Quat(qc, Rc.reshape(-1)); mujoco.mju_quat2Vel(rv, qc, 1.0)
def alpha(r):
    if r < r0 or r >= r3: return 0.0
    if r < r1: x = (r - r0) / (r1 - r0)
    elif r < r2: return 1.0
    else: x = 1 - (r - r2) / (r3 - r2)
    return 0.5 - 0.5 * np.cos(np.pi * x)
def fk(q):
    d.qpos[:] = m.qpos0; d.qpos[qadr] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    return d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3).copy()
maxdq = 0.0; worst = 0.0
for r in range(r0, r3):
    a = alpha(r)
    if a == 0: continue
    q0 = c[r, cols].copy(); p0, M0 = fk(q0)
    qa = np.zeros(4); mujoco.mju_axisAngle2Quat(qa, rv / (np.linalg.norm(rv) + 1e-15), a * np.linalg.norm(rv)) if np.linalg.norm(rv) > 0 else None
    Ra = np.zeros(9); mujoco.mju_quat2Mat(Ra, qa if np.linalg.norm(rv) > 0 else np.array([1., 0, 0, 0])); Ra = Ra.reshape(3, 3)
    # blended correction about the demo part's centre: x -> Ra (x - pD) + pD + a (pA - pD)
    pt = Ra @ (p0 - pD) + pD + a * (pA - pD); Mt = Ra @ M0
    q = q0.copy()
    if NULL:
        fk(q0); el0 = {b: d.xpos[b].copy() for b in NULL}
    for it in range(20 if MATH == 'v1' else 30):
        p, Mx = fk(q)
        e = np.zeros(6); e[:3] = pt - p
        qe = np.zeros(4); qx = np.zeros(4); qt = np.zeros(4)
        mujoco.mju_mat2Quat(qx, Mx.reshape(-1)); mujoco.mju_mat2Quat(qt, Mt.reshape(-1)); mujoco.mju_subQuat(e[3:], qt, qx); e[3:] = Mx @ e[3:]     # local -> world
        if np.linalg.norm(e[:3]) < 1e-6 and np.linalg.norm(e[3:]) < 1e-5: break
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jp, jr, site)
        J = np.vstack([jp[:, dadr], jr[:, dadr]])
        if MATH == 'v1':                           # the first version (screw 3-1's step of the reference was made with it)
            q = q + np.clip(J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), -0.05, 0.05); continue
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6)); dq = Jp @ e
        if NULL:                                   # secondary: keep the elbow / forearm bodies where the demo had them
            N = np.eye(len(dadr)) - Jp @ J
            for b in NULL:
                jb = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, d, jb, None, b)
                Jb = jb[:, dadr]; dq = dq + N @ (Jb.T @ np.linalg.solve(Jb @ Jb.T + 1e-4 * np.eye(3), el0[b] - d.xpos[b])) * 0.5
        q = q + np.clip(dq, -0.05, 0.05)
    worst = max(worst, float(np.linalg.norm(e[:3])))
    maxdq = max(maxdq, float(np.abs(q - q0).max())); c[r, cols] = q
# clamp every arm column to the per-tick limit (from the row before the window)
lim = np.asarray(s.step_limit)[cols]; nclamp = 0
for r in range(max(r0, 1), len(c)):
    dq = c[r, cols] - c[r - 1, cols]; o = np.abs(dq) > lim + 1e-12
    if o.any():
        c[r, cols] = c[r - 1, cols] + np.clip(dq, -lim, lim); nclamp += 1
    elif r >= r3:
        break
s.save_trajectory(c, out)
print(json.dumps(dict(out=out, rows=len(c), correction_mm=((pA - pD) * 1e3).round(2).tolist(), rot_deg=round(float(np.degrees(np.linalg.norm(rv))), 2),
                      max_dq_rad=round(maxdq, 4), ik_residual_mm=round(worst * 1e3, 4), clamped_rows=nclamp, last_clamped_row=r)))
