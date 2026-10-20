import os, sys, time
# Oracle code for `rcb_topogym.py run-oracle --task caliper_sorting --code <this file>` (variation caliper_sorting_vernier;
# docs/rcb/families/caliper_sorting.md). It records the -i01 reference (scripts/robotangle/oracles/caliper-sorting-01-i01).
# run-oracle runs with the repository as the working directory; the planner it imports sits next to this file
# (oracle4.py, plan3.py, scenario3.py, geom.py, aloha_kin.py). The caliper is fixed on its stand (2026-10-18,
# runs/caliper_sorting_design_20261014/STAND_FEASIBILITY.md): the left hand works it, the right brings the sleeves.
#
# The plan is made while the episode runs: oracle4.Episode (the kinematic planner, scene seed 0) yields after each
# sleeve's measurement and receives the two readings, which decide a re-measure (a reading more than 0.12 from every
# label value, at most twice) and the bin (the label nearest to both readings); before each loosening and after each
# turn of the knob it yields ('knob',) and receives the knob's angle, read from the locking screw's mechanism. A reading
# is what the 0.1-unit vernier shows at the slider's place in the simulation: vernier_reading of the slider joint's
# position (as a person reads the vernier on the spot), taken in the middle of the 1 s pause of the left wrist camera
# over the scale.
#
# Control, per 25 ms control step, the plan's 20 ms frames resampled:
#  - hand targets: while the assisted grasp holds a part, the hand goes where the part's planned pose and its pose in the
#    hand (measured when it was taken) put it; the slider's planned place is counted on the fixed caliper; the left
#    hand's path to the slider, to the knob and to the reading pose moves with the slider as it actually is, the right
#    hand's path to a sleeve in the rack with that sleeve (smoothed 0.1 s); the offset is frozen from the last 5 cm of an
#    approach to a part to the way up from it;
#  - the right hand puts the sleeve between the jaws or onto the knife edges only once the sleeve is within 0.3 mm of
#    where the plan holds it before the last approach (at most 3 s);
#  - closing the outside jaws onto a sleeve / opening the knife edges in its bore, guarded: the slider target creeps on at
#    1 mm/s until the slider's jaw presses the sleeve with more than 1 N for 0.25 s; the left hand then holds the slider there 0.3 s
#    and lets go; setting the jaws in the air, it lets go once the slider has arrived (0.2 mm, at most 2 s);
#  - locking on a sleeve (stages 17, 19): the pinch and the turn of the knob push the free slider along the beam (by up
#    to 0.14 mm, away from the sleeve: toward it the sleeve stops it). While the fingers hold the knob and the lock is not
#    yet on, the slider is held through the knob: the left hand's target moves along the beam by HOLD_P x the slider's
#    offset from its place plus a slow integral of it (HOLD_I per step; together at most HOLD_MAX), and the turn waits
#    while the offset is over HOLD_TOL (at most 5 s). Its place: where the hand let go of it; in a bore ID_CLEAR less
#    (the knife edges locked tight in the bore wedge it on them as it is pulled off);
#  - arms: IK near the plan's joints (aloha_kin.Robot.ik_near), joint targets plus feed-forward (inverse dynamics of the
#    planned motion on a contact-free copy of the model, the held part's weight; while a hand moves the slider, the
#    slider's friction along the beam) as a position offset through each drive's gain, and a slow integral of the
#    tracking error (frozen for the hand that holds the slider or pinches the knob, so it does not drag them, and for the
#    right hand while its sleeve waits on the fixed jaw for the closing and the lock (stages 5, 17): growing, it pressed
#    the sleeve on the jaw with 4-11 N by the lift, whose friction twisted the sleeve in the pinch); no arm
#    joint command changes by more than 0.035 rad per step;
#  - grippers: fully closed (0.002) while the plan closes or holds, the plan's opening otherwise;
#  - the locking screw: the left hand pinches the knob physically (no assisted grasp) and turns it; the lock's switch is
#    logged after each knob stage.
# Only joint and gripper targets are commanded; nothing is set in the simulation. The live model and data are only read
# (body, site and joint poses, the grasp welds' activity, contact forces between the jaws and a sleeve, the lock).
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot, Slerp

HERE = "scripts/robotangle/references/caliper_sorting"
sys.path.insert(0, os.path.abspath(HERE))
import aloha_kin as ak                  # noqa: E402
import geom as v2                       # noqa: E402
import oracle4                          # noqa: E402
import plan3 as p3                      # noqa: E402

s = session
w = s.env._world
m, d = w.model, w.data
task = s.env._task
DT = 1.0 / s.control_hz                 # 25 ms
assert abs(DT - 0.025) < 1e-9
F_SLIDER = 1.5                          # N, the slider's friction loss (topogym.assets.geometry.caliper_sorting)
TOUCH_N, TOUCH_STEPS = 1.0, 10          # 'the jaws touch': > 1 N on 10 control steps (0.25 s) in a row
OVER_RATE = 1.0                         # mm/s: after the planned closing / opening, the slider target creeps on
K_I, I_MAX = 0.05, 0.03                 # integral gain per control step, its limit (rad)
TAU_OFF = 0.10                          # s, smoothing of the hand-target offset
CMD_STEP = 0.035                        # rad: the largest change of an arm joint's command per control step
GRIP_CLOSED = 0.002
HOLD_P, HOLD_I, HOLD_MAX, HOLD_TOL = 15.0, 0.05, 2.0, 0.010   # the slider held through the knob: mm / mm, per step, mm, mm
ID_CLEAR = 0.010                        # mm: the knife edges' place in a bore, inside where the hand let go of the slider
LOCK_STAGES = ('17', '19')
PRESSED_STAGES = ('5', '17')            # the right hand's sleeve held on the fixed jaw while the left hand closes and locks
L_CAL_STAGES = ('3', '5', '7', '8', '10', '12', '17', '18', '19', '20', '21')   # the left hand's work on the caliper
KNOB_STAGES = ('17', '18', '19', '20', '21')
ON_SLEEVE = ('4', '5', '6', '9', '10', '11', '17', '19')   # stages with the sleeve at the jaws
HANDS = {'L': 'left', 'R': 'right'}
ARM_J = ('waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate')
AX = p3.T_from(oracle4.R_AXIS, np.zeros(3))   # sleeve body frame in the planner's world-aligned sleeve frame TS


def mj_id(kind, name):
    i = mujoco.mj_name2id(m, kind, name)
    if i < 0:
        raise KeyError(name)
    return i


# ---------------------------------------------------------------- frames: the planner works in the ALOHA mount frame
robot_body = mj_id(mujoco.mjtObj.mjOBJ_BODY, 'robot')
T_MOUNT = np.eye(4)
T_MOUNT[:3, :3] = d.xmat[robot_body].reshape(3, 3)
T_MOUNT[:3, 3] = d.xpos[robot_body]
assert np.allclose(T_MOUNT[:3, :3], np.eye(3)), 'the mount is turned'
T_MOUNT_INV = np.linalg.inv(T_MOUNT)
N_SL = len(task.scenario['sleeves'])
BODY = {'s_caliper': 'caliper::beam', 's_slider': 'caliper::slider', **{f's_sleeve{k}': f'sleeve_{k}::sleeve' for k in range(N_SL)}}
BID = {p: mj_id(mujoco.mjtObj.mjOBJ_BODY, b) for p, b in BODY.items()}
SITE = {h: mj_id(mujoco.mjtObj.mjOBJ_SITE, f'{side}/gripper') for h, side in HANDS.items()}
J_SL = m.jnt_qposadr[mj_id(mujoco.mjtObj.mjOBJ_JOINT, 'caliper::slider')]
WELDS = {}                              # hand -> [(equality id, part)]: the slider and the sleeves (the beam is fixed)
for h, side in HANDS.items():
    WELDS[h] = [(mj_id(mujoco.mjtObj.mjOBJ_EQUALITY, f'grasp/{side}/{e}/{l}'), p) for p, (e, l) in
                {'s_slider': ('caliper', 'slider'), **{f's_sleeve{k}': (f'sleeve_{k}', 'sleeve') for k in range(N_SL)}}.items()]


def T_body(part):
    b = BID[part]
    T = np.eye(4)
    T[:3, :3] = d.xmat[b].reshape(3, 3)
    T[:3, 3] = d.xpos[b]
    return T_MOUNT_INV @ T


def T_site(h):
    T = np.eye(4)
    T[:3, :3] = d.site_xmat[SITE[h]].reshape(3, 3)
    T[:3, 3] = d.site_xpos[SITE[h]]
    return T_MOUNT_INV @ T


def held(h):
    """The part the assisted grasp welds to hand h (its weld active), or None."""
    on = [p for eid, p in WELDS[h] if d.eq_active[eid]]
    return on[0] if on else None


class Arms:
    """The arms' drives: joint targets with feed-forward (inverse dynamics of the desired motion on a contact-free copy of
    the model: inertia, the drives' joint damping; the held part's weight), as the design's Arms (bed-bolt phys3.py)."""

    def __init__(self):
        self.act = {h: [mj_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f'{side}/drive/{j}') for j in ARM_J] for h, side in HANDS.items()}
        self.qadr = {h: np.array([m.jnt_qposadr[mj_id(mujoco.mjtObj.mjOBJ_JOINT, f'{side}/{j}')] for j in ARM_J])
                     for h, side in HANDS.items()}
        self.vadr = {h: np.array([m.jnt_dofadr[mj_id(mujoco.mjtObj.mjOBJ_JOINT, f'{side}/{j}')] for j in ARM_J])
                     for h, side in HANDS.items()}
        self.kp = {h: np.array([m.actuator_gainprm[a, 0] for a in self.act[h]]) for h in HANDS}
        self.frange = {h: np.array([m.actuator_forcerange[a, 1] for a in self.act[h]]) for h in HANDS}
        self.ctrl_lo = {h: np.array([m.actuator_ctrlrange[a, 0] for a in self.act[h]]) for h in HANDS}
        self.ctrl_hi = {h: np.array([m.actuator_ctrlrange[a, 1] for a in self.act[h]]) for h in HANDS}
        import copy
        self.mi = copy.deepcopy(m)              # a private copy: contacts off, for inverse dynamics only
        self.mi.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.di = mujoco.MjData(self.mi)
        self.site = SITE

    def feedforward(self, h, q, qd, qdd, payload=None):
        mi, di = self.mi, self.di
        di.qpos[:] = mi.qpos0
        di.qvel[:] = 0
        di.qacc[:] = 0
        di.qpos[self.qadr[h]] = q
        di.qvel[self.vadr[h]] = qd
        di.qacc[self.vadr[h]] = qdd
        mujoco.mj_inverse(mi, di)
        tau = di.qfrc_inverse[self.vadr[h]].copy()
        if payload is not None:
            mass, com = payload
            jp = np.zeros((3, mi.nv))
            mujoco.mj_jac(mi, di, jp, None, com, mi.site_bodyid[self.site[h]])
            tau += -(jp[:, self.vadr[h]].T @ (mass * mi.opt.gravity))
        return tau

    def command(self, h, q_des, tau_ff):
        """The position target that makes the drive produce tau_ff at the desired joints."""
        return np.clip(q_des + np.clip(tau_ff, -self.frange[h], self.frange[h]) / self.kp[h], self.ctrl_lo[h], self.ctrl_hi[h])

    def q(self, h):
        return d.qpos[self.qadr[h]].copy()


def interp_T(A, B, u):
    T = np.eye(4)
    T[:3, 3] = (1 - u) * A[:3, 3] + u * B[:3, 3]
    T[:3, :3] = Slerp([0, 1], Rot.from_matrix([A[:3, :3], B[:3, :3]]))(u).as_matrix()
    return T


class Resampler:
    """The plan's 20 ms frames at the control's 25 ms, incrementally (the plan grows while the episode runs)."""

    def __init__(self, src, dt_src):
        self.src, self.x, self.step = src, 0.0, DT / dt_src

    def more(self):
        fr, out = self.src, []
        while len(fr) >= 2 and self.x <= len(fr) - 1:
            i0 = min(int(np.floor(self.x)), len(fr) - 2)
            u = self.x - i0
            a, b = fr[i0], fr[i0 + 1]
            g = dict(a)
            g['i'] = i0
            for k in ('qL', 'qR', 'fL', 'fR', 's'):
                g[k] = (1 - u) * a[k] + u * b[k]
            g['TS'] = [interp_T(A, B, u) for A, B in zip(a['TS'], b['TS'])]
            out.append(g)
            self.x += self.step
        return out


# ---------------------------------------------------------------- the plan and the scene it assumes
# CAL_N (debugging only): plan the first CAL_N sleeves; the reference measures and sorts all four
E = oracle4.Episode(int(os.environ.get('CAL_N', 4)), 0)
scen = E.scen
assert [x['cls'] for x in scen['sleeves']] == [x['cls'] for x in task.scenario['sleeves']] and scen['bins'] == task.scenario['bins']
assert all(abs(a['od_mm'] - b['od_mm']) < 1e-12 and abs(a['id_mm'] - b['id_mm']) < 1e-12 and a['slot'] == b['slot']
           for a, b in zip(scen['sleeves'], task.scenario['sleeves']))
assert scen['classes'] == task.scenario['classes'] and scen['unit_mm'] == task.scenario['unit_mm']
TC = E.TC                               # the caliper's frame C on its stand (fixed)
assert np.allclose(T_body('s_caliper'), TC, atol=1e-6), 'the caliper is not where the plan has it'
LOCK = w._mechanism                     # the task's locking-screw mechanism (read only: its angle and switch)
RA = ak.Robot()
A = Arms()
for h in 'LR':      # the planner's kinematics against the simulated arm
    err = np.abs(RA.fk(h, A.q(h)) - T_site(h)).max()
    assert err < 1e-6, f'planner kinematics differ from the simulated {HANDS[h]} arm by {err}'
gen = E.plan()
msg = next(gen)
rs = Resampler(E.frames, oracle4.DT)
fr = rs.more()
plan_T, QD = [], {h: ([], []) for h in 'LR'}


def extend(new):
    """Append resampled plan frames: their hand poses and joint velocities / accelerations (for the feed-forward)."""
    n0 = len(fr) - len(new)
    plan_T.extend((RA.fk('L', f['qL']), RA.fk('R', f['qR'])) for f in new)
    for h in 'LR':
        lo = max(0, n0 - 2)
        q = np.array([f['q' + h] for f in fr[lo:]])
        if len(q) < 3:
            qd = np.zeros_like(q)
            qdd = np.zeros_like(q)
        else:
            qd = np.gradient(q, DT, axis=0)
            qdd = np.gradient(qd, DT, axis=0)
        QD[h][0].extend(list(qd[n0 - lo:]))
        QD[h][1].extend(list(qdd[n0 - lo:]))


extend(fr)
mass = {p: float(m.body_subtreemass[BID[p]]) for p in BODY}
cal_bodies = {BID['s_caliper'], BID['s_slider']}
sl_body = {k: BID[f's_sleeve{k}'] for k in range(N_SL)}
c6 = np.zeros(6)


def jaw_force(k, bodies=None):
    """Normal force (N) between the caliper's jaws (or the given bodies of it) and sleeve k."""
    bodies = cal_bodies if bodies is None else bodies
    tot = 0.0
    for j in range(d.ncon):
        c = d.contact[j]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in bodies and b2 == sl_body[k]) or (b2 in bodies and b1 == sl_body[k]):
            mujoco.mj_contactForce(m, d, j, c6)
            tot += abs(c6[0])
    return tot


H = {}                                        # part pose in the TCP frame, measured when the grasp took it
taken = set()
I = {h: np.zeros(6) for h in 'LR'}
q_prev = {h: None for h in 'LR'}
off = {h: np.eye(4) for h in 'LR'}            # smoothed offset (mount frame): command = off @ plan
closing = {h: False for h in 'LR'}
frozen = {h: None for h in 'LR'}
frames = []
t_wall = time.time()
a_off = DT / TAU_OFF
n, n_last, waited, ok_cnt, dwells = 0, -1, 0, 0, []
settles, settled, n_settle = [], False, 0
s_hold, relax, over, n_touch, s_cmd = None, 0, 0.0, 0, 0.0
s_let, hold_dx, hold_i, n_hold, holds = None, 0.0, 0.0, 0, []   # the slider's place on a sleeve; the hold through the knob
held_L = None
done = False
reading_run, phys_readings, knob_log = [], [], []
cmd_prev = np.asarray(s.robot.targets(), dtype=np.float64)
iL, iR = s.robot.arm_slice['left'], s.robot.arm_slice['right']
gL, gR = s.robot.grip_index['left'], s.robot.grip_index['right']
ended = False

while True:
    if n >= len(fr) - 2 and not done:
        if msg[0] in ('measured', 'remeasured'):
            k_m = msg[1]
            last = lambda k, what: [r for r in phys_readings if r['sleeve'] == k and r['what'] == what][-1]['value']   # noqa: E731
            msg = gen.send((last(k_m, 'OD'), last(k_m, 'ID')))
        elif msg[0] == 'knob':
            msg = gen.send(float(LOCK.read()['screw_angle_deg']))
        else:
            msg = next(gen)
        done = msg[0] == 'done'
        new = rs.more()
        fr.extend(new)
        extend(new)
    if n >= len(fr):
        break
    f = fr[n]
    repeat = n == n_last
    n_last = n
    if held('L') != 's_slider':
        s_hold, over = None, 0.0
    st = f['stage']
    num = st.split()[0]
    k_sl = int(st.split('sleeve ')[1].split(':')[0]) - 1 if 'sleeve ' in st.split(':')[0] else None
    s_act = float(d.qpos[J_SL] * 1000)
    knob_held = num in LOCK_STAGES and f['fL'] < p3.Q_OPEN - 1e-4 and s_let is not None
    if knob_held and not LOCK.read()['locked']:
        hold_i = float(np.clip(hold_i - HOLD_I * (s_act - s_let), -HOLD_MAX, HOLD_MAX))
        hold_dx = float(np.clip(hold_i - HOLD_P * (s_act - s_let), -HOLD_MAX, HOLD_MAX))
    elif num not in LOCK_STAGES:
        hold_dx, hold_i = 0.0, 0.0
    for h in 'LR':
        T_plan = plan_T[n][0 if h == 'L' else 1]
        part = held(h)
        if part == 's_slider':
            s_cmd = f['s'] - (over if num == '5' else -over if num == '10' else 0.0)
            if s_hold is not None:
                s_cmd = s_hold
            T_raw = p3.slider_T(TC, s_cmd) @ np.linalg.inv(H[part])
        elif part is not None:
            k = int(part[len('s_sleeve'):])
            T_raw = (f['TS'][k] @ AX) @ np.linalg.inv(H[part])
        else:
            ref = None
            if h == 'L' and num in L_CAL_STAGES:
                ref = (p3.slider_T(TC, s_act), p3.slider_T(TC, f['s']))
            elif h == 'R' and num == '2' and k_sl is not None:
                ref = (T_body(f's_sleeve{k_sl}') @ AX.T, f['TS'][k_sl])
            T_raw = (ref[0] @ np.linalg.inv(ref[1]) @ T_plan) if ref is not None else T_plan
            if f.get('tgt' + h) is not None:
                if frozen[h] is None:
                    frozen[h] = off[h].copy()
                T_raw = frozen[h] @ T_plan
            if h == 'L' and hold_dx:
                T_raw = p3.T_from(np.eye(3), TC[:3, 0] * hold_dx * 1e-3) @ T_raw
        if f.get('tgt' + h) is None or part is not None:
            frozen[h] = None
        Dr = T_raw @ np.linalg.inv(T_plan)
        Ds = off[h]
        Dn = np.eye(4)
        Dn[:3, 3] = (1 - a_off) * Ds[:3, 3] + a_off * Dr[:3, 3]
        Dn[:3, :3] = Slerp([0, 1], Rot.from_matrix([Ds[:3, :3], Dr[:3, :3]]))(a_off).as_matrix()
        if part is not None:
            Dn = Dr
        off[h] = Dn
    cmd = cmd_prev.copy()
    for h in 'LR':
        T_plan = plan_T[n][0 if h == 'L' else 1]
        T_cmd = off[h] @ T_plan
        q0 = np.asarray(f['q' + h], float)
        if np.allclose(off[h], np.eye(4), atol=1e-9):
            q_des = q0
        else:
            q_des, _, _ = RA.ik_near(h, T_cmd, q0, q0=q_prev[h])
        q_prev[h] = q_des
        payload = None
        part = held(h)
        if part is not None:
            payload = (mass[part], d.xipos[BID[part]].copy())
        z6 = np.zeros(6)
        tau = A.feedforward(h, q_des, z6 if repeat else QD[h][0][n], z6 if repeat else QD[h][1][n], payload)
        if part == 's_slider':      # the slider's friction along the beam, as feed-forward on the hand that moves it
            ds = s_cmd - s_act
            if abs(ds) > 0.02:
                F = np.sign(ds) * (F_SLIDER + 0.5) * TC[:3, 0]
                jp = np.zeros((3, A.mi.nv))
                mujoco.mj_jacSite(A.mi, A.di, jp, None, A.site[h])
                tau = tau + jp[:, A.vadr[h]].T @ F
        pinch = h == 'L' and num in KNOB_STAGES and f['fL'] < p3.Q_OPEN - 1e-4      # the fingers on the knob
        pressed = h == 'R' and num in PRESSED_STAGES                                  # the sleeve on the fixed jaw
        if part != 's_slider' and not pinch and not pressed:
            I[h] = np.clip(I[h] + K_I * (q_des - A.q(h)), -I_MAX, I_MAX)
        q_cmd = A.command(h, q_des + I[h], tau)
        sl = iL if h == 'L' else iR
        cmd[sl] = cmd_prev[sl] + np.clip(q_cmd - cmd_prev[sl], -CMD_STEP, CMD_STEP)
        fp = f['f' + h]
        fprev = fr[n - 1]['f' + h] if n else fp
        if fp < fprev - 1e-7 and fp < p3.Q_OPEN - 1e-4:
            closing[h] = True
        elif fp > fprev + 1e-7:
            closing[h] = False
        cmd[gL if h == 'L' else gR] = GRIP_CLOSED if closing[h] else fp
    if s.step(cmd):
        ended = True
        break
    cmd_prev = np.asarray(s.robot.targets(), dtype=np.float64)
    for h in 'LR':
        part = held(h)
        if part is not None and (h, part) not in taken:
            H[part] = np.linalg.inv(T_site(h)) @ T_body(part)
            taken.add((h, part))
        if part is None:
            taken.difference_update({(h, p) for p in BODY})
    s_act = float(d.qpos[J_SL] * 1000)
    if num in ('5', '10') and held_L == 's_slider' and held('L') is None:
        s_let = s_act - (ID_CLEAR if num == '10' else 0.0)   # the left hand has just let go of the slider on the sleeve
    elif num not in ('5', '10') + LOCK_STAGES:
        s_let = None
    held_L = held('L')
    on = k_sl is not None and num in ON_SLEEVE
    frames.append(dict(t=float(d.time), s=s_act, stage=st, jaw_N=jaw_force(k_sl) if on else 0.0,
                       slider_N=jaw_force(k_sl, {BID['s_slider']}) if on else 0.0, lock=LOCK.read()))
    if len(frames) > 1 and frames[-2]['stage'] != st and frames[-2]['stage'].split()[0] in KNOB_STAGES:
        lk = frames[-2]['lock']
        knob_log.append(dict(stage=frames[-2]['stage'], t=frames[-2]['t'], s=frames[-2]['s'], **lk))
        print('  %s: knob at %.1f deg, slider %s at %.3f mm' % (frames[-2]['stage'][:60], lk['screw_angle_deg'],
              'locked' if lk['locked'] else 'free', frames[-2]['s']), flush=True)
    # reading: the slider where it really is, in the middle of the 1 s pause in front of the vernier
    if num in ('7', '12') and f.get('reading') and k_sl is not None:
        reading_run.append(len(frames) - 1)
    elif reading_run:
        fm = frames[reading_run[len(reading_run) // 2]]
        k_r = int(fm['stage'].split('sleeve ')[1].split(':')[0]) - 1
        phys_readings.append(dict(sleeve=k_r, what='OD' if fm['stage'].startswith('7') else 'ID', s=fm['s'],
                                  value=v2.vernier_reading(fm['s'], scen), locked=fm['lock']['locked'],
                                  stage=fm['stage'], t=fm['t']))
        r = phys_readings[-1]
        print('  reading %s sleeve %d: slider at %.3f mm (%s) -> the vernier reads %.1f (t %.1f s)' % (
            r['what'], k_r + 1, r['s'], 'locked' if r['locked'] else 'free', r['value'], r['t']), flush=True)
        reading_run = []
    if n % 800 == 0:
        print('  t %6.1f s (wall %5.0f s) %-60s held L %-10s R %-10s s %.2f mm' % (
            d.time, time.time() - t_wall, st[:60], held('L'), held('R'), s_act), flush=True)
    # before the last approach to the jaws: the held sleeve within 0.3 mm of where the plan holds it (at most 3 s)
    if f.get('settle') and held('R') is not None and held('R').startswith('s_sleeve') and not settled:
        k = int(held('R')[len('s_sleeve'):])
        e_p = float(np.linalg.norm((T_body(held('R')) @ AX.T)[:3, 3] - f['TS'][k][:3, 3]))
        if e_p > 0.0003 and n_settle < 120:
            n_settle += 1
            continue
        settles.append((st[:2], n_settle * DT, e_p * 1e3))
        settled = True
    if not f.get('settle'):
        settled, n_settle = False, 0
    # guarded closing / opening: once the slider's jaw presses the sleeve with > TOUCH_N for TOUCH_STEPS steps, the slider
    # stops (the fixed jaw already presses it: the right hand holds the sleeve on it)
    if num in ('5', '10') and held('L') == 's_slider' and s_hold is None:
        n_touch = n_touch + 1 if frames[-1]['slider_N'] > TOUCH_N else 0
        if n_touch >= TOUCH_STEPS:
            s_hold, relax = frames[-1]['s'], 0
    else:
        n_touch = 0
    # locking on a sleeve: the knob's turn waits while the slider is off its place (<= 5 s)
    if knob_held and not LOCK.read()['locked'] and abs(s_act - s_let) > HOLD_TOL and n_hold < 200:
        n_hold += 1
        continue
    if num in LOCK_STAGES and n_hold and not knob_held:
        holds.append((st[:2], n_hold * DT, s_let))
        n_hold = 0
    # setting the jaws in the air (stages 3, 8): the plan lets go only once the slider has arrived (0.2 mm, <= 2 s)
    if (num in ('3', '8') and held('L') == 's_slider' and n + 1 < len(fr) and fr[n + 1]['fL'] > f['fL'] + 1e-7
            and abs(frames[-1]['s'] - f['s']) > 0.2 and waited < 80):
        waited += 1
        continue
    # onto a sleeve: the plan lets go of the slider only once the jaws touch, then after 0.3 s holding still
    if num in ('5', '10') and held('L') == 's_slider' and n + 1 < len(fr) and fr[n + 1]['fL'] > f['fL'] + 1e-7:
        if s_hold is None:
            ok_cnt = ok_cnt + 1 if frames[-1]['slider_N'] > TOUCH_N else 0
            if ok_cnt == 0:
                over += OVER_RATE * DT
            if ok_cnt >= TOUCH_STEPS or waited >= 80:
                s_hold = frames[-1]['s']
                relax = 0
            waited += 1
            continue
        if relax < 12:
            relax += 1
            continue
        dwells.append((st[:2], (waited + relax) * DT, frames[-1]['slider_N'], frames[-1]['s']))
    waited, ok_cnt = 0, 0
    n += 1

# the plan is done (both hands off, arms home): hold until the task confirms (its 0.5 s), at most 5 s
hold = np.asarray(s.robot.targets(), dtype=np.float64)
for _ in range(200):
    if ended or s.step(hold):
        break
print('oracle: %d control steps, %.1f s simulated, %.0f s wall' % (len(frames), d.time, time.time() - t_wall))
for dcs in E.decisions:
    print('  decision sleeve %d: readings OD %.1f, ID %.1f -> bin %d (label %s)%s' % (
        dcs['sleeve'] + 1, dcs['od'], dcs['id'], dcs['bin'] + 1, scen['bins'][dcs['bin']],
        ('; measured again %d time(s)' % dcs['remeasures']) if dcs['remeasures'] else ''))
for stg, wt, ep_ in settles:
    print('  stage %s: waited %.2f s for the sleeve to settle (then %.2f mm off)' % (stg.strip(), wt, ep_))
for stg, wt, fN, s_ in dwells:
    print('  stage %s: held %.2f s before letting go of the slider; jaw force %.1f N, slider %.3f mm' % (stg.strip(), wt, fN, s_))
for stg, wt, sl in holds:
    print('  stage %s: the knob\'s turn waited %.2f s for the slider to be back at %.3f mm' % (stg.strip(), wt, sl))
for kt in E.knob_turns:
    print('  knob, stage %s%s: %.1f deg, turned %+.1f, then %.1f deg' % (
        kt['stage'], ' (again)' if kt['again'] else '', kt['before'], kt['turn'], kt.get('after', float('nan'))))
if E.fail:
    print('  IK failures while planning: %d (first: %s)' % (len(E.fail), E.fail[0]))
print('  status', s.status())
