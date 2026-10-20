import numpy as np
# Oracle code for `rcb_topogym.py run-oracle --task bed_bolt --code <this file>` on bed_bolt_phillips, the ratcheting key's
# drives (docs/rcb/families/bed_bolt.md): bed_bolt_square's oracle as it is (the cross recess, like the square, matches
# itself every 90 degrees). run-oracle runs with the repository as the working directory. bed_bolt_crossbar's
# oracle on its kinematic plan (plan.npz, plan.json next to that oracle: the design's oracle2.py, 50 Hz, ALOHA mount
# frame), changed for the ratchet (topogym.assets.geometry.bed_bolt RATCHET; the user 2026-10-18):
#   - the key goes into the recess once; between swings it is not drawn out: each "drawn out, turned back, pushed in"
#     segment of the plan is its turn back with the tip kept in the recess (the plan's key pose moved back along the bolt
#     axis by the plan's draw-out), and the ratchet lets the L turn back while the bit stays;
#   - the first insertion turns the key so that its bit (the L's angle plus the ratchet's) meets the recess, whose
#     symmetry is the drive's (90 deg for the square and the cross); after it the L follows the plan's own angles (its swings from -30 to
#     +30 deg) with no turn about the bolt axis taken from the bolt (the bit and the bolt turn with it);
#   - no waiting for the key to leave the socket, no push-in retries (there are no push-ins after the first).
# The plain keys' oracle (bed_bolt_crossbar's) header follows.
#
# The design's closed-loop replay on the real ALOHA (runs/bed_bolt_design_20261012/scripts/phys3_replay.py, README
# section 9) through the Session: per 40 Hz control step, on the plan resampled to 40 Hz (the key's insertion, every
# lever swing and the final draw-out at half speed, every push-in after a turn-back at a quarter):
#   - the key's path is mapped onto the bolt as it really is before the key goes into the socket (the correction taken
#     once, in full, as each push-in starts, then held); a swing ends at the planned lever angle; the turn-back waits
#     while the key end is still in the socket; before a push-in the arm settles until the key's short tip is within
#     0.2 mm of its command; a push-in that does not reach 3 mm into the socket backs off and pushes again (at most
#     RETRIES times), and each push-in's sideways landing error is learned (70 %) for the next; extra strokes while the
#     head is not seated;
#   - hand targets from part targets: once the assisted grasp holds a part, the hand goes where the part's target and its
#     measured pose in the hand put it;
#   - arms: IK near the plan's joint angles (ik_near: a wrist singularity keeps a small rotation error instead of
#     swinging the joints), then joint targets with feed-forward (inverse dynamics of the planned motion on a contact-free
#     copy of the model, the held part's weight) over the drives' gain, and a slow integral of the tracking error; a
#     held (waiting) frame feeds no velocity forward;
#   - grippers (native finger targets): closed (0.002) on a part while the plan holds it; the right hand, once the
#     thread has caught, loosens to a light pinch 4 mm past where its fingers rest on the shank (the mechanism has
#     released the weld at the catch: the engaged bolt is no assisted-grasp target; the fingers keep pressing) and opens
#     when the plan lets go.
# Only joint and gripper targets are commanded (at most CMD_STEP per arm joint per control step); no pose is ever set.
# The thread (catch, hold, let go) is the task's own mechanism (topogym.tasks.bed_bolt.thread),
# read here as privileged state.
import json, os, sys, copy
import mujoco
from scipy.spatial.transform import Rotation as Rot, Slerp
from topogym.assets.geometry import bed_bolt as geo

HERE = "scripts/robotangle/references/bed_bolt/bed_bolt_crossbar"     # the plan
s = session
w = s.env._world
m, d = w.model, w.data
MECH = w._mechanism
MOUNT = np.r_[-0.1356786, 0.0, float(s.env._task.table_height_m)]
HANDS = {"L": "left", "R": "right"}
ARM_J = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
GRIP_OPEN, GRIP_CLOSED = 0.037, 0.002
TICK = float(w.physics_dt_s)
DT = 1.0 / s.control_hz                       # 25 ms
Q_OPEN_MIN = 0.020
PINCH = 0.004                         # light pinch: finger target this far past where the fingers rest on the shank (the
                                      # design's 2 mm let the bolt droop past 3 degrees under the 1 N m thread's strokes and
                                      # the thread let go at 1.95 turns; with 4 mm it droops at most 6.6 degrees, supported)
SLOW, SLOW_PUSH = 2.0, 2.0            # the slow stages at half speed; push-ins after a turn-back another 2x slower
K_I, I_MAX = 0.05, 0.03               # integral gain per control step, its limit (rad)
DWELL_MAX = 80                        # control steps the arm may settle before a push-in
RETRIES = 12                          # push-ins of one stroke before the oracle gives up on the key (design: 3): a stroke
                                      # swung with the key off the socket turns the bolt by friction (topogym run 1: the
                                      # bolt got 2.5 turns ahead of the plan and the last strokes forced the seat)
GUARD_LAT = 1.0                       # guarded insertion: the tip this far off the socket axis at the mouth -> hold the push
ILC_GAIN = 0.7                        # learned push-in offset: this share of each landing error is added for the next push
CMD_STEP = 0.035                      # rad: the largest change of an arm joint's command per control step (standard's 0.04)
FACE = geo.BOLT["L"] + geo.BOLT["head_h"]   # socket mouth in the bolt frame (m)
SEAT_TOL = 0.0015                     # seated: the slide within this of the full depth; short of it at stage 5 -> extra strokes
DWELL_TOL = 0.0002                    # before a push-in, hold until the key's short tip is this close to its command (m)
SHORT_TIP = np.r_[geo.key_end_points()["short"], 1.0]
STROKE = np.radians(60)
THETA0 = geo.THETA0
THETA_END = THETA0 + 2 * np.pi * (geo.BOLT["L"] - 0.003) / geo.BOLT["lead"]
N_STROKES = int(np.ceil((THETA_END - THETA0) / STROKE - 1e-9))
report = dict(events=[], strokes=[], stages=[])       # stages: [first control step, stage] at every new plan segment
STOP_S = float(os.environ.get("BED_BOLT_STOP_S", "inf"))     # development: stop the run at this simulated time
DRIVE = s.env._task.drive
assert geo.DRIVES[DRIVE].get("ratchet"), "this oracle is the ratcheting key's"
PERIOD = np.radians(geo.DRIVES[DRIVE]["period_deg"])
DEPTH = geo.DRIVES[DRIVE]["depth"]
RATCHET_Q = MECH.ratchet_q


def note(text):
    report["events"].append([round(float(d.time), 3), text])
    print("  t %7.2f  %s" % (d.time, text), flush=True)


# ---------------------------------------------------------------- the design's kinematic model (mount frame)
def rotvec(Rm):
    c = (Rm[0, 0] + Rm[1, 1] + Rm[2, 2] - 1) / 2
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    if c > 0.999999:
        return 0.5 * v
    if c < -0.99:
        return Rot.from_matrix(Rm).as_rotvec()
    ang = np.arccos(c)
    return ang / (2 * np.sin(ang)) * v


class Kin:
    """The ALOHA's two arms in the mount frame (the robot asset's MJCF, base links on the table plane)."""
    MARGIN = 0.15

    def __init__(self):
        spec = mujoco.MjSpec.from_file(os.path.join(os.environ.get("TOPOGYM_ASSET_ROOT", "assets"), "robots", "aloha", "aloha.xml"))
        for side in ("left", "right"):
            spec.body(side + "/base_link").pos[2] = 0.0
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.qadr = {h: np.array([self.m.jnt_qposadr[jid("%s/%s" % (sd, j))] for j in ARM_J]) for h, sd in HANDS.items()}
        self.dadr = {h: np.array([self.m.jnt_dofadr[jid("%s/%s" % (sd, j))] for j in ARM_J]) for h, sd in HANDS.items()}
        self.lim = {h: np.array([self.m.jnt_range[jid("%s/%s" % (sd, j))] for j in ARM_J]) for h, sd in HANDS.items()}
        self.site = {h: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, sd + "/gripper") for h, sd in HANDS.items()}
        self._jp, self._jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))

    def fk(self, h, q):
        self.d.qpos[self.qadr[h]] = q
        mujoco.mj_kinematics(self.m, self.d)
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[self.site[h]].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[self.site[h]]
        return T

    def ik_near(self, h, T, q_ref, q0=None, iters=40, pos_tol=0.001, rot_tol=0.02, mu=1.0):
        """IK that stays near q_ref: |pose error|^2 (position in units of pos_tol, rotation of rot_tol) + mu |q - q_ref|^2."""
        lo, hi = self.lim[h][:, 0] + self.MARGIN, self.lim[h][:, 1] - self.MARGIN
        q_ref = np.asarray(q_ref, float)
        q = np.clip(np.asarray(q_ref if q0 is None else q0, float), lo, hi)
        da, sid = self.dadr[h], self.site[h]
        wv = np.r_[np.full(3, 1 / pos_tol), np.full(3, 1 / rot_tol)]
        for _ in range(iters):
            self.d.qpos[self.qadr[h]] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            e = np.r_[T[:3, 3] - self.d.site_xpos[sid], rotvec(T[:3, :3] @ self.d.site_xmat[sid].reshape(3, 3).T)]
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, sid)
            J = wv[:, None] * np.r_[self._jp[:, da], self._jr[:, da]]
            dq = np.linalg.solve(J.T @ J + (mu + 1e-6) * np.eye(6), J.T @ (wv * e) + mu * (q_ref - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
            if np.abs(dq).max() < 1e-6:
                break
        return q


# ---------------------------------------------------------------- the arms' drives on this session's model
class Arms:
    def __init__(self):
        self.act = {h: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "%s/drive/%s" % (sd, j)) for j in ARM_J]
                    for h, sd in HANDS.items()}
        jq = lambda sd, j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "%s/%s" % (sd, j))]
        jv = lambda sd, j: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "%s/%s" % (sd, j))]
        self.qadr = {h: np.array([jq(sd, j) for j in ARM_J]) for h, sd in HANDS.items()}
        self.vadr = {h: np.array([jv(sd, j) for j in ARM_J]) for h, sd in HANDS.items()}
        self.kp = {h: np.array([m.actuator_gainprm[a, 0] for a in self.act[h]]) for h in HANDS}
        self.frange = {h: np.array([m.actuator_forcerange[a, 1] for a in self.act[h]]) for h in HANDS}
        self.crange = {h: np.array([m.actuator_ctrlrange[a] for a in self.act[h]]) for h in HANDS}
        self.mi = copy.deepcopy(m)                 # contact-free, for inverse dynamics
        self.mi.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.di = mujoco.MjData(self.mi)
        self.hand_body = {h: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, sd + "/gripper_link") for h, sd in HANDS.items()}

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
            mujoco.mj_jac(mi, di, jp, None, com, self.hand_body[h])
            tau += -(jp[:, self.vadr[h]].T @ (mass * mi.opt.gravity))
        return tau

    def target(self, h, q_des, tau_ff):
        """The joint target that makes the PD drive produce tau_ff at q_des."""
        return np.clip(q_des + np.clip(tau_ff, -self.frange[h], self.frange[h]) / self.kp[h],
                       self.crange[h][:, 0], self.crange[h][:, 1])

    def q(self, h):
        return d.qpos[self.qadr[h]].copy()


# ---------------------------------------------------------------- the plan
with np.load(os.path.join(HERE, "plan.npz")) as z:      # read each array once (an NpzFile reloads it on every access)
    P = {k: z[k] for k in z.files}
PJ = json.loads(open(os.path.join(HERE, "plan.json")).read())
STAGES = PJ["stages"]
frames = [dict(t=float(P["t"][i]), qL=P["qL"][i], qR=P["qR"][i], fL=float(P["fL"][i]), fR=float(P["fR"][i]), Tb=P["Tb"][i],
               Tk=P["Tk"][i], theta=None if np.isnan(P["theta"][i]) else float(P["theta"][i]), key_in=bool(P["key_in"][i]),
               stage=STAGES[int(P["stage"][i])]) for i in range(len(P["t"]))]
for g in frames:                # the ratchet: every turn back with the tip kept in the recess (the plan's draw-out undone)
    if "drawn out" in g["stage"]:
        pull = (np.linalg.inv(g["Tb"]) @ g["Tk"] @ np.r_[geo.key_end_points()["short"], 1.0])[0] - (geo.BOLT["L"] + geo.BOLT["head_h"] - DEPTH)
        if pull > 0:
            g["Tk"] = g["Tk"].copy()
            g["Tk"][:3, 3] -= pull * g["Tb"][:3, 0]
        g["key_in"] = True
assert tuple(PJ["place"]) == tuple(geo.PLACE)


def slow_stage(stage):
    return stage.startswith("2 key turned upright") or stage.startswith("3 stroke") or stage.startswith("5 head")


def resample(fr, dt_src=0.02, dt=DT):
    out, x = [], 0.0
    while x <= len(fr) - 1:
        i0 = min(int(np.floor(x)), len(fr) - 2)
        u = x - i0
        a, b = fr[i0], fr[i0 + 1]
        g = dict(a)
        g["i"] = i0
        for key in ("qL", "qR"):
            g[key] = (1 - u) * np.asarray(a[key], float) + u * np.asarray(b[key], float)
        for key in ("Tk", "Tb"):
            T = np.eye(4)
            T[:3, 3] = (1 - u) * a[key][:3, 3] + u * b[key][:3, 3]
            T[:3, :3] = Slerp([0, 1], Rot.from_matrix([a[key][:3, :3], b[key][:3, :3]]))(u).as_matrix()
            g[key] = T
        if a["theta"] is not None and b["theta"] is not None:
            g["theta"] = (1 - u) * a["theta"] + u * b["theta"]
        out.append(g)
        rate = 1.0 / SLOW if slow_stage(a["stage"]) else 1.0
        if a["key_in"] and "drawn out" in a["stage"] and any(not fr[j]["key_in"] for j in range(max(0, i0 - 60), i0)
                                                          if fr[j]["stage"] == a["stage"]):
            rate /= SLOW_PUSH
        x += rate * dt / dt_src
    return out


def spin_about(S, a):
    R = Rot.from_rotvec(S[1] * a).as_matrix()
    W = np.eye(4)
    W[:3, :3] = R
    W[:3, 3] = S[2] - R @ S[2]
    return W


def bolt_T_plan(Th, theta):
    """The bolt's planned pose at spin theta (its tip 3 mm deep at THETA0, then a lead per turn)."""
    a = 0.003 + geo.BOLT["lead"] * (theta - THETA0) / (2 * np.pi)
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec([theta, 0, 0]).as_matrix() @ Rot.from_euler("z", np.pi).as_matrix()
    T[:3, 3] = (a, 0, 0)
    return Th @ T


fr = resample(frames)
RA = Kin()
plan_T = [(RA.fk("L", f["qL"]), RA.fk("R", f["qR"])) for f in fr]
QD = {}
for h in "LR":
    q = np.array([f["q" + h] for f in fr])
    qd = np.gradient(q, DT, axis=0)
    QD[h] = (qd, np.gradient(qd, DT, axis=0))
tip_plan_ax = [(np.linalg.inv(g["Tb"]) @ g["Tk"] @ SHORT_TIP)[0] for g in fr]
A = Arms()
Th = geo.hole_T()                                    # mount frame
BODY = {"bolt": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bolt::bolt"),
        "key": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "key::key")}
PART_BODY = {v: k for k, v in BODY.items()}
SITE = {h: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sd + "/gripper") for h, sd in HANDS.items()}
FINGER_Q = {h: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, sd + "/left_finger")] for h, sd in HANDS.items()}
mass = {p: float(m.body_subtreemass[b]) for p, b in BODY.items()}
iL, iR = list(range(0, 6)), list(range(7, 13))
assert [s.action_layout[i] for i in iL] == ["left/" + j for j in ARM_J]
assert [s.action_layout[i] for i in iR] == ["right/" + j for j in ARM_J]


def T_site(h):
    T = np.eye(4)
    T[:3, :3] = d.site_xmat[SITE[h]].reshape(3, 3)
    T[:3, 3] = d.site_xpos[SITE[h]] - MOUNT
    return T


def T_body(part):
    T = np.eye(4)
    T[:3, :3] = d.xmat[BODY[part]].reshape(3, 3)
    T[:3, 3] = d.xpos[BODY[part]] - MOUNT
    return T


def held(h):
    b = ((w._grasp_manager.read().get("hands") or {}).get(HANDS[h]) or {}).get("binding")
    return str(b["target"][0]) if b else None


def touching_other(part, hand):
    """The part touches something other than this hand (contacts of the tick's last substep)."""
    b = BODY[part]
    for j in range(d.ncon):
        b1, b2 = m.geom_bodyid[d.contact[j].geom1], m.geom_bodyid[d.contact[j].geom2]
        if b in (b1, b2):
            other = b2 if b1 == b else b1
            if not m.body(other).name.startswith(HANDS[hand] + "/"):
                return True
    return False


# ---------------------------------------------------------------- the closed loop
corr, C_hold = np.eye(4), np.eye(4)
phi_bit = None                                      # the ratchet's angle at the first insertion
d_ax0 = None                                        # the bolt's advance against the plan's when the first swing starts
H = {}                                              # part pose in the TCP frame, measured once the assist holds it
taken = set()
I = {h: np.zeros(6) for h in "LR"}
q_prev = {h: None for h in "LR"}
cmd_prev = None
qh0 = None
r_mode = "plan"
TR_hold = TR_plan_hold = None
pinch_q = None
stage_prev = None
sched = list(range(len(fr)))
rep = [i for i in sched if fr[i]["stage"].startswith("3 stroke %d/%d" % (N_STROKES - 1, N_STROKES))]
i5 = next(k for k, i in enumerate(sched) if fr[i]["stage"].startswith("5 head"))
extra, n, k, waited, prev_i = 0, -1, 0, 0, sched[0]
S_sw = None
dwell = 0
last_i = -1
seg_out_seen = push_frozen = False
out_count, out_len = 0, 1
guard, pushing_prev = 0, False
lat_ff = np.zeros(3)
land, push_k0, retries, n_retry = None, None, 0, 0
last = dict(tip_err=0.0, tip_lat=0.0, tip_ax=1.0)
ended = False


def planned_turned(Tk0, Tk1, deg):
    return Rot.from_matrix(Tk1[:3, :3] @ Tk0[:3, :3].T).magnitude() > np.radians(deg)


def lever_angle(g):
    """The plan's lever angle about the bore axis at frame g (the key's -x, from up toward the near side)."""
    v = np.linalg.inv(Th)[:3, :3] @ (-g["Tk"][:3, 0])
    return float(np.arctan2(-v[1], v[2]))


i5_lever = [(j, lever_angle(fr[j])) for j in sched[i5:] if fr[j]["stage"].startswith("5 head")]
seated_skip = False


while k < len(sched) and not ended and d.time < STOP_S:
    i = sched[k]
    # the ratchet turns the bolt a few degrees more than 60 per stroke (no draw-out between them): seated before the plan's
    # last stroke, the strokes left would drive the head into the block. At the end of a turn back, seated: go on with
    # the plan's last segment from its frame whose lever angle is nearest (it turns the key back and draws it out)
    if (not seated_skip and k < i5 and fr[i]["stage"].startswith("3 stroke") and "swings the lever" in fr[i]["stage"]
            and fr[sched[k - 1]]["stage"] != fr[i]["stage"] and MECH.engaged
            and d.qpos[MECH.qa["thr_s"]] >= geo.BOLT["L"] - SEAT_TOL):
        a_now = lever_angle(fr[sched[k - 1]])
        j_best = min(i5_lever, key=lambda x: abs(np.arctan2(np.sin(x[1] - a_now), np.cos(x[1] - a_now))))[0]
        note("head seated (%.2f turns) before stroke %s: on to the plan's last segment at its lever angle %.1f deg (here %.1f)"
             % (MECH.turns(d), fr[i]["stage"][9:15], np.degrees(lever_angle(fr[j_best])), np.degrees(a_now)))
        k = sched.index(j_best, i5)
        i5 = k
        seated_skip = True
        i = sched[k]
    if k == i5 and extra < 40 and MECH.engaged and d.qpos[MECH.qa["thr_s"]] < geo.BOLT["L"] - SEAT_TOL:
        sched[k:k] = rep
        i5 += len(rep)
        extra += 1
        note("head not seated yet (%.1f mm to go): one more stroke" % ((geo.BOLT["L"] - d.qpos[MECH.qa["thr_s"]]) * 1e3))
        i = sched[k]
    k += 1
    n += 1
    f = fr[i]
    if pushing_prev and not ("drawn out" in f["stage"] and f["key_in"]) and held("L") == "key":
        end_c, dep_c = MECH.key_end_in_socket(d)
        if land is not None:
            lat_ff = np.clip(lat_ff + ILC_GAIN * land, -0.004, 0.004)
        entered = end_c == "short" and dep_c >= 0.003
        land = None
        if not entered and retries < RETRIES and push_k0 is not None:
            retries += 1
            n_retry += 1
            note("key did not go into the socket (tip %.1f mm in): back off and push again" % (dep_c * 1e3))
            k = max(push_k0 - 3, 0)
            push_frozen = False
            pushing_prev = False
            continue
        retries = 0
    # (the plain keys' oracle waits here while the key is still in the socket at a turn back; the ratchet turns back in it)
    if (held("L") == "key" and f["key_in"] and not fr[i - 1]["key_in"] and dwell < DWELL_MAX and last["tip_err"] > DWELL_TOL):
        k -= 1                      # about to push the key in: let the arm settle on the pre-insertion pose first
        n -= 1
        i = i - 1
        f = fr[i]
        dwell += 1
    elif not (f["key_in"] and not fr[i - 1]["key_in"]):
        dwell = 0
    if pushing_prev and held("L") == "key" and guard < 80 and last["tip_lat"] > GUARD_LAT and last["tip_ax"] < FACE + 0.001:
        k -= 1                      # guarded insertion: back off to where the planned tip is 0.5 mm outside the mouth
        n -= 1
        j = prev_i
        while j > 0 and fr[j - 1]["stage"] == fr[j]["stage"] and tip_plan_ax[j] < FACE + 0.0005:
            j -= 1
        i = j
        f = fr[i]
        guard += 1
    elif not pushing_prev:
        guard = 0
    prev_i = i
    st = f["stage"]
    TL_plan, TR_plan = plan_T[i]
    new_seg = st != stage_prev
    # the ratchet: the key's path is taken from the bolt only at the first insertion; afterwards the tip stays in the
    # recess and the bolt follows the key (a path re-taken from the loose bolt every step would chase its droop)
    approach = st.startswith("2 key turned upright")
    if new_seg:
        report["stages"].append([int(s.n_steps), st[:60]])
        if "drawn out" in st:
            report["strokes"].append(dict(t=round(float(d.time), 2), stage=st[:40], turns=round(MECH.turns(d), 3),
                                          tilt_deg=round(float(np.degrees(MECH.tilt(d))), 2), retries=n_retry))
        if S_sw is not None:
            corr = spin_about(S_sw, -S_sw[0]) @ corr
            S_sw = None
        if "drawn out" in st:
            C_hold = corr
        seg_out_seen = push_frozen = False
        out_count = 0
        out_len = sum(1 for g in fr[i:i + 2000] if g["stage"] == st and not g["key_in"])
    stage_prev = st
    if "drawn out" in st and not f["key_in"]:
        seg_out_seen = True
        if i != last_i:
            out_count += 1
    pushing = "drawn out" in st and f["key_in"] and seg_out_seen
    if approach and MECH.engaged and qh0 is not None and not (pushing and push_frozen):
        ref = [g for g in fr[i:] if g["theta"] is not None]
        th_ref = ref[0]["theta"] if ref else THETA0
        B_act = T_body("bolt")
        C_now = B_act @ np.linalg.inv(bolt_T_plan(Th, th_ref))
        ax = B_act[:3, 0]
        phi = Rot.from_matrix(C_now[:3, :3]).as_rotvec() @ (-ax)
        if st.startswith("2 key turned upright"):      # the first insertion: the bit (L + ratchet) onto the recess, the
            if phi_bit is None:                         # ratchet's angle taken once (a tip touching the mouth turns the
                phi_bit = float(d.qpos[RATCHET_Q])      # bit; following it would turn the L, and the bit again)
            phi_w = (phi - phi_bit + PERIOD / 2) % PERIOD - PERIOD / 2
        else:                                           # the L at the plan's own angle about the bolt axis
            phi_w = 0.0
        Rw = Rot.from_rotvec(-ax * (phi_w - phi)).as_matrix()
        W = np.eye(4)
        W[:3, :3] = Rw
        W[:3, 3] = B_act[:3, 3] - Rw @ B_act[:3, 3]
        C_now = W @ C_now
        if pushing:
            corr = C_now
            push_frozen = True
        else:                                           # (the ratchet: the turn back follows the bolt as it is)
            corr = C_now
    in_bay = st.startswith("2 key turned upright") or st.startswith("3 ") or st.startswith("4 ") or st.startswith("5 head")
    W_sw = np.eye(4)
    if "swings the lever" in st and MECH.engaged and qh0 is not None and held("L") == "key":
        if S_sw is None:
            ths = [g["theta"] for g in fr[i:i + 500] if g["stage"] == st]
            ax_in = -d.xmat[BODY["bolt"]].reshape(3, 3)[:, 0]
            phi0 = Rot.from_matrix(corr[:3, :3]).as_rotvec() @ ax_in
            S_sw = (phi0, ax_in, d.xpos[BODY["bolt"]] - MOUNT, ths[0], ths[-1])
        u = float(np.clip((f["theta"] - S_sw[3]) / (S_sw[4] - S_sw[3]), 0, 1))
        W_sw = spin_about(S_sw, -S_sw[0] * u)

    def part_target(part, T_des, hand):
        if not touching_other(part, hand) and (part != "key" or MECH.key_end_in_socket(d)[0] is None):
            H_now = np.linalg.inv(T_site(hand)) @ T_body(part)
            H[part][:3, 3] = 0.5 * H[part][:3, 3] + 0.5 * H_now[:3, 3]
            H[part][:3, :3] = Slerp([0, 1], Rot.from_matrix([H[part][:3, :3], H_now[:3, :3]]))(0.5).as_matrix()
        return T_des @ np.linalg.inv(H[part])

    if held("L") == "key":
        K_des = W_sw @ corr @ f["Tk"] if in_bay else f["Tk"]
        if in_bay and not st.startswith("2 "):
            # the bolt's advance against the plan's (a stroke turns it the play less than the plan's 60 deg): the key's
            # path moved along the bore axis by the difference, counted from the first insertion's end
            d_ax = MECH._bolt_axis(d)[0] - (np.linalg.inv(Th) @ f["Tb"])[0, 3]
            if d_ax0 is None:
                d_ax0 = d_ax
            K_des = K_des.copy()
            K_des[:3, 3] += (d_ax - d_ax0) * Th[:3, 0]
        if pushing:
            Bb = T_body("bolt")
            tip_b = np.linalg.inv(Bb) @ T_body("key") @ SHORT_TIP
            e_lat = np.r_[0.0, tip_b[1], tip_b[2]]
            if land is None and tip_b[0] <= FACE + 0.0003:
                land = e_lat.copy()
            K_des = K_des.copy()
            K_des[:3, 3] -= Bb[:3, :3] @ lat_ff
            if push_k0 is None or not pushing_prev:
                push_k0 = k - 1
        TL_cmd = part_target("key", K_des, "L")
    else:
        TL_cmd = TL_plan
    if r_mode == "plan" and held("R") == "bolt" and not MECH.engaged:
        TR_cmd = part_target("bolt", f["Tb"], "R")
    elif r_mode in ("pinch", "open") and TR_hold is not None:
        TR_cmd = TR_hold @ np.linalg.inv(TR_plan_hold) @ TR_plan
    else:
        TR_cmd = TR_plan
    held_frame = i == last_i
    target = {}
    for h, T in (("L", TL_cmd), ("R", TR_cmd)):
        q0 = np.asarray(f["q" + h], float)
        if np.allclose(T, plan_T[i][0 if h == "L" else 1], atol=1e-9):
            q_des = q0
        else:
            q_des = RA.ik_near(h, T, q0, q0=q_prev[h])
        q_prev[h] = q_des
        part = held(h)
        payload = None if part is None else (mass[part], d.xipos[BODY[part]].copy())
        qd_ff = np.zeros(6) if held_frame else QD[h][0][i]
        qdd_ff = np.zeros(6) if held_frame else QD[h][1][i]
        tau = A.feedforward(h, q_des, qd_ff, qdd_ff, payload)
        I[h] = np.clip(I[h] + K_I * (q_des - A.q(h)), -I_MAX, I_MAX)
        target[h] = A.target(h, q_des + I[h], tau)
    last_i = i
    gL = GRIP_CLOSED if f["fL"] < Q_OPEN_MIN else f["fL"]
    if r_mode == "plan":
        gR = GRIP_CLOSED if f["fR"] < Q_OPEN_MIN else f["fR"]
    elif r_mode == "pinch":
        gR = pinch_q
        if f["fR"] >= Q_OPEN_MIN:
            r_mode = "open"
            note("right hand opens and leaves (%.2f thread turns engaged)" % MECH.turns(d))
    if r_mode == "open":
        gR = f["fR"]
    cmd = np.r_[target["L"], target["R"]]
    if cmd_prev is not None:
        cmd = cmd_prev + np.clip(cmd - cmd_prev, -CMD_STEP, CMD_STEP)
    cmd_prev = cmd
    a = np.zeros(14)
    a[iL], a[6], a[iR], a[13] = cmd[:6], gL, cmd[6:], gR
    engaged_before = MECH.engaged
    if s.step(a):
        ended = True
    for h in "LR":                                  # a part just taken: its pose in the TCP frame from now on
        part = held(h)
        if part is not None and (h, part) not in taken:
            H[part] = np.linalg.inv(T_site(h)) @ T_body(part)
            taken.add((h, part))
            note("%s hand holds the %s" % (HANDS[h], part))
        if part is None and any(t[0] == h for t in taken):
            taken.difference_update({t for t in taken if t[0] == h})
            note("%s hand lets go" % HANDS[h])
    if MECH.engaged != engaged_before:
        note("thread %s (%.2f turns)" % ("catches" if MECH.engaged else "lets go", MECH.turns(d)))
    if r_mode == "plan" and MECH.engaged:
        # thread caught: loosen to a light pinch (the commanded opening releases the weld; the fingers keep pressing)
        r_mode = "pinch"
        fq = d.qpos[FINGER_Q["R"]]
        pinch_q = float(np.clip(fq - PINCH, GRIP_CLOSED, GRIP_OPEN))
        TR_hold, TR_plan_hold = T_site("R"), TR_plan
        qh0 = d.qpos[MECH.qa["thr_h"]]
        note("right hand loosens to a light pinch (finger target %.4f, at %.4f)" % (pinch_q, fq))
    pushing_prev = pushing
    tip_b_now = np.linalg.inv(T_body("bolt")) @ T_body("key") @ SHORT_TIP
    last["tip_lat"], last["tip_ax"] = float(np.hypot(tip_b_now[1], tip_b_now[2])), float(tip_b_now[0])
    last["tip_err"] = (float(np.linalg.norm((T_body("key") @ SHORT_TIP)[:3] - (TL_cmd @ H["key"] @ SHORT_TIP)[:3]))
                       if held("L") == "key" else 0.0)

hold = 0
while not ended and hold < 400 and d.time < STOP_S:                     # the plan's end: hold until the goal is confirmed
    hold += 1
    a = np.zeros(14)
    a[iL], a[6], a[iR], a[13] = cmd_prev[:6], gL, cmd_prev[6:], gR
    ended = s.step(a)
depth, radial, ang, _ = MECH._bolt_axis(d)
report.update(control_steps=int(s.n_steps), sim_time_s=round(float(d.time), 2), extra_strokes=extra, retries=n_retry,
              seated=bool(MECH.engaged and d.qpos[MECH.qa["thr_s"]] >= geo.BOLT["L"] - SEAT_TOL), turns=MECH.turns(d),
              tip_depth_mm=round(depth * 1e3, 2), axis_deg=round(float(np.degrees(ang)), 3), status=s.status())
print("REPORT " + json.dumps(report, default=float), flush=True)
