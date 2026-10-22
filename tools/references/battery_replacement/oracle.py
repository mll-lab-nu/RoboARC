import numpy as np
# Oracle code for `rcb_topogym.py run-oracle --task battery_replacement --code <this file>` (variation
# battery_replacement_xbox_aa; design runs/controller_battery_design_20261014, README_v3.md sections 4-5). It records the
# -i01 reference. run-oracle runs with the repository as the working directory; the plan (plan.npz, plan.json; written by
# export_plan.py from the design's kinematic oracle, both arms' joints per 40 Hz control step) sits next to this file.
#
# Every control step, per hand (the design's physics replay, scripts/d3_phys.py, as it ran):
#   - the plan's own joints while the hand neither holds a part nor approaches one;
#   - approaching a part ("(down)", "(in)", pinch, grip): the planned approach carried by the part's actual pose change
#     (a cell's turn about its own axis left out; a pried old cell's pinch level and moved along it, PINCH_SHIFT),
#     blended in from the plan's pose over APPROACH_BLEND, solved by IK near the plan's joints;
#   - holding a part: the part's planned path in its own frame from where the hand actually took it (that offset fades
#     out over 1.5 s, so placements end at the planned poses; the cover lifted onto the tongue's way under the lip,
#     tongue_lift), solved by IK near the plan's joints.
# The command is that target plus the drives' feedforward (inverse dynamics of the arm on the commanded motion, Coulomb
# friction, the held part's weight) over their stiffness, a slow integral of the joint error (held while a hand pushes
# the cover's ridge and while the prying hand waits for the other to take the cell) and, while a hand pushes the cover,
# the push as joint torques (J^T F) over the drive's stiffness; each arm
# joint's command moves at most CMD_STEP per control step. The plan waits at gates for the physics (a grip welded, the
# cover slid, the cell pried up RISE_GATE, the pinching hand settled within SETTLE, a new cell pressed down). Only joint
# and gripper targets are commanded; no pose is set. BR_NOISE_RAD / BR_NOISE_SEED add seeded uniform noise to every arm
# joint command (the robustness check); BR_PINCH / BR_COVER write the pinch and cover-closing diagnostics.
import json, os, sys, time
import mujoco
from scipy.spatial.transform import Rotation as R, Slerp

HERE = "tools/references/battery_replacement"
NP = np.load(os.environ.get("BR_PLAN_NPZ", HERE + "/plan.npz"))
META = json.loads(open(os.environ.get("BR_PLAN", HERE + "/plan.json")).read())
STOP = int(os.environ.get("BR_STOP", "1000000"))          # stop after this many control steps (diagnostics)
Q, F, STAGE, HELD, P = NP["Q"], NP["F"], NP["STAGE"], NP["HELD"], NP["P"]
STAGES, PARTS = META["stages"], tuple(META["parts"])
CELLS = ("old_A", "old_B", "new_1", "new_2")
K_I, I_MAX = .15, .05                   # integral gain per control step, its limit (rad)
PUSH_N, PRESS_N = 10., 5.               # opening the cover: push along the slide, press down
PUSH_N_CLOSE, PRESS_N_CLOSE = 10., 1.   # closing: the pad pushes the ridge, presses lightly
WAIT_MAX, WAIT_LONG = 2., 4.            # s: the longest hold of the plan at a gate (pinch and press: WAIT_LONG)
PINCH = .0015                           # grip command this far inside the planned contact value (about 3 N)
CMD_STEP = .035                         # rad: the largest change of an arm joint's command per control step
TAIL = int(os.environ.get("BR_TAIL", "200"))   # control steps held after the plan, until the goal confirms
DEBUG = {int(x) for x in os.environ.get("BR_DEBUG", "").split(",") if x}
LOG = dict(k=[], held=[], parts=[], tcp=[], target=[], cmd=[], q=[]) if os.environ.get("BR_LOG") else None
TGT = {}
# The force-closure pinch of a pried old cell (the margin: README_v3.md, topogym port). The fingertips cannot go below the
# rib's roof (18.3 mm above the compartment floor) and the cell's axis lies only about 15 mm high, so the contacts sit
# about 30-40 degrees above the cell's mid-plane and the central angle between them is 180 minus their sum. Three rules
# keep it well above the force-closure line (90 degrees at mu 1):
#   - the plan leaves the lever only when the cell is up RISE_GATE (the lever lifts it about 8.0-8.3 mm; at the plan's
#     6.5 mm the hand came down on a cell still rising);
#   - the pinch moves PINCH_SHIFT along the cell (horizontally) toward its raised + end: the cell lies tilted about 5.7
#     degrees, so its axis is about 0.1 mm higher per mm (at 12 mm slot B's wall-side finger met the detent plunger,
#     which stands 0.6 mm above the rim beside slot B);
#   - the fingers close only once the TCP is within SETTLE of the pinch pose (the 40 mm descent leaves it up to 10 mm
#     off, swinging back within about 0.5 s);
#   - the pinch follows the cell level (its shift and turn about the vertical, not a tilt beyond the planned one), and
#     the prying hand's integral is held while the other hand takes the cell (it kept turning the stick, tipping the
#     cell to 11 degrees).
# Measured (oracle pinch log BR_PINCH, 2026-10-16): central angle 100.6 / 89.6 degrees (A / B, B's weld 4.6 s late)
# without them; with them 109.0 / 118.0, and 122-131 / 115-119 over nine runs with BR_NOISE_RAD=1e-3, welded within
# 0.15 s every time. At 10-12 mm the two hands' gripper bases met over slot A in some of those runs.
RISE_GATE = .008
PINCH_SHIFT = {"old_A": .008, "old_B": .008}
APPROACH_BLEND = .4                     # s: an approach's re-anchored target blends in from the plan's (no joint step)
SETTLE = .0005
NOISE = float(os.environ.get("BR_NOISE_RAD", "0"))              # robustness check: uniform noise on every arm joint command
NOISE_RNG = np.random.default_rng(int(os.environ.get("BR_NOISE_SEED", "0")))
PINCH_LOG = open(os.environ["BR_PINCH"], "w") if os.environ.get("BR_PINCH") else None   # diagnostics of the old cells' pinch
COVER_LOG = open(os.environ["BR_COVER"], "w") if os.environ.get("BR_COVER") else None   # diagnostics of the cover's closing

s = session
w = s.env._world
m = w.model
task = s.env._task
SIDES = {"L": "left", "R": "right"}
ARM = {"L": list(range(0, 6)), "R": list(range(7, 13))}
GRIP = {"L": 6, "R": 13}
JQ, KP, DAMP, FRIC, LO, HI = {}, {}, {}, {}, {}, {}
for h in "LR":
    names = [s.action_layout[i] for i in ARM[h]]
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    acts = [next(k for k in range(m.nu) if m.actuator_trnid[k, 0] == j) for j in jids]
    JQ[h] = np.array([m.jnt_qposadr[j] for j in jids])
    KP[h] = np.array([m.actuator_gainprm[a, 0] for a in acts])
    DAMP[h] = np.array([m.dof_damping[m.jnt_dofadr[j]] for j in jids]) / KP[h]
    FRIC[h] = np.array([m.dof_frictionloss[m.jnt_dofadr[j]] for j in jids]) / KP[h]
    LO[h] = np.array([m.jnt_range[j, 0] for j in jids])
    HI[h] = np.array([m.jnt_range[j, 1] for j in jids])
DOF = {h: np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s.action_layout[i])] for i in ARM[h]]) for h in "LR"}
SITE = {h: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{SIDES[h]}/gripper") for h in "LR"}
BODY = {p: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{p}::{'cell' if p in CELLS else p}") for p in PARTS}
FINGER = {h: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{SIDES[h]}/left_finger")] for h in "LR"}
d2 = mujoco.MjData(m)                   # scratch data for kinematics (the live data is only read)
# the drives' feedforward: inverse dynamics of the arms alone (a copy of the model without contacts, equalities, limits
# or friction loss; the links are gravity-compensated as in the scene), as the design's physics replay (phys3.Arms)
import copy
m_ff = copy.deepcopy(m)
m_ff.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
d_ff = mujoco.MjData(m_ff)
GRIPPER_BODY = {h: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{SIDES[h]}/gripper_link") for h in "LR"}
FRANGE = {h: np.array([m.actuator_forcerange[next(k for k in range(m.nu) if m.actuator_trnid[k, 0] == mujoco.mj_name2id(
    m, mujoco.mjtObj.mjOBJ_JOINT, s.action_layout[i])), 1] for i in ARM[h]]) for h in "LR"}


def T_of(pos, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = pos
    return T


# the plan's scene frame in the world: the task's origin (the ALOHA mount on the table) x the plan's placement
x0, y0, yaw = META["place"]
TP = np.eye(4)
TP[:3, :3] = R.from_euler("z", yaw, degrees=True).as_matrix()
TP[:3, 3] = (x0, y0, 0.)
W = np.eye(4)
W[:3, 3] = task.origin()
WS = W @ TP
WS_INV = np.linalg.inv(WS)


def plan_part(k, p):
    """A part's planned pose (scene frame) at control step k."""
    v = P[k, PARTS.index(p)]
    return T_of(v[:3], v[3:])


def part_scene(p):
    """A part's actual pose in the scene frame (from the live data, read only)."""
    T = np.eye(4)
    T[:3, :3] = w.data.xmat[BODY[p]].reshape(3, 3)
    T[:3, 3] = w.data.xpos[BODY[p]]
    return WS_INV @ T


def site_world(h):
    T = np.eye(4)
    T[:3, :3] = w.data.site_xmat[SITE[h]].reshape(3, 3)
    T[:3, 3] = w.data.site_xpos[SITE[h]]
    return T


def fk(h, q):
    """The TCP pose (world) and the TCP site's 6 x 6 Jacobian (translation, rotation) of arm h at joints q."""
    d2.qpos[:] = w.data.qpos
    d2.qpos[JQ[h]] = q
    mujoco.mj_kinematics(m, d2)
    mujoco.mj_comPos(m, d2)
    T = np.eye(4)
    T[:3, :3] = d2.site_xmat[SITE[h]].reshape(3, 3)
    T[:3, 3] = d2.site_xpos[SITE[h]]
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d2, jp, jr, SITE[h])
    return T, np.vstack([jp[:, DOF[h]], jr[:, DOF[h]]])


def feedforward(h, q, qd, qdd, payload=None):
    """Joint torques the drive must add for the desired motion: inverse dynamics of the arm (no contacts), Coulomb
    friction in the direction of motion, and a held part's weight (mass, centre of mass in the world)."""
    d_ff.qpos[:] = w.data.qpos
    d_ff.qpos[JQ[h]] = q
    d_ff.qvel[:] = 0
    d_ff.qacc[:] = 0
    d_ff.qvel[DOF[h]] = qd
    d_ff.qacc[DOF[h]] = qdd
    mujoco.mj_inverse(m_ff, d_ff)
    tau = d_ff.qfrc_inverse[DOF[h]].copy() + FRIC[h] * KP[h] * np.tanh(qd / .01)
    if payload is not None:
        mass, com = payload
        jp = np.zeros((3, m.nv))
        mujoco.mj_jac(m_ff, d_ff, jp, None, com, GRIPPER_BODY[h])
        tau += jp[:, DOF[h]].T @ (mass * -m.opt.gravity)
    return tau


def ik_near(h, T, q0, iters=30):
    """Damped least squares from q0 (the plan's joints): it stays on q0's branch."""
    q = np.array(q0, dtype=float)
    for _ in range(iters):
        T0, J = fk(h, q)
        e = np.r_[T[:3, 3] - T0[:3, 3], R.from_matrix(T[:3, :3] @ T0[:3, :3].T).as_rotvec()]
        if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
        q = np.clip(q + np.clip(dq, -.2, .2), LO[h], HI[h])
    return q


def split_roll(E):
    """E (a pose change in a cell's own frame) -> (E without its turn about the cell's axis x, that turn as 3 x 3)."""
    q = R.from_matrix(E[:3, :3]).as_quat()
    tw = np.array([q[0], 0., 0., q[3]])
    n = np.linalg.norm(tw)
    tw = tw / n if n > 1e-9 else np.array([0, 0, 0, 1.])
    Rt = R.from_quat(tw)
    out = E.copy()
    out[:3, :3] = (R.from_quat(q) * Rt.inv()).as_matrix()
    return out, Rt.as_matrix()


def part_offset(p, P_plan, P_act):
    E = np.linalg.inv(P_plan) @ P_act
    return split_roll(E) if p in CELLS else (E, np.eye(3))


def interp_T(A, B, u):
    M = np.eye(4)
    M[:3, 3] = (1 - u) * A[:3, 3] + u * B[:3, 3]
    M[:3, :3] = Slerp([0, 1], R.from_matrix([A[:3, :3], B[:3, :3]]))(u).as_matrix()
    return M


def held_now(st):
    out = {}
    for h in "LR":
        b = st["grasp"]["hands"][SIDES[h]]["binding"]
        out[h] = None if b is None else b["target"][0]
    return out


def held_plan(k, h):
    i = HELD[k, "LR".index(h)]
    return None if i < 0 else PARTS[i]


# the next part each hand will hold, per plan step (for re-anchoring an approach)
NXT = {h: [None] * len(Q) for h in "LR"}
for h in "LR":
    upcoming = None
    for k in range(len(Q) - 1, -1, -1):
        st_ = STAGES[STAGE[k]]
        if held_plan(k, h) is not None:
            upcoming = (held_plan(k, h), k)
        elif "home" in st_ or "makes room" in st_:
            upcoming = None
        NXT[h][k] = upcoming


def seated_C(slot):
    """The slot's seated cell frame in the scene frame (the compartment frame C of the design, slot A's)."""
    from topogym.assets.geometry import battery_replacement as geo
    return np.linalg.inv(TP) @ geo.seated_T(slot)


COVER_CLOSED = None
# After prying B the left hand puts the stick back into its stand leaning LEAN_DEG toward +x (scene), within the hole's
# free play (the 6 mm shaft stands only its lowest 12 mm in the 6.8 mm hole, about 4.5 degrees of lean): put back
# upright it falls to either side, and leaning toward -x its handle stands in the left hand's way to the cover's pad
# (left/gripper_base met it; the upright handle clears it by about 5 mm). The lean turns about the shaft's middle in the
# hole, LEAN_PIVOT_Z above the table, growing as the stick goes in; the hand's way up after letting go keeps it.
# The tongue's way under the lip: the plan lowers the cover tilted 14 degrees and pushes it +y level, which runs the
# tongue's tip up to 2.3 mm into the ledge in front of the lip (kinematic check of the plan's cover poses against the
# shell, 2026-10-16); the cover stuck there in 2 of 12 runs. The clear way is a gap about 0.15 mm high that runs along the
# tongue (down 14 degrees toward +y) to the open position: while lowering and pushing, the hand lifts the cover by
# tan(14 deg) x (5.7 mm - the planned cover's y from closed) + TONGUE_CLEAR, which is the gap's middle.
TONGUE_SLOPE, TONGUE_OPEN_Y, TONGUE_CLEAR = float(np.tan(np.radians(14.))), .0057, .00008
TONGUE_STAGES = ("7 cover: lower, tilted 14 deg", "7 cover: push the tongue under the lip")


def tongue_lift(k):
    """The cover's lift (m, world z) at plan step k, or 0."""
    if STAGES[STAGE[k]] not in TONGUE_STAGES:
        return 0.
    from topogym.assets.geometry import battery_replacement as geo
    D = np.linalg.inv(W @ geo.cover_closed_T()) @ WS @ plan_part(k, "cover")
    return TONGUE_SLOPE * (TONGUE_OPEN_Y - float(D[1, 3])) + TONGUE_CLEAR


# The last push: the pad pushes the cover's ridge -y only until the cover is RELEASE_DY from latched (the detent plunger
# already rising into the groove), then the left hand lets go: off the ridge +y by RETREAT_Y, then up RETREAT_Z, on its
# own clock (s). The plunger's spring pulls the cover the rest of the way (the groove's edge on the plunger's round top),
# so the goal's 0.5 s hold runs with the hand already leaving and the closed cover in view.
RELEASE_DY, RETREAT_Y, RETREAT_Z = .0009, .0015, .06
RETREAT_T = (.15, .08, .5)              # s: the move off the ridge; the lift's start and duration


def retreat_target(T0, tau):
    """The left hand's retreat pose tau s after letting go of the ridge (world)."""
    u = float(np.clip((tau - RETREAT_T[1]) / RETREAT_T[2], 0., 1.))
    T = T0.copy()
    T[1, 3] += RETREAT_Y * min(1., tau / RETREAT_T[0])
    T[2, 3] += RETREAT_Z * u * u * (3. - 2. * u)
    return T


LEAN_DEG, LEAN_PIVOT_Z = 2.5, .038
LEAN_STAGES = ("4 pry B: into the stand", "4 pry B: let go of the stick", "4 pry B: L hand home (up)")


def stick_lean(k):
    """The world transform that leans the planned stick (and the hand holding it) at step k, or None."""
    stage = STAGES[STAGE[k]]
    if stage not in LEAN_STAGES:
        return None
    v = P[k, PARTS.index("stick")]
    u = 1. if stage != LEAN_STAGES[0] else float(np.clip((.045 - v[2]) / .04, 0., 1.))
    pivot = (WS @ np.r_[v[0], v[1], LEAN_PIVOT_Z, 1.])[:3]
    Rl = WS[:3, :3] @ R.from_euler("y", LEAN_DEG * u, degrees=True).as_matrix() @ WS[:3, :3].T
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = Rl, pivot - Rl @ pivot
    return M


def gate(k, st):
    """Whether the plan may leave step k (only at the end of a segment), and why not."""
    stage, nxt = STAGES[STAGE[k]], STAGES[STAGE[min(k + 1, len(Q) - 1)]]
    if stage == nxt:
        return True, ""
    held = held_now(st)
    for h in "LR":
        p = held_plan(k, h)
        if p is not None and ("pinch" in stage or "grip" in stage) and held[h] != p:
            return False, f"{h} hand waits for the weld on the {p}"
    from topogym.assets.geometry import battery_replacement as geo
    cov = part_scene("cover")
    closed = np.linalg.inv(TP) @ geo.cover_closed_T()
    dy = (cov[1, 3] - closed[1, 3]) * 1000
    if "slide +5.5" in stage and dy < 5.:
        return False, f"cover slid only {dy:.1f} mm"
    if ("slide -y" in stage or "push the ridge -y" in stage) and abs(dy) > .5:
        return False, f"cover still {dy:.1f} mm from latched"
    if "down to the raised cell (in)" in stage or "down beside the cell middle (in)" in stage:
        h = "L" if "pry A" in stage else "R"
        err = float(np.linalg.norm(site_world(h)[:3, 3] - TGT[h][:3, 3]))
        if err > SETTLE:
            return False, f"{h} hand {err * 1000:.2f} mm from the pinch pose"
    if ": lever " in stage:
        cell = "old_A" if "pry A" in stage else "old_B"
        rise = part_scene(cell)[2, 3] - seated_C("A")[2, 3]
        if rise < RISE_GATE:
            return False, f"{cell} raised only {rise * 1000:.1f} mm"
    if "press the + end down" in stage:
        slot = "A" if "into A" in stage else "B"
        seat = seated_C(slot)
        cell = min(("new_1", "new_2"), key=lambda c_: float(np.linalg.norm(plan_part(k, c_)[:3, 3] - seat[:3, 3])))   # v4: by the plan
        M = np.linalg.inv(seat) @ part_scene(cell)
        tilt = np.degrees(np.arcsin(min(1., abs(M[2, 0]))))
        if M[2, 3] * 1000 > .8 or tilt > 2.:
            return False, f"{cell} not down yet ({M[2, 3] * 1000:.1f} mm high, tilt {tilt:.1f} deg)"
    return True, ""


state = dict(I={h: np.zeros(6) for h in "LR"}, cmd=None, hist={h: [] for h in "LR"},
             last_g={h: float(F[0, "LR".index(h)]) for h in "LR"}, prev_held={h: False for h in "LR"},
             take={h: None for h in "LR"}, appr={h: None for h in "LR"}, retreat=None, ended=False)
log = dict(waits=[], takes=[], steps=0)


def control(k):
    """One control step at plan step k; returns True when the episode ended."""
    st = w.read_state()
    stage = STAGES[STAGE[k]]
    held = held_now(st)
    q_act = np.asarray(s.joint_positions(), dtype=float)
    # takes: a new binding records the part in the TCP frame and the actual-vs-planned offset
    for h in "LR":
        p = held[h]
        if p is not None and (state["take"][h] is None or state["take"][h][0] != p):
            H = np.linalg.inv(site_world(h)) @ (WS @ part_scene(p))
            kk = min(len(Q) - 1, k + 1)
            state["take"][h] = (p, H, part_offset(p, plan_part(kk, p), part_scene(p)), float(st["sim_time_s"]))
            log["takes"].append(dict(step=log["steps"], hand=h, part=p, stage=stage))
        if p is None:
            state["take"][h] = None
    if state["retreat"] is None and stage.startswith("7 cover: push the ridge"):
        from topogym.assets.geometry import battery_replacement as geo
        dy = (part_scene("cover")[1, 3] - (np.linalg.inv(TP) @ geo.cover_closed_T())[1, 3])
        if dy <= RELEASE_DY:
            state["retreat"] = (site_world("L").copy(), float(st["sim_time_s"]), state["hist"]["L"][-1].copy())
            log["retreat"] = dict(step=log["steps"], dy_mm=round(float(dy) * 1000, 3), t=float(st["sim_time_s"]))
    cmd = np.zeros(14)
    for h in "LR":
        qo = Q[k, 0:6].copy() if h == "L" else Q[k, 6:12].copy()
        T_plan_w, _ = fk(h, qo)
        T_plan = WS_INV @ T_plan_w
        planned = held_plan(k, h)
        target, approach = None, False
        if held[h] is not None and held[h] == planned and state["take"][h] is not None and state["take"][h][0] == held[h]:
            p, H, (Es, Rt), tk = state["take"][h]
            u = min(1., (float(st["sim_time_s"]) - tk) / 1.5)
            E = interp_T(Es, np.eye(4), u)
            E[:3, :3] = E[:3, :3] @ Rt
            target = WS @ plan_part(k, p) @ E @ np.linalg.inv(H)
            if p == "cover":
                target[2, 3] += tongue_lift(k)
        elif NXT[h][k] is not None and (planned is None or held[h] is None) and \
                any(x in stage for x in ("(down)", "(in)", "pinch", "grip")):
            p, kg = NXT[h][k]
            approach = True
            Pg = plan_part(kg, p)
            Es, _ = part_offset(p, Pg, part_scene(p))
            target = WS @ Pg @ Es @ np.linalg.inv(Pg) @ T_plan
            if p in ("old_A", "old_B") and "raised cell" in stage:
                # level: the planned pinch carried by the cell's shift and its turn about the vertical only (a cell
                # tilted further than planned would tilt the fingers onto the rib's roof), moved along the cell
                Pa = part_scene(p)
                yaw = lambda T_: np.arctan2(T_[1, 0], T_[0, 0])
                M = np.eye(4)
                M[:3, :3] = R.from_euler("z", yaw(Pa) - yaw(Pg)).as_matrix() @ Pg[:3, :3]
                M[:3, 3] = Pa[:3, 3]
                target = WS @ M @ np.linalg.inv(Pg) @ T_plan
                along = (WS @ Pa)[:3, 0] * np.array([1., 1., 0.])     # the cell's + direction, level
                target[:3, 3] += along / np.linalg.norm(along) * PINCH_SHIFT[p]
        lean = stick_lean(k) if h == "L" else None
        if lean is not None:
            target = lean @ (T_plan_w if target is None else target)
        if approach:
            if state["appr"][h] is None or state["appr"][h][0] != NXT[h][k][0]:     # per part (kg moves once it is held)
                state["appr"][h] = (NXT[h][k][0], float(st["sim_time_s"]))
            u = min(1., (float(st["sim_time_s"]) - state["appr"][h][1]) / APPROACH_BLEND)
            target = interp_T(T_plan_w, target, u * u * (3. - 2. * u))
        else:
            state["appr"][h] = None
        retreat = h == "L" and state["retreat"] is not None
        if retreat:                                  # the plan's left hand is no longer followed
            T0, t0, q0 = state["retreat"]
            target = retreat_target(T0, float(st["sim_time_s"]) - t0)
            qo = state["hist"]["L"][-1].copy() if state["hist"]["L"] else q0
        q_des = qo if target is None else ik_near(h, target, qo)
        TGT[h] = T_plan_w if target is None else target
        if log["steps"] + 1 in DEBUG:
            Tq, _ = fk(h, q_des)
            print(json.dumps(dict(step=log["steps"] + 1, hand=h, ik_res_mm=round(float(np.linalg.norm(Tq[:3, 3] - TGT[h][:3, 3])) * 1000, 3),
                                  q_plan=np.round(qo, 4).tolist(), q_des=np.round(q_des, 4).tolist(), q_act=np.round(q_act[ARM[h]], 4).tolist(),
                                  I=np.round(state["I"][h], 4).tolist())), file=sys.stderr, flush=True)
        hist = state["hist"][h]
        hist.append(q_des.copy())
        del hist[:-3]
        qd = (hist[-1] - hist[-2]) * s.control_hz if len(hist) >= 2 else np.zeros(6)
        qdd = (hist[-1] - 2 * hist[-2] + hist[-3]) * s.control_hz ** 2 if len(hist) >= 3 else np.zeros(6)
        payload = None
        if held[h] is not None:
            payload = (float(m.body_subtreemass[BODY[held[h]]]), w.data.xipos[BODY[held[h]]].copy())
        tau = feedforward(h, q_des, qd, np.clip(qdd, -20., 20.), payload)
        off = np.clip(tau, -FRANGE[h], FRANGE[h]) / KP[h]
        if log["steps"] + 1 in DEBUG:
            print(json.dumps(dict(step=log["steps"] + 1, hand=h, stage=stage[:40], dq_ik=np.round(q_des - qo, 4).tolist(),
                                  qdd=np.round(qdd, 2).tolist(), off=np.round(off, 4).tolist(),
                                  err_mm=np.round((site_world(h)[:3, 3] - TGT[h][:3, 3]) * 1000, 2).tolist())), file=sys.stderr, flush=True)
        push = None
        if ("slide" in stage or stage == "7 cover: open") and planned == "cover":
            sy = 1. if stage.startswith("1 cover") else -1.
            push = np.array([0., sy * (PUSH_N if sy > 0 else PUSH_N_CLOSE), -(PRESS_N if sy > 0 else PRESS_N_CLOSE)])
        if stage.startswith("7 cover: push the ridge") and h == "L" and not retreat:
            push = np.array([0., -PUSH_N_CLOSE, -PRESS_N_CLOSE])
        if push is not None:
            _, J = fk(h, q_des)
            off = off + np.clip(J[:3].T @ (WS[:3, :3] @ push), -12.9, 12.9) / KP[h]
        if stage.startswith("7 cover: push the ridge") and h == "L":
            pass                                     # the integral is held during the pad push
        elif planned == "stick" and ("raised cell" in stage or "weld confirmed" in stage):
            pass                                     # and while the other hand takes the pried cell: the stick holds
        elif stage == "7 cover: open" and planned == "cover":
            state["I"][h] = state["I"][h] * .85
        else:
            state["I"][h] = np.clip(state["I"][h] + K_I * (q_des - q_act[ARM[h]]), -I_MAX, I_MAX)
        cmd[ARM[h]] = np.clip(q_des + state["I"][h] + off, LO[h], HI[h])
        # gripper: holding in the plan -> a pinch inside the planned contact value, never wider while the plan holds
        f = float(F[k, "LR".index(h)])
        if planned is None:
            g = f
        else:
            g = min(state["last_g"][h], f - PINCH) if state["prev_held"][h] else f - PINCH
        state["prev_held"][h] = planned is not None
        state["last_g"][h] = g
        cmd[GRIP[h]] = g
    if NOISE:
        for h in "LR":
            cmd[ARM[h]] += NOISE_RNG.uniform(-NOISE, NOISE, 6)
    if state["cmd"] is not None:                     # at most CMD_STEP per arm joint per control step
        for h in "LR":
            cmd[ARM[h]] = state["cmd"][ARM[h]] + np.clip(cmd[ARM[h]] - state["cmd"][ARM[h]], -CMD_STEP, CMD_STEP)
    cmd = cmd.astype(np.float32).astype(np.float64)  # the actions go out as float32: keep the step within CMD_STEP there
    if state["cmd"] is not None:
        arms = np.r_[ARM["L"], ARM["R"]]
        prev, now = state["cmd"][arms], cmd[arms]
        over = np.abs(now - prev) > CMD_STEP
        now[over] = np.nextafter(now[over].astype(np.float32), prev[over].astype(np.float32)).astype(np.float64)
        cmd[arms] = now
    state["cmd"] = cmd.copy()
    log["steps"] += 1
    if LOG is not None:
        LOG["k"].append(k)
        LOG["held"].append([PARTS.index(held[h]) if held[h] else -1 for h in "LR"])
        LOG["parts"].append([part_scene(p)[:3, 3] for p in PARTS])
        LOG["tcp"].append([site_world(h)[:3, 3] for h in "LR"])
        LOG["target"].append([TGT[h][:3, 3] for h in "LR"])
        LOG["cmd"].append(cmd.copy())
        LOG["q"].append(q_act.copy())
    if log["steps"] in DEBUG:
        for h in "LR":
            hs = st["grasp"]["hands"][SIDES[h]]
            T_plan_w, _ = fk(h, Q[k, 0:6] if h == "L" else Q[k, 6:12])
            print(json.dumps(dict(step=log["steps"], k=int(k), stage=stage, hand=h, tcp_err_mm=round(float(np.linalg.norm(site_world(h)[:3, 3] - T_plan_w[:3, 3])) * 1000, 2),
                                  finger=round(float(w.data.qpos[FINGER[h]]), 5), grip_cmd=round(float(cmd[GRIP[h]]), 5),
                                  phase=hs["phase"], intent=hs["intent"], candidate=hs["candidate"], rejection=hs["rejection"],
                                  binding=None if hs["binding"] is None else hs["binding"]["target"][0]), default=str), file=sys.stderr, flush=True)
            c = st["contacts"]
            fl = {i for i in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(SIDES[h] + "/")}
            g_ = getattr(s, "table_guard", None)
            if g_ is not None and g_.points:
                arm_ = SIDES[h]
                low_cmd = g_.lowest(cmd).get(arm_)
                q_only = cmd.copy()
                q_only[ARM[h]] = np.asarray(s.joint_positions(), dtype=float)[ARM[h]]
                low_act = g_.lowest(q_only).get(arm_)
                print(f"    guard {arm_}: lowest of cmd {(low_cmd - g_.table) * 1000:.1f} mm, of actual q {(low_act - g_.table) * 1000:.1f} mm; "
                      f"cmd-q {np.round(cmd[ARM[h]] - q_only[ARM[h]], 4).tolist()}", file=sys.stderr, flush=True)
            from topogym.assets.geometry import battery_replacement as geo
            C0 = W @ geo.compartment_T()
            for a_, b_, pen, frc, pos in zip(c["link_a"], c["link_b"], c["penetration"], c["force"], c["position"]):
                if int(a_) in fl or int(b_) in fl:
                    pc = (np.linalg.inv(C0) @ np.r_[pos, 1])[:3] * 1000
                    print("   ", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(a_)), "-", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b_)),
                          f"pen {pen * 1000:.3f} mm f {np.linalg.norm(frc):.2f} N at C {np.round(pc, 2)}", file=sys.stderr, flush=True)
            if h == "L":
                St = part_scene("stick")
                print("    stick at", np.round(St[:3, 3] * 1000, 2), "axis", np.round(St[:3, 2], 4), file=sys.stderr, flush=True)
            for p_ in ("old_A", "old_B"):
                pc = (np.linalg.inv(C0) @ WS @ part_scene(p_))[:3, 3] * 1000
                print("    cell", p_, "at C", np.round(pc, 2), file=sys.stderr, flush=True)
            tc = (np.linalg.inv(C0) @ site_world(h))[:3, 3] * 1000
            print("    tcp", h, "at C", np.round(tc, 2), "target at C", np.round((np.linalg.inv(C0) @ TGT[h])[:3, 3] * 1000, 2), file=sys.stderr, flush=True)
            if NXT[h][k] is not None:
                p_, kg_ = NXT[h][k]
                E_ = np.linalg.inv(plan_part(kg_, p_)) @ part_scene(p_)
                print("    next part", p_, "offset mm", np.round(E_[:3, 3] * 1000, 2), "rot deg", np.round(np.degrees(R.from_matrix(E_[:3, :3]).as_rotvec()), 2), file=sys.stderr, flush=True)
    if PINCH_LOG is not None and ("raised cell" in stage or "weld confirmed" in stage or ": lever " in stage):
        pinch_record(k, stage, st)
    if COVER_LOG is not None and stage.startswith("7 cover") and "parked" not in stage:
        from topogym.assets.geometry import battery_replacement as geo
        D = np.linalg.inv(W @ geo.cover_closed_T()) @ (WS @ part_scene("cover"))
        hs = st["grasp"]["hands"]["left"]
        forces, shell_pts = [], []
        for i in range(int(w.data.ncon)):
            con = w.data.contact[i]
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            if BODY["cover"] in (b1, b2):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(m, w.data, i, f6)
                other = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2 if b1 == BODY["cover"] else b1)
                forces.append((other.split("::")[-1], round(float(f6[0]), 2)))
                if other.endswith("shell") and f6[0] > .5:
                    pc = (np.linalg.inv(W @ geo.compartment_T()) @ np.r_[con.pos, 1])[:3] * 1000
                    gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, con.geom1 if b1 == BODY["cover"] else con.geom2) or "?"
                    shell_pts.append((gname.split("::")[-1], np.round(pc, 1).tolist(), round(float(f6[0]), 1)))
        agg = {}
        for o, f in forces:
            agg[o] = round(agg.get(o, 0.) + f, 2)
        COVER_LOG.write(json.dumps(dict(step=log["steps"], k=int(k), stage=stage, d_mm=np.round(D[:3, 3] * 1000, 2).tolist(),
                                        rot_deg=np.round(np.degrees(R.from_matrix(D[:3, :3]).as_rotvec()), 2).tolist(),
                                        binding=None if hs["binding"] is None else hs["binding"]["target"][0], phase=hs["phase"],
                                        err_mm=np.round((site_world("L")[:3, 3] - TGT["L"][:3, 3]) * 1000, 2).tolist(),
                                        contacts=agg, shell_pts=shell_pts[:6])) + "\n")
        COVER_LOG.flush()
    return bool(s.step(cmd))


def pinch_record(k, stage, st):
    """One JSON line on the pried cell and the other hand's fingers: the cell's rise and tilt, the TCP in the cell's
    frame, every finger contact (on the cell: its elevation above the cell's horizontal mid-plane), the force-closure
    test of the finger-cell contacts as the grasp assist makes it, the hand's grasp state."""
    from topogym.utils.grasp import force_closure_pair
    from topogym.assets.geometry import battery_replacement as geo
    cell, slot, h = ("old_A", "A", "L") if "pry A" in stage else ("old_B", "B", "R")
    side = SIDES[h]
    c = w.data.xpos[BODY[cell]].copy()
    Rc = w.data.xmat[BODY[cell]].reshape(3, 3).copy()
    ax = Rc[:, 0]
    seat = W @ geo.seated_T(slot)
    zp = np.array([0., 0., 1.]) - ax[2] * ax
    zp /= np.linalg.norm(zp)
    fingers = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}/{f}_finger_link"): f for f in ("left", "right")}
    rows, env = [], []
    for i in range(int(w.data.ncon)):
        con = w.data.contact[i]
        if con.exclude:
            continue
        b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
        for fb, other, sign in ((b1, b2, 1.), (b2, b1, -1.)):
            if fb not in fingers:
                continue
            n_into = sign * np.asarray(con.frame[:3])          # from the finger's geom toward the other one
            if other == BODY[cell]:
                v = np.asarray(con.pos) - c
                vp = v - (v @ ax) * ax
                rows.append(dict(finger=fingers[fb], x_mm=round(float(v @ ax) * 1000, 2),
                                 elev_deg=round(float(np.degrees(np.arcsin(np.clip(vp @ zp / np.linalg.norm(vp), -1, 1)))), 1),
                                 dist_mm=round(float(con.dist) * 1000, 3), pos=np.asarray(con.pos).tolist(), n=n_into.tolist()))
            elif other not in fingers:
                env.append(dict(finger=fingers[fb], body=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other),
                                dist_mm=round(float(con.dist) * 1000, 3),
                                z_C_mm=round(float((np.linalg.inv(W @ geo.compartment_T()) @ np.r_[con.pos, 1])[2]) * 1000, 2)))
    # every other contact of the pinching hand's links, and the pried cell's contacts with the stick and the shell
    others = []
    hand_bodies = {i for i in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(side + "/")}
    stick_b = BODY["stick"]
    for i in range(int(w.data.ncon)):
        con = w.data.contact[i]
        if con.exclude:
            continue
        b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
        pair = {b1, b2}
        finger_cell = bool(pair & set(fingers)) and pair - hand_bodies == {BODY[cell]}
        if (pair & hand_bodies and not pair <= hand_bodies and not finger_cell) or pair == {BODY[cell], stick_b}:
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, w.data, i, f6)
            others.append(dict(a=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1), b=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2),
                               dist_mm=round(float(con.dist) * 1000, 3), f_n=round(float(f6[0]), 2)))
    by = {f: [r for r in rows if r["finger"] == f] for f in ("left", "right")}
    fc = any(force_closure_pair(a_["pos"], a_["n"], b_["pos"], b_["n"], 1.) for a_ in by["left"] for b_ in by["right"])
    central = None
    if by["left"] and by["right"]:
        central = round(max(180. - a_["elev_deg"] - b_["elev_deg"] for a_ in by["left"] for b_ in by["right"]), 1)
    T_cell = np.r_[np.c_[Rc, c], [[0, 0, 0, 1]]]
    T_tcp = np.linalg.inv(T_cell) @ site_world(h)
    err = (site_world(h)[:3, 3] - TGT[h][:3, 3]) * 1000
    hs = st["grasp"]["hands"][side]
    rec = dict(step=log["steps"], k=int(k), t=round(float(st["sim_time_s"]), 4), stage=stage,
               rise_mm=round(float(c[2] - seat[2, 3]) * 1000, 2), tilt_deg=round(float(np.degrees(np.arcsin(abs(ax[2])))), 2),
               cell_C_mm=np.round((np.linalg.inv(W @ geo.compartment_T()) @ np.r_[c, 1])[:3] * 1000, 2).tolist(),
               tcp_in_cell_mm=np.round(T_tcp[:3, 3] * 1000, 2).tolist(), tcp_err_mm=np.round(err, 2).tolist(), finger_q=round(float(w.data.qpos[FINGER[h]]), 5),
               contacts=[{k_: v for k_, v in r.items() if k_ not in ("pos", "n")} for r in rows], env=env, others=others, fc=fc, central_deg=central,
               other_hand_err_mm=np.round((site_world("LR"[h == "L"])[:3, 3] - TGT["LR"[h == "L"]][:3, 3]) * 1000, 2).tolist(),
               phase=hs["phase"], candidate=hs["candidate"], rejection=hs["rejection"],
               binding=None if hs["binding"] is None else hs["binding"]["target"][0])
    PINCH_LOG.write(json.dumps(rec, default=str) + "\n")
    PINCH_LOG.flush()


t0 = time.time()
state["cmd"] = np.asarray(s.hold_action(), dtype=np.float32).astype(np.float64)
# the birth targets hold the arms at home with the fingers at 8.4 mm: open them to the plan's first value over 10 steps
for i in range(10):
    a = state["cmd"].copy()
    for h in "LR":
        a[GRIP[h]] = a[GRIP[h]] + (F[0, "LR".index(h)] - a[GRIP[h]]) * (i + 1) / 10
    state["cmd"] = a
    if s.step(a):
        state["ended"] = True
        break
k, waiting = 0, None
while not state["ended"] and k < len(Q) and log["steps"] < STOP:
    if control(k):
        state["ended"] = True
        break
    ok, why = gate(k, w.read_state())
    now = float(w.read_state()["sim_time_s"])
    if ok:
        if waiting is not None:
            log["waits"].append((STAGES[STAGE[k]], round(now - waiting, 3), "met"))
        waiting = None
        k += 1
    elif waiting is None:
        waiting = now
    elif now - waiting > (WAIT_LONG if ("pinch" in STAGES[STAGE[k]] or "press the + end" in STAGES[STAGE[k]]) else WAIT_MAX):
        log["waits"].append((STAGES[STAGE[k]], round(now - waiting, 3), "gave up: " + why))
        waiting = None
        k += 1
for _ in range(TAIL):
    if state["ended"]:
        break
    if s.step(state["cmd"]):
        state["ended"] = True
if LOG is not None:
    np.savez_compressed(os.environ["BR_LOG"], **{key: np.asarray(v) for key, v in LOG.items()})
res = task.result
_st = w.read_state()
print(json.dumps({p: [round(float(np.linalg.norm(_st["bodies"][p]["linear_velocity_m_s"])) * 1000, 3),
                      round(float(np.linalg.norm(_st["bodies"][p]["angular_velocity_rad_s"])), 4)] for p in PARTS}), file=sys.stderr, flush=True)
print(json.dumps(dict(success=bool(res.task_success), reason=res.reason, metrics={key: v for key, v in res.metrics.items()
      if key in ("swapped", "cover_latched", "recycled", "quiet", "slot_A", "slot_B", "cover_offset_m", "cover_tilt_rad",
                 "plunger_m", "stable_time_s")}), default=str), file=sys.stderr, flush=True)
print(json.dumps(dict(plan_steps=k, control_steps=log["steps"], waits=log["waits"], takes=log["takes"], retreat=log.get("retreat"),
                      wall_s=round(time.time() - t0, 1))), file=sys.stderr, flush=True)
