#!/usr/bin/env python3
"""A faster reference from a demo by a gated time-warp search on the final physics (true-state feedback).

The demo is cut into segments of SEG demo rows. Segment by segment, each candidate speed factor k (e.g. 4, 3, 2, 1.5, 1)
is tried from the current (already warped) state in a forked child: the segment's ctrl rows are resampled at k demo
rows per control tick (linear interpolation of every actuator; the segment's last demo row is always the last target),
executed through rcb_ikea.Session.step (the joint-speed clamp in), and the resulting state is compared with the demo's
own state at the segment's end (the demo track from profile.py). The fastest k whose state passes the gate is taken; the
parent then executes that candidate itself and records the EXECUTED targets (data.ctrl after the clamp), so the clamp
never changes a recorded row on replay. If no k passes, the search backtracks one segment (up to --backtrack deep) and
takes a slower factor there.

Gate at a segment end (all of them): no breakdown; the same retention welds active as in the demo; every part not
touched by the robot within --tol-pos / --tol-rot of its demo pose (world); every part the robot touches within
--tol-held-pos / --tol-held-rot (world) and within --tol-pos / --tol-rot of the demo's pose relative to the nearer hand.
The last --tail-s seconds of the demo (release and hold) are kept at k = 1. The result is verified by a fresh replay
(probe.py) and the grader.

usage: warp_ref.py TASK DEMO.npz TRACK.npz OUT.npz [--ks 4,3,2,1.5,1] [--seg 400] ...
"""
import argparse, json, os, pickle, sys, time
sys.path.insert(0, os.environ["RCB_SHARED"])
import numpy as np, mujoco
import rcb_ikea as R

ap = argparse.ArgumentParser()
ap.add_argument("task"); ap.add_argument("demo"); ap.add_argument("track"); ap.add_argument("out")
ap.add_argument("--ks", default="4,3,2,1.5,1")
ap.add_argument("--seg", type=int, default=400)
ap.add_argument("--tol-pos", type=float, default=0.001)
ap.add_argument("--tol-rot", type=float, default=1.0)
ap.add_argument("--tol-held-pos", type=float, default=0.008)
ap.add_argument("--tol-held-rot", type=float, default=5.0)
ap.add_argument("--tol-rel-pos", type=float, default=0.002)   # a held part relative to its hand
ap.add_argument("--tol-rel-rot", type=float, default=3.0)
ap.add_argument("--weld-slack", type=int, default=1)         # a weld may latch up to this many segments off the demo
ap.add_argument("--tail-s", type=float, default=4.0)
ap.add_argument("--backtrack", type=int, default=4)
ap.add_argument("--hands", default="left_hand,right_hand")
ap.add_argument("--log", default=None)
a = ap.parse_args()
KS = [float(x) for x in a.ks.split(",")]
assert KS[-1] == 1.0

s = R.Session(a.task)
m, d = s.model, s.data
D = R.load_trajectory(a.demo, s.nu, s.timestep)
T = np.load(a.track)
B = int(T["block"]); TQ = T["qpos"]; TEQ = T["eq"]
assert a.seg % B == 0, "segments must end on track blocks"
N = len(D)

# parts: the free root bodies that are not the robot's
robot = set(R.robot_bodies(m).tolist())
parts = []
for j in range(m.njnt):
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        b = int(m.jnt_bodyid[j])
        if b not in robot:
            parts.append((m.body(b).name, b, int(m.jnt_qposadr[j])))
hands = [m.body(h).id for h in a.hands.split(",")]
# axisymmetric parts (the screws): only their axis counts, a spin about it is no error
AXIS = {o["part"]: np.array(o["part_axis"], float) for o in s.S.task.get("operations", []) if o.get("mode") == "screw"}
weld_ids = [i for i in range(m.neq) if m.eq_type[i] == mujoco.mjtEq.mjEQ_WELD]
scratch = mujoco.MjData(m)


def demo_poses(row_end):
    """part and hand world poses of the demo at the end of its row row_end (a multiple of B)."""
    k = row_end // B - 1
    scratch.qpos[:] = TQ[k]
    mujoco.mj_kinematics(m, scratch)
    P = {n: (scratch.xpos[b].copy(), scratch.xquat[b].copy()) for n, b, _ in parts}
    H = [(scratch.xpos[h].copy(), scratch.xquat[h].copy()) for h in hands]
    return P, H, TEQ[k]


def qerr_deg(q1, q2, axis=None):
    if axis is None:
        return float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q1, q2)))))))
    v1 = np.zeros(3); v2 = np.zeros(3)
    mujoco.mju_rotVecQuat(v1, axis, q1); mujoco.mju_rotVecQuat(v2, axis, q2)
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1))))


def rel(p, q, hp, hq):
    qi = np.zeros(4); mujoco.mju_negQuat(qi, hq)
    dp = np.zeros(3); mujoco.mju_rotVecQuat(dp, p - hp, qi)
    dq = np.zeros(4); mujoco.mju_mulQuat(dq, qi, q)
    return dp, dq


def gate(row_end):
    if row_end % B:
        return dict(ok=s.validity.breakdown is None, why=[] if s.validity.breakdown is None else ['breakdown'], score=0.)
    P, H, eq = demo_poses(row_end)
    out = dict(ok=True, why=[])
    if s.validity.breakdown is not None:
        out["ok"] = False; out["why"].append(f"breakdown {s.validity.breakdown}")
    weq = np.array([d.eq_active[i] for i in weld_ids]).astype(bool); deq = np.array([eq[i] for i in weld_ids]).astype(bool)
    kb = row_end // B - 1; sl = a.weld_slack * a.seg // B
    lo = np.array([TEQ[max(0, kb - sl)][i] for i in weld_ids]).astype(bool)      # welds the demo had a slack earlier
    hi = np.array([TEQ[min(len(TEQ) - 1, kb + sl)][i] for i in weld_ids]).astype(bool)  # ... and a slack later
    if not ((weq | ~lo).all() and (~weq | hi).all()):
        out["ok"] = False; out["why"].append(f"welds {weq.tolist()} vs demo {deq.tolist()}")
    touched = {x for pair in s.validity.robot_part_contacts() for x in pair}
    worst = 0.
    for n, b, _ in parts:
        p, q = d.xpos[b], d.xquat[b]
        ax = AXIS.get(n)
        dp_w = float(np.linalg.norm(p - P[n][0])); dr_w = qerr_deg(q, P[n][1], ax)
        if n in touched:
            # relative to the nearer hand, demo vs now
            hi = int(np.argmin([np.linalg.norm(p - d.xpos[h]) for h in hands]))
            r1 = rel(p, q, d.xpos[hands[hi]], d.xquat[hands[hi]]); r0 = rel(P[n][0], P[n][1], *H[hi])
            dp_r = float(np.linalg.norm(r1[0] - r0[0])); dr_r = qerr_deg(r1[1], r0[1], ax)
            bad = dp_w > a.tol_held_pos or dr_w > a.tol_held_rot or dp_r > a.tol_rel_pos or dr_r > a.tol_rel_rot
            score = max(dp_w / a.tol_held_pos, dr_w / a.tol_held_rot, dp_r / a.tol_rel_pos, dr_r / a.tol_rel_rot)
            if bad:
                out["ok"] = False; out["why"].append(f"{n} held: world {1e3*dp_w:.2f}mm {dr_w:.2f}deg rel {1e3*dp_r:.2f}mm {dr_r:.2f}deg")
        else:
            bad = dp_w > a.tol_pos or dr_w > a.tol_rot
            score = max(dp_w / a.tol_pos, dr_w / a.tol_rot)
            if bad:
                out["ok"] = False; out["why"].append(f"{n}: {1e3*dp_w:.2f}mm {dr_w:.2f}deg")
        worst = max(worst, score)
    out["score"] = round(worst, 3)
    return out


def interp(tau):
    """ctrl at fractional demo row index tau (0-based), linear."""
    i0 = int(np.floor(tau)); f = tau - i0
    if f < 1e-12 or i0 + 1 >= N:
        return D[min(i0, N - 1)].copy()
    return (1 - f) * D[i0] + f * D[i0 + 1]


def seg_rows(r0, r1, k):
    if k == 1.0:
        return D[r0:r1].copy()
    m_ = int(np.ceil((r1 - r0) / k))
    rows = [interp(r0 - 1 + j * k) if r0 - 1 + j * k < r1 - 1 else D[r1 - 1].copy() for j in range(1, m_ + 1)]
    rows[-1] = D[r1 - 1].copy()
    return np.array(rows)


def execute(rows):
    rec = []
    for row in rows:
        s.step(row)
        rec.append(d.ctrl.copy())
    return rec


def trial(r0, r1, k):
    """fork: run candidate k from the current state, return the gate."""
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(rfd)
        try:
            execute(seg_rows(r0, r1, k))
            g = gate(r1)
        except Exception as e:  # noqa: BLE001
            g = dict(ok=False, why=[f"exception {e!r}"], score=1e9)
        with os.fdopen(wfd, "wb") as f:
            pickle.dump(g, f)
        os._exit(0)
    os.close(wfd)
    return pid, rfd


def collect(pid, rfd):
    with os.fdopen(rfd, "rb") as f:
        data = f.read()
    os.waitpid(pid, 0)
    return pickle.loads(data) if data else dict(ok=False, why=["child died"], score=1e9)


segs = list(range(0, N, a.seg))
tail0 = max(0, N - int(round(a.tail_s / s.control_dt)))
logf = open(a.log, "a") if a.log else None
stack = []          # per segment taken: (seg index, state before, rec len before, k taken, ks left)
rec = []
i = 0
t0 = time.time()
bt_depth = 0
while i < len(segs):
    r0 = segs[i]; r1 = min(r0 + a.seg, N)
    ks = [1.0] if r0 >= tail0 or r1 % B else KS
    if stack and stack[-1][0] == i:          # re-entered after a backtrack: only slower than before
        _, st, nrec, kprev, ks = stack.pop()
        s.set_state(st); del rec[nrec:]
    st = s.get_state(); st["rows"] = []; nrec = len(rec)
    procs = [(k, *trial(r0, r1, k)) for k in ks]
    res = [(k, collect(pid, fd)) for k, pid, fd in procs]
    ok = [(k, g) for k, g in res if g["ok"]]
    if ok:
        k, g = ok[0]
        rec.extend(execute(seg_rows(r0, r1, k)))
        stack.append((i, st, nrec, k, [x for x in ks if x < k]))
        if len(stack) > 24:
            j = len(stack) - 25
            stack[j] = (stack[j][0], None, stack[j][2], stack[j][3], [])
        line = dict(seg=i, r0=r0, k=k, rows=len(rec), score=g["score"], wall=round(time.time() - t0, 1),
                    fails={str(kk): gg["why"][:2] for kk, gg in res if not gg["ok"]})
        bt_depth = 0
        i += 1
    else:
        line = dict(seg=i, r0=r0, k=None, fails={str(kk): gg["why"][:3] for kk, gg in res})
        # backtrack: the previous segment takes its next slower factor
        while stack and not stack[-1][4]:
            stack.pop()
        if not stack or stack[-1][1] is None or bt_depth >= a.backtrack:
            print(json.dumps(dict(line, stop="no candidate passes; backtrack exhausted")), flush=True)
            break
        bt_depth += 1
        i = stack[-1][0]
        line["backtrack_to"] = i
    print(json.dumps(line), flush=True)
    if logf:
        logf.write(json.dumps(line) + "\n"); logf.flush()
    if i % 50 == 0 or i == len(segs):
        np.savez_compressed(a.out + ".part.npz", ctrl=np.array(rec), substeps=np.int64(R.SUBSTEPS),
                            timestep=np.float64(s.timestep), actuators=np.array(s.actuators), demo_row=np.int64(r1))

done = i >= len(segs)
fc = s.final_check() if done else None
ctrl = np.array(rec)
s.save_trajectory(ctrl, a.out if done else a.out + ".partial.npz")
print(json.dumps(dict(done=done, rows=len(ctrl), demo_rows=N, speedup=round(N / max(1, len(ctrl)), 3),
                      success=None if fc is None else fc["success"],
                      end=None if fc is None else {k: fc["end"][k] for k in ("goal", "released", "at_rest", "breakdown")},
                      settled=None if fc is None else {k: fc["settled"][k] for k in ("goal", "released", "at_rest", "breakdown")},
                      wall=round(time.time() - t0, 1))), flush=True)
