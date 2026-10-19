#!/usr/bin/env python3
"""A faster reference from a demo: an operation-gated time warp on the final physics (true-state feedback).

The demo is cut at its own operation boundaries: phase p ends a little after the demo's p-th retention weld latched
(the demo track from profile.py: the welds at every block end). Each phase is resampled block by block (blocks of B demo
rows): a block in which the robot touches no part (the demo profile's robot-part contact fraction 0) at the free factor
kF, every other block at the contact factor kC (k demo rows per control tick; linear interpolation of every actuator,
the block's last demo row always the last target). Candidates (kF, kC) run in parallel in forked children from the
state the previous phase left; the fastest (fewest rows) that passes the phase gate is taken, and the parent executes it
itself, recording the EXECUTED targets (data.ctrl after the clamp), so the clamp never changes a recorded row on replay.

Phase gate: no breakdown; exactly the demo's welds active at the phase end (the phase's own weld latched by the
mechanism's acceptance gates); every part that is neither welded nor touched by the robot within --tol-pos / --tol-rot
of the demo (screws: their axis only); the time-warped phase takes at most as many ticks as the demo phase.
If no candidate passes, the previous phase is redone with its next slower candidate (--backtrack phases deep).
The last phase (after the last weld: release, retreat, the success hold) runs unwarped. Verify with probe.py (fresh
process) and the grader.

usage: warp_ops.py TASK DEMO.npz TRACK.npz OUT.npz [--cands 4:2,4:1.5,3:1.5,4:1,2:1,1:1] [--after 2]
"""
import argparse, json, os, pickle, sys, time
sys.path.insert(0, os.environ["RCB_SHARED"])
import numpy as np, mujoco
import rcb_ikea as R

ap = argparse.ArgumentParser()
ap.add_argument("task"); ap.add_argument("demo"); ap.add_argument("track"); ap.add_argument("out")
ap.add_argument("--cands", default="4:2,4:1.5,3:1.5,4:1,2:1,1:1")
ap.add_argument("--after", type=int, default=2, help="blocks after the demo's latch block that end the phase")
ap.add_argument("--tol-pos", type=float, default=0.003)
ap.add_argument("--tol-rot", type=float, default=3.0)
ap.add_argument("--backtrack", type=int, default=2)
ap.add_argument("--procs", type=int, default=6)
ap.add_argument("--log", default=None)
a = ap.parse_args()
CANDS = [tuple(float(x) for x in c.split(":")) for c in a.cands.split(",")]
assert CANDS[-1] == (1.0, 1.0)

s = R.Session(a.task)
m, d = s.model, s.data
D = R.load_trajectory(a.demo, s.nu, s.timestep)
T = np.load(a.track)
B = int(T["block"]); TQ = T["qpos"]; TEQ = T["eq"].astype(bool); REC = T["rec"]
RP = REC[:, 3]                      # robot-part contact fraction per block (profile.py column robot_part)
N = len(D)
PARTIAL = len(TQ) * B < N - B           # a partial track: only the phases it covers

robot = set(R.robot_bodies(m).tolist())
parts = []
for j in range(m.njnt):
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        b = int(m.jnt_bodyid[j])
        if b not in robot:
            parts.append((m.body(b).name, b))
AXIS = {o["part"]: np.array(o["part_axis"], float) for o in s.S.task.get("operations", []) if o.get("mode") == "screw"}
weld_ids = [i for i in range(m.neq) if m.eq_type[i] == mujoco.mjtEq.mjEQ_WELD]
weld_body = {i: (m.body(int(m.eq_obj1id[i])).name, m.body(int(m.eq_obj2id[i])).name) for i in weld_ids}
scratch = mujoco.MjData(m)

# phases: the demo's weld latch blocks
W = TEQ[:, weld_ids]
nw = W.sum(1)
latch_blocks = [int(np.argmax(nw >= p)) for p in range(1, len(weld_ids) + 1) if (nw >= p).any()]
ends = [min(len(TQ) - 1, lb + a.after) for lb in latch_blocks]           # block index whose END ends the phase
phase_rows = [(0 if p == 0 else (ends[p - 1] + 1) * B, (ends[p] + 1) * B) for p in range(len(ends))]
if not PARTIAL:
    phase_rows.append(((ends[-1] + 1) * B, N))


def qerr_deg(q1, q2, axis=None):
    if axis is None:
        return float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q1, q2)))))))
    v1 = np.zeros(3); v2 = np.zeros(3)
    mujoco.mju_rotVecQuat(v1, axis, q1); mujoco.mju_rotVecQuat(v2, axis, q2)
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1))))


def gate(row_end):
    kb = row_end // B - 1
    out = dict(ok=True, why=[])
    if s.validity.breakdown is not None:
        out["ok"] = False; out["why"].append(f"breakdown {s.validity.breakdown}")
    weq = np.array([bool(d.eq_active[i]) for i in weld_ids]); deq = W[kb]
    if not np.array_equal(weq, deq):
        out["ok"] = False; out["why"].append(f"welds {weq.astype(int).tolist()} vs demo {deq.astype(int).tolist()}")
    scratch.qpos[:] = TQ[kb]; mujoco.mj_kinematics(m, scratch)
    welded = {n for i in weld_ids if d.eq_active[i] for n in weld_body[i]}
    touched = {x for pair in s.validity.robot_part_contacts() for x in pair}
    worst = 0.
    for n, b in parts:
        if n in welded or n in touched:
            continue
        dp = float(np.linalg.norm(d.xpos[b] - scratch.xpos[b])); dr = qerr_deg(d.xquat[b], scratch.xquat[b], AXIS.get(n))
        worst = max(worst, dp / a.tol_pos, dr / a.tol_rot)
        if dp > a.tol_pos or dr > a.tol_rot:
            out["ok"] = False; out["why"].append(f"{n}: {1e3*dp:.2f}mm {dr:.2f}deg")
    out["score"] = round(worst, 3)
    return out


def interp(tau):
    i0 = int(np.floor(tau)); f = tau - i0
    if f < 1e-12 or i0 + 1 >= N:
        return D[min(i0, N - 1)].copy()
    return (1 - f) * D[i0] + f * D[i0 + 1]


def block_rows(r0, r1, k):
    if k == 1.0:
        return D[r0:r1]
    n = int(np.ceil((r1 - r0) / k))
    rows = np.array([interp(min(r0 - 1 + j * k, r1 - 1)) for j in range(1, n + 1)])
    rows[-1] = D[r1 - 1]
    return rows


def phase_rows_warped(r0, r1, kF, kC):
    out = []
    for b0 in range(r0, r1, B):
        b1 = min(b0 + B, r1)
        kb = b0 // B
        k = kF if (kb < len(RP) and RP[kb] == 0.0) else kC
        out.append(block_rows(b0, b1, k))
    return np.concatenate(out)


def execute(rows):
    rec = []
    for row in rows:
        s.step(row)
        rec.append(d.ctrl.copy())
    return rec


def trial(rows, r1, last):
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(rfd)
        try:
            execute(rows)
            g = gate(r1) if not last else dict(ok=True, why=[], score=0.)
            if last:
                fc = s.final_check()
                g = dict(ok=bool(fc["success"]), why=[] if fc["success"] else [json.dumps({k: fc["settled"][k] for k in ("goal", "released", "at_rest", "breakdown")}, default=str)[:300]], score=0.)
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


logf = open(a.log, "a") if a.log else None


def log(x):
    print(json.dumps(x), flush=True)
    if logf:
        logf.write(json.dumps(x) + "\n"); logf.flush()


log(dict(phases=[(int(x), int(y)) for x, y in phase_rows], latch_rows=[(lb + 1) * B for lb in latch_blocks]))
stack = []        # (phase, state before, rec len before, remaining slower candidates)
rec = []
p = 0
t0 = time.time()
backtracks = 0
while p < len(phase_rows):
    r0, r1 = phase_rows[p]
    last = p == len(phase_rows) - 1 and not PARTIAL
    if stack and stack[-1][0] == p:
        _, st, nrec, cands = stack.pop()
        s.set_state(st); del rec[nrec:]
    else:
        cands = [(1.0, 1.0)] if last else CANDS
    st = s.get_state(); st["rows"] = []; nrec = len(rec)
    plans = [(c, phase_rows_warped(r0, r1, *c)) for c in cands]
    plans.sort(key=lambda x: len(x[1]))
    res = []
    for i in range(0, len(plans), a.procs):
        batch = [(c, rows, *trial(rows, r1, last)) for c, rows in plans[i:i + a.procs]]
        for j, (c, rows, pid, fd) in enumerate(batch):     # shortest first: the first that passes wins
            g = collect(pid, fd)
            res.append((c, rows, g))
            if g["ok"]:
                for _, _, pid2, fd2 in batch[j + 1:]:
                    os.kill(pid2, 9); os.waitpid(pid2, 0); os.close(fd2)
                break
        if any(g["ok"] for _, _, g in res):
            break
    ok = [(c, rows, g) for c, rows, g in res if g["ok"]]
    fails = {f"{c[0]}:{c[1]}": g["why"][:3] for c, _, g in res if not g["ok"]}
    if ok:
        c, rows, g = ok[0]
        rec.extend(execute(rows))
        stack.append((p, st, nrec, [x for x in cands if len(phase_rows_warped(r0, r1, *x)) > len(rows)]))
        log(dict(phase=p, demo_rows=[r0, r1], cand=c, rows=len(rows), total=len(rec), score=g["score"],
                 wall=round(time.time() - t0, 1), fails=fails))
        np.savez_compressed(a.out + ".part.npz", ctrl=np.array(rec), substeps=np.int64(R.SUBSTEPS),
                            timestep=np.float64(s.timestep), actuators=np.array(s.actuators), demo_row=np.int64(r1),
                            phase=np.int64(p))
        p += 1
    else:
        log(dict(phase=p, demo_rows=[r0, r1], cand=None, fails=fails, wall=round(time.time() - t0, 1)))
        while stack and not stack[-1][3]:
            stack.pop()
        if not stack or backtracks >= a.backtrack:
            log(dict(stop="no candidate passes; backtrack exhausted", phase=p))
            break
        backtracks += 1
        p = stack[-1][0]

done = p >= len(phase_rows)
ctrl = np.array(rec)
if done:
    s.save_trajectory(ctrl, a.out)
log(dict(done=done, rows=len(ctrl), demo_rows=N, speedup=round(N / max(1, len(ctrl)), 3), wall=round(time.time() - t0, 1)))
