"""retime_search.py (V3 knot rd, 2026-10-17/18; developed as kr.py): re-time a recorded open-loop reference so it runs in fewer control steps, without changing
the physics, the task or the substeps. Runs inside the release runtime (eai-roboarc:0.2.0[-noinst], its shared tooling at
$KR_TOOLING, default /tooling).

  trace  --inst I --rows R --out O                       replay R from the frozen birth; per row the rope nodes, the
                                                         active grasp welds and the wall time (O/trace.npz, O/trace.json)
  search --inst I --rows R --trace T --out O [opts]      greedy segment re-timing (below); O/result.npz = the new rows
                                                         (the Session's applied targets from the birth), O/log.jsonl
  replay --inst I --rows R --out O.json [--repeat 2]     fresh-process replays (grading=True), per-row timing

Re-timing. The original rows A[0..N) are cut into segments at the ends of its waits (rows where every column holds still).
A candidate re-times one segment [p, q): a monotone warp of the original's row index, sampled once per new control step,
the joint targets interpolated linearly between the original's neighbouring rows (the same joint path, both grippers
included), each original interval i advanced at rate r_i original rows per new row:
  waits (no column moves): r = k_wait;  gripper changes: r = k_grip (1 = as recorded);
  arm moves: r = min(k_move, CAP / d_i), d_i the interval's largest arm joint step, CAP 0.0385 rad (the standard
  service's rule is 0.04 rad per control step), so no new row moves an arm joint more than CAP.
The last new row of a candidate lands exactly on A[q - 1]... A[q] alignment: every candidate ends with the original's
targets of row q, so the scene is compared with the original's at the same commanded pose.
Each segment runs all candidates from the current state (a checkpoint, ckpt.py: rewind-exact) in parallel workers; a
candidate is accepted when, at its end, the rope nodes are within --tol metres of the original's at row q (max over
nodes), the same grasp welds are active, and the episode has not ended badly. The most aggressive accepted candidate
(fewest rows) is kept, its end state becomes the next segment's start. The original is the k = 1 candidate, so a segment
can always fall back to it (its deviation then is the inherited one; --tol-floor lets it pass).
After the last segment the final targets are held until the Session ends the episode (success), at most --hold rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("KR_TOOLING", "/tooling"))
sys.path.insert(0, str(HERE))
CAP = 0.0385
ARM_COLS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIP_COLS = np.array([6, 13])


def session(inst):
    from rcb_topogym import Session
    s = Session("knot_untying", instance=inst, strict=True)
    s.record_start()
    return s


class Probe:
    def __init__(self, s):
        import mujoco
        self.s = s
        w = s.env._world
        self.w = w
        self.ropes = [w._ropes[k] for k in sorted(w._ropes)]
        m = w.model
        self.weld_ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, x["name"]) for x in w.manifest["welds"]],
                                 dtype=np.int64)

    def nodes(self):
        d = self.w.data
        return np.concatenate([r.node_positions(d) for r in self.ropes]).astype(np.float32)

    def welds(self):
        return np.flatnonzero(np.asarray(self.w.data.eq_active)[self.weld_ids]).astype(np.int32)


def load_rows(path):
    z = np.load(path, allow_pickle=False)
    return np.asarray(z["actions"], np.float32)


def save_rows(s, path, meta):
    acts = np.asarray(np.stack(s._recording) if s._recording else np.zeros((0, s.action_dim)), np.float32)
    np.savez(path, actions=acts, task=s.task, topogym_task=s.config["task_id"], variation_id=s.config["variation_id"],
             eval_index=-1 if s.eval_index is None else s.eval_index, scene_seed=int(s.config["seed"]),
             instance_sha=s.instance_sha, action_dim=s.action_dim, action_layout=np.array(s.action_layout),
             gripper_space=s.gripper_space, meta=json.dumps(meta or {}), trajectory_version=2,
             physics_steps_per_action=np.asarray(s._recorded_ticks, np.int64))
    return int(len(acts))


# ------------------------------------------------------------------------------------------------------------- trace
def cmd_trace(a):
    """Writes as it goes (a search can start before the trace ends): parts/n_<k>.npy = the nodes of states k .. k+K-1
    (state i = after i rows), welds.jsonl = one line per state, progress.json = {"available": states written}."""
    rows = load_rows(a.rows)
    t0 = time.time()
    s = session(a.inst)
    build_s = time.time() - t0
    P = Probe(s)
    out = Path(a.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    K = a.part
    N = len(rows)
    buf, welds_f = [P.nodes()], open(out / "welds.jsonl", "w")
    welds_f.write(json.dumps(P.welds().tolist()) + "\n")
    welds, wall = [P.welds().tolist()], np.zeros(N)
    ended, k0 = None, 0
    ck = set(int(x) for x in a.ckpt.split(",")) if a.ckpt else set()
    import ckpt as CK

    def flush(final=False):
        nonlocal buf, k0
        while len(buf) >= K or (final and buf):
            chunk = np.asarray(buf[:K], np.float32)
            np.save(out / "parts" / f"n_{k0:06d}.npy", chunk)
            k0 += len(chunk)
            buf = buf[K:]
        welds_f.flush()
        (out / "progress.tmp").write_text(json.dumps(dict(available=k0, rows=N, part=K)))
        os.replace(out / "progress.tmp", out / "progress.json")
    for i, r in enumerate(rows):
        t = time.perf_counter()
        e = s.step(r)
        wall[i] = time.perf_counter() - t
        buf.append(P.nodes())
        w = P.welds().tolist()
        welds.append(w)
        welds_f.write(json.dumps(w) + "\n")
        if (i + 1) in ck:
            CK.save_checkpoint(s, out / f"ck_{i + 1:05d}.pkl")
        if len(buf) >= K:
            flush()
        if e:
            ended = i + 1
            break
    flush(final=True)
    welds_f.close()
    n = ended or N
    st = s.status()
    np.save(out / "wall.npy", wall[:n])
    res = dict(rows=a.rows, rows_sha256=hashlib.sha256(Path(a.rows).read_bytes()).hexdigest(), n_rows=N, executed=n,
               ended_at=ended, success=st["success"], episode_end=st["episode_end"], state_sha=st["state_sha"],
               instance_sha=s.instance_sha, build_s=round(build_s, 1), wall_s=round(float(wall[:n].sum()), 1),
               s_per_row=round(float(wall[:n].mean()), 4), forcing=st.get("forcing"), review=st.get("review"),
               instability=st.get("instability"), part=K)
    (out / "trace.json").write_text(json.dumps(res, default=str))
    print("TRACE", json.dumps({k: res[k] for k in ("executed", "ended_at", "success", "episode_end", "state_sha", "wall_s",
                                                    "s_per_row", "build_s")}), flush=True)


class Trace:
    """The original's states, read as the trace writes them (waits for a state not written yet)."""
    def __init__(self, d):
        self.d = Path(d)
        self.cache = {}

    def info(self):
        p = self.d / "trace.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def available(self):
        try:
            return json.loads((self.d / "progress.json").read_text())
        except (OSError, ValueError):
            return dict(available=0, part=1)

    def state(self, q, poll=30):
        while True:
            pr = self.available()
            if pr["available"] > q:
                break
            if self.info() is not None and pr["available"] <= q:
                raise IndexError(f"state {q} beyond the trace's {pr['available']}")
            time.sleep(poll)
        K = pr["part"]
        k0 = (q // K) * K
        if k0 not in self.cache:
            self.cache = {k0: np.load(self.d / "parts" / f"n_{k0:06d}.npy")}
        with open(self.d / "welds.jsonl") as f:
            for i, line in enumerate(f):
                if i == q:
                    w = json.loads(line)
                    break
        return self.cache[k0][q - k0], w


# ------------------------------------------------------------------------------------------------------------ re-time
def intervals(A):
    dA = np.abs(np.diff(A.astype(np.float64), axis=0))
    d_arm = dA[:, ARM_COLS].max(1)
    d_grip = dA[:, GRIP_COLS].max(1)
    return d_arm, d_grip


def segments(A, min_len, max_len, wait_min):
    """Cut points q (row counts executed) at the end of waits of >= wait_min rows, segments of min_len..max_len rows."""
    d_arm, d_grip = intervals(A)
    still = (d_arm <= 1e-9) & (d_grip <= 1e-9)
    N = len(A)
    ends = []
    i = 0
    while i < len(still):
        if still[i]:
            j = i
            while j < len(still) and still[j]:
                j += 1
            if j - i >= wait_min:
                ends.append(j + 1)       # rows 0..j executed: the wait's last row is row j
            i = j
        else:
            i += 1
    cuts, last = [], 0
    for q in ends:
        if q - last >= min_len:
            while q - last > max_len:
                last += max_len
                cuts.append(last)
            cuts.append(q)
            last = q
    while N - last > max_len:
        last += max_len
        cuts.append(last)
    if not cuts or cuts[-1] != N:
        if cuts and N - cuts[-1] < min_len // 2:
            cuts[-1] = N
        else:
            cuts.append(N)
    return [(p, q) for p, q in zip([0] + cuts[:-1], cuts)]


def retime(A, p, q, k_wait, k_move, k_grip=1.0, prev=None):
    """New rows for the original's rows p..q-1 (executed rows p+1..q, i.e. targets A[p]..A[q-1]); the segment's
    'time' axis is the original row index tau in [p - 1, q - 1] (tau = p - 1 is the start state's last target, prev)."""
    A64 = A.astype(np.float64)
    pts = np.vstack([A64[p - 1] if p > 0 else (prev if prev is not None else A64[0]), A64[p:q]])   # tau = p-1 .. q-1
    d = np.abs(np.diff(pts, axis=0))
    d_arm, d_grip = d[:, ARM_COLS].max(1), d[:, GRIP_COLS].max(1)
    r = np.full(len(d), float(k_move))
    wait = (d_arm <= 1e-9) & (d_grip <= 1e-9)
    r[wait] = k_wait
    mv = d_arm > 1e-9
    r[mv] = np.minimum(r[mv], CAP / d_arm[mv])
    r[d_grip > 1e-9] = np.minimum(r[d_grip > 1e-9], k_grip)
    r = np.maximum(r, 1e-3)
    if k_move <= 1 and k_wait <= 1 and k_grip <= 1:
        return pts[1:].astype(np.float32)
    cost = np.concatenate([[0.], np.cumsum(1. / r)])      # new-time at each original row boundary
    T = max(1, int(np.ceil(cost[-1] - 1e-9)))
    t = np.arange(1, T + 1) * (cost[-1] / T)               # the last sample at the end
    seg = np.clip(np.searchsorted(cost, t, side="left") - 1, 0, len(d) - 1)
    f = (t - cost[seg]) / np.maximum(cost[seg + 1] - cost[seg], 1e-12)
    out = pts[seg] + (pts[seg + 1] - pts[seg]) * f[:, None]
    out[-1] = pts[-1]
    # guard: no arm joint step above CAP (the warp's rescale by cost[-1]/T can only slow it)
    full = np.vstack([pts[:1], out])
    step = np.abs(np.diff(full, axis=0))[:, ARM_COLS].max(1)
    if step.max() > CAP + 1e-7:
        raise AssertionError(f"retime step {step.max()} > CAP")
    return out.astype(np.float32)


def warp(A, rate, start):
    """The whole route re-timed: rate[i] = original rows per new row over interval i (from target i-1 to target i;
    interval 0 starts at the reset targets `start`); arm joint steps capped at CAP. The last new row is A[-1]."""
    A64 = A.astype(np.float64)
    pts = np.vstack([start[None], A64])
    d = np.abs(np.diff(pts, axis=0))
    d_arm = d[:, ARM_COLS].max(1)
    r = np.asarray(rate, np.float64).copy()
    mv = d_arm > 1e-9
    r[mv] = np.minimum(r[mv], CAP / d_arm[mv])
    r = np.maximum(r, 1e-3)
    cost = np.concatenate([[0.], np.cumsum(1. / r)])
    T = max(1, int(np.ceil(cost[-1] - 1e-9)))
    t = np.arange(1, T + 1) * (cost[-1] / T)
    seg = np.clip(np.searchsorted(cost, t, side="left") - 1, 0, len(d) - 1)
    f = (t - cost[seg]) / np.maximum(cost[seg + 1] - cost[seg], 1e-12)
    out = pts[seg] + (pts[seg + 1] - pts[seg]) * f[:, None]
    out[-1] = pts[-1]
    step = np.abs(np.diff(np.vstack([pts[:1], out]), axis=0))[:, ARM_COLS].max(1)
    if step.max() > CAP + 1e-7:
        raise AssertionError(f"warp step {step.max()} > CAP")
    return out.astype(np.float32)


def cmd_global(a):
    """Global re-timing variants from the original's trace (classes per original row: a wait, a move while the rope is
    quiet in the original (every node slower than --quiet m/s over that row), a move while the rope moves, a gripper
    change), written as <out>/<name>.npz (actions only; replay them with `replay`)."""
    A = load_rows(a.rows)
    if a.upto:                      # only the first --upto rows (e.g. the untie, before a re-planned placement)
        A = A[:a.upto]
    n = len(A)
    if a.trace == "none":           # no trace: every move counts as moving the rope (factor --km)
        v = np.full(n, np.inf)
    else:
        parts = sorted((Path(a.trace) / "parts").glob("n_*.npy"))
        nodes = np.concatenate([np.load(p) for p in parts])
        if len(nodes) < n + 1:
            raise SystemExit(f"trace has {len(nodes) - 1} rows of {n}")
        v = np.linalg.norm(np.diff(nodes[:n + 1], axis=0), axis=2).max(1) / 0.025
        v = np.maximum.reduce([v, np.r_[v[1:], 0], np.r_[0, v[:-1]]])      # the row and its neighbours
    start = np.asarray(json.loads(a.start), np.float64) if a.start else A[0].astype(np.float64)
    pts = np.vstack([start[None], A.astype(np.float64)])
    d = np.abs(np.diff(pts, axis=0))
    d_arm, d_grip = d[:, ARM_COLS].max(1), d[:, GRIP_COLS].max(1)
    wait = (d_arm <= 1e-9) & (d_grip <= 1e-9)
    grip = d_grip > 1e-9
    quiet = v < a.quiet
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cls = dict(wait=int(wait.sum()), grip=int(grip.sum()), move_quiet=int((~wait & ~grip & quiet).sum()),
               move_rope=int((~wait & ~grip & ~quiet).sum()))
    made = []
    # --blocks B: one variant per block and factor set, the factors applied only inside that block of original rows
    # (the rest at the recorded timing): which parts of the route tolerate a re-timing
    blocks = [(0, n)] if not a.blocks else [(n * k // a.blocks, n * (k + 1) // a.blocks) for k in range(a.blocks)]
    if a.from_row:                  # one block: rows from --from-row on re-timed, the rows before it as recorded
        a.blocks, blocks = 1, [(a.from_row, n)]
    for kw in [float(x) for x in a.kw.split(",")]:
        for kq in [float(x) for x in a.kq.split(",")]:
            for km in [float(x) for x in a.km.split(",")]:
              for bi, (b0, b1) in enumerate(blocks):
                r = np.where(wait, kw, np.where(grip, 1., np.where(quiet, kq, km)))
                if a.blocks:    # the rows outside the block exactly as recorded; the block re-timed between them
                    st0 = start if b0 == 0 else A[b0 - 1].astype(np.float64)
                    rows = np.vstack([A[:b0], warp(A[b0:b1], r[b0:b1], st0), A[b1:]]).astype(np.float32)
                else:
                    rows = warp(A, r, start)
                name = f"g_w{kw:g}_q{kq:g}_m{km:g}" + (f"_b{bi}of{a.blocks}" if a.blocks else "")
                z = np.load(a.rows, allow_pickle=False)
                np.savez(out / f"{name}.npz", actions=rows, task=z["task"], topogym_task=z["topogym_task"],
                         variation_id=z["variation_id"], eval_index=z["eval_index"], scene_seed=z["scene_seed"],
                         instance_sha=a.instance_sha or z["instance_sha"], action_dim=z["action_dim"],
                         action_layout=z["action_layout"], gripper_space=z["gripper_space"], trajectory_version=2,
                         physics_steps_per_action=np.full(len(rows), 5, np.int64),
                         meta=json.dumps(dict(source=a.rows, kw=kw, kq=kq, km=km, quiet=a.quiet, classes=cls,
                                              block=[b0, b1] if a.blocks else None, upto=a.upto or None)))
                made.append(dict(name=name, rows=int(len(rows))))
    (out / "variants.json").write_text(json.dumps(dict(classes=cls, variants=made), indent=1))
    print("GLOBAL", json.dumps(dict(classes=cls, n=len(made), rows=[m["rows"] for m in made])), flush=True)


LADDERS = {
    "std": [(1, 1), (1.5, 1.25), (2, 1.5), (3, 1.5), (3, 2), (4, 2.5), (6, 3), (8, 4), (12, 6), (4, 1), (1, 2), (20, 10)],
    "wide": [(1, 1), (2, 1.25), (2, 1.5), (3, 2), (4, 2), (5, 2.5), (6, 3), (8, 3.5), (10, 4), (12, 5), (16, 8), (24, 12)],
    "v2": [(1, 1), (1, 1.5), (1, 2), (1, 2.5), (1, 3), (1, 4), (1, 6), (1.25, 2), (1.5, 1.5), (1.5, 2), (1.5, 3), (2, 2),
           (2, 3), (2, 4), (3, 3), (3, 4), (4, 6), (6, 8), (12, 12)],
}

# ------------------------------------------------------------------------------------------------------------- workers
_S = None
_P = None
_INST = None


def _winit(inst):
    global _S, _P, _INST
    _INST = inst
    _S = session(inst)
    _P = Probe(_S)


def _wrun(job):
    """job: dict(ck, rows, want_nodes, want_welds, out_ck, hold_until_end)"""
    import ckpt as CK
    s, P = _S, _P
    t0 = time.time()
    if job["ck"] is None:
        s.reset()
        s.record_start()
    else:
        CK.restore_checkpoint(s, job["ck"])
    rows = np.load(job["rows"]) if isinstance(job["rows"], str) else job["rows"]
    ended = None
    cancel = job.get("cancel")
    for i, r in enumerate(rows):
        if cancel and i % 10 == 0 and os.path.exists(cancel):
            return dict(name=job["name"], cancelled=True, n_rows=int(len(rows)), episode_end=None, instability=False,
                        wall_s=round(time.time() - t0, 1))
        if s.step(r):
            ended = i + 1
            break
    held = 0
    if ended is None and job.get("hold", 0):
        for _ in range(job["hold"]):
            held += 1
            if s.step(s.hold_action()):
                ended = len(rows) + held
                break
    nodes = P.nodes()
    welds = P.welds().tolist()
    res = dict(name=job["name"], n_rows=int(len(rows)), ended=ended, held=held, episode_end=s.episode_end,
               success=bool(s.success()) if hasattr(s, "success") else None, n_steps=int(s.n_steps),
               instability=s.instability is not None, wall_s=round(time.time() - t0, 1))
    if job.get("want_nodes") is not None:
        dev = np.linalg.norm(nodes - job["want_nodes"], axis=1)
        res.update(dev_max=float(dev.max()), dev_p95=float(np.percentile(dev, 95)), dev_mean=float(dev.mean()),
                   welds_same=sorted(welds) == sorted(job["want_welds"]), welds=welds)
    if job.get("out_ck") and s.episode_end is None:
        CK.save_checkpoint(s, job["out_ck"])
    if s.episode_end is not None and job.get("out_rows"):
        save_rows(s, job["out_rows"], {})
    if job.get("state_sha"):
        res["state_sha"] = s.state_sha()
    return res


def _wrun_i(arg):
    i, job = arg
    return i, _wrun(job)


def cmd_search(a):
    A = load_rows(a.rows)
    TR = Trace(a.trace)
    N = len(A)
    trj = TR.info()
    if trj is not None and trj["executed"] != N:
        print(f"note: the original trace executed {trj['executed']} of {N} rows (ended {trj['episode_end']})", flush=True)
        N = trj["executed"]
        A = A[:N]
    segs = segments(A, a.min_len, a.max_len, a.wait_min)
    if a.final_keep:
        # keep the original's last rows as they are (the release and the settle into success)
        keep = N - a.final_keep
        segs = [(p, min(q, keep)) for p, q in segs if p < keep] + [(keep, N)]
        segs = [(p, q) for p, q in segs if q > p]
    ladder = LADDERS[a.ladder]
    if a.max_k:
        ladder = [x for x in ladder if max(x) <= a.max_k]
    out = Path(a.out)
    (out / "ck").mkdir(parents=True, exist_ok=True)
    (out / "cand").mkdir(exist_ok=True)
    # resume: a log with segment records continues after its last chosen segment (its checkpoint kept)
    cur_ck, new_n, prev, start_si, prev_dev = None, 0, None, 0, 0.
    if (out / "log.jsonl").is_file():
        recs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines() if l.strip()]
        if any(r["kind"] == "end" for r in recs):
            print("SEARCH already ended", flush=True)
            return
        segrecs = [r for r in recs if r["kind"] == "seg"]
        if segrecs:
            last = segrecs[-1]
            ck_last = out / "ck" / f"s{last['si']:03d}_{last['choice']}.pkl"
            if ck_last.is_file() and last["episode_end"] is None and segs[last["si"]] == (last["p"], last["q"]):
                cur_ck, new_n, start_si = str(ck_last), last["new_total"], last["si"] + 1
                prev_dev = float(last.get("dev_max") or 0.)
                prev = np.load(out / "cand" / f"s{last['si']:03d}_{last['choice']}.npy")[-1].astype(np.float64)
                print(f"resume after segment {last['si']} ({new_n} rows for orig {last['q']})", flush=True)
    logf = open(out / "log.jsonl", "a")
    ctx = mp.get_context("fork")
    pool = ctx.Pool(a.workers, initializer=_winit, initargs=(a.inst,))
    t_start = time.time()
    meta = dict(source_rows=a.rows, source_sha256=hashlib.sha256(Path(a.rows).read_bytes()).hexdigest(), tol=a.tol,
                ladder=a.ladder, min_len=a.min_len, max_len=a.max_len, wait_min=a.wait_min, final_keep=a.final_keep, cap=CAP,
                segments=len(segs), resumed_at=start_si, tol_add=a.tol_add, tol_cap=a.tol_cap, split=a.split,
                tol_before=a.tol_before)
    logf.write(json.dumps(dict(kind="start", **meta)) + "\n")
    logf.flush()
    done_rows = None
    for si, (p, q) in enumerate(segs):
        if si < start_si:
            continue
        cands, seen = [], set()
        last = si == len(segs) - 1
        lad = [(1, 1)] if (a.final_keep and last) else ladder
        want_nodes, want_welds = TR.state(q)
        for kw, km in lad:
            rr = retime(A, p, q, kw, km, a.k_grip, prev)
            key = hashlib.sha1(rr.tobytes()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            f = out / "cand" / f"s{si:03d}_w{kw}_m{km}.npy"
            np.save(f, rr)
            cands.append(dict(name=f"w{kw}_m{km}", kw=kw, km=km, rows=str(f), ck=cur_ck,
                              want_nodes=want_nodes, want_welds=want_welds, out_ck=str(out / "ck" / f"s{si:03d}_w{kw}_m{km}.pkl"),
                              hold=a.hold if last else 0, out_rows=str(out / "cand" / f"s{si:03d}_w{kw}_m{km}_end.npz"),
                              n=len(rr)))
        cancel = str(out / "cand" / f"s{si:03d}.cancel")
        for c in cands:
            c["cancel"] = cancel
        order = sorted(range(len(cands)), key=lambda i: cands[i]["n"])
        res = [None] * len(cands)

        def passes(c, r):
            if r is None or r.get("cancelled"):
                return False
            if r["instability"] or (r["episode_end"] is not None and r["episode_end"] != "success"):
                return False
            return r["episode_end"] == "success" or (r["welds_same"] and r["dev_max"] <= limit)
        # the deviation allowed at this segment's end: --tol, or (--tol-add > 0) the deviation inherited at its start
        # plus --tol-add, at most --tol-cap
        limit = a.tol if a.tol_add <= 0 else min(max(a.tol, prev_dev + a.tol_add), a.tol_cap)
        if a.split and q <= a.split:          # before --split (e.g. the untie's end): --tol-before, absolute
            limit = a.tol_before
        decided = False
        for i, r in pool.imap_unordered(_wrun_i, [(i, cands[i]) for i in order], chunksize=1):
            res[i] = r
            if decided:
                continue
            passing = [j for j in range(len(cands)) if passes(cands[j], res[j])]
            if passing:
                best = min(passing, key=lambda j: cands[j]["n"])
                if all(res[j] is not None for j in range(len(cands)) if cands[j]["n"] <= cands[best]["n"]):
                    decided = True
                    Path(cancel).touch()
        for c, r in zip(cands, res):
            r["n"] = c["n"]
        ok = []
        for c, r in zip(cands, res):
            if passes(c, r):
                ok.append((r["n_steps"] if r["episode_end"] == "success" else new_n + c["n"], c, r))
        ident = next((x for x in zip(cands, res) if x[0]["kw"] == 1 and x[0]["km"] == 1), None)
        if not ok:
            # no candidate within the tolerance (an inherited deviation): the least deviating one with the same welds
            good = [(rr.get("dev_max", 9.), cc["n"], i) for i, (cc, rr) in enumerate(zip(cands, res))
                    if not rr.get("cancelled") and rr["episode_end"] is None and not rr["instability"] and rr.get("welds_same")]
            if good:
                i = min(good)[2]
                c, r = cands[i], res[i]
            else:
                c, r = ident
            choice, why = (c, r), "fallback"
        else:
            ok.sort(key=lambda x: x[0])
            choice, why = (ok[0][1], ok[0][2]), "accept"
        c, r = choice
        rec = dict(kind="seg", si=si, p=p, q=q, orig=q - p, choice=c["name"], n=c["n"], why=why, dev_max=r.get("dev_max"), limit=limit,
                   welds_same=r.get("welds_same"), episode_end=r["episode_end"],
                   cands=[dict(name=cc["name"], n=cc["n"], dev_max=None if rr.get("dev_max") is None else round(rr["dev_max"], 5),
                               welds_same=rr.get("welds_same"), end=rr["episode_end"], inst=rr["instability"], wall=rr["wall_s"],
                               cancelled=rr.get("cancelled", False))
                          for cc, rr in zip(cands, res)],
                   new_total=new_n + c["n"], elapsed_s=round(time.time() - t_start))
        logf.write(json.dumps(rec, default=str) + "\n")
        logf.flush()
        print(f"[{time.strftime('%H:%M:%S')}] seg {si}/{len(segs)} {p}-{q} ({q - p}) -> {c['name']} {c['n']} rows "
              f"({why}, dev {r.get('dev_max')}, end {r['episode_end']}); total {new_n + c['n']} for orig {q}", flush=True)
        new_n += c["n"]
        prev_dev = float(r.get("dev_max") or 0.)
        prev = np.load(c["rows"])[-1].astype(np.float64)
        for cc in cands:          # keep only the chosen checkpoint
            if cc is not c and os.path.exists(cc["out_ck"]):
                os.remove(cc["out_ck"])
        cur_ck = c["out_ck"]
        if r["episode_end"] == "success":
            done_rows = c["out_rows"]
            break
    pool.close()
    if done_rows is None and cur_ck is not None:
        # the last segment ended without success: hold from its end state
        r = _hold_tail(a.inst, cur_ck, a.hold, out)
        done_rows = r.get("rows_file")
    summary = dict(kind="end", rows_file=done_rows, total_s=round(time.time() - t_start), **meta)
    if done_rows:
        z = np.load(done_rows)
        summary["new_rows"] = int(len(z["actions"]))
        os.replace(done_rows, out / "result.npz")
        summary["rows_file"] = str(out / "result.npz")
    logf.write(json.dumps(summary) + "\n")
    logf.close()
    print("SEARCH", json.dumps(summary), flush=True)


def _hold_tail(inst, ck, hold, out):
    import ckpt as CK
    s = session(inst)
    CK.restore_checkpoint(s, ck)
    for _ in range(hold):
        if s.step(s.hold_action()):
            break
    if s.episode_end == "success":
        save_rows(s, out / "tail_end.npz", {})
        return dict(rows_file=str(out / "tail_end.npz"))
    return dict(rows_file=None, episode_end=s.episode_end)


# ------------------------------------------------------------------------------------------------------------- replay
def cmd_replay(a):
    z = np.load(a.rows, allow_pickle=False)
    rows = np.asarray(z["actions"], np.float32)
    ticks = np.asarray(z["physics_steps_per_action"]) if "physics_steps_per_action" in z.files else None
    s = session(a.inst)
    outs = []
    for k in range(a.repeat):
        s.reset()
        s.record_start()
        t0 = time.time()
        ended = None
        for i, r in enumerate(rows):
            if s.step(r, _physics_steps=None if ticks is None else int(ticks[i])):
                ended = i + 1
                break
        held = 0
        if ended is None and a.hold:
            for _ in range(a.hold):
                held += 1
                if s.step(s.hold_action()):
                    ended = len(rows) + held
                    break
        st = s.status()
        if a.save_success and st["success"] and k == 0:
            save_rows(s, a.save_success, dict(source=a.rows, held=held))
        outs.append(dict(executed=int(s.n_steps), ended_at=ended, held=held, success=st["success"], episode_end=st["episode_end"],
                         state_sha=st["state_sha"], wall_s=round(time.time() - t0, 1),
                         s_per_row=round((time.time() - t0) / max(1, s.n_steps), 4), forcing=st.get("forcing"),
                         review=st.get("review")))
        print("REPLAY", json.dumps(outs[-1], default=str), flush=True)
    res = dict(rows=a.rows, rows_sha256=hashlib.sha256(Path(a.rows).read_bytes()).hexdigest(), n_rows=int(len(rows)),
               instance=a.inst, instance_sha=s.instance_sha, replays=outs,
               deterministic=len({o["state_sha"] for o in outs}) == 1, all_success=all(o["success"] for o in outs))
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


def cmd_ckinfo(a):
    """A checkpoint's state hash and step count in a fresh Session; --rows-out: its recorded rows from the birth."""
    import ckpt as CK
    s = session(a.inst)
    CK.restore_checkpoint(s, a.ck)
    out = dict(ck=a.ck, n_steps=int(s.n_steps), state_sha=s.state_sha(), episode_end=s.episode_end,
               recorded=len(s._recording))
    if a.rows_out:
        save_rows(s, a.rows_out, {})
    print("CKINFO", json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("global")
    g.add_argument("--rows", required=True)
    g.add_argument("--trace", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--start", default=None, help="the reset targets (JSON list); default the first row")
    g.add_argument("--quiet", type=float, default=0.002)
    g.add_argument("--kw", default="2,3,4")
    g.add_argument("--kq", default="2,3,4,6")
    g.add_argument("--km", default="1,1.25,1.5,2")
    g.add_argument("--blocks", type=int, default=0)
    g.add_argument("--upto", type=int, default=0)
    g.add_argument("--from-row", type=int, default=0)
    g.add_argument("--instance-sha", default=None, help="the birth the rows are replayed on (the made instance)")
    c = sub.add_parser("ckinfo")
    c.add_argument("--inst", required=True)
    c.add_argument("--ck", required=True)
    c.add_argument("--rows-out", default=None)
    t = sub.add_parser("trace")
    t.add_argument("--inst", required=True)
    t.add_argument("--rows", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--ckpt", default="")
    t.add_argument("--part", type=int, default=250)
    s = sub.add_parser("search")
    s.add_argument("--inst", required=True)
    s.add_argument("--rows", required=True)
    s.add_argument("--trace", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--tol", type=float, default=0.003)
    s.add_argument("--tol-add", type=float, default=0.)
    s.add_argument("--tol-cap", type=float, default=0.010)
    s.add_argument("--split", type=int, default=0, help="segments ending at or before this original row use --tol-before")
    s.add_argument("--tol-before", type=float, default=0.0005)
    s.add_argument("--ladder", default="std")
    s.add_argument("--max-k", type=float, default=0)
    s.add_argument("--k-grip", type=float, default=1.0)
    s.add_argument("--min-len", type=int, default=150)
    s.add_argument("--max-len", type=int, default=600)
    s.add_argument("--wait-min", type=int, default=3)
    s.add_argument("--final-keep", type=int, default=0)
    s.add_argument("--hold", type=int, default=600)
    r = sub.add_parser("replay")
    r.add_argument("--inst", required=True)
    r.add_argument("--rows", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--hold", type=int, default=0, help="hold the last targets up to N rows if the rows end first")
    r.add_argument("--save-success", default=None, help="the executed rows (with the hold) of a successful first replay")
    a = ap.parse_args()
    dict(trace=cmd_trace, search=cmd_search, replay=cmd_replay, ckinfo=cmd_ckinfo, global_=cmd_global)[a.cmd.replace("global", "global_")](a)
    os._exit(0)


if __name__ == "__main__":
    main()
