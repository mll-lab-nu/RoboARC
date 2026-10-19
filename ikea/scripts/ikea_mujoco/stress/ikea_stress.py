#!/usr/bin/env python3
"""Stress episodes for the IKEA MuJoCo scenes, read by the RoboTangle stress observer (scripts/robotangle/stress/
observer.py, criteria v2.13b; verdict.py decides). The same methodology as the RoboARC families (docs/rcb/STRESS_HOWTO.md):
random and perturbed action streams on the frozen scene, every MuJoCo step watched for non-finite state, bad warnings,
contacts deeper than 10 mm, energy created, uncaused speed jumps and constraint-energy pops above 3 m/s (v2.11 energy
share). The observer only reads; a run with it ends on the same state hash as a run without it.

The observer was written for a topogym Session. A thin shim (IkeaWorld) gives it the same view of an IKEA scene:
  - entities: the robot = every body tree that carries an actuated joint or tendon; each other movable body tree under
    the world is one task entity (a furniture part, a jaw on its tray); trees without dofs are static;
  - one tick = one control row = rcb_ikea.SUBSTEPS physics steps + the scene's mechanism (S.step), exactly the grader's
    (rcb_ikea.Session.step: ctrl, S.step(SUBSTEPS), the Validity tick);
  - the scene mechanism's latched welds are the observer's weld partners (MechanismWelds), as assisted-grasp welds are;
  - the table: the largest static box whose name has table / bench / desk, joined with the other such boxes whose tops
    are at the same height (vesken's bench tops); the IKEA checks replace two of the observer's (classify()): tunnelling
    = a part's centre of mass inside a top's solid, the workspace = the footprint + 1 m and 1.5 m over the top.

Every command runs in its own process (one scene per process). Environment: IKEA_SCENES (the scenes root, default
/opt/ikea/scenes), RT_STRESS_DIR (the RoboTangle observer + verdict, default <repo>/scripts/robotangle/stress).

  ikea_stress.py checkpoints --task T --ref R.npz --out DIR [--every .05]   # the reference once, unobserved: saved states
  ikea_stress.py refseg --task T --ref R.npz --ckpt DIR --seg K --out F    # reference segment K observed (from ckpt K-1
                                                                            #   or the birth; +overlap; the last + settle)
  ikea_stress.py nop --task T --rows N --out F
  ikea_stress.py episode --task T --strategy S --seed K --out F [--ckpt DIR --ref R.npz] [--rows-dir D] [--no-observe]
  ikea_stress.py pool --jobs J.jsonl --out DIR --procs N [--timeout S]     # episodes, one fresh process each
  ikea_stress.py confirm --task T --rows X.npz --ref R.npz [--ckpt DIR] --out F   # from the birth: ref prefix + rows
  ikea_stress.py shrink --task T --rows X.npz --ckpt DIR --match TEXT --out Y.npz
  ikea_stress.py report DIR [DIR ...] [--md F]
Strategies: random (smooth random targets from the birth), prefix_random (a reference checkpoint, then random), prefix_jerk
(a checkpoint, then step targets to random values, no slew), mutate (the reference with smooth noise and gripper flips,
from a checkpoint), fast (the reference at 2-4x speed from a checkpoint), grip (a checkpoint, then gripper slams while
the arms hold), push (a checkpoint, then 1-3 arm joints ramp slowly by 5-30 % of their range and hold: a deep push).
"""
from __future__ import annotations

import argparse, faulthandler, gzip, hashlib, json, os, pickle, signal, subprocess, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STRESS = Path(os.environ.get("RT_STRESS_DIR", REPO / "scripts/robotangle/stress"))
for p in (str(HERE.parent / "shared"), str(STRESS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import mujoco  # noqa: E402
import rcb_ikea as R  # noqa: E402

TASKS = ("lack", "ikea365", "vesken", "svallet", "eket", "variera", "gladom", "burvik", "stenerik", "stegesund", "kyrre", "uppsta")
STRATEGIES = ("random", "prefix_random", "prefix_jerk", "mutate", "fast", "grip", "push")
WS_MARGIN_M, WS_CEIL_M, WS_FLOOR_M = 1., 1.5, .1   # the IKEA workspace: the footprint + 1 m (beyond both arms' reach;
#   eket's reference carries a panel 0.76 m past the table edge), 1.5 m over the top (vesken's caps: 1.0 m), the floor
OVERLAP = 200          # refseg: rows observed past the segment's end (the next segment's first ticks get history)


# ------------------------------------------------------------------ physics variants (A/B tests of candidate fixes)
VARIANT = os.environ.get("IKEA_STRESS_VARIANT", "")   # "" = the tooling as is; "hand70": Franka Hand forcerange
                                                      # +-70 N and no hand slew (the arm clamp stays)


def make_session(task, scene_dir=None):
    if VARIANT == "hand70":
        R.GRIPPER_SPEED_LIMITS = {}
    elif VARIANT:
        raise SystemExit(f"unknown IKEA_STRESS_VARIANT {VARIANT}")
    sess = R.Session(task, scene_dir)
    if VARIANT == "hand70":
        _hand70(sess)
        reset = sess.reset

        def reset_and_apply():                        # a reset rebuilds the model: apply again
            reset(); _hand70(sess)
        sess.reset = reset_and_apply
    return sess


def _hand70(sess):
    m = sess.model
    for i, n in enumerate(sess.actuators):
        if n.endswith("actuator8"):
            m.actuator_forcerange[i] = (-70., 70.)
            m.actuator_forcelimited[i] = 1


# ------------------------------------------------------------------ the shim the observer reads
class _Link:
    def __init__(self, idx, fixed):
        self.idx, self.is_fixed = int(idx), bool(fixed)


class _Entity:
    def __init__(self, links):
        self.links = links


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _moving(m, b):
    while b > 0:
        if m.body_dofnum[b] > 0:
            return True
        b = m.body_parentid[b]
    return False


def find_table(m, d):
    """The static table / bench boxes: the largest, joined with every other whose top is within 1 mm of its top."""
    boxes = []
    for g in range(m.ngeom):
        name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        bname = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "").lower()
        if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX) or _moving(m, int(m.geom_bodyid[g])):
            continue
        if not any(k in name or k in bname for k in ("table", "bench", "desk")):
            continue
        R_ = np.asarray(d.geom_xmat[g]).reshape(3, 3)
        half = np.abs(R_) @ np.asarray(m.geom_size[g])           # world-axis half extents
        pos = np.asarray(d.geom_xpos[g])
        boxes.append(dict(area=half[0] * half[1], lo=pos[:2] - half[:2], hi=pos[:2] + half[:2], top=pos[2] + half[2],
                          bottom=pos[2] - half[2], name=name or bname))
    if not boxes:
        return None
    best = max(boxes, key=lambda b: b["area"])
    same = [b for b in boxes if abs(b["top"] - best["top"]) < 1e-3]
    lo, hi = np.min([b["lo"] for b in same], 0), np.max([b["hi"] for b in same], 0)
    slabs = [b for b in boxes if b["area"] > .01]          # the tops (legs excluded): the tunnelling check's solids
    return dict(pos=(lo + hi) / 2, size=hi - lo, top=float(best["top"]), name="+".join(b["name"] for b in same),
                slabs=[dict(lo=b["lo"], hi=b["hi"], top=b["top"], bottom=b["bottom"], name=b["name"]) for b in slabs])


class MechanismWelds:
    """The scene mechanism's latched welds, in the form the observer reads its grasp manager's bindings: every active
    body-to-body weld / connect equality makes its part's weld partner the host it is welded to (vesken's caps on their
    trays, lack's legs on the tabletop, stegesund's bolts on the shelf), as an assisted-grasp weld makes a held link's
    partner the hand. Read-only."""
    events = ()

    def __init__(self, m, d):
        self.m, self.d = m, d
        ok = np.isin(np.asarray(m.eq_type), [int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT)])
        ok &= np.isin(np.asarray(m.eq_objtype), [int(mujoco.mjtObj.mjOBJ_BODY), int(mujoco.mjtObj.mjOBJ_XBODY)])
        ok &= (np.asarray(m.eq_obj1id) > 0) & (np.asarray(m.eq_obj2id) > 0)
        self.eqs = np.flatnonzero(ok)

    @property
    def hands(self):
        act = self.eqs[np.asarray(self.d.eq_active)[self.eqs] != 0]
        if not len(act):
            return {}
        # Each weld makes its part's partner the host it is welded to. The host is the body in more active welds (the
        # shelf of stegesund's seat welds, authored part-first: body1 = the bolt or the clamped half, body2 = the shelf;
        # lack's tabletop); on a tie the first body, as the scenes author it (vesken's caps on their trays). The
        # observer reads a binding's links as (partner, part).
        n = Counter(int(b) for e in act for b in (self.m.eq_obj1id[e], self.m.eq_obj2id[e]))
        out = {}
        for e in act:
            b1, b2 = int(self.m.eq_obj1id[e]), int(self.m.eq_obj2id[e])
            host, part = (b2, b1) if n[b2] > n[b1] else (b1, b2)
            out[f"mechanism_{int(e)}"] = {"binding": {"links": (host, part)}}
        return out


class IkeaWorld:
    """What observer.Observer reads from a topogym World, for one rcb_ikea.Session."""

    def __init__(self, sess: "R.Session"):
        self.sess = sess
        self.model, self.data = sess.model, sess.data
        m = self.model
        self.physics_dt_s = R.SUBSTEPS * float(m.opt.timestep)
        self.physics_step = 0
        self._grasp_manager = MechanismWelds(m, self.data)
        robot = set(R.robot_bodies(m).tolist())
        groups = defaultdict(list)
        for b in range(1, m.nbody):
            groups[int(m.body_rootid[b])].append(b)
        self.entities = {"robot": _Entity([_Link(b, not _moving(m, b)) for b in sorted(robot)])}
        for r, bodies in sorted(groups.items()):
            if r in robot or not any(_moving(m, b) for b in bodies):
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, r) or f"body{r}"
            self.entities[name] = _Entity([_Link(b, not _moving(m, b)) for b in bodies])
        self.table = find_table(m, self.data)
        if self.table is None:
            raise RuntimeError("no static table / bench box: the tunnelling and workspace checks need one")
        self.table_height_m = self.table["top"]
        self.env_cfg = _NS(world=_NS(table=_NS(pos=self.table["pos"], size=self.table["size"])))

    def advance(self):
        self.sess.S.step(R.SUBSTEPS)
        self.physics_step += 1


class Observed:
    """An rcb_ikea.Session with the observer on it. step(row) is exactly Session.step(row): ctrl, SUBSTEPS physics steps
    + mechanism (through IkeaWorld.advance, which the observer wraps), the Validity tick, the row recorded."""

    def __init__(self, sess, trace=False, observe=True):
        self.s = sess
        self.w = IkeaWorld(sess)
        self.env = _NS(_world=self.w, _task=_NS(table_height_m=self.w.table_height_m))
        self.monitor = None
        self.obs = None
        self._parts = R.part_bodies(sess.model)
        m = sess.model
        self._frange = np.where(np.asarray(m.actuator_forcelimited).astype(bool), np.asarray(m.actuator_forcerange)[:, 1], np.inf)
        self.ef = dict(min=(np.inf, 0, 0.), rise=0., at=None, from_tick=None, wc_change=None, x=None, sat_ticks=0, final=None,
                       nonpos=(0, 0.), gmax=0., gmax_at=None, gmax_from=None, gmax_wc_change=None, gmax_sat=None)
        self.tunnel_n, self.tunnel_ev = 0, []
        self.ws = dict(beyond_m=(-1e9, 0, None), over_top_m=(-1e9, 0, None), below_floor_m=(-1e9, 0, None))
        if observe:
            from observer import Observer
            self.obs = Observer(self, trace=trace)

    def step(self, row):
        a = np.asarray(row, dtype=np.float64).reshape(-1)
        if a.shape != (self.s.nu,) or not np.isfinite(a).all():
            raise ValueError(f"a row has {self.s.nu} finite entries")
        self.s.data.ctrl[:] = self.s.applied(a) if hasattr(self.s, "applied") else a   # the speed clamp, as Session.step
        self.w.advance()
        self.s.validity.tick()
        self.s.rows.append(a.copy())
        if self.obs is not None:
            self._workspace()
            self._energy_forensics()

    def _energy_forensics(self):
        """For the report, not the verdict: the excess X's largest rise over its running minimum, with the change of the
        constraint work Wc and of the number of saturated actuators over that rise. A rise with Wc unchanged is made by
        the integration of the smooth forces (the drives, gravity), with no contact, limit or equality involved."""
        o = self.obs
        if not o.excess_hist:
            return
        X, t = float(o.excess_hist[-1]), self.w.physics_step
        sat = int(np.sum(np.abs(self.s.data.actuator_force) >= self._frange - 1e-9))
        e = self.ef
        if X < e["min"][0]:
            e["min"] = (X, t, float(o.Wc))
        rise = X - e["min"][0]
        if rise > e["rise"]:
            e.update(rise=rise, at=t, from_tick=e["min"][1], wc_change=float(o.Wc) - e["min"][2], x=X)
        e["sat_ticks"] += int(sat > 0)
        e["final"] = X
        if X <= 0:
            e["nonpos"] = (t, float(o.Wc))
        elif X > e["gmax"]:                         # the global rule's peak: X above 0 since its last tick at <= 0
            e.update(gmax=X, gmax_at=t, gmax_from=e["nonpos"][0], gmax_wc_change=float(o.Wc) - e["nonpos"][1],
                     gmax_sat=sat)

    def _workspace(self):
        """Tunnelling: a part's centre of mass inside a table top's solid (its footprint less 2 cm, 1 cm under its top
        surface, above its bottom), counted per part-tick. The IKEA workspace (replaces the observer's left_workspace, whose box, 0.1 m around the table and 1 m over it,
        the references leave: vesken's caps and eket's back panel are carried higher and wider): each part's centre of
        mass beyond the table footprint (m), its height over the table top and over the floor; the episode's extremes.
        classify() reads them against WS_MARGIN_M / WS_CEIL_M / WS_FLOOR_M (a VIOLENT trigger)."""
        com = np.asarray(self.s.data.xipos)[self._parts]
        t = self.w.table
        beyond = np.max(np.abs(com[:, :2] - t["pos"]) - t["size"] / 2, axis=1)
        k, h = int(np.argmax(beyond)), int(np.argmax(com[:, 2]))
        for sl in t["slabs"]:            # tunnelling: a centre of mass inside a table top's solid (10 mm under its top)
            inside = (np.all((com[:, :2] > sl["lo"] + .02) & (com[:, :2] < sl["hi"] - .02), axis=1)
                      & (com[:, 2] < sl["top"] - .01) & (com[:, 2] > sl["bottom"]))
            if inside.any():
                k = int(np.flatnonzero(inside)[0])
                self.tunnel_n += int(inside.sum())
                if len(self.tunnel_ev) < 3:
                    self.tunnel_ev.append(dict(tick=self.w.physics_step, body=self.s.model.body(int(self._parts[k])).name,
                                               slab=sl["name"], z_mm=round(1000 * float(com[k, 2] - sl["top"]), 2)))
        ws = self.ws
        for key, val, b in (("beyond_m", float(beyond[k]), k), ("over_top_m", float(com[h, 2] - t["top"]), h),
                            ("below_floor_m", -float(com[:, 2].min()), int(np.argmin(com[:, 2])))):
            if val > ws[key][0]:
                ws[key] = (val, self.w.physics_step, self.s.model.body(int(self._parts[b])).name)

    def close(self):
        if self.obs is not None:
            self.obs.uninstall()
            self.obs = None


# ------------------------------------------------------------------ action streams
def ctrl_range(m):
    lo, hi = np.array(m.actuator_ctrlrange[:, 0]), np.array(m.actuator_ctrlrange[:, 1])
    lim = np.asarray(m.actuator_ctrllimited).astype(bool)
    jr = np.full((m.nu, 2), np.nan)
    for a in range(m.nu):
        if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            j = int(m.actuator_trnid[a, 0])
            if m.jnt_limited[j]:
                jr[a] = m.jnt_range[j]
    lo = np.where(lim, lo, np.where(np.isnan(jr[:, 0]), -np.pi, jr[:, 0]))
    hi = np.where(lim, hi, np.where(np.isnan(jr[:, 1]), np.pi, jr[:, 1]))
    return lo, hi


def gripper_mask(names):
    return np.array([n.endswith("actuator8") or n.endswith("gripper") for n in names])


def smooth_random(rng, start, lo, hi, grip, n, dt, amp, vmax):
    """Piecewise-linear random targets from `start`: segments of 0.1-1.5 s, each to a random target within `amp` of the
    range around the start, at most `vmax` rad/s per joint (a gripper: its range in 0.3 s); grippers flip now and then."""
    rows = np.empty((n, len(start)))
    cur = start.copy()
    span = hi - lo
    vstep = np.where(grip, span * dt / .3, vmax * dt)      # a gripper crosses its range in 0.3 s
    i = 0
    while i < n:
        seg = max(1, int(rng.uniform(.1, 1.5) / dt))
        tgt = np.clip(start + rng.uniform(-amp, amp, len(start)) * span, lo, hi)
        tgt[grip] = cur[grip]
        step = np.clip((tgt - cur) / seg, -vstep, vstep)
        for _ in range(seg):
            if i >= n:
                break
            cur = np.clip(cur + step, lo, hi)
            if grip.any() and rng.random() < 1.5 * dt:     # about 1.5 gripper flips per second
                k = int(rng.choice(np.flatnonzero(grip)))
                cur[k] = lo[k] if abs(cur[k] - hi[k]) < abs(cur[k] - lo[k]) else hi[k]
            rows[i] = cur
            i += 1
    return rows


def jerk_stream(rng, start, lo, hi, n, dt):
    """Step targets: hold 0.05-0.6 s, then jump a random 40 % of the actuators to random values in the full range."""
    rows = np.empty((n, len(start)))
    cur = start.copy()
    i = 0
    while i < n:
        hold = max(1, int(rng.uniform(.05, .6) / dt))
        k = rng.random(len(start)) < .4
        cur = np.where(k, rng.uniform(lo, hi), cur)
        rows[i:i + hold] = cur
        i += hold
    return rows[:n]


def smooth_noise(rng, n, dim, sigma, corr_s, dt):
    k = max(1, int(corr_s / dt))
    knots = rng.normal(0, sigma, (n // k + 2, dim))
    x = np.arange(n) / k
    i0 = np.floor(x).astype(int)
    f = (x - i0)[:, None]
    return knots[i0] * (1 - f) + knots[i0 + 1] * f


# ------------------------------------------------------------------ checkpoints
def load_ref(path):
    with np.load(path, allow_pickle=False) as z:
        return np.ascontiguousarray(z["ctrl"], dtype=np.float64)


def ckpt_index(d):
    """index.json (this tool) or ref_check.json (the integration agent's ref_check.py): rows, birth sha, [(row, file,
    sha)] ascending."""
    d = Path(d)
    p = d / "index.json"
    j = json.loads((p if p.exists() else d / "ref_check.json").read_text())
    cps = [dict(row=int(c["row"]), file=c["file"], sha=c.get("state_sha") or c.get("sha")) for c in j["checkpoints"]]
    return dict(rows=int(j["rows"]), birth_sha=j.get("birth_sha"), checkpoints=sorted(cps, key=lambda c: c["row"]))


def load_ckpt(d, c):
    with open(Path(d) / c["file"], "rb") as f:
        return pickle.load(f)


def cmd_checkpoints(a):
    """Replay the reference once without the observer, saving get_state() every `every` of its rows (and its end)."""
    ref = load_ref(a.ref)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = make_session(a.task, a.scene_dir)
    birth = s.state_sha()
    k = max(1, int(round(a.every * len(ref))))
    marks = set(range(k, len(ref), k)) | {len(ref)}
    t0, idx = time.time(), []
    for i, row in enumerate(ref, 1):
        s.step(row)
        if i in marks:
            st = s.get_state()
            st["rows"] = []                             # the prefix is the reference's first i rows; not stored twice
            p = out / f"ckpt_{i:07d}.pkl"
            with open(p, "wb") as f:
                pickle.dump(st, f, protocol=4)
            idx.append(dict(row=i, file=p.name, state_sha=s.state_sha(), goal=bool(s.success().get("success"))))
    fc = s.final_check()
    meta = dict(task=a.task, ref=str(a.ref), ref_sha256=hashlib.sha256(Path(a.ref).read_bytes()).hexdigest(),
                rows=len(ref), birth_sha=birth, end_sha=s.state_sha(), success=bool(fc["success"]),
                end_breakdown=fc["end"]["breakdown"], settled_breakdown=fc["settled"]["breakdown"],
                wall_s=round(time.time() - t0, 1), ms_per_tick=round(1000 * (time.time() - t0) / len(ref), 3),
                mujoco=mujoco.__version__, numpy=np.__version__, host=os.uname().nodename, checkpoints=idx)
    (out / "index.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: v for k, v in meta.items() if k != "checkpoints"}, default=str))
    return 0


# ------------------------------------------------------------------ runs
def classify(rec):
    """verdict.classify (v2.13b), with the observer's left_workspace and tunnel replaced by the IKEA workspace and
    tunnelling checks (Observed._workspace): the observer's tunnel is a centre of mass under the tabletop inside its
    footprint, which a part that falls beside a narrow top (vesken) or under a table on legs (eket, vesken) also is."""
    from verdict import classify as vc
    r = json.loads(json.dumps(rec, default=str))
    r.setdefault("instance", f"{rec['task']}/{rec.get('kind')}")
    c = (r.get("observer") or {}).get("counts") or {}
    c.pop("left_workspace", None)
    if "ikea_tunnel" in rec:                        # the IKEA tunnelling check (inside a top's solid) replaces the observer's
        c.pop("tunnel", None)
        if rec["ikea_tunnel"]["count"]:
            c["tunnel"] = rec["ikea_tunnel"]["count"]
    v = vc(r, "v2.13b")
    ws = rec.get("ikea_workspace") or {}
    out = [f"{k} {ws[k]['value']:.2f} ({ws[k]['body']})" for k, lim in (("beyond_m", WS_MARGIN_M), ("over_top_m", WS_CEIL_M),
                                                                     ("below_floor_m", WS_FLOOR_M))
           if k in ws and ws[k]["value"] > lim]
    if out:
        v["triggers"].append("left_ws: " + ", ".join(out))
        if v["verdict"] == "CLEAN":
            v["verdict"] = "VIOLENT"
    return v


def run_rows(sess, rows, *, task, kind, extra=None, observe=True, trace=None, progress=None, sha_at=None):
    """Run `rows` on `sess` (from its current state) with the observer; the result dict with its verdict. sha_at: also
    record the state hash after that many rows."""
    t0 = time.time()
    ob = Observed(sess, trace=trace is not None, observe=observe)
    fault, n, mid = None, 0, None
    pf = open(progress, "w") if progress else None
    try:
        for row in rows:
            ob.step(row)
            n += 1
            if sha_at is not None and n == sha_at:
                mid = sess.state_sha()
            if pf is not None and n % 25 == 0:
                pf.seek(0); pf.write(f"{n}\n"); pf.flush()
    except Exception as exc:  # noqa: BLE001
        fault = dict(error=repr(exc)[:400], trace=traceback.format_exc()[-1500:], row=n)
    if pf is not None:
        pf.close()
    wall = time.time() - t0
    o = ob.obs.summary() if ob.obs is not None else None
    tr = ob.obs.trace if ob.obs is not None else None
    ws = {k: dict(value=v[0], tick=v[1], body=v[2]) for k, v in ob.ws.items()}
    ef = {k: v for k, v in ob.ef.items() if k not in ("min", "nonpos")}
    ws["observer_left_workspace"] = ((o or {}).get("counts") or {}).get("left_workspace", 0)
    tun = dict(count=ob.tunnel_n, events=ob.tunnel_ev, observer_count=((o or {}).get("counts") or {}).get("tunnel", 0))
    ob.close()          # before status(): ikea365's success report runs a 2 s support probe on the data and restores it
    try:
        st = sess.status()
    except Exception as exc:  # noqa: BLE001
        st = dict(status_error=repr(exc)[:300])
    rec = dict(task=task, kind=kind, rows=int(n), wall_s=round(wall, 2), ms_per_control=round(1000 * wall / max(1, n), 2),
               fault=fault, success=st.get("success"), goal=st.get("goal"), state_sha=st.get("state_sha"),
               sim_time_s=st.get("sim_time_s"), validity_breakdown=st.get("breakdown"),
               max_part_depth_mm=(st.get("terminal") or {}).get("max_part_depth_mm"),
               table=dict(name=ob.w.table["name"], top=ob.w.table["top"], pos=ob.w.table["pos"].tolist(),
                          size=ob.w.table["size"].tolist()),
               entities={k: len(e.links) for k, e in ob.w.entities.items()}, observed=observe,
               ikea_workspace=ws, ikea_tunnel=tun, energy_forensics=ef, mujoco=mujoco.__version__, numpy=np.__version__, host=os.uname().nodename,
               npy_mask=os.environ.get("NPY_DISABLE_CPU_FEATURES", ""), variant=VARIANT,
               slewed_ticks=getattr(sess, "slewed_ticks", None), observer=o)
    if sha_at is not None:
        rec["sha_at"] = dict(row=sha_at, sha=mid)
    rec.update(extra or {})
    if o is not None:
        if n > 0 and o.get("substeps") != n * R.SUBSTEPS:
            rec["harness_error"] = f"the observer saw {o.get('substeps')} MuJoCo steps of {n * R.SUBSTEPS}"
        v = classify(rec)
        rec["verdict"] = dict(verdict=v["verdict"], fails=v["fails"], triggers=v["triggers"], notes=v["notes"])
        if rec.get("harness_error"):
            rec["verdict"]["fails"].append("harness_error")
            rec["verdict"]["verdict"] = "BREAKDOWN"
    if trace is not None and tr is not None:
        with gzip.open(trace, "wt") as f:
            json.dump(tr, f)
    return rec


def make_episode(a, sess):
    """The episode's start state (None = the birth; else loaded into `sess`) and rows, from its seed alone."""
    rng = np.random.default_rng([a.seed, STRATEGIES.index(a.strategy)])
    m = sess.model
    dt = R.SUBSTEPS * float(m.opt.timestep)
    lo, hi = ctrl_range(m)
    grip = gripper_mask(sess.actuators)
    n = int(round(rng.uniform(a.min_s, a.max_s) / dt))
    extra = dict(strategy=a.strategy, seed=a.seed, ckpt_row=0)
    state, start = None, sess.hold_action()
    if a.strategy in ("mutate", "fast") and not a.ckpt:
        raise SystemExit(f"{a.strategy} follows the reference: it needs --ckpt")
    if a.strategy != "random" and a.ckpt:              # without --ckpt the other strategies start at the birth
        idx = ckpt_index(a.ckpt)
        c = idx["checkpoints"][int(rng.integers(len(idx["checkpoints"])))]
        state = load_ckpt(a.ckpt, c)
        extra.update(ckpt_row=c["row"], ckpt_sha=c["sha"])
        sess.set_state(state)
        if sess.state_sha() != c["sha"]:
            raise SystemExit(f"checkpoint {c['row']} does not load to its hash")
        start = sess.data.ctrl.copy()                                  # the ctrl in force at the checkpoint
    if a.strategy in ("random", "prefix_random"):
        amp, vmax = float(rng.choice([.1, .25, .5])), float(rng.choice([.5, 1.5, 3.]))
        extra.update(amp=amp, vmax=vmax)
        rows = smooth_random(rng, start, lo, hi, grip, n, dt, amp, vmax)
    elif a.strategy == "prefix_jerk":
        rows = jerk_stream(rng, start, lo, hi, n, dt)
    elif a.strategy in ("mutate", "fast"):
        ref = load_ref(a.ref)
        r0 = extra["ckpt_row"]
        if a.strategy == "fast":
            f = int(rng.choice([2, 3, 4]))
            seg = ref[r0:r0 + n * f:f].copy()
            extra.update(speedup=f)
        else:
            seg = ref[r0:r0 + n].copy()
            sig = float(rng.choice([.005, .02, .06]))
            nz = smooth_noise(rng, len(seg), seg.shape[1], sig, float(rng.uniform(.05, .5)), dt)
            nz[:, grip] = 0
            seg = seg + nz
            for i in np.flatnonzero(rng.random(len(seg)) < .5 * dt):     # gripper flips, 0.3 s each
                k = int(rng.choice(np.flatnonzero(grip)))
                seg[i:i + int(.3 / dt), k] = hi[k] + lo[k] - seg[i, k]
            extra.update(sigma=sig)
        if len(seg) < n:                                                  # the reference ended: hold its last row
            last = seg[-1:] if len(seg) else start[None]
            seg = np.concatenate([seg, np.repeat(last, n - len(seg), 0)])
        rows = np.clip(seg, lo, hi)
    elif a.strategy == "grip":
        rows = np.repeat(start[None], n, 0)
        i = 0
        while i < n:
            k = int(rng.choice(np.flatnonzero(grip)))
            rows[i:, k] = lo[k] if rng.random() < .5 else hi[k]
            i += max(1, int(rng.uniform(.05, .5) / dt))
    elif a.strategy == "push":
        # a deep push: from a checkpoint (often a part held or seated), the grippers kept closed as they are, 1-3 arm
        # joints ramp slowly (1-3 s) by 5-30 % of their range, then hold: the hand drives what it holds or touches
        # into its neighbours (thin mesh pieces, dowels) under the servo's full force
        rows = np.repeat(start[None], n, 0)
        arm = np.flatnonzero(~grip)
        ks = rng.choice(arm, size=int(rng.integers(1, 4)), replace=False)
        T = int(rng.uniform(1., 3.) / dt)
        ramp = np.clip(np.arange(1, n + 1) / T, 0., 1.)
        for k in ks:
            d = rng.choice([-1., 1.]) * rng.uniform(.05, .3) * (hi[k] - lo[k])
            rows[:, k] = np.clip(start[k] + ramp * d, lo[k], hi[k])
        extra.update(push_joints=[sess.actuators[int(k)] for k in ks], push_s=round(T * dt, 2))
    else:
        raise SystemExit(f"unknown strategy {a.strategy}")
    return state, np.ascontiguousarray(rows, dtype=np.float64), extra


def episode_id(task, strategy, seed):
    return f"{task}__{strategy}__s{seed}"


def cmd_episode(a):
    faulthandler.enable()
    eid = episode_id(a.task, a.strategy, a.seed)
    sess = make_session(a.task, a.scene_dir)
    state, rows, extra = make_episode(a, sess)
    rows_file = None
    if a.rows_dir:                                  # saved before the run: a crash keeps its reproducer
        Path(a.rows_dir).mkdir(parents=True, exist_ok=True)
        rows_file = Path(a.rows_dir) / f"{eid}.npz"
        np.savez_compressed(rows_file, ctrl=rows, ckpt_row=np.int64(extra["ckpt_row"]))
    extra.update(id=eid, rows_file=str(rows_file) if rows_file else None, instance=f"{a.task}/episode")
    rec = run_rows(sess, rows, task=a.task, kind="episode", extra=extra, observe=not a.no_observe, trace=a.trace,
                   progress=a.progress)
    if rows_file is not None and rec.get("verdict", {}).get("verdict") == "CLEAN" and not a.keep_rows:
        rows_file.unlink()
        rec["rows_file"] = None
    _write(a.out, rec)
    return 0


def cmd_index(a):
    """Rebuild a checkpoint dir's index.json from its saved states (no replay): each state's hash and goal, the birth hash,
    and the grader's final_check from the last checkpoint (the reference's end)."""
    ref = load_ref(a.ref)
    out = Path(a.out)
    files = sorted(out.glob("ckpt_*.pkl"))
    s = make_session(a.task, a.scene_dir)
    birth = s.state_sha()
    idx = []
    for p in files:
        with open(p, "rb") as f:
            st = pickle.load(f)
        s.set_state(st)
        idx.append(dict(row=int(p.stem.split("_")[1]), file=p.name, state_sha=s.state_sha(),
                        goal=bool(s.success().get("success"))))
    if idx[-1]["row"] != len(ref):
        raise SystemExit(f"the last checkpoint is row {idx[-1]['row']}, the reference has {len(ref)} rows")
    s.rows = [np.asarray(ref[-1])]                  # final_check holds the last row
    fc = s.final_check()
    meta = dict(task=a.task, ref=str(a.ref), ref_sha256=hashlib.sha256(Path(a.ref).read_bytes()).hexdigest(),
                rows=len(ref), birth_sha=birth, end_sha=idx[-1]["state_sha"], success=bool(fc["success"]),
                end_breakdown=fc["end"]["breakdown"], settled_breakdown=fc["settled"]["breakdown"], rebuilt=True,
                note="breakdown fields cover the settle hold only (the Validity history is the checkpoint's)",
                settled_sha=fc["settled_sha"], mujoco=mujoco.__version__, numpy=np.__version__,
                host=os.uname().nodename, checkpoints=idx)
    (out / "index.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: v for k, v in meta.items() if k != "checkpoints"}, default=str))
    return 0


def cmd_refseg(a):
    """Reference segment K of the checkpoint index, observed: from checkpoint K-1 (the birth for K = 0) to checkpoint K,
    plus OVERLAP rows into the next segment (the last: the grader's settle hold of the last row). Its hash at
    checkpoint K must equal the checkpoint's."""
    ref = load_ref(a.ref)
    idx = ckpt_index(a.ckpt)
    cps = idx["checkpoints"]
    sess = make_session(a.task, a.scene_dir)
    r0 = 0 if a.seg == 0 else cps[a.seg - 1]["row"]
    if a.seg:
        sess.set_state(load_ckpt(a.ckpt, cps[a.seg - 1]))
    start_sha = sess.state_sha()
    expect_start = idx["birth_sha"] if a.seg == 0 else cps[a.seg - 1]["sha"]
    r1 = cps[a.seg]["row"]
    rows = ref[r0:r1]
    last = a.seg == len(cps) - 1
    tail = np.repeat(ref[-1:], int(round(R.SETTLE_S / sess.control_dt)), 0) if last else ref[r1:r1 + OVERLAP]
    rec = run_rows(sess, np.concatenate([rows, tail]), task=a.task, kind="refseg", trace=a.trace, sha_at=len(rows),
                   extra=dict(id=f"{a.task}__ref__seg{a.seg:02d}", instance=f"{a.task}/reference", seg=a.seg, row0=r0,
                              row1=r1, tail=len(tail), tail_kind="settle" if last else "overlap",
                              start_sha_ok=(expect_start is None or start_sha == expect_start),
                              ckpt_sha_ok=None))
    rec["ckpt_sha_ok"] = (rec.get("sha_at") or {}).get("sha") == cps[a.seg]["sha"]
    _write(a.out, rec)
    return 0


def cmd_nop(a):
    sess = make_session(a.task, a.scene_dir)
    rows = np.repeat(sess.hold_action()[None], a.rows, 0)
    _write(a.out, run_rows(sess, rows, task=a.task, kind="nop", trace=a.trace,
                           extra=dict(id=f"{a.task}__nop", instance=f"{a.task}/nop")))
    return 0


def cmd_confirm(a):
    """A candidate from the birth in a fresh process: the reference's first ckpt_row rows unobserved (their hash checked
    against the checkpoint index when given), then the saved rows observed."""
    faulthandler.enable()
    with np.load(a.rows, allow_pickle=False) as z:
        rows = np.ascontiguousarray(z["ctrl"], np.float64)
        r0 = int(z["ckpt_row"]) if "ckpt_row" in z.files else 0      # a plain trajectory: rows from the birth
    sess = make_session(a.task, a.scene_dir)
    t0 = time.time()
    if r0:
        sess.run(load_ref(a.ref)[:r0])
    prefix_sha = sess.state_sha()
    ok = None
    if a.ckpt and r0:
        ok = prefix_sha == next(c["sha"] for c in ckpt_index(a.ckpt)["checkpoints"] if c["row"] == r0)
    sess.validity.reset()
    sess.rows = []
    rec = run_rows(sess, rows, task=a.task, kind="confirm", trace=a.trace, progress=a.progress,
                   extra=dict(id=Path(a.rows).stem + "__confirm", instance=f"{a.task}/confirm", ckpt_row=r0,
                              rows_file=str(a.rows), prefix_wall_s=round(time.time() - t0, 1), prefix_sha=prefix_sha,
                              prefix_matches_ckpt=ok))
    _write(a.out, rec)
    return 0


def cmd_rows(a):
    """A saved stream (rows + ckpt_row) from its checkpoint state, observed: an A/B test of a physics variant on the same
    start state and rows (the state is the stock run's; no prefix replay)."""
    with np.load(a.rows, allow_pickle=False) as z:
        rows, r0 = np.ascontiguousarray(z["ctrl"], np.float64), int(z["ckpt_row"]) if "ckpt_row" in z.files else 0
    sess = make_session(a.task, a.scene_dir)
    if r0:
        sess.set_state(load_ckpt(a.ckpt, next(c for c in ckpt_index(a.ckpt)["checkpoints"] if c["row"] == r0)))
    rec = run_rows(sess, rows, task=a.task, kind="rows", extra=dict(id=Path(a.rows).stem, instance=f"{a.task}/rows",
                                                                     ckpt_row=r0, rows_file=str(a.rows)))
    _write(a.out, rec)
    return 0


def cmd_shrink(a):
    """Shrink a candidate (from its checkpoint, in this process) to the shortest prefix that still fails with --match,
    followed by a 40-row hold; then each arm / gripper column group back to the start ctrl if it still fails."""
    with np.load(a.rows, allow_pickle=False) as z:
        rows, r0 = np.ascontiguousarray(z["ctrl"], np.float64), int(z["ckpt_row"])
    st = None
    if r0:
        st = load_ckpt(a.ckpt, next(c for c in ckpt_index(a.ckpt)["checkpoints"] if c["row"] == r0))
    sess = make_session(a.task, a.scene_dir)
    log = []

    def test(cand):
        sess.reset()
        if st is not None:
            sess.set_state(st)
        cand = np.concatenate([cand, np.repeat(cand[-1:], 40, 0)])
        rec = run_rows(sess, cand, task=a.task, kind="shrink", extra=dict(id="shrink"))
        fails = rec["verdict"]["fails"] + ([rec["fault"]["error"]] if rec.get("fault") else [])
        ok = any(a.match in f for f in fails)
        log.append(dict(rows=len(cand), ok=ok, fails=fails))
        print(json.dumps(log[-1]), flush=True)
        return ok

    if not test(rows):
        raise SystemExit("the candidate does not fail with --match")
    lo, hi = 1, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if test(rows[:mid]):
            hi = mid
        else:
            lo = mid + 1
    rows = rows[:hi]
    # each arm / gripper back to its value at the start, when the stream still fails without its motion
    sess.reset()
    if st is not None:
        sess.set_state(st)
    start = sess.data.ctrl.copy()                   # the ctrl in force at the start state
    names = sess.actuators
    for side in ("left", "right"):
        for grp in ("arm", "gripper"):
            cols = [i for i, n in enumerate(names) if n.startswith(side) and (gripper_mask([n])[0] == (grp == "gripper"))]
            cand = rows.copy()
            cand[:, cols] = start[cols]
            if not np.array_equal(cand, rows) and test(cand):
                rows = cand
                log.append(dict(step=f"{side} {grp} held"))
    final = np.concatenate([rows, np.repeat(rows[-1:], 40, 0)])
    np.savez_compressed(a.out, ctrl=final, ckpt_row=np.int64(r0))
    meta = dict(source=a.rows, match=a.match, ckpt_row=r0, rows=len(final), core_rows=len(rows), log=log,
                sha256=hashlib.sha256(Path(a.out).read_bytes()).hexdigest())
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: v for k, v in meta.items() if k != "log"}))
    return 0


# ------------------------------------------------------------------ pool + report
def cmd_pool(a):
    """Run a jobs file (one JSON per line: task, strategy, seed, [ckpt, ref]) with `procs` workers, one fresh process per
    episode (nice 10). A process that dies without a result (a segfault, ...) gets a crash record: a BREAKDOWN."""
    jobs = [json.loads(l) for l in Path(a.jobs).read_text().splitlines() if l.strip()]
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (out / f"{episode_id(j['task'], j['strategy'], j['seed'])}.json").exists()]
    running = {}
    t0 = time.time()
    me = str(Path(__file__).resolve())

    def launch(j):
        eid = episode_id(j["task"], j["strategy"], j["seed"])
        cmd = ["nice", "-n", "10", sys.executable, me, "episode", "--task", j["task"], "--strategy", j["strategy"],
               "--seed", str(j["seed"]), "--out", str(out / f"{eid}.json"), "--rows-dir", str(out / "rows"),
               "--progress", str(out / "logs" / f"{eid}.progress"), "--min-s", str(j.get("min_s", 3.)),
               "--max-s", str(j.get("max_s", 10.))]
        for k in ("ckpt", "ref", "scene_dir"):
            if j.get(k):
                cmd += [f"--{k.replace('_', '-')}", j[k]]
        log = open(out / "logs" / f"{eid}.log", "w")
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), time.time(), j, eid, log

    while todo or running:
        while todo and len(running) < a.procs:
            p = launch(todo.pop(0))
            running[p[0].pid] = p
        time.sleep(1)
        for pid, (p, ts, j, eid, log) in list(running.items()):
            rc = p.poll()
            if rc is None and time.time() - ts > a.timeout:
                p.kill(); p.wait(); rc = "timeout"
            if rc is None:
                continue
            log.close()
            del running[pid]
            res = out / f"{eid}.json"
            if not res.exists():
                prog = out / "logs" / f"{eid}.progress"
                row = int(prog.read_text().strip() or 0) if prog.exists() else None
                sig = -rc if isinstance(rc, int) and rc < 0 else None
                crash = dict(id=eid, task=j["task"], kind="episode", strategy=j["strategy"], seed=j["seed"],
                             instance=f"{j['task']}/episode", rows_file=str(out / "rows" / f"{eid}.npz"),
                             fault=dict(error=f"process ended without a result: rc={rc}"
                                        + (f" ({signal.Signals(sig).name})" if sig else ""), row_after=row),
                             process_crash=dict(rc=rc, signal=signal.Signals(sig).name if sig else None,
                                                last_progress_row=row,
                                                log_tail=(out / "logs" / f"{eid}.log").read_text()[-3000:]),
                             verdict=dict(verdict="BREAKDOWN" if sig == signal.SIGSEGV else "ERROR",
                                          fails=[f"process crash rc={rc}"], triggers=[], notes=[]))
                _write(res, crash, quiet=True)
            with open(out / "_pool.jsonl", "a") as f:
                f.write(json.dumps(dict(id=eid, rc=rc, wall_s=round(time.time() - ts, 1))) + "\n")
    print(json.dumps(dict(jobs=len(jobs), wall_s=round(time.time() - t0, 1))))
    return 0


def cmd_jobs(a):
    """A jobs file for `pool`: per task (task=count), `count` episodes, the strategies in turn, seeds seed0 + 100000 * (the
    task's index in TASKS) + 0, 1, ... (the rows depend on the seed and strategy only: the same seed in two scenes would
    give the same random stream).
    {ckpt} / {ref} in --ckpt / --ref are replaced by the task."""
    out = []
    for spec in a.tasks:
        t, n = spec.split("=")
        for i in range(int(n)):
            s = a.strategies[i % len(a.strategies)]
            out.append(dict(task=t, strategy=s, seed=a.seed0 + 100000 * TASKS.index(t) + i, ckpt=a.ckpt.format(task=t), ref=a.ref.format(task=t),
                            min_s=a.min_s, max_s=a.max_s))
    Path(a.out).write_text("".join(json.dumps(j) + "\n" for j in out))
    print(f"{len(out)} jobs -> {a.out}")
    return 0


def mechanism_classes(r):
    """A BREAKDOWN's fail rules, each with its culprit, as short class keys for the report (a reading aid, not a verdict)."""
    import re
    v, o = r.get("verdict") or {}, r.get("observer") or {}
    ev = o.get("events") or {}
    ef = r.get("energy_forensics") or {}
    out = []
    if r.get("process_crash"):
        out.append(f"process crash {r['process_crash'].get('signal') or r['process_crash'].get('rc')}")
    for f in v.get("fails", []):
        if f.startswith("task depth") or f.startswith("robot depth"):
            k = "deep_task_contact" if f.startswith("task") else "deep_robot_contact"
            e = (ev.get(k) or [{}])[0]
            out.append(f"{f.split('>')[0]}>10mm {'-'.join(e.get('pair') or ['?'])}")
        elif f.startswith("energy_created(global)"):
            wc = ef.get("gmax_wc_change")
            out.append("energy global" + ("" if wc is None else " (no constraint work: drives' integration)" if abs(wc) < .05
                                          else " (with constraint work)"))
        elif f.startswith("energy_created(integrator)"):
            out.append("energy integrator residual")
        elif f.startswith("energy_created(local)"):
            m = re.search(r"contact (.+?) gave", f)
            c = m.group(1) if m else "?"
            c = re.sub(r"(left|right)_", "*_", c)
            out.append(f"energy local, culprit {c}")
        elif f.startswith("uncaused_jump"):
            e = (ev.get("jump_uncaused") or [{}])[0]
            out.append(f"uncaused jump {re.sub(r'_[0-9]+', '_N', str(e.get('body')))}")
        elif f == "tunnel":
            e = ((r.get("ikea_tunnel") or {}).get("events") or [{}])[0]
            out.append(f"tunnel {e.get('body')} in {e.get('slab')}")
        else:
            out.append(re.sub(r"[0-9.]+", "#", f)[:60])
    return out


def cmd_report(a):
    """Tables over result dirs. --confirmed DIR ...: an episode with a confirm record (<id>__confirm, from the birth in a
    fresh process) is read from it, the episode's state hash checked against it (confirm_sha_ok)."""
    conf = {}
    for d in a.confirmed or []:
        for p in Path(d).glob("*.json"):
            c = json.loads(p.read_text())
            conf[str(c.get("id", "")).replace("__confirm", "")] = c
    rows = []
    for d in a.dirs:
        for p in sorted(Path(d).glob("*.json")):
            if p.name.startswith("_"):
                continue
            r = json.loads(p.read_text())
            c = conf.get(r.get("id"))
            if c is not None and not r.get("process_crash"):
                c = dict(c, id=r["id"], kind=r.get("kind"), strategy=r.get("strategy"),
                         confirm_sha_ok=c.get("state_sha") == r.get("state_sha"))
                r = c
            if r.get("observer") is not None:          # re-read with this file's criteria (the same v2.13b + workspace)
                r["verdict"] = dict(classify(r), **({} if not r.get("harness_error") else dict(verdict="BREAKDOWN")))
            v = r.get("verdict") or {}
            rows.append(dict(id=r.get("id"), task=r.get("task"), kind=r.get("kind"), strategy=r.get("strategy"),
                             verdict=v.get("verdict", "?"), why="; ".join(v.get("fails", []) + v.get("triggers", [])),
                             rows=r.get("rows"), ms=r.get("ms_per_control"), wall=r.get("wall_s") or 0,
                             harness=r.get("harness_error"), ckpt_ok=r.get("ckpt_sha_ok", r.get("confirm_sha_ok")),
                             confirmed=r.get("confirm_sha_ok"),
                             obs_err=(r.get("observer") or {}).get("observer_errors"), ef=r.get("energy_forensics"),
                             classes=mechanism_classes(r) if v.get("verdict") == "BREAKDOWN" else []))
    tab = defaultdict(Counter)
    for r in rows:
        tab[(r["task"], r["strategy"] or r["kind"])][r["verdict"]] += 1
    lines = ["| task | strategy | n | CLEAN | VIOLENT | BREAKDOWN | other |", "|---|---|---|---|---|---|---|"]
    for (t, s), c in sorted(tab.items()):
        other = sum(v for k, v in c.items() if k not in ("CLEAN", "VIOLENT", "BREAKDOWN"))
        lines.append(f"| {t} | {s} | {sum(c.values())} | {c['CLEAN']} | {c['VIOLENT']} | {c['BREAKDOWN']} | {other} |")
    lines += ["", "| id | verdict | why | rows | ms/row |", "|---|---|---|---|---|"]
    for r in rows:
        if r["verdict"] != "CLEAN" or r["harness"] or r["obs_err"] or r["ckpt_ok"] is False:
            ef = r["ef"] or {}
            efs = ""
            if "energy_created(global)" in r["why"] and ef.get("gmax_at") is not None:
                efs = (f" [X > 0 ticks {ef['gmax_from']}-{ef['gmax_at']}: peak {ef['gmax']:.2f} J, Wc change "
                       f"{ef['gmax_wc_change']:+.3f} J, {ef['gmax_sat']} saturated drives]")
            elif "energy_created" in r["why"] and ef.get("at") is not None:
                efs = (f" [X rise {ef['rise']:.2f} J ticks {ef['from_tick']}-{ef['at']}, Wc change {ef['wc_change']:+.3f} J]")
            lines.append(f"| {r['id']} | {r['verdict']} | {r['why'][:300]}{efs} {r['harness'] or ''} | {r['rows']} | {r['ms']} |")
    cls = defaultdict(list)
    for r in rows:
        for c in set(r["classes"]):
            cls[c].append(r)
    lines += ["", "| BREAKDOWN mechanism (per fail rule) | episodes | scenes | examples |", "|---|---|---|---|"]
    for c, rs in sorted(cls.items(), key=lambda x: -len(x[1])):
        sc = Counter(x["task"] for x in rs)
        lines.append(f"| {c} | {len(rs)} | {', '.join(f'{k} {n}' for k, n in sorted(sc.items()))} | "
                     f"{', '.join(x['id'] for x in rs[:3])} |")
    if a.confirmed:
        bd = [r for r in rows if r["verdict"] == "BREAKDOWN"]
        lines.append(f"\nBREAKDOWN confirmed from the birth (same hash): {sum(r['confirmed'] is True for r in bd)} of "
                     f"{len(bd)}; hash differs: {[r['id'] for r in bd if r['confirmed'] is False]}; unconfirmed: "
                     f"{[r['id'] for r in bd if r['confirmed'] is None]}")
    tot = Counter(r["verdict"] for r in rows)
    lines.append(f"\ntotal {len(rows)}: {dict(tot)}; cpu-hours {sum(r['wall'] for r in rows) / 3600:.2f}")
    text = "\n".join(lines)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")
    return 0


def _write(path, rec, quiet=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, path)
    if not quiet:
        v = rec.get("verdict") or {}
        print(json.dumps(dict(id=rec.get("id"), rows=rec.get("rows"), ms=rec.get("ms_per_control"),
                              fault=bool(rec.get("fault")), sha=(rec.get("state_sha") or "")[:16],
                              verdict=v.get("verdict"), fails=v.get("fails"), triggers=v.get("triggers"),
                              harness_error=rec.get("harness_error"), ckpt_sha_ok=rec.get("ckpt_sha_ok"))), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checkpoints"); c.add_argument("--every", type=float, default=.05)
    e = sub.add_parser("episode"); e.add_argument("--strategy", required=True, choices=STRATEGIES)
    e.add_argument("--seed", type=int, required=True); e.add_argument("--ckpt")
    e.add_argument("--min-s", type=float, default=3.); e.add_argument("--max-s", type=float, default=10.)
    e.add_argument("--rows-dir"); e.add_argument("--keep-rows", action="store_true"); e.add_argument("--no-observe", action="store_true")
    e.add_argument("--progress")
    g = sub.add_parser("refseg"); g.add_argument("--ckpt", required=True); g.add_argument("--seg", type=int, required=True)
    n = sub.add_parser("nop"); n.add_argument("--rows", type=int, default=4000)
    f = sub.add_parser("confirm"); f.add_argument("--rows", required=True); f.add_argument("--ckpt"); f.add_argument("--progress")
    x = sub.add_parser("index"); x.add_argument("--task", required=True); x.add_argument("--ref", required=True)
    x.add_argument("--out", required=True); x.add_argument("--scene-dir")
    w = sub.add_parser("rows"); w.add_argument("--rows", required=True); w.add_argument("--ckpt")
    w.add_argument("--task", required=True); w.add_argument("--scene-dir"); w.add_argument("--out", required=True)
    k = sub.add_parser("shrink"); k.add_argument("--rows", required=True); k.add_argument("--ckpt")
    k.add_argument("--match", required=True)
    for x in (c, e, g, n, f, k):
        x.add_argument("--task", required=True)
        x.add_argument("--scene-dir")
        x.add_argument("--out", required=True)
        if x is not n and x is not k:
            x.add_argument("--ref", required=x in (c, g))
        if x is not c and x is not k:
            x.add_argument("--trace")
    p = sub.add_parser("pool"); p.add_argument("--jobs", required=True); p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=4); p.add_argument("--timeout", type=float, default=3600)
    r = sub.add_parser("report"); r.add_argument("dirs", nargs="+"); r.add_argument("--md")
    r.add_argument("--confirmed", nargs="+")
    j = sub.add_parser("jobs"); j.add_argument("--tasks", nargs="+", required=True, help="task=count ...")
    j.add_argument("--seed0", type=int, required=True); j.add_argument("--ckpt", required=True); j.add_argument("--ref", required=True)
    j.add_argument("--strategies", nargs="+", default=list(STRATEGIES)); j.add_argument("--out", required=True)
    j.add_argument("--min-s", type=float, default=3.); j.add_argument("--max-s", type=float, default=10.)
    a = ap.parse_args(argv)
    return dict(checkpoints=cmd_checkpoints, episode=cmd_episode, refseg=cmd_refseg, nop=cmd_nop, confirm=cmd_confirm,
                shrink=cmd_shrink, rows=cmd_rows, index=cmd_index, pool=cmd_pool, report=cmd_report, jobs=cmd_jobs)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
