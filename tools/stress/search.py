"""Fixed-size self-play search for physics breakdowns on one frozen MuJoCo instance, privileged Session.step: the
agent's own command path, with its clip, joint-speed clamp and table guard. One Session per process. Every episode starts
from Search.reset() (Session.reset() to the frozen birth, with MuJoCo's warning counters as in a fresh process), runs a
generated action stream to its end or to the episode's end, and is classified by verdict.py (criteria v2.11) under the
stress observer. After a faulted episode the Session is rebuilt.

usage: search.py --instance <instance.json> --out <dir> --tag <name> --seed <int> (--episodes <n> | --budget <controls>)
                 [--strategy random|directed|mutate|cem|prefix|periodic|mix] [--corpus <manifest.json>]
                 [--reference <privileged.npz>] [--experiment-opt ...]
--episodes stops after n episodes (acceptance rounds are sized in episodes); --budget stops once the controls charged
reach it; with both, whichever comes first.
Strategies:
  random    piecewise-constant random joint and gripper targets (uniform in the limits, or local), and oscillations
  directed  task-directed IK moves on every arm: pull, twist, shake, ram parts and fixtures, slam towards the table,
            throw (release while moving), regrasp; the family's mode list comes from the corpus manifest
  mutate    mutations of this instance's corpus streams and of the manifest's incidents: time compression, amplification
            about the birth targets, reversal, noise, splicing
  cem       the cross-entropy method over 5-waypoint target sequences, maximising the breakdown score
  periodic  sustained periodic inputs: 1-3 joints or grippers (a gripper half the time), each a sinusoid of its own
            frequency (0.5-10 Hz, or the 20 Hz Nyquist square wave at 40 Hz control), amplitude and phase, for 4-20 s.
            Slow, steady energy creation (a soft equality pumped by a gripper at 20 Hz: E1, 0.03 J/s) only shows after
            seconds of a periodic input; random and CEM search missed it in 1.5M controls.
  emptyclose  hard empty closes (claws W4 + rule A, CL8t): grippers commanded shut on nothing while the arm drives the
            hand into its own arm's base, the table or the other hand, then holds it there pressing, sometimes with the
            gripper chattering (BL1 e30's pattern: one hand's fingers pressed together at 40.7 kN on a degenerate row)
  prefix    mid-task states: replay the family's reference from the frozen birth and, at 2-5 random cut points k, run a
            perturbation (random, directed or a mutated stretch of the reference's continuation) from that exact
            state, i.e. episode = reference[:k] + perturbation. Each cut forks the process, so the reference is
            replayed once per pass; the budget is charged with the controls actually simulated (the pass's prefix once,
            plus every perturbation). Without a reference it falls back to random episodes from birth ("prefix-nr").
The episode sequence is a deterministic function of (instance, seed, strategy, corpus, code): the same arguments give the
same episodes, and a smaller budget or episode count gives a prefix of them.
Options from the environment (a round passes them to every worker; the summary records them):
  STRESS_CEM_SEEDED=1     each CEM restart starts from a corpus or incident stream (5 of its rows as the waypoints)
                          instead of the birth targets: CEM around known near-misses.
  STRESS_KEEP_ALL=1       keep every episode's rows in cand/ (not only the candidates): another profile can replay the
                          same streams, branch points included (blanket's GPU-vs-twin verdict cross-check).
  STRESS_OBJECTIVE=fingers  the search score also rewards finger-finger force (/ 2 kN), depth (/ 10 mm) and a small
                          A_ii on a finger row (0.1 / A_ii): it steers CEM and the kept VIOLENT candidates, never a verdict.
Outputs: <out>/<tag>.jsonl (one line per episode), <out>/<tag>.books.jsonl.gz (every episode's full observer summary:
the books and peaks the verdict reads, so any later reading can re-judge every episode, not only the kept ones),
<out>/<tag>.summary.json, and <out>/cand/<tag>_e<k>.npz (+ .json)
for every BREAKDOWN episode (the requested rows up to the episode's end), the 5 highest-scoring VIOLENT ones, and
every episode with a finger-finger force above 500 N, a finger row's A_ii below 0.1, or a degenerate-row signature (a
contact row with A_ii below 0.1 carrying more than 2 kN; when the observer measures them)."""
import _env  # noqa: F401  (before numpy: SIMD masking, threads, sys.path)
import argparse, json, os, sys, time, traceback
from pathlib import Path

import numpy as np

# directed-move modes when the corpus manifest names none for the task's family
DEFAULT_MODES = ["pull", "twist", "shake", "spin", "sweep", "ram", "slam", "regrasp"]


def load_corpus(manifest):
    """The corpus manifest: {"families": {task: {"directed_modes": [...]}}, "cases": [{"base", "input", "incident"}]}
    with "input" relative to the manifest's directory. -> (cases with absolute inputs, families)."""
    if not manifest:
        return [], {}
    m = json.loads(Path(manifest).read_text())
    root = Path(manifest).resolve().parent
    cases = [dict(e, input=str((root / e["input"]).resolve())) for e in m.get("cases", [])]
    return cases, m.get("families", {})


OBJECTIVE = os.environ.get("STRESS_OBJECTIVE", "")
FINGER_KEEP_N = 500.        # finger-finger force that keeps an episode's stream (core: a degenerate row shows as kN at ~1 mm)
FINGER_MIN_A = .1           # A_ii below this on a finger row: a degenerate row (claws W4 + rule A acceptance)
CEM_SEEDED = os.environ.get("STRESS_CEM_SEEDED", "") == "1"
KEEP_ALL = os.environ.get("STRESS_KEEP_ALL", "") == "1"


def score_of(o, verdict):
    pk = lambda k: (o["peaks"].get(k) or {}).get("value", 0.) if o else 0.
    parts = dict(speed=pk("task_speed") / 3, angular=pk("task_angular") / 60,
                 force=max(pk(f"force_{c}") for c in ("task_robot", "task_task", "task_static", "robot_static")) / 2000,
                 depth=max(pk(f"depth_{c}") for c in ("task_robot", "task_task", "task_static", "robot_static", "robot_robot")) / .010,
                 energy=max(pk("energy_excess_j") / .1,
                            min(pk("energy_rise_0p1s_j"), pk("energy_created_j"), pk("contact_net_work_j")) / .25),
                 jump=1 + pk("uncaused_jump_excess_m_s") if pk("uncaused_jump_excess_m_s") > 0 else 0.,
                 pop=pk("pop_speed") / 3)
    if OBJECTIVE == "fingers":
        parts["finger"] = max(pk("force_finger_finger") / 2000, pk("depth_finger_finger") / .010,
                              .1 * max(pk("ff_row_inv_A"), pk("finger_eqlim_inv_A")))
    return max(parts.values()) + (10. if verdict == "BREAKDOWN" else 0.), parts


def finger_peaks(o):
    """The episode's finger-finger force (N) and depth (mm) and the smallest A_ii of a finger row (contacts; joint
    limits and equalities), when the observer measured them."""
    pk = (o or {}).get("peaks") or {}
    out = {}
    n_deg = int(((o or {}).get("counts") or {}).get("degenerate_row", 0))
    if n_deg:
        out.update(degenerate_rows=n_deg, degenerate_force=round(pk["degenerate_force"]["value"], 1),
                   degenerate_where=pk["degenerate_force"]["where"])
    if "force_finger_finger" in pk:
        out.update(ff_force=round(pk["force_finger_finger"]["value"], 1), ff_depth_mm=round(1000 * pk["depth_finger_finger"]["value"], 3),
                   ff_where=pk["force_finger_finger"]["where"])
    for k, name in (("ff_row_inv_A", "ff_min_A"), ("finger_row_inv_A", "finger_min_A"), ("finger_eqlim_inv_A", "finger_eqlim_min_A")):
        if k in pk and pk[k]["value"] > 0:
            out[name] = round(1. / pk[k]["value"], 6)
            out[name + "_where"] = pk[k]["where"]
    return out


class Search:
    def __init__(self, instance, seed, corpus=None, experiment=None, reference=None):
        self.instance = str(instance)
        self.inst = json.loads(Path(instance).read_text())
        self.base = Path(instance).parent.name
        self.experiment = experiment
        self._build()
        self.rng = np.random.default_rng(seed)
        s, r = self.s, self.s.robot
        self.dim = s.action_dim
        self.lo, self.hi = r.lo.copy(), r.hi.copy()
        self.hold0 = s.hold_action().astype(np.float64)
        self.arms = list(r.arms)
        self.arm_slice = dict(r.arm_slice)
        self.grip = dict(r.grip_index)
        self.arm_cols = [i for i in range(self.dim) if i not in self.grip.values()]
        self.grip_open = {a: float(r.native_to_action(a, r.groups[a].grasp.open_position)) for a in self.arms}
        self.grip_closed = {a: float(r.native_to_action(a, r.groups[a].grasp.closed_position)) for a in self.arms}
        self.tcp0 = {a: s.ee_pose(a) for a in self.arms}
        w = s.env._world
        self.task_pos0 = {w.model.body(int(b)).name: np.array(w.data.xipos[int(b)]) for b in self.obs.task}
        self.static_pos = {}
        for name, e in w.entities.items():
            if name in ("robot", "table", "ground", "goal_region"):
                continue
            for l in e.links:
                if self.obs.cat[l.idx] == 0:
                    self.static_pos[f"{name}/{l.name}"] = np.array(w.data.xpos[l.idx])
        self.table_z = self.obs.table_z
        self.base_pos = {}                          # each arm's base link (emptyclose drives a hand into its own base)
        for l in w.entities["robot"].links:
            for a in self.arms:
                if l.name.startswith(f"{a}/") and "base_link" in l.name and a not in self.base_pos:
                    self.base_pos[a] = np.array(w.data.xpos[l.idx])
        cm, families = load_corpus(corpus)
        self.modes = list((families.get(self.inst["task"]) or {}).get("directed_modes") or DEFAULT_MODES)
        self.corpus = [np.load(e["input"])["actions"].astype(np.float64) for e in cm if e["base"] == self.base]
        self.incidents = [np.load(e["input"])["actions"].astype(np.float64) for e in cm if e.get("incident")
                          and np.load(e["input"])["actions"].shape[1] == self.dim]
        self.ref = None
        if reference:
            with np.load(reference, allow_pickle=False) as z:
                ref = np.asarray(z["actions"] if "actions" in z.files else z[z.files[0]], dtype=np.float64)
            if ref.ndim == 2 and ref.shape[1] == self.dim and len(ref) >= 2:
                self.ref = ref

    # ------------------------------------------------------------------ running
    def _build(self):
        """A new Session on the frozen instance and its observer; MuJoCo's warning counters as this fresh process has
        them (reset() puts them back before every episode)."""
        from rcb_topogym import Session
        from observer import Observer
        from replay import apply_options
        self.s = Session(self.inst["task"], instance=self.instance, strict=True)
        apply_options(self.s, self.experiment)     # forensics only, never a verdict (as replay.py --experiment-opt)
        self.obs = Observer(self.s)
        self.warn0 = [(int(w.number), int(w.lastinfo)) for w in self.s.env._world.data.warning]

    def reset(self):
        """The start of every episode: Session.reset() (the frozen birth), MuJoCo's warning counters back to this
        process's fresh values, the observer's per-episode reset. The counters are diagnostics outside
        mjSTATE_INTEGRATION, so the rewind keeps them, and fault detection reads them: the world raises when its
        instability counters change within a tick, but MuJoCo's auto-reset on a bad qacc zeroes every counter and then
        counts that warning once, so after one fault BADQACC stayed at 1 through every later blow-up in the process,
        MuJoCo reset the scene to qpos0 and the episode went on unflagged (K3 three times in one process, 2026-10-10:
        a fault, then twice 268 rows, 14 uncaused jumps, end c2a621e9). The model, every mjtState component and the
        session's own state are otherwise the same after the rewind as in a fresh process."""
        self.s.reset()
        for w, (n, info) in zip(self.s.env._world.data.warning, self.warn0):
            w.number, w.lastinfo = n, info
        self.obs.reset_run()

    def rebuild(self):
        """After a faulted episode: a new Session, so no fault path leaves anything behind (faults are rare)."""
        try:
            self.s.close()
        except Exception:                          # noqa: BLE001  (a faulted env may refuse; the new one is what counts)
            pass
        self._build()

    def run(self, rows):
        """One episode from the frozen birth. -> record (verdict, score, peaks) and the executed requested rows."""
        from verdict import classify
        self.reset()
        s, obs = self.s, self.obs
        fault, executed = None, 0
        t0 = time.time()
        try:
            for row in rows:
                executed += 1
                if s.step(np.asarray(row, dtype=np.float32)):
                    break
        except Exception as exc:
            fault = dict(error=repr(exc)[:300], cause=repr(exc.__cause__)[:300] if exc.__cause__ else None,
                         row=executed - 1)
        wall = time.time() - t0
        try:
            end = s.episode_end
        except Exception:
            end = None
        summ = obs.summary()
        res = dict(id="search", instance=self.instance, observer=summ, fault=fault, episode_end=end,
                   rows=len(rows), control_steps=int(s.n_steps), ms_per_control=round(1000 * wall / max(1, s.n_steps), 1),
                   slew_limited_steps=0, success=None)
        c = classify(res)
        sc, parts = score_of(summ, c["verdict"])
        steps = int(s.n_steps)
        if fault is not None:
            self.rebuild()
        return dict(verdict=c["verdict"], fails=c["fails"], triggers=c["triggers"], score=round(sc, 4),
                    parts={k: round(v, 4) for k, v in parts.items()}, steps=steps, end=end, fault=fault,
                    wall_s=round(wall, 2), speed=c["speed"], ang=c["ang"], task_force=c["task_force"], any_force=c["any_force"],
                    task_depth_mm=c["task_depth_mm"], energy_excess=c["energy_excess"], contact_net=c["contact_net"],
                    pops=c["pops"], grasp_slips=c["grasp_slips"], grasp_lost=c["grasp_lost"],
                    warnings=c.get("warnings") or [], **finger_peaks(summ)), \
            np.asarray(rows[:executed], dtype=np.float32), summ

    # ------------------------------------------------------------------ helpers
    def clipped(self, row):
        return np.clip(row, self.lo, self.hi)

    def hold(self, row, n):
        return [row.copy() for _ in range(max(0, int(n)))]

    def ik_row(self, base_row, arm, pos, quat):
        """base_row with `arm`'s joints solved (stateless IK seeded by base_row) for its TCP at (pos, quat wxyz)."""
        seed = np.asarray(base_row, float)[self.arm_cols]
        out = self.s.kinematics({"op": "ik", "targets": {f"{arm}_tcp": list(map(float, pos)) + list(map(float, quat))},
                                 "seed": seed.tolist()})
        row = base_row.copy()
        q = np.asarray(out["q"], float)
        if np.all(np.isfinite(q)):
            row[self.arm_cols] = q
        return self.clipped(row)

    def rand_dir(self):
        v = self.rng.normal(size=3)
        return v / np.linalg.norm(v)

    def rot_quat(self, quat, angle, axis):
        from scipy.spatial.transform import Rotation
        r = Rotation.from_rotvec(axis * angle) * Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
        x = r.as_quat()
        return np.array([x[3], x[0], x[1], x[2]])

    def ramp(self, a, b, n):
        n = max(1, int(n))
        return [a + (b - a) * (k + 1) / n for k in range(n)]

    # ------------------------------------------------------------------ generators
    def gen_random(self, start=None):
        rng, row, rows = self.rng, (self.hold0 if start is None else np.asarray(start, float)).copy(), []
        L = int(rng.integers(80, 320))
        if rng.random() < .3:                       # oscillation of a few joints between two values
            cols = rng.choice(np.arange(self.dim), size=int(rng.integers(1, 4)), replace=False)
            a, b = row.copy(), row.copy()
            for c in cols:
                a[c], b[c] = rng.uniform(self.lo[c], self.hi[c], size=2)
            period = int(rng.integers(2, 24))
            for k in range(L):
                rows.append(a.copy() if (k // max(1, period // 2)) % 2 == 0 else b.copy())
            return rows + self.hold(rows[-1], 40)
        while len(rows) < L:
            for c in range(self.dim):
                if rng.random() < .5:
                    if rng.random() < .4:
                        row[c] = rng.uniform(self.lo[c], self.hi[c])
                    else:
                        row[c] = row[c] + rng.normal(0, rng.choice([.05, .2, .6]) * (self.hi[c] - self.lo[c]) / 2)
            row = self.clipped(row)
            rows += self.hold(row, int(np.exp(rng.uniform(0, np.log(40)))))
        return rows + self.hold(row, 40)

    def gen_directed(self, start=None):
        rng, row, rows = self.rng, (self.hold0 if start is None else np.asarray(start, float)).copy(), []
        for _ in range(int(rng.integers(1, 5))):
            mode = rng.choice(self.modes)
            n = int(rng.choice([1, 2, 4, 8, 16, 40]))
            arms = list(rng.permutation(self.arms))[:int(rng.integers(1, len(self.arms) + 1))]
            cur = {a: self.s_fk_tcp(row, a) for a in self.arms}
            if mode in ("pull", "slam", "ram", "sweep"):
                new = row.copy()
                for a in arms:
                    p, q = cur[a]
                    if mode == "pull":
                        target = p + self.rand_dir() * rng.uniform(.02, .25)
                    elif mode == "slam":
                        target = p.copy(); target[2] = self.table_z - rng.uniform(0, .05); target[:2] += rng.normal(0, .05, 2)
                    elif mode == "ram":
                        pts = list(self.task_pos0.values()) + list(self.static_pos.values())
                        # no MuJoCo task or static body to ram (blanket: the cloth is not a MuJoCo body): a pull instead
                        target = (pts[int(rng.integers(len(pts)))] + rng.normal(0, .03, 3) if pts
                                  else p + self.rand_dir() * rng.uniform(.02, .25))
                    else:
                        target = p + self.rand_dir() * rng.uniform(.1, .4)
                    new = self.ik_row(new, a, target, q)
                rows += self.ramp(row, new, n); row = new
            elif mode in ("twist", "spin"):
                new = row.copy()
                for a in arms:
                    if mode == "spin":
                        c = self.arm_slice[a].stop - 1
                        new[c] = rng.uniform(self.lo[c], self.hi[c])
                    else:
                        p, q = cur[a]
                        new = self.ik_row(new, a, p, self.rot_quat(q, rng.uniform(-1.2, 1.2), self.rand_dir()))
                rows += self.ramp(row, new, n); row = new
            elif mode == "shake":
                a = arms[0]; p, q = cur[a]; d = self.rand_dir() * rng.uniform(.005, .05)
                plus, minus = self.ik_row(row, a, p + d, q), self.ik_row(row, a, p - d, q)
                period = int(rng.integers(1, 8))
                for k in range(int(rng.integers(10, 80))):
                    rows.append(plus.copy() if (k // period) % 2 == 0 else minus.copy())
                row = rows[-1].copy()
            elif mode in ("throw", "regrasp"):
                a = arms[0]; p, q = cur[a]
                moved = self.ik_row(row, a, p + self.rand_dir() * rng.uniform(.05, .3), q)
                k_open = int(rng.integers(1, max(2, n)))
                part = self.ramp(row, moved, n)
                for k, rr in enumerate(part):
                    if k >= k_open:
                        rr[self.grip[a]] = self.grip_open[a]
                rows += part; row = part[-1].copy()
                if mode == "regrasp":
                    back = self.ik_row(row, a, p, q); back[self.grip[a]] = self.grip_open[a]
                    rows += self.ramp(row, back, int(rng.integers(1, 40))); row = back.copy()
                    row[self.grip[a]] = self.grip_closed[a]
                    rows += self.hold(row, int(rng.integers(5, 30)))
            rows += self.hold(row, int(rng.integers(0, 40)))
        return rows + self.hold(row, 40)

    def gen_emptyclose(self, start=None):
        """Hard empty closes (CL8t): 1-3 bouts of: close one hand's gripper (sometimes both) to the command limit on
        nothing, drive that hand into its own arm's base, the table or the other hand (5 cm past it at most; the other
        hand may come to meet it), and hold it there pressing for 10-120 controls, sometimes with the gripper chattering
        open-shut at 4-20 Hz."""
        rng, row, rows = self.rng, (self.hold0 if start is None else np.asarray(start, float)).copy(), []
        shut = {a: (self.lo if self.grip_closed[a] < self.grip_open[a] else self.hi)[self.grip[a]] for a in self.arms}
        for _ in range(int(rng.integers(1, 4))):
            arm = self.arms[int(rng.integers(len(self.arms)))]
            others = [a for a in self.arms if a != arm]
            new = row.copy()
            for a in [arm] + (others if rng.random() < .3 else []):
                new[self.grip[a]] = shut[a]
            rows += self.ramp(row, new, int(rng.choice([1, 2, 4, 8]))); row = new
            p, q = self.s_fk_tcp(row, arm)
            kinds = ["table"] + (["base"] if arm in self.base_pos else []) + (["other"] if others else [])
            kind = kinds[int(rng.integers(len(kinds)))]
            if kind == "table":
                target = p.copy(); target[2] = self.table_z; target[:2] += rng.normal(0, .03, 2)
            elif kind == "base":
                target = self.base_pos[arm] + rng.normal(0, .03, 3)
            else:
                o = others[int(rng.integers(len(others)))]
                target = self.s_fk_tcp(row, o)[0] + rng.normal(0, .02, 3)
            d = target - p
            target = target + d / max(float(np.linalg.norm(d)), 1e-6) * rng.uniform(0., .05)     # past the obstacle
            new = self.ik_row(row, arm, target, q)
            if kind == "other" and rng.random() < .5:                                           # the other hand meets it
                po, qo = self.s_fk_tcp(row, o)
                new = self.ik_row(new, o, p + rng.normal(0, .02, 3), qo)
            rows += self.ramp(row, new, int(rng.choice([2, 4, 8, 16, 40]))); row = new
            n_hold = int(rng.integers(10, 121))
            if rng.random() < .4:
                period = int(rng.integers(1, 6))
                for k in range(n_hold):
                    r = row.copy()
                    if (k // period) % 2:
                        r[self.grip[arm]] = self.grip_open[arm]
                    rows.append(r)
            else:
                rows += self.hold(row, n_hold)
        return rows + self.hold(row, 40)

    def encode(self, a):
        """A stream as CEM's waypoint vector: M of its rows (sorted random picks), each held for its spacing (1-40)."""
        a = np.asarray(a, float)
        idx = np.sort(self.rng.choice(len(a), size=min(self.M, len(a)), replace=False))
        while len(idx) < self.M:
            idx = np.append(idx, idx[-1])
        gaps = np.diff(np.append(idx, len(a)))
        x = np.zeros((self.M, self.dim + 1))
        x[:, :self.dim] = (a[idx] - self.lo) / np.maximum(self.hi - self.lo, 1e-9)
        x[:, self.dim] = (np.clip(gaps, 1, 40) - 1) / 39.
        return np.clip(x.reshape(-1), 0, 1)

    def s_fk_tcp(self, row, arm):
        """TCP pose (pos, quat wxyz) of the commanded arm joints (stateless FK on the robot model)."""
        q = np.asarray(row, float)[self.arm_cols]
        f = np.asarray(self.s.kinematics({"op": "fk", "q": q.tolist(), "links": [f"{arm}_tcp"]})["poses"][f"{arm}_tcp"], float)
        return f[:3], f[3:7]

    def gen_mutate(self):
        rng = self.rng
        pool = self.corpus + self.incidents
        if not pool:
            return self.gen_random()
        a = pool[int(rng.integers(len(pool)))].copy()
        if len(a) > 400:
            k0 = int(rng.integers(0, len(a) - 300)); a = a[k0:k0 + 300]   # a window (its start state differs; fine)
        for _ in range(int(rng.integers(1, 3))):
            m = rng.choice(["compress", "amplify", "reverse", "noise", "splice"])
            if m == "compress":
                a = a[::int(rng.integers(2, 6))]
            elif m == "amplify":
                a = self.hold0[None] + (a - self.hold0[None]) * rng.uniform(1.3, 2.5)
            elif m == "reverse":
                a = a[::-1]
            elif m == "noise":
                a = a + rng.normal(0, rng.choice([.02, .1, .3]), a.shape) * (self.hi - self.lo)[None] / 2
            else:
                b = pool[int(rng.integers(len(pool)))]
                k = int(rng.integers(1, max(2, len(b))))
                a = np.vstack([a, b[:k]])
        a = np.clip(a, self.lo[None], self.hi[None])
        return list(a) + self.hold(a[-1], 40)

    def gen_periodic(self, start=None):
        rng = self.rng
        row0 = (self.hold0 if start is None else np.asarray(start, float)).copy()
        hz = float(self.inst.get("control_hz") or 40)
        L = int(rng.integers(160, 800))
        cols = [int(c) for c in rng.choice(np.arange(self.dim), size=int(rng.integers(1, 4)), replace=False)]
        if rng.random() < .5:                       # favour a gripper: finger couplings are the usual pumps
            g = int(self.grip[self.arms[int(rng.integers(len(self.arms)))]])
            if g not in cols:
                cols[0] = g
        waves = []
        for c in cols:
            f = float(rng.choice([.5, 1., 2., 5., 10., hz / 2]))
            half = (self.hi[c] - self.lo[c]) / 2
            amp = float(rng.choice([.05, .15, .3, .5])) * half
            center = row0[c] if rng.random() < .5 else float(rng.uniform(self.lo[c] + amp, self.hi[c] - amp))
            waves.append((c, f, amp, center, float(rng.uniform(0, 2 * np.pi))))
        rows = []
        for t in range(L):
            row = row0.copy()
            for c, f, amp, center, phase in waves:
                if f >= hz / 2:                     # the Nyquist frequency: a square wave (a sine would sample one phase)
                    row[c] = center + amp * (1. if t % 2 == 0 else -1.)
                else:
                    row[c] = center + amp * np.sin(2 * np.pi * f * t / hz + phase)
            rows.append(self.clipped(row))
        return rows + self.hold(rows[-1], 40)

    # ------------------------------------------------------------------ prefix: mid-task states
    def gen_ref_mutation(self, k):
        """A mutated stretch of the reference's continuation after row k: amplified about the row at the cut, noise,
        time compression or reversal."""
        rng, start = self.rng, self.ref[k - 1]
        a = self.ref[k:k + int(rng.integers(40, 300))].copy()
        if not len(a):
            a = start[None].repeat(40, axis=0)
        for _ in range(int(rng.integers(1, 3))):
            m = rng.choice(["compress", "amplify", "reverse", "noise"])
            if m == "compress":
                a = a[::int(rng.integers(2, 6))]
            elif m == "amplify":
                a = start[None] + (a - start[None]) * rng.uniform(1.3, 4.0)
            elif m == "reverse":
                a = a[::-1]
            else:
                a = a + rng.normal(0, rng.choice([.02, .1, .3]), a.shape) * (self.hi - self.lo)[None] / 2
        a = np.clip(a, self.lo[None], self.hi[None])
        return list(a) + self.hold(a[-1], 40)

    def gen_tail(self, k):
        mode = self.rng.choice(["random", "directed", "mutate"])
        if mode == "random":
            return self.gen_random(start=self.ref[k - 1])
        if mode == "directed":
            return self.gen_directed(start=self.ref[k - 1])
        return self.gen_ref_mutation(k)

    def episode_record(self, rows_executed, wall, fault):
        """The verdict record of the episode so far (observer summary from its birth), as run() returns it."""
        from verdict import classify
        s, obs = self.s, self.obs
        try:
            end = s.episode_end
        except Exception:
            end = None
        summ = obs.summary()
        res = dict(id="search", instance=self.instance, observer=summ, fault=fault, episode_end=end,
                   rows=len(rows_executed), control_steps=int(s.n_steps), ms_per_control=round(1000 * wall / max(1, s.n_steps), 1),
                   slew_limited_steps=0, success=None)
        c = classify(res)
        sc, parts = score_of(summ, c["verdict"])
        return dict(verdict=c["verdict"], fails=c["fails"], triggers=c["triggers"], score=round(sc, 4),
                    parts={k: round(v, 4) for k, v in parts.items()}, steps=int(s.n_steps), end=end, fault=fault,
                    wall_s=round(wall, 2), speed=c["speed"], ang=c["ang"], task_force=c["task_force"], any_force=c["any_force"],
                    task_depth_mm=c["task_depth_mm"], energy_excess=c["energy_excess"], contact_net=c["contact_net"],
                    pops=c["pops"], grasp_slips=c["grasp_slips"], grasp_lost=c["grasp_lost"],
                    warnings=c.get("warnings") or [], **finger_peaks(summ)), summ

    def prefix_pass(self, record, budget_left=None, episodes_left=None):
        """One pass along the reference: replay it from the frozen birth; at 2-5 random cut points k (drawn before the
        pass), generate a perturbation tail in this process, then fork: the child runs the tail from the exact state
        at k and returns its record, while this process goes on along the reference. record(rec, rows, summ, charge)
        is called per episode, with rows = reference[:k] + the executed tail and charge = the controls simulated for it
        (the prefix advanced since the last cut, plus the tail). The pass stops early once it has charged budget_left
        controls or recorded episodes_left episodes (None: no limit). Returns the controls charged."""
        import pickle
        s, ref, rng = self.s, self.ref, self.rng
        n_cuts = min(int(rng.integers(2, 6)), len(ref) - 1)
        cuts = np.sort(rng.choice(np.arange(1, len(ref)), size=n_cuts, replace=False))
        self.reset()
        done, charged, n_ep = 0, 0, 0
        for k in cuts:
            if (budget_left is not None and charged >= budget_left) or (episodes_left is not None and n_ep >= episodes_left):
                break
            advanced, ended = 0, False
            while done < k:
                ended = bool(s.step(np.asarray(ref[done], dtype=np.float32)))
                done += 1
                advanced += 1
                if ended:
                    break
            if ended:                              # the reference ended its own episode before this cut
                charged += advanced
                break
            tail = self.gen_tail(int(k))
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:                           # the child: the tail from this exact state, then exit at once
                os.close(r)
                try:
                    fault, executed, t0 = None, 0, time.time()
                    try:
                        for row in tail:
                            executed += 1
                            if s.step(np.asarray(row, dtype=np.float32)):
                                break
                    except Exception as exc:
                        fault = dict(error=repr(exc)[:300], cause=repr(exc.__cause__)[:300] if exc.__cause__ else None,
                                     row=int(k) + executed - 1)
                    rows = np.concatenate([np.asarray(ref[:k], dtype=np.float32), np.asarray(tail[:executed], dtype=np.float32)])
                    rec, summ = self.episode_record(rows, time.time() - t0, fault)
                    payload = pickle.dumps((rec, rows, summ, executed))
                except BaseException as exc:      # noqa: BLE001  (report anything to the parent)
                    payload = pickle.dumps(("error", repr(exc)[:300]))
                with os.fdopen(w, "wb") as f:
                    f.write(payload)
                os._exit(0)
            os.close(w)
            with os.fdopen(r, "rb") as f:
                data = f.read()
            os.waitpid(pid, 0)
            got = pickle.loads(data) if data else ("error", "no data from the child")
            if got[0] == "error":
                raise RuntimeError(f"prefix child at row {k}: {got[1]}")
            rec, rows, summ, executed = got
            rec.update(prefix_rows=int(k))
            charge = advanced + executed
            charged += charge
            n_ep += 1
            record(rec, rows, summ, charge)
        return charged

    # CEM over waypoint sequences
    M = 5

    def decode(self, x):
        x = np.clip(x.reshape(self.M, self.dim + 1), 0, 1)
        rows, row = [], self.hold0.copy()
        for j in range(self.M):
            row = self.lo + x[j, :self.dim] * (self.hi - self.lo)
            rows += self.hold(row, 1 + int(round(x[j, self.dim] * 39)))
        return rows + self.hold(row, 40)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instance", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, help="control steps charged (a cap)")
    ap.add_argument("--episodes", type=int, help="episodes (the size of an acceptance worker)")
    ap.add_argument("--strategy", default="mix")
    ap.add_argument("--corpus", help="corpus manifest.json (corpus streams to mutate, incidents, the family's directed modes)")
    ap.add_argument("--reference", help="the instance's reference stream (privileged.npz): the prefix strategy's route")
    ap.add_argument("--experiment-opt", default=os.environ.get("STRESS_EXPERIMENT_OPT"))
    a = ap.parse_args()
    if a.budget is None and a.episodes is None:
        ap.error("give --episodes or --budget")
    out = Path(a.out)
    (out / "cand").mkdir(parents=True, exist_ok=True)
    S = Search(a.instance, a.seed, corpus=a.corpus, experiment=a.experiment_opt, reference=a.reference)
    log = open(out / f"{a.tag}.jsonl", "a")
    import gzip
    books = gzip.open(out / f"{a.tag}.books.jsonl.gz", "at")
    used, ep, t0 = 0, 0, time.time()
    counts = {"BREAKDOWN": 0, "VIOLENT": 0, "CLEAN": 0}
    top_violent = []
    cem = None

    def record(rec, rows, summ, gen, charge=None):
        nonlocal used, ep
        used += rec["steps"] if charge is None else charge
        rec.update(ep=ep, gen=gen, used=used, t=round(time.time() - t0, 1))
        counts[rec["verdict"]] += 1
        log.write(json.dumps(rec, default=str) + "\n"); log.flush()
        books.write(json.dumps(dict(ep=ep, gen=gen, steps=rec["steps"], end=rec.get("end"), fault=rec.get("fault"),
                                    observer=summ), default=str) + "\n"); books.flush()
        keep = rec["verdict"] == "BREAKDOWN"
        if rec["verdict"] == "VIOLENT":
            top_violent.append((rec["score"], ep))
            top_violent.sort(reverse=True)
            keep = (rec["score"], ep) in top_violent[:5]
        # a finger pair pressed past FINGER_KEEP_N, or a finger row with A_ii below FINGER_MIN_A: kept for tracing
        keep = keep or rec.get("ff_force", 0) > FINGER_KEEP_N or rec.get("finger_min_A", 1e9) < FINGER_MIN_A \
            or rec.get("finger_eqlim_min_A", 1e9) < FINGER_MIN_A or rec.get("degenerate_rows", 0) > 0
        keep = keep or KEEP_ALL                 # STRESS_KEEP_ALL=1: every episode's rows (a cross-profile check replays them)
        if keep:
            stem = out / "cand" / f"{a.tag}_e{ep}"
            np.savez(str(stem) + ".npz", actions=rows)
            Path(str(stem) + ".json").write_text(json.dumps(dict(rec, instance=a.instance, observer=summ), default=str, indent=1))
        ep += 1

    strategies = ["random", "directed", "mutate", "cem"] if a.strategy == "mix" else [a.strategy]

    def more():
        return (a.budget is None or used < a.budget) and (a.episodes is None or ep < a.episodes)

    k = 0
    while more():
        strat = strategies[k % len(strategies)]
        k += 1
        try:
            if strat == "cem":
                # one CEM generation per turn: population 12, elites 3; restart after 8 generations
                if cem is None or cem["gen"] >= 8:
                    seeds = S.corpus + S.incidents if CEM_SEEDED else []
                    if seeds:                      # STRESS_CEM_SEEDED: around a known near-miss
                        mu = S.encode(seeds[int(S.rng.integers(len(seeds)))])
                        cem = dict(mu=mu, sd=np.full(mu.shape, .15), gen=0)
                    else:
                        mu = np.concatenate([np.tile(np.r_[(S.hold0 - S.lo) / np.maximum(S.hi - S.lo, 1e-9), .5], S.M)])
                        mu = np.clip(mu + S.rng.normal(0, .15, mu.shape), 0, 1)
                        cem = dict(mu=mu, sd=np.full(mu.shape, .3), gen=0)
                pop = []
                for _ in range(12):
                    if not more():
                        break
                    x = np.clip(cem["mu"] + cem["sd"] * S.rng.normal(size=cem["mu"].shape), 0, 1)
                    rec, rows, summ = S.run(S.decode(x))
                    record(rec, rows, summ, f"cem{cem['gen']}")
                    pop.append((rec["score"], x))
                if len(pop) >= 3:
                    pop.sort(key=lambda t: -t[0])
                    el = np.array([x for _, x in pop[:3]])
                    cem.update(mu=el.mean(axis=0), sd=np.maximum(el.std(axis=0), .05), gen=cem["gen"] + 1)
            elif strat == "prefix" and S.ref is not None:
                S.prefix_pass(lambda rec, rows, summ, charge: record(rec, rows, summ, "prefix", charge),
                              None if a.budget is None else a.budget - used, None if a.episodes is None else a.episodes - ep)
            elif strat == "prefix":                # no reference: random episodes from birth, labelled as such
                rec, rows, summ = S.run(S.gen_random())
                record(rec, rows, summ, "prefix-nr")
            else:
                rows = getattr(S, f"gen_{strat}")()
                rec, rows, summ = S.run(rows)
                record(rec, rows, summ, strat)
        except Exception as exc:
            log.write(json.dumps(dict(error=repr(exc)[:300], strategy=strat, trace=traceback.format_exc()[-800:])) + "\n")
            log.flush()
            try:
                S.rebuild()
            except Exception:
                break
    summary = dict(tag=a.tag, instance=a.instance, seed=a.seed, strategy=a.strategy, budget_controls=a.budget,
                   episodes_quota=a.episodes, used_controls=used, episodes=ep, wall_s=round(time.time() - t0, 1), counts=counts,
                   corpus=a.corpus, reference=a.reference, experiment_opt=a.experiment_opt, provenance=_env.provenance(),
                   objective=OBJECTIVE or None, cem_seeded=CEM_SEEDED, keep_all=KEEP_ALL,
                   joint_speed_limits={k: list(v) for k, v in __import__("topogym.world.mujoco.robot", fromlist=["x"]).JOINT_SPEED_LIMITS_RAD_S.items()})
    books.close()
    (out / f"{a.tag}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
