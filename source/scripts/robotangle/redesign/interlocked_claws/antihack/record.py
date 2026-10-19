"""Replay a row stream open-loop in a MuJoCo Session and record what the anti-hack metrics need.

Reads only (never changes the physics): per control row the effective commanded targets (after the Session's clamps),
the measured joints, both TCP poses, the grasp-assist holdings, finger-object contacts, the task's success flag; every
`--every` rows (and at the first / last row) the object states: rope chain links (positions, quaternions, nodes),
cloth flex vertices, movable rigid bodies; the task's own predicate metrics (scalars); the grasp manager's events.

usage (on the cloud host, env sourced; through launch.sh when longer than ~1 min):
  python -m antihack.record <instance.json> <rows.npz | nop:N> <out.npz> [--every 10] [--hold N] [--no-strict]
         [--max-rows N] [--rows-key actions]
writes <out.npz> (arrays) and <out.json> (summary: success, first success row, episode end, state hashes, events).

The <rows.npz> may be a privileged.npz (actions, action_layout) or any npz with an `actions` (T, 14) array; its
layout must equal the Session's (checked when the file carries action_layout).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _scalars(metrics, prefix="", out=None, depth=0):
    """The JSON-able scalar leaves (and short numeric lists) of a task's metrics dict."""
    out = {} if out is None else out
    if not isinstance(metrics, dict) or depth > 2:
        return out
    for k, v in metrics.items():
        key = f"{prefix}{k}"
        if k == "crossing_sequence" and isinstance(v, (list, tuple)):      # rope_threading: [(ring, sign), ...]
            out[key] = [[str(x[0]), int(x[1])] for x in v][:32]
            continue
        if isinstance(v, (bool, np.bool_)):
            out[key] = bool(v)
        elif isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)):
            out[key] = float(v) if isinstance(v, (float, np.floating)) else int(v)
        elif isinstance(v, str) and len(v) <= 64:
            out[key] = v
        elif isinstance(v, (list, tuple)) and 0 < len(v) <= 16 and all(
                isinstance(x, (int, float, bool, np.integer, np.floating)) for x in v):
            out[key] = [float(x) for x in v]
        elif isinstance(v, dict):
            _scalars(v, key + ".", out, depth + 1)
    return out


class Recorder:
    """Attach to a built Session; call snapshot(row) / per_row(row) as the rows run."""

    def __init__(self, session):
        import mujoco
        self.mj = mujoco
        self.s = session
        self.w = session.env._world
        self.m, self.d = self.w.model, self.w.data
        self.ropes = dict(getattr(self.w, "_ropes", {}) or {})
        self.cloths = dict(getattr(self.w, "_cloths", {}) or {})
        ents = session._entities()
        self.rigid = {}
        for name, ent in ents.items():
            if name in self.ropes or name in self.cloths:
                continue
            fixed = bool(getattr(getattr(ent, "morph", None), "fixed", False))
            links = list(getattr(ent, "links", []) or [])
            if links and not fixed:
                self.rigid[name] = int(links[0].idx)
        # contact bookkeeping: finger bodies per arm, object bodies per entity
        m = self.m
        self.fingers = {}
        for arm in session.robot.arms:
            ids = []
            for f in ("left_finger_link", "right_finger_link"):
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{arm}/{f}")
                if b >= 0:
                    ids.append(b)
            self.fingers[arm] = ids
        self.obj_body = {}
        for name, r in self.ropes.items():
            for b in r.bodies:
                self.obj_body[int(b)] = name
        for name, c in self.cloths.items():
            for b in c.bodies:
                self.obj_body[int(b)] = name
        for name, b in self.rigid.items():
            ent = ents[name]
            for l in ent.links:
                self.obj_body[int(l.idx)] = name
        # static scene obstacles (fixed task entities: rings, stands, posts, weights ...; not the table / ground)
        self.robot_bodies = {int(l.idx) for l in self.w.entities["robot"].links} if "robot" in self.w.entities else set()
        self.obstacle_body = {}
        for name, ent in ents.items():
            if name in self.ropes or name in self.cloths or name in self.rigid:
                continue
            for l in getattr(ent, "links", []) or []:
                self.obstacle_body[int(l.idx)] = name
        self.table_bodies = {int(l.idx) for n in ("table", "ground") if n in self.w.entities
                             for l in self.w.entities[n].links}
        self.events, self._seen = [], set()
        self.samples = []          # (row, {key: array})
        self.metric_samples = []   # (row, {metric: value})

    # -- object state
    def objects(self) -> dict:
        d = self.d
        out = {}
        for name, r in self.ropes.items():
            out[f"rope/{name}/pos"] = np.asarray(d.xpos)[r.bodies].astype(np.float64).copy()
            out[f"rope/{name}/quat"] = np.asarray(d.xquat)[r.bodies].astype(np.float64).copy()
            out[f"rope/{name}/nodes"] = r.node_positions(d).astype(np.float64)
        for name, c in self.cloths.items():
            out[f"cloth/{name}/verts"] = c.vertices(d).astype(np.float64)
        for name, b in self.rigid.items():
            out[f"rigid/{name}/pose"] = np.r_[np.asarray(d.xpos[b]), np.asarray(d.xquat[b])].astype(np.float64)
        return out

    def snapshot(self, row: int) -> None:
        self.samples.append((int(row), self.objects()))
        try:
            self.metric_samples.append((int(row), _scalars(dict(self.s.env._task.result.metrics))))
        except Exception:  # noqa: BLE001 - metrics are evidence only
            self.metric_samples.append((int(row), {}))

    def grasp_events(self) -> None:
        gm = getattr(self.w, "_grasp_manager", None)
        for ev in (gm.events if gm is not None else []):
            seq = ev.get("sequence")
            if seq in self._seen:
                continue
            self._seen.add(seq)
            rec = {k: ev[k] for k in ("physics_step", "hand", "event", "reason") if k in ev}
            tgt = ev.get("target")
            if tgt is None and isinstance(ev.get("binding"), dict):
                tgt = ev["binding"].get("target")
            if tgt is not None:
                rec["target"] = [str(x) for x in (tgt if isinstance(tgt, (list, tuple)) else [tgt])]
            self.events.append(rec)

    def collisions(self) -> tuple:
        """(robot-obstacle contacts, deepest robot-obstacle penetration m, robot-table contacts, object-obstacle
        contacts) at the end of the tick."""
        m, d = self.m, self.d
        n = int(d.ncon)
        if n == 0:
            return 0, 0.0, 0, 0
        b1 = np.asarray(m.geom_bodyid)[np.asarray(d.contact.geom1[:n])]
        b2 = np.asarray(m.geom_bodyid)[np.asarray(d.contact.geom2[:n])]
        dist = np.asarray(d.contact.dist[:n])
        ro = rt = oo = 0
        deep = 0.0
        for x, y, dd in zip(b1, b2, dist):
            x, y = int(x), int(y)
            rx, ry = x in self.robot_bodies, y in self.robot_bodies
            if (rx and y in self.obstacle_body) or (ry and x in self.obstacle_body):
                ro += 1; deep = max(deep, -float(dd))
            elif (rx and y in self.table_bodies) or (ry and x in self.table_bodies):
                rt += 1
            elif (x in self.obj_body and y in self.obstacle_body) or (y in self.obj_body and x in self.obstacle_body):
                oo += 1
        return ro, deep, rt, oo

    def finger_contacts(self) -> dict:
        """{arm: (contacts of its fingers with task objects, fingers touching an object (0-2), object name or '')}."""
        m, d = self.m, self.d
        n = int(d.ncon)
        res = {}
        if n == 0:
            return {arm: (0, 0, "") for arm in self.fingers}
        g1 = np.asarray(d.contact.geom1[:n]); g2 = np.asarray(d.contact.geom2[:n])
        b1 = np.asarray(m.geom_bodyid)[g1]; b2 = np.asarray(m.geom_bodyid)[g2]
        for arm, fb in self.fingers.items():
            count, touching, obj = 0, set(), ""
            for x, y in zip(b1, b2):
                x, y = int(x), int(y)
                if x in fb and y in self.obj_body:
                    count += 1; touching.add(x); obj = self.obj_body[y]
                elif y in fb and x in self.obj_body:
                    count += 1; touching.add(y); obj = self.obj_body[x]
            res[arm] = (count, len(touching), obj)
        return res


def load_rows(spec: str, session, key: str = "actions"):
    if spec.startswith("nop:"):
        return np.repeat(session.hold_action()[None].astype(np.float32), int(spec[4:]), axis=0), None
    z = np.load(spec, allow_pickle=False)
    rows = z[key].astype(np.float32)
    layout = z["action_layout"].tolist() if "action_layout" in z.files else None
    if layout is not None and [str(x) for x in layout] != list(session.action_layout):
        raise ValueError(f"row layout {layout} != session layout {session.action_layout}")
    ticks = z["physics_steps_per_action"] if "physics_steps_per_action" in z.files else None
    return rows, ticks


def run(session, rows, out_npz, *, every: int = 10, hold: int = 0, label: str = "", ticks=None,
        stop_at_end: bool = True, extra: dict | None = None) -> dict:
    """Run `rows` (T, action_dim) from the session's frozen start (reset first) and record. Returns the summary."""
    s = session
    s.reset()
    rec = Recorder(s)
    rows = np.asarray(rows, dtype=np.float32)
    if hold:
        rows = np.concatenate([rows, np.repeat(rows[-1:], int(hold), axis=0)])
    arms = list(s.robot.arms)
    T = len(rows)
    targets = np.zeros((T, s.action_dim), np.float32)
    qmeas = np.zeros((T, s.action_dim), np.float32)
    tcp = np.zeros((T, len(arms), 7), np.float64)
    held = np.zeros((T, len(arms)), np.int8)
    fcont = np.zeros((T, len(arms), 2), np.int16)
    success = np.zeros(T, bool)
    tick = np.zeros(T, np.int64)
    limited = np.zeros(T, bool)
    coll = np.zeros((T, 4), np.float64)          # robot-obstacle contacts, deepest (m), robot-table, object-obstacle
    progress = np.zeros(T, np.float64)
    start_sha = s.state_sha()
    rec.snapshot(0)
    first_success = None
    t0 = time.time()
    n = 0
    for i, row in enumerate(rows):
        stop = s.step(row, _physics_steps=None if ticks is None or i >= len(ticks) else int(ticks[i]))
        n = i + 1
        targets[i] = s._targets
        qmeas[i] = s.robot.positions()
        for k, arm in enumerate(arms):
            p, q = s.ee_pose(arm)
            tcp[i, k, :3], tcp[i, k, 3:] = p, q
            held[i, k] = 1 if s.object_in_hand(arm) else 0
        fc = rec.finger_contacts()
        for k, arm in enumerate(arms):
            fcont[i, k] = fc[arm][:2]
        success[i] = s.success()
        tick[i] = s.physics_steps
        limited[i] = bool(s.last_step_meta.get("limits"))
        coll[i] = rec.collisions()
        progress[i] = float(getattr(s, "_reward", 0.0))
        rec.grasp_events()
        if success[i] and first_success is None:
            first_success = n
        if every and n % every == 0:
            rec.snapshot(n)
        if stop and stop_at_end:
            break
    wall = time.time() - t0
    if not rec.samples or rec.samples[-1][0] != n:
        rec.snapshot(n)
    st = s.status()
    arrays = dict(targets=targets[:n], q_measured=qmeas[:n], tcp=tcp[:n], held=held[:n], finger_contacts=fcont[:n],
                  success=success[:n], physics_step=tick[:n], limited=limited[:n], collisions=coll[:n],
                  progress=progress[:n],
                  sample_rows=np.array([r for r, _ in rec.samples], np.int64),
                  action_layout=np.array(s.action_layout), arms=np.array(arms))
    keys = sorted({k for _, o in rec.samples for k in o})
    for k in keys:
        arrays["obj:" + k] = np.stack([o[k] for _, o in rec.samples])
    # cloth particles at the first / last sample (the task's particle view: flex vertices -> particles)
    for name, c in rec.cloths.items():
        v = arrays["obj:cloth/%s/verts" % name]
        arrays["particles0:" + name] = c.particles_of(v[0])
        arrays["particles1:" + name] = c.particles_of(v[-1])
    for name, r in rec.ropes.items():
        arrays["rope_lengths:" + name] = np.asarray(r.lengths, dtype=np.float64)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    expected = None
    try:
        expected = [[str(x[0]), int(x[1])] for x in s.env._task.layout.required_crossings()]
    except Exception:  # noqa: BLE001 - only rope_threading has a crossing layout
        pass
    collision_summary = dict(robot_obstacle_rows=int((coll[:n, 0] > 0).sum()),
                             robot_obstacle_max_depth_m=float(coll[:n, 1].max()) if n else 0.0,
                             robot_table_rows=int((coll[:n, 2] > 0).sum()),
                             object_obstacle_rows=int((coll[:n, 3] > 0).sum()),
                             obstacles=sorted(set(rec.obstacle_body.values())))
    summary = dict(label=label, rows_in=int(T), rows_run=int(n), hold=int(hold), success=bool(st["success"]),
                   expected_crossings=expected, collisions=collision_summary,
                   max_progress=float(progress[:n].max()) if n else 0.0,
                   first_success_row=first_success, episode_end=st["episode_end"], start_state_sha=start_sha,
                   final_state_sha=st["state_sha"], physics_steps=int(st["physics_steps"]),
                   sim_time_s=float(st["sim_time_s"]), wall_s=round(wall, 1),
                   ms_per_control=round(1e3 * wall / max(1, n), 1), control_dt_s=1.0 / s.control_hz,
                   grasp_events=rec.events, metrics_samples=rec.metric_samples,
                   limited_rows=int(limited[:n].sum()), review=st.get("review"), forcing=st.get("forcing"),
                   instability=st.get("instability"), ropes=list(rec.ropes), cloths=list(rec.cloths),
                   rigid=list(rec.rigid), npz=str(out_npz), **(extra or {}))
    Path(str(out_npz)[:-4] + ".json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("instance"); ap.add_argument("rows"); ap.add_argument("out")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--hold", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--rows-key", default="actions")
    ap.add_argument("--no-strict", action="store_true", help="accept an instance whose state hash differs (realized)")
    ap.add_argument("--ledger", action="store_true", help="replay with the recording's physics-steps ledger")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    from rcb_topogym import Session
    inst = json.loads(Path(a.instance).read_text())
    t0 = time.time()
    s = Session(inst["task"], instance=a.instance, strict=not a.no_strict)
    build_s = time.time() - t0
    rows, ticks = load_rows(a.rows, s, a.rows_key)
    if a.max_rows:
        rows = rows[:a.max_rows]
    out = run(s, rows, a.out, every=a.every, hold=a.hold, label=a.label or Path(a.rows).parent.name,
              ticks=ticks if a.ledger else None,
              extra=dict(instance=str(a.instance), rows_file=str(a.rows), build_s=round(build_s, 1),
                         instance_sha=s.instance_sha, task=inst["task"], variation=s.config.get("variation_id")))
    print(json.dumps({k: v for k, v in out.items() if k not in ("metrics_samples",)}, default=str)[:4000])
    s.close()


if __name__ == "__main__":
    main()
