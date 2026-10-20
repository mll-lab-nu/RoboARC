"""Checks of a redesign instance (build_instance.py), each part in a fresh process from instance.json.

usage (the cloud host, `source KU/work/env_robot.sh`):
  python check_instance.py <instance.json> <out dir> --part goal             the goal predicate on fabricated states
  python check_instance.py <instance.json> <out dir> --part nop --tag a      build, birth vs lab, nop + clamp drift
  python check_instance.py <instance.json> <out dir> --part nop --tag b      the same in a second fresh process
  python check_instance.py <instance.json> <out dir> --part grasp            a scripted weld grasp + a pull on the clamp
  python check_instance.py <instance.json> <out dir> --part summary          checks.json (+ READY_<tier> when all pass)

nop    the Session (frozen scene, strict) built in this process (build time); the born chain's nodes against the lab's
       settled nodes (instance_assets/lab_start/lab_nodes.npz); `--rows` (400) rows holding the birth command: the
       nodes' largest move from birth at 1 s and at the end, the clamped links' drift (the world's
       attachment_residual_m and the links' own displacement) after 200 rows and at the end, the rope's largest speed,
       success / episode end / faults, the state hash at the end.
grasp  one scripted contact_weld_v1 grasp: a vertical pinch on a free strand of the anchored rope 8-30 cm from its
       clamp (the arm whose base is nearer), lift 3 cm, then pull away from the clamp in 2 mm steps until the weld's
       20 ms mean force reaches `--pull-n` (3 N) or 6 cm, hold, return, lower, open, retreat, home. Records
       object_in_hand, the pinched link's lift, the weld force, the clamp drift, the release.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def session(instance):
    from topogym.tasks.knot_untying.redesign import task_rd
    task_rd.register()
    from rcb_topogym import Session
    t0 = time.time()
    s = Session("knot_untying", instance=str(instance), strict=True)
    return s, time.time() - t0


def ropes_of(s):
    return s.env._world._ropes


def nodes(s):
    w = s.env._world
    return {name: r.node_positions(w.data).copy() for name, r in ropes_of(s).items()}


def attached(s):
    """{attachment: (link body ids, their birth positions (world))} from the scene manifest."""
    import mujoco
    w = s.env._world
    m = w.model
    d0 = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d0)
    out = {}
    for name, rec in w.manifest["entities"].items():
        for a in (rec.get("rope") or {}).get("attachments", ()) if rec.get("kind") == "rope_chain" else ():
            ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in a["links"]])
            out[a["name"]] = (ids, d0.xpos[ids].copy())
    return out


def clamp_drift(s, att):
    d = s.env._world.data
    st = s.env._world.read_state()
    return {name: dict(links_max_m=float(np.linalg.norm(d.xpos[ids] - p0, axis=1).max()),
                       residual_m=float(st["ipc"]["attachment_residual_m"][name])) for name, (ids, p0) in att.items()}


def rope_speed(s):
    st = s.env._world.read_state()
    return float(max(np.linalg.norm(np.asarray(f["velocity_m_s"]), axis=1).max() for f in st["fem"].values()))


def metrics_of(s):
    m = dict(s.env._task.result.metrics)
    return {k: v for k, v in m.items() if isinstance(v, (bool, int, float, str, dict))}


# ---------------------------------------------------------------------------------------------------------- nop
def part_nop(a, out):
    s, build_s = session(a.instance)
    inst_dir = Path(a.instance).resolve().parent
    lab = dict(np.load(inst_dir / "instance_assets" / "lab_start" / "lab_nodes.npz"))
    birth = nodes(s)
    vs_lab = {}
    for name, p in birth.items():
        e = np.linalg.norm(p - lab[name], axis=1)
        vs_lab[name] = dict(max_m=float(e.max()), mean_m=float(e.mean()), argmax_node=int(e.argmax()), nodes=len(p))
    att = attached(s)
    res = dict(part="nop", tag=a.tag, instance=str(a.instance), build_s=round(build_s, 1), state_sha_birth=s.instance_sha,
               instance_exact=s.instance_exact, birth_vs_lab=vs_lab, welded_links={k: len(v[0]) for k, v in att.items()},
               clamp_drift_birth=clamp_drift(s, att), rows=a.rows)
    hold = s.hold_action()
    t0 = time.time()
    samples, ended, fault = [], None, None
    try:
        for i in range(a.rows):
            if s.step(hold):
                ended = i
                break
            row = i + 1
            if row in (40, 200, a.rows) or row % 100 == 0:
                now = nodes(s)
                samples.append(dict(row=row, nodes_moved_max_m={k: float(np.linalg.norm(now[k] - birth[k], axis=1).max())
                                                                 for k in now},
                                    clamp=clamp_drift(s, att), rope_speed_m_s=rope_speed(s),
                                    success=bool(s.success()), wall_s=round(time.time() - t0, 1)))
                print(json.dumps(samples[-1]), flush=True)
    except Exception as exc:  # noqa: BLE001  (a fault is a check result)
        fault = repr(exc)[:500]
    st = s.status()
    res.update(samples=samples, ended_at_row=ended, fault=fault, success=bool(st["success"]),
               episode_end=st["episode_end"], instability=st["instability"], state_sha_end=st["state_sha"],
               sim_s=round(st["sim_time_s"], 3), wall_s=round(time.time() - t0, 1),
               metrics_end=metrics_of(s), safety=_safe(s))
    s.close()
    (out / f"nop_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("samples", "metrics_end", "safety")}, default=float))


def _safe(s):
    try:
        sf = s.safety(episode=True)
        return json.loads(json.dumps(sf, default=float))
    except Exception as exc:  # noqa: BLE001
        return dict(error=repr(exc)[:200])


# ---------------------------------------------------------------------------------------------------------- grasp
class Driver:
    """Rows through Session.step: joint-space moves capped per row, Cartesian TCP moves by IK per row."""

    def __init__(self, s, log):
        self.s, self.log = s, log
        self.row = s.hold_action().astype(np.float64)
        self.n = 0
        self.ended = None
        self.hooks = []

    def step(self, row=None):
        if row is not None:
            self.row = np.asarray(row, dtype=np.float64)
        done = self.s.step(self.row.astype(np.float32))
        self.n += 1
        for h in self.hooks:
            h()
        if done and self.ended is None:
            self.ended = dict(row=self.n, episode_end=self.s.status()["episode_end"])
        return done

    def arm_q(self, arm):
        return self.row[self.s.robot.arm_slice[arm]].copy()

    def joints(self, arm, q, max_step=.04):
        sl = self.s.robot.arm_slice[arm]
        start = self.row[sl].copy()
        k = max(1, int(math.ceil(np.abs(np.asarray(q) - start).max() / max_step)))
        for i in range(1, k + 1):
            r = self.row.copy()
            r[sl] = start + (np.asarray(q) - start) * i / k
            self.step(r)

    def gripper(self, arm, width, rows=8):
        gi = self.s.robot.grip_index[arm]
        start = self.row[gi]
        for i in range(1, rows + 1):
            r = self.row.copy()
            r[gi] = start + (width - start) * i / rows
            self.step(r)

    def hold(self, rows):
        for _ in range(rows):
            self.step()

    def ik(self, arm, pos, rot):
        q, ep, er = self.s.ik(arm, pos, rot.as_quat()[[3, 0, 1, 2]])
        return q, ep, er

    def line(self, arm, p0, p1, rot, speed=.08):
        """TCP straight line p0 -> p1 at `speed` (m/s), one IK solve per row from the commanded joints."""
        dist = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
        k = max(1, int(math.ceil(dist / (speed * .025))))
        worst = 0.
        for i in range(1, k + 1):
            p = np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * i / k
            q, ep, er = self.ik(arm, p, rot)
            worst = max(worst, ep)
            r = self.row.copy()
            r[self.s.robot.arm_slice[arm]] = q
            self.step(r)
        return worst


def weld_force(s):
    """The active grasp welds' 20 ms mean translational force (the backend's slip window), per weld id."""
    gm = s.env._world._grasp_manager
    if gm is None:
        return {}
    return {int(e): float(sum(w) / len(w)) for e, w in gm.backend._window.items()}


def choose_station(s, rope, ik_try, *, side, around, span=(.06, .30)):
    """A free strand of `rope` at a material distance in `span` from its clamped end (side "clamp") or from its free end
    (side "free"), tried from `around` outward: room for the fingers (no other strand within 2 cm, no fixture within
    4 cm in plan), lying on the table, reachable (vertical pinch + 5 cm pregrasp)."""
    from topogym.planning.knot_untying.redesign.robot_ik import vertical_pinch
    w = s.env._world
    chain = ropes_of(s)[rope]
    P = chain.node_positions(w.data)
    S = chain.s_nodes
    others = [(n, r.node_positions(w.data)) for n, r in ropes_of(s).items() if n != rope]
    task = s.env._task
    rd = task._scene()
    ox, oy = rd["frame"]["origin_world_xy"]
    fixtures = rd.get("fixtures", [])
    anchor = next(r.get("anchor") for r in rd["ropes"] if r["name"] == rope) or {}
    clamp_start = anchor.get("end", "start") == "start"
    L = float(S[-1])
    order = sorted(np.arange(span[0], span[1] + 1e-9, .01), key=lambda x: abs(x - around))
    tried = []
    for d in order:
        from_start = (side == "clamp") == clamp_start
        sm = d if from_start else L - d
        p = np.array([np.interp(sm, S, P[:, k]) for k in range(3)])
        t = np.array([np.interp(sm + .006, S, P[:, k]) - np.interp(sm - .006, S, P[:, k]) for k in range(3)])
        far = np.abs(S - sm) > .03
        room = float(np.linalg.norm(P[far] - p, axis=1).min())
        for _, Q in others:
            room = min(room, float(np.linalg.norm(Q - p, axis=1).min()))
        fix = min([float(np.hypot(p[0] - (ox + f["pos"][0]), p[1] - (oy + f["pos"][1])))
                   - (float(f.get("radius", 0)) or .5 * float(np.hypot(*f.get("size", [0, 0])[:2])))
                   for f in fixtures] or [1.])
        tried.append(dict(s=round(float(sm), 3), room_m=round(room, 4), fixture_m=round(fix, 4)))
        if room < .02 or fix < .04 or p[2] - task.table_height_m > .02:
            continue
        for flip in (1, -1):
            tcp, rot = vertical_pinch(p, t[:2], flip)
            arm, info = ik_try(tcp, rot)
            if arm is not None:
                return dict(rope=rope, side=side, s=float(sm), point=p.tolist(), tangent=t.tolist(), flip=flip,
                            tcp=tcp.tolist(), arm=arm, ik=info, room_m=room, fixture_m=fix, tried=len(tried))
    raise RuntimeError(f"no {side} station for a smoke grasp on {rope}: {tried[:12]}")


def part_grasp(a, out):
    """Two weld-grasp cycles on the anchored rope: (1) "lift" on a free strand near the free end: pinch, lift 3 cm,
    hold, lower, open; (2) "pull" on the standing part near the clamp: pinch, lift 5 mm, pull away from the clamp until
    the weld's 20 ms mean reaches --pull-n (or 5 cm), hold, ease back, lower, open. Each from home and back home."""
    from topogym.planning.knot_untying.redesign.robot_ik import vertical_pinch
    s, build_s = session(a.instance)
    w = s.env._world
    rd = s.env._task._scene()
    rope = next((r["name"] for r in rd["ropes"] if r.get("anchor")), rd["ropes"][0]["name"])
    chain = ropes_of(s)[rope]
    att = attached(s)
    ox, oy = rd["frame"]["origin_world_xy"]     # the ALOHA bases at task-local (-+0.469, -0.019)
    bases = {arm: np.array([ox + (-.469 if arm == "left" else .469), oy - .019, 0.]) for arm in s.robot.arms}
    log = dict(part="grasp", instance=str(a.instance), build_s=round(build_s, 1), state_sha_birth=s.instance_sha,
               rope=rope, events=[], phases={})
    drv = Driver(s, log)
    trace = []

    losses = []

    def sample():
        f = weld_force(s)
        gm = w._grasp_manager
        for arm, h in gm.hands.items():
            if h["phase"] == "lost" and not any(l["arm"] == arm and l["row"] >= drv.n - 1 for l in losses):
                if not losses or losses[-1]["arm"] != arm or losses[-1].get("cleared"):
                    losses.append(dict(arm=arm, row=drv.n, reason=h.get("rejection"),
                                       weld_n_before=trace[-1]["weld_n"] if trace else None))
            elif h["phase"] != "lost" and losses and losses[-1]["arm"] == arm:
                losses[-1]["cleared"] = True
        trace.append(dict(row=drv.n, held={arm: s.object_in_hand(arm) for arm in s.robot.arms},
                          phase={arm: gm.hands[arm]["phase"] for arm in s.robot.arms},
                          weld_n=max(f.values()) if f else 0., clamp=clamp_drift(s, att)))
    drv.hooks.append(sample)

    def ik_try(tcp, rot):
        for arm in sorted(s.robot.arms, key=lambda x: float(np.linalg.norm(bases[x][:2] - tcp[:2]))):
            q, ep, er = s.ik(arm, tcp, rot.as_quat()[[3, 0, 1, 2]])
            q2, ep2, er2 = s.ik(arm, tcp + [0, 0, .05], rot.as_quat()[[3, 0, 1, 2]])
            if max(ep, ep2) < 1e-3 and max(er, er2) < 2e-2:
                return arm, dict(position_error_m=float(max(ep, ep2)), rotation_error_rad=float(max(er, er2)))
        return None, None

    def ev(phase, event, **kw):
        log["events"].append(dict(phase=phase, row=drv.n, event=event, **kw))

    def cycle(phase, st, lift, pull_n=None):
        arm = st["arm"]
        tcp, rot = vertical_pinch(np.asarray(st["point"]), np.asarray(st["tangent"])[:2], st["flip"])
        gi = s.robot.grip_index[arm]
        wide = min(float(s.robot.hi[gi]), .05)
        k_link = int(np.argmin(np.abs(.5 * (chain.s_nodes[:-1] + chain.s_nodes[1:]) - st["s"])))
        body = int(chain.bodies[k_link])
        z0 = float(w.data.xpos[body][2])
        home_q = drv.arm_q(arm)
        row0 = drv.n
        drv.gripper(arm, wide, rows=6)                                     # open
        q_pre, ep, er = drv.ik(arm, tcp + [0, 0, .05], rot)
        drv.joints(arm, q_pre)                                             # home -> 5 cm above the station
        drv.hold(6)
        ev(phase, "pregrasp", ik_error_m=ep)
        drv.line(arm, tcp + [0, 0, .05], tcp, rot, speed=.06)             # descend
        drv.hold(4)
        drv.gripper(arm, 0., rows=10)                                      # close on the rope
        drv.hold(10)
        ev(phase, "closed", object_in_hand=s.object_in_hand(arm), link=k_link, tcp_z=float(s.ee_pose(arm)[0][2]),
           link_z=float(w.data.xpos[body][2]), tcp_target_z=float(tcp[2]))
        top = tcp + [0, 0, lift]
        drv.line(arm, tcp, top, rot, speed=.05)                            # lift
        drv.hold(20)
        ev(phase, "lifted", object_in_hand=s.object_in_hand(arm), link_lift_m=float(w.data.xpos[body][2]) - z0,
           tcp_z=float(s.ee_pose(arm)[0][2]), link_z=float(w.data.xpos[body][2]), tcp_target_z=float(top[2]),
           weld_n=max(weld_force(s).values() or [0.]))
        p = top
        if pull_n is not None:
            # away from the clamp (clamped end -> station, in plan): 1 mm per row, 0.3 mm once the weld feels 0.5 N
            anchor = next(r.get("anchor") for r in rd["ropes"] if r["name"] == rope) or {}
            end = chain.node_positions(w.data)[0 if anchor.get("end", "start") == "start" else -1]
            away = np.asarray(st["point"], float) - end
            away[2] = 0.
            away /= np.linalg.norm(away)
            peak, pulled, f = 0., 0., 0.
            while pulled < .05 and s.object_in_hand(arm) is not None:
                dx = .001 if f < .5 else .0003
                p = p + dx * away
                pulled += dx
                q, ep, er = drv.ik(arm, p, rot)
                r = drv.row.copy()
                r[s.robot.arm_slice[arm]] = q
                drv.step(r)
                f = max(weld_force(s).values() or [0.])
                peak = max(peak, f)
                if f >= pull_n:
                    break
            drv.hold(12)
            ev(phase, "pulled", pulled_m=round(pulled, 4), peak_weld_n=peak,
               weld_n_after_hold=max(weld_force(s).values() or [0.]), object_in_hand=s.object_in_hand(arm),
               clamp=clamp_drift(s, att))
            drv.line(arm, p, top, rot, speed=.03)                          # ease back over the station
        drv.line(arm, top, tcp + [0, 0, .002], rot, speed=.04)            # lower
        drv.hold(6)
        drv.gripper(arm, wide, rows=6)                                     # open: release
        drv.hold(4)
        ev(phase, "opened", object_in_hand=s.object_in_hand(arm))
        drv.line(arm, tcp + [0, 0, .002], tcp + [0, 0, .06], rot, speed=.08)   # retreat
        drv.joints(arm, home_q)                                            # home
        drv.hold(30)
        rows = [r for r in trace if r["row"] > row0]
        held = [r["row"] for r in rows if r["held"].get(arm)]
        log["phases"][phase] = dict(
            station=st, arm=arm, rows=[row0 + 1, drv.n], held_rows=[min(held), max(held)] if held else None,
            released=bool(held) and s.object_in_hand(arm) is None, weld_peak_n=max((r["weld_n"] for r in rows), default=0.),
            clamp_drift_max_m={k: max(r["clamp"][k]["links_max_m"] for r in rows) for k in att},
            clamp_residual_max_m={k: max(r["clamp"][k]["residual_m"] for r in rows) for k in att})

    t0 = time.time()
    try:
        cycle("lift", choose_station(s, rope, ik_try, side="free", around=.12), lift=.03)
        try:
            st_pull = choose_station(s, rope, ik_try, side="clamp", around=.14)
        except RuntimeError:      # medium (knot agent 10-11): a 0.1 m standing part inside the knot round its post;
            st_pull = choose_station(s, rope, ik_try, side="clamp", around=.30, span=(.06, .80))   # further along
        cycle("pull", st_pull, lift=.005, pull_n=a.pull_n)
    except Exception as exc:  # noqa: BLE001
        import traceback
        log["fault"] = repr(exc)[:500]
        log["fault_trace"] = traceback.format_exc()[-1500:]
    stt = s.status()
    log.update(rows=drv.n, wall_s=round(time.time() - t0, 1), sim_s=round(stt["sim_time_s"], 3), ended=drv.ended,
               success=bool(stt["success"]), episode_end=stt["episode_end"], instability=stt["instability"],
               state_sha_end=stt["state_sha"], rope_speed_end_m_s=rope_speed(s), safety=_safe(s))
    log["losses"] = losses
    log["trace"] = trace[::2]
    s.close()
    (out / "grasp.json").write_text(json.dumps(log, indent=1, default=float))
    print(json.dumps({k: v for k, v in log.items() if k not in ("trace", "safety")}, default=float))


# ---------------------------------------------------------------------------------------------------------- goal
def serpentine(start_xy, length, *, lane=.04, x_max=.36, x_min=-.42, step=.002):
    """A planar path from start_xy (task-local) of `length`: lanes along +x then back, `lane` apart in +y, joined by
    half circles: an untied, uncrossed layout."""
    pts, (x, y), d, total = [np.array(start_xy, float)], start_xy, 1., 0.
    while total < length:
        end = x_max if d > 0 else x_min
        n = max(1, int(abs(end - x) / step))
        for k in range(1, n + 1):
            pts.append(np.array([x + (end - x) * k / n, y]))
        x = end
        c = np.array([x, y + lane / 2])
        for k in range(1, 40):
            a = -np.pi / 2 + np.pi * k / 39
            pts.append(c + lane / 2 * np.array([d * np.cos(a), np.sin(a)]))
        y += lane
        d = -d
        total = float(np.linalg.norm(np.diff(np.asarray(pts), axis=0), axis=1).sum())
    return np.asarray(pts)


def along(path_xy, length, z):
    """The first `length` m of a planar path (resampled every ~1 mm), at height z: (N, 3)."""
    P = np.asarray(path_xy, float)[:, :2]
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if s[-1] < length:
        raise ValueError(f"path {s[-1]:.3f} m shorter than the rope {length:.3f} m")
    u = np.linspace(0., length, max(2, int(length / .001)) + 1)
    return np.column_stack([np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1]), np.full(len(u), z)])


def part_goal(a, out):
    """The goal predicate on fabricated states (no physics): the task declared from the instance's parameters, each
    rope's FEM vertices swept along a centre line (session_kit.rope_birth_from_centerline), at rest, volume intact.
    Cases: the lab start (knotted: not reached); every rope laid along its lab layout path from its clamped end (a
    serpentine when the tier has none: reached); the first rope lifted 15 mm (off the table band: not reached); each
    rope with a region shifted 10 cm out of it (not reached); two ropes: the second 5 mm beside the first (not reached,
    the between-ropes term false)."""
    import tempfile
    import session_kit as sk
    from topogym.tasks.knot_untying.redesign import task_rd
    task_rd.register()
    from topogym.config.registry import task_definition
    inst_path = Path(a.instance).resolve()
    inst = json.loads(inst_path.read_text())
    tp = inst["simulator"]["task_parameters"]
    task = task_definition("knot_untying", inst["simulator"]["variation_id"])(tp, seed=0)
    rd = tp["redesign"]
    top = float(rd["frame"]["table_top_z"])
    ox, oy = rd["frame"]["origin_world_xy"]
    tmp = Path(tempfile.mkdtemp(prefix="goalcheck_"))
    task.declare(tmp, top)
    lab = dict(np.load(inst_path.parent / "instance_assets" / "lab_start" / "lab_nodes.npz"))
    vols = {r["name"]: tmp / task.volume_name(r["name"]) for r in rd["ropes"]}

    def state_of(lines):
        fem = {}
        for name, C in lines.items():
            V, _ = sk.rope_birth_from_centerline(vols[name], C)
            with np.load(vols[name]) as z:
                ms = z["rest"][:, 0].copy()
            fem[name] = dict(position_m=V, velocity_m_s=np.zeros_like(V), material_s_m=ms)
        return dict(fem=fem, ipc=dict(native_valid=True, volume_ratio={n: dict(minimum=1., maximum=1.) for n in lines},
                                      attachment_residual_m={}))

    def run(label, lines):
        reached, failure, m = task.evaluate_state(state_of(lines))
        rec = dict(case=label, reached=bool(reached), failure=failure,
                   ropes={n: {k: v for k, v in r.items() if k in ("planar_clear", "on_table", "quiet", "in_region",
                                                                  "outside_fraction", "minimum_nonlocal_distance_m")}
                          for n, r in m["ropes"].items()},
                   between=m.get("between_ropes"))
        print(json.dumps(rec, default=float), flush=True)
        return rec
    z = top + .004 + .00015          # a rope resting on the table: centre at r + the contact gap
    cases = [run("lab_start", {n: lab[n] for n in vols})]
    paths = (rd.get("source", {}).get("oracle") or {}).get("layout_path") or {}
    laid = {}
    for r in rd["ropes"]:
        n, L = r["name"], float(r["length_m"])
        clamp_end = lab[n][0] if (r.get("anchor") or {}).get("end", "start") == "start" else lab[n][-1]
        start = np.array([clamp_end[0] - ox, clamp_end[1] - oy])
        P = np.asarray(paths[n], float) if n in paths else serpentine(start, L)
        P = np.vstack([start, P]) if np.linalg.norm(P[0] - start) > 1e-3 else P   # from the clamped end
        C = along(P, L, z) + [ox, oy, 0.]
        laid[n] = C if (r.get("anchor") or {}).get("end", "start") == "start" else C[::-1]
    cases.append(run("laid_out", laid))
    first = rd["ropes"][0]["name"]
    lifted = dict(laid)
    lifted[first] = laid[first] + [0., 0., .015]
    cases.append(run(f"{first}_lifted_15mm", lifted))
    expect = [("lab_start", lambda c: not c["reached"]), ("laid_out", lambda c: c["reached"]),
              (f"{first}_lifted_15mm", lambda c: not c["reached"] and not c["ropes"][first]["on_table"])]
    for n in (rd.get("regions") or {}):
        moved = dict(laid)
        moved[n] = laid[n] + [0., .10 if np.mean(rd["regions"][n], axis=0)[1] < 0 else -.10, 0.]
        cases.append(run(f"{n}_shifted_out_of_region", moved))
        expect.append((f"{n}_shifted_out_of_region", lambda c, n=n: not c["reached"] and not c["ropes"][n]["in_region"]))
    if len(laid) > 1:
        a_, b_ = list(laid)[:2]
        over = dict(laid)
        over[b_] = laid[a_][: len(laid[b_])] + [0., .005, 0.]          # rope b 5 mm beside rope a's line
        cases.append(run("ropes_5mm_apart", over))
        expect.append(("ropes_5mm_apart", lambda c: not c["reached"]
                       and not all(v["clear"] for v in (c["between"] or {}).values())))
    by = {c["case"]: c for c in cases}
    verdicts = {name: bool(test(by[name])) for name, test in expect}
    res = dict(part="goal", instance=str(inst_path), cases=cases, expectations=verdicts, ok=all(verdicts.values()))
    (out / "goal.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(dict(part="goal", ok=res["ok"], expectations=verdicts)))


# ---------------------------------------------------------------------------------------------------------- summary
def part_summary(a, out):
    inst = Path(a.instance).resolve()
    tier = json.loads(inst.read_text())["simulator"]["task_parameters"]["redesign"]["tier"]
    na, nb = (json.loads((out / f"nop_{t}.json").read_text()) for t in ("a", "b"))
    gr = json.loads((out / "grasp.json").read_text()) if (out / "grasp.json").is_file() else None
    build = json.loads((inst.parent / "build.json").read_text()) if (inst.parent / "build.json").is_file() else {}
    end40 = next((x for x in na["samples"] if x["row"] == 40), None)
    at200 = next((x for x in na["samples"] if x["row"] == 200), None)
    last = na["samples"][-1] if na["samples"] else None
    checks = dict(
        builds_fresh=dict(ok=bool(na["instance_exact"] and nb["instance_exact"]), build_s=[na["build_s"], nb["build_s"]],
                          instance_exact=[na["instance_exact"], nb["instance_exact"]]),
        birth_vs_lab=dict(ok=all(v["max_m"] <= .001 for v in na["birth_vs_lab"].values()), per_rope=na["birth_vs_lab"]),
        rests=dict(ok=all(v <= .002 for v in (end40 or {}).get("nodes_moved_max_m", {"x": 1.}).values()),
                   moved_1s_m=(end40 or {}).get("nodes_moved_max_m"), moved_end_m=(last or {}).get("nodes_moved_max_m")),
        clamp_drift_200=dict(ok=all(v["links_max_m"] <= 1e-4 for v in (at200 or {}).get("clamp", {"x": {"links_max_m": 1}}).values()),
                             at_200=(at200 or {}).get("clamp"), welded_links=na["welded_links"]),
        nop=dict(ok=(not na["success"] and na["fault"] is None and na["instability"] is None and na["episode_end"] is None
                     and (last or {}).get("rope_speed_m_s", 1.) <= .001),
                 rows=na["rows"], success=na["success"], episode_end=na["episode_end"], fault=na["fault"],
                 rope_speed_end_m_s=(last or {}).get("rope_speed_m_s"), state_sha=na["state_sha_end"],
                 sim_s=na["sim_s"], wall_s=na["wall_s"]),
        determinism=dict(ok=na["state_sha_end"] == nb["state_sha_end"] and na["state_sha_birth"] == nb["state_sha_birth"],
                         a=na["state_sha_end"], b=nb["state_sha_end"], birth=[na["state_sha_birth"], nb["state_sha_birth"]]),
    )
    if gr is not None:
        ev = {(e["phase"], e["event"]): e for e in gr.get("events", [])}
        ph = gr.get("phases", {})
        lift, pull = ph.get("lift", {}), ph.get("pull", {})
        rope = gr.get("rope")
        checks["weld_grasp"] = dict(
            ok=bool(ev.get(("lift", "closed"), {}).get("object_in_hand") == rope
                    and (ev.get(("lift", "lifted"), {}).get("link_lift_m") or 0) > .02
                    and lift.get("released") and gr.get("fault") is None and gr.get("instability") is None
                    and gr.get("episode_end") is None),
            arm=lift.get("arm"), station_s=(lift.get("station") or {}).get("s"),
            object_in_hand=ev.get(("lift", "closed"), {}).get("object_in_hand"),
            link_lift_m=ev.get(("lift", "lifted"), {}).get("link_lift_m"), held_rows=lift.get("held_rows"),
            released=lift.get("released"), weld_peak_n=lift.get("weld_peak_n"), fault=gr.get("fault"),
            episode_end=gr.get("episode_end"), rows=gr.get("rows"), wall_s=gr.get("wall_s"))
        pulled = ev.get(("pull", "pulled"), {})
        checks["clamp_under_pull"] = dict(
            ok=bool(pulled and all(v <= 1e-4 for v in (pull.get("clamp_drift_max_m") or {"x": 1.}).values())
                    and pull.get("released")),
            arm=pull.get("arm"), station_s=(pull.get("station") or {}).get("s"), peak_weld_n=pulled.get("peak_weld_n"),
            weld_n_after_hold=pulled.get("weld_n_after_hold"), pulled_m=pulled.get("pulled_m"),
            held_while_pulling=pulled.get("object_in_hand"), clamp_drift_max_m=pull.get("clamp_drift_max_m"),
            clamp_residual_max_m=pull.get("clamp_residual_max_m"), released=pull.get("released"))
    if (out / "goal.json").is_file():
        g = json.loads((out / "goal.json").read_text())
        checks["goal_predicate"] = dict(ok=g["ok"], reached={c["case"]: c["reached"] for c in g["cases"]},
                                        expectations=g["expectations"])
    previews = sorted(p.name for p in inst.parent.glob("preview_*.png"))
    checks["preview"] = dict(ok=bool(previews), files=previews)
    ok = all(c["ok"] for c in checks.values())
    res = dict(tier=tier, instance=str(inst), state_sha_birth=na["state_sha_birth"], all_ok=ok, checks=checks,
               build=dict((k, build.get(k)) for k in ("make_instance_s", "total_s", "frozen_birth", "weld_targets")),
               written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (inst.parent / "checks.json").write_text(json.dumps(res, indent=1, default=float))
    if ok:
        (inst.parent.parent / f"READY_{tier}").touch()
    print(json.dumps(dict(tier=tier, all_ok=ok, **{k: v["ok"] for k, v in checks.items()})))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("out")
    ap.add_argument("--part", choices=("nop", "grasp", "goal", "summary"), required=True)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--pull-n", type=float, default=3.)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    {"nop": part_nop, "grasp": part_grasp, "goal": part_goal, "summary": part_summary}[a.part](a, out)


if __name__ == "__main__":
    main()
