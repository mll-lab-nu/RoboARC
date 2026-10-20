"""The redesign's untying oracle on the real ALOHA in a MuJoCo rcb_topogym.Session (knot-robot), recorded as rows.

  python run_robot_oracle.py <instance.json> <tier> <out dir> [--budget-min N] [--scene scene_start.json]
  python run_robot_oracle.py --adhoc <released instance.json> primitives <out dir>    (dev: the primitives test)

The decision logic is untie.Untie's (run_oracle2: anchored node deletions with a pin, clear_tail, lift_over_post,
reach-aware draws; then layout into the region where the tier has one), on robot hands (robot_lab.RobotLab +
robot_untie.RobotUntie). Every accepted row is recorded by the Session; after the oracle both hands release and lift,
and the Session runs hold rows until it ends the episode (success) or --hold-s runs out.

Writes <out>/privileged.npz (Session.save_trajectory: actions = the accepted rows, physics_steps_per_action, layout,
instance_sha, meta), <out>/oracle.json (decisions, phases, grasp attempts, events: grasps / releases / slips / reach stops,
max weld forces, timings, final status and state hash), <out>/trace.jsonl (every 8 rows: crossings, weld forces, held
stations, TCP errors), partial rows (trajectory.partial.npz) while it runs.
--adhoc <released instance.json>: no frozen redesign instance (development before knot-inst's): the released knot
instance rebuilt with contact_weld_v1 on every rope link (no clamp, no fixtures); replay_check.py rebuilds the same
Session from <out>/session_spec.json.
"""
from __future__ import annotations

import argparse
import json
import shlex
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def weld_targets_from_scene_dir(scene_dir, entity="rope"):
    with np.load(Path(scene_dir) / f"rope_{entity}.npz") as z:
        lengths = z["lengths"].copy()
    return [dict(entity=entity, link=f"{k:03d}", center_m=[float(lengths[k] / 2), 0., 0.], radius_m=.008)
            for k in range(len(lengths))]


def session_from_spec(spec):
    """A Session from a session spec: {"instance": path} (frozen; the redesign task registered first if needed) or
    {"adhoc_released": path} (the released knot instance + contact_weld_v1 on every link)."""
    if spec.get("adhoc_released"):
        import session_kit as sk
        I = spec["adhoc_released"]
        base = sk.instance_settings(I)
        ga = spec.get("grasp_assist")
        if ga is None:
            ga = dict(base["overrides"].get("grasp_assist") or {})
            # the redesign instances' grasp (knot-inst's build_instance.GRASP = the rope_threading port's settings)
            ga.update(model="contact_weld_v1", stable_time_s=.02, max_tcp_distance_m=.06, max_penetration_m=.006,
                      max_translation_error_m=.025, max_rotation_error_rad=.15, max_finger_opening_fraction=.15,
                      intent_deadband=.002, initial_intent="open",
                      targets=weld_targets_from_scene_dir(Path(I).resolve().parent / "instance_assets/mujoco"))
            spec["grasp_assist"] = ga          # the exact settings travel with the run (replay_check rebuilds them)
        ov = {**base["overrides"], "grasp_assist": ga, "mujoco_birth": base["mujoco_birth"]}
        return sk.make_session(base["task"], base["variation_id"], seed=base["seed"],
                               task_parameters=base["task_parameters"], overrides=ov,
                               gripper_space=base["gripper_space"], frozen_assets_from=I)
    inst = Path(spec["instance"]).resolve()
    register_redesign(inst)
    from rcb_topogym import Session
    return Session(json.loads(inst.read_text())["task"], instance=str(inst), strict=bool(spec.get("strict", True)))


def register_redesign(instance_json):
    """Register the redesign task variation in-process when the instance names one (knot-inst's task_rd)."""
    inst = json.loads(Path(instance_json).read_text())
    vid = (inst.get("simulator") or {}).get("variation_id", "")
    if "_rd_" not in vid and not vid.startswith("rd_"):
        return None
    try:
        from topogym.tasks.knot_untying.redesign import task_rd
    except ImportError:
        return None
    for fn in ("register", "register_all", "register_variations"):
        f = getattr(task_rd, fn, None)
        if callable(f):
            try:
                return f()
            except TypeError:
                return f(vid)
    return None


def scene_for(args, session, instance=None):
    """The lab scene (ropes with anchors, fixtures task-local, regions, oracle layout paths): --scene, else the lab start
    the instance keeps (instance_assets/lab_start/scene_start.json, knot-inst's build_instance), else the instance's
    task_parameters["redesign"] (task_rd: ropes, fixtures, regions; no layout paths), else an empty scene (a released
    rope: no anchor, no fixtures)."""
    if args.scene:
        return json.loads(Path(args.scene).read_text())
    if instance is not None:
        for f in sorted(Path(instance).resolve().parent.glob("instance_assets/lab_start*/scene_start.json")):
            return json.loads(f.read_text())
    tp = dict(session.config.get("task_parameters") or {})
    rd = tp.get("redesign")
    if isinstance(rd, dict) and rd.get("ropes"):
        return dict(rd, ropes=[dict(r, radius=r.get("radius_m", .004)) for r in rd["ropes"]])
    names = list(session.env._world._ropes)
    return dict(ropes=[dict(name=n, anchor=None) for n in names], fixtures=[], regions={})


class Recorder:
    """Partial evidence while a run goes: the accepted rows so far (trajectory.partial.npz, every `every` rows) and the
    decision log (oracle.json, partial)."""

    def __init__(self, session, out, every=200):
        self.s, self.out, self.every = session, Path(out), every
        self.last = 0
        self.dump = None

    def partial(self):
        s = self.s
        if s._recording is None or len(s._recording) - self.last < self.every:
            return
        from topogym.planning.knot_untying.artifacts import save_recording
        save_recording(s, self.out / "trajectory.partial.npz")
        self.last = len(s._recording)
        if self.dump is not None:
            try:
                self.dump()
            except Exception as e:  # noqa: BLE001  (evidence only)
                print("partial dump failed:", repr(e)[:200], flush=True)


def replay_prefix(s, path, log, max_rows=None):
    """--resume: replay a partial run's rows (and tick ledger) open-loop into the recording Session, so the new rows
    continue one row stream from the frozen start (the oracle then decides again on the live rope)."""
    z = np.load(path, allow_pickle=False)
    rows = np.asarray(z["actions"], np.float32)
    ticks = np.asarray(z["physics_steps_per_action"]) if "physics_steps_per_action" in z.files else None
    if list(z["action_layout"]) != list(s.action_layout):
        raise SystemExit("resume: action layout differs")
    t0 = time.time()
    if max_rows is not None:
        rows = rows[:int(max_rows)]
    for i, row in enumerate(rows):
        if s.step(row, _physics_steps=None if ticks is None else int(ticks[i])):
            break
    log(f"resumed: replayed {s.n_steps} rows of {path} in {time.time() - t0:.0f} s (t = {s.sim_time_s:.3f} s)")
    return int(s.n_steps)


def dump(out, U, lab, s, extra):
    from topogym.planning.knot_untying.redesign.mjlab import to_local
    st = s.status()
    m = s.env._task.result.metrics
    rec = dict(extra, status={k: st.get(k) for k in ("success", "sim_steps", "physics_steps", "sim_time_s", "state_sha",
                                                       "episode_end", "review", "forcing")},
               rows=int(s.n_steps), sim_s=round(float(s.sim_time_s), 3), wall_s=round(time.time() - lab.wall0, 1),
               physics_wall_s=round(getattr(lab, "phys_s", 0.), 1),
               crossings=len(lab.crossings()), max_force=lab.max_force, slips=lab.slips, ik_worst_mm={
                   h: round(v * 1000, 3) for h, v in lab.ik_worst.items()},
               tip_below_contact_mm={h: round(v * 1000, 2) for h, v in lab.tip_below.items()},
               gripper_radius_mm={h: round(v * 1000, 1) for h, v in lab.gripper_radius.items()},
               bias_mm={h: (v * 1000).round(2).tolist() for h, v in lab.bias.items()},
               metrics={k: v for k, v in m.items() if isinstance(v, (bool, int, float, str))},
               decisions=U.decisions if U is not None else [], phases=U.phases if U is not None else [],
               grasp_records=getattr(U, "grasp_records", []), routes=getattr(U, "routes", []), events=lab.events,
               reach_stops=[e for e in lab.events if e["kind"] == "reach_stop"], state_hashes=lab.hashes,
               standard_rate=lab.limiter.report() if lab.limiter is not None else None,
               final_centerlines={k: v.round(5).tolist() for k, v in lab.centerlines_local().items()})
    tmp = Path(out) / "oracle.json.tmp"
    tmp.write_text(json.dumps(rec, default=float))
    os.replace(tmp, Path(out) / "oracle.json")
    return rec


# ------------------------------------------------------------------------------------------------ dev: primitives
def run_primitives(lab, U, log):
    """Pinch a free strand at a chosen station, lift 3 cm, carry 10 cm, put down, release; two arms at once; a stroke
    with the other hand pinning. Each step's outcome goes to U.decisions."""
    from topogym.planning.knot_untying.redesign.mjlab import to_local, to_world
    rope = next(iter(lab.rods))
    L = lab.length(rope)

    def stations(hand, need=.015):
        out = []
        for s_ in np.arange(.08, L - .08, .01):
            p = to_local(lab.point(rope, s_))
            if U.reachable(hand, p, .05) and lab.pinch_clearance_at(rope, s_) >= need:
                out.append((float(np.linalg.norm(p[:2] - np.array([-.469 if hand == "left" else .469, -.019]))), s_))
        return [x for _, x in sorted(out)]

    def held_follow(h):
        g = lab.hands[h]["grasp"]
        if g is None:
            return None
        return round(1000 * float(np.linalg.norm(lab.point(g["rope"], g["s"]) - lab.hand_actual(h))), 2)

    # 1) one arm: pinch, lift 3 cm, carry 10 cm, put down, release
    sl = stations("left")
    s_a = sl[len(sl) // 3]
    rec = dict(kind="test_single", hand="left", s=round(s_a, 3))
    try:
        U.take("left", rope, s_a)
        rec["bound"] = lab.hands["left"]["grasp"]
        z0 = float(lab.point(rope, lab.hands["left"]["grasp"]["s"])[2])
        U.lift("left", .03)
        lab.wait(.2)
        g = lab.hands["left"]["grasp"]
        rec["lift_rope_dz_mm"] = None if g is None else round(1000 * (float(lab.point(rope, g["s"])[2]) - z0), 2)
        rec["follow_err_mm_lifted"] = held_follow("left")
        p = lab.hand_pos("left")
        toward = np.r_[np.array([-.469, -.019]) - to_local(p)[:2], 0.]      # carry 10 cm toward the arm's base
        U.goto("left", p + .10 * toward / np.linalg.norm(toward), speed=.08, label="carry")
        lab.wait(.2)
        rec["follow_err_mm_carried"] = held_follow("left")
        rec["held_after_carry"] = lab.hands["left"]["grasp"] is not None
        rec["max_force"] = round(lab.max_force["left"], 3)
        U.drop("left")
        rec["result"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["result"] = f"{type(e).__name__}: {e}"
        if not isinstance(e, Exception) or e.__class__.__name__ in ("Solved", "Budget", "EpisodeEnded"):
            raise
    U.decisions.append(rec)
    log("TEST1", json.dumps(rec, default=float))
    # 2) two arms at once: both pinch, lift together, spread 5 cm each, put down together
    sr = stations("right")
    s_l, s_r = sl[len(sl) // 2], sr[len(sr) // 2]
    if abs(s_l - s_r) < .1:
        s_r = sr[-1]
    rec = dict(kind="test_two_arms", s_left=round(s_l, 3), s_right=round(s_r, 3))
    try:
        U.take("left", rope, s_l)
        U.take("right", rope, s_r)
        rec["bound"] = {h: lab.hands[h]["grasp"] for h in ("left", "right")}
        pl, pr = lab.hand_pos("left"), lab.hand_pos("right")
        lab.move({"left": pl + [0, 0, .03], "right": pr + [0, 0, .03]}, .05, label="lift2")
        axis = (pr - pl) * [1, 1, 0]
        axis /= np.linalg.norm(axis)
        pl, pr = lab.hand_pos("left"), lab.hand_pos("right")
        lab.move({"left": pl - .05 * axis, "right": pr + .05 * axis}, .06, label="spread2")
        lab.wait(.2)
        rec["held_after_spread"] = {h: lab.hands[h]["grasp"] is not None for h in ("left", "right")}
        rec["follow_err_mm"] = {h: held_follow(h) for h in ("left", "right")}
        rec["max_force"] = {h: round(lab.max_force[h], 3) for h in ("left", "right")}
        pl, pr = lab.hand_pos("left"), lab.hand_pos("right")
        z = to_world(np.array([0, 0, .005]))[2]
        lab.move({"left": np.r_[pl[:2], z], "right": np.r_[pr[:2], z]}, .03, label="lower2")
        lab.release("left")
        lab.release("right")
        lab.move({"left": lab.hand_pos("left") + [0, 0, .05], "right": lab.hand_pos("right") + [0, 0, .05]}, .2,
                 label="retreat2")
        rec["result"] = "ok"
    except Exception as e:  # noqa: BLE001
        if e.__class__.__name__ in ("Solved", "Budget", "EpisodeEnded"):
            raise
        rec["result"] = f"{type(e).__name__}: {e}"
        U.release_all()
    U.decisions.append(rec)
    log("TEST2", json.dumps(rec, default=float))
    # 3) a stroke with the other hand pinning: left pins, right pulls along the rope away from the pin
    rec = dict(kind="test_pin_stroke")
    try:
        s_pin = sl[len(sl) // 2]
        cand = [x for x in sr if abs(x - s_pin) > .10]
        s_pull = min(cand, key=lambda x: abs(abs(x - s_pin) - .15)) if cand else sr[0]
        rec.update(s_pin=round(s_pin, 3), s_pull=round(s_pull, 3))
        U.take("left", rope, s_pin, label="grasp_pin")
        U.take("right", rope, s_pull)
        U.lift("right", .02)
        d = lab.point(rope, s_pull) - lab.point(rope, s_pin)
        d[2] = 0
        moved, why = U.stroke("right", d, .15, step=.005, stop_force=3.5)
        rec.update(moved=round(moved, 3), stop=why, max_force={h: round(lab.max_force[h], 3) for h in ("left", "right")},
                   held={h: lab.hands[h]["grasp"] is not None for h in ("left", "right")})
        U.release_all()
        rec["result"] = "ok"
    except Exception as e:  # noqa: BLE001
        if e.__class__.__name__ in ("Solved", "Budget", "EpisodeEnded"):
            raise
        rec["result"] = f"{type(e).__name__}: {e}"
        U.release_all()
    U.decisions.append(rec)
    log("TEST3", json.dumps(rec, default=float))
    U.park()


class StopAfterDecisions(BaseException):
    """--stop-after-decisions: stage 1 of a branched route ends here (rows kept; a BaseException so the planner's
    `except Exception` handlers pass it through)."""


class RowCap(BaseException):
    """--max-rows: a search branch reached its row cap (rows kept; a BaseException as StopAfterDecisions)."""


# ------------------------------------------------------------------------------------------------ the oracle
def first_moves(U, rope, moves, log):
    """--first: moves of the oracle (Untie / RobotUntie methods, unchanged) in a given order before its deletion loop;
    every move is a decision record. run_oracle2 itself lifts a wrap off a post only after a stalled deletion."""
    for mv in moves:
        if not U.visits(rope):
            break
        if mv == "lift_post":
            wraps = U.post_wraps(rope)
            rec = U.lift_over_post(rope, 0, max(wraps, key=lambda w: abs(w["winding"]))) if wraps else None
        elif mv.startswith("lift_exit"):
            # lift_exit[:offset:clear:out] -- a wrap that hugs the post has no pinch room on itself (the gripper meets
            # the post): the same lift_over_post with the pinch moved `offset` past the wrap on the free side and the
            # hand taken higher: the wrap's far side slides up the post and over its top once the hand is at about
            # twice the post's height (a taut rope round a post rises linearly along its path)
            arg = mv.split(":")[1:]
            off, clear, out = (float(arg[i]) if len(arg) > i else d for i, d in enumerate((.05, .10, .10)))
            wraps = U.post_wraps(rope)
            rec = None
            if wraps:
                w = dict(max(wraps, key=lambda x: abs(x["winding"])))
                w["s0"] = w["s1"] = w["s1"] + off if U.free_at_end(rope) else w["s0"] - off
                rec = U.lift_over_post(rope, 0, w, clear=clear, out=out)
                rec["exit_station"] = round(float(w["s0"]), 3)
        elif mv == "loosen":
            rec = U.loosen(rope)
        elif mv == "lift_across":
            rec = U.lift_across(rope, 0)
        elif mv == "clear_tail":
            rec = U.clear_tail(rope, 0)
        elif mv == "draw":
            rec = U.draw_free_end(rope, 0, margin=.05)
        else:
            raise SystemExit(f"--first: unknown move {mv}")
        if rec is None:
            log("first", mv, "nothing to do (no wrap round a post)")
            continue
        rec.setdefault("kind", mv)
        rec["first"] = mv
        rec.setdefault("crossings_after", len(U.lab.crossings()))
        if not any(d is rec for d in U.decisions):      # loosen logs itself when it pushed
            U.decisions.append(rec)
        log("decision", json.dumps({x: rec.get(x) for x in ("kind", "first", "crossings_before", "crossings_after",
                                                             "result", "winding")}, default=float))


def run_oracle(lab, U, scene, log, args):
    order = [r["name"] for r in scene.get("ropes", [])] or list(lab.rods)
    if len(order) > 1:
        order = order[::-1]            # hard: B first (as run_tier.py)
    for name in order:
        if getattr(args, "first", None):
            first_moves(U, name, args.first.split(","), log)
        U.run_oracle2(name, stroke_max=args.stroke, max_decisions=args.max_decisions)

    def goal_read(tag):                # the Session's own goal metrics + the lab's per-rope reading (regions included)
        try:
            gm = lab.goal_metrics(region=scene.get("regions") or None)
        except Exception as e:  # noqa: BLE001
            gm = dict(error=repr(e)[:120])
        rec = dict(tag=tag, t=round(lab.t, 3), row=int(lab.s.n_steps), task=dict(lab.s.env._task.result.metrics), lab=gm)
        U.goal_reads = getattr(U, "goal_reads", []) + [rec]      # (not in U.decisions: a --schedule counts those)
        log("goal_metrics", json.dumps(rec, default=float)[:1500])
    goal_read("after_decisions")
    # untied but strands lying closer than the goal's 2 r in plan (no crossing left): straighten by drawing the free end
    # away from the clamp (Untie's draw_free_end), at most twice
    for name in order:
        for k in range(2):
            m = lab.s.env._task.result.metrics
            ropes = m.get("ropes") or {}
            clear = (ropes.get(name) or {}).get("planar_clear", m.get("planar_clear"))
            if lab.crossings() or clear:
                break
            rec = U.draw_free_end(name, 90 + k, length=.12, margin=.05)
            rec["why"] = "no crossing left, but strands closer than 2 r in plan"
            U.decisions.append(rec)
            log("straighten", json.dumps({x: rec.get(x) for x in ("kind", "result", "moved", "stop", "crossings_after")}))
    # no crossing left but strands still closer than one diameter in plan: the robot's separate_close move
    for name in order:
        m = lab.s.env._task.result.metrics
        ropes = m.get("ropes") or {}
        clear = (ropes.get(name) or {}).get("planar_clear", m.get("planar_clear"))
        if not lab.crossings() and not clear and hasattr(U, "separate_close"):
            recs = U.separate_close(name)
            U.decisions.append(dict(kind="separate_close", rope=name, moves=recs, crossings_after=len(lab.crossings())))
            log("separate", json.dumps(recs, default=str)[:600])
    # untied and planar clear but a strand lies on a fixture (above one diameter + 3 mm): the robot's flatten move
    if getattr(args, "flatten", False):
        for name in order:
            rr = (lab.goal_metrics().get("ropes") or {}).get(name) or {}
            if not lab.crossings() and rr.get("planar_clear") and not rr.get("on_table") and hasattr(U, "flatten_high"):
                recs = U.flatten_high(name)
                U.decisions.append(dict(kind="flatten", rope=name, moves=recs, crossings_after=len(lab.crossings())))
                log("flatten", json.dumps(recs, default=str)[:600])
    if getattr(args, "fixup_if_untied", False) and not lab.crossings():
        # untied: no region layout (it re-tangled hard_nostop inst10); only the parts of a rope outside its goal region
        # (hard: a in the front strip, b in the back strip) are carried back in (Untie.layout_fixup)
        for name in order:
            reg = (scene.get("regions") or {}).get(name)
            if reg and hasattr(U, "layout_fixup"):
                rr = (lab.goal_metrics(region=scene.get("regions")).get("ropes") or {}).get(name) or {}
                if rr.get("in_region") is False:
                    rec = U.layout_fixup(name, reg)
                    log("layout_fixup", json.dumps({k: rec.get(k) for k in ("rope", "outside_fraction", "crossings_after")}
                                                   if isinstance(rec, dict) else rec, default=float)[:400])
        goal_read("after_fixup")
        return
    if getattr(args, "no_layout", False):
        return
    if getattr(args, "layout_if_tangled", False) and not lab.crossings():
        return                         # untied: the region layout only re-tangles
    paths = (scene.get("oracle") or {}).get("layout_path") or {}
    for name in order:
        if name in paths and (scene.get("regions") or {}).get(name):
            U.layout(name, paths[name])
    for name in order:                 # then every rope's parts still outside its region are carried back in (run_tier)
        if (scene.get("regions") or {}).get(name) and hasattr(U, "layout_fixup"):
            U.layout_fixup(name, scene["regions"][name])
    if getattr(args, "flatten", False) or getattr(args, "layout_stop_untied", False):
        if getattr(args, "flatten", False):
            for name in order:
                rr = (lab.goal_metrics().get("ropes") or {}).get(name) or {}
                if not lab.crossings() and rr.get("planar_clear") and not rr.get("on_table") and hasattr(U, "flatten_high"):
                    recs = U.flatten_high(name)
                    log("flatten", json.dumps(recs, default=str)[:600])
        goal_read("after_layout")


def run_layout(lab, U, scene, log, args):
    """--mode layout: only the region layout (RobotUntie.layout) of every rope that has a path, from the current state
    (the birth, or a --resume prefix such as an untied run's rows); at most --layout-max placements per rope."""
    order = [r["name"] for r in scene.get("ropes", [])] or list(lab.rods)
    if len(order) > 1:
        order = order[::-1]
    paths = (scene.get("oracle") or {}).get("layout_path") or {}
    for name in order:
        if name in paths and (scene.get("regions") or {}).get(name):
            rec = U.layout(name, paths[name], max_placements=args.layout_max)
            log("layout", json.dumps({k: rec.get(k) for k in ("rope", "handovers", "crossings_after")}))
    if args.layout_max is None:
        for name in order:
            if (scene.get("regions") or {}).get(name) and hasattr(U, "layout_fixup"):
                rec = U.layout_fixup(name, scene["regions"][name])
                log("layout_fixup", json.dumps({k: rec.get(k) for k in ("rope", "outside_fraction", "crossings_after")}))


BREAKERS = ("pull_free", "pull_both", "route_rule", "route_rule_tangent", "drag")


def run_breaker(lab, U, scene, log, name):
    """The lab's anti-hack breakers (untie.Untie's own methods, as run_tier.py calls them) on robot hands."""
    from topogym.planning.knot_untying.redesign.mjlab import to_local
    ropes = [r["name"] for r in scene.get("ropes", [])] or list(lab.rods)
    rope = ropes[0]
    if name == "pull_free":
        for r in ropes:
            U.run_pull_free(r)
    elif name == "pull_both":
        U.run_pull_both(ropes)
    elif name == "route_rule":
        U.run_route_rule(rope, hold="far", direction="away", max_decisions=8)
    elif name == "route_rule_tangent":
        U.run_route_rule(rope, hold="far", direction="tangent", max_decisions=8)
    elif name == "drag":
        reg = (scene.get("regions") or {}).get(rope)
        if reg:
            c = np.mean(np.asarray(reg, float), axis=0)
            p = to_local(lab.point(rope, lab.length(rope) / 2))
            delta = np.r_[c - p[:2], 0.]
        else:
            delta = np.array([0., .2, 0.])
        U.run_drag(rope, delta)
    else:
        raise SystemExit(f"unknown breaker {name}")
    log("breaker", name, json.dumps([{k: d.get(k) for k in ("kind", "result", "moved", "stop", "crossings_after")}
                                     for d in U.decisions], default=str)[:2000])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pos", nargs="+", metavar="instance.json tier out", help="<instance.json> <tier> <out dir>; with "
                    "--adhoc: <tier|primitives> <out dir>")
    ap.add_argument("--adhoc", default=None, help="released knot instance.json: dev Session with the weld grasp")
    ap.add_argument("--scene", default=None, help="lab scene (scene_start.json) for fixtures / anchors / regions")
    ap.add_argument("--budget-min", type=float, default=None)
    ap.add_argument("--hold-s", type=float, default=12.)
    ap.add_argument("--limit", type=float, default=8.)
    ap.add_argument("--stroke", type=float, default=.18)
    ap.add_argument("--speed", type=float, default=.06)
    ap.add_argument("--free-speed", type=float, default=.25)
    ap.add_argument("--max-decisions", type=int, default=16)
    ap.add_argument("--press-pins", action="store_true", help="pins may also press the over strand (closed fingers)")
    ap.add_argument("--no-pins", action="store_true", help="deletion strokes without pins (tangent, <= 8 cm)")
    ap.add_argument("--mode", choices=("oracle", "primitives", "breaker", "layout"), default=None)
    ap.add_argument("--layout-max", type=int, default=None, help="with --mode layout: at most this many placements")
    ap.add_argument("--breaker", choices=BREAKERS, default=None, help="with --mode breaker: which lab breaker")
    ap.add_argument("--resume", default=None, help="a partial run's rows (trajectory.partial.npz / privileged.npz): "
                    "replayed first, then the oracle continues live")
    ap.add_argument("--resume-rows", type=int, default=None, help="with --resume: replay only its first N rows")
    ap.add_argument("--restore", default=None, help="a checkpoint .pkl (ckpt.py: an earlier run's --checkpoint-decisions "
                    "or --checkpoint-untied state, its rows included) restored instead of replaying rows (--resume); the "
                    "oracle then decides again on the live rope: a branch of the decision-level route search")
    ap.add_argument("--checkpoint-decisions", action="store_true", help="write <out>/ck/dNN.pkl (ckpt.py) + dNN.json "
                    "after every decision record: the branch points of a decision-level route search (--restore)")
    ap.add_argument("--max-rows", type=int, default=None, help="end the run (rows kept, as --stop-after-decisions) once "
                    "the row stream (prefix included) reaches this many rows: a search branch that grew too long")
    ap.add_argument("--first", default=None, help="comma list of the oracle's own moves made before its deletion loop, "
                    "each a decision record (a decision-order variant of the route search): lift_post (the largest wrap "
                    "round a post is lifted off), lift_exit[:offset:clear:out] (the same lift with the pinch past the "
                    "wrap on the free side, for a wrap that hugs the post), loosen, lift_across, clear_tail, draw")
    ap.add_argument("--stop-after-decisions", type=int, default=None, help="end the run once this many decision "
                    "records are logged (rows saved, not solved): stage 1 of a branched route; stage 2 resumes its "
                    "rows (--resume <out>/privileged.npz) with the branch's flags")
    ap.add_argument("--checkpoint-untied", action="store_true", help="write <out>/untied_checkpoint.pkl (ckpt.py) at the "
                    "first decision that leaves no crossing and <out>/final_checkpoint.pkl at the end: a placement lab "
                    "restores them (place_lab.py --restore) instead of replaying the rows")
    ap.add_argument("--schedule", default=None, help="JSON (or a JSON file): [{decisions: n, args: flags}, ...] -- "
                    "variant flags per phase of decisions (re-running a branched line from its birth)")
    # move variants (RobotUntie attributes; defaults = the reference behaviour)
    ap.add_argument("--stroke-lift", type=float, default=.02, help="deletion strokes lift the strand this much (m)")
    ap.add_argument("--pull-offset", type=float, default=.035, help="pull station this far past the crossing (m)")
    ap.add_argument("--high-lift", type=float, default=.065, help="lift over a curb (m); 0: never")
    ap.add_argument("--clamp-keep", type=float, default=.15, help="clear_tail's tail line keeps this far from clamps (m)")
    ap.add_argument("--pin-order", choices=("near", "far"), default="near")
    ap.add_argument("--no-clear-tail", action="store_true", help="no clear_tail after a deletion")
    ap.add_argument("--endgame", action="store_true", help="a stalled deletion ends with the robot's lift_across move")
    ap.add_argument("--gap", type=float, default=None, help="the arms' planned clearance GAP_M (m)")
    ap.add_argument("--real-fingers", action="store_true", help="fixture distances with the fingers at their real "
                    "opening (the planned pad gap, else the live fingers); without it the planner model keeps qpos0's "
                    "over-closed fingers (12-23 mm narrower per side)")
    ap.add_argument("--flatten", action="store_true", help="after the deletions: an untied, planar-clear rope with a strand "
                    "above one diameter + 3 mm (on a fixture) is pinched there, carried 5 cm off the fixture and laid down")
    ap.add_argument("--layout-stop-untied", action="store_true", help="the region layout (run while crossings remain) stops "
                    "at the first placement that leaves no crossing; then the fix-up, flatten and a goal read")
    ap.add_argument("--fixup-if-untied", action="store_true", help="an untied rope gets no region layout, only layout_fixup "
                    "of the ropes with parts outside their goal region (after --flatten)")
    ap.add_argument("--layout-if-tangled", action="store_true", help="run the region layout / fix-up only when crossings "
                    "remain after the deletions (it then works as an endgame; an untied rope is left to the goal)")
    ap.add_argument("--no-layout", action="store_true", help="skip the region layout and its fix-up (the goal has no "
                    "region; the layout re-tangled hard_nostop inst10 from 0 to 3 crossings)")
    ap.add_argument("--stall-adapt-grasp", action="store_true", help="with the stall adaptation also a failed grasp "
                    "(retried with the other hand, then without a pin) and clear_tail's end taken by the other arm "
                    "(= robot_untie f16db2e8..51969a28's --stall-adapt)")
    ap.add_argument("--fast", action="store_true", help="the user's 10-12 speed rule: free moves 0.40 m/s (0.25), "
                    "descents 0.16 (0.12), lifts 0.08 (0.05), drops lowered at 0.06 (0.03) and lifted at 0.3 (0.2), "
                    "presses 0.08 (0.05), tail carries 0.12 (0.08), joint moves 1.5x; strokes keep --speed")
    ap.add_argument("--turbo", type=float, default=1., help="with --fast: its free moves, descents, lifts, drops, presses, "
                    "tail carries and joint moves x this factor, and the lab's 0.2 s minimum move time, 1.2 rad/s turn rate "
                    "and 0.3 s close scaled by it (10-14, route length under the 20 %% rule; strokes keep --speed; every row "
                    "stays within the standard joint speed limit through the runner's limiter)")
    ap.add_argument("--converge-scale", type=float, default=1., help="the lab's converge (a hand held still until its "
                    "contact point settles, before a grasp and after moves) waits at most its max_s x this (10-14, route "
                    "length; it still stops early once the hand has settled)")
    ap.add_argument("--stall-adapt", action="store_true", help="a deletion stroke stopped short changes the next one: "
                    "no pin after an arms stop, turned 40 deg after a fixture stop, the other hand after a reach stop")
    ap.add_argument("--retry-room", action="store_true", help="a pinch the World did not bind retries at the roomiest "
                    "station within 3 cm, from 5 cm above (default: the same station from 1.5 cm)")
    ap.add_argument("--tight", action="store_true", help="lower margins: arms 12 mm (+5 mm next to a holding arm), "
                    "paths 3 mm from the other arm / 2 mm from fixtures, pinches 2.5 mm from fixtures, planned strokes "
                    "4 / 2.5 mm, live stroke stops 3 / 2 mm (--gap / --hold-extra still override)")
    ap.add_argument("--hold-extra", type=float, default=None, help="extra clearance next to a holding arm (m)")
    ap.add_argument("--carry-speed", type=float, default=None, help="layout carries' speed (m/s)")
    ap.add_argument("--pinch-clearance", type=float, default=.010, help="a pinch station needs this much room from "
                    "other rope / fixtures (m; the family's 10 mm)")
    ap.add_argument("--layout-lift", type=float, default=.08, help="layout carries lift the station this much (m): "
                    "high enough that the hanging strand clears rope already laid (knot 10-11: the lab's 5 cm "
                    "serpentine dragged rope across rope)")
    a = ap.parse_args(argv)
    if a.adhoc:
        if len(a.pos) != 2:
            ap.error("--adhoc takes <tier|primitives> <out dir>")
        a.instance, (a.tier, a.out) = None, a.pos
    else:
        if len(a.pos) != 3:
            ap.error("expected <instance.json> <tier> <out dir>")
        a.instance, a.tier, a.out = a.pos
    mode = a.mode or ("primitives" if a.tier == "primitives" else "oracle")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(adhoc_released=a.adhoc) if a.adhoc else dict(instance=str(Path(a.instance).resolve()))
    t0 = time.time()

    def log(*x):
        print(*x, flush=True)
    s = session_from_spec(spec)
    (out / "session_spec.json").write_text(json.dumps(spec))
    build_s = time.time() - t0
    log(f"session built in {build_s:.0f} s; instance_sha {s.instance_sha}; rows {s.n_steps}")
    from topogym.planning.knot_untying.redesign.robot_lab import Budget, EpisodeEnded, RobotLab, Solved
    from topogym.planning.knot_untying.redesign.robot_untie import RobotUntie
    # everything the oracle imports lazily, now: a later rebuild of the dev runtime root (rrt) swaps its directory
    import topogym.planning.knot_untying.artifacts  # noqa: F401
    import topogym.planning.knot_untying.redesign.birth  # noqa: F401
    import topogym.tasks.knot_untying.redesign.geometry  # noqa: F401
    import topogym.tasks.knot_untying.redesign.goal  # noqa: F401
    import topogym.utils.geometry.passage  # noqa: F401
    import topogym.planning.knot_untying.route_rule  # noqa: F401
    scene = scene_for(a, s, None if a.adhoc else a.instance)
    names = sorted(r["name"] for r in scene.get("ropes", []))
    if names and names != sorted(s.env._world._ropes):
        raise SystemExit(f"scene ropes {names} != the Session's {sorted(s.env._world._ropes)}")
    s.record_start()
    if a.restore and a.resume:
        raise SystemExit("--restore or --resume, not both")
    if a.restore:
        import ckpt
        t_r = time.time()
        meta = ckpt.restore_checkpoint(s, a.restore)
        resumed = int(s.n_steps)
        log(f"restored {a.restore} in {time.time() - t_r:.1f} s: rows {s.n_steps} (t = {s.sim_time_s:.3f} s); "
            f"meta {json.dumps(meta, default=str)[:300]}")
    else:
        resumed = replay_prefix(s, a.resume, log, a.resume_rows) if a.resume else 0
    lab = RobotLab(s, scene, hand_limit_n=a.limit, budget_s=None if a.budget_min is None else 60 * a.budget_min,
                   pinch_clearance_m=a.pinch_clearance,
                   log=log, trace_path=out / "trace.jsonl")
    U = RobotUntie(lab, scene, limit=a.limit, speed=a.speed, free_speed=a.free_speed, press_pins=a.press_pins,
                   use_pins=not a.no_pins)
    def apply_flags(f):
        """The move-variant flags onto the live planner (also used by --schedule's later phases)."""
        U.speed, U.free_speed = f.speed, f.free_speed
        U.fast = bool(getattr(f, "fast", False))
        U.turbo = float(getattr(f, "turbo", 1.) or 1.) if U.fast else 1.
        U.descend_speed = .16 * U.turbo if U.fast else .12
        if U.fast:
            U.free_speed = max(U.free_speed, .40 * U.turbo)
        T_ = type(lab)
        lab.move_min_time, lab.rot_speed, lab.close_s = (T_.move_min_time / U.turbo, T_.rot_speed * U.turbo,
                                                         T_.close_s / U.turbo)
        lab.converge_scale = float(getattr(f, "converge_scale", 1.) or 1.)
        U.press_pins, U.use_pins = bool(f.press_pins), not f.no_pins
        U.limit = f.limit
        U.stroke_lift, U.pull_offset = f.stroke_lift, f.pull_offset
        U.high_lift = f.high_lift if f.high_lift > 0 else None
        U.clamp_keep, U.pin_order, U.clear_tail_on = f.clamp_keep, f.pin_order, not f.no_clear_tail
        T, tight = type(U), bool(getattr(f, "tight", False))
        for k, v in (("PATH_NEED_M", .003), ("PATH_FIX_M", .002), ("FIXTURE_GAP_M", .0025), ("STROKE_NEED_M", .004),
                     ("STROKE_FIX_M", .0025), ("STOP_ARMS_M", .003), ("STOP_FIX_M", .002)):
            setattr(U, k, v if tight else getattr(T, k))
        U.GAP_M = f.gap if f.gap is not None else (.012 if tight else T.GAP_M)
        U.HOLD_EXTRA_M = f.hold_extra if f.hold_extra is not None else (.005 if tight else T.HOLD_EXTRA_M)
        U.layout_carry_speed = f.carry_speed
        U.layout_lift = f.layout_lift
        lab.pinch_clearance = float(f.pinch_clearance)
        lab.real_fingers = bool(getattr(f, "real_fingers", False))
        U.endgame = bool(f.endgame)
        U.retry_room = bool(getattr(f, "retry_room", False))
        U.stall_adapt = bool(getattr(f, "stall_adapt", False))
        U.stall_adapt_grasp = bool(getattr(f, "stall_adapt_grasp", False))
        U.layout_stop_untied = bool(getattr(f, "layout_stop_untied", False))

    apply_flags(a)
    if spec.get("instance"):            # the arms' home joints (the instance's reset targets) for home_other
        try:
            rj = json.loads(Path(spec["instance"]).read_text()).get("reset_joint_targets")
            layout = list(s.action_layout)
            if rj:
                U.home_q = {h: np.array([rj[layout.index(f"{h}/{n}")] for n in
                                         ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")])
                            for h in ("left", "right")}
        except Exception as e:  # noqa: BLE001
            log("home_q unavailable:", repr(e)[:120])
    # --schedule: a re-run of a branched line (knot 10-11: final references are re-recorded end to end on a new head).
    # JSON [{"decisions": n1, "args": "<variant flags>"}, {"args": "..."}]: phase i's flags apply until the oracle has
    # logged n_i more decisions, then phase i + 1's (the last phase has no "decisions").
    if a.schedule:
        phases = json.loads(Path(a.schedule).read_text() if Path(a.schedule).is_file() else a.schedule)
        state = dict(i=0, at=0)

        def phase_flags(i):
            return ap.parse_args([*a.pos, *shlex.split(phases[i].get("args", ""))])

        apply_flags(phase_flags(0))

        class _Decisions(list):
            def append(self, x):
                super().append(x)
                i = state["i"]
                if i + 1 < len(phases) and len(self) - state["at"] >= int(phases[i]["decisions"]):
                    state["i"], state["at"] = i + 1, len(self)
                    apply_flags(phase_flags(i + 1))
                    log("schedule", f"phase {i + 1} from decision {len(self)}: {phases[i + 1].get('args', '')}")
        U.decisions = _Decisions(U.decisions)
    if a.checkpoint_untied:             # before the stop hook, so a stage that stops at the untying still writes it
        import ckpt
        ck = dict(done=False)

        class _Checkpoint(type(U.decisions)):
            def append(self, x):
                super().append(x)
                if not ck["done"] and isinstance(x, dict) and x.get("crossings_after") == 0:
                    ck["done"] = True
                    n = ckpt.save_checkpoint(s, out / "untied_checkpoint.pkl", dict(decisions=len(self), row=int(s.n_steps),
                                                                                   kind=x.get("kind"), argv=sys.argv[1:]))
                    log("CHECKPOINT", f"untied at decision {len(self)}, row {s.n_steps}: untied_checkpoint.pkl ({n} bytes)")
        U.decisions = _Checkpoint(U.decisions)
    if a.checkpoint_decisions:          # also before the stop hook: the last record before a stop is a branch point too
        import ckpt
        (out / "ck").mkdir(exist_ok=True)

        def shape():
            """The rope's state for the search: ~250 centre-line points per rope (local frame, m); per rope the free
            tail past the first under-crossing, the tail past every under-crossing (their sum = the rope still to be
            drawn through crossings) and the wraps round posts."""
            out_ = dict(centerline={k: np.asarray(v)[::max(1, len(v) // 250)].round(4).tolist()
                                    for k, v in lab.centerlines_local().items()}, ropes={})
            for rope in lab.rods:
                rem = U.first_under(rope)[1]
                fae, L = U.free_at_end(rope), lab.length(rope)
                out_["ropes"][rope] = dict(first_under_remaining=None if rem is None else round(float(rem), 4),
                                           under_tails=[round(float(L - v["s"] if fae else v["s"]), 4)
                                                        for v in U.visits(rope) if not v["over"]],
                                           wraps=[round(float(w["winding"]), 1) for w in U.post_wraps(rope)])
            return out_
        (out / "ck" / "start.json").write_text(json.dumps(dict(row=int(s.n_steps), crossings=len(lab.crossings()),
                                                               restored_from=a.restore, **shape())))

        class _CheckpointEvery(type(U.decisions)):
            def append(self, x):
                super().append(x)
                n, xd = len(self), (x if isinstance(x, dict) else {})
                meta = dict(decisions=n, row=int(s.n_steps), t=round(float(s.sim_time_s), 3), kind=xd.get("kind"),
                            crossings_after=xd.get("crossings_after"), crossings=len(lab.crossings()),
                            result=str(xd.get("result"))[:160], restored_from=a.restore, resume=a.resume,
                            argv=sys.argv[1:], state_sha=s.state_sha())
                nb = ckpt.save_checkpoint(s, out / "ck" / f"d{n:02d}.pkl", meta)
                (out / "ck" / f"d{n:02d}.json").write_text(json.dumps(dict(meta, **shape()), default=str))
                log("CKPT", f"decision {n} row {s.n_steps} crossings {meta['crossings']} state {meta['state_sha']} "
                            f"({nb} bytes)")
        U.decisions = _CheckpointEvery(U.decisions)
    if a.stop_after_decisions is not None:
        n_stop = int(a.stop_after_decisions)

        class _StopAfter(type(U.decisions)):
            def append(self, x):
                super().append(x)
                if len(self) >= n_stop:
                    raise StopAfterDecisions(len(self))
        U.decisions = _StopAfter(U.decisions)
    rec = Recorder(s, out)
    orig_after = lab._after_step

    def after(label):
        orig_after(label)
        rec.partial()
        if a.max_rows is not None and s.n_steps >= a.max_rows:
            raise RowCap(int(s.n_steps))
    lab._after_step = after
    log(f"fingertips below the pads' contact point: {lab.tip_below}; gripper radius {lab.gripper_radius}; "
        f"crossings at start {len(lab.crossings())}")
    if mode == "breaker" and not a.breaker:
        raise SystemExit("--mode breaker needs --breaker")
    import hashlib
    from topogym.planning.knot_untying.redesign import robot_lab as _rl, robot_untie as _ru, untie as _un
    code = {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()[:16]
            for m in (_rl, _ru, _un)}
    code[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    log("code", json.dumps(code))
    extra = dict(tier=a.tier, mode=mode, breaker=a.breaker, spec=spec, build_s=round(build_s, 1), argv=sys.argv[1:],
                 code=code,
                 crossings_start=len(lab.crossings()), resumed_rows=resumed, partial=True)
    rec.dump = lambda: dump(out, U, lab, s, extra)
    err, solved = None, False
    try:
        if mode == "primitives":
            run_primitives(lab, U, log)
        elif mode == "breaker":
            run_breaker(lab, U, scene, log, a.breaker)
            U.park()
        elif mode == "layout":
            run_layout(lab, U, scene, log, a)
            U.park()
        else:
            run_oracle(lab, U, scene, log, a)
            U.park()
        # hold until the Session confirms the goal (episode end) or the hold budget runs out
        t_end = lab.t + a.hold_s
        while lab.t < t_end:
            lab._step("final_hold")
    except Solved:
        solved = True
        log(f"SOLVED at row {s.n_steps}, t = {s.sim_time_s:.3f} s")
        cur = getattr(U, "current", None)
        if cur is not None and not any(d is cur for d in U.decisions):   # the move the episode ended in
            cur.setdefault("result", "the Session ended the episode in success during this move")
            cur["crossings_after"] = len(lab.crossings())
            U.decisions.append(cur)
    except StopAfterDecisions as e:
        extra["stopped_after_decisions"] = int(e.args[0])
        log("STOPPED", f"after {e.args[0]} decision records at row {s.n_steps} (--stop-after-decisions)")
    except RowCap as e:
        extra["stopped_at_max_rows"] = int(e.args[0])
        log("STOPPED", f"at row {s.n_steps} (--max-rows {a.max_rows})")
    except Budget as e:
        err = f"Budget: {e}"
        log("BUDGET", e)
        try:
            lab.budget_s = None
            U.park()
            for _ in range(int(2. / lab.dt)):
                lab._step("budget_hold")
        except Solved:
            solved = True
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        log("ERROR", err)
        traceback.print_exc()
    if a.checkpoint_untied:
        try:
            ckpt.save_checkpoint(s, out / "final_checkpoint.pkl", dict(row=int(s.n_steps), solved=solved, argv=sys.argv[1:]))
        except Exception as e:  # noqa: BLE001  (evidence only)
            log("final checkpoint failed:", repr(e)[:200])
    acts = s.record_stop()
    extra.update(error=err, solved=solved, partial=False)
    s.save_trajectory(acts, out / "privileged.npz", dict(
        source="knot-robot: the redesign's untying oracle (untie.Untie run_oracle2) on the ALOHA, contact_weld_v1, "
               "rows as accepted by the Session (standard-rate limited)", tier=a.tier, mode=mode, solved=solved,
        error=err, spec=spec))
    res = dump(out, U, lab, s, extra)
    log("RESULT", json.dumps(dict(tier=a.tier, mode=mode, solved=solved, error=err, rows=res["rows"], sim_s=res["sim_s"],
                                  wall_s=res["wall_s"], crossings=res["crossings"], state_sha=res["status"]["state_sha"],
                                  max_force=res["max_force"], slips=res["slips"], reach_stops=len(res["reach_stops"]))))
    s.close()


if __name__ == "__main__":
    main()
