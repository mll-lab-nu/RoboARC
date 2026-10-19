"""The buzz finish on MuJoCo, planned from the measured state with the family's own recipes
(topogym/planning/buzz_wire_3d/recipes.py: acquire, checked_turn, PARTIAL_RELEASE, ACQUIRE_2, final_release), whose
candidate lists were tuned to Genesis' post-release wand poses. Here the candidates follow the measured pose instead:
- the target yaw is the finish pose's handle heading (endpoint_pose(route, 1); the task's angle allows the half turn
  about the handle axis, which keeps the heading);
- a turn tries yaws from the target back toward the wand's current heading (1 deg steps), each with a few xy offsets,
  and takes the first collision-checked one (the recipe's own BuzzIK checks and clearances);
- after a turn that ends within REACH_DEG of the target the tool is released (final_release); otherwise the recipe's
  partial release, its second grasp and a second turn follow; a final pose outside the task's tolerance after release
  starts another grasp-and-turn round (at most ROUNDS).
Every stage is an ordinary BuzzWirePolicy parameter set (make_stage_policy), executed by the caller's Session runner."""
import json

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.assets.geometry.buzz_wire_3d import endpoint_pose
from topogym.planning.buzz_wire_3d import recipes

import os as _os
# the task accepts 10 deg (finish_angle_tolerance_deg); the planned turn's remaining angle below which the final release
# follows (default 4 deg: the final releases so far landed within ~1 deg of the plan); BUZZ_REACH_DEG overrides
REACH_DEG = float(_os.environ.get("BUZZ_REACH_DEG", "4"))
ROUNDS = int(_os.environ.get("BUZZ_ROUNDS", "3"))
# experiment: BUZZ_PARTIAL=deck releases between rounds with the recipe's fast open-and-withdraw onto the deck
# (RELEASE_ONTO_DECK, as centre_release) instead of its PARTIAL_RELEASE


def partial_release():
    if _os.environ.get("BUZZ_PARTIAL") == "deck":
        return recipes.script(recipes.RELEASE_ONTO_DECK)
    return recipes.script(SLOW_PARTIAL_RELEASE if STAGED else recipes.PARTIAL_RELEASE)


# The Robotiq on its spring links' springref 2.62 (from 2026-10-11; robot.SPRING_REFS), after the buzz redesign agent's
# oracle (its topogym/planning/buzz_wire_3d/redesign/oracle.py: DECK_STAGE, the slow release): wide open, the pads now
# hang ~9.7 mm below the TCP and land on the deck before the wand, so a grasp descends with the fingers pre-closed and
# its approach check tolerates the deck slab (role finish_support; the wire, its stops and the robot's other contacts
# stay checked); a release opens slowly and backs off slowly, since a fast open set the parallel-held wand turning.
# BUZZ_STAGED=0: the springref-0 recipes (the -i01 references before 2026-10-11).
STAGED = _os.environ.get("BUZZ_STAGED", "1") != "0"
PRECLOSE = .30                 # the gripper command (rad) the hand closes to above the grasp, as DECK_STAGE
DESCENT_SPEED = .02            # m/s, the pre-closed hand's descent onto the grasp
RETREAT_SPEED = .015           # m/s, the open hand backing off the released wand
LIFT = float(_os.environ.get("BUZZ_LIFT", ".008"))   # m, the held wand lifted off the deck after a grasp (recipe: 5 mm)


def deck_tolerant(env, kin):
    """kin.clearance without the deck slab (the wire_base geoms of role finish_support)."""
    from topogym.planning.buzz_wire_3d.kinematics import manager
    roles = env._world._contact_collector.roles
    geoms = [g for name in ("wire", "wire_base") for link in env._world.entities[name].links for g in link.geoms]
    kin.fixed = manager([o for o, g in zip(kin.fixed_objects, geoms) if roles[g.idx] != "finish_support"])
    return kin


def staged(acquire_params, approach):
    """An acquire stage as transfer only, then the pre-closed descent, the close and the lift (a list of stage
    parameters)."""
    transfer = dict(acquire_params, stop_after_transfer=True)
    return [transfer, recipes.script([
        dict(operation="grip", value=PRECLOSE, seconds=1.),
        dict(operation="tcp", local_translation_m=[0., 0., approach], seconds=approach / DESCENT_SPEED),
        dict(operation="grip", value=.8, seconds=2.),
        dict(operation="hold", seconds=1.),
        dict(operation="tcp", translation_m=[0., 0., LIFT], seconds=2.)])]


def ramp(start, end, seconds, steps=6):
    return [dict(operation="grip", value=float(start + (end - start) * (i + 1) / steps), seconds=seconds / steps)
            for i in range(steps)]


def slow_final_release(env):
    """recipes.final_release, slow: open to the release command over 1.2 s, hold, back off 8 cm at 1.5 cm/s, settle."""
    hand = env._world.read_state()["grasp"]["hands"]["arm"]
    now = .8 * hand["last_fraction"]
    release = min(.25, now - .15)
    return recipes.script([dict(operation="hold", seconds=.5)] + ramp(now, release, 1.2)
                          + [dict(operation="hold", seconds=1.),
                             dict(operation="tcp", local_translation_m=[0., 0., -.08], seconds=.08 / RETREAT_SPEED),
                             dict(operation="hold", seconds=8.)])


SLOW_PARTIAL_RELEASE = [dict(operation="hold", seconds=.5), dict(operation="grip", value=.78, seconds=.75),
                        dict(operation="tcp", local_translation_m=[0., 0., -.08], seconds=.08 / RETREAT_SPEED),
                        dict(operation="hold", seconds=1.), dict(operation="grip", value=.45, seconds=2.)]
OFFSETS = [(.003, .002), (0., 0.), (.002, .003), (.003, 0.), (0., .003), (-.002, .002), (.002, -.002)]


def heading(r):
    x = r.apply([1., 0., 0.])
    return float(np.degrees(np.arctan2(x[1], x[0])))


def target_yaw(env):
    _, end_r = endpoint_pose(env._task.route, 1)
    return heading(end_r)


def wand_yaw(env):
    q = np.asarray(env._world.read_state()["bodies"]["wand"]["quaternion_wxyz"])
    return heading(Rotation.from_quat(q[[1, 2, 3, 0]]))


def wrap(a):
    return (a + 180.) % 360. - 180.


def yaw_candidates(env, *, span=60., step=1.):
    """From the target toward the current heading (the shorter way), at most `span` degrees."""
    goal, now = target_yaw(env), wand_yaw(env)
    delta = wrap(now - goal)
    n = int(min(abs(delta), span) // step) + 1
    return [goal + np.sign(delta) * k * step for k in range(n)], goal, now


def turn(env, output, index, *, final_height, minimum_clearance, speeds, errors, span=60.):
    yaws, goal, now = yaw_candidates(env, span=span)
    params = recipes.checked_turn(env, output, index=index, yaws=yaws, offsets=OFFSETS, final_height=final_height,
                                  minimum_clearance=minimum_clearance, speeds=speeds, errors=errors)
    import json
    chosen = json.loads((output / f"finish_turn_{index}.json").read_text())["yaw"]
    return params, dict(goal_yaw=goal, start_yaw=now, chosen_yaw=chosen, remaining_deg=abs(wrap(chosen - goal)))


def acquire_first(env, output, index):
    """The recipe's first grasp (yaw 225, tilt 45); then its second grasp (ACQUIRE_2), the development runner's finish
    grasps (runner.py: tilt 75 / 105 about the handle) and other approach yaws, while a free-hand path fails."""
    errors = []
    try:
        return recipes.acquire(env, output, index, 225., 45., 0., .04)
    except ValueError as error:
        errors.append(f"225/45: {error}")
    try:
        return recipes.acquire(env, output, **dict(recipes.ACQUIRE_2, index=index))
    except ValueError as error:
        errors.append(f"ACQUIRE_2: {error}")
    for yaw, tilt in ((215., 45.), (235., 45.), (225., 60.), (205., 45.), (245., 45.), (0., 75.), (0., 105.),
                      (180., 75.), (180., 105.), (90., 60.), (270., 60.), (135., 45.), (315., 45.), (265., 60.)):
        try:
            return recipes.acquire(env, output, index, yaw, tilt, 0., .04)
        except ValueError as error:
            errors.append(f"{yaw}/{tilt}: {error}")
    raise ValueError(f"No grasp: {errors}")


def stages(env, output, log=print, *, start_held=False):
    """Yields (name, parameters): rounds of grasp (skipped in round 0 when the wand is already held) and turn toward
    the finish heading, ending 4 mm above the finish pose; the recipe's final release once a turn ends within REACH_DEG
    of the target, its partial release otherwise; at most ROUNDS rounds."""
    # The recipe's turns end with the wand on the deck (final height 0 / 3 mm). MuJoCo's measured grasp holds the wand
    # ~2 mm lower in the hand than Genesis' did, so there the fingertips would reach 1.7 mm into the deck at height 0
    # (debug_turn.py: robot gap -, 0.33, 2.33 mm at 0, 2, 4 mm, every yaw); the turns end 4 mm above the finish pose
    # instead and the release lets the wand down onto the deck. The recipe's planning margin stays (1.4 mm).
    turn1 = dict(final_height=.004, minimum_clearance=.0014, speeds=(.08, .004, .12, .002), errors=(.0008, .001))
    for k in range(ROUNDS + 1):
        if k or not start_held:
            yield f"acquire_{k}", acquire_first(env, output, 10 * k + 1)
        params, info = turn(env, output, 10 * k + 1, **turn1)
        log(dict(turn=k, **info))
        yield f"turn_{k}", params
        if info["remaining_deg"] <= REACH_DEG or k == ROUNDS:
            yield f"final_release_{k}", recipes.final_release(env)
            m = env._task.result.metrics
            log(dict(after_release=k, release_angle=m.get("finish_angle_deg"),
                     release_dist_mm=1e3 * (m.get("distance_to_finish_m") or np.nan)))
            if env._task.result.task_success:
                return
        else:
            yield f"partial_release_{k}", recipes.script(recipes.PARTIAL_RELEASE)


def lower_and_release(env, output, *, log=print):
    """From a held wand with its traversal complete (the ring around the finish post): the simplified oracle's low
    release (SimplifiedBuzzOracle._lower: turn the held wand back about its handle, pitch the handle up, lower it onto the
    post above the finish pose), with its (back, tilt, height) taken from a sweep instead of the per-tier constants: the
    lowest height first, the first plan that passes the family's held-waypoint checks (BuzzIK, 1 mm clearance); then
    its release (hold, open, withdraw along the tool, hold, settle)."""
    import json
    from topogym.planning.buzz_wire_3d.kinematics import BuzzIK
    from topogym.planning.buzz_wire_3d.recipes import measured_grasp
    end_p, _ = endpoint_pose(env._task.route, 1)
    errors = []
    for height in (.006, .008, .010, .012, .015, .020):
        for back, tilt in ((5., 40.), (10., 25.), (0., 30.), (5., 30.), (10., 35.), (0., 40.), (0., 20.), (15., 30.)):
            kin = BuzzIK(env)
            p, r, op, ore = measured_grasp(env, kin)
            intermediate = r * Rotation.from_euler("x", -back, degrees=True)
            seated = intermediate * Rotation.from_euler("y", tilt, degrees=True)
            target = end_p + env._task.origin + [0., 0., height]
            try:
                # 0.5 mm: the held fingers come within ~0.65-0.85 mm of the deck at the low heights (medium, MuJoCo)
                frames = kin.plan_held_waypoints([(p, intermediate), (p, seated), (target, seated)], op, ore,
                                                 minimum_clearance=.0005, allow_wand_support=True)
                refs = kin.joint_references(frames, op, ore, translation_speed=.003, rotation_speed=.12,
                                            allow_wand_support=True)
            except ValueError as error:
                errors.append(f"{height}/{back}/{tilt}: {str(error)[:60]}")
                continue
            (output / "lower_release.json").write_text(json.dumps(dict(height=height, back=back, tilt=tilt,
                                                                       rejected=errors)))
            log(dict(lower_height_mm=1e3 * height, back_deg=back, tilt_deg=tilt, rejected=len(errors)))
            yield "lower", dict(part="joint_reference", plan=refs, progress_error_m=.0008, arrival_error_m=.0006)
            yield "release", recipes.script([
                dict(operation="hold", seconds=1., phase="settle_before_release"),
                dict(operation="grip", value=0., seconds=.5, phase="open_before_withdraw"),
                dict(operation="tcp", local_translation_m=[0., 0., -.1], seconds=1., phase="withdraw_after_release"),
                dict(operation="hold", seconds=2.), dict(operation="settle_gripper")])
            return
    raise ValueError(f"No low release: {errors[:6]} ...")


def level_release(env, output, *, log=print):
    """From a held wand with its traversal complete: turn it level (ring normal straight down, the recipe's half-turn
    convention), its heading kept, centred over the finish post at height h above the finish pose (the lowest h whose
    plan passes the family's held-waypoint checks, 0.5 mm), then release with the recipe's fast open-and-withdraw
    (RELEASE_ONTO_DECK). A level ring centred on the vertical post falls with its full radial clearance around the wire."""
    import json
    from topogym.planning.buzz_wire_3d.kinematics import BuzzIK
    from topogym.planning.buzz_wire_3d.recipes import measured_grasp
    end_p, end_r = endpoint_pose(env._task.route, 1)
    end_p = end_p + env._task.origin
    pitch = float(np.arcsin(end_r.apply([1, 0, 0])[2]))
    errors = []
    for h in (.006, .008, .010, .012, .015, .020, .025, .030, .040):
        for dyaw in (0., 10., -10., 20., -20.):
            kin = BuzzIK(env)
            p, r, op, ore = measured_grasp(env, kin)
            level = (Rotation.from_euler("z", heading(r) + dyaw, degrees=True) * Rotation.from_euler("y", -pitch)
                     * Rotation.from_euler("x", 180., degrees=True))
            try:
                frames = kin.plan_held_waypoints([(np.array([end_p[0], end_p[1], p[2]]), level), (end_p + [0., 0., h], level)],
                                                 op, ore, minimum_clearance=.0005, allow_wand_support=True)
                refs = kin.joint_references(frames, op, ore, translation_speed=.003, rotation_speed=.12,
                                            allow_wand_support=True)
            except ValueError as error:
                errors.append(f"{h}/{dyaw}: {str(error)[:60]}")
                continue
            (output / "level_release.json").write_text(json.dumps(dict(height=h, dyaw=dyaw, rejected=errors)))
            log(dict(level_height_mm=1e3 * h, dyaw_deg=dyaw, rejected=len(errors)))
            yield "level", dict(part="joint_reference", plan=refs, progress_error_m=.0008, arrival_error_m=.0006)
            yield "release", recipes.script(recipes.RELEASE_ONTO_DECK)
            return
    raise ValueError(f"No level release: {errors[:8]} ...")



def centre_release(env, output, *, log=print):
    """From a held wand with its traversal complete: move it horizontally until the ring centre is over the finish post
    (its orientation and height kept, or raised 2 / 4 / 8 mm), then the recipe's fast open-and-withdraw
    (RELEASE_ONTO_DECK): the falling ring keeps its largest clearance around the wire."""
    import json
    from topogym.planning.buzz_wire_3d.kinematics import BuzzIK
    from topogym.planning.buzz_wire_3d.recipes import measured_grasp
    end_p, _ = endpoint_pose(env._task.route, 1)
    end_p = end_p + env._task.origin
    errors = []
    for dz in (0., .002, .004, .008):
        kin = BuzzIK(env)
        p, r, op, ore = measured_grasp(env, kin)
        target = np.array([end_p[0], end_p[1], p[2] + dz])
        try:
            frames = kin.plan_held_waypoints([(target, r)], op, ore, minimum_clearance=.0005, allow_wand_support=True)
            refs = kin.joint_references(frames, op, ore, translation_speed=.003, rotation_speed=.12, allow_wand_support=True)
        except ValueError as error:
            errors.append(f"{dz}: {str(error)[:60]}")
            continue
        (output / "centre_release.json").write_text(json.dumps(dict(dz=dz, offset_mm=(1e3 * (p - end_p)).tolist(), rejected=errors)))
        log(dict(centre_from_mm=[round(float(v), 2) for v in 1e3 * (p[:2] - end_p[:2])], centre_dz_mm=1e3 * dz))
        yield "centre", dict(part="joint_reference", plan=refs, progress_error_m=.0008, arrival_error_m=.0006)
        yield "release", recipes.script(recipes.RELEASE_ONTO_DECK)
        return
    raise ValueError(f"No centred release: {errors}")


GRASPS = [("recipe", 225., 45.), ("acquire_2", None, None), ("g", 215., 45.), ("g", 235., 45.), ("g", 225., 60.),
          ("g", 205., 45.), ("g", 245., 45.), ("g", 0., 75.), ("g", 0., 105.), ("g", 180., 75.), ("g", 180., 105.),
          ("g", 90., 60.), ("g", 270., 60.), ("g", 135., 45.), ("g", 315., 45.), ("g", 265., 60.)]
WITHDRAW = [dict(operation="grip", value=0., seconds=.5, phase="open_after_failed_grasp"),
            dict(operation="tcp", local_translation_m=[0., 0., -.08], seconds=1., phase="withdraw_after_failed_grasp"),
            dict(operation="hold", seconds=1.), dict(operation="settle_gripper")]


def checkpoint(session):
    """A rewind point for development search only (the reference itself is replayed fresh by the grader): the env's
    complete checkpoint of the current tick plus the Session's own counters, recording and targets."""
    env = session.env
    env._checkpoint_current()
    return dict(ref=dict(env._info()["state_ref"]), n=session.n_steps, phys=session.physics_steps, t=session.sim_time_s,
                rec=list(session._recording), ticks=list(session._recorded_ticks), targets=session._targets.copy(),
                last=session.last_step_ticks)


def rewind(session, cp):
    session.env.privileged.rewind_to(cp["ref"])
    session.episode_done, session.episode_end = False, None
    session.n_steps, session.physics_steps, session.sim_time_s = cp["n"], cp["phys"], cp["t"]
    session._recording, session._recorded_ticks = list(cp["rec"]), list(cp["ticks"])
    session._targets, session.last_step_ticks = cp["targets"].copy(), cp["last"]


def plan_grasp(env, output, index, kind, yaw, tilt):
    if kind == "rel":                                   # yaw carries (rel_p, rel_r, q_seed), tilt a label
        if isinstance(yaw, dict):                       # a probe_grid2.py goal: acquire_q's parameters
            return acquire_q(env, output, index, **yaw)
        rel_p, rel_r, q_seed = yaw
        return acquire_rel(env, output, index, rel_p, rel_r, q_seed)
    if kind == "gridh":                                 # yaw carries (yaw, tilt, height), tilt a label
        y_, t_, h_ = yaw
        far_ = t_ >= 75.
        return acquire_general(env, output, index, y_, t_, height=h_, advance=.012 if far_ else 0.,
                               approach=.022 if far_ else .04, adaptive=far_)
    if kind == "acquire_2":
        if STAGED:
            a = recipes.ACQUIRE_2
            return acquire_general(env, output, index, a["yaw"], a["tilt"], advance=a["advance"], approach=a["approach"],
                                   adaptive=a["adaptive"])
        return recipes.acquire(env, output, **dict(recipes.ACQUIRE_2, index=index))
    if kind == "recipe":
        if STAGED:
            return acquire_general(env, output, index, yaw, tilt, advance=0., approach=.04)
        return recipes.acquire(env, output, index, yaw, tilt, 0., .04)
    far = tilt >= 75.
    return acquire_general(env, output, index, yaw, tilt, advance=.012 if far else 0., approach=.022 if far else .04,
                           adaptive=far)


def run_finish(env, session, runner, output, log, *, release=None, start_held=False, grasp_budget=8, extra_grasps=()):
    """The finish executed stage by stage with development backtracking (the grader's fresh replays re-check the
    whole reference): an optional release stage list, then rounds of
      grasp + turn search: grasp candidates (the recipe's two, then the reach-ranked grid) are executed one by one from
        the same rewind point; after each one that holds the wand, the turn is planned (yaw candidates from the target
        back toward the current heading, ending 4 mm above the finish pose) and its remaining angle noted; the first
        pair within REACH_DEG, else the best of `grasp_budget` grasps, is kept (re-executed from the rewind point);
      a failing turn execution is rewound and the next yaw tried;
      release: final within REACH_DEG, else partial, and another round.
    A grasp or turn whose execution fails (no bilateral grasp, the robot touching the wire, which the family's policy
    forbids, a stalled tracking) is rewound. Returns a failure text or None."""
    from topogym.planning.buzz_wire_3d.simplified_policy import make_stage_policy
    turn1 = dict(final_height=.004, minimum_clearance=.0014, speeds=(.08, .004, .12, .002), errors=(.0008, .001))

    def held():
        return bool(env._world.read_state().get("assist_active"))

    def execute(name, params):
        if isinstance(params, list):           # a staged grasp: its stages in order, the first failure ends it
            reason = "policy_done"
            for i, p in enumerate(params):
                reason = execute(f"{name}.{i}", p)
                if reason != "policy_done":
                    break
            return reason
        log(phase="stage_start", stage=name, part=params.get("part"))
        try:
            reason = runner.run_policy(request_id=name, policy=make_stage_policy(env, params),
                                       submission_hash=session.instance_sha, state_fields=("world",))["reason"]
        except RuntimeError as error:
            reason = f"error: {str(error)[:160]}"
        log(phase="stage_end", stage=name, reason=reason)
        return reason

    def done():
        return session.success() or session.episode_end is not None

    def plan_turn(k, yaws, goal, now):
        params = recipes.checked_turn(env, output, index=100 + k, yaws=yaws, offsets=OFFSETS, **turn1)
        chosen = json.loads((output / f"finish_turn_{100 + k}.json").read_text())["yaw"]
        return params, dict(goal_yaw=goal, start_yaw=now, chosen_yaw=chosen, remaining_deg=abs(wrap(chosen - goal)))

    if release is not None:
        for name, params in release(env, output, log=lambda d: log(phase="plan", **d)):
            if execute(name, params) != "policy_done":
                return f"release stage {name} failed"
    tries = 0
    for k in range(ROUNDS + 2):
        if done():
            return None
        turn_plan = None
        if k or not start_held or not held():
            order = ([("rel", g, label) for label, g in extra_grasps] + [("recipe", 225., 45.), ("acquire_2", None, None)]
                     + [("grid", y, t) for y, t in grasp_grid(env, top=40)])
            if _os.environ.get("BUZZ_GRID_LIST"):    # experiment: an explicit yaw:tilt:height list instead
                order = [("gridh", (float(y), float(t), float(h)), f"{y}:{t}:{h}")
                         for y, t, h in (x.split(":") for x in _os.environ["BUZZ_GRID_LIST"].split(","))]
            start = checkpoint(session)
            best, executed = None, 0
            for kind, yaw, tilt in order:
                if executed >= grasp_budget:
                    break
                tries += 1
                try:
                    params = plan_grasp(env, output, 100 + tries, kind, yaw, tilt)
                except ValueError as error:
                    log(phase="plan", grasp=[kind, tilt if kind in ("rel", "gridh") else yaw, tilt], rejected=str(error)[:80])
                    continue
                executed += 1
                if kind in ("rel", "gridh"):
                    yaw = "route" if kind == "rel" else "h"
                reason = execute(f"acquire_{k}_{kind}_{yaw}_{tilt}", params)
                if session.success():
                    return None
                if not (reason == "policy_done" and held() and session.episode_end is None):
                    log(phase="rewind", stage=f"acquire_{k}_{kind}_{yaw}_{tilt}", reason=reason)
                    rewind(session, start)
                    continue
                yaws, goal, now = yaw_candidates(env)
                try:
                    tplan = plan_turn(k, yaws, goal, now)
                except ValueError as error:
                    log(phase="plan", grasp=[kind, yaw, tilt], turn_rejected=str(error)[:80])
                    rewind(session, start)
                    continue
                log(phase="plan", grasp=[kind, yaw, tilt], turn_remaining_deg=tplan[1]["remaining_deg"])
                if best is None or tplan[1]["remaining_deg"] < best[0]:
                    best = (tplan[1]["remaining_deg"], params, kind, yaw, tilt)
                if tplan[1]["remaining_deg"] <= REACH_DEG:
                    turn_plan = tplan
                    break
                rewind(session, start)
            if best is None:
                return f"round {k}: no grasp with a turn"
            if turn_plan is None:              # re-execute the best grasp from the rewind point (deterministic)
                rewind(session, start)
                execute(f"acquire_{k}_best_{best[2]}_{best[3]}_{best[4]}", best[1])
                if not held():
                    return f"round {k}: the kept grasp did not repeat"
        yaws, goal, now = yaw_candidates(env)
        turned = None
        while yaws:
            try:
                params, info = turn_plan if turn_plan is not None else plan_turn(k, yaws, goal, now)
            except ValueError as error:
                log(phase="plan", turn=k, rejected=str(error)[:120])
                break
            turn_plan = None
            log(phase="plan", turn=k, **info)
            cp = checkpoint(session)
            reason = execute(f"turn_{k}_{round(info['chosen_yaw'], 1)}", params)
            if session.success():
                return None
            if reason == "policy_done" and session.episode_end is None:
                turned = info
                break
            log(phase="rewind", stage=f"turn_{k}", reason=reason)
            rewind(session, cp)
            yaws = [y for y in yaws if abs(wrap(y - goal)) > info["remaining_deg"] + 1e-6]
        if turned is None:
            execute(f"partial_release_{k}", partial_release())
            continue
        if turned["remaining_deg"] <= REACH_DEG or k >= ROUNDS:
            execute(f"final_release_{k}", slow_final_release(env) if STAGED else recipes.final_release(env))
            m = env._task.result.metrics
            log(phase="plan", after_release=k, release_angle=m.get("finish_angle_deg"),
                release_dist_mm=1e3 * (m.get("distance_to_finish_m") or np.nan))
            if done():
                return None
        else:
            execute(f"partial_release_{k}", partial_release())
    return None if done() else "rounds exhausted"


GOAL_SEEDS = []          # arm joints to seed the free-hand transfer's IK goal with (e.g. the route's own regrasps)


def seeded_goal(kin, target):
    """The first IK solution of `target` from GOAL_SEEDS that is collision-free (1 mm) with a 0.12 rad joint margin."""
    limits = np.asarray(kin.group.arm_limits_rad)
    for q0 in GOAL_SEEDS:
        seed = kin.seed.copy()
        seed[kin.arm_qindices] = q0
        try:
            q = np.asarray(kin.solve(target, seed), float)
        except ValueError:
            continue
        q[7:] = kin.seed[7:]
        arm = q[kin.arm_qindices]
        if np.minimum(arm - limits[:, 0], limits[:, 1] - arm).min() >= .12 and kin.clearance(q) >= .001:
            return q
    return None


def acquire_general(env, output, index, yaw, tilt, *, fraction=None, height=.014, advance=0., approach=.04,
                    adaptive=False):
    """recipes.acquire with its grasp point (fraction of the grip), height, advance and approach as parameters (the
    recipe: fraction at 48 mm from the ring centre, 14 mm, 0, 40 mm); the same free-hand planning and approach check."""
    from topogym.planning.buzz_wire_3d.kinematics import BuzzIK
    from topogym.planning.buzz_wire_3d.recipes import measured_grasp
    from topogym.utils.io import write_json_atomic
    layout = env._task.layout
    if fraction is None:
        fraction = (.048 - layout.loop_radius_m - layout.stem_length_m) / layout.grip_length_m
    kin = BuzzIK(env)
    p, r, _, _ = measured_grasp(env, kin)
    gr = r * Rotation.from_euler("z", yaw, degrees=True) * Rotation.from_euler("y", tilt, degrees=True) \
        * Rotation.from_euler("z", 180., degrees=True)
    point = layout.loop_radius_m + layout.stem_length_m + layout.grip_length_m * fraction
    gp = p + r.apply([point, 0., 0.]) + [0., 0., height] + gr.apply([0., 0., advance])
    pre = (gp - gr.apply([0., 0., approach]), gr)
    goal = seeded_goal(kin, pre) if GOAL_SEEDS else None
    free_clear = float(_os.environ.get("BUZZ_FREE_CLEAR", ".001"))   # experiment: the free-hand transfer's clearance
    if goal is not None and kin.clearance(goal) < free_clear:
        goal = None
    path = (kin.plan_free_tcp(pre, goal_seed=goal, minimum_clearance=free_clear) if goal is not None
            else kin.plan_free_tcp(pre, minimum_clearance=free_clear))
    seed = np.asarray(path[-1]["qpos"])
    if STAGED:                          # the pre-closed descent: the deck slab may be touched
        deck_tolerant(env, kin)
    for u in np.linspace(0., 1., max(2, int(np.ceil(approach / .0004)) + 1)):
        seed = kin.solve((gp - gr.apply([0., 0., approach * (1 - u)]), gr), seed)
        if kin.clearance(seed) < .001:
            raise ValueError("Selected free-hand path fails the final grasp approach")
    write_json_atomic(output / f"finish_transfer_{index}.json", dict(plan=path, yaw=yaw, tilt=tilt, fraction=fraction,
                                                                      height=height, advance=advance, approach=approach))
    params = dict(part="acquire", plan=path, grasp_y=tilt, grasp_yaw=yaw, grasp_roll=180., grasp_fraction=fraction,
                grasp_height=height, grasp_advance=advance, lift_height=.005, close_on_contact=adaptive,
                settle_timeout_s=45.)
    return staged(params, approach) if STAGED else params


def acquire_rel(env, output, index, rel_p, rel_r, q_seed, *, approach=.04, adaptive=True):
    """recipes.acquire at a given hand-on-wand pose (TCP in the wand frame: rel_p, rel_r, e.g. one of the released
    route's own regrasps), expressed in the acquire stage's parameters (ZYZ yaw / tilt / roll; grip fraction, world
    height and advance along the tool from the measured wand); the free-hand transfer's IK goal is seeded with the given
    arm joints (the route's own collision-free branch at that grasp) instead of random seeds."""
    from topogym.planning.buzz_wire_3d.kinematics import BuzzIK
    from topogym.planning.buzz_wire_3d.recipes import measured_grasp
    from topogym.utils.io import write_json_atomic
    layout = env._task.layout
    kin = BuzzIK(env)
    p, r, _, _ = measured_grasp(env, kin)
    rel_p, rel_r = np.asarray(rel_p, float), rel_r
    yaw, tilt, roll = rel_r.as_euler("ZYZ", degrees=True)
    up = r.inv().apply([0., 0., 1.])
    axis = rel_r.apply([0., 0., 1.])
    point, height, advance = np.linalg.solve(np.column_stack([[1., 0., 0.], up, axis]), rel_p)
    fraction = (point - layout.loop_radius_m - layout.stem_length_m) / layout.grip_length_m
    gr = r * rel_r
    gp = p + r.apply(rel_p)
    goal = kin.seed.copy()
    goal[kin.arm_qindices] = np.asarray(q_seed, float)
    pre = (gp - gr.apply([0., 0., approach]), gr)
    goal = np.asarray(kin.solve(pre, goal), float)
    goal[7:] = kin.seed[7:]
    gap = kin.clearance(goal)
    if gap < .001:
        raise ValueError(f"Route grasp: pre-grasp goal clearance {gap:.6g} m")
    path = kin.plan_free_tcp(pre, goal_seed=goal)
    seed = np.asarray(path[-1]["qpos"])
    if STAGED:                          # the pre-closed descent: the deck slab may be touched
        deck_tolerant(env, kin)
    for u in np.linspace(0., 1., max(2, int(np.ceil(approach / .0004)) + 1)):
        seed = kin.solve((gp - gr.apply([0., 0., approach * (1 - u)]), gr), seed)
        if kin.clearance(seed) < .001:
            raise ValueError("Route grasp fails the final approach")
    write_json_atomic(output / f"finish_transfer_{index}.json", dict(plan=path, yaw=yaw, tilt=tilt, roll=roll,
                      fraction=fraction, height=height, advance=advance, approach=approach, rel_p=rel_p.tolist()))
    params = dict(part="acquire", plan=path, grasp_y=float(tilt), grasp_yaw=float(yaw), grasp_roll=float(roll),
                grasp_fraction=float(fraction), grasp_height=float(height), grasp_advance=float(advance),
                lift_height=.005, close_on_contact=adaptive, settle_timeout_s=45.)
    return staged(params, approach) if STAGED else params


def acquire_q(env, output, index, yaw, tilt, roll, fraction, height, approach, q_goal, *, adaptive=True):
    """acquire_general with the hand's roll about the tool axis and the free-hand transfer's IK goal seeded with given
    arm joints (a collision-free pre-grasp goal found by probe_grid2.py on the same state)."""
    from topogym.planning.buzz_wire_3d.kinematics import BuzzIK
    from topogym.planning.buzz_wire_3d.recipes import measured_grasp
    from topogym.utils.io import write_json_atomic
    layout = env._task.layout
    kin = BuzzIK(env)
    p, r, _, _ = measured_grasp(env, kin)
    gr = r * Rotation.from_euler("z", yaw, degrees=True) * Rotation.from_euler("y", tilt, degrees=True) \
        * Rotation.from_euler("z", roll, degrees=True)
    point = layout.loop_radius_m + layout.stem_length_m + layout.grip_length_m * fraction
    gp = p + r.apply([point, 0., 0.]) + [0., 0., height]
    pre = (gp - gr.apply([0., 0., approach]), gr)
    goal = kin.seed.copy()
    goal[kin.arm_qindices] = np.asarray(q_goal, float)
    goal = np.asarray(kin.solve(pre, goal), float)
    goal[7:] = kin.seed[7:]
    gap = kin.clearance(goal)
    if gap < .001:
        raise ValueError(f"Grid grasp: pre-grasp goal clearance {gap:.6g} m")
    path = kin.plan_free_tcp(pre, goal_seed=goal)
    seed = np.asarray(path[-1]["qpos"])
    if STAGED:                          # the pre-closed descent: the deck slab may be touched
        deck_tolerant(env, kin)
    for u in np.linspace(0., 1., max(2, int(np.ceil(approach / .0004)) + 1)):
        seed = kin.solve((gp - gr.apply([0., 0., approach * (1 - u)]), gr), seed)
        if kin.clearance(seed) < .001:
            raise ValueError("Grid grasp fails the final approach")
    write_json_atomic(output / f"finish_transfer_{index}.json", dict(plan=path, yaw=yaw, tilt=tilt, roll=roll,
                      fraction=fraction, height=height, approach=approach))
    params = dict(part="acquire", plan=path, grasp_y=float(tilt), grasp_yaw=float(yaw), grasp_roll=float(roll),
                grasp_fraction=float(fraction), grasp_height=float(height), grasp_advance=0., lift_height=.005,
                close_on_contact=adaptive, settle_timeout_s=45.)
    return staged(params, approach) if STAGED else params


def grasp_grid(env, top=24):
    """Grasp candidates (yaw every 15 deg x tilt 45..90 about the handle) ranked for reach: the approach pointing away
    from the robot base, 15-65 deg below horizontal (the reachable grasps at the far finish posts; Genesis' own finish
    grasps of medium / hard approach 15-22 deg below horizontal, pointing outward)."""
    import mujoco
    w = env._world
    b = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    base = np.asarray(w.data.xpos[b])
    q = np.asarray(w.read_state()["bodies"]["wand"]["quaternion_wxyz"])
    r = Rotation.from_quat(q[[1, 2, 3, 0]])
    p = np.asarray(w.read_state()["bodies"]["wand"]["position_m"])
    out = p[:2] - base[:2]
    out /= np.linalg.norm(out)
    ranked = []
    for yaw in np.arange(0., 360., 15.):
        for tilt in (45., 60., 75., 90.):
            a = (r * Rotation.from_euler("z", yaw, degrees=True) * Rotation.from_euler("y", tilt, degrees=True)
                 * Rotation.from_euler("z", 180., degrees=True)).apply([0., 0., 1.])
            down = float(np.degrees(np.arcsin(np.clip(-a[2], -1, 1))))
            if not 10. <= down <= 70.:
                continue
            horizontal = a[:2] / max(np.linalg.norm(a[:2]), 1e-9)
            ranked.append((float(horizontal @ out) - .004 * abs(down - 30.), yaw, tilt))
    ranked.sort(reverse=True)
    return [(yaw, tilt) for _, yaw, tilt in ranked[:top]]


def held_turn_release(env, output, *, log=print):
    """From a held wand with its traversal complete (the ring around the finish post): turn it about the vertical
    through its ring centre toward the finish heading (the largest of 100 / 85 / 70 / 55 / 40 % of the way whose plan
    passes the family's held-waypoint checks, 0.5 mm), its ring centred over the post, then the recipe's fast
    open-and-withdraw (RELEASE_ONTO_DECK)."""
    from topogym.planning.buzz_wire_3d.kinematics import BuzzIK
    from topogym.planning.buzz_wire_3d.recipes import measured_grasp
    end_p, _ = endpoint_pose(env._task.route, 1)
    end_p = end_p + env._task.origin
    errors = []
    for frac in (1., .85, .7, .55, .4):
        kin = BuzzIK(env)
        p, r, op, ore = measured_grasp(env, kin)
        delta = wrap(target_yaw(env) - heading(r)) * frac
        rr = Rotation.from_euler("z", delta, degrees=True) * r
        target = np.array([end_p[0], end_p[1], p[2]])
        try:
            frames = kin.plan_held_waypoints([(target, rr)], op, ore, minimum_clearance=.0005, allow_wand_support=True)
            refs = kin.joint_references(frames, op, ore, translation_speed=.003, rotation_speed=.12, allow_wand_support=True)
        except ValueError as error:
            errors.append(f"{frac}: {str(error)[:60]}")
            continue
        (output / "held_turn.json").write_text(json.dumps(dict(fraction=frac, delta_deg=delta, rejected=errors)))
        log(dict(held_turn_deg=round(float(delta), 2), fraction=frac, rejected=len(errors)))
        yield "held_turn", dict(part="joint_reference", plan=refs, progress_error_m=.0008, arrival_error_m=.0006)
        yield "release", recipes.script(recipes.RELEASE_ONTO_DECK)
        return
    raise ValueError(f"No held turn: {errors}")
