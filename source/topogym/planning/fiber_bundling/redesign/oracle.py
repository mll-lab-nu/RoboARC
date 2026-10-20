"""The fiber_bundling redesign oracle (docs/simulator/redesign/fiber_bundling.md), robot level, in the production Session:
the fiber re-planner's stages (tools/references/fiber_bundling/replan.py: the released pickup, the
planner's comb loop, the gated closed-loop feed), with the redesign's own stages added:
- `--route-program`: after the pickup, the route's moves of the held sheath (lift, move, swing about a formboard
  peg, advance toward the ring), so that the cable behind the hand slides through the cradle and wraps the pegs
  as the tier's route requires (the hard tier's lay-over: lift, move across the far peg, lower).
- `--lay`: before the feed advances, roll the held sheath about the ring axis until the task's lay term (the red
  fiber's angle about the ring axis) meets its target.
- `--grasp-link trunk_NN`: the grasped link of a longer sheath (the planner's keyframes assume trunk_04).
- `--rolls`: the comb presses' roll choices (the planner's 0, -70, +70 deg; the arm cannot roll 70 deg at the turned
  rings).

Run it through run_registered.py, with the tooling (rcb_topogym) on PYTHONPATH.

usage: oracle.py <tier> <out_dir> --instance <instance.json> --released <released rows.npz> [--route-program P] [--lay]
       [--grasp-link trunk_07] [re-planner options]
writes <out_dir>/rows.npz (actions) and report.json"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
from rcb_topogym import Session
import topogym

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("tier"); ap.add_argument("out")
ap.add_argument("--instance", required=True, help="the tier's MuJoCo instance.json")
ap.add_argument("--released", required=True, help="released rows (an -i00 privileged.npz): the pickup")
ap.add_argument("--route-program", help="route moves after the pickup (JSON text or file; see route_stage)")
ap.add_argument("--route-speed", type=float, default=.008, help="largest speed of the grasped link's ends, m/s")
ap.add_argument("--lay", action="store_true", help="roll the held sheath to the task's lay target before the feed")
ap.add_argument("--probe-only", action="store_true", help="stop after the roll-range and feed probes")
ap.add_argument("--lay-at-mm", type=float, help="roll to the lay once the feed has carried the front this far past the "
                "ring plane (mm), instead of at the standoff: at hard's ring the roll at the standoff swings the wrist "
                "(link7) into the tall post's cap; further along the axis it clears")
ap.add_argument("--lay-short-deg", type=float, default=0., help="aim the lay roll this much short of the target "
                "(toward the unrolled angle), inside the tolerance: a smaller roll where the wrist's clearance is tight")
ap.add_argument("--lay-sweep-grid", help="the sweep's shifts (mm) and rolls (deg), e.g. '60,70,80/0,-30,-35,-40,20,40'")
ap.add_argument("--front-s-mm", type=float, help="carry the front this far past the ring plane (mm) instead of the "
                "selection's feed front_s")
ap.add_argument("--lay-sweep", action="store_true", help="kinematic only, at the lay roll: the robot's clearance to the "
                "fixtures for front positions along the ring axis x rolls, then stop")
ap.add_argument("--grasp-link", help="the grasped sheath link (default: the planner's trunk_04)")
ap.add_argument("--presses", default=".032,.030,.028,.026,.024")
ap.add_argument("--rolls", help="the comb's absolute rolls, preferred first (the planner's keyframes.roll_choices: 0,-70,70); "
                "at the turned rings the arm cannot roll the held sheath 70 deg, so e.g. 0,-45,45,-70,70")
ap.add_argument("--min-from-up", type=float, default=60., help="with --rolls: the planner's rule, a pressed strand at "
                "least this far from up after the roll (hard: the post caps the comb rolls at about -30 deg, so a strand "
                "near up needs a smaller angle; each press is judged in simulation)")
ap.add_argument("--speed-scale", type=float, default=1., help="scale the planner's motion speeds (stages.SPEEDS: 15 mm/s, "
                "0.25 rad/s joint, 0.35 rad/s rotation) for every planned motion: the comb strokes and the feed's "
                "alignment leg (the 72-min reference budget; the robot's limits are 1.7 m/s TCP, 2.175 rad/s)")
ap.add_argument("--comb-pause", default="1.0,0.5", help="the holds after a comb's first and second guide (s)")
ap.add_argument("--target", type=float, default=.0115, help="comb target (stages.COMB_TARGET_M)")
ap.add_argument("--margin", type=float, default=.0015, help="aim this much inside the target")
ap.add_argument("--guided", type=float, default=.016, help="the planner's guided target (stages.GUIDED_TARGET_M): when no "
                "comb gets every strand within --target but all are within this, feed anyway")
ap.add_argument("--from-rows", help="replay these rows (a previous run's pickup and combs), then verify and feed")
ap.add_argument("--combs-after-prefix", action="store_true", help="after --from-rows, continue the comb loop (a stopped "
                "run's prefix) instead of going straight to the feed")
ap.add_argument("--roll-ranks", default="0", help="candidate combs: the comb roll ranks tried with every press depth "
                "(0 = the preferred roll; e.g. 0,1)")
ap.add_argument("--adopt", action="store_true", help="forked candidate combs whose chosen child carries on as the run "
                "(no re-execution by the parent): candidate quality at about one press of wall time per comb")
ap.add_argument("--skip-strands", action="store_true", help="candidate combs: when no candidate press reduces the "
                "excess, give that strand up and comb the next (as --direct does), instead of ending the comb loop")
ap.add_argument("--direct", action="store_true", help="comb without candidates: each press is planned and executed in "
                "this process (the first --presses depth; a press that does not cut the summed excess deepens that "
                "strand's next press, a strand out of depths is skipped). Half the wall time of the forked candidates, "
                "whose chosen rows the parent re-executes; for the 16-strand tiers")
ap.add_argument("--prefix-rows", type=int, help="only the first N rows of --from-rows")
ap.add_argument("--feed-speed", type=float, default=.01, help="closed-loop feed speed along the ring axis, m/s")
ap.add_argument("--feed-gain", type=float, default=.08, help="lateral correction per control step (integral gain)")
ap.add_argument("--feed-gate", type=float, default=.001, help="advance only while the front is this close to the axis")
ap.add_argument("--near-mm", type=float, default=0., help="slow zone around the ring: from this far before the ring plane "
                "(mm, measured front point; 0 = none) ...")
ap.add_argument("--near-past-mm", type=float, default=10., help="... until the front is this far past it (mm)")
ap.add_argument("--near-speed", type=float, help="feed speed in the slow zone, m/s")
ap.add_argument("--near-gate", type=float, help="lateral gate in the slow zone, m")
ap.add_argument("--force-hold", type=float, help="no advance while the grasp weld's force in the last control step exceeded "
                "this (N, the largest over its MuJoCo steps)")
ap.add_argument("--force-back", type=float, help="above this (N), back the commanded front off by --back-mm per control step")
ap.add_argument("--back-mm", type=float, default=.25)
ap.add_argument("--lag-max", type=float, help="no advance while the commanded front leads the measured one by more than this (m)")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
base = f"fiber-bundling-{a.tier.replace('_', '-')}-i00"
inst = str(Path(a.instance).resolve())
released = Path(a.released)
sel = json.loads((Path(topogym.__file__).resolve().parent / "planning/harbor_bridge/reference_data/fiber_bundling" / a.tier
                  / "selection.json").read_text())
s = Session("fiber_bundling", instance=inst, strict=True)
env = s.env
w = env._world
from topogym.planning.fiber_bundling.stages import plan_comb, plan_feed, COMBS, MAXIMUM_BEND_DEG, connect, measured_offset
import topogym.planning.fiber_bundling.keyframes as _kf
import topogym.planning.fiber_bundling.stages as _st
if a.speed_scale != 1.:                           # faster planned motions (stages.connect reads SPEEDS per call)
    _st.SPEEDS = {k: v * a.speed_scale for k, v in _st.SPEEDS.items()}
PAUSE1, PAUSE2 = (float(x) for x in a.comb_pause.split(","))
if a.grasp_link:                                  # a longer sheath: the planner's keyframes name trunk_04
    _kf.GRASP_LINK = _st.GRASP_LINK = a.grasp_link
if a.rolls:                                       # the comb's roll choices (stages.plan_comb calls roll_choices)
    _ROLLS, _roll_choices = tuple(float(x) for x in a.rolls.split(",")), _st.roll_choices
    _st.roll_choices = lambda fan_deg, current_roll_deg=0., rolls=_ROLLS, minimum_from_up_deg=a.min_from_up: _roll_choices(
        fan_deg, current_roll_deg, rolls=rolls, minimum_from_up_deg=minimum_from_up_deg)
from topogym.planning.fiber_bundling.keyframes import ring_frame, link_pose, GRASP_LINK, rotation
from scipy.spatial.transform import Rotation
from topogym.planning.fiber_bundling.kinematics import FiberIK
from topogym.planning.fiber_bundling.keyframes import strand_measures

STRICT_SOLVE = FiberIK.solve
IK_RELAXED = [0]                                     # planner IK solves that met only the relaxed tolerance


def _solve_or_relaxed(kin, tcp, seed):
    """The planner's IK (0.1 mm, 1e-3 rad); where it fails, the same solver to 1 mm / 1e-2 rad, counted. After the
    route program some comb presses at the turned rings miss the strict tolerance by 0.15-0.5 mm (the joints far from
    their limits): a solver residual, not a reach limit. A press is a few millimetres deep and the comb measures each
    press's result; the feed is closed loop. The roll-range probe stays strict."""
    try:
        return STRICT_SOLVE(kin, tcp, seed)
    except ValueError:
        IK_RELAXED[0] += 1
        p, r = tcp
        return kin.ik.solve_checked(link_name=kin.group.hand_link,
                                    position_world_m=p - r.apply(kin.group.tcp_position_in_hand_m),
                                    orientation_wxyz=r.as_quat()[[3, 0, 1, 2]], joint_dof_indices=kin.indices,
                                    initial_qpos=seed, position_tolerance_m=1e-3, orientation_tolerance_rad=1e-2)


FiberIK.solve = _solve_or_relaxed

layout = env._task.layout
DT = float(w.physics_dt_s) * 10                      # one control step
old = np.asarray(np.load(released)["actions"], np.float32)
rows, report = [], dict(argv=sys.argv[1:], tier=a.tier, instance=inst, topogym=topogym.__file__, released=str(released), stages=[])


def _save_on_exit():
    """Keep the executed rows of a run that stops on an exception (a prefix for the next run)."""
    if not (out / "rows.npz").exists() and rows:
        np.savez(out / "rows_partial.npz", actions=np.asarray(rows, np.float32))
        (out / "report_partial.json").write_text(json.dumps(report, indent=1, default=str))


import atexit
import signal
atexit.register(_save_on_exit)
# SIGTERM ends the run through atexit (its rows kept as a prefix). SIGINT cannot: a detached run inherits it ignored.
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
t_start = time.time()


WELD = [0., False]                                   # [largest weld force in this control step (N), tracking on]


def _install_weld_probe():
    import mujoco as mj
    m, d = w.model, w.data
    welds = [e for e in range(m.neq) if int(m.eq_type[e]) == int(mj.mjtEq.mjEQ_WELD)]
    EQ = int(mj.mjtConstraint.mjCNSTR_EQUALITY)
    orig = mj.mj_step

    def hook(model, data):
        orig(model, data)
        if not WELD[1] or model is not m:
            return
        nf = int(d.nefc)
        if not nf:
            return
        typ, eid, f = np.asarray(d.efc_type[:nf]), np.asarray(d.efc_id[:nf]), np.asarray(d.efc_force[:nf])
        for e in welds:
            if d.eq_active[e]:
                r_ = np.flatnonzero((typ == EQ) & (eid == e))
                if len(r_) >= 3:
                    WELD[0] = max(WELD[0], float(np.linalg.norm(f[r_[:3]])))
    mj.mj_step = hook


_install_weld_probe()


def step(row):
    rows.append(np.asarray(row, np.float32))
    return s.step(rows[-1])


def grasp_phase():
    return w._grasp_manager.hands["arm"]["phase"] if w._grasp_manager is not None else None


def measures():
    return strand_measures(w.read_state(), layout)


def refs_to_rows(refs, grip, pause_s):
    t = np.array([r["time_s"] for r in refs]); q = np.array([r["qpos"][:7] for r in refs])
    n = int(np.ceil(t[-1] / DT))
    out_rows = [np.r_[[np.interp(k * DT, t, q[:, i]) for i in range(7)], grip] for k in range(1, n + 1)]
    out_rows += [out_rows[-1]] * int(round(pause_s / DT))
    return out_rows


def plan_feed_bent(kin, state, front_s, max_offset, standoff=.02, feed_leg=True):
    """plan_feed with the sheath's front link as it is. The planner treats the grasped link and the front link as
    collinear (keyframes.FRONT_M); a wire sheath keeps the bend the combs gave it at that joint (10+ deg), so the
    planner's straight path puts the real front 8 mm off the ring axis, onto the ring. Here the front link (the one
    that carries the strands) is aligned with the ring axis by the smallest rotation (its roll kept) at a standoff
    before the ring, then fed along the axis to front_s past the ring plane, the grasped link following rigidly. The
    real front point's offset from the axis is checked on the feed segment."""
    center, axis = ring_frame(state)
    front_link = f"trunk_{layout.trunk_segments - 1:02d}"
    L = layout.trunk_length_m / layout.trunk_segments
    p4, r4 = link_pose(state, GRASP_LINK)
    p5, r5 = link_pose(state, front_link)
    r_rel, p_rel = r4.inv() * r5, r4.inv().apply(p5 - p4)
    x5 = r5.apply([1., 0., 0.])
    v = np.cross(x5, axis); sv = float(np.linalg.norm(v))
    align = Rotation.from_rotvec(v / sv * np.arctan2(sv, float(x5 @ axis))) if sv > 1e-12 else Rotation.identity()
    r5t = align * r5
    front_now = float((p5 + r5.apply([L, 0., 0.]) - center) @ axis)

    def grasped(front_axial):
        p5t = center + front_axial * axis - L * r5t.apply([1., 0., 0.])
        r4t = r5t * r_rel.inv()
        return p5t - r4t.apply(p_rel), r4t

    pose = lambda label, p, r: dict(label=label, position_m=list(map(float, p)),
                                    quaternion_wxyz=list(map(float, r.as_quat()[[3, 0, 1, 2]])))
    pa, ra = grasped(min(front_now, -standoff))
    pe, re = grasped(front_s)
    offset_p, offset_r = measured_offset(kin, state)
    seed = kin.seed.copy()
    first = connect(kin, [pose("measured_start", p4, r4), pose("aligned_standoff", pa, ra)], offset_p, offset_r)
    if not feed_leg:
        # The closed loop feeds by itself; the planned feed leg's 1 mm clearance check cannot pass a ring closer than
        # 1 mm around the sheath (the redesign's 0.8 / 0.6 mm), so it is not planned.
        return dict(plans=[first], front_axial_start_m=front_now)
    kin.seed = np.asarray(first["refs"][-1]["qpos"])
    try:
        second = connect(kin, [pose("aligned_standoff", first["refs"][-1]["position_m"], rotation(first["refs"][-1]["quaternion_wxyz"])),
                               pose("feed_end", pe, re)], offset_p, offset_r)
    finally:
        kin.seed = seed
    offs = []
    for row in second["refs"]:
        f = np.asarray(row["position_m"]) + rotation(row["quaternion_wxyz"]).apply(p_rel + r_rel.apply([L, 0., 0.])) - center
        offs.append(float(np.linalg.norm(f - (f @ axis) * axis)))
    if max(offs) > max_offset:
        raise ValueError(f"Planned front point leaves the ring axis by {max(offs):.6g} m")
    bend = float(np.degrees(r_rel.magnitude()))
    return dict(plans=[first, second], front_offsets_m=offs, front_link_bend_deg=bend, front_axial_start_m=front_now)


def feed_closed_loop(front_s, hold_s, standoff=.02):
    """The feed, executed closed loop as the planner's own policy does (it waits on a pose gate), and correcting:
    first the planned, collision-checked alignment of the front link (the one carrying the strands) with the ring
    axis at a standoff (plan_feed_bent's first leg); then every control step the measured front point's lateral
    offset from the axis is integrated into a correction of the commanded pose (the planner's IK), and the front
    advances along the axis only while that offset is within the gate. A wire strand whose tip meets the ring pushes
    back with up to its buckling load; open-loop rows let the held sheath drift off the 1.5 mm clearance."""
    state = w.read_state()
    kin = FiberIK(env, link=GRASP_LINK)
    plan = plan_feed_bent(kin, state, front_s, 1., feed_leg=False)  # the alignment leg only
    first = plan["plans"][0]
    for r in refs_to_rows(first["refs"], grip, .5):
        if step(r):
            return "ended during the alignment"
    if a.lay and a.lay_at_mm is None and lay_roll(kin):
        return "ended during the lay roll"
    center, axis = ring_frame(w.read_state())
    front_link = f"trunk_{layout.trunk_segments - 1:02d}"
    L = layout.trunk_length_m / layout.trunk_segments

    def reference():
        """The feed's frame from the current state: the held link relative to the front link, the grasp offset, the
        front link's orientation turned onto the axis, the front's position along it."""
        st = w.read_state()
        p4, r4 = link_pose(st, GRASP_LINK); p5, r5 = link_pose(st, front_link)
        r_rel, p_rel = r4.inv() * r5, r4.inv().apply(p5 - p4)
        offset_p, offset_r = measured_offset(kin, st)
        x5 = r5.apply([1., 0., 0.]); v = np.cross(x5, axis); sv = float(np.linalg.norm(v))
        r5t = (Rotation.from_rotvec(v / sv * np.arctan2(sv, float(x5 @ axis))) if sv > 1e-12 else Rotation.identity()) * r5
        return r_rel, p_rel, offset_p, offset_r, r5t, float((p5 + r5.apply([L, 0., 0.]) - center) @ axis)

    r_rel, p_rel, offset_p, offset_r, r5t, s_des = reference()
    corr, rot_corr, q, held, log = np.zeros(3), np.zeros(3), np.asarray(first["refs"][-1]["qpos"], float), 0, []
    lay_pending = bool(a.lay and a.lay_at_mm is not None)
    gates = dict(advance=0, lateral=0, lag=0, force_hold=0, force_back=0)
    fmax = 0.
    WELD[0], WELD[1] = 0., True
    for k in range(int(round(90. / DT))):
        st = w.read_state()
        p5m, r5m = link_pose(st, front_link)
        f = p5m + r5m.apply([L, 0., 0.]) - center
        lat = f - (f @ axis) * axis
        tilt = np.cross(r5m.apply([1., 0., 0.]), axis)              # the front link's tilt off the axis (rad, about)
        corr = np.clip(corr + a.feed_gain * lat, -.015, .015)
        rot_corr = np.clip(rot_corr + a.feed_gain * tilt, -.1, .1)
        err = float(np.linalg.norm(lat))
        F, WELD[0] = WELD[0], 0.                                     # the weld's largest force in the last control step
        fmax = max(fmax, F)
        s_meas = float(f @ axis)
        if lay_pending and s_meas >= 1e-3 * a.lay_at_mm:   # the late lay roll, then the feed's frame afresh
            lay_pending = False
            report["lay_at_front_mm"] = round(1e3 * s_meas, 2)
            if lay_roll(kin):
                report["feed_log"] = log
                return "ended during the lay roll"
            r_rel, p_rel, offset_p, offset_r, r5t, s_des = reference()
            corr, rot_corr, q = np.zeros(3), np.zeros(3), _q_now()
            continue
        near = a.near_mm > 0 and -1e-3 * a.near_mm <= s_meas <= 1e-3 * a.near_past_mm
        speed = a.near_speed if near and a.near_speed else a.feed_speed
        gate = a.near_gate if near and a.near_gate else a.feed_gate
        if a.force_back is not None and F > a.force_back:          # pushing: back off, keep re-aligning
            s_des -= 1e-3 * a.back_mm
            gates["force_back"] += 1
        elif s_des >= front_s:
            pass
        elif err >= gate:
            gates["lateral"] += 1
        elif a.lag_max is not None and s_des - s_meas > a.lag_max:
            gates["lag"] += 1
        elif a.force_hold is not None and F > a.force_hold:
            gates["force_hold"] += 1
        else:
            s_des = min(front_s, s_des + speed * DT)
            gates["advance"] += 1
        rt = Rotation.from_rotvec(rot_corr) * r5t
        p5t = center + s_des * axis - corr - L * rt.apply([1., 0., 0.])
        r4t = rt * r_rel.inv()
        p4t = p5t - r4t.apply(p_rel)
        try:
            q = np.asarray(solve_relaxed(kin, (p4t + r4t.apply(offset_p), r4t * offset_r), q), float)
        except Exception as e:
            report["feed_log"] = log
            return f"IK failed at step {k}: {repr(e)[:120]}"
        if k % 20 == 0 or (a.force_hold is not None and F > a.force_hold):
            log.append(dict(step=k, s_mm=round(1e3 * s_des, 1), front_mm=round(1e3 * s_meas, 2), lat_mm=round(1e3 * err, 2),
                            corr_mm=round(1e3 * float(np.linalg.norm(corr)), 2),
                            tilt_deg=round(float(np.degrees(np.linalg.norm(tilt))), 2), weld_N=round(F, 1), near=near))
        if step(np.r_[q[:7], grip]):
            report["feed_log"] = log
            report["feed_gates"] = dict(gates, weld_max_N=round(max(fmax, WELD[0]), 1))
            WELD[1] = False
            return "ended"
        if s_des >= front_s:
            held += 1
            if held * DT >= hold_s:
                break
    report["feed_log"] = log
    report["feed_gates"] = dict(gates, weld_max_N=round(fmax, 1))
    WELD[1] = False
    return None


def rolled(p4, r4, c, axis, deg):
    """The held link's pose turned by deg about the ring axis through c."""
    R = Rotation.from_rotvec(np.radians(deg) * axis)
    return c + R.apply(p4 - c), R * r4


def lay_roll(kin, step_deg=1.5, passes=2):
    """Roll the held sheath about the ring axis through the front (the alignment leg put the front link on the axis)
    until the task's lay term is met: the red fiber's root angle about the ring axis from the ring's top. The bundle
    turns about its own front: a held link bent away from the front link (hard's S-route: ~0.1 m off the axis) swings
    around the axis, the front stays on it. Each control step turns at most step_deg. Returns True if the episode
    ended."""
    task = env._task
    for _ in range(passes):
        st = w.read_state()
        center, axis = ring_frame(st)
        term = task.lay_term(st, center, axis)
        need = -float(term["error_deg"])
        if abs(need) > a.lay_short_deg:
            need -= np.sign(need) * a.lay_short_deg
        report.setdefault("lay_roll", []).append(dict(angle_deg=term["angle_deg"], roll_deg=round(need, 2)))
        if abs(need) <= .25 * float(task.config["redesign"]["lay"]["tolerance_deg"]):
            break
        p4, r4 = link_pose(st, GRASP_LINK)
        f = task._front_point(st)
        c = center + float((f - center) @ axis) * axis   # the front's foot on the ring axis: the roll line
        off = (p4 - c) - float((p4 - c) @ axis) * axis
        report["lay_roll"][-1]["held_link_off_axis_mm"] = round(1e3 * float(np.linalg.norm(off)), 1)
        q = kin.seed = _q_now()                      # measured_offset reads the TCP at kin.seed: make it now
        offset_p, offset_r = measured_offset(kin, st)
        if a.lay_sweep:
            grid = {}
            if a.lay_sweep_grid:
                sh_, ro_ = a.lay_sweep_grid.split("/")
                grid = dict(shifts_mm=tuple(float(x) for x in sh_.split(",")), rolls=tuple(float(x) for x in ro_.split(",")))
            lay_sweep(kin, p4, r4, offset_p, offset_r, c, axis, q, **grid)
            (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
            os._exit(0)
        if "roll_range_deg" not in report:
            report["roll_range_deg"] = roll_range(kin, p4, r4, offset_p, offset_r, axis, q, c=c)
            print(json.dumps(dict(stage="roll_range", range_deg=report["roll_range_deg"])), flush=True)
            report["feed_probe"] = feed_probe(kin, p4, r4, offset_p, offset_r, center, axis, q, c=c)
            print(json.dumps(dict(stage="feed_probe", reached_mm=report["feed_probe"])), flush=True)
            if a.probe_only:
                (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
                os._exit(0)
        n = max(1, int(np.ceil(abs(need) / step_deg)))
        for k in range(1, n + 1):
            pk, rk = rolled(p4, r4, c, axis, need * k / n)
            q = np.asarray(solve_relaxed(kin, (pk + rk.apply(offset_p), rk * offset_r), q), float)
            if step(np.r_[q[:7], grip]):
                return True
        for _ in range(int(round(.5 / DT))):          # settle
            if step(rows[-1]):
                return True
    kin.seed = _q_now()                              # the feed's measured_offset must see the rolled hand
    st = w.read_state()
    center, axis = ring_frame(st)
    report["lay_after_roll"] = task.lay_term(st, center, axis)
    print(json.dumps(dict(stage="lay_roll", **report["lay_after_roll"])), flush=True)
    return False


def lay_sweep(kin, p4, r4, offset_p, offset_r, c, axis, q0, shifts_mm=(0, 10, 20, 30, 40, 45, 50, 55),
              rolls=(0, -10, -20, -25, -30, -35, -40, -45, -50), step_deg=2.,
              post_variants=(("as built", 0., True, 0.), ("post -30 mm", .03, True, 0.), ("post -40 mm", .04, True, 0.),
                             ("post -35, peg 1 -10 mm", .035, True, .01), ("post -40, peg 1 -10 mm", .04, True, .01))):
    """Kinematic only: for the held link moved along the ring axis by each shift and then rolled about it, the robot's
    smallest distance to the fixtures (every geom outside the robot and the cable, the table under the robot's base
    excepted), by IK continuation from the current joints, for the tall post as built and with it changed (its cap
    removed, or lowered). The wrist (link7) is rigid with the hand, so its place follows from the roll; the arm's
    redundancy cannot move it."""
    import mujoco as mj
    m, d = w.model, w.data
    bname = lambda b: mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(b)) or ""
    root = next(b for b in range(m.nbody) if bname(b) == "robot")

    def in_robot(b):
        while b > 0:
            if b == root:
                return True
            b = int(m.body_parentid[b])
        return False
    coll = lambda g: int(m.geom_contype[g]) or int(m.geom_conaffinity[g])
    rob = [g for g in range(m.ngeom) if coll(g) and in_robot(int(m.geom_bodyid[g])) and bname(int(m.geom_bodyid[g])) != "link0"]
    fix = [g for g in range(m.ngeom) if coll(g) and not in_robot(int(m.geom_bodyid[g]))
           and not bname(int(m.geom_bodyid[g])).startswith("fiber_bundle")]
    post = [g for g in fix if bname(int(m.geom_bodyid[g])).startswith("formboard_peg_0")]
    post_cap = [g for g in post if bname(int(m.geom_bodyid[g])).endswith("_cap")]
    peg1 = [g for g in fix if bname(int(m.geom_bodyid[g])).startswith("formboard_peg_1")]
    arm_q = [int(m.jnt_qposadr[m.actuator_trnid[i][0]]) for i in range(7)]
    dd, fromto, table = mj.MjData(m), np.zeros(6), {}

    def clearance(q, dz, cap, dz1=0.):
        dd.qpos[:] = d.qpos
        dd.qpos[arm_q] = q[:7]
        mj.mj_kinematics(m, dd)
        best = (1e9, "", "")
        for h in fix:
            if h in post and not cap and h in post_cap:
                continue
            sh = np.array([0., 0., -dz]) if h in post else (np.array([0., 0., -dz1]) if h in peg1 and dz1 else None)
            if sh is not None:
                dd.geom_xpos[h] += sh
            for g in rob:
                dist = mj.mj_geomDistance(m, dd, g, h, .08, fromto)
                if dist < best[0]:
                    best = (dist, bname(int(m.geom_bodyid[g])), bname(int(m.geom_bodyid[h])))
            if sh is not None:
                dd.geom_xpos[h] -= sh
        return [round(1e3 * best[0], 1), best[1], best[2]]

    dd.qpos[:] = d.qpos; mj.mj_kinematics(m, dd)
    where = {bname(b): [round(float(x), 3) for x in dd.xpos[b]] for b in range(m.nbody)
             if bname(b) in ("link7", "hand", "r85_left_spring_link", "r85_right_spring_link", "formboard_peg_0",
                             "formboard_peg_0_cap", "formboard_peg_1", "formboard_peg_1_cap", "bundling_ring")}
    print(json.dumps(dict(stage="lay_sweep_where", bodies=where, axis=[round(float(x), 3) for x in axis],
                          roll_line_point=[round(float(x), 3) for x in c])), flush=True)
    for sh in shifts_mm:
        q = np.asarray(q0, float)
        try:
            for k in range(1, 11):
                pk = p4 + 1e-3 * sh * k / 10 * axis
                q = np.asarray(solve_relaxed(kin, (pk + r4.apply(offset_p), r4 * offset_r), q), float)
        except ValueError:
            table[str(sh)] = "IK fails on the shift"
            continue
        row, cur, q_shift = {}, 0., q.copy()
        for roll in rolls:
            try:
                while abs(cur - roll) > 1e-9:
                    cur = max(roll, cur - step_deg) if roll < cur else min(roll, cur + step_deg)
                    pk, rk = rolled(p4 + 1e-3 * sh * axis, r4, c + 1e-3 * sh * axis, axis, cur)
                    q = np.asarray(solve_relaxed(kin, (pk + rk.apply(offset_p), rk * offset_r), q), float)
            except ValueError:
                row[str(roll)] = "IK fails"
                cur, q = 0., q_shift.copy()                  # the next roll starts again from no roll
                continue
            row[str(roll)] = {name: clearance(q, dz, cap, dz1) for name, dz, cap, dz1 in post_variants}
        table[str(sh)] = row
        print(json.dumps(dict(stage="lay_sweep", shift_mm=sh, clearance_mm=row)), flush=True)
    report["lay_sweep"] = table


def solve_relaxed(kin, tcp, seed, position_m=5e-4, orientation_rad=5e-3):
    """The planner's IK (0.1 mm, 1e-3 rad); where it fails, the same solver to 0.5 mm / 5e-3 rad. The closed-loop
    stages measure and correct what they command, so a sub-millimetre IK residual near the reach limit is harmless."""
    try:
        return STRICT_SOLVE(kin, tcp, seed)
    except ValueError:
        p, r = tcp
        return kin.ik.solve_checked(link_name=kin.group.hand_link,
                                    position_world_m=p - r.apply(kin.group.tcp_position_in_hand_m),
                                    orientation_wxyz=r.as_quat()[[3, 0, 1, 2]], joint_dof_indices=kin.indices,
                                    initial_qpos=seed, position_tolerance_m=position_m,
                                    orientation_tolerance_rad=orientation_rad)


def roll_range(kin, p4, r4, offset_p, offset_r, axis, q0, step_deg=2., limit_deg=120., c=None):
    """Kinematic only: how far the held link can roll about the ring axis from here, each way, by IK continuation
    (the lay target is chosen inside this range)."""
    out = []
    for sign in (-1., 1.):
        q, reached = np.asarray(q0, float), 0.
        for k in range(1, int(limit_deg / step_deg) + 1):
            pk, r = rolled(p4, r4, p4 if c is None else c, axis, sign * step_deg * k)
            try:
                q = np.asarray(STRICT_SOLVE(kin, (pk + r.apply(offset_p), r * offset_r), q), float)
            except ValueError:
                break
            reached = sign * step_deg * k
        out.append(reached)
    return out


def feed_probe(kin, p4, r4, offset_p, offset_r, center, axis, q0, rolls=range(-60, 61, 15), step_m=.004, front_s=.03,
               c=None):
    """Kinematic only: for each roll of the held link about the ring axis, how far along the axis the feed can carry the
    front (mm past the ring plane, relaxed IK) before the arm cannot follow: the lay target is picked among the rolls
    whose feed reaches the end."""
    L = layout.trunk_length_m / layout.trunk_segments
    out = {}
    for roll in rolls:
        q = np.asarray(q0, float)
        pr, r = rolled(p4, r4, p4 if c is None else c, axis, roll)
        reached = None
        try:
            for k in range(1, 61):                       # roll in 60 small steps, then feed
                pk, rk = rolled(p4, r4, p4 if c is None else c, axis, roll * k / 60)
                q = np.asarray(solve_relaxed(kin, (pk + rk.apply(offset_p), rk * offset_r), q), float)
            front = (float((p4 + r.apply([2 * L, 0., 0.]) - center) @ axis) if c is None
                     else float((c - center) @ axis))
            s_ = front
            while s_ < front_s:
                s_ = min(front_s, s_ + step_m)
                pk = pr + (s_ - front) * axis
                q = np.asarray(solve_relaxed(kin, (pk + r.apply(offset_p), r * offset_r), q), float)
                reached = round(1e3 * s_, 1)
        except ValueError:
            pass
        out[str(roll)] = reached
    return out


def _q_now():
    """The robot's measured joints (the IK seed for a stage that starts from the current pose)."""
    return np.asarray(env._robot.entity.get_qpos(), float).reshape(-1).copy()


def route_stage():
    """The route program (--route-program, JSON): moves of the held sheath link, executed in order with IK, each at
    most --route-speed (both ends of the grasped link):
      {"lift": dz}                      raise (or lower) the link by dz (m)
      {"move": [dx, dy]}                translate it in the horizontal plane (m)
      {"swing": {"peg": name, "deg": a}} turn it about the vertical axis through a formboard peg (+: left turn); the
                                         cable behind it wraps the peg ({"to": "ring"} instead of "deg": until the link
                                         points along the ring axis, top view)
      {"advance": standoff}             carry the front point straight to the ring axis, standoff (m) before the ring
                                         plane (the feed's alignment leg then turns it onto the axis)
    then let the cable settle for 1 s and record the task's route terms."""
    prog = json.loads(Path(a.route_program).read_text()) if Path(a.route_program).exists() else json.loads(a.route_program)
    kin = FiberIK(env, link=GRASP_LINK)
    st = w.read_state()
    p, r = link_pose(st, GRASP_LINK)
    offset_p, offset_r = measured_offset(kin, st)
    origin = np.asarray(env._task.origin, float)
    pegs = {pg["name"]: origin + np.r_[np.asarray(pg["xy_m"], float), 0.] for pg in env._task.config["redesign"]["pegs"]}
    L = layout.trunk_length_m / layout.trunk_segments
    q = _q_now()
    log = []
    for mv in prog:
        if "lift" in mv:
            path = lambda u, p0=p, r0=r: (p0 + np.array([0., 0., mv["lift"] * u]), r0)
        elif "move" in mv:
            dxy = np.r_[np.asarray(mv["move"], float), 0.]
            path = lambda u, p0=p, r0=r: (p0 + dxy * u, r0)
        elif "swing" in mv:
            c = pegs[mv["swing"]["peg"]]
            if mv["swing"].get("to") == "ring":       # turn until the link points along the ring axis (top view)
                _, axis = ring_frame(w.read_state())
                x = r.apply([1., 0., 0.])
                deg = float(np.degrees(np.arctan2(axis[1], axis[0]) - np.arctan2(x[1], x[0])))
                deg = (deg + 180.) % 360. - 180.
            else:
                deg = float(mv["swing"]["deg"])
            def path(u, p0=p, r0=r, c=c, deg=deg):
                R = Rotation.from_euler("z", deg * u, degrees=True)
                pp = c + R.apply(p0 - c)
                return np.r_[pp[:2], p0[2]], R * r0
        elif "advance" in mv:
            center, axis = ring_frame(w.read_state())
            goal = center - float(mv["advance"]) * axis
            front = p + r.apply([2 * L, 0., 0.])
            path = lambda u, p0=p, r0=r, d=goal - front: (p0 + d * u, r0)
        else:
            raise ValueError(f"unknown route move {mv}")
        pe, re = path(1.)
        span = max(np.linalg.norm(pe - p), np.linalg.norm(pe + re.apply([L, 0., 0.]) - p - r.apply([L, 0., 0.])))
        n = max(1, int(np.ceil(span / (a.route_speed * DT))))
        for k in range(1, n + 1):
            pk, rk = path(k / n)
            q = np.asarray(solve_relaxed(kin, (pk + rk.apply(offset_p), rk * offset_r), q), float)
            if step(np.r_[q[:7], grip]):
                report["route_log"] = log
                return f"ended during the route ({s.status()['episode_end']})"
        p, r = pe, re
        for _ in range(int(round(.5 / DT))):
            if step(rows[-1]):
                return "ended while settling"
        log.append(dict(move=mv, rows=n, routes=env._task.route_terms(w.read_state())))
        print(json.dumps(dict(stage="route_move", **log[-1])), flush=True)
    for _ in range(int(round(1. / DT))):
        if step(rows[-1]):
            return "ended while settling"
    report["route_log"] = log
    report["route"] = dict(routes=env._task.route_terms(w.read_state()))
    print(json.dumps(dict(stage="route", **report["route"])), flush=True)
    return None


# 1. the old pickup: until the sheath is held and the old rows have stood still for 10 rows (its settle), then stop
held_at, still = None, 0
prefix = None if not a.from_rows else np.asarray(np.load(a.from_rows)["actions"], np.float32)
if prefix is not None and a.prefix_rows:
    prefix = prefix[:a.prefix_rows]
for i, row in enumerate(old if prefix is None else []):
    if step(row):
        raise SystemExit("episode ended during the pickup")
    if held_at is None and grasp_phase() == "holding":
        held_at = i
    if held_at is not None:
        still = still + 1 if np.array_equal(old[i], old[i - 1]) else 0
        if still >= 10 or i > held_at + 400:
            break
if prefix is not None:
    for row in prefix:
        if step(row):
            raise SystemExit("episode ended in the given rows")
grip = float(old[len(rows) - 1][7]) if prefix is None else float(prefix[-1][7])
report["pickup"] = dict(rows=len(rows), held_at_row=held_at, grip=grip, phase=grasp_phase())
print(json.dumps(report["pickup"]), flush=True)
if a.route_program:
    why = route_stage()
    if why:
        report["result"] = why


ADOPT = {}                                           # --adopt: {label: (pid, write end of its pipe)}


def try_stages(cands, adopt=False):
    """Each candidate (label, make_rows, judge) in its own forked copy of this process, all at once: plan + execute,
    judge; each writes rows + verdict. Returns {label: (ok, rows, info)}.
    adopt: the children stay alive after judging; handover() later lets the chosen one carry on as the run (its
    try_stages returns ("adopted", label, rows, decision)) and ends the others and this process."""
    import select
    running = {}
    ADOPT.clear()
    for label, make_rows, judge in cands:
        path = out / f"_cand_{label}.json"
        rfd, wfd = os.pipe() if adopt else (None, None)
        pid = os.fork()
        if pid == 0:
            cand = []
            try:
                cand, info = make_rows()
                ended = False
                for r in cand:
                    if s.step(np.asarray(r, np.float32)):
                        ended = True
                        break
                ok, verdict = judge(ended)
                info.update(verdict)
                res = dict(ok=bool(ok), rows=[list(map(float, r)) for r in cand], info=info)
            except BaseException as e:
                res = dict(ok=False, rows=[], info=dict(error=repr(e)[:400]))
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, default=float))
            tmp.rename(path)                             # atomic: the parent polls for it
            if not adopt or not res["rows"]:
                os._exit(0)
            os.close(wfd)
            ready, _, _ = select.select([rfd], [], [], 4 * 3600.)
            msg = os.read(rfd, 2) if ready else b""
            if msg != b"go":
                os._exit(0)
            decision = json.loads((out / "_adopt.json").read_text())
            return ("adopted", label, res["rows"], decision)
        if adopt:
            os.close(rfd)
            ADOPT[label] = (pid, wfd)
        running[label] = (pid, path)
    results = {}
    for label, (pid, path) in running.items():
        if adopt:
            while not path.exists():                     # a child that died before writing counts as no result
                if os.waitpid(pid, os.WNOHANG)[0] == pid and not path.exists():
                    break
                time.sleep(1.)
        else:
            os.waitpid(pid, 0)
        res = json.loads(path.read_text()) if path.exists() else dict(ok=False, rows=[], info=dict(error="no result"))
        path.unlink(missing_ok=True)
        results[label] = (res["ok"], res["rows"], res["info"])
    return results


def handover(label, decision):
    """--adopt: the chosen candidate's process carries on as the run; the others and this process end here."""
    (out / "_adopt.json").write_text(json.dumps(decision, default=float))
    for lab, (pid, wfd) in ADOPT.items():
        try:
            os.write(wfd, b"go" if lab == label else b"no")
            os.close(wfd)
        except OSError:
            pass
    sys.stdout.flush()
    os._exit(0)


def release_candidates():
    for lab, (pid, wfd) in ADOPT.items():
        try:
            os.write(wfd, b"no")
            os.close(wfd)
        except OSError:
            pass
    ADOPT.clear()


# 2. the combs, as the planner's own loop runs them (stages.stages): while a strand is outside the comb target, comb the
# one farthest from the sheath axis (at most 3 combs per strand). The selection's first combs give the roll and margins.
# A candidate press is accepted when it brings that strand to the aim (target - margin, room for the next combs'
# disturbance) without pushing another past the target; else the press that cuts the summed excess most, if any does.
presses = [float(x) for x in a.presses.split(",")]
par0 = {k: v for k, v in next(e for e in sel if e["step"].startswith("comb_"))["parameters"].items()
        if k not in ("strand", "roll_deg", "d_press_m")}
aim = a.target - a.margin
count = len(measures())


def excess(r):
    return sum(max(0., x - a.target) for x in r)


level, skipped = {}, set()                          # --direct: each strand's press depth index, strands given up
for k in range(0 if (prefix is not None and not a.combs_after_prefix) or "result" in report else 3 * count):
    before = [m["max_radius_m"] for m in measures()]
    if max(before) <= a.target:
        break
    if a.direct:
        left = [j for j in range(count) if before[j] > a.target and j not in skipped]
        if not left:
            report["comb_stop"] = f"comb {k + 1}: every strand outside {a.target} m has had all press depths"
            print(json.dumps(dict(stage="combs", stop=report["comb_stop"])), flush=True)
            if max(before) > a.guided:
                report["result"] = f"{report['comb_stop']}; strands outside the guided {a.guided} m"
            break
        strand = max(left, key=lambda j: before[j])
        d_press = presses[level.get(strand, 0)]
        plan, errors = None, []
        for rank in range(3):                         # the preferred roll, else the next allowed ones
            try:
                plan = plan_comb(FiberIK(env, link=GRASP_LINK), w.read_state(), layout,
                                 dict(par0, d_press_m=d_press, roll_rank=rank), strand=strand)
                break
            except ValueError as e:
                errors.append(repr(e)[:200])
                if "allowed rolls" in str(e):
                    break
        if plan is None:
            info = dict(error=errors[0], errors=errors[1:])
        if plan is not None:
            first, second = plan["guides"]
            cand = refs_to_rows(first["refs"], grip, PAUSE1) + refs_to_rows(second["refs"], grip, PAUSE2)
            for r in cand:
                if step(r):
                    raise SystemExit(f"episode ended in comb {k + 1}")
            after = measures()
            r_ = [m["max_radius_m"] for m in after]
            info = dict(roll_deg=plan["roll_deg"], executed=len(cand), radius_mm=[round(1e3 * x, 2) for x in r_],
                        excess_mm=round(1e3 * excess(r_), 2),
                        max_bend_deg=round(max(m["max_bend_deg"] for m in after), 2), phase=grasp_phase())
            if grasp_phase() != "holding" or info["max_bend_deg"] > MAXIMUM_BEND_DEG:
                report["result"] = f"comb {k + 1}: unsound after the press ({info['phase']}, {info['max_bend_deg']} deg)"
        gained = plan is not None and info["excess_mm"] < 1e3 * excess(before) - .05
        if not gained:                                # deepen this strand's next press, or give it up
            level[strand] = level.get(strand, 0) + 1
            if level[strand] >= len(presses):
                skipped.add(strand)
        print(json.dumps(dict(stage=f"comb_{k + 1:02d}", strand=strand, d_press=d_press, direct=True, gained=gained,
                              **info)), flush=True)
        report["stages"].append(dict(stage=f"comb_{k + 1:02d}", strand=strand, d_press=d_press, direct=True,
                                     gained=gained, info=info))
        if "result" in report:
            break
        continue
    left = [j for j in range(count) if before[j] > a.target and j not in skipped]
    if not left:                                     # candidates: every strand outside the target was given up
        report["comb_stop"] = f"comb {k + 1}: no candidate press reduced the excess for any strand outside {a.target} m"
        print(json.dumps(dict(stage="combs", stop=report["comb_stop"])), flush=True)
        if max(before) > a.guided:
            report["result"] = f"{report['comb_stop']}; strands outside the guided {a.guided} m"
        break
    strand = max(left, key=lambda j: before[j])
    cands = []
    for d_press, rank in [(d, int(rk)) for rk in a.roll_ranks.split(",") for d in presses]:
        p = dict(par0, d_press_m=d_press, roll_rank=rank)

        def make(p=p, strand=strand):
            plan = plan_comb(FiberIK(env, link=GRASP_LINK), w.read_state(), layout, p, strand=strand)
            first, second = plan["guides"]
            cand = refs_to_rows(first["refs"], grip, PAUSE1) + refs_to_rows(second["refs"], grip, PAUSE2)
            return cand, dict(roll_deg=plan["roll_deg"], planned_s=first["refs"][-1]["time_s"] + second["refs"][-1]["time_s"])

        def judge(ended, strand=strand):
            after = measures()
            r = [m["max_radius_m"] for m in after]
            sound = not ended and grasp_phase() == "holding" and max(m["max_bend_deg"] for m in after) <= MAXIMUM_BEND_DEG
            others_ok = all(r[j] <= max(a.target, before[j]) + 5e-4 for j in range(len(r)) if j != strand)
            return sound and r[strand] <= aim and others_ok, dict(
                radius_mm=[round(1e3 * x, 2) for x in r], excess_mm=round(1e3 * excess(r), 2), sound=sound,
                others_ok=others_ok, ended=s.episode_end if ended else None)
        cands.append((f"comb{k}_{d_press}_r{rank}", make, judge))
    results = try_stages(cands, adopt=a.adopt)
    if isinstance(results, tuple):                   # this is the chosen candidate's process: its rows are executed
        _, label, cand_rows, decision = results
        rows.extend(np.asarray(r, np.float32) for r in cand_rows)
        report["stages"].extend(decision["stages"])
        skipped.clear()                              # the bundle changed: strands given up may comb now
        print(json.dumps(dict(stage=f"comb_{k + 1:02d}", executed=len(cand_rows), adopted=label, pid=os.getpid(),
                              radius_mm=[round(1e3 * m["max_radius_m"], 2) for m in measures()])), flush=True)
        continue
    best = None
    entries = []
    labels = [c[0] for c in cands]                   # depth-major per rank: shallowest first, preferred roll first
    for label in labels:
        ok, cand, info = results[label]
        d_press, rank = float(label.split("_")[1]), int(label.split("_r")[-1])
        print(json.dumps(dict(stage=f"comb_{k + 1:02d}", strand=strand, d_press=d_press, roll_rank=rank, ok=ok,
                              **info)), flush=True)
        entries.append(dict(stage=f"comb_{k + 1:02d}", strand=strand, d_press=d_press, roll_rank=rank, ok=ok, info=info))
    report["stages"].extend(entries)
    for label in labels:                             # the shallowest press that meets the aim, else the best excess cut
        ok, cand, info = results[label]
        if ok:
            best = (cand, info, label)
            break
    if best is None:
        for label in labels:
            ok, cand, info = results[label]
            if cand and info.get("sound") and info["excess_mm"] < 1e3 * excess(before) - .05 and (
                    best is None or info["excess_mm"] < best[1]["excess_mm"]):
                best = (cand, info, label)
    if a.adopt:
        if best is not None:
            handover(best[2], dict(stages=entries))     # never returns here
        release_candidates()
    if best is None and a.skip_strands:              # give this strand up for now, comb the next one
        skipped.add(strand)
        print(json.dumps(dict(stage=f"comb_{k + 1:02d}", strand=strand, skipped=True)), flush=True)
        continue
    if best is None:
        if max(before) <= a.guided:
            report["comb_stop"] = f"comb {k + 1}: no press reduced the excess beyond {a.target} m (strand {strand}); all within the guided {a.guided} m"
            print(json.dumps(dict(stage="combs", stop=report["comb_stop"])), flush=True)
        else:
            report["result"] = f"comb {k + 1}: no press reduced the excess beyond {a.target} m (strand {strand})"
        break
    skipped.clear()
    for r in best[0]:
        if step(r):
            raise SystemExit(f"episode ended in comb {k + 1} on the parent's run")
    print(json.dumps(dict(stage=f"comb_{k + 1:02d}", executed=len(best[0]),
                          radius_mm=[round(1e3 * m["max_radius_m"], 2) for m in measures()])), flush=True)
else:
    if (prefix is None or a.combs_after_prefix) and max(m["max_radius_m"] for m in measures()) > a.guided:
        report["result"] = f"strands still outside {a.guided} m after {3 * count} combs"

# 3. verify (1 s) and feed, then hold until the Task succeeds (at most 3 s more)
if "result" not in report:
    for _ in range(int(round(1. / DT))):
        step(rows[-1])
    r = [m["max_radius_m"] for m in measures()]
    report["verified_radius_mm"] = [round(1e3 * x, 2) for x in r]
    np.savez(out / "rows_combed.npz", actions=np.asarray(rows, np.float32))   # a prefix: pickup, route, combs, verify
    report["combed_rows"] = len(rows)
    feed = next(e for e in sel if e["step"] == "feed")["parameters"]
    why = feed_closed_loop(1e-3 * a.front_s_mm if a.front_s_mm is not None else feed["front_s_m"], feed["hold_s"])
    report["feed"] = dict(mode="closed loop", speed_m_s=a.feed_speed, gain=a.feed_gain, gate_m=a.feed_gate, stop=why,
                          near_mm=a.near_mm, near_past_mm=a.near_past_mm, near_speed=a.near_speed, near_gate=a.near_gate,
                          force_hold=a.force_hold, force_back=a.force_back, back_mm=a.back_mm, lag_max=a.lag_max)
    print(json.dumps(dict(stage="feed", **report["feed"], log_tail=report.get("feed_log", [])[-3:])), flush=True)
    ended = why == "ended"
    for _ in range(int(round(3. / DT))):
        if ended:
            break
        ended = step(rows[-1])
    st = s.status()
    report["fed_radius_mm"] = [round(1e3 * m["max_radius_m"], 2) for m in measures()]
    m = dict(getattr(env._task.result, "metrics", None) or {})
    report["final_metrics"] = {k: m.get(k) for k in ("strands_complete", "front_sheath_clear", "maximum_bend_deg",
                                                    "passage_reached", "lay", "routes")}
    report.update(result="success" if st["success"] else f"no success ({st['episode_end']})", status=st)
    print(json.dumps(dict(stage="final", **report["final_metrics"]), default=str), flush=True)
report.update(rows=len(rows), wall_s=round(time.time() - t_start, 1), ik_relaxed_solves=IK_RELAXED[0])
np.savez(out / "rows.npz", actions=np.asarray(rows, np.float32))
(out / "report.json").write_text(json.dumps(report, indent=1, default=str))
print(json.dumps(dict(result=report["result"], rows=len(rows), wall_s=report["wall_s"])), flush=True)
