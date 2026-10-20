"""Plan a fiber_bundling reference on the current physics in the production Session, seeded from the released one (the
fiber agent's re-planner; docs/rcb/families/fiber_bundling.md, "References"):
1. the released rows up to the end of the pickup (the sheath held, the arm still), or the first --prefix-rows rows of
   --from-rows (a previous reference's pickup and combs);
2. the planner's comb loop (topogym/planning/fiber_bundling/stages.py: plan_comb, the strand farthest from the sheath
   axis first, from the measured state), each press deep enough that the strand stays inside once released (candidates
   tried in forked copies of the process, shallowest first; the parent then executes the chosen rows itself);
3. a 1 s verify hold, then the feed, closed loop as the planner's own policy runs it: a planned, collision-checked leg
   aligns the front link with the ring axis; every control step the measured front point's lateral offset is
   integrated into the commanded pose, and the front advances only through its gates (lateral, and optionally a slow
   zone at the ring, the commanded front's lead over the measured one, the grasp weld's force); then hold rows until
   the task succeeds.
Planner references become joint-target rows by time (25 ms), open loop; each guide is followed by the planner's pause.

Run it with the RoboTangle image's Python, as the stress tools (tools/stress/_env.py: this checkout's
topogym and tooling first on sys.path, the task containers' numpy / OpenBLAS SIMD masking, one thread per library).

usage: replan.py <tier> <out_dir> --instance <instance.json> [--from-rows <npz> --prefix-rows N] [feed gates]
writes <out_dir>/rows.npz (actions) and report.json; record the rows afterwards (interlocked_claws/rerecord.py).
The -i01 very_easy / easier feeds (on the exact constraint diagonal): --from-rows the -i01 oracle, --prefix-rows 2941 /
2841, --near-mm 15 --near-past-mm 10 --near-speed .004 --near-gate .0005 --lag-max .003 --force-hold 20 --force-back 26."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "stress"))
import _env  # noqa: F401,E402  (before numpy: SIMD masking, threads, this checkout first on sys.path)
import argparse, json, os, time  # noqa: E402
import numpy as np  # noqa: E402
from rcb_topogym import Session  # noqa: E402
import topogym  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tier"); ap.add_argument("out")
ap.add_argument("--instance", required=True, help="the tier's MuJoCo instance.json")
ap.add_argument("--presses", default=".032,.030,.028,.026,.024")
ap.add_argument("--target", type=float, default=.0115, help="comb target (stages.COMB_TARGET_M)")
ap.add_argument("--margin", type=float, default=.0015, help="aim this much inside the target")
ap.add_argument("--guided", type=float, default=.016, help="the planner's guided target (stages.GUIDED_TARGET_M): when no "
                "comb gets every strand within --target but all are within this, feed anyway")
ap.add_argument("--from-rows", help="replay these rows (a previous run's pickup and combs), then verify and feed")
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
released = HERE.parent / "oracles_genesis" / base / "privileged.npz"          # the released (Genesis) rows: the pickup
sel = json.loads((Path(topogym.__file__).resolve().parent / "planning/harbor_bridge/reference_data/fiber_bundling" / a.tier
                  / "selection.json").read_text())
s = Session("fiber_bundling", instance=inst, strict=True)
env = s.env
w = env._world
from topogym.planning.fiber_bundling.stages import plan_comb, plan_feed, COMBS, MAXIMUM_BEND_DEG, connect, measured_offset
from topogym.planning.fiber_bundling.keyframes import ring_frame, link_pose, GRASP_LINK, rotation
from scipy.spatial.transform import Rotation
from topogym.planning.fiber_bundling.kinematics import FiberIK
from topogym.planning.fiber_bundling.keyframes import strand_measures

layout = env._task.layout
DT = float(w.physics_dt_s) * 10                      # one control step
old = np.asarray(np.load(released)["actions"], np.float32)
rows, report = [], dict(tier=a.tier, instance=inst, provenance=_env.provenance(), stages=[])
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


def plan_feed_bent(kin, state, front_s, max_offset, standoff=.02):
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
    kin = FiberIK(env)
    plan = plan_feed_bent(kin, state, front_s, 1.)                 # the alignment leg (its feed leg is not used)
    first = plan["plans"][0]
    for r in refs_to_rows(first["refs"], grip, .5):
        if step(r):
            return "ended during the alignment"
    center, axis = ring_frame(w.read_state())
    front_link = f"trunk_{layout.trunk_segments - 1:02d}"
    L = layout.trunk_length_m / layout.trunk_segments
    st = w.read_state()
    p4, r4 = link_pose(st, GRASP_LINK); p5, r5 = link_pose(st, front_link)
    r_rel, p_rel = r4.inv() * r5, r4.inv().apply(p5 - p4)
    offset_p, offset_r = measured_offset(kin, st)
    x5 = r5.apply([1., 0., 0.]); v = np.cross(x5, axis); sv = float(np.linalg.norm(v))
    r5t = (Rotation.from_rotvec(v / sv * np.arctan2(sv, float(x5 @ axis))) if sv > 1e-12 else Rotation.identity()) * r5
    s_des = float((p5 + r5.apply([L, 0., 0.]) - center) @ axis)
    corr, rot_corr, q, held, log = np.zeros(3), np.zeros(3), np.asarray(first["refs"][-1]["qpos"], float), 0, []
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
            q = np.asarray(kin.solve((p4t + r4t.apply(offset_p), r4t * offset_r), q), float)
        except Exception as e:
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


def try_stages(cands):
    """Each candidate (label, make_rows, judge) in its own forked copy of this process, all at once: plan + execute,
    judge; each writes rows + verdict. Returns {label: (ok, rows, info)}."""
    running = {}
    for label, make_rows, judge in cands:
        path = out / f"_cand_{label}.json"
        pid = os.fork()
        if pid == 0:
            try:
                cand, info = make_rows()
                ended = False
                for r in cand:
                    if s.step(np.asarray(r, np.float32)):
                        ended = True
                        break
                ok, verdict = judge(ended)
                info.update(verdict)
                path.write_text(json.dumps(dict(ok=bool(ok), rows=[list(map(float, r)) for r in cand], info=info),
                                           default=float))
            except BaseException as e:
                path.write_text(json.dumps(dict(ok=False, rows=[], info=dict(error=repr(e)[:400]))))
            os._exit(0)
        running[label] = (pid, path)
    results = {}
    for label, (pid, path) in running.items():
        os.waitpid(pid, 0)
        res = json.loads(path.read_text()) if path.exists() else dict(ok=False, rows=[], info=dict(error="no result"))
        path.unlink(missing_ok=True)
        results[label] = (res["ok"], res["rows"], res["info"])
    return results


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


for k in range(0 if prefix is not None else 3 * count):
    before = [m["max_radius_m"] for m in measures()]
    if max(before) <= a.target:
        break
    strand = int(np.argmax(before))
    cands = []
    for d_press in presses:
        p = dict(par0, d_press_m=d_press)

        def make(p=p, strand=strand):
            plan = plan_comb(FiberIK(env), w.read_state(), layout, p, strand=strand)
            first, second = plan["guides"]
            cand = refs_to_rows(first["refs"], grip, 1.) + refs_to_rows(second["refs"], grip, .5)
            return cand, dict(roll_deg=plan["roll_deg"], planned_s=first["refs"][-1]["time_s"] + second["refs"][-1]["time_s"])

        def judge(ended, strand=strand):
            after = measures()
            r = [m["max_radius_m"] for m in after]
            sound = not ended and grasp_phase() == "holding" and max(m["max_bend_deg"] for m in after) <= MAXIMUM_BEND_DEG
            others_ok = all(r[j] <= max(a.target, before[j]) + 5e-4 for j in range(len(r)) if j != strand)
            return sound and r[strand] <= aim and others_ok, dict(
                radius_mm=[round(1e3 * x, 2) for x in r], excess_mm=round(1e3 * excess(r), 2), sound=sound,
                others_ok=others_ok, ended=s.episode_end if ended else None)
        cands.append((f"comb{k}_{d_press}", make, judge))
    results = try_stages(cands)
    best = None
    for d_press in presses:                          # the shallowest press that meets the aim, else the best excess cut
        ok, cand, info = results[f"comb{k}_{d_press}"]
        print(json.dumps(dict(stage=f"comb_{k + 1:02d}", strand=strand, d_press=d_press, ok=ok, **info)), flush=True)
        report["stages"].append(dict(stage=f"comb_{k + 1:02d}", strand=strand, d_press=d_press, ok=ok, info=info))
    for d_press in presses:
        ok, cand, info = results[f"comb{k}_{d_press}"]
        if ok:
            best = (cand, info)
            break
    if best is None:
        for d_press in presses:
            ok, cand, info = results[f"comb{k}_{d_press}"]
            if cand and info.get("sound") and info["excess_mm"] < 1e3 * excess(before) - .05 and (
                    best is None or info["excess_mm"] < best[1]["excess_mm"]):
                best = (cand, info)
    if best is None:
        if max(before) <= a.guided:
            report["comb_stop"] = f"comb {k + 1}: no press reduced the excess beyond {a.target} m (strand {strand}); all within the guided {a.guided} m"
            print(json.dumps(dict(stage="combs", stop=report["comb_stop"])), flush=True)
        else:
            report["result"] = f"comb {k + 1}: no press reduced the excess beyond {a.target} m (strand {strand})"
        break
    for r in best[0]:
        if step(r):
            raise SystemExit(f"episode ended in comb {k + 1} on the parent's run")
    print(json.dumps(dict(stage=f"comb_{k + 1:02d}", executed=len(best[0]),
                          radius_mm=[round(1e3 * m["max_radius_m"], 2) for m in measures()])), flush=True)
else:
    if prefix is None and max(m["max_radius_m"] for m in measures()) > a.guided:
        report["result"] = f"strands still outside {a.guided} m after {3 * count} combs"

# 3. verify (1 s) and feed, then hold until the Task succeeds (at most 3 s more)
if "result" not in report:
    for _ in range(int(round(1. / DT))):
        step(rows[-1])
    r = [m["max_radius_m"] for m in measures()]
    report["verified_radius_mm"] = [round(1e3 * x, 2) for x in r]
    feed = next(e for e in sel if e["step"] == "feed")["parameters"]
    why = feed_closed_loop(feed["front_s_m"], feed["hold_s"])
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
    report.update(result="success" if st["success"] else f"no success ({st['episode_end']})", status=st)
report.update(rows=len(rows), wall_s=round(time.time() - t_start, 1))
np.savez(out / "rows.npz", actions=np.asarray(rows, np.float32))
(out / "report.json").write_text(json.dumps(report, indent=1, default=str))
print(json.dumps(dict(result=report["result"], rows=len(rows), wall_s=report["wall_s"])), flush=True)
