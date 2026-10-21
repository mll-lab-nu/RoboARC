#!/usr/bin/env python
"""clock_assembling's rule changes of 2026-10-19 (the metric review, runs/robotangle_mujoco/metric_review_20261019/
clock_zh.md), checked with the parts put in place by their free joints (as a hand would put them; the arms hold their birth
pose) and the rules and the grader stepped as in an episode:

  1. bridge first: the barrel seated, the bridge laid on its pillars, both screws tightened: the bridge is not fixed while
     the other wheels are missing (the review's finding 1);
  2. a screw resting on the bridge under its own weight, turned in place: its turns count (finding 3);
  3. the screws swapped (screw2 in P1's hole, screw1 in P2's): both count and tighten (finding 2);
  4. the right order (wheels, bridge, screws): the bridge is fixed;
  5. from that state, at rest, the arbor wound fast until the click holds about 125 deg: the run passes (finding 5);
  6. the same, with the probe laid against the drum after the winding: the run is void, no success; wound again with the
     probe gone, it passes;
  7. a binding train (the third wheel's spin friction 1e-3 N m) with the drum driven backwards: void, no success
     (finding 4; the drum driven forwards without a contact is not caught by these rules);
  8. after 1, the aligned drop (the review's finding 1, its geometry change (b)): the center wheel released just above the
     unfixed bridge (its lower pivot 2 mm over the bridge's top) with its axis 0, 0.2, 0.4 or 0.6 mm off its hole's does
     not seat (the bridge is solid to the wheels: it rests on the bridge), nor do the center, third and fourth wheels
     released exactly over their holes, and the bridge stays unfixed;
  9. the right order again with every wheel seated about as high as the seat rule allows (put 0.78 mm over its seat; its
     weld keeps it there, 0.74 mm with the weld's give), the bridge laid on its pillars and screwed: it sits flat, and the
     run passes (no arbor touches the bridge, which is solid to the wheels: the arbors collide only up to 0.4 mm under it
     even seated 0.8 mm high).

  python scripts/robotangle/test_clock_rules.py --instance <instance.json> [--out result.json]
Run it where make-instance runs (the RoboTangle MuJoCo image, the SIMD masking, rendering off). Exit 0 when every check
holds; prints one JSON line."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "shared"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    import mujoco
    from rcb_topogym import Session
    from topogym.assets.geometry import clock_gear_train as geo
    s = Session("clock_assembling", instance=a.instance, strict=True)
    w, env = s.env._world, s.env
    m, d, mech, task = w.model, w.data, w._mechanism, s.env._task
    o = mech.o
    cmd = s.robot.command(s.robot.clip(np.asarray(s.hold_action(), dtype=np.float64)))
    J = mujoco.mjtObj.mjOBJ_JOINT
    out, info = {}, {}

    def free(name):
        j = mujoco.mj_name2id(m, J, f"{geo.ENTITY}::{name}_free")
        return int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])

    def put(name, pos, quat=None):
        qa, va = free(name)
        d.qpos[qa:qa + 3] = pos
        if quat is not None:
            d.qpos[qa + 3:qa + 7] = quat
        d.qvel[va:va + 6] = 0.
        mujoco.mj_forward(m, d)

    def yawq(x):
        return np.array([math.cos(x / 2), 0., 0., math.sin(x / 2)])

    def step(n=1):
        res = None
        for _ in range(n):
            res = env.step(cmd)
            if res[2] or res[3]:
                break
        return res

    def hole_world(k):
        bp, BR = mech.bridge_local(d)
        return o + bp + BR @ np.r_[geo.PILLARS[k] + geo.SCREW_OFF, geo.T_BRIDGE]

    def lift_screws(on):
        for sname in ("screw1", "screw2"):
            b = mech.b[sname]
            d.xfrc_applied[b, :] = 0.
            if on:                                          # held as a hand holds it: its weight carried
                d.xfrc_applied[b, 2] = float(m.body_mass[b]) * float(np.linalg.norm(m.opt.gravity))

    def tighten(pairs, limit=120):
        """Turn screws held 0.2 mm over the given holes clockwise, 10 deg a control step, until tight."""
        lift_screws(True)
        yaw, k = 0., 0
        while not all(mech.tight[sn] for sn, _ in pairs) and k < limit:
            yaw -= math.radians(10.)
            for sn, hole in pairs:
                if not mech.tight[sn]:
                    put(sn, hole_world(hole) + np.r_[0., 0., .0002], yawq(yaw))
            step(1)
            k += 1
        lift_screws(False)
        step(10)
        return k

    def seat_wheel(n, z=.0002):
        """n put over its hole with its leaves in the previous wheel's tooth gaps."""
        best = None
        for kk in range(120):
            yw = 2 * math.pi * kk / 120
            put(n, o + np.r_[geo.POS[n], z], yawq(yw))
            if n not in geo.PARENT:
                best = (0., yw)
                break
            e = mech.phase_error(d, n)
            if best is None or abs(abs(e) - .5) < best[0]:
                best = (abs(abs(e) - .5), yw)
        put(n, o + np.r_[geo.POS[n], z], yawq(best[1]))

    def wind(deg):
        """A hand turning the square counterclockwise, fast, until the arbor has advanced deg; then let go."""
        n = 0
        d.qfrc_applied[mech.v_click] = .01
        while mech.max_arbor - mech.arbor0 < math.radians(deg) and n < 400:
            step(1)
            n += 1
        d.qfrc_applied[mech.v_click] = 0.

    def watch(n=320):
        """Up to n control steps (8 s): whether the episode ends in success, and when."""
        for k in range(n):
            r = step(1)
            if r[2] or r[3]:
                return bool(r[4].get("task_success")), k + 1
        return False, None

    # ---- 1-3: bridge first, the screws
    qa, _ = free("barrel")
    put("barrel", o + np.r_[geo.POS["barrel"], .0002], d.qpos[qa + 3:qa + 7].copy())
    step(4)
    put("bridge", o + np.r_[0., 0., geo.H_BRIDGE + .0002], np.array([1., 0., 0., 0.]))
    step(20)
    put("screw1", hole_world("P1") + np.r_[0., 0., .0002], yawq(0.))
    step(20)
    t0, yaw = mech.turns["screw1"], 0.
    for _ in range(12):                                     # resting: 120 deg clockwise in place
        yaw -= math.radians(10.)
        qa, va = free("screw1")
        d.qpos[qa + 3:qa + 7] = yawq(yaw)
        d.qvel[va:va + 6] = 0.
        mujoco.mj_forward(m, d)
        step(1)
    out["2_resting_screw_counts"] = bool(mech.turns["screw1"] - t0 > .3)
    info["2_resting_turns"] = round(mech.turns["screw1"] - t0, 4)
    tighten((("screw2", "P1"), ("screw1", "P2")))
    out["3_swapped_screws_tight"] = bool(all(mech.tight.values()) and mech.hole == {"screw1": "P2", "screw2": "P1"})
    info["3_turns"] = {k: round(v, 4) for k, v in mech.turns.items()}
    step(40)
    out["1_bridge_first_not_fixed"] = bool(all(mech.tight.values()) and not mech.fixed)
    # 8: wheels dropped onto the unfixed bridge from just above it, over their holes
    stA, tsA = w.capture_state(), task.get_state()
    z_drop = geo.H_BRIDGE + geo.T_BRIDGE + geo.LOW_PIVOT_L + .002     # the lower pivot's tip 2 mm over the bridge's top

    def back_to_a():
        w.restore_state(stA)
        task.set_state(tsA)
        env._terminated = env._truncated = False

    sweep = {}
    for off in (0., .0002, .0004, .0006):
        back_to_a()
        seat_wheel("center", z=z_drop)
        qa, _ = free("center")
        d.qpos[qa] += off
        mujoco.mj_forward(m, d)
        step(40)
        p, _ = mech.wheel_local(d, "center")
        sweep[f"{off * 1000:g} mm"] = dict(seated=mech.seated["center"], entered=mech.entered(d, "center"),
                                           height_mm=round(float(p[2]) * 1e3, 2))
    back_to_a()
    for n in ("center", "third", "fourth"):
        seat_wheel(n, z=z_drop)
        step(40)
    drop = dict(seated=dict(mech.seated), entered={n: mech.entered(d, n) for n in geo.WHEELS}, fixed=mech.fixed)
    info["8_aligned_drop"] = dict(center_offset_sweep=sweep, all_three_aligned=drop)
    out["8_aligned_drop_blocked"] = bool(not any(v["seated"] for v in sweep.values())
                                         and not any(drop["seated"][n] for n in ("center", "third", "fourth"))
                                         and not drop["fixed"])

    # ---- 4: the right order
    s.reset()
    for n in geo.WHEELS:
        if n == "barrel":
            qa, _ = free("barrel")
            put("barrel", o + np.r_[geo.POS["barrel"], .0002], d.qpos[qa + 3:qa + 7].copy())
        else:
            seat_wheel(n)
        step(4)
    put("bridge", o + np.r_[0., 0., geo.H_BRIDGE + .0002], np.array([1., 0., 0., 0.]))
    step(20)
    tighten((("screw1", "P1"), ("screw2", "P2")))
    out["4_right_order_fixed"] = bool(mech.fixed and all(mech.seated.values()))
    if not mech.fixed:
        print(json.dumps(dict(ok=False, **out, info=info)), flush=True)
        sys.exit(1)

    # ---- 5-7: the run
    st0, ts0 = w.capture_state(), task.get_state()
    third = mech.jv["third"]["spin"]
    f0 = float(m.dof_frictionloss[third])
    drum = mech.jv["barrel"]["spin"]

    def run(target_deg, jam=False, drive=0., probe=False, rewind=False):
        w.restore_state(st0)
        task.set_state(ts0)
        env._terminated = env._truncated = False
        d.qfrc_applied[:] = 0.
        m.dof_frictionloss[third] = 1e-3 if jam else f0
        wind(target_deg)
        held = math.degrees(mech.max_arbor - mech.arbor0 - geo.CLICK_PITCH)
        if probe:                                           # the probe laid on the plate against the drum's front
            r_drum = geo.pitch_r("barrel") - 1.25 * geo.GEARS["barrel"]["m"] - geo.COL_SHRINK
            put("probe", o + np.r_[geo.POS["barrel"][0] - .045, geo.POS["barrel"][1] - r_drum - geo.PROBE_R + .0001,
                                   geo.PROBE_R + .0001], np.array([1., 0., 0., 0.]))
        d.qfrc_applied[drum] = drive
        res = dict(held_deg=round(held, 1))
        res["success"], res["at_step"] = watch()
        res.update(void=bool(mech.run_void), turns=round(float(mech.read()["run_fourth_turns"]), 3))
        if rewind and not res["success"]:                   # the probe away, wound again: a fresh run
            put("probe", task.origin() + geo.opening_poses()["probe"][0], np.array([1., 0., 0., 0.]))   # on its stand
            wind(target_deg + 30.)
            res["rewound_success"], _ = watch()
        m.dof_frictionloss[third] = f0
        d.qfrc_applied[:] = 0.
        return res

    info["5_from_rest_fast_125"] = r5 = run(143.)
    out["5_from_rest_125_passes"] = bool(r5["success"] and 120. <= r5["held_deg"] < 135.)
    info["6_probe_on_drum"] = r6 = run(170., probe=True, rewind=True)
    out["6_probe_touch_void"] = bool(not r6["success"] and r6["void"] and r6["rewound_success"])
    info["7_jammed_drum_backwards"] = r7 = run(170., jam=True, drive=-.12)
    out["7_backwards_void"] = bool(not r7["success"] and r7["void"])

    # ---- 9: every wheel seated high
    s.reset()
    env._terminated = env._truncated = False
    for n in geo.WHEELS:
        if n == "barrel":
            qa, _ = free("barrel")
            put("barrel", o + np.r_[geo.POS["barrel"], .00078], d.qpos[qa + 3:qa + 7].copy())
        else:
            seat_wheel(n, z=.00078)
        step(4)
    heights = {n: round(float(mech.wheel_local(d, n)[0][2]) * 1e3, 3) for n in geo.WHEELS}
    put("bridge", o + np.r_[0., 0., geo.H_BRIDGE + .0002], np.array([1., 0., 0., 0.]))
    step(20)
    tighten((("screw1", "P1"), ("screw2", "P2")))
    dz, lat, tilt = mech.bridge_seat_error(d)
    r9 = dict(seat_height_mm=heights, fixed=mech.fixed, bridge_dz_mm=round(dz * 1e3, 3), bridge_tilt_deg=round(tilt, 4))
    if mech.fixed:
        wind(170.)
        r9["success"], r9["at_step"] = watch()
        r9.update(turns=round(float(mech.read()["run_fourth_turns"]), 3), stall_s=round(float(mech.read()["run_stall_s"]), 3))
    info["9_high_seats"] = r9
    out["9_high_seats_run"] = bool(all(.7 <= v <= .8 for v in heights.values()) and r9.get("success")
                                   and abs(dz) < .0001 and tilt < .01)
    ok = all(out.values())
    line = json.dumps(dict(ok=ok, **out, info=info))
    if a.out:
        Path(a.out).write_text(line + "\n")
    print(line, flush=True)
    s.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
