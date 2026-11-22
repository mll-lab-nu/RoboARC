"""Live == replay through the table guard (speed lane, 2026-10-13; the fold easier r2 case, row 1328).

A row the guard stops is recorded as the float32 row it stepped. A replay steps that row through the same guard, so the
guard must pass its own output unchanged (idempotence), or the replay commands another target than the live run did and
leaves it. The cloth floor (profiles with `cloth_floor`) judged its stop before the float32 rounding; its floor is not
continuous in the joints (a grid of cloth heights), so a rounded stop could lie below it and be stopped again on replay.

Two checks on one built Session:
  1. property: from a joint row just above the floor, --samples rows that ask to go lower are guarded (G); the guard on
     G (same previous row, same state) must return G unchanged, with no guard event. Pure kinematics, no stepping.
  2. live vs replay: --steps rows that drive each gripper down into its floor are stepped live (recorded), then the
     Session is reset and the recorded rows replayed (Session.replay, what the verifier does): the state hash must be
     the live one, and no replayed row may be changed by the guard.

usage: test_guard_replay.py --instance <instance.json> [--samples 400] [--steps 40] [--seed 0] [--unbound]
  --unbound: an instance not bound to this checkout (dev trees): skip the binding checks, as the speed tools do.
Exit status 0 = both checks pass. Examples: a blanket_fold instance (cloth floor; the CPU twin profile is enough) and a
MuJoCo-only family (rope_threading / knot_untying: the table floor)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--instance", required=True)
ap.add_argument("--samples", type=int, default=400)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--unbound", action="store_true")
a = ap.parse_args()

import rcb_topogym as R  # noqa: E402
if a.unbound:
    import topogym.simulator.frozen_binding as fb
    fb.check_inputs = lambda instance_path, tooling_path, lock: json.loads(Path(instance_path).read_text())
    R.check_runtime = lambda lock: dict(lock.get("runtime") or {})
    if hasattr(R, "_check_render_bridge"):
        R._check_render_bridge = lambda lock: None

f32 = lambda x: np.asarray(x, dtype=np.float32).astype(np.float64)  # noqa: E731
s = R.Session(json.loads(Path(a.instance).read_text())["task"], instance=a.instance, strict=True)
g, robot = s.table_guard, s.robot
if g.limit is None or not g.points:
    sys.exit("this task has no table guard")
cloth = getattr(g, "cloth", None) is not None
print(f"task {s.task}: guard floor = {'cloth' if cloth else 'table'}; arms {sorted(g.points)}", flush=True)


def clearance(row):
    if cloth:
        g._heights = g._cloth_heights()
        return g.clearance(row)
    return {arm: z - g.limit for arm, z in g.lowest(row).items()}


def descent(row, arm, eps=1e-4):
    """The unit joint direction that lowers the arm's gripper fastest (central differences of its clearance)."""
    sl = robot.arm_slice[arm]
    grad = np.zeros(sl.stop - sl.start)
    for j in range(len(grad)):
        up, dn = row.copy(), row.copy()
        up[sl.start + j] += eps
        dn[sl.start + j] -= eps
        grad[j] = (clearance(up)[arm] - clearance(dn)[arm]) / (2 * eps)
    n = np.linalg.norm(grad)
    return None if n == 0 else -grad / n


rng = np.random.default_rng(a.seed)
failures = []


def near_floor(row, arm):
    """row with the arm moved (IK) over the cloth (the particle nearest its TCP, 4 cm above; the table floor: in place),
    then walked down its clearance gradient to <= 3 mm above its floor -> (row, unit descent direction, its slope m/rad)."""
    sl = robot.arm_slice[arm]
    row = row.copy()
    if cloth:
        pos, quat = s.ee_pose(arm)
        P = np.asarray(g.cloth.particles_of(g.cloth.vertices()), dtype=np.float64)
        p = P[np.argmin(np.linalg.norm(P[:, :2] - np.asarray(pos)[:2], axis=1))]
        q, _, _ = s.ik(arm, p + np.r_[0., 0., .04], quat)
        row[sl] = q
        row = f32(robot.clip(row))
    for _ in range(400):
        c = clearance(row)[arm]
        d = descent(row, arm)
        if c <= .003 or d is None:
            break
        row[sl] += d * min(.02, max(.0005, c))
        row = f32(robot.clip(row))
    d, gn = descent(row, arm), None
    if d is not None:
        e = 1e-4
        gn = (clearance(row)[arm] - clearance(row + np.r_[np.zeros(sl.start), d * e, np.zeros(len(row) - sl.stop)])[arm]) / e
    return row, d, gn


# 1. property: guard(guard(raw)) == guard(raw)
start = f32(s._targets)
guarded, targets = 0, {}
for arm in sorted(g.points):
    sl = robot.arm_slice[arm]
    prev, d, gn = near_floor(start, arm)
    c0 = clearance(prev)[arm]
    if d is None or not gn or gn <= 0 or c0 > .006:
        print(f"  {arm}: no floor within reach, skipped", flush=True)
        continue
    targets[arm] = (prev[sl].copy(), d, gn)
    below = 0
    print(f"  {arm}: start {c0 * 1e3:+.2f} mm above its floor, {gn * 1e3:.1f} mm per rad down", flush=True)
    for _ in range(a.samples):
        raw = prev.copy()
        raw[sl] += d * rng.uniform(0., 3 * max(c0, 1e-3) / gn) + rng.normal(0., .1 * max(c0, 1e-3) / gn, sl.stop - sl.start)
        raw = f32(robot.clip(raw))
        below += clearance(raw)[arm] < 0
        G = np.asarray(g.apply(raw, prev), dtype=np.float64)
        if s.last_guard is None:
            continue
        guarded += 1
        G2 = np.asarray(g.apply(G.copy(), prev), dtype=np.float64)
        if s.last_guard is not None or not np.array_equal(G2, G):
            failures.append(dict(check="idempotence", arm=arm, event=s.last_guard,
                                 max_change=float(np.abs(G2 - G).max())))
    print(f"  {arm}: {below} of {a.samples} rows below the floor", flush=True)
if cloth:
    # 1b. the cloth's edge, where the floor jumps (table limit off the cloth, cloth top on it): each arm's TCP 8 mm off
    # the hull point of the cloth nearest it, 1 mm below the local cloth top; rows that move it 0-20 mm onto the cloth
    # are stopped at the jump, where a stop judged before the float32 rounding can round below the floor
    from scipy.spatial import ConvexHull
    P = np.asarray(g.cloth.particles_of(g.cloth.vertices()), dtype=np.float64)
    hull = P[ConvexHull(P[:, :2]).vertices]
    mid = P[:, :2].mean(axis=0)
    radius = float(g.cloth.settings["radius_m"])
    for arm in sorted(g.points):
        sl = robot.arm_slice[arm]
        pos, quat = s.ee_pose(arm)
        e = hull[np.argmin(np.linalg.norm(hull[:, :2] - np.asarray(pos)[:2], axis=1))]
        n = np.r_[(e[:2] - mid) / np.linalg.norm(e[:2] - mid), 0.]
        z = e[2] + radius - .001
        q0, err0, _ = s.ik(arm, np.r_[e[:2], z] + .008 * n, quat)
        q1, err1, _ = s.ik(arm, np.r_[e[:2], z] - .012 * n, quat)
        if max(err0, err1) > .002:
            print(f"  {arm}: edge out of reach (IK error {max(err0, err1) * 1e3:.1f} mm), skipped", flush=True)
            continue
        prev = start.copy()
        prev[sl] = q0
        prev = f32(robot.clip(prev))
        edge = 0
        for _ in range(a.samples):
            raw = prev.copy()
            raw[sl] = prev[sl] + (q1 - q0) * rng.uniform(0., 1.)
            raw = f32(robot.clip(raw))
            G = np.asarray(g.apply(raw, prev), dtype=np.float64)
            if s.last_guard is None:
                continue
            guarded += 1
            edge += 1
            G2 = np.asarray(g.apply(G.copy(), prev), dtype=np.float64)
            if s.last_guard is not None or not np.array_equal(G2, G):
                failures.append(dict(check="idempotence", arm=arm, where="edge", event=s.last_guard,
                                     max_change=float(np.abs(G2 - G).max())))
        print(f"  {arm} at the cloth edge: start {clearance(prev)[arm] * 1e3:+.2f} mm above its floor, {edge} guarded rows",
              flush=True)
print(f"1. property: {guarded} guarded rows, {sum(f['check'] == 'idempotence' for f in failures)} changed by a "
      "second guard", flush=True)

# 2. live vs replay: 30 rows to 2 cm above each arm's near-floor row, then --steps rows pressing down (random depth)
s.reset()
before_live = g.count
s.record_start()
row0 = f32(s._targets)
for i in range(30 + a.steps):
    row = f32(s._targets)
    for arm, (q, d, gn) in targets.items():
        sl = robot.arm_slice[arm]
        if i < 30:
            row[sl] = row0[sl] + (q - d * .02 / gn - row0[sl]) * (i + 1) / 30
        else:
            row[sl] += d * rng.uniform(0., .002) / gn + rng.normal(0., .0002 / gn, sl.stop - sl.start)
    if s.step(f32(robot.clip(row)).astype(np.float32)):
        break
acts = s.record_stop()
ticks = s._saved_recording[1]
live_sha, live_events = s.state_sha(), g.count
s.reset()
before = g.count
changed = []
for i, row in enumerate(acts):
    ended = s.step(row, _physics_steps=int(ticks[i]))
    if not np.array_equal(np.asarray(s._targets, dtype=np.float32), row):
        changed.append(i)
    if ended:
        break
replay_sha = s.state_sha()
print(f"2. live vs replay: {len(acts)} rows, guard events live {live_events - before_live}, replay {g.count - before}; "
      f"rows changed on replay {changed}; live {live_sha} replay {replay_sha}", flush=True)
if changed or replay_sha != live_sha:
    failures.append(dict(check="live_vs_replay", changed=changed, live=live_sha, replay=replay_sha))
print("PASS" if not failures else "FAIL " + json.dumps(failures[:5], default=str))
sys.exit(1 if failures else 0)
