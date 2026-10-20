"""horseshoe_ring on MuJoCo: the one-hand hang from the frozen birth, closed loop (planning tool, no reference).

The right hand takes the right shoe with the tier's recorded grasp (stages.HangPolicy: open, joint approach, Cartesian
descent, close, binding check), lifts it by --lift and, with --shed, pitches its eye end down and levels it again; the
left hand stays at home. The chains and the left
shoe then hang from the held shoe, and the ring, which is around both chains, rests on the hanging shoe or slides down
over it and falls on the table. Reported after --hold seconds: what passes through the ring (bodies' vertices inside its
bore), each body's box over the table, the ring's lowest and highest point, the stress observer's verdict.

  python hang_probe.py <instance.json> <tier> <out.json> [--lift 0.32] [--hold 6] [--shed 30] [--shed-lift 0.08] [--yaw 0]
      [--views prefix]
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[2] / "stress"), str(HERE.parents[2] / "shared")]
import _env  # noqa: F401,E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

DATA = HERE.parents[4] / "topogym/planning/harbor_bridge/reference_data/horseshoe_ring"


def pose(body):
    q = body["quaternion_wxyz"]
    return np.asarray(body["position_m"], float), Rotation.from_quat([q[1], q[2], q[3], q[0]])


def through_ring(task, world):
    """Remainder vertices inside the ring's bore (within 4 mm of its plane), by body: count and mean local position (mm)."""
    bodies, loop = world["bodies"], task.assembly.loops["target_ring"]
    rp, rr = pose(bodies["target_ring"])
    centre, frame = rp + rr.apply(loop.center - task.body_origins["target_ring"]), rr * loop.rotation
    a, b = float(loop.radii[0]) - loop.wire_radius, float(loop.radii[1]) - loop.wire_radius
    out = {}
    for name in task.placement_remainder:
        p, r = pose(bodies[name])
        local = np.asarray(task.placement_vertices[name])
        v = frame.inv().apply(r.apply(local) + p - centre)
        inside = (np.abs(v[:, 2]) < .004) & ((v[:, 0] / a) ** 2 + (v[:, 1] / b) ** 2 < 1)
        if inside.any():
            out[name] = dict(n=int(inside.sum()), local_mm=np.round(1000 * local[inside].mean(0)).tolist())
    return out


def boxes(task, world):
    out = {}
    for name in task.placement_vertices:
        p, r = pose(world["bodies"][name])
        v = r.apply(task.placement_vertices[name]) + p - [0, 0, task.table_height_m]
        out[name] = dict(lo=np.round(v.min(0), 4).tolist(), hi=np.round(v.max(0), 4).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance"); ap.add_argument("tier"); ap.add_argument("out")
    ap.add_argument("--lift", type=float, default=.32)
    ap.add_argument("--hold", type=float, default=6.)
    ap.add_argument("--shed", type=float, default=0., help="pitch the held shoe's eye end down by this angle (deg), hold, level again")
    ap.add_argument("--shed-lift", type=float, default=None, help="make the shed tilt at this lift (m), then lift the rest")
    ap.add_argument("--yaw", type=float, default=0., help="turn the held shoe about the vertical by this angle before the pitch, and back after it")
    ap.add_argument("--views", default=None)
    a = ap.parse_args()
    from observer import Observer
    from verdict import classify
    from rcb_topogym import Session
    from topogym.planning.harbor_bridge.actions import SessionPolicyRunner
    from topogym.planning.horseshoe_ring.kinematics import CheckedJointPolicy
    from topogym.planning.horseshoe_ring.manipulation import ManipulationPlanner
    from topogym.planning.horseshoe_ring.native_contacts import ContactHorseshoeIK
    from topogym.planning.horseshoe_ring.oracle import load_reference
    from topogym.planning.horseshoe_ring.stages import HangPolicy
    import tempfile
    recorded = json.loads((DATA / a.tier / "grasps.json").read_text())
    s = Session("horseshoe_ring", instance=a.instance, strict=True)
    watch = Observer(s)
    env = s.env
    reference, meta = load_reference(DATA / a.tier / "reference.json", env._task.assembly)
    planner = ManipulationPlanner(env, tempfile.mkdtemp(), ContactHorseshoeIK(env, puzzle_clearance_m=.0001, ordinary_limit_m=.001),
                                  reference=reference, fold_end_index=meta["fold_end_index"])
    # the oracle's own hang policy (stages.HangPolicy), waiting for quiet only
    policy = HangPolicy({"right": recorded["right"]["candidates"][0]}, part="pickup", planner=planner)
    policy.lift_m, policy.lift_tracking_m, policy.hang_wait_s, policy.pull_over = a.lift, .004, 60., True
    policy.shed_tilt_deg, policy.shed_hold_s, policy.shed_yaw_deg, policy.shed_lift_m = a.shed, 4., a.yaw, a.shed_lift
    error = [None]

    class Stop:
        def __getattr__(self, name):
            return getattr(inner, name)

        def act(self, observation):
            try:
                return inner.act(observation)
            except (RuntimeError, ValueError) as e:
                error[0] = str(e)[:400]
                return None
    inner = CheckedJointPolicy(policy, env._robot)
    t0 = time.time()
    s.record_start()
    SessionPolicyRunner(s).run_policy(request_id="hang", policy=Stop(), submission_hash="probe", state_fields=("world",))
    rows = []
    for k in range(int(a.hold / env.config.physics_dt_s)):
        s.step(s.hold_action())
        if k % 40 == 39:
            w = env._world.read_state()
            rows.append(dict(t=round((k + 1) * env.config.physics_dt_s, 2), ring=boxes(env._task, w)["target_ring"], through=through_ring(env._task, w)))
    world = env._world.read_state()
    obs = dict(state=dict(env.tick_record()["state"]), robot=env._robot.observe(), privileged=dict(world=world))
    pickle.dump(obs, open(a.out + ".end.pkl", "wb"))
    s.save_trajectory(s.record_stop(), a.out + ".npz")
    if a.views:
        import mujoco
        from PIL import Image
        m, d = env._world.model, env._world.data
        m.vis.headlight.ambient[:] = .55
        m.vis.headlight.diffuse[:] = .6
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        images = []
        for az, el in ((0, -8), (90, -8), (180, -8), (270, -8), (90, -45), (0, -89)):
            cam.lookat[:] = world["bodies"]["horseshoe_left"]["position_m"]
            cam.distance, cam.azimuth, cam.elevation = .6, az, el
            renderer.update_scene(d, cam)
            images.append(renderer.render().copy())
        Image.fromarray(np.concatenate([np.concatenate(images[:3], axis=1), np.concatenate(images[3:], axis=1)], axis=0)).save(a.views + ".png")
    rec = dict(id="hang_probe", instance=str(a.instance), rows=int(s.n_steps), control_steps=int(s.n_steps),
               observer=watch.summary(), fault=None, episode_end=None, success=False)
    v = classify(rec)
    result = dict(tier=a.tier, lift_m=a.lift, shed=dict(tilt_deg=a.shed, lift_m=a.shed_lift, yaw_deg=a.yaw), error=error[0], last_phase=policy.phase, rows=int(s.n_steps), wall_s=round(time.time() - t0, 1),
                  hands={k: (h["binding"]["target"][0] if h["binding"] else None) for k, h in world["grasp"]["hands"].items()},
                  boxes_over_table=boxes(env._task, world), through_ring=through_ring(env._task, world), during_hold=rows,
                  verdict={k: v[k] for k in ("verdict", "fails", "triggers", "notes", "speed", "ang")})
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)
    watch.uninstall()
    s.close()


if __name__ == "__main__":
    main()
