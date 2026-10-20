"""horseshoe_ring hard on MuJoCo: try the hand swap after the support release without the 160 s before it (planning
tool, no reference).

A Session replays a recorded reference up to the row where the left hand has let the upper shoe go, has withdrawn and
the pair is quiet (hard's -i01 of 2026-10-10: row 6440), under the stress observer. From there the oracle's own
operation runs closed loop (HorseshoePolicy.drop_ring_by_hand_swap: the left hand takes the lower shoe, the right hand
opens and carries the ring on its fingers into the goal), with the parameters given here. Reported: the observer's
verdict (verdict.classify), each body's margin to the separator and box over the table, who holds what; the end
observation is saved next to the output (pickle), and MuJoCo raster views at a few phases (MUJOCO_GL=egl). The integral
feedback terms of the earlier stages start at zero here: confirm with the oracle.

  python swap_probe.py <instance.json> <reference.npz> <rows> <out.json> '<json: [{"remainder_shift_m": [0, 0.04, 0]}, ...]>'
"""
import json
import pickle
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[2] / "stress"), str(HERE.parents[3] / "scripts/roboarc/shared")]
import _env  # noqa: F401,E402
import numpy as np  # noqa: E402


class Stop:
    """This operation only: stop when its generator ends or raises."""
    def __init__(self, inner, snap=None):
        self.inner, self.error, self.snap, self.phase = inner, None, snap, None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def act(self, observation):
        try:
            action = self.inner.act(observation)
            if self.snap and self.inner.phase != self.phase:
                self.phase = self.inner.phase
                if self.phase in ("open_right_hand", "settle_ring_on_fingers", "ring_in_goal"):
                    self.snap(self.phase)
            return action
        except (RuntimeError, ValueError) as error:
            self.error = str(error)[:400]
            return None


def views(env, prefix):
    """Six MuJoCo raster views around the right shoe (pictures for the developer; needs MUJOCO_GL=egl and a GPU)."""
    try:
        import mujoco
        from PIL import Image
        m, d = env._world.model, env._world.data
        m.vis.headlight.ambient[:] = .55
        m.vis.headlight.diffuse[:] = .6
        renderer = mujoco.Renderer(m, 480, 640)
    except Exception as error:   # no GL here: the probe still runs
        print("no views:", str(error)[:120], flush=True)
        return None
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def snap(tag):
        images = []
        for az, el in ((0, -8), (90, -8), (180, -8), (270, -8), (90, -45), (0, -89)):
            cam.lookat[:] = env._world.read_state()["bodies"]["horseshoe_right"]["position_m"]
            cam.distance, cam.azimuth, cam.elevation = .55, az, el
            renderer.update_scene(d, cam)
            images.append(renderer.render().copy())
        Image.fromarray(np.concatenate([np.concatenate(images[:3], axis=1), np.concatenate(images[3:], axis=1)], axis=0)).save(f"{prefix}.{tag}.png")
    return snap


def one(instance, reference, rows, case, out, index):
    from observer import Observer
    from verdict import classify
    from rcb_topogym import Session, load_trajectory
    from topogym.planning.harbor_bridge.actions import SessionPolicyRunner
    from topogym.planning.horseshoe_ring.kinematics import CheckedJointPolicy
    from topogym.planning.horseshoe_ring.manipulation import ManipulationPlanner
    from topogym.planning.horseshoe_ring.native_contacts import ContactHorseshoeIK
    from topogym.planning.horseshoe_ring.stages import ContinuationPolicy
    from topogym.planning.rigid_reference.feedback import body_pose, relative_pose, tcp_pose
    from scipy.spatial.transform import Rotation
    s = Session("horseshoe_ring", instance=instance, strict=True)
    watch = Observer(s)
    acts, _ = load_trajectory(reference, "horseshoe_ring", s.action_dim, trusted=True, layout=s.action_layout)
    t0 = time.time()
    s.record_start()
    for row in acts[:rows]:
        s.step(row)
    env = s.env
    world, robot = env._world.read_state(), env._robot.observe()
    held = {"left": "horseshoe_left", "right": "horseshoe_right"}
    tracker = dict(grip={"left": .027, "right": .002}, bias={k: np.zeros(3) for k in held}, angle_bias={k: np.zeros(3) for k in held},
                   held_entities=dict(held), velocity_feedback_state={},
                   offsets={"right": relative_pose(body_pose(world["bodies"]["horseshoe_right"]), tcp_pose(robot["right"]))})
    planner = ManipulationPlanner(env, tempfile.mkdtemp(), ContactHorseshoeIK(env, ordinary_limit_m=.001))
    policy = ContinuationPolicy({"left": {}, "right": {}}, operation="drop_ring_by_hand_swap", tracker=tracker, planner=planner)
    policy.hand_swap_parameters = dict(case)
    wrapped = Stop(CheckedJointPolicy(policy, env._robot), views(env, f"{out}.case{index}"))
    SessionPolicyRunner(s).run_policy(request_id="swap", policy=wrapped, submission_hash="probe", state_fields=("world",))
    for _ in range(80):           # 2 s more
        s.step(s.hold_action())
    task, world = env._task, env._world.read_state()
    bodies = world["bodies"]; region = task.goal_region; boxes = {}
    for name in task.placement_vertices:
        p, q = np.asarray(bodies[name]["position_m"]), bodies[name]["quaternion_wxyz"]
        v = Rotation.from_quat([q[1], q[2], q[3], q[0]]).apply(task.placement_vertices[name]) + p
        boxes[name] = dict(lo=np.round(v.min(0) - [0, 0, task.table_height_m], 4).tolist(), hi=np.round(v.max(0) - [0, 0, task.table_height_m], 4).tolist(),
                           separator_margin_m=round(float(region.separator_offset_m - (v[:, :2] @ np.asarray(region.separator_normal_xy)).max()), 4))
    obs = dict(state=dict(env.tick_record()["state"]), robot=env._robot.observe(), privileged=dict(world=world))
    pickle.dump(obs, open(f"{out}.end{index}.pkl", "wb"))
    s.save_trajectory(s.record_stop(), f"{out}.case{index}.npz")
    if wrapped.snap:
        wrapped.snap("end")
    rec = dict(id="swap_probe", instance=str(instance), rows=int(s.n_steps), control_steps=int(s.n_steps),
               observer=watch.summary(), fault=None, episode_end=None, success=False)
    verdict = classify(rec)
    result = dict(case=case, error=wrapped.error, last_phase=policy.phase, rows=int(s.n_steps), wall_s=round(time.time() - t0, 1),
                  boxes_over_table=boxes,
                  remainder_on_system_side=all(b["separator_margin_m"] >= 0 for n, b in boxes.items() if n != "target_ring"),
                  hands={k: (v["binding"]["target"][0] if v["binding"] else None) for k, v in world["grasp"]["hands"].items()},
                  verdict={k: verdict[k] for k in ("verdict", "fails", "triggers", "notes", "speed", "ang")},
                  peaks={k: (v.get("value") if isinstance(v, dict) else v) for k, v in (rec["observer"].get("peaks") or {}).items()
                         if k in ("task_speed", "task_angular", "task_tip_speed", "force_task_static", "force_task_task", "depth_task_task", "depth_task_static")})
    watch.uninstall()
    s.close()
    return result


def main():
    instance, reference, rows, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    results = []
    for index, case in enumerate(json.loads(sys.argv[5])):
        results.append(one(instance, reference, rows, case, out, index))
        print(json.dumps(results[-1]), flush=True)
        out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
