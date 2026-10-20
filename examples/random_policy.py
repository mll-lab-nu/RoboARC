"""A random policy: small random joint moves around the current targets (a random walk), the cheapest smoke test of
an environment. Run inside the task's image (see docs/API.md, "Running"):

    python examples/random_policy.py roboarc-interlocked-claws-03-i01 --steps 200 --renderer none
"""
import argparse

import roboarc          # before numpy: the graded numpy settings
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--renderer", default="none", choices=("nyx", "mujoco", "none"))
    ap.add_argument("--scale", type=float, default=0.02, help="joint noise per step (rad; gripper: its own units)")
    a = ap.parse_args()
    env = roboarc.make(a.task, renderer=a.renderer)
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=0)
    print("task", env.task.id, "birth", info["birth_hash"], "actions", env.action_space.shape, "cameras", env.camera_names)
    start, n, _ = env.proprio_groups["joint_cmd"]                 # the targets being tracked (the action layout)
    target = obs["proprio"][start:start + n].astype(np.float64)
    low, high = env.action_space.low.astype(np.float64), env.action_space.high.astype(np.float64)
    span = np.where(np.isfinite(high - low), high - low, 1.0)
    for t in range(a.steps):
        target = np.clip(target + rng.normal(0.0, a.scale, target.shape) * np.minimum(span, 1.0), low, high)
        obs, reward, terminated, truncated, info = env.step(target)
        target = np.asarray(info["applied_action"], dtype=np.float64)    # what the rules let through
        if terminated or truncated:
            print("episode end at step", t + 1, info.get("episode_end"))
            break
    print("steps", info["control_step"], "sim time", round(info["sim_time_s"], 3), "s; success", info["success"],
          "; state", env.state_sha())
    env.close()


if __name__ == "__main__":
    main()
