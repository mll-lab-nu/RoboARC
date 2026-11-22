"""Where a VLA (vision-language-action) policy plugs in: camera images + proprioception + the task's instruction in,
a chunk of absolute joint targets out, executed one control step at a time. Replace `MyVLA` with your model.

    python examples/vla_loop.py roboarc-cast-keyhole-02-i01 --renderer nyx --image-size 224 224

Conventions (docs/API.md): obs["rgb"][camera] is (H, W, 3) uint8; obs["proprio"] follows env.proprio_layout (the
standard mode's layout: joint positions, TCP poses in the robot frame, the commanded targets, velocities, gripper
openings, efforts); an action is env.action_layout: per arm its joint targets (rad) then its gripper command. The
environment applies the task's speed rules (clamp, never fail), so a policy may ask for any target.
"""
import argparse

import roboarc          # before numpy
import numpy as np


class MyVLA:
    """A stand-in policy: it holds the current joint targets. A real VLA maps (images, proprio, instruction) to a chunk
    of future actions; `chunk` rows are executed open-loop before the next call (action chunking)."""

    def __init__(self, action_dim: int, chunk: int = 8):
        self.action_dim, self.chunk = action_dim, chunk

    def __call__(self, images: dict, proprio: np.ndarray, instruction: str, cmd: np.ndarray) -> np.ndarray:
        # images: {camera: (H, W, 3) uint8}; proprio: (P,) float32; returns (chunk, action_dim)
        return np.repeat(cmd[None, :], self.chunk, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--renderer", default="nyx", choices=("nyx", "mujoco"))
    ap.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"))
    ap.add_argument("--cameras", nargs="*", help="default: all of the task's cameras")
    ap.add_argument("--max-steps", type=int, default=400)
    a = ap.parse_args()
    env = roboarc.make(a.task, renderer=a.renderer, cameras=a.cameras, image_size=a.image_size)
    obs, info = env.reset()
    policy = MyVLA(env.action_space.shape[0])
    g = env.proprio_groups["joint_cmd"]                      # [start, count, meaning]
    print(env.task.id, "|", env.instruction[:200], "...")
    steps = 0
    while steps < a.max_steps:
        cmd = obs["proprio"][g[0]:g[0] + g[1]]
        chunk = policy(obs["rgb"], obs["proprio"], env.instruction, cmd)
        for row in chunk:
            obs, reward, terminated, truncated, info = env.step(row)
            steps += 1
            if terminated or truncated or steps >= a.max_steps:
                break
        if terminated or truncated:
            break
    print("steps", steps, "success", info["success"], "episode_end", info.get("episode_end"))
    env.close()


if __name__ == "__main__":
    main()
