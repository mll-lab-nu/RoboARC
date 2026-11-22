"""Replay a task's reference trajectory (the privileged mode's solution) through the API and check that it ends on the
state hash the release recorded for it; optionally write a video of the cameras.

    python examples/replay_reference.py roboarc-orbits-02-i01 --renderer none
    python examples/replay_reference.py roboarc-cast-coil-06-i01 --renderer nyx --video /tmp/cast_coil.mp4 --every 4
"""
import argparse
import time

import roboarc          # before numpy
from roboarc.references import hash_matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--renderer", default="none", choices=("nyx", "mujoco", "none"))
    ap.add_argument("--video", help="write an mp4 of env.render() (needs a renderer and imageio)")
    ap.add_argument("--every", type=int, default=4, help="video: one frame per this many control steps")
    a = ap.parse_args()
    actions = roboarc.load_reference(a.task)
    if actions is None:
        raise SystemExit(f"{a.task} has no reference trajectory")
    env = roboarc.make(a.task, renderer=a.renderer, render_mode="rgb_array" if a.video else None)
    obs, info = env.reset()
    writer = None
    if a.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(a.video, fps=max(1, int(env.control_hz / a.every)))
    t0 = time.time()
    for i, row in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(row)
        if writer is not None and i % a.every == 0:
            writer.append_data(env.render())
        if (terminated and env.task.suite == "puzzles") or truncated:
            break
    if writer is not None:
        writer.close()
    end, expected = env.state_sha(), env.task.reference_end_hash
    print(f"{env.task.id}: {i + 1} of {len(actions)} rows in {time.time() - t0:.0f} s, episode_end {info.get('episode_end')}, "
          f"end hash {end[:16]} (recorded {expected}): {'MATCH' if hash_matches(end, expected) else 'DIFFERENT'}")
    env.close()


if __name__ == "__main__":
    main()
