#!/usr/bin/env python
"""Nyx video of a MuJoCo reference run (GPU; the image's Python with Genesis 1.2.1 + gs-nyx; not a CPU check).

A MuJoCo Session on the frozen instance replays the recorded actions with their recorded physics ticks, exactly as the
privileged grader replays them. Every k-th control step the state is rendered: through the render bridge into the
0.2.22 Nyx scene (--renderer host), or by nyx_direct without a Genesis host (--renderer direct, the renderer of the
families whose TASKS renderer is "direct"). The cameras are the observation rig at presentation resolution: same poses,
FOV and Nyx settings, the first camera at 1080 rows and the others in a column on its right, as in today's verifier
videos. At the end the final state hash must equal --expect-sha; otherwise the video is marked failed, never solved.

  python render_reference_video.py --instance I/instance.json --traj R/privileged.npz --out DIR \
      --label "interlocked_claws hard" --expect-sha 87201fbb45cd92cd [--renderer direct] [--max-seconds 45]

Writes DIR/video.mp4 (H.264 CRF 17, 30 fps), DIR/stills/*.png (birth, quarters, end), DIR/manifest.json (speed-up,
stride, frames, hashes, timings, per-frame task-contact depth and body speed for review).

No temporal aliasing (docs/simulator/RENDER_BRIDGE.md, "Reference videos"): the stride aims at a video of at most
--max-seconds at 30 fps, but never renders fewer than --frames-per-sim-second states per simulated second (default 8:
motions up to 4 Hz stay below Nyquist). A long reference makes a longer video rather than skipping more states.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FPS = 30
MAIN_ROWS = 1080


def label_patch(text, width, height=44):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except OSError:
        font = ImageFont.load_default(size=26)
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    w = int(draw.textlength(text, font=font)) + 24
    draw.rectangle([0, 0, min(w, width), height], fill=(0, 0, 0, 120))
    draw.text((12, 7), text, font=font, fill=(255, 255, 255, 235))
    return np.asarray(im)


def blend(frame, patch, x, y):
    h, w = patch.shape[:2]
    alpha = patch[..., 3:4].astype(np.float32) / 255.0
    region = frame[y:y + h, x:x + w].astype(np.float32)
    frame[y:y + h, x:x + w] = (region * (1 - alpha) + patch[..., :3] * alpha).astype(np.uint8)


def scaled_rig(scales):
    """Context: Env.reset builds its cameras at `res * scales[name]` (same pose, FOV, mount); restored afterwards."""
    import contextlib
    import topogym.config.scene as scene_module
    from omegaconf import open_dict
    original = scene_module.load_scene_config

    def load(config):
        cfg = original(config)
        with open_dict(cfg):
            for name, camera in cfg.env.cameras.items():
                w, h = (int(v) for v in camera.res)
                s = scales[name]
                camera.res = [int(round(w * s)), int(round(h * s))]
        return cfg

    @contextlib.contextmanager
    def patched():
        scene_module.load_scene_config = load
        try:
            yield
        finally:
            scene_module.load_scene_config = original
    return patched()


def task_review(world, data, model, body_ids, previous, dt):
    """Deepest task-object contact (mm, MuJoCo dist < 0) and fastest task body (m/s, between rendered frames)."""
    depth = 0.0
    for c in data.contact[:data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if (b1 in body_ids or b2 in body_ids) and c.dist < 0:
            depth = max(depth, -float(c.dist))
    pos = np.array([data.xpos[b] for b in sorted(body_ids)])
    speed = 0.0 if previous is None or dt <= 0 else float(np.linalg.norm(pos - previous, axis=1).max() / dt)
    return depth * 1000, speed, pos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--expect-sha", required=True)
    ap.add_argument("--max-seconds", type=float, default=45.0, help="the video length the stride aims at")
    ap.add_argument("--frames-per-sim-second", type=float, default=8.0,
                    help="at least this many rendered states per simulated second (no aliasing below half of it)")
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--renderer", choices=("host", "direct"), default="host")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "stills").mkdir()
    t_start = time.perf_counter()

    sys.path.insert(0, str(Path(__file__).resolve().parent / "shared"))
    from rcb_topogym import Session, load_trajectory
    import render_bridge
    inst = json.load(open(a.instance))
    s = Session(inst["task"], instance=a.instance, strict=True)
    world = s.env._world
    world.renderer_factory = None                 # the Session's own lazy render host stays unbuilt
    acts, ticks = load_trajectory(a.traj, s.task, s.action_dim, trusted=True, layout=s.action_layout)
    rows = len(acts)
    control_hz = float(s.control_hz)
    if a.frames_per_sim_second < 6.0:
        raise ValueError("--frames-per-sim-second below 6 aliases motions of 3 Hz (RENDER_BRIDGE.md, Reference videos)")
    stride = min(max(math.ceil(control_hz / FPS), math.ceil(rows / (FPS * a.max_seconds))),
                 max(1, math.floor(control_hz / a.frames_per_sim_second)))
    speedup = FPS * stride / control_hz

    from topogym.config.scene import load_scene_config
    names = list(load_scene_config(s.env.config).env.cameras)   # the rig, in observation order
    main_scale = MAIN_ROWS / 480.0
    side_scale = MAIN_ROWS / (480.0 * (len(names) - 1))
    scales = {n: (main_scale if i == 0 else side_scale) for i, n in enumerate(names)}
    t0 = time.perf_counter()
    if a.renderer == "host":
        with scaled_rig(scales):
            bridge = render_bridge.build(a.instance)
    else:
        # nyx_direct reads the rig from the world's scene configuration: a copy at presentation resolution
        import nyx_direct
        from omegaconf import OmegaConf, open_dict
        scaled = OmegaConf.create(OmegaConf.to_container(world.env_cfg, resolve=True))
        with open_dict(scaled):
            for name, camera in scaled.cameras.items():
                w, h = (int(v) for v in camera.res)
                camera.res = [int(round(w * scales[name])), int(round(h * scales[name]))]
        saved, world.env_cfg = world.env_cfg, scaled
        try:
            bridge = nyx_direct.NyxDirect.from_world(world, instance=a.instance)
        finally:
            world.env_cfg = saved
    build_s = time.perf_counter() - t0
    model, data = world.model, world.data
    # The task's moving bodies (every non-robot link with its own degrees of freedom; LinkView.idx is the MuJoCo body).
    task_bodies = {int(link.idx) for name, view in world.entities.items() if name != "robot"
                   for link in getattr(view, "links", []) if int(model.body_dofnum[int(link.idx)]) > 0}
    first = bridge.render()
    res = {n: list(first[n].shape[1::-1]) for n in names}
    width = res[names[0]][0] + (res[names[1]][0] if len(names) > 1 else 0)
    width += width % 2
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{MAIN_ROWS}", "-r", str(FPS),
         "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
         "-threads", "2", "-movflags", "+faststart", str(a.out / "video.mp4")], stdin=subprocess.PIPE)

    # stills on rendered frames only (birth, quarters on the stride grid, the last row), so frame timing stays uniform
    still_rows = sorted({0, rows} | {stride * round(rows * f / stride) for f in (.25, .5, .75)})
    review, previous, frames, render_s = [], None, 0, []

    def emit(step):
        nonlocal previous, frames
        t = time.perf_counter()
        bridge.update(model, data)
        rgb = bridge.render()
        render_s.append(time.perf_counter() - t)
        frame = np.zeros((MAIN_ROWS, width, 3), dtype=np.uint8)
        main = rgb[names[0]]
        frame[:main.shape[0], :main.shape[1]] = main
        y = 0
        for n in names[1:]:
            tile = rgb[n]
            frame[y:y + tile.shape[0], main.shape[1]:main.shape[1] + tile.shape[1]] = tile
            y += tile.shape[0]
        text = f"{a.label}  |  MuJoCo physics, Nyx render  |  {speedup:g}x  |  t {step / control_hz:6.2f} s"
        blend(frame, label_patch(text, main.shape[1]), 0, MAIN_ROWS - 44)
        ffmpeg.stdin.write(frame.tobytes())
        frames += 1
        depth, speed, previous = task_review(world, data, model, task_bodies, previous, stride / control_hz)
        review.append((step, round(depth, 4), round(speed, 4)))
        if step in still_rows:
            from PIL import Image
            Image.fromarray(frame).save(a.out / "stills" / f"step_{step:06d}.png")

    s.reset()
    emit(0)
    physics_s, done, k = 0.0, False, 0
    for k in range(1, rows + 1):
        t = time.perf_counter()
        done = s.step(acts[k - 1], _physics_steps=None if ticks is None else int(ticks[k - 1]))
        physics_s += time.perf_counter() - t
        if k % stride == 0 or k == rows or done:
            emit(k)
        if done:
            break
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    final_sha = s.state_sha()
    status = s.status()
    ok = final_sha == a.expect_sha
    sha256 = hashlib.sha256((a.out / "video.mp4").read_bytes()).hexdigest()
    depths = [d for _, d, _ in review]
    speeds = [v for _, _, v in review]
    manifest = dict(
        label=a.label, instance=a.instance, trajectory=a.traj,
        trajectory_sha256=hashlib.sha256(Path(a.traj).read_bytes()).hexdigest(),
        verdict="replay reproduces the recording" if ok else "FAILED: final state differs from the recording",
        final_state_sha=final_sha, expected_state_sha=a.expect_sha, replay_ok=ok,
        episode_end=status["episode_end"], success=bool(status["success"]), review_events=status["review"],
        rows=rows, rows_replayed=k, control_hz=control_hz, fps=FPS, stride_control_steps=stride, speedup=speedup,
        frames_per_sim_second=control_hz / stride, nyquist_hz=control_hz / (2 * stride),
        frames=frames, video_seconds=frames / FPS, sim_seconds=k / control_hz,
        cameras={n: res[n] for n in names}, layout=f"{names[0]} at {res[names[0]][0]}x{MAIN_ROWS}, "
        + ", ".join(names[1:]) + " stacked on its right", frame_size=[width, MAIN_ROWS],
        encoder=f"libx264 preset medium crf {a.crf} yuv420p", video_sha256=sha256,
        renderer="nyx_direct" if a.renderer == "direct" else "render bridge (Genesis host)",
        bridge_build_s=round(build_s, 1), physics_s=round(physics_s, 1),
        render_s_median=round(float(np.median(render_s)), 4), wall_s=round(time.perf_counter() - t_start, 1),
        review=dict(task_bodies=len(task_bodies), max_task_contact_depth_mm=max(depths),
                    max_task_body_speed_m_s=max(speeds), note="sampled at the rendered frames (every stride steps)"),
        per_frame=review)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k2: v for k2, v in manifest.items() if k2 != "per_frame"}), flush=True)
    bridge.close()
    s.close()
    if not ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
