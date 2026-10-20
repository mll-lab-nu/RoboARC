#!/usr/bin/env python3
"""The target images of a blanket_fold instance (BlanketFoldTask.fold_goal_words: "the finished fold rendered in this
scene, robot at home, from the agent's own cameras"): the frozen fold target's particles (fold_target_path) put on the
Session's cloth, rendered by the Session's own Nyx frames from cam_high and cam_low.

The cloth is a VBD grid whose particles are fixed barycentric combinations of its vertices, and every vertex is one
particle (the cloth file's `node`), so the target particles give the vertices exactly: vertices = target[node]. They are
written into the cloth's state (ClothVBD.set_vertices, at rest); nothing is stepped and nothing else changes: the robot
is at its home pose of the frozen start. The world's render version is bumped and the sensor cache cleared, so the next
frame shows the new cloth state.

The script that rendered the released pictures (blanket-fold 01-05, 2026-10-13) was not kept; this one was written from
the task code on 2026-10-20 and checked against them: on blanket-fold-01-i01 (image roboarc:1.0.0, the task's tooling
and frozen instance, an RTX 4090) the frames differ from the released PNGs by 0.012 / 0.006 grey levels on average
(cam_high / cam_low), 0.006 % / 0.004 % of the pixels by more than 8 levels: Nyx's run-to-run spread.

usage (in the release image, one GPU): python fold_target_images.py <instance.json> <out dir> [--tooling <dir>]
                                       [--not-strict]
  --tooling     the directory of the task's rcb_topogym.py (and render_bridge.py, nyx_direct.py next to it); default
                /opt/tools
  --not-strict  Session(strict=False): for a stand-in instance whose looks were changed by hand (no binding beside it)
  writes <out dir>/target_cam_high.png, target_cam_low.png, start_cam_*.png (the frozen start, for review) and
  target_images.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CAMERAS = ("cam_high", "cam_low")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("out", type=Path)
    ap.add_argument("--tooling", default="/opt/tools")
    ap.add_argument("--not-strict", action="store_true",
                    help="build the Session with strict=False (an instance copy whose looks were changed)")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.tooling)
    from rcb_topogym import Session
    from PIL import Image
    inst = json.loads(Path(a.instance).read_text())
    s = Session(inst["task"], instance=a.instance, **({"strict": False} if a.not_strict else {}))
    if s.n_steps:
        raise SystemExit("the Session is not at its start state")
    a.out.mkdir(parents=True, exist_ok=True)
    world, task = s.env._world, s.env._task
    start = {cam: s.render_frame(cam) for cam in CAMERAS}
    (cloth_name, cloth), = world._cloths.items()
    target = np.asarray(task.fold_target_particles, float)
    vertices = target[cloth.node_particles]
    error = float(np.abs(cloth.particles_of(vertices) - target).max())
    if error > 1e-9:
        raise SystemExit(f"{cloth_name}: the target particles are not the particle view of target[node] ({error:g} m)")
    cloth.set_vertices(vertices)
    world._version += 1             # the renderer shows the new state at its next sync (World.sync_renderer)
    s.env._sensors._key = None      # and the sensor suite renders again (SensorSuite.sample caches by state)
    cfg = task.config
    record = dict(instance=str(a.instance), variation_id=inst["variation_id"], state_sha=s.instance_sha,
                  fold_target_sha256=cfg["fold_target_sha256"],
                  cloth=dict(entity=cloth_name, vertices=len(vertices), particles=len(target),
                             particle_view_error_m=error),
                  appearance=dict(color=cfg.get("color"), texture=cfg.get("texture")),
                  frames="the frozen start with the cloth set to the fold target at rest, nothing stepped", cameras={})
    for cam in CAMERAS:
        img = s.render_frame(cam)
        name = f"target_{cam}.png"
        Image.fromarray(img).save(a.out / name, optimize=True)
        Image.fromarray(start[cam]).save(a.out / f"start_{cam}.png", optimize=True)
        record["cameras"][cam] = dict(file=name, image_size=list(img.shape[1::-1]),
                                      rgb_sha256=hashlib.sha256(img.tobytes()).hexdigest())
    s.close()
    (a.out / "target_images.json").write_text(json.dumps(record, indent=1))
    print("TARGET_IMAGES " + json.dumps(record["cameras"]), flush=True)


if __name__ == "__main__":
    main()
