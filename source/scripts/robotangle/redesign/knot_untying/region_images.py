#!/usr/bin/env python3
"""The region images of a knot_untying redesign region tier (task_rd.KnotUntyingRedesignTask.REGION_IMAGES; knot-robot
2026-10-14): the frozen instance's start scene as the agent's cameras cam_high and cam_low render it (the Session's
Nyx frames, robot at home), with every goal region's border drawn in as the placement tasks draw theirs in the scene
(topogym/assets/geometry/goal_region.py: a 2 mm wide line of 20 mm dashes and 10 mm gaps, colour (.1, .85, .2)), the
line's inner edge on the region's border: the containment limit the task's words name.

The redesign scene does not draw its regions, so the line is painted onto the rendered frame: every dash is a 2 mm x
20 mm quad lying 0.3 mm above the tabletop, projected with the camera's own calibration (Session.camera_info:
intrinsic_cv, extrinsic_cv) at 4 x 4 subpixels, and painted only where the rendered z-depth (Session.render_depth) shows
the tabletop there or nothing at all: a rope, fixture or arm in front of the line hides it. Nothing in the instance,
the scene or its physics changes; the frames show the start state after one hold row (see main()).

usage (in the release image, one GPU): python region_images.py <instance.json> <out dir> [--tooling <dir>]
  --tooling  the directory of the task's rcb_topogym.py (and render_bridge.py next to it); default /opt/tools
  writes <out dir>/region_cam_high.png, region_cam_low.png and region_images.json (what was drawn, per camera)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CAMERAS = ("cam_high", "cam_low")
LINE_M, DASH_M, GAP_M, LIFT_M = .002, .02, .01, .0003    # goal_region.write_goal_region's marker
COLOUR = np.array([.1, .85, .2]) * 255.
SS = 4                                                     # subpixels per pixel and axis
DEPTH_TOL_M = .002                                         # the line counts as seen when nothing is nearer than this


def dashes(poly):
    """The dash quads (k, 4, 2) of an axis-aligned rectangle's border, task frame: as goal_region.write_goal_region,
    the line's centre half a line width outside the border, so that its inner edge is the border."""
    p = np.asarray(poly, float)
    lo, hi = p.min(axis=0), p.max(axis=0)
    centre, half = (lo + hi) / 2, (hi - lo) / 2 + LINE_M / 2
    quads = []
    for axis in (0, 1):
        other = 1 - axis
        for sign in (-1, 1):
            start = -half[axis]
            while start < half[axis]:
                end = min(start + DASH_M, half[axis])
                a = np.zeros(2)
                b = np.zeros(2)
                a[axis], b[axis] = start, end
                a[other] = b[other] = sign * half[other]
                w = np.zeros(2)
                w[other] = LINE_M / 2
                quads.append(np.array([a - w, b - w, b + w, a + w]) + centre)
                start = end + GAP_M
    return np.array(quads)


def paint(rgb, depth, K, E, quads_world, table_z):
    """rgb (H, W, 3) uint8 with the dashes painted where they are seen; -> (image, record)."""
    H, W = depth.shape
    R, t = E[:, :3], E[:, 3]
    # the tabletop plane z = table_z + LIFT in camera coordinates: n_c . p = c
    n_c = R @ np.array([0., 0., 1.])
    c = float(n_c @ t + (table_z + LIFT_M))
    from PIL import Image, ImageDraw
    mask = Image.new("L", (W * SS, H * SS), 0)
    draw = ImageDraw.Draw(mask)
    drawn = 0
    for q in quads_world:
        P = np.c_[q, np.full(len(q), table_z + LIFT_M)]
        pc = P @ R.T + t
        if (pc[:, 2] <= .05).any():
            continue
        uv = (pc @ K.T)[:, :2] / pc[:, 2:3]
        draw.polygon([(float(u * SS), float(v * SS)) for u, v in uv], fill=255)
        drawn += 1
    cover = np.asarray(mask, np.float64).reshape(H, SS, W, SS).mean(axis=(1, 3)) / 255.
    # the plane's z-depth at every pixel centre: the ray K^-1 (u, v, 1) meets n_c . p = c at s = c / (n_c . d)
    u, v = np.meshgrid(np.arange(W) + .5, np.arange(H) + .5)
    d = np.linalg.solve(K, np.stack([u.ravel(), v.ravel(), np.ones(u.size)]))
    s = c / (n_c @ d)
    plane_z = (s * d[2]).reshape(H, W)
    seen = (depth <= 0) | (depth >= plane_z - DEPTH_TOL_M)
    alpha = cover * seen
    out = rgb.astype(np.float64) * (1 - alpha[..., None]) + COLOUR * alpha[..., None]
    rec = dict(dashes=int(len(quads_world)), dashes_in_front=drawn, line_px=round(float(cover.sum()), 1),
               hidden_px=round(float((cover * ~seen).sum()), 1))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("out", type=Path)
    ap.add_argument("--tooling", default="/opt/tools")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.tooling)
    from rcb_topogym import Session
    inst = json.loads(Path(a.instance).read_text())
    rd = inst["simulator"]["task_parameters"]["redesign"]
    regions = rd.get("regions") or {}
    if not regions:
        raise SystemExit(f"{inst['variation_id']}: no goal region (nothing to draw)")
    origin = np.asarray(rd["frame"]["origin_world_xy"], float)
    table_z = float(rd["frame"]["table_top_z"])
    quads = np.concatenate([dashes(poly) + origin for poly in regions.values()])
    s = Session(inst["task"], instance=a.instance)
    if s.n_steps:
        raise SystemExit("the Session is not at its start state")
    a.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    # A frame sampled before the first step can show the rope's declared rest shape (a straight tube) instead of its
    # birth: the renderer takes the chain state at its first sync, after that frame (seen 2026-10-14 on the medium
    # tier). One hold row (25 ms, the arms at home) brings the frames to the world's state.
    before = {cam: s.render_frame(cam) for cam in CAMERAS}
    s.step(s.hold_action())
    info = s.camera_info(list(CAMERAS))
    record = dict(instance=str(a.instance), variation_id=inst["variation_id"], state_sha=s.instance_sha,
                  frames="after one hold row (the start state, 25 ms later)", state_after_row=s.state_sha(),
                  regions={k: np.asarray(v, float).tolist() for k, v in regions.items()},
                  marker=dict(line_m=LINE_M, dash_m=DASH_M, gap_m=GAP_M, colour=[.1, .85, .2],
                              inner_edge="the region's border"), cameras={})
    for cam in CAMERAS:
        rgb = s.render_frame(cam)
        depth = s.render_depth(cam)
        K = np.asarray(info[cam]["intrinsic_cv"], float)
        E = np.asarray(info[cam]["extrinsic_cv"], float)
        img, rec = paint(rgb, depth, K, E, quads, table_z)
        name = f"region_{cam}.png"
        Image.fromarray(img).save(a.out / name, optimize=True)
        Image.fromarray(rgb).save(a.out / f"start_{cam}.png", optimize=True)   # the frame without the line (review)
        Image.fromarray(before[cam]).save(a.out / f"row0_{cam}.png", optimize=True)   # sampled before the first step
        record["cameras"][cam] = dict(file=name, image_size=list(img.shape[1::-1]), **rec)
    s.close()
    (a.out / "region_images.json").write_text(json.dumps(record, indent=1))
    print("REGION_IMAGES " + json.dumps(record["cameras"]), flush=True)


if __name__ == "__main__":
    main()
