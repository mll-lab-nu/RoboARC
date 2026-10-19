"""Preview sheets of the fiber_bundling redesign tiers: CPU-only offscreen renders (the soft-body redesign tooling's
previews.render, EGL on Mesa's software device), framed on the cable and its fixtures. docs/rcb/redesign/
fiber_bundling.md. The start state by default; with --actions, the states after the given rows of a stream, replayed
in the production Session (run it through run_registered.py then).

usage: preview.py <instance.json> <out_dir> --prefix <tier> [--title T]   -> <prefix>_<view>.png, <prefix>_sheet.png
       preview.py <instance.json> <out_dir> --prefix <name> --actions rows.npz --rows N [N ...]
                                                                       -> <prefix>_r<N>_<view>.png, <prefix>_r<N>_sheet.png
"""
import argparse

from previews.render import setup_cpu_gl

# Views of the fiber scene (the cable axis at z 1.16, the robot at x -0.57 facing +x): from the operator's side, from
# behind the robot, from above, and a close-up of the ring with the strand fan.
VIEWS = {
    "front": dict(azimuth=90.0, elevation=-22.0, distance=0.62),
    "robot": dict(azimuth=0.0, elevation=-30.0, distance=0.70),
    "top": dict(azimuth=90.0, elevation=-89.0, distance=0.60),
    "ring": dict(azimuth=150.0, elevation=-18.0, distance=0.30),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("instance"); ap.add_argument("out")
    ap.add_argument("--prefix", required=True); ap.add_argument("--title", default=None)
    ap.add_argument("--actions", help="a stream (rows.npz): render the states after --rows instead of the start")
    ap.add_argument("--rows", type=int, nargs="+", default=[])
    ap.add_argument("--hide-robot", action="store_true", help="hide the arm and gripper (the route seen from the top)")
    a = ap.parse_args(argv)
    setup_cpu_gl()
    import numpy as np
    import mujoco
    from previews.render import load_instance_state, render_state, save_views
    if not a.actions:
        model, data, info = load_instance_state(a.instance)
        sheet(model, data, a.out, a.prefix, a.title or a.prefix, a.hide_robot)
        return
    from rcb_topogym import Session
    s = Session("fiber_bundling", instance=a.instance, strict=True)
    w = s.env._world
    rows = np.asarray(np.load(a.actions)["actions"], np.float32)
    marks = sorted(set(min(r, len(rows) - 1) for r in a.rows))
    for n, row in enumerate(rows[:marks[-1] + 1]):
        ended = s.step(row)
        if n in marks or ended:
            sheet(w.model, w.data, a.out, f"{a.prefix}_r{n}", f"{a.title or a.prefix}: after row {n}", a.hide_robot)
        if ended:
            break


def sheet(model, data, out, prefix, title, hide_robot=False):
    import copy
    import numpy as np
    import mujoco
    from previews.render import render_state, save_views
    data = copy.copy(data)                       # a live Session's data stays untouched (its solver warm start too)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
    # The cable's capsules are its only geometry and sit in MuJoCo's collision group 3, hidden by default: show them.
    cable = {b for b, n in enumerate(names) if n.startswith("fiber_bundle::")}
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in cable:
            model.geom_group[g] = 2
    if hide_robot:                               # the arm's subtree into group 5, which the renderer does not show
        root = names.index("robot")
        def in_robot(b):
            while b > 0:
                if b == root:
                    return True
                b = int(model.body_parentid[b])
            return False
        for g in range(model.ngeom):
            if in_robot(int(model.geom_bodyid[g])):
                model.geom_group[g] = 5
    mujoco.mj_forward(model, data)
    def body(prefix):
        return np.asarray(data.xpos[next(b for b, n in enumerate(names) if n.startswith(prefix))])
    ring, cradle = body("bundling_ring"), body("cable_cradle_pad")
    centre = tuple(float(x) for x in 0.5 * (ring + cradle))
    imgs = {}
    for name, view in VIEWS.items():
        look = tuple(float(x) for x in ring) if name == "ring" else centre
        imgs.update({name: render_state(model, data, views=(view,), lookat=look)["view0"]})
    for path in save_views(imgs, out, prefix, title=title):
        print(path, flush=True)


if __name__ == "__main__":
    main()
