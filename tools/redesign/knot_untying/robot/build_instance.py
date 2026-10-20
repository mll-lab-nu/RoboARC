"""Freeze a redesigned knot_untying tier as a robot-level MuJoCo instance (rcb_topogym make-instance, in-process).

usage (the cloud host, `source KU/work/env_robot.sh`): python build_instance.py <lab start dir> <tier> <out dir>
  <lab start dir>  the tier's object-level lab start: scene_start.json + start_state.npz (KU/out/tiers/<tier>_v6, easy_v7)
  <out dir>        gets instance.json + instance_assets/ (the frozen MuJoCo scene, the rope birth, the lab start) + build.json

What it does:
  1. rebuilds the lab (mjlab.Lab, as run_tier.py does), loads the start (velocities zeroed) and reads every rope's settled
     node centre line (world frame);
  2. the scene as task parameters (task_rd.py): the frame, the ropes (lab length, radius, anchor -> its clamp plate),
     the fixtures as in scene_start.json, the regions;
  3. the birth: each rope's own volume (a straight tube of its length, as the task declares it) swept along the lab
     centre line (session_kit.rope_birth_from_centerline) -> instance_assets/rope_birth/manifest.json;
  4. the grasp: contact_weld_v1 with a target on every link of every rope, both hands (the rope_threading port's
     grasp settings);
  5. make-instance (variation knot_untying_rd_<tier>, gripper_space "width"), then the birth path made relative, the
     generated volumes kept in instance_assets/frozen_assets/ and the lab start in instance_assets/lab_start/.
Needs the core patch knot_redesign_rope_attachments (AttachmentSpec on MuJoCo) in the runtime tree.
"""
import argparse, os
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

GRASP = dict(model="contact_weld_v1", stable_time_s=.02, max_tcp_distance_m=.06, max_penetration_m=.006,
             max_translation_error_m=.025, max_rotation_error_rad=.15, max_finger_opening_fraction=.15,
             intent_deadband=.002, initial_intent="open")      # the rope_threading port's (release_recipe overrides)
WELD_RADIUS_M = .008


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lab_start(start_dir):
    """(scene_start dict, {rope: settled nodes (n + 1, 3) world}, lab info) of a lab start."""
    from topogym.planning.knot_untying.redesign.mjlab import Lab
    start_dir = Path(start_dir)
    scene = json.loads((start_dir / "scene_start.json").read_text())
    lab = Lab(dict(ropes=[dict(name=r["name"], centerline=np.asarray(r["centerline"]), radius=r["radius"])
                          for r in scene["ropes"]], fixtures=scene.get("fixtures", [])))
    lab.load_state(start_dir / "start_state.npz", velocities=False)
    nodes = {name: rod.nodes(lab.d).copy() for name, rod in lab.rods.items()}
    info = dict(lengths_m={name: float(rod.s_nodes[-1]) for name, rod in lab.rods.items()},
                links={name: int(rod.n) for name, rod in lab.rods.items()}, crossings=len(lab.crossings()),
                goal=lab.goal_metrics(scene.get("regions") or None))
    return scene, nodes, info


def clamp_plate(scene, rope):
    """The clamp plate fixture of `rope`: clamp_<rope>_plate, else the clamp box nearest its anchored end."""
    fixtures = scene.get("fixtures", [])
    names = {f["name"]: f for f in fixtures}
    if f"clamp_{rope['name']}_plate" in names:
        return f"clamp_{rope['name']}_plate"
    end = np.asarray(rope["centerline"][0 if rope["anchor"].get("end", "start") == "start" else -1], float)
    plates = [f for f in fixtures if f.get("role") == "clamp" and "plate" in f["name"]]
    if not plates:
        raise ValueError(f"no clamp plate for rope {rope['name']}")
    return min(plates, key=lambda f: float(np.linalg.norm(np.asarray(f["pos"][:2]) - end[:2])))["name"]


def redesign_parameters(scene, tier, info, start_dir):
    fr = scene["frame"]
    ropes = []
    for r in scene["ropes"]:
        rec = dict(name=r["name"], radius_m=float(r["radius"]), length_m=info["lengths_m"][r["name"]])
        if r.get("anchor"):
            rec["anchor"] = dict(end=r["anchor"].get("end", "start"), links=int(r["anchor"].get("links", 2)),
                                 rigid=clamp_plate(scene, r))
        ropes.append(rec)
    keep = ("name", "role", "kind", "pos", "size", "radius", "height", "yaw_deg", "rgba")
    fixtures = [{k: f[k] for k in keep if k in f} for f in scene.get("fixtures", [])]
    start_dir = Path(start_dir)
    return dict(tier=tier, frame=dict(origin_world_xy=[float(v) for v in fr["origin_world_xy"]],
                                      table_top_z=float(fr["table_top_z"])),
                ropes=ropes, fixtures=fixtures, regions={k: [[float(x), float(y)] for x, y in v]
                                                         for k, v in (scene.get("regions") or {}).items()},
                fixture_friction=.3,
                # inst9 (main 10-11 21:50): cylinder fixtures (posts) become 32-facet convex prisms (<= 0.06 mm off the
                # cylinder): MuJoCo's cylinder-convex rim contact flipped its normal and let a finger slice 11 mm into a
                # post (easy_v8 w1_e10, the FB3 class). RD_CYLINDER_FIXTURES=cylinder keeps the old primitive.
                cylinder_fixtures=os.environ.get("RD_CYLINDER_FIXTURES", "cylinder"),   # inst10: core b2b751580 builds robot-touchable cylinders as 32-facet prisms itself
                source=dict(lab_start=str(start_dir.resolve()), design=scene.get("design"), notes=scene.get("notes"),
                            scene_start_sha256=sha256(start_dir / "scene_start.json"),
                            start_state_sha256=sha256(start_dir / "start_state.npz"),
                            crossings=info["crossings"], oracle=scene.get("oracle")))


def node_error(V, volume, nodes):
    """The chain nodes a birth gives (rope_chain.material_frames at the chain's stations) vs the lab nodes (m)."""
    from topogym.world.mujoco.rope_chain import _rings, _volume, material_frames
    vol = _volume(volume)
    st = vol["rest"][_rings(vol), 0].mean(axis=1)
    n = len(nodes) - 1
    born, _, _ = material_frames(V, vol, np.linspace(st[0], st[-1], n + 1))
    d = np.linalg.norm(born - nodes, axis=1)
    return dict(max_m=float(d.max()), mean_m=float(d.mean()), argmax=int(d.argmax()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("tier")
    ap.add_argument("out")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "instance_assets").mkdir(parents=True)
    from topogym.tasks.knot_untying.redesign import task_rd
    task_rd.register()
    import session_kit as sk
    from topogym.assets.geometry.rope_volume import write_rope_volume

    scene, nodes, info = lab_start(a.start)
    rd = redesign_parameters(scene, a.tier, info, a.start)
    # the lab start, kept with the instance (provenance; check_instance compares the birth with it)
    lab_dir = out / "instance_assets" / "lab_start"
    lab_dir.mkdir()
    for f in ("scene_start.json", "start_state.npz"):
        shutil.copy2(Path(a.start) / f, lab_dir / f)
    np.savez(lab_dir / "lab_nodes.npz", **{name: v for name, v in nodes.items()})

    # births: each rope's volume (generated as the task's declare() does) swept along the lab centre line
    work = out / "_build"
    work.mkdir()
    births, birth_check, vols = {}, {}, {}
    for r in rd["ropes"]:
        vol = work / task_rd.KnotUntyingRedesignTask.volume_name(r["name"])
        write_rope_volume(vol, np.array([[0., 0., 0.], [r["length_m"], 0., 0.]]), r["radius_m"])
        V, stretch = sk.rope_birth_from_centerline(vol, nodes[r["name"]])
        births[r["name"]] = V
        vols[r["name"]] = vol
        birth_check[r["name"]] = dict(node_error=node_error(V, vol, nodes[r["name"]]), **stretch,
                                      vertices=int(len(V)), lab_links=info["links"][r["name"]])
    manifest = sk.write_rope_birth(out / "instance_assets" / "rope_birth", births,
                                   note=f"knot_untying redesign {a.tier}: the lab start's settled chain ({a.start})")
    # the grasp: a weld target on every link of every rope (the scene's chain: round(L / 4.5 mm) links)
    targets = []
    for r in rd["ropes"]:
        n = max(2, int(round(r["length_m"] / task_rd.LINK_M)))
        targets += [dict(entity=r["name"], link=f"{k:03d}", center_m=[r["length_m"] / n / 2, 0., 0.],
                         radius_m=WELD_RADIUS_M) for k in range(n)]
    overrides = {"mujoco_birth": str(manifest), "grasp_assist": dict(GRASP, targets=targets)}
    t1 = time.time()
    import rcb_topogym as rt
    tp = work / "task_parameters.json"
    tp.write_text(json.dumps({"redesign": rd}))
    ov = work / "overrides.json"
    ov.write_text(json.dumps(overrides))
    args = argparse.Namespace(task="knot_untying", eval_index=None, variation=task_rd.variation_id(a.tier), scene_seed=0,
                              task_parameters=str(tp), simulator_overrides=str(ov), ipc_settled_state=None,
                              gripper_space="width", out=str(out / "instance.json"))
    rt.cmd_make_instance(args)
    make_s = time.time() - t1
    inst = json.loads((out / "instance.json").read_text())
    inst["simulator"]["mujoco_birth"] = "instance_assets/rope_birth/" + Path(manifest).name
    frozen = out / "instance_assets" / "frozen_assets"
    frozen.mkdir()
    for name, vol in vols.items():
        shutil.copy2(vol, frozen / vol.name)
        if inst["asset_identity"].get(vol.name) != sha256(vol):
            raise RuntimeError(f"the generated volume {vol.name} differs from the one the task declared")
    inst["redesign_build"] = dict(tool="tools/redesign/knot_untying/robot/build_instance.py",
                                  lab_start=rd["source"]["lab_start"], patch="knot_redesign_rope_attachments")
    (out / "instance.json").write_text(json.dumps(inst, indent=1))
    shutil.rmtree(work)
    # the born chain as frozen (qpos0 of the frozen scene) against the lab nodes
    import mujoco
    from topogym.world.mujoco.scene import load_scene
    model, man = load_scene(out / "instance_assets" / "mujoco")
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d)
    frozen_check = {}
    for name in nodes:
        rec = man["entities"][name]
        bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, l["body"]) for l in rec["links"]]
        with np.load(out / "instance_assets" / "mujoco" / rec["rope"]["file"]) as z:
            lengths = z["lengths"]
        P = d.xpos[bodies]
        born = np.vstack([P, P[-1] + d.xmat[bodies[-1]].reshape(3, 3)[:, 0] * lengths[-1]])
        err = np.linalg.norm(born - nodes[name], axis=1)
        frozen_check[name] = dict(links=int(rec["rope"]["links"]), link_m=rec["rope"]["link_m"],
                                  attachments=rec["rope"].get("attachments"), max_node_error_m=float(err.max()),
                                  mean_node_error_m=float(err.mean()), birth_chord_error_m=rec["rope"]["birth_chord_error_m"])
    res = dict(tier=a.tier, out=str(out), state_sha=inst["state_sha"], variation=inst["variation_id"],
               lab=dict(start=str(Path(a.start).resolve()), **{k: v for k, v in info.items() if k != "goal"}),
               birth_precheck=birth_check, frozen_birth=frozen_check, weld_targets=len(targets),
               make_instance_s=round(make_s, 1), total_s=round(time.time() - t0, 1))
    (out / "build.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
