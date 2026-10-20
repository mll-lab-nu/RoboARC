"""Build MuJoCo instances of the redesigned rope_threading tiers (rcb_topogym make-instance in-process; rcb_topogym =
scripts/roboarc/shared on PYTHONPATH).

  register(tier)        the variation id (the records and profiles are registered by topogym.config.variation_records /
                        topogym.world.mujoco.profiles through topogym.tasks.rope_threading.redesign.registration)
  build(tier, out_dir)  the rope volume (the task's own generator) frozen as the instance's frozen asset, the birth = its
                        spawn on the table, contact_weld_v1 on every rope link, the simulator settings of the MuJoCo
                        rope_threading references (0.04 s controls, 64 substeps) -> <out_dir>/instance.json +
                        instance_assets/ (make-instance)

usage: python -m topogym.planning.rope_threading.redesign.session_build <tier> <out_dir> [--nop 25]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from topogym.planning.rope_threading.redesign import scenes as S

import os

# the simulator settings of the MuJoCo rope_threading references (rope-threading-*-i00 instances); grasp_assist's targets
# are made per tier (one per rope link)
SIM_SETTINGS = {"physics_dt_s": 0.04, "solver_substeps": 64, "control_decimation": 1, "reset_physics_steps": 0,
                "physics_engine": "mujoco"}
GRASP_SETTINGS = {"model": "contact_weld_v1", "stable_time_s": 0.02, "max_tcp_distance_m": 0.06, "max_penetration_m": 0.006,
                  "max_translation_error_m": 0.025, "max_rotation_error_rad": 0.15, "max_finger_opening_fraction": 0.15,
                  "intent_deadband": 0.002, "initial_intent": "open"}


def use_tmp():
    """Session temporary folders under RR_TMP when it is set (else the default temporary folder)."""
    import tempfile
    d = os.environ.get("RR_TMP")
    if d:
        Path(d).mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = d
        tempfile.tempdir = d
        return d
    return tempfile.gettempdir()
TABLE_Z = 0.9196429


def variation_id(tier):
    return f"rope_threading_rd_{tier}"


def register(tier):
    """The tier's variation id; its record and MuJoCo profile come from the product registration."""
    from topogym.config import variation_records
    vid = variation_id(tier)
    if not any(r["task_id"] == "rope_threading" and r["variation_id"] == vid for r in variation_records.RECORDS):
        raise KeyError(f"{vid} is not registered (topogym.config.variation_records)")
    return vid


def frozen_volume(tier, folder):
    """The task's own rope volume for the tier (+ elastic_rest = spawn, as rope_reference_shape 'initial' writes it)."""
    from topogym.assets.geometry.rope_volume import write_rope_volume
    from topogym.tasks.rope_threading.redesign.task import RedesignLayout, DEV_PARAMETERS
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    from topogym.tasks.rope_threading.redesign import frozen
    layout = RedesignLayout(frozen.load(tier))
    path = write_rope_volume(folder / "rope_volume.npz", layout.centerline(), S.ROPE_RADIUS_M, **DEV_PARAMETERS["rope_mesh"])
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    np.savez_compressed(path, **arrays, elastic_rest=arrays["spawn"])
    return Path(path)


def build(tier, out_dir):
    import argparse
    import rcb_topogym as rt
    from topogym.world.initial_state import save_numeric_archive
    out = Path(out_dir).resolve()
    if out.exists():
        shutil.rmtree(out)
    use_tmp()
    vid = register(tier)
    vol = frozen_volume(tier, out / "instance_assets" / "frozen_assets")
    with np.load(vol, allow_pickle=False) as data:
        spawn = data["spawn"].copy()
        st = data["rest"][data["surface_ring_indices"], 0].mean(axis=1)
    birth = spawn + np.array([0.0, 0.0, TABLE_Z])
    manifest = save_numeric_archive(out / "birth", {"fem": {"rope": np.asarray(birth, dtype=np.float64)}},
                                    dict(source="rope_threading redesign session_build", note=f"{tier} start"))
    # the frozen volume replaces whatever this process's TetGen writes in declare() (rcb_topogym's frozen-asset rule)
    sha = hashlib.sha256(vol.read_bytes()).hexdigest()
    rt._install_frozen_assets(out, {"rope_volume.npz": sha})
    L = float(st[-1] - st[0])
    n = max(2, int(round(L / 0.0055)))
    grasp = dict(GRASP_SETTINGS, targets=[dict(entity="rope", link=f"{k:03d}", center_m=[L / n / 2, 0., 0.], radius_m=.008)
                                          for k in range(n)])
    overrides = {**SIM_SETTINGS, "grasp_assist": grasp, "mujoco_birth": str(manifest)}
    kit = out / "_kit_inputs"
    kit.mkdir(exist_ok=True)
    (kit / "overrides.json").write_text(json.dumps(overrides))
    rt.cmd_make_instance(argparse.Namespace(task="rope_threading", eval_index=None, variation=vid, scene_seed=0,
                                            task_parameters=None, simulator_overrides=str(kit / "overrides.json"),
                                            ipc_settled_state=None, gripper_space="native", out=str(out / "instance.json")))
    inst = json.loads((out / "instance.json").read_text())
    scene = json.loads((out / "instance_assets" / "mujoco" / "scene.json").read_text())
    rope = scene["entities"]["rope"]["rope"]
    return dict(tier=tier, variation_id=vid, instance=str(out / "instance.json"), state_sha=inst.get("state_sha"),
                links=rope["links"], link_m=rope["link_m"], rope_length_m=L, birth=str(manifest),
                asset_identity_volume=inst.get("asset_identity", {}).get("rope_volume.npz") == sha,
                entities=sorted(scene["entities"]))


def nop_check(tier, instance, rows=25):
    """Open the frozen instance in a Session (as the graders do), hold the home pose for `rows` controls, evaluate."""
    import rcb_topogym as rt
    register(tier)
    s = rt.Session("rope_threading", instance=instance, strict=False)
    home = np.asarray(json.loads(Path(instance).read_text())["reset_joint_targets"], dtype=np.float32)
    t = []
    import time
    for i in range(rows):
        t0 = time.time()
        done = s.step(home)
        t.append(time.time() - t0)
        if done:
            break
    st = s.status()
    s.close()
    return dict(rows=rows, ms_per_control=round(1000 * float(np.mean(t[3:] if len(t) > 3 else t)), 1),
                success=st.get("success"), episode_end=st.get("episode_end"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tier")
    ap.add_argument("out")
    ap.add_argument("--nop", type=int, default=0)
    a = ap.parse_args()
    rep = build(a.tier, a.out)
    print(json.dumps(rep), flush=True)
    if a.nop:
        print(json.dumps(nop_check(a.tier, rep["instance"], a.nop)), flush=True)
