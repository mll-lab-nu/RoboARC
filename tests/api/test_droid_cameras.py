"""roboarc.vla.droid_cameras: DROID's camera poses on the one-arm puzzles. Without a simulator unless marked `sim`."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import roboarc
from roboarc.vla import droid_cameras as D
from roboarc.vla.droid_adapter import DroidAdapter
from roboarc.vla.stub_server import HoldPolicy

from test_vla import _serve

PROFILE = (Path(__file__).resolve().parents[2] / "source/topogym/config/profiles/camera_configs"
           / "panda_robotiq_observation.yaml")
MOUNT = [-0.57, 0.0, 0.9196429]          # the one-arm puzzles' robot base (world_tabletop_v1.yaml)


def scene(cameras=None, mount=MOUNT):
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    cams = cameras if cameras is not None else OmegaConf.to_container(OmegaConf.load(PROFILE).cameras)
    return OmegaConf.create({"env": {"cameras": cams, "robot_mount": {"pos": list(mount), "euler": [0.0, 0.0, 0.0]}}})


def test_poses_are_those_of_task_28s_camera_files():
    """The camera files of the diagnostic runs (progress report 28, rounded to 0.1 mm), from sim-evals' numbers."""
    s = scene()
    D.apply(s)
    c = s.env.cameras
    want = {"front_oblique": ([-0.52, -0.57, 1.5796], [-0.3317, -0.4232, 1.398], [0.4773, 0.3725, 0.7959]),
            "agentview": ([-0.52, 0.57, 1.5796], [-0.3317, 0.4232, 1.398], [0.4773, -0.3725, 0.7959])}
    for name, (pos, lookat, up) in want.items():
        assert np.allclose(c[name].pos, pos, atol=1e-4) and np.allclose(c[name].lookat, lookat, atol=1e-4)
        assert np.allclose(c[name].up, up, atol=1e-4) and c[name].fov == pytest.approx(71.51, abs=0.01)
        assert list(c[name].res) == [848, 480]
    a = c.wrist.attach
    assert a.enabled and a.link == "hand"
    assert np.allclose(a.offset_pos, [-0.074, 0.031, -0.0226], atol=1e-4)
    assert np.allclose(a.offset_lookat, [0.0199, 0.0333, 0.2623], atol=1e-4)
    assert np.allclose(a.offset_up, [-0.9496, -0.016, 0.313], atol=1e-4)
    assert c.wrist.fov == pytest.approx(56.74, abs=0.01)


def test_exterior_cameras_follow_the_robot_base():
    s = scene(mount=[-0.4, 0.1, 1.0])
    D.apply(s)
    assert np.allclose(s.env.cameras.front_oblique.pos, [-0.35, -0.47, 1.66])


def test_other_robots_are_refused():
    with pytest.raises(ValueError, match="wrist"):
        D.apply(scene({"cam_high": {}, "agentview": {}, "front_oblique": {}}))


def _camera_info(task: str, droid: bool) -> dict:
    code = ("import json, sys, roboarc\n"
            + ("from roboarc.vla import droid_cameras; droid_cameras.install()\n" if droid else "")
            + f"env = roboarc.make({task!r}, renderer='none')\n"
            "print(json.dumps({c: (v['cam2world_gl'], v['intrinsic_cv']) for c, v in env.unwrapped.camera_info().items()}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.sim
def test_installed_poses_reach_the_calibration_strictly():
    """In the graded runtime (strict): the calibration that RoboARC reports holds DROID's poses in the robot frame."""
    task = roboarc.task_info(os.environ.get("ROBOARC_TEST_DROID", "lock-picking-03")).id
    old, new = _camera_info(task, False), _camera_info(task, True)
    pos, quat = D.EXTERIOR["front_oblique"]
    T = np.asarray(new["front_oblique"][0])
    assert np.allclose(T[:3, 3], pos, atol=1e-6) and np.allclose(T[:3, :3], D._pose(pos, quat)[:3, :3], atol=1e-6)
    fy = np.asarray(new["front_oblique"][1])[1, 1]
    assert fy == pytest.approx(240 / np.tan(np.radians(D.EXTERIOR_FOV / 2)), rel=1e-6)
    assert not np.allclose(np.asarray(old["front_oblique"][0]), T)


@pytest.mark.sim
def test_physics_is_unchanged(tmp_path):
    """The runner with and without --camera-poses droid (rendering off, the adapter around HoldPolicy): the same birth
    and the same end state."""
    task = roboarc.task_info(os.environ.get("ROBOARC_TEST_DROID", "lock-picking-03")).id
    from roboarc.vla.run import main
    port, stop = _serve(DroidAdapter(HoldPolicy(horizon=15)))
    try:
        for run_id, extra in (("task", []), ("droid", ["--camera-poses", "droid"])):
            argv = ["--task", task, "--policy", f"ws://127.0.0.1:{port}", "--out", str(tmp_path), "--run-id", run_id,
                    "--renderer", "none", "--policy-hz", "15", "--max-steps", "200", *extra]
            assert main(argv) == 0
    finally:
        stop()
    res = {r: json.loads((tmp_path / task / r / "s0" / "e1" / "result.json").read_text()) for r in ("task", "droid")}
    assert res["task"]["infra_error"] == res["droid"]["infra_error"] == 0
    assert res["task"]["birth_hash"] == res["droid"]["birth_hash"] and res["task"]["live_sha"] == res["droid"]["live_sha"]
    assert json.loads((tmp_path / task / "droid" / "config.json").read_text())["camera_poses"] == "droid"
