"""roboarc.vla (docs/VLA.md): openpi's conventions on RoboARC's robots, the websocket protocol, the stub policy server and
the episode verdict. Without a simulator (stand-in robots built from the tasks' real robot specs) unless marked `sim`;
the protocol tests need msgpack and websockets (`pip install -e .[vla]`), the image tests Pillow. When openpi_client is
installed the bytes and the image resizing are compared with its own; when openpi's source is on the path
(OPENPI_SRC=<openpi>/src) its WebsocketPolicyServer serves this client."""
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import roboarc
from roboarc.vla import conventions as C

from conftest import sim_task

ALOHA_GRIP = {"command_space": "native", "closed_target": 0.002, "open_target": 0.037}       # robot_spec() of aloha-2
ROBOTIQ_GRIP = {"command_space": "native", "closed_target": 0.8, "open_target": 0.0}          # franka-robotiq-2f85


class FakeEnv:
    """The parts of a puzzle environment a convention reads, for a robot with these arms (J joints + a gripper each)."""

    def __init__(self, arms, joints, grip, layout_names=None, cameras=("cam_high", "cam_low", "cam_left_wrist",
                                                                      "cam_right_wrist"), moving=()):
        self.arms = list(arms)
        n = len(arms) * (joints + 1)
        self.action_layout = layout_names or [f"{a}/j{i}" if i < joints else f"{a}_native_gripper"
                                              for a in arms for i in range(joints + 1)]
        slices = {a: slice(k * (joints + 1), k * (joints + 1) + joints) for k, a in enumerate(arms)}
        grip_index = {a: k * (joints + 1) + joints for k, a in enumerate(arms)}
        robot = SimpleNamespace(arm_slice=slices, grip_index=grip_index)
        self.session = SimpleNamespace(robot=robot, robot_spec=lambda: {"grippers": {a: dict(grip) for a in arms}})
        self.proprio_groups = {"joint_pos": [0, n, ""]}
        self.task = SimpleNamespace(suite="puzzles", id="fake")
        self.control_hz = 40.0
        self.all_camera_names = list(cameras)
        self._moving = set(moving)

    @property
    def unwrapped(self):
        return self

    def camera_info(self):
        return {c: {"moves": c in self._moving} for c in self.all_camera_names}


def aloha_env():
    return FakeEnv(["left", "right"], 6, ALOHA_GRIP)


def droid_env():
    return FakeEnv(["arm"], 7, ROBOTIQ_GRIP, cameras=("agentview", "front_oblique", "wrist"), moving=("wrist",))


def _ref(task):
    rows = roboarc.load_reference(task, "standard")
    if rows is None:
        pytest.skip(f"{task} has no reference")
    return rows


# ------------------------------------------------------------------------------------------------------- conventions
def test_aloha_round_trip_is_exact_on_a_reference():
    """Rows -> openpi actions (gripper normalised) -> rows gives the reference's float32 rows bit for bit: a policy that
    returns a reference reaches its recorded end state (checked end to end in the sim test and docs/VLA.md)."""
    conv = C.AlohaConvention(aloha_env())
    rows = _ref("interlocked-claws-08")
    acts = conv.from_rows(rows.astype(np.float64))
    assert acts[:, [6, 13]].min() >= -1e-9 and acts[:, [6, 13]].max() <= 1 + 1e-9    # 0 = closed .. 1 = open
    assert np.array_equal(conv.to_rows(acts).astype(np.float32), rows)


def test_droid_round_trip_and_gripper_direction():
    conv = C.DroidConvention(droid_env())
    rows = _ref("hedgehog-in-cage-02")
    acts = conv.from_rows(rows.astype(np.float64))
    assert acts.shape == (len(rows), 8)
    assert np.array_equal(conv.to_rows(acts).astype(np.float32), rows)
    closed = np.zeros((1, 8)); closed[0, 7] = 1.0                              # DROID: 1 = closed
    assert conv.to_rows(closed)[0, 7] == pytest.approx(0.8)
    assert conv.cameras == {"observation/exterior_image_1_left": "front_oblique",
                            "observation/exterior_image_2_left": "agentview", "observation/wrist_image_left": "wrist"}


def test_aloha_observation_layout():
    env = aloha_env()
    conv = C.AlohaConvention(env)
    proprio = np.arange(14, dtype=np.float32) * 0.1
    proprio[[6, 13]] = [0.002, 0.037]                                           # measured gripper: closed, open
    rgb = {c: np.full((480, 848, 3), 200, np.uint8) for c in env.all_camera_names}
    o = conv.observation({"rgb": rgb, "proprio": proprio}, "do it")
    assert set(o) == {"state", "images", "prompt"} and o["prompt"] == "do it"
    assert o["state"].shape == (14,) and o["state"][6] == pytest.approx(0.0, abs=1e-6) and o["state"][13] == pytest.approx(1.0)
    assert np.allclose(o["state"][:6], proprio[:6])
    assert set(o["images"]) == set(C.ALOHA_CAMERAS)
    for img in o["images"].values():
        assert img.shape == (3, 224, 224) and img.dtype == np.uint8
        assert img[:, 0, :].max() == 0 and img[:, 112, :].min() == 200          # letterboxed: black bands, image inside


def test_droid_observation_layout():
    env = droid_env()
    conv = C.DroidConvention(env)
    proprio = np.zeros(8, np.float32); proprio[7] = 0.8                          # closed
    rgb = {c: np.zeros((480, 848, 3), np.uint8) for c in env.all_camera_names}
    o = conv.observation({"rgb": rgb, "proprio": proprio}, "p")
    assert o["observation/exterior_image_1_left"].shape == (224, 224, 3)       # DROID sends HWC
    assert o["observation/joint_position"].shape == (7,)
    assert o["observation/gripper_position"] == pytest.approx([1.0])


def test_one_action_per_control_step_by_default():
    conv = C.AlohaConvention(aloha_env())
    chunk = np.tile(np.linspace(0, 1, 30)[:, None], (1, 14))
    conv.begin_chunk(chunk, 25)
    rows = []
    while (r := conv.next_row(np.zeros(14))) is not None:
        rows.append(r)
    assert len(rows) == 25
    assert np.allclose([r[0] for r in rows], chunk[:25, 0])


def test_resampled_chunk_keeps_its_duration():
    """policy_hz=50 on a 40 Hz robot: 25 actions (0.5 s) take 20 control steps; joints interpolate, the gripper holds."""
    conv = C.AlohaConvention(aloha_env(), policy_hz=50.0)
    chunk = np.zeros((25, 14)); chunk[:, 0] = np.arange(25); chunk[:, 6] = (np.arange(25) >= 10)
    conv.begin_chunk(chunk, 25)
    rows = []
    while (r := conv.next_row(np.zeros(14))) is not None:
        rows.append(r)
    assert len(rows) == 20
    assert [r[0] for r in rows[:3]] == pytest.approx([0.0, 1.25, 2.5])
    k = 8                                                  # u = 10.0: action 10 (the gripper opens there)
    assert rows[k][6] == pytest.approx(0.037) and rows[k - 1][6] == pytest.approx(0.002)


def test_droid_velocity_integrates_from_the_measured_joints():
    conv = C.DroidConvention(droid_env(), velocity=True)
    assert conv.policy_hz == C.DROID_CONTROL_HZ
    chunk = np.zeros((8, 8)); chunk[:, 0] = 0.5; chunk[:, 1] = 3.0; chunk[:, 7] = 0.7
    conv.begin_chunk(chunk, 8)
    proprio = np.zeros(8); proprio[:7] = 0.1
    rows = []
    while (r := conv.next_row(proprio)) is not None:
        rows.append(r)
    assert len(rows) == math.ceil(8 * 40 / 15)                    # 8 DROID steps of 1/15 s at 40 Hz
    # clipped to [-1, 1] (openpi's client), then x 0.2 rad from the measured joints; gripper binarised (0.7 -> closed)
    assert rows[0][0] == pytest.approx(0.1 + 0.5 * 0.2) and rows[0][1] == pytest.approx(0.1 + 1.0 * 0.2)
    assert rows[0][7] == pytest.approx(0.8)
    assert np.array_equal(rows[0], rows[1])                       # held within one DROID step


def test_bad_chunks_are_refused():
    conv = C.AlohaConvention(aloha_env())
    with pytest.raises(ValueError):
        conv.begin_chunk(np.zeros((4, 7)), 4)
    with pytest.raises(ValueError):
        conv.begin_chunk(np.full((4, 14), np.nan), 4)
    conv.begin_chunk(np.zeros((4, 32)), 4)                        # padded model outputs: the first D dimensions


def test_prompts_are_short_task_sentences():
    tasks = roboarc.list_tasks()
    for t in tasks:
        p = C.task_prompt(t)
        assert p and p[-1] in ".!?" and len(p) <= 400 and "\n" not in p, (t.id, p)
        assert C.task_prompt(t, full=True).startswith(p)
    assert C.task_prompt(roboarc.task_info("interlocked-claws-08")) == \
        "Separate and place the entire key piece inside the green dashed goal region."


def test_resize_with_pad_matches_openpi():
    pytest.importorskip("PIL")
    img = (np.arange(480 * 848 * 3) % 251).astype(np.uint8).reshape(480, 848, 3)
    out = C.resize_with_pad(img, 224, 224)
    assert out.shape == (224, 224, 3)
    rh = int(480 / (848 / 224))
    top = int((224 - rh) / 2)
    assert out[:top].max() == 0 and out[top + rh:].max() == 0
    try:
        from openpi_client import image_tools
    except ImportError:
        return
    assert np.array_equal(out, image_tools.resize_with_pad(img, 224, 224))


# ----------------------------------------------------------------------------------------------------- the protocol
def test_msgpack_numpy_bytes():
    pytest.importorskip("msgpack")
    from roboarc.vla import wire
    obs = {"state": np.arange(14, dtype=np.float64), "images": {"cam_high": np.zeros((3, 224, 224), np.uint8)},
           "prompt": "x", "n": np.int64(3), "roboarc": {"control_step": 5}}
    back = wire.unpackb(wire.packb(obs))
    assert np.array_equal(back["state"], obs["state"]) and back["state"].dtype == np.float64
    assert back["images"]["cam_high"].shape == (3, 224, 224) and back["n"] == 3 and back["prompt"] == "x"
    try:
        from openpi_client import msgpack_numpy
    except ImportError:
        return
    assert msgpack_numpy.packb(obs) == wire.packb(obs)
    assert np.array_equal(msgpack_numpy.unpackb(wire.packb(obs))["state"], obs["state"])


def _serve(policy, metadata=None):
    pytest.importorskip("websockets")
    pytest.importorskip("msgpack")
    from roboarc.vla.stub_server import serve_in_thread
    return serve_in_thread(policy, metadata=metadata)


def test_stub_server_hold_and_replay():
    from roboarc.vla.stub_server import HoldPolicy, ReplayPolicy
    from roboarc.vla.wire import WebsocketClientPolicy
    port, stop = _serve(HoldPolicy(horizon=5), {"name": "hold"})
    try:
        client = WebsocketClientPolicy(f"ws://127.0.0.1:{port}", wait_s=10)
        assert client.get_server_metadata() == {"name": "hold"}
        out = client.infer({"state": np.arange(14.0), "prompt": "p"})
        assert out["actions"].shape == (5, 14) and np.array_equal(out["actions"][3], np.arange(14.0))
        assert "infer_ms" in out["server_timing"]
        out = client.infer({"observation/joint_position": np.ones(7), "observation/gripper_position": np.array([0.5])})
        assert out["actions"].shape == (5, 8) and out["actions"][0, 7] == 0.5
        client.close()
    finally:
        stop()
    acts = np.arange(30.0)[:, None] * np.ones((1, 8))
    port, stop = _serve(ReplayPolicy(acts, horizon=10))
    try:
        client = WebsocketClientPolicy(f"127.0.0.1:{port}", wait_s=10)
        a = client.infer({"roboarc": {"control_step": 25}})["actions"]
        assert a[:, 0].tolist() == [25, 26, 27, 28, 29, 29, 29, 29, 29, 29]       # past the end: the last action
        client.close()
    finally:
        stop()


def test_server_errors_reach_the_client():
    from roboarc.vla.wire import WebsocketClientPolicy

    class Broken:
        def infer(self, obs):
            raise KeyError("no such key")

    port, stop = _serve(Broken())
    try:
        client = WebsocketClientPolicy(f"ws://127.0.0.1:{port}", wait_s=10)
        with pytest.raises(RuntimeError, match="no such key"):
            client.infer({"state": np.zeros(3)})
    finally:
        stop()


def test_openpi_server_interop():
    """openpi's own WebsocketPolicyServer (OPENPI_SRC=<openpi>/src, with openpi_client installed) serves this client."""
    src = os.environ.get("OPENPI_SRC")
    if not src:
        pytest.skip("OPENPI_SRC is not set")
    sys.path.insert(0, src)
    server_mod = pytest.importorskip("openpi.serving.websocket_policy_server")
    import asyncio
    import threading
    import socket
    from roboarc.vla.stub_server import HoldPolicy
    from roboarc.vla.wire import WebsocketClientPolicy
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    server = server_mod.WebsocketPolicyServer(HoldPolicy(horizon=4), host="127.0.0.1", port=port,
                                              metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]})
    threading.Thread(target=lambda: asyncio.run(server.run()), daemon=True).start()
    client = WebsocketClientPolicy(f"ws://127.0.0.1:{port}", wait_s=20)
    assert client.get_server_metadata()["reset_pose"][1] == -1.5
    out = client.infer({"state": np.arange(14.0), "images": {}, "prompt": "p"})
    assert out["actions"].shape == (4, 14) and "infer_ms" in out["server_timing"]


# ------------------------------------------------------------------------------------------------------ the verdict
def test_verdict_is_the_standard_rule():
    from roboarc.vla.run import verdict
    live = {"live_success": True, "live_sha": "a", "infra_error": False, "review": None}
    assert verdict(live, {"replay_success": True, "replay_sha": "a"}, "puzzles")["success"] == 1
    v = verdict(live, {"replay_success": True, "replay_sha": "b"}, "puzzles")
    assert v["success"] == 1 and v["deterministic"] == 0                   # reported, not required (GPU families)
    assert verdict(live, {"replay_success": False, "replay_sha": "a"}, "puzzles")["success"] == 0
    v = verdict(live, None, "puzzles")
    assert v["success"] == 0 and v["grader_error"] == 1 and v["infra_error"] == 1
    v = verdict({**live, "review": {"count": 1}}, {"replay_success": True, "replay_sha": "a"}, "puzzles")
    assert v["success"] == 0 and v["success_pending_review"] == 1
    v = verdict(live, {"replay_success": True, "replay_sha": "a", "replay_episode_end": "simulator_instability"}, "puzzles")
    assert v["success"] == 0 and v["infra_error"] == 1
    assert verdict({**live, "live_success": False}, {"replay_success": True}, "puzzles")["success"] == 0
    assert verdict(live, None, "ikea")["success"] == 1


def test_parallel_writers_of_one_json_file(tmp_path):
    """Runners of one run id write config.json at once; in containers every runner's pid is the same (1), so the
    temporary names must not depend on it. Threads of one process share a pid too."""
    from concurrent.futures import ThreadPoolExecutor
    from roboarc.vla.run import _write_json
    path = tmp_path / "config.json"
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: [_write_json(path, {"writer": i, "n": n}) for n in range(50)], range(8)))
    assert json.loads(path.read_text())["n"] == 49 and [p.name for p in tmp_path.iterdir()] == ["config.json"]


def test_default_step_limit():
    from roboarc.vla.run import default_max_steps
    t = roboarc.task_info("interlocked-claws-08")
    assert default_max_steps(t) == 8600                  # 2 x 4277 rows, rounded up to 100
    assert default_max_steps(roboarc.task_info("fiber-bundling-05")) == 12000     # no reference


# ------------------------------------------------------------------------------------------- end to end (simulator)
@pytest.mark.sim
def test_episode_end_to_end(tmp_path):
    """The runner in fresh processes against the stub server (rendering off: black images): the reference served in
    chunks succeeds live and in the grader's replay and ends on the recorded hash; holding fails."""
    task = roboarc.task_info(sim_task())
    if task.suite != "puzzles" or task.reference is None:
        pytest.skip("the end-to-end test runs on a puzzle with a reference")
    from roboarc.vla.run import main
    from roboarc.vla.stub_server import HoldPolicy, ReplayPolicy
    import subprocess
    ref = tmp_path / "ref.npz"                  # in its own process: one environment per process
    assert subprocess.call([sys.executable, "-m", "roboarc.vla.run", "--task", task.id, "--export-reference", str(ref)]) == 0
    with np.load(ref) as z:
        acts = z["actions"]
    seeds = "0,1" if 1 in task.seeds else "0"         # seed 1: the same rows succeed on the appearance seed
    for name, policy, steps in (("replay", ReplayPolicy(acts, horizon=30), None), ("hold", HoldPolicy(30), 200)):
        port, stop = _serve(policy, {"version": {"model": name}})
        try:
            argv = ["--task", task.id, "--policy", f"ws://127.0.0.1:{port}", "--out", str(tmp_path / "out"),
                    "--run-id", name, "--renderer", "none"] + (["--max-steps", str(steps)] if steps else [])
            assert main(argv + ["--seed", seeds if name == "replay" else "0"]) == 0
        finally:
            stop()
        run = tmp_path / "out" / task.id / name
        res = json.loads((run / "s0" / "e1" / "result.json").read_text())
        assert json.loads((run / "config.json").read_text())["config_key"].startswith("vla:")
        assert res["server_metadata"] == {"version": {"model": name}}, res      # the served model's pinned versions
        if name == "replay":
            assert res["success"] == 1 and res["replay_success"] == 1 and res["deterministic"] == 1, res
            if task.reference_end_hash:
                assert res["live_sha"].startswith(task.reference_end_hash), res
            if seeds != "0":
                r1 = json.loads((run / "s1" / "e1" / "result.json").read_text())
                assert r1["success"] == 1 and r1["deterministic"] == 1 and r1["birth_hash"] != res["birth_hash"], r1
        else:
            assert res["success"] == 0 and res["steps"] == 200 and res["infra_error"] == 0, res
