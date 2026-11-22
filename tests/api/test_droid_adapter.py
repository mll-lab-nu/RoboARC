"""roboarc.vla.droid_adapter: DROID's wrist convention and start pose around a DROID-trained policy, and the models'
own inputs and outputs (droid_models.py, without their libraries). Without a simulator unless marked `sim` (the runner
against the adapter, rendering off)."""
import importlib.util
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import roboarc
from roboarc import _runtime
from roboarc.vla import conventions as C
from roboarc.vla.droid_adapter import DROID_RESET, JOINTS, WRIST_OFFSET, DroidAdapter
from roboarc.vla.droid_models import (IMAGES, OpenpiModel, cosmos3_actions, cosmos3_request, droid_eef_9d,
                                      groot_actions, groot_request, molmoact2_actions, molmoact2_inputs)
from roboarc.vla.stub_server import HoldPolicy

from conftest import puzzle_image_only
from test_vla import _serve, droid_env

HOME = np.array([0, np.pi / 8, 0, -5 * np.pi / 8, 0, 3 * np.pi / 4, -np.pi / 4])   # Franka puzzles' start, RoboARC's q7
RESET = DROID_RESET + np.eye(7)[6] * WRIST_OFFSET                                    # DROID's reset in RoboARC's q7


def obs(q, control_step):
    return {JOINTS: np.asarray(q, dtype=np.float64), "observation/gripper_position": np.array([0.0]), "prompt": "p",
            "roboarc": {"control_step": control_step}}


class Recorder(HoldPolicy):
    """HoldPolicy that keeps every observation it is asked with."""

    def __init__(self):
        super().__init__(horizon=4)
        self.seen = []

    def infer(self, o):
        self.seen.append(o)
        return super().infer(o)


class Refuse:
    def infer(self, o):
        raise AssertionError("the policy was asked while the arm moves to DROID's reset")


def test_roboarc_droid_convention_passes_q7_as_measured():
    """The adapter's wrist offset assumes RoboARC's droid convention passes q7 through unchanged; if RoboARC ever maps
    the wrist itself, this fails and the adapter's offset must go."""
    conv = C.DroidConvention(droid_env())
    proprio = np.zeros(8)
    proprio[:7] = HOME
    assert np.allclose(conv.observation({"proprio": proprio}, "p")[JOINTS], HOME)
    assert np.allclose(conv.to_rows(np.r_[HOME, 0.0][None, :])[0, :7], HOME)


def test_policy_sees_droid_wrist_and_actions_map_back():
    rec = Recorder()
    out = DroidAdapter(rec).infer(obs(HOME, 100))                     # 2.5 s at 40 Hz: the move is over
    seen = rec.seen[0]
    assert "roboarc" not in seen
    assert seen[JOINTS][6] == pytest.approx(-np.pi / 2)              # DROID's reading of RoboARC's start wrist
    assert np.allclose(out["actions"][:, :7], HOME)                  # holding maps back to RoboARC's joints exactly


def test_start_moves_on_a_line_toward_droid_reset_without_the_policy():
    a = DroidAdapter(Refuse()).infer(obs(HOME, 0))["actions"]
    assert a.shape == (15, 8) and np.all(a[:, 7] == 0.0)             # gripper open
    path = np.vstack([HOME, a[:, :7]])
    assert np.allclose(np.abs(np.diff(path, axis=0)).max(axis=1), 1.0 / 15)   # 1 rad/s at 15 Hz on the farthest joint
    moving = RESET != HOME
    frac = (a[:, :7] - HOME)[:, moving] / (RESET - HOME)[moving]
    assert np.allclose(frac, frac[:, :1])                             # every joint at the same fraction: a line


def test_move_reaches_droid_reset_then_the_policy_takes_over():
    """Through RoboARC's own chunk player at 15 Hz with horizon 8, the joints reaching every row."""
    conv = C.DroidConvention(droid_env(), policy_hz=15.0)
    rec = Recorder()
    adapter = DroidAdapter(rec)
    q, step = HOME.copy(), 0
    while step < 100:
        conv.begin_chunk(adapter.infer(obs(q, step))["actions"], 8)
        while (row := conv.next_row(np.zeros(8))) is not None:
            q, step = row[:7], step + 1
    assert np.allclose(q, RESET) and rec.seen == []
    adapter.infer(obs(q, step))
    assert len(rec.seen) == 1


def test_velocity_move_reaches_droid_reset_through_the_velocity_convention():
    """droid-velocity: the move is sent as DROID's velocities, integrated by RoboARC's own player from the measured
    joints (here the joints reach every target); the policy's velocities keep q7 as they are (a constant offset)."""
    conv = C.DroidConvention(droid_env(), velocity=True)
    rec = Recorder()
    adapter = DroidAdapter(rec, velocity=True)
    q, step = HOME.copy(), 0
    while step < 100:
        conv.begin_chunk(adapter.infer(obs(q, step))["actions"], 8)
        proprio = np.r_[q, 0.0]
        while (row := conv.next_row(proprio)) is not None:
            q, step = row[:7], step + 1
            proprio = np.r_[q, 0.0]
    assert np.allclose(q, RESET) and rec.seen == []
    out = adapter.infer(obs(q, step))
    assert np.allclose(out["actions"][:, :7], rec.seen[0][JOINTS])   # the policy's actions untouched: no wrist offset


def test_the_control_step_is_required():
    with pytest.raises(ValueError, match="control_step"):
        DroidAdapter(Refuse()).infer({JOINTS: HOME})


def test_molmoact2_gets_droid_frames_in_its_slot_order():
    """The runner pads the 848 x 480 cameras to 320 x 320 (--image-size 320); MolmoAct2 gets DROID's 320 x 180 frames
    back without a padding row, exterior 1 and wrist (as MolmoAct2's own DROID clients), and DROID's joints + gripper."""
    cams = [np.full((480, 848, 3), 50 * (i + 1), np.uint8) for i in range(3)]
    o = {k: C.resize_with_pad(c, 320, 320) for k, c in zip(IMAGES, cams)}
    o.update({JOINTS: DROID_RESET, "observation/gripper_position": np.array([0.25]), "prompt": "p"})
    images, state = molmoact2_inputs(o)
    assert [im.shape for im in images] == [(180, 320, 3)] * 2
    assert [(int(im.min()), int(im.max())) for im in images] == [(50, 50), (150, 150)]
    assert state.dtype == np.float32 and np.allclose(state, np.r_[DROID_RESET, 0.25])


def test_openpi_gripper_is_binarised_at_one_half():
    """pi0.5's gripper actions become 0 or 1 at 0.5, as openpi's DROID client and sim-evals' execute them; the joints
    and the other outputs stay as the policy gave them."""
    raw = np.zeros((15, 8), np.float32)
    raw[:, :7] = 0.3
    raw[:, 7] = np.linspace(0.0, 1.0, 15)
    m = OpenpiModel("cfg", "ckpt", "sha")
    m.policy = SimpleNamespace(infer=lambda o: {"actions": raw.copy(), "policy_timing": {"infer_ms": 1.0}})
    out = m.infer({})
    a = out["actions"]
    assert a.dtype == raw.dtype and np.array_equal(a[:, :7], raw[:, :7]) and out["policy_timing"] == {"infer_ms": 1.0}
    assert np.array_equal(a[:, 7], (raw[:, 7] > 0.5).astype(np.float32))


def test_molmoact2_actions_clip_only_the_gripper():
    raw = np.zeros((1, 15, 8))
    raw[0, :, 6] = -0.4
    raw[0, :, 7] = np.linspace(-0.2, 1.3, 15)
    a = molmoact2_actions(raw)
    assert a.shape == (15, 8) and np.all(a[:, 6] == -0.4)
    assert a[:, 7].min() == 0.0 and a[:, 7].max() == 1.0 and np.allclose(a[3:12, 7], raw[0, 3:12, 7])


# Two frames of Isaac-GR00T's demo_data/droid_sample (episode 0, frames 0 and 400): DROID's joints and the data's eef_9d
DROID_SAMPLE = [([-0.22476, -0.4210602, -0.1281129, -2.3547568, -0.1962341, 2.2180023, 0.0263882],
                 [0.418337, -0.1657044, 0.5165336, -0.2482533, -0.2758719, -0.9285823, 0.9672262, -0.1233633,
                  -0.2219347]),
                ([-0.1130906, 0.3409959, 0.0495907, -2.1415112, 0.1455035, 2.5047667, -0.2688557],
                 [0.5729071, -0.0376473, 0.2453221, 0.0839255, -0.042899, -0.9955482, 0.9920771, 0.0973273, 0.079439])]


# GR00T's eef_9d reads Euler angles with scipy, which the IKEA images ship without (the puzzle image and the CI lock
# have it). Skipped in such an image only: on a host a missing scipy still fails these tests
skip_in_an_image_without_scipy = pytest.mark.skipif(
    (_runtime.in_puzzle_image() or _runtime.in_ikea_image()) and importlib.util.find_spec("scipy") is None,
    reason="this image ships without scipy, which GR00T's eef_9d needs")


@skip_in_an_image_without_scipy
@pytest.mark.parametrize("q, eef", DROID_SAMPLE)
def test_groot_eef_9d_matches_droid_data(q, eef):
    assert np.allclose(droid_eef_9d(q), eef, atol=1e-6)


@skip_in_an_image_without_scipy
def test_groot_request_has_droid_frames_and_state():
    """Exterior 1 and the wrist as DROID's 320 x 180 frames (batch 1, one time step), DROID's eef_9d, gripper and
    joints as float32, and the prompt."""
    cams = [np.full((480, 848, 3), 50 * (i + 1), np.uint8) for i in range(3)]
    o = {k: C.resize_with_pad(c, 320, 320) for k, c in zip(IMAGES, cams)}
    r = groot_request({**o, JOINTS: DROID_RESET, "observation/gripper_position": np.array([0.25]), "prompt": "p"})
    video = r["video"]
    assert sorted(video) == ["exterior_image_1_left", "wrist_image_left"]
    assert all(v.shape == (1, 1, 180, 320, 3) and v.dtype == np.uint8 for v in video.values())
    assert np.all(video["exterior_image_1_left"] == 50) and np.all(video["wrist_image_left"] == 150)
    state = r["state"]
    assert {k: v.shape for k, v in state.items()} == {"eef_9d": (1, 1, 9), "gripper_position": (1, 1, 1),
                                                      "joint_position": (1, 1, 7)}
    assert all(v.dtype == np.float32 for v in state.values())
    assert np.allclose(state["eef_9d"][0, 0], droid_eef_9d(DROID_RESET), atol=1e-6)
    assert np.allclose(state["joint_position"][0, 0], DROID_RESET) and state["gripper_position"][0, 0, 0] == 0.25
    assert r["language"] == {"annotation.language.language_instruction": [["p"]]}


def test_groot_actions_join_joints_and_binarised_gripper():
    out = {"joint_position": np.full((1, 40, 7), 0.3, np.float32),
           "gripper_position": np.linspace(0, 1, 40, dtype=np.float32).reshape(1, 40, 1)}
    a = groot_actions(out)
    assert a.shape == (40, 8) and np.allclose(a[:, :7], 0.3)
    assert np.array_equal(a[:, 7], (np.linspace(0, 1, 40) > 0.5).astype(float))


def test_cosmos3_request_has_three_droid_frames():
    """Both exterior cameras and the wrist as DROID's 640 x 360 frames, the joints (7,) and gripper (1,) as float32."""
    cams = [np.full((480, 848, 3), 50 * (i + 1), np.uint8) for i in range(3)]
    o = {k: C.resize_with_pad(c, 640, 640) for k, c in zip(IMAGES, cams)}
    r = cosmos3_request({**o, JOINTS: DROID_RESET, "observation/gripper_position": np.array([0.25]), "prompt": "p"})
    assert [r[k].shape for k in IMAGES] == [(360, 640, 3)] * 3
    assert [int(r[k].min()) for k in IMAGES] == [int(r[k].max()) for k in IMAGES] == [50, 100, 150]
    assert r[JOINTS].dtype == r["observation/gripper_position"].dtype == np.float32
    assert np.allclose(r[JOINTS], DROID_RESET) and r["observation/gripper_position"].tolist() == [0.25]
    assert r["prompt"] == "p"


def test_cosmos3_actions_binarise_the_gripper():
    action = np.c_[np.full((32, 7), 0.3, np.float32), np.linspace(0, 1, 32, dtype=np.float32)]
    a = cosmos3_actions(action)
    assert a.shape == (32, 8) and a.dtype == np.float64 and np.allclose(a[:, :7], 0.3)
    assert np.array_equal(a[:, 7], (np.linspace(0, 1, 32) > 0.5).astype(float))


@pytest.mark.sim
@puzzle_image_only
def test_episode_reaches_droid_reset(tmp_path):
    """The runner against the adapter around HoldPolicy on a one-arm puzzle (rendering off): the arm reaches DROID's
    reset in the first 2.5 s, then holds there; no infrastructure error."""
    task = roboarc.task_info(os.environ.get("ROBOARC_TEST_DROID", "buzz-wire-3d-06"))
    from roboarc.vla.run import main
    port, stop = _serve(DroidAdapter(HoldPolicy(horizon=15)))
    try:
        argv = ["--task", task.id, "--policy", f"ws://127.0.0.1:{port}", "--out", str(tmp_path), "--run-id", "adapter",
                "--renderer", "none", "--convention", "droid", "--policy-hz", "15", "--horizon", "8",
                "--max-steps", "200", "--seed", "0"]
        assert main(argv) == 0
    finally:
        stop()
    ep = tmp_path / task.id / "adapter" / "s0" / "e1"
    res = json.loads((ep / "result.json").read_text())
    assert res["infra_error"] == 0 and res["steps"] == 200, res
    with np.load(ep / "chunks.npz") as z:
        steps, length, actions = z["control_step"], z["length"], z["actions"]
    chunks = np.split(actions, np.cumsum(length)[:-1])
    last_move = max(i for i, s in enumerate(steps) if s < 100)
    assert np.allclose(chunks[last_move][:, :7], RESET, atol=1e-5)               # the move ends on DROID's reset
    assert np.abs(chunks[last_move + 1][:, :7] - RESET).max() < 0.02          # then the arm holds there (measured)
