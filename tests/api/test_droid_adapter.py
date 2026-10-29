"""roboarc.vla.droid_adapter: DROID's wrist convention and start pose around a DROID-trained policy. Without a simulator
unless marked `sim` (the runner against the adapter, rendering off)."""
import json
import os

import numpy as np
import pytest

import roboarc
from roboarc.vla import conventions as C
from roboarc.vla.droid_adapter import DROID_RESET, JOINTS, WRIST_OFFSET, DroidAdapter
from roboarc.vla.stub_server import HoldPolicy

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


@pytest.mark.sim
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
