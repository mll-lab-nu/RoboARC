"""One task's environment end to end, inside its image (marker sim): spaces, reset determinism (the graded birth
state), clamping instead of failing, state snapshots, the success predicate on the reference, the deliverable."""
import json
from pathlib import Path

import numpy as np
import pytest

import roboarc
from roboarc.references import load_reference

from conftest import sim_task

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def env():
    e = roboarc.make(sim_task(), renderer="none")
    yield e
    e.close()


def test_spaces(env):
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert set(obs) == {"proprio"}                          # renderer="none": no images
    assert obs["proprio"].shape == (len(env.proprio_layout),)
    assert env.action_space.shape == (len(env.action_layout),)
    for name, (start, n, _) in env.proprio_groups.items():
        assert 0 <= start and start + n <= len(env.proprio_layout), name
    assert env.control_hz > 0 and env.max_episode_steps == env.task.max_episode_steps
    json.dumps(env.describe(), default=str)


def test_birth_is_the_graded_instance(env):
    """reset() always restores the same birth state, the one the task's own frozen record names."""
    _, info = env.reset()
    birth = info["birth_hash"]
    assert birth == env.state_sha() == env.birth_hash
    for _ in range(5):
        env.step(env.action_space.sample())
    _, info = env.reset(seed=123)
    assert info["birth_hash"] == birth and env.state_sha() == birth
    u = env.unwrapped
    if env.task.suite == "puzzles":
        frozen = json.loads(Path(u.instance_path).read_text())["state_sha"]
    else:
        frozen = u.binding["initial_state_sha256"]
    assert frozen.startswith(birth) or birth.startswith(frozen)


def test_hold_is_deterministic(env):
    hashes = []
    for _ in range(2):
        obs, _ = env.reset()
        hold = env.unwrapped.session.hold_action()
        for _ in range(20):
            obs, r, term, trunc, info = env.step(hold)
        hashes.append(env.state_sha())
    assert hashes[0] == hashes[1]


def test_out_of_range_actions_are_clamped_not_failed(env):
    env.reset()
    big = np.where(np.isfinite(env.action_space.high), env.action_space.high, 0) + 10.0
    obs, r, term, trunc, info = env.step(big.astype(env.action_space.dtype))
    applied = np.asarray(info["applied_action"], dtype=np.float64)
    assert np.all(applied <= env.action_space.high + 1e-6)
    assert info["speed_limited"] or info.get("clipped")
    with pytest.raises(ValueError):
        env.step(np.zeros(env.action_space.shape[0] + 1))
    with pytest.raises(ValueError):
        env.step(np.full(env.action_space.shape, np.nan))


def test_standard_joint_speed_rule(env):
    if env.task.suite != "puzzles":
        pytest.skip("IKEA: the speed clamp is in rcb_ikea.Session.applied (tested by the IKEA scenes)")
    u = env.unwrapped
    env.reset()
    hold = np.asarray(u.session.hold_action(), dtype=np.float64)
    a = hold.copy()
    a[u._arm_cols[0]] += 0.3                                # 0.3 rad in one step: the rule allows 0.04
    obs, r, term, trunc, info = env.step(a)
    step = np.asarray(info["applied_action"], dtype=np.float64) - hold
    assert info["speed_limited"]
    assert abs(step[u._arm_cols[0]]) <= u.joint_step_limit_rad + 1e-6


def test_get_set_state_continues_identically(env):
    env.reset()
    rng = np.random.default_rng(0)
    hold = np.asarray(env.unwrapped.session.hold_action(), dtype=np.float64)
    rows = [hold + rng.normal(0, 0.01, hold.shape) * (np.isfinite(env.action_space.high)) for _ in range(30)]
    for a in rows[:10]:
        env.step(a)
    snap = env.get_state()
    for a in rows[10:]:
        env.step(a)
    want = env.state_sha()
    how = env.set_state(snap)
    assert how in ("checkpoint", "replay")
    for a in rows[10:]:
        env.step(a)
    assert env.state_sha() == want
    env.reset()                                              # the snapshot's checkpoint is gone: the replay path
    assert env.set_state(snap) in ("replay", "checkpoint")
    for a in rows[10:]:
        env.step(a)
    assert env.state_sha() == want


def test_reference_reaches_success(env, tmp_path):
    """The task's own success predicate on its reference (short tasks only: ROBOARC_TEST_* picks them)."""
    rows = load_reference(env.task.id)
    if rows is None:
        pytest.skip("no reference")
    if len(rows) > 100_000:
        pytest.skip("long reference: covered by the heavy bit-identity tests")
    env.reset()
    success = False
    for a in rows:
        obs, r, term, trunc, info = env.step(a)
        success = success or info["success"]
        if term and env.task.suite == "puzzles":
            break
    if env.task.suite == "ikea":
        fc = env.evaluate()["final_check"]
        assert fc["success"]
    else:
        assert success and info["episode_end"] == "success"
        traj = env.trajectory()
        path = env.save_trajectory(tmp_path / "trajectory.npz")
        with np.load(path) as z:
            assert np.array_equal(z["actions"], traj)
        st = env.unwrapped.grade(traj)                       # the privileged verifier's own replay
        assert st["success"] and st["strict_goal"] is not None
    assert env.task.reference_end_hash is None or env.state_sha().startswith(env.task.reference_end_hash) or not success


def test_ee_pose_wrapper(env):
    """EEPoseAction (puzzles): a delta of +1 cm in z per step lifts the first arm's TCP; the plan says ok."""
    if env.task.suite != "puzzles":
        pytest.skip("the EE wrapper is for the puzzles")
    from roboarc.wrappers import EEPoseAction
    u = env.unwrapped
    w = EEPoseAction(env, mode="delta")
    obs, _ = w.reset()
    g = env.proprio_groups["tcp_pose"][0]
    z0 = float(obs["proprio"][g + 2])
    a = np.zeros(w.action_space.shape, np.float64)
    hold = u.session.hold_action()
    for i, arm in enumerate(u.arms):
        a[8 * i + 7] = hold[u.session.robot.grip_index[arm]]
    a[2] = 0.01
    for _ in range(10):
        obs, r, term, trunc, info = w.step(a)
    assert info["ee_plan"][0]["status"] == "ok", info["ee_plan"]
    assert float(obs["proprio"][g + 2]) - z0 > 0.005


def test_camera_info_frames(env):
    """camera_info(): cam2world_gl is the inverse of extrinsic_cv (robot frame) with the OpenGL axis flip; the principal
    point is the image centre; the static cameras see both TCPs in front of them (robot-frame poses agree)."""
    obs, _ = env.reset()
    info = env.camera_info()
    assert set(info) == set(env.unwrapped.all_camera_names)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    g = env.proprio_groups["tcp_pose"][0]
    tcps = [obs["proprio"][g + 7 * k:g + 7 * k + 3].astype(np.float64) for k in range(len(env.unwrapped.arms))]
    for name, v in info.items():
        E = np.eye(4); E[:3] = np.asarray(v["extrinsic_cv"], dtype=np.float64)
        assert np.allclose(np.linalg.inv(E) @ flip, np.asarray(v["cam2world_gl"], dtype=np.float64), atol=1e-6), name
        K = np.asarray(v["intrinsic_cv"], dtype=np.float64)
        W, H = v["image_size"]
        assert abs(K[0, 2] - (W - 1) / 2) <= 1.0 and abs(K[1, 2] - (H - 1) / 2) <= 1.0, name
        if not v["moves"]:
            for p in tcps:
                pc = E[:3, :3] @ p + E[:3, 3]
                assert pc[2] > 0, name
                u, w = (K @ (pc / pc[2]))[:2]
                assert 0 <= u < W and 0 <= w < H, (name, u, w)
