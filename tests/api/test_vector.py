"""The vector environment (roboarc.make_vec, preview): the worker protocol with a stand-in environment anywhere, and
one task's environments end to end inside its image (marker sim)."""
from functools import partial

import numpy as np
import pytest

import roboarc
from roboarc import spaces
from roboarc.registry import list_tasks
from roboarc.vector import VecEnv

from conftest import sim_task


class _CountEnv:
    """A stand-in with the RoboARCEnv attributes the vector environment uses: the state is a counter, the episode
    terminates at `end` steps, and step() raises after the end (as RoboARCEnv does)."""

    def __init__(self, end=3, offset=0.0):
        self.end, self.offset = end, offset
        self.observation_space = spaces.Dict({"proprio": spaces.Box(-np.inf, np.inf, (2,), np.float32),
                                               "rgb": spaces.Dict({"cam": spaces.Box(0, 255, (4, 5, 3), np.uint8)})})
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.step_after_end = False
        self._n = 0
        self._terminated = self._truncated = False

    def _obs(self):
        return {"proprio": np.array([self._n, self.offset], np.float32),
                "rgb": {"cam": np.full((4, 5, 3), self._n, np.uint8)}}

    def reset(self, seed=None, options=None):
        self._n = 0
        self._terminated = self._truncated = False
        return self._obs(), {"birth_hash": "b"}

    def step(self, action):
        if self._terminated:
            raise RuntimeError("the episode has ended")
        self._n += 1
        self._terminated = self._n >= self.end
        return self._obs(), float(self._terminated), self._terminated, False, {"success": self._terminated}

    def observe(self):
        return self._obs()

    def boom(self):
        raise ValueError("boom")

    def close(self):
        pass


def _venv(**kw):
    fns = [partial(_CountEnv, end=e, offset=float(i)) for i, e in enumerate((2, 3))]
    return VecEnv(fns, start_method="fork", **kw)


def test_batches_and_ended_envs():
    with _venv() as v:
        assert v.num_envs == 2 and v.single_action_space.shape == (2,)
        obs, infos = v.reset()
        assert obs["proprio"].shape == (2, 2) and obs["rgb"]["cam"].shape == (2, 4, 5, 3)
        assert [i["birth_hash"] for i in infos] == ["b", "b"]
        a = np.zeros((2, 2), np.float32)
        v.step(a)
        obs, rew, term, trunc, infos = v.step(a)              # env 0 ends here
        assert list(term) == [True, False] and list(rew) == [1.0, 0.0]
        obs, rew, term, trunc, infos = v.step(a)              # env 0 is not stepped again; env 1 ends
        assert infos[0] == {"ended": True} and obs["proprio"][0, 0] == 2 and rew[0] == 0.0
        assert list(term) == [True, True] and obs["proprio"][1, 0] == 3
        obs, _ = v.reset(indices=[0])
        assert obs["proprio"].shape == (1, 2) and obs["proprio"][0, 0] == 0
        assert v.get_attr("offset") == [0.0, 1.0]
        v.set_attr("offset", 5.0, indices=[1])
        assert v.observe()["proprio"][:, 1].tolist() == [0.0, 5.0]


def test_autoreset():
    with _venv(autoreset=True) as v:
        v.reset()
        a = np.zeros((2, 2), np.float32)
        v.step(a)
        obs, rew, term, _, infos = v.step(a)
        assert term[0] and obs["proprio"][0, 0] == 0                     # the new episode's first observation
        assert infos[0]["final_observation"]["proprio"][0] == 2 and "reset_info" in infos[0]


def test_errors_name_the_environment():
    with _venv() as v:
        v.reset()
        with pytest.raises(RuntimeError, match="environment 0"):
            v.call("boom")
        with pytest.raises(ValueError, match="shape"):
            v.step(np.zeros((3, 2)))
        assert v.call("observe")[1]["proprio"][1] == 1.0                 # the workers still answer
    with pytest.raises(RuntimeError, match="closed"):
        v.reset()


class _EnvSeen(_CountEnv):
    """A stand-in that records, at construction in its worker, the graded numpy settings of the worker's environment."""

    def __init__(self):
        import os
        from roboarc._runtime import GRADED_ENV
        super().__init__()
        self.seen = {k: os.environ.get(k) for k in GRADED_ENV}


def test_workers_get_the_graded_numpy_settings(monkeypatch):
    """A parent that imported numpy before roboarc lacks the graded settings in os.environ; each worker imports numpy
    afresh and must start with them (the parent's own os.environ is left as it was)."""
    import os
    from roboarc._runtime import GRADED_ENV
    for k in GRADED_ENV:
        monkeypatch.delenv(k, raising=False)
    with VecEnv([_EnvSeen, _EnvSeen], start_method="fork") as v:
        assert v.get_attr("seen") == [dict(GRADED_ENV)] * 2
    assert all(k not in os.environ for k in GRADED_ENV)


def test_seed_arguments():
    task = next(t for t in list_tasks() if t.suite == "puzzles" and len(t.seeds) == 3)
    with pytest.raises(ValueError, match="instance seeds"):
        roboarc.make_vec(task.id, 2, instance_seeds=[0, 7])
    with pytest.raises(ValueError, match="entries"):
        roboarc.make_vec(task.id, 3, instance_seeds=[0, 1])
    with pytest.raises(ValueError, match="num_envs"):
        roboarc.make_vec(task.id, 0)


@pytest.mark.sim
def test_sim_envs_match():
    """Three workers of the image's test task, seeds 0, 0 and 1: the two seed-0 workers stay bit-identical (birth and
    state hashes), and the seed-1 worker's robot follows the same physics (its proprioception equals theirs; its state
    record differs from seed 0's only in the instance id)."""
    task = sim_task()
    ref = roboarc.load_reference(task)
    with roboarc.make_vec(task, 3, instance_seeds=[0, 0, 1], renderer="none") as v:
        obs, infos = v.reset()
        assert obs["proprio"].shape[0] == 3
        assert infos[0]["birth_hash"] == infos[1]["birth_hash"]
        rows = ref[:50] if ref is not None else np.zeros((50,) + v.single_action_space.shape)
        for row in rows:
            obs, _, term, trunc, _ = v.step(np.repeat(np.asarray(row)[None], 3, axis=0))
        shas = v.call("state_sha")
        assert shas[0] == shas[1]
        assert np.array_equal(obs["proprio"][0], obs["proprio"][1])
        assert np.array_equal(obs["proprio"][0], obs["proprio"][2])
        assert v.get_attr("instance_seed") == [0, 0, 1]
