"""Several RoboARC environments at once (preview): one environment per worker process, stepped together.

    import roboarc
    venv = roboarc.make_vec("orbits-02", 4, instance_seeds="all", renderer="nyx", image_size=(224, 224))
    obs, infos = venv.reset()                       # obs["proprio"]: (4, P); obs["rgb"][cam]: (4, H, W, 3)
    obs, rewards, terminated, truncated, infos = venv.step(actions)        # actions: (4, action_dim)
    venv.close()

The simulators are one scene per process, so every environment lives in its own worker process (started with "spawn":
no simulator state is inherited) and the batch is gathered from all workers each step. Each worker builds its
environment with roboarc.make() and the same options, so physics, rules and images are the single environment's.
Gymnasium is not needed (the published images do not ship it); the attributes follow gymnasium.vector where they apply
(num_envs, single_observation_space, single_action_space)."""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import traceback
from functools import partial

import numpy as np

from ._runtime import GRADED_ENV

_START_LOCK = threading.Lock()      # os.environ is patched around each worker's start (its GPU)


def _make_env(task_id, kwargs):
    from .core import make
    return make(task_id, **kwargs)


def _ended(env) -> bool:
    return bool(env._truncated or (env._terminated and not env.step_after_end))


def _worker(conn, env_fn) -> None:
    """One worker: build the environment, then answer commands until "close". Every answer is ("ok", value) or
    ("error", traceback text); the environment is closed when the worker ends."""
    env = None
    try:
        env = env_fn()
        conn.send(("ok", {"observation_space": env.observation_space, "action_space": env.action_space,
                          "describe": env.describe() if hasattr(env, "describe") else {}}))
    except BaseException:  # noqa: BLE001
        conn.send(("error", traceback.format_exc()))
        conn.close()
        return
    last_obs = None
    while True:
        try:
            cmd, args = conn.recv()
        except EOFError:
            break
        try:
            if cmd == "reset":
                last_obs, info = env.reset(**args)
                out = (last_obs, info)
            elif cmd == "step":
                if _ended(env):     # an ended episode is not stepped again: its last observation, no reward
                    out = (last_obs, 0.0, bool(env._terminated), bool(env._truncated), {"ended": True})
                else:
                    last_obs, reward, term, trunc, info = env.step(args["action"])
                    if args["autoreset"] and (term or trunc):
                        info = dict(info, final_observation=last_obs)
                        last_obs, reset_info = env.reset()
                        info["reset_info"] = reset_info
                    out = (last_obs, reward, term, trunc, info)
            elif cmd == "observe":
                last_obs = env.observe()
                out = last_obs
            elif cmd == "call":
                out = getattr(env, args["name"])(*args["args"], **args["kwargs"])
            elif cmd == "getattr":
                out = getattr(env, args["name"])
            elif cmd == "setattr":
                setattr(env, args["name"], args["value"])
                out = None
            elif cmd == "close":
                conn.send(("ok", None))
                break
            else:
                raise ValueError(f"unknown command {cmd!r}")
            conn.send(("ok", out))
        except BaseException:  # noqa: BLE001
            conn.send(("error", traceback.format_exc()))
    try:
        env.close()
    finally:
        conn.close()


def _stack(items: list):
    """A list of observations (dicts of arrays, nested one level for rgb / depth) -> one dict of stacked arrays."""
    first = items[0]
    if isinstance(first, dict):
        return {k: _stack([it[k] for it in items]) for k in first}
    return np.stack([np.asarray(it) for it in items])


class VecEnv:
    """`env_fns`: one picklable callable per environment, each returning a RoboARC environment (roboarc.make_vec builds
    them). The environments must share one observation and action space (one task; any of its instance seeds).

    step(actions) -> (obs, rewards (N,), terminated (N,), truncated (N,), infos (list of N dicts)). An environment whose
    episode has ended is not stepped again until reset (its entry repeats its last observation, reward 0, and
    infos[i]["ended"] is True), unless autoreset=True: then it is reset in the step that ended it, the returned
    observation is the new episode's first, and infos[i] holds "final_observation" and "reset_info"."""

    def __init__(self, env_fns, *, autoreset: bool = False, gpus=None, start_method: str = "spawn"):
        if not env_fns:
            raise ValueError("no environments")
        self.num_envs = len(env_fns)
        self.autoreset = bool(autoreset)
        self.closed = False
        ctx = mp.get_context(start_method)
        gpus = None if gpus is None else [str(g) for g in gpus]
        self._conns, self._procs = [], []
        try:
            for i, fn in enumerate(env_fns):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, fn), daemon=True, name=f"roboarc-env-{i}")
                # a spawned worker starts with the parent's os.environ of this moment: its GPU, and numpy's graded
                # settings (roboarc/__init__.py), which a parent that imported numpy before roboarc does not hold;
                # the worker imports numpy afresh, so it gets them (a value the parent sets itself is kept)
                patch = {k: v for k, v in GRADED_ENV.items() if k not in os.environ}
                if gpus:
                    patch["CUDA_VISIBLE_DEVICES"] = gpus[i % len(gpus)]
                with _START_LOCK:
                    saved = {k: os.environ.get(k) for k in patch}
                    os.environ.update(patch)
                    try:
                        p.start()
                    finally:
                        for k, v in saved.items():
                            if v is None:
                                os.environ.pop(k, None)
                            else:
                                os.environ[k] = v
                child.close()
                self._conns.append(parent)
                self._procs.append(p)
            ready = self._gather(range(self.num_envs))
        except BaseException:
            self.close(terminate=True)
            raise
        self.single_observation_space = ready[0]["observation_space"]
        self.single_action_space = ready[0]["action_space"]
        self.descriptions = [r["describe"] for r in ready]
        shapes = {(tuple(r["action_space"].shape), str(r["observation_space"])) for r in ready}
        if len(shapes) != 1:
            self.close()
            raise ValueError("the environments do not share one observation and action space")

    # ---------------------------------------------------------------- plumbing
    def _gather(self, indices) -> list:
        """Receive one answer from each of `indices`; raise on the first worker error (after receiving them all)."""
        out, errors = [], []
        for i in indices:
            try:
                status, value = self._conns[i].recv()
            except EOFError:
                self._procs[i].join(5)
                status, value = "error", f"worker {i} exited (exit code {self._procs[i].exitcode})"
            if status == "error":
                errors.append(f"environment {i}:\n{value}")
            out.append(value)
        if errors:
            raise RuntimeError("roboarc vector environment: " + "\n".join(errors))
        return out

    def _send(self, indices, cmd, args) -> list:
        self._check_open()
        for i in indices:
            self._conns[i].send((cmd, args))
        return self._gather(indices)

    def _indices(self, indices):
        return list(range(self.num_envs)) if indices is None else [int(i) for i in indices]

    def _check_open(self):
        if self.closed:
            raise RuntimeError("the vector environment is closed")

    # ---------------------------------------------------------------- API
    def reset(self, *, seed=None, indices=None, options=None):
        """Reset all environments (or `indices`) to their frozen instances -> (obs of those environments, infos)."""
        self._check_open()
        idx = self._indices(indices)
        seeds = [None] * len(idx) if seed is None else [seed + k for k in range(len(idx))]
        for i, s in zip(idx, seeds):
            self._conns[i].send(("reset", {"seed": s, "options": options}))
        res = self._gather(idx)
        return _stack([r[0] for r in res]), [r[1] for r in res]

    def step(self, actions):
        """One control step of every environment; `actions` (num_envs, action_dim)."""
        a = np.asarray(actions)
        if a.shape != (self.num_envs,) + tuple(self.single_action_space.shape):
            raise ValueError(f"actions must have shape {(self.num_envs,) + tuple(self.single_action_space.shape)}, "
                             f"got {a.shape}")
        self._check_open()
        for i in range(self.num_envs):
            self._conns[i].send(("step", {"action": a[i], "autoreset": self.autoreset}))
        res = self._gather(range(self.num_envs))
        obs = _stack([r[0] for r in res])
        return (obs, np.array([r[1] for r in res], dtype=np.float64), np.array([r[2] for r in res], dtype=bool),
                np.array([r[3] for r in res], dtype=bool), [r[4] for r in res])

    def observe(self, indices=None):
        """The observations of the current states (images included), without stepping (see RoboARCEnv.observe)."""
        return _stack(self._send(self._indices(indices), "observe", {}))

    def call(self, name: str, *args, indices=None, **kwargs) -> list:
        """env.<name>(*args, **kwargs) in every environment (or `indices`) -> the list of results, e.g.
        call("success"), call("state_sha"), call("get_state"), call("evaluate")."""
        return self._send(self._indices(indices), "call", {"name": name, "args": args, "kwargs": kwargs})

    def get_attr(self, name: str, indices=None) -> list:
        return self._send(self._indices(indices), "getattr", {"name": name})

    def set_attr(self, name: str, value, indices=None) -> None:
        """Set an attribute in every environment (or `indices`), e.g. set_attr("step_images", False)."""
        self._send(self._indices(indices), "setattr", {"name": name, "value": value})

    def close(self, terminate: bool = False, timeout: float = 30.0) -> None:
        if self.closed:
            return
        self.closed = True
        for conn, p in zip(self._conns, self._procs):
            if not terminate and p.is_alive():
                try:
                    conn.send(("close", None))
                    if conn.poll(timeout):
                        conn.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for conn, p in zip(self._conns, self._procs):
            p.join(timeout if not terminate else 0.1)
            if p.is_alive():
                p.terminate()
                p.join(5)
            conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close(terminate=True)
        except Exception:  # noqa: BLE001
            pass

    def __len__(self):
        return self.num_envs


def make_vec(task_id: str, num_envs: int, *, instance_seeds=0, gpus=None, autoreset: bool = False,
             start_method: str = "spawn", **make_kwargs) -> VecEnv:
    """`num_envs` environments of one task, each in its own worker process (preview; docs/API.md).

    instance_seeds  0 (default: every environment on the released instance), one seed for all, "all" (the task's
                    instance seeds in turn: 0, 1, 2, 0, ... the appearance seeds, same physics), or a list of num_envs
                    seeds.
    gpus            GPU indices assigned to the workers in turn (each worker sees only its GPU: CUDA_VISIBLE_DEVICES);
                    default: every worker sees the parent's GPUs.
    autoreset       reset an environment in the step that ends its episode (see VecEnv).
    Further keyword arguments are roboarc.make()'s (renderer, cameras, depth, image_size, rules, ...)."""
    from .registry import resolve
    n = int(num_envs)
    if n < 1:
        raise ValueError("num_envs must be at least 1")
    info = resolve(task_id)
    if isinstance(instance_seeds, str):
        if instance_seeds != "all":
            raise ValueError("instance_seeds must be an int, a list of ints or 'all'")
        seeds = [list(info.seeds)[i % len(info.seeds)] for i in range(n)]
    elif isinstance(instance_seeds, (int, np.integer)):
        seeds = [int(instance_seeds)] * n
    else:
        seeds = [int(s) for s in instance_seeds]
        if len(seeds) != n:
            raise ValueError(f"instance_seeds has {len(seeds)} entries for {n} environments")
    bad = sorted({s for s in seeds if s not in info.seeds})
    if bad:
        raise ValueError(f"{info.id} allows the instance seeds {list(info.seeds)}, not {bad}")
    fns = [partial(_make_env, info.id, dict(make_kwargs, instance_seed=s)) for s in seeds]
    return VecEnv(fns, autoreset=autoreset, gpus=gpus, start_method=start_method)
