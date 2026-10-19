"""knot (10-13, main: start placement labs from a saved full state instead of replaying the untie rows). A checkpoint =
the env's current tick completed (EpisodeState: the MuJoCo world capture (mjSTATE_INTEGRATION, eq_data, the grasp
manager's and slip windows' state, contact evidence, plastic rest angles), the robot controller's state, the task's
state, the RNG) + the Session's counters, controller targets and recorded rows (pickle). restore_checkpoint puts it
into a fresh Session of the same instance (the world id re-bound) and records it as the current tick, so the
Session's own rewinds work from it."""
import copy
import dataclasses
import pickle
from pathlib import Path

import numpy as np

SESSION_FIELDS = ("n_steps", "physics_steps", "sim_time_s", "last_step_ticks", "_reward", "episode_end", "episode_done")
CODEC = "knot_place_checkpoint/1"


def save_checkpoint(s, path, meta=None):
    env = s.env
    env._checkpoint_current()
    st = env._history.get(env._state_id)
    blob = dict(codec=CODEC, instance_sha=s.instance_sha, episode_state=st,
                session={f: copy.deepcopy(getattr(s, f)) for f in SESSION_FIELDS},
                recording=[np.array(r) for r in s._recording], ticks=list(s._recorded_ticks),
                targets=np.array(s._targets, copy=True), meta=meta or {})
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return Path(path).stat().st_size


def restore_checkpoint(s, path):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    if blob.get("codec") != CODEC:
        raise ValueError(f"{path}: not a {CODEC} file")
    if blob["instance_sha"] != s.instance_sha:
        raise ValueError(f"{path}: a checkpoint of instance {blob['instance_sha']}, not {s.instance_sha}")
    env = s.env
    st = blob["episode_state"]
    env._world.restore_state(dataclasses.replace(st.world, world_id=env._world._world_id))
    env._robot.restore_state(st.robot)
    env._task.set_state(copy.deepcopy(st.task))
    env._rng.bit_generator.state = copy.deepcopy(st.rng_state)
    env._control_step = st.info.control_step
    env._terminated, env._truncated = st.info.terminated, st.info.truncated
    env._sensors.invalidate()
    env._save_state()
    env._checkpoint_current()
    for f, v in blob["session"].items():
        setattr(s, f, copy.deepcopy(v))
    s._recording[:] = [np.array(r) for r in blob["recording"]]
    s._recorded_ticks[:] = list(blob["ticks"])
    s._targets = np.array(blob["targets"], copy=True)
    s.instability = None
    s.last_step_meta = {}
    return blob["meta"]
