"""A DROID-trained policy (pi0.5-DROID, MolmoAct2-DROID, GR00T-N1.7-DROID, Cosmos3-Nano-Policy-DROID:
droid_models.py) on RoboARC's one-arm puzzles (one Franka arm with a Robotiq 2F-85, the `droid` conventions), served
like stub_server's policies: no change to the runner or the tasks. The adapter knows no task: from the observation's joints and control step only, it makes the
robot DROID's on the two points the convention passes through as they are:

    wrist   RoboARC mounts the Robotiq on the Franka hand frame, -45 deg about the flange axis (topogym's
            assets/robot.py, build_robotiq_panda); DROID mounts it on the flange. The same gripper pose reads
            q7_DROID = q7_RoboARC - pi/4: the policy sees DROID's q7 and its actions are mapped back.
    start   DROID's episodes start at its reset joints (droid/robot_env.py). For the first MOVE_S simulated seconds
            (the observation's "roboarc"/"control_step") the policy is not asked: each chunk moves the arm on a straight
            joint-space line toward DROID's reset, re-planned from the measured joints, with the gripper open.

The cameras are the runner's: `--camera-poses droid` (droid_cameras.py) puts them where DROID's are.

    python -m roboarc.vla.droid_adapter --model jointpos --port 8000   # in the model's environment, RoboARC on PYTHONPATH
    python -m roboarc.vla.run --task <task> --policy ws://HOST:8000 --convention droid --policy-hz 15 --horizon 8 \
        --camera-poses droid

scripts/vla_droid.sh runs both in one command. --model: jointpos, velocity, polaris, molmoact2, groot, cosmos3, hold
(droid_models.py). The wrist offset applies to joint positions only: DROID's joint velocities (velocity) go as they
are."""
from __future__ import annotations

import argparse
import logging

import numpy as np

from .conventions import DROID_CONTROL_HZ, DROID_MAX_JOINT_DELTA
from .droid_models import JOINTS, MODELS
from .stub_server import serve

DROID_RESET = np.array([0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0])  # droid/robot_env.py
WRIST_OFFSET = np.pi / 4                  # q7_RoboARC - q7_DROID for the same gripper pose
CONTROL_HZ = 40.0                         # the puzzles' control rate
MOVE_S, MOVE_RAD_S, CHUNK = 2.5, 1.0, 15  # the move to DROID's reset: simulated seconds, speed, actions per chunk


class DroidAdapter:
    """Wraps `policy` (a DroidModel, or anything with infer(obs) -> {"actions": (H, 8)} in DROID's units) with
    DROID's wrist convention and start pose. Keeps no state between requests."""

    def __init__(self, policy, *, velocity: bool = False):
        self.policy = policy
        self.velocity = bool(velocity)         # DROID's joint velocities (droid-velocity) instead of joint positions

    def infer(self, obs: dict) -> dict:
        meta = obs.get("roboarc")
        if not isinstance(meta, dict) or "control_step" not in meta:
            raise ValueError("DroidAdapter needs the observation's roboarc/control_step (run without --no-roboarc-obs)")
        model_obs = {k: v for k, v in obs.items() if k != "roboarc"}
        q = np.array(model_obs[JOINTS], dtype=np.float64).reshape(-1)
        q[6] -= WRIST_OFFSET
        model_obs[JOINTS] = q
        if meta["control_step"] / CONTROL_HZ < MOVE_S:
            plan = toward_reset(q)
            out = {"actions": velocities(q, plan) if self.velocity else plan}
        else:
            out = dict(self.policy.infer(model_obs))
        actions = np.array(out["actions"], dtype=np.float64)
        if not self.velocity:
            actions[:, 6] += WRIST_OFFSET
        out["actions"] = actions
        return out


def toward_reset(q: np.ndarray) -> np.ndarray:
    """(CHUNK, 8) DROID actions from joints `q`: the straight joint-space line to DROID's reset at MOVE_RAD_S on the
    joint that moves most, then held there; the gripper open."""
    d = DROID_RESET - q
    far = float(np.abs(d).max())
    k = np.arange(1, CHUNK + 1)
    frac = np.minimum(k * MOVE_RAD_S / DROID_CONTROL_HZ / far, 1.0) if far > 0 else np.ones(CHUNK)
    return np.concatenate([q + frac[:, None] * d, np.zeros((CHUNK, 1))], axis=1)


def velocities(q: np.ndarray, plan: np.ndarray) -> np.ndarray:
    """A joint-position plan (H, 8) from joints `q` as DROID's normalised velocities: each action moves the joints by
    velocity x DROID_MAX_JOINT_DELTA from where they are (droid-velocity); the gripper column stays."""
    out = plan.copy()
    out[:, :7] = np.diff(np.vstack([q, plan[:, :7]]), axis=0) / DROID_MAX_JOINT_DELTA
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m roboarc.vla.droid_adapter", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default="jointpos")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    model = MODELS[a.model]
    model.load()
    meta = {"policy": f"{model.name} + roboarc DroidAdapter", "model": a.model, "version": model.version,
            "convention": "droid-velocity" if model.velocity else "droid", "policy_hz": DROID_CONTROL_HZ, "move_s": MOVE_S,
            "droid_reset": DROID_RESET.tolist(), "wrist_offset_rad": WRIST_OFFSET}
    serve(DroidAdapter(model, velocity=model.velocity), a.host, a.port, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
