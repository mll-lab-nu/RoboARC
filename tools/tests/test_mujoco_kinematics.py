#!/usr/bin/env python
"""MuJoCo kinematics queries (topogym/world/mujoco/views.py: forward_kinematics, inverse_kinematics) against Genesis' on
the same robot, through the planning tools' own solvers (topogym.solvers.genesis_kinematics GenesisFKSolver and
GenesisIKSolver.solve, as every EE_ABS oracle calls them).

  genesis  a Session on a Genesis instance of the robot: FK of fixed arm configurations (home + fixed offsets), then IK of
           each one's hand pose from fixed seeds (home, and the target perturbed); writes the cases.
  mujoco   a Session on a MuJoCo instance of the same robot: the same FK and IK; per case the FK pose error, both IK
           residuals and the chosen branch (the two solutions within 1e-3 rad on every arm joint: the same branch);
           every query run twice (bit-identical); --expect: a previous mujoco output, which this fresh process must
           reproduce bit for bit.

  python tools/tests/test_mujoco_kinematics.py genesis --task cast_keyhole --instance G/instance.json --out G.json
  python tools/tests/test_mujoco_kinematics.py mujoco --task cast_keyhole --instance M/instance.json --cases G.json \
      --out M.json [--expect M0.json]
(ALOHA: any ALOHA family, e.g. cast_keyhole; Panda + Robotiq: fiber_bundling. Both instances of one task: the same mount.)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/roboarc/shared"))

# ALOHA (6 arm joints): offsets from home
OFFSETS = ([0., 0., 0., 0., 0., 0.], [.3, -.2, .25, .4, -.3, .5], [-.4, .3, -.35, -.6, .45, -.8],
           [.15, .45, -.5, .9, .6, 1.2], [-.25, -.35, .6, -.3, -.5, -1.5],
           # far from the home seed: large forearm roll / wrist rotate, where the IK may take another wrist branch
           [.9, .9, -.2, -1.2, 1.2, .9], [.95, .85, -.2, 1.95, -.6, -2.2], [-.5, .2, .3, -2.2, -.9, 2.8],
           [.6, -.3, .4, 2.8, 1.2, .5])
# Panda (7 arm joints): home + offsets, then absolute poses near singularities: joint 6 near 0 or pi (the wrist: joint 5
# and joint 7 axes nearly parallel), joint 4 near its -0.0698 rad limit (the elbow nearly stretched)
PANDA_OFFSETS = ([0.] * 7, [.3, -.2, .25, .4, -.3, .5, .2], [-.4, .3, -.35, -.6, .45, -.8, -.5],
                 [.6, .3, -.4, .5, .9, -.6, 1.2], [-.7, -.3, .5, -.4, -1.1, .7, -1.3])
PANDA_POSES = ([.1, .4, -.1, -1.9, .4, .03, -.8], [.3, .5, .2, -1.6, -.6, .06, .4], [-.2, .3, .1, -2.2, 1., 3.1, 1.5],
               [0., .3, 0., -.12, 0., 1.2, 0.], [.2, .6, -.2, -.1, .3, .05, -.5])
NEAR = np.array([.05, -.04, .03, -.06, .05, -.07, .05])


def setup(task, instance, strict):
    from rcb_topogym import Session
    s = Session(task, instance=instance, strict=strict)     # the Genesis side: any instance of the robot
    robot = s.env._robot
    return s, robot, robot.entity


def hand_pose(fk, entity, robot, side, q):
    from topogym.solvers.genesis_kinematics import GenesisFKSolver
    pose = GenesisFKSolver(entity).solve(q)
    i = entity.get_link(robot.groups[side].hand_link).idx_local
    return (np.asarray(pose["position_world_m"]).reshape(-1, 3)[i].copy(),
            np.asarray(pose["orientation_wxyz"]).reshape(-1, 4)[i].copy())


def ik(entity, robot, side, position, quat, seed):
    from topogym.solvers.genesis_kinematics import GenesisIKSolver
    group = robot.groups[side]
    t0 = time.perf_counter()
    q = GenesisIKSolver(entity).solve(link_name=group.hand_link, position_world_m=position, orientation_wxyz=quat,
                                      joint_dof_indices=robot._indices[side][:len(group.arm_joints)],
                                      initial_qpos=seed, max_solver_iterations=100)
    return np.asarray(q, dtype=np.float64).reshape(-1), time.perf_counter() - t0


def arm(robot, side, q):
    return np.asarray(q)[robot._q_indices[side][:len(robot.groups[side].arm_joints)]]


def with_arm(robot, side, base, values):
    q = np.array(base, dtype=np.float64)
    q[robot._q_indices[side][:len(robot.groups[side].arm_joints)]] = values
    return q


def error(a, b):
    (pa, qa), (pb, qb) = a, b
    angle = (Rotation.from_quat(np.asarray(qa)[[1, 2, 3, 0]]).inv() * Rotation.from_quat(np.asarray(qb)[[1, 2, 3, 0]])).magnitude()
    return float(np.linalg.norm(np.asarray(pa) - pb)), float(angle)


def cases(robot, entity):
    base = np.asarray(entity.get_qpos(), dtype=np.float64).reshape(-1)
    out = []
    for side, group in robot.groups.items():
        home = arm(robot, side, base)
        lo, hi = np.asarray(group.arm_limits_rad, dtype=np.float64).T
        if len(home) == 6:
            targets = [np.clip(home + offset, lo + .05, hi - .05) for offset in OFFSETS]
        else:
            targets = ([np.clip(home + offset, lo + .005, hi - .005) for offset in PANDA_OFFSETS]
                       + [np.clip(pose, lo + .005, hi - .005) for pose in PANDA_POSES])
        for k, target in enumerate(targets):
            for label, seed in (("home", home), ("near", np.clip(target + NEAR[:len(home)], lo, hi))):
                out.append(dict(side=side, case=k, seed_label=label, target=target, seed=seed, base=base))
    return out


def run_genesis(a):
    s, robot, entity = setup(a.task, a.instance, False)
    rows = []
    for c in cases(robot, entity):
        q_target = with_arm(robot, c["side"], c["base"], c["target"])
        p, quat = hand_pose(None, entity, robot, c["side"], q_target)
        q, dt = ik(entity, robot, c["side"], p, quat, with_arm(robot, c["side"], c["base"], c["seed"]))
        got = hand_pose(None, entity, robot, c["side"], q)
        rows.append(dict(side=c["side"], case=c["case"], seed_label=c["seed_label"], target_arm=c["target"].tolist(),
                         seed_arm=c["seed"].tolist(), hand_position=p.tolist(), hand_wxyz=quat.tolist(),
                         ik_arm=arm(robot, c["side"], q).tolist(), residual=error(got, (p, quat)), ik_s=dt))
    Path(a.out).write_text(json.dumps(dict(engine="genesis", instance=a.instance, cases=rows), indent=1))
    print(json.dumps(dict(out=a.out, cases=len(rows))))
    s.close()


def run_mujoco(a):
    s, robot, entity = setup(a.task, a.instance, True)
    ref = json.loads(Path(a.cases).read_text())["cases"]
    base = np.asarray(entity.get_qpos(), dtype=np.float64).reshape(-1)
    live = (s.env._world.data.qpos.copy(), s.env._world.data.qvel.copy())
    rows, bits = [], []
    for r in ref:
        side = r["side"]
        q_target = with_arm(robot, side, base, r["target_arm"])
        fk = hand_pose(None, entity, robot, side, q_target)
        fk_error = error(fk, (np.asarray(r["hand_position"]), np.asarray(r["hand_wxyz"])))
        seed = with_arm(robot, side, base, r["seed_arm"])
        q1, dt = ik(entity, robot, side, np.asarray(r["hand_position"]), np.asarray(r["hand_wxyz"]), seed)
        q2, _ = ik(entity, robot, side, np.asarray(r["hand_position"]), np.asarray(r["hand_wxyz"]), seed)
        got = hand_pose(None, entity, robot, side, q1)
        mine, theirs = arm(robot, side, q1), np.asarray(r["ik_arm"])
        rows.append(dict(side=side, case=r["case"], seed_label=r["seed_label"], fk_error=fk_error,
                         residual_mujoco=error(got, (np.asarray(r["hand_position"]), np.asarray(r["hand_wxyz"]))),
                         residual_genesis=r["residual"], branch_diff_rad=float(np.max(np.abs(mine - theirs))),
                         same_branch=bool(np.max(np.abs(mine - theirs)) < 1e-3),
                         to_target_rad=dict(mujoco=float(np.max(np.abs(mine - r["target_arm"]))),
                                            genesis=float(np.max(np.abs(theirs - r["target_arm"])))),
                         repeat_identical=bool(np.array_equal(q1, q2)), ik_s=dt, genesis_ik_s=r["ik_s"],
                         ik_arm=mine.tolist(), ik_arm_hex=[float(x).hex() for x in mine]))
        bits.append([float(x).hex() for x in q1])
    untouched = bool(np.array_equal(live[0], s.env._world.data.qpos) and np.array_equal(live[1], s.env._world.data.qvel))
    summary = dict(cases=len(rows), fk_max_m=max(r["fk_error"][0] for r in rows),
                   fk_max_rad=max(r["fk_error"][1] for r in rows),
                   residual_mujoco_max=[max(r["residual_mujoco"][0] for r in rows), max(r["residual_mujoco"][1] for r in rows)],
                   residual_genesis_max=[max(r["residual_genesis"][0] for r in rows), max(r["residual_genesis"][1] for r in rows)],
                   same_branch=sum(r["same_branch"] for r in rows), repeat_identical=all(r["repeat_identical"] for r in rows),
                   live_state_untouched=untouched, ik_ms_median=1000 * float(np.median([r["ik_s"] for r in rows])),
                   genesis_ik_ms_median=1000 * float(np.median([r["genesis_ik_s"] for r in rows])))
    if a.expect:
        old = json.loads(Path(a.expect).read_text())["bits"]
        summary["fresh_process_identical"] = old == bits
    Path(a.out).write_text(json.dumps(dict(engine="mujoco", instance=a.instance, summary=summary, cases=rows, bits=bits),
                                      indent=1))
    print(json.dumps(summary))
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("genesis", "mujoco"))
    ap.add_argument("--task", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases")
    ap.add_argument("--expect")
    a = ap.parse_args()
    (run_genesis if a.stage == "genesis" else run_mujoco)(a)


if __name__ == "__main__":
    main()
