"""Oracle of the window-change Hedgehog (variation hedgehog_in_cage_window_change) on a frozen Session.

The plan (window_change_plan.json, next to this file) has three stages, one per grasped spike: S8, S10 and a third one
(S5 in the current plan: which spike is within reach depends on where the second set-down lands). In each stage:
  - the free arm motions (to the pregrasp, the straight approach, the retreat) are joint rows, executed as they are;
  - the arm lags its commanded rows (up to ~15 mm at the hand on the long moves), so before every close and every release
    it holds until the arm has caught up: the fingers close around the spike, and the ball is let go from a still hand;
  - the gripper closes until the assisted weld holds;
  - the held route is a list of ball poses in the cage frame. It is solved through IK with the hand-in-ball pose that
    the weld actually took, so the ball itself follows the planned route even when the grasp landed a little off;
  - after a release the ball falls onto its next rest; the next stage starts only if it lies where the plan expects.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PLAN = Path(__file__).with_name("window_change_plan.json")
FREE_STEP_RAD = .02      # largest joint-target change per control step on the free rows
HELD_STEP_RAD = .006     # slower while holding: the held routes pass the bars with 0.05-0.3 mm to spare
SETTLE_STEPS = 100       # 2.5 s for a released ball to come to rest
GOAL_HOLD_STEPS = 120    # 3 s: the task needs 2 s of its final conditions
REST_TOLERANCE = (.0015, np.radians(2.))
ARM_TOLERANCE_RAD = .001  # caught up: every arm joint within this of its command


def _pose(entity):
    p = np.asarray(entity.get_pos(), float).reshape(3)
    q = np.asarray(entity.get_quat(), float).reshape(4)
    return p, Rotation.from_quat(q[[1, 2, 3, 0]])


class WindowChangeOracle:
    def __init__(self, session, plan=None, log=print):
        self.s, self.log = session, log
        self.plan = plan or json.loads(PLAN.read_text())
        robot = session.env._robot
        self.entity, self.group = robot.entity, robot.groups["arm"]
        self.arm_dofs, self.arm_qs = list(robot._indices["arm"][:7]), list(robot._q_indices["arm"][:7])
        from topogym.solvers.genesis_kinematics import GenesisIKSolver
        self.ik = GenesisIKSolver(self.entity)
        world = session.env._world
        self.world, self.ball = world, world.entities["hedgehog_ball"]
        self.cage = _pose(world.entities["hedgehog_cage"])
        self.width = lambda native: session.robot.native_to_width("arm", native)
        self.w_closed = self.width(.8)
        self.q_cmd = np.asarray(session.joint_positions(), float)[:7]
        self.w_cmd = float(np.asarray(session.joint_positions(), float)[7])
        self.report = dict(stages=[])

    # -- low level
    def step_to(self, q, width, max_step):
        q0, w0 = self.q_cmd.copy(), self.w_cmd
        n = max(1, int(np.ceil(np.max(np.abs(np.asarray(q) - q0)) / max_step)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            self.s.step(np.r_[q0 + t * (np.asarray(q) - q0), w0 + t * (width - w0)])
        self.q_cmd, self.w_cmd = np.asarray(q, float).copy(), float(width)

    def rows(self, rows, width, max_step=FREE_STEP_RAD):
        for q in rows:
            self.step_to(q, width, max_step)

    def hold(self, steps):
        for _ in range(steps):
            self.s.step(np.r_[self.q_cmd, self.w_cmd])

    def held(self):
        return bool(self.world.read_state().get("assist_active"))

    def ball_in_cage(self):
        p, r = _pose(self.ball)
        return self.cage[1].inv().apply(p - self.cage[0]), self.cage[1].inv() * r

    def solve(self, hand):
        seed = np.asarray(self.entity.get_qpos(), float).reshape(-1).copy()
        seed[self.arm_qs] = self.q_cmd
        q = self.ik.solve_checked(link_name=self.group.hand_link, position_world_m=hand[0],
                                  orientation_wxyz=hand[1].as_quat()[[3, 0, 1, 2]], joint_dof_indices=self.arm_dofs,
                                  initial_qpos=seed)
        return np.asarray(q, float).reshape(-1)[self.arm_qs]

    # -- stage pieces
    def wait_width(self, max_steps=200):
        """The Robotiq drive is slow on MuJoCo (about 13 mm/s): hold until the measured opening stops changing (it settles
        short of the nominal width: 16.2 instead of 22.8 mm at native 0.6)."""
        last = [float(self.s.gripper_width("arm"))]
        for _ in range(max_steps):
            self.s.step(np.r_[self.q_cmd, self.w_cmd])
            last.append(float(self.s.gripper_width("arm")))
            if len(last) > 4 and max(last[-5:]) - min(last[-5:]) < 5e-5:
                return True
        return False

    def settle(self, max_steps=80):
        """Hold until every arm joint is within ARM_TOLERANCE_RAD of its command."""
        for _ in range(max_steps):
            if np.max(np.abs(np.asarray(self.s.joint_positions(), float)[:7] - self.q_cmd)) < ARM_TOLERANCE_RAD:
                return True
            self.s.step(np.r_[self.q_cmd, self.w_cmd])
        return False

    def close(self, width_from, max_steps=200):
        """Command closed and hold until the assisted weld takes the part (the proof run closed for 3 s)."""
        for k in range(max_steps):
            self.w_cmd = width_from + (self.w_closed - width_from) * min(1., (k + 1) / 10)
            self.s.step(np.r_[self.q_cmd, self.w_cmd])
            if self.held():
                self.hold(8)
                return True
        return self.held()

    def hand_in_ball(self):
        hp, hr = _pose(self.entity.get_link(self.group.hand_link))
        bp, br = _pose(self.ball)
        return br.inv().apply(hp - bp), br.inv() * hr

    def follow(self, route, hib):
        hp, hr = hib
        worst = 0.
        for row in route:
            pc, rc = np.asarray(row[:3]), Rotation.from_quat(row[3:])
            bw = (self.cage[0] + self.cage[1].apply(pc), self.cage[1] * rc)
            self.step_to(self.solve((bw[0] + bw[1].apply(hp), bw[1] * hr)), self.w_closed, HELD_STEP_RAD)
            p, r = self.ball_in_cage()
            worst = max(worst, float(np.linalg.norm(p - pc)))
        return worst

    def at(self, row):
        p, r = self.ball_in_cage()
        dp = float(np.linalg.norm(p - np.asarray(row[:3])))
        da = float((r * Rotation.from_quat(row[3:]).inv()).magnitude())
        return dp, da

    # -- whole flow
    def run(self):
        first = self.plan["stages"][0]
        self.step_to(self.q_cmd, self.width(first["approach_native"]), FREE_STEP_RAD)   # pre-close at home
        self.wait_width()
        for stage in self.plan["stages"]:
            entry = dict(name=stage["name"], spike=stage["spike"])
            self.report["stages"].append(entry)
            dp, da = self.at(stage["held_route"][0])
            entry["start_offset"] = dict(position_mm=round(dp * 1e3, 2), rotation_deg=round(np.degrees(da), 2))
            if dp > REST_TOLERANCE[0] or da > REST_TOLERANCE[1]:
                entry["result"] = "ball_not_at_planned_start"
                self.log(entry)
                return self.report
            w_open = self.width(stage["approach_native"])
            self.rows(stage["to_pregrasp"], w_open)
            self.wait_width()
            self.rows(stage["approach"], w_open)
            self.settle()
            if not self.close(w_open):
                entry["result"] = "weld_not_engaged"
                self.log(entry)
                return self.report
            hib = self.hand_in_ball()
            entry["route_tracking_max_mm"] = round(self.follow(stage["held_route"], hib) * 1e3, 2)
            dp, da = self.at(stage["held_route"][-1])
            entry["end_offset"] = dict(position_mm=round(dp * 1e3, 2), rotation_deg=round(np.degrees(da), 2))
            if stage["release"]:
                self.settle()
                self.step_to(self.q_cmd, w_open, FREE_STEP_RAD)
                self.wait_width()
                self.rows(stage["retreat"], w_open)
                self.hold(SETTLE_STEPS)
                p, r = self.ball_in_cage()
                entry["rest"] = dict(position_m=p.tolist(), quaternion_xyzw=r.as_quat().tolist())
            else:
                self.hold(GOAL_HOLD_STEPS)
            entry["result"] = "done"
            entry["status"] = {k: v for k, v in self.s.status().items() if k in ("success", "episode_end", "sim_steps")}
            self.log(entry)
        self.report["success"] = bool(self.s.status().get("success"))
        return self.report


def run_oracle(session, log=print):
    """Entry point for run-oracle code and the Harbor bridge."""
    return WindowChangeOracle(session, log=log).run()
