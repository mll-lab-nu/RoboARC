"""Closed-loop reference tooling for BURVIK (privileged): a Session that records the EXECUTED rows (the targets
Session.applied produces, so the speed clamp never changes them on replay), Cartesian corrections of the arms through
damped least squares on the TCP Jacobians, and the mechanism / success readouts.

    from orc import Orc
    o = Orc(scenes_root)                 # birth, or o.load(state_pkl, rows_npz)
    o.run(rows)                          # base rows as given (clamped and recorded)
    o.run(rows, offset=fn)               # base rows + a TCP offset fn(k) -> {"left": dx3, "right": dx3} (world, m)
    o.track(...)                         # helpers below
"""
import os, sys, pickle, json
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
GRIP_WAIT = float(os.environ.get("BURVIK_GRIP_WAIT", "2"))   # ticks of hand slew a row may lag before the arms wait


def setup(code_root, scenes_root):
    os.environ["IKEA_SCENES"] = scenes_root
    p = os.path.join(code_root, "scripts/ikea_mujoco/shared")
    if p not in sys.path:
        sys.path.insert(0, p)
    import rcb_ikea
    return rcb_ikea


class Orc:
    def __init__(self, R, task="burvik"):
        self.R = R
        self.s = R.Session(task)
        self.m, self.d = self.s.model, self.s.data
        self.rows = []                                     # executed rows
        names = self.s.actuators
        self.arm = {side: [names.index(f"{side}_actuator{i}") for i in range(1, 8)] for side in ("left", "right")}
        self.grip = {side: names.index(f"{side}_actuator8") for side in ("left", "right")}
        self.qadr = {side: [int(self.m.jnt_qposadr[self.m.actuator_trnid[a, 0]]) for a in self.arm[side]] for side in self.arm}
        self.vadr = {side: [int(self.m.jnt_dofadr[self.m.actuator_trnid[a, 0]]) for a in self.arm[side]] for side in self.arm}
        self.site = {side: self.m.site(f"{side}_tcp").id for side in self.arm}
        self.mech = self.s.S.mech

    # ---- state
    def save(self):
        st = self.s.get_state()
        return dict(state=st, rows=list(self.rows))

    def restore(self, snap):
        self.s.set_state(snap["state"])
        self.rows = list(snap["rows"])

    # ---- stepping
    def step(self, row):
        row = np.asarray(row, dtype=np.float64)
        if GRIP_WAIT > 0:
            # the Franka Hand slew (rcb_ikea.GRIPPER_WIDTH_SPEED_M_S): while a hand's command is more than GRIP_WAIT
            # ticks of its slew away from the row's, the arms hold their last targets and only the hand moves, so a
            # grasp or a release completes before the arm goes on (a slow ramp within GRIP_WAIT ticks is not held)
            g = [self.grip["left"], self.grip["right"]]
            tgt = np.clip(row[g], self.s._clo[g], self.s._chi[g])
            while np.any(np.abs(tgt - self.d.ctrl[g]) > GRIP_WAIT * self.s.step_limit[g] + 1e-9):
                w = np.array(self.rows[-1]) if self.rows else self.d.ctrl.copy()
                w[g] = tgt
                self._step(w)
        self._step(row)

    def _step(self, row):
        a = self.s.applied(np.asarray(row, dtype=np.float64))
        self.s.step(a)
        self.rows.append(a.copy())

    def dls(self, side, dx, drot=None, lam=1e-3):
        """joint increment (7) moving the side's TCP by dx (world, m) and its orientation by drot (rotation vector)."""
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site[side])
        cols = self.vadr[side]
        if drot is None:
            J = jp[:, cols]; e = np.asarray(dx)
        else:
            J = np.vstack([jp[:, cols], jr[:, cols]]); e = np.concatenate([dx, drot])
        return J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), e)

    def run(self, rows, offset=None, hold_rot=True):
        """the base rows, each with the arm targets shifted so that the TCPs move by offset(k) (world) relative to where
        the base row would put them: the shift is integrated incrementally (the change of the offset since the last row,
        mapped through the current Jacobian), so a constant offset adds no motion."""
        acc = {side: np.zeros(7) for side in self.arm}
        prev = {side: np.zeros(3) for side in self.arm}
        for k, row in enumerate(rows):
            row = np.array(row, dtype=np.float64)
            if offset is not None:
                off = offset(k)
                mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
                for side, dx in off.items():
                    dx = np.asarray(dx, float)
                    acc[side] += self.dls(side, dx - prev[side], np.zeros(3) if hold_rot else None)
                    prev[side] = dx
            for side in self.arm:
                row[self.arm[side]] += acc[side]
            self.step(row)

    # ---- readouts
    def body(self, n):
        b = self.m.body(n).id
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def active(self):
        return [k for k in self.mech.eq if self.mech.active(k)]

    def metrics(self, key):
        return self.mech.metrics.get(key, {})

    def write(self, path):
        np.savez_compressed(path, ctrl=np.array(self.rows), substeps=np.int64(self.R.SUBSTEPS), timestep=np.float64(self.s.timestep),
                            actuators=np.array(self.s.actuators))
