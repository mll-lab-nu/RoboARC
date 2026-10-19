"""Closed-loop pose servo of a part held rigidly by one arm: each control tick the arm's targets move by the damped
least-squares joint step that drives the part toward a desired pose (world), the other arm following given base rows.

  part pose error: e_p = p* - p, e_w = rotvec(R* R^T); the held TCP's twist: v = e_p + e_w x (p_tcp - p), w = e_w
  dq = clip(DLS(J_tcp, [kp v, kr w]), the speed limit x 0.9) added to the arm's last target.
"""
import numpy as np
import mujoco


def rotvec(Rm):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Rm.reshape(9))
    if q[0] < 0: q = -q
    s = np.linalg.norm(q[1:])
    return np.zeros(3) if s < 1e-12 else q[1:] / s * 2 * np.arctan2(s, q[0])


class PoseServo:
    def __init__(self, orc, side, part, kp=0.08, kr=0.08, vmax=0.0004, wmax=0.002):
        self.o, self.side, self.part = orc, side, part
        self.b = orc.m.body(part).id
        self.kp, self.kr, self.vmax, self.wmax = kp, kr, vmax, wmax
        self.lim = orc.s.step_limit[orc.arm[side]] * 0.9
        self.last_lead = None
        self.lead_max = None          # bound on the target's TCP lead over the actual TCP (m): no windup against contact

    def clamp_lead(self, target):
        """the arm's new joint targets with their TCP lead over the actual joints bounded by lead_max."""
        if self.lead_max is None:
            return target
        o = self.o; m, d = o.m, o.d
        eq = target - d.qpos[o.qadr[self.side]]
        jp = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jp, None, o.site[self.side])
        lead = float(np.linalg.norm(jp[:, o.vadr[self.side]] @ eq)); self.last_lead = lead
        return target if lead <= self.lead_max else d.qpos[o.qadr[self.side]] + eq * (self.lead_max / lead)

    def errors(self, p_star, R_star):
        d = self.o.d
        p = d.xpos[self.b].copy(); Rm = d.xmat[self.b].reshape(3, 3)
        return p_star - p, rotvec(R_star @ Rm.T)

    def tick(self, base_row, p_star, R_star, extra=None):
        """one control tick: base_row for the other arm (and the grippers), this arm servoed; returns (|e_p|, |e_w|)."""
        o, d = self.o, self.o.d
        mujoco.mj_kinematics(o.m, d); mujoco.mj_comPos(o.m, d)
        e_p, e_w = self.errors(p_star, R_star)
        v = self.kp * e_p; w = self.kr * e_w
        nv, nw = np.linalg.norm(v), np.linalg.norm(w)
        if nv > self.vmax: v *= self.vmax / nv
        if nw > self.wmax: w *= self.wmax / nw
        tcp = d.site_xpos[o.site[self.side]]
        v_tcp = v + np.cross(w, tcp - d.xpos[self.b])
        dq = o.dls(self.side, v_tcp, w)
        dq = np.clip(dq, -self.lim, self.lim)
        row = np.array(base_row, dtype=np.float64)
        prev = o.rows[-1] if o.rows else d.ctrl.copy()
        row[o.arm[self.side]] = self.clamp_lead(prev[o.arm[self.side]] + dq)
        if extra:
            for i, val in extra.items():
                row[i] = val
        o.step(row)
        return float(np.linalg.norm(e_p)), float(np.linalg.norm(e_w))
