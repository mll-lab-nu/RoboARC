#!/usr/bin/env python3
"""stegesund closed-loop reference under the Franka Hand slew (see README.md).

The author's speed-limited demo is the phase guide. Its rows are executed through rcb_ikea's clamp (arm joints and, now,
the hand at 0.455357 units per tick), plus a per-arm joint offset that the closed-loop corrections leave behind:
  - grip wait: a demo row is held until both hands' applied command equals the row's, so a grasp or release completes
    before the arms go on (the demo's 0.638-unit ramps take 1.4x the ticks);
  - guide tracks: in each track window the held part's pose relative to its reference part is steered onto the guide's
    (the demo replayed without the hand slew, where it assembles the stand) through the holding arm's TCP twist and a
    damped least-squares Jacobian; the increments add up in the arm's offset;
  - TCP tracks ('tcp' as the part): the TCP steered onto the guide's TCP pose in a part's frame (grasp approaches), or
    onto the guide's TCP path moved by one constant rigid correction ('fixed:<part>', the hex-key drives);
  - perturb: a tiny constant joint-target shift over a row range, a search knob where the demo itself is chaotic;
  - drives: a hex-key drive that stops short gets the demo's last turn repeated (drive_fix);
  - the offset is ramped out once the arm's hand has opened again after the window (ramp_delay ticks later).
The track windows and the chain of resumed runs that made stegesund.npz are in recipe/.
Every stage ends with a checkpoint (all executed rows + the full state); --from resumes from one.
The recorded rows are the EXECUTED targets (data.ctrl after the clamp), so a replay applies every row unchanged.

  build.py --out DIR [--from STAGE] [--to STAGE] [--set key=value ...]      (DIR/ref.npz)
"""
from __future__ import annotations
import argparse, json, math, os, pickle, sys, time
import numpy as np
import mujoco
import rcb_ikea as R

ARM = {'left': np.arange(0, 7), 'right': np.arange(8, 15)}
GRIP = {'left': 7, 'right': 15}
GL = [7, 15]
PARTS = ['half_A', 'half_B', 'shelf', 'rod', 'bolt_1', 'bolt_2', 'bolt_3', 'bolt_4', 'hex_key']
DEMO = os.environ.get('STEG_DEMO', 'runs/demo.npz')
GUIDE = os.environ.get('STEG_GUIDE', 'runs/guide.npz')

CFG = dict(tk_kp=0.02, tk_kr=0.02, tk_cap=0.5, ramp=400, grip_wait=1, lam=1e-3,
           noslew=0, perturb=0.0, perturb_j=0, perturb_from=47500, perturb_to=10**9,
           drives=0, d_seat=0.0003, d_turns=3, d_hold=4000, ramp_delay=0)
# (demo rows [r0, r1), arm, part, reference part or 'world')
TRACKS: list = []
STAGES = [('carry', 44000), ('press0', 47500), ('press', 60000), ('shelf', 72000), ('bolt1in', 92000), ('drive1', 131000),
          ('bolt3in', 152000), ('drive3', 183000), ('flip', 223000), ('bolt2in', 241000), ('drive2', 274000),
          ('bolt4in', 293000), ('drive4', 323000), ('stand', 347000), ('end', None)]
# the guide's seat-weld events (demo rows): bolt -> (armed, welded)
GUIDE_EVENTS = {'bolt_1': (89761, 120413), 'bolt_3': (158542, 172840), 'bolt_2': (239636, 265493), 'bolt_4': (291219, 313202)}


def qmul(a, b):
    w, x, y, z = a; W, X, Y, Z = b
    return np.array([w*W-x*X-y*Y-z*Z, w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qmat(q):
    M = np.zeros(9); mujoco.mju_quat2Mat(M, np.asarray(q, dtype=np.float64)); return M.reshape(3, 3)


def rotvec(q):
    q = q / np.linalg.norm(q)
    if q[0] < 0: q = -q
    ang = 2 * math.atan2(np.linalg.norm(q[1:]), q[0])
    return q[1:] / (np.linalg.norm(q[1:]) + 1e-300) * ang


class Builder:
    def __init__(self, out, cfg, tracks):
        self.out, self.cfg = out, cfg; os.makedirs(out, exist_ok=True)
        self.s = R.Session('stegesund'); self.m, self.d = self.s.model, self.s.data
        if cfg.get('noslew'):                                 # a study only: the pre-rule clamp (the guide's conditions)
            self.s.step_limit[GL] = np.inf; self.s._limited = np.isfinite(self.s.step_limit)
        self.demo = np.load(DEMO)['ctrl']
        if os.path.isdir(GUIDE):                              # the guide as .npy files, memory-mapped (shared page cache)
            self.gpos, self.gquat, self.gtcp = (np.load(os.path.join(GUIDE, k + '.npy'), mmap_mode='r') for k in ('pos', 'quat', 'tcp'))
        else:
            z = np.load(GUIDE); self.gpos, self.gquat, self.gtcp = z['pos'], z['quat'], z['tcp']
        self.pid = {p: self.m.body(p).id for p in PARTS}
        self.site = {a: self.m.site(a + '/tcp').id for a in ARM}
        self.dof = {a: np.array([int(self.m.jnt_dofadr[self.m.joint(f'{a}/joint{k}').id]) for k in range(1, 8)]) for a in ARM}
        self.jp = np.zeros((3, self.m.nv)); self.jr = np.zeros((3, self.m.nv))
        self.track_at = {}
        for tr in tracks:
            r0, r1, arm, part, ref = tr[:5]; opt = tr[5] if len(tr) > 5 else {}
            for i in range(r0, r1):
                self.track_at.setdefault(i, []).append((arm, part, ref, opt))
        self.track_end = {}
        for tr in tracks:
            if not (len(tr) > 5 and tr[5].get('no_ramp')):
                self.track_end.setdefault(tr[1], []).append(tr[2])
        self.ex: list = []; self.i = 0; self.trace = []; self.pidv = np.array([self.pid[p] for p in PARTS])
        self.fq = [self.m.joint(j).qposadr[0] for j in ('left/finger_joint1', 'left/finger_joint2', 'right/finger_joint1', 'right/finger_joint2')]
        self.off = {a: np.zeros(7) for a in ARM}; self.ramps = []; self.pending = {}; self.log = []
        self.keep = {a: np.zeros(7) for a in ARM}             # whole turns on j7 left by a drive fix (never ramped)
        self.rewind = {}; self.drives_done = set(); self.fixed = {}
        self.drive_at = {v[2]: k for k, v in DRIVES.items()}

    # ---- state
    def save(self, name):
        st = dict(s=self.s.get_state(), ex=list(self.ex), i=self.i, off={a: v.copy() for a, v in self.off.items()},
                  ramps=list(self.ramps), pending=dict(self.pending), log=list(self.log), cfg=self.cfg,
                  keep={a: v.copy() for a, v in self.keep.items()}, rewind=dict(self.rewind), drives_done=set(self.drives_done),
                  fixed=dict(self.fixed))
        with open(os.path.join(self.out, f'ck_{name}.pkl'), 'wb') as f:
            pickle.dump(st, f)
        if self.trace:
            np.savez_compressed(os.path.join(self.out, f'trace_{name}.npz'), row=np.array([t[0] for t in self.trace]),
                                tick=np.array([t[1] for t in self.trace]), pos=np.array([t[2] for t in self.trace]),
                                quat=np.array([t[3] for t in self.trace]), tcp_pos=np.array([t[4] for t in self.trace]),
                                tcp_quat=np.array([t[5] for t in self.trace]), fing=np.array([t[6] for t in self.trace]))
            self.trace = []

    def load(self, path):
        with open(path, 'rb') as f:
            st = pickle.load(f)
        self.s.set_state(st['s']); self.ex = list(st['ex']); self.i = st['i']; self.off = st['off']
        self.ramps = st['ramps']; self.pending = st['pending']; self.log = st['log']
        self.keep = st.get('keep', self.keep); self.rewind = st.get('rewind', {}); self.drives_done = st.get('drives_done', set())
        self.fixed = st.get('fixed', {})

    def note(self, **kw):
        kw.update(row=self.i, tick=len(self.ex)); self.log.append(kw); print(json.dumps(kw, default=str), flush=True)

    # ---- measurements
    def rel(self, part, ref, pos=None, quat=None):
        """part pose in ref's frame: (p, q wxyz); from the guide's arrays when pos/quat are given"""
        if pos is None:
            pos, quat = self.d.xpos, self.d.xquat
            P = lambda n: (pos[self.pid[n]], quat[self.pid[n]])
        else:
            P = lambda n: (pos[PARTS.index(n)], quat[PARTS.index(n)])
        pp, qp = P(part)
        if ref == 'world':
            return np.array(pp, dtype=np.float64), np.array(qp, dtype=np.float64)
        pr, qr = P(ref)
        return qmat(qr).T @ (pp - pr), qmul(qconj(qr), qp)

    def jac(self, arm):
        mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site[arm])
        return np.vstack([self.jp[:, self.dof[arm]], self.jr[:, self.dof[arm]]])

    def track_dq(self, i, arm, part, ref, opt=None):
        """joint increment of `arm` that moves `part` by kp x its error against the guide (in ref's frame) and turns it
        by kr x its angle error, through the TCP twist (lever arm from the part origin) and a damped LS Jacobian"""
        c = self.cfg; opt = opt or {}
        if part == 'tcp':
            return self.approach_dq(i, arm, ref, opt)
        gp, gq = self.rel(part, ref, self.gpos[i], self.gquat[i])
        p, q = self.rel(part, ref)
        Rr = np.eye(3) if ref == 'world' else self.d.xmat[self.pid[ref]].reshape(3, 3)
        pw = np.asarray(opt.get('pw', (1., 1., 1.)))          # per-axis weights of the position error, ref frame
        e = (gp - p) * pw
        if 'emax' in opt:                                     # one-sided errors (ref frame), e.g. never pull a part back out
            e = np.minimum(e, np.asarray(opt['emax'])); e = np.maximum(e, np.asarray(opt['emin']))
        dp_w = Rr @ e * opt.get('kp', c['tk_kp'])
        rv = rotvec(qmul(gq, qconj(q / np.linalg.norm(q))))
        if 'free_axis' in opt:                                # no correction about this axis (ref frame), e.g. a bolt's spin
            ax = np.asarray(opt['free_axis'], dtype=np.float64); ax = ax / np.linalg.norm(ax)
            rv = rv - ax * (rv @ ax)
        om = Rr @ rv * opt.get('kr', c['tk_kr'])
        v = dp_w + np.cross(om, self.d.site_xpos[self.site[arm]] - self.d.xpos[self.pid[part]])
        J = self.jac(arm)
        dq = J.T @ np.linalg.solve(J @ J.T + c['lam'] ** 2 * np.eye(6), np.r_[v, om])
        cap = self.s.step_limit[ARM[arm]] * c['tk_cap']
        return np.clip(dq, -cap, cap)

    def approach_dq(self, i, arm, ref, opt):
        """joint increment of `arm` that moves its TCP by kp x its error against the guide's TCP pose in ref's frame (an
        approach to a part, or a push with the closed hand), rotation by kr x the angle error"""
        c = self.cfg; k = 0 if arm == 'left' else 1
        if ref.startswith('fixed:'):
            # the guide's TCP path moved by a constant rigid correction: the part's actual pose against the guide's at
            # the window's first tick (e.g. a hex-key drive centred on the bolt where it really is)
            part = ref[6:]; key = (arm, part, opt.get('r0'))
            if key not in self.fixed:
                pid = self.pid[part]
                Ra, pa = self.d.xmat[pid].reshape(3, 3).copy(), self.d.xpos[pid].copy()
                Rg, pg = qmat(self.gquat[i, PARTS.index(part)]), np.array(self.gpos[i, PARTS.index(part)])
                Dr, Dp = np.eye(3), np.zeros(3)
                if 'held' in opt:
                    # the held tool's grasp against the guide's: the tool (not the hand) follows the corrected path
                    h = self.pid[opt['held']]
                    Rk, pk = self.d.xmat[h].reshape(3, 3), self.d.xpos[h]
                    Rt0, pt0 = self.d.site_xmat[self.site[arm]].reshape(3, 3), self.d.site_xpos[self.site[arm]]
                    Rkg, pkg = qmat(self.gquat[i, PARTS.index(opt['held'])]), np.array(self.gpos[i, PARTS.index(opt['held'])])
                    Rtg, ptg = qmat(self.gtcp[i, k, 3:]), np.array(self.gtcp[i, k, :3])
                    Ho_R, Ho_p = Rk.T @ Rt0, Rk.T @ (pt0 - pk)          # tcp in the tool frame, ours
                    Hg_R, Hg_p = Rkg.T @ Rtg, Rkg.T @ (ptg - pkg)       # the guide's
                    Dr = Hg_R.T @ Ho_R; Dp = Hg_R.T @ (Ho_p - Hg_p)      # D = Hg^-1 Ho
                self.fixed[key] = (Ra @ Rg.T, pa - Ra @ Rg.T @ pg, Dr, Dp)
            Rc, tc, Dr, Dp = (self.fixed[key] + (np.eye(3), np.zeros(3)))[:4]
            Rtg = qmat(self.gtcp[i, k, 3:]); ptg = np.array(self.gtcp[i, k, :3])
            gp = Rc @ (Rtg @ Dp + ptg) + tc; gR = Rc @ Rtg @ Dr
            Rt = self.d.site_xmat[self.site[arm]].reshape(3, 3)
            v = (gp - self.d.site_xpos[self.site[arm]]) * opt.get('kp', c['tk_kp'])
            qe = np.zeros(4); mujoco.mju_mat2Quat(qe, (gR @ Rt.T).reshape(-1))
            om = rotvec(qe) * opt.get('kr', c['tk_kr'])
            J = self.jac(arm)
            dq = J.T @ np.linalg.solve(J @ J.T + c['lam'] ** 2 * np.eye(6), np.r_[v, om])
            cap = self.s.step_limit[ARM[arm]] * c['tk_cap']
            return np.clip(dq, -cap, cap)
        if ref == 'world':
            Rr, pr = np.eye(3), np.zeros(3); gRr, gpr = np.eye(3), np.zeros(3)
        else:
            Rr, pr = self.d.xmat[self.pid[ref]].reshape(3, 3), self.d.xpos[self.pid[ref]]
            gRr, gpr = qmat(self.gquat[i, PARTS.index(ref)]), self.gpos[i, PARTS.index(ref)]
        gp = gRr.T @ (self.gtcp[i, k, :3] - gpr); gR = gRr.T @ qmat(self.gtcp[i, k, 3:])
        Rt = self.d.site_xmat[self.site[arm]].reshape(3, 3)
        p = Rr.T @ (self.d.site_xpos[self.site[arm]] - pr); Rc = Rr.T @ Rt
        pw = np.asarray(opt.get('pw', (1., 1., 1.)))
        v = Rr @ ((gp - p) * pw) * opt.get('kp', c['tk_kp'])
        qe = np.zeros(4); mujoco.mju_mat2Quat(qe, (gR @ Rc.T).reshape(-1))
        om = Rr @ rotvec(qe) * opt.get('kr', c['tk_kr'])
        J = self.jac(arm)
        dq = J.T @ np.linalg.solve(J @ J.T + c['lam'] ** 2 * np.eye(6), np.r_[v, om])
        cap = self.s.step_limit[ARM[arm]] * c['tk_cap']
        return np.clip(dq, -cap, cap)

    # ---- acting
    def target(self, i):
        row = self.demo[i].copy()
        if self.cfg['perturb'] and self.cfg['perturb_from'] <= i < self.cfg['perturb_to']:
            row[int(self.cfg['perturb_j'])] += self.cfg['perturb']      # a search knob: a tiny constant target shift
        for arm in ARM:
            row[ARM[arm]] += self.off[arm] + self.keep[arm]
        return row

    def step(self, row):
        self.s.step(row); self.ex.append(self.d.ctrl.copy())
        if len(self.ex) % 50 == 0:
            tq = np.zeros((2, 4))
            for k, a in enumerate(ARM):
                mujoco.mju_mat2Quat(tq[k], self.d.site_xmat[self.site[a]])
            self.trace.append((self.i, len(self.ex), self.d.xpos[self.pidv].copy(), self.d.xquat[self.pidv].copy(),
                               np.array([self.d.site_xpos[self.site[a]] for a in ARM]), tq, self.d.qpos[self.fq].copy()))

    def tick_ramps(self):
        t = len(self.ex); keep = []
        for arm, t0, n, o0 in self.ramps:
            k = min(1., max(0., (t - t0) / n))
            self.off[arm] = o0 * (1 - k)
            if k < 1: keep.append((arm, t0, n, o0))
        self.ramps = keep

    def control(self, i):
        self.tick_ramps()
        for arm, part, ref, opt in self.track_at.get(i, ()):
            if 'until_weld' in opt and self.s.S.mech.status()[opt['until_weld']]['welded']:
                continue                                      # the bolt has seated: stop steering
            self.off[arm] = self.off[arm] + self.track_dq(i, arm, part, ref, opt)

    def run_to(self, end):
        demo = self.demo; end = len(demo) if end is None else min(end, len(demo))
        while self.i < end:
            i = self.i
            if self.cfg['drives'] and i in self.drive_at and i not in self.drives_done:
                self.drives_done.add(i); drive_fix(self, self.drive_at[i]); continue
            for arm, (air, take) in list(self.rewind.items()):
                if i == air:                                  # key in the air: ramp the offset to whole turns on j7
                    o = self.off[arm].copy(); keep = np.zeros(7)
                    keep[6] = 2 * math.pi * round(o[6] / (2 * math.pi))
                    self.keep[arm] = self.keep[arm] + keep; self.off[arm] = o - keep
                    n = max(1, min(self.cfg['ramp'], take - air - 50))
                    self.ramps.append((arm, len(self.ex), n, self.off[arm].copy())); del self.rewind[arm]
                    self.note(ev='rewind', arm=arm, keep_turns=round(keep[6] / (2 * math.pi)))
            for arm in self.track_end.get(i, ()):
                self.pending[arm] = True
            for arm in list(self.pending):
                g = GRIP[arm]
                if (i >= 40 and demo[i][g] == demo[i - 1][g] and demo[i - 40][g] < demo[i][g] - 1
                        and abs(self.d.ctrl[g] - demo[i][g]) < 1e-12):
                    # ramp_delay: the hand first moves away from the part it let go (a ramp right at the release
                    # can drag the part, e.g. twist a hex key in the other hand)
                    self.ramps.append((arm, len(self.ex) + int(self.cfg['ramp_delay']), self.cfg['ramp'], self.off[arm].copy()))
                    del self.pending[arm]
                    self.note(ev='ramp', arm=arm)
            self.control(i)
            self.step(self.target(i))
            while self.cfg['grip_wait'] and np.any(np.abs(demo[i][GL] - self.d.ctrl[GL]) > 1e-12):
                self.control(i)
                self.step(self.target(i))
            self.i += 1


# hex-key drives (demo rows): bolt -> (driving arm, spin start, spin end, guide weld row, row from which the key is out of
# the socket and in the air, row by which the other hand grasps it)
DRIVES = {'bolt_1': ('right', 106849, 117483, 120413, 121200, 123500),
          'bolt_3': ('left', 162050, 172684, 172840, 173900, 175300),
          'bolt_2': ('left', 251794, 265400, 265493, 266900, 267900),
          'bolt_4': ('right', 302536, 315002, 313202, 315300, 316700)}


def drive_fix(b, bolt):
    """Closed-loop end of a hex-key drive (drives=1). At the demo's spin end, if the bolt has not reached the guide's
    seated position along its hole axis (the key slips in the socket on our slightly different bolt pose), the demo's
    last turn of the driving arm is repeated (j7 + 2 pi per repeat, the same direction), the key following the bolt's
    advance along the hole axis (DLS), until the bolt is within d_seat of the guide's seated position or welded, at
    most d_turns turns; then the row is held until the seat weld latches (at most d_hold ticks). The demo resumes at the
    spin end with the extra rotation in the arm's offset; the residual (off minus whole turns on j7) is ramped out
    while the key is in the air, before the other hand takes it, so the hand is back on the demo's pose mod 2 pi."""
    c = b.cfg; demo = b.demo; arm, s0, s1, wrow, air, take = DRIVES[bolt]
    A = ARM[arm]; j7 = A[6]
    mech = b.s.S.mech
    sh, bo = b.pid['shelf'], b.pid[bolt]
    gS = qmat(b.gquat[wrow, 2]); gseat = gS.T @ (b.gpos[wrow, PARTS.index(bolt)] - b.gpos[wrow, 2])
    gS0 = qmat(b.gquat[s0, 2]); gstart = gS0.T @ (b.gpos[s0, PARTS.index(bolt)] - b.gpos[s0, 2])
    ax = gseat - gstart; ax = ax / np.linalg.norm(ax)          # the hole axis (shelf frame), the bolt's advance direction

    def axial():
        Rs = b.d.xmat[sh].reshape(3, 3)
        return float((Rs.T @ (b.d.xpos[bo] - b.d.xpos[sh])) @ ax), Rs
    sgn = np.sign(demo[s1, j7] - demo[s0, j7])
    k = s1 - 1
    while abs(demo[k, j7] - (demo[s1, j7] - sgn * 2 * math.pi)) > 0.01 and k > s0:
        k -= 1
    turn = list(range(k, s1))
    a0, _ = axial(); short = float(gseat @ ax) - a0
    b.note(ev='drive', bolt=bolt, short_mm=round(short * 1e3, 3), welded=mech.status()[bolt]['welded'], turn_rows=len(turn))
    if short > c['d_seat'] and not mech.status()[bolt]['welded']:
        base = b.d.ctrl.copy() - demo[s1]                    # the arms' current offsets (both arms)
        follow = np.zeros(7); m = 0; done = False; prev, _ = axial()
        while m < c['d_turns'] and not done:
            m += 1
            for r in turn:
                row = demo[r] + base; row[j7] += sgn * 2 * math.pi * m; row[A] += follow
                b.step(row)
                cur, Rs = axial()
                v = Rs @ ax * (cur - prev); prev = cur        # the key follows the bolt along the hole axis
                J = b.jac(arm)
                follow = follow + J.T @ np.linalg.solve(J @ J.T + c['lam'] ** 2 * np.eye(6), np.r_[v, 0, 0, 0])
                if float(gseat @ ax) - cur <= c['d_seat'] or mech.status()[bolt]['welded']:
                    done = True; break
        hold = b.d.ctrl.copy(); n = 0
        while not mech.status()[bolt]['welded'] and n < c['d_hold']:
            b.step(hold); n += 1
        cur, _ = axial()
        b.note(ev='drive_done', bolt=bolt, turns=m, short_mm=round((float(gseat @ ax) - cur) * 1e3, 3), hold=n,
               welded=mech.status()[bolt]['welded'])
        for a in ARM:
            b.off[a] = b.d.ctrl[ARM[a]] - demo[s1][ARM[a]] - b.keep[a]
        b.rewind[arm] = (air, take)
    b.i = s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--from', dest='frm', default=None)
    ap.add_argument('--to', default=None); ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--tracks', default=None, help='JSON file with the track list (default: TRACKS)')
    a = ap.parse_args()
    cfg = dict(CFG)
    for kv in a.set:
        k, v = kv.split('='); cfg[k] = type(CFG[k])(float(v))
    tracks = json.load(open(a.tracks)) if a.tracks else TRACKS
    b = Builder(a.out, cfg, tracks)
    names = [s for s, _ in STAGES]; k0 = 0
    if a.frm:
        b.load(a.frm); k0 = next(k for k, (n, e) in enumerate(STAGES) if e is None or e > b.i)
    t0 = time.time()
    for name, end in STAGES[k0:]:
        b.run_to(end); b.save(name)
        rep = b.s.success()['assembled']
        k = min(b.i, len(b.demo)) - 1
        dev = {p: round(float(np.linalg.norm(b.d.xpos[b.pid[p]] - b.gpos[k, PARTS.index(p)])) * 1e3, 2) for p in PARTS}
        rdev = {}
        for p in PARTS:
            qe = qmul(b.gquat[k, PARTS.index(p)], qconj(b.d.xquat[b.pid[p]]))
            rdev[p] = round(math.degrees(np.linalg.norm(rotvec(qe))), 3)
        st = b.s.S.mech.status()
        ev_ok = all(st[f]['armed'] == (k >= a0) and st[f]['welded'] == (k >= w0) for f, (a0, w0) in GUIDE_EVENTS.items())
        score = max(v for p, v in dev.items() if p != 'hex_key')
        rel = {}                                              # frame members vs the shelf: mm + 250 mm x rotation (rad)
        for p in ('half_A', 'half_B'):
            gp_, gq_ = b.rel(p, 'shelf', b.gpos[k], b.gquat[k]); pp_, qq_ = b.rel(p, 'shelf')
            rel[p] = round(float(np.linalg.norm(pp_ - gp_)) * 1e3 + 250 * float(np.linalg.norm(rotvec(qmul(gq_, qconj(qq_))))), 3)
        b.note(ev='stage', stage=name, wall=round(time.time() - t0, 1), score=score, ev_ok=ev_ok, rel=rel, dev_mm=dev, rdev_deg=rdev,
               rep={k: v for k, v in rep.items() if k in ('half_B', 'shelf', 'rod', 'stand_upright_tilt_deg')},
               bolts={k: rep[k]['ok'] for k in rep if k.startswith('bolt')}, fast=b.s.S.mech.status())
        if a.to == name:
            break
    if not a.to or a.to == 'end':
        fc = b.s.final_check()
        b.note(ev='final', success=fc['success'], report=fc['end']['report'], term=fc['end']['terminal'],
               settled=fc['settled']['success'], sha=fc['state_sha'], settled_sha=fc['settled_sha'],
               slewed_ticks=b.s.slewed_ticks)
        b.s.save_trajectory(np.array(b.ex), os.path.join(a.out, 'ref.npz'))


if __name__ == '__main__':
    main()
