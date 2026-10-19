#!/usr/bin/env python3
"""Build the variera closed-loop reference under the Franka Hand slew (see README.md).

The old reference (the author's demo time-dilated to the joint-speed limits + the release tail, sha256 a2653e57...) is
the phase source: its rows are executed with a per-arm offset, and a row whose hand command the slewed hand has not
reached yet is held (grip_wait), so a grasp or release completes before the arm goes on. The guide (rec_guide.py) is the
same rows replayed without the hand slew, where they pass: per source row, the parts' and both TCPs' world poses.
  - held tracks (TRACKS): while a hand carries a side frame, the frame's true pose is servoed toward the guide's (world)
    by damped-least-squares IK through the TCP's lever arm; once the frame rests on the bench, only its x, y and yaw
    (no push into the bench). The increments stay in the arm's offset and are ramped out after the release (RAMPS).
  - soft releases (SOFT): a side frame stands on an 18 mm base; the slowed opening kicks it into rocking and it settles
    up to 4 mm off. Instead the hand opens to 5 units below the zero-force command (the servo command whose target is
    the fingers' present opening), holds 200 ticks, opens 0.5 mm per finger beyond it and holds 400 ticks, then opens.
  - finish drives (FINISH): where the demo's screwdriver spin ends with the screw head not seated, joint 7 keeps
    spinning (0.0065 rad per tick) while the TCP follows the screw down by the pitch and the tip is kept on the true
    screw axis, until the seat gap is below 0.03 mm; then joint 7's target is set to the joint (zero torque) before the
    demo lifts. The extra turns stay in the offset; joint 7's is ramped to a whole quarter turn (the cross recess)
    during the transfer.
  - release compensation (release_search, comp=1) was tried and is off: moving a held frame on the bench makes it rock.
Rows are the EXECUTED targets (data.ctrl after the clamp), so the clamp never changes a row on replay.

  build.py --guide G.npz --src OLD.npz --out DIR [--upto ROW] [--resume CK.pkl] [--set k=v ...]
"""
from __future__ import annotations
import argparse, json, math, os, pickle, sys, time
import numpy as np
from vlib import Rig, Guide, ARM, GRIP, held_track_dq

CFG = dict(tk_kp=0.02, tk_kr=0.02, tk_cap=0.5, ramp=400, grip_wait=1,
           comp_iters=5, comp_tol=0.08e-3, lead=800, settle=300, comp=0,
           soft=1, sr_light=5.0, sr_h1=200, sr_gap=0.5, sr_h2=400, logevery=0,
           fin=1, fin_tol=0.03e-3, fin_rate=0.0065, fin_max=6000, fin_kc=0.05, fin_push=5e-9, fin_push_rate=2.5e-7,
           fin_push_max=0.5e-3, fin_untorque=100, fin_ramp_after=1500, fin_ramp=2000)
# held tracks: (src rows [r0, r1), arm, part, frame)
TRACKS = [(1900, 4646, 'left', 'frame_1', 'world'),
          (8300, 10734, 'right', 'frame_2', 'world')]
# release compensation: (src row where the hand starts to open, arm, part)
RELEASES = [(4646, 'left', 'frame_1'), (10734, 'right', 'frame_2')]
# soft releases: (src row where the hand starts to open, arm, last src row of the demo's opening)
SOFT = [(4646, 'left', 4846), (10734, 'right', 10934)]
# closed-loop finish of the screwdriver drives (src row where the demo's left joint-7 spin ends)
FINISH = [73903, 120466, 173337, 227337]
# offsets ramped out (src row, arm): after the release's opening
RAMPS = [(5100, 'left'), (11200, 'right')]


class Builder:
    def __init__(self, a, cfg):
        self.out = a.out; self.cfg = cfg; os.makedirs(self.out, exist_ok=True)
        self.g = Rig(); self.G = Guide(a.guide)
        import rcb_ikea as R
        self.demo = R.load_trajectory(a.src, 16); self.i = 0
        self.track_at = {}
        for r0, r1, arm, obj, fr in TRACKS:
            for i in range(r0, r1):
                self.track_at.setdefault(i, []).append((arm, obj, fr))
        self.rel_at = {r: (arm, obj) for r, arm, obj in RELEASES}
        self.ramp_at = {r: arm for r, arm in RAMPS}
        self.soft_at = {r: (arm, r1) for r, arm, r1 in SOFT}
        self.fin_at = set(FINISH); self.fin_ramp_at = {}
        self.off = {a: np.zeros(7) for a in ARM}; self.ramps = []
        self.comp = {}; self.log = []; self.t0 = time.time(); self.done_rel = set()

    # ---- state
    def state(self):
        return dict(fin_ramp_at=dict(self.fin_ramp_at), g=self.g.get(), i=self.i, off={a: v.copy() for a, v in self.off.items()}, ramps=list(self.ramps),
                    comp={k: v.copy() for k, v in self.comp.items()}, log=list(self.log), done_rel=set(self.done_rel))

    def restore(self, st):
        self.g.set(st['g']); self.i = st['i']; self.off = {a: v.copy() for a, v in st['off'].items()}
        self.ramps = list(st['ramps']); self.comp = {k: v.copy() for k, v in st['comp'].items()}
        self.log = list(st['log']); self.done_rel = set(st['done_rel']); self.fin_ramp_at = dict(st.get('fin_ramp_at', {}))

    def save(self, name):
        pickle.dump(self.state(), open(os.path.join(self.out, f'ck_{name}.pkl'), 'wb'))

    def note(self, **kw):
        kw.update(row=self.i, tick=len(self.g.ex), wall=round(time.time() - self.t0, 1))
        self.log.append(kw); print(json.dumps(kw, default=str), flush=True)

    # ---- acting
    def target(self, i):
        row = self.demo[i].copy()
        for arm in ARM:
            row[ARM[arm]] += self.off[arm]
        return row

    def tick_ramps(self):
        t = len(self.g.ex); keep = []
        for r in self.ramps:
            arm, t0, n, o0 = r[:4]; o1 = r[4] if len(r) > 4 else np.zeros(7)
            k = min(1., (t - t0) / n)
            self.off[arm] = o0 + (o1 - o0) * k
            if k < 1: keep.append(r)
        self.ramps = keep

    def tracks(self, i, note=False):
        c = self.cfg
        for arm, obj, fr in self.track_at.get(i, ()):
            cap = self.g.lim[ARM[arm]] * c['tk_cap']
            dq, ep, ea = held_track_dq(self.g, self.G, i, arm, obj, fr, c['tk_kp'], c['tk_kr'], cap,
                                       dp=self.comp.get(obj))
            self.off[arm] += dq
            if note:
                self.note(ev='track', arm=arm, obj=obj, err_mm=round(1e3 * ep, 3), err_deg=round(ea, 3))

    def row(self, i):
        if i in self.ramp_at:
            arm = self.ramp_at[i]
            self.ramps = [r for r in self.ramps if r[0] != arm]
            self.ramps.append((arm, len(self.g.ex), self.cfg['ramp'], self.off[arm].copy()))
            self.note(ev='ramp', arm=arm, off=np.round(self.off[arm], 5).tolist())
        self.tick_ramps(); self.tracks(i, note=(i % 400 == 0 and i in self.track_at))
        g = self.g; demo = self.demo
        g.step(self.target(i))
        gl = [GRIP[a] for a in ARM]
        while self.cfg['grip_wait'] and np.any(np.abs(demo[i][gl] - g.ctrl[gl]) > 1e-12):
            self.tick_ramps(); self.tracks(i)
            g.step(self.target(i))

    def soft_release(self, i, arm, r1):
        """Open the hand in steps instead of the demo's ramp (the arms hold row i): to `sr_light` command units below
        the zero-force command (the servo's command whose target is the fingers' present opening), hold sr_h1 ticks;
        to sr_gap mm per finger beyond it, hold sr_h2 ticks (the fingers stay as stops around the part while it
        settles); then to the demo's open command (row r1). The demo rows i..r1 move no arm joint."""
        g = self.g; c = self.cfg; m = g.m; k = GRIP[arm]
        gain, b0, kL = float(m.actuator_gainprm[k, 0]), float(m.actuator_biasprm[k, 0]), -float(m.actuator_biasprm[k, 1])
        L = float(g.d.actuator_length[k]); z0 = (kL * L - b0) / gain
        prof = [(z0 - c['sr_light'], c['sr_h1']), (z0 + kL * c['sr_gap'] * 1e-3 / gain, c['sr_h2']), (self.demo[r1][k], 0)]
        for cmd, hold in prof:
            row = self.target(i); row[k] = cmd
            while abs(g.ctrl[k] - cmd) > 1e-12:
                self.tick_ramps(); g.step(row); row = self.target(i); row[k] = cmd
            for _ in range(hold):
                self.tick_ramps(); g.step(row); row = self.target(i); row[k] = cmd
        self.note(ev='soft_release', arm=arm, z0=round(z0, 3), prof=[(round(a, 3), b) for a, b in prof])
        self.i = r1 + 1

    def run(self, end):
        while self.i < end:
            i = self.i
            if self.cfg['soft'] and i in self.soft_at:
                self.soft_release(i, *self.soft_at[i]); continue
            for r, (arm, obj) in self.rel_at.items():
                if self.cfg['comp'] and i == r - self.cfg['lead'] and r not in self.done_rel:
                    self.release_search(r, arm, obj)
            if i in self.fin_ramp_at:
                o1 = np.zeros(7); j = self.off['left'][6]; o1[6] = round(j / (math.pi / 2)) * (math.pi / 2)
                self.ramps = [r for r in self.ramps if r[0] != 'left']
                self.ramps.append(('left', len(self.g.ex), self.cfg['fin_ramp'], self.off['left'].copy(), o1))
                self.note(ev='fin_ramp', off=np.round(self.off['left'], 5).tolist(), to_j7=round(o1[6], 5))
            self.row(i); self.i += 1
            if self.cfg['fin'] and i in self.fin_at:
                self.finish_drive(i)
            if self.cfg['logevery'] and i % self.cfg['logevery'] == 0:
                self.note(ev='err', **self.guide_err(i))

    def tip_screw(self):
        """The screw nearest the screwdriver tip (its body origin): name, lateral offset of the tip from the screw axis
        (world vector), the screw axis."""
        g = self.g; tip = g.d.xpos[g.bid['screwdriver']]; best = None
        for s in ('screw_1', 'screw_2', 'screw_3', 'screw_4'):
            p, Rm = g.pose(s); ax = Rm[:, 2]; v = tip - p; lat = v - (v @ ax) * ax
            if best is None or np.linalg.norm(v) < best[3]:
                best = (s, lat, ax, float(np.linalg.norm(v)))
        return best[:3]

    def seat_gap(self, s):
        return float(self.g.mon.evaluate()['screws'][s]['gap_m'])

    def finish_drive(self, i):
        """The demo's spin has ended (row i held): if the driven screw's head is not seated, keep spinning the left joint
        7 at fin_rate per tick (the correction kept in the left offset) while the TCP follows the screw down by the pitch
        (damped-least-squares IK on joints 1-6, no rotation), the tip kept on the true screw axis, and a light extra push
        (fin_push per tick, at most fin_push_max) while the screw's descent stalls; until the seat gap < fin_tol. Then
        joint 7's target is set to the joint (zero torque) for fin_untorque ticks before the demo lifts."""
        g = self.g; c = self.cfg; s, lat, ax = self.tip_screw(); gap0 = gap = self.seat_gap(s); k = 0; push = 0.; extra = 0.
        dz = 0.0005 * c['fin_rate'] / (2 * math.pi); hist = [gap]
        while gap > c['fin_tol'] and k < c['fin_max']:
            s_, lat, ax = self.tip_screw()
            p_ = push if extra < c['fin_push_max'] else 0.; extra += p_
            v = -ax * (dz + p_) - c['fin_kc'] * lat
            J = g.jac('left')[:, :6]; x = np.r_[v, np.zeros(3)]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), x)
            self.off['left'][:6] += np.clip(dq, -g.lim[:6] * 0.25, g.lim[:6] * 0.25)
            self.off['left'][6] += c['fin_rate']
            g.step(self.target(i)); k += 1
            if k % 20 == 0:
                gap = self.seat_gap(s); hist.append(gap)
                if len(hist) > 10 and hist[-11] - gap < 0.3 * 10 * 20 * dz:      # descent stalls: push a little more
                    push = min(push + c['fin_push'], c['fin_push_rate'])
        q7 = float(g.d.qpos[g.qadr['left'][6]])
        self.off['left'][6] = q7 - self.demo[i][6]
        for _ in range(c['fin_untorque']):
            g.step(self.target(i))
        gap1 = self.seat_gap(s)
        self.fin_ramp_at[i + c['fin_ramp_after']] = True
        self.note(ev='finish', screw=s, ticks=k, gap0_mm=round(1e3 * gap0, 4), gap_mm=round(1e3 * gap, 4), gap_after_mm=round(1e3 * gap1, 4),
                  push_um_tick=round(1e6 * push, 4), extra_mm=round(1e3 * extra, 4), off=np.round(self.off['left'], 5).tolist())

    def guide_err(self, i):
        """Each body's world position error to the guide (mm) and its angle to the guide's orientation (deg)."""
        out = {}
        for b in self.G.names:
            p, Rm = self.g.pose(b); pg, Rg = self.G.body(i, b)
            out[b] = [round(1e3 * float(np.linalg.norm(p - pg)), 2), round(math.degrees(math.acos(max(-1., min(1., (np.trace(Rg.T @ Rm) - 1) / 2)))), 2)]
        return out

    def part_err(self, obj, i):
        p, _ = self.g.pose(obj); pg, _ = self.G.body(i, obj)
        return p - pg

    def release_search(self, r, arm, obj):
        """From here (r - lead), replay the release with the held target moved by comp until the part lands within tol."""
        c = self.cfg; st0 = self.state(); best = None; end = r + c['settle']
        comp = np.zeros(3)
        for k in range(c['comp_iters']):
            self.restore(st0); self.comp[obj] = comp.copy(); self.done_rel.add(r)
            while self.i < end:
                if self.cfg['soft'] and self.i in self.soft_at:
                    self.soft_release(self.i, *self.soft_at[self.i]); continue
                self.row(self.i); self.i += 1
            e = self.part_err(obj, end - 1); e[2] = 0.
            n = float(np.linalg.norm(e))
            self.note(ev='release_try', obj=obj, k=k, comp_mm=np.round(1e3 * comp, 3).tolist(), err_mm=np.round(1e3 * e, 3).tolist())
            if best is None or n < best[0]:
                best = (n, comp.copy(), self.state())
            if n < c['comp_tol']:
                break
            comp = comp - e
        self.restore(best[2])
        self.note(ev='release', obj=obj, comp_mm=np.round(1e3 * best[1], 3).tolist(), err_mm=round(1e3 * best[0], 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--guide', required=True); ap.add_argument('--src', required=True)
    ap.add_argument('--upto', type=int, default=0); ap.add_argument('--ck', type=int, default=20000)
    ap.add_argument('--resume', default=None); ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    cfg = dict(CFG)
    for kv in a.set:
        k, v = kv.split('='); cfg[k] = type(CFG[k])(float(v))
    b = Builder(a, cfg)
    if a.resume:
        b.restore(pickle.load(open(a.resume, 'rb')))
    end = a.upto or len(b.demo)
    while b.i < end:
        nxt = min(end, (b.i // a.ck + 1) * a.ck)
        b.run(nxt)
        b.save('%07d' % b.i)
        st = b.g.mon.evaluate()
        b.note(ev='ck', frames={f: [round(1e3 * v['lateral_m'], 3), round(v['angle_deg'], 3)] for f, v in st['frames'].items()},
               acc={'%s|%d' % k: round(v['turns'], 2) for k, v in b.g.mon.acc.items()})
    ex = np.array(b.g.ex)
    b.g.s.save_trajectory(ex, os.path.join(a.out, 'ref.npz'))
    if end == len(b.demo):
        fc = b.g.s.final_check()
        b.note(ev='final', success=fc['success'], end={k: fc['end'][k] for k in ('goal', 'released', 'at_rest', 'breakdown')},
               settled={k: fc['settled'][k] for k in ('success', 'goal', 'released', 'at_rest')},
               report={k: v for k, v in fc['end']['report'].items() if k != 'screws'},
               screws={s: {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in v.items()} for s, v in fc['end']['report']['screws'].items()},
               sha=fc['state_sha'], settled_sha=fc['settled_sha'], rows=len(ex), slewed=b.g.s.slewed_ticks)
        json.dump(fc, open(os.path.join(a.out, 'final.json'), 'w'), default=str)


if __name__ == '__main__':
    main()
