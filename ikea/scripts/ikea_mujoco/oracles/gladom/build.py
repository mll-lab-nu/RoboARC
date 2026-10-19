#!/usr/bin/env python3
"""Build the gladom closed-loop reference under the Franka Hand slew (see README.md).

The 0.3.0 reference (the author's demo time-dilated to the arm limits) is the phase guide. Its rows are executed through
rcb_ikea's clamp plus a left-arm offset that the closed loop leaves behind:
  * grip wait: a row whose hand command the slewed hand has not reached yet is held (the arm waits for the hand);
  * in-hand compensation: the slower close leaves the hex driver up to ~2.7 mm off the hand axis (the demo's grasp has
    it centred), so the wrist spin would orbit its tip around the screw socket, and the off-centre driver topples when
    it is dropped back into its stand. While the left hand holds the driver, its tip offset in the TCP frame is
    measured from the TRUE state whenever the driver touches nothing but the fingers (frozen while it touches the stand
    or a screw), and the left arm is offset (damped least squares on the TCP Jacobian, no rotation) so that the TCP
    moves by minus that offset: the tip is back on the demo's hand axis, also while j7 spins. The offset is kept until
    the hand is fully open again plus `hold_open` ticks (the driver has dropped), then ramped out;
  * drive to seat: at the end of each driver spin (`DRIVES`), if the screw is not latched yet, the row is held and j7
    keeps turning the same way at `spin_rate` x its limit until the screw's TRUE gap is below `seat_gap`, then holds
    still until the retention weld latches (the scene's own gates). The extra turn stays on j7 until the driver is out
    of the socket, then goes back to the nearest whole turn (the hand's pose is the demo's again).
Checkpoints (all executed rows + the full state) at the stage rows; --from resumes one, bit-exact.
  build.py --out DIR [--from CK.pkl] [--to ROW] [--set key=value ...]   (DIR/ref.npz: the executed rows)"""
from __future__ import annotations
import argparse, json, math, os, pickle, sys, time
import numpy as np
from gl_lib import Rig, ARM, GRIP, SCREWS, demo_rows

CFG = dict(k=0.3, cap=0.5, lam=1e-3, ramp=400, hold_open=200, spin_rate=0.8, seat_gap=0.0004, max_spin=4000,
           max_hold=800, unwind=0.5, out_mm=10.0, open_cmd=170.0, pk=0.05, pr=0.05, place_max=600, lower_v=2e-5, lower_max=0.015, lower_hold=40, every=200)
STAGES = [20000, 30600, 49000, 60500, 93800, 144400]          # demo rows: after each screw's driver release (checkpoints)
# the end of each driver spin in the demo (left j7 strokes of 6-7 turns) and its screw
DRIVES = {28908: 's1-123451', 57880: 's2-122202', 103008: 's3-122202-left', 153438: 's3-122202-right'}
# the demo rows just before the left hand lets go of a screw over the stand's screw slot (0 -> 172.1, regrasp station)
SLOT_RELEASES = (10471, 37237, 77587, 132090)
SLOT = np.array([0.2, -0.265])                  # the screw slot's centre (tool_stand_s0/s1/s3/s4 inner faces, scene.xml)
PIVOT_Z = 0.006                                 # servo pivot on the screw axis above its origin (inside the slot)
# the screw head's hex socket centre above the screw origin (scene.xml <screw>_head_hex*: 23.8-25 mm on the long screw
# 123451, 12.8-14 mm on the three 122202)
SOCKET_Z = {'screw_long': 0.0244, 'screw_bottom': 0.0134, 'screw_ring_l': 0.0134, 'screw_ring_r': 0.0134}
J7 = ARM['left'][6]


class Builder:
    def __init__(self, out, cfg):
        self.out, self.cfg = out, cfg; os.makedirs(out, exist_ok=True)
        self.g = Rig(); self.demo = demo_rows(); self.i = 0
        self.off = np.zeros(7); self.ramp = None; self.est = None; self.opened = 0; self.hticks = 0
        self.spin = 0.; self.spun = None
        self.g.lfing = [self.g.m.body(n).id for n in ('left_left_finger', 'left_right_finger')]
        self.logf = open(os.path.join(out, 'log.jsonl'), 'a')

    KEYS = ('off', 'ramp', 'est', 'opened', 'hticks', 'spin', 'spun')

    def save(self, name):
        st = dict(g=self.g.get(), i=self.i, cfg=self.cfg, **{k: getattr(self, k) for k in self.KEYS})
        pickle.dump(st, open(os.path.join(self.out, f'ck_{name}.pkl'), 'wb'))

    def load(self, path):
        st = pickle.load(open(path, 'rb'))
        self.g.set(st['g']); self.i = st['i']        # a diag.py checkpoint (nothing applied yet) has only these two
        for k in self.KEYS:
            if k in st: setattr(self, k, st[k])

    def note(self, **kw):
        kw.update(row=self.i, tick=len(self.g.ex)); self.logf.write(json.dumps(kw, default=str) + '\n'); self.logf.flush()

    # ---- the in-hand compensation
    def held(self):
        """The left hand's command is closed on the driver (closed on nothing: width ~0)."""
        g = self.g
        return (g.d.ctrl[GRIP['left']] < 1.0 and g.width('left') > 0.01 and
                np.linalg.norm(g.d.xpos[g.bid['hex_driver']] - g.d.site_xpos[g.site['left']]) < 0.15)

    def free(self):
        """The driver touches nothing but the left fingers."""
        g = self.g; n = g.d.ncon
        if n == 0:
            return True
        b1 = g.m.geom_bodyid[g.d.contact.geom1[:n]]; b2 = g.m.geom_bodyid[g.d.contact.geom2[:n]]
        hd = g.bid['hex_driver']; other = np.where(b1 == hd, b2, np.where(b2 == hd, b1, -1))
        other = other[other >= 0]
        return bool(np.all(np.isin(other, g.lfing)))

    def socket(self, key):
        """Driver tip relative to the screw's socket: (lateral m, axial m)."""
        g = self.g; part = SCREWS[key]
        zs = g.axis(part); dl = g.d.xpos[g.bid['hex_driver']] - (g.d.xpos[g.bid[part]] + SOCKET_Z[part] * zs)
        ax = float(dl @ zs); return float(np.linalg.norm(dl - ax * zs)), ax

    def control(self):
        """Per tick: update the left-arm offset (compensation or ramp) and the extra j7 turn."""
        g = self.g; c = self.cfg
        if self.held():
            R = g.d.site_xmat[g.site['left']].reshape(3, 3); ptcp = g.d.site_xpos[g.site['left']]
            if self.free():
                e = (R.T @ (g.d.xpos[g.bid['hex_driver']] - ptcp))[:2]
                if self.est is None:
                    self.note(ev='comp_on', est_mm=np.round(1e3 * e, 3).tolist())
                self.est = e
            if self.est is not None:
                d = -R @ np.r_[self.est, 0.]
                J = g.jac('left'); x = np.r_[d, 0., 0., 0.]
                tgt = J.T @ np.linalg.solve(J @ J.T + c['lam'] ** 2 * np.eye(6), x)
                cap = g.s.step_limit[ARM['left']] * c['cap']
                self.off += np.clip(c['k'] * (tgt - self.off), -cap, cap); self.hticks += 1
            self.ramp = None; self.opened = 0
        else:
            if self.est is not None:
                self.note(ev='comp_off', est_mm=np.round(1e3 * self.est, 3).tolist(), off=np.round(self.off, 5).tolist())
                self.est = None
            if self.ramp is None and np.any(self.off != 0):
                self.opened = self.opened + 1 if g.d.ctrl[GRIP['left']] >= c['open_cmd'] else 0
                if self.opened >= c['hold_open']:
                    self.ramp = [len(g.ex), self.off.copy()]; self.note(ev='ramp')
            if self.ramp is not None:
                k = min(1., (len(g.ex) - self.ramp[0]) / c['ramp'])
                self.off = self.ramp[1] * (1 - k)
                if k >= 1: self.ramp = None
        # the extra turn of the last drive back to the nearest whole turn once the driver is out of the socket
        if self.spun is not None and np.hypot(*self.socket(self.spun)) > 1e-3 * c['out_mm']:
            goal = 2 * math.pi * round(self.spin / (2 * math.pi)); st = c['unwind'] * g.s.step_limit[J7]
            self.spin = goal if abs(goal - self.spin) <= st else self.spin + math.copysign(st, goal - self.spin)
            if self.spin == goal:
                self.note(ev='unwound', spin=self.spin); self.spun = None

    def target(self):
        row = self.demo[self.i].copy(); row[ARM['left']] += self.off; row[J7] += self.spin; return row

    def tick(self):
        self.control(); self.g.step(self.target())

    def seat(self, key):
        """Drive to seat (see the module docstring)."""
        g = self.g; c = self.cfg; self.spun = None
        lat, ax = self.socket(key)
        if g.mech.active(key) or lat > 0.0015 or abs(ax) > 0.004:
            self.note(ev='seat_skip', key=key, engaged=g.mech.active(key), lat_mm=round(1e3 * lat, 3), ax_mm=round(1e3 * ax, 3))
            return
        sg = math.copysign(1., self.demo[self.i][J7] - self.demo[self.i - 200][J7]); n = 0
        while g.screw(key)['gap'] > c['seat_gap'] and n < c['max_spin'] and not g.mech.active(key):
            self.spin += sg * c['spin_rate'] * g.s.step_limit[J7]; self.tick(); n += 1
        m = 0
        while not g.mech.active(key) and m < c['max_hold']:
            self.tick(); m += 1
        self.spun = key
        s = g.screw(key)
        self.note(ev='seat', key=key, spin_ticks=n, hold_ticks=m, engaged=g.mech.active(key), spin=round(self.spin, 4),
                  gap_mm=round(1e3 * s['gap'], 4), turns=round(s['turns'], 3))

    def place(self):
        """Before the hand lets go of a screw over the slot: the held screw is steered onto the slot's axis and upright
        (TRUE pose, pivot inside the slot), the row held, so the slowly opening fingers drop it straight."""
        g = self.g; c = self.cfg
        ptcp = g.d.site_xpos[g.site['left']]
        key = min(SCREWS, key=lambda k: np.linalg.norm(g.d.xpos[g.bid[SCREWS[k]]] - ptcp)); part = SCREWS[key]
        def err():
            z = g.axis(part); p = g.d.xpos[g.bid[part]] + PIVOT_Z * z
            cr = np.cross(z, [0., 0., 1.]); sn = np.linalg.norm(cr)
            return p, np.r_[SLOT - p[:2], 0.], cr / (sn + 1e-300) * math.atan2(sn, z[2])
        p, e, rv = err()
        if np.linalg.norm(p - ptcp) > 0.05 or np.linalg.norm(e) > 0.004:
            self.note(ev='place_skip', part=part, lat_mm=round(1e3 * float(np.linalg.norm(e)), 3)); return
        e0, t0 = 1e3 * float(np.linalg.norm(e)), math.degrees(float(np.linalg.norm(rv))); n = 0
        cap = g.s.step_limit[ARM['left']] * c['cap']
        while n < c['place_max'] and (np.linalg.norm(e) > 5e-5 or np.linalg.norm(rv) > math.radians(0.1)):
            om = c['pr'] * rv; v = c['pk'] * e + np.cross(om, g.d.site_xpos[g.site['left']] - p)
            J = g.jac('left'); x = np.r_[v, om]
            self.off += np.clip(J.T @ np.linalg.solve(J @ J.T + c['lam'] ** 2 * np.eye(6), x), -cap, cap)
            self.tick(); n += 1; p, e, rv = err()
        # lower the screw until it rests on the stand (its head on the slot's walls), so the fingers, which open
        # slower than the screw falls, never catch its head (it tips over otherwise)
        stand = [g.m.geom(f'tool_stand_s{k}').id for k in range(7)]
        pb = g.bid[part]; fing = set(g.lfing)
        def touch():
            nc = g.d.ncon; c1 = g.d.contact.geom1[:nc]; c2 = g.d.contact.geom2[:nc]
            b1 = g.m.geom_bodyid[c1]; b2 = g.m.geom_bodyid[c2]
            st1 = np.isin(c1, stand); st2 = np.isin(c2, stand)
            screw = bool(np.any((st1 & (b2 == pb)) | (st2 & (b1 == pb))))
            finger = bool(np.any((st1 & np.isin(b2, list(fing))) | (st2 & np.isin(b1, list(fing)))))
            return screw, finger
        dz = 0.; m = 0; why = 'max'
        while dz < c['lower_max']:
            sc, fi = touch()
            if sc or fi:
                why = 'screw' if sc else 'finger'; break
            om = c['pr'] * rv; v = c['pk'] * e + np.r_[0., 0., -c['lower_v']] + np.cross(om, g.d.site_xpos[g.site['left']] - p)
            J = g.jac('left'); x = np.r_[v, om]
            self.off += np.clip(J.T @ np.linalg.solve(J @ J.T + c['lam'] ** 2 * np.eye(6), x), -cap, cap)
            self.tick(); dz += c['lower_v']; m += 1; p, e, rv = err()
        for _ in range(c['lower_hold']):
            self.tick()
        self.note(ev='place', part=part, ticks=n, lat_mm=[round(e0, 3), round(1e3 * float(np.linalg.norm(e)), 3)],
                  tilt_deg=[round(t0, 3), round(math.degrees(float(np.linalg.norm(rv))), 3)], lower_mm=round(1e3 * dz, 3),
                  lower_ticks=m, stop=why)

    def run_to(self, end):
        g = self.g; gl = [GRIP[a] for a in ARM]
        engaged = {k: g.mech.active(k) for k in SCREWS}
        while self.i < end:
            if self.i in DRIVES:
                self.seat(DRIVES[self.i])
            if self.i in SLOT_RELEASES:
                self.place()
            self.tick()
            while np.any(np.abs(self.demo[self.i][gl] - g.d.ctrl[gl]) > 1e-12):   # the hand slew: hold the row
                self.tick()
            self.i += 1
            for k in SCREWS:
                if g.mech.active(k) and not engaged[k]:
                    engaged[k] = True; self.note(ev='engaged', key=k, turns=round(float(g.mech.turns[k]), 3))
            if self.i % self.cfg['every'] == 0:
                r = g.summary(); R = g.d.site_xmat[g.site['left']].reshape(3, 3)
                r['rel_mm'] = np.round(1e3 * R.T @ (g.d.xpos[g.bid['hex_driver']] - g.d.site_xpos[g.site['left']]), 3).tolist()
                r.update(ev='sum', off=np.round(self.off, 5).tolist(), spin=round(self.spin, 4),
                         est=None if self.est is None else np.round(1e3 * self.est, 3).tolist())
                self.note(**r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--from', dest='frm'); ap.add_argument('--to', type=int)
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    cfg = dict(CFG)
    for kv in a.set:
        k, v = kv.split('='); cfg[k] = type(CFG[k])(float(v))
    b = Builder(a.out, cfg)
    if a.frm:
        b.load(a.frm)
    t0 = time.time(); end = a.to or len(b.demo)
    for s in [x for x in STAGES if b.i < x <= end] + [end]:
        b.run_to(s)
        b.save(str(s))
        b.note(ev='stage', stage=s, engaged={k: b.g.mech.active(k) for k in SCREWS}, comp_ticks=b.hticks,
               wall=round(time.time() - t0, 1))
    if end == len(b.demo):
        fc = b.g.s.final_check()
        b.note(ev='final', success=fc['success'], end=fc['end'], settled=fc['settled'], sha=fc['state_sha'],
               settled_sha=fc['settled_sha'], rows=len(b.g.ex), wall=round(time.time() - t0, 1))
        b.g.s.save_trajectory(np.array(b.g.ex), os.path.join(a.out, 'ref.npz'))


if __name__ == '__main__':
    main()
