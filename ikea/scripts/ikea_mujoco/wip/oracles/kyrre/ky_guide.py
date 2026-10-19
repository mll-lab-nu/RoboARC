#!/usr/bin/env python3
"""KYRRE closed-loop reference (privileged mode, rcb_ikea.Session, the final physics: the cylinder->prism rule, the arm
joint-speed clamp, the welded-part contact switch-off). A port of oracles/uppsta's uppsta_oracle.py (the pattern of
oracles/vesken/vesken_oracle.py) to kyrre.

Phase guide: the author's demo (within the joint-speed limits as sent) and the guide run: the same rows in the scene
without the prism rule (track_np.py), where the demo assembles the stool. The guide run gives every free body's pose
every 50 rows (REF/track.npz) and, with the demo's gripper commands, each arm's grasp intervals and the part each one
carries (REF/sched.json, mkguide.py); REF/events.json the welds by row.

Segment k = demo rows [B_{k-1}, B_k), the boundaries being the arms' release rows. Per arm and row, from the TRUE state:
  approach (<= `approach` rows before a grasp): the TCP is offset by where the part really is versus the guide run
      (translation, plus its rotation when small), so the hand closes on the part the same way;
  carry (a grasp interval of a carried part, after `grip_delay` rows): an integral loop moves the TCP offset (position
      and orientation) so that the part follows its guide pose; at its place on the seat the error is the part's pose
      relative to the seat versus the guide run's. The hex key at a screw: its guide pose relative to the seat, turned
      about the screw's axis by the screw's spin versus the guide's modulo 60 degrees, the key tip servoed onto the
      screw's axis;
  finish: where the guide run's weld of a screw has latched but ours has not, ky_lib.ratchet drives the screw to its
      weld (strokes inside the free sector of the hand's turn, the arm included), and the demo resumes;
  idle: the offset decays to zero (then exactly zero: the demo row as it is).
Offsets become joint targets by damped least squares IK on left_tcp / right_tcp. Every emitted row is within the
joint-speed limits of the previous applied target (a longer step is split over the ticks it needs), so
Session.applied() never changes a recorded row.
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np
import mujoco
import rcb_ikea as R

ARMS = ('left', 'right')
import ky_lib as KL
ARM_CTRL = {'left': list(range(0, 7)), 'right': list(range(8, 15))}
GRIP = {'left': 7, 'right': 15}
SPEED_MARGIN = 0.999
NEAR_SEAT = 0.03           # a part within this of its place on the seat (guide run) is tracked relative to the seat
NEAR_KEY = 0.06            # the hex key within this of a screw (guide run) is the key at that screw


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def qmul(a, b):
    w, x, y, z = a; W, X, Y, Z = b
    return np.array([w*W-x*X-y*Y-z*Z, w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qrot(q, v):
    return qmul(qmul(q, np.r_[0., v]), qconj(q))[1:]


def rotvec(q):
    q = q / np.linalg.norm(q)
    if q[0] < 0: q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-12: return np.zeros(3)
    return q[1:] / s * 2 * math.atan2(s, q[0])


def rotmat(v):
    th = np.linalg.norm(v)
    if th < 1e-12: return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


class IK:
    def __init__(self, model):
        self.m = model; self.d = mujoco.MjData(model)
        self.site = {a: model.site(f'{a}_tcp').id for a in ARMS}
        j = lambda a, i: model.joint(f'{a}_joint{i}').id
        self.qadr = {a: [model.jnt_qposadr[j(a, i)] for i in range(1, 8)] for a in ARMS}
        self.dadr = {a: [model.jnt_dofadr[j(a, i)] for i in range(1, 8)] for a in ARMS}
        self.lo = {a: np.array([model.jnt_range[j(a, i)][0] if model.jnt_limited[j(a, i)] else -1e9 for i in range(1, 8)]) for a in ARMS}
        self.hi = {a: np.array([model.jnt_range[j(a, i)][1] if model.jnt_limited[j(a, i)] else 1e9 for i in range(1, 8)]) for a in ARMS}
        self.jp = np.zeros((3, model.nv)); self.jr = np.zeros((3, model.nv))
        # only the arm chains matter for the TCP: kinematics of the whole model is the cost; keep it, it is exact

    def fk(self, arm, q):
        self.d.qpos[self.qadr[arm]] = q
        mujoco.mj_kinematics(self.m, self.d)
        s = self.site[arm]
        return self.d.site_xpos[s].copy(), self.d.site_xmat[s].reshape(3, 3).copy()

    def solve(self, arm, q0, p_goal, R_goal, iters=20):
        q = np.array(q0, float); s = self.site[arm]
        for _ in range(iters):
            self.d.qpos[self.qadr[arm]] = q
            mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
            p = self.d.site_xpos[s]; Rm = self.d.site_xmat[s].reshape(3, 3)
            ep = p_goal - p; Re = R_goal @ Rm.T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            if np.linalg.norm(ep) < 1e-7 and np.linalg.norm(er) < 1e-6:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, s)
            J = np.vstack([self.jp[:, self.dadr[arm]], self.jr[:, self.dadr[arm]]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[ep, er])
            q = np.clip(q + dq, self.lo[arm], self.hi[arm])
        return q


class Guide:
    """The guide run: every free body's pose every `step` rows (track.npz of diag.py --track)."""

    def __init__(self, ref_dir):
        z = np.load(os.path.join(ref_dir, 'track.npz'))
        self.rows = z['rows']; self.pose = z['body_pose']; self.names = [str(x) for x in z['bodies']]
        self.step = int(self.rows[1] - self.rows[0]); assert self.rows[0] == self.step
        self.idx = {n: k for k, n in enumerate(self.names)}
        self.events = json.load(open(os.path.join(ref_dir, 'events.json')))       # [(row, event)] of the mechanism

    def at(self, name, i):
        """pose after row i (i = rows done): linear in position, nearest in rotation"""
        j = self.idx[name]; k = i / self.step - 1; n = len(self.rows)
        k0 = min(max(int(math.floor(k)), 0), n - 1); k1 = min(k0 + 1, n - 1); w = min(max(k - k0, 0.), 1.)
        p = (1 - w) * self.pose[k0, j, :3] + w * self.pose[k1, j, :3]
        q = self.pose[k0 if w < 0.5 else k1, j, 3:]
        return p, q

    def welded_at(self, i):
        return sorted({e['fastener'] for r, e in self.events if r <= i and 'fastener' in e})


class Runner:
    def __init__(self, demo_path, ref_dir, scene_dir=None):
        self.demo = np.ascontiguousarray(np.load(demo_path)['ctrl'], dtype=np.float64)
        self.g = Guide(ref_dir)
        self.sched = json.load(open(os.path.join(ref_dir, 'sched.json')))
        self.s = R.Session('kyrre', scene_dir)
        self.m, self.d = self.s.model, self.s.data
        self.mech = self.s.S.mech
        self.parts = list(self.s.S.task['parts']) + ['hex_key']
        self.bid = {n: self.m.body(n).id for n in self.parts}
        self.screws = [o['part'] for o in self.mech.ops if o['mode'] == 'screw']
        self.opkey = {o['part']: o['key'] for o in self.mech.ops}
        self.target = {o['part']: np.asarray(o['target_pos'], float) for o in self.mech.ops}
        self.ik = IK(self.m)
        self.lim = self.s.step_limit * SPEED_MARGIN
        self.fin = np.isfinite(self.lim)
        self.off = {a: dict(p=np.zeros(3), r=np.zeros(3)) for a in ARMS}
        self.gb = self.m.geom_bodyid
        self.robot = np.zeros(self.m.nbody, bool); self.robot[R.robot_bodies(self.m)] = True
        self.rr_max = (0., None, 0)
        self.p = {}
        self.set_sched(self.sched)
        bounds = sorted({it['oe'] for a in ARMS for it in self.sched[a] if it['oe'] < len(self.demo)})
        self.bounds = bounds + [len(self.demo)]
        tp = os.path.join(ref_dir, 'tempo.npz')          # optional: demo rows advanced per emitted row (integer >= 1)
        self.tempo = np.load(tp)['S'].astype(np.int64) if os.path.isfile(tp) else None
        self.tau = 0                                      # the demo row the controller is at

    def set_sched(self, sched):
        """per arm (cs, oe, part, os): os = the row the demo's gripper starts opening for the release"""
        self.sched = sched; self.ivs = {}
        self.twirl = {(a, it['cs']) for a in ARMS for it in sched[a] if it.get('twirl')}
        for a in ARMS:
            g = self.demo[:, GRIP[a]]; out = []
            for it in sched[a]:
                os_ = min(it['oe'], len(g) - 1)
                while os_ > it['cs'] and g[os_ - 1] < g[os_]:
                    os_ -= 1
                out.append((it['cs'], it['oe'], it['part'], os_))
            self.ivs[a] = out
        self.press_state = {}

    # ------------------------------------------------------------ state
    def checkpoint(self):
        st = self.s.get_state(); rows = np.array(st.pop('rows')); st['rows'] = []
        return dict(st=st, rows=rows, off={a: {k: v.copy() for k, v in o.items()} for a, o in self.off.items()},
                    rr_max=self.rr_max, tau=self.tau)

    def restore(self, ck):
        st = dict(ck['st']); st['rows'] = list(ck['rows'])
        self.s.set_state(st)
        self.off = {a: {k: v.copy() for k, v in o.items()} for a, o in ck['off'].items()}; self.rr_max = ck['rr_max']
        self.tau = ck.get('tau', len(ck['rows']))

    # ------------------------------------------------------------ errors
    def host(self, name, i):
        """the body a part is tracked relative to at guide row i: the seat for a part at its place on the seat (within
        NEAR_SEAT of its operation's target in the guide run), else None (world)"""
        if name == 'seat' or name not in self.target:
            return None
        pg, qg = self.g.at(name, i); pb, qb = self.g.at('seat', i)
        return 'seat' if float(np.linalg.norm(pg - (pb + qrot(qb, self.target[name])))) < NEAR_SEAT else None

    def _on_host(self, h, pr, qr, i):
        """a guide pose mapped onto the true host h (the same pose relative to h as in the guide run)"""
        bb = self.bid[h]; pbr, qbr = self.g.at(h, i)
        rel_p = qrot(qconj(qbr), pr - pbr); rel_q = qmul(qconj(qbr), qr)
        return self.d.xpos[bb] + qrot(self.d.xquat[bb], rel_p), qmul(self.d.xquat[bb], rel_q)

    def part_error(self, name, i, ctl=False):
        """(position error, rotation error) of a part versus the guide run, in world axes; relative to the board when it
        is at the board. The hex key at a screw: its guide pose relative to the board, turned about the screw's axis
        (through the true screw) by the screw's spin versus the guide's taken modulo 60 degrees (the hexagon), so the
        key meets the socket as it is now and still hangs the way it did in the guide run"""
        d = self.d; b = self.bid[name]
        pr, qr = self.g.at(name, i)
        h = self.host(name, i)
        if h is not None:
            pr, qr = self._on_host(h, pr, qr, i)
        if name == 'hex_key' and self.p.get('key_on_screw', True):
            pk, qk = self.g.at(name, i)
            tips = (pk, pk + qrot(qk, KL.KEY_SHORT_TIP))           # kyrre: either tip of the L key may be the one in a socket
            sd = {s_: min(float(np.linalg.norm(t - self.g.at(s_, i)[0])) for t in tips) for s_ in self.screws}
            s_ = min(sd, key=sd.get)
            if sd[s_] < NEAR_KEY and not self.welded_part(s_):
                if h is None:                                   # the key's guide pose relative to the seat
                    pr, qr = self._on_host('seat', *self.g.at(name, i), i)
                psg, qsg = self._on_host('seat', *self.g.at(s_, i), i)
                ps, qs = d.xpos[self.bid[s_]], d.xquat[self.bid[s_]]
                ax = qrot(qs, np.array([0., 0., 1.]))
                spin = float(rotvec(qmul(qs, qconj(qsg))) @ ax)
                spin = (spin + math.pi / 6) % (math.pi / 3) - math.pi / 6
                if self.p.get('key_spin', 'mod') == 'zero':     # the guide's orientation relative to the board
                    spin = 0.
                tip = self._key_tip(ps, ax)
                # once the key is in the socket the offset is held (the screw then turns with the key; a wrap of the
                # modulo would jump the target by 60 degrees)
                held = getattr(self, '_kspin', None)
                if tip is not None and held is not None:
                    spin = held
                elif ctl:
                    self._kspin = spin if tip is not None else None
                qz = np.r_[math.cos(spin / 2), math.sin(spin / 2) * ax]
                pr = ps + qrot(qz, pr - psg); qr = qmul(qz, qr)
                e = d.xpos[b] - pr
                if tip is not None and self.p.get('tip_servo', True):
                    # at the socket: the lateral error is the key tip's offset from the hole's axis (the guide's screw
                    # axis on the true board: the guide's tip is on it, and a key held there keeps the screw upright as
                    # it turns), the axial error the guide's
                    axg = qrot(qsg, np.array([0., 0., 1.]))
                    lat = tip - psg; lat = lat - (lat @ axg) * axg
                    e = (e @ axg) * axg + lat
                er = rotvec(qmul(d.xquat[b], qconj(qr)))
                if tip is not None and self.p.get('flat_servo', True):
                    # kyrre: at the socket the key's turn about the screw axis is servoed on its flats against the TRUE
                    # socket's (ky_lib.Rig.key_vs), not on the guide's spin: a key forced a few degrees off the flats
                    # wedges in the socket, then stays in it when the demo lifts it and turns the screw back
                    _, hk, _, hx = self.rig().key_vs(s_)
                    # above the socket the flats are lined up faster (x flat_gain), so the key enters on them
                    er = er - (er @ ax) * ax + (self.p.get('flat_gain', 8.) if hk > 4e-4 else 1.) * hx * ax
                return e, er
        er = rotvec(qmul(d.xquat[b], qconj(qr)))
        if name in self.screws and self.p.get('screw_hex', True):
            # kyrre: a screw's spin about its own axis does not matter while it is carried (the finger twirls and the hex
            # key meet its socket as it is, modulo 60 degrees, from the true state): only its tilt is corrected, so the
            # hand does not turn about the screw axis and the arm keeps the demo's posture past the legs
            ax = qrot(d.xquat[b], np.array([0., 0., 1.]))
            sp = float(er @ ax)
            er = er - sp * ax
        return d.xpos[b] - pr, er

    def _key_tip(self, ps, ax):
        """the centre of the hex key's geom nearest the screw axis when it is within TIP_ZONE of the head along the
        axis and within 3 mm of the axis (the short arm over the socket), else None"""
        if not hasattr(self, '_key_geoms'):
            kb = self.bid['hex_key']
            self._key_geoms = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == kb]
        best = None
        for g in self._key_geoms:
            v = self.d.geom_xpos[g] - ps; z = float(v @ ax); r = float(np.linalg.norm(v - z * ax))
            if 0 < z < 0.03 and r < 0.003 and (best is None or r < best[0]):
                best = (r, self.d.geom_xpos[g].copy())
        return None if best is None else best[1]

    def welded_part(self, part):
        k = self.opkey.get(part)
        return bool(k is not None and self.mech.active(k))

    def placed(self):
        """the parts whose operation's weld has latched (the mechanism holds them in their place on the seat)"""
        return {p_ for p_ in self.opkey if self.welded_part(p_)}

    def dev(self, i):
        """mm: each part's deviation from the guide run; 0 for a part success.py already accepts in its place (a peg
        pushed a little deeper than in the guide run is as good)"""
        out = {}; ok = self.placed()
        gap, lat, ang = self.mech.geometry_all()[:3]; opi = {o['part']: j for j, o in enumerate(self.mech.ops)}
        for n in self.parts:
            e, er = self.part_error(n, i)
            # position (mm), and for a part with a place (not the key) its tilt too: 0.3 mm per degree
            rd = math.degrees(float(np.linalg.norm(er))) if n in self.target else 0.
            out[n] = 0.0 if n in ok else round(max(1000 * float(np.linalg.norm(e)), 0.3 * rd), 2)
            if n in self.screws and n not in ok and lat[opi[n]] < 2e-3 and ang[opi[n]] < 15. and 0 < gap[opi[n]] < 9e-3:
                # a screw standing in its thread: only its lateral error counts (how deep it is threaded at this row is the
                # drive's business: the finish drives it to its weld)
                out[n] = round(1000 * float(lat[opi[n]]), 2)
        return out

    def screw_spin_dev(self, i):
        """degrees: each screw's spin about its axis versus the guide run, modulo 60 (the hex socket the key must meet)"""
        out = {}
        for n in self.screws:
            _, er = self.part_error(n, i)
            ax = qrot(self.d.xquat[self.bid[n]], np.array([0., 0., 1.]))
            out[n] = round(abs(math.degrees(float(er @ ax))), 2)
        return out

    def mode(self, arm, i):
        for cs, oe, part, os_ in self.ivs[arm]:
            if cs <= i < oe:
                if part is None or (arm, cs) in self.twirl:   # a twirl turns the part in the hand: the offset is kept
                    return 'frozen', None, cs
                if self.p.get('press') and part in self.p.get('press_parts', (part,)) and i >= os_ - self.p.get('press_rows', 800):
                    return 'press', part, (cs, os_)
                return ('carry' if i >= cs + self.p.get('grip_delay', 300) else 'frozen'), part, cs
            if i < cs:
                if part is not None and cs - i <= self.p.get('approach', 1500):
                    return 'approach', part, cs
                return 'idle', None, None
        return 'idle', None, None

    # ------------------------------------------------------------ control
    def control(self, i, tcp_ref):
        p = self.p
        modes = {a: self.mode(a, i) for a in ARMS}
        for arm in ARMS:
            o = self.off[arm]
            md, part, cs = modes[arm]
            if md == 'approach' and p.get('ff', True):
                b = self.bid[part]
                pa, qa = self.d.xpos[b], self.d.xquat[b]
                pr, qr = self.g.at(part, i)
                dr = rotvec(qmul(qa, qconj(qr)))
                if np.linalg.norm(dr) > p.get('ff_rot_max', 0.25) or not p.get('ff_rot', True) or (arm, cs) in self.twirl:
                    dr = np.zeros(3)
                tp, tR = tcp_ref[arm]
                tgt_p = pa + rotmat(dr) @ (tp - pr) - tp + np.asarray(p.get('gofs', (0, 0, 0)), float) * \
                    smooth(1 - (cs - i) / max(1., p.get('gofs_rows', 600.)))
                a = p.get('ff_rate', 1 / 300.)
                o['p'] += a * (tgt_p - o['p']); o['r'] += a * (dr - o['r'])
            elif md == 'carry':
                e, erot = self.part_error(part, i, ctl=True)
                n_hold = 2 if any(modes[x][0] == 'carry' and modes[x][1] == part for x in ARMS if x != arm) else 1
                g = p.get('g', 0.005) / n_hold; gr = p.get('gr', 0.0025) / n_hold
                o['p'] -= g * e
                if p.get('rot', True):
                    o['r'] -= gr * erot
                npn = np.linalg.norm(o['p']); lim = p.get('max_off', 0.03)
                if npn > lim: o['p'] *= lim / npn
                nr = np.linalg.norm(o['r']); rl = 0.6 if part in self.screws else 0.3
                if nr > rl: o['r'] *= rl / nr
            elif md == 'press':
                # before the release: push the part `press` m further along its insertion direction (the guide run's
                # motion of the part relative to the board over the last 1500 rows), the tracking loop held
                cs, os_ = cs
                key = (arm, cs)
                if key not in self.press_state:
                    pr1, _ = self.g.at(part, os_); pr0, _ = self.g.at(part, max(cs, os_ - 1500))
                    pb, qb = self.g.at('seat', os_)
                    dl = qrot(qconj(qb), pr1 - pr0)                          # in the seat frame
                    dw = qrot(self.d.xquat[self.bid['seat']], dl)
                    nd = np.linalg.norm(dw)
                    self.press_state[key] = (o['p'].copy(), dw / nd if nd > 1e-6 else np.zeros(3))
                p0, dw = self.press_state[key]
                n = p.get('press_rows', 800)
                o['p'] = p0 + p['press'] * dw * smooth((i - (os_ - n)) / max(1., 0.6 * n))
            elif md == 'frozen':
                pass
            else:
                k = 1 / p.get('decay', 600.)
                o['p'] *= (1 - k); o['r'] *= (1 - k)
                if np.linalg.norm(o['p']) < 2e-6 and np.linalg.norm(o['r']) < 1e-5:
                    o['p'][:] = 0; o['r'][:] = 0

    def emit(self, row):
        row = np.clip(np.asarray(row, float), self.s._clo, self.s._chi)
        prev = self.d.ctrl.copy(); dlt = row - prev
        r = float(np.max(np.abs(dlt[self.fin]) / self.lim[self.fin]))
        n = max(1, int(math.ceil(r - 1e-12)))
        for k in range(1, n + 1):
            self.s.step(row if k == n else prev + dlt * (k / n))
            self._rr()

    def _rr(self):
        d = self.d; n = d.ncon
        if not n: return
        b1 = self.gb[d.contact.geom1[:n]]; b2 = self.gb[d.contact.geom2[:n]]
        sel = np.flatnonzero(self.robot[b1] & self.robot[b2])
        if len(sel):
            adr = d.contact.efc_address[sel]
            f = np.where(adr >= 0, d.efc_force[np.maximum(adr, 0)], 0.)
            j = int(np.argmax(f))
            if f[j] > self.rr_max[0]:
                self.rr_max = (float(f[j]), (self.m.body(int(b1[sel[j]])).name, self.m.body(int(b2[sel[j]])).name), len(self.s.rows))

    def step_row(self, i, guide_at=None):
        row = self.demo[i].copy()
        tcp_ref = {a: self.ik.fk(a, row[ARM_CTRL[a]]) for a in ARMS}
        self.control(i if guide_at is None else guide_at, tcp_ref)
        for a in ARMS:
            dp = self.off[a]['p']; dr = self.off[a]['r']
            if np.linalg.norm(dp) > 0 or np.linalg.norm(dr) > 0:
                tp, tR = tcp_ref[a]
                row[ARM_CTRL[a]] = self.ik.solve(a, row[ARM_CTRL[a]], tp + dp, rotmat(dr) @ tR)
        self.emit(row)

    # ------------------------------------------------------------ one segment
    def run_seg(self, k, p):
        self.p = p
        self.press_state = {}; self._kspin = None; self.finish_log = []
        r0 = self.bounds[k - 1] if k > 0 else 0
        r1 = self.bounds[k]
        if self.tempo is None and not p.get('tempo_max'):
            for i in range(r0, r1):
                self.step_row(i)
                self.finish(i)
            self.tau = r1
        else:                                             # tempo: demo rows i, i+S, ... (S from the profile, capped)
            assert self.tau == r0, (self.tau, r0)
            cap = int(p.get('tempo_max', 4))
            while self.tau < r1:
                S = 1 if self.tempo is None else int(min(self.tempo[self.tau], cap))
                S = max(1, min(S, r1 - self.tau))
                self.step_row(self.tau + S - 1, guide_at=self.tau)
                self.tau += S
        welded = sorted(self.placed())
        dev = self.dev(r1)
        look = int(min(p.get('look', 2500), (self.bounds[k + 1] if k + 1 < len(self.bounds) else r1) - r1))
        dev_look = None; spin_look = self.screw_spin_dev(r1)
        if look > 0:                    # a part released at r1 must stay: the next rows (default parameters), then restore
            ck = self.checkpoint(); pp = self.p; self.p = {}
            for i in range(r1, r1 + look):
                self.step_row(i)
            dev_look = self.dev(r1 + look); spin_look = self.screw_spin_dev(r1 + look)
            self.restore(ck); self.p = pp
        released = sorted({it[2] for a in ARMS for it in self.ivs[a] if it[1] == r1 and it[2]})
        return dict(seg=k, r0=r0, r1=r1, rows=len(self.s.rows), tau=self.tau, welded=welded, welded_guide=self.g.welded_at(r1),
                    max_dev=max(dev.values()), worst=max(dev, key=dev.get), dev=dev, rr=self.rr_max,
                    dev_look=dev_look, max_dev_look=max(dev_look.values()) if dev_look else None, released=released,
                    spin_look=spin_look, finish=self.finish_log,
                    breakdown=self.s.validity.breakdown, depth_mm=round(1000 * self.s.validity.max_part_depth_m, 3),
                    params=p)

    def finish(self, i):
        """At the guide run's weld row of a screw (events.json): if our screw's weld has not latched, the key hand drives it
        to its weld (ky_lib.ratchet, strokes inside ky_lib.arm_sector's free sector), then the key arm's offset is set to
        where the hand now is (the carry loop takes it back onto the guide)."""
        for row, e in self.g.events:
            if row != i + 1 or e.get('fastener') not in self.screws or self.welded_part(e['fastener']):
                continue
            s_ = e['fastener']; rig = self.rig()
            arm = min(ARMS, key=lambda a: float(np.linalg.norm(self.d.site_xpos[self.ik.site[a]] - self.d.xpos[self.bid['hex_key']])))
            if 'hex_key' not in rig.touching(arm):
                self.finish_log.append(dict(row=i, screw=s_, skipped='no hand holds the key')); continue
            sec = KL.arm_sector(rig, s_, arm, margin_deg=5.)
            out = KL.ratchet(rig, arm, s_, sector=sec, max_strokes=self.p.get('finish_strokes', 24))
            out['sector_deg'] = [round(math.degrees(x)) for x in sec]; out['row'] = i
            self.finish_log.append(out)
            tp, tR = self.ik.fk(arm, self.demo[i][ARM_CTRL[arm]])
            pn = self.d.site_xpos[self.ik.site[arm]].copy(); Rn = self.d.site_xmat[self.ik.site[arm]].reshape(3, 3)
            # the commanded TCP now (the targets' FK) versus the demo row's: the offset that reproduces it
            cp, cR = self.ik.fk(arm, self.d.ctrl[ARM_CTRL[arm]])
            self.off[arm]['p'] = cp - tp
            Re = cR @ tR.T; self.off[arm]['r'] = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])

    def rig(self):
        if not hasattr(self, '_rig'):
            self._rig = KL.Rig(session=self.s)
        return self._rig

    def hold_rest(self, sec):
        row = self.d.ctrl.copy()
        for _ in range(int(round(sec / self.s.control_dt))):
            self.s.step(row); self._rr()


CANDIDATES = [                       # in order of preference: the first that passes is kept (search.py)
    dict(),
    dict(g=0.01, gr=0.005),
    dict(rot=False, ff_rot=False),
    dict(g=0.0025, gr=0.00125),
    dict(gofs=(0, 0, -0.002)),
    dict(press=0.003),                # a part pushed deeper than the guide before its release (a dowel that tips over)
    dict(press=0.005),
    dict(press=0.003, g=0.01, gr=0.005),
    dict(press=0.015),                # kyrre: a screw is let go upright 28 mm over its hole; a lower release lands it truer
    dict(press=0.02),
    dict(press=0.02, g=0.01, gr=0.005),
]
