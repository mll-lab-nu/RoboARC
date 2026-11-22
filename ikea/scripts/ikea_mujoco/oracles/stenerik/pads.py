#!/usr/bin/env python3
"""stenerik pad phase with a closed-loop press per pad (robot actions only; the scene, its gates and the physics are
untouched). From a builder checkpoint after the six screw cycles, the demo's peel / carry / press rows are followed with
centre_hook (each hand's descent onto a pad still on the backing centred on that pad, true state: the Franka Hand slew)
and pad_hook (true-state steering of the carried pad onto its site). When a free pad (off the backing, not bonded) sits on a
free site within the alignment gates and a hand presses it (pad-support contact > F_ON), the demo row is held and the
pressing arm's target is servoed along the site normal on the TRUE pad-support contact force (M.force: the scene's own
reading) toward F_TGT, with a penetration guard, until the scene's adhesive weld latches; then held HOLD ticks, the
push offset ramped out while the row is still held, and the demo resumes.
usage: pads.py --ck CK --out DIR [--upto DEMO_ROW]   (CK: build.py's checkpoint after the sixth screw cycle, or a pre_/bond_ one)
Writes DIR/ref.npz (all executed rows from the birth), DIR/log.json, DIR/pre_<pad>.pkl (state just before each press),
DIR/bond_<pad>.pkl (after each bond), and the grader's final_check at the end."""
import argparse, json, os, pickle, time
import numpy as np
from st_lib import Rig, Follower, demo_rows, pad_hook, pad_view, ARM

F_ON = float(os.environ.get('PP_F_ON', '0.4'))        # N: a hand is pressing (instantaneous pad-support force)
F_TGT = float(os.environ.get('PP_F_TGT', '1.8'))      # N: servo target (the gate is a 48 ms mean > 1 N for 0.30 s)
STEP = float(os.environ.get('PP_STEP', '4e-6'))       # m per tick, the largest push increment
KP = float(os.environ.get('PP_KP', '3e-6'))           # m per tick per N of force error
GAP_MIN = float(os.environ.get('PP_GAP_MIN', '-6e-5'))  # m: back off before the |gap| < 0.1 mm gate
MAXT = int(os.environ.get('PP_MAXT', '4000'))         # ticks per press attempt
HOLD = int(os.environ.get('PP_HOLD', '40'))           # ticks held after the latch
RAMP = int(os.environ.get('PP_RAMP', '200'))          # ticks to remove the push offset (row still held)
# The demo's press phases (its own gripper commands): each pad is carried closed, released on its site (open), and the
# fingers close again (the end of that re-close is the row below); the press follows. A press is only taken over from
# there to +2000 rows, so a pad still gripped in its carry is never pushed.
PRESS_FROM = [217571, 228440, 239974, 251553]
PRESS_ARM = {'pad_1': 'left', 'pad_2': 'left', 'pad_3': 'right', 'pad_4': 'right'}   # the demo's pressing hand per pad
# If at PRESS_FROM + REACH_AT the pad sits on its site but no hand touches it (the demo's press passes above it), that hand
# is lowered along the site normal by the same force servo until it bears on the pad (at most REACH_MAX of travel).
REACH_AT = int(os.environ.get('PP_REACH_AT', '1000'))
REACH_MAX = float(os.environ.get('PP_REACH_MAX', '0.006'))
# Pace: outside each pad's placement and press (from SLOW_BEFORE rows before the demo's release on the site to the end of
# the press window) the demo is followed every STRIDE-th row (Rig.move still keeps every tick within the joint-speed
# limits): the demo's peel and carry moves are far below the limits, and the welded screws' thread contacts make every
# tick of this phase expensive. STRIDE 1 = the demo's own pace.
RELEASE_AT = [217149, 228018, 239552, 251131]
STRIDE = int(os.environ.get('PP_STRIDE', '2'))
SLOW_BEFORE = int(os.environ.get('PP_SLOW_BEFORE', '800'))


CLOSE_AT = {212203: 'left', 223093: 'left', 234648: 'right', 246207: 'right'}   # the demo starts closing on pad k
APPROACH = int(os.environ.get('PP_APPROACH', '600'))   # demo rows before CLOSE_AT in which the approaching hand is centred
CENTRE = int(os.environ.get('PP_CENTRE', '1'))   # 0: the demo's own approach


def centre_hook(rig, fol, gain=0.05, cap=2e-5, near=0.03):
    """While a hand descends onto a pad still on the backing (the APPROACH rows before the demo closes it on that pad):
    steer the hand in its own x-y plane so the pad's centre is on the TCP's approach axis (true state). Under the slew the
    demo's grasp of pad 1 arrives 2.8 mm off centre along the closing direction (the old reference: 1.5 mm), so one finger
    lands on the pad's top edge instead of beside it, presses the pad into the backing (6-10 N) and the closing fingers
    squeeze it out. The correction accumulates in that arm's follower offset (ramped out with the carry)."""
    M = rig.M; d = rig.d

    def hook(i, r):
        w = next(((r0, arm) for r0, arm in CLOSE_AT.items() if r0 - APPROACH <= i < r0), None)
        if not CENTRE or w is None:
            return r
        arm = w[1]; tcp = d.site_xpos[rig.site[arm]]; Rt = d.site_xmat[rig.site[arm]].reshape(3, 3)
        cand = [p for p in M.pads if M.active('backing_%s' % p)]
        if not cand:
            return r
        p = min(cand, key=lambda q: float(np.linalg.norm(rig.pose(q)[0] - tcp)))
        v = Rt.T @ (rig.pose(p)[0] - tcp)
        if np.linalg.norm(v) > near or p in rig.touching(arm):
            return r
        e = Rt @ np.array([v[0], v[1], 0.]); dv = gain * e; n = np.linalg.norm(dv)
        if n > cap: dv *= cap / n
        dq = rig.ik(arm, dv)
        if arm == 'left': fol.offL = fol.offL + dq
        else: fol.off = fol.off + dq
        r = r.copy(); r[ARM[arm]] += dq
        return r
    return hook


def stride(i):
    return 1 if any(r - SLOW_BEFORE <= i < r0 + 2000 for r, r0 in zip(RELEASE_AT, PRESS_FROM)) else STRIDE


def fol_state(fol):
    return dict(off=fol.off.copy(), offL=np.array(fol.offL).copy(), ramp=fol.ramp, rampL=fol.rampL)


def fol_set(fol, s):
    fol.off = np.array(s['off']); fol.offL = np.array(s['offL']); fol.ramp = s['ramp']; fol.rampL = s['rampL']


def candidate(rig, done, i):
    """(arm, pad, support, type) when a free pad sits on a free site within the alignment gates and a hand touches it."""
    from ikea_runtime import rotate, quat_angle_deg, qmul
    if not any(r0 <= i < r0 + 2000 for r0 in PRESS_FROM):
        return None
    M = rig.M; g = M.g
    for p in M.pads:
        if p in done or M.active('backing_%s' % p) or any(M.active('adhesive_%s_%s' % (s, p)) for s in M.supports):
            continue
        arm = next((a for a in ('left', 'right') if p in rig.touching(a)), None)
        reach = arm is None and any(i == r0 + REACH_AT for r0 in PRESS_FROM)
        if reach:
            arm = PRESS_ARM[p]
        if arm is None:
            continue
        pp, qp = M.pose(p); qb = qmul(qp, M.pad_box[p])
        for s in M.supports:
            for typ in ('flat', 'slope'):
                if '%s|%s' % (s, typ) in M.pad_at:
                    continue
                tp, tq, qs = M.site_target(s, typ); ax = rotate(qs, [0, 0, 1]); dl = pp - tp; gap = float(dl @ ax)
                lat = float(np.linalg.norm(dl - gap * ax))
                if abs(gap) < 3e-4 and lat < g['lateral_m'] and quat_angle_deg(qb, tq) < g['axis_deg'] and (reach or M.force(p, s) > F_ON):
                    return arm, p, s, typ
    return None


def hand_gap(rig, arm, p):
    """The smallest distance (mm) between the arm's finger / hand geoms and pad p's geoms (mj_geomDistance)."""
    import mujoco
    m, d = rig.m, rig.d; pb = rig.bid[p]; fr = np.zeros(6); best = 1e9
    pg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == pb]
    hg = [g for g in range(m.ngeom) if m.geom_bodyid[g] in rig.finger[arm]]
    for a in hg:
        for b in pg:
            best = min(best, mujoco.mj_geomDistance(m, d, a, b, 0.05, fr))
    return round(best * 1e3, 3)


def press(rig, fol, i, cand, ph):
    """Hold demo row i; servo the pressing arm along the site normal on the true pad-support force until the bond."""
    from ikea_runtime import rotate
    M = rig.M; arm, p, s, typ = cand; key = 'pad|%s|%s|%s' % (p, s, typ)
    _, _, qs = M.site_target(s, typ); n = rotate(qs, [0, 0, 1])
    push = np.zeros(7); out = dict(row=i, arm=arm, pad=p, site='%s|%s' % (s, typ), k0=len(rig.ex)); trace = []
    bonded = lambda: M.active('adhesive_%s_%s' % (s, p))
    t = 0; t_bond = None; fmax = 0.; travel = 0.; out['touch0'] = p in rig.touching(arm); out['gap_hand_mm'] = hand_gap(rig, arm, p)
    while t < MAXT:
        f = M.force(p, s); met = M.metrics.get(key, {}); gap = met.get('gap_m', 0.)
        fmax = max(fmax, f)
        dz = float(np.clip(KP * (F_TGT - f), -STEP, STEP))
        if gap < GAP_MIN:
            dz = -STEP
        if travel > REACH_MAX and f < 0.2:
            out['reach_failed'] = True; break
        other = [b for b in rig.touching(arm) if b != p]
        if other and f < 0.2:
            out['blocked'] = other; break
        travel += dz
        dq = rig.ik(arm, -n * dz)
        push += dq
        if arm == 'left': fol.offL = fol.offL + dq
        else: fol.off = fol.off + dq
        r = ph(i, fol.row(i))                # the pad's lateral / orientation steering stays on
        rig.move(r); t += 1
        if t % 10 == 0:
            trace.append([t, round(f, 3), round(met.get('contact_N', 0.), 3), round(gap * 1e3, 4), round(met.get('lateral_m', 0.) * 1e3, 3),
                          round(met.get('dwell_s', 0.), 4), round(met.get('speed_m_s', 0.), 4)])
        if bonded() and t_bond is None:
            t_bond = t
        if t_bond is not None and t - t_bond >= HOLD:
            break
    out.update(bonded=bool(bonded()), ticks=t, t_bond=t_bond, fmax=round(fmax, 3), trace=trace[-60:])
    # the push removed with the row still held: the bonded pad stays put
    for k in range(RAMP):
        dq = -push / RAMP
        if arm == 'left': fol.offL = fol.offL + dq
        else: fol.off = fol.off + dq
        rig.move(fol.row(i))
    out['view'] = pad_view(rig, p); out['k1'] = len(rig.ex)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ck', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--upto', type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rig = Rig(); D = demo_rows(); fol = Follower(rig, D); t0 = time.time()
    st = pickle.load(open(a.ck, 'rb')); rig.set(st)
    if 'fol' in st:
        fol_set(fol, st['fol'])
    else:
        fol.off = np.array(st.get('fol_off', np.zeros(7)))
    i = int(st['demo_i']); end = a.upto or len(D)
    done = set(st.get('pads_done', [])); log = dict(start=i, ck=a.ck, presses=list(st.get('presses', [])))
    ph0 = pad_hook(rig, fol, 0, len(D)); ch = centre_hook(rig, fol)
    ph = lambda k, r: ch(k, ph0(k, r))

    def save(name, i_next):
        s2 = rig.get(); s2['fol'] = fol_state(fol); s2['demo_i'] = i_next; s2['pads_done'] = sorted(done)
        s2['presses'] = log['presses']; rig.save(s2, os.path.join(a.out, name))

    while i < end:
        rig.move(ph(i, fol.row(i)))
        c = candidate(rig, done, i)
        if c is not None:
            save('pre_%s.pkl' % c[1], i)            # resume here = re-run this demo row's tick, then the press
            out = press(rig, fol, i, c, ph); log['presses'].append(out); done.add(c[1])
            print('press', json.dumps({k: v for k, v in out.items() if k != 'trace'}, default=float), round(time.time() - t0), flush=True)
            if out['bonded']:
                fol.ramp_left(i + 200, i + 1200); fol.ramp_out(i + 200, i + 1200)
                save('bond_%s.pkl' % c[1], i + 1)
            else:
                print('trace', out['trace'], flush=True)
            json.dump(log, open(os.path.join(a.out, 'log.json'), 'w'), default=float)
        i += stride(i)
        if i % 2000 < stride(i):
            print('row', i, len(rig.ex), dict(M_pad_at=rig.M.pad_at), round(time.time() - t0), flush=True)
    np.savez_compressed(os.path.join(a.out, 'ref.npz'), ctrl=np.array(rig.ex), substeps=np.int64(4),
                        timestep=np.float64(rig.s.timestep), actuators=np.array(rig.s.actuators))
    save('end.pkl', i)
    log.update(rows=len(rig.ex), ops=dict(rig.M.ops), wall=time.time() - t0, pad_at=dict(rig.M.pad_at), grip_wait_ticks=rig.grip_wait_ticks,
               active=[n for n in rig.M.eq if rig.M.active(n)])
    if i >= len(D):
        fc = rig.s.final_check()
        log['final'] = dict(success=fc['success'], state_sha=fc['state_sha'], settled_sha=fc['settled_sha'],
                            end={k: fc['end'][k] for k in ('success', 'goal', 'released', 'at_rest', 'breakdown')},
                            settled={k: fc['settled'][k] for k in ('success', 'goal', 'released', 'at_rest', 'breakdown')},
                            failures=fc['end']['report'].get('failures'), sfail=fc['settled']['report'].get('failures'),
                            terminal=fc['end']['terminal'])
    json.dump(log, open(os.path.join(a.out, 'log.json'), 'w'), default=float)
    print('DONE', len(rig.ex), json.dumps(log['ops'], default=float), json.dumps(log.get('final'), default=float), round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
