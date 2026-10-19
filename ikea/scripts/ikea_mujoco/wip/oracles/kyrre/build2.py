#!/usr/bin/env python3
"""kyrre closed-loop reference builder (final physics: prisms, the arm clamp, the welded-part contact switch-off). The
author's demo is the phase guide, executed within the joint-speed limits (Rig.move); the true state closes the loop:
  place   while a hand carries the cycle's screw to its hole (before the thread), the screw is steered onto the hole's
          true axis (ky_lib.place_hook); the offset is ramped out after the release;
  drive   the demo's own hex-key strokes (its lever strokes are robust on the final physics), but driven to the weld:
          if the demo's drive window ends (its own weld row on the author's geometry) before the screw's weld latches,
          its last stroke cycle (from one stroke end to the next) is repeated until it does; a weld that latches early
          ends the window at the end of that stroke;
  resume  the demo resumes at its own resume row (s1) with the arms' offsets ramped out.
usage: build2.py --out DIR [--from-ck CK] [--upto-cycle K]; checkpoints DIR/ck_<k>_<tag>.pkl; DIR/ref.npz at the end."""
import argparse, json, math, os, time
import numpy as np
from ky_lib import Rig, Follower, demo_rows, place_hook, key_hook, ratchet, handle_yaw, arm_sector, ARM

CYCLES = json.load(open(os.environ.get('KY_CYCLES', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cycles.json'))))['cycles']
MAX_EXTRA = int(os.environ.get('KY_MAX_EXTRA', '24'))
# Pace: inside a drive window, while the key is lifted over the screw (its tip more than FREE_H above the screw head, on
# the axis within 5 mm: the demo's return swing between strokes) the demo is followed every STRIDE-th row (Rig.move still
# keeps every tick within the joint-speed limits); everywhere else every row.
STRIDE = int(os.environ.get('KY_STRIDE', '1'))
FREE_H = float(os.environ.get('KY_FREE_H', '1.5e-3'))
KEY_HOOK = int(os.environ.get('KY_KEY_HOOK', '0'))
# a key tip below ENG_H above the screw head (and within 2 mm of its axis) is in a stroke: the demo drives some screws with
# the tip fully in the socket (about -0.2 mm), others jammed higher in it (about +1.0 mm); between strokes it lifts the
# key to about +2 mm
ENG_H = float(os.environ.get('KY_ENG_H', '1.5e-3'))


def hole(rig, o):
    ps, Rs = rig.pose(o['host']); return ps + Rs @ np.array(o['target_pos']), Rs @ np.array(o['axis'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--from-ck', default=None)
    ap.add_argument('--upto-cycle', type=int, default=99)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rig = Rig(); D = demo_rows(); fol = Follower(rig, D); t0 = time.time()
    i, k = 0, 0; log = dict(cycles=[])
    if a.from_ck:
        st = rig.load(a.from_ck); i, k = st['demo_i'], st['cycle']; fol.set_state(st['fol']); log = st.get('log', log)

    def save(tag):
        st = rig.get(); st.update(demo_i=i, cycle=k, fol=fol.state(), log=log)
        rig.save(st, os.path.join(a.out, 'ck_%d_%s.pkl' % (k, tag)))
        json.dump(log, open(os.path.join(a.out, 'log.json'), 'w'), default=float)

    def engaged(s):
        lat, h, ang, hx = rig.key_vs(s)
        return h < ENG_H and np.linalg.norm(lat) < 2e-3

    while k < len(CYCLES) and k < a.upto_cycle:
        c = CYCLES[k]; s = c['screw']; o = rig.ops[rig.opi[s]]; W, s1 = c['demo_weld'], c['s1']
        cl = dict(screw=s, start_row=len(rig.ex), demo_i=i); save('start')
        hk = place_hook(rig, fol, s, hole(rig, o)); released = None
        kh = key_hook(rig, fol, s, c['key_arm']) if KEY_HOOK else (lambda j, r: r)
        ends = []; zs = []; yaws = []; was = False; welded_at = None; took = None
        # follow the demo to its weld row W (the place hook steering the carried screw; stroke ends recorded)
        def free(s):
            lat, h, ang, hx = rig.key_vs(s)
            return h > FREE_H and h < 0.03 and np.linalg.norm(lat) < 5e-3

        while i < W:
            rig.move(kh(i, hk(i, fol.row(i)))); i += STRIDE if (STRIDE > 1 and free(s)) else 1
            if released is None and hk.steered and s not in rig.touching(hk.steered) and 0 < rig.op_view(s)[0] < 8e-3:
                released = i; fol.ramp_out(i + 100, i + 700, arms=(hk.steered,)); cl['released'] = i; cl['steered'] = hk.steered
            if took is None and c['key_arm'] and 'hex_key' in rig.touching(c['key_arm']) and rig.key_vs(s)[1] < 8e-3 \
                    and np.linalg.norm(rig.key_vs(s)[0]) < 4e-3:
                took = i; cl['key_over_screw'] = i; save('take')
                if os.environ.get('KY_STOP_TAKE'): print('stop at take', i, flush=True); return
            e = engaged(s)
            if was and not e: ends.append(i); zs.append(rig.pose(s)[0].copy()); yaws.append(('end', handle_yaw(rig, s), i))
            if e and not was: yaws.append(('start', handle_yaw(rig, s), i))
            was = e
            if rig.welded(s) and welded_at is None:
                welded_at = i; cl['welded_demo_row'] = i
            if welded_at is not None and not e:
                break                                      # early weld: the window ends at the end of this stroke
        cl['strokes_in_window'] = len(ends); cl['gap_at_W'] = rig.op_view(s)[0]
        print('cycle', k, s, 'window to', i, 'strokes', len(ends), 'welded', welded_at, 'gap %.3f' % (rig.op_view(s)[0] * 1e3), 'rows', len(rig.ex), round(time.time() - t0), flush=True)
        save('window')
        # the finish: the demo's strokes end at its own weld row W (on the author's geometry); if the screw's weld has not
        # latched, ky_lib.ratchet drives it to the weld with strokes of the demo's own arc (the handle's free sector: a
        # multiple of 60 deg, measured from the window's strokes), starting with the key turned back to a stroke's start
        extra = 0
        if not rig.welded(s):
            arcs = [abs((b[1] - a[1] + math.pi) % (2 * math.pi) - math.pi) for a, b in zip(yaws, yaws[1:]) if a[0] == 'start' and b[0] == 'end']
            arcs = [x for x in arcs if x > math.radians(20)]
            arc = float(np.median(arcs)) if arcs else math.pi / 3
            delta = (math.pi / 3) * max(1, int((arc + math.radians(10)) // (math.pi / 3)))
            cl['finish_arc_deg'] = [round(math.degrees(arc), 1), round(math.degrees(delta))]
            # the free sector of the turn, the arm included (ky_lib.arm_sector), from where the demo's last stroke left
            # the key; the ratchet's strokes stay inside it
            sec = arm_sector(rig, s, c['key_arm'], margin_deg=5.); cl['finish_sector_deg'] = [round(math.degrees(x)) for x in sec]
            rl = []; out = ratchet(rig, c['key_arm'], s, delta=delta, sector=sec, log=rl, max_strokes=MAX_EXTRA)
            extra = out['strokes']; cl['finish'] = out; cl['finish_log'] = rl[-3:]
            print('  finish', json.dumps(out), cl['finish_arc_deg'], flush=True)
        cl['extra_strokes'] = extra
        if not rig.welded(s):
            cl['fail'] = 'not welded after %d extra strokes' % extra; log['cycles'].append(cl); save('fail'); print('FAIL', k, cl); return
        i = s1; fol.set_from(i); fol.ramp_out(i + 1, i + 1500)
        cl['resume'] = i; cl['rows'] = len(rig.ex); log['cycles'].append(cl); k += 1
        cl['seat_mm'] = [round(float(x) * 1e3, 2) for x in rig.pose('seat')[0]]
        print('cycle', k - 1, 'done', json.dumps({x: cl.get(x) for x in ('strokes_in_window', 'extra_strokes', 'rows', 'seat_mm')}), round(time.time() - t0), flush=True)
        if np.linalg.norm(rig.pose('seat')[0][:2]) > 5e-3:
            print('FAIL seat moved', cl['seat_mm'], flush=True); save('seatmoved'); return
        save('done')
    while i < len(D):
        rig.move(fol.row(i)); i += 1
    for t in range(4000):
        rig.step(rig.ctrl)
        if t % 200 == 199:
            term = rig.s.validity.terminal()
            if term['at_rest'] and term['released']: break
    fc = rig.s.final_check()
    log['final'] = dict(success=fc['success'], sha=fc['state_sha'], settled=fc['settled_sha'],
                        end={x: fc['end'][x] for x in ('goal', 'released', 'at_rest', 'breakdown')},
                        report={x: fc['end']['report'].get(x) for x in ('welds_engaged', 'operations_ok', 'heads_ok', 'rigid', 'released', 'hold_s')})
    np.savez_compressed(os.path.join(a.out, 'ref.npz'), ctrl=np.array(rig.ex), substeps=np.int64(4),
                        timestep=np.float64(rig.s.timestep), actuators=np.array(rig.s.actuators))
    log['rows'] = len(rig.ex); log['wall'] = time.time() - t0
    save('end'); print('DONE', len(rig.ex), json.dumps(log['final'], default=str), round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
