#!/usr/bin/env python3
"""stenerik closed-loop reference builder (final physics). The demo is the phase guide, executed within the joint-speed
limits (Rig.move: a row farther than the limit is reached over the ticks it needs); per screw cycle the true state
closes the loop:
  place  while the right hand carries a screw to its insert, the screw is steered onto the insert axis (lateral error +
         tilt, damped-least-squares IK, accumulated in a right-arm offset), the offset ramped out after the release;
  drive  from the demo row where the screwdriver arrives over the screw: st_lib.drive (centre on the true axis, lower,
         spin with a 5-15 N push, zero torque at the seat until the retention weld engages, 0.2 s hold); the demo then
         resumes at the end of its own spin with the right arm offset to where the arm is, ramped out on the way back.
The Franka Hand slew (rcb_ikea.GRIPPER_WIDTH_SPEED_M_S): Rig.move holds each row until the hand commands have reached
it (st_lib.GRIP_WAIT), so every grasp and release completes before the demo goes on.
Usage: build.py --out DIR [--upto N] [--from-ck CK --start ROW] ; writes DIR/ref.npz (the executed rows), DIR/log.json,
checkpoints DIR/ck_<row>.pkl at each cycle boundary."""
import argparse, json, os, sys, time
import numpy as np
from st_lib import Rig, Follower, demo_rows, drive, place_hook, tip_hook, align_state, untorque, centre, pad_hook

# demo rows of each cycle (the right hand): screw grab, screw release, driver arrives over the screw (s0), the end of
# the demo's own spin (s1), driver released after it is put back; k = the insert the screw goes to
STOP_S0 = int(os.environ.get('ST_STOP_S0', '-1'))
MODE = os.environ.get('ST_MODE', 'demo_spin')   # demo_spin: the demo's own spin with the tip kept on the axis
CYCLES = json.loads(os.environ.get('ST_CYCLES', 'null')) or [
    dict(k=0, grab=30980, rel=35200, s0=38960, s1=48240, drel=50080),
    dict(k=1, grab=51940, rel=56160, s0=59920, s1=69740, drel=71680),
    dict(k=2, grab=73540, rel=77560, s0=81320, s1=90020, drel=91920, mode='drive'),   # under the Franka Hand slew the
    #   released screw lands perched on the insert's thread start and leans ~7 deg (the old reference's landed meshed):
    #   driven closed loop, the drive stands it up first (st_lib.straighten)
    dict(k=3, grab=97960, rel=102100, s0=105860, s1=114560, drel=116480),
    dict(k=4, grab=118340, rel=122500, s0=126240, s1=173420, drel=180620, mode='drive'),  # the demo's re-grip ratchet
    #   advances this screw only ~1 mm on the final physics: driven closed loop instead, the demo resumes after its ratchet
    dict(k=5, grab=182300, rel=188720, s0=193600, s1=203560, drel=206120, mode='drive'),  # the demo arrives 2.5 mm
    #   off the axis with its wings 37 deg off the slots: driven closed loop
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--upto', type=int, default=0)
    ap.add_argument('--from-ck', default=None); ap.add_argument('--start', type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rig = Rig(); D = demo_rows(); fol = Follower(rig, D); log = dict(cycles=[]); t0 = time.time()
    end = a.upto or len(D); i = a.start
    if a.from_ck:
        st = rig_load = __import__('pickle').load(open(a.from_ck, 'rb')); rig.set(st); fol.off = np.array(st.get('fol_off', np.zeros(7)))
    for c in CYCLES:
        if c['s1'] <= i:
            continue
        cl = dict(c)
        # follow to the grab; then carry with the place hook from 40% of the way to the release
        h0 = c['grab'] + int(0.4 * (c['rel'] - c['grab']))
        fol.run(i, h0); i = h0
        tcp = rig.d.site_xpos[rig.site['right']]
        held = sorted([s for s in rig.screws if s in rig.touching('right')], key=lambda s: np.linalg.norm(rig.pose(s)[0] - tcp))
        cl['held'] = held
        if held:
            hk = place_hook(rig, fol, held[0], c['k'], h0, c['rel'])
            fol.run(h0, c['rel'], hook=hk)
        else:
            fol.run(h0, c['rel'])
        i = c['rel']
        cl['placed'] = {s: [float(x) for x in rig.screw_state(s)] for s in held}
        fol.ramp_out(c['rel'] + 200, c['rel'] + 800)
        fol.run(i, c['s0']); i = c['s0']
        s = rig.nearest_screw_to_tip(); cl['driven'] = s
        st = rig.get(); st['fol_off'] = fol.off.tolist(); st['demo_i'] = c['s0']
        rig.save(st, os.path.join(a.out, 'cks0_%07d.pkl' % c['s0']))
        print('s0', c['k'], s, [round(float(x), 5) for x in rig.screw_state(s)], json.dumps(align_state(rig)), round(time.time() - t0), flush=True)
        if STOP_S0 == c['k']:
            print('stopped at s0', c['k'], s, flush=True); return
        lat, h = rig.tip(s); cl['tip_at_s0'] = [float(np.linalg.norm(lat)) * 1e3, h * 1e3]
        if c.get('mode', MODE) == 'demo_spin':
            if cl['tip_at_s0'][0] > 1.0:       # the demo arrives off the true axis: centre first (the demo row held)
                cl['precentre'] = centre(rig, s, max(h, 1.5e-3)); fol.set_from(c['s0'])
            fol.run(c['s0'], c['s1'], hook=tip_hook(rig, fol, s, c['s0'], c['s1']))
            cl['after_demo_spin'] = [float(x) for x in rig.screw_state(s)]
        if not rig.screw_state(s)[4]:
            dlog = []; out = drive(rig, s, dlog); cl['drive'] = out; cl['drive_log'] = dlog[-3:]
        untorque(rig)
        cl['align'] = align_state(rig)
        print('cycle', c['k'], s, json.dumps({k: cl.get(k) for k in ('tip_at_s0', 'after_demo_spin', 'drive', 'align')}, default=float), round(time.time() - t0), flush=True)
        st = rig.get(); st['fol_off'] = fol.off.tolist(); st['demo_i'] = c['s1']
        rig.save(st, os.path.join(a.out, 'cks1_%07d.pkl' % c['s1']))
        fol.set_from(c['s1']); i = c['s1'] + 1
        fol.ramp_out(c['s1'] + 300, c['drel'] - 300)
        fol.run(i, c['drel'] + 100); i = c['drel'] + 100
        cl['rows'] = len(rig.ex); cl['ops'] = dict(rig.M.ops)
        log['cycles'].append(cl)
        st = rig.get(); st['fol_off'] = fol.off.tolist(); st['demo_i'] = i
        rig.save(st, os.path.join(a.out, 'ck_%07d.pkl' % i))
        json.dump(log, open(os.path.join(a.out, 'log.json'), 'w'), default=float)
        if i >= end:
            break
    if i < end:
        # the pads: the demo's peel and press, each carried pad steered onto its site (pad_hook, left arm); after each
        # bond the left offset is ramped out on the way to the next pad
        ph = pad_hook(rig, fol, i, end); nb = [len(rig.M.pad_at)]; log['pads'] = []

        def hook(k, r):
            r = ph(k, r)
            if len(rig.M.pad_at) > nb[0]:
                nb[0] = len(rig.M.pad_at); fol.ramp_left(k + 200, k + 1200); fol.ramp_out(k + 200, k + 1200)
                log['pads'].append(dict(row=k, pad_at=dict(rig.M.pad_at), offL=fol.offL.round(4).tolist()))
                print('pad bonded', k, dict(rig.M.pad_at), flush=True)
            return r
        fol.run(i, end, hook=hook); i = end
    np.savez_compressed(os.path.join(a.out, 'ref.npz'), ctrl=np.array(rig.ex), substeps=np.int64(4),
                        timestep=np.float64(rig.s.timestep), actuators=np.array(rig.s.actuators))
    log['rows'] = len(rig.ex); log['ops'] = dict(rig.M.ops); log['wall'] = time.time() - t0; log['grip_wait_ticks'] = rig.grip_wait_ticks
    log['active'] = [n for n in rig.M.eq if rig.M.active(n)]
    json.dump(log, open(os.path.join(a.out, 'log.json'), 'w'), default=float)
    print('DONE', len(rig.ex), json.dumps(log['ops']), round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
