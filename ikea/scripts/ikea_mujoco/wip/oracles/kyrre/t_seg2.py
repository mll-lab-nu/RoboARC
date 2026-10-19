"""From a checkpoint saved at row R0 (t_seg.py), continue to STOP printing every K rows: key vs screw, screw turn, offsets,
guide key height vs guide screw."""
import sys, json, pickle, math, numpy as np
sys.path.insert(0, 'tools')
import ky_guide as U
from ky_lib import DemoTrack, KEY_SHORT_TIP
ck, r0, stop, screw, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
run = U.Runner('refs/kyrre_demo.npz', 'ref_g2'); run.restore(pickle.load(open(ck, 'rb'))); run.p = {}; run.press_state = {}; run._kspin = None
T = DemoTrack('runs/track_np/track.npz', run.rig()); rig = run.rig()
yp = rig.screw_turn(screw); turn = 0.
for i in range(r0, stop):
    run.step_row(i)
    y = rig.screw_turn(screw); turn += -((y - yp + math.pi) % (2 * math.pi) - math.pi); yp = y
    if i % K == 0:
        lat, h, a_, hx = rig.key_vs(screw)
        d2 = T.at(i); kp = d2.xpos[rig.key]; kR = d2.xmat[rig.key].reshape(3, 3); sp = d2.xpos[rig.bid[screw]]; sR = d2.xmat[rig.bid[screw]].reshape(3, 3)
        tips = [kp + kR @ KEY_SHORT_TIP, kp]; tip = min(tips, key=lambda t: np.linalg.norm(t - sp)); hg = float((tip - sp) @ sR[:, 2])
        print(i, 'h %.2f (guide %.2f) lat %.2f hx %.1f | turn %.1f | gap %.3f | offL p %s r %s' % (h * 1e3, hg * 1e3, np.linalg.norm(lat) * 1e3, math.degrees(hx), math.degrees(turn),
              run.mech.geometry_all()[0][[o['part'] for o in run.mech.ops].index(screw)] * 1e3, (run.off['left']['p'] * 1e3).round(2).tolist(), run.off['left']['r'].round(3).tolist()), flush=True)
