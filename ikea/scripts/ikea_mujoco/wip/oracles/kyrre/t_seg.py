"""Run a segment's demo rows with the guide runner (params P) from a checkpoint up to row STOP (no finish); save; report
the cycle screw and the seat every K rows."""
import sys, json, pickle, math, numpy as np
sys.path.insert(0, 'tools')
import ky_guide as U
ck, stop, screw, K, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
P = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
run = U.Runner('refs/kyrre_demo.npz', 'ref_g2'); run.restore(pickle.load(open(ck, 'rb'))); run.p = P; run.press_state = {}; run._kspin = None
p0 = run.d.xpos[run.bid['seat']].copy(); j = [o['part'] for o in run.mech.ops].index(screw)
for i in range(run.tau, stop):
    run.step_row(i)
    if i % K == 0:
        gap, lat, ang = run.mech.geometry_all()[:3]
        lat_k, h, a_, hx = run.rig().key_vs(screw)
        print(i, 'gap %.3f lat %.3f tilt %.2f | key h %.2f lat %.2f hx %.1f | seat moved %.2f' % (gap[j] * 1e3, lat[j] * 1e3, ang[j], h * 1e3, np.linalg.norm(lat_k) * 1e3, math.degrees(hx), np.linalg.norm(run.d.xpos[run.bid['seat']] - p0) * 1e3), flush=True)
run.tau = stop
pickle.dump(run.checkpoint(), open(out, 'wb'))
print('saved', out)
