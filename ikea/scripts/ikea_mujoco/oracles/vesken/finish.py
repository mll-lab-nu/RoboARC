"""From the last op checkpoint: the demo's tail (offsets ramped out), a rest hold, then save the reference + final_check.
usage: finish.py demo.npz ref_dir op22.pkl out.npz [params.json]"""
import sys, json, pickle, time, numpy as np, vesken_oracle as V
demo, ref, ck, out = sys.argv[1:5]
tail = json.load(open(sys.argv[5]))['tail'] if len(sys.argv) > 5 else dict(g=0.02, gr=0.01, rest_s=0.5)
run = V.Runner(demo, ref)
run.restore(pickle.load(open(ck, 'rb')))
t = time.time()
run.tail(tail)
fc = run.s.final_check()
run.s.save_trajectory(path=out)
e, st = fc['end'], fc['settled']
print(json.dumps(dict(success=fc['success'], rows=len(run.s.rows), end={k: e[k] for k in ('goal', 'released', 'at_rest', 'breakdown')},
      settled={k: st[k] for k in ('goal', 'released', 'at_rest', 'breakdown')}, passed=e['report']['operations_passed'], failed=[o for o in e['report'].get('operations', []) if not o.get('passed')] if isinstance(e['report'].get('operations'), list) else e['report'].get('operations'),
      orient=e['report']['orientation_deg'], floor=e['report']['floor_contact_N'], term=e['terminal'], sha=fc['state_sha'],
      settled_sha=fc['settled_sha'], rr=run.rr_max, wall=time.time() - t), default=str))
