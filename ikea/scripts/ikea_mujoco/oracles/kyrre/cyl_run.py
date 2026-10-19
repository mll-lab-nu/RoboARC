"""Fresh replay of a kyrre trajectory with progress (welds latched, clamp ticks), end-state pickle and final_check.
usage: cyl_run.py TASK TRAJ OUTPREFIX [--every N] [--from PKL]"""
import json, os, pickle, sys, time
import numpy as np
import rcb_ikea as R
task, traj, out = sys.argv[1:4]
every = int(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else 20000
ck_every = int(sys.argv[sys.argv.index('--ckpt-every') + 1]) if '--ckpt-every' in sys.argv else 0
ck_from = int(sys.argv[sys.argv.index('--ckpt-from') + 1]) if '--ckpt-from' in sys.argv else 0
t0 = time.time()
s = R.Session(task); init = s.state_sha()
c = R.load_trajectory(traj, s.nu, s.timestep)
start = 0
if '--from' in sys.argv:
    st = pickle.load(open(sys.argv[sys.argv.index('--from') + 1], 'rb')); s.set_state(st); start = len(s.rows)
log = open(out + '.log', 'a')
def note(**k):
    k.update(wall_s=round(time.time() - t0, 1)); log.write(json.dumps(k, default=str) + "\n"); log.flush()
note(event='start', task=task, rows=len(c), initial_state_sha=init, start=start, traj_sha=R.file_sha(traj),
     cpu=[l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0], host=os.uname().nodename)
m = s.S.mech
for i in range(start, len(c)):
    s.step(c[i])
    if ck_every and i + 1 >= ck_from and (i + 1) % ck_every == 0:
        os.makedirs(out + '.ck', exist_ok=True); pickle.dump(s.get_state(), open(out + '.ck/%07d.pkl' % (i + 1), 'wb'))
    if (i + 1) % every == 0:
        act = [k for k in m.eq if m.active(k)] if hasattr(m, 'eq') else None
        b = s.validity.terminal()['breakdown']
        note(row=i + 1, welds=len(act) if act is not None else None, slewed=s.slewed_ticks, breakdown=b)
        if b is not None:
            break
pickle.dump(s.get_state(), open(out + '.end.pkl', 'wb'))
fc = s.final_check()
res = dict(task=task, traj_sha256=R.file_sha(traj), rows=len(c), initial_state_sha=init, success=fc["success"],
           state_sha=fc["state_sha"], settled_sha=fc["settled_sha"], slewed_ticks=s.slewed_ticks,
           end={k: fc["end"][k] for k in ("goal", "released", "at_rest", "breakdown")},
           settled={k: fc["settled"][k] for k in ("goal", "released", "at_rest", "breakdown")},
           end_terminal=fc["end"]["terminal"], settled_terminal=fc["settled"]["terminal"],
           wall_s=round(time.time() - t0, 1), host=os.uname().nodename)
json.dump(res, open(out + '.json', 'w'), indent=1, default=str)
note(event='done', success=fc["success"], state_sha=fc["state_sha"], settled_sha=fc["settled_sha"], slewed=s.slewed_ticks)
print(json.dumps({k: res[k] for k in ('success', 'state_sha', 'settled_sha', 'slewed_ticks', 'end', 'settled', 'wall_s')}, default=str))
