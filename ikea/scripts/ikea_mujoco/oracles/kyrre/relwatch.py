"""max relative drift/angle of the assembled stool (success monitor) row by row from a checkpoint, plus robot contacts.
usage: relwatch.py TRAJ CK.pkl TO [EVERY]"""
import json, pickle, sys
import rcb_ikea as R
traj, ck, to = sys.argv[1], sys.argv[2], int(sys.argv[3]); ev = int(sys.argv[4]) if len(sys.argv) > 4 else 50
s = R.Session('kyrre'); c = R.load_trajectory(traj, s.nu, s.timestep); s.set_state(pickle.load(open(ck, 'rb'))); i = len(s.rows)
mon = s.S.monitor; m, d = s.model, s.data; prev = list(mon.max_rel)
while i < to:
    s.step(c[i]); i += 1
    if i % ev == 0:
        cur = list(mon.max_rel); t = s.validity.terminal()
        if cur != prev or t['robot_part_contacts']:
            print(i, [round(cur[0] * 1e3, 3), round(cur[1], 3)], t['robot_part_contacts'][:4])
        prev = cur
