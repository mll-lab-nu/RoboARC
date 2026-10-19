"""Op-by-op search driver: from the checkpoint of op k-1, try CANDIDATES for op k in parallel; keep the first that welds."""
import json, os, pickle, sys, time, argparse
import multiprocessing as mp
import numpy as np
import vesken_oracle as V

ap = argparse.ArgumentParser()
ap.add_argument('--demo', default='refs/vesken.npz')
ap.add_argument('--ref', default='tools/ref')
ap.add_argument('--dir', default='ck')
ap.add_argument('--from-op', type=int, default=0)
ap.add_argument('--to-op', type=int, default=22)
ap.add_argument('--workers', type=int, default=6)
ap.add_argument('--cands', default=None, help='json list overriding CANDIDATES')
ap.add_argument('--dev-max', type=float, default=3.0)
ap.add_argument('--ax-max', type=float, default=2.0, help='deg: the axis error at the weld (a tilted column moves its top under the next tray)')
ap.add_argument('--lat-max', type=float, default=1.0, help='mm: the lateral error at the weld (margin under the 0.8 mm gate: the welds are soft)')
ap.add_argument('--op-cands', default=None, help='json file {op: [candidates]}: the candidates of those ops instead')
ap.add_argument('--trial-rows', type=int, default=150000, help='a candidate that executes more rows than this is dropped')
ap.add_argument('--box-rot-max', type=float, default=180.0, help='feet and caps not placed yet: largest rotation (deg, modulo quarter turns about the part axis) from the reference run')
ap.add_argument('--box-dev-max', type=float, default=60.0, help='feet and caps (the parts box): the approach feed-forward finds them')
a = ap.parse_args()
os.makedirs(a.dir, exist_ok=True)
RUN = V.Runner(a.demo, a.ref)
CANDS0 = json.loads(a.cands) if a.cands else V.CANDIDATES
OPC = {int(k): v for k, v in json.load(open(a.op_cands)).items()} if a.op_cands else {}
CANDS = CANDS0
CK = None

class TooLong(Exception):
    pass

def trial(args):
    k, idx = args
    RUN.restore(CK) if CK is not None else None
    t = time.time(); log = []
    n0 = len(RUN.s.rows); emit = RUN.emit
    def capped(row):
        if len(RUN.s.rows) - n0 > a.trial_rows:
            raise TooLong()
        emit(row)
    RUN.emit = capped
    try:
        res = RUN.run_op(k, CANDS[idx], log)
    except TooLong:
        return dict(key=RUN.ops[k]['key'], idx=idx, ok=False, waited=0, rows=len(RUN.s.rows), disturb_mm=0, disturb_part=None, gap_mm=None,
                    lat_mm=None, ax=None, wall=round(time.time() - t, 1), rr=RUN.rr_max, params=CANDS[idx], dev={n: 1e9 for n in RUN.parts}, log=['too_long'], metrics=None), None
    res['wall'] = round(time.time() - t, 1); res['idx'] = idx; res['log'] = log
    import math
    H = RUN.plan['ops'][k]['H']
    res['box_rot'] = {n: round(math.degrees(float(np.linalg.norm(V.sym_rotvec(RUN.d.xquat[RUN.bid[n]], RUN.pref(n, H)[1])))), 1)
                      for n in [x['part'] for x in RUN.ops[k + 1:]] if n.startswith(('foot', 'cap'))}
    ck = RUN.checkpoint() if res['ok'] else None
    return res, ck

for k in range(a.from_op, a.to_op + 1):
    CANDS = OPC.get(k, CANDS0)
    prevp = os.path.join(a.dir, f'op{k-1:02d}.pkl')
    CK = pickle.load(open(prevp, 'rb')) if k > 0 else None
    if k == 0:
        CK = RUN.checkpoint()
    with mp.get_context('fork').Pool(min(a.workers, len(CANDS))) as pool:
        results = pool.map(trial, [(k, i) for i in range(len(CANDS))])
    best = None
    later = [x['part'] for x in RUN.ops[k + 1:]]
    for res, ck in results:
        res['later_dev'] = max([res['dev'][n] for n in later if not n.startswith(('foot', 'cap'))], default=0.)
        res['box_dev'] = max([res['dev'][n] for n in later if n.startswith(('foot', 'cap'))], default=0.)
        brot = max(res.get('box_rot', {}).values(), default=0. if 'box_rot' in res else 1e9)
        dv = sorted(res['dev'].items(), key=lambda x: -x[1])[:4]
        print(json.dumps({x: res[x] for x in ('key', 'idx', 'ok', 'waited', 'rows', 'disturb_mm', 'disturb_part', 'gap_mm', 'lat_mm', 'ax', 'wall', 'rr')}, default=str), 'dev', dv, flush=True)
        print('   later_dev', res['later_dev'], 'box_dev', res['box_dev'], 'box_rot', res.get('box_rot'), flush=True)
        if res['ok'] and (res['lat_mm'] or 0) < a.lat_max and (res['ax'] or 0) < a.ax_max and res['later_dev'] < a.dev_max \
                and res['box_dev'] < a.box_dev_max and brot < a.box_rot_max:
            res['score'] = res['lat_mm'] + res['ax'] / 2 + res['later_dev'] / 10      # the cleanest weld, then the least disturbance
            if best is None or res['score'] < best[0]['score']:
                best = (res, ck)
    json.dump([r for r, _ in results], open(os.path.join(a.dir, f'op{k:02d}_results.json'), 'w'), default=str, indent=1)
    if best is None:
        print('FAIL op', k, flush=True); sys.exit(1)
    print('KEEP op', k, best[0]['key'], 'cand', best[0]['idx'], best[0]['params'], flush=True)
    pickle.dump(best[1], open(os.path.join(a.dir, f'op{k:02d}.pkl'), 'wb'))
