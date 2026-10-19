#!/usr/bin/env python3
"""Segment-by-segment search driver for uppsta_oracle.py: from the checkpoint of segment k-1, run segment k with its kept
parameters (--params) or the candidates; keep the first that passes the gate (every part within --dev-max mm of its guide
pose, the screws' welds as in the guide run, no breakdown), unless a later one is at least 0.5 mm closer. With --params,
a segment whose kept parameters fail falls back to the same parameters at the demo's pace (tempo_max 1, when the ref dir
has a tempo profile), (tempo_max 2 first),
then to every candidate at the profile's tempo and at the demo's pace (--fallback 1).
Checkpoints in --dir. The reference (2026-10-18), every segment from the birth:
  python search.py --demo uppsta_interp.npz --ref REFDIR --dir ck --from-seg 0 --params params.json --spin-max-seg 8:25 \
      --rcb-ikea <rcb_ikea.py with the Franka Hand slew>
  (REFDIR = the guide's track.npz, sched.json, events.json and ref/tempo.npz; the image's scene, image 0.3.1)"""
import json, os, pickle, sys, time, argparse
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--demo', required=True); ap.add_argument('--ref', required=True); ap.add_argument('--dir', default='ck')
ap.add_argument('--from-seg', type=int, default=0); ap.add_argument('--to-seg', type=int, default=10 ** 6)
ap.add_argument('--workers', type=int, default=8); ap.add_argument('--cands', default=None)
ap.add_argument('--dev-max', type=float, default=3.0)
ap.add_argument('--spin-max', type=float, default=8.0, help='deg: a free screw\'s spin vs the guide, modulo 60, after the look-ahead')
ap.add_argument('--end-max', type=float, default=10.0, help='mm: the worst deviation allowed at the segment end itself'); ap.add_argument('--scene-dir', default=None)
ap.add_argument('--rcb-ikea', default=os.path.join(HERE, '..', '..', 'shared'))
ap.add_argument('--oracle', default=HERE)
ap.add_argument('--start-ck', default=None, help='checkpoint of segment from-seg - 1 (default DIR/seg<k-1>.pkl)')
ap.add_argument('--dev-max-seg', default='', help='per-segment dev-max overrides, e.g. 12:5,13:4')
ap.add_argument('--spin-max-seg', default='', help='per-segment spin-max overrides, e.g. 8:25')
ap.add_argument('--fallback', type=int, default=1)
ap.add_argument('--params', default=None, help='params.json: a fixed parameter set per segment (no search)')
ap.add_argument('--follow', default=None, help='a guide run still being written (diag.py --track --flush DIR): before each '
                'segment wait until its track covers the segment, then rebuild the schedule into --ref')
a = ap.parse_args()
sys.path[:0] = [os.path.abspath(a.rcb_ikea), a.oracle]
import uppsta_oracle as U
import mksched
os.makedirs(a.dir, exist_ok=True)
RUN = U.Runner(a.demo, a.ref, a.scene_dir)
CANDS = json.loads(a.cands) if a.cands else U.CANDIDATES
FIXED = json.load(open(a.params))['segs'] if a.params else None
CK = None
SPINSEG = {int(k): float(v) for k, v in (s.split(':') for s in a.spin_max_seg.split(',') if s)}
DEVSEG = {int(k): float(v) for k, v in (s.split(':') for s in a.dev_max_seg.split(',') if s)}
print('segments', len(RUN.bounds), RUN.bounds, flush=True)


def params_of(k, idx):
    """idx -1: the kept parameters; -3: the same at tempo_max 2; -2: at the demo's pace (tempo_max 1); 0..: a candidate;
    100+i: candidate i at the demo's pace"""
    if idx == -1: return dict(FIXED[k])
    if idx == -2: return dict(FIXED[k], tempo_max=1)
    if idx == -3: return dict(FIXED[k], tempo_max=2)
    if idx >= 100: return dict(CANDS[idx - 100], tempo_max=1)
    return dict(CANDS[idx])


def trial(args):
    k, idx = args
    RUN.restore(CK)
    t = time.time()
    res = RUN.run_seg(k, params_of(k, idx))
    res['wall'] = round(time.time() - t, 1); res['idx'] = idx
    return res, RUN.checkpoint()


LOOSE = {'hex_key': 10., 'hammer': 10.}   # tools success.py does not place: their deviation counts a tenth


def score(res):
    """mm: a part released at the segment end must stay: its deviation from the guide after the look-ahead rows (the
    next segment's first rows); every other part may be in a contact transient at the boundary or already in the next
    operation's hands, so the smaller of its deviations at the end and after the look-ahead counts. A part pressed
    deeper before its release has an allowance (press mm + 1.5 mm). Returns (score, the worst at the end itself)."""
    extra = 1000 * res['params'].get('press', 0) + 1.5 if res['params'].get('press') else 0.
    rel = set(res.get('released', []))
    look = res.get('dev_look') or res['dev']
    worst = 0.
    for n, v in res['dev'].items():
        w = look[n] if n in rel else min(v, look[n])
        w = w / LOOSE.get(n, 1.)
        worst = max(worst, w - (extra if n in rel else 0.))
    end = max([v / LOOSE.get(n, 1.) for n, v in res['dev'].items() if n not in rel], default=0.)   # a released part may
    # still be settling at the end row: its look-ahead deviation is the test
    return worst, end


def gate(res):
    sc, end = score(res)
    res['score'] = round(sc, 2); res['score_end'] = round(end, 2)
    spin_ok = all(v < SPINSEG.get(res['seg'], a.spin_max) for n, v in (res.get('spin_look') or {}).items() if n in res.get('released', []) and n not in res['welded'])
    dmax = DEVSEG.get(res['seg'], a.dev_max)
    return (res['breakdown'] is None and res['welded'] == res['welded_guide'] and sc < dmax
            and end < a.end_max and spin_ok)


for k in range(a.from_seg, min(a.to_seg + 1, len(RUN.bounds))):
    if a.follow:
        follow(k)
    ckp = a.start_ck if (k == a.from_seg and a.start_ck) else os.path.join(a.dir, f'seg{k-1:03d}.pkl')
    CK = pickle.load(open(ckp, 'rb')) if k > 0 else RUN.checkpoint()
    if FIXED:                               # the kept parameters first; the candidates only if they fail the gate
        results = [trial((k, -1))]
        for alt in ((-3, -2) if 'tempo_max' not in FIXED[k] else ()):
            if any(gate(r) for r, _ in results) or not a.fallback or RUN.tempo is None:
                break
            print('fixed params fail seg', k, '-> tempo_max', 2 if alt == -3 else 1, flush=True)
            results += [trial((k, alt))]
        if not any(gate(r) for r, _ in results) and a.fallback:
            print('fixed params fail seg', k, '-> candidates', flush=True)
            idx = list(range(len(CANDS))) + ([100 + i for i in range(len(CANDS))] if RUN.tempo is not None else [])
            with mp.get_context('fork').Pool(min(a.workers, len(idx))) as pool:
                results += pool.map(trial, idx and [(k, i) for i in idx])
    else:                                   # the default first (in this process); the others only if it fails the gate
        results = [trial((k, 0))]
        if not gate(results[0][0]) and len(CANDS) > 1:
            with mp.get_context('fork').Pool(min(a.workers, len(CANDS) - 1)) as pool:
                results += pool.map(trial, [(k, i) for i in range(1, len(CANDS))])
    best = None
    for res, ck in results:
        dv = sorted(res['dev'].items(), key=lambda x: -x[1])[:4]
        gate(res)
        print(json.dumps({x: res[x] for x in ('seg', 'idx', 'r0', 'r1', 'rows', 'score', 'max_dev', 'max_dev_look', 'worst', 'spin_look', 'welded', 'welded_guide',
                                              'breakdown', 'depth_mm', 'wall', 'rr')}, default=str), 'dev', dv, flush=True)
        if gate(res) and (best is None or res['score'] < best[0]['score'] - 0.5):
            best = (res, ck)
    json.dump([r for r, _ in results], open(os.path.join(a.dir, f'seg{k:03d}_results.json'), 'w'), default=str, indent=1)
    if best is None:
        print('FAIL seg', k, flush=True)
        r0 = min(results, key=lambda x: x[0]['score'])
        pickle.dump(r0[1], open(os.path.join(a.dir, f'seg{k:03d}_best_failed.pkl'), 'wb'))
        sys.exit(1)
    print('KEEP seg', k, 'cand', best[0]['idx'], best[0]['params'], 'score', best[0]['score'], 'max_dev', best[0]['max_dev'],
          'look', best[0]['max_dev_look'], flush=True)
    pickle.dump(best[1], open(os.path.join(a.dir, f'seg{k:03d}.pkl'), 'wb'))
    json.dump(best[0]['params'], open(os.path.join(a.dir, f'seg{k:03d}_params.json'), 'w'))
