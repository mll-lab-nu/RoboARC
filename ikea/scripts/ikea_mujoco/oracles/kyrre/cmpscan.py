"""Compare one op's trace in scan A (grip-wait rows) vs scan C (demo rows), aligned by the demo row index map.
usage: cmpscan.py scanA.jsonl scanC.jsonl idxmap.json OP C_FROM C_TO [STEP]"""
import json, sys, bisect
a = [json.loads(l) for l in open(sys.argv[1]) if '"r"' in l]
c = [json.loads(l) for l in open(sys.argv[2]) if '"r"' in l]
idx = json.load(open(sys.argv[3]))   # demo row -> A row (every 500th demo row)
op, c0, c1 = sys.argv[4], int(sys.argv[5]), int(sys.argv[6]); step = int(sys.argv[7]) if len(sys.argv) > 7 else 2000
ar = [x['r'] for x in a]
for x in c:
    if not (c0 <= x['r'] <= c1) or x['r'] % step: continue
    ra = idx[str(x['r'])]; j = min(bisect.bisect_left(ar, ra), len(a) - 1); y = a[j]
    print(x['r'], ra, y['r'], 'C', x['w'], x['ops'][op], x['L'], x['R'], round(x['gL']), round(x['gR']), '| A', y['w'], y['ops'][op], y['L'], y['R'], round(y['gL']), round(y['gR']))
