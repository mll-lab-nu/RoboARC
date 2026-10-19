"""203 stress jobs for kyrre (7 strategies x 29, uppsta's mix and windows), the same seeds for both scene variants.
usage: mkjobs.py CKPT_DIR REF OUT.jsonl"""
import json, sys
ck, ref, out = sys.argv[1:4]
S = ("random", "prefix_random", "prefix_jerk", "mutate", "fast", "grip", "push")
with open(out, "w") as f:
    for i in range(203):
        f.write(json.dumps(dict(task="kyrre", strategy=S[i % 7], seed=770000 + i, ckpt=ck, ref=ref, min_s=3.0, max_s=10.0)) + "\n")
