#!/bin/bash
# Rebuild stegesund.npz: the chain of resumed builder runs that made it (each step resumes the previous step's stage
# checkpoint). OUT=<dir>; PYTHON with rcb_ikea on PYTHONPATH, IKEA_SCENES, STEG_DEMO (the shipped demo = the old
# oracles/stegesund/stegesund.npz, sha256 2181d216...) and STEG_GUIDE (rec_guide.py's output) set, numpy SIMD masked.
set -euo pipefail
H=$(cd "$(dirname "$0")/.." && pwd); R=$H/recipe; O=${OUT:?}; PY=${PYTHON:-python}
b() { $PY $H/build.py "$@"; }
b --out $O/1 --to press0 --tracks $R/t1_carry.json
b --out $O/2 --from $O/1/ck_press0.pkl --to press --tracks $R/t1_carry.json --set perturb=1e-06 perturb_j=1 perturb_from=47500
b --out $O/3 --from $O/2/ck_press.pkl --to shelf --tracks $R/t0_none.json --set perturb_to=72000
b --out $O/4 --from $O/3/ck_shelf.pkl --to bolt1in --tracks $R/t2_bolt1.json --set drives=1
b --out $O/5 --from $O/4/ck_bolt1in.pkl --to drive3 --tracks $R/t3_r4.json --set drives=1
b --out $O/6 --from $O/5/ck_drive3.pkl --to flip --tracks $R/t4_r6a.json --set drives=1
b --out $O/7 --from $O/6/ck_flip.pkl --to bolt2in --tracks $R/t4_r6a.json --set drives=1
b --out $O/8 --from $O/7/ck_bolt2in.pkl --to stand --tracks $R/t5_r11.json --set drives=1 ramp_delay=600
b --out $O/9 --from $O/8/ck_stand.pkl --tracks $R/t6_r12b.json --set drives=1 ramp_delay=600
sha256sum $O/9/ref.npz
