#!/usr/bin/env python3
"""Fragility probe of an open-loop reference: write a copy of the trajectory with one control entry changed by --eps
(row --row, actuator --col), everything else equal. Replayed on the UNCHANGED scene (no prism rule), it shows whether
the reference survives a perturbation far smaller than any physics change.

usage: python perturb_traj.py IN.npz OUT.npz --row 1000 --col 0 --eps 1e-6
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--row", type=int, default=1000); ap.add_argument("--col", type=int, default=0)
ap.add_argument("--eps", type=float, default=1e-6)
a = ap.parse_args()
z = dict(np.load(a.inp))
c = z["ctrl"].astype(np.float64).copy()
c[a.row, a.col] += a.eps
z["ctrl"] = c
np.savez(a.out, **z)
