"""Remove (or scale) one frequency band from chosen command columns of a reference: per column, the straight line from
its first to its last value is taken out, the band's FFT bins are multiplied by the scale (0: removed), the line is
put back (zero phase; every other frequency unchanged; the end values move by the band's circular leakage, a few mrad
at most on the claws). Writes actions + action_layout.

usage: notch_ref.py <in.npz> <out.npz> --cols left/wrist_rotate,right/wrist_rotate [--band 1.35,1.9] [--scale 0]
       [--fs 40] [--length N]
--length: first pad the rows with copies of the last one (or cut them) to N rows; the band is removed over that length.
The claws -i01 references: --length 792 (easy), 1416 (medium), 16202 (hard), claws v5's lengths (the released oracle
padded with its last row), on the released oracle tools/references/oracles_genesis/interlocked-claws-<tier>-i00/privileged.npz."""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--cols", required=True)
ap.add_argument("--band", default="1.35,1.9"); ap.add_argument("--scale", type=float, default=0.); ap.add_argument("--fs", type=float, default=40.)
ap.add_argument("--length", type=int, default=0)
a = ap.parse_args()
z = np.load(a.src)
acts = z["actions"].astype(np.float64)
if a.length:
    acts = np.concatenate([acts, np.repeat(acts[-1:], max(0, a.length - len(acts)), axis=0)])[:a.length]
layout = [str(x) for x in z["action_layout"]]
lo, hi = (float(v) for v in a.band.split(","))
n = len(acts)
f = np.fft.rfftfreq(n, 1 / a.fs)
band = (f >= lo) & (f <= hi)
report = {}
for name in a.cols.split(","):
    j = layout.index(name)
    x = acts[:, j]
    line = np.linspace(x[0], x[-1], n)
    X = np.fft.rfft(x - line)
    removed = X.copy(); removed[~band] = 0
    X[band] *= a.scale
    y = np.fft.irfft(X, n=n) + line
    report[name] = dict(band_rms_before=round(float(np.sqrt(np.mean(np.fft.irfft(removed, n=n) ** 2))), 5),
                        max_change=round(float(np.abs(y - x).max()), 5), end_change=[round(float(y[0] - x[0]), 6), round(float(y[-1] - x[-1]), 6)])
    acts[:, j] = y
np.savez(a.out, actions=acts.astype(np.float32), action_layout=np.array(layout))
print(json.dumps(dict(src=a.src, out=a.out, band=[lo, hi], scale=a.scale, cols=report)))
