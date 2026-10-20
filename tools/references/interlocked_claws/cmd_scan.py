"""Narrowband oscillation in reference commands (a dither, a wiggle). For every column of each reference's actions,
in 10 s windows (hop 5 s; the control rate 40 Hz unless the argument ends in @<Hz>): the linear trend removed, the
spectral peak above 0.5 Hz, the RMS of the window band-passed to that peak +-0.2 Hz, and the peak's share of the power
above 0.5 Hz. A window is flagged when that narrowband RMS exceeds the threshold (5 mrad for joints; 5 % of the column's
range for grippers), the peak holds at least 40 % of the power above 0.5 Hz and it is a true spectral peak (above the
three bins below it; the falling edge of a slow motion's spectrum just above 0.5 Hz is not). Per column with a
flagged window: how many, their span, the median peak frequency, the largest narrowband RMS and amplitude (RMS x sqrt 2),
and the class: "tone" when the peak holds at least 90 % of the power above 0.5 Hz in the median flagged window (a
sinusoid riding on the motion), else "motion" (the fast part of ordinary moves).

usage: cmd_scan.py <label=reference.npz> [...]   -> one JSON line per reference"""
import json, sys
import numpy as np

TH_RAD, TH_GRIP_FRAC, SHARE, WIN_S, HOP_S, LO = .005, .05, .4, 10., 5., .5


def scan(path, fs=40.):
    z = np.load(path, allow_pickle=True)
    a = z["actions"].astype(np.float64)
    layout = [str(x) for x in z["action_layout"]] if "action_layout" in z.files else [f"c{j}" for j in range(a.shape[1])]
    n, W, H = len(a), int(WIN_S * fs), int(HOP_S * fs)
    out = dict(rows=n, fs_hz=round(fs, 3), seconds=round(n / fs, 1), flagged={})
    if n < W:
        W = n
    f = np.fft.rfftfreq(W, 1 / fs)
    hann = np.hanning(W)
    for j, name in enumerate(layout):
        x = a[:, j]
        grip = "grip" in name
        th = TH_GRIP_FRAC * max(float(np.ptp(x)), 1e-9) if grip else TH_RAD
        hits = []
        for s0 in range(0, max(1, n - W + 1), H):
            seg = x[s0:s0 + W]
            if len(seg) < W:
                break
            t = np.arange(W)
            seg = seg - np.polyval(np.polyfit(t, seg, 1), t)
            P = np.abs(np.fft.rfft(seg * hann)) ** 2
            hi = f > LO
            if not hi.any() or P[hi].sum() <= 0:
                continue
            k = np.flatnonzero(hi)[np.argmax(P[hi])]
            if k < 3 or P[k] <= P[k - 3:k].max():
                continue
            band = (f >= f[k] - .2) & (f <= f[k] + .2)
            share = float(P[band & hi].sum() / P[hi].sum())
            X = np.fft.rfft(seg)
            X[~band] = 0
            r = float(np.sqrt(np.mean(np.fft.irfft(X, n=W) ** 2)))
            if r > th and share >= SHARE:
                hits.append((s0, float(f[k]), r, share))
        if hits:
            out["flagged"][name] = dict(windows=len(hits), span_s=[round(hits[0][0] / fs, 1), round((hits[-1][0] + W) / fs, 1)],
                                        freq_hz=round(float(np.median([h[1] for h in hits])), 2),
                                        rms_max=round(max(h[2] for h in hits), 4), amp_max=round(max(h[2] for h in hits) * 2 ** .5, 4),
                                        share_median=round(float(np.median([h[3] for h in hits])), 2),
                                        cls="tone" if np.median([h[3] for h in hits]) >= .9 else "motion",
                                        unit="gripper units" if grip else "rad")
    return out


for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    path, fs = (path.rsplit("@", 1)[0], float(path.rsplit("@", 1)[1])) if "@" in path else (path, 40.)
    try:
        print(json.dumps(dict(ref=label, path=path, **scan(path, fs))), flush=True)
    except Exception as exc:
        print(json.dumps(dict(ref=label, path=path, error=repr(exc))), flush=True)
