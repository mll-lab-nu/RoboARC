"""Per-frame checks of a kinematic plan of the caliper on its stand (oracle4.py's pickle): the instance's MuJoCo model
with both arms at the plan's joints, the slider at the plan's opening and every sleeve where the plan has it (every
`stride`-th frame), and per stage the smallest distances (mm, negative: overlap) between
  arms      any left-arm and any right-arm collision geom
  L-other   the left arm and the caliper (but the slider while the left hand holds or takes it, the knob while it pinches
            it), the stand, the table, the sleeves
  R-other   the right arm and the caliper, the stand, the table, the sleeves it does not hold or take, the bins
  sleeve    the sleeve the right hand holds and the stand, the table, the other sleeves, the bins, the rack; the caliper
            too, except in the stages where it is meant to touch it (4, 5, 17 / 9, 10, 19)
plus each joint's smallest distance to its limits. usage: check4.py <instance scene dir> <plan pkl> [stride]"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aloha_kin as ak                  # noqa: E402
import oracle4                          # noqa: E402

sd, pkl = Path(sys.argv[1]), sys.argv[2]
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 2
root = os.environ['TOPOGYM_ASSET_ROOT'].rstrip('/') + '/'
spec = mujoco.MjSpec.from_string((sd / 'scene.xml').read_text().replace('@robot-assets@/', root))
spec.meshdir = str(sd)
m = spec.compile()
d = mujoco.MjData(m)
mujoco.mj_kinematics(m, d)


def jadr(n):
    return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])


def bname(b):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or ''


OFF = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot')].copy()
J = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
QA = {h: [jadr(f'{s}/{j}') for j in J] for h, s in (('L', 'left'), ('R', 'right'))}
FA = {h: [jadr(f'{s}/{j}') for j in ('left_finger', 'right_finger')] for h, s in (('L', 'left'), ('R', 'right'))}
J_SL = jadr('caliper::slider')
plan = pickle.load(open(pkl, 'rb'))
fr = plan['frames']
N = len(plan['scenario']['sleeves'])
J_SLEEVE = [jadr(f'sleeve_{k}::root') for k in range(N)]
coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]


def G(pred):
    return [g for g in coll if pred(bname(m.geom_bodyid[g]))]


ARM = {h: G(lambda b, s=s: b.startswith(s + '/') and 'base_link' not in b and 'shoulder' not in b) for h, s in (('L', 'left'), ('R', 'right'))}
ARM_ALL = {h: G(lambda b, s=s: b.startswith(s + '/')) for h, s in (('L', 'left'), ('R', 'right'))}
CAL = {p: G(lambda b, p=p: b == f'caliper::{p}') for p in ('beam', 'slider', 'screw')}
STAND = G(lambda b: b.startswith('caliper_stand'))
TABLE = G(lambda b: b == 'table')
RACK = G(lambda b: b.startswith('sleeve_rack'))
BINS = G(lambda b: b.startswith('bin_'))
SLV = [G(lambda b, k=k: b == f'sleeve_{k}::sleeve') for k in range(N)]
TC_model = np.eye(4)
cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'caliper::beam')
TC_model[:3, :3], TC_model[:3, 3] = d.xmat[cb].reshape(3, 3), d.xpos[cb] - OFF
err = np.abs(TC_model - fr[0]['TC']).max()
assert err < 1e-6, f'the plan\'s caliper differs from the instance\'s by {err}'
R_AX = oracle4.R_AXIS


def gap(A, B, distmax=0.05):
    """(smallest distance, the closest pair's bodies)"""
    best, pair = distmax, None
    if not A or not B:
        return best, pair
    pA, pB = d.geom_xpos[A], d.geom_xpos[B]
    g = np.linalg.norm(pA[:, None] - pB[None], axis=2) - m.geom_rbound[A][:, None] - m.geom_rbound[B][None]
    ft = np.zeros(6)
    for i, j in zip(*np.nonzero(g < distmax)):
        x = mujoco.mj_geomDistance(m, d, A[i], B[j], distmax, ft)
        if x < best:
            best, pair = x, (bname(m.geom_bodyid[A[i]]), bname(m.geom_bodyid[B[j]]))
    return best, pair


RK = ak.Robot()
worst = {}
for n, f in enumerate(fr):
    if n % stride:
        continue
    for h in 'LR':
        d.qpos[QA[h]] = f['q' + h]
        d.qpos[FA[h]] = f['f' + h]
    d.qpos[J_SL] = f['s'] * 1e-3
    for k in range(N):
        T = f['TS'][k]
        d.qpos[J_SLEEVE[k]:J_SLEEVE[k] + 3] = OFF + T[:3, 3]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (T[:3, :3] @ R_AX).flatten())
        d.qpos[J_SLEEVE[k] + 3:J_SLEEVE[k] + 7] = q
    mujoco.mj_kinematics(m, d)
    num = f['stage'].split()[0]
    held_R = f['heldR'] if f['heldR'] is not None else f['tgtR']
    L_on = f['heldL'] or f['tgtL']
    cal_L = CAL['beam'] + ([] if L_on == 'slider' else CAL['slider']) + ([] if L_on == 'knob' else CAL['screw'])
    other_sl = [g for k in range(N) if k != held_R for g in SLV[k]]
    row = dict(arms=gap(ARM_ALL['L'], ARM_ALL['R']),
               L_other=gap(ARM['L'], cal_L + STAND + TABLE + sum(SLV, [])),
               R_other=gap(ARM['R'], CAL['beam'] + CAL['slider'] + CAL['screw'] + STAND + TABLE + BINS + RACK + other_sl))
    if isinstance(held_R, int) and f['heldR'] is not None:
        touch_ok = num in ('4', '5', '17', '9', '10', '19')
        row['sleeve'] = gap(SLV[held_R], STAND + TABLE + other_sl + BINS + RACK + ([] if touch_ok else sum(CAL.values(), [])))
    row['margin'] = (min(RK.margin(h, f['q' + h]) for h in 'LR'), None)
    if os.environ.get('CHECK_STAGE') == num and f['t'] < float(os.environ.get('CHECK_UNTIL', '1e9')):
        print(f"   t {f['t']:7.2f} L-other {row['L_other'][0] * 1e3:7.1f} {row['L_other'][1]} fL {f['fL']:.4f}", flush=True)
    for kk, (v, pair) in row.items():
        w = worst.setdefault(num, {})
        if kk not in w or v < w[kk][0]:
            w[kk] = (v, f['t'], pair)
order = sorted(worst, key=lambda x: int(x))
print('stage   arms mm (t)       L-other mm        R-other mm        sleeve mm         joint margin rad')
for num in order:
    w = worst[num]
    cell = lambda k, sc=1e3: ('%6.1f (%5.1f)' % (w[k][0] * sc, w[k][1])) if k in w else '      -       '    # noqa: E731
    print(f'{num:>5}  {cell("arms")}  {cell("L_other")}  {cell("R_other")}  {cell("sleeve")}  {cell("margin", 1.0)}')
allv = {k: min(worst[n][k][0] for n in order if k in worst[n]) for k in ('arms', 'L_other', 'R_other', 'sleeve', 'margin')}
print('overall:', {k: round(v * (1 if k == 'margin' else 1e3), 2) for k, v in allv.items()})
print('closest pairs under 3 mm:')
for num in order:
    for k, (v, t, pair) in worst[num].items():
        if k != 'margin' and v < 0.003:
            print(f'   stage {num:>3} {k:8s} {v * 1e3:6.1f} mm at t {t:6.1f} s: {pair}')
