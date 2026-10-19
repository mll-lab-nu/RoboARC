"""Official 8-step Cast Coil assembly (the v2 very_easy tier): piece poses and hand poses from the solution record.

The layout is the one verified kinematically on 2026-10-09
(runs/test_results/cast_coil_revision_20261009/official_easy_ik_no_handle):
- the assembled 3 x 4 x 4 cell block has its centre at ASSEMBLY_CENTRE and both piece frames in the resting
  orientation of the task (window axis along world +x towards the arms, upside down);
- every route step is shared by the hands: gold takes GOLD_SHARE of each motion and silver the rest, so that silver's
  pose relative to gold follows the route exactly (gold: half of every turn about the window axis, a quarter of every
  turn about the in-plane axis, half of every slide);
- the assembly is the route (a disassembly of silver relative to gold, in gold's frame) played backwards;
- each hand pinches its piece's grasp cell from outside along the window axis: TCP x (the fingers) along the
  approach axis, closing across piece z, the finger pads at the grasp pad point.
Poses are (position, scipy Rotation) pairs in metres; a piece pose maps piece-frame points to the world.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from topogym.assets.geometry.cast_coil_official import BLOCK_CENTRE_CELLS, block_centre_cells  # noqa: F401

ASSEMBLY_CENTRE = np.array([-.1356786, .20, .9196429+.26])
REST_ORIENTATION = Rotation.from_matrix(np.column_stack(([0., 1., 0.], [1., 0., 0.], [0., 0., -1.])))
GOLD_SHARE = dict(y=.5, x=.25, z=.5, slide=.5)
PAD_AHEAD_OF_TCP_M = .01381
AXES = dict(x=np.array([1., 0, 0]), y=np.array([0., 1, 0]), z=np.array([0., 0, 1]))


def compose(a, b):
    # Observation arrays are read-only and scipy's Rotation.apply needs a writable buffer: copy.
    return np.array(a[0], float) + a[1].apply(np.array(b[0], float)), a[1]*b[1]


def inverse(a):
    r = a[1].inv()
    return -r.apply(np.array(a[0], float)), r


def interpolate(a, b, u):
    turn = Slerp([0, 1], Rotation.concatenate([a[1], b[1]]))(u)
    return np.array(a[0], float) + u*(np.array(b[0], float)-a[0]), turn


def step_pose(step, fraction, cell_m):
    """Rigid motion (in gold's frame, metres) of `fraction` of one route step."""
    if step["kind"] == "slide":
        return np.asarray(step["vector"], float)*cell_m*fraction, Rotation.identity()
    turn = Rotation.from_rotvec(AXES[step["axis"]]*np.deg2rad(step["degrees"])*fraction)
    pivot = np.asarray(step["through"], float)*cell_m
    return pivot - turn.apply(pivot), turn


def gold_step_pose(step, fraction, cell_m):
    """Gold's share of one route step, opposite to silver's motion, in gold's own frame."""
    share = GOLD_SHARE["slide" if step["kind"] == "slide" else step["axis"]]
    return step_pose(step, -share*fraction, cell_m)


def disassembly(record, *, degrees_per_sample=2.25, metres_per_sample=.002):
    """[(step index, gold pose in its assembled frame, silver pose relative to gold)], from assembled to apart."""
    cell = float(record["cell_m"])
    gold, relative = (np.zeros(3), Rotation.identity()), (np.zeros(3), Rotation.identity())
    samples = [(-1, gold, relative)]
    for index, step in enumerate(record["route"]):
        if step["kind"] == "slide":
            count = max(1, int(np.ceil(np.linalg.norm(step["vector"])*cell/metres_per_sample)))
        else:
            count = max(1, int(np.ceil(abs(step["degrees"])/degrees_per_sample)))
        g0, r0 = gold, relative
        for k in range(1, count+1):
            u = k/count
            u = u**3*(10+u*(-15+6*u))
            gold = compose(g0, gold_step_pose(step, u, cell))
            relative = compose(step_pose(step, u, cell), r0)
            samples.append((index, gold, relative))
    return samples


def assembled_gold_pose(record, centre=ASSEMBLY_CENTRE, orientation=REST_ORIENTATION):
    """Gold's world pose when the assembled block is centred at `centre` (silver's is the same)."""
    return np.asarray(centre, float) - orientation.apply(block_centre_cells(record)*float(record["cell_m"])), orientation


def assembly(record, **sampling):
    """[(assembly step 1..8, gold world pose, silver world pose)], from the start (apart) to assembled.

    A record with anchor "hands" (the lattice designs) shifts both pieces together at every sample so that the
    midpoint of the two grasp TCPs stays where it is when assembled: the hands share the reach of a turn about a
    line far from the block, and silver's pose relative to gold is unchanged."""
    world = assembled_gold_pose(record)
    out = []
    for index, gold, relative in reversed(disassembly(record, **sampling)):
        step = len(record["route"]) - index if index >= 0 else len(record["route"])
        gold_world = compose(world, gold)
        out.append((step, gold_world, compose(gold_world, relative)))
    if record.get("anchor") == "hands":
        tcp = {piece: grasp_tcp_in_piece(record, piece) for piece in ("gold", "silver")}

        def middle(gold, silver):
            return (compose(gold, tcp["gold"])[0] + compose(silver, tcp["silver"])[0])/2
        anchor = middle(out[-1][1], out[-1][2])
        out = [(step, (g[0] + anchor - middle(g, s), g[1]), (s[0] + anchor - middle(g, s), s[1])) for step, g, s in out]
    return out


def grasp_tcp_in_piece(record, piece):
    """The hand's TCP pose in its piece's frame: fingers along the approach axis, closing across piece z."""
    meta = record["pieces"][piece]
    approach = np.asarray(meta["approach_axis"], float)
    closing = np.array([0., 0., 1.]) if approach[1] > 0 else np.array([0., 0., -1.])
    turn = Rotation.from_matrix(np.column_stack((approach, closing, np.cross(approach, closing))))
    pad = np.asarray(meta["grasp_pad_cells"], float)*float(record["cell_m"])
    return pad - PAD_AHEAD_OF_TCP_M*approach, turn


def carry(rest, start, assembly_turns, centre_in_piece, samples):
    """Poses from `rest` (lifted) to `start`: first to the start's y and height while turning through
    `assembly_turns` backwards (the orientations the piece passes during the assembly, from its start to assembled;
    the carry goes from assembled-like to the start, so the wrists wind up exactly what the assembly unwinds), then
    along x to the start.

    Returns two lists of poses; the piece point `centre_in_piece` moves on straight segments."""
    c0, c2 = rest[0] + rest[1].apply(centre_in_piece), start[0] + start[1].apply(centre_in_piece)
    c1 = np.array([c0[0], c2[1], c2[2]])

    def turned(v):
        x = (1-v)*(len(assembly_turns)-1)
        i = min(int(x), len(assembly_turns)-2)
        return Slerp([0, 1], Rotation.concatenate([assembly_turns[i], assembly_turns[i+1]]))(x-i)

    first, second = [], []
    for k in range(1, samples+1):
        v = k/samples
        v = v**3*(10+v*(-15+6*v))
        turn = turned(v)
        first.append((c0 + v*(c1-c0) - turn.apply(centre_in_piece), turn))
    final = turned(1.)
    for k in range(1, samples//2+1):
        v = k/(samples//2)
        v = v**3*(10+v*(-15+6*v))
        second.append((c1 + v*(c2-c1) - final.apply(centre_in_piece), final))
    return first, second
