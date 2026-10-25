"""Orbits with the cord as a rigid capsule chain: the layout, the MJCF of the fixture, the free handle and the closed
chain, and the chain's joint state (the rigid-chain settings orbits_{easy,medium,hard}_chain; the development Session
of topogym.planning.orbits.chain builds the same MJCF).

Local task frame: x along the fixture, the ball post at -x, the handle at +x, z up from the table top; the entity is
posed like the task (yaw_deg about z, base-centre anchor). Fixture: the base box, the posts as capsules, the stopper
ball and each ring as RING_SEGMENTS capsules (primitives: Genesis 1.2.1's primitive cylinder lets a capsule drift on
its face). The handle (a convex elliptic prism) hangs on eye 0 by a ball joint; the cord is two branches of
ball-jointed capsules from the middle link (the root, a free joint), the second branch tied to the handle's eye 1 by a
connect equality: cord and handle form one closed loop. Each bending joint has a spring k = E pi r^4 / 4 / L towards
the authored pose, so the chain is authored straight at REST_AT, high above the fixture, where no part touches another
(Genesis drops the collisions of an entity's parts that touch in its authored pose); a state is written into the
joints (chain_qpos).

A state is dict(points=(N+1, 3) local cord points from eye 0 to eye 1, handle_pos=(3,), handle_quat=(4,) wxyz).
"""
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation as R

from .orbits import OrbitsLayout

RING_SEGMENTS = 32
CORD_KG_M3 = 650.          # the FEM cord's density
HANDLE_KG_M3 = 700.
FRICTION = '.3 .005 .0001'  # the FEM cord's 0.3; Genesis takes the larger of two geoms' values (fingers: 1.0)
SOLREF = '.005 1'           # the stiffest MuJoCo/Genesis accept at dt 2.5 ms (time constant >= 2 dt)
# The loop's tie at eye 1 (the connect). A soft equality held under a tie force f stretches
# r = f invweight (1 - d) / d (dmax tc dampratio)^2 / d (MuJoCo's impedance d, time constant tc; invweight the two
# links', here the last link's 1063 + the handle's 26 = 1089 /kg). With SOLREF and MuJoCo's default solimp
# (.9 .95) that is 1.36 mm per N, measured in Genesis 1.2.1 (scripts/oracle_runners/orbits_chain/tie_stiffness.py:
# the handle pulled 0 -> 3 N, 1.43 mm at 1.06 N through the tie; past the task's 2 mm at ~1.5 N). dampratio .5 and
# d .98 flat: 0.136 mm per N (0.15 mm at 1.07 N, ~0.41 mm at 3 N through the tie), no NaN, no slowdown; tc is already
# the 2 dt floor; d .99-.999 gave NaN when a slack cord went taut, more CG iterations (200) did not help. Genesis reads
# both from the MJCF (the built solver row checked). Its RigidEquality.idx of this connect is not the solver row (it
# reports 0, the robot's first finger equality): find the row by type and links in equalities_info.
TIE_SOLREF = '.005 .5'
TIE_SOLIMP = '.98 .98 .001 .5 2'
REST_AT = (0., 0., 1.)      # local point high above the fixture where the straight rest chain is authored (never stepped)
# assisted grasp (contact_weld_v1): every link is a target, a sphere round its middle small enough that a hand
# touching one link's middle does not touch the next one's (a hand on two targets gets no weld), and the handle
GRASP_LINK_R = .008
GRASP_HANDLE_R = .045


@dataclass(frozen=True)
class ChainLayout(OrbitsLayout):
    """OrbitsLayout with ring 1 larger by r1_dr and its centre r1_dr further -x (its +x rim stays on P1), and ring 2
    and every post / ring after it (P2 .. P_n, rings 2 .. n) r2_dx further +x (ring_x follows post_x)."""
    r1_dr: float = 0.
    r2_dx: float = 0.

    @property
    def post_x(self):
        x = super().post_x.copy()
        x[2:] += self.r2_dx
        return x

    @property
    def ring_x(self):
        x = super().ring_x.copy()
        if len(x):
            x[0] -= self.r1_dr
        return x


def ring_radius(lay, k):
    """Ring k's radius (k from 0: ring 1 larger by r1_dr)."""
    return lay.ring_radius_m+(getattr(lay, 'r1_dr', 0.) if k == 0 else 0.)


def chain_layout(geometry, linear_scale, *, fixture_scale, ball_scale, cord_radius_m, base_drop_m, ring1_grow_m,
                 handle_x_m, ring2_dx_m=0., ring_drop_m=0., ring_stack_cut_m=0.):
    """The task's geometry scaled by linear_scale, then the fixture (every length but the cord's radius) by
    fixture_scale; the ball's radius by ball_scale, the cord's radius cord_radius_m, the base base_drop_m thinner (the
    rings, posts' tops and ball where they were), ring 1 larger by ring1_grow_m (ChainLayout.r1_dr), ring 2 and
    everything after it ring2_dx_m further +x (ChainLayout.r2_dx), the handle's centre at handle_x_m; every ring (the
    posts' tops and the ball with them) ring_drop_m lower and the rings' stack ring_stack_cut_m shorter (the lowest
    ring stays, the step between rings ring_stack_cut_m / (n - 1) less)."""
    lay = OrbitsLayout(**geometry).scaled(linear_scale)
    lay = replace(lay.scaled(fixture_scale), cord_radius_m=lay.cord_radius_m)
    lay = replace(lay, cord_radius_m=cord_radius_m, ball_radius_m=lay.ball_radius_m*ball_scale,
                  base_height_m=lay.base_height_m-base_drop_m, handle_x_m=handle_x_m,
                  lowest_ring_z_m=lay.lowest_ring_z_m-ring_drop_m,
                  ring_step_m=lay.ring_step_m-ring_stack_cut_m/(lay.ring_count-1))
    return ChainLayout(**asdict(lay), r1_dr=ring1_grow_m, r2_dx=ring2_dx_m)


def links_for(length, link_m):
    return max(8, int(round(length/link_m)))


def _v(x):
    return ' '.join(f'{a:.9g}' for a in np.asarray(x, float).ravel())


def _q(rot):
    return _v(rot.as_quat()[[3, 0, 1, 2]])


def eye_offsets(lay):
    """The two cord ends in the handle frame (origin at the handle's centre): the eyes, at the FEM cord's end height."""
    s = lay.template_scale
    zc = lay.clearance_m+lay.handle_height_m/2
    return np.array([[e[0]-lay.handle_x_m, e[1], lay.clearance_m+.006*s-zc] for e in lay.eyes])


def handle_outline(lay, count=48):
    """The handle mesh's vertices in its own frame (an elliptic prism)."""
    th = np.arange(count)*2*np.pi/count
    ring = np.column_stack([lay.handle_length_m/2*np.cos(th), lay.handle_width_m/2*np.sin(th)])
    h = lay.handle_height_m/2
    return np.array([[x, y, z] for z in (-h, h) for x, y in ring])


def link_stiffness(lay, link_m, E):
    return E*np.pi*lay.cord_radius_m**4/4/link_m


def grasp_targets(n, link_m):
    """[(link name in the 'orbits_chain' entity, centre in the link frame, radius)] for every link and the handle."""
    return [(f'link_{i:03d}', (link_m/2, 0., 0.), GRASP_LINK_R) for i in range(n)]+[('handle', (0., 0., 0.), GRASP_HANDLE_R)]


def scene_xml(lay, n, link_m, yaw_deg, E):
    """MJCF of the fixture, the handle and the closed chain in the entity frame (local frame turned by yaw_deg).

    The chain is authored straight along local x at REST_AT: both simulators pull a ball joint's spring towards its
    authored (qpos0) pose, so the rest shape is straight; a state's shape is written into the joints at run time
    (chain_qpos). The root is the middle link k; links k-1 .. 0 run back to eye 0, where the handle hangs by a ball
    joint without stiffness; links k+1 .. n-1 run on to eye 1, tied to the handle's eye-1 site by a site connect.
    """
    turn = R.from_euler('z', yaw_deg, degrees=True)
    root = ET.Element('mujoco', model='orbits_chain')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    d = ET.SubElement(root, 'default')
    ET.SubElement(d, 'geom', friction=FRICTION, solref=SOLREF, condim='3', contype='1', conaffinity='1')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'mesh', name='handle', vertex=_v(handle_outline(lay)))
    world = ET.SubElement(root, 'worldbody')
    wood = dict(rgba='.55 .35 .18 1')
    ET.SubElement(world, 'geom', name='base', type='box', pos=_v(turn.apply([0, 0, lay.base_height_m/2])),
                  quat=_q(turn), size=_v([lay.base_length_m/2, lay.base_width_m/2, lay.base_height_m/2]), **wood)
    tops = [lay.ball_z, *lay.ring_z]
    for i, (x, z) in enumerate(zip(lay.post_x, tops)):
        a, b = turn.apply([[x, 0, lay.base_height_m], [x, 0, z]])
        ET.SubElement(world, 'geom', name=f'post_{i}', type='capsule', fromto=_v([a, b]), size=_v(lay.post_radius_m), **wood)
    ET.SubElement(world, 'geom', name='stopper_ball', type='sphere', pos=_v(turn.apply([lay.post_x[0], 0, lay.ball_z])),
                  size=_v(lay.ball_radius_m), **wood)
    for k, (x, z) in enumerate(zip(lay.ring_x, lay.ring_z)):
        a = np.arange(RING_SEGMENTS+1)*2*np.pi/RING_SEGMENTS
        r = ring_radius(lay, k)
        c = turn.apply(np.column_stack([x+r*np.cos(a), r*np.sin(a), np.full(len(a), z)]))
        for j in range(RING_SEGMENTS):
            ET.SubElement(world, 'geom', name=f'ring_{k+1}_{j:02d}', type='capsule', fromto=_v([c[j], c[j+1]]),
                          size=_v(lay.ring_tube_radius_m), **wood)
    # sliding friction only (condim 3): Genesis 1.2.1 has no torsional or rolling friction, so MuJoCo must not hold a
    # pinched cord against turning better than Genesis does; chain_mj adds noslip iterations for the sliding hold
    cord = dict(type='capsule', size=_v(lay.cord_radius_m), density=_v(CORD_KG_M3), rgba='.95 .78 .1 1')
    joint = dict(type='ball', stiffness=_v(link_stiffness(lay, link_m, E)), damping='2e-4', armature='1e-6')
    k = n//2
    L = _v([link_m, 0, 0])

    def link(parent, i, pos, quat=None):
        body = ET.SubElement(parent, 'body', name=f'link_{i:03d}', pos=pos, **({'quat': quat} if quat else {}))
        if i == k:
            ET.SubElement(body, 'freejoint', name='root')
        else:
            ET.SubElement(body, 'joint', name=f'bend_{i:03d}', **joint)
        ET.SubElement(body, 'geom', name=f'cord_{i:03d}', fromto=_v([0, 0, 0, link_m, 0, 0]), **cord)
        return body

    start = np.asarray(REST_AT, float)+[(k-n/2)*link_m, 0, 0]
    mid = link(world, k, _v(turn.apply(start)), _q(turn))
    parent = mid
    for i in range(k+1, n):                       # on to eye 1
        parent = link(parent, i, L)
    ET.SubElement(parent, 'site', name='cord_end', pos=L, size='.001')
    parent = link(mid, k-1, '0 0 0', '0 0 0 1')    # back to eye 0: turned half round about z, then straight on
    for i in range(k-2, -1, -1):
        parent = link(parent, i, L)
    eyes = eye_offsets(lay)
    handle = ET.SubElement(parent, 'body', name='handle', pos=_v(np.array([link_m, 0, 0])-eyes[0]))
    ET.SubElement(handle, 'joint', name='handle_eye', type='ball', pos=_v(eyes[0]), damping='1e-5', armature='1e-6')
    ET.SubElement(handle, 'geom', name='handle', type='mesh', mesh='handle', density=_v(HANDLE_KG_M3), rgba='.62 .42 .22 1')
    ET.SubElement(handle, 'site', name='eye_1', pos=_v(eyes[1]), size='.001')
    eq = ET.SubElement(root, 'equality')
    ET.SubElement(eq, 'connect', name='eye_1', site1='cord_end', site2='eye_1', solref=TIE_SOLREF, solimp=TIE_SOLIMP)
    contact = ET.SubElement(root, 'contact')
    for i in (*range(3), *range(n-3, n)):       # the cord ends sit in the eyes
        ET.SubElement(contact, 'exclude', body1='handle', body2=f'link_{i:03d}')
    ET.indent(root)
    return ET.tostring(root, encoding='unicode')


def _minrot(a, b):
    return R.align_vectors([b], [a])[0]


def chain_qpos(lay, state, yaw_deg=0.):
    """{joint name: qpos} putting scene_xml's chain (entity frame, root position without the entity's origin) and
    handle into `state`: each link turned from its parent's frame by the least rotation onto its own direction."""
    turn = R.from_euler('z', yaw_deg, degrees=True)
    Q = turn.apply(np.asarray(state['points'], float))
    hq = state['handle_quat']
    hrot = turn*R.from_quat([hq[1], hq[2], hq[3], hq[0]])
    n = len(Q)-1
    k = n//2
    out, rots = {}, {}
    rots[k] = _minrot([1., 0., 0.], Q[k+1]-Q[k])
    out['root'] = np.r_[Q[k], rots[k].as_quat()[[3, 0, 1, 2]]]
    for i in range(k+1, n):
        rots[i] = _minrot(rots[i-1].apply([1., 0., 0.]), Q[i+1]-Q[i])*rots[i-1]
        out[f'bend_{i:03d}'] = (rots[i-1].inv()*rots[i]).as_quat()[[3, 0, 1, 2]]
    base = rots[k]*R.from_quat([0., 0., 1., 0.])
    for i in range(k-1, -1, -1):
        rots[i] = _minrot(base.apply([1., 0., 0.]), Q[i]-Q[i+1])*base
        out[f'bend_{i:03d}'] = (base.inv()*rots[i]).as_quat()[[3, 0, 1, 2]]
        base = rots[i]
    out['handle_eye'] = (rots[0].inv()*hrot).as_quat()[[3, 0, 1, 2]]
    return out


def chain_points(pos, mats, link_m, k):
    """Cord points (eye 0 to eye 1) from the links' origins and rotation matrices (n, 3, 3): branch 0 ends at its
    links' far ends, branch 1 starts at their origins."""
    ends = pos+np.einsum('nij,j->ni', mats, [link_m, 0., 0.])
    return np.vstack([ends[:k], pos[k:], ends[-1:]])
