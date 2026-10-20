#!/usr/bin/env python3
"""The two opt-in contact_weld_v1 rules of topogym/world/mujoco/grasp.py (2026-10-14), on small MuJoCo scenes driven
through the real MujocoGraspBackend and GraspManager; the world and robot around them are stand-ins exposing only what
the backend reads (model, data, links, the grasp welds, the tick's contact readout from MujocoWorld._read_contacts).

ForceClosureGraspTarget (controller battery design, runs/controller_battery_design_20261014 sections 3.3, 8 D2): an AA
cell (radius 7.25 mm) in a slot with 1.0 mm side clearance and the rim 4 mm above the cell's top, pinched from above by
the two ALOHA fingertip collision hulls (aloha.xml's finger meshes as 64-vertex hulls, as the MuJoCo scenes collide
them). For each finger opening the gripper is lowered until a finger touches something, then each finger closes until
it touches the cell and presses 0.05 mm into it. Seated, the best such pinch has a central angle of 65.8 deg between
the two contacts (the design's 3-D check: 66.6-71.7 deg): with the check the cell is never welded, without it it is.
Lifted 6 mm (the pried cell), a 12 mm opening pinches at 108.5 deg and the cell is welded with the check on.

MultiRopeGraspAssistConfig (wire splice design, runs/wire_splice_design_20261014 section 8 item 5): two wire tips (ball
joint chains of 2 mm capsule links, radius 1.78 mm) between two finger pads. One behind the other (each pad touches one
wire, the wires press together) the plain rule finds no candidate; side by side (both pads touch both wires) it finds
several and calls them ambiguous. With weld_each_rope both become one grasp welding each wire's touched link nearest
the TCP to the same hand, and the two welds carry both tips when the hand moves.

  PYTHONPATH=. python tests/test_mujoco_grasp_opt_in.py
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import unittest

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from topogym.utils.grasp import (ForceClosureGraspTarget, GraspAssistConfig, GraspTarget, MultiRopeGraspAssistConfig,
                                 WeakHoldGraspTarget, force_closure_pair)
from topogym.world.mujoco.grasp import MujocoGraspBackend
from topogym.world.mujoco.world import MujocoWorld
from topogym.world.physics.grasp_manager import GraspManager

ALOHA = ROOT / "assets/robots/aloha/aloha.xml"
Q_CLOSED = .0079            # the ALOHA finger joint at which the two pads meet on the TCP plane
DT, SUBSTEPS = .0025, 4     # a World tick and its MuJoCo steps (tabletop_aloha)
WELD = 'solref="0.00125 1" solimp="0.9 0.95 0.001 0.5 2"'     # profiles._RIGID weld
DOWN = "0.70710678 0 0.70710678 0"      # hand frame: approach (x) down, closing (y) along world y, width (z) along x
CELL_R, CELL_HALF, SIDE_GAP, RIM = .00725, .02475, .001, .0185
WIRE_R, WIRE_L, WIRE_N = .00178, .002, 8


def finger_hulls():
    """The ALOHA finger collision hulls (64 vertices) in the TCP frame (site gripper) with the pads touching: 'a' on
    the -y side, 'b' on the +y side."""
    spec = mujoco.MjSpec.from_file(str(ALOHA))
    for mesh in spec.meshes:
        if "finger" in mesh.file:
            mesh.maxhullvert = 64
    m = spec.compile()
    d = mujoco.MjData(m)
    for side in ("left", "right"):
        d.qpos[m.jnt_qposadr[m.joint(f"left/{side}_finger").id]] = Q_CLOSED
    mujoco.mj_forward(m, d)
    site = m.site("left/gripper").id
    frame, origin = d.site_xmat[site].reshape(3, 3), d.site_xpos[site]
    out = {}
    for side in ("left", "right"):
        body = m.body(f"left/{side}_finger_link").id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != body or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or \
                    (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            mesh = m.geom_dataid[g]
            vertices = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
            graph = m.mesh_graph[m.mesh_graphadr[mesh]:]
            hull = vertices[graph[2 + int(graph[0]):2 + 2 * int(graph[0])]]
            local = (hull @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] - origin) @ frame
            out["a" if local[:, 1].mean() < 0 else "b"] = local
    return out


class Link:
    def __init__(self, world, body):
        self.world, self.idx, self.is_fixed = world, body, False

    def get_pos(self):
        return np.array(self.world.data.xpos[self.idx])

    def get_quat(self):
        return np.array(self.world.data.xquat[self.idx])


class Entity:
    def __init__(self, world, bodies, n_dofs):
        self._links = {name: Link(world, world.model.body(body).id) for name, body in bodies.items()}
        self.links, self.n_dofs, self.world = list(self._links.values()), n_dofs, world

    def get_link(self, name):
        return self._links[name]

    def get_qpos(self):
        return np.array(self.world.data.qpos)

    def get_joint(self, name):
        return SimpleNamespace(qs_idx_local=[int(self.world.model.joint(name).qposadr[0])])


class World:
    """What MujocoGraspBackend reads of a MujocoWorld."""

    def __init__(self, xml, entities, targets, ropes=()):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._cloths, self._ropes, self.physics_step = {}, {name: None for name in ropes}, 0
        self.entities = {name: Entity(self, bodies, 6) for name, bodies in entities.items()}
        self.manifest = dict(welds=[dict(name=weld_name(t), hand="right", target=[t.entity, t.link]) for t in targets])
        self.forward()

    def forward(self):
        mujoco.mj_forward(self.model, self.data)
        self._readout = MujocoWorld._read_contacts(self)

    def contacts(self):
        return self._readout["contacts"]


class Robot:
    def __init__(self, world):
        self.description = SimpleNamespace(robot_id="aloha")
        self.groups = dict(right=SimpleNamespace(
            hand_link="hand", tcp_position_in_hand_m=(0., 0., 0.), gripper_field="gripper",
            finger_joints=("finger_a", "finger_b"),
            grasp=SimpleNamespace(finger_links=("finger_a", "finger_b"), open_position=.02, closed_position=0.)))
        self.entity = Entity(world, dict(hand="hand", finger_a="finger_a", finger_b="finger_b"), 2)
        self.gripper = .01          # the gripper command (closed fraction 0.5); the configurations start closing

    def capture_state(self):
        return SimpleNamespace(targets=dict(right=dict(gripper=self.gripper)))


def weld_name(target):
    return f"grasp/right/{target.entity}/{target.link}"


def welds_xml(targets, body_of):
    return "".join(f'<weld name="{weld_name(t)}" body1="hand" body2="{body_of(t)}" active="false" {WELD}/>'
                   for t in targets)


def assisted(world, config):
    world.physics_step = 0          # a new episode clock for a new manager
    backend = MujocoGraspBackend(world, Robot(world), config)
    return backend, GraspManager(backend, config)


def tick(world, manager, step=None):
    """One World tick: the manager's tick around either MuJoCo steps (with the backend's per-step sampling) or, for a
    scene held still, one forward pass."""
    world.physics_step += 1
    manager.before_tick(world.physics_step)
    if step is None:
        world.forward()
    else:
        for _ in range(SUBSTEPS):
            step()
            mujoco.mj_step(world.model, world.data)
            manager.backend.sample()
        world._readout = MujocoWorld._read_contacts(world)
        mujoco.mj_kinematics(world.model, world.data)
    manager.after_tick(DT)


# --------------------------------------------------------------------------------------------- the battery
def battery_world(lift):
    hulls = finger_hulls()
    meshes = "".join(f'<mesh name="m_{k}" vertex="{" ".join(f"{x:.9g}" for x in v.ravel())}"/>' for k, v in hulls.items())
    wall_y, half = CELL_R + SIDE_GAP + .0015, RIM / 2
    block = f'<geom name="block" type="box" size=".02 .003 {lift / 2}" pos="0 0 {lift / 2}"/>' if lift else ""
    target = GraspTarget("cell", "cell", (0., 0., 0.), .035)
    xml = f"""<mujoco><option timestep="{DT / SUBSTEPS}"/><asset>{meshes}</asset><worldbody>
 <geom name="floor" type="box" size=".05 .03 .002" pos="0 0 -.002"/>
 <geom name="wall_p" type="box" size=".03 .0015 {half}" pos="0 {wall_y} {half}"/>
 <geom name="wall_m" type="box" size=".03 .0015 {half}" pos="0 {-wall_y} {half}"/>{block}
 <body name="hand" mocap="true" pos="0 0 .1" quat="{DOWN}">
  <body name="finger_a"><joint name="finger_a" type="slide" axis="0 -1 0"/><geom name="finger_a" type="mesh" mesh="m_a"/></body>
  <body name="finger_b"><joint name="finger_b" type="slide" axis="0 1 0"/><geom name="finger_b" type="mesh" mesh="m_b"/></body>
 </body>
 <body name="cell" pos="0 0 {CELL_R + lift}"><freejoint/>
  <geom name="cell" type="cylinder" size="{CELL_R} {CELL_HALF}" quat="{DOWN}" mass=".023"/></body>
</worldbody><equality>{welds_xml([target], lambda t: "cell")}</equality></mujoco>"""
    return World(xml, dict(cell=dict(cell="cell")), [target])


def pinch(world, opening):
    """Lower the gripper, fingers `opening` apart, until a finger touches something; close each finger until it touches
    the cell and press it 0.05 mm in. Returns the central angle (deg) between the two finger contacts on the cell, or
    None when a finger cannot reach the cell or rests on something else."""
    m, d = world.model, world.data
    fa, fb, cell = (m.geom(n).id for n in ("finger_a", "finger_b", "cell"))
    env = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0]
    qa, qb = m.joint("finger_a").qposadr[0], m.joint("finger_b").qposadr[0]
    scratch = np.zeros(6)

    def gap(finger, geoms):
        return min(mujoco.mj_geomDistance(m, d, finger, g, .01, scratch) for g in geoms)

    def place(z, sa, sb):
        d.mocap_pos[0] = (0., 0., z)
        d.qpos[qa], d.qpos[qb] = sa, sb
        mujoco.mj_forward(m, d)

    s = opening / 2
    low, high = 0., .08
    for _ in range(40):
        z = (low + high) / 2
        place(z, s, s)
        clear = min(gap(fa, env + [cell]), gap(fb, env + [cell])) > 0
        low, high = (low, z) if clear else (z, high)
    z, closed = high, {}
    for finger, name in ((fa, "a"), (fb, "b")):
        def at(x):
            place(z, x if name == "a" else s, x if name == "b" else s)
        at(-.002)
        if gap(finger, [cell]) > 0:
            return None
        low, high = -.002, s
        for _ in range(40):
            x = (low + high) / 2
            at(x)
            low, high = (low, x) if gap(finger, [cell]) > 0 else (x, high)
        at(high)
        if gap(finger, env) <= 0:
            return None
        closed[name] = high - 5e-5
    place(z, closed["a"], closed["b"])
    if min(gap(fa, env), gap(fb, env)) <= 0:
        return None
    world.forward()
    c = world.contacts()
    centre = d.xpos[m.body("cell").id]
    angles = []
    for finger in (m.geom_bodyid[fa], m.geom_bodyid[fb]):
        rows = np.flatnonzero(((c["link_a"] == finger) & (c["link_b"] == m.body("cell").id)) |
                              ((c["link_b"] == finger) & (c["link_a"] == m.body("cell").id)))
        if not len(rows):
            return None
        p = c["position"][rows[0]] - centre
        angles.append(np.degrees(np.arctan2(p[2], abs(p[1]))))
    return 180. - sum(angles)


def cell_config(force_closure):
    target = (ForceClosureGraspTarget("cell", "cell", (0., 0., 0.), .035, friction_coefficient=1.) if force_closure
              else GraspTarget("cell", "cell", (0., 0., 0.), .035))
    return GraspAssistConfig("contact_weld_v1", (target,), initial_intent="close")


class ForceClosure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seated = battery_world(0.)
        cls.lifted = battery_world(.006)

    def acquire(self, world, config, ticks=12):
        backend, manager = assisted(world, config)
        for _ in range(ticks):
            tick(world, manager)
        state = manager.hands["right"]
        welded = bool(world.data.eq_active[world.model.equality(weld_name(config.targets[0])).id])
        if state["binding"] is not None:
            manager.backend.release(state["binding"])       # leave the scene's welds inactive for the next case
        return state, welded

    def test_pair_rule(self):
        mu = 1.
        for phi in (10., 44., 46., 80.):        # a cylinder pinched at phi above its mid-plane on both sides
            p, q = np.radians(phi), np.radians(phi)
            a, b = (-np.cos(p), 0., np.sin(p)), (np.cos(q), 0., np.sin(q))
            na, nb = (np.cos(p), 0., -np.sin(p)), (-np.cos(q), 0., -np.sin(q))
            self.assertEqual(force_closure_pair(a, na, b, nb, mu), phi <= 45., phi)
        # the same pinch with the two points 10 mm apart along the cylinder's axis (y): one line contact per finger
        p = np.radians(40.)
        a, b = (-np.cos(p), -.005, np.sin(p)), (np.cos(p), .005, np.sin(p))
        self.assertTrue(force_closure_pair(a, (np.cos(p), 0., -np.sin(p)), b, (-np.cos(p), 0., -np.sin(p)), mu))
        self.assertFalse(force_closure_pair((0, 0, 0), (1, 0, 0), (0, 0, 0), (-1, 0, 0), mu))
        self.assertTrue(force_closure_pair((0, 0, 0), (1, 0, 0), (.01, .02, 0), (-1, 0, 0), mu))    # opposed faces
        self.assertFalse(force_closure_pair((0, 0, 0), (0, 0, -1), (.01, 0, 0), (0, 0, -1), mu))    # both on top
        self.assertTrue(force_closure_pair((0, 0, 0), (0, 1, 0), (0, 1, 0), (0, -.6, .8), 1.5))    # 36.9 deg < 56.3

    def test_seated_cell_is_never_welded_with_the_check(self):
        angles = {}
        for opening in np.arange(.002, .0096, .0005):
            central = pinch(self.seated, opening)
            if central is None:
                continue
            angles[round(opening * 1000, 1)] = central
            state, welded = self.acquire(self.seated, cell_config(True))
            self.assertFalse(welded, opening)
            self.assertEqual((state["binding"], state["rejection"]), (None, "no_bilateral_candidate"))
            state, welded = self.acquire(self.seated, cell_config(False))        # the plain rule, as before
            self.assertTrue(welded, opening)
            self.assertEqual(state["phase"], "holding")
        self.assertGreaterEqual(len(angles), 10)
        self.assertLess(max(angles.values()), 72.)          # 65.8 deg at an 8 mm opening

    def test_lifted_cell_is_welded_with_the_check(self):
        central = pinch(self.lifted, .012)
        self.assertGreater(central, 100.)                   # 108.5 deg
        state, welded = self.acquire(self.lifted, cell_config(True))
        self.assertTrue(welded)
        self.assertEqual(state["phase"], "holding")
        self.assertEqual(tuple(state["binding"]["target"]), ("cell", "cell"))

    def test_decision_follows_the_central_angle(self):
        """mu 1: a pinch closes force exactly when the central angle reaches 90 deg (2 deg kept clear of the edge)."""
        cases = 0
        for world, openings in ((self.seated, (.005, .008)), (self.lifted, np.arange(.008, .0131, .0005))):
            for opening in openings:
                central = pinch(world, opening)
                if central is None or abs(central - 90.) < 2.:
                    continue
                backend = MujocoGraspBackend(world, Robot(world), cell_config(True))
                self.assertEqual(backend.candidates("right") == [("cell", "cell")], central >= 90., (opening, central))
                cases += 1
        self.assertGreaterEqual(cases, 8)


def weak_config(firm=.268, travel=.06, mu=1., weak=True):
    target = (WeakHoldGraspTarget("cell", "cell", (0., 0., 0.), .035, friction_coefficient=mu, firm_friction_coefficient=firm,
                                  weak_hold_travel_m=travel) if weak else
              ForceClosureGraspTarget("cell", "cell", (0., 0., 0.), .035, friction_coefficient=mu))
    return GraspAssistConfig("contact_weld_v1", (target,), initial_intent="close")


def carry(world, dz):
    """The hand (a mocap body) and the cell moved up together by dz, as a held cell is carried."""
    m, d = world.model, world.data
    d.mocap_pos[0, 2] += dz
    d.qpos[m.jnt_qposadr[m.body("cell").jntadr[0]] + 2] += dz
    world.forward()


class WeakHold(unittest.TestCase):
    """WeakHoldGraspTarget (controller battery, GRASP_PROPOSAL.md rule C, 2026-10-18): a pried cell's fingertip pinch
    (central angle about 108 deg: force closure at mu 1, not at 0.268) welds, but its hold is lost once the cell has
    moved 60 mm; a pinch across the cell's middle (central angle near 180 deg) is firm and has no such limit."""

    def hold(self, world, config, ticks=12):
        backend, manager = assisted(world, config)
        for _ in range(ticks):
            tick(world, manager)
        return manager, manager.hands["right"]

    def test_weak_pinch_is_lost_beyond_the_travel(self):
        world = battery_world(.006)
        central = pinch(world, .012)
        self.assertTrue(100. < central < 150., central)          # 108.5 deg
        manager, state = self.hold(world, weak_config())
        self.assertEqual(state["phase"], "holding")
        b = state["binding"]
        self.assertFalse(b["firm_hold"])
        np.testing.assert_allclose(b["weak_hold_origin_m"], world.data.xpos[world.model.body("cell").id])
        carry(world, .05)
        tick(world, manager)
        self.assertEqual(state["phase"], "holding")             # 50 mm: still held
        carry(world, .02)
        tick(world, manager)
        self.assertEqual((state["phase"], state["rejection"], state["binding"]), ("lost", "weak_hold_travel", None))
        self.assertFalse(world.data.eq_active[world.model.equality(weld_name(weak_config().targets[0])).id])

    def test_firm_pinch_has_no_travel_limit(self):
        world = battery_world(.025)                              # the cell's middle above the walls: fingers on its sides
        central = pinch(world, .016)
        self.assertGreater(central, 155., central)
        manager, state = self.hold(world, weak_config())
        b = state["binding"]
        self.assertEqual((state["phase"], b["firm_hold"], b["weak_hold_origin_m"]), ("holding", True, None))
        carry(world, .2)
        tick(world, manager)
        self.assertEqual(state["phase"], "holding")

    def test_force_closure_at_the_firm_coefficient_alone(self):
        """A ForceClosureGraspTarget at mu 0.268 (the new cells' rule): the 108 deg pinch never welds, the side pinch does."""
        weak, firm = battery_world(.006), battery_world(.025)
        pinch(weak, .012)
        pinch(firm, .016)
        _, state = self.hold(weak, weak_config(mu=.268, weak=False))
        self.assertEqual((state["binding"], state["rejection"]), (None, "no_bilateral_candidate"))
        _, state = self.hold(firm, weak_config(mu=.268, weak=False))
        self.assertEqual(state["phase"], "holding")

    def test_fields_are_checked(self):
        with self.assertRaises(ValueError):
            WeakHoldGraspTarget("c", "c", friction_coefficient=1., firm_friction_coefficient=1.)
        with self.assertRaises(ValueError):
            WeakHoldGraspTarget("c", "c", weak_hold_travel_m=0.)
        self.assertEqual(list(asdict(WeakHoldGraspTarget("c", "c")))[-3:],
                         ["friction_coefficient", "firm_friction_coefficient", "weak_hold_travel_m"])


# --------------------------------------------------------------------------------------------- the wires
def wires_world(layout, ropes=()):
    """Two vertical wires of WIRE_N links each, rooted on a fixed base, their tips between two finger pads (12 mm tall,
    16 mm wide; the hand frame's origin, the TCP, at the pads' centre 5 mm below the tips: the centre of link 5).
    'behind': one behind the other along the closing direction, pressed 0.05 mm together, each pad 0.05 mm into its
    own wire. 'side': side by side across it, 1 mm apart, both pads 0.05 mm into both."""
    if layout == "behind":
        centres, pad = [(0., -(WIRE_R - 2.5e-5)), (0., WIRE_R - 2.5e-5)], 2 * WIRE_R - 7.5e-5
    else:
        centres, pad = [(-(WIRE_R + 5e-4), 0.), (WIRE_R + 5e-4, 0.)], WIRE_R - 5e-5
    tip = .05
    base = tip - WIRE_N * WIRE_L
    bits = {"A": 'contype="2" conaffinity="5"', "B": 'contype="4" conaffinity="3"'}     # no self contact
    bodies, targets = "", []
    for (x, y), name in zip(centres, "AB"):
        chain = ""
        for k in reversed(range(WIRE_N)):
            joint = f'<joint name="{name}_j{k}" type="ball" stiffness=".02" damping=".0005" armature="2e-6"/>'
            chain = (f'<body name="{name}_{k}" pos="0 0 {WIRE_L if k else 0}">{joint}'
                     f'<geom type="capsule" fromto="0 0 0 0 0 {WIRE_L}" size="{WIRE_R}" mass=".00018" {bits[name]}/>'
                     f'{chain}</body>')
        bodies += f'<body name="{name}_base" pos="{x} {y} {base}">{chain}</body>'
        targets += [GraspTarget(f"wire_{name}", f"{name}_{k}", (0., 0., WIRE_L / 2), .004) for k in range(WIRE_N)]
    xml = f"""<mujoco><option timestep="{DT / SUBSTEPS}" gravity="0 0 0"/><worldbody>{bodies}
 <body name="hand" mocap="true" pos="0 0 {tip - .005}" quat="{DOWN}">
  <body name="finger_a"><joint name="finger_a" type="slide" axis="0 -1 0"/>
   <geom type="box" size=".006 .0025 .008" pos="0 {-(pad + .0025)} 0" mass=".01"/></body>
  <body name="finger_b"><joint name="finger_b" type="slide" axis="0 1 0"/>
   <geom type="box" size=".006 .0025 .008" pos="0 {pad + .0025} 0" mass=".01"/></body>
 </body></worldbody>
<actuator><position joint="finger_a" kp="2000"/><position joint="finger_b" kp="2000"/></actuator>
<equality>{welds_xml(targets, lambda t: t.link)}</equality></mujoco>"""
    entities = {f"wire_{n}": {f"{n}_{k}": f"{n}_{k}" for k in range(WIRE_N)} for n in "AB"}
    return World(xml, entities, targets, ropes), targets


def rope_config(targets, bundle, **kwargs):
    if bundle:
        return MultiRopeGraspAssistConfig("contact_weld_v1", tuple(targets), initial_intent="close",
                                          weld_each_rope=("wire_A", "wire_B"), **kwargs)
    return GraspAssistConfig("contact_weld_v1", tuple(targets), initial_intent="close")


def touching(world):
    """The (finger, wire) pairs in contact, and whether the two wires touch."""
    m, c = world.model, world.contacts()
    owner = {m.body(f"{n}_{k}").id: n for n in "AB" for k in range(WIRE_N)}
    owner.update({m.body(f).id: f[-1] for f in ("finger_a", "finger_b")})
    pairs = {tuple(sorted((owner.get(int(x)), owner.get(int(y))), key=str)) for x, y in zip(c["link_a"], c["link_b"])}
    return {p for p in pairs if None not in p and p != ("A", "B")}, ("A", "B") in pairs


class RopePinch(unittest.TestCase):
    def run_ticks(self, world, config, ticks=12):
        backend, manager = assisted(world, config)
        for _ in range(ticks):
            tick(world, manager)
        return backend, manager, manager.hands["right"]

    def active(self, world):
        m, d = world.model, world.data
        return sorted(m.equality(e).name for e in range(m.neq) if d.eq_active[e])

    def test_plain_rule_rejects_both_layouts(self):
        for layout, ropes, reason in (("behind", (), "no_bilateral_candidate"), ("side", (), "ambiguous_candidates"),
                                      ("side", ("wire_A", "wire_B"), "ambiguous_candidates")):     # as rope chains
            world, targets = wires_world(layout, ropes)
            fingers = {("A", "a"), ("B", "b")} if layout == "behind" else {("A", "a"), ("A", "b"), ("B", "a"), ("B", "b")}
            self.assertEqual(touching(world), (fingers, layout == "behind"))
            _, _, state = self.run_ticks(world, rope_config(targets, False))
            self.assertEqual((state["binding"], state["rejection"]), (None, reason), (layout, ropes))
            self.assertEqual(self.active(world), [])

    def test_bundle_welds_one_link_of_each_wire(self):
        for layout in ("behind", "side"):
            world, targets = wires_world(layout)
            backend, manager, state = self.run_ticks(world, rope_config(targets, True))
            # the TCP is at the centre of link 5 (of 8, 2 mm each): each wire's touched link nearest it
            expected = (("wire_A", "A_5"), ("wire_B", "B_5"))
            self.assertEqual(state["phase"], "holding", layout)
            self.assertEqual(tuple(state["binding"]["target"]), expected, layout)
            self.assertEqual([p["target"] for p in state["binding"]["parts"]], list(expected))
            self.assertEqual(self.active(world), ["grasp/right/wire_A/A_5", "grasp/right/wire_B/B_5"])
            backend.robot.gripper = .015            # an opening command releases both welds
            tick(world, manager)
            self.assertEqual([e["event"] for e in manager.events], ["released"])
            self.assertEqual(self.active(world), [])

    def test_bundle_needs_the_wires_pressed_together(self):
        """One behind the other but 1 mm apart: each pad touches its own wire, the wires do not touch: no grasp."""
        world, targets = wires_world("behind")
        m, d = world.model, world.data
        for name, dy in (("A_base", -.0005), ("B_base", .0005)):
            m.body_pos[m.body(name).id, 1] += dy
        for joint in ("finger_a", "finger_b"):             # each pad opens with its wire
            d.qpos[m.joint(joint).qposadr[0]] += .0005
        world.forward()
        self.assertEqual(touching(world), ({("A", "a"), ("B", "b")}, False))
        _, _, state = self.run_ticks(world, rope_config(targets, True))
        self.assertEqual((state["binding"], state["rejection"]), (None, "no_bilateral_candidate"))

    def test_both_welds_carry_the_tips_and_open_releases_them(self):
        world, targets = wires_world("behind")
        backend, manager, state = self.run_ticks(world, rope_config(targets, True))
        m, d = world.model, world.data
        tips = [m.body(n).id for n in ("A_5", "B_5")]      # the welded links
        before = d.xpos[tips].copy()
        d.ctrl[:] = [d.qpos[m.joint(j).qposadr[0]] for j in ("finger_a", "finger_b")]
        start = d.mocap_pos[0].copy()
        for k in range(40):                     # the hand moves 2 mm across, along the wires' width direction
            d.mocap_pos[0] = start + (0.002 * min(1., (k + 1) / 30), 0., 0.)
            tick(world, manager, step=lambda: None)
        self.assertEqual(manager.hands["right"]["phase"], "holding")
        moved = d.xpos[tips, 0] - before[:, 0]
        self.assertTrue(np.all(moved > .0015), moved)
        backend.release(manager.hands["right"]["binding"])
        self.assertEqual(self.active(world), [])

    def test_rope_rotation_limit(self):
        """A bundle weld 0.3 rad off its acquired rotation: lost under the plain 0.15 rad, kept under 0.6 rad."""
        from scipy.spatial.transform import Rotation
        for limit, expected in ((None, "relative_pose_error"), (.6, None)):
            world, targets = wires_world("behind")
            backend, _, state = self.run_ticks(world, rope_config(targets, True, rope_max_rotation_error_rad=limit))
            binding = deepcopy(state["binding"])
            q = np.asarray(binding["parts"][0]["relative_pose"]["quaternion_wxyz"])
            turned = Rotation.from_quat(q[[1, 2, 3, 0]]) * Rotation.from_rotvec((.3, 0., 0.))
            binding["parts"][0]["relative_pose"]["quaternion_wxyz"] = turned.as_quat()[[3, 0, 1, 2]].tolist()
            self.assertIsNone(backend.check_health(state["binding"]))
            self.assertEqual(backend.check_health(binding), expected, limit)
            backend.release(state["binding"])


class PlainConfigurations(unittest.TestCase):
    def test_serialized_fields_and_backend_paths_unchanged(self):
        """A configuration without the opt-in classes serializes as before (its asdict is in every tick record and in
        each frozen instance's state_sha) and leaves the backend's opt-in code unreached."""
        self.assertEqual(list(asdict(GraspTarget("e", "l"))), ["entity", "link", "center_m", "radius_m"])
        self.assertEqual(list(asdict(GraspAssistConfig())), [
            "model", "targets", "stable_time_s", "max_tcp_distance_m", "max_penetration_m", "max_translation_error_m",
            "max_rotation_error_rad", "max_finger_opening_fraction", "intent_deadband", "initial_intent"])
        world = battery_world(0.)
        backend = MujocoGraspBackend(world, Robot(world), cell_config(False))
        self.assertEqual((backend._closure, backend._each_rope, backend._weak), ({}, frozenset(), {}))
        self.assertEqual(list(asdict(ForceClosureGraspTarget("e", "l"))),
                         ["entity", "link", "center_m", "radius_m", "friction_coefficient"])
        self.assertFalse(hasattr(backend, "conflicts"))     # the GraspManager keeps its equality test

    def test_simulator_config_reads_the_opt_in_classes_back(self):
        from topogym.simulator.config import SimulatorConfig
        import json
        _, targets = wires_world("behind")
        for config in (rope_config(targets, True, rope_max_rotation_error_rad=.6), cell_config(True), cell_config(False),
                       weak_config()):
            read = SimulatorConfig(task_id="horseshoe_ring", grasp_assist=json.loads(json.dumps(asdict(config))))
            self.assertEqual(read.grasp_assist, config)
            self.assertIs(type(read.grasp_assist), type(config))
            self.assertEqual([type(t) for t in read.grasp_assist.targets], [type(t) for t in config.targets])


if __name__ == "__main__":
    unittest.main()
