"""Per-family MuJoCo settings: what MuJoCo needs beyond the task's own declarations. A family port adds its entry here
(docs/rcb/MUJOCO_CORE.md); the profile is frozen with every instance's scene (scene.json), so changing it here never
changes an existing instance.

Fields:
  solver   MuJoCo <option>: integrator, solver, iterations, ls_iterations, tolerance, cone, impratio.
  collision  MuJoCo <option> ccd_iterations / ccd_tolerance, the same for every family (_COLLISION below); optional
           multiccd: False writes <flag multiccd="disable"/> (MuJoCo's multi-point convex collision off for that scene;
           absent: on, MuJoCo's default).
  contact  every collision geom: solref [timeconst, dampratio] / solimp, the task bodies' and table's condim and friction
           (sliding from the task material's friction, else `friction`; torsional, rolling). The robot keeps its MJCF's
           condim and friction (ALOHA: condim 6, mu 1); margin optional.
  weld     the assisted grasp welds' solref / solimp.
  mjcf_equality  optional {solref, solimp}: replaces the authored softness of an MJCF entity's <equality><connect>.
  birth_overlap_exclude  optional True: an MJCF entity's same-root, non-parent body pairs that overlap in the authored
           pose are excluded from contact (Genesis' neutral-overlap filter; mjcf_entity.py).
  robot    armature: every robot joint's armature (Genesis imports MJCF joints with its default armature 0.1);
           self_contact: {solref, solimp} of contacts between two robot geoms (solmix 0 on the robot's geoms, so its
           contacts with anything else take the other geom's parameters); the own_contact geoms (grasp pads) keep their
           own class and solmix 1 unless self_contact_pads (their grasps keep the mean of both classes);
           coupling: {solref, solimp} of the robot MJCF's joint couplings (equality joint: the ALOHA's second finger
           follows the driven one; MJCF default 20 ms);
           maxhullvert: the robot's collision meshes collide as convex hulls of at most this many vertices (MuJoCo's
           maxhullvert; visual meshes are untouched), maxhullvert_skip: mesh-name prefixes that keep their full hull.
  static_contact  {soft, keep}, the contacts of static bodies (the table, the ground, task links fixed to the world;
           docs/rcb/MUJOCO_CORE.md has the mixing table). soft: a class (SOFT_CONTACT) the static bodies not kept carry
           with solmix 0: a robot geom with robot.self_contact (needed) touching them gets it, a grasp pad keeps its own
           class, a task body its own parameters. keep: {entity: reason} static bodies that keep their own parameters
           and win every contact with them (solmix 1e9), for thin static geometry that a softer contact would let a robot
           geom pass through. soft None: every static body keeps its parameters and wins. Absent: no change.
  coacd    None: a mesh collides as its convex hull; else CoACD settings for a non-convex collision mesh.
  split    optional {entity: "module:function"}: the family's exact convex decomposition of that entity's non-convex
           collision mesh (function(vertices, faces) -> [(vertices, faces)]), checked against the declared mesh
           (scene._family_parts: volume, containment, coverage).
  prism_partition  True: a link whose collision meshes are convex prisms of one extrusion (slabs of a plate) collides
           as an exact re-partition of the same solid into fat convex prisms, without slivers (prisms.py); a link of
           several slab groups and other convex pieces re-partitions each group. {"fatten_m": w}: then pieces thinner
           than w are re-cut exactly with a neighbour (prisms.fatten).
  rope     a FEM rope (EntitySpec "fem_elastic") as a rigid capsule chain: link length, rod integration rules, node joints,
           self-contact window, robot contact (rope_chain.py, ROPE_DEFAULTS).
  capsules optional: a URDF cylinder with a sphere of its radius on each end cap collides as one capsule (the same solid).
  drag     optional quadratic air drag on light links: {links: fnmatch pattern on body names, k: N s^2/m^2}; the world
           applies f = -k |v| v at each link's centre of mass once per tick (MujocoWorld._apply_drag).
  joints   optional MJCF joint attribute rules: [{match: fnmatch pattern on joint names, attribute: value, ...}].
  mjcf     optional per MJCF entity: {entity: {contact: {attribute: value}}} on every geom of that entity.
  pairs    optional explicit contacts between task bodies: [{bodies: [[fnmatch patterns], [fnmatch patterns]] on MuJoCo
           body names, condim, friction (5 values), solref, solimp, margin, gap, optional solreffriction and reason}]. The
           contacts MuJoCo would make between a body of the first list and one of the second become <pair>s with exactly
           these parameters (no solmix, no priority), and <exclude> drops the automatic ones (SceneBuilder._pairs;
           scene.json "pairs" counts them).
  plastic  optional elastoplastic ball joints: {joints: fnmatch pattern on joint names, yield_angle_deg}. Each named joint
           needs a stiffness (set with `joints`): its spring is elastic up to the yield angle, and past it the rest
           orientation follows the joint (MujocoWorld._plastic_return, before every MuJoCo step). The rest orientations
           are state.
"""
from __future__ import annotations

import math
from copy import deepcopy

# Convex collision run to convergence, every family: at MuJoCo's
# defaults (35 iterations, tolerance 1e-6) GJK / EPA returned wrong depths and normals for thin and curved shapes (claws
# medium: a 7.56 mm bar passed through another, the validator's M4; fiber: a 130 mm contact, F3 / F4). 500 / 1e-10 fixes
# them at no measurable cost (claws, 600 controls per tier at the same load: 20.0-53.7 ms per control with either value).
_COLLISION = dict(ccd_iterations=500, ccd_tolerance=1e-10)

# The soft contact class, for contacts that need no task precision (robot.self_contact, static_contact.soft): a
# position-driven robot pressed into something rigid through a nearly hard contact created energy (validator M6: an arm
# folded onto itself, +22 J; F1 / F2: the Robotiq against fiber's cradle and fixed terminal, 854 J). solref 5 ms,
# overdamped (dampratio 2), solimp .9 .95 (2026-10-10 sweep, docs/rcb/MUJOCO_CORE.md: MuJoCo's default 20 ms let a
# Robotiq sink 10-12 mm into a thin wire or plate; at dampratio 1 the stiffer classes broke down again). It is for
# thick bodies: on thin static geometry (buzz's 1.47 mm wire) any class softer than the task's lets a hull pass the axis
# (B1), and such a body keeps its own contact (static_contact.keep).
# Constant impedance (solimp .95 .95) since 2026-10-11: with .9 -> .95 over 1 mm, an arm driven into a static fixture
# through several coupled soft rows created energy (rope_threading kt1_3 e116, fiber FB5b e131, knot easy_v8 e245;
# docs/rcb/MUJOCO_CORE.md, "The soft class's impedance"). Frozen scenes carry SOFT_CONTACT_V1; MujocoWorld.build
# migrates them (scene.migrate_contacts).
SOFT_CONTACT = dict(solref=[.005, 2.], solimp=[.95, .95, .001, .5, 2.])
SOFT_CONTACT_V1 = dict(solref=[.005, 2.], solimp=[.9, .95, .001, .5, 2.])

# L12's bake-off profile `mujoco-claws-allcontact125-source-anchor` (2026-10-09): Newton / 100 iterations,
# implicitfast, elliptic cone with impratio 10 (the ALOHA model's own setting), every contact at a 1.25 ms time constant,
# the weld at 1.25 ms, Genesis' armature 0.1 on the robot joints; v2 (2026-10-09, core, on the validate agent's 95 claws
# breakdown reproducers, each a fresh Session under its observer: energy balance, depth, causality):
#  - the ALOHA's collision hulls at most 64 vertices: its gripper hulls had 1911 (wrist mount) and 7003 (D405 camera)
#    vertices, its fingers 346 / 348, and MuJoCo's multi-point convex collision (multiccd, on by default in 3.14)
#    produced bogus deep contacts with wrong normals on them (robot self-collision +65 J, a gripper driven 21 mm into a
#    claw); at 128 vertices they come back (7 of 95 breakdowns). The capped hulls stay inside the full ones, within
#    0.1-2.0 mm (fingers 0.6 / 1.0 mm).
#  - contacts overdamped (dampratio 2) and nearly hard (solimp dmax .99): a piece pressed by kN drive forces sinks
#    (1 - d) / d * timeconst^2 * F / m deep, 5x less at .99 than at .95; the remaining uncaused jumps went with dampratio 2.
#  - timeconst stays 2 dt (1.25 ms): the claws are thin (medium bars 7.56 mm, hard tubes cut into 1.7 mm CoACD slices)
#    and must not sink to half their thickness; at 2.5 ms abuse reached 7.5 mm and 4 of 95 broke down (0 of 95 here).
# Result: 61 -> 0 of 95 breakdowns, the deepest task contact under abuse 21.5 -> 3.9 mm, per-control time unchanged.
# v3: robot-robot contacts get MuJoCo's default softness (robot.self_contact, 20 ms / .9 .95, solmix 0: robot-task and
# robot-table contacts unchanged): with the task's nearly hard contact an arm folded onto itself at its torque caps
# created energy. The references touch no robot-robot pair (same state hashes).
_RIGID = dict(
    solver=dict(integrator="implicitfast", solver="Newton", iterations=100, ls_iterations=50, tolerance=1e-8,
                cone="elliptic", impratio=10.),
    collision=_COLLISION,
    contact=dict(solref=[.00125, 2.], solimp=[.9, .99, .001, .5, 2.], condim=3, friction=.3, table_friction=1.,
                 torsional_friction=.005, rolling_friction=.0001, margin=0.),
    weld=dict(solref=[.00125, 1.], solimp=[.9, .95, .001, .5, 2.]),
    robot=dict(armature=.1, maxhullvert=64, self_contact=dict(solref=[.02, 1.], solimp=[.9, .95, .001, .5, 2.])),
    coacd=None,
    memory="256M",
)

# CoACD settings of L12's strict decomposition of the hard claws (threshold .002: 2944 / 2862 hulls, sampled aperture
# contraction at most 0.37 / 0.30 mm).
_COACD_STRICT = dict(threshold=.002, resolution=2000, mcts_nodes=20, mcts_iterations=150, mcts_max_depth=3, seed=0,
                     preprocess_mode="off", max_convex_hull=-1, merge=False)

# fiber_bundling: the released fiber physics as Genesis 1.2.1 runs it (Newton, 50 iterations,
# pyramidal cones, implicit-fast integration, 0.625 ms steps). Geoms without their own contact parameters take Genesis'
# constraint_timeconst 10 ms and its default friction 1; the fiber tree keeps bundle.xml's (2 ms, .95 .99 .001, mu .35) and
# the Robotiq pads theirs (4 ms, .95 .99 .001; their collision boxes end at the visual pad face). The robot as Genesis
# imports its MJCF: armature 0.1, springs about 0, contact priority ignored. The robot's other collision geoms take the
# pads' impedance (.95 .99 .001): MuJoCo's soft contact lets a light chain link squeezed by the arm sink by about
# (1 - d) / d * timeconst^2 * F / m; at Genesis' averaged d = .97 a lead link sank 10.8 mm into link3 (corpus a2b4fc2f,
# hard); at .99 it stays at 3.0 mm, and the released references do not change (no robot-cable contact but the pads).
# Evidence (Genesis vs MuJoCo on the released references, the corpus): tmp/2026-10-09_robotangle_mujoco/fiber/STATUS.md.
# v3 (2026-10-09, fiber agent; the validate agent's 16 fiber breakdown reproducers F1-F5 / S1, my 22-stream corpus and
# the 4 references, each a fresh Session under the validator's observer, mujoco 3.14.0):
#  - every contact at 4 dt (2.5 ms), dampratio 1: the robot's (its pads too), the fixtures', the cable's. Softer contacts let
#    links sink deeper and hull normals flip (robot and fixture contacts at 10 ms: 10 of 14 reproducers broke down, at 20
#    ms 13, at 2.5 ms 2); the corpus' deepest task contact 9.6 -> 2-4 mm.
#  - the Robotiq's collision hulls at most 128 vertices (base 803, base_mount 620 before): inside the full hulls within
#    0.03-0.58 mm. The Panda arm keeps its full hulls (at 128 link1 would lose 21 mm).
#  - air drag on the fine fibres, k = 5e-5 N s^2/m^2 (still air on a 2.4 mm x 8 mm link is 1.4e-5): whipped to 34-49 m/s
#    they otherwise outrun every contact at 0.625 ms steps; at 1 m/s a link feels 5e-5 N (0.17 % of its speed per tick).
#  - every static body soft (static_contact.soft, robot.self_contact: SOFT_CONTACT), the cradle's 10 mm plates, its 8 mm
#    pad, the 8 mm ring tube and the 12 mm post included: robot-static and robot-robot contacts get the soft class, the
#    cable's contacts keep its 2.5 ms (solmix). The 16 reproducers and 20 corpus streams under the validator's observer:
#    0 breakdowns (2 with the cradle, ring and post kept exact, 3 without static_contact); robot-static depth at most
#    3.8 mm (the fixed terminal box), 2.8 mm against a 10 mm cradle plate (half of it is 5 mm).
# v4 (2026-10-10): the cable as wire. The user (2026-10-09): it bends like everyday iron wire under force and stays bent;
# nothing bends before contact.
#  - Every cable joint is elastoplastic (profile plastic, MujocoWorld._plastic_return): a stiff spring, elastic up to
#    0.4 deg. Past its yield torque its rest orientation follows, so a bent joint stays bent with at most 0.4 deg of
#    elastic give, and below the yield nothing creeps.
#  - The yield angle is iron wire's. Annealed iron yields at a strain of about 0.1 % (yield stress ~200 MPa, E 200 GPa).
#    A wire of radius 1.2 mm then bends elastically up to a curvature of 0.001 / 1.2 mm = 0.83 /m: 0.38 deg over a strand
#    segment of 8 mm. At 1 deg (2.6x springier) a bent strand sprang back 4.5 mm.
#  - Yield torques: strand 1e-3 N m (a strand tip starts to bend at about 14 mN), sheath 0.1 N m.
#  - Damping: critical, 2 sqrt(k I), I = armature + link: strand 7.8e-5 (I 1.04e-8 kg m2), sheath 0.024 (I 9.45e-6). The
#    spring is explicit and the damping implicit: stable while (w dt)^2 < 4 + 4 w dt, w dt < 4.8; the stiffest, a strand
#    tip, has w dt = 2.35.
#  - The cable's own contacts at 2 dt, overdamped (solref 1.25 ms, dampratio 2): the same stiffness as 2.5 ms at
#    dampratio 1 (k ~ 1 / (timeconst dampratio)^2), twice the damping. The arm pressing a trunk link at ~530 N stored
#    ~0.5 J in the soft penetration and the contact released it into the light cable (core, the reproducer S1 cem_e52:
#    pair net +0.47 J, X rise 0.19 J; overdamped: <= 0 J, 0.04 J).
#  - Why not v3's dry friction (frictionloss, impedance .995): it crept under every held load. After a reference's
#    success the bundle settled 4-5 mm in ~30 s, to the frictionless equilibrium. No friction setting held it (time
#    constant, impedance, impratio, noslip), and the stiffer settings broke reproducers.
_YIELD_DEG = .4


def _plastic_rule(match, yield_torque, **attrs):
    return dict(match=match, stiffness=yield_torque / math.radians(_YIELD_DEG), **attrs)


FIBER_PLASTIC_JOINTS = [_plastic_rule("fiber_bundle::fiber_bend_*", 1e-3, damping=7.8e-5),
                        _plastic_rule("fiber_bundle::lead_bend_*", .1, damping=.024),
                        _plastic_rule("fiber_bundle::trunk_bend_*", .1, damping=.024)]

_FIBER = dict(
    solver=dict(integrator="implicitfast", solver="Newton", iterations=50, ls_iterations=50, tolerance=1e-8,
                cone="pyramidal", impratio=1.),
    collision=_COLLISION,
    contact=dict(solref=[.0025, 1.], solimp=[.9, .95, .001, .5, 2.], condim=3, friction=1., table_friction=1.,
                 torsional_friction=.005, rolling_friction=.0001, margin=0.),
    weld=dict(solref=[.00125, 1.], solimp=[.9, .95, .001, .5, 2.]),
    robot=dict(armature=.1, no_priority=True, pad_collision=True,
               contact=dict(solimp=[.95, .99, .001, .5, 2.]), maxhullvert=128, maxhullvert_skip=("link", "hand"),
               self_contact=SOFT_CONTACT),
    static_contact=dict(soft=SOFT_CONTACT, keep={}),
    mjcf=dict(fiber_bundle=dict(contact=dict(solref=[.00125, 2.]))),
    # FB2: a Panda pushing at its drive caps drove the light front sheath link (about
    # 1.9 g of effective mass per contact row under the exact constraint diagonal) 15 mm into the ring's 4 mm tube within
    # one tick. The sheath against the ring gets its own class, the mixed contact's parameters but nearly hard
    # (impedance .999 / .9999); its other contacts and the strands' keep theirs.
    # FB3: the general convex collider (GJK/EPA) on the
    # sheath's flat front cap against one of the ring tube's short cylinders (1.7 mm long) returned, for single MuJoCo
    # steps, a witness 5-14 mm deep along that cylinder's own axis (the ring's tangent) where the true overlap was
    # 0.3 mm, and the near-hard pair then pushed 20-115 kN. So the sheath meets only the tube's 64 spheres (1.72 mm
    # apart: a union of spheres with dents <= 0.093 mm; analytic sphere x cylinder / sphere x sphere contacts). The
    # strands keep the whole tube.
    pairs=[dict(bodies=[["fiber_bundle::trunk_*"], ["bundling_ring::ring"]], geom_types=[None, ["sphere"]], condim=3,
                friction=[1., 1., .005, .0001, .0001], solref=[.00125, 2.], solimp=[.999, .9999, .001, .5, 2.],
                margin=0., gap=0., reason="FB2/FB3: the sheath must not sink into the ring's thin tube; sphere segments only")],
    drag=dict(links="fiber_bundle::fiber_*", k=5e-5),
    joints=FIBER_PLASTIC_JOINTS,
    plastic=dict(joints="fiber_bundle::*_bend_*", yield_angle_deg=_YIELD_DEG),
    coacd=None,
    memory="256M",
)

# claws v4 (2026-10-10): the soft class on robot-robot contacts and, opted in, on the robot against the table and the
# ground (static_contact.soft); the pieces keep the nearly hard contact everywhere. On the 97 reproducers 0 breakdowns
# stay 0, violent 38 -> 26, the largest contact force 4.9 -> 1.4 kN, robot contact depth 5.1 -> 2.75 mm; the 42-stream
# corpus keeps 0 flags (deepest contact 2.11 -> 1.11 mm); the references keep their state hashes.
# claws v5 (2026-10-10): the ALOHA's follower finger on a stiff coupling (robot.coupling, ALOHA_COUPLING). The MJCF's
# finger equality keeps MuJoCo's default 20 ms, and against the driven finger it created energy: a gripper command
# switched between 0.021 and 0.037 m every control made the left coupling deliver +0.21 J (+0.32 J with the hand in free
# space) and claws easier / medium broke down. The real gripper moves both fingers from one
# servo through a rigid linkage; at 2 dt, overdamped, the coupling only dissipates (-0.02 / -0.03 J on the same rows).
ALOHA_COUPLING = dict(solref=[.00125, 2.], solimp=[.9, .95, .001, .5, 2.])
_CLAWS = dict(_RIGID, robot=dict(_RIGID["robot"], self_contact=SOFT_CONTACT, coupling=ALOHA_COUPLING),
              static_contact=dict(soft=SOFT_CONTACT))
# claws v6 (2026-10-10, W4): the grasp welds at 4 dt (2.5 ms). On the exact constraint diagonal a weld runs at its
# nominal time constant; at 2 dt (1.25 ms, MuJoCo's floor), beside the 2 dt contacts and finger coupling, it ratcheted a
# jam's load (the validator's CL6, claws easy periodic_e38: 22 -> 149 N in 20 ms, then the slip released the piece into
# the finger hull, 11.3 mm deep). At 4 dt that stream stays CLEAN (0.5 mm), also when switched in mid-jam (1.1 mm), and
# every claws reference succeeds at its recorded row (new state hashes).
_CLAWS_WELD = dict(_RIGID["weld"], solref=[.0025, 1.])

# hedgehog_in_cage (hedgehog agent, 2026-10-09): a rigid ball of convex pieces (an icosphere core, flat-ended dowel
# spikes) in a fixed cage of convex pieces, Panda + Robotiq. The claws' rigid contact policy v2 (the task's own Genesis
# constraint time constant is the same 1.25 ms). The robot as Genesis imports it (fiber's import: the Robotiq pads keep
# their own contact parameters); the Robotiq's collision hulls capped at 128 vertices (at most 0.58 mm inside the full
# hulls, fiber's measurement), the Panda arm's kept whole.
# hedgehog_v4 / hedgehog_v3_waisted: robot-robot contacts on the soft class
# (robot.self_contact = SOFT_CONTACT, as claws, keyhole and coil). A random arm target folded the wrist until link5 pressed
# the open Robotiq's right coupler; on the springref 2.62 that contact grew to 166 kN in 9 ticks (stress R1,
# runs/stress_20261011/robotiq_springref_coupler_breakdown/).
_HEDGEHOG = dict(_RIGID, robot=dict(armature=.1, own_contact=True, no_priority=True,
                                    pad_collision=True, maxhullvert=128, maxhullvert_skip=("link", "hand"),
                                    self_contact=SOFT_CONTACT))

# cast_keyhole (keyhole agent, 2026-10-09): two 9 mm cast plates (ALOHA, contact_weld_v1), declared as exact convex slab
# prisms (topogym.assets.geometry.cast_keyhole: scanline trapezoids, no CoACD). The claws v2 contacts (2 dt overdamped,
# dmax .99, converged CCD, ALOHA hulls <= 64 vertices): the task asks for Genesis' 1 ms constraint time constant, MuJoCo's
# floor is 2 dt. The slabs are cut at every outline vertex height, down to 0.002 mm thin (hard gold: 12 of 155 under
# 0.1 mm); prism_partition collides the same solid as fat exact convex prisms (thinnest 0.46 mm, hard gold 155 -> 54).
# keyhole_v2: frictional creep. A piece resting interlocked on its partner must come to rest (Genesis
# holds it): at claws v2's solimp dmin .9 (impedance .9 at small depths) the soft friction
# let medium's interlocked gold creep 11 mm in 6 s, still sliding at 3.6 mm/s; contacts nearly hard at every depth (dmin
# .99) and friction 5x stiffer than normal (impratio 50, MuJoCo's remedy for slip on elliptic cones): 1.8 mm, 0.05 mm/s.
# keyhole_v3: the ALOHA's follower finger on claws v5's stiff coupling (robot.coupling,
# ALOHA_COUPLING; porting guide trap 10, the E1 fix every ALOHA family takes: the MJCF's finger equality at MuJoCo's
# default 20 ms created energy against the driven finger under a fast gripper oscillation). The contacts, solimp and
# impratio stay keyhole_v2's; the references are re-recorded on it.
# The same solid re-cut where the partition leaves pieces under 0.5 mm (prism_partition fatten_m, as hedgehog): hard's
# gold had three interior slivers 0.456 / 0.457 / 0.463 mm wide (neighbouring pieces on both sides of the thin
# direction, so the plate is not thin there; a contact pushed ~0.23 mm into one could flip its normal, porting guide
# trap 4). Re-cut exactly with a neighbour each, hard's thinnest piece is 1.934 mm; the other tiers have none under
# 0.5 mm (easy's thinnest stays 0.581 mm) and keep their pieces. Birth, a 440-control nop and the keyhole channel
# (300 poses about the assembled one against the declared meshes) are unchanged on hard.
# Accepted (2026-10-10): a slow creep at birth on very_easy and easier. The released standing assembly topples at birth
# and comes to rest within 0.45 s; on these two tiers the gold then leans on the silver and the table with its friction
# saturated (|f_t| / (mu f_n) = 1.0 at both contacts after 10 s), and the pieces slide on steadily at 1.1 / 0.95 um/s
# (relative, 50-60 s after birth; about 0.6 mm over a 10 min episode). It is the limit of Coulomb friction in that pose,
# not the solver: a 1e-10 tolerance changes nothing, noslip 10 slows it to 0.35 um/s at a higher step cost and does not
# stop it. 50x below the 0.05 mm/s accepted for medium's interlocked gold above. easy, medium, hard: <= 0.08 um/s.
# keyhole_v4: robot-robot contacts on the soft class (robot.self_contact = SOFT_CONTACT, as claws
# v4). On _RIGID's 20 ms class a position-driven ALOHA link sank more than 10 mm into the other arm under random joint
# targets (the acceptance round on df00ee35b: easier random_e57, a left finger 11.4 mm into the right gripper base at
# 163 N; hard random_e47, a left finger 13.7 mm into a right finger; reproducer J1). On the 5 ms class the same streams
# read 3.4 / 4.7 mm and CLEAN. The five references touch no robot-robot pair: births and final state hashes unchanged.
_KEYHOLE = dict(_RIGID, prism_partition=dict(fatten_m=.0005), solver=dict(_RIGID["solver"], impratio=50.),
                contact=dict(_RIGID["contact"], solimp=[.99, .99, .001, .5, 2.]),
                robot=dict(_RIGID["robot"], self_contact=SOFT_CONTACT, coupling=ALOHA_COUPLING))

# The rope families: knot_untying and rope_threading, the FEM rope as a rigid capsule chain
# (rope_chain.py; evidence in tmp/2026-10-09_robotangle_mujoco/rope). The claws contact policy v4 (the soft class on
# robot-robot and robot-static contacts), the rope's own friction (its material's friction_mu 0.3) against the table,
# itself and the fixtures (MuJoCo takes the larger of two geoms' friction: the ALOHA's fingers keep mu 1), the ALOHA as
# claws imports it.
# knot_chain_v2 (2026-10-10): claws_v5's stiff follower-finger coupling (robot.coupling, inherited from _CLAWS), the
# chain's bend stops, nearly hard node connects (impedance .9999; v1: .95-.99, ~0.1 mm per newton of tension) and the
# rope's contacts acting within its IPC contact distance (geom margin 0.2 mm: resting contacts keep a positive gap).
# Chain integrity under the robot's real loads (in-Session rows at the servo caps, the node gap after every MuJoCo step):
# two-arm pulls 0.005-0.014 mm, +-1.5 rad wrist twists 0.007-0.071 mm, whips at link speeds of 2.6 m/s 0.031-0.093 mm,
# flat under a steady load. No real grasp passes a rope more than ~8 N: a grasp weld on a rope link is lost to its pose
# error limit at ~8 N, the friction pinch slips below 1 N, and a knot's loop slides through its crossing at ~5 N. A
# test-only stiff weld held a 14.2 N two-arm pull at 0.030 mm. Pulling further leaves the arms' reach.
# The tasks' failure "rope_volume_invalid" (an inverted or collapsed FEM volume, persisting 0.25 s under the Session's
# rule R2) never fires on these scenes: its physical cause, a crushed tetrahedral volume, does not exist on a chain of
# rigid links, and the chain reports volume ratio 1. Nothing replaces the rule (protocol unchanged).
_ROPE = dict(_CLAWS, contact=dict(_CLAWS["contact"], friction=.3, table_friction=.3), memory="512M",
             rope=dict(link_m=.0045, inertia_rule=4., damping_rule=.3, exclude_links=3,
                       joint=dict(solref=[.00125, 1.], solimp=[.9999, .9999, .001, .5, 2.]), robot_contact="full",
                       contact_margin_m=.0002))

# buzz_wire_3d (buzz agent, 2026-10-09): a steady-hand toy. Its wire and the wand's loop and stem are authored as chains
# of URDF cylinders with a sphere of the same radius on each end cap: exact capsules ("capsules": one MuJoCo capsule per
# cylinder + end-sphere triple, the same solid), so wand-on-wire touches come from MuJoCo's analytic capsule-capsule test
# at distance <= 0 (no margin, no hull). The deck is convex (its convex tiers and feet). Contacts as claws (_RIGID: the
# Genesis scene's constraint_timeconst 1.25 ms, overdamped, nearly hard) with Genesis' default friction 1 on the wire and
# deck (the wand's own .3; both engines take the larger of a pair); the Panda + Robotiq as fiber imports it (the MJCF's
# own pad contacts, no priority), its Robotiq hulls capped at 128 vertices (within 0.58 mm of the full
# hulls; the Panda arm and hand meshes keep their hulls). The grasp weld nearly hard (solimp .99 / .999): it anchors at
# the hand origin, ~0.2 m from the 6 g wand, and at the claws' .9 / .95 its rotation rows gave way ~1 deg under the pads'
# ~3 N squeeze, so the held wand drifted 2-3 mm in the hand and medium's released route touched the wire (Genesis on the
# same rows: <= 0.33 mm, 0.15 deg); at .99 / .999 the drift is <= 0.08 mm, 0.03 deg. Every static body keeps its (the
# task's own) parameters and wins its contacts (static_contact soft None): the validator's B1 / B1b (a Robotiq pad, 3.5 g,
# driven into the 1.47 mm wire: 121 / 46 kN, +64 / +13 J) came from the pads' own softer 4 ms class averaged in, which let
# the pad's box face pass the wire's axis (2.0 mm deep); with it both are violent but causal (10.2 / 3.2 kN, 0.4 / 0.3 mm,
# <= 0.006 J). MuJoCo-default soft robot-static contacts made them worse (robot depth > 10 mm, +97 / +494 J).
_BUZZ = dict(
    _RIGID,
    contact=dict(_RIGID["contact"], friction=1.),
    static_contact=dict(soft=None),
    weld=dict(solref=[.00125, 1.], solimp=[.99, .999, .001, .5, 2.]),
    robot=dict(armature=.1, own_contact=True, no_priority=True, pad_collision=True,
               contact=dict(solimp=[.95, .99, .001, .5, 2.]), maxhullvert=128, maxhullvert_skip=("link", "hand")),
    capsules=True,
)

# blanket_flatten: the PBD cloth as a 2D flex over contact-sphere vertices (cloth.py; evidence in
# tmp/2026-10-09_robotangle_mujoco/blanket). The flex on the authored 51 x 41 grid, whose nodes are PBD particles (the
# birth is exact there; the PBD's sub-grid crumples are smoothed, the particles <= 6.3 mm off the coarse surface);
# membrane 3.3 kN/m, bending 3.3e-6 N m ("s3e7": the very_easy and easy virtual-hand routes succeed as on Genesis, at
# 8.45 / 12.75 s against 8.6 / 12.75 s); spheres r 4.5 mm (2 r above the grid's 8.49 mm quad diagonal: no holes); the
# table's static mu 0.5. The discrete integrator (mujoco 3.14 integrates flex elasticity only there) with CG / 1e-6 up
# to 200 iterations (below 50 the cloth jitters at rest; the discrete step's own M+K solve runs under the same cap: at
# 50 it stopped short every step, mjWARN_INERTIA; at 200 it converges, +4 % per control), one 2.5 ms step per tick; the
# ALOHA as claws v5 imports it (its finger coupling).
_CLOTH = dict(_CLAWS, solver=dict(integrator="discrete", solver="CG", iterations=200, ls_iterations=20, tolerance=1e-6,
                                  cone="elliptic", impratio=10.),
              contact=dict(_CLAWS["contact"], table_friction=.5), memory="1G",
              cloth=dict(mesh="visual", sphere_radius_m=.0045, young=3e7, thickness=1.1e-4))
# The switch to the PBD particle mesh (8181 vertices: exact for every particle, ~6x the cost per step; the spheres at
# r 2.9 mm, under half the 6 mm spacing, since MuJoCo 3.14's contact excludes are too slow): map
# ("blanket_flatten", None) to it.
_CLOTH_FINE = dict(_CLOTH, cloth=dict(_CLOTH["cloth"], mesh="particles", sphere_radius_m=.0029, young=9e6))

# cast_coil_v2_very_easy: the claws' contact policy v2 (_RIGID) for the design-6
# pieces, 35 mm bars of boxes and one round pin each, 1.5 mm design clearance; the pieces' friction is the task's (0.35).
# v2 (2026-10-10): the ALOHA's follower finger on claws v5's stiff coupling (ALOHA_COUPLING, the E1 fix), re-recorded
# together with the exact constraint diagonal.
# v3 (2026-10-11): robot-robot contacts on the soft class (robot.self_contact = SOFT_CONTACT), as coil_legacy_v3 and
# keyhole_v4, whose acceptance rounds on _RIGID's 20 ms class had an ALOHA link pressed 10.1-16.2 mm into the other arm.
_CAST_COIL_V2 = dict(_RIGID, name="cast_coil_official_v3",
                     robot=dict(_RIGID["robot"], self_contact=SOFT_CONTACT, coupling=ALOHA_COUPLING))

# lock_picking: the chain-lock tiers (lock_picking_<tier> since 2026-10-10, when they replaced the
# original tiers; until then lock_picking_v2_<tier>) on the claws' near-hard, overdamped task
# contacts (_RIGID): the key's shaft has 0.75-1 mm of radial play in the bores and its tabs 1.5-4 deg in the slots, so no
# contact may sink deep. The Panda+Robotiq is imported as for fiber, without a hull cap: capped at 64 vertices the
# Panda's arm hulls move by up to 21 mm, more than the wrist's 8 mm clearance over the key tray.
# 2026-10-10: the lock's own robot settings, the values of fiber's at F1 (e9e54207d) written out, so that fiber's tuning
# no longer changes the lock's scenes (it did twice: the soft self-contact class, then F1).
# lock_chain_v2 (2026-10-11): the Robotiq's spring links at the MJCF's springref 2.62 (robot.SPRING_REFS; v1 imported
# them about 0, springref_zero) and the grasp's finger-contact release (grasp.WELD_CONTACT_LOST_S), as every Panda +
# Robotiq profile. The births are unchanged; the references' rows reach success 28 rows (0.7 s) earlier on every tier,
# with new final states.
_LOCK = dict(_RIGID, name="lock_chain_v2",
             robot=dict(armature=.1, no_priority=True, pad_collision=True,
                        contact=dict(solimp=[.95, .99, .001, .5, 2.]), maxhullvert=128, maxhullvert_skip=("link", "hand"),
                        self_contact=dict(solref=[.005, 2.], solimp=[.9, .95, .001, .5, 2.])))

# horseshoe_ring (2026-10-09): five free steel bodies on the table, ALOHA, contact_weld_v1 (no static task body).
# Each horseshoe is an open band swept as 56 convex pieces (each the hull of two consecutive bevelled 16 x 8 mm
# cross-sections, so each piece is its own exact solid; the thinnest 6.1 mm along the arc, medium) plus two terminal
# eyes. The eyes, the two chain links and the ring are closed wire loops authored as a cylinder per centreline segment
# and a sphere on every vertex (topogym/assets/geometry/horseshoe_ring.py: write_body), wire radius 2.8 mm (eyes, ring)
# and 1.4 mm (chain links; the template's 1.4 / 0.7 mm at scale 2). Every collision shape is convex as declared: no
# mesh collides as the hull of a non-convex solid, so the eyes', links' and ring's bores stay open without split or
# CoACD.
#  - the claws contacts (_CLAWS, claws_v5: the shared rigid policy, 1.25 ms overdamped, solimp dmax .99, converged CCD,
#    ALOHA hulls <= 64 vertices; the soft class on robot-robot and robot-table contacts; the follower-finger coupling). The task asks for Genesis' 1 ms
#    constraint time constant, MuJoCo's floor is 2 dt. The chain wire is 2.8 mm thick, so a contact must stay well
#    below 1.4 mm deep.
#  - capsules: each loop collides as exact capsules (a cylinder and the two spheres on its end caps) instead of three
#    primitives under convex collision. A closed loop shares each sphere between two segments, so every other segment
#    becomes a capsule and the segments between them stay cylinders whose ends lie inside the capsules' caps: the same
#    solid, with every sphere absorbed. The shortest leftover cylinder is 1.51 mm long (an eye's tight end).
# Five tiers at rest (nop 400 + 40 hold controls under the stress observer): deepest contact 0.005 mm, task bodies at
# most 4.3 mm/s, no energy created, every body within 0.04 mm of its birth pose, nop not a success.
_HORSESHOE = dict(_CLAWS, capsules=True)

# cast_coil, the five released Genesis tiers (2026-10-09, kept and ported as they are): two cast pieces (ALOHA,
# contact_weld_v1), each declared as exact disjoint collision boxes (topogym.assets.geometry.cast_coil: orthogonal CSG
# partitioned on the coordinate planes), which MuJoCo collides as box primitives: no hull, no CoACD, no slivers. Thinnest
# declared box: very_easy 9.9 mm, easier 7.5 mm (comb / baffle, scale 0.5); easy, medium, hard 0.75 mm (a bridge across a
# layer gap). The assembled clearances (0.3 mm comb / baffle, 0.75 mm between layers) stay exact. The claws v2 contacts
# (2 dt overdamped, dmax .99, converged CCD, ALOHA hulls <= 64 vertices): the task asks for Genesis' 1 ms constraint time
# constant, MuJoCo's floor is 2 dt; its Genesis-only max_collision_pairs and noslip_iterations have no MuJoCo meaning.
# Keyed per tier, beside the design-6 redesign's own key ("cast_coil", "cast_coil_v2_very_easy").
# coil_legacy_v2 (2026-10-10): the ALOHA's follower finger on claws v5's stiff coupling (robot.coupling, ALOHA_COUPLING;
# porting guide trap 10, the E1 fix every ALOHA family takes); the references are re-recorded on it.
# coil_legacy_v3: robot-robot contacts on the soft class (robot.self_contact = SOFT_CONTACT), as
# keyhole_v4: on _RIGID's 20 ms class an ALOHA link sank 10.7-13.9 mm into another robot link (the acceptance round on
# df00ee35b: easier random_e79 and random_e141, hard cem_e45; reproducer J2), 2.5-3.3 mm on the 5 ms class. The five
# references touch no robot-robot pair: births and final state hashes unchanged.
_COIL_LEGACY = dict(_RIGID, name="coil_legacy_v3",
                    robot=dict(_RIGID["robot"], self_contact=SOFT_CONTACT, coupling=ALOHA_COUPLING))

# Rigid link chains: slotted_post_ring's chain settings (very_easy / easier and easy / medium / hard_chain: #108's
# fixture-and-chain file; the FEM settings' chains: the 10-07 chain of fem_chain.py) and orbits very_easy / easier
#. The cords are the team's 10-07
# chain standard, kept as the task files author it (16.5 mm capsule links on ball joints, damping 2e-4; armature 1e-6,
# slotted, and 1e-5, orbits; orbits' bend stiffness k = E pi r^4 / (4 L) at E 100 MPa = 0.058 N m/rad per joint; friction
# 0.3, condim 3). What MuJoCo adds (one set of numbers for every chain, for knot / rope to start from):
#  - time step 0.625 ms (2.5 ms ticks of 4 substeps, rcb_topogym simulator_config): the claws / fiber step. The links are
#    light (slotted 0.41 g, orbits 0.12 g) and their armature dominates their swing (slotted 1e-6 kg m2 is ~26x a link's
#    own inertia about its joint, orbits 1e-5 ~1000x); orbits' bend spring then swings at sqrt(k / armature) = 76 rad/s,
#    0.05 rad per step. The near-hard contacts below need the step: MuJoCo floors a time constant at 2 dt.
#  - contacts (mjcf on the chain entities, _CHAIN_GEOMS: every chain, bead, disc, ring and handle geom; the fixtures and
#    table through `contact`): the claws v2 class, 1.25 ms (2 dt), overdamped (dampratio 2), solimp .9 .99 .001. A link
#    pressed by the arm sinks about (1 - d) / d * timeconst^2 * F / m: 0.4 mm for 10 N on a 0.41 g slotted link at dmax
#    .99. MuJoCo's default 20 ms (what the authored cord.xml / ring.xml would get) lets it sink 250x deeper. The authored
#    1-2 ms of #108 and orbits are Genesis values below MuJoCo's 2 dt floor.
#  - assisted grasp welds: 1.25 ms, dampratio 1 (the claws weld).
#  - robot: armature 0.1 (Genesis' import), ALOHA collision hulls <= 64 vertices, robot-robot and robot-static contacts
#    in the soft class (claws v4).
_CHAIN = _CLAWS
_CHAIN_GEOMS = dict(contact=dict(solref=_RIGID["contact"]["solref"], solimp=_RIGID["contact"]["solimp"]))
# The slotted FEM settings' chains (cord_model "chain", tasks/slotted_post_ring/fem_chain.py): base and post are URDF
# meshes; the post (a rectangle with the rounded through-slot as a hole, one extrusion) collides as exact fat convex
# prisms (slotted_post_chain.post_parts: 82 pieces, thinnest 0.86 mm); its hull would seal the slot.
# slotted_chain_fem_v2 / slotted_chain108_v2 (2026-10-12): v1 + orbits' speed clamp (speed_clamp, below) on the chain's
# bodies at 3 m/s / 120 rad/s. The acceptance round's prefix streams on hard_chain (prefix12_e6, prefix7_e2 on ac8dafe6a):
# the chain flung at 3.9-4.9 m/s, disc_left lands on the table (164-166 N), links whip within the tick and rule 6 misses
# by up to 0.03 J (uncaused jumps), as orbits' O1. The five references' maxima are 1.61 m/s and 55 rad/s. The clamp
# covers every moving body of the chain entity (links, beads, discs and the ring, as orbits' covers the handle): the
# references' maxima are taken over all of them, so no reference motion is damped.
_SLOTTED_CLAMP = dict(v_m_s=3., w_rad_s=120., rate=.5)
_SLOTTED_FEM_CHAIN = dict(_CHAIN, name="slotted_chain_fem_v2", mjcf=dict(cord=_CHAIN_GEOMS, target_ring=_CHAIN_GEOMS),
                          split={"post": "topogym.assets.geometry.slotted_post_chain:post_parts"},
                          speed_clamp=dict(_SLOTTED_CLAMP, links="cord::*"))
# slotted #108's chain file (easy / medium / hard_chain; very_easy / easier): one MJCF, the
# fixture boxes exact primitives in "<entity>::world".
# slotted_chain108_v3: the same profile; the label marks the chain file's ball-joint armature 1e-5
# (was 1e-6; slotted_chain.ARMATURE), the fix for the stress breakdowns S1 and prefix4_e20.
_SLOTTED_108 = dict(_CHAIN, name="slotted_chain108_v3", mjcf=dict(slotted_chain=_CHAIN_GEOMS),
                    speed_clamp=dict(_SLOTTED_CLAMP, links="slotted_chain::*"))
# orbits: the ring of the FEM settings' chains (a polygonal torus) collides as its 128 exact convex segments
# (orbits_chain.ring_parts; thinnest 4.1 mm), not as its hull, which would close the hole; easier's ring is declared as
# 48 convex capsule meshes (orbits/below_easy.py ring_collision_urdf), which collide as they are. The two links at the middle
# tie overlap at birth and are excluded (birth_overlap_exclude, Genesis' neutral-overlap filter). The tie (<connect>) is
# nearly hard: 1.25 ms, dampratio 1, solimp .99 .99 (the authored 5 ms tie is Genesis' soft constraint, which let it
# stretch past the 2 mm rule). The posts (8.2 mm across), the ring tube and the stopper ball keep the task's
# contact against the robot (static_contact.keep): a softer class lets a gripper sink through thin static geometry (B1).
# orbits_chain_v3 / v4 / fem_v2 (2026-10-11): v1 / v2 / fem_v1 + the speed clamp for MJCF chain links (speed_clamp,
# world.py MujocoWorld._clamp_apply): extra damping on a cord link only above 4 m/s or 150 rad/s, rate 0.5. The stress
# session's prefix_e1 on very_easy (the reference's first 2654 rows, then the cord flung at 8 m/s): the handle lands on
# the ground (90 N) and is spun up to 120 rad/s, twelve links gain 0.9-1.7 m/s within that tick, and the cord's energy
# balance of the tick misses by 0.046 J (uncaused jumps). At 6 m/s / 300 rad/s it still breaks (the landing whips links from 5
# to 8 m/s within one tick); at 4 / 150 and at 3 / 120 the same stream lands without an uncaused jump. The references'
# maxima are 2.16 m/s and 80 rad/s (easier), so they never reach it and replay bit-identically.
# orbits_chain_v5 / v6 / fem_v3: the clamp at slotted's 3 m/s / 120 rad/s (was 4 / 150). The easier
# acceptance round on 315526e0be broke down once (prefix7_e2: the cord flung at 5.4 m/s, the handle spinning ~124 rad/s,
# lands on a static at tick 69696; three links jump with no partner and rule 6 misses by ~4 mJ). Traced on the
# integrator's books: the static contacts did -93.93 mJ (the observer books -93.85), the cord's kinetic energy changed
# by -89.33 mJ (-88.69 without armature): the step's own energy error on a fast-spinning tree's landing, +3.9 mJ in that
# tick (+0.4 to +1.2 mJ per tick in the free flight before), plus 0.64 mJ of armature energy released into the links; no
# contact created energy. At 3 / 120 the stream is VIOLENT in two fresh processes (no uncaused jump), O1 too; the two
# references never activate the clamp (0 substeps) and replay to their recorded hashes.
def _orbits(name, rings, *, ring_parts=True):
    """The orbits chain profile of a setting with `rings` rings (rings + 1 posts). ring_parts: the rings collide as
    their mesh's exact convex segments (the FEM settings' chains); False: as the task declares them (easier since
    a loop of 48 convex capsule meshes along the centreline)."""
    keep = {**{f"post_{i+1}": "8.2 mm post" for i in range(rings+1)}, **{f"ring_{i+1}": "ring tube" for i in range(rings)},
            "stopper_ball": "stopper ball"}
    return dict(_CHAIN, name=name, birth_overlap_exclude=True, mjcf=dict(cord=_CHAIN_GEOMS),
                split={f"ring_{i+1}": "topogym.assets.geometry.orbits_chain:ring_parts" for i in range(rings)}
                if ring_parts else {},
                mjcf_equality=dict(solref=[.00125, 1.], solimp=[.99, .99, .001, .5, 2.]),
                static_contact=dict(soft=SOFT_CONTACT, keep=keep),
                speed_clamp=dict(links="cord::*", v_m_s=3., w_rad_s=120., rate=.5))


# blanket_flatten on a VBD cloth: MuJoCo steps
# the ALOHA (as claws imports it) and the table only; the cloth is Newton's VBD on the authored 51 x 41 grid
# (cloth_vbd.py: its settings there). Selected by name (SimulatorConfig.mujoco_profile); the flex (blanket_v1) stays the
# family's default until the VBD backend's acceptance.
# The step: vbd_fast.FastSolverVBD (lane S's deterministic 4-thread CPU step, bit-identical at any thread count and on
# Intel / AMD) with plastic bending (crease memory): a dihedral beyond yield_angle from its rest angle moves the rest
# angle toward it at `rate` per 2.5 ms step (8 /s), capped at max_rest. So folds hold, a turned-back flap stays open,
# and nothing unfolds by itself (lane S's study, 2026-10-11: the only setting of edge_ke 6.5e-3 / 1.41e-3, yield .3 / .6,
# rate .005 / .02 where folds hold and stay open).
# The step's coarse correction: before every 2nd iteration a Newton step on the
# translations of mesh-graph patches (radius 4 hops, 100 Gauss-Seidel sweeps), table friction as a Coulomb capacity;
# without it, table friction kept Gauss-Seidel from carrying a drag across the sheet (37 N at the pins for a 0.71 N
# load). Plastic hinges touching a pinned vertex do not yield (skip_pinned). friction_eps 1e-3 m/s: half the fold creep
# of 1e-2.
_CLOTH_VBD = dict(_CLAWS, contact=dict(_CLAWS["contact"], table_friction=.5),
                  cloth=dict(backend="vbd", edge_ke=6.5e-3, iterations=20, friction_eps=1e-3,
                             solver="topogym.world.mujoco.vbd_fast:FastSolverVBD",
                             solver_kwargs=dict(threads=4, coarse_every=2, coarse_patch_hops=4, coarse_sweeps=100,
                                                plastic=dict(yield_angle=.6, rate=.02, max_rest=2.7, skip_pinned=True))))

# The same cloth on one CUDA device (user decision 2026-10-12: the blanket references are recorded on GPU, within the
# 72 min budget): vbd_gpu.GpuDetSolverVBD, the CPU step above moved to the GPU (the same kernels, kwargs and material),
# bit-identical across fresh processes and GPU classes (sm_80 / sm_89 / sm_90 / sm_120), FMA contraction off in the
# process. Not bit-identical to the CPU step (the self-contact terms are summed in a canonical order instead of the
# BVH's), so a reference replays on the profile it was recorded on: blanket_vbd (CPU) stays for the CPU references. A
# frozen scene built on blanket_vbd runs on this step (cloth_vbd.PLACEMENT_KEYS: device, solver and solver_kwargs from
# the profile, the material from the scene); the GPU step ignores threads.
# cloth_floor: under the gripper's non-pad geoms the floor is the local cloth
# stack top, so a kinematic gripper cannot crush a stack into the table (the cloth is one-way coupled: MuJoCo never
# feels it; validator blanket easier4 directed_e0: the gripper base on the table at 227 N over an 18 mm stack).
# mujoco_substeps 4: MuJoCo at 0.625 ms, 4 steps per 2.5 ms cloth step (world.advance), as
# every other ALOHA family steps MuJoCo; at 2.5 ms MuJoCo's refsafe clamps the finger coupling's 1.25 ms time constant to
# 5 ms and the coupling returned energy. The cloth's colliders and
# pins take the fresh kinematics after the 4th step.
# two_way: the cloth's contact forces on every robot body go back to MuJoCo (cloth_coupling.py:
# linearly implicit over the cloth step, no pull, friction within 0.4 G_n, 2x actuator torque cap); the cloth floor
# stays as the second line of defence. blanket_vbd (CPU, the frozen CPU references) stays one-way.
# Adaptive iterations, in blanket_vbd_gpu and blanket_vbd_cpu_twin (blanket_vbd, the
# frozen CPU references, stays without): after the 20 iterations, blocks of 20 more (to 80) while the last iteration's
# largest vertex update exceeds 0.1 mm (vbd_fast / vbd_gpu `adaptive`). 20 iterations do not converge in a crumpled
# stack at rest and the unconverged steps ratchet membrane energy up (validator wave 2, hard4 mutate_e3 on the twin: X
# rise 0.268 J). 8 perturbed copies on the twin: breakdowns 1/8 -> 0/8, rise median 0.067 -> 0.056 J, max 0.268 -> 0.133 J,
# +12 % iterations; on routes +0.02-0.2 % iterations. It does not change the pressed-pile class (easy4 directed_e6, GPU:
# 7/8 -> 6/8 breakdowns, max 2.28 -> 0.92 J), which stays a criteria / scope question.
_ADAPTIVE = dict(tol=1e-4, block=20, max_iterations=80)
_CLOTH_VBD_GPU = dict(_CLOTH_VBD, cloth_floor=True, mujoco_substeps=4, cloth=dict(
    _CLOTH_VBD["cloth"], device="cuda", solver="topogym.world.mujoco.vbd_gpu:GpuDetSolverVBD",
    solver_kwargs=dict({k: v for k, v in _CLOTH_VBD["cloth"]["solver_kwargs"].items() if k != "threads"},
                       adaptive=_ADAPTIVE), two_way={}))

# The GPU profile's physics on the CPU step: the cloth
# floor, MuJoCo at 4 steps per cloth step and two-way coupling as blanket_vbd_gpu, the cloth on FastSolverVBD at 1 thread
# (bit-identical at any thread count; infra 10-12 on Quest: 1 thread per process and 1 process per core is 3x the
# throughput of 4 threads).
# Not bit-identical to the GPU step (the self-contact sums' order), so its runs are judged by verdict, not by hash; a GPU
# reference's rows (or a GPU checkpoint: capture() is host arrays) drive it. blanket_vbd (the frozen CPU references)
# stays as it was.
_CLOTH_VBD_CPU_TWIN = dict(_CLOTH_VBD, cloth_floor=True, mujoco_substeps=4, cloth=dict(
    _CLOTH_VBD["cloth"], two_way=_CLOTH_VBD_GPU["cloth"]["two_way"],
    solver_kwargs=dict(_CLOTH_VBD["cloth"]["solver_kwargs"], threads=1, adaptive=_ADAPTIVE)))

# Profiles chosen by name (SimulatorConfig.mujoco_profile) instead of the family's default, with the families they serve.
NAMED_PROFILES = {
    "blanket_vbd": dict(_CLOTH_VBD, name="blanket_vbd", tasks=("blanket_flatten", "blanket_fold")),
    "blanket_vbd_gpu": dict(_CLOTH_VBD_GPU, name="blanket_vbd_gpu", tasks=("blanket_flatten", "blanket_fold")),
    "blanket_vbd_cpu_twin": dict(_CLOTH_VBD_CPU_TWIN, name="blanket_vbd_cpu_twin", tasks=("blanket_flatten", "blanket_fold")),
}

PROFILES = {
    # Every Panda + Robotiq profile from 2026-10-11 on: the Robotiq's spring links at the MJCF's springref 2.62
    # (robot.SPRING_REFS, set at every load, frozen scenes included; before: about 0, the Genesis import).
    ("buzz_wire_3d", None): dict(_BUZZ, name="buzz_v2"),
    ("fiber_bundling", None): dict(_FIBER, name="fiber_v5"),
    ("cast_coil", "cast_coil_v2_very_easy"): _CAST_COIL_V2,
    ("cast_coil", "cast_coil_v2_easier"): _CAST_COIL_V2,
    ("cast_coil", "cast_coil_v2_easy"): _CAST_COIL_V2,
    ("cast_coil", "cast_coil_v2_design"): _CAST_COIL_V2,
    ("lock_picking", "lock_picking_very_easy"): _LOCK,
    ("lock_picking", "lock_picking_easier"): _LOCK,
    ("lock_picking", "lock_picking_easy"): _LOCK,
    ("lock_picking", "lock_picking_medium"): _LOCK,
    ("lock_picking", "lock_picking_hard"): _LOCK,
    **{("slotted_post_ring", v): _SLOTTED_108
       for v in ("slotted_post_ring_very_easy", "slotted_post_ring_easier", "slotted_post_ring_easy_chain",
                 "slotted_post_ring_medium_chain", "slotted_post_ring_hard_chain")},
    ("orbits", "orbits_very_easy"): _orbits("orbits_chain_v5", 0),
    ("orbits", "orbits_easier"): _orbits("orbits_chain_v6", 1, ring_parts=False),
    # The FEM settings with their cord as the chain (task parameter cord_model "chain"; the chain's joints from the FEM
    # material, slotted_post_ring/common.py and orbits/common.py): the same scene construction as the chain settings above.
    **{("slotted_post_ring", v): _SLOTTED_FEM_CHAIN
       for v in ("slotted_post_ring_easy", "slotted_post_ring_medium", "slotted_post_ring_hard",
                 "slotted_post_ring_clearance36")},
    ("orbits", "orbits_easy"): _orbits("orbits_chain_fem_v3", 2),
    ("orbits", "orbits_medium"): _orbits("orbits_chain_fem_v3", 3),
    ("orbits", "orbits_hard"): _orbits("orbits_chain_fem_v3", 4),
    ("orbits", "orbits_reference85"): _orbits("orbits_chain_fem_v3", 4),
    # very_easy .. easy, hard: the cages declare slab groups and wedges down to 0.027 mm thick (sliver hulls); they collide
    # as an exact fat re-partition (thinnest piece 1.26 mm; hard 0.32 mm, the width of an island its top rim declares).
    ("hedgehog_in_cage", None): dict(_HEDGEHOG, name="hedgehog_v4", prism_partition=dict(fatten_m=.0015)),
    # medium: the waisted cage is one non-convex solid (Genesis: SDF); MuJoCo collides its exact convex pieces (426,
    # thinnest 2.1 mm), built from the cage's own primitives (topogym/assets/geometry/hedgehog_ellipse.py).
    ("hedgehog_in_cage", "hedgehog_in_cage_medium"): dict(
        _HEDGEHOG, name="hedgehog_v3_waisted",
        split={"hedgehog_cage": "topogym.assets.geometry.hedgehog_ellipse:waisted_cage_hulls"}),
    # Bird in Cage (an experimental setting, 2026-10-11): hedgehog_v4's solver and robot, exact primitives and convex
    # pieces (topogym.assets.geometry.hedgehog_owl; no partition, split or CoACD), every task contact on the family's
    # class (1.25 ms, dampratio 2, solimp .9 .99) with wood's friction 0.4. The owl (a free body with its wings on a hinge;
    # the hinge's dry friction is the owl's own, owl.xml) and the rod are MJCF entities; the cage a fixed URDF whose bars
    # and rung are capsules (`capsules`: a cylinder and its two end spheres): on MuJoCo 3.14 a mesh against a cylinder
    # returned a few normals pointing into the mesh (some 0.3 mm deep), and a capsule 11.6 mm into a cylinder crashed the
    # process (geometry docstring, cage_parts).
    ("hedgehog_in_cage", "hedgehog_in_cage_owl"): dict(
        _HEDGEHOG, name="hedgehog_owl_v1", capsules=True,
        mjcf={entity: dict(contact=dict(solref=_RIGID["contact"]["solref"], solimp=_RIGID["contact"]["solimp"], condim=3,
                                        friction=[.4, _RIGID["contact"]["torsional_friction"],
                                                  _RIGID["contact"]["rolling_friction"]]))
              for entity in ("owl", "rod")}),
    # cast_cage: Hanayama's Cast Cage at scale 4 and a push rod (tasks/hedgehog_in_cage/cast_cage.py),
    # on the family's contacts and robot (_HEDGEHOG).
    #  - capsules: the rod's shaft is declared as a cylinder with a sphere of its radius on each end cap and collides as
    #    that exact capsule (porting guide trap 7); its handle is a box.
    #  - no prism_partition: the cage's 262 and the star's 7 collision meshes are the design's exact convex pieces, each
    #    colliding as its own hull. The thinnest are 14 wall strips 1.20-1.49 mm wide at the windows' sides, thin along
    #    the wall (77-138 mm tall, 11 mm through the wall); the family's re-partitioned cages are 1.26 mm (hard 0.32 mm)
    #    at their thinnest. The family entry's prism_partition (fatten_m 1.5 mm) leaves these pieces as they are: 262
    #    pieces in and out, no re-cut, thinnest 1.20 mm. A normal could flip only past half a strip's width (0.6 mm) along
    #    the wall; the -i01 instance's 10 s nop under the stress observer is CLEAN, its deepest task contact 0.043 mm over
    #    17,600 substeps (the star at rest on the cage).
    ("hedgehog_in_cage", "hedgehog_in_cage_cast_cage"): dict(_HEDGEHOG, name="hedgehog_cast_cage_v1", capsules=True),
    ("cast_keyhole", None): dict(_KEYHOLE, name="keyhole_v4"),
    # The double-maze setting (an experimental setting, 2026-10-12): its own key, a copy of keyhole_v4's settings (the
    # five tiers' accepted profile is left as it is). The plates are the family's slab prisms (cast_keyhole_double_maze).
    # keyhole_double_maze_v2 (2026-10-12): v1 with MuJoCo's multi-point convex collision off (collision.multiccd False).
    # The 1000-episode acceptance on f2449335c broke down once (cem_e157): the dropped silver plate tumbling over the
    # table's -y edge got wrong contacts from multiccd's extra points (the table box against convex prisms of the plate at
    # the box's edge): a 5.8 mm piece read 1.44 mm deep where the exact penetration was 0.33 mm, a 0.59 mm piece 1.56 mm
    # for 0.003 mm, the normals 84-90 deg off (sideways instead of up); 6-9.8 kN along them for 17 substeps created 1.44 J.
    # The same states without multiccd give the exact depths and normals. 40 states within 0.5 mm / 0.5 deg of the event:
    # 9 of 40 created over 0.25 J with multiccd, 0 of 40 without (40 of 40 had wrong contacts, then 0). The round's 148
    # table-related streams: 1 breakdown and 63 wrong contacts with multiccd, 0 and 0 without; the most a contact pair
    # created 1.44 J -> 0.19 J (a finger on the silver, 0.19 J with multiccd too). Each convex piece then touches with one
    # point (a plate lying flat: 57 table points instead of 216, resting just as still); the reference's contacts and
    # tracking are unchanged.
    ("cast_keyhole", "cast_keyhole_double_maze"): dict(_KEYHOLE, name="keyhole_double_maze_v2",
                                                       collision=dict(_COLLISION, multiccd=False)),
    ("blanket_flatten", None): dict(_CLOTH, name="blanket_v1"),
    ("interlocked_claws", None): dict(_CLAWS, name="claws_v6", weld=_CLAWS_WELD),
    ("interlocked_claws", "interlocked_claws_hard"): dict(_CLAWS, name="claws_v6_coacd002", coacd=_COACD_STRICT, memory="1G",
                                                          weld=_CLAWS_WELD),
    # Hanayama Cast ABC (an experimental setting, 2026-10-11): claws_v6, each piece colliding as its exact convex
    # decomposition (topogym.assets.geometry.cast_abc), so the plate's windows and notches and the C's gap stay open.
    ("interlocked_claws", "interlocked_claws_cast_abc"): dict(
        _CLAWS, name="claws_v6_cast_abc", weld=_CLAWS_WELD,
        split={"ab_plate": "topogym.assets.geometry.cast_abc:plate_parts",
               "c_ring": "topogym.assets.geometry.cast_abc:ring_parts"}),
    # Hanayama Cast Rotor (an experimental setting, 2026-10-11): claws_v6, each piece colliding as its exact convex
    # decomposition (topogym.assets.geometry.claws_rotor), so the windows, slits, groove and notches stay open.
    ("interlocked_claws", "interlocked_claws_rotor"): dict(
        _CLAWS, name="claws_v6_rotor", weld=_CLAWS_WELD,
        split={"piece_0": "topogym.assets.geometry.claws_rotor:rotor_parts",
               "piece_1": "topogym.assets.geometry.claws_rotor:rotor_parts"}),
    # bed_bolt (2026-10-16): claws_v6, the values the design's sandbox ran the ALOHA with (runs/bed_bolt_design_20261012,
    # phys3.py); the bolt's and key's own contact parameters are in their MJCF (topogym.assets.geometry.bed_bolt).
    # bed_bolt_v2 (2026-10-17, the threaded hole): the bolt entity's one static body, the thread's stop plate (thr_stop),
    # keeps its own nearly hard contact and wins (static_contact.keep): at the soft class a 200 N push on a loose bolt
    # pressed its tip 3-5 mm into the plate (into a 1 mm plate: popped, 1.6 J created). No other body changes.
    ("bed_bolt", None): dict(_CLAWS, name="bed_bolt_v2", weld=_CLAWS_WELD,
                             static_contact=dict(_CLAWS["static_contact"],
                                                 keep={"bolt": "the thread's stop plate: a loose bolt pushed onto it"})),
    # clock_assembling (2026-10-16): claws_v6, the values the design's demo ran the ALOHA with
    # (runs/watch_assembly_design_20261014, README_v3.md 7.3); the movement's own contact parameters, joint limits and
    # welds are in its MJCF (topogym.assets.geometry.clock_gear_train).
    ("clock_assembling", None): dict(_CLAWS, name="clock_v1", weld=_CLAWS_WELD),
    # battery_replacement (2026-10-16): claws_v6 (the values the design's sandbox ran the ALOHA with:
    # runs/controller_battery_design_20261014, scripts/d3_phys.py). MuJoCo's multi-point convex collision stays on: with it
    # off a cell lying on a box (the recycling box's floor) touches at one point and rocks without end (0.3 rad/s at rest).
    # The parts' own contact parameters are in their MJCF (topogym.assets.geometry.battery_replacement).
    ("battery_replacement", None): dict(_CLAWS, name="battery_replacement_v1", weld=_CLAWS_WELD),
    # Hanayama Cast Trinity (an experimental setting, 2026-10-12): claws_v6, each of the three pieces colliding as its
    # exact convex decomposition (topogym.assets.geometry.cast_trinity), so the windows, slots and head gaps stay open.
    # caliper_sorting (2026-10-16): claws_v6's contacts, welds and robot; the caliper's slider joint gets the design's
    # armature and a stiff friction constraint (runs/caliper_sorting_design_20261014, README_v3.md section 2: below its
    # 1.5 N breakaway it still creeps 0.03 mm in 5 s at 0.05 N; a stiffer one broke the two-hand control), and the inside
    # jaws' 0.3 mm knife edges are exact capsules.
    # caliper_sorting_v2 (2026-10-17): the locking screw's knob joint caliper::screw holds its friction loss (0.02 N m)
    # and its stops nearly hard, as bed_bolt's (docs/rcb/MUJOCO_CORE.md, task mechanisms: at MuJoCo's defaults a joint's
    # friction loss creeps under a load below it and a stop gives), with a small armature for the 2 g knob.
    # caliper_sorting_v3 (2026-10-18, the caliper fixed on its stand): the beam on its joint caliper::mount, held by its
    # limits within 1 um (topogym.assets.geometry.caliper_sorting.build_caliper), a hard limit as the knob's.
    ("caliper_sorting", None): dict(
        _CLAWS, name="caliper_sorting_v3", weld=_CLAWS_WELD, capsules=True,
        joints=[dict(match="caliper::mount", armature=1e-5, solreflimit=[.00125, 2.], solimplimit=[.99, .999, .001, .5, 2.]),
                dict(match="caliper::slider", armature=.001, solreffriction=[.00125, 1.],
                     solimpfriction=[.99, .999, .001, .5, 2.]),
                dict(match="caliper::screw", armature=1e-5, solreffriction=[.00125, 1.],
                     solimpfriction=[.99, .999, .001, .5, 2.], solreflimit=[.00125, 2.],
                     solimplimit=[.99, .999, .001, .5, 2.])]),
    ("interlocked_claws", "interlocked_claws_cast_trinity"): dict(
        _CLAWS, name="claws_v6_cast_trinity", weld=_CLAWS_WELD,
        split={"trinity_t": "topogym.assets.geometry.cast_trinity:t_parts",
               "trinity_h": "topogym.assets.geometry.cast_trinity:h_parts",
               "trinity_b": "topogym.assets.geometry.cast_trinity:b_parts"}),
    # knot_chain_v5 (2026-10-10): the rod's rotational inertia floor 12 -> 24 k dt^2 (inertia_rule), its damping
    # 0.15 I / dt as v4 (damping_rule). The GPU search found a fourth blow-up (K3, knot easy, e99), on v4 with the exact
    # constraint diagonal: the arms fling the rope; in free flight its end whip-cracks (link 0: 7.6 -> 18 m/s in 4 ms)
    # and curls past the 0.9 rad bend stops to 1.2 rad; links 13-25 then grow exponentially (17 -> 700 m/s in 6 MuJoCo
    # steps). Robot contacts and solver iterations play no part (robot-robot pairs off and 5x iterations: the same
    # fault). From the state 10 rows before it: inertia_rule 12 blows up, 15, 16, 18 and 27 hold; damping off (the
    # headroom formula's 12) blows up earlier, so the damping helps and the floor is the lever. v5 holds at 1.5x the
    # effective stiffness (rule 16). Earlier: v4 (12, damping 0.3 -> 0.15) for e2, the rope pulled taut between the
    # hands; v3 (8) for e0 and e1, a pinched tail and mid-rope sliding; v2 (4) the first. Cost: bending waves run 6.4 %
    # slower at 0.2 m wavelength and 20 % at 0.1 m (v4: 3.4 % / 12 %, v2: 1.2 % / 4.5 %); the floor is 25x the links'
    # physical rotational inertia; the rod damping is 3.6 k dt.
    # knot_chain_v6 (2026-10-11): v5 + the speed clamp (rope_chain.py, RopeChain._clamp) at 2 m/s / 150 rad/s, rate 0.5.
    # The GPU search's two v5 blockers (easier e79: a bad qacc at 2560 m/s; easy top029: an uncaused 14 -> 15.4 m/s jump)
    # are a whipped free tail spinning at hundreds of rad/s, where the node connect rows pump energy into it (their net
    # work grows ~ w^4). The thresholds are 3x the references' maxima (0.63 m/s, 48 rad/s); no reference reaches them,
    # so the references replay bit-identically.
    # knot_chain_v7 (2026-10-11): v6 + the tension guard (rope_chain.py, RopeChain._tension_guard) at 30 E I / l^2
    # (14.9 N; t_ref the same, decay 0.99 per step). The GPU search's knot easier top054_b1_e229 and top004_b0_e84: the
    # left wrist and gripper base pinch the rope; the chain's tension reaches ~38 N, past the explicit rod's critical
    # ~30 N, and a mode grows ~3x per MuJoCo step to a bad qacc before the speed clamp sees a fast link. The guard
    # raises the pinched links' rotational inertia (up to 2.5x on e229), keeping their angular momentum. The
    # references peak at 1.1 N (2.2 E I / l^2): they replay bit-identically.
    ("knot_untying", None): dict(
        _ROPE, name="knot_chain_v7", rope=dict(_ROPE["rope"], inertia_rule=24., damping_rule=.15,
                                               speed_clamp=dict(v_m_s=2., w_rad_s=150., rate=.5),
                                               tension_guard=dict(t_on=30., t_ref=30., decay=.99))),
    # rope_threading: the released rope is 100x stiffer (E 5e6, rest = its initial shape) and touches only the fingers;
    # links of 5.5 mm (its mesh spacing); the rings collide as their exact convex tube segments (64 per closed ring, 32 for
    # the open cradle; rope_threading.ring_hulls), the stands and bases as their own convex meshes. The assisted grasp is
    # contact_weld_v1 on the rope link nearest the TCP (the instances' targets: every link), so a pinch must close on
    # the rope; the released ipc_soft_band_v1 bound it with the pads up to 6 mm away. The references were recorded live on
    # MuJoCo with the pinch closed.
    # rope_chain_v2 (2026-10-11): v1 + the speed clamp as knot_chain_v6, at 6 m/s / 300 rad/s (the references' maxima:
    # 2.64 m/s, 113 rad/s); no reference reaches them.
    # rope_chain_v3 (2026-10-11): v2 + the tension guard as knot_chain_v7, at 30 E I / l^2 of this rope (~1000 N: never
    # reached by a weld that slips at 8 N; a guard for any future pinch).
    # The rings, their stands and bases (entities R1.., R1_stand.., R1_base..; a tier has two or three) keep the task's
    # nearly hard contact against the robot (static_contact.keep, 2026-10-11): soft, an arm driven into a ring tube
    # through several soft rows created energy (gpusearch kt1_3 e116: the wrist on R3, +2.2 J; with the soft class's
    # constant impedance still +2.5 J); kept, the five rope streams of that search are VIOLENT or CLEAN. The rope meets
    # them with the same class it had (the rope's own parameters are the task's), so its contacts are unchanged.
    ("rope_threading", None): dict(
        _ROPE, name="rope_chain_v3", rope=dict(_ROPE["rope"], link_m=.0055, robot_contact="fingers",
                                               speed_clamp=dict(v_m_s=6., w_rad_s=300., rate=.5),
                                               tension_guard=dict(t_on=30., t_ref=30., decay=.99)),
        static_contact=dict(_ROPE["static_contact"], keep={"R[0-9]*": "thin static rings: an arm driven into a tube "
                                                                      "through soft rows created energy (e116)"}),
        split={ring: "topogym.assets.geometry.rope_threading:ring_hulls" for ring in ("R1", "R2", "R3")}),
    # The released tiers themselves (variation ids, geometry, goals and rules unchanged), keyed per tier so that later
    # designs under new variation ids can take their own profile.
    **{("horseshoe_ring", f"horseshoe_ring_{tier}"): dict(_HORSESHOE, name="horseshoe_v1")
       for tier in ("very_easy", "easier", "easy", "medium", "hard")},
    **{("cast_coil", f"cast_coil_{tier}"): _COIL_LEGACY for tier in ("very_easy", "easier", "easy", "medium", "hard")},
}


def profile_for(task_id: str, variation_id: str, name: str | None = None) -> dict:
    """The family's profile (a variation-specific entry first), or the named one (NAMED_PROFILES). No entry: the family
    is not ported to MuJoCo."""
    if name is not None:
        profile = NAMED_PROFILES.get(name)
        if profile is None or task_id not in profile["tasks"]:
            raise ValueError(f"MuJoCo profile {name!r} does not serve {task_id}")
        return deepcopy(profile)
    if task_id == "rope_threading" and (variation_id or "").startswith("rope_threading_rd_"):
        # the redesigned tiers (bulkhead lacing): the family's profile with the tier's rings split into their tube
        # segments and the family's ring rule (static_contact.keep) on the tier's own ring entities
        from topogym.tasks.rope_threading.redesign.registration import profile_entry
        return deepcopy(profile_entry(variation_id[len("rope_threading_rd_"):], PROFILES[(task_id, None)]))
    for key in ((task_id, variation_id), (task_id, None)):
        if key in PROFILES:
            return deepcopy(PROFILES[key])
    raise NotImplementedError(f"{task_id} ({variation_id}) has no MuJoCo profile: the family is not ported")
