"""blanket_flatten hard, the Genesis tier ported to blanket_vbd_gpu (V3, 2026-10-17): its route (routes.route_for
"hard5").

The released route's page turns (routes.HARD stage A) take the twist out of the band on VBD too, but the released stage
B and the Genesis PBD creep that finished the route do not carry over (blanket_flatten.md, "hard: blocked"). The new
route runs as a sequence of parts (robot_oracle.run_sequence); after its declared stages a part may hand over to a lab
program on the redesigned tiers' planner hook (redesign_routes.LabProgramPlanner), which measures the live cloth and
lays it out with the redesign's measured fix-ups (re-centring on the flat layout, the family's fold layouts, the
finishing relays). The flat layout is the redesigned tiers' (centre (-0.1357, 0.05)): the arm bases' covers leave the
sheet a window only ~5 cm wider per side than its 0.60 m length, so it must end close to axis-aligned at that x, and y
0.05 keeps its far edge inside the arms' top-down reach.
"""
from __future__ import annotations

HARD_TIER = dict(name="hard_port", coverage_target=.95, strain_guard=1.5, obstacles=())


def recenter_only(sim, tier, rounds=2):
    from .redesign.reverse import recenter
    return recenter(sim, rounds=rounds)


def recenter_fold_relay(sim, tier):
    from .redesign.reverse import recenter, fold_layouts, finish_relay
    return recenter(sim, rounds=2) + fold_layouts(sim, max_stages=3) + finish_relay(sim)


PROGRAMS = dict(recenter2=recenter_only, recenter_fold_relay=recenter_fold_relay)


def planner(name):
    from .redesign_routes import LabProgramPlanner
    return LabProgramPlanner(PROGRAMS[name], lambda p: (HARD_TIER,), name=f"hard_port_{name}")


# ------------------------------------------------------------------------------------------------ the route (hard5)
# Found on the branching lab (the work notes of the V3 blanket hard port, lab L6 g2) on the -i01 birth
# c7e8fe0051099141 (blanket_vbd_gpu): three parts, each run from the state the previous one left
# (robot_oracle.run_sequence).
#  1. The released route's page turns (routes.HARD stage A), fast approaches: the twist is out; the sheet lies on its
#     far long edge's strip, tucked under along the whole length, with a standing pleat across the width at v ~ -0.06
#     (4-7 cm tall) and the near-right corner folded (coverage 0.571 after a 3 s settle).
#  2. Hang and lay (as medium's): both hands pinch the near side (left (-0.20, -0.22), right (0.05, -0.17): low,
#     single-layer, exposed), lift 12 cm, rise to 20 cm moving 3 cm toward the robot so the near part hangs and the
#     pleat opens, hold 2 s, lay it 9 cm toward the robot 3 cm above the table; then the measured re-centring
#     (redesign reverse.recenter, two rounds: the near pair, then the far pair, dragged to the flat layout's places):
#     the sheet lies flat and close to axis-aligned (coverage 0.905), only the far-left strip still tucked under.
#  3. The measured slide toward the robot (robot_oracle.slide_stage, far side to y 0.16: 13 cm): the tucked strip's
#     bottom layer stays on the table while the sheet above it moves away from the crease, so the strip rolls out
#     (a strip's drop-then-lay at its crease, or a lift-and-drag, did not open it: the plastic crease holds it shut);
#     then the measured re-centring, fold layouts and finishing relays while the goal is not met (it is met at the
#     first settle: coverage 0.958, 28 mm, 2813 rows).
# A fold layout on the far-left corner before the slide turned the sheet ~20 deg into the left arm base's cover (L2 h1).
HANG = dict(grasp_uv=dict(left=(-.20, -.22), right=(.05, -.17)), segments=(
    dict(name="lift", duration_s=1.5, by=dict(left=(0., 0., .12), right=(0., 0., .12))),
    dict(name="rise", duration_s=2., by=dict(left=(0., -.03, .20), right=(0., -.03, .20))),
    dict(name="hold", duration_s=2., by=dict(left=(0., -.03, .20), right=(0., -.03, .20))),
    dict(name="lay", duration_s=4., by=dict(left=(0., -.09, .03), right=(0., -.09, .03)))))
FOOTING = dict(half_span_m=.010, radius_m=.006, tolerance_m=.003)


def route():
    """routes.route_for("hard5"): a fresh sequence (its planners hold one episode's state)."""
    from .routes import HARD
    common = dict(fast=True, footing=dict(FOOTING), planner_rounds=80, planner_settle_s=3., settle_s=20.)
    return dict(max_strain_ratio=1.5, sequence=(
        dict(common, stages=(HARD["stages"][0],), settle_s=3.),
        dict(common, stages=(HANG,), planner=planner("recenter2"), settle_s=.1),
        dict(common, stages=(dict(measured="slide", far_y_m=.16),), planner=planner("recenter_fold_relay"),
             base_clearance=dict(HARD["base_clearance"])),
    ))
