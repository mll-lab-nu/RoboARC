"""The buzz_wire_3d rd oracle's held segments per tier (moved out of topogym/tasks/buzz_wire_3d/redesign/tiers.py:
the simulator package carries no solution hints). Used by run_oracle.py and chain.py."""

# The oracle's held segments per tier (from the kinematic analysis, tools/redesign/buzz_wire_3d/segplan.py):
# the grasp of each segment and where it ends (a park on a rest sleeve, or the drop on the insulated finish post).
ORACLE_PLANS = {
    "very_easy": dict(pick="top:180:0.9:-10:-2", place="top:180:0.9:-10:-2", segments=[
        dict(grasp="top:180:0.9:-10:-2", s1=0.2676, end="park"),
        dict(grasp="side:180:0.55", s1=1.01, end="drop")]),
    # from the all-pairs chains (tools/redesign/buzz_wire_3d/chain.py; the cloud host out/chains/<tier>.json)
    "easier": dict(pick="top:180:0.9:-10:-2", place="top:180:0.9:-10:-2", segments=[
        dict(grasp="top:180:0.9:-10:-2", s1=0.6212, end="park"),
        dict(grasp="sideb:0:0.55", s1=0.9438, end="drop")]),
    "easy": dict(pick="top:180:0.9:-10:-2", place="top:180:0.9:-10:-2", segments=[
        dict(grasp="top:180:0.9:-10:-2", s1=0.2319, end="park"),
        dict(grasp="side:180:0.8", s1=0.7418, end="park"),
        dict(grasp="sideb:0:0.55", s1=0.9577, end="park"),
        dict(grasp="sideb:0:0.8", s1=1.2808, end="drop")]),
}
