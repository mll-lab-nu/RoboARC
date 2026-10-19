"""The redesigned blanket_flatten tiers' start-state programs (inverse generation: fold programs applied to the flat
sheet) and the tier records built from them. Authoring only (FlexSim labs, planning, scripts); the simulator
package keeps only the geometry helpers (topogym.tasks.blanket_flatten.redesign.tiers). Frozen instances carry
their start state as arrays."""
from __future__ import annotations

import numpy as np

from topogym.planning.blanket_flatten.redesign.programs import page_turn_path
from topogym.tasks.blanket_flatten.redesign.tiers import (CENTRE_SHIFT, CX, CY, FENCE, GRIP_IN, W, _mat_side,  # noqa: F401
                                                         corner, fold, fold_world)


def turn(hands_points, crease_a, crease_b, duration_s=4., **kw):
    return dict(name="turn", duration_s=duration_s,
                hands={h: dict(path=page_turn_path(p, crease_a, crease_b, **kw)) for h, p in hands_points.items()})


VERY_EASY = dict(
    name="very_easy", title="pinwheel of stacked corners (two ordered pairs)", coverage_target=.95, strain_guard=1.5,
    obstacles=(),
    generate=[
        fold("fold_NR", (.06, -.24), (.3, 0.), (.3, -.24)),
        fold("fold_FR_over_NR", (0., .24), (.3, -.06), (.3, .24)),
        fold("fold_FL", (-.06, .24), (-.3, 0.), (-.3, .24)),
        fold("fold_NL_over_FL", (0., -.24), (-.3, .06), (-.3, -.24), settle_s=3.),
    ],
)

# ----------------------------------------------------------------------------------------------- tier: easier
EASIER = dict(
    name="easier", title="half fold over a tucked corner, top corner folded", coverage_target=.95, strain_guard=1.5,
    obstacles=(),
    tuck=dict(a=(-.15, -.24), b=(-.3, -.09), side=(-.3, -.24)),
    generate=[
        fold("half_fold", (-.32, 0.), (.32, 0.), (0., .24), duration_s=4.),
        fold("fold_top_corner", (.14, -.24), (.3, -.08), (.3, -.24), material=lambda uv: uv[:, 1] > .02, settle_s=3.),
    ],
)

# ----------------------------------------------------------------------------------------------- tier: easy
EASY = dict(
    name="easy", title="long-axis half fold (relay across the centre), top corner folded", coverage_target=.95,
    strain_guard=1.5, obstacles=(),
    generate=[
        fold("half_fold_left_over_right", (0., -.25), (0., .25), (-.3, 0.), duration_s=4.),
        fold("fold_top_corner", (.16, .24), (.3, .10), (.3, .24), material=lambda uv: uv[:, 0] < -.02, settle_s=3.),
    ],
)

# ----------------------------------------------------------------------------------------------- tier: medium
MEDIUM = dict(
    name="medium", title="long-axis letter fold with an inner corner (two ordered relays)", coverage_target=.95,
    strain_guard=1.5, obstacles=(),
    generate=[
        fold("fold_NR_corner", (.16, -.24), (.3, -.10), (.3, -.24)),
        fold("right_third_over_middle", (.10, -.25), (.10, .25), (.3, 0.), duration_s=3.5),
        fold("left_third_over_top", (-.10, -.25), (-.10, .25), (-.3, 0.), duration_s=3.5, settle_s=3.),
    ],
)

# ----------------------------------------------------------------------------------------------- tier: hard
HARD = dict(
    name="hard", title="folded in four over a tucked corner, against a fence", coverage_target=.95, strain_guard=1.5,
    obstacles=(FENCE,),
    tuck=dict(a=(.18, -.24), b=(.3, -.12), side=(.3, -.24)),
    generate=[
        fold("half_fold_left_over_right", (0., -.25), (0., .25), (-.3, 0.), duration_s=4.),
        fold("width_fold_far_over_near", (-.32, 0.), (.32, 0.), (.15, .24), duration_s=4., settle_s=3.),
    ],
)
HARD_SCROLL = dict(
    name="hard_scroll", title="double scroll against a fence (v1b, not unwound by patch grasps)", coverage_target=.97,
    strain_guard=1.5, obstacles=(FENCE,),
    rolls=(dict(edge="far", rolled_m=.17, r0=.015, spacing=.0105), dict(edge="near", rolled_m=.17, r0=.015, spacing=.0105)),
    generate=[],
)

# ----------------------------------------------------------------------------------------------- v2 ladder (04:55)
X_FOLDS = [
    fold("big_NR_corner", (-.10, -.24), (.30, .16), (.30, -.24)),
    fold("big_NL_corner_over", (.10, -.24), (-.30, .16), (-.30, -.24), settle_s=2.),
]
EASY_X = dict(
    name="easy", title="crossed big corners (two reach-forced relays)", coverage_target=.95, strain_guard=1.5,
    obstacles=(), generate=list(X_FOLDS[:1]) + [dict(X_FOLDS[1], fold=dict(X_FOLDS[1]["fold"], settle_s=3.))],
)
MEDIUM_X = dict(
    name="medium", title="crossed big corners under two far-corner flaps", coverage_target=.95, strain_guard=1.5,
    obstacles=(), generate=X_FOLDS + [
        fold("far_right_corner", (.12, .24), (.30, .06), (.30, .24)),
        fold("far_left_corner", (-.12, .24), (-.30, .06), (-.30, .24), settle_s=3.)],
)
HARD_X = dict(
    name="hard", title="crossed big corners under a far-half fold, against a fence",
    coverage_target=.95, strain_guard=1.5, obstacles=(FENCE,),
    generate=X_FOLDS + [fold("far_half_over", (-.32, .015), (.32, .015), (0., .24), duration_s=4., settle_s=3.)],
)

TIERS = dict(very_easy=VERY_EASY, easier=EASIER, easy=EASY_X, medium=MEDIUM_X, hard=HARD_X,
             easy_longaxis=EASY, medium_letter=MEDIUM, hard_quarter=HARD, hard_scroll=HARD_SCROLL)

# ----------------------------------------------------------------------------------------------- v3 ladder
EASIER3 = dict(
    name="easier3", title="half fold over a tucked corner (legs 0.21), top corner folded", coverage_target=.95,
    strain_guard=1.5, obstacles=(),
    tuck=dict(a=(-.09, -.24), b=(-.3, -.03), side=(-.3, -.24)),
    generate=[
        fold("half_fold", (-.32, 0.), (.32, 0.), (0., .24), duration_s=4.),
        fold("fold_top_corner", (.14, -.24), (.3, -.08), (.3, -.24), material=lambda uv: uv[:, 1] > .02, settle_s=3.),
    ],
)
EASY3 = dict(
    name="easy3", title="half fold over two tucked corners (legs 0.20), both top corners folded", coverage_target=.95,
    strain_guard=1.5, obstacles=(),
    tucks=(dict(a=(-.10, -.24), b=(-.3, -.04), side=(-.3, -.24)), dict(a=(.10, -.24), b=(.3, -.04), side=(.3, -.24))),
    generate=[
        fold("half_fold", (-.32, 0.), (.32, 0.), (0., .24), duration_s=4.),
        fold("fold_top_corner_R", (.14, -.24), (.3, -.08), (.3, -.24), material=lambda uv: uv[:, 1] > .02),
        fold("fold_top_corner_L", (-.14, -.24), (-.3, -.08), (-.3, -.24), material=lambda uv: uv[:, 1] > .02,
             settle_s=3.),
    ],
)
MEDIUM3 = dict(
    name="medium3", title="far corners sandwiched inside a half fold over two tucked corners", coverage_target=.95,
    strain_guard=1.5, obstacles=(),
    tucks=(dict(a=(-.12, -.24), b=(-.3, -.06), side=(-.3, -.24)), dict(a=(.12, -.24), b=(.3, -.06), side=(.3, -.24))),
    generate=[
        fold("far_corner_L", (-.14, .24), (-.3, .08), (-.3, .24)),
        fold("far_corner_R", (.14, .24), (.3, .08), (.3, .24)),
        fold("half_fold", (-.32, 0.), (.32, 0.), (0., .24), duration_s=4., settle_s=3.),
    ],
)
TIERS.update(easier3=EASIER3, easy3=EASY3, medium3=MEDIUM3)
VERY_EASY3 = dict(
    name="very_easy3", title="ordered corner pair + corner flap + tucked corner", coverage_target=.95, strain_guard=1.5,
    obstacles=(),
    tuck=dict(a=(.09, -.24), b=(.3, -.03), side=(.3, -.24)),
    generate=[
        fold("fold_FR", (.06, .24), (.3, 0.), (.3, .24)),
        fold("fold_FL", (-.06, .24), (-.3, 0.), (-.3, .24)),
        fold("fold_NL_over_FL", (0., -.24), (-.3, .06), (-.3, -.24), settle_s=3.),
    ],
)
TIERS.update(very_easy3=VERY_EASY3)
TUCK21 = dict(name="tuck21", title="near-left corner tucked under (legs 0.21), nothing else", coverage_target=.95,
              strain_guard=1.5, obstacles=(), tuck=dict(a=(-.09, -.24), b=(-.3, -.03), side=(-.3, -.24)), generate=[])
TIERS.update(tuck21=TUCK21)
for _k in [f'tuck21_l{a}d{b}' for a in (12, 16, 20) for b in (12, 17, 22)]:
    TIERS[_k] = TUCK21
def double_corner(tag, cu, cv, small=.12, big=.24):
    su, sv = np.sign(cu), np.sign(cv)
    return [
        fold(f"small_{tag}", (cu - su * small, cv), (cu, cv - sv * small), (cu, cv)),
        fold(f"big_{tag}", (cu - su * big, cv), (cu, cv - sv * big), (cu, cv), settle_s=2.),
    ]


VERY_EASY4 = dict(
    name="very_easy4", title="ordered corner pair + far-right flap + near-right double corner fold", coverage_target=.95,
    strain_guard=1.5, obstacles=(),
    generate=double_corner("NR", .3, -.24) + [
        fold("fold_FR", (.06, .24), (.3, 0.), (.3, .24)),
        fold("fold_FL", (-.06, .24), (-.3, 0.), (-.3, .24)),
        fold("fold_NL_over_FL", (0., -.24), (-.3, .06), (-.3, -.24), settle_s=3.),
    ],
)
TIERS.update(very_easy4=VERY_EASY4)
for _k in ['tuck21dx_l14d17', 'tuck21dx_l18d17', 'tuck21dx_l24d17', 'tuck21dx_l18d24', 'tuck21dx_l24d24',
           'tuck21dx_pin_l14', 'tuck21dx_pin_l20']:
    TIERS[_k] = TUCK21
_HALF_FOLD = fold("half_fold", (-.32, 0.), (.32, 0.), (0., .24), duration_s=4.)
_TOP_R = fold("fold_top_corner_R", (.14, -.24), (.3, -.08), (.3, -.24), material=lambda uv: uv[:, 1] > .02)
_TOP_L = fold("fold_top_corner_L", (-.14, -.24), (-.3, -.08), (-.3, -.24), material=lambda uv: uv[:, 1] > .02,
              settle_s=3.)
EASIER4 = dict(
    name="easier4", title="half fold over a near-left double corner fold, top corner folded", coverage_target=.95,
    strain_guard=1.5, obstacles=(),
    generate=double_corner("NL", -.3, -.24, small=.10, big=.20) + [_HALF_FOLD, dict(_TOP_R, fold=dict(_TOP_R["fold"], settle_s=3.))],
)
EASY4 = dict(
    name="easy4", title="half fold over two near double corner folds, both top corners folded", coverage_target=.95,
    strain_guard=1.5, obstacles=(),
    generate=double_corner("NL", -.3, -.24, small=.10, big=.20) + double_corner("NR", .3, -.24, small=.10, big=.20)
    + [_HALF_FOLD, _TOP_R, _TOP_L],
)
MEDIUM4 = dict(
    name="medium4", title="far corners sandwiched in a half fold over two near double corner folds", coverage_target=.95,
    strain_guard=1.5, obstacles=(),
    generate=double_corner("NL", -.3, -.24, small=.10, big=.20) + double_corner("NR", .3, -.24, small=.10, big=.20) + [
        fold("far_corner_L", (-.14, .24), (-.3, .08), (-.3, .24)),
        fold("far_corner_R", (.14, .24), (.3, .08), (.3, .24)),
        dict(_HALF_FOLD, fold=dict(_HALF_FOLD["fold"], settle_s=3.))],
)
TIERS.update(easier4=EASIER4, easy4=EASY4, medium4=MEDIUM4)
HARD4 = dict(
    name="hard4", title="four double corner folds, the far two sandwiched in a half fold, against a fence",
    coverage_target=.95, strain_guard=1.5, obstacles=(FENCE,),
    generate=double_corner("NL", -.3, -.24, small=.10, big=.20) + double_corner("NR", .3, -.24, small=.10, big=.20)
    + double_corner("FL", -.3, .24, small=.08, big=.16) + double_corner("FR", .3, .24, small=.08, big=.16)
    + [dict(_HALF_FOLD, fold=dict(_HALF_FOLD["fold"], settle_s=3.))],
)
TIERS.update(hard4=HARD4)
TIERS.update(easy4b=EASY4, medium4b=MEDIUM4)
TIERS.update(hard4b=HARD4)
TIERS.update(very_easy4c=VERY_EASY4, very_easy4d=VERY_EASY4)
TIERS.update(easy4c=EASY4, easy4e=EASY4, easy4f=EASY4, easy4g=EASY4, medium4c=MEDIUM4, easy4h=EASY4, medium4d=MEDIUM4, medium4e=MEDIUM4, medium4f=MEDIUM4)
TIERS.update(very_easy4c_fin=VERY_EASY4, easier4_fin=EASIER4, easy4f_fin=EASY4, hard4b_fin=HARD4, medium4e_fin=MEDIUM4)
TIERS.update(very_easy4c_fin2=VERY_EASY4, easier4_fin2=EASIER4, easy4f_fin2=EASY4, hard4b_fin2=HARD4, medium4e_fin2=MEDIUM4, easy4f_fin3=EASY4,
             very_easy4c_fin3=VERY_EASY4, medium4e_fin3=MEDIUM4,
             hard4b_fin4=HARD4, easy4f_fin4=EASY4, very_easy4c_fin4=VERY_EASY4, medium4e_fin4=MEDIUM4,
             very_easy4c_fin5=VERY_EASY4, medium4e_fin5=MEDIUM4, easy4f_fin5=EASY4)
