"""blanket_fold tiers (a fold-to-shape family).

Same cloth, scene and robot as blanket_flatten (the coarse MuJoCo flex of its port; FlexSim in the redesign labs).
The goal is a TARGET STATE made by inverse generation: the family's generation tool turns rigid pages of the flat
sheet (fold(): material crease a-b, the side containing side_uv) and lets it settle. The predicate is
predicate.evaluate (v3). Starts: the flat sheet, or a blanket_flatten v4 start (the robot must flatten it first).

Ladder: TIERS below, one target state per tier.
"""
from __future__ import annotations

import numpy as np

from topogym.tasks.blanket_flatten.redesign.tiers import CX, CY, corner, fold, fold_world


def w(u, v):
    return (CX + u, CY + v)


def bisector(c, t, L=.8):
    c, t = np.asarray(c, float), np.asarray(t, float)
    m, d = (c + t) / 2, (t - c) / np.linalg.norm(t - c)
    n = np.array([-d[1], d[0]])
    return tuple(m - L * n), tuple(m + L * n)


HALF = [fold("half", (-.3, 0.), (.3, 0.), (0., .24))]
HALF_CREASES = [((-.3, 0.), (.3, 0.))]

C_FL = corner(-.3, .24)
DOG_T = (C_FL[0] + .10, C_FL[1] - .14)          # the named point the far-left corner is laid on
DOG_A, DOG_B = bisector(C_FL, DOG_T)
LETTER = [fold("third_far", (-.3, .08), (.3, .08), (0., .24)),
          fold_world("third_near", w(-.3, -.08), w(.3, -.08), w(0., -.2))]
LETTER_CREASES = [((-.3, .08), (.3, .08)), ((-.3, -.08), (.3, -.08))]

TIERS = dict(
    very_easy=dict(name="very_easy", start=None,
                   target=HALF, creases=HALF_CREASES),
    easier=dict(name="easier", start=None,
                target=[fold("dog_ear", DOG_A, DOG_B, (-.3, .24))] + HALF,
                creases=[(DOG_A, DOG_B)] + HALF_CREASES, named_point=dict(DOG=DOG_T)),
    easy=dict(name="easy", 
              start=dict(state="very_easy4_v4dx", flatten="very_easy4c"), target=HALF, creases=HALF_CREASES),
    medium=dict(name="medium", start=None,
                target=LETTER, creases=LETTER_CREASES, status="two-fold technique lab"),
    hard=dict(name="hard", 
              start=dict(state="hard4_v4dx", flatten="hard4b"), target=LETTER, creases=LETTER_CREASES,
              obstacles="hard4", status="depends on medium"),
)

C_TOP_NOW = (C_FL[0], -C_FL[1])                    # the far-left corner after the half fold (flat-layout coords)
DOG2_T = (C_TOP_NOW[0] + .10, C_TOP_NOW[1] + .10)   # its named target on the top layer
DOG2_A, DOG2_B = bisector(C_TOP_NOW, DOG2_T)
TOP_LAYER = (lambda vuv: vuv[:, 1] > 0.)            # the far half's material = the top layer after the half fold
TIERS["easier2"] = dict(name="easier2", 
                        start=None,
                        target=HALF + [fold_world("dog_ear_top", w(*DOG2_A), w(*DOG2_B), w(*C_TOP_NOW), material=TOP_LAYER)],
                        creases=HALF_CREASES + [(tuple(np.array(DOG2_A) * [1, -1]), tuple(np.array(DOG2_B) * [1, -1]))])

# ---------------------------------------------------------------------------------------------- ladder v2 (18:40)
LETTER_FLIP = [fold("third_far", (-.3, .08), (.3, .08), (0., .24)),
               fold_world("stack_flip", w(-.3, -.08), w(.3, -.08), w(0., 0.))]
TIERS["easier_v1"] = dict(TIERS["easier"], name="easier_v1")
TIERS["medium_v1"] = dict(TIERS["medium"], name="medium_v1")
TIERS["hard_v1"] = dict(TIERS["hard"], name="hard_v1")
TIERS["easier"] = dict(name="easier", 
                       start=None, target=LETTER_FLIP, creases=LETTER_CREASES)
TIERS["medium"] = dict(name="medium", 
                       start=dict(state="easier4_v4dx", flatten="easier4_fin2"), target=LETTER_FLIP, creases=LETTER_CREASES)
TIERS["hard"] = dict(name="hard", 
                     start=dict(state="hard4_v4dx", flatten="hard4b_fin2"), target=LETTER_FLIP, creases=LETTER_CREASES,
                     obstacles="hard4")

TIERS["easy_v1"] = dict(TIERS["easy"], name="easy_v1")
TIERS["easy"] = dict(name="easy", 
                     start=dict(state="easier4_v4dx", flatten="easier4_fin2"), target=HALF, creases=HALF_CREASES)

from topogym.tasks.blanket_flatten.redesign.tiers import FENCE
NEAR_FENCE = dict(name="fence_near", centre_m=(CX, CY - .24 - .012 - .02), half_m=(.24, .02, .03))
TIERS["hard_v2"] = dict(TIERS["hard"], name="hard_v2")
TIERS["hard"] = dict(name="hard", 
                     start=dict(state="easier4_v4dx", flatten="easier4_fin2"), target=LETTER_FLIP, creases=LETTER_CREASES,
                     obstacles=(FENCE, NEAR_FENCE))
TIERS["hard_nearfence"] = dict(TIERS["hard"], name="hard_nearfence",
                               
                               obstacles=(NEAR_FENCE,))

TIERS["hard_v3_two_fences"] = dict(TIERS["hard"], name="hard_v3_two_fences")
TIERS["hard_flex"] = dict(TIERS["hard_nearfence"], name="hard_flex", status="flex fallback")
TIERS["hard"] = dict(TIERS["hard_v2"], name="hard", status="intended; flex-limited (corner ears), to re-run on VBD")

# ---------------------------------------------------------------------------------------------- ladder B (10-11 03:45)
C_TOP_NOW_R = (-C_TOP_NOW[0], C_TOP_NOW[1])             # the far-right corner after the half fold
DOG2R_T = (-DOG2_T[0], DOG2_T[1])
DOG2R_A, DOG2R_B = bisector(C_TOP_NOW_R, DOG2R_T)
DOG_TOP_L = fold_world("dog_ear_top", w(*DOG2_A), w(*DOG2_B), w(*C_TOP_NOW), material=TOP_LAYER)
DOG_TOP_R = fold_world("dog_ear_top_r", w(*DOG2R_A), w(*DOG2R_B), w(*C_TOP_NOW_R), material=TOP_LAYER)
_CR_L = (tuple(np.array(DOG2_A) * [1, -1]), tuple(np.array(DOG2_B) * [1, -1]))
_CR_R = (tuple(np.array(DOG2R_A) * [1, -1]), tuple(np.array(DOG2R_B) * [1, -1]))
TIERS["b_very_easy"] = dict(TIERS["very_easy"], name="b_very_easy", ladder="B")
TIERS["b_easier"] = dict(TIERS["easier2"], name="b_easier", ladder="B")
TIERS["b_easy"] = dict(TIERS["easy"], name="b_easy", ladder="B")
TIERS["b_medium"] = dict(name="b_medium", ladder="B",
                         start=dict(state="easier4_v4dx", flatten="easier4_fin2"), target=HALF + [DOG_TOP_L],
                         creases=HALF_CREASES + [_CR_L])
TIERS["b_hard"] = dict(name="b_hard", ladder="B",
                       start=dict(state="hard4_v4dx", flatten="hard4b_fin2"), target=HALF + [DOG_TOP_L, DOG_TOP_R],
                       creases=HALF_CREASES + [_CR_L, _CR_R], obstacles="hard4")
TIERS["b_hard_fold"] = dict(TIERS["b_hard"], name="b_hard_fold", start=None, obstacles=None)
TIERS["b_hard_two_dog_ears"] = dict(TIERS["b_hard"], name="b_hard_two_dog_ears", status="dropped 04:55 (target roll)")
TIERS["b_hard"] = dict(TIERS["b_medium"], name="b_hard", start=dict(state="hard4_v4dx", flatten="hard4b_fin2"),
                       obstacles="hard4")
