"""The planner's scene: the task's own scenario per scene seed (topogym.assets.geometry.caliper_sorting.scenario: the
caliper's unit and zero, four sleeves of two classes in rack slots, the labels' class order, the stand's shift), with the
sleeves as the planner keeps them (geom.make_sleeves). The hand-held design's robot-motion noise and caliper re-grasps
went with the hand-held caliper (2026-10-18, the caliper on its stand)."""
from __future__ import annotations

import geom as sv

CLS = ('A', 'B', 'C', 'D')


def scene(seed):
    scen = dict(sv.G.scenario(seed))
    scen['sleeves'] = sv.make_sleeves(scen)
    return scen
