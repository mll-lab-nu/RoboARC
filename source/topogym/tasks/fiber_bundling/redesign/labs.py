"""Mechanics labs of the fiber_bundling redesign: each tier's new fixture and goal terms on the released easier bundle
(three symmetric strands), so that one mechanic is tested without the 16-strand bundle. Not tiers.
docs/rcb/redesign/fiber_bundling.md, "Feasibility of the key mechanics".
"""
from topogym.assets.geometry.fiber_bundling_simplified import SymmetricFiberLayout
from topogym.tasks.base import merge_parameters

from .tasks import TIERS, RedesignFiberTask


def _lab(name, tier, **extra):
    cfg = merge_parameters(TIERS[tier], dict(geometry=dict(fiber_count=3)))
    cfg = merge_parameters(cfg, extra)
    attrs = dict(variation_id=f"fiber_bundling_lab_{name}", difficulty=tier, family_id=f"lab_{name}",
                 model_revision="fiber_redesign_lab_v0", layout_class=SymmetricFiberLayout,
                 defaults=merge_parameters(RedesignFiberTask.defaults, cfg), __module__=__name__)
    return type("FiberBundlingLab" + "".join(w.capitalize() for w in name.split("_")) + "Task", (RedesignFiberTask,), attrs)


# lay: the released easier scene with easier's lay term
FiberBundlingLabLayTask = _lab("lay", "easier", geometry=dict(ring_inner_radius_m=.0135))
# turned: easy's closer, turned ring and its lay
FiberBundlingLabTurnedTask = _lab("turned", "easy")
# seated: medium's longer sheath, peg, ring and lay
FiberBundlingLabSeatedTask = _lab("seated", "medium")
# lay-over: hard's pegs, ring and lay
FiberBundlingLabLayoverTask = _lab("layover", "hard")
LAB_CLASSES = dict(lay=FiberBundlingLabLayTask, turned=FiberBundlingLabTurnedTask, seated=FiberBundlingLabSeatedTask,
                   layover=FiberBundlingLabLayoverTask)
