"""Freeze the fiber_bundling redesign tiers as MuJoCo instances (rcb_topogym make-instance in-process), with the
released grasp assistance (contact_weld_v1 on the second-to-last sheath link) and the variations registered in-process
(the soft-body redesign tooling's session_kit, tools/redesign on dev/robotangle_softbody_redesign, must be
on PYTHONPATH). docs/rcb/redesign/fiber_bundling.md.

usage: build.py <out_dir> [tier | lab_<name> ...]   -> <out_dir>/<name>/instance.json (+ instance_assets/)
"""
import json
import sys
from pathlib import Path

from session_kit import freeze_instance, register_variation
from topogym.tasks.fiber_bundling.redesign.labs import LAB_CLASSES
from topogym.tasks.fiber_bundling.redesign.tasks import TIER_CLASSES

CLASSES = {**TIER_CLASSES, **{f"lab_{k}": v for k, v in LAB_CLASSES.items()}}

GRASP = {"model": "contact_weld_v1", "stable_time_s": 0.02, "max_tcp_distance_m": 0.06, "max_penetration_m": 0.002,
         "max_translation_error_m": 0.01, "max_rotation_error_rad": 0.15, "max_finger_opening_fraction": 0.15,
         "intent_deadband": 0.002, "initial_intent": "open"}          # the released instances' grasp_assist


def register(name):
    cls = CLASSES[name]
    register_variation("fiber_bundling", cls.variation_id, f"{cls.__module__}:{cls.__name__}", difficulty=cls.difficulty,
                       family_id=cls.family_id, model_revision=cls.model_revision)
    return cls


def grasp_override(cls):
    segments = int(cls.defaults["geometry"].get("trunk_segments", 6))
    length = float(cls.defaults["geometry"].get("trunk_length_m", .24))
    target = {"entity": "fiber_bundle", "link": f"trunk_{segments - 2:02d}", "center_m": [length / segments / 2, 0., 0.],
              "radius_m": .025}
    return {"grasp_assist": dict(GRASP, targets=[target])}


def main(out, tiers):
    out = Path(out)
    for tier in tiers:
        cls = register(tier)
        res = freeze_instance("fiber_bundling", cls.variation_id, out / tier / "instance.json",
                              overrides=grasp_override(cls), gripper_space="native")
        print(json.dumps(dict(tier=tier, **res)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or list(TIER_CLASSES))
