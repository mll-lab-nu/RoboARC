"""fiber_bundling redesign (step 3, proposal): harder tiers for the Panda + Robotiq on the wire cable (fiber_v4).

docs/rcb/redesign/fiber_bundling.md has the design. Every tier keeps the family's predicate (all strand stations and
the sheath front through the bundling ring, tip first, bends <= 60 deg, no gripper-strand contact, a 0.5 s stable
window) and adds target-configuration terms on top of it:
- `lay`: the marked strand's root sits within a tolerance of an angle around the ring axis, measured from the ring's
  top (the world vertical projected onto the ring plane, opposite the ring's post): the bundle's roll is specified,
  like a keyed connector;
- `pegs`: formboard pegs the sheath must pass on a given side (top view), close enough to be routed by them and
  beside them (below the peg's top, not over it);
and changes the scene: 16 strands, a closer-fitting ring, the ring turned about the vertical, pegs on the shortcuts.

The scene is a rigid transform of the family's own fixture (the ring and its mount turned about the vertical through
its centre and moved), plus primitive pegs; the cable model, the cradle, the terminal and the robot are the
family's. Positions in `redesign` are relative to the task origin (the trunk root on the cable axis).
"""
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.tasks.base import EntitySpec, merge_parameters
from topogym.tasks.fiber_bundling.common import FiberBundlingTask

RING_PARTS = ("bundling_ring", "ring_mount_base", "ring_mount_post")


def same_colour_strands(marked, count):
    """The other strands drawn in the marked strand's palette colour (COLORS[i % len(COLORS)], the palette
    strand_palette_shift rotates only past entry 0), in index order."""
    from topogym.assets.geometry.fiber_bundling import COLORS
    return [i for i in range(count) if i != marked and i % len(COLORS) == marked % len(COLORS)]
PEG_COLOR = (.30, .34, .38)
CAP_COLOR = (.18, .20, .23)
CAP_HEIGHT_M = .006
# Smooth pegs (a nylon formboard peg on a PVC sheath): a contact takes the larger friction of its two geoms, so the
# sheath's own (0.35) applies instead of the family's default 1.0 for task bodies (capstan friction e^(mu theta):
# 13x at mu 1.0, 2.5x at 0.35 for a 150 deg contact arc).
PEG_FRICTION = .3


def _yaw(deg):
    return Rotation.from_euler("z", deg, degrees=True)


class RedesignFiberTask(FiberBundlingTask):
    """A redesign tier: the family's scene and predicate with `redesign` fixtures and goal terms."""
    instruction = ('Manipulate only the thick sheath. Route it as the formboard pegs require, then pass every fiber '
                   'tip first through the copper ring until the roots and front sheath have passed, without sharp '
                   'bends, with one of the two red fibers at the stated position around the ring (seen from the entry side). Any gripper contact with a fine strand fails '
                   'the task; keep the rear terminal anchored.')
    variation_id = "fiber_bundling_rd_base"
    family_id = "redesign"
    model_revision = "fiber_redesign_v0"
    difficulty = "very_easy"
    fixed_parameters = ()
    defaults = merge_parameters(FiberBundlingTask.defaults, dict(
        geometry=dict(fiber_count=16, ring_inner_radius_m=.0135),
        redesign=dict(
            # the ring (and its mount) turned by yaw_deg about the vertical through its released centre, then moved
            # so that its centre is center_m (None: the released centre)
            ring=dict(yaw_deg=0., center_m=None),
            # [{"name", "xy_m": [x, y], "radius_m", "top_m" (above the cable axis), "side": "left"|"right",
            #   "reach_m" (largest sheath-to-peg surface gap that still counts as routed),
            #   "cap_m" (the head's overhang beyond the shaft: a formboard nail; a cable wrapped under tension
            #   cannot slip over it)}]
            pegs=[],
            # {"strand": i, "angle_deg": a (0 = the ring's top; right-handed about the ring axis, the feed direction),
            #  "tolerance_deg": t}
            lay=None)))

    # ------------------------------------------------------------------------------------------------ the scene
    def declare(self, asset_dir, table_height_m):
        spec = super().declare(asset_dir, table_height_m)
        rd = self.config["redesign"]
        ring = rd["ring"]
        R = _yaw(float(ring["yaw_deg"]))
        pivot = np.asarray(self.ring_center, float)                     # the released ring centre
        target = pivot if ring["center_m"] is None else self.origin + np.asarray(ring["center_m"], float)
        entities = []
        for e in spec.entities:
            if e.name in RING_PARTS:
                p = target + R.apply(np.asarray(e.position_m, float) - pivot)
                q = R * Rotation.from_quat(np.asarray(e.quaternion_wxyz, float)[[1, 2, 3, 0]])
                e = replace(e, position_m=tuple(map(float, p)), quaternion_wxyz=tuple(map(float, q.as_quat()[[3, 0, 1, 2]])))
                if e.name == "bundling_ring":
                    self.ring_center = np.asarray(e.position_m, float)
            entities.append(e)
        for peg in rd["pegs"]:
            top = float(self.origin[2] + peg["top_m"])
            h = top - self.table_height_m
            c = self.origin + np.r_[np.asarray(peg["xy_m"], float), 0.]
            entities.append(EntitySpec(peg["name"], "cylinder", None, (float(c[0]), float(c[1]), self.table_height_m + h / 2),
                                       fixed=True, material={"friction": PEG_FRICTION},
                                       morph_options={"radius": float(peg["radius_m"]), "height": h},
                                       surface={"color": PEG_COLOR}))
            if float(peg.get("cap_m", 0.)) > 0.:
                entities.append(EntitySpec(f"{peg['name']}_cap", "cylinder", None,
                                           (float(c[0]), float(c[1]), top - CAP_HEIGHT_M / 2), fixed=True,
                                           material={"friction": PEG_FRICTION},
                                           morph_options={"radius": float(peg["radius_m"]) + float(peg["cap_m"]),
                                                          "height": CAP_HEIGHT_M},
                                           surface={"color": CAP_COLOR}))
        return replace(spec, entities=tuple(entities))

    # ------------------------------------------------------------------------------------------------ the goal
    def public_instruction(self):
        """The family's public goal description with the redesign's requirements added to its success conditions
        (no world coordinates: pegs by their order from the cradle and height, sides and angles relative to the
        sheath and the ring)."""
        text = super().public_instruction()
        extra = self.redesign_requirements()
        anchor = "\n- Keep all final conditions satisfied together"   # the family's last success condition
        if not extra or anchor not in text:
            return text
        return text.replace(anchor, "".join(f"\n- {x}" for x in extra) + anchor, 1)

    def redesign_requirements(self):
        rd = self.config["redesign"]
        out = []
        pegs = sorted(rd["pegs"], key=lambda pg: pg["xy_m"][0])
        ordinal = ["first", "second", "third", "fourth"]
        for i, peg in enumerate(pegs):
            which = "the formboard peg" if len(pegs) == 1 else f"the {ordinal[i]} formboard peg from the cradle"
            out.append(f"Route the sheath past {which} (its top {1e3 * peg['top_m']:.0f} mm above the sheath's start "
                       f"axis) with the peg on the sheath's {peg['side']}, looking along the sheath from the rear "
                       f"terminal toward the ring. The sheath must lie beside the peg, below its top, within "
                       f"{1e3 * float(peg.get('reach_m', .02)):.0f} mm of its surface.")
        lay = rd["lay"]
        if lay:
            a, tol = float(lay["angle_deg"]), float(lay["tolerance_deg"])
            minutes = int(round((a / 30. % 12.) * 60.)) % 720
            clock = f"{(minutes // 60) or 12}:{minutes % 60:02d}"
            turn = "clockwise" if a > 0 else "counterclockwise"
            out.append(f"Seen from the ring's entry side, looking through the ring in the feed direction, the root of "
                       f"one of the two red fibers must end {abs(a):g} deg {turn} from the top of the ring (about {clock} on a clock "
                       f"face), within ±{tol:g} deg.")
        return out

    def _sheath_polyline(self, state):
        """Top view of the sheath's centre line: lead links near the cradle, then every trunk link, then the front."""
        links = state["articulations"]["fiber_bundle"]["links"]
        names = [n for n in links if n.startswith("lead_")]
        names = sorted(names, key=lambda n: int(n.split("_")[1]))[-4:]
        names += [f"trunk_{i:02d}" for i in range(self.layout.trunk_segments)]
        pts = [np.asarray(links[n]["position_m"], float) for n in names]
        pts.append(self._front_point(state))
        return np.asarray(pts)

    def route_terms(self, state):
        """Per peg: the side of the sheath it lies on (top view, looking from the terminal towards the front), the
        surface gap between peg and sheath at the closest approach, whether the sheath passes there below the peg's
        top (beside the peg, not over it), and whether it counts as routed."""
        out = []
        P3 = self._sheath_polyline(state)
        P = P3[:, :2]
        for peg in self.config["redesign"]["pegs"]:
            c = (self.origin[:2] + np.asarray(peg["xy_m"], float))
            best = None
            for a, b, za, zb in zip(P[:-1], P[1:], P3[:-1, 2], P3[1:, 2]):
                ab = b - a
                t = float(np.clip(np.dot(c - a, ab) / max(np.dot(ab, ab), 1e-12), 0., 1.))
                q = a + t * ab
                d = float(np.linalg.norm(c - q))
                if best is None or d < best[0]:
                    cross = ab[0] * (c - q)[1] - ab[1] * (c - q)[0]
                    best = (d, "left" if cross > 0 else "right", float(za + t * (zb - za)))
            gap = best[0] - float(peg["radius_m"]) - self.layout.trunk_radius_m
            below = best[2] <= float(self.origin[2] + peg["top_m"]) - self.layout.trunk_radius_m
            ok = best[1] == peg["side"] and gap <= float(peg.get("reach_m", .02)) and below
            out.append(dict(name=peg["name"], side=best[1], required=peg["side"], gap_m=round(gap, 5),
                            below_top=bool(below), routed=bool(ok)))
        return out

    def lay_term(self, state, center, normal):
        """The marked colour's root angle about the ring axis from the ring's top (deg, right-handed about the feed
        direction), and whether it is within tolerance. The texts name the strand by its colour ("one of the two red
        fibers"): every strand with the marked strand's palette colour (strand i has palette entry i % len(COLORS);
        strand_palette_shift never moves entry 0, so with 16 strands strands 0 and 8 are red) may meet the rule. The
        marked strand itself is reported when it is within tolerance or when none is, so a trajectory that met the
        rule with it keeps exactly its earlier record; otherwise the first other strand of that colour within tolerance."""
        lay = self.config["redesign"]["lay"]
        if not lay:
            return None
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        up = np.array([0., 0., 1.]) - n[2] * n
        up /= np.linalg.norm(up)
        left = np.cross(n, up)
        paths = self._strand_points(state)
        axis_point = self._front_point(state)
        marked = int(lay["strand"])

        def term(i):
            v = paths[i][-1] - axis_point
            v -= np.dot(v, n) * n
            angle = float(np.degrees(np.arctan2(np.dot(v, left), np.dot(v, up))))
            err = (angle - float(lay["angle_deg"]) + 180.) % 360. - 180.
            return dict(strand=i, angle_deg=round(angle, 2), error_deg=round(err, 2),
                        within=bool(abs(err) <= float(lay["tolerance_deg"])))

        first = term(marked)
        if first["within"]:
            return first
        for i in same_colour_strands(marked, len(paths)):
            other = term(i)
            if other["within"]:
                return other
        return first

    def evaluate_transition(self, state, task_state):
        reached, failure, metrics, nxt = super().evaluate_transition(state, task_state)
        if failure is not None or not metrics:
            return reached, failure, metrics, nxt
        ring = state["bodies"]["bundling_ring"]
        center = np.asarray(ring["position_m"], float)
        q = np.asarray(ring["quaternion_wxyz"], float)
        normal = Rotation.from_quat(q[[1, 2, 3, 0]]).apply([0., 0., 1.])
        lay = self.lay_term(state, center, normal)
        routes = self.route_terms(state)
        metrics = dict(metrics, lay=lay, routes=routes, passage_reached=bool(reached))
        reached = bool(reached and (lay is None or lay["within"]) and all(r["routed"] for r in routes))
        return reached, failure, metrics, nxt


# ---------------------------------------------------------------------------------------------------- the tiers
# The tiers' data (inverse generation).
TIERS = {
    "very_easy": {'geometry': {'fiber_count': 16, 'ring_inner_radius_m': 0.0135},
     'redesign': {'ring': {'yaw_deg': 0.0, 'center_m': [0.35, 0.0, 0.0]}, 'pegs': [], 'lay': None}},
    "easier": {'geometry': {'fiber_count': 16, 'ring_inner_radius_m': 0.0135},
     'redesign': {'ring': {'yaw_deg': 0.0, 'center_m': [0.35, 0.0, 0.0]},
                  'pegs': [],
                  'lay': {'strand': 0, 'angle_deg': -30.0, 'tolerance_deg': 20.0}}},
    "easy": {'geometry': {'fiber_count': 16, 'ring_inner_radius_m': 0.0128},
     'redesign': {'ring': {'yaw_deg': 30.0, 'center_m': [0.3215, 0.1086, 0.0]},
                  'pegs': [],
                  'lay': {'strand': 0, 'angle_deg': -130.0, 'tolerance_deg': 20.0}}},
    "medium": {'geometry': {'fiber_count': 16, 'ring_inner_radius_m': 0.0128, 'trunk_length_m': 0.36, 'trunk_segments': 9},
     'center_xy_m': [-0.22, -0.25],
     'redesign': {'ring': {'yaw_deg': 40.0, 'center_m': [0.3994, 0.1034, 0.0]},
                  'pegs': [{'name': 'formboard_peg_0',
                            'xy_m': [0.2, -0.03],
                            'radius_m': 0.012,
                            'top_m': 0.045,
                            'side': 'left',
                            'reach_m': 0.02,
                            'cap_m': 0.005}],
                  'lay': {'strand': 0, 'angle_deg': -140.0, 'tolerance_deg': 15.0}}},
    "hard": {'geometry': {'fiber_count': 16, 'ring_inner_radius_m': 0.0126, 'trunk_length_m': 0.36, 'trunk_segments': 9},
     'center_xy_m': [-0.22, -0.25],
     'redesign': {'ring': {'yaw_deg': 40.0, 'center_m': [0.4168, 0.088, 0.0]},
                  'pegs': [{'name': 'formboard_peg_0',
                            'xy_m': [0.12, -0.03],
                            'radius_m': 0.012,
                            'top_m': 0.085,
                            'side': 'right',
                            'reach_m': 0.02,
                            'cap_m': 0.005},
                           {'name': 'formboard_peg_1',
                            'xy_m': [0.2, -0.06],
                            'radius_m': 0.012,
                            'top_m': 0.035,
                            'side': 'left',
                            'reach_m': 0.02,
                            'cap_m': 0.005}],
                  'lay': {'strand': 0, 'angle_deg': -130.0, 'tolerance_deg': 12.0}}},
}


def _tier_class(tier, cfg):
    name = "FiberBundlingRedesign" + "".join(w.capitalize() for w in tier.split("_")) + "Task"
    attrs = dict(variation_id=f"fiber_bundling_rd_{tier}", difficulty=tier, family_id=f"redesign_{tier}",
                 model_revision="fiber_redesign_v0", defaults=merge_parameters(RedesignFiberTask.defaults, cfg),
                 __module__=__name__)
    return type(name, (RedesignFiberTask,), attrs)


FiberBundlingRedesignVeryEasyTask = _tier_class("very_easy", TIERS["very_easy"])
FiberBundlingRedesignEasierTask = _tier_class("easier", TIERS["easier"])
FiberBundlingRedesignEasyTask = _tier_class("easy", TIERS["easy"])
FiberBundlingRedesignMediumTask = _tier_class("medium", TIERS["medium"])
FiberBundlingRedesignHardTask = _tier_class("hard", TIERS["hard"])
TIER_CLASSES = dict(very_easy=FiberBundlingRedesignVeryEasyTask, easier=FiberBundlingRedesignEasierTask,
                    easy=FiberBundlingRedesignEasyTask, medium=FiberBundlingRedesignMediumTask,
                    hard=FiberBundlingRedesignHardTask)
