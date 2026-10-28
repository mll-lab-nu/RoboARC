"""The redesigned knot_untying tiers as robot-level MuJoCo tasks (knot-inst, 2026-10-10).

One task class for the five tiers (very_easy, easier, easy, medium, hard). The scene travels in the task parameters
(task_parameters["redesign"]), so a frozen instance describes itself and the stress tools rebuild it:

  redesign = {
    "tier": "medium",
    "frame": {"origin_world_xy": [x, y], "table_top_z": z},     # task-local frame: origin between the ALOHA bases
    "ropes": [{"name": "rope", "radius_m": .004, "length_m": 1.76,
               "anchor": {"end": "start", "links": 2, "rigid": "clamp_rope_plate"}}],
    "fixtures": [{"name", "role" (clamp | post | curb), "kind" (box | cylinder), "pos" (task-local centre, z above
                  the table top), "size" [sx, sy, sz] | "radius" + "height", "yaw_deg", "rgba"}],
    "regions": {"rope": [[x, y], ...]},                           # task-local polygons; {} without a layout goal
    "fixture_friction": .3,
    "source": {...},                                              # provenance (the lab start), not read by the task
  }

What it declares (existing protocol concepts only):
  - every rope a FEM rope (EntitySpec "fem_elastic", the family's material): its volume a straight material tube of the
    rope's length and radius (topogym.assets.geometry.rope_volume), so the MuJoCo chain has the lab rope's links; the
    shape comes from the instance's birth (SimulatorConfig.mujoco_birth: the lab start's settled chain);
  - every fixture a fixed rigid box / cylinder (posts, a curb, the clamp's plate and legs), the lab's friction;
  - every rope's anchored end an AttachmentSpec to its clamp plate (MuJoCo: the end links welded for good, needs the
    core patch knot_redesign_rope_attachments).
The goal (goal terms only; nothing here ends an episode): for every rope the family's own terms (common.py: no projected
crossings or overlaps at 2 r on the material centre line, on the table within the family's z band, quiet at the
family's 1 mm/s, an intact volume), the ropes kept 2 r apart in plan (two ropes), each rope's centre line inside its
region (tiers with a region), and the family's release settling (robot clear) over every rope.

The variation records are in topogym.config.variation_records (registration.variation_records); register() adds them
in-process where an older tree lacks them (idempotent).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from topogym.assets.materials import rope_fem_parameters, rope_ipc_parameters
from topogym.assets.scene_visuals import surface
from topogym.evaluation.task_metrics import ipc_integrity, soft_speed
from topogym.tasks.base import AttachmentSpec, EntitySpec, TaskSpec, merge_parameters
from topogym.tasks.knot_untying import common as _common
from topogym.tasks.knot_untying.common import KnotUntyingTask
from topogym.tasks.scene_presets import record_preset

TIERS = ("very_easy", "easier", "easy", "medium", "hard")
LINK_M = .0045          # the knot profile's chain link (topogym/world/mujoco/profiles.py knot_chain_v5, rope.link_m)
# The fixtures' looks of appearance seeds 1 and 2 (task parameter fixture_finish; scene_variation's visual axis), by
# fixture role: the posts and the curb oiled walnut, then blue powder-coated steel; the clamp black-oxide steel, then red
# powder-coated steel. Each is >= 20 CIEDE2000 from the released colours, from each other seed's, from the green goal
# region and from the rope of its seed, and >= 15 from every table top. "released": each fixture's own rgba.
FIXTURE_FINISHES = {
    "walnut_black_oxide": dict(post=dict(color=(.20, .11, .06, 1.), metallic=0., roughness=.55),
                               curb=dict(color=(.20, .11, .06, 1.), metallic=0., roughness=.55),
                               clamp=dict(color=(.035, .035, .04, 1.), metallic=.6, roughness=.45)),
    "blue_red_powder_coat": dict(post=dict(color=(.06, .16, .42, 1.), metallic=.3, roughness=.45),
                                 curb=dict(color=(.06, .16, .42, 1.), metallic=.3, roughness=.45),
                                 clamp=dict(color=(.55, .07, .05, 1.), metallic=.3, roughness=.4)),
}


def variation_id(tier: str) -> str:
    return f"knot_untying_rd_{tier}"


def _yaw_quat(yaw_deg) -> tuple:
    h = np.radians(float(yaw_deg)) / 2
    return (float(np.cos(h)), 0., 0., float(np.sin(h)))


def _seg_dist(A, B, C, D):
    """Planar segment-segment distances (0 where they cross), vectorized: AB[i] against CD[i]."""
    def cr(x, y):
        return x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]

    def point_segment(P, X, Y):
        E = Y - X
        t = np.clip(np.einsum("ij,ij->i", P - X, E) / np.maximum(np.einsum("ij,ij->i", E, E), 1e-30), 0, 1)
        return np.linalg.norm(P - (X + t[:, None] * E), axis=1)
    U, V, CA = B - A, D - C, C - A
    den = cr(U, V)
    with np.errstate(divide="ignore", invalid="ignore"):
        t, u = cr(CA, V) / den, cr(CA, U) / den
    hit = (np.abs(den) > 1e-15) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    d = np.minimum.reduce([point_segment(A, C, D), point_segment(B, C, D), point_segment(C, A, B), point_segment(D, A, B)])
    return np.where(hit, 0., d)


def mutual_planar_distance(a_xy, b_xy) -> float:
    """The exact smallest top-view distance between two polylines (0 where they cross): candidate segment pairs from a
    k-d tree on the segment midpoints within r + half of each longest segment, r doubled from 2 cm until a pair lies
    within r (then it is the global minimum) or r spans both curves."""
    from scipy.spatial import cKDTree
    a, b = np.asarray(a_xy, float)[:, :2], np.asarray(b_xy, float)[:, :2]
    A0, A1, B0, B1 = a[:-1], a[1:], b[:-1], b[1:]
    reach = .5 * (np.linalg.norm(A1 - A0, axis=1).max() + np.linalg.norm(B1 - B0, axis=1).max())
    ta, tb = cKDTree(.5 * (A0 + A1)), cKDTree(.5 * (B0 + B1))
    both = np.vstack([a, b])
    span = float(np.linalg.norm(both.max(0) - both.min(0)))
    r = .02
    while True:
        pairs = ta.sparse_distance_matrix(tb, r + reach, output_type="ndarray")
        if len(pairs):
            d = float(_seg_dist(A0[pairs["i"]], A1[pairs["i"]], B0[pairs["j"]], B1[pairs["j"]]).min())
            if d <= r or r >= span:
                return d
        elif r >= span:
            return float("inf")
        r *= 2.


class KnotUntyingRedesignTask(KnotUntyingTask):
    """A redesigned knot_untying tier (scene in task_parameters["redesign"]); see the module docstring."""
    variation_id = "knot_untying_rd"
    family_id = "knot_untying_redesign"
    model_revision = "knot_untying_rd_v1_mujoco_chain"
    difficulty = None
    defaults = merge_parameters(KnotUntyingTask.defaults, {"redesign": {}, "fixture_finish": "released"})
    hash_when_changed = (*KnotUntyingTask.hash_when_changed, "fixture_finish")
    instruction = ("Untie every knot and spread each rope on the table so that nonadjacent rope segments have no "
                   "crossings or overlaps when viewed from above. A clamped rope end stays clamped.")

    # ------------------------------------------------------------------ the scene
    def _scene(self) -> dict:
        rd = self.config.get("redesign") or {}
        for key in ("frame", "ropes"):
            if key not in rd:
                raise ValueError(f"knot_untying redesign: task_parameters.redesign lacks {key!r}")
        names = [r["name"] for r in rd["ropes"]]
        if not names or len(set(names)) != len(names):
            raise ValueError("knot_untying redesign: rope names must be distinct and present")
        fixtures = {f["name"] for f in rd.get("fixtures", ())}
        for r in rd["ropes"]:
            anchor = r.get("anchor")
            if anchor and anchor.get("rigid") not in fixtures:
                raise ValueError(f"rope {r['name']}: its anchor names no fixture ({anchor.get('rigid')})")
        bad = set(rd.get("regions") or {}) - set(names)
        if bad:
            raise ValueError(f"regions of unknown ropes: {sorted(bad)}")
        return rd

    @property
    def rope_names(self) -> tuple:
        return tuple(r["name"] for r in self._scene()["ropes"])

    @staticmethod
    def volume_name(rope: str) -> str:
        """The generated volume's file name (the frozen-asset hook matches by name): the family's for a rope "rope"."""
        return "rope_volume.npz" if rope == "rope" else f"rope_volume_{rope}.npz"

    def declare(self, asset_dir, table_height_m):
        from topogym.assets.geometry.rope_volume import write_rope_volume
        self.table_height_m = float(table_height_m)
        rd = self._scene()
        frame = rd["frame"]
        if abs(float(frame["table_top_z"]) - self.table_height_m) > 1e-6:
            raise ValueError(f"redesign frame table top {frame['table_top_z']} != the scene's {self.table_height_m}")
        ox, oy = (float(v) for v in frame["origin_world_xy"])
        self.origin_xy = np.array([ox, oy])
        preset = record_preset(self, asset_dir)
        entities, attachments, self.ropes = [], [], {}
        for r in rd["ropes"]:
            name, length, radius = r["name"], float(r["length_m"]), float(r["radius_m"])
            volume = Path(asset_dir) / self.volume_name(name)
            # a straight material tube of the rope's length: the chain's links follow from it (as many as the lab's);
            # the shape is the birth's (SimulatorConfig.mujoco_birth)
            write_rope_volume(volume, np.array([[0., 0., 0.], [length, 0., 0.]]), radius)
            # through the family module: rcb_topogym's frozen-asset hook replaces the file with a frozen instance's copy
            cmap = _common.load_centerline_map(volume)
            self.ropes[name] = dict(centerline_map=cmap, radius=radius, length=length)
            entities.append(EntitySpec(
                name, "mesh", None, (ox, oy, self.table_height_m), physics_model="fem_elastic", volume_path=volume,
                material=rope_fem_parameters(), surface=surface(r.get("finish", self.config["rope_finish"])),
                morph_options={"convexify": False, "decimate": False}))
            anchor = r.get("anchor")
            if anchor:
                # the lab's anchor: its first `links` chain links; the band ends exactly at their end (a chain of
                # round(L / LINK_M) links, the scene's own rule), so the welded links are the lab's
                n = max(2, int(round(length / LINK_M)))
                attachments.append(AttachmentSpec(f"clamp_{name}", name, anchor["rigid"], anchor.get("end", "start"),
                                                  int(anchor["links"]) * length / n))
        friction = float(rd.get("fixture_friction", .3))
        for f in rd.get("fixtures", ()):
            x, y, z = (float(v) for v in f["pos"])
            if f["kind"] == "box":
                opts = {"size": tuple(float(v) for v in f["size"])}
            elif f["kind"] == "cylinder" and rd.get("cylinder_fixtures") == "prism":
                # a convex 32-facet prism of the cylinder (vertices on the radius: <= 0.06 mm inside it at r 12 mm);
                # MuJoCo's cylinder-convex rim contact can flip its normal (a finger sliced 11 mm into a post)
                opts = {}
                mesh_path = Path(asset_dir) / f"{f['name']}_prism.obj"
                _write_prism(mesh_path, float(f["radius"]), float(f["height"]), 32)
            elif f["kind"] == "cylinder":
                opts = {"radius": float(f["radius"]), "height": float(f["height"])}
            else:
                raise ValueError(f"fixture {f['name']}: kind {f['kind']} (box or cylinder)")
            rgba = f.get("rgba", ".45 .42 .38 1")
            rgba = tuple(float(v) for v in (rgba.split() if isinstance(rgba, str) else rgba))
            finish = self.config["fixture_finish"]
            look = dict(color=rgba) if finish == "released" else dict(FIXTURE_FINISHES[finish][f["role"]])
            prism = f["kind"] == "cylinder" and rd.get("cylinder_fixtures") == "prism"
            entities.append(EntitySpec(
                f["name"], "mesh" if prism else f["kind"], str(mesh_path) if prism else None,
                (ox + x, oy + y, self.table_height_m + z),
                _yaw_quat(f.get("yaw_deg", 0.)), fixed=True, material={"friction": friction}, morph_options=opts,
                surface=look))
        self.regions = {k: np.asarray(v, dtype=float) for k, v in (rd.get("regions") or {}).items()}
        # the family's single-rope attributes, for code that reads them (its first rope)
        first = rd["ropes"][0]["name"]
        self.centerline_map = self.ropes[first]["centerline_map"]
        return TaskSpec(
            tuple(entities),
            # the family's FEM / IPC options (they configure a Genesis rope; the MuJoCo chain takes the profile's)
            {"fem": {}, "ipc": dict(rope_ipc_parameters(), contact_eps_velocity=.001)},
            dict(**preset, task_id=self.task_id, fem_rope_profile="v13_material_mesh", redesign_tier=rd.get("tier"),
                 ropes=[dict(name=r["name"], length_m=float(r["length_m"])) for r in rd["ropes"]],
                 required_reference_shape="straight", goal_evaluation_complete=bool(self.goal_complete)),
            attachments=tuple(attachments))

    # ------------------------------------------------------------------ the goal
    def final_settling_spec(self):
        from topogym.evaluation.terminal import SettlingSpec
        return SettlingSpec(fem_targets=self.rope_names, clearance_m=.001,
                            robot_contact_policy="fixed_contact_active_clearance_v1",
                            maximum_speed_m_s=self.config["terminal"]["maximum_speed_m_s"])

    def rope_centerline(self, state, name):
        """(material centre-line points (N, 3) world, their material stations (N,)) of rope `name`."""
        from topogym.utils.geometry.rope import sample_material_centerline
        indices, material_s = self.ropes[name]["centerline_map"]
        points = sample_material_centerline(state["fem"][name], indices, material_s)
        stations = material_s[indices]
        return points, (stations.mean(axis=1) if stations.ndim == 2 else stations)

    def evaluate_state(self, state):
        cfg = self.config["terminal"]
        ropes, ok, lines = {}, True, {}
        worst_speed = 0.
        for name in self.rope_names:
            soft = state["fem"][name]
            radius = self.ropes[name]["radius"]
            points, stations = self.rope_centerline(state, name)
            lines[name] = points
            clear, geometry = _common.planar_clearance(points, stations, 2 * radius)
            z = np.asarray(soft["position_m"])[:, 2]
            on_table = bool(z.min() >= self.table_height_m - cfg["penetration_tolerance_m"]
                            and z.max() <= self.table_height_m + 2 * radius + cfg["table_clearance_m"])
            quiet, speed = soft_speed(soft, cfg["maximum_speed_m_s"])
            worst_speed = max(worst_speed, speed["maximum_speed_m_s"])
            rec = dict(planar_clear=bool(clear), on_table=on_table, quiet=bool(quiet), **geometry, **speed)
            if name in self.regions:
                from .goal import inside
                within = inside(points[:, :2] - self.origin_xy, self.regions[name])
                rec.update(in_region=bool(within.all()), outside_fraction=float(1 - within.mean()))
            ropes[name] = rec
            ok &= rec["planar_clear"] and on_table and rec["quiet"] and rec.get("in_region", True)
        names = list(lines)
        between = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = mutual_planar_distance(lines[names[i]], lines[names[j]])
                need = self.ropes[names[i]]["radius"] + self.ropes[names[j]]["radius"]
                between[f"{names[i]}|{names[j]}"] = dict(minimum_planar_distance_m=d, clear=bool(d >= need))
                ok &= d >= need
        intact = ipc_integrity(state, tuple(names))
        metrics = dict(
            planar_clear=all(r["planar_clear"] for r in ropes.values()),
            on_table=all(r["on_table"] for r in ropes.values()),
            quiet=all(r["quiet"] for r in ropes.values()),
            maximum_speed_m_s=worst_speed, ropes=ropes, between_ropes=between,
            in_regions=all(r.get("in_region", True) for r in ropes.values()),
            evaluation_status="available" if intact is not None else "unavailable",
            thresholds_calibrated=False, goal_evaluation_complete=bool(self.goal_complete))
        if len(names) == 1:     # the family's flat keys (minimum_nonlocal_distance_m, p99_speed_m_s, ...)
            metrics.update({k: v for k, v in ropes[names[0]].items() if k not in metrics})
        return (bool(ok and intact is True), "rope_volume_invalid" if intact is False else None, metrics)

    # The region images a region tier's task ships beside its instruction (/app/goal/, both modes): the start scene seen
    # from the agent's cameras cam_high and cam_low with each goal region's dashed line drawn in (the scene itself does
    # not draw it). scripts/robotangle/redesign/knot_untying/region_images.py renders them.
    REGION_IMAGES = ("region_cam_high.png", "region_cam_low.png")

    def public_instruction(self) -> str:
        """The family's goal words (describe_goal) with this design's: the clamp and, on the region tiers, the goal
        region as a note after the task sentence; the region's containment and the ropes' separation as success
        conditions before the confirmation line, where the texts take them (scripts/robotangle/package_text_tool.py).
        No world coordinates (user 2026-10-13): a region is named by its dashed green line and its size, and the
        region images show where it lies; left / right and near / far as seen in the overhead camera cam_high (near =
        the lower side of its image, the smaller task-frame y)."""
        text = super().public_instruction()
        rd = self._scene()
        ropes = rd["ropes"]
        names = [r["name"] for r in ropes]
        regions = rd.get("regions") or {}
        note, rules = [], []
        if any(r.get("anchor") for r in ropes):
            note.append("One end of every rope is clamped to the table under a staple and stays clamped.")
        if regions:
            one = len(regions) == 1
            if one:
                (name,) = regions
                whom = "the rope" if len(names) == 1 else f"rope `{name}`"
                note.append(f"Lay {whom} inside the green dashed goal region.")
            else:
                order = sorted(regions, key=lambda k: float(np.mean(np.asarray(regions[k], float)[:, 1])))
                note.append("Lay each rope inside its green dashed goal region: " + ", ".join(
                    f"rope `{k}` inside the {_ordinal(i, len(order))} one" for i, k in enumerate(order)) + ".")
            note.append(("The goal region is not drawn in the scene: the region images " if one else
                         "The goal regions are not drawn in the scene: the region images ")
                        + f"`/app/goal/{self.REGION_IMAGES[0]}` (from the overhead camera cam_high) and "
                        + f"`/app/goal/{self.REGION_IMAGES[1]}` (from the low oblique camera cam_low) show the start "
                        + ("scene with the region's dashed line drawn in." if one else
                           "scene with their dashed lines drawn in."))
            for name, poly in regions.items():
                p = np.asarray(poly, float)
                w, d = (float(v) * 1000 for v in p.max(axis=0) - p.min(axis=0))
                rect = len(p) == 4 and all(min(abs(float(p[i, 0] - p[(i + 1) % 4, 0])),
                                               abs(float(p[i, 1] - p[(i + 1) % 4, 1]))) < 1e-9 for i in range(4))
                whom = "the rope" if len(names) == 1 else f"rope `{name}`"
                where = "the green dashed goal region" if one else "its green dashed goal region"
                shape = (f"it encloses a rectangle {w:.6g} mm wide (left to right in cam_high's image) and {d:.6g} mm "
                         "deep" if rect else f"it encloses an outline {w:.6g} mm wide and {d:.6g} mm deep at most")
                rules.append(f"Put the whole centre line of {whom} inside {where}: the inner edge of the 2 mm wide dashed "
                             f"green line is the containment limit; {shape}. Every point of the centre line counts; the "
                             "rope's own thickness may reach over the line.")
        if len(names) > 1:
            rr = {float(r["radius_m"]) for r in ropes}
            gap = (f"at least one rope diameter ({2000 * rr.pop():.6g} mm)" if len(rr) == 1
                   else "at least the sum of their radii")
            rules.append(f"Keep the ropes apart: in the top-down view the centre lines of different ropes must be {gap} "
                         "apart.")
        head, sep, rest = text.partition("\n\n")
        if note:
            head = head + " " + " ".join(note)
        anchor = "- Keep all final conditions satisfied together"
        if rules:
            if anchor not in rest:
                raise ValueError("knot_untying redesign: the family's goal words lack the confirmation line")
            rest = rest.replace(anchor, "\n".join("- " + r for r in rules) + "\n" + anchor, 1)
        return head + sep + rest

    def config_origin(self):
        return np.asarray(self._scene()["frame"]["origin_world_xy"], float)


def _ordinal(i: int, n: int) -> str:
    """The i-th of n regions in cam_high's near-to-far order, in words."""
    if n == 2:
        return ("near", "far")[i]
    k = i + 1
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')} nearest"


def _tier_class(tier: str):
    name = "KnotUntyingRd" + "".join(w.capitalize() for w in tier.split("_")) + "Task"
    return type(name, (KnotUntyingRedesignTask,), dict(variation_id=variation_id(tier), difficulty=tier,
                                                       __module__=__name__))


for _tier in TIERS:
    _cls = _tier_class(_tier)
    globals()[_cls.__name__] = _cls


def register() -> list:
    """Append the five redesign variations to the variation records of this process (idempotent), as
    session_kit.register_variation does; every process that builds or loads a redesign instance calls it first."""
    from topogym.config import variation_records
    out = []
    for tier in TIERS:
        vid = variation_id(tier)
        cls = "KnotUntyingRd" + "".join(w.capitalize() for w in tier.split("_")) + "Task"
        row = next((r for r in variation_records.RECORDS if r["task_id"] == "knot_untying" and r["variation_id"] == vid),
                   None)
        if row is None:
            row = dict(task_id="knot_untying", variation_id=vid, difficulty=tier, family_id="knot_untying_redesign",
                       model_revision=KnotUntyingRedesignTask.model_revision,
                       implementation=f"{__name__}:{cls}", aliases=[], status="integrated", reason=None,
                       validation="redesign_dev", pool="redesign")
            variation_records.RECORDS = (*variation_records.RECORDS, row)
        out.append(row)
    return out


def _write_prism(path, radius, height, n):
    """A closed convex n-gon prism (OBJ, centred at the origin, axis z) with its vertices on the circle of `radius`."""
    a = np.linspace(0., 2 * np.pi, n, endpoint=False)
    ring = np.c_[radius * np.cos(a), radius * np.sin(a)]
    v = [(x, y, -height / 2) for x, y in ring] + [(x, y, height / 2) for x, y in ring]
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j), (i, n + j, n + i)]
    f += [(0, k + 1, k) for k in range(1, n - 1)] + [(n, n + k, n + k + 1) for k in range(1, n - 1)]
    lines = ["v %.9f %.9f %.9f" % p for p in v] + ["f %d %d %d" % (a_ + 1, b_ + 1, c_ + 1) for a_, b_, c_ in f]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
