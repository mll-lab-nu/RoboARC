"""Seeded scene variation: one seed picks a difficulty tier and one pose/appearance draw.

``variation_id="seeded"`` resolves the tier as ``seed % 3`` (``VariantRng.tier``) and
overrides the leaves listed in ``AXES`` with values drawn from named streams of the
same seed. Only placement, orientation, colour and material leaves are listed, never
sizes or geometry, so every draw is a rigid move or a re-finish of the nominal scene.
Explicit tier names keep the nominal scene; nothing here runs for them.

Axis kinds:
- ``offset``: add U[lo, hi] to the tier's own default, per component (``None`` keeps
  that component);
- ``choice``: pick one entry of ``values`` with equal probability.
"""
from copy import deepcopy
from collections.abc import Mapping

import numpy as np

from topogym.tasks.base import _parameter_at
from topogym.utils.variant_rng import VariantRng

SCENE_VARIATION_VERSION = 1

# task_id -> axes shared by its three tiers (each offset applies to the tier's own default).
AXES = {
    "horseshoe_ring": (
        # Goal region pinned to -y; bounds keep the reach proxy within +2 cm (historical scene-variation geometry audit).
        dict(path="center_xy_m", kind="offset", range=[[-.03, .04], [-.01, .03]]),
        dict(path="yaw_deg", kind="offset", range=[-10., 10.]),
        # The ring, the horseshoes and the chains (no text names their colours): seed 1 a red painted ring on black
        # oxide horseshoes, seed 2 a dark coated ring on bronze ones; every ring stands apart from the horseshoes at
        # its seed (brass, 18 CIEDE2000 from the copper ring, left the ring values on 2026-10-27).
        dict(path="target_ring_finish", kind="choice", values=["copper", "red_painted", "coated_metal"], visual=True),
        dict(path="shoe_finish", kind="choice", values=["steel", "black_oxide", "bronze"], visual=True),
    ),
    "cast_coil": (
        # Both pieces move together; the silver offset stays in world XY. +y moves the
        # silver piece (already at y ~ 0.23) away from both arms, so y only decreases.
        dict(path="center_xy_m", kind="offset", range=[[-.03, .03], [-.03, 0.]]),
        dict(path="yaw_deg", kind="offset", range=[-10., 10.]),
        # Both pieces' colour and material (scene_visuals.CAST_PIECE_FINISHES): red and blue anodised aluminium, then
        # black and orange powder coat.
        dict(path="piece_finish", kind="choice", values=["cast", "anodised", "powder_coat"], visual=True),
    ),
    "cast_keyhole": (
        # Inside the reference policy's IK tolerance (+-20 mm); goal region pinned to +y.
        dict(path="center_xy_m", kind="offset", range=[[-.02, .02], [-.02, .02]]),
        dict(path="yaw_deg", kind="offset", range=[-5., 5.]),
        # Both pieces' colour and material (scene_visuals.CAST_PIECE_FINISHES): red and blue anodised aluminium, then
        # black and orange powder coat.
        dict(path="piece_finish", kind="choice", values=["cast", "anodised", "powder_coat"], visual=True),
    ),
    "hedgehog_in_cage": (
        # Panda base at x = -0.57: +x/+y moves away from it (reach proxy), so both lean negative.
        dict(path="center_xy_m", kind="offset", range=[[-.03, .01], [-.02, .01]]),
        dict(path="yaw_deg", kind="offset", range=[-5., 5.]),
        dict(path="wood_tint", kind="offset", range=[-.15, 0.], visual=True),
    ),
    "buzz_wire_3d": (
        # -y pushes the finish-side IK past tolerance and the far corner is 0.88 m from
        # the Panda base, so the fixture only moves -x/+y. No yaw: route frames are unrotated.
        dict(path="center_xy_m", kind="offset", range=[[-.015, 0.], [0., .015]]),
        dict(path="wood_tint", kind="offset", range=[-.2, 0.], visual=True),
        # Copper or brass; the steel loop stays distinct from either.
        dict(path="rail_rgb", kind="choice", values=[[.72, .33, .16], [.78, .60, .25]], visual=True),
    ),
    "fiber_bundling": (
        # The scene sits near x ~ 0.2, ~0.8 m from the Panda base: no +x.
        # Strand fan/bend/curl are already drawn from the episode seed by FiberLayout.
        dict(path="center_xy_m", kind="offset", range=[[-.02, 0.], [-.02, .02]]),
        # The copper ring and the red strand stay (the instructions name them); the jacket is black, dark grey, navy
        # or dark green, and the other strands' colours trade places.
        dict(path="sheath_color", kind="choice", visual=True, values=[
            [.026, .029, .032], [.24, .25, .27], [.05, .09, .22], [.07, .17, .10]]),
        dict(path="strand_palette_shift", kind="choice", values=[0, 2, 4], visual=True),
    ),
    "lock_picking": (
        # The chain lock (2026-10-10, it replaced the original tiers): each tier's geometry, stops and key tray are one
        # fixed design (tasks/lock_picking/chain.py, fixed_parameters), so the lock does not move; its
        # rotor count differs by tier (3 or 4), so the rotor colours keep their order. Only the base's finish varies.
        dict(path="base_finish", kind="choice", values=["coated_metal", "engineering_plastic", "silver"], visual=True),
    ),
    "orbits": (
        dict(path="yaw_deg", kind="offset", range=[-5., 5.]),
        # Goal region pinned to -y: the +y side already holds the handle.
        dict(path="base_center_xy_m", kind="offset", range=[[-.03, .03], [-.02, .02]]),
        # The handle stays wooden (the instruction names it).
        dict(path="cord_finish", kind="choice", values=["hemp", "orange_rope", "golden_rope"], visual=True),
    ),
    "rope_threading": (
        # The nominal scene fills 95-96% of cam_high/cam_low, and a turn swings the
        # 1.1 m rope's far end (5 deg -> ~5 cm), so this envelope is small.
        dict(path="geometry.translation_m", kind="offset", range=[[-.01, .02], [-.01, 0.], None]),
        dict(path="geometry.yaw_deg", kind="offset", range=[-2., 2.]),
        # Ring colours stay: the instruction orders the rings cyan, green, purple. The rope is hemp, blue or black, apart
        # from every ring and grommet (orange sat near the red grommet, gold near the amber ring; 2026-10-21).
        dict(path="rope_finish", kind="choice", values=["hemp", "blue_rope", "black_rope"], visual=True),
    ),
    "slotted_post_ring": (
        # FEM volumes cannot rotate, so translation only; -y beyond ~2.5 cm pushes the discs out of frame.
        dict(path="center_xy_m", kind="offset", range=[[-.05, .05], [-.02, .05]]),
        dict(path="cord_finish", kind="choice", values=["hemp", "orange_rope", "golden_rope"], visual=True),
    ),
    "interlocked_claws": (
        # No pose axes: the start is a two-hand pregrasp tied to arm joint angles and weld points.
        # Every piece's colour and material (InterlockedClawsTask.piece_finishes).
        dict(path="piece_finish", kind="choice", values=["painted", "anodised", "powder_coat"], visual=True),
    ),
    "blanket_flatten": (
        # Drawn before freezing: each seeded instance gets its own prepared archive
        # (scripts/prepare_blanket_initial_states.py --seeded). Coverage is footprint
        # area / 0.288 m^2, independent of position; -y would crowd the home grippers.
        dict(path="color", kind="choice", visual=True, values=[
            [.22, .46, .70], [.70, .20, .18], [.22, .52, .30], [.80, .62, .18], [.45, .47, .50]]),
        dict(path="texture.pattern", kind="choice", values=["grid", "stripes"], visual=True),
        dict(path="center_xy_m", kind="offset", range=[[-.04, .04], [0., .03]]),
    ),
    "blanket_fold": (
        # Looks only (the fold target images must be rendered in the drawn look); the cloth stays where its frozen
        # start and target put it, so there is no centre axis.
        dict(path="color", kind="choice", visual=True, values=[
            [.22, .46, .70], [.70, .20, .18], [.22, .52, .30], [.80, .62, .18], [.45, .47, .50]]),
        dict(path="texture.pattern", kind="choice", values=["grid", "stripes"], visual=True),
    ),
    "knot_untying": (
        # The knot sits at world x = 0, 13.6 cm right of the table centre: shift leans -x.
        # Its tails point -y, away from the arm line (y = -0.019): -y and a positive turn
        # push them out of reach (reach proxy), so both bounds are asymmetric.
        dict(path="origin_xy_m", kind="offset", range=[[-.06, .02], [-.01, .04]]),
        # Success counts projected crossings, invariant under a turn about z.
        dict(path="yaw_deg", kind="offset", range=[-30., 15.]),
        dict(path="rope_finish", kind="choice", values=["hemp", "orange_rope", "golden_rope"], visual=True),
    ),
    # The fixed-scene families below vary appearance only (appearance(); the user, 2026-10-20): every axis re-colours
    # visual geoms or meshes, never a colliding one; the values leave out the default and their order is the looks of
    # appearance seeds 1, 2, ... Colours the texts name stay (battery cells and red "+" marks; caliper labels, scale and
    # the knob's white line; Trinity's green goal region). Battery's pry stick follows props_finish since 2026-10-21 (its
    # texts no longer name its colours).
    "bed_bolt": (
        dict(path="wood_finish", kind="choice", values=["walnut", "white_painted", "white_oak"], visual=True),
        dict(path="bolt_finish", kind="choice", values=["zinc", "yellow_zinc"], visual=True),
        dict(path="key_finish", kind="choice", values=["black_oxide", "blue_coated"], visual=True),
    ),
    "caliper_sorting": (
        # The four sleeves stay alike (one finish for all); bins keep the white label cards.
        dict(path="fixture_finish", kind="choice", values=["forest_green", "safety_orange", "wood"], visual=True),
        dict(path="bin_finish", kind="choice", values=["dark_red", "dark_green"], visual=True),
        dict(path="sleeve_finish", kind="choice", values=["brass", "black_oxide"], visual=True),
    ),
    "clock_assembling": (
        dict(path="movement_finish", kind="choice", values=["rhodium", "gilt"], visual=True),
        dict(path="stand_finish", kind="choice", values=["walnut", "white_plastic"], visual=True),
        dict(path="screw_finish", kind="choice", values=["gilt", "steel"], visual=True),
    ),
    "wire_splicing": (
        # The texts name the wires red (A) and blue (B) and the post's yellow rod: the insulation and the rod stay; one
        # wire's bare core turns tinned at a time, so the two cores also tell apart up close.
        dict(path="terminal_finish", kind="choice", values=["white", "black", "grey"], visual=True),
        dict(path="post_finish", kind="choice", values=["steel", "green", "black"], visual=True),
        dict(path="core_finish", kind="choice", values=["copper", "b_tinned", "a_tinned"], visual=True),
    ),
    "battery_replacement": (
        dict(path="controller_finish", kind="choice", values=["white", "red", "blue"], visual=True),
        dict(path="tray_finish", kind="choice", values=["grey", "white"], visual=True),
        dict(path="box_finish", kind="choice", values=["blue", "green"], visual=True),
        dict(path="props_finish", kind="choice", values=["wood", "black"], visual=True),
    ),
}


def _defines(tree, path):
    for name in path.split("."):
        if not isinstance(tree, Mapping) or name not in tree:
            return False
        tree = tree[name]
    return True


def axes(task_class):
    """The task's axes whose leaf the class defines (cast_coil's v2 designs have no centre or yaw parameter)."""
    return tuple(axis for axis in AXES.get(task_class.task_id, ()) if _defines(task_class.defaults, axis["path"]))


def _stream_name(task_class, axis):
    return f"{'visual/' if axis.get('visual') else ''}{task_class.task_id}/{axis['path']}"


def _plain(value):
    """JSON-safe Python values; numpy scalars break the instance_id hash."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


def _assign(tree, path, value):
    *parents, leaf = path.split(".")
    for name in parents:
        tree = tree.setdefault(name, {})
    tree[leaf] = value


def draw(task_class, seed):
    """Nested task-parameter overrides for ``seed``; a pure function of (seed, axis name)."""
    rng = VariantRng(seed)
    values = {}
    for axis in axes(task_class):
        stream = rng.stream(_stream_name(task_class, axis))
        if axis["kind"] == "offset":
            base = _parameter_at(task_class.defaults, axis["path"])
            if np.ndim(base) == 0:
                lo, hi = axis["range"]
                value = float(base) + float(stream.uniform(lo, hi))
            else:
                value = [float(b) if r is None else float(b) + float(stream.uniform(*r))
                         for b, r in zip(base, axis["range"], strict=True)]
        elif axis["kind"] == "choice":
            value = axis["values"][int(stream.integers(len(axis["values"])))]
        else:
            raise ValueError(f"Unknown scene variation kind: {axis['kind']!r}")
        _assign(values, axis["path"], _plain(value))
    return values


def appearance(task_class, seed):
    """Task parameters of appearance seed ``seed`` for a fixed-scene base whose task seed stays 0, visual axes only: {}
    for seed 0 (the scene as its reference was recorded); else each choice axis takes its values other than the default
    in order, seed 1 the first (so two consecutive seeds differ on every choice axis: independent draws for seeds 1 and 2
    can coincide), and each offset axis the seed's draw."""
    if seed == 0:
        return {}
    drawn, values = draw(task_class, seed), {}
    for axis in axes(task_class):
        if not axis.get("visual"):
            continue
        if axis["kind"] == "choice":
            default = _plain(_parameter_at(task_class.defaults, axis["path"]))
            others = [v for v in axis["values"] if _plain(v) != default]
            value = others[(seed - 1) % len(others)]
        else:
            value = _parameter_at(drawn, axis["path"])
        _assign(values, axis["path"], _plain(value))
    values["scene_look"] = scene_look(task_class, seed)
    return values


def scene_look(task_class, seed):
    """The scene's look of appearance seed ``seed`` >= 1 (topogym.world.look; the user, 2026-10-27: white light that
    changes colour temperature, the table top and the floor change finish). Each base draws, once for all its seeds,
    which seed is warm and which brighter and the orders of key lights, table tops and floors; seed S takes the S-th of
    each, so seeds 1 and 2 differ in every part. The bands leave a gap round the released look (about 6000 K,
    brightness 1, key light 0, height 2.40 m) so that no seed looks like seed 0, and stop short of dark, glaring or
    strongly tinted scenes (renders of cast-keyhole-02, 2026-10-27: below 3600 K the green goal dashes turn yellow,
    6500-7000 K is not told from seed 0, brightness moves the frame little under the Reinhard tone mapper): warm
    3600-4300 K or cool 8000-10000 K, brightness 0.65-0.78 or 1.25-1.50, height 1.9-2.4 m."""
    from topogym.world.look import FLOOR_FINISHES, SCENE_LOOK_VERSION, TABLE_EXCLUDED, TABLE_FINISHES
    if seed < 1:
        raise ValueError("seed 0 has no scene_look: it is the released look")
    name = f"visual/scene_look/{task_class.task_id}/{task_class.variation_id}"
    shared, own = VariantRng(0).stream(name), VariantRng(seed).stream(name)
    warm_seed, bright_seed = (int(v) for v in shared.integers(2, size=2))
    keys = shared.permutation(3) + 1
    tables = shared.permutation(sorted(t for t in list(TABLE_FINISHES)[1:]
                                       if t not in TABLE_EXCLUDED.get(task_class.task_id, ())))
    floors = shared.permutation(sorted(list(FLOOR_FINISHES)[1:]))
    warm, bright = seed % 2 == warm_seed, seed % 2 == bright_seed
    lighting = dict(key=int(keys[(seed - 1) % 3]),
                    height_m=round(float(own.uniform(1.9, 2.4)), 3),
                    cct_k=round(float(own.uniform(3600., 4300.) if warm else own.uniform(8000., 10000.))),
                    brightness=round(float(own.uniform(1.25, 1.50) if bright else own.uniform(.65, .78)), 3))
    return dict(version=SCENE_LOOK_VERSION, lighting=lighting, table=str(tables[(seed - 1) % len(tables)]),
                floor=str(floors[(seed - 1) % len(floors)]))


def within_envelope(task_class, config, *, atol=1e-9):
    """True if ``config`` equals the tier defaults except for in-envelope axis leaves."""
    listed = {axis["path"]: axis for axis in axes(task_class)}
    for path, axis in listed.items():
        value, base = _parameter_at(config, path), _parameter_at(task_class.defaults, path)
        if axis["kind"] == "choice":
            if _plain(value) not in [_plain(v) for v in axis["values"]] and _plain(value) != _plain(base):
                return False
            continue
        ranges = [axis["range"]] if np.ndim(base) == 0 else axis["range"]
        delta = np.atleast_1d(np.asarray(value, float) - np.asarray(base, float))
        for d, r in zip(delta, ranges, strict=True):
            if (r is None and abs(d) > atol) or (r is not None and not r[0] - atol <= d <= r[1] + atol):
                return False

    def rest(tree, prefix=""):
        # Everything outside the listed axis leaves, flattened by dotted path.
        out = {}
        for key, value in tree.items():
            path = f"{prefix}{key}"
            if path in listed or path == "scene_look":      # the scene's look (scene_look) is not a scene variation
                continue
            if isinstance(value, Mapping):
                out.update(rest(value, path + "."))
            else:
                out[path] = value
        return out

    return rest(_plain_tree(config)) == rest(_plain_tree(task_class.defaults))


def _plain_tree(tree):
    return {k: _plain_tree(v) if isinstance(v, Mapping) else _plain(v) for k, v in tree.items()}
