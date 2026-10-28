"""The scene's look under an appearance seed: room lighting, table top, its frame and floor (task parameter
``scene_look``).

Seed 0 has no ``scene_look``: the compact_lab presentation as released. Seeds 1, 2, ... carry one (drawn by
topogym.tasks.scene_variation.scene_look), which load_scene_config applies to the renderer config only:

  lighting  the four ceiling point lights of compact_lab keep their x, y; ``key`` names the one that is the key light
            (3x the fill intensity, the only one casting shadows; compact_lab's is light 0), ``height_m`` the height
            of all four, ``cct_k`` their colour temperature (white light, the Planckian locus), ``brightness`` scales
            the four lights and the environment map together; the environment map takes the lights' colour as tint.
  table     a TABLE_FINISHES name for the table top ("table");
  floor     a FLOOR_FINISHES name for the room's floor ("lab_floor");
  frame     a FRAME_FINISHES name for the table top's rim and apron ("table_frame*").

A finish is a material class (wood, metal, stone, plastic; floors wood, tile, concrete, carpet) and either a plain
colour with roughness and metalness or a textured PBR material: colour, normal and ARM (ambient occlusion, roughness,
metalness) maps of a CC0 Poly Haven texture, its tile size (``tile_m``: the texture's real width) setting how often it
repeats over the table top or the floor. The maps are visual assets (topogym.assets.library.visual_asset_path).

Nothing here touches a colliding geom, the MuJoCo model or the task's own objects (their colours and materials are the
task's appearance axes). ``pbr_materials`` tells nyx_direct to draw the roughness and metalness that MJCF materials and
mesh surfaces declare, which the released look leaves out (albedo only), so seed 0's frames stay as they were.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

SCENE_LOOK_VERSION = 1

def _texture(cls, name, tile_m, **more):
    """A textured finish of Poly Haven texture ``name`` (CC0; its 1k maps <name>_{diff,nor_gl,arm}_1k.jpg). Nyx
    multiplies the ARM map's roughness and metalness by the factors, which stay 1."""
    return dict(cls=cls, color=[1., 1., 1.], metallic=1., roughness=1., diffuse_map=f"{name}_diff_1k.jpg",
                normal_map=f"{name}_nor_gl_1k.jpg", arm_map=f"{name}_arm_1k.jpg", tile_m=tile_m, **more)


# Table tops; the first is compact_lab's own plastic laminate (seed 0's). Seeds >= 1 draw a class other than plastic,
# seeds 1 and 2 two different classes (the user, 2026-10-27: wood to metal or plastic, as RoboDojo's table materials).
# lum: cam_high's mean luma of cast-keyhole-02 with this top under the released lights (2026-10-27): of seeds 1 and 2,
# the one with the darker top takes the brighter light. A painted blue steel plate (0.31, with glare) was left out, and
# granite tiles (lum 0.38: 3 of 16 seeds drawn on it fell below the 0.35 exposure floor, the darkest a black rope on
# their speckle, renders of 2026-10-28) were dropped.
TABLE_FINISHES = {
    "grey_laminate": dict(cls="plastic", color=[.83, .84, .82], metallic=0., roughness=.42, lum=.62),
    "ash_veneer": _texture("wood", "ash_veneer", 1.0, lum=.50),
    "cherry_veneer": _texture("wood", "cherry_veneer", 1.0, lum=.54),
    "coated_pine": _texture("wood", "coated_pine", .74, lum=.33),
    "brushed_steel": dict(cls="metal", color=[.72, .72, .73], metallic=1., roughness=.35, lum=.53),
    "satin_aluminium": dict(cls="metal", color=[.80, .81, .83], metallic=1., roughness=.55, lum=.55),
    "terrazzo": _texture("stone", "terrazzo_tiles", 2.0, lum=.43),
    "brushed_concrete": _texture("stone", "brushed_concrete", 2.5, lum=.40),
}
# Table tops a family's seeds never draw: those its objects would not stand out from (task_id -> names).
TABLE_EXCLUDED: dict[str, tuple] = {
    # its wrist cameras look down onto the table at close range past dark wooden parts (the owl): the darkest top
    # left hedgehog-07's seed 1 wrist view below the exposure floor (0.338, 2026-10-28)
    "hedgehog_in_cage": ("coated_pine",),
}
# The dim seed's lowest brightness per family where its own parts fill much of the frame (default 0.65): bed_bolt's
# frame takes a third of cam_high, and a charcoal frame on brushed concrete under 0.74 stayed at mean luma 0.315.
DIM_FLOOR: dict[str, float] = {"bed_bolt": .9}
# A table top whose lum is below this is dark: a dim seed drawn on it dims to 0.85-0.93 only (all families).
DARK_TOP_LUM = .45
# Floors; the first is compact_lab's blue-grey tiles (seed 0's), its grout lines staying under every floor. Seeds 1 and 2
# draw two different classes.
FLOOR_FINISHES = {
    "blue_grey_tile": dict(cls="tile", color=[.32, .37, .39], metallic=0., roughness=.70),
    "parquet": _texture("wood", "diagonal_parquet", 2.36),
    "beige_tiles": _texture("tile", "interior_tiles", 1.9),
    "terracotta_tiles": _texture("tile", "terracotta_floor_tiles", 2.08),
    "concrete": _texture("concrete", "brushed_concrete", 2.5),
    "carpet": _texture("carpet", "poly_wool_herringbone", .27),
}
# The table top's rim and apron ("table_frame*"); the first is compact_lab's dark wood (seed 0's). Seeds 1 and 2 draw two
# different ones (the user, 2026-10-27: the frame changes colour and material too).
FRAME_FINISHES = {
    "dark_wood": dict(cls="wood", color=[1., 1., 1.], metallic=0., roughness=.45, diffuse_map="wood_table_001_diff_1k.png",
                      normal_map="wood_table_001_nor_gl_1k.png"),
    "light_wood": dict(cls="wood", color=[1., 1., 1.], metallic=1., roughness=1., diffuse_map="ash_veneer_diff_1k.jpg",
                       normal_map="ash_veneer_nor_gl_1k.jpg", arm_map="ash_veneer_arm_1k.jpg"),
    "black_powder_coat": dict(cls="metal", color=[.04, .04, .045], metallic=.6, roughness=.45),
    "aluminium": dict(cls="metal", color=[.80, .81, .83], metallic=1., roughness=.35),
    "white_edge_band": dict(cls="plastic", color=[.88, .88, .86], metallic=0., roughness=.30),
    "grey_rubber": dict(cls="rubber", color=[.22, .23, .24], metallic=0., roughness=.85),
    "blue_painted_steel": dict(cls="metal", color=[.10, .22, .50], metallic=.4, roughness=.40),
}
KEY_INTENSITY, FILL_INTENSITY = 3., 1.          # compact_lab's key and fill lights
LIMITS = dict(height_m=(1.5, 2.5), cct_k=(2000., 10000.), brightness=(.5, 1.5))   # sanity bounds, not the draw's bands


def cct_rgb(kelvin: float) -> list[float]:
    """Linear sRGB of a black body at ``kelvin`` (CIE 1931 xy on the Planckian locus by Kim et al. 2002's cubic
    fit, 1667-25000 K), scaled so its largest channel is 1."""
    t = float(kelvin)
    if not 1667. <= t <= 25000.:
        raise ValueError(f"colour temperature {t} K is outside 1667-25000 K")
    if t <= 4000.:
        x = -.2661239e9 / t**3 - .2343589e6 / t**2 + .8776956e3 / t + .179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + .2226347e3 / t + .240390
    if t <= 2222.:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - .20219683
    elif t <= 4000.:
        y = -.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - .16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - .37001483
    xyz = np.array([x / y, 1., (1. - x - y) / y])
    rgb = np.array([[3.2406, -1.5372, -.4986], [-.9689, 1.8758, .0415], [.0557, -.2040, 1.0570]]) @ xyz
    rgb = np.clip(rgb, 0., None)
    return [round(float(v), 4) for v in rgb / rgb.max()]


def check_scene_look(look) -> dict:
    """``look`` as a plain dict, or ValueError when it is not a scene_look of this version."""
    if not isinstance(look, Mapping) or set(look) != {"version", "lighting", "table", "floor", "frame"}:
        raise ValueError("scene_look must have exactly version, lighting, table, floor and frame")
    if look["version"] != SCENE_LOOK_VERSION:
        raise ValueError(f"scene_look version {look['version']!r} is not {SCENE_LOOK_VERSION}")
    lighting = look["lighting"]
    if not isinstance(lighting, Mapping) or set(lighting) != {"key", "height_m", "cct_k", "brightness"}:
        raise ValueError("scene_look.lighting must have exactly key, height_m, cct_k and brightness")
    if isinstance(lighting["key"], bool) or lighting["key"] not in (0, 1, 2, 3):
        raise ValueError("scene_look.lighting.key must be 0, 1, 2 or 3")
    for name, (lo, hi) in LIMITS.items():
        value = lighting[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
            raise ValueError(f"scene_look.lighting.{name} must be in [{lo}, {hi}]")
    if look["table"] not in TABLE_FINISHES or look["floor"] not in FLOOR_FINISHES or look["frame"] not in FRAME_FINISHES:
        raise ValueError(f"unknown scene_look table {look['table']!r}, floor {look['floor']!r} or frame {look['frame']!r}")
    return dict(version=look["version"], lighting=dict(lighting), table=look["table"], floor=look["floor"],
                frame=look["frame"])


def apply_scene_look(scene, look) -> None:
    """Apply ``look`` to a composed tabletop config (load_scene_config): the renderer's lights and environment map,
    and three finish rules (env.appearance, after the presentation's) for the table top, the floor and the table's
    rim and apron."""
    from omegaconf import OmegaConf, open_dict
    look = check_scene_look(look)
    lighting, renderer = look["lighting"], scene.env.renderer
    rgb = cct_rgb(lighting["cct_k"])
    released = OmegaConf.to_container(renderer.lights, resolve=True)
    if len(released) != 4 or any(light["type"] != "point" for light in released):
        raise ValueError("scene_look expects compact_lab's four point lights")
    lights = [dict(type="point", pos=[float(light["pos"][0]), float(light["pos"][1]), float(lighting["height_m"])],
                   intensity=(KEY_INTENSITY if i == lighting["key"] else FILL_INTENSITY) * float(lighting["brightness"]),
                   color=rgb, shadow=i == lighting["key"]) for i, light in enumerate(released)]
    with open_dict(renderer):
        renderer.lights = lights
        renderer.env_intensity = float(renderer.env_intensity) * float(lighting["brightness"])
        renderer.env_tint = rgb
        renderer.pbr_materials = True
    from topogym.world.lab import compact_lab_layout
    floor = next(item for item in compact_lab_layout() if item["name"] == "lab_floor")
    current = scene.env.get("appearance")
    rules = [] if current is None else list(OmegaConf.to_container(current, resolve=True))
    rules += [_rule("table", TABLE_FINISHES[look["table"]], scene.env.world.table.size),
              _rule("lab_floor", FLOOR_FINISHES[look["floor"]], floor["size"]),
              _rule("table_frame*", FRAME_FINISHES[look["frame"]], None)]
    with open_dict(scene.env):
        scene.env.appearance = rules


def _rule(pattern, finish, size):
    """nyx_direct's finish rule of ``finish`` on the box of ``size`` (x, y, z): a texture repeats once per tile_m."""
    rule = {k: v for k, v in finish.items() if k not in ("cls", "tile_m", "lum")}
    if "tile_m" in finish:
        rule["uv_scale"] = [round(float(size[0]) / finish["tile_m"], 4), round(float(size[1]) / finish["tile_m"], 4)]
    return dict(pattern=pattern, **rule)
