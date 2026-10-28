"""The scene's look under an appearance seed: room lighting, table top and floor (task parameter ``scene_look``).

Seed 0 has no ``scene_look``: the compact_lab presentation as released. Seeds 1, 2, ... carry one (drawn by
topogym.tasks.scene_variation.scene_look), which load_scene_config applies to the renderer config only:

  lighting  the four ceiling point lights of compact_lab keep their x, y; ``key`` names the one that is the key light
            (3x the fill intensity, the only one casting shadows; compact_lab's is light 0), ``height_m`` the height
            of all four, ``cct_k`` their colour temperature (white light, the Planckian locus), ``brightness`` scales
            the four lights and the environment map together; the environment map takes the lights' colour as tint.
  table     a TABLE_FINISHES name for the table top ("table");
  floor     a FLOOR_FINISHES name for the room's floor ("lab_floor").

Nothing here touches a colliding geom, the MuJoCo model or the task's own objects (their colours and materials are the
task's appearance axes). ``pbr_materials`` tells nyx_direct to draw the roughness and metalness that MJCF materials and
mesh surfaces declare, which the released look leaves out (albedo only), so seed 0's frames stay as they were.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

SCENE_LOOK_VERSION = 1

# Table tops, linear RGB; the first is compact_lab's own (seed 0's), which no seed >= 1 draws. All light to mid tones:
# a dark top would hide the dark ropes and cords several families use.
TABLE_FINISHES = {
    "grey_laminate": dict(color=[.83, .84, .82], metallic=0., roughness=.42),
    "beige_laminate": dict(color=[.78, .66, .48], metallic=0., roughness=.45),
    "maple_laminate": dict(color=[.72, .50, .30], metallic=0., roughness=.45),
    "pale_blue_laminate": dict(color=[.45, .58, .78], metallic=0., roughness=.40),
    "warm_grey_stone": dict(color=[.40, .38, .35], metallic=0., roughness=.65),
    "mid_grey_laminate": dict(color=[.30, .31, .32], metallic=0., roughness=.45),
    "brushed_steel": dict(color=[.72, .72, .73], metallic=1., roughness=.45),
}
# Table tops a family's seeds never draw: those its objects would not stand out from (task_id -> names).
TABLE_EXCLUDED: dict[str, tuple] = {}
# Floors; the first is compact_lab's blue-grey tiles (seed 0's). The grout lines stay.
FLOOR_FINISHES = {
    "blue_grey_tile": dict(color=[.32, .37, .39], metallic=0., roughness=.70),
    "warm_grey_tile": dict(color=[.42, .40, .37], metallic=0., roughness=.60),
    "sand_tile": dict(color=[.58, .52, .42], metallic=0., roughness=.60),
    "dark_slate": dict(color=[.17, .18, .19], metallic=0., roughness=.55),
    "terracotta_tile": dict(color=[.50, .30, .22], metallic=0., roughness=.65),
    "light_grey_vinyl": dict(color=[.62, .63, .62], metallic=0., roughness=.35),
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
    if not isinstance(look, Mapping) or set(look) != {"version", "lighting", "table", "floor"}:
        raise ValueError("scene_look must have exactly version, lighting, table and floor")
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
    if look["table"] not in TABLE_FINISHES or look["floor"] not in FLOOR_FINISHES:
        raise ValueError(f"unknown scene_look table {look['table']!r} or floor {look['floor']!r}")
    return dict(version=look["version"], lighting=dict(lighting), table=look["table"], floor=look["floor"])


def apply_scene_look(scene, look) -> None:
    """Apply ``look`` to a composed tabletop config (load_scene_config): the renderer's lights and environment map,
    and two finish rules (env.appearance, after the presentation's) for the table top and the floor."""
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
    current = scene.env.get("appearance")
    rules = [] if current is None else list(OmegaConf.to_container(current, resolve=True))
    rules += [dict(pattern="table", **TABLE_FINISHES[look["table"]]),
              dict(pattern="lab_floor", **FLOOR_FINISHES[look["floor"]])]
    with open_dict(scene.env):
        scene.env.appearance = rules
