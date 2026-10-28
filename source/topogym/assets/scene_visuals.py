"""Explicit non-physical finishes for the default task scenes."""
from copy import deepcopy

SURFACES = {
    "engineering_plastic": dict(color=(.72, .75, .73, 1.), metallic=0., roughness=.65),
    "coated_metal": dict(color=(.10, .16, .19, 1.), metallic=.3, roughness=.5),
    "silver": dict(color=(.70, .74, .78, 1.), metallic=.9, roughness=.3),
    "copper": dict(color=(.72, .33, .16, 1.), metallic=.9, roughness=.36),
    "rubber": dict(color=(.026, .029, .032, 1.), metallic=0., roughness=.8),
    "hemp": dict(color=(.65, .39, .13, 1.), metallic=0., roughness=.9),
    "orange_rope": dict(color=(.82, .27, .045, 1.), metallic=0., roughness=.88),
    "golden_rope": dict(color=(.92, .61, .035, 1.), metallic=0., roughness=.88),
}
# Roles added after scene_preset.json began recording SURFACES: kept out of it so every existing scene's record stays
# byte-identical; record_preset adds the ones a scene's parameters name. brass: Horseshoe's third target-ring look
# (appearance seeds, 2026-10-20 to 10-27; 18 CIEDE2000 from copper). red_painted: Horseshoe's target ring at appearance
# seed 1 (2026-10-27). black_oxide, bronze: Horseshoe's horseshoes and chains at appearance seeds 1 and 2; brass,
# gunmetal: Lock Picking's key; teak, green_powder_coat: its base; blue_painted, plywood: its stop frames and key tray
# (2026-10-27). blue_rope, black_rope: Rope Threading's rope looks of appearance seeds 1 and 2 (2026-10-21), apart from
# every ring and grommet colour.
LATE_SURFACES = {
    "brass": dict(color=(.78, .60, .25, 1.), metallic=.9, roughness=.35),
    "red_painted": dict(color=(.55, .05, .06, 1.), metallic=.3, roughness=.4),
    "black_oxide": dict(color=(.04, .04, .045, 1.), metallic=.7, roughness=.5),
    "bronze": dict(color=(.45, .28, .12, 1.), metallic=.9, roughness=.4),
    "teak": dict(color=(.42, .24, .10, 1.), metallic=0., roughness=.6),
    "green_powder_coat": dict(color=(.07, .25, .12, 1.), metallic=0., roughness=.6),
    "blue_painted": dict(color=(.20, .50, .85, 1.), metallic=0., roughness=.5),
    "plywood": dict(color=(.55, .38, .20, 1.), metallic=0., roughness=.8),
    "gunmetal": dict(color=(.16, .17, .19, 1.), metallic=.8, roughness=.45),
    "blue_rope": dict(color=(.04, .12, .60, 1.), metallic=0., roughness=.88),
    "black_rope": dict(color=(.03, .03, .035, 1.), metallic=0., roughness=.88),
}


def surface(role):
    """Fresh surface options; never mutate physical friction/density/constraints."""
    try:
        return deepcopy({**SURFACES, **LATE_SURFACES}[role])
    except KeyError as exc:
        raise ValueError(f"Unknown task visual role: {role}") from exc


def finish_generated_urdf(path, role):
    """Finish generated visuals only; leave shared generators and colliders alone."""
    from pathlib import Path
    import xml.etree.ElementTree as ET
    import numpy as np
    import trimesh
    path = Path(path).resolve()
    options = surface(role)
    tree = ET.parse(path)
    files = []
    for visual in tree.findall(".//visual"):
        node = visual.find("geometry/mesh")
        if node is None:
            raise ValueError("Expected a generated mesh visual")
        source = Path(node.get("filename"))
        if not source.is_absolute():
            source = path.parent/source
        mesh = trimesh.load(source, force="mesh", process=False)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(mesh.vertices), 2)),
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=options["color"], metallicFactor=options["metallic"],
                roughnessFactor=options["roughness"]))
        target = source.with_name(source.stem+"_"+role+".glb")
        mesh.export(target)
        node.set("filename", target.relative_to(path.parent).as_posix())
        for old in list(visual.findall("material")):
            visual.remove(old)
        files.append(target.name)
    ET.indent(tree)
    tree.write(path, encoding="unicode", xml_declaration=True)
    return dict(role=role, files=files, surface=options)


def scale_glb_color(path, rgb_scale, *, materials=None, color=None, metallic=None, roughness=None, texture=True):
    """Rescale (or with ``color`` replace) GLB base colours in place; buffers stay byte-identical.
    ``metallic`` / ``roughness`` replace the factors; ``texture=False`` drops the base colour texture (a painted or
    metal finish over a wood mesh).

    Only the glTF JSON chunk is rewritten, so textures and geometry are untouched.
    ``materials`` limits the edit to materials whose name contains one of the
    given substrings (unnamed materials match only when ``materials`` is None).
    ``roughness`` / ``metallic`` (None: kept) also set those factors of the edited materials.
    """
    import json
    import struct
    from pathlib import Path
    path = Path(path)
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    length, kind = struct.unpack_from("<I4s", data, 12)
    if magic != b"glTF" or kind != b"JSON":
        raise ValueError(f"Not a binary glTF file: {path}")
    document = json.loads(data[20:20+length])
    for material in document.get("materials", []):
        name = material.get("name") or ""
        if materials is not None and not any(key in name for key in materials):
            continue
        pbr = material.setdefault("pbrMetallicRoughness", {})
        base = list(color) if color is not None else list(pbr.get("baseColorFactor", [1., 1., 1., 1.]))
        scale = [float(rgb_scale)]*3 if isinstance(rgb_scale, (int, float)) else [float(v) for v in rgb_scale]
        pbr["baseColorFactor"] = [min(1., float(base[i])*scale[i]) for i in range(3)] + [float(base[3])]
        if metallic is not None:
            pbr["metallicFactor"] = float(metallic)
        if roughness is not None:
            pbr["roughnessFactor"] = float(roughness)
        if not texture:
            pbr.pop("baseColorTexture", None)
    chunk = json.dumps(document, separators=(",", ":")).encode()
    chunk += b" "*(-len(chunk) % 4)
    rest = data[20+length:]
    path.write_bytes(struct.pack("<4sII", b"glTF", version, 20+len(chunk)+len(rest))
                     + struct.pack("<I4s", len(chunk), b"JSON") + chunk + rest)


def scale_urdf_color(path, rgb_scale, *, color=None):
    """Rescale (or with ``color`` replace) a URDF's visual material colours in place, as scale_glb_color does a GLB's.

    A URDF visual's own ``<material><color>`` is the colour Nyx draws for its mesh,
    not the mesh file's baseColorFactor, so a tint must change both. Only the
    ``rgba`` values inside ``<visual>`` change; the rest of the file keeps its bytes.
    """
    import re
    from pathlib import Path
    path = Path(path)

    def colour(match):
        base = [float(v) for v in match.group(2).split()]
        if color is not None:
            base[:3] = color[:3]
        return match.group(1) + " ".join(str(min(1., base[i]*float(rgb_scale))) for i in range(3)) \
            + f" {base[3]:g}" + match.group(3)

    def visual(match):
        return re.sub(r'(<color\s+rgba=")([^"]+)(")', colour, match.group(0))
    path.write_text(re.sub(r"<visual\b.*?</visual>", visual, path.read_text(), flags=re.S))


# Finishes of the Cast puzzles' two pieces (cast_coil, cast_keyhole; task parameter piece_finish, appearance seeds): per
# finish, each piece's (colour, roughness, metallic), keyed by the metal it has in the released look ("cast": the
# generated gold and silver, roughness 0.30-0.34, metallic 0.80-0.82). No text names a piece's colour (2026-10-27): the
# gold piece turns red anodised aluminium, then matte black powder coat; the silver piece blue anodised aluminium, then
# orange powder coat. Each piece's three colours, the two pieces of one seed and every piece and the green goal region
# stay >= 20 CIEDE2000 apart. Nyx draws a generated piece's URDF colour with its GLB's roughness and metallic factors
# (renders of cast-keyhole-02, 2026-10-27: the entity surface's roughness and metallic do not reach a URDF entity), so
# both files change.
CAST_PIECE_FINISHES = {
    "cast": None,
    "anodised": dict(gold=((.55, .05, .06), .35, .6), silver=((.05, .15, .55), .35, .6)),
    "powder_coat": dict(gold=((.035, .035, .04), .75, 0.), silver=((.80, .25, .03), .7, 0.)),
}


def refinish_cast_piece(urdf, metal, finish):
    """Give a generated Cast piece (its URDF and the GLBs beside it), gold or silver in the released look, its
    CAST_PIECE_FINISHES finish, in place."""
    from pathlib import Path
    color, roughness, metallic = CAST_PIECE_FINISHES[finish][metal]
    urdf = Path(urdf)
    for path in sorted(urdf.parent.glob("*.glb")):
        scale_glb_color(path, 1., color=(*color, 1.), roughness=roughness, metallic=metallic)
    scale_urdf_color(urdf, 1., color=color)


def localize_generated_urdfs(folder):
    """Make generated mesh references reproducible across episode directories."""
    import os
    from pathlib import Path
    import xml.etree.ElementTree as ET
    folder = Path(folder).resolve()
    for path in sorted(folder.rglob("*.urdf")):
        tree = ET.parse(path)
        for node in tree.findall(".//mesh"):
            source = Path(node.get("filename"))
            if not source.is_absolute():
                source = path.parent/source
            source = source.resolve()
            if not source.is_relative_to(folder) or not source.is_file():
                raise ValueError(f"Generated mesh is outside its asset folder: {source}")
            node.set("filename", Path(os.path.relpath(source, path.parent)).as_posix())
        ET.indent(tree)
        tree.write(path, encoding="unicode", xml_declaration=True)


def visual_collision_urdf(folder, name):
    """Keep textured UV seams out of collision topology and inertia integration.

    The existing OBJ is the exact collision mesh. Genesis computes the missing
    inertia from that mesh and the same Rigid material density as a Mesh morph.
    """
    from pathlib import Path
    import xml.etree.ElementTree as ET
    folder = Path(folder)
    for suffix in ("obj", "glb"):
        if not (folder/f"{name}.{suffix}").is_file():
            raise FileNotFoundError(folder/f"{name}.{suffix}")
    root = ET.Element("robot", name=name)
    link = ET.SubElement(root, "link", name=name)
    for tag, suffix in (("visual", "glb"), ("collision", "obj")):
        geom = ET.SubElement(ET.SubElement(link, tag), "geometry")
        ET.SubElement(geom, "mesh", filename=f"{name}.{suffix}")
    ET.indent(root)
    path = folder/f"{name}.urdf"
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)
    return path
