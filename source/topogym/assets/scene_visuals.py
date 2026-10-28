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
# (appearance seeds, 2026-10-20), apart from the silver horseshoes and the default copper ring. blue_nylon_rope,
# black_polyester_rope: rope and cord looks of appearance seeds 1 and 2 (2026-10-27): another colour
# and another fibre than hemp or polypropylene (braided nylon has a sheen, polyester a softer one), apart from every
# ring and grommet colour and the green goal region, and >= 15 CIEDE2000 from every table top a seed draws.
LATE_SURFACES = {
    "brass": dict(color=(.78, .60, .25, 1.), metallic=.9, roughness=.35),
    "blue_nylon_rope": dict(color=(.04, .12, .60, 1.), metallic=0., roughness=.45),
    "black_polyester_rope": dict(color=(.03, .03, .035, 1.), metallic=0., roughness=.6),
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


def scale_glb_color(path, rgb_scale, *, materials=None, color=None):
    """Rescale (or with ``color`` replace) GLB base colours in place; buffers stay byte-identical.

    Only the glTF JSON chunk is rewritten, so textures and geometry are untouched.
    ``materials`` limits the edit to materials whose name contains one of the
    given substrings (unnamed materials match only when ``materials`` is None).
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
    chunk = json.dumps(document, separators=(",", ":")).encode()
    chunk += b" "*(-len(chunk) % 4)
    rest = data[20+length:]
    path.write_bytes(struct.pack("<4sII", b"glTF", version, 20+len(chunk)+len(rest))
                     + struct.pack("<I4s", len(chunk), b"JSON") + chunk + rest)


def scale_urdf_color(path, rgb_scale):
    """Rescale a URDF's visual material colours in place, as scale_glb_color does a GLB's.

    A URDF visual's own ``<material><color>`` is the colour Nyx draws for its mesh,
    not the mesh file's baseColorFactor, so a tint must change both. Only the
    ``rgba`` values inside ``<visual>`` change; the rest of the file keeps its bytes.
    """
    import re
    from pathlib import Path
    path = Path(path)

    def colour(match):
        base = [float(v) for v in match.group(2).split()]
        return match.group(1) + " ".join(str(min(1., base[i]*float(rgb_scale))) for i in range(3)) \
            + f" {base[3]:g}" + match.group(3)

    def visual(match):
        return re.sub(r'(<color\s+rgba=")([^"]+)(")', colour, match.group(0))
    path.write_text(re.sub(r"<visual\b.*?</visual>", visual, path.read_text(), flags=re.S))


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
