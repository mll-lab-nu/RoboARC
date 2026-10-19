"""Official-style Cast Coil pieces (the official pieces): square-bar cells plus one round pin per piece.

From the pieces' record (resource variations/cast_coil/design_6.json): every cell becomes a box and every pin
a cylinder, along the piece's y axis by default: a pin entry is [x centre, z centre, y start, y end] in cells, or with a
fifth entry "x" or "z" a pin along that axis, [y centre, z centre, x start, x end, "x"] or [x centre, y centre, z start,
z end, "z"] (the other two coordinates in order). Each exposed face (no cell of the same piece behind it) is set back by half the
clearance, so the assembled pieces keep `clearance_m` between them while the seams between neighbouring cells of one
piece stay flush (a set-back seam would be a groove the other piece's edges could catch on). One URDF per piece: a
single link, named like the piece's entity, whose frame is the piece's cell frame in metres, with the boxes and the pin
as collision primitives, the inertia of the uniform-density piece and one visual mesh (GLB: the union of the boxes
plus the pin, flat shaded, in the Genesis tiers' gold and silver metal finish); assembled, both piece frames coincide.
"""
from pathlib import Path

import numpy as np

# The Genesis tiers' finish (legacy/cast_coil_genesis_20261009: the coil template's colours, PBR metallic 0.82,
# roughness 0.34), so the render bridge's Nyx frames show the pieces as today's frames do.
COLOURS = dict(gold=(.53, .39, .12, 1.), silver=(.39, .43, .47, 1.))
METAL = dict(metallicFactor=.82, roughnessFactor=.34)
PIN_SECTIONS = 64
PIECE_NAMES = dict(silver="coil_silver", gold="coil_gold")
BLOCK_CENTRE_CELLS = np.array([1.5, 2., 2.])
PIN_PLANE = ((1, 2), (0, 2), (0, 1))   # the two coordinates across a pin along x, y, z, in a pin entry's order


def block_centre_cells(record):
    """The assembled block's centre in the piece frame (cells): the record's, else the official 3 x 4 x 4 block's.
    Here, not in topogym/planning, because the task places its pieces with it and the image drops topogym/planning."""
    return np.asarray(record.get("block_centre_cells", BLOCK_CENTRE_CELLS), float)


def piece_primitives(record, piece):
    """[(kind, centre_m, size)] in the piece frame: ("box", centre, half extents) or ("cylinder", from, to, radius)."""
    b, gap = float(record["cell_m"]), float(record["clearance_m"])
    cells = {tuple(c) for c in record["pieces"][piece]["cells"]}
    shapes = []
    for c in sorted(cells):
        low, high = np.asarray(c, float), np.asarray(c, float)+1
        for axis in range(3):
            step = np.eye(3, dtype=int)[axis]
            if tuple(np.asarray(c)-step) not in cells:
                low[axis] += gap/2/b
            if tuple(np.asarray(c)+step) not in cells:
                high[axis] -= gap/2/b
        shapes.append(("box", (low+high)/2*b, (high-low)/2*b, c))
    radius = float(record["pin_radius_cells"])*b - gap/2
    for pin in record["pieces"][piece]["pins"]:
        axis = "xyz".index(pin[4]) if len(pin) > 4 else 1
        (u, v), start = PIN_PLANE[axis], np.zeros(3)
        start[u], start[v], start[axis] = pin[0], pin[1], pin[2]
        end = start.copy()
        end[axis] = pin[3]
        start, end = start*b, end*b
        start[axis] += gap/2
        end[axis] -= gap/2
        shapes.append(("cylinder", start, end, radius))
    return shapes


def pin_axis(pin):
    """0, 1 or 2: the axis a ("cylinder", start, end, radius) primitive runs along."""
    return int(np.argmax(np.abs(np.asarray(pin[2], float)-np.asarray(pin[1], float))))


def pin_free_segments(pin, boxes):
    """[(start, end)] along its axis of a pin outside the piece's cells. The pins pass through whole cells: where the
    axis is inside a cell, the pin's cross-section is too (a partial overlap is refused)."""
    _, start, end, radius = pin
    axis = pin_axis(pin)
    across = PIN_PLANE[axis]
    y0, y1 = sorted((float(start[axis]), float(end[axis])))
    covered = []
    for _, centre, half, _ in boxes:
        lo, hi = np.asarray(centre)-np.asarray(half), np.asarray(centre)+np.asarray(half)
        inside = [lo[a] <= start[a] <= hi[a] for a in across]
        if not all(inside):
            continue
        if any(start[a]-radius < lo[a]-1e-12 or start[a]+radius > hi[a]+1e-12 for a in across):
            raise ValueError("A pin's cross-section leaves a cell it passes through: partial overlaps are not modelled")
        if hi[axis] > y0 and lo[axis] < y1:
            covered.append((max(lo[axis], y0), min(hi[axis], y1)))
    free, at = [], y0
    for lo, hi in sorted(covered):
        if lo > at+1e-12:
            free.append((at, lo))
        at = max(at, hi)
    if at < y1-1e-12:
        free.append((at, y1))
    return free


def _solids(shapes):
    """(volume, centre, inertia of a unit-density solid about its centre) of every box and free pin segment."""
    boxes = [s for s in shapes if s[0] == "box"]
    solids = []
    for _, centre, half, _ in boxes:
        a, b, c = 2*np.asarray(half, float)
        v = a*b*c
        solids.append((v, np.asarray(centre, float), v/12*np.diag([b*b+c*c, a*a+c*c, a*a+b*b])))
    for pin in (s for s in shapes if s[0] == "cylinder"):
        r, axis = float(pin[3]), pin_axis(pin)
        for y0, y1 in pin_free_segments(pin, boxes):
            length = y1-y0
            v = np.pi*r*r*length
            side = v*(3*r*r+length*length)/12
            centre, own = np.array(pin[1], float), np.full(3, side)
            centre[axis], own[axis] = (y0+y1)/2, v*r*r/2
            solids.append((v, centre, np.diag(own)))
    return solids


def piece_volume(shapes):
    """The piece's volume: its cells plus the parts of its pins outside them."""
    return sum(v for v, _, _ in _solids(shapes))


def piece_points(shapes, ring=16):
    """Corners of every box and points on both end circles of every pin, in the piece frame (metres)."""
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    points = []
    for shape in shapes:
        if shape[0] == "box":
            points.append(shape[1] + corners*shape[2])
        else:
            angle = np.linspace(0, 2*np.pi, ring, endpoint=False)
            (u, v), circle = PIN_PLANE[pin_axis(shape)], np.zeros((ring, 3))
            circle[:, u], circle[:, v] = np.cos(angle), np.sin(angle)
            circle *= shape[3]
            points += [shape[1]+circle, shape[2]+circle]
    return np.concatenate(points)


def piece_inertia(shapes, density):
    """(mass, centre of mass, inertia tensor about it) of the piece at one density, in the piece frame: its cells and the
    parts of its pins outside them, each counted once."""
    solids = _solids(shapes)
    mass = density*sum(v for v, _, _ in solids)
    centre = sum(v*c for v, c, _ in solids)/sum(v for v, _, _ in solids)
    inertia = np.zeros((3, 3))
    for v, c, own in solids:
        d = c-centre
        inertia += density*(own + v*(float(d @ d)*np.eye(3) - np.outer(d, d)))
    return mass, centre, inertia


def piece_visual(shapes, rgba):
    """The piece's visual mesh in the piece frame: the exact union of its boxes (every exposed face of their
    coordinate-plane partition, so coplanar neighbours weld without T-junctions) plus its pins, flat shaded, with the
    metal finish METAL in colour `rgba`."""
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals
    boxes = np.array([[np.asarray(c)-h, np.asarray(c)+h] for _, c, h, _ in (s for s in shapes if s[0] == "box")])
    axes = [np.unique(boxes[:, :, i]) for i in range(3)]
    centres = np.stack(np.meshgrid(*[(a[1:]+a[:-1])/2 for a in axes], indexing="ij"), axis=-1)
    solid = np.any(np.all((centres[..., None, :] >= boxes[:, 0]) & (centres[..., None, :] <= boxes[:, 1]), axis=-1),
                   axis=-1)
    vertices, faces = [], []
    for index in np.argwhere(solid):
        low = np.array([axes[k][index[k]] for k in range(3)])
        high = np.array([axes[k][index[k]+1] for k in range(3)])
        for axis in range(3):
            u, v = (axis+1) % 3, (axis+2) % 3
            for sign in (-1, 1):
                neighbour = index.copy()
                neighbour[axis] += sign
                if 0 <= neighbour[axis] < solid.shape[axis] and solid[tuple(neighbour)]:
                    continue
                corners = np.tile(low, (4, 1))
                corners[:, axis] = low[axis] if sign < 0 else high[axis]
                corners[[1, 2], u] = high[u]
                corners[[2, 3], v] = high[v]
                if sign < 0:
                    corners = corners[::-1]
                n = len(vertices)
                vertices.extend(corners)
                faces.extend(((n, n+1, n+2), (n, n+2, n+3)))
    meshes = [trimesh.Trimesh(vertices=vertices, faces=faces, process=True)]
    for _, start, end, radius in (s for s in shapes if s[0] == "cylinder"):
        meshes.append(trimesh.creation.cylinder(radius=radius, segment=np.array([start, end]), sections=PIN_SECTIONS))
    mesh = trimesh.util.concatenate(meshes)
    # Flat faces must not interpolate normals across the right-angle edges.
    mesh.unmerge_vertices()
    mesh.visual = TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)), material=PBRMaterial(baseColorFactor=rgba, **METAL))
    return mesh


def piece_urdf(name, shapes, density, rgba, visual):
    def numbers(values):
        return " ".join(f"{float(v):.17g}" for v in values)
    mass, centre, inertia = piece_inertia(shapes, density)
    lines = [f'<robot name="{name}">', f'  <link name="{name}">',
             f'    <inertial><origin xyz="{numbers(centre)}" rpy="0 0 0"/><mass value="{mass:.17g}"/>'
             f'<inertia ixx="{inertia[0, 0]:.17g}" ixy="{inertia[0, 1]:.17g}" ixz="{inertia[0, 2]:.17g}" '
             f'iyy="{inertia[1, 1]:.17g}" iyz="{inertia[1, 2]:.17g}" izz="{inertia[2, 2]:.17g}"/></inertial>',
             f'    <visual name="{name}"><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{visual}"/>'
             f'</geometry><material name="{name}_metal"><color rgba="{numbers(rgba)}"/></material></visual>']
    for shape in shapes:
        if shape[0] == "box":
            c = shape[3]
            origin = f'<origin xyz="{numbers(shape[1])}" rpy="0 0 0"/>'
            geometry = f'<geometry><box size="{numbers(2*np.asarray(shape[2]))}"/></geometry>'
            label = f"cell_{c[0]}_{c[1]}_{c[2]}"
        else:
            # rpy turns the cylinder's z axis onto the pin's: (0, 90 deg, 0) onto x, (-90 deg, 0, 0) onto y
            rpy = (f"0 {np.pi/2:.17g} 0", f"{-np.pi/2:.17g} 0 0", "0 0 0")[pin_axis(shape)]
            origin = f'<origin xyz="{numbers((shape[1]+shape[2])/2)}" rpy="{rpy}"/>'
            geometry = (f'<geometry><cylinder radius="{shape[3]:.17g}" '
                        f'length="{float(np.linalg.norm(shape[2]-shape[1])):.17g}"/></geometry>')
            label = "pin"
        lines.append(f'    <collision name="{label}">{origin}{geometry}</collision>')
    lines += ["  </link>", "</robot>", ""]
    return "\n".join(lines)


def generate_assets(folder, record):
    """Write one URDF and its visual GLB per piece; return the per-piece report (paths, mass, volume, points, grasp
    pad)."""
    folder = Path(folder)
    report = dict(kind=record["kind"], cell_m=record["cell_m"], clearance_m=record["clearance_m"], bodies={})
    for piece, name in PIECE_NAMES.items():
        shapes = piece_primitives(record, piece)
        volume = piece_volume(shapes)
        density = float(record["piece_mass_kg"])/volume
        path = folder/name/f"{name}.urdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        piece_visual(shapes, COLOURS[piece]).export(path.with_suffix(".glb"), include_normals=True)
        path.write_text(piece_urdf(name, shapes, density, COLOURS[piece], path.with_suffix(".glb").name))
        meta = record["pieces"][piece]
        report["bodies"][name] = dict(
            piece=piece, urdf=str(path), mass_kg=float(record["piece_mass_kg"]), volume_m3=volume, density_kg_m3=density,
            boxes=sum(s[0] == "box" for s in shapes), pins=sum(s[0] == "cylinder" for s in shapes),
            grasp_cell=list(meta["grasp_cell"]),
            grasp_pad_m=(np.asarray(meta["grasp_pad_cells"], float)*float(record["cell_m"])).tolist(),
            points_m=piece_points(shapes))
    return report
