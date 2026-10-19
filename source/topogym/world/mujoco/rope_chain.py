"""Ropes on MuJoCo: a FEM rope declaration (EntitySpec physics_model "fem_elastic") becomes a rigid capsule chain.

Why a chain, and why in maximal coordinates (2026-10-09/10, rope agent; evidence in tmp/2026-10-09_robotangle_mujoco/rope):
  - A rigid chain cannot be crushed the way the FEM rope was (the source of most bogus rope failures on Genesis).
  - A chain of ball joints in generalized coordinates is MuJoCo's cable, but a 1 m rope needs 220-340 links of 4.5 mm,
    and every link touches the table: a contact on link k involves all dofs from the root to k, so the Newton solver's
    Hessian is dense (100 links: 224-525 ms per 0.625 ms step). Free links joined by connect equalities keep every
    Jacobian row local: the sparse Newton solve takes 5.7 ms at 220 links, 7.5 ms at 300 (1.3 ms once the rope rests).

The model of one rope:
  - links: capsules of the rope's radius between nodes at equal material spacing (`link_m`), each a free body named
    `<entity>::<k>` (k = 000, 001, ... in material order); the end links' caps end at the rope's material ends; mass
    rho pi r^2 per material length.
  - joints: a connect equality at every node (a ball joint as a stiff soft constraint).
  - bending and twist: a discrete elastic rod. At node k the deviation of the relative rotation R_k^T R_k+1 from its rest
    value, theta = log(Q0_k^T R_k^T R_k+1) (in link k+1's frame), stores U = 1/2 theta' K theta, K = diag(G J, E I, E I) / l
    (E, nu of the rope's material, G = E / 2 (1 + nu), I = pi r^4 / 4, J = 2 I, l the node's material spacing). The couple
    is applied exactly (conservative: -J_r^-T(theta) K theta on k+1, the opposite on k) as generalized applied forces
    (qfrc_applied, the free joints' angular dofs), computed before every MuJoCo step at the current pose (a couple lagged
    by one step pumps energy into a spring). Rest: Q0 = identity for a straight rest, else the rest shape's own frames.
  - link frames are the rope's material frames: x along the link, y the direction of the FEM rings' first vertex (the
    rings carry the material orientation), so twist and bending are measured against the same material directions in the
    rest shape and at birth.
  - integration: the couples are explicit, so each link's rotational inertia is raised to at least `inertia_rule` *
    k * dt^2 about each axis (k: the link's stiffness about that axis; 2 was the stability edge, the default 4 keeps a
    factor 2), and the rod damps the relative angular velocity with `damping_rule` * I / dt per axis (explicit damping
    is limited by the same inertia). The extra rotational inertia changes the rope's long waves by < 4 % (0.2 m waves).
  - contacts: every link collides with everything but its `exclude_links` neighbours on each side, from a distance of
    `contact_margin_m` (geom margin; default 0.2 mm, the IPC rope's contact distance d_hat, materials.rope_ipc_parameters):
    the contacts act within that distance as IPC's barrier does, so a resting contact keeps a positive gap (a link on a
    box or mesh under its own weight: +0.14 / +0.16 mm) instead of MuJoCo's soft penetration (-0.06 / -0.04 mm without
    the margin). The tasks' rules need that: a released rope may touch the robot's fixed parts but never intersect them
    (knot_untying's exact release check, 1 nm), and strands lying side by side stay a rope diameter apart in plan.
  - bend stops: a rope cannot fold tighter than its thickness allows, and the excluded neighbours would let it. Every link
    carries a sphere at its centre that touches only the spheres of its two neighbours (explicit contact pairs), sized so
    they meet when their joint bends `bend_stop_rad` (0.9 rad over 4.5 mm links: a bend radius of 5 mm, 1.25 rope radii;
    a hairpin then keeps its strands 10 mm apart, more than the rope's 8 mm). Without them a pulled loop collapsed into a
    single-joint kink past 1.5 rad under 1 N and blew the rod up.

The speed clamp (`speed_clamp`, knot_chain_v6 / rope_chain_v2, 2026-10-11): a link spinning far faster than any reference
lets the node connect rows pump energy into it. An explicit step moves a connect's two anchors along their tangents, so
every MuJoCo step opens a gap of about w^2 l dt^2 that the soft row closes into kinetic energy (about w^4: a positive
feedback). The GPU search found it twice on knot_chain_v5 (a whipped free tail: easier e79, the rows' net work +0.04 ->
+200 mJ per step over 30 steps, then a bad qacc at 2560 m/s; easy top029, an uncaused 14 -> 15.4 m/s jump). The clamp
adds damping only on the excess: on a link whose centre-of-mass speed exceeds v_m_s, a force -rate m / dt (1 - v_m_s /
|v|) v at its centre of mass; on a link whose spin exceeds w_rad_s, a couple -rate I / dt (1 - w_rad_s / |w|) w (body
frame). Both oppose the motion, so it only removes energy, and it sits in qfrc_applied outside the rod's elastic part,
where the stress observer books it as dissipation. Below both thresholds it writes nothing: a run that never reaches
them is bit-identical to one without the clamp (every -i01 reference: 0 activations). Its activations are counted
(RopeChain.clamp_stats, rope_clamp_stats). It is read from the family's live profile at load, frozen scenes included
(their manifests keep the profile they were built with; the chain's model is unchanged, so the v5 births stand).

The tension guard (`tension_guard`, knot_chain_v7 / rope_chain_v3, 2026-10-11): the explicit rod is stable only while
the chain tension at a link stays below a critical value that grows with the link's rotational inertia (the cinch test:
~30 N locally at the v5 floor). A robot pinching the rope between its wrist and gripper base (GPU search, knot easier
top054_b1_e229 and top004_b0_e84) stretches the chain past it: a mode growing ~3x per MuJoCo step through the rod and
the node connects, then a bad qacc, before any link is fast enough for the speed clamp. Before every MuJoCo step the
guard reads each node connect's force from the previous step; a link whose connects carry more than t_on (in units of
the rod's E I / l^2: 0.50 N on the knot rope, 33 N on rope_threading's) gets its rotational inertia raised to
I0 * T / t_ref, keeping its angular momentum (per body axis) and its centre-of-mass
velocity, so its kinetic energy can only drop; the scale relaxes by `decay` per step, keeping the spin (the energy
drops again). The guard removes energy only, and never writes while every link stays at or below t_on (the -i01
references peak at 2.2 E I / l^2, 1.1 N, on knot_untying and 0.09, 3.1 N, on rope_threading): such a run is
bit-identical to one without it. Its activations are counted (clamp_stats tg_*).

The FEM view (`RopeChain.fem_state`): the rope's FEM vertices (the frozen volume's), carried by the chain with
linear-blend skinning on material arc length (as the render bridge's bind_chain): a vertex at material s blends the two
links whose centres bracket s, keeping its birth offset in each link's frame. Task predicates read it unchanged
(state["fem"][name]: position_m, velocity_m_s, material_s_m). The IPC integrity evidence of a rigid chain is constant:
native_valid True, volume ratio 1 (rigid links do not change volume).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROPE_FORMAT = "robotangle-mujoco-rope/1"
SEP = "::"

# Profile "rope" keys and their defaults.
ROPE_DEFAULTS = dict(
    link_m=.0045,                 # target material length of a link (chord error at the knot instances' births <= 0.07 mm)
    inertia_rule=4.,              # rotational inertia floor: I >= inertia_rule * k * dt^2 (per axis)
    damping_rule=.3,              # rod damping: c = damping_rule * I / dt (per axis)
    exclude_links=3,              # no contact between links closer than this along the chain
    bend_stop_rad=.9,             # bend stop: neighbouring links touch once their joint bends this far (None: off)
    joint=dict(solref=[.00125, 1.], solimp=[.9999, .9999, .001, .5, 2.]),   # the node connect equalities (nearly hard:
                                  # MuJoCo's softness is mass-normalised, light links make soft connects stretch ~0.1 mm/N)
    robot_contact="full",         # "full": every robot geom touches the rope; "fingers": only the finger links' geoms
    contact_margin_m=.0002,       # the rope's contacts act from this distance (the IPC rope's d_hat); None: none
    rest=None,                    # None: the volume's elastic rest (straight unless elastic_rest is stored); "birth"
    speed_clamp=None,             # dict(v_m_s, w_rad_s, rate): extra damping on a link's speed above v_m_s / spin above
                                  # w_rad_s (RopeChain._clamp); None: off
    tension_guard=None,           # dict(t_on, t_ref, decay): a link's rotational inertia raised with its connects'
                                  # tension above t_on E I / l^2 (RopeChain._tension_guard); None: off
)


# ----------------------------------------------------------------------------------------------- geometry helpers
def birth_from_archive(manifest_path, name: str) -> np.ndarray:
    """The FEM vertex positions of soft entity `name` in a frozen IPC settled state (ipc_settled_state.py archive)."""
    from topogym.world.initial_state import load_numeric_archive
    payload, _ = load_numeric_archive(Path(manifest_path))
    fem = payload.get("fem") or {}
    if name not in fem:
        raise ValueError(f"the birth archive {manifest_path} has no FEM entity {name!r} ({sorted(fem)})")
    return np.asarray(fem[name], dtype=np.float64)


def _volume(path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def _rings(volume) -> np.ndarray:
    if "surface_ring_indices" not in volume:
        raise NotImplementedError("a rope chain needs a volume with material surface rings (v13 rope profile)")
    rings = np.asarray(volume["surface_ring_indices"], dtype=np.int64)
    if rings.ndim != 2 or rings.shape[1] < 3:
        raise ValueError("invalid rope surface rings")
    return rings


def material_frames(points, volume, s_nodes):
    """Node positions (n + 1, 3) and link frames (n, 3, 3) of the rope in configuration `points` (its FEM vertices):
    the material centreline (ring centroids, as the predicates sample it) at the node stations, and for every link
    x = its axis, y = the material direction of the rings' first vertex, projected and averaged over the link."""
    p = np.asarray(points, dtype=np.float64)
    rest = np.asarray(volume["rest"], dtype=np.float64)
    rings = _rings(volume)
    st = rest[rings, 0].mean(axis=1)
    centre = p[rings].mean(axis=1)
    nodes = np.column_stack([np.interp(s_nodes, st, centre[:, i]) for i in range(3)])
    # material orientation of every ring: its first vertex's offset from the centroid, turned back by that vertex's
    # authored angle in the rest cross-section (so y is the rest frame's +y in every ring)
    off = p[rings[:, 0]] - centre
    phi = np.arctan2(rest[rings[:, 0], 2], rest[rings[:, 0], 1])
    axes = np.diff(nodes, axis=0)
    lengths = np.linalg.norm(axes, axis=1)
    if np.any(lengths < 1e-9):
        raise ValueError("degenerate rope link")
    x = axes / lengths[:, None]
    R = np.empty((len(x), 3, 3))
    # tangent at each ring (centreline differences) to turn the offset about
    tangent = np.gradient(centre, st, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    o = off - np.einsum("ij,ij->i", off, tangent)[:, None] * tangent
    o /= np.linalg.norm(o, axis=1, keepdims=True)
    b = np.cross(tangent, o)
    y_ring = np.cos(phi)[:, None] * o - np.sin(phi)[:, None] * b     # rotate o by -phi about the tangent
    for k in range(len(x)):
        sel = (st >= s_nodes[k] - 1e-12) & (st <= s_nodes[k + 1] + 1e-12)
        if not sel.any():
            sel = np.zeros(len(st), bool)
            sel[int(np.argmin(np.abs(st - 0.5 * (s_nodes[k] + s_nodes[k + 1]))))] = True
        y = y_ring[sel].sum(axis=0)
        y -= np.dot(y, x[k]) * x[k]
        n = np.linalg.norm(y)
        if n < 1e-9:
            raise ValueError("rope material frame undefined (rings aligned with the link)")
        y /= n
        R[k] = np.column_stack([x[k], y, np.cross(x[k], y)])
    return nodes, R, lengths


def _quat_wxyz(R) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


# ----------------------------------------------------------------------------------------------- scene building
def add_rope_chain(builder, spec, birth) -> None:
    """Write rope `spec` as a capsule chain into the SceneBuilder (bodies, connects, contact excludes) and its data file
    rope_<name>.npz into the scene directory; register the manifest record (kind "rope_chain")."""
    import xml.etree.ElementTree as ET
    from .scene import numbers
    if birth is None:
        raise ValueError(f"{spec.name}: building a MuJoCo rope needs its frozen birth (SimulatorConfig.mujoco_birth)")
    if spec.volume_path is None:
        raise ValueError(f"{spec.name}: a FEM rope needs its volume")
    settings = {**ROPE_DEFAULTS, **dict(builder.profile.get("rope") or {})}
    volume = _volume(spec.volume_path)
    rest = np.asarray(volume["rest"], dtype=np.float64)
    birth = np.asarray(birth, dtype=np.float64)
    if birth.shape != rest.shape:
        raise ValueError(f"{spec.name}: the birth has {birth.shape} vertices, the volume {rest.shape}")
    radius = float(volume["radius_m"])
    material = dict(spec.material)
    E, nu, rho = float(material["E"]), float(material.get("nu", .3)), float(material.get("rho", 650.))
    mu = float(material.get("friction_mu", builder.profile["contact"]["friction"]))
    rings = _rings(volume)
    st = rest[rings, 0].mean(axis=1)
    L = float(st[-1] - st[0])
    n = max(2, int(round(L / float(settings["link_m"]))))
    s_nodes = np.linspace(st[0], st[-1], n + 1)
    nodes, R, lengths = material_frames(birth, volume, s_nodes)
    # rest relative rotations: the volume's elastic rest (straight unless stored), or the birth itself
    if settings.get("rest") == "birth":
        rest_rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
    elif "elastic_rest" in volume and not np.array_equal(volume["elastic_rest"], rest):
        _, R0, _ = material_frames(volume["elastic_rest"], volume, s_nodes)
        rest_rel = np.einsum("nji,njk->nik", R0[:-1], R0[1:])
    else:
        rest_rel = np.repeat(np.eye(3)[None], n - 1, axis=0)
    dt = float(builder.env_cfg.sim.dt) / int(builder.env_cfg.sim.substeps)
    G = E / (2 * (1 + nu))
    I_area = np.pi * radius ** 4 / 4
    mat_len = np.diff(s_nodes)
    mass = rho * np.pi * radius ** 2 * mat_len
    k_twist, k_bend = G * 2 * I_area / lengths, E * I_area / lengths            # per link (its own length)
    I_axis = np.maximum(mass * radius ** 2 / 2, settings["inertia_rule"] * k_twist * dt ** 2)
    I_perp = np.maximum(mass * (3 * radius ** 2 + lengths ** 2) / 12, settings["inertia_rule"] * k_bend * dt ** 2)
    node_l = 0.5 * (mat_len[:-1] + mat_len[1:])
    K = np.stack([G * 2 * I_area / node_l, E * I_area / node_l, E * I_area / node_l], axis=1)
    rule = float(settings["damping_rule"])
    C = np.stack([rule * np.minimum(I_axis[:-1], I_axis[1:]) / dt,
                  rule * np.minimum(I_perp[:-1], I_perp[1:]) / dt,
                  rule * np.minimum(I_perp[:-1], I_perp[1:]) / dt], axis=1)
    contact = builder._contact_attrs("task", mu)
    if settings.get("contact_margin_m"):
        contact = {**contact, "margin": numbers([settings["contact_margin_m"]])}
    bits = {}
    if settings["robot_contact"] == "fingers":
        # the rope touches the static scene, itself and the fingers only (MuJoCo collides two geoms when either one's
        # contype shares a bit with the other's conaffinity): rope 2 / 3, the robot's other geoms 4 / 5, everything else
        # (the fingers, the table, the fixtures) keeps 1 / 1. Rope-rope, rope-scene, rope-finger, robot-robot and
        # robot-scene contacts stay; rope-robot contacts outside the fingers are gone.
        bits = dict(contype="2", conaffinity="3")
    elif settings["robot_contact"] != "full":
        raise ValueError(f"unknown rope robot_contact {settings['robot_contact']!r}")
    links, joints = [], []
    for k in range(n):
        name = f"{spec.name}{SEP}{k:03d}"
        body = ET.SubElement(builder.world, "body", name=name, pos=numbers(nodes[k]), quat=numbers(_quat_wxyz(R[k])))
        ET.SubElement(body, "freejoint", name=f"{name}{SEP}root")
        ET.SubElement(body, "inertial", pos=numbers([lengths[k] / 2, 0., 0.]), mass=numbers([mass[k]]),
                      diaginertia=numbers([I_axis[k], I_perp[k], I_perp[k]]))
        x0 = radius if k == 0 else 0.
        x1 = lengths[k] - radius if k == n - 1 else lengths[k]
        if x1 - x0 < 1e-6:          # an end link shorter than the radius: the cap alone
            x0 = x1 = 0.5 * (x0 + x1)
            ET.SubElement(body, "geom", name=f"{name}{SEP}g", type="sphere", pos=numbers([x0, 0., 0.]),
                          size=numbers([radius]), **contact, **bits)
        else:
            ET.SubElement(body, "geom", name=f"{name}{SEP}g", type="capsule", fromto=numbers([x0, 0., 0., x1, 0., 0.]),
                          size=numbers([radius]), **contact, **bits)
        links.append(dict(name=f"{k:03d}", body=name, parent=None))
        joints.append(dict(name=f"{k:03d}", mj=f"{name}{SEP}root"))
    joint = settings["joint"]
    for k in range(n - 1):
        ET.SubElement(builder.equality, "connect", name=f"{spec.name}{SEP}node{k + 1:03d}",
                      body1=f"{spec.name}{SEP}{k:03d}", body2=f"{spec.name}{SEP}{k + 1:03d}",
                      anchor=numbers([lengths[k], 0., 0.]), solref=numbers(joint["solref"]), solimp=numbers(joint["solimp"]))
    for i in range(n):
        for j in range(i + 1, min(n, i + 1 + int(settings["exclude_links"]))):
            ET.SubElement(builder.contact, "exclude", body1=f"{spec.name}{SEP}{i:03d}", body2=f"{spec.name}{SEP}{j:03d}")
    stop = settings.get("bend_stop_rad")
    if stop:
        # neighbouring link centres are L cos(theta / 2) apart when their joint bends theta: 2 r_stop apart at `stop`
        r_stop = 0.5 * float(np.mean(lengths)) * float(np.cos(0.5 * float(stop)))
        stop_attrs = {k: v for k, v in contact.items() if k in ("solref", "solimp")}
        for k in range(n):
            body = next(b for b in builder.world.findall("body") if b.get("name") == f"{spec.name}{SEP}{k:03d}")
            ET.SubElement(body, "geom", name=f"{spec.name}{SEP}{k:03d}{SEP}stop", type="sphere", size=numbers([r_stop]),
                          pos=numbers([lengths[k] / 2, 0., 0.]), contype="0", conaffinity="0", group="3",
                          rgba="0 0 0 0")
        for k in range(n - 1):
            ET.SubElement(builder.contact, "pair", geom1=f"{spec.name}{SEP}{k:03d}{SEP}stop",
                          geom2=f"{spec.name}{SEP}{k + 1:03d}{SEP}stop", condim="1", **stop_attrs)
    if bits:
        _finger_bits(builder)
    data = f"rope_{spec.name}.npz"
    np.savez(builder.out / data, s_nodes=s_nodes, lengths=lengths, rest_rel=rest_rel, K=K, C=C, birth=birth,
             elements=np.asarray(volume["elements"], dtype=np.int64), material_s=rest[:, 0].copy(), radius=radius)
    volume_sha = hashlib.sha256(Path(spec.volume_path).read_bytes()).hexdigest()
    centre = birth[rings].mean(axis=1)
    k_seg = np.clip(np.searchsorted(s_nodes, st, side="right") - 1, 0, n - 1)
    A, B = nodes[k_seg], nodes[k_seg + 1]
    t = np.clip(np.einsum("ij,ij->i", centre - A, B - A) / np.einsum("ij,ij->i", B - A, B - A), 0., 1.)
    chord = float(np.linalg.norm(centre - (A + t[:, None] * (B - A)), axis=1).max())
    builder.entities[spec.name] = dict(
        kind="rope_chain", root=f"{spec.name}{SEP}000", links=links, joints=joints, free_joint=None, collision=True,
        rope=dict(format=ROPE_FORMAT, file=data, links=n, link_m=float(L / n), radius_m=radius, E=E, nu=nu, rho=rho,
                  friction=mu, volume_sha256=volume_sha, declared_position_m=list(map(float, spec.position_m)),
                  settings=json.loads(json.dumps(settings)), dt_s=dt, birth_chord_error_m=chord,
                  bend_stop_radius_m=r_stop if stop else None,
                  inertia_raised=dict(axis=int(np.sum(I_axis > mass * radius ** 2 / 2)),
                                      perpendicular=int(np.sum(I_perp > mass * (3 * radius ** 2 + lengths ** 2) / 12))),
                  rest="straight" if np.allclose(rest_rel, np.eye(3)) else "shaped"))


def _finger_bits(builder) -> None:
    """robot_contact "fingers": the robot's collision geoms outside the finger links move to contype 4 / conaffinity 5 (see
    add_rope_chain). The robot is in the scene before any task entity, so nothing else needs changing."""
    fingers = {f for g in builder.description.groups for f in g.grasp.finger_links}
    robot = {l["body"] for l in builder.entities["robot"]["links"]}
    defaults = {}
    for d in builder.default.iter("default"):          # class defaults that set the robot geoms' bits
        g = d.find("geom")
        if g is not None:
            defaults[d.get("class")] = (g.get("contype"), g.get("conaffinity"))
    for body in builder.root.iter("body"):
        name = body.get("name") or ""
        if name not in robot or name in fingers:
            continue
        for g in body.findall("geom"):
            ct, ca = g.get("contype"), g.get("conaffinity")
            if ct is None or ca is None:
                dct, dca = defaults.get(g.get("class"), (None, None))
                ct, ca = ct if ct is not None else dct, ca if ca is not None else dca
            if (ct or "1") == "0" and (ca or "1") == "0":
                continue                                # visual
            if (ct or "1") != "1" or (ca or "1") != "1":
                raise NotImplementedError(f"robot geom of {name} has collision bits {ct} / {ca}: not ported")
            g.set("contype", "4")
            g.set("conaffinity", "5")


# ----------------------------------------------------------------------------------------------- runtime
def _log_so3(D):
    tr = np.clip((np.trace(D, axis1=1, axis2=2) - 1) / 2, -1., 1.)
    th = np.arccos(tr)
    v = np.stack([D[:, 2, 1] - D[:, 1, 2], D[:, 0, 2] - D[:, 2, 0], D[:, 1, 0] - D[:, 0, 1]], axis=1)
    small = th < 1e-6
    f = np.where(small, .5 + th * th / 12, th / (2 * np.where(small, 1., np.sin(th))))
    return v * f[:, None]


def _cross(u, v):
    """np.cross of (n, 3) arrays, its arithmetic (a1 b2 - a2 b1, ...) without its axis handling: the same bits."""
    out = np.empty(np.broadcast_shapes(u.shape, v.shape))
    np.subtract(u[:, 1] * v[:, 2], u[:, 2] * v[:, 1], out=out[:, 0])
    np.subtract(u[:, 2] * v[:, 0], u[:, 0] * v[:, 2], out=out[:, 1])
    np.subtract(u[:, 0] * v[:, 1], u[:, 1] * v[:, 0], out=out[:, 2])
    return out


def _jr_inv_T(theta, g):
    """J_r^{-T}(theta) g: J_r^{-1} = I + 1/2 [theta] + c [theta]^2, so its transpose is I - 1/2 [theta] + c [theta]^2."""
    th = np.linalg.norm(theta, axis=1)
    small = th < 1e-4
    ths = np.where(small, 1., th)
    c = np.where(small, 1 / 12 + th * th / 720, 1 / ths ** 2 - (1 + np.cos(ths)) / (2 * ths * np.sin(ths)))
    tg = _cross(theta, g)
    return g - .5 * tg + c[:, None] * _cross(theta, tg)


class RopeChain:
    """One rope of a built MuJoCo world: its rod couples (apply, before every MuJoCo step), its FEM view (fem_state) and
    its constant integrity evidence."""

    def __init__(self, world, name: str, record: dict, scene_dir):
        import mujoco
        m = world.model
        self.world, self.name, self.record = world, name, record
        info = record["rope"]
        with np.load(Path(scene_dir) / info["file"], allow_pickle=False) as data:
            self.s_nodes = data["s_nodes"].copy()
            self.lengths = data["lengths"].copy()
            self.rest_rel = data["rest_rel"].copy()
            self.K, self.C = data["K"].copy(), data["C"].copy()
            self.birth = data["birth"].copy()
            self.elements = data["elements"].copy()
            self.material_s = data["material_s"].copy()
        self.bodies = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, l["body"]) for l in record["links"]])
        if np.any(self.bodies < 0):
            raise ValueError(f"{name}: scene lacks rope links")
        jnt = np.array([int(m.body_jntadr[b]) for b in self.bodies])
        if np.any(jnt < 0) or np.any(np.asarray(m.jnt_type)[jnt] != mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"{name}: rope links must be free bodies")
        self.dof = np.array([int(m.jnt_dofadr[j]) for j in jnt])
        self.rot_dofs = self.dof[:, None] + np.arange(3, 6)[None]
        self.lin_dofs = self.dof[:, None] + np.arange(3)[None]
        # the clamp is a runtime rule of the family's live profile, frozen scenes included (as the exact diagonal and
        # the finger pads, world.py): a frozen birth's manifest keeps the profile it was built with
        live = (getattr(world, "profile", None) or {}).get("rope") or {}
        clamp = live["speed_clamp"] if "speed_clamp" in live else (info.get("settings") or {}).get("speed_clamp")
        self._clamp_cfg = None
        if clamp:
            self._clamp_cfg = (float(clamp["v_m_s"]), float(clamp["w_rad_s"]), float(clamp.get("rate", .5)) / float(m.opt.timestep))
            self._mass = np.asarray(m.body_mass)[self.bodies].copy()
            self._inertia = np.asarray(m.body_inertia)[self.bodies].copy()
            self._ipos = np.asarray(m.body_ipos)[self.bodies].copy()
        self._tg = None
        guard = live["tension_guard"] if "tension_guard" in live else (info.get("settings") or {}).get("tension_guard")
        if guard:
            eq2joint = np.full(int(m.neq), -1, np.int64)       # node connect equality -> its joint (links k, k + 1)
            for k in range(len(self.bodies) - 1):
                e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"{name}{SEP}node{k + 1:03d}")
                if e < 0:
                    raise ValueError(f"{name}: no node connect {k + 1}")
                eq2joint[e] = k
            # tension in units of the rod's bending stiffness per length, E I / l^2 at each joint (knot: 0.50 N): the
            # critical tension scales with it, the inertia floor being inertia_rule * k dt^2
            unit = self.K[:, 1] / (0.5 * (self.lengths[:-1] + self.lengths[1:]))
            self._tg = dict(on=float(guard["t_on"]), ref=float(guard["t_ref"]), decay=float(guard.get("decay", .99)),
                            unit=unit, eq2joint=eq2joint, I0=np.asarray(m.body_inertia)[self.bodies].copy(),
                            ipos=np.asarray(m.body_ipos)[self.bodies].copy(), scale=np.ones(len(self.bodies)))
        self.clamp_stats = dict(substeps=0, link_substeps=0, speed_link_substeps=0, spin_link_substeps=0,
                                first_tick=None, last_tick=None, max_speed_m_s=0., max_spin_rad_s=0., loss_j=0.,
                                tg_link_substeps=0, tg_max_scale=1., tg_max_tension_n=0., tg_first_tick=None,
                                tg_loss_j=0.)
        self.energy = 0.
        self.elastic_qfrc = np.zeros((len(self.bodies), 3))   # the last apply's elastic couples (body-frame dofs)
        self._skin = None
        self._skin_bodies = None      # fem_state: each vertex's two link bodies

    # -- the rod
    def _theta(self, d):
        """The links' rotation matrices and every joint's rotation vector from its rest, at d's kinematics."""
        R = np.asarray(d.xmat)[self.bodies].reshape(-1, 3, 3)
        Q = np.einsum("nji,njk->nik", R[:-1], R[1:])
        return R, _log_so3(np.einsum("nji,njk->nik", self.rest_rel, Q))

    def _links(self, R, tau):
        """A couple tau on link k+1 (its frame) and -tau on link k, per joint -> the links' body-frame generalized forces."""
        tau_world = np.einsum("nij,nj->ni", R[1:], tau)
        total = np.zeros((len(self.bodies), 3))
        total[1:] += tau_world
        total[:-1] -= tau_world
        return np.einsum("nji,nj->ni", R, total)

    def apply(self, d) -> None:
        """Write the rod's couples for the current pose and velocity into d.qfrc_applied (the links' angular dofs, body
        frame). Refreshes the kinematics first: mj_step leaves d.xmat at the previous step's qpos. The elastic part (the
        gradient of energy below) is kept in elastic_qfrc; the rest is the rod's damping."""
        import mujoco
        mujoco.mj_kinematics(self.world.model, d)
        if self._tg is not None:
            self._tension_guard(d, np.asarray(d.xmat)[self.bodies].reshape(-1, 3, 3))
        R, theta = self._theta(d)
        g = self.K * theta
        tau = -_jr_inv_T(theta, g)                                   # on link k+1, its frame
        self.elastic_qfrc = self._links(R, tau)
        w = np.asarray(d.qvel)[self.rot_dofs]                       # body-frame angular velocities
        w_world = np.einsum("nij,nj->ni", R, w)
        rel = np.einsum("nji,nj->ni", R[1:], w_world[1:] - w_world[:-1])    # relative, in link k+1's frame
        qa = np.asarray(d.qfrc_applied)
        qa[self.rot_dofs] = self._links(R, tau - self.C * rel)
        if self._clamp_cfg is not None:
            self._clamp(d, R, qa)
        self.energy = float(0.5 * np.sum(theta * g))

    def _clamp(self, d, R, qa) -> None:
        """The speed clamp (module docstring): damping on the excess speed / spin of the links above the thresholds, added
        to qfrc_applied (a force at each centre of mass: the linear dofs and its moment about the link origin; a couple).
        Writes nothing while no link exceeds them (its linear dofs are cleared once after an activation)."""
        v_c, w_c, k = self._clamp_cfg
        qv = np.asarray(d.qvel)
        w_b = qv[self.rot_dofs]                                       # body frame
        r_b = self._ipos
        v_com = qv[self.lin_dofs] + np.einsum("nij,nj->ni", R, _cross(w_b, r_b))
        sp = np.linalg.norm(v_com, axis=1)
        om = np.linalg.norm(w_b, axis=1)
        hot_v, hot_w = sp > v_c, om > w_c
        if not (hot_v.any() or hot_w.any()):
            if qa[self.lin_dofs].any():                                # the previous activation's forces
                qa[self.lin_dofs] = 0.
            return
        fv = np.where(hot_v, 1. - v_c / np.where(hot_v, sp, 1.), 0.)
        fw = np.where(hot_w, 1. - w_c / np.where(hot_w, om, 1.), 0.)
        F = -(k * self._mass * fv)[:, None] * v_com                  # world, at the centre of mass
        F_b = np.einsum("nji,nj->ni", R, F)
        couple = -(k * fw)[:, None] * self._inertia * w_b             # body frame
        qa[self.lin_dofs] = F
        qa[self.rot_dofs] += couple + _cross(r_b, F_b)
        st = self.clamp_stats
        tick = int(getattr(self.world, "physics_step", 0)) + 1
        st["substeps"] += 1
        st["link_substeps"] += int(np.sum(hot_v | hot_w))
        st["speed_link_substeps"] += int(hot_v.sum())
        st["spin_link_substeps"] += int(hot_w.sum())
        st["first_tick"] = tick if st["first_tick"] is None else st["first_tick"]
        st["last_tick"] = tick
        st["max_speed_m_s"] = max(st["max_speed_m_s"], float(sp.max()))
        st["max_spin_rad_s"] = max(st["max_spin_rad_s"], float(om.max()))
        st["loss_j"] -= float(np.sum(F * v_com) + np.sum(couple * w_b)) * self.world.model.opt.timestep   # its power at v0

    def _tension_guard(self, d, R) -> None:
        """The tension guard (module docstring): each link's rotational inertia I0 * max(1, T / t_ref) while its node
        connects carry T > t_on (the previous step's connect forces, in units of E I / l^2), relaxing by `decay` per
        step. Raising keeps the link's angular momentum (per body axis) and its centre-of-mass velocity, relaxing keeps
        its spin: the kinetic energy only drops (counted in tg_loss_j; the observer sees it as dissipation). Writes
        nothing while every scale is 1 and no link exceeds t_on."""
        import mujoco
        g = self._tg
        n = len(self.bodies)
        nefc = int(d.nefc)
        T = np.zeros(n - 1)
        if nefc:
            types = np.asarray(d.efc_type[:nefc])
            eq = types == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            joint = np.where(eq, g["eq2joint"][np.where(eq, np.asarray(d.efc_id[:nefc]), 0)], -1)
            sel = joint >= 0
            if sel.any():
                f = np.asarray(d.efc_force[:nefc])[sel]
                T = np.sqrt(np.bincount(joint[sel], weights=f * f, minlength=n - 1))
        tau = T / g["unit"]                 # the tension in units of E I / l^2
        Tl = np.zeros(n)
        Tl[:-1] = tau
        Tl[1:] = np.maximum(Tl[1:], tau)
        old = g["scale"]
        new = np.maximum(np.where(Tl > g["on"], Tl / g["ref"], 1.), np.maximum(1., old * g["decay"]))
        new = np.where(new < 1. + 1e-9, 1., new)
        st = self.clamp_stats
        if np.array_equal(new, old):
            st["tg_link_substeps"] += int(np.sum(old != 1.))
            return
        qv = np.asarray(d.qvel)
        ch = np.flatnonzero(new != old)
        up = ch[new[ch] > old[ch]]
        if len(up):                         # keep the angular momentum (body frame, per axis) and the COM velocity
            w0 = qv[self.rot_dofs[up]].copy()
            w1 = w0 * (old[up] / new[up])[:, None]
            ip, Ru = g["ipos"][up], R[up]
            vcom = qv[self.lin_dofs[up]] + np.einsum("nij,nj->ni", Ru, _cross(w0, ip))
            I_old, I_new = g["I0"][up] * old[up][:, None], g["I0"][up] * new[up][:, None]
            st["tg_loss_j"] += float(0.5 * np.sum(I_old * w0 * w0 - I_new * w1 * w1))
            qv[self.rot_dofs[up]] = w1
            qv[self.lin_dofs[up]] = vcom - np.einsum("nij,nj->ni", Ru, _cross(w1, ip))
        down = ch[new[ch] < old[ch]]
        if len(down):                       # keep the spin: the rotational energy drops with the inertia
            w0 = qv[self.rot_dofs[down]]
            st["tg_loss_j"] += float(0.5 * np.sum(g["I0"][down] * (old[down] - new[down])[:, None] * w0 * w0))
        np.asarray(self.world.model.body_inertia)[self.bodies[ch]] = g["I0"][ch] * new[ch][:, None]
        g["scale"] = new
        st["tg_link_substeps"] += int(np.sum(new != 1.))
        st["tg_max_scale"] = max(st["tg_max_scale"], float(new.max()))
        st["tg_max_tension_n"] = max(st["tg_max_tension_n"], float(T.max()))
        if st["tg_first_tick"] is None:
            st["tg_first_tick"] = int(getattr(self.world, "physics_step", 0)) + 1

    def joint_energy(self, d) -> np.ndarray:
        """Each joint's elastic energy 1/2 theta^T K theta at d's kinematics (the caller keeps d.xmat current)."""
        _, theta = self._theta(d)
        return 0.5 * np.sum(self.K * theta * theta, axis=1)

    # -- the FEM view
    def _binding(self):
        """Linear-blend skinning on material arc length, bound at birth (the frozen scene's body poses = qpos0)."""
        if self._skin is None:
            import mujoco
            m = self.world.model
            d0 = mujoco.MjData(m)
            mujoco.mj_kinematics(m, d0)
            pos = np.asarray(d0.xpos)[self.bodies].copy()
            rot = np.asarray(d0.xmat)[self.bodies].reshape(-1, 3, 3).copy()
            n = len(self.bodies)
            centres = 0.5 * (self.s_nodes[:-1] + self.s_nodes[1:])
            s = self.material_s
            hi = np.clip(np.searchsorted(centres, s), 1, n - 1)
            lo = hi - 1
            t = np.clip((s - centres[lo]) / (centres[hi] - centres[lo]), 0., 1.)[:, None]
            local_lo = np.einsum("nji,nj->ni", rot[lo], self.birth - pos[lo])
            local_hi = np.einsum("nji,nj->ni", rot[hi], self.birth - pos[hi])
            self._skin = (lo, hi, t, local_lo, local_hi)
        return self._skin

    def fem_state(self, d, velocities) -> dict:
        """state["fem"][name] as Genesis' read_vertices gives it, from the chain (skinned vertices and velocities).
        Each vertex's two link bodies are gathered once (np.take); the values are those of indexing the links first."""
        lo, hi, t, a_loc, b_loc = self._binding()
        if self._skin_bodies is None:
            self._skin_bodies = (self.bodies[lo], self.bodies[hi])
        bl, bh = self._skin_bodies
        xpos, xmat = np.asarray(d.xpos), np.asarray(d.xmat)
        ra = np.einsum("nij,nj->ni", np.take(xmat, bl, axis=0).reshape(-1, 3, 3), a_loc)
        rb = np.einsum("nij,nj->ni", np.take(xmat, bh, axis=0).reshape(-1, 3, 3), b_loc)
        pa, pb = ra + np.take(xpos, bl, axis=0), rb + np.take(xpos, bh, axis=0)
        ang, lin = velocities
        va = np.take(lin, bl, axis=0) + _cross(np.take(ang, bl, axis=0), ra)
        vb = np.take(lin, bh, axis=0) + _cross(np.take(ang, bh, axis=0), rb)
        return dict(position_m=pa + t * (pb - pa), velocity_m_s=va + t * (vb - va),
                    active=np.ones(len(self.material_s), dtype=bool), material_s_m=self.material_s.copy(),
                    vertex_indices=np.arange(len(self.material_s), dtype=np.int32))

    def integrity(self) -> dict:
        return dict(minimum=1., maximum=1.)

    def node_positions(self, d) -> np.ndarray:
        """The chain's nodes (n + 1, 3): every link's origin and the last link's end."""
        P = np.asarray(d.xpos)[self.bodies]
        end = P[-1] + np.asarray(d.xmat)[self.bodies[-1]].reshape(3, 3)[:, 0] * self.lengths[-1]
        return np.vstack([P, end])


# ----------------------------------------------------------------------------------------------- energy books
# The rod acts through qfrc_applied, so MuJoCo's own energy (data.energy) leaves its elastic energy out. These give it to
# an energy balance that books the rod as stored energy rather than outside work (scripts/robotangle/stress/observer.py);
# checked against the path-integrated work of the couples by scripts/robotangle/test_rope_energy.py.
def rope_energy(world, data=None) -> float:
    """The rods' elastic energy, 1/2 theta^T K theta over every joint of every rope chain of a MuJoCo world, at `data`'s
    kinematics (default the world's own data; the caller keeps its xmat current). 0 without ropes."""
    ropes = getattr(world, "_ropes", None) or {}
    d = world.data if data is None else data
    return float(sum(float(np.sum(r.joint_energy(d))) for r in ropes.values()))


def rope_link_energy(world, data=None) -> dict:
    """The rod energy stored at each rope link's joints, body id -> J: a joint's energy counts for both its links (as a
    stored-energy book counts a constraint row shared by two bodies). Empty without ropes."""
    ropes = getattr(world, "_ropes", None) or {}
    d = world.data if data is None else data
    out = {}
    for r in ropes.values():
        e = r.joint_energy(d)
        per = np.zeros(len(r.bodies))
        per[:-1] += e
        per[1:] += e
        out.update({int(b): float(v) for b, v in zip(r.bodies, per)})
    return out


def rope_clamp_stats(world) -> dict:
    """Each rope's speed-clamp activations since the world was built (RopeChain.clamp_stats): name -> counts. Only ropes
    whose profile sets speed_clamp."""
    ropes = getattr(world, "_ropes", None) or {}
    return {name: dict(r.clamp_stats) for name, r in ropes.items() if r._clamp_cfg is not None}


def rope_elastic_qfrc(world):
    """The rods' elastic couples as the last RopeChain.apply wrote them into qfrc_applied: an nv vector (the rope links'
    angular dofs; zeros elsewhere), the part of qfrc_applied whose potential rope_energy gives. None without ropes."""
    ropes = getattr(world, "_ropes", None) or {}
    if not ropes:
        return None
    out = np.zeros(world.model.nv)
    for r in ropes.values():
        out[r.rot_dofs] = r.elastic_qfrc
    return out
