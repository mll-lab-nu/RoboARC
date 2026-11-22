"""CS (`cstate`, "+state"): the information condition of the deformable tasks (ablation experiment E3).

The deformable analogue of C1G (scripts/ablations/cond_info.py): a Standard task whose service also gives, in the
agent-facing robot frame (the frame of every pose the service gives: never the simulator's world frame, which would
disclose the table height and the robot's mounting),
  - the task objects' poses in every response's `meta["objects"]` and the rigid parts' and fixtures' collision shapes
    behind `GET /geometry`, exactly as C1G gives them (cond_info's own code: patch_log, patch_objects, the C1G geometry
    block), the deformable bodies left out of `geo["parts"]`;
  - the current shape of the deformable body in every response's `meta["deformables"]`, and its layout (what never
    changes: link names, grid, triangles, the shapes of its rigid and welded links, its grasp regions) in
    `geo["deformables"]`:
      * "chains" (rope_threading, knot_untying: a rope_chain rope; orbits: the cord; fiber_bundling: the sheath and the
        fibres): every chain of capsule links as its centreline, one point per link plus one, and its radius, in link
        order. Every link is given (no downsampling: 75 to 545 links, at most about 15 kB of JSON per response), so no
        crossing can be lost between samples. A rigid link of the body that is not a capsule (the cord's handle, the
        sheath's stiff front segments) by its pose; a link welded to the world by its shapes, once, in /geometry.
      * "cloth" (blanket_flatten, blanket_fold): the simulation mesh's vertices (the authored grid, 51 x 41 = 2091 on
        every released blanket: no downsampling, about 60 kB per response) with the grid's shape; the triangles in
        /geometry.
Nothing else: no velocities of single points, no joint readings, no tension, no grasp phase, no task state (no crossing
count, no stage, no progress, no success). The service reads only MjData's kinematics (refreshed at every tick's end) and
the cloth's vertex view; it writes nothing, so the episode's state hash and its fresh-process replay are unchanged.

Interface (scripts/ablations/make_variants.py, the lead's framework): `make(src, dst)` edits dst, a full copy of the
released Standard task src, in place; the framework then writes the task.toml name, the [metadata] ablation line, DIGEST
and the content-derived mtimes. `leak_check(dst)` lists what is wrong with a generated directory (empty: clean). Every
edit applies exactly once, anything unexpected raises (the framework then removes dst). Deterministic: no time stamps, no
randomness. The field check against the simulator (CPU for the chain families, a GPU for the cloth) is
scripts/ablations/check_cstate.py."""
from __future__ import annotations

import pathlib
import re
import sys
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cond_info as ci  # noqa: E402  -- C00 / C10 / C1G: the log patch, the object poses, the geometry block

TOKEN = "cstate"

# task.toml [metadata] family -> (the deformable entities: name -> "chains" | "cloth", the noun the text uses)
FAMILIES = {
    "blanket_flatten": ({"blanket": "cloth"}, "blanket"),
    "blanket_fold": ({"blanket": "cloth"}, "blanket"),
    "fiber_bundling": ({"fiber_bundle": "chains"}, "fiber bundle"),
    "knot_untying": ({"rope": "chains"}, "rope"),
    "orbits": ({"cord": "chains"}, "cord"),
    "rope_threading": ({"rope": "chains"}, "rope"),
}

# ---- server.py --------------------------------------------------------------------------------------------------
# The C1G geometry block (cond_info._geo_only_fn) with the deformable bodies left out of geo["parts"] (a cloth has no
# links at all, and a chain's free links would be given "at the start", which says nothing about where they are now) and
# out of geo["grasp_regions"] (a cloth's soft patch has no link; a chain's regions are under geo["deformables"]).
GEO_LOOP_OLD = "    for name, ent in S._entities().items():\n        fixed, links, base = "
GEO_LOOP_NEW = ("    for name, ent in S._entities().items():\n"
                "        if name in DEFORMABLE:   # cstate: under geo[\"deformables\"] (layout) and meta[\"deformables\"] (now)\n"
                "            continue\n"
                "        fixed, links, base = ")
GEO_REGION_OLD = '    for t in (grasp.get("config") or {}).get("targets") or ():\n'
GEO_REGION_NEW = ('    for t in (grasp.get("config") or {}).get("targets") or ():\n'
                  '        if t.get("entity") in DEFORMABLE:   # cstate: a deformable body\'s regions are under geo["deformables"]\n'
                  '            continue\n')
GEO_BODY_OLD = 'GEOMETRY_BODY = json.dumps(geometry_robot(), separators=(",", ":")).encode()\n'
GEO_BODY_NEW = ('GEOMETRY_BODY = json.dumps({**geometry_robot(), "deformables": deformables_static()},\n'
                '                           separators=(",", ":")).encode()   # cstate: + the deformable bodies\' layout\n')
STATE_FN = r'''
# ---------------------------------------------------------------- cstate (ablation E3, "+state"): the deformable body
# Every response's meta["deformables"] (deformables_robot) is the current shape of the task's deformable bodies, in the
# agent-facing frame (the robot frame, as every pose of this service); GET /geometry's "deformables" (deformables_static,
# built once at startup from the model alone) is their layout, which never changes. DEFORMABLE (the generator writes it
# per family): name -> "cloth" (the simulation mesh's vertices, with the grid's shape when the mesh is the authored
# regular grid) or "chains" (capsule chains: a rope_chain rope's links in material order, or an articulated body's links
# that each carry one capsule, chained along its kinematic tree; its other links by pose, or by their shapes once when
# welded to the world). A chain is its centreline: link k's capsule axis runs from point k (its end at the link's joint
# with the link before it) to point k + 1 (the next link's joint; the last point: the last link's far end). Reads only:
# MjData's kinematics (the world refreshes them at every tick's end) and the cloth's vertex view; no task state, no
# success, no contacts are read, and nothing is written, so the state hash and the fresh-process replay are unchanged.
DEFORMABLE = @DEFORMABLE@
_LAYOUT: dict = {}          # name -> what deformables_robot reads (bodies, link-frame axis points, the cloth's grid)


def _robot_points(X) -> list:
    """World points (n, 3) -> the robot frame, 0.01 mm (meta["objects"] precision)."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3) @ T_RW[:3, :3].T + T_RW[:3, 3]
    return [[round(float(v), 5) for v in x] for x in X]


def _geom_shape(g: int) -> dict:
    """Geom g's shape (geometry_robot's encoding), pos / quat relative to its body."""
    import mujoco
    from scipy.spatial import ConvexHull
    m = S.env._world.model; T = mujoco.mjtGeom; t, s = int(m.geom_type[g]), m.geom_size[g]
    if t == T.mjGEOM_BOX: out = {"type": "box", "half_extents": _rr(s)}
    elif t == T.mjGEOM_SPHERE: out = {"type": "sphere", "radius": _rr(s[:1])[0]}
    elif t == T.mjGEOM_CAPSULE: out = {"type": "capsule", "radius": _rr(s[:1])[0], "half_length": _rr(s[1:2])[0]}
    elif t == T.mjGEOM_CYLINDER: out = {"type": "cylinder", "radius": _rr(s[:1])[0], "half_length": _rr(s[1:2])[0]}
    elif t == T.mjGEOM_ELLIPSOID: out = {"type": "ellipsoid", "radii": _rr(s)}
    elif t == T.mjGEOM_MESH:
        i = int(m.geom_dataid[g]); a = int(m.mesh_vertadr[i])
        v = np.asarray(m.mesh_vert[a:a + int(m.mesh_vertnum[i])], dtype=np.float64)
        try:
            v = v[ConvexHull(v).vertices]
        except Exception:              # noqa: BLE001 -- a flat mesh has no 3D hull: all its points
            pass
        out = {"type": "convex", "vertices": [_rr(x) for x in v]}
    else:
        raise RuntimeError(f"cstate: geom {g} has an unsupported type {t}")
    n = m.geom(g).name.split("::", 1)[-1]
    if n: out["name"] = n
    out["pos"], out["quat"] = _rr(m.geom_pos[g]), _rr(m.geom_quat[g])
    return out


def _collision_geoms(b: int) -> list:
    m = S.env._world.model; a = int(m.body_geomadr[b])
    return [g for g in range(a, a + int(m.body_geomnum[b])) if m.geom_contype[g] or m.geom_conaffinity[g]]


def _axis(b: int, g: int) -> tuple:
    """Capsule (or end-cap sphere) g of body b: its axis end points in b's frame, the one at the link's joint with the
    link before it first, and its radius. That joint is the body's one hinge or ball joint, at its anchor (the cord's
    second half is authored with its joints at the far end of each link's capsule), else the body's origin (a free body:
    a rope_chain link, in material order from its origin; a body welded to its parent)."""
    import mujoco
    m = S.env._world.model; t = int(m.geom_type[g]); c = np.asarray(m.geom_pos[g], dtype=np.float64)
    r = float(m.geom_size[g][0])
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return c, c.copy(), r
    if t != mujoco.mjtGeom.mjGEOM_CAPSULE:
        raise RuntimeError(f"cstate: chain link body {b} carries geom type {t}, not a capsule")
    j, nj = int(m.body_jntadr[b]), int(m.body_jntnum[b])
    anchor = np.asarray(m.jnt_pos[j], dtype=np.float64) if nj == 1 and int(m.jnt_type[j]) != mujoco.mjtJoint.mjJNT_FREE \
        else np.zeros(3)
    z = M.quat_to_mat(np.asarray(m.geom_quat[g], dtype=np.float64))[:, 2] * float(m.geom_size[g][1])
    a, e = c - z, c + z
    return (a, e, r) if np.linalg.norm(a - anchor) <= np.linalg.norm(e - anchor) else (e, a, r)


def _chains_layout(name: str) -> tuple:
    import mujoco
    w = S.env._world; m, d = w.model, w.data
    ent = S._entities()[name]
    names = {int(l.idx): l.name for l in ent.links}
    rope = getattr(w, "_ropes", {}).get(name)
    if rope is not None:                      # a rope_chain rope: its free links in material order
        chains = [[int(b) for b in rope.bodies]]
    else:                                     # an articulated body: maximal paths of capsule links in its tree
        caps = {b for b in names if len(_collision_geoms(b)) == 1
                and int(m.geom_type[_collision_geoms(b)[0]]) == mujoco.mjtGeom.mjGEOM_CAPSULE}
        kids: dict = {}
        for b in sorted(caps):
            kids.setdefault(int(m.body_parentid[b]), []).append(b)
        chains = []
        for b in sorted(caps):
            p = int(m.body_parentid[b])
            if p in caps and len(kids.get(p, ())) == 1:
                continue                      # b continues its parent's chain
            ch = [b]
            while len(kids.get(ch[-1], ())) == 1:
                ch.append(kids[ch[-1]][0])
            chains.append(ch)
    if not chains:
        raise RuntimeError(f"cstate: {name} has no capsule chain")
    in_chain = {b for ch in chains for b in ch}
    lay = {"kind": "chains", "chains": [], "rigid": {}}
    static = {"kind": "capsule_chains", "chains": [], "rigid_links": {}, "welded_links": {}, "grasp_regions": []}
    where = {}                                # body -> (chain index, link index)
    for i, ch in enumerate(chains):
        axes = []
        for k, b in enumerate(ch):
            gs = _collision_geoms(b)
            if len(gs) != 1:
                raise RuntimeError(f"cstate: chain link {names.get(b, b)} of {name} has {len(gs)} collision geoms")
            axes.append(_axis(b, gs[0]))
            where[b] = (i, k)
        radii = sorted({round(r, 6) for _, _, r in axes})
        lay["chains"].append({"bodies": np.asarray(ch, dtype=np.int64), "P": np.stack([a for a, _, _ in axes]),
                              "D": np.stack([e for _, e, _ in axes]), "radius": radii[0] if len(radii) == 1 else radii})
        p = int(m.body_parentid[ch[0]])
        static["chains"].append({"links": [names[b] for b in ch], "parent": names.get(p),
                                 "radius": lay["chains"][-1]["radius"]})
    for b in sorted(names):
        gs = _collision_geoms(b)
        if b in in_chain or not gs:
            continue
        if int(m.body_weldid[b]) == 0:        # welded to the world: its shapes in the agent-facing frame, once
            shapes = []
            for g in gs:
                s = _geom_shape(g); q_b = np.asarray(d.xquat[b], dtype=np.float64)
                pos, quat = to_robot(np.asarray(d.xpos[b], dtype=np.float64) + M.quat_to_mat(q_b) @ np.asarray(m.geom_pos[g]),
                                     _qmul(q_b, np.asarray(m.geom_quat[g], dtype=np.float64)))
                s["pos"], s["quat"] = _rr(pos), _rr(quat)
                shapes.append(s)
            static["welded_links"][names[b]] = {"shapes": shapes}
        else:                                 # a rigid link that moves: its pose now in meta, its shapes in its own frame
            lay["rigid"][names[b]] = b
            static["rigid_links"][names[b]] = {"shapes": [_geom_shape(g) for g in gs]}
    by_name = {v: k for k, v in names.items()}
    grasp = w.read_state().get("grasp") or {}
    for t in (grasp.get("config") or {}).get("targets") or ():
        if t.get("entity") != name or "link" not in t:
            continue
        b = by_name[t["link"]]; c = np.asarray(t["center_m"], dtype=np.float64); r = round(float(t["radius_m"]), 6)
        if b in where:                        # on a chain link: where along its capsule axis
            i, k = where[b]; ch = lay["chains"][i]; a, e = ch["P"][k], ch["D"][k]; u = e - a
            s = float(np.dot(c - a, u) / np.dot(u, u)) if np.dot(u, u) > 0 else 0.0
            off = float(np.linalg.norm(c - (a + s * u)))
            static["grasp_regions"].append({"link": t["link"], "chain": i, "segment": k, "t": round(s, 4),
                                            "off_axis_m": round(off, 6), "radius": r})
        else:
            static["grasp_regions"].append({"link": t["link"], "center": _rr(c), "radius": r})
    for k in ("rigid_links", "welded_links", "grasp_regions"):
        if not static[k]:
            del static[k]
    return lay, static


def _cloth_layout(name: str) -> tuple:
    c = S.env._world._cloths[name]
    rest, faces = np.asarray(c.rest, dtype=np.float64), np.asarray(c.faces, dtype=np.int64)
    xs, ys = np.unique(np.round(rest[:, 0], 6)), np.unique(np.round(rest[:, 1], 6))
    ix, iy = np.searchsorted(xs, np.round(rest[:, 0], 6)), np.searchsorted(ys, np.round(rest[:, 1], 6))
    grid = spacing = None
    if (np.ptp(rest[:, 2]) < 1e-9 and len(xs) * len(ys) == len(rest) and len(xs) > 1 and len(ys) > 1
            and np.array_equal(ix * len(ys) + iy, np.arange(len(rest)))):
        grid, spacing = [int(len(xs)), int(len(ys))], [round(float(np.diff(xs).mean()), 6), round(float(np.diff(ys).mean()), 6)]
    radius = getattr(c, "contact_radius_m", None)
    static = {"kind": "cloth", "n_vertices": int(len(rest)), "grid_shape": grid, "spacing_m": spacing,
              "triangles": faces.tolist(), **({"contact_radius_m": round(float(radius), 6)} if radius is not None else {})}
    return {"kind": "cloth", "grid_shape": grid}, static


def deformables_static() -> dict:
    """GET /geometry's "deformables": each deformable body's layout; also builds _LAYOUT for deformables_robot."""
    out = {}
    for name, kind in sorted(DEFORMABLE.items()):
        if name not in S._entities():
            raise RuntimeError(f"cstate: no deformable entity {name!r} in this scene")
        lay, static = _cloth_layout(name) if kind == "cloth" else _chains_layout(name)
        _LAYOUT[name] = lay; out[name] = static
    return out


def deformables_robot() -> dict:
    """meta["deformables"]: each deformable body's current shape, in the agent-facing frame."""
    w = S.env._world; d = w.data; out = {}
    for name, lay in _LAYOUT.items():
        if lay["kind"] == "cloth":
            out[name] = {"kind": "cloth", "vertices": _robot_points(w._cloths[name].vertices(d)), "grid_shape": lay["grid_shape"]}
            continue
        xp = np.asarray(d.xpos, dtype=np.float64); xm = np.asarray(d.xmat, dtype=np.float64).reshape(-1, 3, 3)
        chains = []
        for ch in lay["chains"]:
            B = ch["bodies"]
            P = xp[B] + np.einsum("nij,nj->ni", xm[B], ch["P"])
            end = xp[B[-1]] + xm[B[-1]] @ ch["D"][-1]
            chains.append({"points": _robot_points(np.vstack([P, end[None]])), "radius": ch["radius"]})
        rec = {"kind": "chains", "chains": chains}
        if lay["rigid"]:
            rec["links"] = {}
            for n, b in lay["rigid"].items():
                p, q = to_robot(np.asarray(d.xpos[b], dtype=np.float64), np.asarray(d.xquat[b], dtype=np.float64))
                rec["links"][n] = {"position": [round(float(x), 5) for x in p], "orientation_wxyz": [round(float(x), 6) for x in q]}
        out[name] = rec
    return out

'''
STATE_META_OLD = '"objects": objects_robot()}   # probing v2 C10: the task objects\' poses, robot frame'
STATE_META_NEW = ('"objects": objects_robot(), "deformables": deformables_robot()}   # probing v2 C10 + cstate: the poses, '
                  'then the deformable bodies\' shapes')
STATE_FRAME_NEW = ('"the scene\'s poses and the deformable bodies\' shapes are in every response\'s meta (objects, deformables) and '
                   'the part shapes behind GET /geometry; every pose, camera extrinsic and kinematics answer of this service is in it "')
STATE_SPEC_NEW = (ci.GEO_SPEC_NEW
                  + '    "deformable_state": {"where": "meta deformables in every response of /submit and /observe", "layout": '
                    '"GET /geometry deformables"},   # cstate\n')

# ---- instruction.md ---------------------------------------------------------------------------------------------
RULE_HEAD = "**Object poses, part shapes and the @NOUN@'s state are given (this variant).**"
RULE_END = ("The part shapes are behind `r.geometry()` and the @NOUN@'s current state is in every response's "
            "`meta[\"deformables\"]` (both under `Part shapes and deformable state`, below). No other internal state of the "
            "simulator is given: no joint readings, no grasp phase, no task progress or history. The safety readings are in "
            "`meta[\"safety\"]` (listed under Action types, below). The simulator itself, its source, the success "
            "implementation and its verdicts, forward rollouts and the reference solution are not exposed; the task "
            "requirements above still apply in full.")
PARA_NEW = ("their shapes are given as well (`Part shapes and deformable state`, below), and the @NOUN@'s current state, "
            "but no other internal state: the rest comes from the cameras.")
POSITIONS_NEW = "the scene's poses are in `meta[\"objects\"]` and `meta[\"deformables\"]`"
CLIENT_LINE = 'obs.meta["deformables"]'
CLIENT_NEW = (ci.GEO_CLIENT_NEW
              + CLIENT_LINE.ljust(46) + "# this variant, every response: the @NOUN@'s current shape (below)\n")
SECTION_HEAD = ("### Part shapes (this variant)\n", "### Part shapes and deformable state (this variant)\n")
SECTION_ONLY = ("Only the task's parts and fixtures are there:",
                "Only the task's rigid parts and fixtures are there (the @NOUN@ is under `geo[\"deformables\"]`, below):")
SECTION_NOT_GIVEN = r"\*\*Not given\.\*\* [^\n]*\n\n"
NOT_GIVEN = ("**Not given.** Nothing else of the simulator's internal state is reported: no speeds of single points, no "
             "joint readings, no forces or tensions, no grasp phase, no task progress or history, and not how far any success "
             "condition is met; contacts and contact forces are not reported, beyond the safety readings. The rest comes from "
             "the cameras and your own measurements.\n\n")
CHAIN_POINTS = "one point per link plus one, in link order: link k's capsule axis runs from point k to point k + 1"
STATE_TEXT = {
    "rope": ("**Deformable state.** Every response's `meta[\"deformables\"][\"rope\"]` is the rope's current shape, in the "
             "world frame of `spec[\"world\"]`: `chains[0][\"points\"]` is its centreline (" + CHAIN_POINTS + ") and "
             "`chains[0][\"radius\"]` its radius (m). The rope is the chain of capsules of that radius between consecutive "
             "points; the end caps reach one radius beyond the first and the last point. `geo[\"deformables\"][\"rope\"]` "
             "gives the chain's link names and its grasp regions (`segment` k: the capsule between points k and k + 1, at "
             "the fraction `t` along it).\n\n"),
    "chains": ("**Deformable state.** Every response's `meta[\"deformables\"][\"@NAME@\"]` is the @NOUN@'s current shape, in "
               "the world frame of `spec[\"world\"]`: `chains`, one entry per chain of capsule links, each with its "
               "centreline `points` (" + CHAIN_POINTS + ") and its `radius` (m); a chain is the capsules of that radius "
               "between consecutive points. `links[link]` gives the `position` and `orientation_wxyz` of each rigid link of "
               "the @NOUN@ that is not a capsule. `geo[\"deformables\"][\"@NAME@\"]` gives the layout: each chain's link "
               "names and the link it hangs from (`parent`); the rigid links' shapes in their own frames (`rigid_links`: a "
               "point x of a shape is at T(link) · (`pos` + R(`quat`) · x)); the shapes of its links fixed to the world, in "
               "the world frame (`welded_links`); and its grasp regions (on a chain: `segment` k, the capsule between its "
               "points k and k + 1, at the fraction `t` along it; on a rigid link: `center` in that link's frame). @FAMILY@"
               "\n\n"),
    "cloth": ("**Deformable state.** Every response's `meta[\"deformables\"][\"blanket\"]` is the blanket's current shape, in "
              "the world frame of `spec[\"world\"]`: `vertices`, the positions of the simulation mesh's vertices, and "
              "`grid_shape` [A, B]: the mesh is a regular grid of A x B vertices, and vertex (a, b) is `vertices[a * B + b]`. "
              "In the flat blanket vertex (a, b) lies a · `spacing_m[0]` and b · `spacing_m[1]` from the corner vertex "
              "(0, 0), a along the blanket's long side and b along its short side. Each grid cell is split into two "
              "triangles, (a, b), (a + 1, b), (a + 1, b + 1) and (a, b), (a + 1, b + 1), (a, b + 1). "
              "`geo[\"deformables\"][\"blanket\"]` gives `grid_shape`, `spacing_m`, the `triangles` (vertex indices) and "
              "`contact_radius_m`, how far the cloth's contact surface lies from its vertices. The cloth points the "
              "requirements are measured at are points of these triangles at fixed barycentric weights (the grid vertices "
              "among them).\n\n"),
}
FAMILY_TEXT = {
    "orbits": ("Here the cord is two chains, both starting at the wooden handle (the rigid link `handle`, "
               "`links[\"handle\"]`); their far ends are tied to each other: the middle of the cord loop."),
    "fiber_bundling": ("Here the chains are the sheath (`lead_00` onwards, from the rear terminal; `lead_00` is fixed) and "
                       "one chain per fine fiber (`fiber_<i>_00` at the sheath's front to the tip, the last link); the "
                       "sheath's stiff front segments (`trunk_<k>`) are rigid links, and the rear terminal is welded to the "
                       "world."),
}
LEFT = (r"(?i)no privileged state|no shapes|nothing reports|no positions of the scene|only the safety readings|"
        r"where the\s+objects are|no object poses")
TOML = ("no privileged state", "object poses, part shapes and deformable state given")


def family_of(task_dir: pathlib.Path) -> str:
    meta = tomllib.loads((task_dir / "task.toml").read_text()).get("metadata") or {}
    fam = meta.get("family")
    if meta.get("mode") != "standard":
        raise SystemExit(f"{task_dir.name}: cstate is a Standard-mode condition (mode {meta.get('mode')!r})")
    if fam not in FAMILIES:
        raise SystemExit(f"{task_dir.name}: no cstate representation for the family {fam!r} (known: {', '.join(FAMILIES)})")
    return fam


def server_block(family: str) -> str:
    """The code that goes before meta_common: the C1G geometry block with the deformable bodies left out, and the
    deformable state."""
    deformable = dict(sorted(FAMILIES[family][0].items()))
    t = ci._geo_only_fn()
    t = ci.sub1(t, GEO_LOOP_OLD, GEO_LOOP_NEW, "C1G block: the parts loop")
    t = ci.sub1(t, GEO_REGION_OLD, GEO_REGION_NEW, "C1G block: the grasp regions loop")
    t = ci.sub1(t, GEO_BODY_OLD, STATE_FN.replace("@DEFORMABLE@", repr(deformable)).lstrip("\n") + "\n" + GEO_BODY_NEW,
                "C1G block: GEOMETRY_BODY")
    return t


def section(family: str) -> str:
    entities, noun = FAMILIES[family]
    (name, kind), = entities.items()
    t = ci.sub1(ci.GEO_SECTION, *SECTION_HEAD, "C1G section heading")
    t = ci.sub1(t, SECTION_ONLY[0], SECTION_ONLY[1].replace("@NOUN@", noun), "C1G section: parts only")
    text = STATE_TEXT["cloth" if kind == "cloth" else ("rope" if name == "rope" else "chains")]
    text = text.replace("@NAME@", name).replace("@NOUN@", noun).replace(" @FAMILY@", " " + FAMILY_TEXT.get(family, "")).rstrip(" ")
    text = text.replace(" \n\n", "\n\n")
    return ci.sub1(t, SECTION_NOT_GIVEN, text + NOT_GIVEN, "C1G section: not given", regex=True)


def patch_state(src: pathlib.Path, dst: pathlib.Path) -> None:
    """C1G's geometry and the client's Robot.geometry() with the deformable bodies apart, their state in every
    response, the texts. Runs after cond_info.patch_log and cond_info.patch_objects."""
    name = src.name
    family = family_of(src)
    noun = FAMILIES[family][1]
    srv = dst / "environment" / "sim" / "server.py"
    t = srv.read_text()
    t = ci.sub1(t, ci.OBJ_FN_ANCHOR, server_block(family) + ci.OBJ_FN_ANCHOR.lstrip("\n"), f"{name} server.py cstate block")
    t = ci.sub1(t, STATE_META_OLD, STATE_META_NEW, f"{name} server.py meta_common (deformables)")
    t = ci.sub1(t, ci.FULL_GET_OLD, ci.FULL_GET_NEW, f"{name} server.py /geometry")
    t = ci.sub1(t, ci.FULL_FRAME_OLD, STATE_FRAME_NEW, f"{name} server.py world frame")
    t = ci.sub1(t, ci.FULL_SPEC_OLD, STATE_SPEC_NEW, f"{name} server.py spec")
    if '"state": state_robot()' in t:
        raise SystemExit(f"{name}: cstate must not report meta state")
    compile(t, str(srv), "exec")
    srv.write_text(t)
    cli = (dst / "environment" / "rcb_client.py").read_text()
    for old, new, what in ci.FULL_CLI:
        cli = ci.sub1(cli, old, new, f"{name} rcb_client.py {what}")
    compile(cli, str(dst / "environment" / "rcb_client.py"), "exec")
    (dst / "environment" / "rcb_client.py").write_text(cli)
    ins = (dst / "instruction.md").read_text()
    ins = ci.sub1(ins, ci.GEO_RULE_HEAD[0], RULE_HEAD.replace("@NOUN@", noun), f"{name} instruction (Rules heading)")
    ins = ci.sub1(ins, ci.FULL_RULE_END_OLD, RULE_END.replace("@NOUN@", noun), f"{name} instruction (Rules end)", regex=True)
    if ci.FULL_PARA_OLD in ins:
        ins = ci.sub1(ins, ci.FULL_PARA_OLD, PARA_NEW.replace("@NOUN@", noun), f"{name} instruction (Objects here)")
    if ci.GEO_POSITIONS[0] in ins:
        ins = ci.sub1(ins, ci.GEO_POSITIONS[0], POSITIONS_NEW, f"{name} instruction (spec world)")
    ins = ci.sub1(ins, ci.FULL_CLIENT_ANCHOR, CLIENT_NEW.replace("@NOUN@", noun) + ci.FULL_CLIENT_ANCHOR,
                  f"{name} instruction (client)")
    ins = ci.sub1(ins, "## External material\n", section(family) + "## External material\n", f"{name} instruction (section)")
    (dst / "instruction.md").write_text(ins)
    toml = (dst / "task.toml").read_text()
    (dst / "task.toml").write_text(ci.sub1(toml, *TOML, f"{name} task.toml description (cstate)"))
    problems = leak_check(dst)
    if problems:
        raise SystemExit(f"{name}: " + "; ".join(problems))


def make(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Edit dst (a full copy of the released Standard task src) into CS: C00's log patch, C10's object poses, C1G's part
    shapes without the deformable bodies, and the deformable bodies' state."""
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    family_of(src)
    ci.patch_log(src, dst)
    ci.patch_objects(src, dst)
    patch_state(src, dst)


# ---- leak check (static; the field check against the simulator is check_cstate.py) -----------------------------
# Names the cstate block must never read: the task's state and verdicts, rewards, contacts, the safety monitor.
BLOCK_FORBIDDEN = re.compile(r"_task\b|task_state|_reward|success|episode_end|\bmonitor\b|contacts?\(|safety|state_robot\(|"
                             r"\bmechanism\b|_RULES_KEY")
# A coordinate tuple with a decimal point in the agent's text (rcb_limited.world_text's rule, without its robot exception)
COORDS = re.compile(r"\(\s*-?\d+\.\d+\s*,\s*-?\d+(?:\.\d+)?\s*(?:,\s*-?\d+(?:\.\d+)?\s*)?\)")


def leak_check(dst: pathlib.Path) -> list[str]:
    """What is wrong with a generated cstate directory (empty: clean): the service, the client and the text."""
    dst = pathlib.Path(dst); out = []
    try:
        family = family_of(dst)
    except SystemExit as exc:
        return [str(exc)]
    entities, noun = FAMILIES[family]
    t = (dst / "environment" / "sim" / "server.py").read_text()
    try:
        compile(t, "server.py", "exec")
    except SyntaxError as exc:
        out.append(f"server.py does not compile: {exc}")
    if t.count("\ndef deformables_robot() -> dict:\n") != 1 or t.count("deformables_robot()}") != 1:
        out.append("server.py: deformables_robot is not defined once and called once in meta_common")
    if f"DEFORMABLE = {dict(sorted(entities.items()))!r}\n" not in t:
        out.append(f"server.py: DEFORMABLE is not {dict(sorted(entities.items()))!r}")
    if '"state": state_robot()' in t or "state_robot()}" in t:
        out.append("server.py reports meta state")
    a, b = t.find("# ---------------------------------------------------------------- cstate"), t.find("\nGEOMETRY_BODY = ")
    block = t[a:b] if 0 <= a < b else ""
    if not block:
        out.append("server.py: no cstate block before GEOMETRY_BODY")
    else:
        code = "\n".join(line.split("#", 1)[0] for line in block.splitlines())   # the code, not its comments
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        bad = sorted(set(BLOCK_FORBIDDEN.findall(code)))
        if bad:
            out.append(f"server.py: the cstate block reads {bad}")
        if "to_robot(" not in code or "T_RW" not in code:
            out.append("server.py: the cstate block does not move its points into the robot frame")
    if "/geometry" not in t or "def geometry(self)" not in (dst / "environment" / "rcb_client.py").read_text():
        out.append("GET /geometry or Robot.geometry() missing")
    ins = (dst / "instruction.md").read_text()
    left = re.findall(LEFT, ins)
    if left:
        out.append(f"instruction: statements against the given state left: {left}")
    if 'meta["state"]' in ins:
        out.append("instruction points at meta state")
    for want in ('meta["deformables"]', "### Part shapes and deformable state (this variant)", 'meta["objects"]'):
        if want not in ins:
            out.append(f"instruction lacks {want!r}")
    i = ins.find("### Part shapes and deformable state (this variant)")
    new_text = ins[i:ins.find("## External material\n", i)] if i >= 0 else ""
    if COORDS.search(new_text):
        out.append(f"instruction: a coordinate in the variant's section: {COORDS.search(new_text).group(0)}")
    toml = (dst / "task.toml").read_text()
    if "no privileged state" in toml or TOML[1] not in toml:
        out.append("task.toml description not updated")
    return out
