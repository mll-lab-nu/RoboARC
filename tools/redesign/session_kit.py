"""Session kit: build and run a robot-level MuJoCo Session for a redesign scene, in-process, with no file edits outside
the redesign dirs (no registry / tier / rcb_topogym change).

The three soft-body families already run on MuJoCo in rcb_topogym.TASKS (physics_engine "mujoco", their profiles in
topogym/world/mujoco/profiles.py by family). A MuJoCo rope or cloth needs a frozen **birth** (SimulatorConfig.
mujoco_birth): ropes an initial-state numeric archive holding the FEM rope vertices (`{"fem": {name: (N, 3)}}`), cloths
an instance-like JSON holding the PBD particles (`soft_state.<name>.position_m`). A redesign scene is therefore:
  (1) a variation of an existing tier (its task class and task_parameters), or a new task class in the family's
      redesign package registered in-process (register_variation: appends a variation record, like
      rcb_topogym._register_cube does for its fixture);
  (2) a new birth for the deformable (rope_birth_from_centerline + write_rope_birth, or cloth_birth_json), plus
      whatever rigid obstacles the task class declares;
  (3) Session(task, variation_id=..., seed=..., task_parameters=..., simulator_overrides={"mujoco_birth": ...}).
freeze_instance() then writes a frozen instance (instance.json + instance_assets/mujoco/...) that every tool here takes
(`--new-instance`), and that replays deterministically.

    import session_kit as sk
    sk.use_lane_tmp("knot_untying")                         # Session temp dirs under $RD_WORK/<lane>/tmp
    base = sk.instance_settings("<instances>/knot-untying-easy-i00/instance.json")
    V = sk.load_rope_birth(base["mujoco_birth"])["rope"]    # the released birth (FEM vertices)
    manifest = sk.write_rope_birth(out_dir / "birth", {"rope": V + [0.05, 0, 0]}, note="easy shifted 5 cm")
    s = sk.make_session("knot_untying", base["variation_id"], seed=base["seed"],
                        task_parameters=base["task_parameters"], overrides={"mujoco_birth": str(manifest)})
    s.step(s.hold_action()); print(s.status())
    sk.freeze_instance(..., out_json)                        # a frozen instance for the harness / breakers / previews

Run (the cloud host, env sourced; a Session build takes ~5-50 s):
    python -m session_kit example-knot <out_dir>          # the example above, + previews of the new scene
    python -m session_kit describe <instance.json>         # what to pass to make_session to rebuild an instance
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def use_lane_tmp(lane: str) -> str:
    """Keep the Session's temporary episode folders inside the lane's work tree: $RD_WORK/<lane>/tmp (RD_WORK defaults to
    <the system temp dir>/roboarc-redesign)."""
    d = Path(os.environ.get("RD_WORK") or Path(tempfile.gettempdir()) / "roboarc-redesign") / lane / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(d)
    tempfile.tempdir = str(d)
    return str(d)


def register_variation(task_id: str, variation_id: str, implementation: str, *, difficulty: str,
                       family_id: str = "redesign", model_revision: str | None = None, aliases=()) -> dict:
    """Append a variation record in-process (topogym.config.variation_records.RECORDS); idempotent.
    implementation: "topogym.tasks.<family>.redesign.<module>:<TaskClass>" (a TaskDefinition subclass)."""
    from topogym.config import variation_records
    for row in variation_records.RECORDS:
        if row["task_id"] == task_id and row["variation_id"] == variation_id:
            return row
    rec = dict(task_id=task_id, variation_id=variation_id, difficulty=difficulty, family_id=family_id,
               model_revision=model_revision or f"{variation_id}_redesign_v0", implementation=implementation,
               aliases=list(aliases), status="integrated", reason=None, validation="redesign_dev", pool="redesign")
    variation_records.RECORDS = (*variation_records.RECORDS, rec)
    return rec


def instance_settings(instance_json) -> dict:
    """What make_session needs to rebuild a frozen instance's scene: variation, seed, task_parameters, the simulator
    overrides it was made with (mujoco_birth resolved to an absolute path), gripper space."""
    p = Path(instance_json).resolve()
    inst = json.loads(p.read_text())
    sim = dict(inst["simulator"])
    birth = sim.get("mujoco_birth")
    if birth and not Path(birth).is_absolute():
        birth = str(p.parent / birth)
    tp = dict(sim.get("task_parameters") or {})
    for k in ("initial_state_path",):
        if tp.get(k) and not Path(tp[k]).is_absolute():
            tp[k] = str(p.parent / tp[k])
    keep = {k: sim[k] for k in ("physics_dt_s", "solver_substeps", "control_decimation", "reset_physics_steps",
                                "grasp_assist", "physics_engine") if k in sim}
    return dict(task=inst["task"], variation_id=sim["variation_id"], seed=int(sim["seed"]), task_parameters=tp or None,
                overrides=keep, mujoco_birth=birth, gripper_space=inst.get("gripper_space", "width"),
                state_sha=inst.get("state_sha"))


def use_frozen_assets(instance_json) -> list:
    """Make the next Session builds use an instance's frozen generated assets (its rope volume: a regenerated volume
    can differ by a few tetrahedra, 40747 vs 40769 vertices on knot hard, and a birth must match it vertex for vertex).
    The same hook rcb_topogym installs for a bound instance. -> the frozen file names."""
    import rcb_topogym as rt
    p = Path(instance_json).resolve()
    inst = json.loads(p.read_text())
    rt._install_frozen_assets(p.parent, inst.get("asset_identity") or {})
    return sorted(getattr(rt, "_FROZEN_ASSETS", {}))


def make_session(task: str, variation_id: str, *, seed: int = 0, task_parameters: dict | None = None,
                 overrides: dict | None = None, gripper_space: str = "native", register: dict | None = None,
                 frozen_assets_from: str | None = None):
    """A fresh (unfrozen) MuJoCo Session; register: kwargs of register_variation for a new task class;
    frozen_assets_from: an instance.json whose frozen generated assets (rope volume) the build must use."""
    if register:
        register_variation(**register)
    if frozen_assets_from:
        use_frozen_assets(frozen_assets_from)
    from rcb_topogym import Session
    return Session(task, variation_id=variation_id, seed=seed, task_parameters=task_parameters,
                   simulator_overrides=overrides or None, gripper_space=gripper_space, strict=False)


def freeze_instance(task: str, variation_id: str, out_json, *, seed: int = 0, task_parameters: dict | None = None,
                    overrides: dict | None = None, gripper_space: str = "native", register: dict | None = None) -> dict:
    """rcb_topogym make-instance in-process: <out_json> + instance_assets/ (the frozen MuJoCo scene). The birth file
    stays where it is (an absolute path in the instance); copy it next to the instance first to keep it self-contained."""
    import argparse
    import rcb_topogym as rt
    if register:
        register_variation(**register)
    out = Path(out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / "_kit_inputs"
    tmp.mkdir(exist_ok=True)
    ov = tmp / "overrides.json"; ov.write_text(json.dumps(overrides or {}))
    tp = None
    if task_parameters:
        tp = tmp / "task_parameters.json"; tp.write_text(json.dumps(task_parameters))
    args = argparse.Namespace(task=task, eval_index=None, variation=variation_id, scene_seed=int(seed),
                              task_parameters=None if tp is None else str(tp), simulator_overrides=str(ov),
                              ipc_settled_state=None, gripper_space=gripper_space, out=str(out))
    rt.cmd_make_instance(args)
    inst = json.loads(out.read_text())
    return dict(instance=str(out), state_sha=inst.get("state_sha"))


# ------------------------------------------------------------------------------------------------ births
def load_rope_birth(manifest_path) -> dict:
    """{name: FEM vertices (N, 3)} of a rope birth archive (an IPC settled state or write_rope_birth's)."""
    from topogym.world.initial_state import load_numeric_archive
    payload, _ = load_numeric_archive(Path(manifest_path))
    return {k: np.asarray(v, dtype=np.float64) for k, v in (payload.get("fem") or {}).items()}


def write_rope_birth(folder, ropes: dict, note: str = "") -> Path:
    """A rope birth archive {"fem": {name: (N, 3)}} -> <folder>/manifest.json (pass it as mujoco_birth)."""
    from topogym.world.initial_state import save_numeric_archive
    payload = {"fem": {k: np.asarray(v, dtype=np.float64) for k, v in ropes.items()}}
    return save_numeric_archive(Path(folder), payload, dict(source="redesign session_kit", note=note))


def rope_birth_from_centerline(volume_npz, centerline, *, up=(0.0, 0.0, 1.0), roll: float = 0.0) -> tuple:
    """The rope volume's FEM vertices swept along a new centre line: every vertex keeps its rest material coordinate s
    (rest x) and cross-section offset (rest y, z), placed at C(s) + y N(s) + z B(s) with a parallel-transport frame
    (N starts horizontal, perpendicular to the first tangent and `up`, rolled by `roll`). The centre line is
    re-parametrised by arc length over the rope's material length. -> (vertices (N, 3), info: stretch of the given
    line vs the material length)."""
    with np.load(volume_npz, allow_pickle=False) as z:
        rest = np.asarray(z["rest"], dtype=np.float64)
        rings = np.asarray(z["surface_ring_indices"], dtype=np.int64) if "surface_ring_indices" in z.files else None
    s_v = rest[:, 0]
    st = rest[rings, 0].mean(axis=1) if rings is not None else np.unique(s_v)
    s0, s1 = float(st[0]), float(st[-1])
    C = np.asarray(centerline, dtype=np.float64)
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    L = s1 - s0
    u = (arc / arc[-1]) * L + s0           # material coordinate of each centre-line sample
    T = np.gradient(C, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    n0 = np.cross(np.asarray(up, float), T[0])
    if np.linalg.norm(n0) < 1e-6:
        n0 = np.cross([1.0, 0, 0], T[0])
    n0 /= np.linalg.norm(n0)
    if roll:
        n0 = n0 * np.cos(roll) + np.cross(T[0], n0) * np.sin(roll)
    N = np.zeros_like(C); N[0] = n0
    for i in range(1, len(C)):            # parallel transport (rotation taking T[i-1] to T[i])
        v = np.cross(T[i - 1], T[i]); c = float(T[i - 1] @ T[i])
        if np.linalg.norm(v) < 1e-12:
            N[i] = N[i - 1]
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            Rm = np.eye(3) + vx + vx @ vx / (1 + c)
            N[i] = Rm @ N[i - 1]
        N[i] -= (N[i] @ T[i]) * T[i]; N[i] /= np.linalg.norm(N[i])
    B = np.cross(T, N)
    cy, cz = rest[:, 1], rest[:, 2]
    if rings is not None:                  # the rest centre line (ring centroids) may be offset from y = z = 0
        cy0 = np.interp(s_v, st, rest[rings, 1].mean(axis=1)); cz0 = np.interp(s_v, st, rest[rings, 2].mean(axis=1))
        cy, cz = cy - cy0, cz - cz0
    P = np.stack([np.interp(s_v, u, C[:, k]) for k in range(3)], axis=1)
    Nv = np.stack([np.interp(s_v, u, N[:, k]) for k in range(3)], axis=1)
    Bv = np.stack([np.interp(s_v, u, B[:, k]) for k in range(3)], axis=1)
    V = P + cy[:, None] * Nv + cz[:, None] * Bv
    return V, dict(material_length_m=L, centerline_length_m=float(arc[-1]), stretch=float(arc[-1] / L))


def cloth_birth_json(path, name: str, particles, task_parameters: dict | None = None) -> Path:
    """A cloth birth (cloth.birth_from_instance reads soft_state.<name>.position_m): particles (8181, 3) of the
    task's PBD particle set, e.g. a family generator's folded state."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    P = np.asarray(particles, dtype=np.float64)
    p.write_text(json.dumps(dict(soft_state={name: dict(position_m=P.tolist())},
                                 simulator=dict(task_parameters=task_parameters or {}), state_sha=None,
                                 source="redesign session_kit")))
    return p


# ------------------------------------------------------------------------------------------------ scene specs
# A redesign scene as data: an existing variation's task class (its predicate unchanged) + extra rigid entities
# (posts, blocks, staples, walls, weights) + a new birth of the deformable. build_scene_session(spec) registers the
# variation in-process and returns a robot-level MuJoCo Session; the tooling CLIs take it as
#   --builder session_kit:build_scene_session --builder-args @scene.json
# scene.json:
#   {"task": "knot_untying", "variation_id": "knot_untying_rd_medium", "base_instance": "<an instance.json of the
#    family: task class, task_parameters, simulator settings and the default birth come from it>",
#    "entities": [{"name": "post1", "kind": "cylinder" | "box" | "sphere", "pos": [x, y, z], "quat": [w, x, y, z],
#                  "radius": r, "height": h, "size": [sx, sy, sz], "fixed": true, "on_table": true,
#                  "rho": 800 (movable only), "friction": 0.6, "rgba": [r, g, b, a]}],
#    "birth": {"rope": {"centerline": [[x, y, z], ...]} | {"vertices_npy": path} | {"archive": manifest.json}}
#           | {"cloth": {"particles_npy": path}}   (omitted: the base instance's birth)}
# "pos" is the entity centre in world coordinates; with "on_table" its z is replaced by table + half height.
_ENTITY_KEYS = {"name", "kind", "pos", "quat", "radius", "height", "size", "fixed", "on_table", "rho", "friction", "rgba"}


def _extra_entity_specs(extra, table_height_m):
    from topogym.tasks.base import EntitySpec
    out = []
    for e in extra:
        bad = set(e) - _ENTITY_KEYS
        if bad:
            raise ValueError(f"entity {e.get('name')}: unknown keys {sorted(bad)}")
        kind = e["kind"]
        pos = [float(x) for x in e["pos"]]
        if kind == "box":
            opts = {"size": tuple(float(x) for x in e["size"])}
            half = opts["size"][2] / 2
        elif kind == "cylinder":
            opts = {"radius": float(e["radius"]), "height": float(e["height"])}
            half = opts["height"] / 2
        elif kind == "sphere":
            opts = {"radius": float(e["radius"])}
            half = opts["radius"]
        else:
            raise ValueError(f"entity {e['name']}: kind must be box, cylinder or sphere")
        if e.get("on_table"):
            pos[2] = float(table_height_m) + half
        fixed = bool(e.get("fixed", True))
        material = {"friction": float(e.get("friction", 0.6))}
        if not fixed:
            material["rho"] = float(e.get("rho", 800.0))
        out.append(EntitySpec(e["name"], kind, None, position_m=tuple(pos),
                              quaternion_wxyz=tuple(float(x) for x in e.get("quat", (1, 0, 0, 0))), fixed=fixed,
                              material=material, morph_options=opts,
                              surface={"color": tuple(float(x) for x in e.get("rgba", (.45, .42, .38, 1.)))}))
    return out


def _declare_with_extra(self, asset_dir, table_height_m):
    from dataclasses import replace
    base = type(self).__mro__[1]
    spec = base.declare(self, asset_dir, table_height_m)
    extra = _extra_entity_specs(type(self)._rd_extra, table_height_m)
    names = {e.name for e in spec.entities}
    clash = [e.name for e in extra if e.name in names]
    if clash:
        raise ValueError(f"extra entities clash with the task's own: {clash}")
    return replace(spec, entities=(*spec.entities, *extra))


def scene_variation(task: str, variation_id: str, base_variation: str, extra=(), difficulty: str | None = None) -> str:
    """Register `variation_id` = `base_variation`'s task class + extra rigid entities; returns the implementation."""
    from importlib import import_module
    from topogym.config.registry import variation_registration
    rec = variation_registration(task, base_variation)
    module, clsname = rec["implementation"].split(":")
    Base = getattr(import_module(module), clsname)
    cname = "RD_" + "".join(c if c.isalnum() else "_" for c in variation_id)
    cls = type(cname, (Base,), dict(variation_id=variation_id, model_revision=f"{variation_id}_rd_v0",
                                     difficulty=difficulty or getattr(Base, "difficulty", None),
                                     _rd_extra=list(extra), declare=_declare_with_extra))
    globals()[cname] = cls
    impl = f"{__name__}:{cname}"
    register_variation(task, variation_id, impl, difficulty=difficulty or rec.get("difficulty") or "redesign",
                       family_id="redesign_scene")
    return impl


def _birth_override(birth: dict | None, base: dict, work: Path) -> str | None:
    if not birth:
        return base["mujoco_birth"]
    if "rope" in birth:
        b = birth["rope"]
        if b.get("archive"):
            return str(b["archive"])
        if b.get("vertices_npy"):
            V = np.load(b["vertices_npy"])
        else:
            inst_dir = Path(base["instance"]).resolve().parent
            vol = inst_dir / "instance_assets" / "frozen_assets" / "rope_volume.npz"
            if not vol.is_file():
                raise ValueError(f"no frozen rope volume next to {base['instance']}; pass vertices_npy")
            V, _ = rope_birth_from_centerline(vol, np.asarray(b["centerline"], float))
        man = write_rope_birth(work / f"birth_{abs(hash(json.dumps(birth, sort_keys=True, default=str))) % 10**10}",
                               {"rope": V}, note="scene spec birth")
        return str(man)
    if "cloth" in birth:
        P = np.load(birth["cloth"]["particles_npy"])
        return str(cloth_birth_json(work / "cloth_birth.json", birth["cloth"].get("name", "blanket"), P,
                                    base["task_parameters"]))
    raise ValueError("birth must name rope or cloth")


def knot_tail_for_length(variation_id: str, target_length_m: float, task: str = "knot_untying") -> tuple:
    """The knot layout's `tail_length_m` (a free task parameter) that makes the variation's rope `target_length_m`
    long, so a redesign rope of another length gets a FEM volume of its own length. -> (tail_length_m, length_m)."""
    from topogym.assets.geometry.knot_untying import KnotLayout
    from topogym.config.registry import task_definition
    cls = task_definition(task, variation_id)
    geom = dict((cls.defaults or {}).get("geometry") or {})

    def length(tail):
        P = np.asarray(KnotLayout(**{**geom, "tail_length_m": float(tail)}).centerline(), float)
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    lo = 3 * float(geom.get("planar_scale_m", .032)) + 1e-4
    hi = 3.0
    if length(lo) > target_length_m:
        raise ValueError(f"{variation_id}: the shortest rope ({length(lo):.3f} m) exceeds {target_length_m} m")
    for _ in range(60):
        mid = .5 * (lo + hi)
        lo, hi = (mid, hi) if length(mid) < target_length_m else (lo, mid)
    return .5 * (lo + hi), length(.5 * (lo + hi))


def scene_from_knot_lane(scene_start_json, base_instance, variation_id: str) -> dict:
    """A scene spec from the knot lane's scene_start.json (its local frame: origin_world_xy + table_top_z; fixtures as
    boxes / cylinders with yaw; the settled start centre line): same fixtures, birth = the settled centre line, the
    rope length = the authored one. Base it on an instance whose knot family honours `tail_length_m` (overhand_trefoil:
    knot MEDIUM; figure_eight: hard): only those families' rope length can be set (the loop families' is fixed); the
    knot shape itself comes from the birth, the predicate is the family's own (shape-agnostic)."""
    from scipy.spatial.transform import Rotation
    d = json.loads(Path(scene_start_json).read_text())
    fr = d["frame"]
    ox, oy = fr["origin_world_xy"]; tz = fr["table_top_z"]

    def world(p):
        return [float(p[0]) + ox, float(p[1]) + oy, float(p[2]) + tz]
    ents = []
    for f in d.get("fixtures", []):
        e = dict(name=f["name"], kind=f["kind"], pos=world(f["pos"]))
        if f.get("yaw_deg"):
            q = Rotation.from_euler("z", f["yaw_deg"], degrees=True).as_quat()
            e["quat"] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        if f["kind"] == "box":
            e["size"] = [float(x) for x in f["size"]]
        elif f["kind"] == "cylinder":
            e["radius"], e["height"] = float(f["radius"]), float(f["height"])
        rgba = f.get("rgba")
        if isinstance(rgba, str):
            rgba = [float(x) for x in rgba.split()]
        if rgba:
            e["rgba"] = rgba
        ents.append(e)
    rope = d["ropes"][0]["name"]
    C = [world(p) for p in d["start_state"]["centerlines"][rope]]
    return dict(task="knot_untying", variation_id=variation_id, base_instance=str(base_instance), entities=ents,
                birth=dict(rope=dict(centerline=C)), rope_length_m=float(d["authored_report"]["lengths_m"][0]),
                source=str(scene_start_json))


def build_scene_session(spec=None, **kw):
    """A robot-level MuJoCo Session of a scene spec (dict, or a path to its JSON; see above)."""
    if spec is None:
        spec = kw
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text())
    base = instance_settings(spec["base_instance"])
    base["instance"] = spec["base_instance"]
    tp = spec.get("task_parameters", base["task_parameters"])
    frozen = spec["base_instance"]
    if spec.get("rope_length_m"):                  # a rope of another length: its own FEM volume (no frozen volume)
        tail, L = knot_tail_for_length(base["variation_id"], float(spec["rope_length_m"]), base["task"])
        tp = {**(tp or {}), "geometry": {**((tp or {}).get("geometry") or {}), "tail_length_m": tail}}
        frozen = None
        spec = {**spec, "_rope_volume": dict(tail_length_m=tail, length_m=L)}
    # per process: two jobs building the same spec at once must not share the generated volume / birth files
    work = (Path(spec.get("work_dir") or os.environ.get("TMPDIR") or tempfile.gettempdir())
            / f"scene_{spec['variation_id']}_{os.getpid()}_{int(__import__('time').time())}")
    work.mkdir(parents=True, exist_ok=True)
    scene_variation(base["task"], spec["variation_id"], base["variation_id"], spec.get("entities", ()),
                    spec.get("difficulty"))
    if frozen is None and spec.get("birth", {}).get("rope", {}).get("centerline") is not None:
        birth = _rope_birth_from_task_volume(base, spec, tp, work)
    else:
        birth = _birth_override(spec.get("birth"), base, work)
    overrides = {**base["overrides"], **({"mujoco_birth": birth} if birth else {}), **spec.get("overrides", {})}
    return make_session(base["task"], spec["variation_id"], seed=base["seed"], task_parameters=tp, overrides=overrides,
                        gripper_space=base["gripper_space"], frozen_assets_from=frozen)


def _rope_birth_from_task_volume(base, spec, task_parameters, work: Path) -> str:
    """The rope volume the task will declare for these parameters (generated here, as declare() does), swept along
    the spec's centre line -> a birth archive."""
    from topogym.assets.geometry.knot_untying import KnotLayout
    from topogym.assets.geometry.rope_volume import write_rope_volume
    from topogym.config.registry import task_definition
    cls = task_definition(base["task"], base["variation_id"])
    geom = {**dict((cls.defaults or {}).get("geometry") or {}), **dict((task_parameters or {}).get("geometry") or {})}
    layout = KnotLayout(**geom)
    vol = write_rope_volume(work / "rope_volume.npz", layout.centerline(), layout.rope_radius_m)
    V, info = rope_birth_from_centerline(vol, np.asarray(spec["birth"]["rope"]["centerline"], float))
    man = write_rope_birth(work / f"birth_len{info['material_length_m']:.4f}_{len(V)}", {"rope": V},
                           note=f"scene spec birth, stretch {info['stretch']:.5f}")
    (work / "birth_info.json").write_text(json.dumps(info))
    return str(man)


# ------------------------------------------------------------------------------------------------ CLI
def _example_knot(out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    use_lane_tmp(os.environ.get("RD_LANE", "tooling"))
    src = os.environ.get("KIT_INSTANCE")
    if not src:
        raise SystemExit("set KIT_INSTANCE to a knot_untying easy instance.json")
    base = instance_settings(src)
    V = load_rope_birth(base["mujoco_birth"])["rope"]
    shift = np.array([0.05, 0.0, 0.0])
    man = write_rope_birth(out / "birth", {"rope": V + shift}, note="knot easy birth shifted +5 cm in x")
    import time
    t0 = time.time()
    s = make_session(base["task"], base["variation_id"], seed=base["seed"], task_parameters=base["task_parameters"],
                     overrides={**base["overrides"], "mujoco_birth": str(man)}, gripper_space=base["gripper_space"])
    build = time.time() - t0
    for _ in range(10):
        s.step(s.hold_action())
    st = s.status()
    w = s.env._world
    rope = w._ropes["rope"]
    nodes = rope.node_positions(w.data)
    res = dict(source_instance=src, birth=str(man), build_s=round(build, 1), status=dict(success=st["success"],
               state_sha=st["state_sha"], episode_end=st["episode_end"]), scene_dir=str(w.scene_dir),
               rope_centroid=nodes.mean(axis=0).round(4).tolist())
    (out / "example_knot.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    # previews of the new scene's start state
    try:
        from previews.render import setup_cpu_gl
        setup_cpu_gl()
        from previews.render import highlight, render_state, save_views
        import mujoco
        m, d = w.model, mujoco.MjData(w.model)
        d.qpos[:] = w.data.qpos
        highlight(m, w.manifest)
        save_views(render_state(m, d, lookat=tuple(nodes.mean(axis=0))), out, "example_knot", title="session_kit example")
    except Exception as exc:  # noqa: BLE001
        print("preview skipped:", repr(exc))
    s.close()


def _scene_check(spec_path, out_dir, rows=10):
    """Build a scene spec's Session, step `rows` hold rows, write <out>/scene_check.json and previews."""
    import time
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    use_lane_tmp(os.environ.get("RD_LANE", "tooling"))
    t0 = time.time()
    s = build_scene_session(spec_path)
    build = time.time() - t0
    for _ in range(rows):
        s.step(s.hold_action())
    st = s.status()
    w = s.env._world
    m = dict(s.env._task.result.metrics)
    res = dict(spec=str(spec_path), build_s=round(build, 1), success=st["success"], episode_end=st["episode_end"],
               state_sha=st["state_sha"], entities=sorted(w.manifest["entities"]),
               metrics={k: v for k, v in m.items() if isinstance(v, (bool, int, float, str))})
    (out / "scene_check.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str)[:2000])
    try:
        from previews.render import setup_cpu_gl
        setup_cpu_gl()
        import mujoco
        from previews.render import highlight, object_lookat, render_state, save_views
        mdl, d = w.model, mujoco.MjData(w.model)
        d.qpos[:] = w.data.qpos
        mujoco.mj_forward(mdl, d)
        highlight(mdl, w.manifest)
        save_views(render_state(mdl, d, lookat=object_lookat(mdl, d, w.manifest)), out, "scene_check",
                   title=f"scene spec {Path(spec_path).name}")
    except Exception as exc:  # noqa: BLE001
        print("preview skipped:", repr(exc))
    s.close()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("example-knot"); e.add_argument("out")
    d = sub.add_parser("describe"); d.add_argument("instance")
    sc = sub.add_parser("scene"); sc.add_argument("spec"); sc.add_argument("out")
    sc.add_argument("--rows", type=int, default=10)
    a = ap.parse_args(argv)
    if a.cmd == "example-knot":
        _example_knot(a.out)
    elif a.cmd == "scene":
        _scene_check(a.spec, a.out, a.rows)
    else:
        print(json.dumps(instance_settings(a.instance), indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
