#!/usr/bin/env python3
"""State identity by part (RoboARC 1.2.0 appearance seeds, the per-seed evidence; docs/benchmarks/roboarc_seeds_release.md):
the tick record Session.state_sha hashes, split into its
top-level parts (world, robot, task, ...), each hashed the same way (sha256 of canonical JSON, 16 hex), so that a seed's
birth or END state can be compared with seed 0's: everything but `task` (instance_id, the task RNG) must be equal.

In the release image (PYTHONPATH = a task's tests/ dir, which holds the pinned tooling, and /opt/topogym):
  python state_parts.py birth  --task <family> --instance <instance.json>                    # one JSON line
  python state_parts.py replay --task <family> --instance <instance.json> --traj <npz> [--fresh N]
        the trajectory replayed strictly, as the graders replay it (Session.replay(grading=True)) in N fresh processes
        (default 2, as grade.py --repeat 2): per replay the end state's part hashes, state_sha, success; `deterministic`
        = all replays end on one state
On the host (stdlib):
  python3 state_parts.py compare <seed-0 json> <seed-S json>     # exit 1 unless every part but `task` is equal
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys

IGNORED = ("task",)        # the parts a seed may change (instance_id, its RNG); everything else is physics
# the built MuJoCo model's physics (no rgba, materials, textures, lights, visual options): a seed must build the same
MODEL_PHYSICS = ("qpos0", "body_parentid", "body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat",
                 "body_gravcomp", "jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited", "jnt_stiffness",
                 "jnt_margin", "jnt_solref", "jnt_solimp", "dof_damping", "dof_armature", "dof_frictionloss", "dof_solref",
                 "dof_solimp", "geom_type", "geom_bodyid", "geom_dataid", "geom_size", "geom_pos", "geom_quat", "geom_friction",
                 "geom_contype", "geom_conaffinity", "geom_condim", "geom_solref", "geom_solimp", "geom_solmix", "geom_margin",
                 "geom_gap", "geom_priority", "mesh_vert", "mesh_face", "mesh_vertadr", "mesh_faceadr", "pair_geom1",
                 "pair_geom2", "pair_friction", "pair_solref", "pair_solimp", "exclude_signature", "eq_type", "eq_obj1id",
                 "eq_obj2id", "eq_data", "eq_active0", "eq_solref", "eq_solimp", "tendon_stiffness", "tendon_damping",
                 "tendon_range", "actuator_trntype", "actuator_trnid", "actuator_gainprm", "actuator_biasprm",
                 "actuator_ctrlrange", "actuator_forcerange", "actuator_gear", "flex_vert", "flex_elem", "flex_edge",
                 "flex_young", "flex_poisson", "flex_damping", "flex_friction", "flex_contype", "flex_conaffinity")
OPT_PHYSICS = ("timestep", "gravity", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance", "cone", "jacobian",
               "solver", "iterations", "ls_iterations", "noslip_iterations", "integrator", "disableflags", "enableflags",
               "o_margin", "o_solref", "o_solimp", "o_friction", "density", "viscosity", "wind", "magnetic")


def model_physics(s) -> str | None:
    """sha16 over the built MjModel's physics arrays (MODEL_PHYSICS, OPT_PHYSICS) that exist in this MuJoCo build."""
    import numpy as np
    m = getattr(getattr(s.env, "_world", None), "model", None)
    if m is None:
        return None
    h = hashlib.sha256()
    for src, names in ((m, MODEL_PHYSICS), (m.opt, OPT_PHYSICS)):
        for n in names:
            if hasattr(src, n):
                a = np.ascontiguousarray(np.asarray(getattr(src, n)))
                h.update(n.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()[:16]


def sha16(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:16]


def record(rt, s) -> dict:
    """Session.state_sha's record (rcb_topogym.Session.state_sha, the same normalization)."""
    rec = rt._plain(s.env.tick_record())
    for k in ("episode_id", "branch_id", "state_id", "parent_state_id"):
        rec["state"].pop(k, None)
    contacts = rec.get("world", {}).get("contacts", {})
    if contacts and isinstance(contacts.get("link_a"), list):
        n = len(contacts["link_a"])
        if all(isinstance(v, list) and len(v) == n for v in contacts.values()):
            rows = [{k: v[i] for k, v in contacts.items()} for i in range(n)]
            rec["world"]["contacts"] = sorted(rows, key=lambda r: json.dumps(r, sort_keys=True))
    return rec


def parts(rt, s) -> dict:
    rec = record(rt, s)
    out = dict(state_sha=s.state_sha(), parts={k: sha16(v) for k, v in sorted(rec.items())},
               physics_sha=sha16({k: v for k, v in rec.items() if k not in IGNORED}),
               # the task part without its instance id (an appearance seed's own): the task RNG, result and metrics
               task_but_instance_id=sha16({k: v for k, v in rec.get("task", {}).items() if k != "instance_id"}))
    assert sha16(rec) == out["state_sha"], "the record is not the one Session.state_sha hashes"
    return out


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def session(rt):
    """Session(strict) of the instance; its frozen generated assets (a rope's volume) installed as Session installs them
    for a bound instance, so an instance without a runtime_binding.json beside it builds the same files."""
    path = os.path.abspath(arg("--instance"))
    rt._install_frozen_assets(os.path.dirname(path), json.load(open(path)).get("asset_identity") or {})
    return rt.Session(arg("--task"), instance=path, strict=True)


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "compare":
        a, b = (json.load(open(p)) for p in sys.argv[2:4])
        pa = a.get("parts") or a["replays"][0]["parts"]
        pb = b.get("parts") or b["replays"][0]["parts"]
        differ = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
        ok = set(differ) <= set(IGNORED) and set(pa) == set(pb)
        print(json.dumps(dict(equal_but_task=ok, parts_differing=differ)))
        return 0 if ok else 1
    import rcb_topogym as rt
    if cmd == "birth":
        s = session(rt)
        print("PARTS " + json.dumps(dict(kind="birth", instance_exact=bool(s.instance_exact),
                                         instance_matches=bool(s.instance_matches),
                                         assets_exact=bool(getattr(s, "assets_exact", False)),
                                         runtime_binding=getattr(s, "runtime_binding", None) is not None,
                                         model_physics_sha=model_physics(s), **parts(rt, s))), flush=True)
        s.close()
        return 0
    if cmd == "_one":                       # one replay in this process
        s = session(rt)
        acts, ticks = rt.load_trajectory(arg("--traj"), arg("--task"), s.action_dim, trusted=True, layout=s.action_layout)
        st = s.replay(acts, physics_steps=ticks, grading=True)
        print("ONE " + json.dumps(dict(success=bool(st["success"]), rows=int(len(acts)), birth=s.instance_sha,
                                       instance_exact=bool(s.instance_exact), **parts(rt, s))), flush=True)
        s.close()
        return 0
    if cmd == "replay":
        outs = []
        for _ in range(int(arg("--fresh", "2"))):
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "_one", *sys.argv[2:]], capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("ONE ")]
            if p.returncode or not line:
                print("ERROR " + json.dumps(dict(rc=p.returncode, tail=(p.stdout + p.stderr)[-1500:])), flush=True)
                return 2
            outs.append(json.loads(line[-1][4:]))
        print("PARTS " + json.dumps(dict(kind="end", replays=outs, all_success=all(o["success"] for o in outs),
                                         deterministic=len({o["state_sha"] for o in outs}) == 1)), flush=True)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
