#!/usr/bin/env python3
"""Appearance seeds of a frozen MuJoCo instance, derived from its seed-0 instance without making the scene again.

A seed changes the looks only (the user, 2026-10-20): task parameters topogym.tasks.scene_variation.appearance(<task
class>, S) over the base's own, task and scene seed unchanged, physics identical. Seed S's instance is seed 0's with
  - simulator.task_parameters merged with appearance(cls, S);
  - instance_assets/ copied byte for byte: the frozen MuJoCo scene, the birth and settled-state archives, the frozen
    generated assets (rope volumes) and the render host stay seed 0's files;
  - the fields a build computes, recomputed by building that frozen scene under those parameters (rcb_topogym.Session,
    the frozen generated assets installed as for a bound instance): state_sha, asset_identity (the generated render-side
    assets), effective_task_parameters, text, public_goal, success_rule, scene_variation, evaluable_for_task_success;
  - derived_from: seed 0's state_sha and canonical sha256, S, the appearance parameters and the source that derived it.
topogym_source stays seed 0's: the binding's creation_source names the source that made the frozen scene and births.
The derivation stops unless the physical fields (objects, soft_state, reset_joint_targets, action layout, gripper space,
control rate, resolved variation) equal seed 0's, the effective task parameters differ from seed 0's only on
appearance(cls, S)'s keys, and the birth's tick record (what state_sha hashes) differs from seed 0's, built in the same
process from the seed-0 instance (strict: its state and assets reproduce), in task.instance_id only. Seeds are written
to <out>/seeds/<S>/ (instance.json + instance_assets/), the layout release_rebuild.py, stage_instance.py --seed-file and
assets.py read; one line `DERIVED {json}` per seed on stdout.

usage, inside the release's image (as make-instance: the image's source, the shared tooling at --tooling):
  python derive_seed_instance.py <seed-0 instance dir> <out dir> <S> [<S> ...] [--tooling /tooling] [--check]
    --check  load each written seed again in a fresh process (Session strict: the state and assets reproduce)
on the host, every base of a directory of instances (docker, one container per base, standard library only):
  python3 derive_seed_instance.py --image <image> --tooling <shared dir> --instances <dir> --out <dir> --seeds 1,2 \\
      [--only <base,...>] [--gpu-profiles blanket_vbd_gpu] [--gpu 0] [--cuda-compat <dir>] [--jobs 1] [--check]
    writes <out>/<base>/seeds/<S>/ and <out>/logs/derive_<base>.log; a GPU-profile base runs on --gpu (--cuda-compat:
    a directory of CUDA forward-compatibility libraries mounted at /compat, for a host whose driver is older than the
    image's CUDA).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

MUJOCO_ENV = (("NPY_DISABLE_CPU_FEATURES", "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR"),
              ("OPENBLAS_CORETYPE", "Haswell"))   # as make-instance in release_rebuild.py (mk_tasks.py MUJOCO_ENV)
# the fields of the physical state: a derived seed's must equal seed 0's
PHYSICAL = ("objects", "soft_state", "reset_joint_targets", "action_dim", "action_layout", "gripper_space", "control_hz",
            "resolved_variation_id")
# the fields a build computes that may follow the looks: recomputed for the seed
RECOMPUTED = ("state_sha", "asset_identity", "effective_task_parameters", "text", "public_goal", "success_rule",
              "scene_variation", "evaluable_for_task_success")


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def files_of(d: pathlib.Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*"))
            if p.is_file()}


def birth_record(rt, s) -> dict:
    """The tick record Session.state_sha hashes (episode identities out, contacts in a canonical order)."""
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


def sha16(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:16]


def leaves(x, pre: str = "") -> dict:
    """{dotted path: value} of a JSON tree."""
    if isinstance(x, (dict, list)) and x:
        out = {}
        for k, v in (x.items() if isinstance(x, dict) else enumerate(x)):
            out.update(leaves(v, f"{pre}{k}."))
        return out
    return {pre[:-1]: x}


def build(rt, path: pathlib.Path):
    """Session(strict) of a frozen instance, its archives and frozen generated assets installed as Session does for a
    bound instance (rope volumes: rcb_topogym._install_frozen_assets, by seed 0's asset_identity)."""
    inst = json.loads(path.read_text())
    rt._accept_realized_identities()
    rt._install_frozen_assets(path.parent, inst["asset_identity"] if "asset_identity" in inst
                              else inst["derived_from"]["frozen_identity"])
    return rt.Session(inst["task"], instance=str(path), strict=True)


def derive_one(rt, src: pathlib.Path, out: pathlib.Path, seed: int, rec0: dict) -> dict:
    """Seed `seed` of the seed-0 instance in `src` (its birth record rec0), written to out/seeds/<seed>/; the
    derivation's record."""
    from topogym.config.registry import task_definition
    from topogym.tasks.base import merge_parameters
    from topogym.tasks.goal_description import TASK_IDS, describe_goal
    from topogym.tasks.scene_variation import appearance
    from topogym.utils.io import to_jsonable
    t0 = time.time()
    inst0 = json.loads((src / "instance.json").read_text())
    if inst0["simulator"].get("physics_engine") != "mujoco" or inst0["variation_id"] == "seeded":
        raise SystemExit(f"{src}: a frozen MuJoCo instance of a fixed scene is needed")
    looks = appearance(task_definition(inst0["topogym_task"], inst0["variation_id"]), seed)
    tp0 = inst0["simulator"]["task_parameters"]
    final, tmp = out / "seeds" / str(seed), out / "seeds" / f".{seed}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        shutil.copytree(src / "instance_assets", tmp / "instance_assets")
        new = copy.deepcopy(inst0)
        new["simulator"]["task_parameters"] = merge_parameters(tp0, looks)
        # the build: seed 0's frozen instance with the seed's parameters and no recorded state or assets to compare
        (tmp / "instance.json").write_text(json.dumps(dict(
            {k: v for k, v in new.items() if k not in ("state_sha", "asset_identity")},
            derived_from=dict(frozen_identity=inst0.get("asset_identity") or {}))))
        s = build(rt, tmp / "instance.json")
        try:
            task = s.env._task
            eff = to_jsonable(task.config)
            for k in ("initial_state_path", "fold_target_path"):    # instance-relative, as make-instance records them
                if k in eff:
                    eff[k] = inst0["effective_task_parameters"][k]
            got = json.loads(json.dumps(to_jsonable(dict(
                state_sha=s.instance_sha, asset_identity=s.asset_identity, effective_task_parameters=eff, text=s.spec["text"],
                public_goal=describe_goal(task) if task.task_id in TASK_IDS else None, success_rule=rt.success_rule(s),
                scene_variation=s.env._scene_variation, evaluable_for_task_success=rt.success_enabled(task),
                objects=s.objects(), soft_state=s._soft_state(),
                reset_joint_targets=[round(float(x), 6) for x in s.joint_positions()], action_dim=s.action_dim,
                action_layout=s.action_layout, gripper_space=s.gripper_space, control_hz=s.control_hz,
                resolved_variation_id=s.env._variation["variation_id"]))))
            rec = birth_record(rt, s)
            if sha16(rec) != s.instance_sha:
                raise SystemExit(f"{src} seed {seed}: the birth record is not the one Session.state_sha hashes")
        finally:
            s.close()
        bad = [k for k in PHYSICAL if got[k] != inst0.get(k)]
        if bad:
            raise SystemExit(f"{src} seed {seed}: physical fields differ from seed 0: {bad}")
        looked = sorted(k for k in set(eff) | set(inst0["effective_task_parameters"])
                        if eff.get(k) != inst0["effective_task_parameters"].get(k))
        if not set(looked) <= set(looks):
            raise SystemExit(f"{src} seed {seed}: effective task parameters differ from seed 0 beyond the looks: "
                             f"{sorted(set(looked) - set(looks))}")
        l0, l1 = leaves(rec0), leaves(rec)
        differ = sorted(k for k in set(l0) | set(l1) if k not in l0 or k not in l1 or l0[k] != l1[k])
        if not set(differ) <= {"task.instance_id"}:
            raise SystemExit(f"{src} seed {seed}: the birth differs from seed 0's beyond task.instance_id: {differ[:20]}")
        if files_of(tmp / "instance_assets") != files_of(src / "instance_assets"):
            raise SystemExit(f"{src} seed {seed}: instance_assets/ is not seed 0's byte for byte")
        new.update({k: got[k] for k in RECOMPUTED})
        new["derived_from"] = dict(state_sha=inst0["state_sha"], instance_sha256=canonical(inst0), appearance_seed=seed,
                                   appearance=looks, source=rt._source(), tool="scripts/robotangle/derive_seed_instance.py")
        new = json.loads(json.dumps(to_jsonable(new)))
        (tmp / "instance.json").write_text(json.dumps(new, indent=1))
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)     # nothing half-written under seeds/
        raise
    a0, a1 = inst0.get("asset_identity") or {}, got["asset_identity"]
    return dict(seed=seed, appearance=looks, state_sha=got["state_sha"], seed0_state_sha=inst0["state_sha"],
                birth_leaves=len(l1), birth_leaves_differing=differ,
                world_sha=sha16({k: v for k, v in rec.items() if k != "task"}),
                seed0_world_sha=sha16({k: v for k, v in rec0.items() if k != "task"}), instance_sha256=canonical(new),
                fields_changed=sorted(k for k in RECOMPUTED if got[k] != inst0.get(k)),
                assets_changed=sorted(n for n in set(a0) | set(a1) if a0.get(n) != a1.get(n)),
                instance_assets_files=len(files_of(final / "instance_assets")), seconds=round(time.time() - t0, 1))


CHECK = """import json, pathlib, sys
sys.path[:0] = [sys.argv[1], sys.argv[3]]
import rcb_topogym as rt
import derive_seed_instance as d
s = d.build(rt, pathlib.Path(sys.argv[2]))
print("CHECK " + json.dumps(dict(instance_exact=s.instance_exact, assets_exact=s.assets_exact, state_sha=s.instance_sha,
                                 runtime_binding=s.runtime_binding is not None)), flush=True)
s.close()
"""


def in_image(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--tooling", default="/tooling")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if any(s < 1 for s in a.seeds):
        ap.error("seeds >= 1 (seed 0 is the instance itself)")
    sys.path.insert(0, a.tooling)
    import rcb_topogym as rt
    s0 = build(rt, a.src.resolve() / "instance.json")      # strict: seed 0's state and assets reproduce in this image
    rec0 = birth_record(rt, s0)
    s0.close()
    for seed in a.seeds:
        rec = derive_one(rt, a.src.resolve(), a.out.resolve(), seed, rec0)
        print("DERIVED " + json.dumps(rec), flush=True)
    for seed in a.seeds if a.check else ():   # a fresh process per seed: the written instance reproduces
        path = a.out.resolve() / "seeds" / str(seed) / "instance.json"
        r = subprocess.run([sys.executable, "-c", CHECK, a.tooling, str(path), str(pathlib.Path(__file__).resolve().parent)],
                           capture_output=True, text=True, cwd="/tmp")
        line = next((x for x in r.stdout.splitlines() if x.startswith("CHECK ")), None)
        want = json.loads(path.read_text())["state_sha"]
        if r.returncode or line is None or json.loads(line[6:])["state_sha"] != want:
            sys.stderr.write(r.stdout[-2000:] + r.stderr[-4000:])
            raise SystemExit(f"{path}: does not load again as written (Session strict)")
        print(f"CHECK seed {seed} " + line[6:], flush=True)
    return 0


def docker_derive(image: str, tooling: pathlib.Path, src: pathlib.Path, out: pathlib.Path, seeds: list[int],
                  log: pathlib.Path, *, gpu: str | None = None, cuda_compat: pathlib.Path | None = None,
                  name: str | None = None, check: bool = False) -> float:
    """One base's seeds in `image` (src: its seed-0 instance dir, mounted read-only; out/seeds/<S>/ written), as
    release_rebuild.py runs make-instance; the wall seconds. Raises SystemExit (see `log`) when the derivation stops."""
    here = pathlib.Path(__file__).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = [*(f"{k}={v}" for k, v in MUJOCO_ENV), "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
           "RCB_TORCH_THREADS=1", "HOME=/tmp", "XDG_CACHE_HOME=/tmp/cache"]
    dev = ["-e", "CUDA_VISIBLE_DEVICES="]
    if gpu is not None:
        dev = ["--gpus", f"device={gpu}", "-e", "NVIDIA_DRIVER_CAPABILITIES=all"]
        if cuda_compat is not None:
            dev += ["-v", f"{cuda_compat.resolve()}:/compat:ro", "-e", "LD_LIBRARY_PATH=/compat"]
    cmd = ["docker", "run", "--rm", *(["--name", name] if name else []), "--network", "none",
           "--user", f"{os.getuid()}:{os.getgid()}", *dev, *(x for e in env for x in ("-e", e)),
           "-v", f"{tooling.resolve()}:/tooling:ro", "-v", f"{here.parent}:/derive:ro",
           "-v", f"{src.resolve()}:/src:ro", "-v", f"{out.resolve()}:/out", "-w", "/tmp", image,
           "python", f"/derive/{here.name}", "/src", "/out", *map(str, seeds), *(["--check"] if check else [])]
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"derive {src.name} seeds {seeds} failed ({r.returncode}); see {log}")
    return time.time() - t0


def on_host(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--tooling", type=pathlib.Path, required=True)
    ap.add_argument("--instances", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--only")
    ap.add_argument("--gpu-profiles", default="blanket_vbd_gpu")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--cuda-compat", type=pathlib.Path)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    seeds = sorted({int(x) for x in a.seeds.split(",")})
    bases = sorted(p.parent.name for p in a.instances.glob("*/instance.json"))
    if a.only:
        missing = set(a.only.split(",")) - set(bases)
        if missing:
            ap.error(f"--only: no instance.json for {sorted(missing)} in {a.instances}")
        bases = [b for b in bases if b in a.only.split(",")]
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    gpu_profiles = set(a.gpu_profiles.split(","))

    def one(base: str) -> tuple[str, float | str]:
        prof = json.loads((a.instances / base / "instance.json").read_text())["simulator"].get("mujoco_profile") or ""
        try:
            return base, round(docker_derive(a.image, a.tooling, a.instances / base, a.out / base, seeds,
                                             a.out / "logs" / f"derive_{base}.log",
                                             gpu=a.gpu if prof in gpu_profiles else None, cuda_compat=a.cuda_compat,
                                             name=f"derive-seed-{base}", check=a.check), 1)
        except SystemExit as e:
            return base, f"FAILED: {e}"
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        walls = dict(pool.map(one, bases))
    (a.out / "derive_walls.json").write_text(json.dumps(walls, indent=1) + "\n")
    failed = {b: w for b, w in walls.items() if isinstance(w, str)}
    print(json.dumps(dict(bases=len(bases), seeds=seeds, failed=failed,
                          seconds=sum(w for w in walls.values() if not isinstance(w, str))), indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(on_host(sys.argv[1:]) if "--image" in sys.argv else in_image(sys.argv[1:]))
