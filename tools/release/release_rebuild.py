#!/usr/bin/env python3
"""The RoboARC release rebuild in one command (docs/history/MUJOCO_MIGRATION.md, Release): on a docker host with harbor,
from a commit of this repository, without a GPU.

usage: tools/release/release_rebuild.sh <head> --version <v> --onto <commit> [options]
       tools/release/release_rebuild.sh --manifest <freeze.json> [options]
Run it from a clone of this repository (the worktrees are made from it). Both <head> and --onto are commits of this
repository: <head> gives the simulator source (source/), the release inputs (tools/release/) and the pinned tooling
(scripts/roboarc/shared/); --onto is the tree the release files and task dirs are written to (often the same commit).
  <head>          the commit to rebuild the release on; tools/release/mujoco_release.json there names the MuJoCo
                  bases and how their instances are made
  --manifest      the post-freeze inputs in one file (records/freeze_manifests/freeze_manifest.example.json): head (the
                  frozen commit), version, onto, families, hf_revision, gpu, lanes, jobs, and per base (new name):
                  reference (privileged.npz; relative paths are relative to the manifest), its sha256, expect (end
                  state prefix), instance (a made instance directory to use instead of making it), step_s (the wall
                  time per control step the texts state, measured on MuJoCo). Options on the command line win.
  --bench         the benchmark the MuJoCo tasks belong to (default roboarc; BENCH in scripts/<bench>/mk_tasks.py): its
                  files are scripts/<bench>/ (release.json, assets.json, packages/, oracles/, shared/),
                  images/<bench>/VERSION, the image ghcr.io/mll-lab-nu/<bench>:<version> and the task dirs
                  tasks/<bench>-<base>-<mode>. A new base's texts start from its template's final package in the
                  render tree (scripts/<bench>/packages/<template>-<mode>, the base's name replaced)
  --onto          required: render the task dirs on this commit (one whose scripts/<bench>/mk_tasks.py renders MuJoCo
                  bases): the release files are written there, with <head>'s tooling
                  (shared/rcb_topogym.py, render_bridge.py, nyx_direct.py) laid over it; then the repository checks
                  of the changed tasks and `harbor add --scan` (tasks/dataset.toml, where the tree still has one)
  --instances     made instance directories, <dir>/<new base>/ (instance.json + instance_assets/): used instead of
                  making them (a family's own make, a dry run); the report marks them, and an instance made on another
                  source than the image's is allowed only here
  --gpu           the GPU (device index) for the make-instance runs and the Harbor trials of a base on a GPU profile
                  (mujoco_release.json gpu_profiles, e.g. blanket_vbd_gpu; default 0)
  --version       the image version: images/<bench>/VERSION, the image label (RELEASE_VERSION, as images/<bench>/build.sh
                  passes it) and the tags ghcr.io/mll-lab-nu/<bench>:<v> and <bench>:<v> (local only; refused if
                  the tag exists, unless --replace-tag)
  --tag           a local tag for the release image instead of those two (the image without instances: <tag>-noinst),
                  e.g. to re-make a released version next to its image without touching its tag
  --work          the work directory (default ./release-rebuild-<head9>): repo/ (a worktree of <head> where every
                  generated file is written and committed on the local branch release-rebuild/<head9>), instances/ (the
                  instances made, in the HF dataset's layout), cache/ (CoACD hulls), harbor/ (the Harbor run), logs/,
                  REPORT.md and report.json
  --families      only these families of mujoco_release.json (comma-separated)
  --only          only these new bases (comma-separated; a dry run of single tiers; with --harbor-only, Harbor on
                  only these bases of the previous run)
  --references    reference trajectories, <dir>/<base>/privileged.npz (the new or the released base name); default
                  oracles/<new base>/privileged.npz in the tree
  --expect        a JSON file {base: end state (prefix)}: the report compares the oracles' end states with it
  --hf-revision   the dataset commit that holds the instances of a previous run in --work (the upload is a separate,
                  gated step: REPORT.md prints it); those instances are used again, not made again. Without it,
                  assets.json keeps its revision and its new pins name files that are not uploaded
  --lanes, --jobs concurrent Harbor trials (default 3) and make-instance runs (default 4)
  --no-harbor     stop after the checks
A manifest base with "reference": null has no reference solution (user 2026-10-13): no oracles/ copy, release.json
evidence {} and oracle "none" (its texts say so), no birth comparison, and Harbor runs nop only for it (harbor_nop/).
Steps:
  1. a worktree of <head>; the image without MuJoCo instances (source/images/roboarc/Dockerfile built from source/, as
     images/roboarc/build.sh does, with an empty `instances` context): <bench>:<v>-noinst
  2. every MuJoCo base's instance made inside it (rcb_topogym.py make-instance, the MuJoCo tasks' SIMD masking, no
     network), then the release image with them: ghcr.io/mll-lab-nu/<bench>:<v>; the runtime of both must be
     the same (installed_runtime)
  3. the release written: VERSION; packages/<new>-* (a base the render tree already has packages for keeps those
     texts, only its instance lock is new (or kept: below); a new base without packages gets its template's final
     packages (mujoco_release.json new_bases "template": a base of the render tree, or its Genesis-era name, which
     scripts/<bench>/roboarc_names.json maps), the name replaced: edit its texts in packages/ and run again);
     oracles/<new>/ (the reference; standard.npz is the same file: a MuJoCo standard oracle streams the reference's
     rows and mk_tasks.py reads only its actions; unless the tree's oracles/<new>/standard.npz is a reference re-timed
     to the standard service's joint-speed rule, meta "speed_rule": then that file, standard_reference()); release.json (runtime, tooling, the MuJoCo base records with their
     evidence: this run); assets.json (the made instances' pins); the release image's archive (images/archive.sh,
     <work>/image/) and its manifest digest in images/pins.json, which the tasks name; every task dir rendered
     (mk_tasks.py).
     A base that comes out unchanged (the same frozen instance, references, runtime and tooling, so the same record
     but for its provenance words) keeps its lock and its release.json record; when every base does, release.json keeps
     its release commit too: re-making a release with its own instances and references changes no task file
  4. mk_tasks.py --check (every base) and --assets <work> --check (the MuJoCo bases' instance bytes); the render tree's
     generator is driven only through its interface (mk_tasks.PACKAGE_FORMAT "final-texts/1", MUJOCO_ENV)
  5. a local commit, then harbor_check.py --tasks: oracle and nop, both modes, rendering off, on that commit (a base
     on a GPU profile keeps its one-GPU reservation)
Facts the texts state are read in the release image, per base, from its instance and the tooling (release.json):
grasp_rules (Session.grasp_rules() of the instance's configuration), physics_dt_s (the contact-review window in
ticks), gpu_physics (a GPU profile), grasp ([metadata] grasp) and grasp_regions (the grader-details line per mode, the
head's tools/release/topogym_rules.py): the texts state the instance's grasp, not the released Genesis one; step_s comes from
the manifest.
  6. REPORT.md: PASS when every check and trial is as expected (oracle 1, nop 0, the evidence's claims hold)
Instance bytes are only reproducible from the same make: CoACD's hulls differ between hosts in their last bits and
vertex order (claws hard, host A against a cloud host: <= 5e-13 m; the oracle's end state is the same), and scene.json records
whether each hull came from the hull cache. So the uploaded instances are the ones a run made, and the run with
--hf-revision uses them.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import platform
import re
import shutil
import socket
import subprocess
import sys
import zipfile

BENCH = "roboarc"          # the default --bench
TOOLING = ("rcb_topogym.py", "render_bridge.py", "nyx_direct.py")   # <head>'s shared/ tooling, laid over the --onto tree
MAKE_ONE = '''import json, shutil, sys
from pathlib import Path
sys.path.insert(0, "/tooling")
import rcb_topogym as rt
job = json.load(open(sys.argv[1]))
released = Path("/opt/robotangle/instances") / job["released"]
inst = json.loads((released / "instance.json").read_text())
out = Path(job["out"]).parent
host = out / "instance_assets" / "render_host"
if job["render_host"]:      # the released instance: the render host, and a birth source inside the new instance
    shutil.copytree(released / "instance_assets", host / "instance_assets")
    shutil.copy2(released / "instance.json", host / "instance.json")
    shutil.copy2(job["released_binding"], host / "runtime_binding.json")
if job["frozen_assets"]:    # the released generated assets (a rope's tet volume): declare() uses them, not this CPU's
    shutil.copytree(released / "instance_assets" / "frozen_assets", out / "instance_assets" / "frozen_assets")
    rt._install_frozen_assets(out, inst.get("asset_identity") or {})
over = dict(job["simulator_overrides"])
over.update({k: inst["simulator"][k] for k in job["released_simulator_keys"]})
weld = job["grasp_weld_links"]
if weld:                    # the assisted grasp: contact_weld_v1 on every link of the entity, the released tolerances
    import numpy as np
    with np.load(released / "instance_assets" / "frozen_assets" / f"{weld['entity']}_volume.npz") as vol:
        st = vol["rest"][vol["surface_ring_indices"], 0].mean(axis=1)
    L = float(st[-1] - st[0])
    n = max(2, int(round(L / weld["link_m"])))
    g = inst["simulator"]["grasp_assist"]
    keep = ("stable_time_s", "max_tcp_distance_m", "max_penetration_m", "max_translation_error_m", "max_rotation_error_rad",
            "max_finger_opening_fraction", "intent_deadband", "initial_intent")
    over["grasp_assist"] = dict(model="contact_weld_v1", **{k: g[k] for k in keep if k in g}, targets=[
        dict(entity=weld["entity"], link=f"{k:03d}", center_m=[L / n / 2, 0., 0.], radius_m=weld["radius_m"])
        for k in range(n)])
def resolve(v):
    if v == "@released_instance":
        return str(released / "instance.json")
    if isinstance(v, str) and v.startswith("@render_host/"):
        return str(host / v[len("@render_host/"):])
    return v
over = {k: resolve(v) for k, v in over.items()}
argv = ["make-instance", "--task", job["task"], "--variation", job["variation"], "--scene-seed", str(job["scene_seed"]),
        "--gripper-space", job["gripper_space"], "--out", job["out"]]
if over:
    json.dump(over, open(job["overrides_file"], "w"), indent=1)
    argv += ["--simulator-overrides", job["overrides_file"]]
if job["released_task_parameters"]:
    json.dump(inst["simulator"]["task_parameters"], open(job["task_parameters_file"], "w"), indent=1)
    argv += ["--task-parameters", job["task_parameters_file"]]
print(" ".join(argv), flush=True)
sys.argv = ["rcb_topogym.py", *argv]
rt.main()                   # in this process: the frozen-asset hook above applies
if job["render_host"]:      # recorded as a port of the released instance: the birth relative to the instance, the source
    new = json.loads(Path(job["out"]).read_text())
    birth = Path(new["simulator"].get("mujoco_birth") or "")
    if birth.is_absolute() and out in birth.parents:
        new["simulator"]["mujoco_birth"] = birth.relative_to(out).as_posix()
    new["ported_from"] = dict(state_sha=inst["state_sha"], asset_identity=inst["asset_identity"],
                              genesis_version=inst.get("genesis_version"))
    Path(job["out"]).write_text(json.dumps(new, indent=1))
'''


FACTS = '''import json, sys, types
sys.path.insert(0, "/tooling")
import rcb_topogym as M
sys.path.insert(0, "/scripts")
import topogym_rules as R           # the head's grader-details text (the grasp-region line per mode)
from topogym.simulator.config import SimulatorConfig
gpu_profiles, out = json.loads(sys.argv[1]), {}


def chain_facts(inst, cfg, scene):
    """A rope / cord chain's facts for the texts (RoboTangle-New's chain_words): its links (count, length, radius) from
    the frozen scene, its joints' springs, bend stops, plasticity, speed clamp and tension guard from the profile."""
    import xml.etree.ElementTree as ET
    import numpy as np
    from topogym.world.mujoco import profiles
    prof = profiles.profile_for(inst["task"], inst.get("resolved_variation_id") or inst["variation_id"], cfg.mujoco_profile)
    root = ET.parse(scene).getroot()
    if prof.get("rope"):
        from topogym.world.mujoco.rope_chain import ROPE_DEFAULTS
        r, ent, kind = {**ROPE_DEFAULTS, **prof["rope"]}, "rope", "rope"
    elif (prof.get("speed_clamp") or {}).get("links"):
        r, ent, kind = {}, prof["speed_clamp"]["links"].split("::")[0], "mjcf"
    else:
        return None
    caps = [g for g in root.iter("geom") if g.get("type") == "capsule" and g.get("name", "").startswith(ent + "::")
            and g.get("fromto")]
    if kind == "mjcf":     # the cord's links only: the capsules of the bodies on a ball joint (not a ring's segments)
        caps = [g for bd in root.iter("body") if bd.get("name", "").startswith(ent + "::")
                and any(j.get("type") == "ball" for j in bd.findall("joint"))
                for g in bd.findall("geom") if g.get("type") == "capsule" and g.get("fromto")]
    ft = np.array([[float(v) for v in g.get("fromto").split()] for g in caps])
    link = float(np.median(np.linalg.norm(ft[:, 3:] - ft[:, :3], axis=1))) if len(caps) else None
    ks = [float(j.get("stiffness", 0.)) for bd in root.iter("body") if bd.get("name", "").startswith(ent + "::")
          for j in bd.findall("joint") if j.get("type") == "ball"]
    return dict(kind=kind, entity=ent, profile=prof.get("name"), links=len(caps),
                link_mm=None if link is None else round(1e3 * link, 2),
                radius_mm=round(1e3 * float(caps[0].get("size").split()[0]), 3) if caps else None,
                bend_stop_rad=r.get("bend_stop_rad"), stiffness_nm_rad=round(max(ks), 6) if ks else 0.,
                speed_clamp=r.get("speed_clamp") or prof.get("speed_clamp"), tension_guard=r.get("tension_guard"),
                plastic=prof.get("plastic"))
for b in sys.argv[2:]:
    inst = json.load(open(f"/work/instances/{b}/instance.json"))
    cfg = SimulatorConfig(**inst["simulator"])
    fake = types.SimpleNamespace(env=types.SimpleNamespace(config=cfg))    # grasp_rules reads env.config only
    ga = inst["simulator"].get("grasp_assist") or {}           # [metadata] grasp (topogym_mk_task.grasp_words)
    if ga.get("model", "none") == "none":
        grasp = "physical (no assisted grasping)"
    else:
        parts = ", ".join(sorted({t["entity"] for t in ga.get("targets", [])}))
        grasp = (f"assisted (his {ga['model']} pre-weld: the grippers start closed on {parts})" if ga.get("initial_intent") == "close"
                 else f"assisted (his {ga['model']} grasp regions on {parts})")
    regions = {m: R.grasp_regions(inst["task"], inst, limited=m == "standard") for m in ("privileged", "standard")}
    chain = chain_facts(inst, cfg, f"/work/instances/{b}/instance_assets/mujoco/scene.xml")
    out[b] = dict(grasp_rules=M.Session.grasp_rules(fake), physics_dt_s=float(cfg.physics_dt_s),
                  gpu_physics=(cfg.mujoco_profile or "") in gpu_profiles, mujoco_profile=cfg.mujoco_profile,
                  robot_id=cfg.robot_id, grasp=grasp, grasp_regions={m: v for m, v in regions.items() if v},
                  **(dict(chain=chain) if chain else {}))
print(json.dumps(out))
'''


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run(cmd: list[str], log: pathlib.Path | None = None, cwd: pathlib.Path | None = None, check: bool = True) -> str:
    """Run a command; its output goes to log (and is returned when there is no log)."""
    if log is None:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if check and r.returncode:
            raise SystemExit(f"failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout}{r.stderr}")
        return r.stdout
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if check and r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(cmd)}; see {log}")
    return ""


def module(path: pathlib.Path):
    sys.path[:0] = [str(path.parent), str(path.parents[1])]
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def image_id(ref: str) -> str | None:
    r = subprocess.run(["docker", "image", "inspect", ref, "--format", "{{.Id}}"], capture_output=True, text=True)
    return r.stdout.strip() or None


def files_of(d: pathlib.Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): sha(p.read_bytes()) for p in sorted(d.rglob("*")) if p.is_file()}


def npz_text(path: pathlib.Path, name: str) -> str | None:
    """A text scalar of an .npz (a reference's instance_sha.npy: the birth it was recorded on), read without numpy."""
    with zipfile.ZipFile(path) as z:
        if name not in z.namelist():
            return None
        raw = z.read(name)
    size = 4 if raw[6] >= 2 else 2                      # .npy: magic, version, header length, header, data
    return raw[8 + size + int.from_bytes(raw[8:8 + size], "little"):].decode("utf-32-le").rstrip("\0")


def canonical(inst: dict) -> str:
    return sha(json.dumps(inst, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def bases_of(spec: dict, families: list[str] | None) -> list[dict]:
    """The MuJoCo bases to rebuild: released base, new base, make-instance job."""
    fam = spec["families"]
    unknown = set(families or []) - set(fam)
    if unknown:
        raise SystemExit(f"not in mujoco_release.json: {sorted(unknown)}")
    out = []
    for f in families or list(fam):
        r = fam[f]
        for tier in r.get("tiers", []):     # a tier's new base is named by its number in the family (numbers), not its tier
            stem, fd = f"{f.replace('_', '-')}-{tier.replace('_', '-')}", f.replace("_", "-")
            num = r.get("numbers", {}).get(tier)
            out.append(dict(family=f, tier=tier, old=stem + spec["suffix"]["from"],
                            new=(f"{fd}-{num}" if num else stem) + spec["suffix"]["to"],
                            variation=r.get("variations", {}).get(tier, f"{f}_{tier}"),
                            released_simulator_keys=r.get("released_simulator_keys", []),
                            simulator_overrides=r.get("simulator_overrides", {}),
                            render_host=r.get("render_host", False), frozen_assets=r.get("frozen_assets", False),
                            released_task_parameters=r.get("released_task_parameters", False),
                            grasp_weld_links=r.get("grasp_weld_links"),
                            task=f, template=None, design=r.get("design"),
                            **{**spec["make_instance"], **r.get("make_instance", {})}))
        for new, n in r.get("new_bases", {}).items():   # a new design: no released base; its texts start from a template's
            unknown = set(n) - {"template", "variation", "tier", "task", "scene_seed", "gripper_space", "design", "note",
                                "simulator_overrides"}
            if unknown or not n.get("template") or not n.get("variation"):
                raise SystemExit(f"{f} new_bases {new}: needs template and variation (unknown keys {sorted(unknown)})")
            out.append(dict(family=f, tier=n.get("tier", n["variation"].removeprefix(f"{f}_")), old=None, new=new,
                            variation=n["variation"], released_simulator_keys=[],
                            simulator_overrides=n.get("simulator_overrides", {}),   # fixed values only (no released instance)
                            render_host=False, frozen_assets=False, released_task_parameters=False, grasp_weld_links=None,
                            task=n.get("task", f),
                            template=n["template"], design=n.get("design"),
                            **{**spec["make_instance"], **r.get("make_instance", {}),
                               **{k: n[k] for k in ("scene_seed", "gripper_space") if k in n}}))
    names = [b["new"] for b in out]
    if len(set(names)) != len(names):
        raise SystemExit(f"a base is listed twice: {sorted({n for n in names if names.count(n) > 1})}")
    return out


def reference(b: dict, refs: pathlib.Path | None, repo: pathlib.Path, named: dict | None = None) -> pathlib.Path:
    if named and b["new"] in named:        # the manifest's path for this base
        if not named[b["new"]].is_file():
            raise SystemExit(f"{b['new']}: the manifest's reference {named[b['new']]} is missing")
        return named[b["new"]]
    cands = ([refs / n / "privileged.npz" for n in (b["new"], b["old"]) if n] if refs else []) + \
            [repo / "scripts/roboarc/oracles" / b["new"] / "privileged.npz"]
    for c in cands:
        if c.is_file():
            return c
    raise SystemExit(f"{b['new']}: no reference (looked for {', '.join(map(str, cands))})")


def standard_reference(repo: pathlib.Path, base: str, privileged: bytes) -> bytes:
    """The standard oracle's rows: the reference itself, unless the tree holds a standard reference re-timed to the
    standard service's joint-speed rule (scripts/roboarc/oracles/<base>/standard.npz whose meta has "speed_rule":
    the reference's rows with every arm-joint step within JOINT_STEP_LIMIT_RAD, 2026-10-14) on the reference's birth:
    a tree file recorded on another birth (its instance_sha is not the reference's, e.g. a copy from before a
    re-freeze) is not the standard oracle of this reference. The privileged reference is never replaced by it."""
    p = repo / "scripts/roboarc/oracles" / base / "standard.npz"
    if not p.is_file() or p.read_bytes() == privileged:
        return privileged
    with zipfile.ZipFile(p) as z:
        raw = z.read("meta.npy") if "meta.npy" in z.namelist() else b""
    meta = {}
    if raw[:6] == b"\x93NUMPY" and b"'descr': '<U" in raw[:128]:     # a 0-d unicode array (np.savez of a str), parsed without numpy
        start = (10 + int.from_bytes(raw[8:10], "little")) if raw[6] == 1 else (12 + int.from_bytes(raw[8:12], "little"))
        meta = json.loads(raw[start:].decode("utf-32-le").rstrip("\x00") or "{}")
    if "speed_rule" not in meta:
        return privileged
    born, ref_born = npz_text(p, "instance_sha.npy"), npz_text(io.BytesIO(privileged), "instance_sha.npy")
    return privileged if None not in (born, ref_born) and born != ref_born else p.read_bytes()


def speed_rule_record(repo: pathlib.Path, base: str) -> dict:
    """A speed-rule standard reference's record (tools/release/standard_speed_rule.json): `words`, the sentence the
    standard evidence states (release.json `speed_rule`, which mk_tasks.py renders into the standard README's Reference
    solution), and `end_state`, the end it reaches (the evidence's, and the end the Harbor standard oracle must reach:
    not the privileged reference's)."""
    recs = json.loads((repo / "tools/release/standard_speed_rule.json").read_text())["bases"]
    if base not in recs:
        raise SystemExit(f"{base}: its standard oracle is a speed-rule reference, but tools/release/"
                         "standard_speed_rule.json has no record of it (words, end_state)")
    return recs[base]


def standard_expect(a, base: str) -> str | None:
    """The end the Harbor standard oracle must reach: a speed-rule standard reference's own (its oracles/<base>/
    standard.npz differs from privileged.npz in the release files), else None (the privileged reference's end)."""
    d = a.out / "scripts" / a.bench / "oracles" / base
    if not (d / "standard.npz").is_file() or (d / "standard.npz").read_bytes() == (d / "privileged.npz").read_bytes():
        return None
    return speed_rule_record(a.repo, base)["end_state"]


def make_instances(a, bases: list[dict], noinst: str, mujoco_env) -> dict:
    """make-instance for every base inside the image without instances, in parallel; {base: wall seconds}."""
    inst, mk = a.work / "instances", a.work / "mk"
    shutil.rmtree(inst, ignore_errors=True)
    for d in (inst, mk, a.work / "cache"):
        d.mkdir(parents=True, exist_ok=True)
    for b in bases:                    # made elsewhere (--instances / the manifest): copied as they are
        if b["new"] in a.premade:
            shutil.copytree(a.premade[b["new"]], inst / b["new"], symlinks=False)
    bases = [b for b in bases if b["new"] not in a.premade]
    (mk / "make_one.py").write_text(MAKE_ONE)
    env = [*(f"{k}={v}" for k, v in mujoco_env), "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
           "RCB_TORCH_THREADS=1", "HOME=/tmp", "XDG_CACHE_HOME=/work/cache"]
    tooling = a.repo / "scripts/roboarc/shared"

    def one(b: dict) -> tuple[str, float]:
        job = {k: b[k] for k in ("variation", "released_simulator_keys", "simulator_overrides", "scene_seed",
                                 "gripper_space", "render_host", "frozen_assets", "released_task_parameters",
                                 "grasp_weld_links")}
        job.update(task=b["task"], released=b["old"] or b["template"], out=f"/work/instances/{b['new']}/instance.json",
                   overrides_file=f"/work/mk/{b['new']}.overrides.json",
                   task_parameters_file=f"/work/mk/{b['new']}.task_parameters.json")
        if b["render_host"]:     # the released task's binding goes with the released instance into the render host
            released_binding = a.repo / "tasks" / f"robotangle-{b['old']}-privileged/environment/runtime_binding.json"
            if not released_binding.is_file():   # the Genesis RoboTangle task dirs are not carried in this repository
                raise SystemExit(f"{b['new']}: render_host needs {released_binding} (the released Genesis task's binding), "
                                 "which this repository does not carry: pass the made instance with --instances or the "
                                 "manifest's instance")
            shutil.copyfile(released_binding, mk / f"{b['new']}.released_binding.json")
            job["released_binding"] = f"/work/mk/{b['new']}.released_binding.json"
        (inst / b["new"]).mkdir()
        (mk / f"{b['new']}.json").write_text(json.dumps(job, indent=1) + "\n")
        t0 = datetime.datetime.now()
        gpu = (b["simulator_overrides"].get("mujoco_profile") or "") in a.gpu_profiles   # a GPU profile: its make needs one
        dev = ["--gpus", f"device={a.gpu}"] if gpu else ["-e", "CUDA_VISIBLE_DEVICES="]
        run(["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}", *dev,
             *(x for e in env for x in ("-e", e)), "-v", f"{tooling}:/tooling:ro", "-v", f"{a.work}:/work", noinst,
             "python", "/work/mk/make_one.py", f"/work/mk/{b['new']}.json"], log=a.work / "logs" / f"make_{b['new']}.log")
        return b["new"], (datetime.datetime.now() - t0).total_seconds()

    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        return dict(pool.map(one, bases))


def template_base(rb: pathlib.Path, template: str) -> str:
    """A new base's template as a base of the render tree: its own name, or the RoboARC name roboarc_names.json gives
    its Genesis-era name (<family>-<tier>-i00)."""
    if (rb / "packages" / f"{template}-privileged").is_dir():
        return template
    names = json.loads((rb / "roboarc_names.json").read_text())
    old = "robotangle-" + re.sub(r"-i00$", "-i01", template)
    if old not in names:
        raise SystemExit(f"template {template}: neither a base of scripts/{rb.name}/packages nor in roboarc_names.json")
    return names[old]["new"].removeprefix(rb.name + "-")


def write_release(a, bases: list[dict], head: str, runtime: dict, noinst: dict, facts: dict) -> dict:
    """VERSION, packages/, oracles/, release.json and assets.json for the made instances, in the render tree (a.out: the
    --onto worktree, else <head>'s); the old task dirs removed. Returns what it did: the bases whose packages it wrote
    from their template's ("written": edit their texts) and the bases it found unchanged ("kept", with the record fields
    only this run's provenance words would change)."""
    rb = a.out / "scripts" / a.bench                        # this benchmark's release files
    new_bench = True                                # always a benchmark of its own (the Genesis in-place release is gone)
    (a.out / "images" / a.bench / "VERSION").write_text(a.version + "\n")
    rel = json.loads((rb / "release.json").read_text())
    assets = json.loads((rb / "assets.json").read_text())
    tooling = sha((rb / "shared/rcb_topogym.py").read_bytes())
    ev_file = a.repo / "tools/release/release_evidence.json"
    shipped = json.loads(ev_file.read_text()) if ev_file.is_file() else {}
    src8, tool8 = runtime["source"]["sha256"][:8], tooling[:8]
    host = f"{a.host_label}, {cpu_model()}, no GPU"
    records, written, kept = {}, [], {}
    same_release = rel["runtime"] == {**rel["runtime"], **runtime} and rel["tooling_sha256"] == tooling

    def texts_of(src: pathlib.Path, old: str, new: str) -> dict:
        """A package's files: texts with the base's name replaced; binary files (a fold's goal-state images,
        environment/goal/*.png) byte for byte."""
        out = {}
        for p in sorted(src.rglob("*")):
            if p.is_file():
                b = p.read_bytes()
                try:
                    out[p.relative_to(src)] = (b.decode().replace(old, new), p.stat().st_mode)
                except UnicodeDecodeError:
                    out[p.relative_to(src)] = (b, p.stat().st_mode)
        return out
    # a new base's template texts: its template's final packages in the render tree, the base's name replaced
    template_texts = {}
    for b in bases:
        if b["template"] and not all((rb / "packages" / f"{b['new']}-{m}").is_dir() for m in ("privileged", "standard")):
            t = template_base(rb, b["template"])
            for m in ("privileged", "standard"):
                template_texts[b["new"], m] = texts_of(rb / "packages" / f"{t}-{m}", t, b["new"])
    # a base the benchmark already releases (its packages/<new>-* in the render tree) keeps those texts; a text change
    # of a released base is made there. Only its instance lock is written again.
    onto_texts = {b["new"] for b in bases if new_bench and all((rb / "packages" / f"{b['new']}-{m}").is_dir()
                                                               for m in ("privileged", "standard"))}
    for b in bases:
        old, new, d = b["old"], b["new"], a.work / "instances" / b["new"]
        inst = json.loads((d / "instance.json").read_text())
        if new not in onto_texts:
            written.append(new)
        for mode in ("privileged", "standard") if new not in onto_texts else ():
            dst = rb / "packages" / f"{new}-{mode}"
            if not b["template"]:         # a released base needs its packages in the render tree
                raise SystemExit(f"{new}: no packages/{new}-{mode} in the render tree (a released base keeps its texts there)")
            texts = dict(template_texts[new, mode])   # a new base: its template's final texts, under the new name
            shutil.rmtree(dst, ignore_errors=True)
            for rel_path, (s, st_mode) in texts.items():
                (dst / rel_path).parent.mkdir(parents=True, exist_ok=True)
                (dst / rel_path).write_bytes(s) if isinstance(s, bytes) else (dst / rel_path).write_text(s)
                os.chmod(dst / rel_path, st_mode & 0o777)
            if old and old != new and not new_bench:
                shutil.rmtree(rb / "packages" / f"{old}-{mode}", ignore_errors=True)
        assets_files = {p.relative_to(d).as_posix(): sha(p.read_bytes()) for p in sorted((d / "instance_assets").rglob("*"))
                        if p.is_file()}
        lock_file = rb / "packages" / f"{new}-privileged/runtime_binding.upstream.json"
        prev_lock = json.loads(lock_file.read_text()) if new in onto_texts and lock_file.is_file() else {}
        # the same frozen instance as the render tree binds (a premade instance used again): its lock, which records the
        # image it was made in, stays
        same_instance = (prev_lock.get("instance_sha256") == canonical(inst)
                         and all(assets_files.get(k) == v for k, v in prev_lock.get("asset_files", {}).items()))
        if not same_instance:
            lock = dict(schema_version=1, instance_sha256=canonical(inst), asset_files=assets_files, tooling_sha256=tooling,
                        creation_source=inst["topogym_source"], runtime=runtime, image=noinst)
            lock_file.write_text(json.dumps(lock, indent=2) + "\n")
        prev = rel["bases"].get(new) or {}
        no_ref = new in a.no_oracle
        if old and old != new and not new_bench:
            shutil.rmtree(rb / "oracles" / old, ignore_errors=True)
        if no_ref:                       # no reference solution: no oracles/, release.json evidence {} (oracle none)
            shutil.rmtree(rb / "oracles" / new, ignore_errors=True)
        else:
            ref = reference(b, a.references, a.repo, a.ref_paths)
            data = ref.read_bytes()
            per_mode = dict(privileged=data, standard=standard_reference(a.repo, new, data))
            (rb / "oracles" / new).mkdir(parents=True, exist_ok=True)
            for mode in ("privileged", "standard"):
                if (rb / "oracles" / new / f"{mode}.npz") != ref:
                    (rb / "oracles" / new / f"{mode}.npz").write_bytes(per_mode[mode])
        # The record form: the image and the date, never a host or a job name. A new or changed reference's record is
        # the Harbor trial that passed it (tools/release/release_evidence.json, its provenance not copied); without one
        # it is not_run: no score is claimed
        given = (shipped.get("bases") or {}).get(new) or {}
        evidence = {} if no_ref else {
            m: dict({k: v for k, v in given[m].items() if k != "provenance"}, trajectory_sha256=sha(per_mode[m]))
            if m in given else dict(status="not_run", trajectory_sha256=sha(per_mode[m]))
            for m in ("privileged", "standard")}
        if not no_ref and per_mode["standard"] != per_mode["privileged"]:   # a standard reference within the speed rule
            rec = speed_rule_record(a.repo, new)
            evidence["standard"].update(speed_rule=rec["words"], end_state=rec["end_state"])
        if new_bench and not no_ref:    # a mode whose reference the render tree already releases keeps its record (its
            released_ev = (rel["bases"].get(new) or {}).get("evidence") or {}   # status and where it was graded): only
            for m in ("privileged", "standard"):                                # a new or changed reference is this run's
                if (released_ev.get(m) or {}).get("trajectory_sha256") == evidence[m]["trajectory_sha256"]:
                    evidence[m] = released_ev[m]
        record = dict(
            creation_source=inst["topogym_source"]["sha256"][:8],
            evidence=evidence,
            **(dict(oracle="none") if no_ref else {}),
            # a premade instance (--instances) was not made in this image: the 0.1.0 rename's wording (audit S7)
            family=b["family"], instance_package=(
                f"MuJoCo instance made before this image was built (TopoGym source {inst['topogym_source']['sha256'][:8]}; "
                f"its bytes are bound into the release image ghcr.io/mll-lab-nu/{a.bench}:{a.version}, "
                "tools/release/release_rebuild.py --instances)" if new in a.premade else
                f"MuJoCo instance made in the release image (ghcr.io/mll-lab-nu/{a.bench}:{a.version}, "
                f"tools/release/release_rebuild.py; make-instance {b['variation']}, scene seed {b['scene_seed']})"),
            tier=b["tier"], physics_engine="mujoco", release_commit=head,
            # a released base keeps its nop record; a new one takes release_evidence.json's, else not_run
            nop=(((rel["bases"].get(new) or {}).get("nop") or {k: v for k, v in (shipped.get("bases") or {}).get(new, {})
                                                                 .get("nop", {"status": "not_run"}).items() if k != "provenance"})
                 if new_bench else   # main's record form (the 0.1.0 rename step), no host
                 dict(run=f"{a.version}, harbor run -a nop with rendering off (scripts/roboarc/harbor_check.py)",
                      date=datetime.date.today().isoformat(), host=host)),
            **(dict(template=b["template"], variation=b["variation"]) if b["template"] else
               dict(ported_from=old) if new_bench else
               dict(replaces=old if old != new else rel["bases"][new]["replaces"])),
            **(dict(design=b["design"]) if b["design"] else {}),
            **{k: facts[new][k] for k in ("grasp_rules", "physics_dt_s", "grasp", "grasp_regions", "chain")
               if facts[new].get(k)},
            **(dict(gpu_physics=True) if facts[new]["gpu_physics"] else {}),
            # the step time measured on this instance stays its step time when the manifest names none
            **(dict(step_s=a.step_s[new]) if new in a.step_s else
               dict(step_s=prev["step_s"]) if same_instance and prev.get("step_s") else {}),
            **(later_release(a, rb, rel, new, evidence, no_ref) if new_bench else {}))
        diff = sorted(k for k in set(prev) | set(record) if prev.get(k) != record.get(k))
        if same_release and same_instance and prev and set(diff) <= {"instance_package", "release_commit"}:
            record, kept[new] = prev, diff      # unchanged: its record as released (where and when it was made)
        records[old or new] = (new, record)
        for prefix in {f"instances/{n}/" for n in (old, new) if n and (n == new or not new_bench)}:
            assets["files"] = {k: v for k, v in assets["files"].items() if not k.startswith(prefix)}
        assets["files"].update({f"instances/{new}/{k}": v for k, v in files_of(d).items()})
        if old and not new_bench:
            assets["instances"].pop(old, None)
        assets["instances"][new] = {"instance_sha256": canonical(inst)}
        for t in a.out.glob(f"tasks/robotangle-{old}-*") if old and not new_bench else ():
            shutil.rmtree(t)
    # release.json: the records in place (a new base where its released one was), one runtime and tooling for all
    new_of, by_new = {o: n for o, (n, _) in records.items()}, dict(records.values())
    bases_out = {}
    for k, v in rel["bases"].items():
        bases_out[new_of.get(k, k)] = by_new.get(new_of.get(k, k), v)
    rel["bases"] = {**bases_out, **{n: r for n, r in by_new.items() if n not in bases_out}}
    rel["runtime"].update(runtime)
    rel["tooling_sha256"] = tooling
    if not (same_release and len(kept) == len(bases)):   # the commit the release is built from (unless it re-made itself)
        rel["release_commit"] = rel["template_commit"] = head
    (rb / "release.json").write_text(json.dumps(rel, indent=1, ensure_ascii=False) + "\n")
    assets["files"] = dict(sorted(assets["files"].items()))
    assets["instances"] = dict(sorted(assets["instances"].items()))
    if a.hf_revision:
        assets["revision"] = a.hf_revision
    (rb / "assets.json").write_text(json.dumps(assets, indent=1, ensure_ascii=False) + "\n")
    return dict(written=written, kept=kept)


EMPTY_FAILURES = re.compile(r"(?m)^## Task failure conditions\n\s*\n(?=## )")


def later_release(a, rb: pathlib.Path, rel: dict, new: str, evidence: dict, no_ref: bool) -> dict:
    """A base the render tree's release.json already has (a later release of it): its first release date ("released",
    the date its Change Log's "new task" entry keeps) and this release's entries ("changes", per mode) for what changed in
    its task dir: a reference where it had none, a new reference of a mode, and the empty failure heading the generator
    now drops (main's rendered instruction.md has one). The entries of earlier releases ("changes", "released") are
    kept: a base's Change Log is its whole history."""
    old = rel["bases"].get(new)
    if not old:                       # a new base: its Change Log entry is dated by this release
        return dict(released=datetime.date.today().isoformat())
    today, v = datetime.date.today().isoformat(), f"{a.bench} {a.version}"
    earlier = list(old.get("changes") or [])
    changes = []
    old_ev = old.get("evidence") or {}
    if not no_ref and (old.get("oracle") == "none" or not old_ev):
        changes.append(dict(date=today, mode="both", text=(
            f"{v}: a reference solution (`solution/`, release.json oracle `none` -> `full`), recorded on MuJoCo from the "
            "frozen instance; the frozen instance and the task text are unchanged.")))
    elif not no_ref:
        for m in ("privileged", "standard"):
            if (old_ev.get(m) or {}).get("trajectory_sha256") != evidence[m]["trajectory_sha256"]:
                what = ("replaced by one within the standard service's joint-speed rule (release.json `speed_rule`)"
                        if evidence[m].get("speed_rule") else "replaced")
                changes.append(dict(date=today, mode=m, text=(
                    f"{v}: the {m} reference (`solution/`) {what}; the frozen instance and the task text are unchanged.")))
    for m in ("privileged", "standard"):
        f = a.out / "tasks" / f"{a.bench}-{new}-{m}" / "instruction.md"
        if f.is_file() and EMPTY_FAILURES.search(f.read_text()):
            changes.append(dict(date=today, mode=m, text=(
                f"{v}: instruction.md drops the empty `## Task failure conditions` heading (no rule was left under it; "
                "instruction audit V6); the rules are unchanged.")))
    changes = earlier + [c for c in changes if c not in earlier]
    released = old.get("released") or (old.get("nop") or {}).get("date")
    return dict(**(dict(released=released) if released else {}),
                **(dict(changes=changes) if changes else {}))


def drop_missing(tasks: pathlib.Path) -> list[str]:
    """Remove the dataset.toml entries of task dirs that are gone (a replaced -i00 base); returns their names."""
    path = tasks / "dataset.toml"
    head, *blocks = path.read_text().split("\n[[tasks]]\n")
    keep, dropped = [], []
    for b in blocks:
        name = b.split('name = "', 1)[1].split('"', 1)[0].split("/", 1)[1]
        (keep if (tasks / name).is_dir() else dropped).append(b if (tasks / name).is_dir() else name)
    path.write_text("\n[[tasks]]\n".join([head, *keep]))
    return dropped


def cpu_model() -> str:
    for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown CPU"


def harbor_table(a, bases: list[dict], hc) -> tuple[list[dict], bool]:
    """Per base and mode: oracle and nop rewards, the oracle's evidence fields and end states."""
    results = {}
    for run_dir in ("harbor", "harbor_nop"):
        if (a.work / run_dir / "results.json").is_file():
            results.update({(r["task"], r["agent"]): r for r in json.loads((a.work / run_dir / "results.json").read_text())})
    rows, ok = [], True
    for b in bases:
        for mode in ("privileged", "standard"):
            task = f"{a.bench}-{b['new']}-{mode}"
            o, n = results.get((task, "oracle"), {}), results.get((task, "nop"), {})
            if b["new"] in getattr(a, "no_oracle", ()):   # no reference solution: nop 0 is the whole trial evidence
                good = bool(n.get("pass"))
                ok &= good
                rows.append(dict(base=b["new"], mode=mode, oracle="none", nop=n.get("reward"), ends=[], expected=None,
                                 evidence_holds=None, oracle_s=None, nop_s=n.get("wall_s"), exception=n.get("exception"),
                                 ok=good))
                continue
            rw = o.get("rewards") or {}
            ends = sorted({s for s in hc.end_states(a.work / "harbor/jobs" / f"{task}__oracle", o.get("trial")) if s})
            claims = rw.get("deterministic") == 1 and (rw.get("all_success") == 1 if mode == "privileged"
                                                      else rw.get("live_success") == rw.get("replay_success") == 1)
            want = ((standard_expect(a, b["new"]) if mode == "standard" else None)   # a speed-rule standard reference's end
                    or (a.expect or {}).get(b["new"]) or (a.expect or {}).get(b["old"]))
            match = None if not want else (len(ends) == 1 and ends[0].startswith(want))
            good = bool(o.get("pass") and n.get("pass") and claims and match is not False)
            ok &= good
            rows.append(dict(base=b["new"], mode=mode, oracle=o.get("reward"), nop=n.get("reward"), ends=ends,
                             expected=want, evidence_holds=claims, oracle_s=o.get("wall_s"), nop_s=n.get("wall_s"),
                             exception=o.get("exception") or n.get("exception"), ok=good))
    return rows, ok


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("head", nargs="?")
    ap.add_argument("--version")
    ap.add_argument("--manifest", type=pathlib.Path)
    ap.add_argument("--onto")
    ap.add_argument("--bench")
    ap.add_argument("--instances", type=pathlib.Path)
    ap.add_argument("--gpu")
    ap.add_argument("--work", type=pathlib.Path)
    ap.add_argument("--families")
    ap.add_argument("--only")
    ap.add_argument("--references", type=pathlib.Path)
    ap.add_argument("--expect", type=pathlib.Path)
    ap.add_argument("--hf-revision")
    ap.add_argument("--lanes", type=int)
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--harbor", default="harbor")
    ap.add_argument("--host-label", default=socket.gethostname().split(".")[0])
    ap.add_argument("--replace-tag", action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("--no-harbor", action="store_true")
    ap.add_argument("--harbor-only", action="store_true", help="only step 5 and the report, on --work's previous run")
    a = ap.parse_args(argv)
    m = json.loads(a.manifest.read_text()) if a.manifest else {}
    mdir = a.manifest.resolve().parent if a.manifest else pathlib.Path.cwd()
    for k, default in (("head", None), ("version", None), ("onto", None), ("bench", BENCH), ("families", None), ("only", None), ("hf_revision", None),
                       ("gpu", "0"), ("lanes", 3), ("jobs", 4)):
        if getattr(a, k) is None:
            v = m.get(k, default)
            setattr(a, k, ",".join(v) if k in ("families", "only") and isinstance(v, list) else v)
    if not a.head or not a.version:
        ap.error("the head and --version are needed (on the command line or in the manifest)")
    if a.bench == "robotangle":
        ap.error("--bench robotangle (the Genesis in-place release) is not supported here")
    if not a.onto and not a.harbor_only:
        ap.error("--onto <commit> is required: the tree the release files and the task dirs are rendered on")
    path = lambda v: (mdir / v).resolve()  # noqa: E731
    per = m.get("bases", {})
    a.ref_paths = {b: path(r["reference"]) for b, r in per.items() if r.get("reference")}
    # "reference": null: a base without a reference solution (user 2026-10-13): no oracle, no birth comparison, nop only
    a.no_oracle = {b for b, r in per.items() if "reference" in r and r["reference"] is None}
    for b, r in per.items():             # a reference named with its sha256 must be that file
        if r.get("reference") and r.get("sha256") and sha(a.ref_paths[b].read_bytes()) != r["sha256"]:
            raise SystemExit(f"{b}: {a.ref_paths[b]} is not the manifest's sha256 {r['sha256']}")
    a.premade = {b: path(r["instance"]) for b, r in per.items() if r.get("instance")}
    if a.instances:
        a.premade.update({d.name: d.resolve() for d in a.instances.iterdir() if (d / "instance.json").is_file()})
    a.step_s = {b: float(r["step_s"]) for b, r in per.items() if r.get("step_s")}
    expect = {b: r["expect"] for b, r in per.items() if r.get("expect")}
    here = pathlib.Path(__file__).resolve().parents[2]
    head = run(["git", "-C", str(here), "rev-parse", "--verify", f"{a.head}^{{commit}}"]).strip()
    a.work = (a.work or pathlib.Path(f"release-rebuild-{head[:9]}")).resolve()
    a.references = a.references.resolve() if a.references else None
    a.expect = {**expect, **(json.loads(a.expect.read_text()) if a.expect else {})} or None
    if a.harbor_only:                    # a previous run's render tree, image and report: Harbor again (e.g. after an infra error)
        report = json.loads((a.work / "report.json").read_text())
        a.repo, a.out = pathlib.Path(report.get("repo") or a.work / "repo"), a.work / ("render" if report.get("onto") else "repo")
        bases = [dict(new=n, old=i["replaces"]) for n, i in report["instances"].items()]
        if a.only:                       # a subset of that run's bases
            bases = [b for b in bases if b["new"] in a.only.split(",")]
        return harbor_step(a, here, report, bases, report.get("gpu_bases", []), report["images"]["release"]["reference"],
                           module(here / "scripts/roboarc/harbor_check.py"))
    a.repo = a.work / "repo"
    image = a.tag or f"ghcr.io/mll-lab-nu/{a.bench}:{a.version}"
    noinst_tag = f"{a.tag}-noinst" if a.tag else f"{a.bench}:{a.version}-noinst"
    tags = [image] if a.tag else [image, f"{a.bench}:{a.version}"]
    if image_id(image) and not a.replace_tag:
        raise SystemExit(f"{image} exists here; pick another --version or --tag, or pass --replace-tag")
    (a.work / "logs").mkdir(parents=True, exist_ok=True)
    report = dict(head=head, version=a.version, work=str(a.work), started=datetime.datetime.now().isoformat(timespec="seconds"))

    # 1. the worktree of <head> and the image without instances
    if a.repo.exists():
        run(["git", "-C", str(here), "worktree", "remove", "--force", str(a.repo)])
    branch = f"release-rebuild/{head[:9]}-{a.work.name}"   # one per work dir: worktrees cannot share a branch
    run(["git", "-C", str(here), "worktree", "add", "-B", branch, str(a.repo), head])
    if not (a.repo / "tools/release/mujoco_release.json").is_file() or not (a.repo / "source/topogym").is_dir():
        raise SystemExit(f"{a.head}: not a commit of this repository's layout (source/topogym, tools/release/)")
    ctx = a.repo / "source"                 # the image build context, as images/roboarc/build.sh uses it
    spec = json.loads((a.repo / "tools/release/mujoco_release.json").read_text())
    # GPU profiles: the head's list and this checkout's (a head from before the list names none)
    a.gpu_profiles = set(spec.get("gpu_profiles", [])) | set(json.loads(
        (here / "tools/release/mujoco_release.json").read_text()).get("gpu_profiles", []))
    bases = bases_of(spec, a.families.split(",") if a.families else None)
    unknown = (set(a.premade) | set(a.ref_paths) | set(a.step_s)) - {b["new"] for b in bases}
    if unknown:
        raise SystemExit(f"the manifest / --instances name bases that are not being rebuilt: {sorted(unknown)}")
    if a.only:
        missing = set(a.only.split(",")) - {b["new"] for b in bases}
        if missing:
            raise SystemExit(f"--only: not in mujoco_release.json (or not in --families): {sorted(missing)}")
        bases = [b for b in bases if b["new"] in a.only.split(",")]
        keep = {b["new"] for b in bases}         # a whole-release manifest narrowed by --only: its other bases left out
        for d in (a.premade, a.ref_paths, a.step_s):
            for k in set(d) - keep:
                del d[k]
        a.no_oracle &= keep
        a.expect = {k: v for k, v in a.expect.items() if k in keep} if a.expect else a.expect
    # the render tree: the --onto worktree with <head>'s tooling laid over it, or <head>'s own
    a.out, out_branch = a.repo, branch
    if a.onto:
        onto = run(["git", "-C", str(here), "rev-parse", "--verify", f"{a.onto}^{{commit}}"]).strip()
        a.out, out_branch = a.work / "render", f"release-render/{head[:9]}-{a.work.name}"
        if a.out.exists():
            run(["git", "-C", str(here), "worktree", "remove", "--force", str(a.out)])
        run(["git", "-C", str(here), "worktree", "add", "-B", out_branch, str(a.out), onto])
        if not (a.out / "scripts" / a.bench / "mk_tasks.py").is_file():
            raise SystemExit(f"--onto {a.onto}: no scripts/{a.bench}/mk_tasks.py")
        for n in TOOLING:
            shutil.copy2(a.repo / "scripts/roboarc/shared" / n, a.out / "scripts" / a.bench / "shared" / n)
        report.update(onto=onto)
    for b in bases:
        if b["new"] not in a.no_oracle:
            reference(b, a.references, a.repo, a.ref_paths)        # every reference exists before anything is built
        rb = a.out / "scripts" / a.bench
        has_texts = all((rb / "packages" / f"{b['new']}-{m}").is_dir() for m in ("privileged", "standard"))
        if not has_texts:                 # every base has texts: its own packages, or (a new base) its template's
            if not b["template"]:
                raise SystemExit(f"{b['new']}: no packages/{b['new']}-* in the render tree")
            t = template_base(rb, b["template"])
            if not all((rb / "packages" / f"{t}-{m}").is_dir() for m in ("privileged", "standard")):
                raise SystemExit(f"{b['new']}: its template {t} has no packages in the render tree")
        if a.hf_revision and not (a.work / "instances" / b["new"] / "instance.json").is_file():
            raise SystemExit(f"--hf-revision uses the instances of a previous run in {a.work}: {b['new']} has none")
    # harbor_check.py is this checkout's (the runner), mk_tasks.py the render tree's; MUJOCO_ENV is the same in both
    gen = f"scripts/{a.bench}/mk_tasks.py"     # the render tree's generator
    mk, hc = module(a.out / gen), module(here / "scripts/roboarc/harbor_check.py")
    if getattr(mk, "PACKAGE_FORMAT", None) != "final-texts/1":
        raise SystemExit(f"--onto {a.onto}: scripts/{a.bench}/mk_tasks.py does not render final packages "
                         "(PACKAGE_FORMAT \"final-texts/1\")")
    empty = a.work / "empty"
    empty.mkdir(exist_ok=True)
    build = ["docker", "buildx", "build", "--load", "-f", "images/roboarc/Dockerfile",
             "--build-arg", f"RELEASE_VERSION={a.version}"]       # as images/<bench>/build.sh builds it
    run([*build, "--build-context", f"instances={empty}", "-t", noinst_tag, "."], log=a.work / "logs/build_noinst.log",
        cwd=ctx)
    noinst = dict(reference=noinst_tag, id=image_id(noinst_tag), repo_digests=[])

    def installed(ref: str) -> dict:
        out = run(["docker", "run", "--rm", "--network", "none", "-v",
                   f"{a.repo / 'scripts/roboarc/shared'}:/tooling:ro", ref, "python", "-c", hc.RUNTIME])
        return json.loads(out.strip().splitlines()[-1])

    rt = installed(noinst_tag)

    # 2. the instances, made inside it (or, with --hf-revision, the uploaded ones of the previous run); the release image
    walls = {} if a.hf_revision else make_instances(a, bases, noinst_tag, mk.MUJOCO_ENV)
    walls.update({n: None for n in a.premade})
    instances = {}
    for b in bases:
        d = a.work / "instances" / b["new"]
        inst = json.loads((d / "instance.json").read_text())
        assert inst["simulator"]["physics_engine"] == "mujoco", b["new"]
        if b["new"] not in a.premade:
            assert inst["topogym_source"]["sha256"] == rt["runtime"]["source"]["sha256"], f"{b['new']}: made on another source"
        recorded = (None if b["new"] in a.no_oracle else
                    npz_text(reference(b, a.references, a.repo, a.ref_paths), "instance_sha.npy"))
        if recorded not in (None, inst["state_sha"]):
            raise SystemExit(f"{b['new']}: birth {inst['state_sha']}, its reference was recorded on {recorded}")
        instances[b["new"]] = dict(replaces=b["old"] or f"none (new; template {b['template']})",
                                   variation=b["variation"], birth=inst["state_sha"],
                                   reference_birth=recorded,
                                   instance_sha256=canonical(inst), files=len(files_of(d)),
                                   bytes=sum(p.stat().st_size for p in d.rglob("*") if p.is_file()),
                                   make_s=("made elsewhere (--instances), source " + inst["topogym_source"]["sha256"][:8]
                                           if b["new"] in a.premade else round(walls[b["new"]], 1) if walls
                                           else "uploaded, used again"))
    run([*build, "--build-context", f"instances={a.work / 'instances'}", *(x for t in tags for x in ("-t", t)), "."],
        log=a.work / "logs/build_release.log", cwd=ctx)
    rel_rt = installed(image)
    assert rel_rt == rt, "the release image's runtime differs from the one the instances were made on"
    facts = json.loads(run(["docker", "run", "--rm", "--network", "none", "-e", "CUDA_VISIBLE_DEVICES=", "-v",
                            f"{a.repo / 'scripts/roboarc/shared'}:/tooling:ro", "-v", f"{a.repo / 'tools/release'}:/scripts:ro",
                            "-v", f"{a.work}:/work", image, "python",
                            "-c", FACTS, json.dumps(sorted(a.gpu_profiles)), *(b["new"] for b in bases)]).strip().splitlines()[-1])
    gpu_bases = [b["new"] for b in bases if facts[b["new"]]["gpu_physics"]]
    report.update(facts=facts, gpu_bases=gpu_bases)
    runtime = dict(rt["runtime"])
    report.update(images=dict(noinst=noinst, release=dict(reference=image, id=image_id(image))),
                  runtime=runtime, instances=instances)
    # the release image's deterministic archive (the bytes images/push_all.sh pushes) and its pin: the tasks name the
    # image by that manifest digest (images/pins.json in the render tree; the same image gives the same digest)
    if (a.out / "images/pins.json").is_file():
        reg = f"ghcr.io/mll-lab-nu/{a.bench}:{a.version}"
        arch = a.work / "image" / f"{a.bench}-{a.version}.oci.tar"
        made = json.loads(run(["bash", str(here / "images/archive.sh"), image, str(arch), reg]))
        assert made["config_digest"] == report["images"]["release"]["id"], made
        pins_path = a.out / "images/pins.json"
        pins = json.loads(pins_path.read_text())
        entry = dict(name=a.bench, ref=f"{reg}@{made['manifest_digest']}", id=made["config_digest"], archive=arch.name)
        k = next((i for i, p in enumerate(pins["images"]) if p["name"] == a.bench), None)
        if k is None:
            pins["images"].insert(0, dict(entry, made_by="images/roboarc/build.sh, then images/archive.sh"))
        else:
            pins["images"][k].update(entry)
        pins_path.write_text(json.dumps(pins, indent=1) + "\n")
        report["images"]["release"].update(archive=str(arch), manifest_digest=made["manifest_digest"])

    # 3. the release files, every task dir rendered; 4. the checks
    done = write_release(a, bases, head, runtime, noinst, facts)
    report.update(packages_written=done["written"], unchanged_bases=done["kept"])
    py = sys.executable
    run([py, gen], log=a.work / "logs/mk_tasks.log", cwd=a.out)
    checks = {}
    for name, cmd in (("mk_tasks.py --check, all bases", [py, gen, "--check"]),
                      ("mk_tasks.py --check, MuJoCo instance bytes", [py, gen, "--assets", str(a.work), "--check",
                                                 *(b["new"] for b in bases)])):
        r = subprocess.run(cmd, cwd=a.out, capture_output=True, text=True)
        checks[name] = dict(ok=r.returncode == 0, output=(r.stdout + r.stderr).strip().splitlines()[-5:])
    if a.onto:                          # the repository's own checks of the changed tasks, and the dataset manifest
        changed = sorted({f"tasks/{p.split('/')[1]}" for p in run(["git", "-C", str(a.out), "status", "--porcelain",
                                                                    "--untracked-files=all", "tasks"]).split()
                          if p.startswith(f"tasks/{a.bench}-")} & {f"tasks/{d.name}" for d in (a.out / "tasks").iterdir()})
        report["changed_tasks"] = len(changed)
        if (a.out / "tasks/dataset.toml").is_file():   # main since #215 keeps no tasks/dataset.toml (per-task DIGEST)
            r = subprocess.run([*a.harbor.split(), "add", "--scan", "."], cwd=a.out / "tasks", capture_output=True, text=True)
            checks["harbor add --scan (tasks/dataset.toml)"] = dict(ok=r.returncode == 0,
                                                                    output=(r.stdout + r.stderr).strip().splitlines()[-3:])
            dropped = drop_missing(a.out / "tasks")      # `harbor add` keeps the entries of removed dirs (the -i00 bases)
            report["dataset_dropped"] = dropped
        env = dict(os.environ, PATH=f"{pathlib.Path(py).parent}:{os.environ['PATH']}")   # python3 >= 3.11 (tomllib)
        for name, cmd in (("ci_checks/run_all.sh", ["bash", "ci_checks/run_all.sh", *changed]),
                          ("task_conventions --check", [py, "scripts/task_conventions.py", "--check", *changed]),
                          ("check-resources", [py, "ci_checks/check-resources.py", *changed]),
                          ("check-cutoffs", [py, "ci_checks/check-cutoffs.py", *changed])):
            r = subprocess.run(cmd, cwd=a.out, capture_output=True, text=True, env=env)
            (a.work / "logs" / f"check_{name.replace('/', '_').replace(' ', '_')}.log").write_text(r.stdout + r.stderr)
            if name == "ci_checks/run_all.sh":   # its repo-wide part also re-renders other benchmarks (their deps, not ours)
                fails = [x for x in (r.stdout + r.stderr).splitlines() if x.startswith("FAIL")]
                ours = [x for x in fails if a.bench in x or "dataset" in x or "tooling" in x or "Traceback" in x]
                checks[name] = dict(ok=not ours, output=ours[-5:] or [f"{len(fails)} FAIL lines, none about this benchmark, "
                                                                     "the dataset manifest or the tooling (see the log)"])
                continue
            checks[name] = dict(ok=r.returncode == 0, output=(r.stdout + r.stderr).strip().splitlines()[-5:])
    report["checks"] = checks
    run(["git", "-C", str(a.out), "add", "-A"])
    made = [b["new"] for b in bases if b["new"] not in a.premade]
    pre = [b["new"] for b in bases if b["new"] in a.premade]
    msg = (f"RoboARC release rebuild on {head[:9]} (image {a.version}): "
           + "; ".join(([f"MuJoCo instances made in the release image, {', '.join(made)}"] if made else [])
                       + ([f"premade MuJoCo instances (--instances) bound into it, {', '.join(pre)}"] if pre else []))
           + f"\n\ntools/release/release_rebuild.py; report: {a.work}/REPORT.md\n")
    run(["git", "-C", str(a.out), "commit", "-q", "--allow-empty", "-m", msg])   # a release re-made as it was: no change
    commit = run(["git", "-C", str(a.out), "rev-parse", "HEAD"]).strip()
    report.update(commit=commit, branch=out_branch,
                  diffstat=run(["git", "-C", str(a.out), "diff", "--shortstat", "HEAD~1", commit]).strip())

    return harbor_step(a, here, report, bases, gpu_bases, image, hc)


def harbor_step(a, here: pathlib.Path, report: dict, bases: list[dict], gpu_bases: list[str], image: str, hc) -> int:
    """5. Harbor on the render tree's commit, then the report (also run alone: --harbor-only)."""
    ok = all(c["ok"] for c in report["checks"].values())
    if not a.no_harbor and ok:
        if (a.work / "harbor").exists():
            (a.work / "harbor").rename(a.work / f"harbor.{datetime.datetime.now():%Y%m%dT%H%M%S}")
        rc = 0
        for run_dir, agents, sel in (("harbor", "oracle,nop", [b for b in bases if b["new"] not in a.no_oracle]),
                                     ("harbor_nop", "nop", [b for b in bases if b["new"] in a.no_oracle])):
            if not sel:                  # bases without a reference solution: nop only, in their own run directory
                continue
            r = subprocess.run([sys.executable, str(here / "scripts/roboarc/harbor_check.py"), "--tasks", str(a.out / "tasks"),
                                "--image", image, "--run", str(a.work / run_dir), "--harbor", a.harbor, "--lanes", str(a.lanes),
                                "--agents", agents, "--gpu-bases", ",".join(gpu_bases), "--bench", a.bench,
                                *(b["new"] for b in sel)], cwd=here,
                               stdout=open(a.work / f"logs/{run_dir}.log", "w"), stderr=subprocess.STDOUT)
            rc = rc or r.returncode
        rows, harbor_ok = harbor_table(a, bases, hc)
        report["harbor"] = dict(rc=rc, rows=rows)
        ok &= harbor_ok and rc == 0
    report.update(ok=ok, finished=datetime.datetime.now().isoformat(timespec="seconds"))
    (a.work / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (a.work / "REPORT.md").write_text(report_md(report, a, bases))
    print((a.work / "REPORT.md").read_text())
    return 0 if ok else 1


def report_md(r: dict, a, bases: list[dict]) -> str:
    rt = r["runtime"]
    out = [f"# RoboARC release rebuild on {r['head'][:9]}: {'PASS' if r['ok'] else 'FAIL'}", "",
           f"- head `{r['head']}`, image version `{a.version}`, {r['started']} - {r['finished']}",
           f"- images (local only): release `{r['images']['release']['reference']}` = `{r['images']['release']['id']}`; "
           f"made in `{r['images']['noinst']['reference']}` = `{r['images']['noinst']['id']}`",
           f"- runtime: source `{rt['source']['sha256']}` ({rt['source']['files']} files), python {rt['python']}, "
           + ", ".join(f"{k} {v}" for k, v in rt["dependencies"].items()),
           f"- commit `{r['commit']}` on the local branch `{r['branch']}` in `{a.out}` ({r['diffstat']})"
           + (f", on `{r['onto']}`" if r.get("onto") else ""), "",
           "## Instances (made in the image without instances)", "",
           "| base | replaces | variation | birth (state_sha) | = reference's | instance sha256 | files | bytes | make s |",
           "|---|---|---|---|---|---|---|---|---|"]
    out += [f"| {n} | {i['replaces']} | {i['variation']} | `{i['birth']}` | {'yes' if i['reference_birth'] else 'not recorded'} | "
            f"`{i['instance_sha256'][:16]}` | {i['files']} | {i['bytes']} | {i['make_s']} |" for n, i in r["instances"].items()]
    out += ["", "## Checks", ""] + [f"- {k}: {'ok' if c['ok'] else 'FAIL'}"
                                     + ("" if c["ok"] else ": " + " / ".join(c["output"])) for k, c in r["checks"].items()]
    if "harbor" in r:
        out += ["", "## Harbor (oracle and nop, rendering off)", "",
                "| task | oracle | nop | oracle end state | expected | evidence claims hold | oracle s | nop s | ok |",
                "|---|---|---|---|---|---|---|---|---|"]
        out += [f"| {x['base']}-{x['mode']} | {x['oracle']} | {x['nop']} | {' / '.join(x['ends'])} | {x['expected'] or ''} | "
                f"{'n/a (no reference)' if x['oracle'] == 'none' else 'yes' if x['evidence_holds'] else 'NO'} | {x['oracle_s']} | {x['nop_s']} | {'yes' if x['ok'] else 'NO'} |"
                for x in r["harbor"]["rows"]]
        out += ["", f"Run: `{a.work}/harbor` (results.md, results.json, jobs/); release.json's evidence names its oracle "
                "trials."]
    else:
        out += ["", "Harbor: not run (--no-harbor, or a check failed)."]
    out += ["", "## Gated, not done here", ""]
    if a.hf_revision:
        out += [f"- HF: the instances are the dataset's commit `{a.hf_revision}`, which assets.json pins."]
    elif len(r.get("unchanged_bases") or {}) == len(r["instances"]):
        out += ["- HF: nothing to upload: every instance is the one assets.json already pins (every base unchanged)."]
    else:
        out += ["- HF upload of the instances (the user's OK first; upload only, nothing deleted). NOT UPLOADED: "
                "assets.json's new pins name files that are not in the dataset at its revision.",
                f"  `hf upload eai-v2-internal/robotangle-assets {a.work}/instances instances --repo-type dataset "
                f"--commit-message \"RoboARC MuJoCo instances made in ghcr.io/mll-lab-nu/{a.bench}:{a.version} ({r['head'][:9]})\"`",
                f"  then run this again with `--work {a.work} --hf-revision <the new dataset commit>`, which uses these "
                "instances again (and runs Harbor again on the final bytes)."]
    out += ["- Image push, and the push of the commit: the user's OK."]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
