#!/usr/bin/env python3
"""The RoboTangle release rebuild in one command (docs/rcb/MUJOCO_MIGRATION.md, Release): on a docker host with harbor,
from a commit of this repository, without a GPU.

usage: scripts/robotangle/release_rebuild.sh <head> --version <v> [options]
       scripts/robotangle/release_rebuild.sh --manifest <freeze.json> [options]
  <head>          the commit to rebuild the release on; scripts/robotangle/mujoco_release.json there names the MuJoCo
                  bases and how their instances are made
  --manifest      the post-freeze inputs in one file (scripts/robotangle/freeze_manifest.example.json): head (the
                  frozen commit), version, onto, families, hf_revision, gpu, lanes, jobs, and per base (new name):
                  reference (privileged.npz; relative paths are relative to the manifest), its sha256, expect (end
                  state prefix), instance (a made instance directory to use instead of making it), step_s (the wall
                  time per control step the texts state, measured on MuJoCo). Options on the command line win.
  --bench         the benchmark the MuJoCo tasks belong to (default robotangle-new, user decision 2026-10-12: RoboTangle on
                  MuJoCo is a benchmark of its own; one parameter, here and BENCH in scripts/<bench>/mk_tasks.py on main):
                  with --onto, its files are scripts/<bench>/ (release.json, assets.json, packages/, oracles/, shared/),
                  images/<bench>/VERSION, the image ghcr.io/embodied-first-exam/<bench>:<version> and the task dirs
                  tasks/<bench>-<base>-<mode>; RoboTangle's own files and tasks (scripts/robotangle/, robotangle-*) are
                  read (the released texts each new base ports), never changed. --bench robotangle: the earlier in-place
                  release (an -i01 base replaces its -i00 in RoboTangle)
  --onto          render the task dirs on this commit (main, or a main-based branch whose mk_tasks.py renders MuJoCo
                  bases) instead of on <head>: the release files are written there, with <head>'s tooling
                  (shared/rcb_topogym.py, render_bridge.py, nyx_direct.py) laid over it; then the repository checks
                  of the changed tasks and `harbor add --scan` (tasks/dataset.toml, where the tree still has one)
  --instances     made instance directories, <dir>/<new base>/ (instance.json + instance_assets/): used instead of
                  making them (a family's own make, a dry run); the report marks them, and an instance made on another
                  source than the image's is allowed only here
  --gpu           the GPU (device index) for the make-instance runs and the Harbor trials of a base on a GPU profile
                  (mujoco_release.json gpu_profiles, e.g. blanket_vbd_gpu; default 0)
  --version       the image version: images/<bench>/VERSION and the tags ghcr.io/embodied-first-exam/<bench>:<v> and
                  <bench>:<v> (local only; refused if the tag exists, unless --replace-tag)
  --work          the work directory (default ./release-rebuild-<head9>): repo/ (a worktree of <head> where every
                  generated file is written and committed on the local branch release-rebuild/<head9>), instances/ (the
                  instances made, in the HF dataset's layout), cache/ (CoACD hulls), harbor/ (the Harbor run), logs/,
                  REPORT.md and report.json
  --families      only these families of mujoco_release.json (comma-separated)
  --only          only these new bases (comma-separated; a dry run of single tiers)
  --references    reference trajectories, <dir>/<base>/privileged.npz (the new or the released base name); default
                  oracles/<new base>/privileged.npz in the tree
  --expect        a JSON file {base: end state (prefix)}: the report compares the oracles' end states with it
  --hf-revision   the dataset commit that holds the instances of a previous run in --work (the upload is a separate,
                  gated step: REPORT.md prints it); those instances are used again, not made again. Without it,
                  assets.json keeps its revision and its new pins name files that are not uploaded
  --lanes, --jobs concurrent Harbor trials (default 3) and make-instance / seed derivation runs (default 4)
  --appearance-seeds  also each base's instance for these appearance seeds (comma-separated, >= 1; manifest
                  "appearance_seeds": a list), derived from the base's instance (made in this run or premade) in the
                  image without instances (derive_seed_instance.py, next to this file): the task parameters
                  topogym.tasks.scene_variation.appearance(<task class>, S) over the instance's own, instance_assets/
                  byte for byte (frozen MuJoCo scene, birth archives, frozen generated assets), the build's fields
                  (state_sha, asset_identity, ...) recomputed; the birth must equal the base's but for task.instance_id
                  (the user, 2026-10-20: a seed varies appearance only, one reference serves every seed). Written to
                  instances/<base>/seeds/<S>/ (instance.json + instance_assets/), which the image copies with its base
                  (/opt/robotangle/instances/<base>/seeds/<S>/) and assets.json pins (files under
                  instances/<base>/seeds/<S>/, instances[<base>].seeds[<S>] instance_sha256); a premade base that brings
                  its own seeds/<S>/ keeps them. Seed 0 is the base instance itself, made as without this option. For
                  each seed S the release files also get (write_seed_files):
                  packages/<base>-privileged/runtime_binding.s<S>.upstream.json, the seed instance's lock (mk_tasks.py:
                  runtime_binding.s<S>.json), and oracles/<base>/privileged.s<S>.npz, the reference re-stamped (restamp:
                  the same rows, instance_sha = seed S's birth; mk_tasks.py: solution/trajectory.s<S>.npz); and, for a
                  base whose packages show goal pictures (environment/goal/), the seed's own in
                  packages/<base>-<mode>/tests/seeds/<S>/goal/, pinned in assets.json under
                  instances[<base>].seeds[<S>].goal (seed_goal_pictures)
  --goal-pictures the appearance seeds' goal pictures rendered elsewhere, <dir>/<base>/seeds/<S>/ (the renderer's
                  pictures and its record *_images.json; manifest "goal_pictures"), used instead of rendering them in
                  this run (the render tree's scripts/<bench>/goal_images/<renderer>, in the release image on --gpu)
  --no-harbor     stop after the checks
A manifest base with "reference": null has no reference solution (user 2026-10-13): no oracles/ copy, release.json
evidence {} and oracle "none" (its texts say so), no birth comparison, and Harbor runs nop only for it (harbor_nop/).
Steps:
  1. a worktree of <head>; the image without MuJoCo instances (images/robotangle/Dockerfile.mujoco, an empty
     `instances` context): eai-<bench>:<v>-noinst
  2. every MuJoCo base's instance made inside it (rcb_topogym.py make-instance, the MuJoCo tasks' SIMD masking, no
     network), then the release image with them: ghcr.io/embodied-first-exam/<bench>:<v>; the runtime of both must be
     the same (installed_runtime)
  3. the release written: VERSION; packages/<new>-* (the released base's texts, renamed, and the made instance's lock;
     with --onto a benchmark of its own, a base the render tree already has packages for keeps those texts, only its
     lock is new;
     a family whose mujoco_release.json entry names package_text, a directory of the tree, takes the files of its
     <package_text>/<released base>-<mode>/ over the released ones: the task texts of a changed design; a family's
     new_bases are tasks of a new design: each takes its template's released texts, renamed, with the files of its
     <package_text>/<new base>-<mode>/ over them, and leaves the template as it is; a template of another family: its
     family (task.toml) replaced by the new base's task where the grader and the sidecar are started (tests/test.sh
     --family, the standard compose's RCB_TASK), and refused while any other text still names it);
     oracles/<new>/ (the reference; standard.npz is the same file: a MuJoCo standard oracle streams the reference's
     rows and mk_tasks.py reads only its actions; unless the tree's oracles/<new>/standard.npz is a reference re-timed
     to the standard service's joint-speed rule, meta "speed_rule": then that file, standard_reference()); release.json (runtime, tooling, the MuJoCo base records with their
     evidence: this run); assets.json (the made instances' pins); with --appearance-seeds the seeds' locks, re-stamped
     references and goal pictures (rendered in the release image unless --goal-pictures); every task dir rendered
     (mk_tasks.py)
  4. mk_tasks.py --check (every base) and --assets <work> --check (the MuJoCo bases' instance bytes)
  5. a local commit, then harbor_check.py --tasks: oracle and nop, both modes, rendering off, on that commit (a base
     on a GPU profile keeps its one-GPU reservation)
Facts the texts state are read in the release image, per base, from its instance and the tooling (release.json):
grasp_rules (Session.grasp_rules() of the instance's configuration), physics_dt_s (the contact-review window in
ticks), gpu_physics (a GPU profile), grasp ([metadata] grasp) and grasp_regions (the grader-details line per mode, the
head's scripts/topogym_rules.py): the texts state the instance's grasp, not the released Genesis one; step_s comes from
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
import tomllib
import zipfile

BENCH = "robotangle-new"   # the default --bench: RoboTangle on MuJoCo, its own benchmark (rename: here and on main)
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
params = inst["simulator"]["task_parameters"] if job["released_task_parameters"] else None
if job["released_task_parameters"]:
    json.dump(params, open(job["task_parameters_file"], "w"), indent=1)
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


def restamp(ref: bytes, birth: str) -> bytes:
    """A reference re-stamped with another birth (an appearance seed's state_sha; the user, 2026-10-20: a seed changes
    the looks only, so the reference's rows serve every seed): instance_sha.npy's text replaced under its own .npy header
    (the birth padded to the header's width, never longer), every other entry byte for byte, in the same order and with
    the same names, dates and compression. Standard library only (npz_text's layout)."""
    zin = zipfile.ZipFile(io.BytesIO(ref))
    raw = zin.read("instance_sha.npy")
    size = 4 if raw[6] >= 2 else 2
    head = 8 + size + int.from_bytes(raw[8:8 + size], "little")
    new = birth.encode("utf-32-le")
    if b"'descr': '<U" not in raw[:head] or len(new) > len(raw) - head:
        raise SystemExit(f"restamp: the reference's instance_sha.npy cannot hold the birth {birth!r}")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zout:
        for info in zin.infolist():           # fresh records: the reader's extra fields (zip64) are not carried over
            zi = zipfile.ZipInfo(info.filename, info.date_time)
            zi.compress_type, zi.external_attr = info.compress_type, info.external_attr
            zout.writestr(zi, raw[:head] + new.ljust(len(raw) - head, b"\0") if info.filename == "instance_sha.npy"
                          else zin.read(info.filename))
    data = out.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        same = z.namelist() == zin.namelist() and all(z.read(n) == zin.read(n) for n in z.namelist() if n != "instance_sha.npy")
    if not same or npz_text(io.BytesIO(data), "instance_sha.npy") != birth:
        raise SystemExit(f"restamp: the re-stamped reference is not the reference with birth {birth}")
    return data


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
                            package_text=r.get("package_text"), task=f, template=None, design=r.get("design"),
                            **{**spec["make_instance"], **r.get("make_instance", {})}))
        for new, n in r.get("new_bases", {}).items():   # a new design: no released base; its texts start from a template's
            unknown = set(n) - {"template", "variation", "tier", "task", "scene_seed", "gripper_space", "package_text",
                                "design", "note", "simulator_overrides"}
            if unknown or not n.get("template") or not n.get("variation"):
                raise SystemExit(f"{f} new_bases {new}: needs template and variation (unknown keys {sorted(unknown)})")
            out.append(dict(family=f, tier=n.get("tier", n["variation"].removeprefix(f"{f}_")), old=None, new=new,
                            variation=n["variation"], released_simulator_keys=[],
                            simulator_overrides=n.get("simulator_overrides", {}),   # fixed values only (no released instance)
                            render_host=False, frozen_assets=False, released_task_parameters=False, grasp_weld_links=None,
                            package_text=n.get("package_text", r.get("package_text")), task=n.get("task", f),
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
            [repo / "scripts/robotangle/oracles" / b["new"] / "privileged.npz"]
    for c in cands:
        if c.is_file():
            return c
    raise SystemExit(f"{b['new']}: no reference (looked for {', '.join(map(str, cands))})")


def standard_reference(repo: pathlib.Path, base: str, privileged: bytes) -> bytes:
    """The standard oracle's rows: the reference itself, unless the tree holds a standard reference re-timed to the
    standard service's joint-speed rule (scripts/robotangle/oracles/<base>/standard.npz whose meta has "speed_rule":
    the reference's rows with every arm-joint step within JOINT_STEP_LIMIT_RAD, 2026-10-14) on the reference's birth:
    a tree file recorded on another birth (its instance_sha is not the reference's, e.g. a copy from before a
    re-freeze) is not the standard oracle of this reference. The privileged reference is never replaced by it."""
    p = repo / "scripts/robotangle/oracles" / base / "standard.npz"
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
    """A speed-rule standard reference's record (scripts/robotangle/standard_speed_rule.json): `words`, the sentence the
    standard evidence states (release.json `speed_rule`, which mk_tasks.py renders into the standard README's Reference
    solution), and `end_state`, the end it reaches (the evidence's, and the end the Harbor standard oracle must reach:
    not the privileged reference's)."""
    recs = json.loads((repo / "scripts/robotangle/standard_speed_rule.json").read_text())["bases"]
    if base not in recs:
        raise SystemExit(f"{base}: its standard oracle is a speed-rule reference, but scripts/robotangle/"
                         "standard_speed_rule.json has no record of it (words, end_state)")
    return recs[base]


# The onto tree's mk_tasks.py words a standard reference re-timed to the joint-speed rule (release.json standard evidence
# with "speed_rule"; dev/speedrule-refs-main 31f7a28beb). Laid over a generator that lacks it, as the tooling is; every
# anchor must still be there (a generator that rewords them fails here instead of silently losing the words).
SPEED_RULE_DOC = ("  oracles/       the references (privileged.npz; standard.npz is the same file)\n",
                  "  oracles/       the references (privileged.npz; standard.npz is the same file, or, where release.json's standard evidence\n"
                  "                 has \"speed_rule\", the reference re-timed to the standard service's joint-speed rule)\n")
SPEED_RULE_ANCHOR = '        return ("`solution/solve.sh` streams the recorded rows of the standard-mode oracle episode "\n'
SPEED_RULE_BRANCH = (
    '        if ev.get("speed_rule"):     # 2026-10-14: a standard reference re-timed to the joint-speed rule (oracles/<base>/standard.npz)\n'
    '            return ("`solution/solve.sh` streams the rows of the standard reference (`solution/oracle_qpos.npy`, one "\n'
    '                    "`joint_position` row per control step, `hold`) through the client (`solution/oracle_replay.py`): "\n'
    '                    f"{ev[\'speed_rule\']} That episode scored 1 (live and fresh replay, end state `{ev[\'end_state\']}`) on "\n'
    '                    f"{run_words(ev)}.")\n')


def speed_rule_wording(gen: pathlib.Path) -> bool:
    """Gives the onto tree's generator the speed-rule wording unless it has it; True when it was added."""
    s = gen.read_text()
    if 'ev.get("speed_rule")' in s:
        return False
    if s.count(SPEED_RULE_DOC[0]) != 1 or s.count(SPEED_RULE_ANCHOR) != 1:
        raise SystemExit(f"{gen}: the speed-rule wording's anchors changed (ref_solution's standard return, the oracles/ line)")
    gen.write_text(s.replace(SPEED_RULE_DOC[0], SPEED_RULE_DOC[1]).replace(SPEED_RULE_ANCHOR, SPEED_RULE_BRANCH + SPEED_RULE_ANCHOR))
    return True


# The onto tree's mk_tasks.py drops a "## Task failure conditions" section left empty (2026-10-17 instruction audit V6):
# audit_words' B4 removes a chain base's rope-volume rules, the only failure rule of knot_untying and rope_threading,
# so the heading stood alone (main's 26 released knot/rope dirs too). Laid over the generator as the speed-rule wording
# is, inside audit_words, so every text it renders (README and instruction, both modes) loses such a heading.
EMPTY_FAILURES_ANCHOR = ('    if mode == "standard":\n'
                         '        s = re.sub(r"(?m)^- Grading: the verifier replays your trajectory twice[^\\n]*\\n", "", s)\n')
EMPTY_FAILURES_RULE = ('    s = re.sub(r"(?m)^## Task failure conditions\\n\\s*\\n(?=## )", "", s)   '
                       '# no empty failure section (audit V6, 2026-10-17)\n')


# The onto tree's mk_tasks.py dates a task's Change Log entry "new task of <bench>" from its nop record, which a later
# release writes again: a base released before would read as new on the later date (V2 release, 2026-10-17). A record
# with "released" keeps that date, and a record's "changes" ({date, mode: privileged | standard | both, text}) are the
# later releases' entries of that task, after the fixes. Laid over the generator as the rules above are.
HISTORY_DATE = ('    date = rec["nop"]["date"]\n    if rec.get("template"):\n',
                '    date = rec.get("released") or rec["nop"]["date"]   # a base released before keeps its date (V2)\n'
                '    if rec.get("template"):\n')
HISTORY_LOG = ("    log += fix_changes(base, mode)\n",
               "    log += fix_changes(base, mode)\n"
               "    log += [f\"- {c['date']}: {c['text']}\" for c in rec.get(\"changes\", ()) if c[\"mode\"] in (mode, \"both\")]   # later releases (V2)\n")


def history_rule(gen: pathlib.Path) -> bool:
    """Gives the onto tree's generator the released-date and later-change entries unless it has them; True when added."""
    s = gen.read_text()
    if HISTORY_DATE[1] in s:
        return False
    if s.count(HISTORY_DATE[0]) != 1 or s.count(HISTORY_LOG[0]) != 1:
        raise SystemExit(f"{gen}: the Change Log anchors changed (mujoco_changes' date, rewrite_readme's fix_changes)")
    gen.write_text(s.replace(HISTORY_DATE[0], HISTORY_DATE[1]).replace(HISTORY_LOG[0], HISTORY_LOG[1]))
    return True


def empty_failures_rule(gen: pathlib.Path) -> bool:
    """Gives the onto tree's generator the empty-failure-section rule unless it has it; True when it was added."""
    s = gen.read_text()
    if EMPTY_FAILURES_RULE in s:
        return False
    if s.count(EMPTY_FAILURES_ANCHOR) != 1:
        raise SystemExit(f"{gen}: the empty-failure-section rule's anchor changed (audit_words' standard Grading line)")
    gen.write_text(s.replace(EMPTY_FAILURES_ANCHOR, EMPTY_FAILURES_RULE + EMPTY_FAILURES_ANCHOR))
    return True


def standard_expect(a, base: str) -> str | None:
    """The end the Harbor standard oracle must reach: a speed-rule standard reference's own (its oracles/<base>/
    standard.npz differs from privileged.npz in the release files), else None (the privileged reference's end)."""
    d = a.out / "scripts" / a.bench / "oracles" / base
    if not (d / "standard.npz").is_file() or (d / "standard.npz").read_bytes() == (d / "privileged.npz").read_bytes():
        return None
    return speed_rule_record(a.repo, base)["end_state"]


def make_instances(a, bases: list[dict], noinst: str, mujoco_env) -> dict:
    """make-instance for every base inside the image without instances, in parallel, then each base's appearance seeds
    (derive_seeds); {base or <base>/seeds/<S>: wall seconds}."""
    inst, mk = a.work / "instances", a.work / "mk"
    shutil.rmtree(inst, ignore_errors=True)
    for d in (inst, mk, a.work / "cache"):
        d.mkdir(parents=True, exist_ok=True)
    for b in bases:                    # made elsewhere (--instances / the manifest): copied as they are
        if b["new"] in a.premade:
            shutil.copytree(a.premade[b["new"]], inst / b["new"], symlinks=False)
    made = [b for b in bases if b["new"] not in a.premade]
    (mk / "make_one.py").write_text(MAKE_ONE)
    env = [*(f"{k}={v}" for k, v in mujoco_env), "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
           "RCB_TORCH_THREADS=1", "HOME=/tmp", "XDG_CACHE_HOME=/work/cache"]
    tooling = a.repo / "scripts/robotangle/shared"

    def one(b: dict) -> tuple[str, float]:
        name = tag = b["new"]
        job = {k: b[k] for k in ("variation", "released_simulator_keys", "simulator_overrides", "scene_seed",
                                 "gripper_space", "render_host", "frozen_assets", "released_task_parameters",
                                 "grasp_weld_links")}
        job.update(task=b["task"], released=b["old"] or b["template"], out=f"/work/instances/{name}/instance.json",
                   overrides_file=f"/work/mk/{tag}.overrides.json",
                   task_parameters_file=f"/work/mk/{tag}.task_parameters.json")
        if b["render_host"]:     # the released task's binding goes with the released instance into the render host
            shutil.copyfile(a.repo / "tasks" / f"robotangle-{b['old']}-privileged/environment/runtime_binding.json",
                            mk / f"{b['new']}.released_binding.json")
            job["released_binding"] = f"/work/mk/{b['new']}.released_binding.json"
        (inst / name).mkdir(parents=True, exist_ok=True)
        (mk / f"{tag}.json").write_text(json.dumps(job, indent=1) + "\n")
        t0 = datetime.datetime.now()
        gpu = (b["simulator_overrides"].get("mujoco_profile") or "") in a.gpu_profiles   # a GPU profile: its make needs one
        dev = ["--gpus", f"device={a.gpu}"] if gpu else ["-e", "CUDA_VISIBLE_DEVICES="]
        run(["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}", *dev,
             *(x for e in env for x in ("-e", e)), "-v", f"{tooling}:/tooling:ro", "-v", f"{a.work}:/work", noinst,
             "python", "/work/mk/make_one.py", f"/work/mk/{tag}.json"], log=a.work / "logs" / f"make_{tag}.log")
        return name, (datetime.datetime.now() - t0).total_seconds()

    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        walls = dict(pool.map(one, made))
    return {**walls, **derive_seeds(a, bases, noinst)} if a.appearance_seeds else walls


def derive_seeds(a, bases: list[dict], image: str) -> dict:
    """--appearance-seeds: each base's seeds derived from its instance in <work>/instances/<base>/ (made in this run or
    premade) in the image without instances, as make-instance runs (derive_seed_instance.py: instance_assets/ byte for
    byte, appearance(cls, S) task parameters, the build's fields recomputed; it stops unless the birth is the base's but
    for task.instance_id). A premade base's own seeds/<S>/ are kept. A GPU-profile instance derives on --gpu.
    {<base>/seeds/<S>: wall seconds of the base's derivation per seed}."""
    dsi = module(pathlib.Path(__file__).resolve().parent / "derive_seed_instance.py")

    def one(b: dict) -> dict:
        d = a.work / "instances" / b["new"]
        todo = [s for s in a.appearance_seeds if not (d / "seeds" / str(s) / "instance.json").is_file()]
        if not todo:
            return {}
        gpu = (json.loads((d / "instance.json").read_text())["simulator"].get("mujoco_profile") or "") in a.gpu_profiles
        wall = dsi.docker_derive(image, a.repo / "scripts/robotangle/shared", d, d, todo,
                                 a.work / "logs" / f"derive_{b['new']}.log", gpu=a.gpu if gpu else None)
        return {f"{b['new']}/seeds/{s}": wall / len(todo) for s in todo}
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        return {k: v for r in pool.map(one, bases) for k, v in r.items()}


def write_seed_files(a, rb: pathlib.Path, new: str, d: pathlib.Path, ref: bytes | None, lock: dict) -> dict:
    """One base's appearance-seed release files (--appearance-seeds; without it only the clean-up, a no-op on a tree
    without them): per seed S, the lock of instances/<new>/seeds/<S>/ (packages/<new>-privileged/
    runtime_binding.s<S>.upstream.json: seed 0's lock with that instance's canonical sha256, asset files and creation
    source) and, where the base has a reference, the reference re-stamped with the seed's birth (oracles/<new>/
    privileged.s<S>.npz; the standard oracle streams only the rows, which every seed shares). A previous release's seed
    files of the base go first, its packages' tests/seeds/ too (seed_goal_pictures writes them). Returns the base's
    assets.json `seeds` ({} without seeds)."""
    for p in [*(rb / "packages" / f"{new}-privileged").glob("runtime_binding.s*.upstream.json"),
              *(rb / "oracles" / new).glob("privileged.s*.npz")]:
        p.unlink()
    for mode in ("privileged", "standard"):
        shutil.rmtree(rb / "packages" / f"{new}-{mode}" / "tests" / "seeds", ignore_errors=True)
    pins = {}
    for s in a.appearance_seeds:
        sd = d / "seeds" / str(s)
        si = json.loads((sd / "instance.json").read_text())
        files = {p.relative_to(sd).as_posix(): sha(p.read_bytes()) for p in sorted((sd / "instance_assets").rglob("*"))
                 if p.is_file()}
        (rb / "packages" / f"{new}-privileged" / f"runtime_binding.s{s}.upstream.json").write_text(json.dumps(
            dict(lock, instance_sha256=canonical(si), asset_files=files, creation_source=si["topogym_source"]), indent=2) + "\n")
        if ref is not None:
            (rb / "oracles" / new / f"privileged.s{s}.npz").write_bytes(restamp(ref, si["state_sha"]))
        pins[str(s)] = {"instance_sha256": canonical(si)}
    return pins


# The goal pictures of a task family whose instruction shows some (packages' environment/goal/: blanket_fold's
# target_cam_*.png, knot_untying's region tiers' region_cam_*.png) are rendered for an appearance seed by the render
# tree's scripts/<bench>/goal_images/<renderer> (in the release image, one GPU); each writes the pictures and a record
# <kind>_images.json that names the birth it rendered (state_sha).
GOAL_RENDERERS = {"blanket_fold": "fold_target_images.py", "knot_untying": "region_images.py"}


def goal_command(a, base: str, seed: int, renderer: str, image: str, mujoco_env) -> list[str]:
    """The render of one seed's goal pictures: in the release image on --gpu, from a writable copy of the seed's instance
    (instances mounted read-only), into <work>/goal/<base>/seeds/<seed>/."""
    env = [*(f"{k}={v}" for k, v in mujoco_env), "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
           "RCB_TORCH_THREADS=1", "HOME=/tmp", "XDG_CACHE_HOME=/work/cache"]
    return ["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}", "--gpus", f"device={a.gpu}",
            *(x for e in env for x in ("-e", e)), "-v", f"{a.repo / 'scripts/robotangle/shared'}:/tooling:ro",
            "-v", f"{a.out / 'scripts' / a.bench / 'goal_images'}:/goal_images:ro",
            "-v", f"{a.work / 'instances'}:/instances:ro", "-v", f"{a.work}:/work", image,
            "sh", "-c", 'cp -a "$1" /tmp/instance && exec python "/goal_images/$2" /tmp/instance/instance.json "$3" --tooling /tooling',
            "render", f"/instances/{base}/seeds/{seed}", renderer, f"/work/goal/{base}/seeds/{seed}"]


def seed_goal_pictures(a, bases: list[dict], image: str, mujoco_env) -> None:
    """--appearance-seeds: each seed's goal pictures for the bases whose packages show some (environment/goal/; the user,
    2026-10-20: rendered in the seed's looks, committed in both task dirs as tests/seeds/<S>/goal/, which
    scripts/task_seed.py materialize copies over environment/goal/). They come from --goal-pictures <dir>/<base>/seeds/<S>/,
    or are rendered in this run into <work>/goal/<base>/seeds/<S>/ (goal_command; a picture already there is used again).
    The run stops when a picture is missing or its renderer's record names another birth than the seed's. Written to
    packages/<base>-<mode>/tests/seeds/<S>/goal/ and pinned in assets.json, instances[<base>].seeds[<S>].goal."""
    rb = a.out / "scripts" / a.bench
    assets = json.loads((rb / "assets.json").read_text())
    for b in bases:
        new = b["new"]
        pkgs = [rb / "packages" / f"{new}-{m}" for m in ("privileged", "standard")]
        names = [sorted(p.name for p in (q / "environment/goal").glob("*") if p.is_file()) for q in pkgs]
        if not any(names):
            continue
        if names[0] != names[1]:
            raise SystemExit(f"{new}: the twins show other goal pictures ({names[0]} / {names[1]})")
        renderer = GOAL_RENDERERS.get(b["task"])
        script = rb / "goal_images" / (renderer or "")
        for s in a.appearance_seeds:
            birth = json.loads((a.work / "instances" / new / "seeds" / str(s) / "instance.json").read_text())["state_sha"]
            src = (a.goal_pictures or a.work / "goal") / new / "seeds" / str(s)
            cmd = goal_command(a, new, s, renderer or "<renderer>", image, mujoco_env)
            if not a.goal_pictures and not all((src / n).is_file() for n in names[0]):
                if not renderer or not script.is_file():
                    raise SystemExit(f"{new} seed {s}: its goal pictures {names[0]} need a renderer: no "
                                     f"{script if renderer else 'GOAL_RENDERERS entry for ' + b['task']} (or pass --goal-pictures)")
                shutil.rmtree(src, ignore_errors=True)
                run(cmd, log=a.work / "logs" / f"goal_{new}.seed{s}.log")
            missing = [n for n in names[0] if not (src / n).is_file()]
            records = [json.loads(p.read_text()) for p in sorted(src.glob("*_images.json"))] if src.is_dir() else []
            if missing or len(records) != 1 or records[0].get("state_sha") != birth:
                raise SystemExit(f"{new} seed {s}: goal pictures in {src}: " + (f"missing {missing}" if missing else
                                 f"{len(records)} renderer records (*_images.json), want one naming the birth {birth}"
                                 + (f", it names {records[0].get('state_sha')}" if len(records) == 1 else ""))
                                 + f"; render them with: {' '.join(cmd)}")
            for q in pkgs:
                (q / "tests/seeds" / str(s) / "goal").mkdir(parents=True, exist_ok=True)
                for n in names[0]:
                    shutil.copyfile(src / n, q / "tests/seeds" / str(s) / "goal" / n)
            assets["instances"][new]["seeds"][str(s)]["goal"] = {n: sha((src / n).read_bytes()) for n in names[0]}
    (rb / "assets.json").write_text(json.dumps(assets, indent=1, ensure_ascii=False) + "\n")


def write_release(a, bases: list[dict], head: str, runtime: dict, uipc_builds: dict, noinst: dict, facts: dict) -> None:
    """VERSION, packages/, oracles/, release.json and assets.json for the made instances, in the render tree (a.out: the
    --onto worktree, else <head>'s); the old task dirs removed. Task texts of a changed or new design come from <head>
    (package_text)."""
    rb = a.out / "scripts" / a.bench                        # this benchmark's release files
    released = a.out / "scripts/robotangle/packages"         # RoboTangle's released texts (a new benchmark reads them)
    new_bench = a.bench != "robotangle"
    if new_bench and not released.is_dir():   # main without the Genesis RoboTangle tasks (deleted 2026-10-13, #187): the
        released = a.repo / "scripts/robotangle/packages"   # head keeps their released texts (a new or newly ported base's template)
    (a.out / "images" / a.bench / "VERSION").write_text(a.version + "\n")
    rel = json.loads((rb / "release.json").read_text())
    assets = json.loads((rb / "assets.json").read_text())
    tooling = sha((rb / "shared/rcb_topogym.py").read_bytes())
    ev_file = a.repo / "scripts/robotangle/release_evidence.json"
    shipped = json.loads(ev_file.read_text()) if ev_file.is_file() else {}
    src8, tool8 = runtime["source"]["sha256"][:8], tooling[:8]
    host = f"{a.host_label}, {cpu_model()}, no GPU"
    records = {}

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
    # a new base's template texts, read before a renamed base of the template's family removes its released packages
    template_texts = {(b["new"], m): texts_of((released if new_bench else rb / "packages") / f"{b['template']}-{m}",
                                              b["template"], b["new"])
                      for b in bases if b["template"] for m in ("privileged", "standard")}
    # a base the benchmark already releases (its packages/<new>-* in the render tree, e.g. main's 0.1.0 bases) keeps
    # those texts: they carry what main changed since (the 0.1.0 rename's doc paths, later text fixes), which the
    # head's RoboTangle packages and package_text do not; a text change of a released base is made there. Only its
    # instance lock is written again.
    onto_texts = {b["new"] for b in bases if new_bench and all((rb / "packages" / f"{b['new']}-{m}").is_dir()
                                                               for m in ("privileged", "standard"))}
    for b in bases:
        old, new, d = b["old"], b["new"], a.work / "instances" / b["new"]
        inst = json.loads((d / "instance.json").read_text())
        for mode in ("privileged", "standard") if new not in onto_texts else ():
            dst = rb / "packages" / f"{new}-{mode}"
            if b["template"]:             # a new base: its template's released texts, under the new name
                texts, key = dict(template_texts[new, mode]), new
                if not texts or not b.get("package_text"):
                    raise SystemExit(f"{new}: needs its template's packages/{b['template']}-{mode} and its own texts "
                                     "(package_text)")
            else:                         # the released base's texts, under the new name
                src = (released if new_bench else rb / "packages") / f"{old}-{mode}"
                if not src.is_dir():
                    src = dst
                texts, key = texts_of(src, old, new), old
            if b.get("package_text"):     # the base's own task texts over those (its design changed, or is new)
                own = a.repo / b["package_text"] / f"{key}-{mode}"
                if not own.is_dir():
                    raise SystemExit(f"{new}: package_text has no {own}")
                texts.update(texts_of(own, old or b["template"], new))
            if b["template"]:             # a template of another family: its grader and sidecar start the new task
                tf = tomllib.loads(template_texts[new, mode][pathlib.Path("task.toml")][0])["metadata"]["family"]
                if tf != b["task"]:
                    for rel_path, line in ((pathlib.Path("tests/test.sh"), "--family {} "),
                                           (pathlib.Path("environment/docker-compose.yaml"), "RCB_TASK: {}\n")):
                        if rel_path in texts:
                            s, st_mode = texts[rel_path]
                            texts[rel_path] = (s.replace(line.format(tf), line.format(b["task"])), st_mode)
                    left = sorted(str(p) for p, (s, _) in texts.items()
                                  if isinstance(s, str) and tf in s.replace(b["task"], ""))
                    if left:
                        raise SystemExit(f"{new}-{mode}: its template's family {tf} is still named in {left} (a new base "
                                         f"of {b['task']}: its package_text replaces every task line of the template)")
            shutil.rmtree(dst, ignore_errors=True)
            for rel_path, (s, st_mode) in texts.items():
                (dst / rel_path).parent.mkdir(parents=True, exist_ok=True)
                (dst / rel_path).write_bytes(s) if isinstance(s, bytes) else (dst / rel_path).write_text(s)
                os.chmod(dst / rel_path, st_mode & 0o777)
            if old and old != new and not new_bench:
                shutil.rmtree(rb / "packages" / f"{old}-{mode}", ignore_errors=True)
        assets_files = {p.relative_to(d).as_posix(): sha(p.read_bytes()) for p in sorted((d / "instance_assets").rglob("*"))
                        if p.is_file()}
        lock = dict(schema_version=1, instance_sha256=canonical(inst), asset_files=assets_files, tooling_sha256=tooling,
                    creation_source=inst["topogym_source"], runtime=runtime, image=noinst)
        (rb / "packages" / f"{new}-privileged/runtime_binding.upstream.json").write_text(json.dumps(lock, indent=2) + "\n")
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
            ev = dict(deterministic=1, release=f"{a.version} (dev/robotangle_mujoco {head[:9]})", source=src8, success=1,
                      tooling=tool8)
        if new_bench:   # main's record form (the 0.1.0 rename step): the image and the date, never a host or a job name.
            # A new or changed reference's record is the Harbor trial that passed it (scripts/robotangle/
            # release_evidence.json, its provenance not copied); without one it is not_run: no score is claimed
            given = (shipped.get("bases") or {}).get(new) or {}
            evidence = {} if no_ref else {
                m: dict({k: v for k, v in given[m].items() if k != "provenance"}, trajectory_sha256=sha(per_mode[m]))
                if m in given else dict(status="not_run", trajectory_sha256=sha(per_mode[m]))
                for m in ("privileged", "standard")}
        else:
            evidence = {} if no_ref else {m: dict(attempt=f"{a.host_label}:{a.work}/harbor/jobs/{a.bench}-{new}-{m}__oracle",
                                                  **ev, trajectory_sha256=sha(per_mode[m])) for m in ("privileged", "standard")}
        if not no_ref and per_mode["standard"] != per_mode["privileged"]:   # a standard reference within the speed rule
            rec = speed_rule_record(a.repo, new)
            evidence["standard"].update(speed_rule=rec["words"], end_state=rec["end_state"])
        if new_bench and not no_ref:    # a mode whose reference the render tree already releases keeps its record (its
            released_ev = (rel["bases"].get(new) or {}).get("evidence") or {}   # status and where it was graded): only
            for m in ("privileged", "standard"):                                # a new or changed reference is this run's
                if (released_ev.get(m) or {}).get("trajectory_sha256") == evidence[m]["trajectory_sha256"]:
                    evidence[m] = released_ev[m]
        records[old or new] = (new, dict(
            creation_source=inst["topogym_source"]["sha256"][:8],
            evidence=evidence,
            **(dict(oracle="none") if no_ref else {}),
            # a premade instance (--instances) was not made in this image: the 0.1.0 rename's wording (audit S7)
            family=b["family"], instance_package=(
                f"MuJoCo instance made before this image was built (TopoGym source {inst['topogym_source']['sha256'][:8]}; "
                f"its bytes are bound into the release image eai-{a.bench}:{a.version}, "
                "scripts/robotangle/release_rebuild.py --instances)" if new in a.premade else
                f"MuJoCo instance made in the release image (eai-{a.bench}:{a.version}, "
                f"scripts/robotangle/release_rebuild.py; make-instance {b['variation']}, scene seed {b['scene_seed']})"),
            tier=b["tier"], physics_engine="mujoco", release_commit=head,
            # a released base keeps its nop record; a new one takes release_evidence.json's, else not_run
            nop=(((rel["bases"].get(new) or {}).get("nop") or {k: v for k, v in (shipped.get("bases") or {}).get(new, {})
                                                                 .get("nop", {"status": "not_run"}).items() if k != "provenance"})
                 if new_bench else   # main's record form (the 0.1.0 rename step), no host
                 dict(run=f"{a.version}, harbor run -a nop with rendering off (scripts/robotangle/harbor_check.py)",
                      date=datetime.date.today().isoformat(), host=host)),
            **(dict(template=b["template"], variation=b["variation"]) if b["template"] else
               dict(ported_from=old) if new_bench else
               dict(replaces=old if old != new else rel["bases"][new]["replaces"])),
            **(dict(design=b["design"]) if b["design"] else {}),
            **{k: facts[new][k] for k in ("grasp_rules", "physics_dt_s", "grasp", "grasp_regions", "chain")
               if facts[new].get(k)},
            **(dict(gpu_physics=True) if facts[new]["gpu_physics"] else {}),
            **(dict(step_s=a.step_s[new]) if new in a.step_s else {}),
            **(later_release(a, rb, rel, new, evidence, no_ref) if new_bench else {})))
        for prefix in {f"instances/{n}/" for n in (old, new) if n and (n == new or not new_bench)}:
            assets["files"] = {k: v for k, v in assets["files"].items() if not k.startswith(prefix)}
        assets["files"].update({f"instances/{new}/{k}": v for k, v in files_of(d).items()})
        if old and not new_bench:
            assets["instances"].pop(old, None)
        assets["instances"][new] = {"instance_sha256": canonical(inst)}
        seeds = write_seed_files(a, rb, new, d, None if no_ref else per_mode["privileged"], lock)
        if seeds:                         # the appearance seeds' instances (their files are under instances/<new>/seeds/)
            assets["instances"][new]["seeds"] = seeds
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
    if new_bench:                         # the benchmark's own release: the dev/robotangle_mujoco commit it is built from
        rel["release_commit"] = rel["template_commit"] = head
        (a.out / "images" / a.bench / "Dockerfile").write_text(
            main_dockerfile((a.repo / "images/robotangle/Dockerfile.mujoco").read_text(), a.bench))
    assert {n: b["uipc_build_sha256"] for n, b in rel["uipc_builds"].items()} == uipc_builds, "uipc builds changed"
    (rb / "release.json").write_text(json.dumps(rel, indent=1, ensure_ascii=False) + "\n")
    assets["files"] = dict(sorted(assets["files"].items()))
    assets["instances"] = dict(sorted(assets["instances"].items()))
    if a.hf_revision:
        assets["revision"] = a.hf_revision
    (rb / "assets.json").write_text(json.dumps(assets, indent=1, ensure_ascii=False) + "\n")


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


DOCKERFILE_HEAD = """# {bench} base image: generated by scripts/robotangle/release_rebuild.py on dev/robotangle_mujoco from that branch's
# images/robotangle/Dockerfile.mujoco at the release commit (scripts/{bench}/release.json release_commit); do not edit.
# Build context: images/{bench}/context/, prepared by scripts/{bench}/fetch_image_context.sh (BUILD_CONTEXT; the release
# commit's simulator source, robot assets and image helpers, and the frozen instances from the HF dataset pinned in
# scripts/{bench}/assets.json); then scripts/build_images.sh. The only change from Dockerfile.mujoco: the instances
# are the context's instances/ directory instead of a named build context.
"""


def main_dockerfile(head_dockerfile: str, bench: str) -> str:
    """images/<bench>/Dockerfile on main: the head's Dockerfile.mujoco with the instances in the build context."""
    a = "RUN --mount=type=bind,from=instances,target=/tmp/mj-instances"
    assert head_dockerfile.count(a) == 1, "Dockerfile.mujoco: the instances mount changed"
    return DOCKERFILE_HEAD.format(bench=bench) + head_dockerfile.replace(
        a, "RUN --mount=type=bind,source=instances,target=/tmp/mj-instances")


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
    ap.add_argument("--appearance-seeds")
    ap.add_argument("--goal-pictures", type=pathlib.Path)
    ap.add_argument("--harbor", default="harbor")
    ap.add_argument("--host-label", default=socket.gethostname().split(".")[0])
    ap.add_argument("--replace-tag", action="store_true")
    ap.add_argument("--no-harbor", action="store_true")
    ap.add_argument("--harbor-only", action="store_true", help="only step 5 and the report, on --work's previous run")
    a = ap.parse_args(argv)
    m = json.loads(a.manifest.read_text()) if a.manifest else {}
    mdir = a.manifest.resolve().parent if a.manifest else pathlib.Path.cwd()
    for k, default in (("head", None), ("version", None), ("onto", None), ("bench", BENCH), ("families", None), ("only", None), ("hf_revision", None),
                       ("gpu", "0"), ("lanes", 3), ("jobs", 4), ("appearance_seeds", None)):
        if getattr(a, k) is None:
            v = m.get(k, default)
            setattr(a, k, ",".join(map(str, v)) if k in ("families", "only", "appearance_seeds") and isinstance(v, list) else v)
    a.appearance_seeds = sorted({int(x) for x in (a.appearance_seeds or "").split(",") if x.strip()})
    if any(x < 1 for x in a.appearance_seeds):
        ap.error("--appearance-seeds: seeds >= 1 (seed 0 is the base instance)")
    if not a.head or not a.version:
        ap.error("the head and --version are needed (on the command line or in the manifest)")
    if a.bench != "robotangle" and not a.onto and not a.harbor_only:
        ap.error(f"--bench {a.bench}: its files live on main (scripts/{a.bench}/): pass --onto <main-based commit>")
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
    if a.goal_pictures is None and m.get("goal_pictures"):
        a.goal_pictures = path(m["goal_pictures"])
    a.goal_pictures = a.goal_pictures.resolve() if a.goal_pictures else None
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
        return harbor_step(a, here, report, bases, report.get("gpu_bases", []), report["images"]["release"]["reference"],
                           module(here / "scripts/robotangle/harbor_check.py"))
    a.repo = a.work / "repo"
    image = f"ghcr.io/embodied-first-exam/{a.bench}:{a.version}"
    noinst_tag = f"{a.bench}:{a.version}-noinst"
    if image_id(image) and not a.replace_tag:
        raise SystemExit(f"{image} exists here; pick another --version or pass --replace-tag")
    (a.work / "logs").mkdir(parents=True, exist_ok=True)
    report = dict(head=head, version=a.version, work=str(a.work), started=datetime.datetime.now().isoformat(timespec="seconds"))

    # 1. the worktree of <head> and the image without instances
    if a.repo.exists():
        run(["git", "-C", str(here), "worktree", "remove", "--force", str(a.repo)])
    branch = f"release-rebuild/{head[:9]}-{a.work.name}"   # one per work dir: worktrees cannot share a branch
    run(["git", "-C", str(here), "worktree", "add", "-B", branch, str(a.repo), head])
    if (a.repo / "source/scripts/robotangle").is_dir():   # the RoboARC repository: the dev source lives under source/
        a.repo = a.repo / "source"
    spec = json.loads((a.repo / "scripts/robotangle/mujoco_release.json").read_text())
    # GPU profiles: the head's list and this checkout's (a head from before the list names none)
    a.gpu_profiles = set(spec.get("gpu_profiles", [])) | set(json.loads(
        (here / "scripts/robotangle/mujoco_release.json").read_text()).get("gpu_profiles", []))
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
        gen = a.out / "scripts" / a.bench / "mk_tasks.py"
        if not gen.is_file() or "def mujoco_lines" not in gen.read_text():
            raise SystemExit(f"--onto {a.onto}: no scripts/{a.bench}/mk_tasks.py that renders MuJoCo bases")
        for n in TOOLING:
            shutil.copy2(a.repo / "scripts/robotangle/shared" / n, a.out / "scripts" / a.bench / "shared" / n)
        report.update(onto=onto, speed_rule_wording_added=speed_rule_wording(gen),
                      empty_failures_rule_added=empty_failures_rule(gen), history_rule_added=history_rule(gen))
    for b in bases:
        if b["new"] not in a.no_oracle:
            reference(b, a.references, a.repo, a.ref_paths)        # every reference exists before anything is built
        if b["template"]:                             # and every new base's texts
            for mode in ("privileged", "standard"):
                need = [a.repo / "scripts/robotangle/packages" / f"{b['template']}-{mode}"] + \
                       ([a.repo / b["package_text"] / f"{b['new']}-{mode}"] if b["package_text"] else [])
                if not b["package_text"] or not all(p.is_dir() for p in need):
                    raise SystemExit(f"{b['new']}: needs its template's texts and its own (package_text): "
                                     f"{', '.join(str(p) for p in need if not p.is_dir()) or 'no package_text'}")
        if a.hf_revision and not (a.work / "instances" / b["new"] / "instance.json").is_file():
            raise SystemExit(f"--hf-revision uses the instances of a previous run in {a.work}: {b['new']} has none")
        src = a.work / "instances" / b["new"]
        lacking = [s for s in a.appearance_seeds if a.hf_revision and not (src / "seeds" / str(s) / "instance.json").is_file()]
        if lacking:                       # the uploaded instances, used again: their appearance seeds come with them
            raise SystemExit(f"{b['new']}: --appearance-seeds {','.join(map(str, lacking))}: {src} has no seeds/<S>/instance.json")
    # harbor_check.py is this checkout's (the runner), mk_tasks.py the render tree's; MUJOCO_ENV is the same in both
    gen = f"scripts/{a.bench if a.onto else 'robotangle'}/mk_tasks.py"     # the render tree's generator
    if a.appearance_seeds and "def seeds_of_base" not in (a.out / gen).read_text():
        raise SystemExit(f"--appearance-seeds: the render tree's {gen} does not render appearance seeds (seeds_of_base)")
    mk, hc = module(a.out / gen), module(here / "scripts/robotangle/harbor_check.py")
    empty = a.work / "empty"
    empty.mkdir(exist_ok=True)
    build = ["docker", "buildx", "build", "-f", "images/robotangle/Dockerfile.mujoco"]
    run([*build, "--build-context", f"instances={empty}", "-t", noinst_tag, "."], log=a.work / "logs/build_noinst.log",
        cwd=a.repo)
    noinst = dict(reference=noinst_tag, id=image_id(noinst_tag), repo_digests=[])

    def installed(ref: str) -> dict:
        out = run(["docker", "run", "--rm", "--network", "none", "-v",
                   f"{a.repo / 'scripts/robotangle/shared'}:/tooling:ro", ref, "python", "-c", hc.RUNTIME])
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
        seeds = {}
        for s in a.appearance_seeds:      # each appearance seed's instance: same make, appearance(cls, S) parameters
            si = json.loads((d / "seeds" / str(s) / "instance.json").read_text())
            assert si["simulator"]["physics_engine"] == "mujoco", f"{b['new']} seed {s}"
            if b["new"] not in a.premade:
                assert si["topogym_source"]["sha256"] == rt["runtime"]["source"]["sha256"], f"{b['new']} seed {s}: made on another source"
            seeds[str(s)] = dict(birth=si["state_sha"], instance_sha256=canonical(si),
                                 task_parameters=si["simulator"].get("task_parameters"),
                                 simulator_keys_differing=sorted(k for k in set(si["simulator"]) | set(inst["simulator"])
                                                                 if k != "task_parameters"
                                                                 and si["simulator"].get(k) != inst["simulator"].get(k)),
                                 make_s=round(walls[f"{b['new']}/seeds/{s}"], 1) if f"{b['new']}/seeds/{s}" in walls
                                 else "made elsewhere / uploaded")
        if seeds:
            instances[b["new"]]["seeds"] = seeds
    run([*build, "--build-context", f"instances={a.work / 'instances'}", "-t", image, "-t", f"{a.bench}:{a.version}",
         "."], log=a.work / "logs/build_release.log", cwd=a.repo)
    rel_rt = installed(image)
    assert rel_rt == rt, "the release image's runtime differs from the one the instances were made on"
    facts = json.loads(run(["docker", "run", "--rm", "--network", "none", "-e", "CUDA_VISIBLE_DEVICES=", "-v",
                            f"{a.repo / 'scripts/robotangle/shared'}:/tooling:ro", "-v", f"{a.repo / 'scripts'}:/scripts:ro",
                            "-v", f"{a.work}:/work", image, "python",
                            "-c", FACTS, json.dumps(sorted(a.gpu_profiles)), *(b["new"] for b in bases)]).strip().splitlines()[-1])
    gpu_bases = [b["new"] for b in bases if facts[b["new"]]["gpu_physics"]]
    report.update(facts=facts, gpu_bases=gpu_bases)
    runtime = dict(rt["runtime"])
    uipc = rt["uipc_builds"]
    report.update(images=dict(noinst=noinst, release=dict(reference=image, id=image_id(image))),
                  runtime=runtime, instances=instances)

    # 3. the release files, every task dir rendered; 4. the checks
    write_release(a, bases, head, runtime, uipc, noinst, facts)
    if a.appearance_seeds:               # the seeds' goal pictures (rendered in the release image); the report names
        seed_goal_pictures(a, bases, image, mk.MUJOCO_ENV)       # their re-stamped references and pictures
        rb = a.out / "scripts" / a.bench
        pins = json.loads((rb / "assets.json").read_text())
        for b in bases:
            for s, x in instances[b["new"]].get("seeds", {}).items():
                ref = rb / "oracles" / b["new"] / f"privileged.s{s}.npz"
                x.update(reference_sha256=sha(ref.read_bytes()) if ref.is_file() else None,
                         goal=pins["instances"][b["new"]]["seeds"][s].get("goal"))
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
                ours = [x for x in fails if "robotangle" in x or "dataset" in x or "tooling" in x or "Traceback" in x]
                checks[name] = dict(ok=not ours, output=ours[-5:] or [f"{len(fails)} FAIL lines, none about RoboTangle, "
                                                                     "the dataset manifest or the tooling (see the log)"])
                continue
            checks[name] = dict(ok=r.returncode == 0, output=(r.stdout + r.stderr).strip().splitlines()[-5:])
    report["checks"] = checks
    run(["git", "-C", str(a.out), "add", "-A"])
    made = [b["new"] for b in bases if b["new"] not in a.premade]
    pre = [b["new"] for b in bases if b["new"] in a.premade]
    msg = (f"RoboTangle release rebuild on {head[:9]} (image {a.version}): "
           + "; ".join(([f"MuJoCo instances made in the release image, {', '.join(made)}"] if made else [])
                       + ([f"premade MuJoCo instances (--instances) bound into it, {', '.join(pre)}"] if pre else []))
           + f"\n\nscripts/robotangle/release_rebuild.py; report: {a.work}/REPORT.md\n")
    run(["git", "-C", str(a.out), "commit", "-q", "-m", msg])
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
            r = subprocess.run([sys.executable, str(here / "scripts/robotangle/harbor_check.py"), "--tasks", str(a.out / "tasks"),
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
    out = [f"# RoboTangle release rebuild on {r['head'][:9]}: {'PASS' if r['ok'] else 'FAIL'}", "",
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
    seeded = {n: i["seeds"] for n, i in r["instances"].items() if i.get("seeds")}
    if seeded:                            # --appearance-seeds: instances/<base>/seeds/<S>/
        out += ["", "## Appearance seeds (instances/<base>/seeds/<S>/: derived from the base's instance, appearance(cls, S) task parameters)", "",
                "| base | seed | birth (state_sha) | instance sha256 | task parameters | simulator keys differing | derive s | "
                "reference re-stamped (oracles/<base>/privileged.s<S>.npz) | goal pictures |",
                "|---|---|---|---|---|---|---|---|---|"]
        out += [f"| {n} | {s} | `{x['birth']}` | `{x['instance_sha256'][:16]}` | `{json.dumps(x['task_parameters'])[:120]}` | "
                f"{', '.join(x['simulator_keys_differing']) or 'none'} | {x['make_s']} | "
                f"{'`' + x['reference_sha256'][:16] + '`' if x.get('reference_sha256') else 'none'} | "
                f"{', '.join(f'{k} `{v[:12]}`' for k, v in (x.get('goal') or {}).items()) or 'none'} |"
                for n, ss in seeded.items() for s, x in ss.items()]
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
    else:
        out += ["- HF upload of the instances (the user's OK first; upload only, nothing deleted). NOT UPLOADED: "
                "assets.json's new pins name files that are not in the dataset at its revision.",
                f"  `hf upload eai-v2-internal/robotangle-assets {a.work}/instances instances --repo-type dataset "
                f"--commit-message \"RoboTangle MuJoCo instances made in eai-{a.bench}:{a.version} ({r['head'][:9]})\"`",
                f"  then run this again with `--work {a.work} --hf-revision <the new dataset commit>`, which uses these "
                "instances again (and runs Harbor again on the final bytes)."]
    out += ["- Image push, and the push of the commit: the user's OK."]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
