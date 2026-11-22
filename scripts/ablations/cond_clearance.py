"""E4 of the ablation experiments: clearance-relaxed variants of rigid puzzles (tokens clr2x, clr4x; privileged and
standard), made by scripts/ablations/make_variants.py through `make(src, dst, factor)`.

What a variant changes. The one geometric parameter that sets a puzzle's clearance (RELAXATIONS: the bed bolt's blind
bore round the bolt's shank, the lock plates' bore round the key's shaft, the cast coil's design gap between the two
assembled pieces) is relaxed so that the clearance becomes `factor` (2 or 4) times the original; nothing else is edited:
the robot, the start state of every part, the grader and its thresholds, the time limit, and the texts except the
dimensions they state (the lock texts state both radii and what they allow). One exception, the cast coil: its goal's
assembled-pose tolerance (1 mm, set below the 1.5 mm gap) scales with the gap, in the grader's line, the sentence that
states it and the privileged excerpt that quotes it (_coil says why). Only relaxing: the bore grows, the coil's cells
shrink, so every relaxed part is a subset of the original solid and the original reference motion still fits. Every
variant is certified by replaying the ORIGINAL reference solution on it (the oracle runs of bash-runners/ablations/), and
scripts/ablations/check_clr{2,4}x.py measures the clearance in its frozen scene.

How. The geometry of a RoboARC puzzle is built by the simulator source in the release image (/opt/topogym) and frozen
into the base's instance (instance.json + instance_assets/: the MuJoCo scene), which the image carries and every task
stages by name (/opt/robotangle/instances/<base>, checked against the task's runtime_binding.json); the camera images
are drawn from the same generated assets. A variant therefore needs the relaxed source and an instance frozen from it:
  1. the parameter's line in the family's source file is edited (Relaxation.edits: old -> new[factor]; the image's file
     must have Edit.sha256), and the patched tree's SOURCE.json is computed by the image's own rule;
  2. the instance is made from the patched source inside the pinned release image, CPU only and without network, as the
     release pipeline makes a base's instance (scripts/ablations/clearance_freeze.py: `rcb_topogym.py make-instance`,
     scene seed 0, the base's variation and gripper space, the MuJoCo tasks' SIMD masking);
  3. the task directory gets both under `ablation_e4/` in every build context that stages the instance (privileged:
     environment/ and tests/; standard: environment/sim/), and the containers use them in place of the image's: the
     privileged Dockerfiles copy them over /opt/topogym and /opt/robotangle/instances/<base> before stage_instance.py
     runs, the standard sim service mounts them there read-only. runtime_binding.json names the new instance, assets and
     source (creation and runtime), so every Session checks that it runs the relaxed source on the relaxed scene;
  4. seeds: a variant has seed 0 only (`[metadata.seeds]` allowed and oracle [0]; the appearance seeds' bindings and
     re-stamped references are removed), since only seed 0's instance is relaxed. The privileged reference is re-stamped with the
     variant's birth state hash when the relaxation changes it (tools/release/restamp.py: only instance_sha.npy changes);
     the README's title, twin and seed row and the limits files' task name are what scripts/task_conventions.py writes.
No new image: the variants run on the pinned images (images/pins.json). Making a variant needs docker and the pinned
`roboarc` image (RCB_ABLATION_IMAGE names another reference to the same image id, e.g. a host-local registry's); the
freeze reproduces the released instances byte for byte from the unedited source, and scripts/ablations/digests/
clr{2,4}x.tsv lock every variant's DIGEST.

Python >= 3.11, standard library only (the image does the rest)."""
from __future__ import annotations

import atexit
import dataclasses
import functools
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
TOOLING = REPO / "scripts" / "roboarc" / "shared"
E4 = "ablation_e4"                 # the directory a variant's relaxed source and instance live in, per build context
FACTORS = (2, 4)
# a measured clearance must be the intended one within this (mm): the frozen meshes are snap-rounded to 1e-7 m grids
TOLERANCE_MM = 1e-3
# the MuJoCo tasks' numpy / OpenBLAS SIMD masking (scripts/roboarc/mk_tasks.py MUJOCO_ENV): every CPU class freezes the
# same bytes
MUJOCO_ENV = (("NPY_DISABLE_CPU_FEATURES", "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR"),
              ("OPENBLAS_CORETYPE", "Haswell"))


# ---------------------------------------------------------------- measuring a clearance in a frozen scene (scene.xml)
def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split()]


def _obj_vertices(path: pathlib.Path) -> list[tuple[float, ...]]:
    return [tuple(map(float, line.split()[1:4])) for line in path.read_text().splitlines() if line.startswith("v ")]


def measure_bore(scene: pathlib.Path) -> dict:
    """bed_bolt: the blind bore's radius (the inner faces of its 16 wall boxes, `bed_frame::bore_<i>`, round the bore
    axis: their mean centre) minus the bolt shank's radius (the prism mesh of `bolt::bolt_shank`, about its own axis)."""
    root = ET.parse(scene).getroot()
    walls = [g for g in root.iter("geom") if re.fullmatch(r"bed_frame::bore_\d+", g.get("name") or "")]
    if len(walls) != 16:
        raise SystemExit(f"{scene}: {len(walls)} bore walls, not 16")
    pos = [_floats(g.get("pos")) for g in walls]
    half = {_floats(g.get("size"))[1] for g in walls}
    if len(half) != 1:
        raise SystemExit(f"{scene}: bore walls of unequal thickness")
    c = [sum(p[i] for p in pos) / len(pos) for i in range(3)]
    radial = [math.hypot(p[1] - c[1], p[2] - c[2]) for p in pos]
    if max(radial) - min(radial) > 1e-9 or max(abs(p[0] - c[0]) for p in pos) > 1e-9:
        raise SystemExit(f"{scene}: the bore walls are not a ring about one axis")
    bore = sum(radial) / len(radial) - half.pop()
    shank = [g for g in root.iter("geom") if g.get("name") == "bolt::bolt_shank"]
    mesh = [m for m in root.iter("mesh") if shank and m.get("name") == shank[0].get("mesh")]
    if len(mesh) != 1:
        raise SystemExit(f"{scene}: no single bolt shank mesh")
    v = _obj_vertices(scene.parent / mesh[0].get("file"))
    r = [math.hypot(x, y) for x, y, _ in v]
    if max(r) - min(r) > 1e-9:
        raise SystemExit(f"{scene}: the shank prism is not round about its axis")
    return dict(clearance_mm=(bore - max(r)) * 1e3, bore_radius_mm=bore * 1e3, shank_radius_mm=max(r) * 1e3)


def _mesh_vertices(scene: pathlib.Path, root, name: str) -> list[tuple[float, ...]]:
    m = [x for x in root.iter("mesh") if x.get("name") == name]
    if len(m) != 1:
        raise SystemExit(f"{scene}: no single mesh {name}")
    if m[0].get("file"):
        return _obj_vertices(scene.parent / m[0].get("file"))
    f = _floats(m[0].get("vertex"))
    return [tuple(f[i:i + 3]) for i in range(0, len(f), 3)]


def measure_lock_bore(scene: pathlib.Path) -> dict:
    """lock_picking: the plates' bore radius (the smallest distance from the lock axis, each plate frame's x axis, of the
    convex pieces of every plate: the rotor discs `lock_disc_<i>::disc` and the END plate `lock_goal::disc`) minus the
    key shaft's radius (the key piece that runs along the key's x axis through its own axis: the largest distance of its
    vertices from that axis)."""
    root = ET.parse(scene).getroot()
    plates = [b for b in root.iter("body") if re.fullmatch(r"lock_disc_\d+::disc|lock_goal::disc", b.get("name") or "")]
    if len(plates) < 2:
        raise SystemExit(f"{scene}: {len(plates)} lock plates")
    bores = []
    for b in plates:
        v = [p for g in b.findall("geom") if g.get("type") == "mesh" for p in _mesh_vertices(scene, root, g.get("mesh"))]
        bores.append(min(math.hypot(y, z) for _, y, z in v))
    key = [b for b in root.iter("body") if b.get("name") == "lock_key::key"]
    if len(key) != 1:
        raise SystemExit(f"{scene}: no single key body")
    shaft = None
    for g in key[0].findall("geom"):
        v = _mesh_vertices(scene, root, g.get("mesh"))
        r = [math.hypot(y, z) for _, y, z in v]
        length = max(p[0] for p in v) - min(p[0] for p in v)
        if min(r) < 1e-9 and (shaft is None or length > shaft[0]):
            shaft = (length, max(r))
    if shaft is None:
        raise SystemExit(f"{scene}: no key shaft on the key axis")
    if max(bores) - min(bores) > 1e-6:
        raise SystemExit(f"{scene}: the plates' bores differ ({min(bores)} .. {max(bores)} m)")
    return dict(clearance_mm=(min(bores) - shaft[1]) * 1e3, bore_radius_mm=min(bores) * 1e3, shaft_radius_mm=shaft[1] * 1e3,
                plates=len(plates))


def measure_coil_gap(scene: pathlib.Path) -> dict:
    """cast_coil v2: the gap the two assembled pieces keep between them (cast_coil_official: every exposed face of a cell
    box is set back by half of it, seams between a piece's own cells stay flush), from the cell boxes of both pieces: a
    box spans the cell, the cell minus half the gap or the cell minus the gap along each axis."""
    root = ET.parse(scene).getroot()
    extents = []
    for name in ("coil_gold::coil_gold", "coil_silver::coil_silver"):
        body = [b for b in root.iter("body") if b.get("name") == name]
        if len(body) != 1:
            raise SystemExit(f"{scene}: no body {name}")
        boxes = [g for g in body[0].iter("geom") if g.get("type") == "box"]
        if not boxes:
            raise SystemExit(f"{scene}: {name} has no cell boxes")
        extents += [2 * s for g in boxes for s in _floats(g.get("size"))]
    cell, low = max(extents), min(extents)
    gap = cell - low
    for e in extents:
        if min(abs(e - cell), abs(e - (cell - gap / 2)), abs(e - low)) > 1e-9:
            raise SystemExit(f"{scene}: a cell box {e} m is not the cell minus 0, half or all of the gap")
    return dict(clearance_mm=gap * 1e3, cell_mm=cell * 1e3)


MEASURES = dict(bore=measure_bore, lock_bore=measure_lock_bore, coil_gap=measure_coil_gap)


# ---------------------------------------------------------------- the puzzles and their relaxations
@dataclasses.dataclass(frozen=True)
class Edit:
    file: str                    # a source file under /opt/topogym, e.g. topogym/assets/geometry/bed_bolt.py
    sha256: str                  # the release image's copy of that file
    old: str                     # the text replaced (exactly once in the file)
    new: dict                    # factor -> its replacement


@dataclasses.dataclass(frozen=True)
class Relaxation:
    base: str                    # the released base, e.g. roboarc-bed-bolt-01-i01
    constraint: str              # the paper's rigid constraint type
    task: str                    # the family key (make-instance --task)
    variation: str               # make-instance --variation (the base's, as tools/release/mujoco_release.json makes it)
    gripper_space: str           # make-instance --gripper-space
    parameter: str               # what is relaxed, in words
    edits: tuple                 # Edit, ...: the relaxation in the simulator source
    measure: str                 # MEASURES key
    clearance_mm: float          # the original clearance, measured on the released frozen instance
    texts: dict = dataclasses.field(default_factory=dict)   # factor -> ((old, new), ...): dimensions the texts state

    def instance_name(self) -> str:
        return self.base.removeprefix("roboarc-")


_BORE = "RB = .0054                      # bore radius (shank radius 5.0 mm)"
_BED_BOLT_PY = "11c53ead8fec4e96743de2d90bcc23f964eb89fc08c07ce7bee5a3896102dabd"


def _bed_bolt(n: str, variation: str) -> Relaxation:
    return Relaxation(
        f"roboarc-bed-bolt-{n}-i01", "assembly", "bed_bolt", variation, "native",
        "the blind bore's radius RB (5.4 mm round the bolt's 5.0 mm shank): bore minus shank radius 0.4 mm -> 0.8 / 1.6 mm",
        (Edit("topogym/assets/geometry/bed_bolt.py", _BED_BOLT_PY, _BORE,
              {2: _BORE.replace(".0054 ", ".0058 ") + "; ablation E4: bore minus shank radius x2 (0.4 -> 0.8 mm)",
               4: _BORE.replace(".0054 ", ".0066 ") + "; ablation E4: bore minus shank radius x4 (0.4 -> 1.6 mm)"}),),
        "bore", 0.4)


_LOCK = '            "bore_radius": .0075, "slot_radius": .045, "shallow_radius": .028, "slot_half_deg": float(slot_half_deg),'
_LOCK_PY = "1d98800badc9c031ad01046b01c71aef68bed7581efbddbd762de77ba9426c18"


def _lock(n: str, variation: str, shaft_mm: float, rotors: int) -> Relaxation:
    """lock_picking (chain locks, topogym/tasks/lock_picking/chain.py): every plate's bore radius (7.5 mm round the key's
    shaft). The texts state both radii and what they allow: the shaft axis' largest offset, (bore - shaft) + 0.01 mm, and
    the tilt that offset allows over the stack (from the first rotor plate's face to the END plate's, rotors x 120 + 10 mm),
    the tooling's rule, recomputed here as the released texts state it (0.31 / 0.18 degrees)."""
    c0 = 7.5 - shaft_mm
    stack = rotors * 120. + 10.
    def words(bore):
        off = bore - shaft_mm + .01
        tilt = math.degrees(2 * off / stack)
        return (f"bore radius {bore:g} mm, shaft radius {shaft_mm:g} mm.",
                f"may exceed {bore:g} mm by at most 0.01 mm: about {off:.2f} mm of offset when the key is straight, which "
                f"over the stack allows only about {tilt:.2f} degrees of tilt.")
    texts = {}
    for k in FACTORS:
        old, new = words(7.5), words(shaft_mm + k * c0)
        texts[k] = tuple(zip(old, new))
    return Relaxation(
        f"roboarc-lock-picking-{n}-i01", "narrow passage", "lock_picking", variation, "native",
        f"every plate's bore radius (7.5 mm round the key's {shaft_mm:g} mm shaft): bore minus shaft radius {c0:g} mm -> "
        f"{2 * c0:g} / {4 * c0:g} mm",
        (Edit("topogym/tasks/lock_picking/chain.py", _LOCK_PY, _LOCK,
              {k: _LOCK.replace('"bore_radius": .0075,', f'"bore_radius": {(shaft_mm + k * c0) / 1000:.5f},')
               + f"  # ablation E4: bore minus shaft radius x{k} ({c0:g} -> {k * c0:g} mm)" for k in FACTORS}),),
        "lock_bore", c0, texts)


_COIL = "        return read_json(self.geometry_resource)\n"
_COIL_PY = "4927c2e32ca41047178f5aa1574444afc20668bff16e8229129af4fccb41cf40"
_COIL_GOAL = ('        reached = (position_error < .001 and rotation_error < .02) if self.config["objective"] == "assemble" '
              'else separation > .01')
_COIL_COMMON_PY = "30cf31f0986503660d8c90d6139e1d6e5e8acec176e0a886b3341364526caa0e"
_COIL_WORDS = ("Align the pieces' matching assembly frames with relative position error below 1 mm and relative angular error "
               "below 0.02 rad.")
_GOAL_DESCRIPTION_PY = "3c243c1b6bce056d1f8ab8656f620c48c40cb6bb37922f9530983ffc048c1acd"


def _coil(n: str, variation: str) -> Relaxation:
    """cast_coil v2 (the lattice designs): the design's clearance_m, the gap the assembled pieces keep (every exposed cell
    face set back by half of it, the pins' radius by half of it), x2 / x4. The goal's assembled-pose tolerance (the
    pieces' frames within 1 mm, set below the 1.5 mm gap) scales with it: with the gap alone relaxed the reference
    assembles the pieces but they come to rest 1.22 mm apart at x2, inside the larger gap and outside the 1 mm. That one
    grader number, and the sentence of the texts that states it, change with the clearance; nothing else of the grader."""
    edits = (
        Edit("topogym/tasks/cast_coil/cast_coil_v2_very_easy.py", _COIL_PY, _COIL,
             {k: f"        return dict(read_json(self.geometry_resource), clearance_m={0.0015 * k:g})   "
                 f"# ablation E4: gap x{k} (1.5 -> {1.5 * k:g} mm)\n" for k in FACTORS}),
        Edit("topogym/tasks/cast_coil/common.py", _COIL_COMMON_PY, _COIL_GOAL,
             {k: _COIL_GOAL.replace("position_error < .001", f"position_error < .001 * {k}")
                 + f"  # ablation E4: the assembled-pose tolerance x{k} with the gap" for k in FACTORS}),
        Edit("topogym/tasks/goal_description.py", _GOAL_DESCRIPTION_PY, _COIL_WORDS,
             {k: _COIL_WORDS.replace("below 1 mm", f"below {k} mm") for k in FACTORS}),
    )
    texts = {k: ((_COIL_WORDS, _COIL_WORDS.replace("below 1 mm", f"below {k} mm")),) for k in FACTORS}
    return Relaxation(
        f"roboarc-cast-coil-{n}-i01", "interlocking", "cast_coil", variation, "native",
        "the design's clearance_m, the gap the two assembled pieces keep (1.5 mm): 1.5 mm -> 3 / 6 mm, with the goal's "
        "assembled-pose tolerance 1 mm -> 2 / 4 mm",
        edits, "coil_gap", 1.5, texts)


RELAXATIONS: dict[str, Relaxation] = {r.base: r for r in (
    _bed_bolt("01", "bed_bolt_crossbar"),
    _bed_bolt("03", "bed_bolt_square"),
    _lock("04", "lock_picking_medium", 6.5, 3),
    _lock("05", "lock_picking_hard", 6.75, 4),
    _coil("06", "cast_coil_v2_very_easy"),
    _coil("07", "cast_coil_v2_easier"),
)}

# The six bases E4 runs (scripts/ablations/experiments.py reads them), two per rigid constraint type. The narrow-passage
# pair is lock picking, where the agent's main-evaluation results leave no room (3/3): every narrow-passage base with room
# breaks at x4 (buzz wire: the loop cannot grow x4 round the held tool; the owl, its bars thinned or its widest gap
# widened: the open-loop reference depends on where released parts come to rest; the cast cage: its pieces are too thin
# to widen its windows). The interlocking pair is cast coil with its assembled-pose tolerance
# scaled with the gap (_coil): no interlocking puzzle passes x4 with its grader untouched.
BASES = list(RELAXATIONS)


# ---------------------------------------------------------------- freezing the relaxed instance (docker, CPU)
def pinned_image() -> tuple[str, str]:
    """(reference, image id) of the release image the tasks are built FROM (images/pins.json, name roboarc)."""
    pins = json.loads((REPO / "images" / "pins.json").read_text())["images"]
    p = [x for x in pins if x["name"] == "roboarc"]
    if len(p) != 1:
        raise SystemExit("images/pins.json: no single roboarc image")
    return p[0]["ref"], p[0]["id"]


def _docker_image() -> str:
    ref, want = pinned_image()
    use = os.environ.get("RCB_ABLATION_IMAGE") or ref
    got = subprocess.run(["docker", "image", "inspect", use, "--format", "{{.Id}}"], capture_output=True, text=True)
    if got.returncode != 0:
        raise SystemExit(f"E4 needs the release image {use} (docker pull {ref}; or RCB_ABLATION_IMAGE=<a local reference "
                         f"to image {want}>): {got.stderr.strip()[:200]}")
    # The classic image store reports the config digest (the pinned id), the containerd image store the manifest digest.
    if got.stdout.strip() not in (want, ref.rsplit("@", 1)[-1]):
        raise SystemExit(f"{use} is image {got.stdout.strip()}, not the pinned {want} (images/pins.json)")
    return use


def _run_freeze(r: Relaxation, factor: int, out: pathlib.Path) -> None:
    """clearance_freeze.py in the release image: out/{instance/, SOURCE.json, topogym/<file>}."""
    image = _docker_image()
    out.mkdir(parents=True, exist_ok=True)
    job = dict(edits=[dict(file=e.file, sha256=e.sha256, old=e.old, new=e.new[factor]) for e in r.edits], task=r.task,
               variation=r.variation, gripper_space=r.gripper_space, out="/out")
    (out / "job.json").write_text(json.dumps(job, indent=1) + "\n")
    env = [*MUJOCO_ENV, ("CUDA_VISIBLE_DEVICES", ""), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
           ("OPENBLAS_NUM_THREADS", "1"), ("RCB_TORCH_THREADS", "1"), ("HOME", "/tmp"), ("XDG_CACHE_HOME", "/tmp/cache"),
           ("RCB_RENDER", "off"), ("RCB_SESSION_LOG", ""), ("PYTHONPATH", "")]
    cmd = ["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "8g",
           "--label", "rcb-ablation=e4-freeze", "--user", f"{os.getuid()}:{os.getgid()}",
           *(x for k, v in env for x in ("-e", f"{k}={v}")),
           "-v", f"{TOOLING}:/tooling:ro", "-v", f"{HERE}:/e4:ro", "-v", f"{out}:/out", image,
           "python", "/e4/clearance_freeze.py", "/out/job.json"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    (out / "freeze.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0 or "FROZEN " not in p.stdout:
        raise SystemExit(f"{r.base} x{factor}: freezing the relaxed instance failed (log {out / 'freeze.log'}):\n"
                         + (p.stdout + p.stderr)[-1500:])
    (out / "job.json").unlink()


@functools.cache
def frozen(base: str, factor: int) -> pathlib.Path:
    """The relaxed source and instance of base at factor (made once per process; RCB_ABLATION_E4_CACHE keeps them)."""
    r = RELAXATIONS[base]
    cache = os.environ.get("RCB_ABLATION_E4_CACHE")
    if cache:
        root = pathlib.Path(cache)
    else:                                   # this process's own, removed when it ends
        root = pathlib.Path(tempfile.mkdtemp(prefix="rcb-e4-"))
        atexit.register(shutil.rmtree, root, True)
    out = root / f"{r.instance_name()}-x{factor}"
    if not (out / "instance" / "instance.json").is_file():
        shutil.rmtree(out, ignore_errors=True)
        _run_freeze(r, factor, out)
    got = measure(r, out / "instance")
    want = factor * r.clearance_mm
    if abs(got["clearance_mm"] - want) > TOLERANCE_MM:
        raise SystemExit(f"{base} x{factor}: the frozen clearance is {got['clearance_mm']:.6f} mm, not {want:.6f} mm")
    return out


def measure(r: Relaxation, instance: pathlib.Path) -> dict:
    return MEASURES[r.measure](instance / "instance_assets" / "mujoco" / "scene.xml")


# ---------------------------------------------------------------- editing the task directory
def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _files(d: pathlib.Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def binding(old: dict, instance: pathlib.Path, source: dict) -> dict:
    """The runtime binding of the relaxed instance: old's, with the instance, its assets and the source (creation and
    runtime) replaced; tooling, render bridge, nyx_direct, image and dependencies stay."""
    inst = json.loads((instance / "instance.json").read_text())
    if inst.get("topogym_source") != source:
        raise SystemExit("the relaxed instance does not name the relaxed source")
    new = dict(old)
    new["instance_sha256"] = canonical(inst)
    new["asset_files"] = {"instance_assets/" + k: v for k, v in _files(instance / "instance_assets").items()}
    new["creation_source"] = dict(source)
    new["runtime"] = dict(old["runtime"], source=dict(source))
    return new


def _sub1(text: str, old: str, new: str, what: str) -> str:
    _once(text, old, what)
    return text.replace(old, new)


def _once(text: str, part: str, what: str) -> None:
    if text.count(part) != 1:
        raise SystemExit(f"{what}: expected 1 match of {part[:70]!r}, found {text.count(part)}")


def _install(frozen_dir: pathlib.Path, ctx: pathlib.Path) -> None:
    """ctx/ablation_e4/: SOURCE.json, topogym/<the edited file>, instance/ (files 0644, directories 0755)."""
    dst = ctx / E4
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    dst.mkdir(parents=True)
    shutil.copyfile(frozen_dir / "SOURCE.json", dst / "SOURCE.json")
    shutil.copytree(frozen_dir / "topogym", dst / "topogym")
    shutil.copytree(frozen_dir / "instance", dst / "instance")
    for p in [dst, *dst.rglob("*")]:
        os.chmod(p, 0o755 if p.is_dir() else 0o644)


BINDING_COPY = "COPY runtime_binding.json runtime_binding.s1.json runtime_binding.s2.json seed /tmp/rcb-seed/\n"


def _dockerfile(path: pathlib.Path, r: Relaxation) -> None:
    """A privileged image: the relaxed source over /opt/topogym and the relaxed instance in place of the image's, before
    stage_instance.py stages (and checks) it; seed 0's binding only."""
    t = path.read_text()
    over = (f"# ablation E4 (scripts/ablations/cond_clearance.py): the clearance-relaxed simulator source (one edited file,\n"
            f"# its SOURCE.json) and the instance frozen from it, in place of the image's; runtime_binding.json names both\n"
            f"COPY {E4}/topogym/ /opt/topogym/topogym/\n"
            f"COPY {E4}/SOURCE.json /opt/topogym/SOURCE.json\n"
            f"RUN rm -rf /opt/robotangle/instances/{r.instance_name()}\n"
            f"COPY {E4}/instance/ /opt/robotangle/instances/{r.instance_name()}/\n")
    t = _sub1(t, BINDING_COPY, over + "COPY runtime_binding.json seed /tmp/rcb-seed/\n", str(path))
    _once(t, f"python /opt/robotangle/stage_instance.py {r.instance_name()} ", f"{path} stage line")
    path.write_text(t)


def _compose(path: pathlib.Path, r: Relaxation) -> None:
    """The standard sim service mounts the relaxed source and instance over the image's (read-only)."""
    t = path.read_text()
    rels = [e.file.split("/", 1)[1] for e in r.edits]
    mounts = ("      - ./sim:/srv:ro\n"
              f"      # ablation E4 (scripts/ablations/cond_clearance.py): the clearance-relaxed source and its instance\n"
              + "".join(f"      - ./sim/{E4}/topogym/{rel}:/opt/topogym/topogym/{rel}:ro\n" for rel in rels)
              + f"      - ./sim/{E4}/SOURCE.json:/opt/topogym/SOURCE.json:ro\n"
              f"      - ./sim/{E4}/instance:/opt/robotangle/instances/{r.instance_name()}:ro\n")
    t = _sub1(t, "      - ./sim:/srv:ro\n", mounts, str(path))
    _once(t, f"stage_instance.py {r.instance_name()} /tmp/robotangle-instance --binding /srv/runtime_binding.json "
             "--seed-file /srv/seed", f"{path} entrypoint")
    path.write_text(t)


def _restamp(traj: pathlib.Path, sha: str) -> None:
    """The reference's birth stamp (instance_sha.npy) set to sha, every other archive member byte for byte
    (tools/release/restamp.py)."""
    sys.path.insert(0, str(REPO / "tools" / "release"))
    import restamp
    with zipfile.ZipFile(traj) as z:
        old = restamp.npy_str(z.read(restamp.MEMBER))[1]
    if old == sha:
        return
    tmp = traj.with_name(traj.name + ".restamp")
    restamp.restamp(traj, sha, tmp)
    tmp.replace(traj)


def _seeds(task_toml: pathlib.Path) -> None:
    """Seed 0 only: `allowed` and `oracle` of [metadata.seeds] become [0] (the default must be 0)."""
    t = task_toml.read_text()
    seeds = tomllib.loads(t)["metadata"]["seeds"]
    if seeds.get("default") != 0 or 0 not in (seeds.get("oracle") or []):
        raise SystemExit(f"{task_toml}: unexpected seeds {seeds}")
    head, sep, rest = t.partition("[metadata.seeds]\n")
    if not sep:
        raise SystemExit(f"{task_toml}: no [metadata.seeds]")
    table, nxt, tail = rest.partition("\n[")         # the table ends at the next one
    for key in ("allowed", "oracle"):
        found = re.findall(rf"(?m)^{key} = .*$", table)
        if len(found) != 1:
            raise SystemExit(f"{task_toml}: no single `{key} =` line in [metadata.seeds]")
        table = table.replace(found[0], f"{key} = [0]")
    task_toml.write_text(head + sep + table + nxt + tail)


def _normal_form(src: pathlib.Path, dst: pathlib.Path) -> None:
    """What scripts/task_conventions.py normalize writes for the variant's name and seeds (so that `--check` passes):
    the README's title, twin and seed row, and the task name in the resource-limits files."""
    other = "standard" if src.name.endswith("-privileged") else "privileged"
    twin = lambda name: name[: name.rindex("-") + 1] + other  # noqa: E731
    readme = dst / "README.md"
    t = _sub1(readme.read_text(), f"# {src.name}\n", f"# {dst.name}\n", f"{readme} title")
    t = _sub1(t, f"| Twin | `{twin(src.name)}` |", f"| Twin | `{twin(dst.name)}` |", f"{readme} twin")
    t = _sub1(t, " / seed 0 (allowed seeds: [0, 1, 2]; §2.13) |", " / seed 0 |", f"{readme} seed row")
    readme.write_text(t)
    for f in sorted([*dst.glob("environment/rcb_monitor/limits.json"), *dst.glob("tests/rcb_limits.json")]):
        f.write_text(_sub1(f.read_text(), f'"task": "{src.name}"', f'"task": "{dst.name}"', str(f)))


def split(name: str) -> tuple[str, str]:
    for mode in ("privileged", "standard"):
        if name.endswith("-" + mode):
            return name[: -len(mode) - 1], mode
    raise SystemExit(f"{name}: not a <base>-<mode> task directory")


def make(src: pathlib.Path, dst: pathlib.Path, factor: int) -> None:
    """Edit dst (a full copy of the released task src) into its clearance-x<factor> variant (module docstring)."""
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    base, mode = split(src.name)
    if base not in RELAXATIONS:
        raise SystemExit(f"{base}: not an E4 base ({', '.join(RELAXATIONS)})")
    if factor not in FACTORS:
        raise SystemExit(f"factor {factor}: E4 relaxes x2 or x4")
    r = RELAXATIONS[base]
    fz = frozen(base, factor)
    source = json.loads((fz / "SOURCE.json").read_text())
    inst = json.loads((fz / "instance" / "instance.json").read_text())
    for p in sorted(dst.rglob("runtime_binding.s*.json")):
        p.unlink()
    places = ["environment", "tests"] if mode == "privileged" else ["environment/sim"]
    bindings = sorted(dst.rglob("runtime_binding.json"))
    if not bindings:
        raise SystemExit(f"{src.name}: no runtime_binding.json")
    for b in bindings:
        b.write_text(json.dumps(binding(json.loads(b.read_text()), fz / "instance", source), indent=2) + "\n")
    for place in places:
        _install(fz, dst / place)
    if mode == "privileged":
        _dockerfile(dst / "environment" / "Dockerfile", r)
        _dockerfile(dst / "tests" / "Dockerfile", r)
        for p in sorted((dst / "solution").glob("trajectory.s*.npz")):
            p.unlink()
        _restamp(dst / "solution" / "trajectory.npz", inst["state_sha"])
    else:
        _compose(dst / "environment" / "docker-compose.yaml", r)
        d = dst / "tests" / "Dockerfile"
        d.write_text(_sub1(d.read_text(), BINDING_COPY, "COPY runtime_binding.json seed /tmp/rcb-seed/\n", str(d)))
    _seeds(dst / "task.toml")
    _normal_form(src, dst)
    _texts(dst, r, factor)


def _texts(dst: pathlib.Path, r: Relaxation, factor: int) -> None:
    """The dimensions the texts state (Relaxation.texts: the instruction states each once, the README may), then the
    source the texts quote follows the edit: the privileged instruction's implementation excerpt (evaluate_state,
    verbatim) shows the relaxed line wherever it shows an edited one (the coil's goal tolerance)."""
    for old, new in r.texts.get(factor, ()):
        f = dst / "instruction.md"
        f.write_text(_sub1(f.read_text(), old, new, str(f)))
        f = dst / "README.md"
        if old in f.read_text():
            f.write_text(_sub1(f.read_text(), old, new, str(f)))
    for e in r.edits:
        for f in (dst / "instruction.md", dst / "README.md"):
            if e.old in f.read_text():
                f.write_text(_sub1(f.read_text(), e.old, e.new[factor], str(f)))


def leak_check(dst: pathlib.Path) -> list[str]:
    """A standard variant adds nothing the agent's container sees: the relaxed files live in the sim service's context
    only (environment/sim/ablation_e4) and the agent image copies none of them; the instruction differs from the base
    task's only in the dimensions Relaxation.texts restates (radii and tolerances in mm, no position), checked when the
    released task lies next to the variant (tasks/, make_variants.py's default)."""
    dst = pathlib.Path(dst)
    _, mode = split(dst.name)
    bad = []
    if mode == "standard":
        places = sorted(p.relative_to(dst).as_posix() for p in dst.rglob(E4) if p.is_dir())
        if places != [f"environment/sim/{E4}"]:
            bad.append(f"{E4} in {places}")
        if E4 in (dst / "environment" / "Dockerfile").read_text():
            bad.append("the agent's image copies the relaxed files")
        m = re.fullmatch(r"(.+)-clr(\d)x-standard", dst.name)
        released = dst.parent / f"{m.group(1)}-standard" / "instruction.md" if m and m.group(1) in RELAXATIONS else None
        if released is not None and released.is_file():
            want = released.read_text()
            for old, new in RELAXATIONS[m.group(1)].texts.get(int(m.group(2)), ()):
                want = want.replace(old, new)
            if (dst / "instruction.md").read_text() != want:
                bad.append("the instruction differs from the released task's beyond the restated dimensions")
    return bad


# ---------------------------------------------------------------- the check of a made variant (check_clr2x.py, check_clr4x.py)
def check(variant: pathlib.Path, factor: int) -> dict:
    """The clearance measured in the variant's own frozen scene against factor x the original, and the relaxed files
    against what the variant binds (runtime_binding.json in every place)."""
    variant = pathlib.Path(variant)
    name = variant.name
    token = f"-clr{factor}x-"
    if token not in name:
        raise SystemExit(f"{name}: not a clr{factor}x variant")
    base, mode = name.split(token)[0], name.split(token)[1]
    r = RELAXATIONS[base]
    places = ["environment", "tests"] if mode == "privileged" else ["environment/sim"]
    rows = []
    for place in places:
        d = variant / place / E4
        got = measure(r, d / "instance")
        src = json.loads((d / "SOURCE.json").read_text())
        for b in sorted(variant.rglob("runtime_binding.json")):
            want = json.loads(b.read_text())
            if want["instance_sha256"] != canonical(json.loads((d / "instance" / "instance.json").read_text())):
                raise SystemExit(f"{b}: binds another instance than {d / 'instance'}")
            if want["runtime"]["source"] != src or want["creation_source"] != src:
                raise SystemExit(f"{b}: binds another source than {d / 'SOURCE.json'}")
        for e in r.edits:
            if (d / "topogym" / e.file.split("/", 1)[1]).read_text().count(e.new[factor]) != 1:
                raise SystemExit(f"{d}: the relaxed {e.file} is not the x{factor} edit")
        rows.append(dict(place=place, **got))
    want = factor * r.clearance_mm
    ok = all(abs(x["clearance_mm"] - want) <= TOLERANCE_MM for x in rows)
    return dict(variant=name, base=base, factor=factor, parameter=r.parameter, original_mm=r.clearance_mm,
                intended_mm=want, measured_mm=rows[0]["clearance_mm"], places=rows, ok=ok)


def check_main(argv: list[str], factor: int) -> int:
    if len(argv) != 1:
        print(f"usage: check_clr{factor}x.py <variant task dir>", file=sys.stderr)
        return 2
    try:
        out = check(pathlib.Path(argv[0]), factor)
    except SystemExit as e:
        print(f"FAIL {argv[0]}: {e}")
        return 1
    print(("ok " if out["ok"] else "FAIL ") + json.dumps(out))
    return 0 if out["ok"] else 1
