#!/usr/bin/env python3
"""Render the RoboTangle tasks (formerly TopoGym; docs/benchmarks/robotangle.md):
tasks/robotangle-<family>-<tier>-i00-privileged and its standard twin -standard (eai-standard/2.4).

usage: python3 scripts/robotangle/mk_tasks.py [--check] [--list] [--assets <dir>] [base ...]
       python3 scripts/robotangle/mk_tasks.py --dev <dev.json> --out <dir> base ...
  no base   every base in release.json
  --check   write nothing; exit 1 if a task would change, or if an identity check fails
  --list    print the bases with their evidence
  --assets  a local copy of the pinned HF dataset (scripts/robotangle/assets.py fetch <dir>): also check the frozen
            instances' bytes against the runtime bindings (without it, the pins in assets.json are compared)
  --dev     local dev task dirs, written to --out (never tasks/): dev.json names a locally built image ("image":
            reference, id; "runtime" and "uipc_builds" as its installed_runtime() reports them) and, per base, a local
            frozen instance directory ("instance") and the reference trajectory both modes replay ("reference"); the
            binding then names that instance, image and runtime (scripts/robotangle/harbor_check.sh writes dev.json)
A MuJoCo instance (simulator.physics_engine "mujoco", known from the local instance) makes a MuJoCo task: the numpy /
OpenBLAS SIMD masking in every container that runs a Session (docs/simulator/MUJOCO_CORE.md), render_bridge.py next to every
copy of the tooling (its frames, docs/simulator/RENDER_BRIDGE.md) and its sha256 in the binding (render_bridge_sha256). The
families whose MuJoCo tasks render with nyx_direct.py (NYX_DIRECT: rcb_topogym.TASKS renderer "direct") ship that file
next to the tooling too, and its sha256 in the binding (nyx_direct_sha256).

Inputs, all in the repository (Python standard library only, plus scripts/task_conventions.normalize):
  release.json   the release (dev/jianwen commit, public simulator source, runtime) and, per base, where its frozen
                 instance comes from and which recorded oracle runs are its evidence (scripts/robotangle/import_evidence.py)
  shared/        the release's task templates (harbor_templates/topogym at release.json template_commit), verbatim:
                 the tooling rcb_topogym.py, the graders, the sim service adapter (server.py, replay.py, finalize.sh),
                 the standard oracle player
  scripts/eai_standard/v2_4/ + scripts/eai_standard/
                 the eai-standard/2.4 layer (docs/STANDARD_MODE_2_4.md, issue #89): rcb_limited.py and rcb_client.py
                 from v2_4/, rcb1.py, run.sh and the client pins from scripts/eai_standard/. The release called its
                 layer "2.3"; that number went to the realtime capability (#86), so the layer and the texts say 2.4
  packages/      per task, the upstream texts of the graded package (task.toml, README.md, instruction.md, Dockerfiles,
                 compose files, test.sh) and its upstream runtime binding; the generator applies the RoboTangle rename
                 (user-facing names only), the image name and the evidence wording to them
  oracles/       the recorded oracle trajectories of the evidence runs (privileged deliverable; standard episode)
  assets.json    the frozen instances (instance.json + instance_assets/, up to 20 MB per file) are not in git: they are
                 in the private HF dataset eai-v2-internal/robotangle-assets, pinned here by commit and sha256
                 (scripts/robotangle/assets.py). The image carries them (images/robotangle/Dockerfile); a task keeps its
                 runtime_binding.json, which names the instance's canonical sha256 and every asset's sha256, and stages
                 its instance from the image (images/robotangle/stage_instance.py: privileged environment/ and tests/
                 Dockerfiles, the standard sim service's entrypoint), where the tooling checks the binding again.
Nothing at run time imports repository code: the tooling and the sim layer are copies (PROTOCOL D9).
"""
from __future__ import annotations
import argparse, filecmp, hashlib, json, os, pathlib, re, shutil, sys, tempfile, zipfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASKS = ROOT / "tasks"
SHARED, PKG, ORC = HERE / "shared", HERE / "packages", HERE / "oracles"
sys.path.insert(0, str(ROOT / "scripts"))
import task_conventions as TC  # noqa: E402  the protocol's conventions: every generator ends with normalize(task_dir)

REL = json.loads((HERE / "release.json").read_text())
NOP = json.loads((HERE / "nop_records.json").read_text())
ASSETS = json.loads((HERE / "assets.json").read_text())        # the pinned HF dataset (scripts/robotangle/assets.py)
STAGED = "/tmp/robotangle-instance"                             # where the standard sim stages its instance   # nop runs on this release's build (very_easy / easier)
VERSION = (ROOT / "images/robotangle/VERSION").read_text().strip()
IMAGE = f"ghcr.io/mll-lab-nu/eai-robotangle:{VERSION}"
ROBOT = {"aloha": "aloha-2", "panda_robotiq": "franka-robotiq-2f85"}
ROBOT_WORDS = {"(aloha, Genesis)": "(ALOHA 2, Genesis)", "(panda_robotiq, Genesis)": "(Franka Panda + Robotiq 2F-85, Genesis)"}
DOCS = [("docs/rcb/STANDARD_MODE_2_3.md", "docs/STANDARD_MODE_2_4.md"), ("docs/STANDARD_MODE.md", "docs/STANDARD_MODE_2_2.md"),
        ("docs/simulator/TOPOGYM_PROTOCOL.md", "docs/benchmarks/robotangle.md"), ("docs/PROTOCOL.md", "docs/PROTOCOL.md"),
        ("docs/TAXONOMY.md", "docs/TAXONOMY.md"), ("docs/rcb/LIMITED_MODE_PROTOCOL.md", "docs/LIMITED_MODE_PROTOCOL.md")]
IMAGE_RX = re.compile(r"(?:[\w.-]+/)+rcb-topogym:[\w.-]+")
EAI = ROOT / "scripts/eai_standard"
LAYER = {"environment/rcb_client.py": EAI / "v2_4/rcb_client.py", "environment/rcb1.py": EAI / "rcb1.py",
         "environment/client-requirements.txt": EAI / "client-requirements.txt",
         "environment/sim/rcb_limited.py": EAI / "v2_4/rcb_limited.py", "environment/sim/rcb1.py": EAI / "rcb1.py",
         "environment/sim/run.sh": EAI / "run.sh"}   # the eai-standard/2.4 layer, copied unchanged (check-tooling-identity.sh)
PROTO = ("eai-standard/2.3", "eai-standard/2.4")   # the release's name of the layer -> its registered version (#89)
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def dash(s: str) -> str:
    return s.replace("_", "-")


def text(s: str) -> str:
    """User-facing rename and paths (README, instruction, task.toml); code identifiers (topogym, rcb_topogym) stay."""
    s = re.sub(r"\btopogym-([a-z0-9-]+-i\d\d-(?:privileged|standard))\b", r"robotangle-\1", s)
    s = IMAGE_RX.sub(IMAGE, s)
    for a, b in DOCS:
        s = s.replace(a, b)
    for a, b in ROBOT_WORDS.items():
        s = s.replace(a, b)
    s = s.replace("TopoGym", "RoboTangle").replace(*PROTO)
    s = re.sub(r"\bhis (?=[a-z]+(?:_[a-z0-9]+)*_v\d)", "the ", s)
    return s.replace("his qualification record", "the author's qualification record")


MUJOCO_WORDS = (("(ALOHA 2, Genesis)", "(ALOHA 2, MuJoCo)"),
                ("(Franka Panda + Robotiq 2F-85, Genesis)", "(Franka Panda + Robotiq 2F-85, MuJoCo)"),
                ("RoboTangle on Genesis 1.2.1", "RoboTangle on MuJoCo 3.14.0"),
                ("`topogym`) on Genesis 1.2.1 physics", "`topogym`) on MuJoCo 3.14.0 physics"),
                ('"genesis",', '"mujoco",'), ("(Genesis initialises once)", "(one simulator per process)"))


def is_mujoco(rec: dict) -> bool:
    """A base whose frozen instance runs on MuJoCo (release.json physics_engine; docs/simulator/MUJOCO_CORE.md)."""
    return rec.get("physics_engine") == "mujoco"


def mujoco_text(s: str, rec: dict) -> str:
    """A MuJoCo base's texts name the engine it runs on (the physics port's only text change; rules and goals stay)."""
    if not is_mujoco(rec):
        return s
    for a, b in MUJOCO_WORDS:
        s = s.replace(a, b)
    return s


def toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def standard_task_text(s: str) -> str:
    """Describe the existing standard robot fields without public simulator diagnostics."""
    robot_signals = (
        "- Robot command bounds: joint and gripper targets use the configured ranges and tabletop stops. "
        "Command clipping is reported in the existing `meta[\"clipped\"]` entries, with the action `row`, joint "
        "`index` and a plain `reason` about the robot's own command. Clipping does not end the episode.\n"
        "- Robot signals: the groups in `spec[\"proprio_groups\"]` are `joint_pos`, `tcp_pose`, `joint_cmd`, "
        "`joint_vel`, `gripper_opening` and `joint_effort`. Positions and velocities come from robot encoders; "
        "gripper opening is the encoder-derived finger gap. `joint_effort` is applied drive effort or the declared "
        "PD drive estimate, in the specified joint order and units.\n")
    endings = (
        "- Episode endings use the existing `meta[\"done\"]` and `meta[\"reason\"]` fields: task-rule failures "
        "use `task_failed`; simulator errors use `error` with a generic explanation.\n")
    s = re.sub(r"(?m)^(- Robot safety .*\n)", lambda m: m.group(1) + robot_signals + endings, s)
    s = re.sub(r"(?m)^- Simulator breakdown .*\n", "", s)
    s = s.replace('in the evidence (`meta["safety"]`)', 'in the private grading evidence')
    s = re.sub(r" Every response's `meta\[\"safety\"\]` reports, since the\n.*?"
               r"table-guard stops, against the limits above\.", "", s, flags=re.S)
    return s


def run_words(e: dict) -> str:
    return f"{e['release']} (source {e['source']}, tooling {e['tooling']}; lab run {pathlib.Path(e['attempt']).name.removeprefix('attempt-')})"


def evidence_text(base: str, rec: dict) -> str:
    ev = rec["evidence"]
    if not ev:
        return ("Oracle pending: the author is fixing this tier's oracle; it follows in a later pull request. "
                "Human review (PROTOCOL §2.7); nop pending.")
    runs = "; ".join(f"{m}: {run_words(ev[m])}" for m in ("privileged", "standard") if m in ev)
    note = rec.get("note")
    return (f"Human review (PROTOCOL §2.7). No oracle run on this build (dev/jianwen {REL['release_commit'][:9]}, source "
            f"{REL['runtime']['source']['sha256'][:8]}, tooling {REL['tooling_sha256'][:8]}). Positive examples: the "
            f"author's oracle scored 1 in both modes on {runs}" + (f"; {note}" if note else "") +
            ". solution/ replays the recorded trajectories.")


def since_evidence(rec: dict) -> str:
    """What changed between the evidence build and this release (release.json; docs/benchmarks/robotangle.md, Evidence)."""
    end = " The oracle was not re-run on this build (docs/benchmarks/robotangle.md, Evidence)."
    if rec["family"] == "blanket_flatten" and REL.get("blanket_change"):
        return REL["blanket_change"] + end
    shared = REL.get("release_change", "")
    if rec.get("note"):          # graded on an earlier build than merged-v18 / its equivalent (the documented equivalence)
        return f"Between that build and merged-v18: {rec['note']}. After merged-v18, {shared}." + end
    return f"This family's task, geometry and asset files are unchanged from that build to this release; {shared}." + end


def ref_solution(mode: str, rec: dict) -> str:
    ev = rec["evidence"].get(mode)
    if is_mujoco(rec) and ev:
        if mode == "privileged":
            route = ("this design's oracle, recorded on MuJoCo" if rec.get("template") or rec.get("design") else
                     "the released route re-recorded on MuJoCo with a held tail, docs/simulator/MUJOCO_CORE.md")
            return (f"`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `{ev['trajectory_sha256'][:16]}…`) to "
                    f"`/app/output/`: the MuJoCo reference ({route}), which the verifier scored 1 in two fresh-process "
                    f"replays on {run_words(ev)}.")
        return ("`solution/solve.sh` streams the recorded rows of the standard-mode oracle episode "
                "(`solution/oracle_qpos.npy`, one `joint_position` row per 25 ms control step, `hold`; the same rows as the "
                "privileged reference) through the client (`solution/oracle_replay.py`). That episode scored 1 (live and "
                f"fresh replay) on {run_words(ev)}.")
    if not ev:
        return ("None in this release: the author is fixing this tier's oracle (it follows in a later pull request). "
                "`oracle = \"none\"`, human review (PROTOCOL §2.7).")
    same = since_evidence(rec)
    if mode == "privileged":
        return (f"`solution/solve.sh` copies `solution/trajectory.npz` (sha256 `{ev['trajectory_sha256'][:16]}…`) to "
                f"`/app/output/`: the trajectory the author's oracle (the RoboTangle planner on dev/jianwen) produced on "
                f"{run_words(ev)}, where the verifier scored it 1 in two fresh-process replays. " + same)
    return ("`solution/solve.sh` streams the recorded rows of the author's standard-mode oracle episode "
            "(`solution/oracle_qpos.npy`, one `joint_position` row per 25 ms control step, `hold`) through the client "
            f"(`solution/oracle_replay.py`). That episode scored 1 (live and fresh replay) on {run_words(ev)}. " + same)


def nop_words(base: str, mode: str, rec: dict) -> str:
    if is_mujoco(rec) and rec.get("nop"):
        n = rec["nop"]
        return f"`nop` scored 0 on this base in both modes on this build ({n['run']}; {n['date']}, {n['host']})."
    if rec["tier"] in ("easy", "medium", "hard"):
        return ("`nop` scored 0 on this base in both modes on the merged-v18 candidate build (lab `align_v18/cand_v18`, the "
                "pipeline's Harbor nop checks, 2026-10-05; frozen instance state identical); not yet re-run on this build.")
    run = NOP["runs"].get(base.removesuffix("-i00"), {}).get(mode)
    built = NOP["build"].split()[1]                       # "release <commit> image ..." (scripts/robotangle/nop_records.json)
    here = REL["release_commit"].startswith(built)
    if run and run["clean"] and run["reward"] == 0:
        return (f"`nop` scored 0 on {'this release' if here else 'the previous release'}'s build ({built}; {NOP['date']}, "
                f"{NOP['host']}; image {NOP['build'].split(' sha256:')[1][:12]}…, with the audited libuipc wheel)"
                + ("." if here else "; not yet re-run on this build."))
    if run:
        return (f"`nop`: no valid run yet. On the build of {built} an RTX 4090 (compute capability 8.9) refused this task's "
                "frozen PBD state, which was prepared on an RTX PRO 6000 Blackwell (compute capability 12.0, 188 SMs). "
                + (REL.get("blanket_nop_note") or "") + " Pending (docs/benchmarks/robotangle.md, GPU portability).")
    return f"`nop`: pending (`harbor run -p tasks/robotangle-{base}-{mode} -a nop`, docs/benchmarks/robotangle.md)."


def provenance(base: str, rec: dict) -> str:
    if is_mujoco(rec):
        rows = [f"- Frozen instance: a MuJoCo instance (`simulator.physics_engine` \"mujoco\", its scene frozen under "
                f"`instance_assets/mujoco/`), built by TopoGym source `{rec['creation_source']}` ({rec['instance_package']}); "
                f"bound to this release's runtime (source `{REL['runtime']['source']['sha256'][:8]}`, tooling "
                f"`{REL['tooling_sha256'][:8]}`)."]
    else:
        rows = [f"- Frozen instance: built by TopoGym source `{rec['creation_source']}` (graded package "
            f"`{rec['instance_package']}/topogym-{base}-privileged` on the lab host); bound to this release's runtime "
            f"(source `{REL['runtime']['source']['sha256'][:8]}`, tooling `{REL['tooling_sha256'][:8]}`). An exact-restore "
            "preflight on this build is pending (docs/benchmarks/robotangle.md, Pending)."]
    rows.append(f"- Frozen instance files (`instance.json`, `instance_assets/`): not in this directory. They are in the private HF "
                f"dataset `{ASSETS['repo']}` at `{ASSETS['revision'][:12]}`, under `instances/{base}/` "
                "(scripts/robotangle/assets.json pins each file's sha256). The image carries them, and "
                "`/opt/robotangle/stage_instance.py` copies this task's instance next to its `runtime_binding.json`, "
                "which names the instance's canonical sha256 and every asset's sha256. The tooling checks them again at "
                "every session start.")
    if is_mujoco(rec):
        rows.append(f"- Release: dev/robotangle_mujoco `{rec['release_commit'][:9]}`, image `{IMAGE}` "
                    "(images/robotangle/Dockerfile.mujoco).")
    else:
        rows.append(f"- Release: dev/jianwen `{REL['release_commit'][:9]}` (task templates `{REL['template_commit'][:9]}`), "
                    f"image `{IMAGE}` (images/robotangle/).")
    rows.append("- Name: RoboTangle is the published name of TopoGym (2026-10-07). Code identifiers keep the old name: the "
                "simulator package `topogym`, the tooling `rcb_topogym.py`, `TOPOGYM_*` variables.")
    return "\n".join(rows)


def rewrite_toml(s: str, base: str, rec: dict) -> str:
    s = mujoco_text(text(s), rec)
    s = re.sub(r'(?m)^simulator = "topogym"', 'simulator = "robotangle"', s)
    s = re.sub(r'(?m)^robot = "([a-z_]+)"', lambda m: f'robot = "{ROBOT.get(m.group(1), m.group(1))}"', s)
    s = re.sub(r'(?m)^author_name = "jianwen"', 'author_name = "Jianwen Lyu"', s)
    s = s.replace('authors = [{ name = "jianwen" }]', 'authors = [{ name = "Jianwen Lyu" }]')
    s = s.replace('"topogym",', '"robotangle",')
    s = re.sub(r'(?m)^tooling_version = "\d+"', f'tooling_version = "{REL["tooling_version"]}"', s)
    s = re.sub(r'(?m)^oracle_evidence = .*\n', "", s)
    if {"privileged", "standard"} <= set(rec["evidence"]):   # a passing oracle run in both modes (docs/benchmarks/robotangle.md)
        s = re.sub(r'(?m)^oracle = .*\n', 'oracle = "full"\n', s, count=1)
    else:
        s = re.sub(r'(?m)^oracle = .*\n', f'oracle = "none"\noracle_evidence = {toml_str(evidence_text(base, rec))}\n', s, count=1)
    return s


MUJOCO_CHANGE = ("- 2026-10-10: physics moved from Genesis 1.2.1 to MuJoCo 3.14.0 (docs/simulator/MUJOCO_CORE.md, "
                 "docs/history/MUJOCO_MIGRATION.md): a new frozen instance of the same scene, goal and rules, a new reference, "
                 "the numpy / OpenBLAS SIMD masking in every container that runs a Session, and the render bridge next to "
                 "the tooling (Nyx frames unchanged, docs/simulator/RENDER_BRIDGE.md). This task replaces `robotangle-{old}-{mode}` "
                 "under PROTOCOL §2.8 (MuJoCo physics).")
# A released base whose design changed with the port (mujoco_release.json design: one line, what changed).
DESIGN_CHANGE = ("- {date}: physics moved from Genesis 1.2.1 to MuJoCo 3.14.0 (docs/simulator/MUJOCO_CORE.md, "
                 "docs/history/MUJOCO_MIGRATION.md), and the task's design changed: {design}. A new frozen instance, task text, "
                 "rules, goal and reference, the numpy / OpenBLAS SIMD masking in every container that runs a Session, and "
                 "the render bridge next to the tooling (docs/simulator/RENDER_BRIDGE.md). This task replaces "
                 "`robotangle-{old}-{mode}` under PROTOCOL §2.8.")
# A new base (mujoco_release.json new_bases): no released task, so no earlier change log.
NEW_TASK = ("- {date}: new task on MuJoCo 3.14.0 (docs/simulator/MUJOCO_CORE.md): variation `{variation}` of {family}{design}. "
            "Its task files start from `robotangle-{template}-{mode}`, with this design's own task text, rules and goal, "
            "its own frozen instance and its own reference.")


def changes(mode: str, rec: dict | None = None) -> list[str]:
    """The README Change Log (PROTOCOL §2.8: published tasks fixed in place)."""
    if rec is not None and rec.get("template"):
        return [NEW_TASK.format(date=rec["nop"]["date"], variation=rec["variation"], family=rec["family"],
                                design=f": {rec['design']}" if rec.get("design") else "", template=rec["template"],
                                mode=mode)]
    grader = ("The standard grader counts a fresh replay that crashed without writing its verdict (e.g. CUDA out of memory) "
              "as a grader error; it counted as instance_mismatch." if mode == "standard" else "The graders are unchanged.")
    return [
        "- 2026-10-08: base image 0.2.19 -> 0.2.21, now public. The simulator, the frozen instances and the audited libuipc "
        "wheel are unchanged: on compute capability 8.9 and 12.x (e.g. RTX 4090, L40S, RTX PRO 6000) the physics runs the "
        "same libuipc code as before. New: the NVIDIA Vulkan ICD and EGL vendor files, and a second libuipc build of the "
        "audited recipe for compute capability 8.6 (RTX A6000 class), which the tooling selects at Session start on every "
        "other GPU, so knot_untying and rope_threading run there too. The runtime binding names both builds, and the tooling "
        "hash changes with the selection. The compose files reserve one GPU without `runtime: nvidia`, which with the "
        f"image's NVIDIA_VISIBLE_DEVICES=all could expose every GPU (Nyx needs exactly one). {grader} The scene, frozen "
        "instance and pass rule are unchanged, so the task keeps its name.",
        "- 2026-10-08 (2): the instance now also starts on a CPU of another instruction-set class. The asset generators run "
        "on numpy and OpenBLAS, whose SIMD kernels depend on the CPU: on a host without AVX-512 (e.g. AWS L4 / L40S "
        "instances) the pinned generator writes the frozen asset files with other bytes (floating-point noise up to 1e-8 m, "
        "face order), and the Session refused the instance (IPC and stored PBD states also refused their archive). Such a "
        "host now realizes the frozen recipe (`instance_realized`): the same asset files from the generator the runtime "
        "binding pins, archives that differ from this scene only in those bytes, and the frozen state, exactly or within "
        "the pose tolerance; the live episode and the fresh replay realize it alike. The graders"
        + (" and the sim service" if mode == "standard" else "") + " record `instance_realized` and `assets_exact`. A host "
        "that reproduces the frozen bytes runs exactly as before. The tooling hash changes; the image, scene, frozen instance "
        "and pass rule are unchanged, so the task keeps its name.",
        "- 2026-10-08 (3): base image 0.2.21 -> 0.2.22. The image adds, for the eight IPC instances (knot_untying, "
        "rope_threading), the rope volume mesh each was frozen with (`instance_assets/frozen_assets/rope_volume.npz`, the sha256 "
        "its asset identity records), and the tooling uses it instead of regenerating it: tetgen's interior points depend on "
        "the CPU's floating-point rounding, and on CPUs without AVX-512 knot_untying medium and hard got another mesh, which "
        "their frozen settled state does not fit. Nothing else in the image changes. The scene, frozen instance and pass rule are unchanged, so the task "
        "keeps its name.",
    ] + ([] if rec is None or not is_mujoco(rec) else
         [DESIGN_CHANGE.format(date=rec["nop"]["date"], design=rec["design"], old=rec["replaces"], mode=mode)]
         if rec.get("design") else [MUJOCO_CHANGE.format(old=rec["replaces"], mode=mode)])


def rewrite_readme(s: str, base: str, mode: str, rec: dict) -> str:
    s = mujoco_text(text(s), rec)
    if mode == "standard":
        s = standard_task_text(s)
    s = s.replace("from `tests/instance.json`", "from the frozen instance (`/tests/instance.json`, staged from the image and "
                        "checked against `tests/runtime_binding.json`)")
    s = re.sub(r"(?ms)^## Reference solution\n.*?(?=^## )", "## Reference solution\n" + ref_solution(mode, rec) + "\n\n", s)
    nop = nop_words(base, mode, rec)
    s = re.sub(r"(?ms)^(## Verification\n.*?)(?=^## )", lambda m: m.group(1).rstrip() + "\n\n" + nop + "\n\n", s)
    return s.rstrip() + "\n\n### Provenance\n" + provenance(base, rec) + "\n\n### Change Log\n" + "\n".join(changes(mode, rec)) + "\n"


def binding(base: str, inst_dir: pathlib.Path | None = None, bridge: bool = False, direct: bool = False) -> bytes:
    """The task's runtime binding: the upstream lock (instance and asset hashes) with this release's tooling and runtime.
    The frozen instance must be the pinned one in assets.json; with inst_dir (a local copy), its bytes are checked too."""
    lock = json.loads((PKG / f"{base}-privileged" / "runtime_binding.upstream.json").read_text())
    pinned = ASSETS["instances"].get(base, {}).get("instance_sha256")
    assert lock["instance_sha256"] == pinned, f"{base}: the bound instance is not the one pinned in assets.json"
    prefix = f"instances/{base}/"
    pinned_assets = {k[len(prefix):]: v for k, v in ASSETS["files"].items() if k.startswith(prefix + "instance_assets/")}
    # Frozen generated assets (image 0.2.22: knot_untying's rope volume meshes, rcb_topogym._install_frozen_assets), added to
    # the instance after the upstream lock: each is a generated file whose sha256 the instance's asset_identity records.
    frozen = {k: v for k, v in pinned_assets.items() if k.startswith("instance_assets/frozen_assets/")}
    lock["asset_files"] = {**lock["asset_files"], **frozen}
    assert lock["asset_files"] == pinned_assets, f"{base}: instance_assets pinned in assets.json differ from the binding"
    if inst_dir is not None and frozen:
        identity = json.loads((inst_dir / "instance.json").read_text()).get("asset_identity") or {}
        for rel, want in frozen.items():
            assert identity.get(pathlib.PurePosixPath(rel).name) == want, f"{base}: {rel} is not the generated asset it records"
    if inst_dir is not None:
        canon = json.dumps(json.loads((inst_dir / "instance.json").read_text()), sort_keys=True, separators=(",", ":"), allow_nan=False)
        assert lock["instance_sha256"] == sha(canon.encode()), f"{base}: instance.json is not the bound instance"
        assets = {p.relative_to(inst_dir).as_posix(): sha(p.read_bytes()) for p in sorted((inst_dir / "instance_assets").rglob("*")) if p.is_file()}
        assert lock["asset_files"] == assets, f"{base}: instance_assets differ from the binding"
    lock["tooling_sha256"] = sha((SHARED / "rcb_topogym.py").read_bytes())
    lock["runtime"] = REL["runtime"]
    # both libuipc builds the tooling chooses from by the GPU's compute capability (rcb_topogym.select_uipc_build), by their
    # uipc-build.json sha256; the runtime's own uipc_build_sha256 stays the audited build, the image's default
    lock["uipc_builds"] = {name: b["uipc_build_sha256"] for name, b in sorted(REL["uipc_builds"].items())}
    if bridge:     # a MuJoCo task: the render bridge next to the tooling (rcb_topogym._check_render_bridge)
        lock["render_bridge_sha256"] = sha((SHARED / "render_bridge.py").read_bytes())
    if direct:     # its family renders with nyx_direct.py (NYX_DIRECT)
        lock["nyx_direct_sha256"] = sha((SHARED / "nyx_direct.py").read_bytes())
    return (json.dumps(lock, indent=2) + "\n").encode()


def put(dst: pathlib.Path, data: bytes, exe: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    os.chmod(dst, 0o755 if exe else 0o644)


def copy_tree(src: pathlib.Path, dst: pathlib.Path) -> None:
    for p in sorted(src.rglob("*")):
        if p.is_file():
            put(dst / p.relative_to(src), p.read_bytes())
        elif p.is_dir():
            (dst / p.relative_to(src)).mkdir(parents=True, exist_ok=True)


GPU_OLD = ("# Nyx renders through Vulkan, so the container needs the NVIDIA graphics libraries as well as compute: `runtime: nvidia`\n"
           "# with NVIDIA_DRIVER_CAPABILITIES=all (`capabilities: [gpu]` alone injects compute only).\n")
GPU_NEW = ("# Nyx renders through Vulkan: the device reservation below passes exactly one GPU (Nyx needs exactly one), and\n"
           "# NVIDIA_DRIVER_CAPABILITIES=all makes the container toolkit mount the driver's graphics libraries, which the image's\n"
           "# Vulkan ICD and EGL vendor files name (images/robotangle/Dockerfile). No `runtime: nvidia`: with the image's\n"
           "# NVIDIA_VISIBLE_DEVICES=all it can expose every GPU of the host.\n")


def gpu_lines(name: str, s: str) -> str:
    """Compose files: one GPU through the device reservation only, as images/robopaint/ (no `runtime: nvidia`)."""
    if not name.endswith("docker-compose.yaml"):
        return s
    assert s.count(GPU_OLD) == 1 and len(re.findall(r"(?m)^\s*runtime: nvidia\n", s)) == 1, f"{name}: unexpected GPU block"
    s = re.sub(r"(?m)^\s*runtime: nvidia\n", "", s.replace(GPU_OLD, GPU_NEW))
    assert "count: 1" in s and "capabilities: [gpu]" in s and "NVIDIA_DRIVER_CAPABILITIES: all" in s, name
    return s


def image_lines(s: str) -> str:
    s = re.sub(r"(?m)^FROM\s+" + IMAGE_RX.pattern, "FROM " + IMAGE, s)
    return re.sub(r"(?m)^(\s*image:\s*)" + IMAGE_RX.pattern, lambda m: m.group(1) + IMAGE, s)


STAGE = "RUN python /opt/robotangle/stage_instance.py {base} {dest} --purge   # the frozen instance, from the image (scripts/robotangle/assets.json), checked against runtime_binding.json"
ENTRY = ('    # The frozen instance comes from the image (scripts/robotangle/assets.json): the entrypoint stages it next to a copy of\n'
         '    # runtime_binding.json and checks it (images/robotangle/stage_instance.py), then runs the command.\n'
         '    entrypoint: ["sh", "-c", "python /opt/robotangle/stage_instance.py {base} ' + STAGED + ' --binding /srv/runtime_binding.json && exec \\"$$@\\"", "stage"]\n')


def staged(name: str, s: str, base: str) -> str:
    """Task files that stage the frozen instance from the image instead of copying it from the task directory."""
    def sub(a: str, b: str) -> str:
        assert s.count(a) == 1, f"{base} {name}: expected one {a!r}"
        return s.replace(a, b)
    if name == "environment/Dockerfile" and "COPY instance.json /app/instance.json\n" in s:
        s = sub("COPY instance.json /app/instance.json\n", "")
        s = sub("COPY instance_assets/ /app/instance_assets/", STAGE.format(base=base, dest="/app"))
    elif name == "tests/Dockerfile" and "instance_assets" in s:
        s = sub("COPY grade.py instance.json rcb_topogym.py test.sh /tests/", "COPY grade.py rcb_topogym.py test.sh /tests/")
        s = sub("COPY instance_assets/ /tests/instance_assets/", STAGE.format(base=base, dest="/tests"))
    elif name == "environment/docker-compose.yaml" and "RCB_INSTANCE: /srv/instance.json" in s:
        s = sub("RCB_INSTANCE: /srv/instance.json", f"RCB_INSTANCE: {STAGED}/instance.json")
        s = sub("    command: sh /srv/run.sh\n", ENTRY.format(base=base) + "    command: sh /srv/run.sh\n")
    return s


MUJOCO_ENV = (("NPY_DISABLE_CPU_FEATURES", "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR"),
              ("OPENBLAS_CORETYPE", "Haswell"))   # docs/simulator/MUJOCO_CORE.md: the containers of a MuJoCo task
NYX_DIRECT = ("blanket_flatten", "blanket_fold", "buzz_wire_3d", "cast_coil", "cast_keyhole", "fiber_bundling",
              "hedgehog_in_cage", "horseshoe_ring", "interlocked_claws", "knot_untying", "lock_picking", "orbits",
              "rope_threading", "slotted_post_ring")   # rcb_topogym.TASKS renderer "direct"


def nyx_direct(rec: dict) -> bool:
    """A base whose family's MuJoCo tasks render with nyx_direct.py: the file ships next to every copy of the tooling and
    the binding names it."""
    return rec["family"] in NYX_DIRECT


def mujoco_lines(name: str, s: str, direct: bool = False) -> str:
    """A MuJoCo task: the SIMD masking in the images that run a Session (privileged agent and verifier: the RoboTangle
    image) and in the standard sim service, and render_bridge.py (with `direct`, nyx_direct.py too) copied next to the
    privileged tooling."""
    def sub(a: str, b: str) -> str:
        assert s.count(a) == 1, f"{name}: expected one {a!r}"
        return s.replace(a, b)
    env = ("# MuJoCo task: numpy / OpenBLAS SIMD masking, so every CPU class generates the frozen bytes (docs/simulator/MUJOCO_CORE.md)\n"
           "ENV " + " ".join(f"{k}={v}" for k, v in MUJOCO_ENV) + "\n")
    if name in ("environment/Dockerfile", "tests/Dockerfile") and f"FROM {IMAGE}\n" in s:
        s = sub(f"FROM {IMAGE}\n", f"FROM {IMAGE}\n" + env)
        if name == "environment/Dockerfile":
            s = sub("COPY rcb_topogym.py /opt/tools/rcb_topogym.py\n",
                    "COPY rcb_topogym.py /opt/tools/rcb_topogym.py\nCOPY render_bridge.py /opt/tools/render_bridge.py\n"
                    + ("COPY nyx_direct.py /opt/tools/nyx_direct.py\n" if direct else ""))
        else:
            s = sub("COPY grade.py rcb_topogym.py test.sh /tests/", "COPY grade.py rcb_topogym.py render_bridge.py "
                    + ("nyx_direct.py " if direct else "") + "test.sh /tests/")
    elif name == "environment/docker-compose.yaml" and "\n  sim:\n" in s:
        s = sub("      RCB_TASK: ", "".join(f"      {k}: {v}\n" for k, v in MUJOCO_ENV) + "      RCB_TASK: ")
    return s


def dev_binding(inst_dir: pathlib.Path, dev: dict, bridge: bool = False, direct: bool = False) -> bytes:
    """--dev: the binding of a local frozen instance (canonical sha256, every asset's sha256) to the dev image's runtime
    and identity, with the same keys as binding()."""
    inst = json.loads((inst_dir / "instance.json").read_text())
    canon = json.dumps(inst, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assets = {p.relative_to(inst_dir).as_posix(): sha(p.read_bytes()) for p in sorted((inst_dir / "instance_assets").rglob("*"))
              if p.is_file() and "__pycache__" not in p.parts}
    lock = dict(schema_version=1, instance_sha256=sha(canon.encode()), asset_files=assets,
                tooling_sha256=sha((SHARED / "rcb_topogym.py").read_bytes()), creation_source=inst["topogym_source"],
                runtime=dev["runtime"], image=dev["image"], uipc_builds=dev["uipc_builds"])
    if bridge:
        lock["render_bridge_sha256"] = sha((SHARED / "render_bridge.py").read_bytes())
    if direct:
        lock["nyx_direct_sha256"] = sha((SHARED / "nyx_direct.py").read_bytes())
    return (json.dumps(lock, indent=2) + "\n").encode()


def render(base: str, out: pathlib.Path, inst_src: pathlib.Path | None = None, dev: dict | None = None) -> list[pathlib.Path]:
    rec = REL["bases"][base]
    dirs = []
    mujoco = is_mujoco(rec) or (inst_src is not None and
                                json.loads((inst_src / "instance.json").read_text())["simulator"].get("physics_engine") == "mujoco")
    reference = pathlib.Path(dev["bases"][base]["reference"]) if dev else None
    for mode in ("privileged", "standard"):
        t = out / f"robotangle-{base}-{mode}"; up = PKG / f"{base}-{mode}"; dirs.append(t)
        rd = lambda n: (up / n).read_text()  # noqa: E731
        put(t / "task.toml", rewrite_toml(rd("task.toml"), base, rec).encode())
        put(t / "README.md", rewrite_readme(rd("README.md"), base, mode, rec).encode())
        instruction = text(rd("instruction.md"))
        if mode == "standard":
            instruction = standard_task_text(instruction)
        instruction = instruction.replace("RoboTangle", "RoboTangle (simulator package `topogym`)", 1)
        put(t / "instruction.md", mujoco_text(instruction, rec).encode())
        for n in ("environment/Dockerfile", "environment/docker-compose.yaml", "tests/Dockerfile", "tests/docker-compose.yaml"):
            if (up / n).is_file():
                text_n = staged(n, gpu_lines(n, image_lines(rd(n))), base)
                put(t / n, (mujoco_lines(n, text_n, nyx_direct(rec)) if mujoco else text_n).encode())
        put(t / "tests/test.sh", (up / "tests/test.sh").read_bytes(), exe=True)
        direct = mujoco and nyx_direct(rec)
        lock = binding(base, inst_src, mujoco, direct) if dev is None else dev_binding(inst_src, dev, mujoco, direct)
        tool = (SHARED / "rcb_topogym.py").read_bytes()
        bridge = (SHARED / "render_bridge.py").read_bytes()
        if mode == "privileged":
            for d in ("environment", "tests"):
                put(t / d / "runtime_binding.json", lock)
                put(t / d / "rcb_topogym.py", tool)
                if mujoco:
                    put(t / d / "render_bridge.py", bridge)
                if direct:
                    put(t / d / "nyx_direct.py", (SHARED / "nyx_direct.py").read_bytes())
            put(t / "tests/grade.py", (SHARED / "privileged/grade.py").read_bytes())
            traj = reference or ORC / base / "privileged.npz"
            if traj.is_file():
                put(t / "solution/trajectory.npz", traj.read_bytes())
                put(t / "solution/solve.sh", (
                    "#!/bin/bash\n# Reference solution (privileged): the trajectory the author's oracle produced and the verifier scored 1\n"
                    "# on the evidence run named in README.md (Reference solution). Uses nothing but the recorded file.\n"
                    "set -euo pipefail\nmkdir -p /app/output\ncp /solution/trajectory.npz /app/output/trajectory.npz\n").encode(), exe=True)
        else:
            for rel, src in LAYER.items():
                put(t / rel, src.read_bytes())
            for n in ("server.py", "replay.py", "finalize.sh"):
                put(t / "environment/sim" / n, (SHARED / "standard/sim" / n).read_bytes())
            put(t / "environment/sim/runtime_binding.json", lock)
            put(t / "environment/sim/rcb_topogym.py", tool)
            if mujoco:
                put(t / "environment/sim/render_bridge.py", bridge)
            if direct:
                put(t / "environment/sim/nyx_direct.py", (SHARED / "nyx_direct.py").read_bytes())
            put(t / "tests/grade.py", (SHARED / "standard/grade.py").read_bytes())
            put(t / "tests/runtime_binding.json", lock)
            traj = reference or ORC / base / "standard.npz"
            if traj.is_file():
                with zipfile.ZipFile(traj) as z:
                    put(t / "solution/oracle_qpos.npy", z.read("actions.npy"))
                put(t / "solution/oracle_meta.json", (json.dumps({"K": 1, "interpolate": "hold", "rows": "the recorded "
                    "joint_position rows of the evidence episode (README.md, Reference solution)"}, indent=1) + "\n").encode())
                put(t / "solution/oracle_replay.py", (SHARED / "standard/solution/oracle_replay.py").read_bytes())
                put(t / "solution/solve.sh", (SHARED / "standard/solution/solve.sh").read_bytes(), exe=True)
    for t in dirs:
        TC.normalize(t)          # leaves the layer alone: scripts/eai_standard/ is a shared source (task_conventions)
    return dirs


def snapshot(d: pathlib.Path) -> dict:
    return {p.relative_to(d).as_posix(): (p.read_bytes(), 0o755 if os.stat(p).st_mode & 0o100 else 0o644)   # exec bit only
            for p in sorted(d.rglob("*")) if p.is_file()}


def dev_main(a) -> int:
    """--dev: render the named bases from dev.json into --out (local only; never tasks/, nothing checked against release.json's
    instance pins)."""
    global IMAGE
    if a.out is None or not a.bases or a.check or a.assets:
        raise SystemExit("--dev needs --out and the bases, and takes neither --check nor --assets")
    dev = json.loads(a.dev.read_text())
    resolve = lambda v: (a.dev.parent / v).resolve()  # noqa: E731  (dev.json paths: absolute, or relative to dev.json)
    dev["bases"] = {b: {k: str(resolve(v)) for k, v in r.items()} for b, r in dev["bases"].items()}
    IMAGE = dev["image"]["reference"]
    a.out.mkdir(parents=True, exist_ok=True)
    for b in a.bases:
        for m in ("privileged", "standard"):
            shutil.rmtree(a.out / f"robotangle-{b}-{m}", ignore_errors=True)
        for t in render(b, a.out, pathlib.Path(dev["bases"][b]["instance"]), dev):
            print(f"dev task {t}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--assets", type=pathlib.Path, help="a local copy of the pinned HF dataset (assets.py fetch)")
    ap.add_argument("--dev", type=pathlib.Path, help="dev.json: local dev task dirs (see usage)")
    ap.add_argument("--out", type=pathlib.Path, help="with --dev: where the dev task dirs go")
    ap.add_argument("bases", nargs="*")
    a = ap.parse_args(argv)
    if a.dev is not None:
        return dev_main(a)
    bases = a.bases or sorted(REL["bases"])
    if a.list:
        for b in bases:
            r = REL["bases"][b]
            print(f"{b:34s} instance {r['creation_source']}  " + "  ".join(f"{m}: {e['release']}:{e['source']}" for m, e in r["evidence"].items()) or "oracle pending")
        return 0
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for b in bases:
            inst = a.assets / "instances" / b if a.assets else None
            if inst is not None and not (inst / "instance.json").is_file():
                bad.append(f"{b}: no frozen instance at {inst}"); continue
            for t in render(b, tmp / "out", inst):
                cur = TASKS / t.name
                if a.check:
                    if not cur.is_dir() or snapshot(cur) != snapshot(t):
                        bad.append(f"tasks/{t.name}: differs from a fresh render (run scripts/robotangle/mk_tasks.py)")
                else:
                    if cur.exists():
                        shutil.rmtree(cur)
                    shutil.copytree(t, cur, symlinks=False)
        known = {f"robotangle-{b}-{m}" for b in REL["bases"] for m in ("privileged", "standard")}
        stray = sorted(p.name for p in TASKS.glob("robotangle-*") if p.name not in known)
        bad += [f"tasks/{n}: not in scripts/robotangle/release.json" for n in stray]
        tools = {sha(p.read_bytes()) for p in TASKS.glob("robotangle-*/**/rcb_topogym.py")}
        if tools - {sha((SHARED / "rcb_topogym.py").read_bytes())}:
            bad.append("a task's rcb_topogym.py differs from scripts/robotangle/shared/rcb_topogym.py")
    for b in bad:
        print("FAIL", b)
    if not bad:
        print(f"ok robotangle: {2 * len(bases)} task dirs {'match a fresh render' if a.check else 'rendered'} "
              f"(release {REL['release_commit'][:9]}, image {IMAGE})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
