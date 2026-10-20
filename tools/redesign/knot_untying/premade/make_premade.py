"""The knot_untying rd bases ship premade frozen instances (rope rd's precedent; docs/rcb/HARBOR_FREEZE_RUNBOOK.md "Bases
made elsewhere"): make one INSIDE the release image (bound in its runtime) from the exact inputs of a frozen source
instance: its task parameters, its rope birth archive and its grasp, as build_instance.py passed them to make-instance.
Nothing is derived again (no lab rebuild, no birth sweep). knot-untying-06-i01 = inst10/very_easy, knot-untying-09-i01 =
inst10/medium_loose (their instance.json sha256 and birth archive sha256 are in mujoco_release.json's notes). Run it
twice and compare the two outputs byte for byte; the state must equal the source's.
What the made instance does not carry (2026-10-17, V2 release leak probe: a privileged agent reads instance.json and its
assets in /app, and the Session holds the task parameters): the source's task_parameters["redesign"]["source"] (the lab
start's provenance, which task_rd.py never reads: its oracle plan, "order" / "pin" / "layout_path", its notes, the lab
path), the lab_start/ copy (provenance, read by nothing; its scene_start.json repeats the oracle plan), and internal
host paths (redesign_build.lab_start, the birth manifest's metadata note). They are recorded as hashes only
(redesign_build.premade.dropped). The birth arrays, the scene (frame, ropes, fixtures, regions) and the grasp are the
source's, unchanged, so the state must still equal the source's.
usage (in the release noinst image, its shared tooling at /tooling): python make_premade.py <source instance dir> <out dir>"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/tooling")
import rcb_topogym as rt  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


src, out = Path(sys.argv[1]), Path(sys.argv[2])
inst = json.loads((src / "instance.json").read_text())
if out.exists():
    shutil.rmtree(out)
(out / "instance_assets").mkdir(parents=True)
# the birth archive the override names (its arrays byte for byte; the manifest without the lab's host path in its
# metadata note); no lab_start/ (provenance only, never read; it repeats the oracle plan)
shutil.copytree(src / "instance_assets" / "rope_birth", out / "instance_assets" / "rope_birth")
birth = out / "instance_assets" / "rope_birth" / "manifest.json"
src_birth = json.loads(birth.read_text())
note = (src_birth.get("metadata") or {}).get("note")
if note and "(" in note:
    src_birth["metadata"]["note"] = note.split(" (", 1)[0]
    birth.write_text(json.dumps(src_birth, indent=2) + "\n")
work = out.parent / (out.name + ".mk")
work.mkdir(exist_ok=True)
tp, ov = work / "task_parameters.json", work / "overrides.json"
params = json.loads(json.dumps(inst["simulator"]["task_parameters"]))
dropped_source = params["redesign"].pop("source", None)      # provenance; task_rd.py reads frame/ropes/fixtures/regions
tp.write_text(json.dumps(params))
ov.write_text(json.dumps({"mujoco_birth": str(birth), "grasp_assist": inst["simulator"]["grasp_assist"]}))
sys.argv = ["rcb_topogym.py", "make-instance", "--task", inst["task"], "--variation", inst["variation_id"],
            "--scene-seed", str(inst["seed"]), "--gripper-space", inst["gripper_space"], "--task-parameters", str(tp),
            "--simulator-overrides", str(ov), "--out", str(out / "instance.json")]
print(" ".join(sys.argv[1:]), flush=True)
rt.main()
new = json.loads((out / "instance.json").read_text())
new["simulator"]["mujoco_birth"] = "instance_assets/rope_birth/manifest.json"     # as build_instance.py records it
frozen = out / "instance_assets" / "frozen_assets"
frozen.mkdir()
for f in sorted((src / "instance_assets" / "frozen_assets").iterdir()):
    shutil.copy2(f, frozen / f.name)
    if new["asset_identity"].get(f.name) != sha(f):
        raise SystemExit(f"{f.name}: the source's generated volume differs from the one this make declared")
build = {k: v for k, v in (inst.get("redesign_build") or {}).items() if k != "lab_start"}   # no host path
canon = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()  # noqa: E731
new["redesign_build"] = dict(build, premade=dict(
    tool="scripts/robotangle/redesign/knot_untying/premade/make_premade.py (inside the release image)",
    source_state_sha=inst["state_sha"], source_instance_sha256=sha(src / "instance.json"), task_parameters_sha256=sha(tp),
    dropped=dict(task_parameters_redesign_source_sha256=canon(dropped_source) if dropped_source else None,
                 lab_start_sha256={f.name: sha(f) for f in sorted((src / "instance_assets" / "lab_start").iterdir())}
                 if (src / "instance_assets" / "lab_start").is_dir() else None,
                 redesign_build_lab_start=bool((inst.get("redesign_build") or {}).get("lab_start")),
                 birth_manifest_sha256=sha(src / "instance_assets" / "rope_birth" / "manifest.json")),
    birth_sha256={f.name: sha(f) for f in sorted((out / "instance_assets" / "rope_birth").iterdir())},
    grasp_assist_sha256=hashlib.sha256(json.dumps(inst["simulator"]["grasp_assist"], sort_keys=True).encode()).hexdigest()))
(out / "instance.json").write_text(json.dumps(new, indent=1))
same = {k: new.get(k) == inst.get(k) for k in sorted(set(new) | set(inst))}
print("RESULT " + json.dumps(dict(state_sha=new["state_sha"], source_state_sha=inst["state_sha"],
                                  equal_state=new["state_sha"] == inst["state_sha"],
                                  differing_keys=[k for k, v in same.items() if not v],
                                  task_parameters_sha256=sha(tp))), flush=True)
