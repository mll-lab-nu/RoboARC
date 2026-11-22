#!/usr/bin/env python3
"""Freeze one clearance-relaxed RoboARC instance inside the release image (ablation E4; scripts/ablations/cond_clearance.py
runs it, one container per base and factor, CPU only, no network).

    python /e4/clearance_freeze.py <job.json>

job.json: {"edits": [{"file": "topogym/...py", "sha256": <the image's file>, "old": <text>, "new": <text>}, ...],
"task": <family>, "variation": <variation id>, "gripper_space": "native" | "width", "out": <directory>}

1. The image's simulator source (/opt/topogym: SOURCE.json and topogym/) is copied to /tmp/tg, and each edit's `file`
   there gets its edit old -> new (the image's file must have `sha256`, and `old` must occur in it exactly once).
2. /tmp/tg/SOURCE.json is the copy's identity by the rule of topogym.simulator.frozen_binding.installed_runtime (the
   sha256 of {"topogym/<path>": sha256} over every file but __pycache__), so the patched tree is a bound source like the
   image's own.
3. `rcb_topogym.py make-instance` (the tooling at /tooling, the task directories' copy) builds and freezes the scene from
   the patched source, as the release pipeline makes a base's instance (tools/release/release_rebuild.py MAKE_ONE: the
   variation, scene seed 0, the gripper space, no overrides), with the MuJoCo tasks' SIMD masking set by the caller.
Writes <out>/instance/ (instance.json + instance_assets/), <out>/SOURCE.json and <out>/topogym/<file> (each patched
file), and prints one line `FROZEN {json}`. Standard library + the image's own packages only."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import sys


def tree_source(root: pathlib.Path) -> dict:
    """topogym.simulator.frozen_binding.installed_runtime's source identity of root (root/topogym, no __pycache__)."""
    files = {}
    for p in sorted((root / "topogym").rglob("*")):
        if "__pycache__" in p.parts or p.is_symlink() or not p.is_file():
            continue
        files["topogym/" + p.relative_to(root / "topogym").as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files=len(files))


def main(argv: list[str]) -> int:
    job = json.loads(pathlib.Path(argv[1]).read_text())
    out = pathlib.Path(job["out"])
    tg = pathlib.Path("/tmp/tg")
    shutil.rmtree(tg, ignore_errors=True)
    shutil.copytree("/opt/topogym", tg, ignore=shutil.ignore_patterns("__pycache__"))
    for e in job["edits"]:
        target = tg / e["file"]
        data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() != e["sha256"]:
            raise SystemExit(f"{e['file']}: the image's file is not the one the relaxation was written for")
        text = data.decode()
        if text.count(e["old"]) != 1:
            raise SystemExit(f"{e['file']}: the edited text occurs {text.count(e['old'])} times, not once")
        target.write_text(text.replace(e["old"], e["new"]))
    source = tree_source(tg)
    (tg / "SOURCE.json").write_text(json.dumps(source, indent=1) + "\n")
    os.environ["TOPOGYM_SOURCE"] = str(tg / "SOURCE.json")
    sys.path[:0] = ["/tooling", str(tg)]
    for name in [m for m in sys.modules if m == "topogym" or m.startswith("topogym.")]:
        del sys.modules[name]
    import rcb_topogym as rt
    inst = out / "instance"
    shutil.rmtree(inst, ignore_errors=True)
    inst.mkdir(parents=True)
    sys.argv = ["rcb_topogym.py", "make-instance", "--task", job["task"], "--variation", job["variation"],
                "--scene-seed", "0", "--gripper-space", job["gripper_space"], "--out", str(inst / "instance.json")]
    rt.main()
    made = json.loads((inst / "instance.json").read_text())
    if made.get("topogym_source") != source:
        raise SystemExit("the made instance does not name the patched source")
    import topogym
    if pathlib.Path(topogym.__file__).resolve().parents[1] != tg:
        raise SystemExit(f"topogym was imported from {topogym.__file__}, not from the patched copy")
    (out / "SOURCE.json").write_text(json.dumps(source, indent=1) + "\n")
    for e in job["edits"]:
        patched = out / "topogym" / e["file"].split("/", 1)[1]
        patched.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tg / e["file"], patched)
    print("FROZEN " + json.dumps(dict(state_sha=made["state_sha"], source=source)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
