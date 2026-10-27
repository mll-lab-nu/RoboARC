#!/usr/bin/env python3
"""Refresh a frozen MuJoCo instance whose task identity changed but whose physics did not (RoboARC 2.1.0: the blanket
families' terminal revision, release_settling_v1 -> v2_active_parts, enters the task's instance_id and with it the birth's
state hash), without making the scene again.

The refreshed instance is the old one with
  - instance_assets/ copied byte for byte (the frozen MuJoCo scene, the birth and settled-state archives, the frozen
    generated assets);
  - the fields a build computes recomputed by building that frozen scene in this image (as derive_seed_instance.py does
    for an appearance seed): state_sha, asset_identity, effective_task_parameters, text, public_goal, success_rule,
    scene_variation, evaluable_for_task_success;
  - refreshed_from: the old state_sha and canonical sha256, the source that refreshed it and this tool.
topogym_source stays the old one: the binding's creation_source names the source that made the frozen scene and births.
The refresh stops unless the physical fields (objects, soft_state, reset_joint_targets, action layout, gripper space,
control rate, resolved variation) equal the old instance's and instance_assets/ is the old one's byte for byte. The
birth's tick record is printed by part (each part's sha16, as tools/release/state_parts.py): compare it with the old
instance's birth in the old image (state_parts.py birth) to see which parts the identity change touched.

usage, inside the image of the source the instance is refreshed for (as make-instance; the shared tooling at --tooling):
  python refresh_instance.py <instance dir> <out dir> [--tooling /tooling] [--check]
    writes <out>/instance.json and <out>/instance_assets/; one line `REFRESHED {json}` on stdout
    --check  load the written instance again in a fresh process (Session strict: the state and assets reproduce)
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import derive_seed_instance as dsi  # noqa: E402

CHECK = """import json, pathlib, sys
sys.path[:0] = [sys.argv[1], sys.argv[3]]
import rcb_topogym as rt
import derive_seed_instance as d
s = d.build(rt, pathlib.Path(sys.argv[2]))
print("CHECK " + json.dumps(dict(instance_exact=s.instance_exact, assets_exact=s.assets_exact, state_sha=s.instance_sha)), flush=True)
s.close()
"""


def refresh(rt, src: pathlib.Path, out: pathlib.Path) -> dict:
    from topogym.tasks.goal_description import TASK_IDS, describe_goal
    from topogym.utils.io import to_jsonable
    old = json.loads((src / "instance.json").read_text())
    if old["simulator"].get("physics_engine") != "mujoco":
        raise SystemExit(f"{src}: a frozen MuJoCo instance is needed")
    tmp = out.parent / f".{out.name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        shutil.copytree(src / "instance_assets", tmp / "instance_assets")
        (tmp / "instance.json").write_text(json.dumps(dict(
            {k: v for k, v in old.items() if k not in ("state_sha", "asset_identity")},
            derived_from=dict(frozen_identity=old.get("asset_identity") or {}))))
        s = dsi.build(rt, tmp / "instance.json")
        try:
            task = s.env._task
            eff = to_jsonable(task.config)
            for k in ("initial_state_path", "fold_target_path"):    # instance-relative, as make-instance records them
                if k in eff:
                    eff[k] = old["effective_task_parameters"][k]
            got = json.loads(json.dumps(to_jsonable(dict(
                state_sha=s.instance_sha, asset_identity=s.asset_identity, effective_task_parameters=eff, text=s.spec["text"],
                public_goal=describe_goal(task) if task.task_id in TASK_IDS else None, success_rule=rt.success_rule(s),
                scene_variation=s.env._scene_variation, evaluable_for_task_success=rt.success_enabled(task),
                objects=s.objects(), soft_state=s._soft_state(),
                reset_joint_targets=[round(float(x), 6) for x in s.joint_positions()], action_dim=s.action_dim,
                action_layout=s.action_layout, gripper_space=s.gripper_space, control_hz=s.control_hz,
                resolved_variation_id=s.env._variation["variation_id"]))))
            rec = dsi.birth_record(rt, s)
            if dsi.sha16(rec) != s.instance_sha:
                raise SystemExit(f"{src}: the birth record is not the one Session.state_sha hashes")
        finally:
            s.close()
        bad = [k for k in dsi.PHYSICAL if got[k] != old.get(k)]
        if bad:
            raise SystemExit(f"{src}: physical fields differ from the old instance: {bad}")
        if dsi.files_of(tmp / "instance_assets") != dsi.files_of(src / "instance_assets"):
            raise SystemExit(f"{src}: instance_assets/ is not the old one's byte for byte")
        new = copy.deepcopy(old)
        new.update({k: got[k] for k in dsi.RECOMPUTED})
        new["refreshed_from"] = dict(state_sha=old["state_sha"], instance_sha256=dsi.canonical(old), source=rt._source(),
                                     tool="tools/release/refresh_instance.py")
        new = json.loads(json.dumps(to_jsonable(new)))
        (tmp / "instance.json").write_text(json.dumps(new, indent=1))
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dict(state_sha=got["state_sha"], old_state_sha=old["state_sha"], instance_sha256=dsi.canonical(new),
                parts={k: dsi.sha16(v) for k, v in sorted(rec.items())},
                fields_changed=sorted(k for k in dsi.RECOMPUTED if got[k] != old.get(k)))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--tooling", default="/tooling")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.tooling)
    import rcb_topogym as rt
    rec = refresh(rt, a.src.resolve(), a.out.resolve())
    print("REFRESHED " + json.dumps(rec), flush=True)
    if a.check:
        path = a.out.resolve() / "instance.json"
        r = subprocess.run([sys.executable, "-c", CHECK, a.tooling, str(path), str(pathlib.Path(__file__).resolve().parent)],
                           capture_output=True, text=True, cwd="/tmp")
        line = next((x for x in r.stdout.splitlines() if x.startswith("CHECK ")), None)
        if r.returncode or line is None or json.loads(line[6:])["state_sha"] != rec["state_sha"]:
            sys.stderr.write(r.stdout[-2000:] + r.stderr[-4000:])
            raise SystemExit(f"{path}: does not load again as written (Session strict)")
        print("CHECK " + line[6:], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
