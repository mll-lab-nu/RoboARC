"""Freeze a redesigned claws tier as a frozen instance (rcb_topogym make-instance in-process, after registering the
tier's variation): python freeze.py <tier> <out_dir>  ->  <out_dir>/instance.json + instance_assets/ (the frozen MuJoCo
scene). Tools that load the instance in other processes need the variation registered there too: put
`scripts/robotangle/redesign/interlocked_claws/overlay` first on PYTHONPATH (its sitecustomize registers all five)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tier")
    ap.add_argument("out_dir")
    a = ap.parse_args(argv)
    from topogym.tasks.interlocked_claws.redesign.task import register
    import rcb_topogym as rt
    vid = register(a.tier)
    out = Path(a.out_dir).resolve() / "instance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    ov = out.parent / "_inputs_overrides.json"
    ov.write_text("{}")
    args = argparse.Namespace(task="interlocked_claws", eval_index=None, variation=vid, scene_seed=0,
                              task_parameters=None, simulator_overrides=str(ov), ipc_settled_state=None,
                              gripper_space="native", out=str(out))
    rt.cmd_make_instance(args)
    inst = json.loads(out.read_text())
    print(json.dumps(dict(tier=a.tier, instance=str(out), state_sha=inst.get("state_sha"))))


if __name__ == "__main__":
    main()
