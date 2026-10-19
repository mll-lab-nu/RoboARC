"""make-instance in-process for a buzz_wire_3d variation (a redesign tier, registered in-process, or a released tier):
<out>/instance.json + instance_assets/ (the frozen MuJoCo scene). usage: freeze.py <variation_id> <out instance.json>"""
import argparse, json, sys
from pathlib import Path
import rdreg

ap = argparse.ArgumentParser()
ap.add_argument("variation"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rdreg.register()
import rcb_topogym as rt
out = Path(a.out).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
ns = argparse.Namespace(task="buzz_wire_3d", eval_index=None, variation=a.variation, scene_seed=a.seed,
                        task_parameters=None, simulator_overrides=None, ipc_settled_state=None, gripper_space="native",
                        out=str(out))
rt.cmd_make_instance(ns)
inst = json.loads(out.read_text())
print(json.dumps(dict(instance=str(out), state_sha=inst.get("state_sha"), variation=inst["simulator"]["variation_id"])))
