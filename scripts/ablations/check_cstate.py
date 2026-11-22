#!/usr/bin/env python3
"""Check one generated CS (`cstate`) variant beyond oracle / nop: the static leak check, then the field check against the
simulator (scripts/ablations/cstate_field_check.py, run in the task's own simulator image). Exit 0 when both pass.

    python3 scripts/ablations/check_cstate.py tasks/roboarc-rope-threading-06-i01-cstate-standard
    python3 scripts/ablations/check_cstate.py --gpu 1 tasks/roboarc-*-cstate-standard        # all 18, one after another

The field check builds the frozen scene in a container of the image the task's `sim` service names
(environment/docker-compose.yaml), runs the variant's own service code against it (the robot frame, meta["objects"],
GET /geometry, meta["deformables"]), replays the first reference rows and compares, at control steps 0, 25 and 100,
what the agent would get with the simulator's own state (cstate_field_check.py says how). The chain families (rope,
cord, fibre bundle) run on the CPU; a blanket's cloth steps on a GPU, so the cloth families need one (`--gpu N`, default
the environment's RCB_CHECK_GPU, else 0). About 1-3 minutes per task. It needs Docker and the image (images/pins.json).

Options: --static-only (no Docker: the leak check only); --image REF (another tag of the same image id);
--steps 0,25,100; --json DIR (one report per task, <DIR>/<task>.json); --cloth-on-cpu (development only: runs a blanket on the CPU
twin of its cloth step, which is not the task's physics; never a substitute for the GPU check)."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cond_state_deform as cs  # noqa: E402

ENV_KEEP = ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "RCB_TASK", "PYTHONPATH")


def sim_service(task: pathlib.Path) -> dict:
    """The `sim` service of the task's compose file: its image, its environment (the keys the scene build reads) and the
    base its entrypoint stages."""
    text = (task / "environment" / "docker-compose.yaml").read_text()
    sim = text[text.index("\n  sim:\n"):]
    sim = sim[:re.search(r"\n  [A-Za-z_#]", sim[1:]).start() + 1] if re.search(r"\n  [A-Za-z_#]", sim[1:]) else sim
    image = re.search(r"\n    image: (\S+)", sim).group(1)
    base = re.search(r"stage_instance\.py (\S+) /tmp/robotangle-instance", sim).group(1)
    env = dict(re.findall(r"\n      ([A-Z_]+): \"?([^\"\n#]*?)\"?\s*(?:#[^\n]*)?(?=\n)", sim))
    return {"image": image, "base": base, "env": {k: env[k] for k in ENV_KEEP if k in env}}


def field_check(task: pathlib.Path, gpu: str | None, image: str | None, steps: str, cloth_on_cpu: bool) -> dict:
    svc = sim_service(task)
    family = cs.family_of(task)
    cloth = any(k == "cloth" for k in cs.FAMILIES[family][0].values())
    name = re.sub(r"[^a-z0-9_.-]", "-", task.name.lower())[:60]
    cmd = ["docker", "run", "--rm", "--name", f"cstate-check-{name}-{os.getpid()}", "--cpus", "4", "--memory", "12g",
           "--shm-size", "2g", "--network", "none",
           "-e", "RCB_INSTANCE=/tmp/robotangle-instance/instance.json", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]
    for k, v in svc["env"].items():
        cmd += ["-e", f"{k}={v}"]
    if cloth and not cloth_on_cpu:
        cmd += ["--gpus", f"device={gpu}"]
    cmd += ["-v", f"{task / 'environment' / 'sim'}:/srv:ro", "-v", f"{task}:/task:ro", "-v", f"{HERE}:/ablations:ro",
            "--entrypoint", "sh", image or svc["image"], "-c",
            f"python /opt/robotangle/stage_instance.py {svc['base']} /tmp/robotangle-instance --binding /srv/runtime_binding.json "
            f"--seed-file /srv/seed >/dev/null && exec python -u /ablations/cstate_field_check.py --task /task --steps {steps}"
            + (" --cloth-on-cpu" if cloth_on_cpu else "")]
    p = subprocess.run(cmd, capture_output=True, text=True)
    line = next((x for x in p.stdout.splitlines() if x.startswith("CSTATE_REPORT ")), None)
    if line is None:
        return {"ok": False, "fails": [f"no report from the field check (docker exit {p.returncode})"],
                "stderr": (p.stderr or p.stdout)[-3000:]}
    return json.loads(line[len("CSTATE_REPORT "):])


def check(task: pathlib.Path, a) -> bool:
    meta = tomllib.loads((task / "task.toml").read_text()).get("metadata") or {}
    if meta.get("ablation") not in (None, cs.TOKEN):
        print(f"FAIL {task.name}: not a {cs.TOKEN} variant (ablation {meta.get('ablation')!r})"); return False
    problems = cs.leak_check(task)
    print(f"{'ok  ' if not problems else 'FAIL'} {task.name}: leak check" + (": " + "; ".join(problems) if problems else ""))
    if problems:
        return False
    if a.static_only:
        return True
    rep = field_check(task, a.gpu, a.image, a.steps, a.cloth_on_cpu)
    if a.json:
        a.json.mkdir(parents=True, exist_ok=True)
        (a.json / f"{task.name}.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    worst = {}
    for cp in rep.get("checkpoints", ()):
        for k, v in cp.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, float):
                        worst[f"{k}.{kk}"] = max(worst.get(f"{k}.{kk}", 0.0), vv)
    steps = [cp["step"] for cp in rep.get("checkpoints", ())]
    print(f"{'ok  ' if rep.get('ok') else 'FAIL'} {task.name}: field check at steps {steps}"
          + ("" if not rep.get("cloth_on_cpu") else " (cloth on the CPU twin: a code-path smoke test only)")
          + "".join(f"\n       {k} {v:.3g}" for k, v in sorted(worst.items()))
          + "".join(f"\n       {f}" for f in rep.get("fails", ())) + (f"\n{rep['stderr']}" if rep.get("stderr") else ""),
          flush=True)
    return bool(rep.get("ok"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tasks", type=pathlib.Path, nargs="+", help="generated variant directories tasks/<base>-cstate-standard")
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--gpu", default=os.environ.get("RCB_CHECK_GPU", "0"))
    ap.add_argument("--image", default=None)
    ap.add_argument("--steps", default="0,25,100")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--cloth-on-cpu", action="store_true")
    a = ap.parse_args()
    bad = [t.name for t in a.tasks if not check(t.resolve(), a)]
    if len(a.tasks) > 1:
        print(f"{len(a.tasks) - len(bad)} of {len(a.tasks)} passed" + (": failed " + ", ".join(bad) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
