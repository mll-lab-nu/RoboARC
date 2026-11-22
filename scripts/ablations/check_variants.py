#!/usr/bin/env python3
"""Check every task variant of an ablation experiment before any agent runs on it (docs/ABLATIONS.md, step 1).

    python3 scripts/ablations/check_variants.py run E1 [--gpus 0,1,2,3] [--per-gpu 4] [--render on|off]
    python3 scripts/ablations/check_variants.py gate E1      # exit 0 only if E1's report passes for the variants as they are
    python3 scripts/ablations/check_variants.py show E1      # the report as a table

`run` (bash-runners/ablations/check_variants.sh calls it after building the variants):
1. every variant dir of the experiment exists and its DIGEST is the one this checkout builds and the one locked in
   scripts/ablations/digests/ (the variants we validated); a missing or different one stops the check;
2. per variant, Harbor runs the reference solution (`-a oracle`, must score 1) and the do-nothing agent (`-a nop`, must
   score 0; a task without a reference solution, `[metadata] oracle = "none"`, nop only), each from its own copy of the
   task at seed 0, as the batch runner makes it (scripts/task_seed.py
   materialize, mtimes stamped) and pinned to one GPU (scripts/harbor/pin_gpu.py); up to --per-gpu trials share a GPU,
   a deformable task's trial has one alone (scripts/harbor/gpu_slots.py classify). A trial passes with the expected
   reward, no infrastructure error, the service log present (scripts/roboarc/harbor_check.py's rule) and, for the
   oracle, the task's own determinism check (`deterministic`) and both the live and the replayed success;
3. a condition with a check of its own, scripts/ablations/check_<token>.py or check_<module>.py <variant dir> (E4's
   measured clearance, E3's state fields), must exit 0; an experiment with a GPU check of its own (EXPERIMENT_CHECKS:
   E4's check_clearance_render.py, the relaxed scenes as the cameras render them) runs it once after the trials, and
   it must exit 0 too (its pictures are for a person to look at: send_checks.sh sends them);
4. the report: jobs/ablations/checks/<E>.json (machine-readable) and <E>.md, rewritten after every trial. Running `run`
   again keeps the passed trials of unchanged variants and redoes the rest.
`--render off` checks without a GPU (RCB_RENDER=off and no device reservation, scripts/roboarc/harbor_check.py
render_off; a GPU-physics family keeps its GPU): the maintainers' pre-check; the gate takes only a report with rendering
on. `gate` is what bash-runners/ablations/run_E<n>.sh asks before it starts a batch. Python >= 3.11, standard library."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import pathlib
import shutil
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "scripts"), str(REPO / "scripts" / "harbor"), str(REPO / "scripts" / "roboarc")]
import experiments  # noqa: E402
import gpu_slots  # noqa: E402
import harbor_check  # noqa: E402
import make_variants as mv  # noqa: E402
import task_conventions as tc  # noqa: E402
import task_seed  # noqa: E402

CHECKS = REPO / "jobs" / "ablations" / "checks"
AGENTS = ("oracle", "nop")
# an experiment's check of its own that needs the GPU and rendering (rendering on only), run once after its variants'
# trials: (script, its arguments; {gpu} is the first GPU of --gpus). Its exit code counts like a variant's check; a person
# looks at what it writes (send_checks.sh sends it).
EXPERIMENT_CHECKS = {"E4": ("check_clearance_render.py", ("--out", "jobs/ablations/checks/e4_render", "--gpu", "{gpu}"))}
GPU_PHYSICS_FAMILIES = ("blanket_flatten", "blanket_fold")   # their physics runs on the GPU (rendering off keeps it)
LOCK = threading.Lock()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sh(cmd: list[str], **kw) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, **kw).stdout.strip()


def environment(render: str) -> dict:
    """What the report records about the run: checkout, machine, Harbor, the images the tasks pin."""
    git = lambda *a: sh(["git", "-C", str(REPO), *a])  # noqa: E731
    pins = json.loads((REPO / "images" / "pins.json").read_text())["images"]
    images = {p["ref"]: sh(["docker", "image", "inspect", p["ref"], "--format", "{{.Id}}"]) or None for p in pins}
    try:
        host = json.loads(sh([sys.executable, str(REPO / "scripts" / "harbor" / "host_info.py")]) or "{}")
    except ValueError:
        host = {}
    return {"commit": git("rev-parse", "HEAD") or None, "branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
            "local_changes": bool(git("status", "--porcelain", "--untracked-files=no")), "harbor": sh(["harbor", "--version"]),
            "render": render, "images": images, "host": host}


def current(eid: str) -> list[dict]:
    """The experiment's variants as they are on disk: name, condition, DIGEST (stated, recomputed, locked)."""
    out = []
    for v in experiments.variants(eid):
        d = REPO / "tasks" / v.task
        stated = (d / "DIGEST").read_text().strip() if (d / "DIGEST").is_file() else None
        out.append({"task": v.task, "base": v.base, "condition": v.token, "label": mv.REGISTRY[v.token].label, "mode": v.mode,
                    "exists": d.is_dir(), "digest": tc.digest(d) if d.is_dir() else None, "digest_file": stated,
                    "digest_locked": mv.load_lock(v.token).get(v.task)})
    return out


def has_oracle(v: dict) -> bool:
    """Whether the variant's task has a reference solution for seed 0 (`[metadata] oracle` full or partial, its seeds)."""
    return 0 in task_seed.seeds(REPO / "tasks" / v["task"])["oracle"]


def digest_ok(v: dict) -> bool:
    return bool(v["exists"] and v["digest"] == v["digest_file"] and (v["digest_locked"] in (None, v["digest"])))


class Gpus:
    """Units per GPU: a rigid task's trial takes one, a deformable task's trial all of them (a GPU alone)."""

    def __init__(self, gpus: list[str], share: int):
        self.free = {g: share for g in gpus}
        self.share = share
        self.cv = threading.Condition()

    def take(self, units: int) -> str | None:
        if not self.free:
            return None
        with self.cv:
            while True:
                g = max(self.free, key=lambda k: (self.free[k], -int(k)))
                if self.free[g] >= units:
                    self.free[g] -= units
                    return g
                self.cv.wait()

    def give(self, g: str | None, units: int) -> None:
        if g is None:
            return
        with self.cv:
            self.free[g] += units
            self.cv.notify_all()


def trial(a, eid: str, v: dict, agent: str, gpus: Gpus) -> dict:
    """One Harbor trial of a variant: its own seed-0 copy, pinned to a GPU (rendering on) or rendering off."""
    task = REPO / "tasks" / v["task"]
    job = f"check-{eid.lower()}-{agent}-{v['task']}"
    work = CHECKS / eid / "copies" / job
    jobs = CHECKS / eid / "jobs"
    for p in (work, jobs / job):
        if p.exists():
            shutil.rmtree(p)
    work.mkdir(parents=True)
    jobs.mkdir(parents=True, exist_ok=True)
    family = gpu_slots.classify(task, None, a.per_gpu)[3]
    units = gpu_slots.classify(task, None, a.per_gpu)[0]
    copy = task_seed.materialize(task, 0, work / "tasks")
    if a.registry_mirror:                     # a host that has the pinned images only from a mirror (the same digests)
        for f in [*copy.glob("*/Dockerfile"), *copy.glob("*/docker-compose.yaml")]:
            f.write_text(f.read_text().replace("ghcr.io/mll-lab-nu/", f"{a.registry_mirror}/mll-lab-nu/"))
    gpu = None
    if a.render == "off":
        off = work / "off" / copy.name
        off.parent.mkdir(parents=True)
        harbor_check.render_off(copy, off, keep_gpu=family in GPU_PHYSICS_FAMILIES)
        copy = off
        units = units if family in GPU_PHYSICS_FAMILIES else 0
    if units:
        gpu = gpus.take(units)
        if gpu is not None:
            pin = [sys.executable, str(REPO / "scripts/harbor/pin_gpu.py"), "--gpu", gpu, str(copy)]
            subprocess.run(pin, check=True, capture_output=True)
            subprocess.run(pin[:-1] + ["--check", str(copy)], check=True, capture_output=True)
    subprocess.run([sys.executable, str(REPO / "scripts/stamp_mtimes.py"), "--full", str(copy.parent)], check=True,
                   capture_output=True)
    cmd = [*a.harbor.split(), "run", "-p", str(copy), "-a", agent, "-o", str(jobs), "--job-name", job, "-n", "1", "-k", "1", "-y"]
    t0 = time.time()
    try:
        with open(CHECKS / eid / "logs" / f"{job}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=work).returncode
    finally:
        gpus.give(gpu, units)
    r = dict(job=job, agent=agent, rc=rc, gpu=gpu, wall_s=round(time.time() - t0, 1), finished=now(),
             expected=harbor_check.EXPECTED[agent], **harbor_check.reward_of(jobs / job))
    rw = r.get("rewards") or {}
    ok = (r.get("reward") is not None and float(r["reward"]) == r["expected"] and int(rw.get("infra_error") or 0) == 0
          and int(rw.get("missing_log") or 0) == 0)
    if agent == "oracle":
        for k in ("deterministic", "live_success", "replay_success"):
            if k in rw and float(rw[k]) != 1.0:
                ok = False
                r.setdefault("why", []).append(f"{k}={rw[k]}")
    r["pass"] = ok
    r["evidence"] = {k: rw[k] for k in harbor_check.EVIDENCE if k in rw}
    r.pop("rewards", None)
    shutil.rmtree(work, ignore_errors=True)
    return r


def extra_check(v: dict) -> dict | None:
    """The condition's own check, if it has one: scripts/ablations/check_<token>.py or check_<module>.py <variant dir>."""
    script = next((s for s in (HERE / f"check_{v['condition']}.py", HERE / f"check_{mv.REGISTRY[v['condition']].module}.py")
                   if s.is_file()), None)
    if script is None:
        return None
    p = subprocess.run([sys.executable, str(script), str(REPO / "tasks" / v["task"])], capture_output=True, text=True)
    return {"script": f"scripts/ablations/{script.name}", "rc": p.returncode, "pass": p.returncode == 0,
            "output": (p.stdout + p.stderr)[-2000:]}


def load(eid: str) -> dict:
    p = CHECKS / f"{eid}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def save(eid: str, rep: dict) -> None:
    rep["variants"] = sorted(rep["variants"], key=lambda v: v["task"])
    for v in rep["variants"]:
        v["pass"] = bool(digest_ok(v) and all((v.get(a) or {}).get("pass") for a in AGENTS)
                         and (v.get("extra") is None or v["extra"].get("pass")))
    rep["n_variants"], rep["n_pass"] = len(rep["variants"]), sum(v["pass"] for v in rep["variants"])
    rep["all_pass"] = (rep["n_variants"] == len(experiments.variants(eid)) and rep["n_pass"] == rep["n_variants"]
                       and all(x.get("pass") for x in (rep.get("experiment_checks") or {}).values())
                       and (eid not in EXPERIMENT_CHECKS or rep.get("environment", {}).get("render") != "on"
                            or bool(rep.get("experiment_checks"))))
    rep["updated"] = now()
    tmp = CHECKS / f"{eid}.json.tmp"
    tmp.write_text(json.dumps(rep, indent=1) + "\n")
    tmp.replace(CHECKS / f"{eid}.json")
    (CHECKS / f"{eid}.md").write_text(table(rep))


def table(rep: dict) -> str:
    e = rep.get("environment", {})
    lines = [f"# {rep['experiment']}: variant checks", "",
             f"Checkout {e.get('commit')} ({e.get('branch')}{', local changes' if e.get('local_changes') else ''}), "
             f"harbor {e.get('harbor')}, rendering {e.get('render')}; updated {rep.get('updated')}.", "",
             f"**{'ALL PASS' if rep.get('all_pass') else 'NOT PASSED'}**: {rep.get('n_pass')} / "
             f"{len(experiments.variants(rep['experiment']))} variants pass.", "",
             "| variant | condition | DIGEST as locked | oracle (1) | nop (0) | own check | pass | oracle wall s |",
             "|---|---|---|---|---|---|---|---|"]
    for v in rep["variants"]:
        o, n, x = v.get("oracle") or {}, v.get("nop") or {}, v.get("extra")
        lines.append(f"| {v['task']} | {v['label']} | {'yes' if digest_ok(v) else 'NO'} | {o.get('reward', '-')}"
                     f"{'' if o.get('pass', True) else ' FAIL'} | {n.get('reward', '-')}{'' if n.get('pass', True) else ' FAIL'} | "
                     f"{'-' if x is None else ('ok' if x.get('pass') else 'FAIL')} | {'yes' if v.get('pass') else 'NO'} | "
                     f"{o.get('wall_s', '-')} |")
    for name, x in (rep.get("experiment_checks") or {}).items():
        lines += ["", f"Experiment check `scripts/ablations/{name}`: {'pass' if x.get('pass') else 'FAIL'} (exit {x.get('rc')}); "
                      f"its output: `{x.get('out')}`."]
    return "\n".join(lines) + "\n"


def run(a) -> int:
    eid = a.experiment
    for d in ("logs", "jobs", "copies"):
        (CHECKS / eid / d).mkdir(parents=True, exist_ok=True)
    cur = current(eid)
    bad = [v["task"] for v in cur if not digest_ok(v)]
    if bad:
        print(f"{len(bad)} variant(s) missing or not as validated (DIGEST): {', '.join(bad[:5])}{' ...' if len(bad) > 5 else ''}\n"
              f"build them first: python3 scripts/ablations/make_variants.py build --experiment {eid}", file=sys.stderr)
        return 2
    prior = load(eid)
    old = {v["task"]: v for v in prior.get("variants", []) if prior.get("environment", {}).get("render") == a.render}
    rep = {"experiment": eid, "purpose": experiments.EXPERIMENTS[eid].purpose, "created": prior.get("created") or now(),
           "environment": {**environment(a.render), "registry_mirror": a.registry_mirror}, "expected": {"oracle": 1, "nop": 0},
           "variants": []}
    todo = []
    for v in cur:
        prev = old.get(v["task"])
        entry = dict(v)
        for agent in AGENTS:
            if agent == "oracle" and not has_oracle(v):
                entry[agent] = {"agent": agent, "skipped": True, "pass": True,
                                "why": "the task has no reference solution for seed 0 ([metadata] oracle): nop only"}
                continue
            if prev and prev.get("digest") == v["digest"] and (prev.get(agent) or {}).get("pass"):
                entry[agent] = prev[agent]            # an unchanged variant's passed trial is kept
            else:
                todo.append((v, agent))
        keep = prev and prev.get("digest") == v["digest"] and (prev.get("extra") or {}).get("pass")
        entry["extra"] = prev["extra"] if keep else extra_check(v)   # an unchanged variant's passed own check is kept
        rep["variants"].append(entry)
    save(eid, rep)
    by = {v["task"]: v for v in rep["variants"]}
    gpus = Gpus([g for g in a.gpus.split(",") if g != ""] if a.render == "on" or a.gpus else [], a.per_gpu)
    if a.render == "on" and not gpus.free:
        print("--render on needs GPUs (--gpus, default: every GPU nvidia-smi lists)", file=sys.stderr)
        return 2
    order = sorted(todo, key=lambda x: (x[1] != "oracle", x[0]["task"]))      # oracles first: the long ones
    skipped = sum(1 for v in rep["variants"] if (v.get("oracle") or {}).get("skipped"))
    print(f"{eid}: {len(cur)} variants, {len(todo)} trials to run ({len(cur) * 2 - len(todo) - skipped} kept"
          f"{f', {skipped} without a reference solution: nop only' if skipped else ''}), rendering {a.render}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=a.lanes) as pool:
        futs = {pool.submit(trial, a, eid, v, agent, gpus): (v, agent) for v, agent in order}
        for f in cf.as_completed(futs):
            v, agent = futs[f]
            try:
                r = f.result()
            except Exception as exc:  # noqa: BLE001 -- a trial that could not even start is a failed check, recorded
                r = {"agent": agent, "pass": False, "error": repr(exc), "finished": now()}
            with LOCK:
                by[v["task"]][agent] = r
                save(eid, rep)
            print(f"{v['task']} {agent}: reward {r.get('reward')} -> {'pass' if r.get('pass') else 'FAIL'} "
                  f"({r.get('wall_s', '-')} s)", flush=True)
    if eid in EXPERIMENT_CHECKS and a.render == "on":
        name, args = EXPERIMENT_CHECKS[eid]
        prev_x = (prior.get("experiment_checks") or {}).get(name) or {}
        unchanged = prev_x.get("pass") and prev_x.get("digests") == {v["task"]: v["digest"] for v in cur}
        if unchanged:
            rep.setdefault("experiment_checks", {})[name] = prev_x
        else:
            first = next((g for g in a.gpus.split(",") if g), "0")
            argv = [x.replace("{gpu}", first) for x in args]
            print(f"{eid}: scripts/ablations/{name} {' '.join(argv)}", flush=True)
            p = subprocess.run([sys.executable, str(HERE / name), *argv], capture_output=True, text=True, cwd=REPO)
            rep.setdefault("experiment_checks", {})[name] = {
                "rc": p.returncode, "pass": p.returncode == 0, "finished": now(), "args": argv,
                "out": argv[argv.index("--out") + 1] if "--out" in argv else None,
                "digests": {v["task"]: v["digest"] for v in cur}, "output": (p.stdout + p.stderr)[-3000:]}
            print((p.stdout + p.stderr)[-3000:], flush=True)
        save(eid, rep)
    print(table(rep))
    print(f"report: {CHECKS / (eid + '.json')}")
    return 0 if rep["all_pass"] else 1


def gate(eid: str, quiet: bool = False) -> int:
    """0: the experiment's report says every variant passed with rendering on, for the variants exactly as they are."""
    rep = load(eid)
    why = []
    if not rep:
        why.append(f"no report jobs/ablations/checks/{eid}.json: run bash-runners/ablations/check_variants.sh {eid} first")
    else:
        if rep.get("environment", {}).get("render") != "on":
            why.append("the report was made with rendering off: run the check on this GPU machine")
        got = {v["task"]: v for v in rep.get("variants", [])}
        for v in current(eid):
            r = got.get(v["task"])
            if r is None:
                why.append(f"{v['task']}: not checked")
            elif not r.get("pass"):
                why.append(f"{v['task']}: did not pass")
            elif not digest_ok(v) or r.get("digest") != v["digest"]:
                why.append(f"{v['task']}: changed since its check (DIGEST)")
        if eid in EXPERIMENT_CHECKS:
            x = (rep.get("experiment_checks") or {}).get(EXPERIMENT_CHECKS[eid][0])
            if not (x and x.get("pass")):
                why.append(f"scripts/ablations/{EXPERIMENT_CHECKS[eid][0]} has not passed")
            elif x.get("digests") != {v["task"]: v["digest"] for v in current(eid)}:
                why.append(f"scripts/ablations/{EXPERIMENT_CHECKS[eid][0]} ran on other variants (DIGEST)")
    if why and not quiet:
        print(f"{eid} may not start: {len(why)} problem(s)\n  " + "\n  ".join(why[:20]), file=sys.stderr)
    return 1 if why else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("run", "gate", "show"))
    ap.add_argument("experiment", choices=tuple(experiments.EXPERIMENTS))
    ap.add_argument("--render", choices=("on", "off"), default="on")
    ap.add_argument("--gpus", default=None, help="GPU indices, comma-separated (default: every GPU nvidia-smi lists)")
    ap.add_argument("--per-gpu", type=int, default=4, help="rigid-task trials per GPU (a deformable task's trial: one GPU)")
    ap.add_argument("--lanes", type=int, default=None, help="trials at once (default: GPUs x --per-gpu; --render off: 3)")
    ap.add_argument("--harbor", default="harbor")
    ap.add_argument("--registry-mirror", default="", help=argparse.SUPPRESS)   # maintainers' hosts: images from a mirror
    a = ap.parse_args(argv)
    if a.cmd == "gate":
        return gate(a.experiment)
    if a.cmd == "show":
        rep = load(a.experiment)
        print(table(rep) if rep else f"no report for {a.experiment}")
        return 0 if rep else 1
    if a.gpus is None:
        a.gpus = sh(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]).replace(" ", "").replace("\n", ",") \
            if a.render == "on" else ""
    if a.lanes is None:
        a.lanes = max(1, len([g for g in a.gpus.split(",") if g]) * a.per_gpu) if a.render == "on" else 3
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
