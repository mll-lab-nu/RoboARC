"""roboarc selftest: tasks end to end through Harbor without a model. The reference solution (Harbor's `oracle`
agent) must score 1 and doing nothing (`nop`) must score 0. Each trial is the steps a hand-written check uses:
scripts/task_seed.py materialize (a copy at the seed, file times fixed), scripts/harbor/clamp_cpus.py, optionally
scripts/harbor/pin_gpu.py, then `harbor run`; the score is Harbor's verifier/reward.txt. Standard library only."""
from __future__ import annotations

import ast
import queue
import shutil
import struct
import subprocess
import threading
import time
import zipfile
from pathlib import Path

from . import _common as C

EXPECTED = {"oracle": 1.0, "nop": 0.0}      # Harbor's agents: the task's own solution, and an agent that does nothing


def reference_rows(path) -> int:
    """The number of rows (control steps) of a reference trajectory: an .npz whose `actions` (puzzles) or `ctrl` (IKEA)
    array has one row per step. Read from the array's header, without numpy."""
    with zipfile.ZipFile(path) as z:
        name = "actions.npy" if "actions.npy" in z.namelist() else "ctrl.npy"
        with z.open(name) as f:
            if f.read(6) != b"\x93NUMPY":
                raise ValueError(f"{path}: {name} is not a numpy array")
            major = f.read(2)[0]
            (n,) = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))
            return int(ast.literal_eval(f.read(n).decode("latin1"))["shape"][0])


def family_of(task) -> str:
    """The family a task is checked for: its puzzle family; the IKEA scenes together are one, `ikea`."""
    return "ikea" if task.suite == "ikea" else task.family


def families() -> dict[str, list]:
    from ..registry import list_tasks
    out: dict[str, list] = {}
    for t in list_tasks():
        out.setdefault(family_of(t), []).append(t)
    return out


def shortest(tasks: list):
    """The task with the shortest reference solution (fewest rows; the id breaks a tie)."""
    with_ref = [t for t in tasks if t.reference]
    return min(with_ref or tasks, key=lambda t: (reference_rows(t.reference) if t.reference else 0, t.id))


def select(names: list[str], family_arg: str | None) -> list:
    """The tasks to check: the ones named, else the shortest-reference task of each family asked for."""
    from ..registry import resolve
    if names:
        if family_arg:
            raise C.CliError("selftest: give tasks or --families, not both")
        return [resolve(n) for n in names]
    known = families()
    want = C.csv(family_arg) or ["all"]
    if want == ["all"]:
        want = sorted(known)
    picked = []
    for f in want:
        key = f.replace("-", "_")
        if key not in known:
            raise C.CliError(f"selftest: no family {f!r}; the families: all, {', '.join(sorted(known))}")
        picked.append(shortest(known[key]))
    return picked


def plan(a) -> list[dict]:
    modes, agents, seeds = C.csv(a.modes), C.csv(a.agents), C.csv(a.seeds)
    for name, given, allowed in (("--modes", modes, C.MODES), ("--agents", agents, tuple(EXPECTED))):
        bad = [x for x in given if x not in allowed]
        if bad or not given:
            raise C.CliError(f"selftest: {name} takes {', '.join(allowed)} (comma-separated); got "
                             f"{','.join(given) or 'nothing'}")
    if not seeds or not all(s.isdigit() for s in seeds):
        raise C.CliError("selftest: --seeds takes seeds, comma-separated, e.g. 0 or 0,1,2")
    trials = []
    for t in select(a.tasks, a.families):
        for mode in modes:
            d = t.task_dirs.get(mode)
            if d is None:
                raise C.CliError(f"selftest: {t.id} has no {mode} task directory")
            for agent in agents:
                for seed in seeds:
                    name = Path(d).name
                    trials.append({"job": f"selftest-{name}-{agent}-s{seed}", "task": name, "dir": d, "mode": mode,
                                   "agent": agent, "seed": int(seed), "expected": EXPECTED[agent]})
    return trials


def reward_of(job_dir: Path) -> float | None:
    """The score Harbor's verifier wrote for the job's one trial (<job>/<trial>/verifier/reward.txt), or None."""
    for f in sorted(job_dir.glob("*/verifier/reward.txt")):
        try:
            return float(f.read_text().strip())
        except ValueError:
            return None
    return None


def run_trial(tr: dict, jobs: Path, gpu: str | None) -> dict:
    """Run one trial unless its job already has a score; returns the trial with reward, ok, seconds and a note."""
    job_dir, log, t0 = jobs / tr["job"], jobs / f"{tr['job']}.log", time.time()
    out = dict(tr, reward=reward_of(job_dir), seconds=0, note="")
    if out["reward"] is not None:
        out["note"] = "scored by an earlier run"
    elif job_dir.exists():
        out["note"] = (f"{job_dir} exists without a score (an earlier run that did not finish): move it away to run "
                       "the trial again")
    else:
        work = jobs / ".selftest" / tr["job"]
        shutil.rmtree(work, ignore_errors=True)          # the copy of a run that was interrupted
        steps = [C.command("scripts/task_seed.py", "materialize", tr["dir"], "--seed", tr["seed"], "--out", work),
                 C.command("scripts/harbor/clamp_cpus.py", work / tr["task"])]
        if gpu is not None:
            steps.append(C.command("scripts/harbor/pin_gpu.py", "--gpu", gpu, work / tr["task"]))
        steps.append(["harbor", "run", "-p", str(work), "-i", tr["task"], "-a", tr["agent"], "-n", "1", "-k", "1", "-y",
                      "-o", str(jobs), "--job-name", tr["job"]])
        with open(log, "w") as f:
            for cmd in steps:
                f.write("+ " + C.quoted(cmd) + "\n")
                f.flush()
                try:
                    rc = C.run(cmd, stdout=f, stderr=subprocess.STDOUT)
                except FileNotFoundError:
                    rc, out["note"] = 127, f"{cmd[0]} not found (roboarc doctor)"
                if rc != 0:
                    step = "harbor" if cmd[0] == "harbor" else Path(cmd[1]).name
                    out["note"] = out["note"] or f"{step} exit {rc}: {log}"
                    break
        shutil.rmtree(work, ignore_errors=True)
        out["reward"], out["seconds"] = reward_of(job_dir), round(time.time() - t0)
        if out["reward"] is None and not out["note"]:
            out["note"] = f"no verifier/reward.txt under {job_dir}: {log}"
    out["ok"] = out["reward"] == tr["expected"]
    return out


def _selftest(a) -> int:
    trials = plan(a)
    jobs = Path(a.jobs).resolve() if a.jobs else C.repo() / "jobs" / "selftest"
    gpus = C.csv(a.gpus)
    if any(not g.isdigit() for g in gpus):
        raise C.CliError("selftest: --gpus takes GPU indices, comma-separated, e.g. 0 or 0,1")
    n_tasks = len({t["task"].rsplit("-", 1)[0] for t in trials})
    print(f"selftest: {len(trials)} trials over {n_tasks} task(s), {a.lanes} at a time"
          + (f", on GPU {','.join(gpus)}" if gpus else "") + f"; jobs and logs in {jobs}")
    if a.print:
        for t in trials:
            print(f"  {t['job']}: {t['task']}, agent {t['agent']}, seed {t['seed']}, expected score {t['expected']:g}")
        return 0
    jobs.mkdir(parents=True, exist_ok=True)
    todo: queue.Queue = queue.Queue()
    for t in trials:
        todo.put(t)
    done, lock, stop = [], threading.Lock(), threading.Event()

    def lane(k: int) -> None:
        gpu = gpus[k % len(gpus)] if gpus else None
        while not stop.is_set():
            try:
                tr = todo.get_nowait()
            except queue.Empty:
                return
            try:
                r = run_trial(tr, jobs, gpu)
            except Exception as e:           # reported as this trial's failure; the lane goes on
                r = dict(tr, reward=None, ok=False, seconds=0, note=f"{type(e).__name__}: {e}")
            with lock:
                done.append(r)
                score = "none" if r["reward"] is None else f"{r['reward']:g}"
                print(f"[{len(done)}/{len(trials)}] {'ok  ' if r['ok'] else 'FAIL'}  {r['task']}  {r['agent']}  "
                      f"seed {r['seed']}  score {score} (expected {r['expected']:g})  {r['seconds']} s"
                      + (f"  {r['note']}" if r["note"] else ""), flush=True)

    threads = [threading.Thread(target=lane, args=(k,), daemon=True) for k in range(min(a.lanes, len(trials)))]
    for t in threads:
        t.start()
    try:
        for t in threads:
            while t.is_alive():
                t.join(0.5)
    except KeyboardInterrupt:        # Ctrl-C reaches the running harbor too; no further trial starts
        stop.set()
        print("selftest: interrupted. Scored trials are kept: to continue, move away the job directory of the trial that "
              f"was running (under {jobs}) and run the same command again")
        return 130
    try:
        (jobs / ".selftest").rmdir()        # the copies' directory, empty again
    except OSError:
        pass
    bad = [r for r in done if not r["ok"]]
    print(f"selftest: {len(done) - len(bad)} of {len(trials)} trials have the expected score"
          + ("" if not bad else "; not: " + ", ".join(r["job"] for r in bad)))
    return 1 if bad or len(done) != len(trials) else 0


def add_parsers(sub) -> None:
    p = sub.add_parser("selftest", formatter_class=C.Formatter,
                       help="through Harbor: the reference solution must score 1, doing nothing 0", description="""\
Check tasks end to end through Harbor, without a model or an API key. For every task, mode and seed chosen, the
reference solution (Harbor's agent `oracle`) must score 1 and doing nothing (Harbor's agent `nop`) must score 0.

Each trial is one Harbor job, made the way the batch runner makes its jobs:
  1. scripts/task_seed.py materialize: a copy of the task directory at the seed, every file's time derived from its
     content (so Docker's build cache cannot take one task's file for another's);
  2. scripts/harbor/clamp_cpus.py: the copy's CPU caps are clamped to this host's CPU count; with --gpus,
     scripts/harbor/pin_gpu.py pins the copy's containers to one GPU;
  3. harbor run -p <copy> -i <task>-<mode> -a <agent> -n 1 -k 1 -y -o <jobs> --job-name <job>
     with <job> = selftest-<task>-<mode>-<agent>-s<seed>; its output goes to <jobs>/<job>.log;
  4. the score is read from <jobs>/<job>/<trial>/verifier/reward.txt.
A job that has its score is not run again: after an interruption, the same command continues. A job directory without
a score (a trial that did not finish) is reported as failed until you move it away.

Needs Docker with the NVIDIA Container Toolkit, Harbor and the images (`roboarc doctor` checks all of them).
The privileged reference trial of interlocked-claws-08 takes about 6 minutes on an RTX 4090.
Exit status: 0 when every trial has its expected score, 1 otherwise.""", epilog="""\
examples:
  roboarc selftest interlocked-claws-08                 # one task: 2 modes x 2 agents = 4 trials
  roboarc selftest --families orbits,ikea --print       # show what would run
  roboarc selftest --families all --lanes 2 --gpus 0,1  # one task per family (release 2.2.1: 20 tasks, 80 trials)""")
    p.add_argument("tasks", metavar="TASK", nargs="*", help="the tasks to check, by id or alias (default: one task "
                   "per family, see --families)")
    p.add_argument("--families", metavar="FAMILY[,FAMILY...]", help="without TASK: check, for each of these families, its task "
                   "with the shortest reference solution. Family names as `roboarc list` prints them, with `_` or `-`; "
                   "`ikea` stands for the IKEA scenes as one family; `all` for every family (default: all: every "
                   "puzzle family and ikea)")
    p.add_argument("--modes", metavar="MODE[,MODE]", default="privileged,standard", help="the modes to check: privileged, "
                   "standard (default: privileged,standard)")
    p.add_argument("--agents", metavar="AGENT[,AGENT]", default="oracle,nop", help="oracle (the reference solution, must "
                   "score 1), nop (does nothing, must score 0) (default: oracle,nop)")
    p.add_argument("--seeds", metavar="SEED[,SEED...]", default="0", help="the appearance seeds to check, each 0, 1 or 2. "
                   "A reference solution is validated at the seeds its task.toml lists ([metadata.seeds] oracle): 0 "
                   "for every task, 1 and 2 for some (default: 0)")
    p.add_argument("--jobs", metavar="DIR", help="Harbor's jobs directory, where the trials, their logs and scores go "
                   "(default: jobs/selftest in the checkout, which git ignores)")
    p.add_argument("--lanes", metavar="N", type=C.positive_int, default=1, help="trials at a time, an integer. Unlike "
                   "roboarc eval, selftest does not schedule: each trial needs the CPUs and memory its task declares, "
                   "and a rope or cloth trial should not share its GPU, so one lane per GPU always fits (default: 1)")
    p.add_argument("--gpus", metavar="GPU[,GPU...]", help="GPU indices as nvidia-smi numbers them. Lane k runs its trials "
                   "on the k-th of them, in turn (default: none given, and Docker puts every trial on GPU 0)")
    p.add_argument("--print", action="store_true", help="print the trials that would run, and run nothing")
    p.set_defaults(fn=_selftest)
