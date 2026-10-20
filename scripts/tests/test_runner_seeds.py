#!/usr/bin/env python3
"""Seeds in scripts/run_agent_batch.sh (docs/RUNNING.md "Seeds"): a base whose tasks allow more than one seed gets K seeds
drawn without replacement, the same in both modes, one job per (task, mode, seed) from the copy scripts/task_seed.py
materializes, with -k 1; a base with one allowed seed runs as before (one job per task and mode, -k K). Checked on a
throwaway repository (two RoboARC task pairs, one made to allow seeds 0, 1, 2 with build = "runtime") with a stand-in
`harbor` on PATH that records what it was started with and writes one graded trial per attempt; the results
(write_task_results.py: one folder per seed, the task's README with its success over its seeds) and the batch summary
are read back. Also task_seed.assign itself. Standard library, bash and python >= 3.11 (tomllib); no docker, no
Harbor, no GPU."""
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import task_seed  # noqa: E402

fails = []
SEEDED, SINGLE = "roboarc-blanket-flatten-01-i01", "roboarc-blanket-flatten-02-i01"


def jload(p: Path):
    return json.loads(p.read_text()) if p.is_file() else {}


def text(p: Path) -> str:
    return p.read_text() if p.is_file() else ""


def check(cond, what):
    if not cond:
        fails.append(what)


def test_assign():
    for key in ("b/x", "b/y", "other/x"):
        s = task_seed.assign([0, 1, 2], 3, key)
        check(sorted(s) == [0, 1, 2], f"K = 3 of 3 seeds: all three, once each ({key}: {s})")
        check(task_seed.assign([0, 1, 2], 3, key) == s, f"the same key gives the same seeds ({key})")
        s7 = task_seed.assign([0, 1, 2], 7, key)
        check(len(s7) == 7 and sorted(s7[:3]) == sorted(s7[3:6]) == [0, 1, 2], f"K = 7: every seed once per pass ({key}: {s7})")
    check(len({tuple(task_seed.assign([0, 1, 2], 3, f"b/{i}")) for i in range(60)}) == 6, "the key changes the order")
    # each trial's seed is still uniform: over many keys, every position takes every seed about a third of the time
    n = 3000
    pos = [collections.Counter() for _ in range(3)]
    for i in range(n):
        for j, s in enumerate(task_seed.assign([0, 1, 2], 3, f"batch/base-{i}")):
            pos[j][s] += 1
    check(all(abs(c[s] / n - 1 / 3) < 0.04 for c in pos for s in (0, 1, 2)), f"per-trial marginals not uniform: {pos}")
    r = task_seed.assign({"from": 10, "to": 14}, 5, "k")
    check(sorted(r) == [10, 11, 12, 13, 14], f"a range: without replacement ({r})")
    a = task_seed.assign("any", 3, "k")
    check(len(set(a)) == 3 and all(0 <= x < task_seed.ANY for x in a), f'"any": three distinct seeds ({a})')
    check(task_seed.assign([0], 3, "k") == [0, 0, 0], "one allowed seed: every trial on it")


FAKE_HARBOR = r'''#!/usr/bin/env python3
# stand-in for `harbor run`: records the job and writes one graded trial per attempt (success when the seed is even)
import json, os, pathlib, sys, tomllib
a = sys.argv[1:]
opt = lambda n: a[a.index(n) + 1]
jobs, job, k = pathlib.Path(opt("-o")), opt("--job-name"), int(opt("-k"))
jd = jobs / job
jd.mkdir(parents=True, exist_ok=True)
(jd / "result.json").write_text("{}")
for t in [a[i + 1] for i, x in enumerate(a) if x == "-i"]:
    copy = pathlib.Path("tasks") / t
    md = tomllib.loads((copy / "task.toml").read_text())
    seed = md["metadata"]["seeds"]["default"]
    rec = {"job": job, "task": t, "k": k, "default": seed, "cpus": md["environment"]["cpus"],
           "seed_files": {p.relative_to(copy).as_posix(): p.read_text().strip() for p in sorted(copy.rglob("seed"))}}
    with open(os.environ["FAKE_HARBOR_LOG"], "a") as f:
        f.write(json.dumps(rec) + "\n")
    for i in range(k):
        td = jd / f"{t[:32]}__{len(list(jd.iterdir()))}x{i}"
        (td / "verifier").mkdir(parents=True)
        (td / "config.json").write_text(json.dumps({"task": {"path": f"tasks/{t}", "source": "tasks"}}))
        (td / "result.json").write_text(json.dumps({"id": f"{job}-{i}", "task_name": t, "started_at": f"2026-10-20T00:00:0{i}Z",
                                                    "agent_result": {"n_input_tokens": 1, "n_output_tokens": 1}}))
        (td / "verifier" / "reward.json").write_text(json.dumps({"success": int(seed % 2 == 0)}))
'''


def make_repo(root: Path) -> None:
    """A throwaway checkout: the runner, its scripts and docs/easy_set.toml, the two task pairs."""
    shutil.copytree(REPO / "scripts", root / "scripts", symlinks=True,
                    ignore=shutil.ignore_patterns("roboarc", "eai_standard", "tests", "__pycache__"))
    (root / "docs").mkdir()
    shutil.copy(REPO / "docs" / "easy_set.toml", root / "docs" / "easy_set.toml")
    for b in (SEEDED, SINGLE):
        for m in ("privileged", "standard"):
            shutil.copytree(REPO / "tasks" / f"{b}-{m}", root / "tasks" / f"{b}-{m}", symlinks=True)
    for m in ("privileged", "standard"):
        p = root / "tasks" / f"{SEEDED}-{m}" / "task.toml"
        s = p.read_text()
        s2 = re.sub(r"(?m)^allowed = \[0\]\noracle = \[0\]\n", 'allowed = [0, 1, 2]\noracle = [0, 1, 2]\nbuild = "runtime"\n', s)
        assert s2 != s, p
        p.write_text(s2)
    (root / "bin").mkdir()
    (root / "bin" / "harbor").write_text(FAKE_HARBOR)
    os.chmod(root / "bin" / "harbor", 0o755)


def run(root: Path, batch: str, *bases: str, **env) -> subprocess.CompletedProcess:
    e = dict(os.environ, PATH=f"{root / 'bin'}:{os.path.dirname(sys.executable)}:{os.environ['PATH']}",
             FAKE_HARBOR_LOG=str(root / "harbor.jsonl"), JOBS=str(root / "jobs"), RESULTS=str(root.parent / "results"),
             LOG_MIRROR="0", MEM_GUARD="0", HOST_CPUS="4", MODEL="openai/test-model", RUN_ID="codex-test_model-xhigh")
    e.update(env)
    return subprocess.run(["bash", str(root / "scripts" / "run_agent_batch.sh"), batch, *bases], env=e,
                          capture_output=True, text=True, timeout=600)


def harbor_log(root: Path) -> list[dict]:
    p = root / "harbor.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []


def test_runner():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "repo"
        make_repo(root)
        r = run(root, "b1", SEEDED, SINGLE, K="3")
        check(r.returncode == 0, f"runner exit {r.returncode}: {r.stdout[-2000:]} {r.stderr[-2000:]}")
        want = task_seed.assign([0, 1, 2], 3, f"b1/{SEEDED}")
        recs = harbor_log(root)
        by_job = {x["job"]: x for x in recs}
        for m, sim in (("privileged", "environment"), ("standard", "environment/sim")):
            task = f"{SEEDED}-{m}"
            jobs = [x for x in recs if x["task"] == task]
            check([x["default"] for x in jobs] == want, f"{m}: seeds {[x['default'] for x in jobs]}, assigned {want}")
            for x in jobs:
                s = x["default"]
                check(x["job"] == f"b1-{m}-{SEEDED}-s{s}", f"job name {x['job']}")
                check(x["k"] == 1, f"{x['job']}: -k {x['k']}, not 1")
                check(x["cpus"] == 4, f"{x['job']}: the copy's CPU cap {x['cpus']} was not clamped to HOST_CPUS")
                files = {} if s == 0 else {f"{sim}/seed": str(s), "tests/seed": str(s)}
                check(x["seed_files"] == files, f"{x['job']}: seed files {x['seed_files']}, want {files}")
                rec = jload(root / "jobs" / x["job"] / "seed.json")
                check(rec == {"task": task, "seed": s, "slot": f"s{s}", "source": "scripts/task_seed.py materialize"},
                      f"{x['job']}/seed.json: {rec}")
        for m in ("privileged", "standard"):
            x = by_job.get(f"b1-{m}-{SINGLE}")
            check(x is not None and x["k"] == 3 and x["seed_files"] == {} and x["default"] == 0,
                  f"single-seed base, {m}: one job, -k 3, the task as it is ({x})")
            check(not (root / "jobs" / f"b1-{m}-{SINGLE}" / "seed.json").exists(), "a single-seed job has no seed.json")
        check(len(recs) == 8, f"{len(recs)} harbor jobs, want 3 seeds x 2 modes + 2")
        check(not list((root / "tasks").rglob("seed")), "tasks/ itself was changed")
        check(not (root / "jobs" / ".tasks").exists() or not any((root / "jobs" / ".tasks").iterdir()), "job copies left behind")
        plan = jload(root / "jobs" / ".plan" / "b1.json")
        check(plan.get("seeds") == {SEEDED: want}, f"plan seeds {plan.get('seeds')}")
        check(sorted(j["job"] for j in plan.get("jobs", [])) == sorted(x["job"] for x in recs), "plan jobs != jobs run")
        check(all(re.search(r"-s(\d+)$", j["job"]) and j.get("seed") == int(re.search(r"-s(\d+)$", j["job"]).group(1))
                  for j in plan.get("jobs", []) if SEEDED in j["job"]), "plan job seeds")
        log = text(root / "jobs" / "b1.log")
        check(f"seeds {SEEDED}: {' '.join(map(str, want))} (K 3 per mode" in log, "the assignment is not in the batch log")
        # results: one folder per seed, the task's README with its success over its seeds
        res = Path(td) / "results" / "b1"
        for m in ("privileged", "standard"):
            d = res / f"{SEEDED}-{m}" / "codex-test_model-xhigh"
            for s in want:
                cfg = jload(d / f"s{s}" / "config.json")
                check(cfg.get("seed") == s and cfg.get("seed_source") == "runner copy" and cfg.get("attempts") == 1
                      and not cfg.get("attempts_over_cap") and cfg.get("result") == ("success" if s % 2 == 0 else "fail"),
                      f"{m} s{s}/config.json: {cfg}")
            readme = text(d / "README.md")
            check("Success over its seeds (results only): 2 / 3; 0 voided." in readme, f"task README: {readme}")
            single = res / f"{SINGLE}-{m}" / "codex-test_model-xhigh"
            check((single / "config.json").is_file() and not list(single.glob("s*/")), f"single-seed results layout: {single}")
        summ = text(root / "jobs" / "b1.summary.md")
        check(summ.count(f"| {SEEDED}-privileged (seed ") == 3, "summary rows do not show the seed")
        check(f"{SEEDED}-privileged 2 / 3 (seeds 0: success, 1: fail, 2: success)" in summ, f"summary per-task line: {summ}")
        # an infrastructure retry: the job moved aside with its claim, the same command again -> the same seed, alone
        s = want[1]
        job = f"b1-standard-{SEEDED}-s{s}"
        if (root / "jobs" / job).is_dir():
            shutil.move(root / "jobs" / job, root / "aside")
            os.rmdir(root / "jobs" / ".claims" / job)
        n0 = len(recs)
        r = run(root, "b1", SEEDED, SINGLE, K="3")
        recs = harbor_log(root)
        check(r.returncode == 0 and [x["job"] for x in recs[n0:]] == [job] and recs[-1]["default"] == s,
              f"rerun: {[x['job'] for x in recs[n0:]]}")
        # K beyond the allowed seeds: another shuffled pass, -p2 slots
        r = run(root, "b2", SEEDED, K="4", MODES="privileged")
        want4 = task_seed.assign([0, 1, 2], 4, f"b2/{SEEDED}")
        jobs4 = [x["job"] for x in harbor_log(root) if x["job"].startswith("b2-")]
        exp4 = [f"b2-privileged-{SEEDED}-s{x}" for x in want4[:3]] + [f"b2-privileged-{SEEDED}-s{want4[3]}-p2"]
        check(r.returncode == 0 and jobs4 == exp4, f"K = 4: {jobs4}, want {exp4}")
        # SEQ=0 cannot hold a copy per seed
        r = run(root, "b3", SEEDED, K="3", SEQ="0")
        check(r.returncode == 2 and "use SEQ=1" in r.stdout + r.stderr, f"SEQ=0 with seeds: exit {r.returncode}")


if __name__ == "__main__":
    test_assign()
    test_runner()
    if fails:
        print("\n".join(f"FAIL {f}" for f in fails))
        print(f"{len(fails)} failure(s)")
        sys.exit(1)
    print("ok runner seeds: K seeds without replacement per base, the same in both modes, one -k 1 job per seed from its "
          "materialized copy; single-seed bases unchanged; results per seed with the task's success over its seeds; a "
          "rerun keeps the seed")
