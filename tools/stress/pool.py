"""Run replay jobs as parallel fresh processes. MuJoCo is single-threaded, so this runs one process per core.

usage: pool.py <jobs.json> <results dir> [--workers N] [--timeout S]
jobs.json: [{"id", "instance", "actions" (npz path or nop:N), "hold"?: 40, "trace"?: false, "experiment_opt"?: str}]
Writes <results dir>/<id>.json (and <id>.trace.json.gz, <id>.log), plus <results dir>/_pool.json with the exit codes and
wall times. A job whose result exists is skipped, so a rerun resumes."""
import argparse, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_jobs(jobs, results, workers=8, timeout=10800., echo=True):   # 3 h: the reference wall limit (round.py)
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (results / f"{j['id']}.json").exists()]
    status = {}

    def one(j):
        cmd = [sys.executable, str(HERE / "replay.py"), "--instance", str(j["instance"]), "--actions", str(j["actions"]),
               "--out", str(results / f"{j['id']}.json"), "--id", j["id"], "--hold", str(j.get("hold", 40))]
        if j.get("trace"):
            cmd += ["--trace", str(results / f"{j['id']}.trace.json.gz")]
        if j.get("experiment_opt"):
            cmd += ["--experiment-opt", j["experiment_opt"]]
        t0 = time.time()
        with open(results / f"{j['id']}.log", "w") as log:
            try:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        status[j["id"]] = dict(rc=rc, wall_s=round(time.time() - t0, 1))
        if echo:
            print(json.dumps(dict(id=j["id"], **status[j["id"]])), flush=True)

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(one, todo))
    old = json.loads((results / "_pool.json").read_text()) if (results / "_pool.json").exists() else {}
    (results / "_pool.json").write_text(json.dumps({**old, **status}, indent=1))
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("jobs")
    ap.add_argument("results")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=10800)
    a = ap.parse_args()
    run_jobs(json.loads(Path(a.jobs).read_text()), a.results, a.workers, a.timeout)


if __name__ == "__main__":
    main()
