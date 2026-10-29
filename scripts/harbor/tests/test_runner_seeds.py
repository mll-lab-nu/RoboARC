"""Tests for run_agent_batch.sh's trials and seeds (docs/RUNNING.md "Seeds", "Results") with a fake `harbor`: no docker,
no model, no network. The runner and its helpers run from a copy of the repository's scripts/ next to fixture tasks
made from a committed task pair: one that allows seeds 0, 1, 2 (`build = "runtime"`) and one that allows its default
seed only.

    python3 scripts/harbor/tests/test_runner_seeds.py     # stdlib only (Python 3.11+)

- K trials of a task get distinct seeds (the same in both modes), each its own Harbor job (-k 1), its copy made from that
  seed, and seed.json in the job; the results are one folder per trial, t<k>/, with the seed in config.json, and an
  index of the task's trials; a single-seed task with K > 1 is not over the retry cap (each trial is its own);
- a trial voided for infrastructure, moved aside and run again, gets its seed again and is recorded as the second
  attempt of the same trial;
- SEQ=0: one job per mode and trial, every task of it from its own seed;
- K=1: the job name is the one of the runner before seeds;
- GPU_SCHED=1 (the default): each job's copy has every NVIDIA device pinned to the GPU the host's ledger gave it, and
  <job>/placement.json records it (a task without a GPU device: none); GPU_SCHED=0 leaves the copies as they are;
- the machine (host_info.py, a fake nvidia-smi): logged as host:, GPUs:, software: lines, written into every job as
  host.json, and in each result's config.json as `host`, with the name and driver of the job's GPU;
- PHASE_SPLIT=1 (the default): Harbor gets the agent-end marker plugin (scripts/harbor/plugins on its PYTHONPATH); when
  a trial's agent phase ends the lane starts the next trial while the first validates on its GPU, and ends only when
  both have their results, with every slot released; a deformable trial waits for a GPU on which none validates;
  PHASE_SPLIT=0, GPU_SCHED=0 or a harbor without --plugin: one trial after the other, as before.
"""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
import task_seed  # noqa: E402

FAKE_HARBOR = '''#!{py}
"""A stand-in for `harbor run`: records its arguments and each task copy's seed files, writes one finished trial per task
(no reward.json, an infrastructure failure, when the job is $FAKE_INFRA_JOB). With --plugin (the agent-end marker,
plugins/phase_marker.py) it writes the marker file its --pk out= names when the "agent phase" ends, then "validates" for
$FAKE_VERIFY_S seconds before the trial's results; `run --help` lists --plugin unless $FAKE_NO_PLUGIN is set."""
import json, os, pathlib, re, sys, time, uuid
a = sys.argv[1:]
if a == ["--version"]:
    print("0.0.0-fake")
    sys.exit(0)
if "--help" in a:
    print("Usage: harbor run [OPTIONS]\\n  --job-name TEXT" + ("" if os.environ.get("FAKE_NO_PLUGIN") else "\\n  --plugin TEXT"))
    sys.exit(0)
val = lambda k: a[a.index(k) + 1]
tasks = [a[i + 1] for i, x in enumerate(a) if x == "-i"]
jobs, job, root = pathlib.Path(val("-o")), val("--job-name"), pathlib.Path(val("-p"))
seeds = {{}}
for t in tasks:
    seeds[t] = {{str(p.relative_to(root / t)): p.read_text().strip() for p in sorted((root / t).rglob("seed")) if p.is_file()}}
devices = {{t: sorted(m.group(0).strip() for d in ("environment", "tests") for p in sorted((root / t / d).glob("*compose*.y*ml"))
                     for m in re.finditer(r"(?m)^[ \\t]*(count|device_ids):.*$", p.read_text())) for t in tasks}}
with open(os.environ["FAKE_HARBOR_LOG"], "a") as f:
    f.write(json.dumps({{"argv": a, "job": job, "seeds": seeds, "devices": devices, "pythonpath": os.environ.get("PYTHONPATH")}}) + "\\n")
trials = {{t: jobs / job / f"{{t[:32]}}__{{uuid.uuid4().hex[:7]}}" for t in tasks}}
for d in trials.values():                     # Harbor makes the trial's directories when it starts
    (d / "verifier").mkdir(parents=True)
    (d / "agent").mkdir()
if "--plugin" in a:
    pk = dict(a[i + 1].split("=", 1) for i, x in enumerate(a) if x == "--pk")
    out = pathlib.Path(pk["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({{"job": pk.get("job"), "agent_finished_at": "2026-10-21T00:01:10Z"}}) + "\\n")
    time.sleep(float(os.environ.get("FAKE_VERIFY_S", "0")))
for t, d in trials.items():
    (d / "result.json").write_text(json.dumps({{
        "id": uuid.uuid4().hex, "task_name": f"roboarc/{{t}}", "started_at": "2026-10-21T00:00:00Z",
        "agent_execution": {{"started_at": "2026-10-21T00:00:10Z", "finished_at": "2026-10-21T00:01:10Z"}},
        "agent_result": {{"n_input_tokens": 1000, "n_cache_tokens": 0, "n_output_tokens": 50, "cost_usd": None}}}}))
    if os.environ.get("FAKE_INFRA_JOB") != job:
        (d / "verifier" / "reward.json").write_text(json.dumps({{"success": 0}}))
'''


FAKE_NVIDIA_SMI = "#!/bin/sh\nprintf '0, NVIDIA Fake GPU, 580.95.05, 46068\\n1, NVIDIA Fake GPU, 580.95.05, 46068\\n'\n"


def pair_bases():
    """Two committed task pairs (base names), the smallest of each kind: a puzzle base (three appearance seeds since
    1.2.0) and an IKEA base (made a single-seed task here: one_seed)."""
    size = lambda d: sum(f.stat().st_size for f in d.rglob("*") if f.is_file())  # noqa: E731
    out = []
    for kind in ("puzzle", "ikea"):
        for d in sorted((REPO / "tasks").glob("roboarc-*-privileged"), key=size):
            b = d.name.removesuffix("-privileged")
            if b.startswith("roboarc-ikea") == (kind == "ikea") and (REPO / "tasks" / f"{b}-standard" / "task.toml").is_file():
                out.append(b)
                break
    if len(out) != 2:
        raise unittest.SkipTest("no puzzle or no IKEA task pair in tasks/")
    return out


def three_seeds(task: pathlib.Path) -> None:
    """Seeds 0, 1, 2 staged at run time (a task the generator rendered with seeds has them already)."""
    p = task / "task.toml"
    if 'build = "runtime"' in p.read_text():
        return
    s = re.sub(r"(?m)^allowed = .*$", "allowed = [0, 1, 2]", p.read_text(), count=1)
    s = re.sub(r"(?m)^oracle = \[.*$", 'oracle = [0, 1, 2]\nbuild = "runtime"', s, count=1)
    p.write_text(s)
    places = ["environment/sim", "tests"] if task.name.endswith("-standard") else ["environment", "tests"]
    for pl in places:
        (task / pl / "seed").write_text("0\n")


def one_seed(task: pathlib.Path) -> None:
    """Its default seed only, without seed files (every committed task allows three seeds since 2026-10-27)."""
    p = task / "task.toml"
    p.write_text(re.sub(r'(?m)^build = "runtime"\n', "", re.sub(r"(?m)^allowed = .*$", "allowed = [0]", p.read_text(), count=1)))
    for f in task.rglob("seed"):
        f.unlink()


class RunnerSeeds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.multi, cls.single = pair_bases()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        repo = self.root / "repo"
        (repo / "scripts").mkdir(parents=True)
        for f in (REPO / "scripts").iterdir():
            if f.is_file():
                shutil.copy2(f, repo / "scripts" / f.name)
        for d in ("harbor", "rcb_monitor", "eai_standard"):
            shutil.copytree(REPO / "scripts" / d, repo / "scripts" / d, ignore=shutil.ignore_patterns("__pycache__"))
        (repo / "docs").mkdir()
        shutil.copy2(REPO / "docs" / "easy_set.toml", repo / "docs" / "easy_set.toml")
        for b in (self.multi, self.single):
            for m in ("privileged", "standard"):
                shutil.copytree(REPO / "tasks" / f"{b}-{m}", repo / "tasks" / f"{b}-{m}", symlinks=True)
        for m in ("privileged", "standard"):
            three_seeds(repo / "tasks" / f"{self.multi}-{m}")
            one_seed(repo / "tasks" / f"{self.single}-{m}")
        self.repo = repo
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        harbor = bin_dir / "harbor"
        harbor.write_text(FAKE_HARBOR.format(py=sys.executable))
        harbor.chmod(0o755)
        (bin_dir / "nvidia-smi").write_text(FAKE_NVIDIA_SMI)
        (bin_dir / "nvidia-smi").chmod(0o755)
        self.log = self.root / "harbor.jsonl"
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("GATEWAY", "USAGE_GATEWAY", "TOKEN_BUDGET", "OPENAI_", "ANTHROPIC_", "CODEX_", "DEEPSEEK_", "FAKE_"))}
        self.env.update(PATH=f"{bin_dir}:{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}",
                        FAKE_HARBOR_LOG=str(self.log), MODEL="openai/test-model", EFFORT="high",
                        JOBS=str(self.root / "jobs"), RESULTS=str(self.root / "results"),
                        SLOTS_DIR=str(self.root / "slots"), GPUS="0,1", HOST_CPUS="64", MEM_HEADROOM_GB="0")
        self.run_id = "codex-test_model-high"

    def tearDown(self):
        self.tmp.cleanup()

    def run_batch(self, *bases, **env):
        e = dict(self.env, **env)
        p = subprocess.run(["bash", str(self.repo / "scripts/harbor/run_agent_batch.sh"), "b", *bases], env=e,
                           capture_output=True, text=True, timeout=900, cwd=self.root)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p

    def calls(self):
        return [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []

    def result(self, task, k):
        return json.loads((self.root / "results" / "b" / task / self.run_id / f"t{k}" / "config.json").read_text())

    def expected(self, base, k):
        return task_seed.assign(self.repo / "tasks" / f"{base}-privileged", "b", k)

    def test_k3_distinct_seeds_per_trial(self):
        self.run_batch(self.multi, self.single, K="3")
        calls = self.calls()
        self.assertEqual(len(calls), 12)                          # 2 bases x 3 trials x 2 modes, one job each
        for c in calls:
            self.assertEqual(c["argv"][c["argv"].index("-k") + 1], "1")
        want = self.expected(self.multi, 3)
        self.assertEqual(sorted(want), [0, 1, 2])
        for m, places in (("privileged", ["environment/seed", "tests/seed"]), ("standard", ["environment/sim/seed", "tests/seed"])):
            task = f"{self.multi}-{m}"
            for k in (1, 2, 3):
                job = f"b-{m}-{self.multi}-t{k}"
                (c,) = [c for c in calls if c["job"] == job]
                self.assertEqual(c["seeds"][task], {p: str(want[k - 1]) for p in places})
                rec = json.loads((self.root / "jobs" / job / "seed.json").read_text())
                self.assertEqual((rec["trial"], rec["trials"], rec["tasks"][task]["seed"]), (k, 3, want[k - 1]))
                cfg = self.result(task, k)
                self.assertEqual((cfg["trial"], cfg["trials"], cfg["seed"], cfg["seed_source"]), (k, 3, want[k - 1], "runner"))
                self.assertEqual((cfg["attempts"], cfg["attempts_over_cap"], cfg["result"]), (1, False, "fail"))
                self.assertIs(cfg["digest_match"], True)
            idx = json.loads((self.root / "results" / "b" / task / self.run_id / "trials.json").read_text())
            self.assertEqual([(x["trial"], x["seed"]) for x in idx["trials"]], [(k, want[k - 1]) for k in (1, 2, 3)])
        for m in ("privileged", "standard"):                       # F5: a single-seed task, K = 3, three trials
            task = f"{self.single}-{m}"
            for k in (1, 2, 3):
                (c,) = [c for c in calls if c["job"] == f"b-{m}-{self.single}-t{k}"]
                self.assertEqual(c["seeds"][task], {})            # the task as it is: no seed file
                cfg = self.result(task, k)
                self.assertEqual((cfg["seed"], cfg["attempts"], cfg["attempts_over_cap"]), (0, 1, False))
        summary = (self.root / "jobs" / "b.summary.md").read_text()
        self.assertIn("Per task over its trials", summary)
        self.assertIn("| trial | seed |", summary)

    def test_retry_keeps_seed(self):
        job = f"b-privileged-{self.multi}-t2"
        self.run_batch(self.multi, MODES="privileged", K="2", FAKE_INFRA_JOB=job)
        task = f"{self.multi}-privileged"
        first = self.result(task, 2)
        self.assertEqual(first["result"], "not_a_result")
        jobs = self.root / "jobs"
        aside = self.root / "aside"
        aside.mkdir()
        shutil.move(str(jobs / job), str(aside / job))             # RUNNING.md: move the voided job and its claim aside
        (jobs / ".claims" / job).rmdir()
        self.run_batch(self.multi, MODES="privileged", K="2")
        reruns = [c for c in self.calls() if c["job"] == job]
        self.assertEqual(len(reruns), 2)
        self.assertEqual(reruns[0]["seeds"], reruns[1]["seeds"])     # the retry runs from the same seed
        cfg = self.result(task, 2)
        self.assertEqual((cfg["result"], cfg["attempts"], cfg["attempts_over_cap"]), ("fail", 2, False))
        self.assertEqual(cfg["seed"], self.expected(self.multi, 2)[1])
        led = json.loads((self.root / "results" / "b" / task / self.run_id / "t2" / "attempts.json").read_text())["attempts"]
        self.assertEqual([a["status"] for a in led], ["not_a_result", "fail"])

    def test_seq0(self):
        self.run_batch(self.multi, self.single, MODES="privileged", K="2", SEQ="0")
        calls = self.calls()
        self.assertEqual(sorted(c["job"] for c in calls), ["b-privileged-t1", "b-privileged-t2"])
        want = self.expected(self.multi, 2)
        for c in calls:
            k = int(c["job"][-1])
            self.assertEqual(c["seeds"][f"{self.multi}-privileged"]["environment/seed"], str(want[k - 1]))
            self.assertEqual(c["seeds"][f"{self.single}-privileged"], {})
        for k in (1, 2):
            self.assertEqual(self.result(f"{self.multi}-privileged", k)["seed"], want[k - 1])
            self.assertEqual(self.result(f"{self.single}-privileged", k)["attempts_over_cap"], False)

    def test_gpu_placement(self):
        sys.path.insert(0, str(REPO / "scripts" / "harbor"))
        import gpu_slots
        slots = self.root / "slots"
        slots.mkdir()
        me = os.getpid()                                           # GPU 0 taken whole by a trial of another live runner
        (slots / "state.json").write_text(json.dumps({"holds": {"other": {
            "pid": me, "start": gpu_slots.proc_start(me), "gpu": "0", "units": 4, "cpus": 0, "mem_gb": 0}},
            "waiting": {}, "draining": []}))
        self.run_batch(self.multi, self.single)
        calls = self.calls()
        self.assertEqual(len(calls), 4)
        for c in calls:
            (task,) = c["devices"]
            units = gpu_slots.classify(self.repo / "tasks" / task, None, 4)[0]
            place = json.loads((self.root / "jobs" / c["job"] / "placement.json").read_text())
            self.assertEqual(place["units"], units)
            if units == 0:                                         # IKEA privileged: no GPU device, none given
                self.assertIsNone(place["gpu"])
                self.assertEqual(c["devices"][task], [])
            else:
                self.assertEqual(place["gpu"], "1")
                self.assertTrue(c["devices"][task])
                self.assertEqual(set(c["devices"][task]), {'device_ids: ["1"]'})
            self.assertIs(self.result(task, 1)["digest_match"], True)      # the pin is in the copy only
            self.assertEqual(self.result(task, 1)["gpu"], place["gpu"])
        state = json.loads((slots / "state.json").read_text())
        self.assertEqual(list(state["holds"]), ["other"])          # every job's slot released

    def test_host_record(self):
        self.run_batch(self.multi, MODES="privileged")
        (c,) = self.calls()
        log = (self.root / "jobs" / "b.log").read_text()
        self.assertIn("GPUs:       0, NVIDIA Fake GPU, 580.95.05, 46068 MiB; 1, NVIDIA Fake GPU, 580.95.05, 46068 MiB", log)
        self.assertRegex(log, r"(?m)^\[[^]]+\] host:\s+.+, \d+ CPUs, \d+ GB RAM$")
        self.assertRegex(log, r"(?m)^\[[^]]+\] software:\s+.*; harbor 0\.0\.0-fake$")
        host = json.loads((self.root / "jobs" / c["job"] / "host.json").read_text())
        self.assertEqual([g["name"] for g in host["gpus"]], ["NVIDIA Fake GPU"] * 2)
        cfg = self.result(f"{self.multi}-privileged", 1)
        self.assertEqual(cfg["host"], host)
        self.assertIn(cfg["gpu"], ("0", "1"))
        self.assertEqual((cfg["gpu_name"], cfg["gpu_driver"]), ("NVIDIA Fake GPU", "580.95.05"))

    def test_gpu_sched_off(self):
        self.run_batch(self.multi, MODES="privileged", GPU_SCHED="0")
        (c,) = self.calls()
        (devs,) = c["devices"].values()
        self.assertTrue(devs)
        self.assertTrue(all(d.startswith("count:") for d in devs))         # the task's own reservation, as before
        self.assertFalse((self.root / "jobs" / c["job"] / "placement.json").exists())
        self.assertNotIn("--plugin", c["argv"])                              # no phase split without the ledger
        self.assertIn("PHASE_SPLIT needs GPU_SCHED=1", (self.root / "jobs" / "b.log").read_text())

    def log_index(self, needle):
        lines = (self.root / "jobs" / "b.log").read_text().splitlines()
        hits = [i for i, x in enumerate(lines) if needle in x]
        self.assertTrue(hits, f"no log line with {needle!r}")
        return hits[0]

    def assert_ended_clean(self, task, jobs):
        for k in (1, 2):
            self.assertEqual(self.result(task, k)["result"], "fail")          # both trials have their results
        self.assertEqual(json.loads((self.root / "slots" / "state.json").read_text())["holds"], {})
        fin = self.log_index("finished (no unclaimed job left)")
        self.assertTrue(all(self.log_index(f"done {j} ") < fin for j in jobs))

    def test_phase_split_next_trial_during_validation(self):
        task = f"{self.single}-standard"                                      # IKEA standard: 1 unit (its sim's GPU)
        self.run_batch(self.single, MODES="standard", K="2", GPUS="0", FAKE_VERIFY_S="6")
        j1, j2 = (f"b-standard-{self.single}-t{k}" for k in (1, 2))
        self.assertLess(self.log_index(f"start {j2}:"), self.log_index(f"done {j1} "))   # t2 reasons while t1 validates
        self.assertIn("agent phase ended; it validates on GPU 0 (keeps 2 CPUs, 2 GB)",
                      (self.root / "jobs" / "b.log").read_text())
        for c in self.calls():
            argv = c["argv"]
            self.assertEqual(argv[argv.index("--plugin") + 1], "phase_marker:PhaseMarker")
            self.assertIn(f"out={self.root}/jobs/.phase/{c['job']}.json", argv)
            self.assertIn("/scripts/harbor/plugins", c["pythonpath"])
            place = json.loads((self.root / "jobs" / c["job"] / "placement.json").read_text())
            self.assertEqual((place["phase_split"], place["gpu_resident"], place["validate_cpus"],
                              place["validate_memory_gb"]), (True, 8, 2, 2))
        self.assert_ended_clean(task, (j1, j2))

    def test_phase_split_deformable_takes_another_gpu(self):
        self.run_batch(self.multi, MODES="privileged", K="2", FAKE_VERIFY_S="6")   # GPUS 0,1
        j1, j2 = (f"b-privileged-{self.multi}-t{k}" for k in (1, 2))
        self.assertLess(self.log_index(f"start {j2}:"), self.log_index(f"done {j1} "))
        gpus = {json.loads((self.root / "jobs" / j / "placement.json").read_text())["gpu"] for j in (j1, j2)}
        self.assertEqual(gpus, {"0", "1"})                                    # not beside t1's validation
        self.assert_ended_clean(f"{self.multi}-privileged", (j1, j2))

    def test_phase_split_deformable_waits_on_one_gpu(self):
        self.run_batch(self.multi, MODES="privileged", K="2", GPUS="0", FAKE_VERIFY_S="4")
        j1, j2 = (f"b-privileged-{self.multi}-t{k}" for k in (1, 2))
        self.assertLess(self.log_index(f"done {j1} "), self.log_index(f"start {j2}:"))   # GPU 0 empties first
        self.assertIn("none is empty (GPU 0: 0/4, 1 validating", (self.root / "jobs" / "b.log").read_text())
        self.assert_ended_clean(f"{self.multi}-privileged", (j1, j2))

    def test_phase_split_off(self):
        self.run_batch(self.single, MODES="standard", K="2", GPUS="0", FAKE_VERIFY_S="3", PHASE_SPLIT="0")
        j1, j2 = (f"b-standard-{self.single}-t{k}" for k in (1, 2))
        self.assertLess(self.log_index(f"done {j1} "), self.log_index(f"start {j2}:"))
        self.assertTrue(all("--plugin" not in c["argv"] for c in self.calls()))
        self.assertIs(json.loads((self.root / "jobs" / j1 / "placement.json").read_text())["phase_split"], False)
        self.assert_ended_clean(f"{self.single}-standard", (j1, j2))

    def test_harbor_without_plugins(self):
        self.run_batch(self.single, MODES="standard", K="2", GPUS="0", FAKE_VERIFY_S="3", FAKE_NO_PLUGIN="1")
        j1, j2 = (f"b-standard-{self.single}-t{k}" for k in (1, 2))
        self.assertIn("this harbor has no --plugin", (self.root / "jobs" / "b.log").read_text())
        self.assertLess(self.log_index(f"done {j1} "), self.log_index(f"start {j2}:"))
        self.assertTrue(all("--plugin" not in c["argv"] for c in self.calls()))

    def test_k1_job_name_unchanged(self):
        self.run_batch(self.single, MODES="privileged")
        (c,) = self.calls()
        self.assertEqual(c["job"], f"b-privileged-{self.single}")
        cfg = self.result(f"{self.single}-privileged", 1)
        self.assertEqual((cfg["trial"], cfg["trials"], cfg["seed"]), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
