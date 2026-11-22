"""Tests for bash-runners/ablations/ with a fake scripts/harbor/run_agent_batch.sh that records how each lane ran it: no
docker, no harbor, no GPU. Standard library only:

    python3 scripts/ablations/tests/test_runners.py

- run_E<n>.sh refuses to start (exit 2, no lane) without a passing variant check for the variants as they are, with a
  check made with rendering off, after a variant changed, without RESULTS, or with a MODEL that gives another run id;
- with a passing check it starts 4 lanes per GPU (LANES overrides), each running the runner over the experiment's
  batches in order with the fixed settings (AGENT, EFFORT=high, K=3, SEEDS=0, MODES, the usage gateway with
  GATEWAY_RAW=0) and the experiment's bases; it writes <RESULTS>/<batch>/ablation.json and ablation_checks.json;
- a lane that the runner stopped (STOP lane <pid> in the batch's log) does not go on to the next batch.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts" / "ablations"), str(REPO / "scripts")]
import experiments  # noqa: E402
import task_conventions as tc  # noqa: E402

FAKE_RUNNER = r'''#!/usr/bin/env bash
python3 - "$$" "$@" <<'PY'
import json, os, sys
keep = ("AGENT", "MODEL", "EFFORT", "K", "SEEDS", "MODES", "ENV_FILE", "USAGE_GATEWAY", "GATEWAY_RAW", "JOBS", "RESULTS")
pid, batch, bases = sys.argv[1], sys.argv[2], sys.argv[3:]
with open(os.path.join(os.environ["FAKE_RUNNER_LOG"], f"{pid}.json"), "w") as f:
    f.write(json.dumps({"batch": batch, "bases": bases, "env": {k: os.environ.get(k) for k in keep}}))
if os.environ.get("FAKE_STOP") == batch:
    os.makedirs(os.environ["JOBS"], exist_ok=True)
    with open(os.path.join(os.environ["JOBS"], f"{batch}.log"), "a") as f:
        f.write(f"[x] STOP lane {pid}: a model-account failure\n")
PY
'''


class Runners(unittest.TestCase):
    E = "E5"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name)
        repo = self.repo = root / "repo"
        shutil.copytree(REPO / "bash-runners", repo / "bash-runners")
        shutil.copytree(REPO / "scripts", repo / "scripts", ignore=shutil.ignore_patterns("__pycache__", "digests"))
        (repo / "scripts" / "ablations" / "digests").mkdir()
        shutil.copytree(REPO / "images", repo / "images", ignore=shutil.ignore_patterns("*.tar", "*.oci.tar"))
        runner = repo / "scripts" / "harbor" / "run_agent_batch.sh"
        runner.write_text(FAKE_RUNNER)
        runner.chmod(0o755)
        (repo / ".env").write_text("OPENAI_API_KEY=x\n")
        # the experiment's variants: tiny task dirs with their DIGEST, locked
        self.variants = [v.task for v in experiments.variants(self.E)]
        lock = {}
        for t in self.variants:
            d = repo / "tasks" / t
            d.mkdir(parents=True)
            (d / "task.toml").write_text(f'[task]\nname = "roboarc/{t}"\n')
            dg = tc.digest(d)
            (d / "DIGEST").write_text(dg + "\n")
            lock.setdefault(t.split("-")[-2], {})[t] = dg
        for token, rows in lock.items():
            (repo / "scripts" / "ablations" / "digests" / f"{token}.tsv").write_text(
                "".join(f"{k}\t{v}\n" for k, v in sorted(rows.items())))
        self.report(render="on")
        bin_dir = root / "bin"
        bin_dir.mkdir()
        (bin_dir / "nvidia-smi").write_text("#!/bin/sh\necho 'GPU 0: NVIDIA Fake GPU'\necho 'GPU 1: NVIDIA Fake GPU'\n")
        (bin_dir / "nvidia-smi").chmod(0o755)
        self.log = root / "lanes"
        self.log.mkdir()
        self.results = root / "results"
        self.env = {k: v for k, v in os.environ.items() if k not in (
            "AGENT", "MODEL", "EFFORT", "RESULTS", "ENV_FILE", "JOBS", "LANES", "GPUS", "SEEDS", "K", "MODES", "FAKE_STOP")}
        self.env.update(PATH=f"{bin_dir}:{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}",
                        FAKE_RUNNER_LOG=str(self.log), RESULTS=str(self.results), STAGGER="0")

    def tearDown(self):
        self.tmp.cleanup()

    def report(self, render: str, passed: bool = True):
        c = self.repo / "jobs" / "ablations" / "checks"
        c.mkdir(parents=True, exist_ok=True)
        vs = [{"task": t, "digest": (self.repo / "tasks" / t / "DIGEST").read_text().strip(), "pass": passed}
              for t in self.variants]
        (c / f"{self.E}.json").write_text(json.dumps({"experiment": self.E, "environment": {"render": render},
                                                      "variants": vs, "all_pass": passed}))

    def run_e(self, **env):
        e = dict(self.env, **env)
        return subprocess.run(["bash", str(self.repo / "bash-runners" / "ablations" / f"run_{self.E}.sh")], env=e,
                              capture_output=True, text=True, timeout=120)

    def lanes(self, n, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            got = [json.loads(p.read_text()) for p in self.log.glob("*.json")]
            if len(got) >= n:
                time.sleep(0.5)
                return [json.loads(p.read_text()) for p in self.log.glob("*.json")]
            time.sleep(0.2)
        return [json.loads(p.read_text()) for p in self.log.glob("*.json")]

    def test_starts_lanes(self):
        p = self.run_e()
        self.assertEqual(p.returncode, 0, p.stderr)
        got = self.lanes(8)
        self.assertEqual(len(got), 8)                       # 4 per GPU, 2 GPUs; E5 has one batch
        e = experiments.EXPERIMENTS[self.E]
        for g in got:
            self.assertEqual(g["batch"], "ablation-e5-r1")
            self.assertEqual(g["bases"], e.runner_bases())
            env = g["env"]
            self.assertEqual((env["AGENT"], env["MODEL"], env["EFFORT"], env["K"], env["SEEDS"], env["MODES"], env["GATEWAY_RAW"]),
                             ("codex", "gpt-6-astra", "high", "3", "0", "standard", "0"))
            self.assertEqual(env["USAGE_GATEWAY"], str(self.repo / "scripts" / "usage_gateway"))
        meta = json.loads((self.results / "ablation-e5-r1" / "ablation.json").read_text())
        self.assertEqual((meta["experiment"], meta["repeat"], len(meta["trials"])), ("E5", 1, 30))
        self.assertEqual(meta["results_dataset"], "https://huggingface.co/datasets/RoboARC/RoboARC-ablation-outputs")
        self.assertTrue((self.results / "ablation-e5-r1" / "ablation_checks.json").is_file())

    def test_smoke(self):
        p = subprocess.run(["bash", str(self.repo / "bash-runners" / "ablations" / "smoke.sh"), self.E], env=self.env,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr)
        got = self.lanes(2)
        first = experiments.EXPERIMENTS[self.E].bases()[0]
        self.assertEqual(len(got), 2)                       # E5: the first task's two conditions, one lane each
        for g in got:
            self.assertEqual(g["batch"], "ablation-smoke-e5")
            self.assertEqual(g["bases"], [f"{first}-obsv1", f"{first}-obsv2"])
            self.assertEqual(g["env"]["JOBS"], str(self.repo / "jobs" / "smoke"))
        meta = json.loads((self.results / "ablation-smoke-e5" / "ablation.json").read_text())
        self.assertEqual([r["task"] for r in meta["trials"]], [f"{first}-obsv1-standard", f"{first}-obsv2-standard"])
        self.assertFalse((self.results / "ablation-e5-r1").exists())

    def test_model_spellings(self):
        p = self.run_e(MODEL="openai/gpt-6-astra", LANES="1")
        self.assertEqual(p.returncode, 0, p.stderr)
        p = self.run_e(MODEL="openai/gpt-6.1-sol", LANES="1")
        self.assertEqual(p.returncode, 2)
        self.assertIn("run id", p.stderr)

    def test_refusals(self):
        self.report(render="off")
        self.assertEqual(self.run_e().returncode, 2)
        self.report(render="on", passed=False)
        self.assertEqual(self.run_e().returncode, 2)
        self.report(render="on")
        t = self.repo / "tasks" / self.variants[0] / "task.toml"
        t.write_text(t.read_text() + "# changed\n")
        p = self.run_e()
        self.assertEqual(p.returncode, 2)
        self.assertIn("DIGEST", p.stderr)
        self.assertEqual(list(self.log.glob("*.json")), [])
        e = dict(self.env); e.pop("RESULTS")
        self.assertEqual(subprocess.run(["bash", str(self.repo / "bash-runners/ablations/run_E5.sh")], env=e,
                                        capture_output=True).returncode, 2)

    def test_send_results(self):
        """bash-runners/ablations/send_results.sh: scripts/send_results.sh per batch, to the ablations' dataset by default,
        the experiment and repeat in NOTES; SMOKE=1 sends the smoke batch from jobs/smoke."""
        fake = self.repo / "scripts" / "send_results.sh"
        fake.write_text('#!/usr/bin/env bash\npython3 -c \'import json, os, sys; print(json.dumps({"argv": sys.argv[1:], '
                        '**{k: os.environ.get(k) for k in ("HF_DATASET", "NOTES", "JOBS", "RESULTS")}}))\' "$@" >> '
                        f'{self.log}/send.jsonl\n')
        fake.chmod(0o755)
        for b in ("ablation-e5-r1", "ablation-smoke-e5"):
            (self.results / b).mkdir(parents=True)
            (self.results / b / "ablation.json").write_text("{}")
        send = ["bash", str(self.repo / "bash-runners/ablations/send_results.sh"), "E5", "tester"]
        self.assertEqual(subprocess.run(send, env=dict(self.env, NOTES="none"), capture_output=True).returncode, 0)
        self.assertEqual(subprocess.run(send, env=dict(self.env, SMOKE="1"), capture_output=True).returncode, 0)
        self.assertEqual(subprocess.run(send, env=dict(self.env, HF_DATASET="me/other"), capture_output=True).returncode, 0)
        rows = [json.loads(l) for l in (self.log / "send.jsonl").read_text().splitlines()]
        self.assertEqual([r["argv"] for r in rows], [["ablation-e5-r1", "tester"], ["ablation-smoke-e5", "tester"],
                                                     ["ablation-e5-r1", "tester"]])
        self.assertEqual([r["HF_DATASET"] for r in rows], ["RoboARC/RoboARC-ablation-outputs"] * 2 + ["me/other"])
        self.assertTrue(rows[0]["NOTES"].startswith("ablation E5 (") and "repeat 1 of 1" in rows[0]["NOTES"])
        self.assertIn("smoke 1 of 1", rows[1]["NOTES"])
        self.assertEqual(rows[1]["JOBS"], str(self.repo / "jobs" / "smoke"))

    def test_lane_stops_on_account_failure(self):
        jobs = pathlib.Path(self.tmp.name) / "jobs"
        lane = self.repo / "bash-runners" / "ablations" / "lane.sh"
        e = dict(self.env, JOBS=str(jobs), FAKE_STOP="b1")
        p = subprocess.run(["bash", str(lane), "b1 b2 b3", "base-a"], env=e, capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 1)
        self.assertEqual([json.loads(f.read_text())["batch"] for f in self.log.glob("*.json")], ["b1"])
        e["FAKE_STOP"] = ""
        p = subprocess.run(["bash", str(lane), "b1 b2 b3", "base-a"], env=e, capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0)
        self.assertEqual(sorted(json.loads(f.read_text())["batch"] for f in self.log.glob("*.json")), ["b1", "b1", "b2", "b3"])


if __name__ == "__main__":
    unittest.main()
