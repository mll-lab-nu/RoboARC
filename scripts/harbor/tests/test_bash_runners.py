"""Tests for bash-runners/ (docs/COLLABORATOR_SETUP.md, section 5) with a fake run_agent_batch.sh that records how each
lane was started: no docker, no harbor, no GPU.

    python3 scripts/harbor/tests/test_bash_runners.py     # stdlib only (Python 3.11+)

- run_seed<s>.sh starts 4 lanes per GPU (of GPUS when set; LANES overrides), each the runner with the section's
  settings: K=3, SEEDS=<s>, the repository's usage gateway with GATEWAY_RAW=0, the given MODEL, EFFORT, RESULTS, every
  task base by default (or the bases given), ENV_FILE <repo>/.env and JOBS <repo>/jobs/agents by default; each lane's
  output in its own file next to JOBS;
- MODEL, EFFORT (never `default`) and RESULTS are required, ENV_FILE must exist, start_lanes.sh needs SEEDS: else exit 2
  and no lane starts.
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

FAKE_RUNNER = '''#!/usr/bin/env bash
python3 - "$@" <<'PY'
import json, os, sys
keep = ("AGENT", "MODEL", "EFFORT", "K", "SEEDS", "ENV_FILE", "USAGE_GATEWAY", "GATEWAY_RAW", "JOBS", "RESULTS", "MODES")
rec = {"argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in keep}}
with open(os.path.join(os.environ["FAKE_RUNNER_LOG"], f"{os.getppid()}.json"), "w") as f:
    f.write(json.dumps(rec))
PY
'''


class BashRunners(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        repo = self.root / "repo"
        shutil.copytree(REPO / "bash-runners", repo / "bash-runners")
        runner = repo / "scripts" / "harbor" / "run_agent_batch.sh"
        runner.parent.mkdir(parents=True)
        runner.write_text(FAKE_RUNNER)
        runner.chmod(0o755)
        (repo / "scripts" / "usage_gateway").mkdir()
        for b in ("roboarc-a-01-i01", "roboarc-b-02-i01"):
            for m in ("privileged", "standard"):
                (repo / "tasks" / f"{b}-{m}").mkdir(parents=True)
        (repo / ".env").write_text("OPENAI_API_KEY=x\n")
        self.repo = repo
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        (bin_dir / "nvidia-smi").write_text("#!/bin/sh\necho 'GPU 0: NVIDIA Fake GPU'\necho 'GPU 1: NVIDIA Fake GPU'\n")
        (bin_dir / "nvidia-smi").chmod(0o755)
        self.log = self.root / "lanes"
        self.log.mkdir()
        self.env = {k: v for k, v in os.environ.items()
                    if k not in ("AGENT", "MODEL", "EFFORT", "RESULTS", "ENV_FILE", "JOBS", "LANES", "GPUS", "SEEDS", "K")}
        self.env.update(PATH=f"{bin_dir}:{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}",
                        FAKE_RUNNER_LOG=str(self.log), MODEL="openai/test-model", EFFORT="high",
                        RESULTS=str(self.root / "results"))

    def tearDown(self):
        self.tmp.cleanup()

    def start(self, script, *args, **env):
        return subprocess.run(["bash", str(self.repo / "bash-runners" / script), *args], env=dict(self.env, **env),
                              capture_output=True, text=True, timeout=60, cwd=self.root)

    def lanes(self, n):
        """The records of the n lanes started (they run in the background)."""
        for _ in range(300):
            if len(list(self.log.iterdir())) >= n:
                break
            time.sleep(0.1)
        time.sleep(0.5)
        recs = [json.loads(p.read_text()) for p in sorted(self.log.iterdir())]
        self.assertEqual(len(recs), n)
        return recs

    def test_one_script_per_seed(self):
        for s in (0, 1, 2):
            with self.subTest(seed=s):
                p = self.start(f"run_seed{s}.sh", "b")
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                for rec in self.lanes(8):                                   # 2 GPUs, 4 lanes each
                    self.assertEqual(rec["argv"], ["b", "roboarc-a-01-i01", "roboarc-b-02-i01"])
                    self.assertEqual(rec["env"], {
                        "AGENT": "codex", "MODEL": "openai/test-model", "EFFORT": "high", "K": "3", "SEEDS": str(s),
                        "ENV_FILE": str(self.repo / ".env"), "USAGE_GATEWAY": str(self.repo / "scripts" / "usage_gateway"),
                        "GATEWAY_RAW": "0", "JOBS": str(self.repo / "jobs" / "agents"), "RESULTS": str(self.root / "results"),
                        "MODES": None})
                outs = sorted(p.name for p in (self.repo / "jobs").glob(f"b.seeds-{s}.lane*.out"))
                self.assertEqual(outs, [f"b.seeds-{s}.lane{k}.out" for k in range(1, 9)])
                shutil.rmtree(self.log)
                self.log.mkdir()

    def test_lanes_bases_and_other_settings(self):
        p = self.start("run_seed1.sh", "b", "roboarc-b-02-i01", GPUS="1", MODES="standard", AGENT="claude-code",
                       JOBS=str(self.root / "jobs2" / "agents"))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        for rec in self.lanes(4):                                           # GPUS=1: one GPU
            self.assertEqual(rec["argv"], ["b", "roboarc-b-02-i01"])
            self.assertEqual((rec["env"]["MODES"], rec["env"]["AGENT"], rec["env"]["SEEDS"]), ("standard", "claude-code", "1"))
            self.assertEqual(rec["env"]["JOBS"], str(self.root / "jobs2" / "agents"))
        self.assertTrue((self.root / "jobs2" / "b.seeds-1.lane4.out").is_file())
        shutil.rmtree(self.log)
        self.log.mkdir()
        p = self.start("run_seed2.sh", "b", LANES="1")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.lanes(1)

    def test_refused(self):
        for script, env, needle in (("run_seed0.sh", {"MODEL": ""}, "MODEL must be set"),
                                    ("run_seed0.sh", {"EFFORT": ""}, "EFFORT must be set"),
                                    ("run_seed0.sh", {"EFFORT": "default"}, "never default"),
                                    ("run_seed0.sh", {"RESULTS": ""}, "RESULTS must be set"),
                                    ("run_seed0.sh", {"ENV_FILE": str(self.root / "none.env")}, "no such file"),
                                    ("start_lanes.sh", {}, "SEEDS must be set")):
            with self.subTest(env=env):
                p = self.start(script, "b", **env)
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertIn(needle, p.stderr)
        p = self.start("run_seed0.sh")                                      # no batch name: the usage
        self.assertEqual(p.returncode, 2)
        self.assertIn("run_seed0.sh, run_seed1.sh and run_seed2.sh", p.stdout)
        time.sleep(0.5)
        self.assertEqual(list(self.log.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
