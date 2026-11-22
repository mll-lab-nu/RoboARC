"""Tests for scripts/ablations/check_variants.py with a fake harbor (no docker, no GPU): each variant gets an oracle and a
nop run from its own seed-0 copy pinned to a GPU; the pass rule (expected reward, no infrastructure error, the oracle's
determinism and live / replay success); the report; a second run keeps the passed trials and redoes the failed ones;
the gate. Standard library only:

    python3 scripts/ablations/tests/test_check_variants.py
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts" / "ablations"), str(REPO / "scripts")]
import experiments  # noqa: E402
import task_conventions as tc  # noqa: E402

COMPOSE = """services:
  sim:
    image: x
    environment:
      NVIDIA_DRIVER_CAPABILITIES: all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""
FAKE_HARBOR = r'''#!/usr/bin/env python3
import json, os, pathlib, re, sys
a = sys.argv[1:]
task, agent, out, job = (pathlib.Path(a[a.index("-p") + 1]), a[a.index("-a") + 1], pathlib.Path(a[a.index("-o") + 1]),
                         a[a.index("--job-name") + 1])
compose = (task / "environment" / "docker-compose.yaml").read_text()
fail = os.environ.get("FAKE_FAIL", "")
ok = agent == "oracle"
if fail and fail in task.name and agent == "oracle":
    ok = False
t = out / job / f"{task.name[:30]}__x"
(t / "verifier").mkdir(parents=True)
rw = {"success": 1.0 if ok else 0.0, "infra_error": 0, "deterministic": 1, "live_success": 1.0 if ok else 0.0,
      "replay_success": 1.0 if ok else 0.0}
(t / "result.json").write_text(json.dumps({"verifier_result": {"rewards": rw}}))
(t / "verifier" / "reward.txt").write_text(str(rw["success"]))
(t / "pinned.txt").write_text(",".join(re.findall(r'device_ids: \["(\d+)"\]', compose)))
'''


class Check(unittest.TestCase):
    E = "E5"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name)
        repo = self.repo = root / "repo"
        shutil.copytree(REPO / "scripts", repo / "scripts", ignore=shutil.ignore_patterns("__pycache__", "digests", "tests"))
        (repo / "scripts" / "ablations" / "digests").mkdir()
        shutil.copytree(REPO / "images", repo / "images", ignore=shutil.ignore_patterns("*.tar"))
        self.variants = [v.task for v in experiments.variants(self.E)]
        locks = {}
        for t in self.variants:
            d = repo / "tasks" / t
            (d / "environment").mkdir(parents=True)
            (d / "task.toml").write_text(f'[task]\nname = "roboarc/{t}"\n\n[metadata]\nfamily = "bed_bolt"\noracle = "full"\n')
            (d / "environment" / "docker-compose.yaml").write_text(COMPOSE)
            dg = tc.digest(d)
            (d / "DIGEST").write_text(dg + "\n")
            locks.setdefault(t.split("-")[-2], []).append(f"{t}\t{dg}\n")
        for token, rows in locks.items():
            (repo / "scripts" / "ablations" / "digests" / f"{token}.tsv").write_text("".join(rows))
        self.harbor = root / "harbor"
        self.harbor.write_text(FAKE_HARBOR)
        self.harbor.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}")

    def tearDown(self):
        self.tmp.cleanup()

    def cv(self, *args, **env):
        return subprocess.run([sys.executable, str(self.repo / "scripts/ablations/check_variants.py"), *args],
                              env=dict(self.env, **env), capture_output=True, text=True, timeout=600)

    def report(self):
        return json.loads((self.repo / "jobs/ablations/checks" / f"{self.E}.json").read_text())

    def test_run_rerun_gate(self):
        bad = self.variants[3]
        p = self.cv("run", self.E, "--gpus", "0,1", "--per-gpu", "2", "--harbor", str(self.harbor), FAKE_FAIL=bad)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        rep = self.report()
        self.assertEqual((rep["n_variants"], rep["n_pass"], rep["all_pass"]), (30, 29, False))
        failed = [v for v in rep["variants"] if not v["pass"]]
        self.assertEqual([v["task"] for v in failed], [bad])
        self.assertFalse(failed[0]["oracle"]["pass"])
        self.assertTrue(failed[0]["nop"]["pass"])
        for v in rep["variants"]:
            self.assertIn(v["oracle"]["gpu"], ("0", "1"))
            job = self.repo / "jobs/ablations/checks" / self.E / "jobs" / v["oracle"]["job"]
            self.assertEqual(next(job.glob("*/pinned.txt")).read_text(), v["oracle"]["gpu"])   # the copy, pinned
        self.assertEqual((self.repo / "tasks" / bad / "environment/docker-compose.yaml").read_text(), COMPOSE)
        self.assertEqual(self.cv("gate", self.E).returncode, 1)
        # the second run: only the failed oracle runs again
        p = self.cv("run", self.E, "--gpus", "0,1", "--harbor", str(self.harbor))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("1 trials to run (59 kept)", p.stdout)
        self.assertTrue(self.report()["all_pass"])
        self.assertEqual(self.cv("gate", self.E).returncode, 0)
        # a changed variant is no longer covered
        t = self.repo / "tasks" / self.variants[0] / "task.toml"
        t.write_text(t.read_text() + "# x\n")
        g = self.cv("gate", self.E)
        self.assertEqual(g.returncode, 1)
        self.assertIn("DIGEST", g.stderr)

    def test_no_reference_solution_nop_only(self):
        t = self.repo / "tasks" / self.variants[1] / "task.toml"
        t.write_text(t.read_text().replace('oracle = "full"\n', 'oracle = "none"\n'))
        (t.parent / "DIGEST").write_text(tc.digest(t.parent) + "\n")
        lock = self.repo / "scripts/ablations/digests" / f"{self.variants[1].split('-')[-2]}.tsv"
        lock.write_text("".join(l if not l.startswith(self.variants[1] + "\t") else f"{self.variants[1]}\t{tc.digest(t.parent)}\n"
                                for l in lock.read_text().splitlines(keepends=True)))
        p = self.cv("run", self.E, "--gpus", "0", "--harbor", str(self.harbor))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("59 trials to run (0 kept, 1 without a reference solution: nop only)", p.stdout)
        v = next(x for x in self.report()["variants"] if x["task"] == self.variants[1])
        self.assertTrue(v["oracle"]["skipped"] and v["pass"] and v["nop"]["pass"])
        self.assertEqual(self.cv("gate", self.E).returncode, 0)

    def test_render_off_is_not_a_gate(self):
        p = self.cv("run", self.E, "--render", "off", "--harbor", str(self.harbor))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertTrue(self.report()["all_pass"])
        job = self.repo / "jobs/ablations/checks" / self.E / "jobs" / self.report()["variants"][0]["oracle"]["job"]
        self.assertEqual(next(job.glob("*/pinned.txt")).read_text(), "")     # no GPU: rendering off, no reservation
        g = self.cv("gate", self.E)
        self.assertEqual(g.returncode, 1)
        self.assertIn("rendering off", g.stderr)

    def test_unbuilt_variants_refused(self):
        shutil.rmtree(self.repo / "tasks" / self.variants[0])
        p = self.cv("run", self.E, "--gpus", "0", "--harbor", str(self.harbor))
        self.assertEqual(p.returncode, 2)
        self.assertIn("make_variants.py build", p.stderr)


class CheckE4(Check):
    """E4: its own per-variant checks (check_clr2x.py / check_clr4x.py) and its experiment check, the GPU render check
    (check_clearance_render.py), which must pass for the gate; a passed render check is kept while no variant changes."""
    E = "E4"

    def setUp(self):
        super().setUp()
        rb = self.repo / "scripts" / "ablations"
        for k in (2, 4):
            (rb / f"check_clr{k}x.py").write_text("import sys\nprint('ok {}')\n")
        (rb / "check_clearance_render.py").write_text(
            "import os, pathlib, sys\n"
            "pathlib.Path(os.environ['FAKE_RENDER_LOG']).open('a').write(' '.join(sys.argv[1:]) + '\\n')\n"
            "print('ok render'); sys.exit(int(os.environ.get('FAKE_RENDER_RC', '0')))\n")
        self.render_log = pathlib.Path(self.tmp.name) / "render.log"

    def test_run_rerun_gate(self):
        env = dict(FAKE_RENDER_LOG=str(self.render_log))
        p = self.cv("run", self.E, "--gpus", "2,3", "--harbor", str(self.harbor), FAKE_RENDER_RC="1", **env)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        rep = self.report()
        self.assertEqual((rep["n_variants"], rep["n_pass"], rep["all_pass"]), (24, 24, False))
        self.assertTrue(all(v["extra"]["pass"] for v in rep["variants"]))       # check_clr<k>x.py, per variant
        self.assertEqual(self.render_log.read_text().split(), ["--out", "jobs/ablations/checks/e4_render", "--gpu", "2"])
        g = self.cv("gate", self.E)
        self.assertEqual(g.returncode, 1)
        self.assertIn("check_clearance_render.py has not passed", g.stderr)
        p = self.cv("run", self.E, "--gpus", "2,3", "--harbor", str(self.harbor), **env)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.cv("gate", self.E).returncode, 0)
        p = self.cv("run", self.E, "--gpus", "2,3", "--harbor", str(self.harbor), FAKE_RENDER_RC="1", **env)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)            # unchanged variants: the pass is kept
        self.assertEqual(len(self.render_log.read_text().splitlines()), 2)

    test_no_reference_solution_nop_only = None
    test_unbuilt_variants_refused = None


if __name__ == "__main__":
    unittest.main()
