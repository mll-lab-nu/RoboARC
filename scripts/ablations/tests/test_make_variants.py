"""Tests for scripts/ablations/make_variants.py and experiments.py: the registry, the trial counts of E1..E5, the
framework's task.toml edits and DIGEST, determinism, the locked DIGESTs, and that a failing edit leaves nothing behind.
Standard library only; builds a few variants of committed tasks in a temporary directory:

    python3 scripts/ablations/tests/test_make_variants.py
"""
import pathlib
import sys
import tempfile
import tomllib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts" / "ablations"), str(REPO / "scripts"), str(REPO / "scripts" / "harbor")]
import experiments  # noqa: E402
import make_variants as mv  # noqa: E402
import task_conventions as tc  # noqa: E402
import task_names  # noqa: E402

BASE = "roboarc-bed-bolt-01-i01"


class Registry(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(set(mv.REGISTRY), {"q1c00", "q1c10", "q1c0g", "q1c1g", "cstate", "clr2x", "clr4x", "obsv1", "obsv2"})
        for c in mv.REGISTRY.values():
            self.assertRegex(c.token, r"^[a-z0-9]+$")
            self.assertTrue(set(c.modes) <= {"privileged", "standard"})
            self.assertNotIn('"', c.note)          # it goes into a TOML string unescaped

    def test_experiments(self):
        want = {"E1": 156, "E2": 120, "E3": 36, "E5": 30}
        for eid, n in want.items():
            self.assertEqual(len(experiments.plan(eid)), n, eid)
            self.assertTrue(all(row["seed"] == 0 for row in experiments.plan(eid)))
            for t in experiments.EXPERIMENTS[eid].conditions:
                self.assertIn(t, mv.REGISTRY)
        try:
            n4 = len(experiments.plan("E4"))
        except SystemExit:
            n4 = None                            # cond_clearance.py not merged yet
        if n4 is not None:
            self.assertEqual(n4, 24)
            self.assertEqual(sum(want.values()) + n4, experiments.TOTAL_TRIALS)
        for eid in ("E1", "E2", "E3", "E5"):
            for b in experiments.EXPERIMENTS[eid].bases():
                self.assertTrue((REPO / "tasks" / f"{b}-standard").is_dir(), b)

    def test_batches_and_run_ids(self):
        self.assertEqual(experiments.EXPERIMENTS["E1"].batches(), ["ablation-e1-r1", "ablation-e1-r2", "ablation-e1-r3"])
        self.assertEqual(experiments.EXPERIMENTS["E2"].batches(), ["ablation-e2-r1", "ablation-e2-r2"])
        self.assertEqual({e.run_id for e in experiments.EXPERIMENTS.values()},
                         {"codex-gpt6_astra-high", "claude_code-opus_5_5-high"})

    def test_names_parse(self):
        for eid in ("E1", "E3", "E5"):
            for v in experiments.variants(eid):
                self.assertEqual(task_names.base(v.task), f"{v.base}-{v.token}")
                self.assertEqual(task_names.mode(v.task), v.mode)


class Build(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_toml_digest_and_determinism(self):
        d, digest = mv.make_one(mv.REGISTRY["q1c10"], BASE, "standard", REPO / "tasks", self.out)
        self.assertEqual(d.name, f"{BASE}-q1c10-standard")
        t = tomllib.loads((d / "task.toml").read_text())
        self.assertEqual(t["task"]["name"], f"roboarc/{d.name}")
        self.assertTrue(t["task"]["description"].startswith(f"[{mv.REGISTRY['q1c10'].note}] STANDARD MODE"))
        self.assertEqual(t["metadata"]["ablation"], "q1c10")
        self.assertEqual((d / "DIGEST").read_text().strip(), digest)
        self.assertEqual(tc.digest(d), digest)
        _, again = mv.make_one(mv.REGISTRY["q1c10"], BASE, "standard", REPO / "tasks", self.out)
        self.assertEqual(again, digest)
        lock = mv.load_lock("q1c10").get(d.name)
        if lock is not None:
            self.assertEqual(lock, digest, "digests/q1c10.tsv")

    def test_failure_leaves_nothing(self):
        with self.assertRaises(SystemExit):
            mv.make_one(mv.REGISTRY["q1c00"], "roboarc-no-such-base-i01", "standard", REPO / "tasks", self.out)
        bad = mv.Condition("q1c00", "C00", ("standard",), "cond_info", (("condition", "nope"),), "x")
        with self.assertRaises(ValueError):
            mv.make_one(bad, BASE, "standard", REPO / "tasks", self.out)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_unmerged_module(self):
        c = mv.Condition("zz", "Z", ("standard",), "cond_not_here", (), "x")
        with self.assertRaises(SystemExit):
            mv.make_one(c, BASE, "standard", REPO / "tasks", self.out)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_locks_are_current(self):
        """Every locked DIGEST of one base per lock file is what this checkout builds (the full set: check_variants.sh)."""
        for f in sorted(mv.DIGESTS.glob("*.tsv")):
            token = f.stem
            if token not in mv.REGISTRY:
                continue
            lock = mv.load_lock(token)
            if not lock:
                continue
            name = sorted(lock)[0]
            mode = task_names.mode(name)
            base = task_names.base(name)[: -len(token) - 1]
            try:
                d, digest = mv.make_one(mv.REGISTRY[token], base, mode, REPO / "tasks", self.out)
            except SystemExit as exc:
                # a module not merged yet; E4's build, which re-freezes the scene in the release image (Docker), on a
                # host without the image: check_variants.sh builds and checks the full set where it runs
                if "not on this branch yet" in str(exc) or "needs the release image" in str(exc):
                    continue
                raise
            self.assertEqual(digest, lock[name], name)


if __name__ == "__main__":
    unittest.main()
