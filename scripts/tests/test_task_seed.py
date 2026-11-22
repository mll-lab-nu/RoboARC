"""Tests for scripts/task_seed.py (assign, materialize) and ci_checks/check-seed-files.py, on fixture task pairs made
from a committed task (docs/RUNNING.md "Seeds"). Standard library only:

    python3 scripts/tests/test_task_seed.py
"""
import collections
import hashlib
import importlib.util
import json
import pathlib
import re
import shutil
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import task_seed  # noqa: E402
sys.path.insert(0, str(REPO / "scripts" / "roboarc"))
import mk_tasks  # noqa: E402

spec = importlib.util.spec_from_file_location("check_seed_files", REPO / "ci_checks" / "check-seed-files.py")
check_seed_files = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_seed_files)


def a_pair(ikea: bool = False):
    """A committed privileged / standard pair (base name): a puzzle base (three appearance seeds since 1.2.0), or with
    ikea=True an IKEA base (three seeds of one frozen scene since 2026-10-27: the standard mode's lighting)."""
    for d in sorted((REPO / "tasks").glob("roboarc-ikea-*-privileged" if ikea else "roboarc-*-privileged")):
        if not ikea and d.name.startswith("roboarc-ikea-"):
            continue
        if (d.with_name(d.name.replace("-privileged", "-standard")) / "task.toml").is_file():
            return d.name.removesuffix("-privileged")
    raise unittest.SkipTest("no task pair in tasks/")


def tree(d: pathlib.Path) -> dict:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


def three_seeds(task: pathlib.Path, oracle="[0, 1, 2]") -> None:
    """Make a copied task allow seeds 0, 1, 2 built at run time, with the files check-seed-files.py asks for (a task the
    generator rendered with seeds has them already)."""
    p = task / "task.toml"
    if 'build = "runtime"' in p.read_text():          # rendered with seeds: only the oracle seeds are set here
        p.write_text(re.sub(r"(?m)^oracle = \[.*$", f"oracle = {oracle}", p.read_text(), count=1))
        return
    s = re.sub(r"(?m)^allowed = .*$", "allowed = [0, 1, 2]", p.read_text(), count=1)
    s = re.sub(r"(?m)^oracle = \[.*$", f"oracle = {oracle}\nbuild = \"runtime\"", s, count=1)
    p.write_text(s)
    for place in task_seed.places(task):
        d = task / place
        (d / "seed").write_text("0\n")
        if (d / "runtime_binding.json").is_file():
            for k in (1, 2):
                (d / f"runtime_binding.s{k}.json").write_text(f'{{"seed": {k}}}\n')
    if (task / "solution" / "trajectory.npz").is_file():
        for k in (1, 2):
            shutil.copyfile(task / "solution" / "trajectory.npz", task / "solution" / f"trajectory.s{k}.npz")
    # the images stage the seed: the generator's own text edits (scripts/roboarc/mk_tasks.py seed_texts)
    mode = "standard" if task.name.endswith("-standard") else "privileged"
    base = task.name.removeprefix("roboarc-").removesuffix(f"-{mode}")
    names = ("environment/Dockerfile", "tests/Dockerfile") if mode == "privileged" else \
        ("environment/docker-compose.yaml", "tests/Dockerfile")
    texts = {n: (task / n).read_text() for n in names}
    texts["task.toml"] = ""
    mk_tasks.seed_texts(base, mode, texts, [1, 2], None)
    for n in names:
        (task / n).write_text(texts[n])


class Assign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(cls.tmp.name)
        cls.base = a_pair()
        cls.p = root / f"{cls.base}-privileged"
        cls.s = root / f"{cls.base}-standard"
        for t in (cls.p, cls.s):
            shutil.copytree(REPO / "tasks" / t.name, t, symlinks=True)
            three_seeds(t)
        cls.one = root / f"{a_pair(ikea=True)}-privileged"          # a task that allows its default seed only
        shutil.copytree(REPO / "tasks" / cls.one.name, cls.one, symlinks=True)
        p = cls.one / "task.toml"
        p.write_text(re.sub(r'(?m)^build = "runtime"\n', "", re.sub(r"(?m)^allowed = .*$", "allowed = [0]", p.read_text(),
                                                                      count=1)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_distinct_and_fixed(self):
        a = task_seed.assign(self.p, "b1", 3)
        self.assertEqual(sorted(a), [0, 1, 2])
        self.assertEqual(a, task_seed.assign(self.p, "b1", 3))            # fixed by batch + base
        self.assertEqual(a, task_seed.assign(self.s, "b1", 3))            # the twins get the same seeds
        self.assertEqual(task_seed.assign(self.p, "b1", 2), a[:2])        # a trial's seed does not depend on K
        self.assertEqual(task_seed.assign(self.p, "b1", 7), a + a + a[:1])  # past the last seed the order starts over
        orders = {tuple(task_seed.assign(self.p, f"batch{i}", 3)) for i in range(60)}
        self.assertGreater(len(orders), 3)                                # the batch name changes the order

    def test_uniform_marginal(self):
        n = 3000
        for k in range(3):
            c = collections.Counter(task_seed.assign(self.p, f"b{i}", 3)[k] for i in range(n))
            for seed in (0, 1, 2):
                self.assertTrue(0.29 < c[seed] / n < 0.38, (k, c))

    def test_single_seed(self):
        self.assertEqual(task_seed.assign(self.one, "b1", 3), [0, 0, 0])

    def test_range(self):
        t = self.p.with_name(self.p.name.replace("-privileged", "-x-privileged"))
        shutil.copytree(self.p, t, symlinks=True)
        p = t / "task.toml"
        p.write_text(re.sub(r"(?m)^allowed = .*$", "allowed = { from = 10, to = 13 }", p.read_text(), count=1))
        a = task_seed.assign(t, "b", 6)
        self.assertEqual(sorted(a[:4]), [10, 11, 12, 13])
        self.assertTrue(all(10 <= x <= 13 for x in a))
        self.assertEqual(len(set(a[4:])), 2)


class Materialize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.base = a_pair()
        self.src = self.root / "src"
        for m in ("privileged", "standard"):
            t = self.src / f"{self.base}-{m}"
            shutil.copytree(REPO / "tasks" / t.name, t, symlinks=True)
            three_seeds(t)
            g = t / "environment" / "goal"
            if g.exists():
                shutil.rmtree(g)
            g.mkdir()
            (g / "target_cam_0.png").write_bytes(b"seed 0 image")
        p = self.src / f"{self.base}-privileged"
        for k in (1, 2):                    # the goal images of seeds 1 and 2: the standard twin finds them in the holder
            d = p / "tests" / "seeds" / str(k) / "goal"
            d.mkdir(parents=True)
            (d / "target_cam_0.png").write_bytes(f"seed {k} image".encode())

    def tearDown(self):
        self.tmp.cleanup()

    def test_seed_files_goal_and_toml(self):
        for m, places in (("privileged", ["environment", "tests"]), ("standard", ["environment/sim", "tests"])):
            t = self.src / f"{self.base}-{m}"
            self.assertEqual(task_seed.places(t), places)
            dst = task_seed.materialize(t, 2, self.root / f"out-{m}")
            for place in places:
                self.assertEqual((dst / place / "seed").read_text(), "2\n")
            self.assertEqual((dst / "environment" / "goal" / "target_cam_0.png").read_bytes(), b"seed 2 image")
            self.assertFalse((dst / "tests" / "seeds").exists())
            self.assertRegex((dst / "task.toml").read_text(), r"(?m)^default = 2$")
            d0 = task_seed.materialize(t, 0, self.root / f"out0-{m}")
            for place in places:
                self.assertEqual((d0 / place / "seed").read_text(), "0\n")
            self.assertEqual((d0 / "environment" / "goal" / "target_cam_0.png").read_bytes(), b"seed 0 image")

    def test_refused(self):
        t = self.src / f"{self.base}-privileged"
        with self.assertRaises(SystemExit):
            task_seed.materialize(t, 3, self.root / "o3")                      # not allowed
        shutil.rmtree(t / "tests" / "seeds" / "1")
        with self.assertRaises(SystemExit):
            task_seed.materialize(t, 1, self.root / "o1")                      # no goal images of seed 1
        self.assertFalse((self.root / "o1" / t.name).exists())

    def test_check_seed_files(self):
        for m in ("privileged", "standard"):
            t = self.src / f"{self.base}-{m}"
            self.assertEqual(check_seed_files.check(t), [], m)
        t = self.src / f"{self.base}-standard"
        (t / "tests" / "runtime_binding.s2.json").unlink()
        (t / "environment" / "sim" / "seed").write_text("1\n")
        errs = check_seed_files.check(t)
        self.assertTrue(any("tests/runtime_binding.s2.json missing" in e for e in errs), errs)
        self.assertTrue(any("environment/sim/seed is '1'" in e for e in errs), errs)
        p = self.src / f"{self.base}-privileged"
        (p / "tests" / "seeds" / "2" / "goal" / "target_cam_0.png").rename(p / "tests" / "seeds" / "2" / "goal" / "x.png")
        if (p / "solution" / "trajectory.s1.npz").is_file():
            (p / "solution" / "trajectory.s1.npz").unlink()
            self.assertTrue(any("trajectory.s1.npz missing" in e for e in check_seed_files.check(p)))
        self.assertTrue(any("not the files of environment/goal/" in e for e in check_seed_files.check(p)))

    def test_ikea(self):
        """A committed IKEA task: materialized on its default seed it is the task, byte for byte; on seed 2 its places
        carry the seed. Its seeds need no bindings, only the standard rig's lighting of every seed."""
        base = a_pair(ikea=True)
        for m in ("privileged", "standard"):
            t = REPO / "tasks" / f"{base}-{m}"
            dst = task_seed.materialize(t, 0, self.root / f"plain-{m}")
            self.assertEqual(tree(dst), tree(t))
            self.assertEqual(check_seed_files.check(t), [])
            d2 = task_seed.materialize(t, 2, self.root / f"two-{m}")
            for place in task_seed.places(t):
                self.assertEqual((d2 / place / "seed").read_text(), "2\n")
        t = self.root / f"{base}-standard"
        shutil.copytree(REPO / "tasks" / t.name, t, symlinks=True)
        rig = t / "environment" / "sim" / "rig.json"
        spec = json.loads(rig.read_text())
        del spec["seeds"]["2"]
        rig.write_text(json.dumps(spec))
        self.assertEqual(check_seed_files.check(t), ["environment/sim/rig.json: no lighting of seed 2 (seeds)"])


if __name__ == "__main__":
    unittest.main()
