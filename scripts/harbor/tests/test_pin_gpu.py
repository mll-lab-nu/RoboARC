"""Tests for scripts/harbor/pin_gpu.py on copies of committed tasks: no GPU, no docker.

    python3 scripts/harbor/tests/test_pin_gpu.py     # stdlib only (Python 3.11+)

- every NVIDIA device of a copy's compose files (environment/ and tests/: main or sim, rcb_monitor, the verifier)
  becomes `device_ids: ["N"]` and no `count:` is left; --check passes then and fails before;
- a copy without an NVIDIA device (IKEA privileged) is left byte for byte;
- a count other than 1, and a task of the repository itself, are refused.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
TOOL = REPO / "scripts" / "harbor" / "pin_gpu.py"


def first(pattern):
    return next(iter(sorted((REPO / "tasks").glob(pattern))), None)


class Pin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def copy(self, src):
        if src is None:
            self.skipTest("the task kind this test copies is not in tasks/")
        return pathlib.Path(shutil.copytree(src, self.root / src.name))

    def run_tool(self, *args):
        return subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True)

    def composes(self, t):
        return [p for d in ("environment", "tests") for p in (t / d).glob("*compose*.y*ml")]

    def test_puzzle_privileged_and_standard(self):
        for src in (first("roboarc-cast-coil-*-privileged"), first("roboarc-cast-coil-*-standard")):
            t = self.copy(src)
            self.assertEqual(self.run_tool("--gpu", 2, "--check", t).returncode, 1)    # not pinned yet
            p = self.run_tool("--gpu", 2, t)
            self.assertEqual(p.returncode, 0, p.stderr)
            text = "".join(c.read_text() for c in self.composes(t))
            self.assertNotIn("count:", text)
            want = 3 if src.name.endswith("-privileged") else 2    # main + rcb_monitor + verifier | sim + rcb_monitor
            self.assertEqual(text.count('device_ids: ["2"]'), want)
            self.assertEqual(self.run_tool("--gpu", 2, "--check", t).returncode, 0)
            self.assertEqual(self.run_tool("--gpu", 3, "--check", t).returncode, 1)    # pinned to another GPU

    def test_no_gpu_copy_unchanged(self):
        t = self.copy(first("roboarc-ikea-*-privileged"))
        before = {c: c.read_bytes() for c in self.composes(t)}
        p = self.run_tool("--gpu", 1, t)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual({c: c.read_bytes() for c in self.composes(t)}, before)
        self.assertEqual(self.run_tool("--gpu", 1, "--check", t).returncode, 0)

    def test_refusals(self):
        t = self.copy(first("roboarc-cast-coil-*-privileged"))
        c = t / "environment" / "docker-compose.yaml"
        c.write_text(c.read_text().replace("count: 1", "count: 2", 1))
        p = self.run_tool("--gpu", 0, t)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("exactly one GPU", p.stderr)
        repo_task = first("roboarc-cast-coil-*-privileged")
        p = self.run_tool("--gpu", 0, repo_task)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("a task of the repository itself", p.stderr)


if __name__ == "__main__":
    unittest.main()
