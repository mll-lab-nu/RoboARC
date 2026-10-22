"""Tests for tools/release/restamp.py on a committed privileged reference (standard library; numpy, when installed,
reads the result back):

    python3 tools/tests/test_restamp.py

- the re-stamped archive differs from the reference in `instance_sha.npy` only, and names the new hash;
- --check names any other change (an action row, a zip entry's date) and a wrong hash;
- a malformed hash and an existing output are refused.
"""
import pathlib
import subprocess
import sys
import tempfile
import unittest
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "release"))
import restamp  # noqa: E402


def a_reference() -> pathlib.Path:
    for p in sorted((REPO / "tasks").glob("roboarc-*-privileged/solution/trajectory.npz"), key=lambda p: p.stat().st_size):
        return p
    raise unittest.SkipTest("no committed privileged reference")


class Restamp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        self.src = a_reference()

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_the_hash_changes(self):
        out = self.d / "s1.npz"
        old = restamp.restamp(self.src, "0123456789abcdef", out)
        self.assertRegex(old, r"^[0-9a-f]{16}$")
        self.assertEqual(restamp.check(self.src, out, "0123456789abcdef"), [])
        self.assertNotEqual(restamp.check(self.src, out, old), [])
        try:
            import numpy as np
        except ImportError:
            return
        z = np.load(out)
        self.assertEqual(str(z["instance_sha"]), "0123456789abcdef")
        ref = np.load(self.src)
        self.assertTrue((z["actions"] == ref["actions"]).all())
        self.assertEqual(int(z["scene_seed"]), int(ref["scene_seed"]))

    def test_check_names_other_changes(self):
        out = self.d / "s1.npz"
        restamp.restamp(self.src, "0123456789abcdef", out)
        members = restamp.read(out)
        bad = self.d / "bad.npz"
        with zipfile.ZipFile(bad, "w") as z:
            for info, data in members:
                if info.filename == "actions.npy":
                    data = data[:-1] + bytes([data[-1] ^ 1])
                z.writestr(info, data)
        self.assertTrue(any("actions.npy: bytes differ" in b for b in restamp.check(self.src, bad)))
        dated = self.d / "dated.npz"
        with zipfile.ZipFile(dated, "w") as z:
            for info, data in members:
                info.date_time = (2026, 10, 21, 0, 0, 0)
                z.writestr(info, data)
        self.assertTrue(any("zip entry changed" in b for b in restamp.check(self.src, dated)))

    def test_refused(self):
        tool = REPO / "tools/release/restamp.py"
        p = subprocess.run([sys.executable, str(tool), str(self.src), "--sha", "XYZ", "--out", str(self.d / "x.npz")],
                           capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        (self.d / "y.npz").write_bytes(b"")
        p = subprocess.run([sys.executable, str(tool), str(self.src), "--sha", "0" * 16, "--out", str(self.d / "y.npz")],
                           capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("exists", p.stderr)


if __name__ == "__main__":
    unittest.main()
