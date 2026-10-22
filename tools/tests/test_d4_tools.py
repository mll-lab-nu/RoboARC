"""Tests of the appearance-seed evidence tools that run without the image (standard library only):

    python3 tools/tests/test_d4_tools.py

- tools/release/state_parts.py compare: equal but `task` passes, any other part differing (or a part missing) fails;
- tools/release/d4_table.py: one row per `D4 {...}` line, the pass count, bases without a result named.
"""
import json, pathlib, subprocess, sys, tempfile, unittest

REL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "release"


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


class Tools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def parts(self, name, **parts):
        base = dict(world="w0", robot="r0", targets="t0", state="s0", task="k0")
        base.update(parts)
        p = self.d / name
        p.write_text(json.dumps({"kind": "end", "replays": [{"parts": base, "state_sha": "x", "success": True}]}))
        return p

    def test_compare(self):
        a = self.parts("a.json")
        self.assertEqual(run(REL / "state_parts.py", "compare", a, self.parts("b.json", task="k1")).returncode, 0)
        p = run(REL / "state_parts.py", "compare", a, self.parts("c.json", world="w1"))
        self.assertEqual(p.returncode, 1)
        self.assertIn("world", json.loads(p.stdout)["parts_differing"])
        c = self.d / "birth.json"
        c.write_text(json.dumps({"kind": "birth", "parts": dict(world="w0", robot="r0", targets="t0", state="s0")}))
        self.assertEqual(run(REL / "state_parts.py", "compare", a, c).returncode, 1)     # a part missing

    def test_table(self):
        out = self.d / "out"
        out.mkdir()
        row = dict(base="b-01-i01", seed=1, birth="ab", a_files=True, a_fields=True, a_model=True, b_strict=True,
                   c_birth=True, c_birth_parts_differing=["task"], d_restamp=True, ok=True)
        (out / "b-01-i01.log").write_text("noise\nD4 " + json.dumps(row) + "\nD4 " + json.dumps(dict(row, seed=2, ok=False)) + "\nrc=0\n")
        (out / "c-01-i01.log").write_text("Traceback\nrc=1\n")
        p = run(REL / "d4_table.py", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("1 / 2 seed instances pass", p.stdout)
        self.assertIn("c-01-i01", p.stdout)
        self.assertEqual(len(json.loads((out / "d4.json").read_text())), 2)


if __name__ == "__main__":
    unittest.main()
