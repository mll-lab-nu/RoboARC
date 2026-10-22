"""Tests for images/pull_all.sh with a fake `docker` (no network, no registry), standard library only:

    python3 images/tests/test_pull_all.py

- --list prints "<ref> <id>" for every pin of the repository's images/pins.json, in its order;
- a malformed pin (a tag without its digest, an id that is not a sha256) stops before any pull;
- every image is pulled by its pinned reference and checked against its pinned id: a different id fails the run.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "images/pull_all.sh"
FAKE_DOCKER = """#!{py}
import json, os, sys
log = os.environ["FAKE_DOCKER_LOG"]
with open(log, "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:3] == ["image", "inspect"]:
    ids = json.loads(os.environ["FAKE_DOCKER_IDS"])
    print(ids.get(sys.argv[-1], ""))
"""
A = "ghcr.io/example/base:1.0@sha256:" + "a" * 64
B = "ghcr.io/example/sim:1.0@sha256:" + "b" * 64
IA, IB = "sha256:" + "1" * 64, "sha256:" + "2" * 64


class PullAll(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        (self.d / "bin").mkdir()
        docker = self.d / "bin" / "docker"
        docker.write_text(FAKE_DOCKER.format(py=sys.executable))
        docker.chmod(0o755)
        self.log = self.d / "docker.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def pins(self, images):
        p = self.d / "pins.json"
        p.write_text(json.dumps({"images": images}))
        return p

    def run_it(self, *args, ids=None):
        env = dict(os.environ, PATH=f"{self.d / 'bin'}:{os.environ['PATH']}", FAKE_DOCKER_LOG=str(self.log),
                   FAKE_DOCKER_IDS=json.dumps(ids or {}))
        return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=60)

    def calls(self):
        return [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_list_of_the_repository_pins(self):
        p = self.run_it("--list")
        self.assertEqual(p.returncode, 0, p.stderr)
        want = [f"{x['ref']} {x['id']}" for x in json.loads((REPO / "images/pins.json").read_text())["images"]]
        self.assertEqual(p.stdout.splitlines(), want)
        self.assertEqual(self.calls(), [])

    def test_malformed_pins_stop_before_any_pull(self):
        for bad in ({"name": "x", "ref": "ghcr.io/example/base:1.0", "id": IA},
                    {"name": "x", "ref": A, "id": "1234"}, {"name": "x"}):
            p = self.run_it("--pins", str(self.pins([{"name": "ok", "ref": B, "id": IB}, bad])))
            self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
            self.assertEqual(self.calls(), [])
        self.assertEqual(self.run_it("--pins", str(self.d / "missing.json")).returncode, 2)

    def test_pull_in_order_and_check_ids(self):
        pins = self.pins([{"name": "base", "ref": A, "id": IA}, {"name": "sim", "ref": B, "id": IB}])
        p = self.run_it("--pins", str(pins), ids={A: IA, B: IB})
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        pulls = [c[-1] for c in self.calls() if c[0] == "pull"]
        self.assertEqual(pulls, [A, B])
        self.log.unlink()
        p = self.run_it("--pins", str(pins), ids={A: IA, B: "sha256:" + "9" * 64})
        self.assertEqual(p.returncode, 1)
        self.assertIn(f"FAIL {B}: image id", p.stderr)


if __name__ == "__main__":
    unittest.main()
