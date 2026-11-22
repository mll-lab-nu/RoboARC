#!/usr/bin/env python3
"""scripts/vla_droid.sh --gpu with a recording `docker`, `curl` and model environment (no GPU, no image, no network,
a few seconds):

    python3 scripts/tests/test_vla_droid.py

- without --gpu every container gets `--gpus all`, as before, and the model process's CUDA variables are untouched;
- --gpu N gives every container (the episodes, the `hold` stand-in) `--gpus device=N` and the model process, which
  runs on the host, CUDA_VISIBLE_DEVICES=N in nvidia-smi's numbering (CUDA_DEVICE_ORDER=PCI_BUS_ID); a UUID passes too;
- --gpu takes one GPU: a list is refused before anything starts.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/vla_droid.sh"
FAKE = """#!{py}
import json, os, sys, time
tool, log = os.path.basename(sys.argv[0]), os.environ["FAKE_LOG"]
server = "roboarc.vla.droid_adapter" in sys.argv
if tool == "curl":          # the health check: ready once the model server has started
    for _ in range(200):
        if os.path.exists(log) and '"server": true' in open(log).read():
            sys.exit(0)
        time.sleep(0.05)
    sys.exit(1)
with open(log, "a") as f:
    f.write(json.dumps({{"tool": tool, "argv": sys.argv[1:], "server": server,
                         "cuda": [os.environ.get("CUDA_DEVICE_ORDER"), os.environ.get("CUDA_VISIBLE_DEVICES")]}}) + "\\n")
if server:                  # runs until the script stops it
    time.sleep(120)
print(tool, "done")
"""


class Gpu(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        for tool in ("bin/docker", "bin/curl", "openpi/.venv/bin/python"):
            f = self.d / tool
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(FAKE.format(py=sys.executable))
            f.chmod(0o755)
        (self.d / "vla-lib/websockets").mkdir(parents=True)
        self.log = self.d / "log.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def run_it(self, *args):
        if self.log.exists():
            self.log.unlink()
        env = {k: v for k, v in os.environ.items() if not k.startswith("CUDA_")}
        env.update(PATH=f"{self.d / 'bin'}:{env['PATH']}", FAKE_LOG=str(self.log), VLA_LIB=str(self.d / "vla-lib"),
                   OPENPI=str(self.d / "openpi"))
        p = subprocess.run(["bash", str(SCRIPT), "--task", "lock-picking-03", "--out", str(self.d / "out"), *args],
                           capture_output=True, text=True, env=env, timeout=120)
        calls = [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []
        return p, calls

    def gpus(self, calls):
        """The --gpus value of every `docker run`."""
        runs = [c["argv"] for c in calls if c["tool"] == "docker" and c["argv"][0] == "run"]
        return [a[a.index("--gpus") + 1] for a in runs]

    def test_default_is_unchanged(self):
        p, calls = self.run_it("--model", "hold", "--seed", "0,1")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.gpus(calls), ["all"] * 3)             # the hold server and two episodes
        p, calls = self.run_it("--model", "jointpos")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.gpus(calls), ["all"])
        self.assertEqual([c["cuda"] for c in calls if c["tool"] == "python"], [[None, None]])

    def test_one_gpu_for_the_server_and_the_episodes(self):
        p, calls = self.run_it("--model", "hold", "--seed", "0,1", "--gpu", "2")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.gpus(calls), ["device=2"] * 3)
        p, calls = self.run_it("--model", "jointpos", "--gpu", "2")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.gpus(calls), ["device=2"])
        self.assertEqual([c["cuda"] for c in calls if c["tool"] == "python"], [["PCI_BUS_ID", "2"]])
        uuid = "GPU-8f6d5c2e-1b7a-4c3d-9e0f-123456789abc"
        p, calls = self.run_it("--policy", "ws://127.0.0.1:8000", "--gpu", uuid)
        self.assertEqual((p.returncode, self.gpus(calls)), (0, ["device=" + uuid]))

    def test_one_gpu_only(self):
        for bad in ("0,1", "all", "device=0"):
            p, calls = self.run_it("--model", "hold", "--gpu", bad)
            self.assertEqual((p.returncode, calls), (2, []), bad)
            self.assertIn("--gpu takes one GPU", p.stderr)


if __name__ == "__main__":
    unittest.main()
