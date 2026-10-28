"""Tests for scripts/harbor/host_info.py, the machine record of a batch (docs/RUNNING.md "Results"), with a fake
nvidia-smi, docker and harbor on PATH: no GPU needed.

    python3 scripts/harbor/tests/test_host_info.py     # stdlib only

- every GPU nvidia-smi lists, with its name, driver and memory; the docker and harbor versions; the CPUs and memory;
- --lines writes the host:, GPUs: and software: lines in MANIFEST.md's form (scripts/send_results.sh);
- without nvidia-smi, docker or harbor the record says so (no GPU, null versions) and the tool still succeeds.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "host_info.py"
FAKES = {
    "nvidia-smi": "#!/bin/sh\nprintf '0, NVIDIA L40S, 580.95.05, 46068\\n1, NVIDIA L40S, 580.95.05, 46068\\n'\n",
    "docker": "#!/bin/sh\necho 'Docker version 28.3.3, build 980b856'\n",
    "harbor": "#!/bin/sh\necho '0.25.1'\n",
}


class HostInfo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bin = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args, fakes=FAKES):
        for name, body in fakes.items():
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)
        p = subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True, timeout=60,
                           env=dict(os.environ, PATH=str(self.bin)))
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout

    def test_record(self):
        h = json.loads(self.run_tool())
        self.assertEqual(h["gpus"], [{"index": "0", "name": "NVIDIA L40S", "driver": "580.95.05", "memory_mib": 46068},
                                     {"index": "1", "name": "NVIDIA L40S", "driver": "580.95.05", "memory_mib": 46068}])
        self.assertEqual((h["docker"], h["harbor"]), ("Docker version 28.3.3, build 980b856", "0.25.1"))
        self.assertEqual(h["cpus"], len(os.sched_getaffinity(0)))
        self.assertIsInstance(h["memory_gb"], int)
        self.assertTrue(h["cpu_model"])

    def test_lines(self):
        host, gpus, software = self.run_tool("--lines").splitlines()
        self.assertRegex(host, r"^host:\s+.+, \d+ CPUs, \d+ GB RAM$")
        self.assertEqual(gpus, "GPUs:       0, NVIDIA L40S, 580.95.05, 46068 MiB; 1, NVIDIA L40S, 580.95.05, 46068 MiB")
        self.assertEqual(software, "software:   Docker version 28.3.3, build 980b856; harbor 0.25.1")
        # MANIFEST.md's reading (the dashboard's): <index>, <name>, <driver> per GPU, ";" between GPUs
        self.assertEqual([re.split(r",\s*", g)[:3] for g in gpus.split(None, 1)[1].split("; ")],
                         [["0", "NVIDIA L40S", "580.95.05"], ["1", "NVIDIA L40S", "580.95.05"]])

    def test_missing_tools(self):
        h = json.loads(self.run_tool(fakes={}))
        self.assertEqual((h["gpus"], h["docker"], h["harbor"]), ([], None, None))
        self.assertIn("GPUs:       none", self.run_tool("--lines", fakes={}))


if __name__ == "__main__":
    unittest.main()
