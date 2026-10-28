"""Tests for scripts/harbor/gpu_slots.py, the host's trial slots (docs/RUNNING.md "GPUs and lanes"): no GPU, no docker.

    python3 scripts/harbor/tests/test_gpu_slots.py     # stdlib only (Python 3.11+)

- up to 4 rigid-task trials share a GPU, packed onto the fullest GPU that has room; a deformable-task trial gets an
  empty GPU of its own, and a task without a GPU device takes none;
- a deformable trial that does not fit is recorded as waiting: the empty GPUs are kept for it and a busy GPU is
  drained (no new rigid trial) until it is empty, then the deformable trial gets it;
- CPUs and memory: a trial starts only when the reserved CPUs and memory plus its own fit the host; with nothing else
  held it always starts;
- the slots of a runner that has ended are dropped, and release frees a job's or a runner's slots;
- classify reads a task's family, its declared minimums and whether its compose files reserve a GPU.
"""
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
TOOL = REPO / "scripts" / "harbor" / "gpu_slots.py"
sys.path.insert(0, str(TOOL.parent))
import gpu_slots  # noqa: E402

ME = __import__("os").getpid()


def dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


class Slots(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def acquire(self, job, units, cpus=1, mem=1, gpus="0,1", host_cpus=64, host_mem=256, pid=ME, headroom=None):
        args = [sys.executable, str(TOOL), "acquire", "--dir", self.dir, "--job", job, "--runner-pid", str(pid),
                "--units", str(units), "--cpus", str(cpus), "--mem-gb", str(mem), "--gpus", gpus, "--share", "4",
                "--host-cpus", str(host_cpus), "--host-mem-gb", str(host_mem)]
        if headroom is not None:
            args += ["--headroom-gb", str(headroom)]
        p = subprocess.run(args, capture_output=True, text=True)
        self.assertIn(p.returncode, (0, 3), p.stderr)
        return (p.stdout.strip() if p.returncode == 0 else None), p.stderr.strip()

    def release(self, job):
        subprocess.run([sys.executable, str(TOOL), "release", "--dir", self.dir, "--job", job], check=True)

    def state(self):
        return json.loads((pathlib.Path(self.dir) / "state.json").read_text())

    def test_rigid_trials_pack_four_per_gpu(self):
        got = [self.acquire(f"r{i}", 1)[0] for i in range(9)]
        self.assertEqual(got[:8], ["0"] * 4 + ["1"] * 4)
        self.assertIsNone(got[8])                                  # both GPUs hold 4: the ninth waits

    def test_deformable_trial_gets_an_empty_gpu(self):
        self.assertEqual(self.acquire("r0", 1)[0], "0")
        self.assertEqual(self.acquire("d0", 4)[0], "1")            # GPU 0 is busy: the empty one
        gpu, why = self.acquire("d1", 4)
        self.assertIsNone(gpu)
        self.assertIn("none is empty", why)
        self.assertIn("d1", self.state()["waiting"])

    def test_waiting_deformable_drains_a_gpu(self):
        self.acquire("r0", 1)                                      # GPU 0: 1 rigid
        self.acquire("d0", 4)                                      # GPU 1: deformable
        self.assertIsNone(self.acquire("d1", 4)[0])                # waits: GPU 0 is drained
        self.assertEqual(self.state()["draining"], ["0"])
        gpu, why = self.acquire("r1", 1)                           # no rigid trial goes to the drained GPU
        self.assertIsNone(gpu)
        self.assertIn("draining", why)
        self.release("r0")
        self.assertEqual(self.acquire("d1", 4)[0], "0")            # empty now: the deformable trial gets it
        self.assertNotIn("d1", self.state()["waiting"])

    def test_empty_gpu_is_kept_for_a_waiting_deformable(self):
        self.acquire("d0", 4, gpus="0,1")
        self.acquire("r0", 1, gpus="0,1")                          # GPU 1
        st = self.state()
        st["waiting"]["d9"] = {"pid": ME, "start": gpu_slots.proc_start(ME), "t": time.time()}
        (pathlib.Path(self.dir) / "state.json").write_text(json.dumps(st))
        self.release("d0")                                         # GPU 0 empties while d9 waits
        self.assertEqual(self.acquire("r1", 1)[0], "1")            # rigid: onto GPU 1 (room), not the kept GPU 0
        for i in range(2, 4):
            self.acquire(f"r{i}", 1)
        gpu, why = self.acquire("r9", 1)                           # GPU 1 full, GPU 0 kept: wait
        self.assertIsNone(gpu)
        self.assertIn("kept for", why)
        self.assertEqual(self.acquire("d9", 4)[0], "0")

    def test_single_gpu_alternates(self):
        for i in range(3):
            self.assertEqual(self.acquire(f"r{i}", 1, gpus="0")[0], "0")
        self.assertIsNone(self.acquire("d0", 4, gpus="0")[0])      # waits; GPU 0 drains
        self.assertIsNone(self.acquire("r3", 1, gpus="0")[0])
        for i in range(3):
            self.release(f"r{i}")
        self.assertEqual(self.acquire("d0", 4, gpus="0")[0], "0")

    def test_no_gpu_task(self):
        gpu, _ = self.acquire("ikea", 0)
        self.assertEqual(gpu, "")
        self.assertIsNone(self.state()["holds"]["ikea"]["gpu"])

    def test_cpu_and_memory_admission(self):
        self.assertEqual(self.acquire("a", 1, cpus=4, mem=7, host_cpus=8)[0], "0")
        self.assertEqual(self.acquire("b", 1, cpus=4, mem=7, host_cpus=8)[0], "0")
        gpu, why = self.acquire("c", 1, cpus=4, mem=7, host_cpus=8)
        self.assertIsNone(gpu)
        self.assertIn("CPUs: needs 4, 8 of 8 reserved", why)
        gpu, why = self.acquire("m", 0, cpus=0, mem=7, host_mem=20, headroom=4)   # 14 of 16 GB reserved
        self.assertIsNone(gpu)
        self.assertIn("memory", why)

    def test_alone_always_starts(self):
        self.assertEqual(self.acquire("big", 1, cpus=100, mem=1000, host_cpus=8, host_mem=16)[0], "0")
        self.assertIsNone(self.acquire("next", 1, cpus=1, mem=1, host_cpus=8, host_mem=16)[0])

    def test_dead_runner_slots_are_dropped(self):
        st = {"holds": {"old": {"pid": dead_pid(), "start": "1", "gpu": "0", "units": 4, "cpus": 1, "mem_gb": 1}},
              "waiting": {}, "draining": []}                       # a runner that has ended (killed: no release)
        (pathlib.Path(self.dir) / "state.json").write_text(json.dumps(st))
        self.assertEqual(self.acquire("d", 4, gpus="0")[0], "0")   # its GPU is free again
        self.assertNotIn("old", self.state()["holds"])

    def test_release_by_runner(self):
        self.acquire("a", 1)
        self.acquire("b", 1)
        subprocess.run([sys.executable, str(TOOL), "release", "--dir", self.dir, "--runner-pid", str(ME)], check=True)
        self.assertEqual(self.state()["holds"], {})

    def test_waiting_expires(self):
        self.acquire("r", 1, gpus="0")
        self.acquire("d", 4, gpus="0")
        st = self.state()
        st["waiting"]["d"]["t"] -= gpu_slots.WAIT_TTL_S + 1
        (pathlib.Path(self.dir) / "state.json").write_text(json.dumps(st))
        self.assertEqual(self.acquire("r2", 1, gpus="0")[0], "0")  # no longer drained

    def test_classify_tasks(self):
        tasks = REPO / "tasks"
        rigid = next(iter(sorted(tasks.glob("roboarc-cast-coil-*-privileged"))), None)
        cloth = next(iter(sorted(tasks.glob("roboarc-blanket-fold-*-privileged"))), None)
        ikea_p = next(iter(sorted(tasks.glob("roboarc-ikea-*-privileged"))), None)
        ikea_s = next(iter(sorted(tasks.glob("roboarc-ikea-*-standard"))), None)
        if not all((rigid, cloth, ikea_p, ikea_s)):
            self.skipTest("the task families this test reads are not in tasks/")
        deform = set(gpu_slots.DEFORMABLE.split())
        self.assertEqual(gpu_slots.classify(rigid, deform, 4)[:3], (1, 4.0, 7.0))
        self.assertEqual(gpu_slots.classify(cloth, deform, 4)[0], 4)
        self.assertEqual(gpu_slots.classify(ikea_p, deform, 4)[0], 0)      # no GPU device in its compose files
        self.assertEqual(gpu_slots.classify(ikea_s, deform, 4)[0], 1)      # the sim sidecar has one
        self.assertEqual(gpu_slots.classify(rigid, deform, 4, 0.5, 0.5)[1:3], (2.0, 3.5))


if __name__ == "__main__":
    unittest.main()
