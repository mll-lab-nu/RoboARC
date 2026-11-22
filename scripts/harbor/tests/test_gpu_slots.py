"""Tests for scripts/harbor/gpu_slots.py, the host's trial slots (docs/RUNNING.md "GPUs and lanes"): no GPU, no docker.

    python3 scripts/harbor/tests/test_gpu_slots.py     # stdlib only (Python 3.11+)

- up to 4 rigid-task trials share a GPU, packed onto the fullest GPU that has room; a deformable-task trial gets an
  empty GPU of its own, and a task without a GPU device takes none;
- a deformable trial that does not fit is recorded as waiting: the empty GPUs are kept for it and a busy GPU is
  drained (no new rigid trial) until it is empty, then the deformable trial gets it;
- CPUs and memory: a trial starts only when the reserved CPUs and memory plus its own fit the host; with nothing else
  held it always starts;
- the slots of a runner that has ended are dropped, and release frees a job's or a runner's slots;
- a trial whose agent phase ended validates on its GPU (`phase`): it frees its reasoning unit and keeps the verifier's
  CPUs and memory only, its hold passes to the process that waits for its validation, a GPU holds at most --resident
  trials in either phase, and a deformable trial still needs a GPU on which no trial reasons or validates;
- classify reads a task's family, its declared minimums and whether its compose files reserve a GPU; a GPU of its own
  for the families of scripts/task_resources.toml's own_gpu classes (or of --deformable), and no classification for a
  RoboARC task whose family is in no class; with --phase validate the verifier's CPUs and memory.
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

    def acquire(self, job, units, cpus=1, mem=1, gpus="0,1", host_cpus=64, host_mem=256, pid=ME, headroom=None,
                resident=None):
        args = [sys.executable, str(TOOL), "acquire", "--dir", self.dir, "--job", job, "--runner-pid", str(pid),
                "--units", str(units), "--cpus", str(cpus), "--mem-gb", str(mem), "--gpus", gpus, "--share", "4",
                "--host-cpus", str(host_cpus), "--host-mem-gb", str(host_mem)]
        if headroom is not None:
            args += ["--headroom-gb", str(headroom)]
        if resident is not None:
            args += ["--resident", str(resident)]
        p = subprocess.run(args, capture_output=True, text=True)
        self.assertIn(p.returncode, (0, 3), p.stderr)
        return (p.stdout.strip() if p.returncode == 0 else None), p.stderr.strip()

    def release(self, job):
        subprocess.run([sys.executable, str(TOOL), "release", "--dir", self.dir, "--job", job], check=True)

    def phase(self, job, pid=ME, cpus=1, mem=1):
        return subprocess.run([sys.executable, str(TOOL), "phase", "--dir", self.dir, "--job", job, "--pid", str(pid),
                               "--cpus", str(cpus), "--mem-gb", str(mem)], capture_output=True, text=True)

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

    def test_validation_frees_the_reasoning_unit(self):
        for i in range(4):
            self.assertEqual(self.acquire(f"r{i}", 1, gpus="0")[0], "0")
        self.assertIsNone(self.acquire("r4", 1, gpus="0")[0])          # four trials reason on GPU 0
        self.assertEqual(self.phase("r0").returncode, 0)                # r0's agent phase ended: it validates there
        h = self.state()["holds"]["r0"]
        self.assertEqual((h["gpu"], h["units"], h["phase"]), ("0", 0, "validate"))
        self.assertEqual(self.acquire("r4", 1, gpus="0")[0], "0")       # its reasoning unit is free

    def test_resident_cap(self):
        for i in range(4):
            self.acquire(f"r{i}", 1, gpus="0", resident=5)
        self.phase("r0")
        self.assertEqual(self.acquire("r4", 1, gpus="0", resident=5)[0], "0")   # 4 reasoning + 1 validating
        self.phase("r1")
        gpu, why = self.acquire("r5", 1, gpus="0", resident=5)        # a unit is free, but 5 trials are on GPU 0
        self.assertIsNone(gpu)
        self.assertIn("validating", why)
        for i in range(2, 5):                                           # the default: twice the share (8)
            self.phase(f"r{i}")
        got = [self.acquire(f"s{i}", 1, gpus="0")[0] for i in range(4)]
        self.assertEqual(got[:3], ["0"] * 3)                            # 5 validating + 3 reasoning
        self.assertIsNone(got[3])

    def test_deformable_needs_a_gpu_without_validation(self):
        self.acquire("r0", 1)                                           # GPU 0
        self.phase("r0")                                                # validates on GPU 0
        self.assertEqual(self.acquire("d0", 4)[0], "1")                 # not GPU 0, where r0 validates
        self.phase("d0")
        gpu, why = self.acquire("d1", 4)                                # d0 validates on GPU 1: no GPU is empty
        self.assertIsNone(gpu)
        self.assertIn("none is empty", why)
        self.release("r0")
        self.assertEqual(self.acquire("d1", 4)[0], "0")

    def test_rigid_reasoning_beside_a_validating_deformable(self):
        self.assertEqual(self.acquire("d0", 4, gpus="0")[0], "0")
        self.assertIsNone(self.acquire("r0", 1, gpus="0")[0])           # a deformable trial reasons alone
        self.release("r0")
        self.phase("d0")
        self.assertEqual(self.acquire("r0", 1, gpus="0")[0], "0")       # it validates: rigid trials may reason there

    def test_validation_releases_cpus_and_memory(self):
        self.acquire("a", 1, cpus=4, mem=7, host_cpus=10)
        self.acquire("b", 1, cpus=4, mem=7, host_cpus=10)
        self.assertIsNone(self.acquire("c", 1, cpus=4, mem=7, host_cpus=10)[0])
        self.phase("a", cpus=2, mem=5)                                  # the verifier's 2 CPUs: 6 of 10 reserved
        self.assertEqual(self.acquire("c", 1, cpus=4, mem=7, host_cpus=10)[0], "0")
        self.assertEqual((self.state()["holds"]["a"]["cpus"], self.state()["holds"]["a"]["mem_gb"]), (2, 5))

    def test_validation_hold_belongs_to_its_process(self):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            self.acquire("a", 1)
            self.assertEqual(self.phase("a", pid=p.pid).returncode, 0)
            subprocess.run([sys.executable, str(TOOL), "release", "--dir", self.dir, "--runner-pid", str(ME)], check=True)
            self.assertIn("a", self.state()["holds"])                   # the lane's own release leaves it
        finally:
            p.kill()
            p.wait()
        self.acquire("b", 1)                                            # its process ended: dropped
        self.assertNotIn("a", self.state()["holds"])

    def test_phase_never_grows_and_needs_a_hold(self):
        self.acquire("a", 1, cpus=2, mem=3)
        self.assertEqual(self.phase("a", cpus=4, mem=8).returncode, 0)
        h = self.state()["holds"]["a"]
        self.assertEqual((h["cpus"], h["mem_gb"]), (2, 3))
        r = self.phase("nojob")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no hold", r.stderr)

    def test_ledger_without_phases_reads_as_reasoning(self):
        st = {"holds": {"old": {"pid": ME, "start": gpu_slots.proc_start(ME), "gpu": "0", "units": 1, "cpus": 1,
                                "mem_gb": 1}}, "waiting": {}, "draining": []}   # written before phases existed
        (pathlib.Path(self.dir) / "state.json").write_text(json.dumps(st))
        got = [self.acquire(f"r{i}", 1, gpus="0")[0] for i in range(4)]
        self.assertEqual(got, ["0", "0", "0", None])

    def test_classify_tasks(self):
        tasks = REPO / "tasks"
        first = lambda pat: next(iter(sorted(tasks.glob(pat))), None)  # noqa: E731
        rigid = first("roboarc-cast-coil-*-privileged")
        bolt = first("roboarc-bed-bolt-*-privileged")                       # a rigid family released after the classes
        wire = first("roboarc-wire-splicing-*-standard")                    # rope / chain
        cloth = first("roboarc-blanket-fold-*-privileged")
        ikea_p = first("roboarc-ikea-*-privileged")
        ikea_s = first("roboarc-ikea-*-standard")
        if not all((rigid, bolt, wire, cloth, ikea_p, ikea_s)):
            self.skipTest("the task families this test reads are not in tasks/")
        self.assertEqual(gpu_slots.classify(rigid, None, 4)[:3], (1, 4.0, 7.0))
        self.assertEqual(gpu_slots.classify(bolt, None, 4)[:3], (1, 4.0, 7.0))
        self.assertEqual(gpu_slots.classify(wire, None, 4)[:2], (4, 4.0))
        self.assertEqual(gpu_slots.classify(cloth, None, 4)[0], 4)
        self.assertEqual(gpu_slots.classify(ikea_p, None, 4)[0], 0)        # no GPU device in its compose files
        self.assertEqual(gpu_slots.classify(ikea_s, None, 4)[0], 1)        # the sim sidecar has one
        self.assertEqual(gpu_slots.classify(rigid, None, 4, 0.5, 0.5)[1:3], (2.0, 3.5))
        self.assertEqual(gpu_slots.classify(rigid, {"cast_coil"}, 4)[0], 4)  # --deformable overrides the classes
        self.assertEqual(gpu_slots.classify(cloth, set(), 4)[0], 1)
        # the validation phase: the verifier's declared CPUs and memory (scripts/task_resources.toml)
        std = first("roboarc-cast-coil-*-standard")
        self.assertEqual(gpu_slots.validate_needs(rigid), (2.0, 5.0))
        self.assertEqual(gpu_slots.validate_needs(std), (2.0, 6.0))
        self.assertEqual(gpu_slots.validate_needs(cloth), (2.0, 5.0))
        self.assertEqual(gpu_slots.validate_needs(ikea_p), (3.0, 3.0))
        self.assertEqual(gpu_slots.validate_needs(ikea_s), (2.0, 2.0))
        self.assertEqual(gpu_slots.validate_needs(rigid, 0.5, 2), (1.0, 10.0))
        r = subprocess.run([sys.executable, str(TOOL), "classify", str(std), "--phase", "validate"],
                           capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "2 6")

    def test_every_family_has_a_class(self):
        classes, required = gpu_slots.family_classes()
        own = {f for f, (_, o) in classes.items() if o}
        self.assertEqual(own, {"blanket_flatten", "blanket_fold", "fiber_bundling", "knot_untying", "orbits",
                               "rope_threading", "slotted_post_ring", "wire_splicing"})
        tasks = sorted((REPO / "tasks").glob("roboarc-*/task.toml"))
        if not tasks:
            self.skipTest("no tasks/")
        for t in tasks:
            if __import__("re").search(required, t.parent.name):
                with self.subTest(task=t.parent.name):
                    gpu_slots.classify(t.parent, None, 4)                  # raises for a family without a class

    def test_family_without_class(self):
        task = pathlib.Path(self.dir) / "roboarc-new-family-01-i01-privileged"
        (task / "environment").mkdir(parents=True)
        (task / "task.toml").write_text('[metadata]\nfamily = "new_family"\n\n[metadata.resources]\nmin_cpus = 2.0\n')
        with self.assertRaisesRegex(ValueError, "new_family.*no class"):
            gpu_slots.classify(task, None, 4)
        with self.assertRaisesRegex(ValueError, "no class"):
            gpu_slots.classify(task, {"new_family"}, 4)                    # an override does not give it resources
        r = subprocess.run([sys.executable, str(TOOL), "classify", str(task)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("no class", r.stderr)
        other = pathlib.Path(self.dir) / "roboarc-ikea-new-i00-privileged"  # classes_required does not find IKEA
        other.mkdir()
        (other / "task.toml").write_text('[metadata]\nfamily = "ikea_new"\n')
        self.assertEqual(gpu_slots.classify(other, None, 4)[0], 0)


if __name__ == "__main__":
    unittest.main()
