"""Tests for scripts/harbor/plugins/phase_marker.py, the Harbor job plugin that marks the end of a trial's agent phase
(docs/RUNNING.md "GPUs and lanes"): no docker, no model.

    python3 scripts/harbor/tests/test_phase_marker.py     # stdlib only; the Harbor check needs Harbor importable

- on the agent-end event the plugin writes its marker file: the job, the trial and task names, the agent phase's start
  and end; an error while writing never reaches the trial;
- with Harbor importable (HARBOR_PYTHON, or the interpreter of `harbor` on PATH): Harbor's own plugin loader accepts the
  class with the runner's arguments, and the plugin registers on the job's agent-end event.
"""
import asyncio
import datetime as dt
import json
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest

HERE = pathlib.Path(__file__).resolve().parent
PLUGINS = HERE.parents[0] / "plugins"
sys.path.insert(0, str(PLUGINS))
sys.path.insert(0, str(HERE))
import phase_marker  # noqa: E402
from test_dsh_acp import harbor_python  # noqa: E402


class FakeJob:
    def __init__(self):
        self.hooks = []

    def on_agent_ended(self, cb):
        self.hooks.append(cb)
        return self


def event(trial="t__abc", task="roboarc-x-privileged"):
    t0 = dt.datetime(2026, 10, 29, 1, 0, tzinfo=dt.timezone.utc)
    ae = types.SimpleNamespace(started_at=t0, finished_at=t0 + dt.timedelta(minutes=30))
    return types.SimpleNamespace(trial_name=trial, task_name=task, timestamp=t0 + dt.timedelta(minutes=30, seconds=1),
                                 result=types.SimpleNamespace(agent_execution=ae))


class Marker(unittest.TestCase):
    def test_marker_on_agent_end(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / ".phase" / "b-privileged-x.json"
            p, job = phase_marker.PhaseMarker(out=str(out), job="b-privileged-x"), FakeJob()
            asyncio.run(p.on_job_start(job))
            self.assertEqual(len(job.hooks), 1)
            self.assertFalse(out.exists())                     # nothing before the agent ends
            asyncio.run(job.hooks[0](event()))
            rec = json.loads(out.read_text())
            self.assertEqual((rec["job"], rec["trial"], rec["task"]), ("b-privileged-x", "t__abc", "roboarc-x-privileged"))
            self.assertEqual(rec["agent_finished_at"], "2026-10-29T01:30:00+00:00")
            self.assertEqual(list(out.parent.iterdir()), [out])   # written whole: no temporary file left

    def test_write_error_stays_in_the_plugin(self):
        with tempfile.TemporaryDirectory() as d:
            blocker = pathlib.Path(d) / "file"
            blocker.write_text("")
            p, job = phase_marker.PhaseMarker(out=str(blocker / "m.json"), job="j"), FakeJob()   # a file as its directory
            asyncio.run(p.on_job_start(job))
            asyncio.run(job.hooks[0](event()))                 # no exception
            asyncio.run(job.hooks[0](types.SimpleNamespace()))    # an event without fields: no exception either


class HarborLoader(unittest.TestCase):
    def test_harbor_accepts_the_plugin(self):
        py = harbor_python()
        if not py or subprocess.run([py, "-c", "import harbor"], capture_output=True).returncode != 0:
            self.skipTest("Harbor is not importable (set HARBOR_PYTHON to its interpreter)")
        with tempfile.TemporaryDirectory() as d:
            code = textwrap.dedent(f"""
                import asyncio, sys
                sys.path.insert(0, {str(PLUGINS)!r})
                from harbor.cli.job_plugins import attach_job_plugin
                from harbor.job import Job
                assert hasattr(Job, "on_agent_ended")
                class J:
                    hooks = []
                    def on_agent_ended(self, cb):
                        self.hooks.append(cb); return self
                j = J()
                asyncio.run(attach_job_plugin(j, "phase_marker:PhaseMarker", kwargs={{"out": {d + '/m.json'!r}, "job": "j"}}))
                assert len(j.hooks) == 1
                print("ok")
            """)
            p = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(p.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
