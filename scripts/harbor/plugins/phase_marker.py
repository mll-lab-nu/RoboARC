"""The moment a trial's agent phase ends, as a file the batch runner can watch: a Harbor job plugin.

    PYTHONPATH=scripts/harbor/plugins harbor run ... --plugin phase_marker:PhaseMarker --pk out=<file>

scripts/harbor/run_agent_batch.sh passes it to every `harbor run` (PHASE_SPLIT=1, docs/RUNNING.md "GPUs and lanes").
Harbor runs a trial's agent, then collects its artifacts (a standard task's sim sidecar replays and records the video
there), stops the agent's environment and runs the verifier; the trial's result.json is written only at the very end.
On Harbor's agent-end event (job.on_agent_ended) this plugin writes <file>, a JSON object: the job, the trial and task
names, when the agent phase started and ended, and when the marker was written. The runner then counts the trial as
validating (scripts/harbor/gpu_slots.py phase) and lets the lane start the next trial's agent phase.

It never stops a trial: an error while writing is printed and swallowed (the runner then waits for the trial to end, as
it does without the marker). It imports nothing from Harbor, so a Harbor release that renames a model class cannot break
the import; the event's fields are read with getattr.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def iso(t) -> str | None:
    return t.isoformat() if hasattr(t, "isoformat") else (str(t) if t is not None else None)


class PhaseMarker:
    def __init__(self, out: str, job: str | None = None, **_):
        self.out = Path(out)
        self.job = job

    async def on_job_start(self, job) -> None:
        if self.job is None:
            self.job = getattr(getattr(job, "config", None), "job_name", None)
        job.on_agent_ended(self.agent_ended)

    async def on_job_end(self, job_result) -> None:
        return None

    async def agent_ended(self, event) -> None:
        try:
            result = getattr(event, "result", None)
            ae = getattr(result, "agent_execution", None)
            rec = {"job": self.job, "trial": getattr(event, "trial_name", None), "task": getattr(event, "task_name", None),
                   "agent_started_at": iso(getattr(ae, "started_at", None)),
                   "agent_finished_at": iso(getattr(ae, "finished_at", None)), "marked_at": iso(getattr(event, "timestamp", None))}
            self.out.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.out.with_name(f"{self.out.name}.{os.getpid()}")
            tmp.write_text(json.dumps(rec) + "\n")
            tmp.replace(self.out)
        except Exception as exc:  # noqa: BLE001 - a marker must never fail the trial
            print(f"phase_marker: could not write {self.out}: {exc!r}", file=sys.stderr)
