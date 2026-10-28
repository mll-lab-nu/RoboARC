#!/usr/bin/env python3
"""Start Harbor's ACP runner with the instruction read from a file, and set the session's reasoning effort (in the trial's
container, with the runner's own Python; scripts/harbor/agents/dsh_acp.py uploads and starts it).

    acp_prompt_file.py --instruction-file F --runner R [--reasoning-effort E] -- <the runner's other arguments>

- The instruction stays off every command line: this process reads it, puts it into the runner's argv inside this
  process only (sys.argv) and runs the runner here (runpy). /proc/<pid>/cmdline keeps the file's path, so an agent that
  runs `pkill -f <a word of the task>` cannot match the harness (the exam's issue #228).
- With --reasoning-effort: Harbor's runner sets the model only. The ACP connection it opens (acp.spawn_agent_process) is
  wrapped so that after session/new and after each model change the session's `reasoning_effort` option is set to E,
  when the session advertises that option with E among its values (dsh-acp: per model). Every attempt goes to
  <logs dir>/dsh-effort.json; a session without the option, or without E, is recorded and the run goes on.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import runpy
import sys
import time
from pathlib import Path

EFFORT_ID = "reasoning_effort"


def option_values(option) -> list[str]:
    """The selectable values of an ACP session config option (flat or grouped), as Harbor's runner reads them."""
    out: list[str] = []
    for item in getattr(option, "options", None) or []:
        value = getattr(item, "value", None)
        if value:
            out.append(value)
            continue
        out += [v for v in (getattr(g, "value", None) for g in getattr(item, "options", None) or []) if v]
    return out


class EffortLog:
    def __init__(self, path: Path | None, effort: str):
        self.path, self.effort, self.attempts = path, effort, []

    def add(self, entry: dict) -> None:
        self.attempts.append({"t": round(time.time(), 3), **entry})
        if self.path is not None:
            self.path.write_text(json.dumps({"requested": self.effort, "attempts": self.attempts}, indent=1) + "\n")


class EffortConnection:
    """The runner's ACP connection, with the reasoning effort set after session/new and after each model change."""

    def __init__(self, inner, effort: str, log: EffortLog):
        self._inner, self._effort, self._log = inner, effort, log
        self._session = None
        self._options = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def _apply(self, when: str) -> None:
        option = next((o for o in self._options or [] if getattr(o, "id", None) == EFFORT_ID), None)
        if option is None:
            self._log.add({"when": when, "skipped": f"the session advertises no {EFFORT_ID} option"})
            return
        values = option_values(option)
        if values and self._effort not in values:
            self._log.add({"when": when, "skipped": f"{self._effort} is not among {values}"})
            return
        try:
            response = await self._inner.set_config_option(session_id=self._session, config_id=EFFORT_ID, value=self._effort)
        except Exception as e:                    # recorded; the run goes on with the session's own effort
            self._log.add({"when": when, "error": f"{type(e).__name__}: {e}"})
            return
        self._options = getattr(response, "config_options", None) or self._options
        self._log.add({"when": when, "set": self._effort})

    async def new_session(self, *args, **kwargs):
        session = await self._inner.new_session(*args, **kwargs)
        self._session = getattr(session, "session_id", None)
        self._options = getattr(session, "config_options", None)
        await self._apply("session/new")
        return session

    async def set_config_option(self, *args, **kwargs):
        response = await self._inner.set_config_option(*args, **kwargs)
        self._options = getattr(response, "config_options", None) or self._options
        if kwargs.get("config_id") != EFFORT_ID:
            await self._apply(f"after setting {kwargs.get('config_id')}")
        return response

    async def set_session_model(self, *args, **kwargs):
        response = await self._inner.set_session_model(*args, **kwargs)
        await self._apply("after session/set_model")
        return response


def wrap_connection(effort: str, log: EffortLog) -> None:
    """Make acp.spawn_agent_process yield an EffortConnection (the runner imports it from acp when it starts)."""
    import acp

    original = acp.spawn_agent_process

    @contextlib.asynccontextmanager
    async def spawn_agent_process(*args, **kwargs):
        async with original(*args, **kwargs) as (conn, process):
            yield EffortConnection(conn, effort, log), process

    acp.spawn_agent_process = spawn_agent_process


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--instruction-file", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--reasoning-effort")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- and the runner's other arguments")
    a = ap.parse_args(argv)
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    instruction = Path(a.instruction_file).read_text(encoding="utf-8")
    if a.reasoning_effort:
        logs = next((r.split("=", 1)[1] for r in rest if r.startswith("--logs-dir=")), None)
        wrap_connection(a.reasoning_effort, EffortLog(Path(logs) / "dsh-effort.json" if logs else None, a.reasoning_effort))
    sys.argv = [a.runner, f"--instruction={instruction}", *rest]
    runpy.run_path(a.runner, run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
