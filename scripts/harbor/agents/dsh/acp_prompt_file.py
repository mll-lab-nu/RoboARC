#!/usr/bin/env python3
"""Start Harbor's ACP runner with the instruction read from a file, set the session's reasoning effort and check its model
(in the trial's container, with the runner's own Python; scripts/harbor/agents/dsh_acp.py uploads and starts it).

    acp_prompt_file.py --instruction-file F --runner R [--reasoning-effort E] -- <the runner's other arguments>

- The instruction stays off every command line: this process reads it, puts it into the runner's argv inside this
  process only (sys.argv) and runs the runner here (runpy). /proc/<pid>/cmdline keeps the file's path, so an agent that
  runs `pkill -f <a word of the task>` cannot match the harness (the exam's issue #228).
- The ACP connection Harbor's runner opens (acp.spawn_agent_process) is wrapped. With --reasoning-effort (the runner
  sets the model only): after session/new and after each model change the session's `reasoning_effort` option is set to
  E, when the session advertises that option with E among its values (dsh-acp: per model); a session without the
  option, or without E, is recorded and the run goes on. With HARBOR_ACP_REQUESTED_MODEL: the prompt goes out only when
  the session's `model` option is that value; otherwise the runner fails (the runner itself goes on with the agent's
  default when no option matches). Both go to <logs dir>/dsh-session.json.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

EFFORT_ID = "reasoning_effort"
MODEL_ID = "model"


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


class SessionLog:
    def __init__(self, path: Path | None, effort: str | None):
        self.path, self.data = path, {"requested": effort, "attempts": []}

    def write(self) -> None:
        if self.path is not None:
            self.path.write_text(json.dumps(self.data, indent=1) + "\n")

    def add(self, entry: dict) -> None:
        self.data["attempts"].append({"t": round(time.time(), 3), **entry})
        self.write()


class SessionConnection:
    """The runner's ACP connection, with the reasoning effort set after session/new and after each model change, and the
    prompt held back unless the session's model is the requested one."""

    def __init__(self, inner, effort: str | None, model: str | None, log: SessionLog):
        self._inner, self._effort, self._model, self._log = inner, effort, model, log
        self._session = None
        self._options = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _option(self, option_id: str):
        return next((o for o in self._options or [] if getattr(o, "id", None) == option_id), None)

    async def _apply(self, when: str) -> None:
        if not self._effort:
            return
        option = self._option(EFFORT_ID)
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

    async def prompt(self, *args, **kwargs):
        if self._model:
            current = getattr(self._option(MODEL_ID), "current_value", None)
            self._log.data["model"] = {"requested": self._model, "current": current}
            self._log.write()
            if current != self._model:
                raise RuntimeError(f"the session's model is {current!r}, not the requested {self._model!r}: no prompt sent")
        return await self._inner.prompt(*args, **kwargs)


def wrap_connection(effort: str | None, model: str | None, log: SessionLog) -> None:
    """Make acp.spawn_agent_process yield a SessionConnection (the runner imports it from acp when it starts)."""
    import acp

    original = acp.spawn_agent_process

    @contextlib.asynccontextmanager
    async def spawn_agent_process(*args, **kwargs):
        async with original(*args, **kwargs) as (conn, process):
            yield SessionConnection(conn, effort, model, log), process

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
    logs = next((r.split("=", 1)[1] for r in rest if r.startswith("--logs-dir=")), None)
    wrap_connection(a.reasoning_effort, os.environ.get("HARBOR_ACP_REQUESTED_MODEL"),
                    SessionLog(Path(logs) / "dsh-session.json" if logs else None, a.reasoning_effort))
    sys.argv = [a.runner, f"--instruction={instruction}", *rest]
    runpy.run_path(a.runner, run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
