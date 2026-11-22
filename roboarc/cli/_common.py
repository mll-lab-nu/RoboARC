"""What the subcommands share: where the checkout's scripts are, how they are started, and small argument helpers.
Standard library only (the host-side subcommands run on a host without numpy)."""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .. import _runtime

MODES = ("privileged", "standard")


class CliError(Exception):
    """A problem to report in one sentence: main() prints `roboarc: <message>` and exits with `code`."""

    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.code = code


class Formatter(argparse.RawDescriptionHelpFormatter):
    """Descriptions and examples as written; option help wrapped to the terminal, at most 120 columns."""

    def __init__(self, prog):
        super().__init__(prog, max_help_position=28, width=min(shutil.get_terminal_size((120, 24)).columns, 120))

    def _format_action(self, action):
        text = super()._format_action(action)
        if isinstance(action, argparse._SubParsersAction):     # the subcommands only, not a line repeating the metavar
            text = text.split("\n", 1)[1]
        return text


def repo() -> Path:
    """The RoboARC checkout the package was installed from (`pip install -e .`), or ROBOARC_REPO."""
    try:
        return _runtime.repo_root()
    except FileNotFoundError as e:
        raise CliError(str(e)) from None


def script(rel: str) -> Path:
    """A script of the checkout, by its repository path."""
    p = repo() / rel
    if not p.is_file():
        raise CliError(f"{rel} is not in this checkout ({repo()})")
    return p


def command(rel: str, *args) -> list[str]:
    """The argv that runs a script of the checkout: bash for a shell script, this interpreter for a Python one."""
    p = script(rel)
    return ([sys.executable] if p.suffix == ".py" else ["bash"]) + [str(p), *(str(a) for a in args)]


def child_env(changes: dict | None = None) -> dict:
    """The environment a script is started with: the caller's own, unchanged (without the numpy settings that importing
    roboarc added to this process), then `changes` applied (a value of None removes the name)."""
    env = {k: v for k, v in os.environ.items() if k not in _runtime.ENV_ADDED}
    for k, v in (changes or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    return env


def run(argv: list[str], changes: dict | None = None, **kw) -> int:
    """Run a command in the foreground, on this terminal, and return its exit status."""
    return subprocess.call(argv, env=child_env(changes), **kw)


def spawn(argv: list[str], changes: dict | None = None, **kw) -> subprocess.Popen:
    """Start a command and return at once (the caller waits for it)."""
    return subprocess.Popen(argv, env=child_env(changes), **kw)


def capture(argv: list[str], timeout: float = 60) -> tuple[int | None, str]:
    """(exit status, output) of a quick command: its stdout when it succeeds, stdout and stderr when it fails;
    (None, reason) when it is not installed or does not end."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=child_env())
    except FileNotFoundError:
        return None, f"{argv[0]} not found"
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"{argv[0]}: {type(e).__name__}"
    return p.returncode, (p.stdout if p.returncode == 0 else p.stdout + p.stderr).strip()


def csv(text: str | None) -> list[str]:
    """`a,b,c` (or `a b c`) as a list; empty for None."""
    return [x for x in (text or "").replace(",", " ").split() if x]


def quoted(argv: list[str], changes: dict | None = None) -> str:
    """One shell line: the variables a command is started with, then the command."""
    return " ".join([f"{k}={shlex.quote(str(v))}" for k, v in (changes or {}).items() if v is not None]
                    + [shlex.quote(str(a)) for a in argv])


def positive_int(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive integer")
    return n
