"""The consent step before the images come onto a machine. The images contain third-party software and assets under
their own licences (LICENSES/THIRD_PARTY.md); a user accepts them once, with --accept-licences or by answering the
question on a terminal, and the answer is kept in a small file under the user's configuration directory. Standard
library only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from . import _common as C

LICENCES = "LICENSES/THIRD_PARTY.md"            # the file a user accepts, by its repository path
FLAG = "--accept-licences"
VARIABLE = "ROBOARC_ACCEPT_LICENCES"            # =1: accepted. The scripts under images/ read it; so does `require`


def consent_file() -> Path:
    """$XDG_CONFIG_HOME/roboarc/licences-accepted.json (default ~/.config/roboarc/licences-accepted.json)."""
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "roboarc" / "licences-accepted.json"


def accepted() -> bool:
    try:
        return json.loads(consent_file().read_text()).get("accepted") is True
    except (OSError, ValueError, AttributeError):
        return False


def record(how: str) -> None:
    """Keep the acceptance: when, how, and the sha256 of the licences file that was accepted."""
    import roboarc
    path = consent_file()
    text = C.script(LICENCES).read_bytes()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"accepted": True, "licences": LICENCES, "sha256": hashlib.sha256(text).hexdigest(),
                                    "roboarc": roboarc.__version__, "how": how,
                                    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
        print(f"third-party licences accepted ({LICENCES}); recorded in {path}", file=sys.stderr)
    except OSError as e:         # e.g. a read-only home directory: the acceptance holds for this run
        print(f"third-party licences accepted ({LICENCES}); not recorded ({path}: {e})", file=sys.stderr)


def require(flag: bool, what: str) -> dict:
    """Return when the user has accepted the third-party licences (earlier, with the flag or the variable, or now at
    the prompt); otherwise raise the CliError that stops `what` (e.g. "images pull") before it touches an image.
    Returns the variable that tells the image scripts so, for the environment they are started with."""
    told = {VARIABLE: "1"}
    if accepted():
        return told
    if flag or os.environ.get(VARIABLE) == "1":
        record(FLAG if flag else f"{VARIABLE}=1")
        return told
    if not sys.stdin.isatty():
        raise C.CliError(f"{what}: the third-party licences in {LICENCES} have not been accepted on this machine yet: "
                         f"read that file, then run the command again with {FLAG}")
    print(f"The RoboARC images contain third-party software and assets under their own licences. They are listed in\n"
          f"  {C.repo() / LICENCES}\nRead that file before you answer.", file=sys.stderr)
    try:
        answer = input("Do you accept these licences? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        raise C.CliError(f"{what}: the third-party licences were not accepted, so nothing was done", code=1)
    record("prompt")
    return told
