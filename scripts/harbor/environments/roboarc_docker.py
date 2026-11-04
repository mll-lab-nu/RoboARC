"""Harbor's Docker environment with the egress sidecar's read timeout raised to the usage gateway's: a Harbor environment.

    PYTHONPATH=scripts/harbor/environments harbor run ... -e roboarc_docker:RoboARCDocker

scripts/harbor/run_agent_batch.sh passes it to every `harbor run` when the usage gateway is on (USAGE_GATEWAY,
docs/RUNNING.md "Usage recording"). Every RoboARC task declares an egress allowlist, so Harbor puts its egress sidecar
(gost, a transparent proxy) between the agent and the network. The gateway is plain HTTP, so the sidecar reads its
traffic, and gost waits at most `readTimeout` for the upstream's response headers: unset in Harbor's gost.yaml (0.23.0
to 0.25.x), it is 15 s. A model call whose provider took longer than that to answer was cut (the agent saw ECONNRESET,
the provider still finished and billed it): Jianwen's report of 2026-10-31, 171 of 172 resets 15.0-15.9 s after the
request started, with a Copilot proxy and a rate-limiting Azure endpoint. Response bodies were never cut.

This class builds the sidecar from a copy of the installed Harbor's sidecar folder whose gost.yaml sets the handler's
`readTimeout` to READ_TIMEOUT, the gateway's own upstream timeout (scripts/usage_gateway/gateway.py), so the sidecar
never gives up before the gateway does. Nothing else changes: the allowlist, the network policy and the rest of the
folder are the installed Harbor's, and Harbor's own install is never edited. The copy lives in the user's cache
(~/.cache/roboarc/egress-sidecar-<hash>/, named by the source folder and the patch), and Harbor names the sidecar image
by its build folder's contents, so the patched image is built once per Harbor version. A gost.yaml that already sets
`readTimeout` is used as it is; one whose handler block is not the expected one stops the import with an error instead
of running unpatched.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment

READ_TIMEOUT = "900s"
HANDLER = "    handler:\n      type: red\n      metadata:\n"   # the handler block of Harbor's gost.yaml


def patched_gost(text: str) -> str:
    """gost.yaml with the handler's readTimeout set (unchanged if it sets one already)."""
    if "readTimeout" in text:
        return text
    if text.count(HANDLER) != 1:
        raise RuntimeError("roboarc_docker: the egress sidecar's gost.yaml has no single `handler: type: red metadata:` "
                           "block; this Harbor's sidecar is not the one this class patches")
    return text.replace(HANDLER, HANDLER + f"        readTimeout: {READ_TIMEOUT}\n")


def patched_context(src: Path, cache: Path) -> Path:
    """A copy of the sidecar build folder `src` with the patched gost.yaml, under `cache` (made once, then reused)."""
    text = patched_gost((src / "gost.yaml").read_text())
    if text == (src / "gost.yaml").read_text():
        return src
    h = hashlib.sha256(text.encode())
    for p in sorted(q for q in src.rglob("*") if q.is_file()):
        h.update(str(p.relative_to(src)).encode() + b"\0" + p.read_bytes())
    dst = cache / f"egress-sidecar-{h.hexdigest()[:16]}"
    if not dst.is_dir():
        cache.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".egress-sidecar-", dir=cache))
        shutil.copytree(src, tmp / "c", symlinks=True)
        (tmp / "c" / "gost.yaml").write_text(text)
        try:
            os.replace(tmp / "c", dst)      # another process may have made it meanwhile: theirs is the same
        except OSError:
            pass
        shutil.rmtree(tmp, ignore_errors=True)
    return dst


_CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "roboarc"


class RoboARCDocker(DockerEnvironment):
    _EGRESS_CONTROL_SIDECAR_CONTEXT_PATH = patched_context(DockerEnvironment._EGRESS_CONTROL_SIDECAR_CONTEXT_PATH, _CACHE)
