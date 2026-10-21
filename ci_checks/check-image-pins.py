#!/usr/bin/env python3
"""Every image a task names is pinned by digest (images/README.md): each `FROM` of a Dockerfile and each compose
`image:` in a task dir is `<repo>[:<tag>]@sha256:<64 hex>`, and it is one of
  - RoboARC's own images, `ghcr.io/mll-lab-nu/...`, exactly as images/pins.json pins them (the digest of the
    archive images/push_all.sh pushes), or
  - the standard client base `docker.io/library/python...@sha256:...`, exactly scripts/eai_standard/client.env's
    CLIENT_IMAGE.
A bare tag, another digest, another registry: a failure. Repository-wide (no arguments): images/pins.json itself (each
reference pinned, an image id per entry, unique names and archives), the monitor's reference
(scripts/rcb_monitor/inject.py IMAGE) and the IKEA sim recipe's base (ikea/images/ikea-mujoco/sim/Dockerfile FROM) are
pins of images/pins.json. The sim recipe names its base as it was built, under the former registry
`ghcr.io/embodied-first-exam/` (an input of the sim image, so it keeps its text): the same name, tag and digest.

Usage: check-image-pins.py [task dirs...]   (default: every task under tasks/ plus the repository-wide checks)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = "ghcr.io/mll-lab-nu/"
FORMER_REGISTRY = "ghcr.io/embodied-first-exam/"   # the images' name up to 1.1.0 (same digests)
CLIENT_REPO = "docker.io/library/python"
DIGEST = re.compile(r"^(?P<name>[a-z0-9][^@\s]*)@sha256:[0-9a-f]{64}$")
PINS = json.loads((ROOT / "images/pins.json").read_text())["images"]
PINNED = {p["ref"] for p in PINS}
CLIENT = re.search(r"^CLIENT_IMAGE=(\S+)", (ROOT / "scripts/eai_standard/client.env").read_text(), re.M).group(1)


def refs(task: Path) -> list[tuple[str, str]]:
    """(file, reference) for every FROM of every Dockerfile and every compose `image:` in the task dir."""
    out = []
    for f in sorted(task.rglob("*")):
        if not f.is_file():
            continue
        if f.name == "Dockerfile" or f.name.startswith("Dockerfile."):
            text = re.sub(r"\\\n", " ", f.read_text())
            stages = []
            for m in re.finditer(r"^\s*FROM\s+(?:--\S+\s+)*(\S+)(?:\s+AS\s+(\S+))?", text, re.M | re.I):
                if m.group(1) not in stages:             # an earlier stage of the same file is not an image
                    out.append((str(f.relative_to(task)), m.group(1)))
                if m.group(2):
                    stages.append(m.group(2))
        elif re.fullmatch(r"(docker-)?compose(\.[\w-]+)?\.ya?ml", f.name):
            out += [(str(f.relative_to(task)), i.strip("'\""))
                    for i in re.findall(r"^\s*image:\s*(\S+)", f.read_text(), re.M)]
    return out


def problem(ref: str) -> str | None:
    if not DIGEST.match(ref):
        return "not pinned by digest (<repo>:<tag>@sha256:<digest>)"
    if ref.startswith(REGISTRY):
        return None if ref in PINNED else "not a pin of images/pins.json (another tag or digest)"
    if ref.startswith(CLIENT_REPO + ":") or ref.startswith(CLIENT_REPO + "@"):
        return None if ref == CLIENT else f"not the client base of scripts/eai_standard/client.env ({CLIENT})"
    return f"outside {REGISTRY} and the client base {CLIENT_REPO}"


def repository() -> list[str]:
    bad = []
    names = [p.get("name") for p in PINS]
    archives = [p.get("archive") for p in PINS]
    if len(set(names)) != len(names) or len(set(archives)) != len(archives):
        bad.append("images/pins.json: names and archives must be unique")
    for p in PINS:
        if not (DIGEST.match(p.get("ref", "")) and p["ref"].startswith(REGISTRY)):
            bad.append(f"images/pins.json: {p.get('ref')!r} is not {REGISTRY}<name>:<tag>@sha256:<digest>")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", p.get("id", "")):
            bad.append(f"images/pins.json: {p.get('ref')}: `id` must be the image id (sha256:<64 hex>)")
    sys.path.insert(0, str(ROOT / "scripts"))
    from rcb_monitor import inject                       # noqa: E402  the monitor sidecar every task gets
    if inject.IMAGE not in PINNED:
        bad.append(f"scripts/rcb_monitor/inject.py IMAGE {inject.IMAGE} is not a pin of images/pins.json")
    sim = ROOT / "ikea/images/ikea-mujoco/sim/Dockerfile"
    for base in re.findall(r"^FROM\s+(\S+)", sim.read_text(), re.M):
        if not base.startswith(FORMER_REGISTRY) or REGISTRY + base[len(FORMER_REGISTRY):] not in PINNED:
            bad.append(f"ikea/images/ikea-mujoco/sim/Dockerfile FROM {base} is not a pin of images/pins.json (under {FORMER_REGISTRY})")
    return bad


def main() -> int:
    args = sys.argv[1:]
    tasks = [Path(a).resolve() for a in args] or sorted(p for p in (ROOT / "tasks").iterdir() if (p / "task.toml").is_file())
    bad, n = [], 0
    for t in tasks:
        for rel, ref in refs(t):
            n += 1
            why = problem(ref)
            if why:
                bad.append(f"{t.name}/{rel}: {ref}: {why}")
    if not args:
        bad += repository()
    for b in bad:
        print("FAIL", b)
    if not bad:
        print(f"ok image pins: {n} image references in {len(tasks)} task dir(s), each pinned by digest"
              + ("" if args else f"; images/pins.json ({len(PINS)} images) agrees with the monitor and IKEA sim pins"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
