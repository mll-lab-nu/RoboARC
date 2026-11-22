#!/usr/bin/env python3
"""RoboARC's pinned images (images/pins.json, images/README.md): list them, check the local Docker images or a
directory of OCI archives against the pins, pull the images that are missing. images/pull_all.sh is the entry point.

usage:
  python3 images/pins.py list     [--pins F]                          "<ref> <id>" per image, in pull order
  python3 images/pins.py check    [--pins F] [--json]                 the local images against the pins; pulls nothing
  python3 images/pins.py pull     [--pins F] [--accept-licences]      pull what is missing, by digest, then check
  python3 images/pins.py archives --archives DIR [--pins F] [--json]  OCI archives against the pins; no Docker
  python3 images/pins.py licences WHAT [--accept-licences]            exit 0 when the licences are accepted

A pin has the reference the tasks name, <repo>:<tag>@sha256:<manifest digest>, and the image id (the config digest).
An image is local with its pin when Docker resolves the pinned reference and reports as its id
  - the pinned id: the classic image store (storage driver overlay2), whose image id is the config digest, or
  - the pinned manifest digest: the containerd image store (what a new install of Docker Engine 29 uses), whose image
    id is the digest of the image's manifest. The manifest names its config by digest, so the pinned id follows.
An image that is local under its tag only (loaded from an archive, or built) is reported apart: the tasks name the
digest, and the classic store resolves a digest only for an image pulled from a registry.

Pulling or building the images needs your acceptance of the third-party licences in LICENSES/THIRD_PARTY.md:
--accept-licences, or ROBOARC_ACCEPT_LICENCES=1 in the environment. Listing and checking need none, and `pull` asks
for it only when an image has to be pulled. `licences` is that rule for the build scripts.

Exit: 0 every image (archive) is its pin; 1 some are missing or differ, or a pull failed; 2 usage, unreadable pins,
Docker not usable; 3 the licences are not accepted. Standard library only, Python >= 3.8.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = re.compile(r"^[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}$")
ID = re.compile(r"^sha256:[0-9a-f]{64}$")
ACCEPT = "ROBOARC_ACCEPT_LICENCES"
# what to do about a failed pull, by the kind of docker's error (classify)
HINTS = {
    "denied": "{host} refused the pull. The packages are public and need no login, but an old login that Docker still "
              "holds for {host} is refused in place of an anonymous pull: `docker logout {host}`, then run this again. "
              "If the packages are not public yet, log in once with a GitHub token that has the read:packages scope: "
              "`echo <token> | docker login {host} -u <github user> --password-stdin`.",
    "denied-other": "{host} refused the pull: log in to it (`docker login {host}`), or pull without an old login "
                    "(`docker logout {host}`), then run this again.",
    "network": "{host} could not be reached: check the network (and the proxy the Docker daemon uses), then run this "
               "again. Layers that were downloaded are kept.",
    "disk": "Docker's storage is full: the four images take about 19 GB (`docker system df` shows what is there).",
    "daemon": "the Docker daemon cannot be reached: is it running, and is your user in the `docker` group?",
}


def say(*a) -> None:
    print(*a, flush=True)


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def die(msg: str) -> None:
    eprint(msg)
    raise SystemExit(2)


def load_pins(path) -> list[dict]:
    """The pins in the file's order, each with repo, tag and digest split out of its reference. A malformed entry
    stops everything (exit 2) before any image is touched."""
    try:
        images = json.loads(pathlib.Path(path).read_text())["images"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        die(f"pull_all: cannot read the pins in {path}: {e}")
    if not isinstance(images, list) or not images:
        die(f"pull_all: {path} pins no image")
    pins = []
    for p in images:
        p = p if isinstance(p, dict) else {}
        ref, iid = p.get("ref", ""), p.get("id", "")
        if not isinstance(ref, str) or not isinstance(iid, str) or not REF.match(ref) or not ID.match(iid):
            die(f"pull_all: {p.get('name', '?')}: not a pinned <repo>:<tag>@sha256 reference with an image id")
        name, digest = ref.split("@")
        repo, tag = name.rsplit(":", 1)
        pins.append({"name": p.get("name", repo.rsplit("/", 1)[-1]), "ref": ref, "id": iid, "repo": repo, "tag": tag,
                     "digest": digest, "archive": p.get("archive", "")})
    return pins


# ---- the local Docker images

class DockerError(Exception):
    pass


def docker(*args: str, **kw) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["docker", *args], universal_newlines=True, **kw)
    except FileNotFoundError:
        raise DockerError("docker is not on PATH") from None


def image_id(ref: str) -> str | None:
    """The image id Docker reports for a reference; None when it has no such image."""
    p = docker("image", "inspect", "--format", "{{.Id}}", ref, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode == 0:
        return p.stdout.strip()
    if "no such image" in p.stderr.lower():
        return None
    raise DockerError(p.stderr.strip() or f"docker image inspect exited {p.returncode}")


def judge(pin: dict, by_ref: str | None, by_tag: str | None) -> tuple[str, str | None]:
    """One pin against the ids Docker reports for the pinned reference and for <repo>:<tag> (None: no such image).
    Returns (status, the id seen): ok (the pinned reference resolves, to the pinned id or the pinned manifest digest),
    differs (it resolves to something else), tag-only (the pinned image is local, but only under its tag), missing."""
    same = (pin["id"], pin["digest"])
    if by_ref is not None:
        return ("ok" if by_ref in same else "differs"), by_ref
    return ("tag-only" if by_tag in same else "missing"), by_tag


def look(pin: dict) -> tuple[str, str | None]:
    by_ref = image_id(pin["ref"])
    return judge(pin, by_ref, None if by_ref is not None else image_id(f"{pin['repo']}:{pin['tag']}"))


def line(pin: dict, status: str, got: str | None) -> str:
    ref, tag = pin["ref"], f"{pin['repo']}:{pin['tag']}"
    if status == "ok":      # the id as `docker images` shows it
        how = "" if got == pin["id"] else (" (the manifest digest, the image id of Docker's containerd image store; "
                                           f"pinned id {pin['id'][7:19]})")
        return f"ok   {ref} = {got[7:19]}{how}"
    if status == "differs":
        return f"FAIL {ref}: image id {got}, pinned {pin['id']}"
    if status == "tag-only":
        return (f"FAIL {ref}: local as {tag} only, not under the digest the tasks name: pull it, or serve its archive "
                "from a local registry (images/README.md)")
    return f"FAIL {ref}: not local" + (f" ({tag} is another image, {got})" if got else "")


def store() -> tuple[str, str]:
    """(the Docker daemon's image store: containerd, classic or unknown; its storage driver)."""
    p = docker("info", "--format", "{{.Driver}} {{.DriverStatus}}", stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    driver = p.stdout.split(" ", 1)[0].strip()
    if p.returncode != 0 or not driver:
        return "unknown", ""
    return ("containerd" if "io.containerd.snapshotter" in p.stdout else "classic"), driver


def classify(err: str) -> str:
    """The kind of a failed `docker pull`, from docker's error text: daemon, disk, network, denied or other."""
    e = err.lower()
    if "cannot connect to the docker daemon" in e or "docker.sock" in e or "docker daemon socket" in e:
        return "daemon"
    if "no space left on device" in e:
        return "disk"
    if re.search(r"no such host|i/o timeout|connection re(fused|set)|tls handshake|network is unreachable|"
                 r"no route to host|temporary failure in name resolution|proxyconnect|deadline exceeded|"
                 r"request canceled", e):
        return "network"
    if re.search(r"unauthorized|denied|authentication required|forbidden|\b40[13]\b|no basic auth credentials", e):
        return "denied"
    return "other"


def report(bad: bool) -> int:
    if bad:
        eprint("pull_all: some images are missing or differ (above)")
    else:
        say("pull_all: every pinned image is local with its pinned id")
    return 1 if bad else 0


def show(pin: dict, status: str, got: str | None) -> bool:
    """Print an image's line (a failure on stderr); True when it is not its pin."""
    (say if status == "ok" else eprint)(line(pin, status, got))
    return status != "ok"


def cmd_check(a) -> int:
    rows = [(p, *look(p)) for p in load_pins(a.pins)]
    kind, driver = store()
    if a.json:
        ok = all(s == "ok" for _, s, _ in rows)
        print(json.dumps({"ok": ok, "store": kind, "driver": driver,
                          "images": [{"name": p["name"], "ref": p["ref"], "id": p["id"], "status": s, "local_id": g}
                                     for p, s, g in rows]}, indent=1))
        return 0 if ok else 1
    bad = [show(*r) for r in rows]
    say(f"Docker image store: {kind}" + (f" ({driver})" if driver else ""))
    return report(any(bad))


def cmd_pull(a) -> int:
    rows = [(p, *look(p)) for p in load_pins(a.pins)]
    # a reference that resolves to another image is not mended by a pull of the same digest: only what is absent
    if any(s in ("missing", "tag-only") for _, s, _ in rows) and not accepted(a):
        return refuse("pull_all: pulling the images")
    progress = sys.stdout.isatty()      # docker draws its progress on a terminal; in a log, one line per image
    bad, kinds = False, {}
    for p, s, got in rows:
        if s in ("missing", "tag-only"):
            say(f"pull {p['ref']}")
            r = docker("pull", *([] if progress else ["-q"]), p["ref"],
                       stdout=None if progress else subprocess.DEVNULL, stderr=subprocess.PIPE)
            if r.returncode != 0:
                said = [x for x in r.stderr.strip().splitlines() if x.strip()] or ["no message"]
                eprint(f"FAIL {p['ref']}: docker pull failed: {said[0]}")
                kinds.setdefault(classify(r.stderr), p["ref"].split("/", 1)[0])
                bad = True
                continue
            s, got = look(p)
        bad |= show(p, s, got)
    for kind, host in kinds.items():
        if kind == "denied" and host != "ghcr.io":
            kind = "denied-other"
        if kind in HINTS:
            eprint("pull_all: " + HINTS[kind].format(host=host))
    return report(bad)


def cmd_list(a) -> int:
    for p in load_pins(a.pins):
        print(p["ref"], p["id"])
    return 0


# ---- the licences

def accepted(a) -> bool:
    return bool(getattr(a, "accept_licences", False)) or os.environ.get(ACCEPT) == "1"


def refuse(what: str) -> int:
    eprint(f"{what} needs your acceptance of the third-party licences in LICENSES/THIRD_PARTY.md: read it, then run "
           f"this again with --accept-licences (or {ACCEPT}=1).")
    return 3


def cmd_licences(a) -> int:
    return 0 if accepted(a) else refuse(a.what)


# ---- OCI archives

def check_archive(path, pin: dict) -> list[str]:
    """What is wrong with an OCI archive as the pinned image (nothing: it is that image). One pass over the archive:
    index.json names exactly one manifest, its digest is the pinned one, its config is the pinned image id, and every
    blob it names is there under the sha256 of its bytes."""
    blobs, index = {}, None             # blob file name -> (sha256 of its bytes, the bytes of a small one)
    try:
        with tarfile.open(path, "r|") as t:
            for m in t:
                name = m.name[2:] if m.name.startswith("./") else m.name
                if not m.isfile() or not (name == "index.json" or name.startswith("blobs/sha256/")):
                    continue
                f, h, small = t.extractfile(m), hashlib.sha256(), b""
                while True:
                    chunk = f.read(1 << 22)
                    if not chunk:
                        break
                    h.update(chunk)
                    if m.size <= 1 << 20:
                        small += chunk
                if name == "index.json":
                    index = small
                else:
                    blobs[name.rsplit("/", 1)[1]] = (h.hexdigest(), small)
    except (OSError, tarfile.TarError) as e:
        return [f"cannot be read ({type(e).__name__}: {e})"]
    try:
        (man,) = json.loads(index or b"")["manifests"]
        manifest = json.loads(blobs[man["digest"][7:]][1])
        config = manifest["config"]["digest"]
        need = [config] + [layer["digest"] for layer in manifest["layers"]]
    except (KeyError, ValueError, TypeError) as e:
        return [f"not an OCI archive of one image ({type(e).__name__}: {e})"]
    bad = []
    if man["digest"] != pin["digest"]:
        bad.append(f"manifest {man['digest']} is not the pinned {pin['digest']}")
    if config != pin["id"]:
        bad.append(f"config {config} is not the pinned image id {pin['id']}")
    wrong = sorted(n for n, (h, _) in blobs.items() if h != n)
    missing = [d for d in need if d[7:] not in blobs]
    if wrong:
        bad.append("blobs whose sha256 is not their name: " + ", ".join(wrong))
    if missing:
        bad.append("blobs the manifest names are missing: " + ", ".join(missing))
    return bad


def cmd_archives(a) -> int:
    d, rows = pathlib.Path(a.archives), []
    for p in load_pins(a.pins):
        path = d / p["archive"] if p["archive"] else None
        if path is None:
            problems = ["the pin names no archive"]
        elif not path.is_file():
            problems = ["missing"]
        else:
            problems = check_archive(path, p)
        rows.append((p, problems))
        if a.json:
            continue
        what = p["archive"] or p["name"]
        if problems:
            eprint(f"FAIL {what}: " + "; ".join(problems))
        else:
            say(f"ok   {what} = {p['ref']}")
    bad = any(problems for _, problems in rows)
    if a.json:
        print(json.dumps({"ok": not bad, "archives": [{"name": p["name"], "archive": p["archive"], "ref": p["ref"],
                                                        "status": "fail" if problems else "ok", "problems": problems}
                                                       for p, problems in rows]}, indent=1))
    elif bad:
        eprint(f"archives: some archives in {d} are missing or differ from their pins (above)")
    else:
        say("archives: every archive is its pinned image")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="images/pins.py", description="RoboARC's pinned images (images/README.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("list", cmd_list), ("check", cmd_check), ("pull", cmd_pull), ("archives", cmd_archives),
                     ("licences", cmd_licences)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        if name != "licences":
            s.add_argument("--pins", default=str(ROOT / "images/pins.json"))
        if name in ("check", "archives"):
            s.add_argument("--json", action="store_true")
        if name in ("pull", "licences"):
            s.add_argument("--accept-licences", "--accept-licenses", action="store_true")
        if name == "archives":
            s.add_argument("--archives", required=True, metavar="DIR")
        if name == "licences":
            s.add_argument("what", help="what needs the acceptance, e.g. 'images/roboarc/build.sh: building the image'")
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except DockerError as e:
        eprint(f"pull_all: Docker cannot be used: {e}")
        if classify(str(e)) == "daemon":
            eprint("pull_all: " + HINTS["daemon"])
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
