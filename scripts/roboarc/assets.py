#!/usr/bin/env python3
"""RoboARC's frozen MuJoCo instances: the private Hugging Face dataset eai-v2-internal/robotangle-assets, pinned by
commit and by the sha256 of every file in scripts/roboarc/assets.json. They are kept out of git (21 477 files); the
release image carries them (images/roboarc/build.sh).

usage:
  python3 scripts/roboarc/assets.py fetch <dir>            download the pinned instances into <dir> and check them
  python3 scripts/roboarc/assets.py check <dir>            check a copy in the dataset's layout (<dir>/instances/<base>/)
  python3 scripts/roboarc/assets.py check-instances <dir>  check a build context: one directory per base
                                                           (<dir>/<base>/), exactly the pinned bases and files
  python3 scripts/roboarc/assets.py pin <dir> <rev>        maintainers: write assets.json from a local copy of revision
                                                           <rev> (its instances/)

assets.json: repo, repo_type, private, revision; files = {"instances/<base>/<path>": sha256} for every file of every
pinned instance; instances = {<base>: {"instance_sha256": the canonical sha256 of its instance.json}}. Task
directories keep only runtime_binding.json, which pins the same hashes (scripts/roboarc/mk_tasks.py checks that they
agree); the image stages a task's instance with /opt/robotangle/stage_instance.py.

The dataset at that revision also holds the Genesis base image's build inputs (image/); they are not pinned here and
never fetched (the release image builds FROM the published base image, images/roboarc/README.md).

Access: a member of eai-v2-internal, logged in with `hf auth login` or with HF_TOKEN in the environment. The token is
never printed. fetch needs huggingface_hub (pip install huggingface_hub); everything else uses the standard library.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PIN = HERE / "assets.json"
PREFIX = "instances/"


def file_sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha(p: pathlib.Path) -> str:
    """topogym.simulator.frozen_binding.canonical_hash of a JSON file (what runtime_binding.json names)."""
    v = json.loads(p.read_text())
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check(d: pathlib.Path, pin: dict) -> list[str]:
    """A copy in the dataset's layout: every pinned file present with its sha256, every instance.json canonical."""
    bad = []
    for rel, want in sorted(pin["files"].items()):
        p = d / rel
        if not p.is_file():
            bad.append(f"{rel}: missing")
        elif file_sha(p) != want:
            bad.append(f"{rel}: sha256 differs from scripts/roboarc/assets.json")
    for base, rec in sorted(pin["instances"].items()):
        p = d / "instances" / base / "instance.json"
        if p.is_file() and canonical_sha(p) != rec["instance_sha256"]:
            bad.append(f"instances/{base}/instance.json: canonical sha256 differs")
    return bad


def check_instances(d: pathlib.Path, pin: dict) -> list[str]:
    """An image build context (one directory per base): the pinned bases with exactly their pinned files. The image
    copies every directory and file it finds, so an extra one is an error too."""
    want = {rel[len(PREFIX):]: sha for rel, sha in pin["files"].items()}
    have = {p.relative_to(d).as_posix(): p for p in sorted(d.rglob("*")) if p.is_file()}
    bad = [f"{rel}: not pinned in scripts/roboarc/assets.json" for rel in sorted(set(have) - set(want))]
    bad += [f"{rel}: missing" for rel in sorted(set(want) - set(have))]
    bad += [f"{rel}: sha256 differs from scripts/roboarc/assets.json" for rel in sorted(set(want) & set(have))
            if file_sha(have[rel]) != want[rel]]
    bad += [f"{b}: a directory without pinned files" for b in sorted(p.name for p in d.iterdir() if p.is_dir())
            if b not in pin["instances"]]
    return bad


def fetch(d: pathlib.Path, pin: dict) -> None:
    """The pinned bases' directories only (instances/<base>/**), at the pinned revision."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("fetch needs huggingface_hub: pip install huggingface_hub (then `hf auth login`)")
    d.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id=pin["repo"], repo_type="dataset", revision=pin["revision"], local_dir=str(d),
                          allow_patterns=[f"{PREFIX}{b}/**" for b in sorted(pin["instances"])])
    except Exception as e:  # noqa: BLE001  the message names the repository, never the token
        sys.exit(f"download of {pin['repo']}@{pin['revision'][:12]} failed ({type(e).__name__}): this needs read access "
                 "to the private dataset (a member of eai-v2-internal; `hf auth login` or HF_TOKEN)")


def pin_from(d: pathlib.Path, rev: str, old: dict | None = None) -> dict:
    """assets.json for the instances of a local copy of revision rev (d/instances/<base>/), in the committed layout."""
    files = {p.relative_to(d).as_posix(): file_sha(p) for p in sorted((d / "instances").rglob("*"))
             if p.is_file() and ".cache" not in p.parts}
    inst = {p.parent.name: {"instance_sha256": canonical_sha(p)} for p in sorted((d / "instances").glob("*/instance.json"))}
    if not inst:
        raise SystemExit(f"{d}/instances: no instance")
    old = old if old is not None else (json.loads(PIN.read_text()) if PIN.is_file() else {})
    return {"repo": old.get("repo", "eai-v2-internal/robotangle-assets"), "repo_type": "dataset", "private": True,
            "revision": rev, "files": files, "instances": inst}


def dumps(pin: dict) -> str:
    return json.dumps(pin, indent=1) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("fetch", "check", "check-instances", "pin") or (argv[0] == "pin") != (len(argv) == 3):
        print(__doc__, file=sys.stderr)
        return 2
    d = pathlib.Path(argv[1])
    if argv[0] == "pin":
        PIN.write_text(dumps(pin_from(d, argv[2])))
        print(f"wrote {PIN}")
        return 0
    pin = json.loads(PIN.read_text())
    if argv[0] == "fetch":
        fetch(d, pin)
    bad = check_instances(d, pin) if argv[0] == "check-instances" else check(d, pin)
    for b in bad:
        print("FAIL", b)
    if not bad:
        print(f"ok {pin['repo']}@{pin['revision'][:12]}: {len(pin['files'])} files match scripts/roboarc/assets.json ({d})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
