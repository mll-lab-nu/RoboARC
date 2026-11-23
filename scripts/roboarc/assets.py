#!/usr/bin/env python3
"""RoboARC's frozen MuJoCo instances: the Hugging Face dataset RoboARC/RoboARC-assets
(https://huggingface.co/datasets/RoboARC/RoboARC-assets), every file pinned by its sha256 in
scripts/roboarc/assets.json. They are kept out of git (65 208 files); the release image carries them
(images/roboarc/build.sh).

usage:
  python3 scripts/roboarc/assets.py fetch <dir>            download the pinned instances into <dir> and check them
  python3 scripts/roboarc/assets.py check <dir>            check a copy in the dataset's layout (<dir>/instances/<base>/)
  python3 scripts/roboarc/assets.py check-instances <dir>  check a build context: one directory per base
                                                           (<dir>/<base>/), exactly the pinned bases and files
  python3 scripts/roboarc/assets.py pin <dir> <rev>        maintainers: write assets.json from a local copy of revision
                                                           <rev> (its instances/)

assets.json: repo, repo_type, private, revision; files = {"instances/<base>/<path>": sha256} for every file of every
pinned instance; instances = {<base>: {"instance_sha256": the canonical sha256 of its instance.json}}. A base with
appearance seeds (docs/RUNNING.md "Seeds") also has each seed's instance under instances/<base>/seeds/<S>/ (its files in
`files`, like seed 0's) and instances.<base>.seeds = {"<S>": {"instance_sha256": ...}}; a base without seeds has no
`seeds` key, so the pins of earlier releases keep their bytes. A seed's goal pictures (a base whose instruction shows
some) are not dataset files: the task dirs commit them (mk_tasks.py) and the seed's `goal` = {<file name>: sha256}
pins them; `pin` keeps a seed's `goal` while the seed's instance is the same. Task
directories keep only runtime_binding.json, which pins the same hashes (scripts/roboarc/mk_tasks.py checks that they
agree); the image stages a task's instance with /opt/robotangle/stage_instance.py.

Source: assets.json's "repo" and "revision" name the dataset revision the pins were written from. PUBLISHED maps that
revision to the public dataset commit that holds exactly the pinned files, and fetch downloads from it;
ROBOARC_ASSETS_SOURCE=<repo>@<revision> names another copy instead (another name, or a later revision that holds every
pinned file). Whatever the source, each file must have its pinned sha256. The dataset is public: fetch needs no token,
and the token is never printed.

fetch: with git and git-lfs installed, one shallow git fetch of the revision (its text files in one pack) and
`git lfs pull` for the rest, in <dir> as a git work tree; otherwise, or when that fails, huggingface_hub's
snapshot_download, one request per file (pip install huggingface_hub). Everything else uses the standard library.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PIN = HERE / "assets.json"
PREFIX = "instances/"
SOURCE = "ROBOARC_ASSETS_SOURCE"
PUBLIC_REPO = "RoboARC/RoboARC-assets"
# assets.json "revision" -> (repo, revision) of the public dataset commit with exactly the files that pin lists
PUBLISHED = {"b2cf9b552f310ef4cfbd91df62ccfa837757c93a": (PUBLIC_REPO, "e65dcbf87eb4164cae2ea92becca0ed477f4c35c")}
HUB = "https://huggingface.co/datasets/"


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
        for rel, want in instance_pins(base, rec):
            p = d / rel
            if p.is_file() and canonical_sha(p) != want:
                bad.append(f"{rel}: canonical sha256 differs")
    return bad + schema(pin)


def instance_pins(base: str, rec: dict) -> list[tuple[str, str]]:
    """(path of each instance.json of a base, its pinned canonical sha256): seed 0, then every seed."""
    out = [(f"{PREFIX}{base}/instance.json", rec["instance_sha256"])]
    for s, srec in sorted((rec.get("seeds") or {}).items(), key=lambda kv: int(kv[0])):
        out.append((f"{PREFIX}{base}/seeds/{s}/instance.json", srec["instance_sha256"]))
    return out


def schema(pin: dict) -> list[str]:
    """The pin agrees with itself: every pinned instance (and seed) has its instance.json among the files, every
    seeds/<S>/ directory among the files is a pinned seed, seeds are positive integers."""
    bad = []
    for base, rec in sorted(pin["instances"].items()):
        for s in rec.get("seeds") or {}:
            if not str(s).isdigit() or int(s) < 1:
                bad.append(f"instances.{base}.seeds: {s!r} is not a positive integer")
        for rel, _ in instance_pins(base, rec):
            if rel not in pin["files"]:
                bad.append(f"{rel}: pinned instance without its file in `files`")
    for rel in pin["files"]:
        parts = rel.split("/")
        if len(parts) > 4 and parts[2] == "seeds" and parts[3] not in (pin["instances"].get(parts[1], {}).get("seeds") or {}):
            bad.append(f"{rel}: a file of seed {parts[3]}, which instances.{parts[1]}.seeds does not pin")
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
    for base, rec in sorted(pin["instances"].items()):
        for rel, want in instance_pins(base, rec):
            p = d / rel[len(PREFIX):]
            if p.is_file() and canonical_sha(p) != want:
                bad.append(f"{rel[len(PREFIX):]}: canonical sha256 differs")
    return bad + schema(pin)


def source(pin: dict) -> tuple[str, str]:
    """(repo, revision) fetch downloads from: those of ROBOARC_ASSETS_SOURCE=<repo>@<revision>, else the public copy of
    the pinned revision (PUBLISHED), else the pinned repo and revision."""
    s = os.environ.get(SOURCE, "")
    repo, at, rev = s.partition("@")
    if s and not (repo and at and rev):
        sys.exit(f"{SOURCE} must be <repo>@<revision>, not {s!r}")
    return (repo, rev) if s else PUBLISHED.get(pin["revision"], (pin["repo"], pin["revision"]))


def fetch_git(d: pathlib.Path, repo: str, rev: str) -> bool:
    """The revision's files with git and git-lfs: a shallow fetch of the commit, a forced checkout of it in d, then the
    LFS files under instances/. False when git or git-lfs is missing or a step fails (fetch then uses huggingface_hub);
    git never prompts for a password."""
    if not shutil.which("git") or subprocess.run(["git", "lfs", "version"], capture_output=True).returncode != 0:
        return False
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    g = ["git", "-C", str(d)]
    remote = ["remote", "set-url" if (d / ".git").is_dir() else "add", "origin", HUB + repo]
    steps = ([] if (d / ".git").is_dir() else [["git", "init", "-q", str(d)]]) + [
        g + remote, g + ["fetch", "-q", "--depth", "1", "origin", rev],
        g + ["checkout", "-q", "-f", "--detach", "FETCH_HEAD"], g + ["lfs", "pull", f"--include={PREFIX}**"]]
    for step in steps:
        if subprocess.run(step, env=env).returncode != 0:
            print(f"git could not fetch {repo}@{rev[:12]}: using huggingface_hub", file=sys.stderr)
            return False
    return True


def fetch(d: pathlib.Path, pin: dict) -> None:
    """The revision of source() into d: with git (fetch_git) when it can, else the pinned bases' directories
    (instances/<base>/**) with huggingface_hub, which reads a public dataset without a token (and sends one when it has
    one: `hf auth login`, HF_TOKEN)."""
    repo, rev = source(pin)
    d.mkdir(parents=True, exist_ok=True)
    if fetch_git(d, repo, rev):
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("fetch needs huggingface_hub (pip install huggingface_hub), or git with git-lfs")
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", revision=rev, local_dir=str(d),
                          allow_patterns=[f"{PREFIX}{b}/**" for b in sorted(pin["instances"])])
    except Exception as e:  # noqa: BLE001  the message names the repository, never the token
        sys.exit(f"download of {repo}@{rev[:12]} failed ({type(e).__name__}). The dataset is read without a token, and "
                 "a token the Hub rejects stops the download: unset HF_TOKEN or run `hf auth logout`. "
                 f"{SOURCE}=<repo>@<revision> downloads from another copy")


def pin_from(d: pathlib.Path, rev: str, old: dict | None = None) -> dict:
    """assets.json for the instances of a local copy of revision rev of the public dataset (d/instances/<base>/), in the
    committed layout."""
    files = {p.relative_to(d).as_posix(): file_sha(p) for p in sorted((d / "instances").rglob("*"))
             if p.is_file() and ".cache" not in p.parts}
    inst = {p.parent.name: {"instance_sha256": canonical_sha(p)} for p in sorted((d / "instances").glob("*/instance.json"))}
    if not inst:
        raise SystemExit(f"{d}/instances: no instance")
    old = old if old is not None else (json.loads(PIN.read_text()) if PIN.is_file() else {})
    for p in sorted((d / "instances").glob("*/seeds/*/instance.json"), key=lambda p: (p.parents[2].name, int(p.parent.name))):
        base, s = p.parents[2].name, p.parent.name
        if base not in inst or not s.isdigit() or int(s) < 1:
            raise SystemExit(f"{p}: a seed instance of a base without its own instance, or seed {s!r} not a positive integer")
        rec = {"instance_sha256": canonical_sha(p)}
        was = old.get("instances", {}).get(base, {}).get("seeds", {}).get(s, {})
        if was.get("goal") and was.get("instance_sha256") == rec["instance_sha256"]:   # its goal pictures, rendered from it
            rec["goal"] = was["goal"]
        inst[base].setdefault("seeds", {})[s] = rec
    return {"repo": PUBLIC_REPO, "repo_type": "dataset", "private": False, "revision": rev, "files": files,
            "instances": inst}


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
    repo, rev = source(pin) if argv[0] == "fetch" else (pin["repo"], pin["revision"])
    if argv[0] == "fetch":
        fetch(d, pin)
    bad = check_instances(d, pin) if argv[0] == "check-instances" else check(d, pin)
    for b in bad:
        print("FAIL", b)
    missing = sum(b.endswith(": missing") for b in bad)
    if argv[0] == "fetch" and missing:      # the download ended without an error, yet files are absent
        print(f"{missing} of the {len(pin['files'])} pinned files are not in {d}: {repo}@{rev[:12]} does not hold them, "
              f"or could not be read. images/roboarc/README.md (Build) says where every pinned file is ({SOURCE}, --instances)")
    if not bad:
        where = f" {repo}@{rev[:12]}" if argv[0] == "fetch" else ""
        print(f"ok{where}: {len(pin['files'])} files match scripts/roboarc/assets.json ({d})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
