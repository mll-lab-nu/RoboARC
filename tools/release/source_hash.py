#!/usr/bin/env python3
"""SOURCE.json from a checkout, without Docker: the simulator-source hash that the release image installs
(/opt/topogym/SOURCE.json) and every task's runtime_binding.json names (scripts/roboarc/release.json runtime.source).

    python3 tools/release/source_hash.py           # print it
    python3 tools/release/source_hash.py --check   # and exit 1 unless it equals release.json's runtime.source

How: the git-tracked files of source/topogym/, filtered as source/.dockerignore filters the build context (no
agents/, planning/, data/, deferred/legacy/redundants/rollout_simulation dirs, no *.md, no __pycache__ / *.pyc, no
.env* / .secrets), are copied to a temporary directory and the image's own build step, source/images/topogym/prune.py,
runs on it. Standard library only. A change to any of the hashed files (443 in release 1.0.0) changes the hash, and
with it every task's binding: that needs a new image version (docs/rcb/README.md, source/README.md)."""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXCLUDED_TOP = {"agents", "planning", "data"}                                   # source/.dockerignore: topogym/<dir>
EXCLUDED_ANY = {"deferred", "legacy", "redundants", "rollout_simulation"}       # topogym/**/<dir>


def excluded(rel: str) -> bool:
    parts = rel.split("/")                                                      # topogym/...
    return (parts[1] in EXCLUDED_TOP or any(p in EXCLUDED_ANY for p in parts[1:-1]) or "__pycache__" in parts
            or rel.endswith((".md", ".pyc")) or parts[-1].startswith(".env") or ".secrets" in parts)


def source_json() -> dict:
    files = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z", "source/topogym"]).decode().split("\0")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for f in filter(None, files):
            rel = f[len("source/"):]
            if not excluded(rel):
                (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / f, tmp / rel)
        subprocess.run([sys.executable, str(ROOT / "source/images/topogym/prune.py"), str(tmp)], check=True,
                       stdout=subprocess.DEVNULL)
        return json.loads((tmp / "SOURCE.json").read_text())


def main() -> int:
    got = source_json()
    print(json.dumps(got))
    if "--check" in sys.argv[1:]:
        want = json.loads((ROOT / "scripts/roboarc/release.json").read_text())["runtime"]["source"]
        if got != want:
            print(f"SOURCE.json differs from scripts/roboarc/release.json runtime.source {want}", file=sys.stderr)
            return 1
        print("ok: equals scripts/roboarc/release.json runtime.source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
