#!/usr/bin/env python3
"""Frozen bytes: no path that ci_checks/frozen-bytes.json lists differs from the baseline recorded there.

Between releases the task directories, the files the generators copy into them or render them from, and the inputs of
the released images do not change, comments and docstrings included: one changed character re-renders tasks, gives them
new DIGESTs and splits results that were graded on one release (docs/DEVELOPING.md). frozen-bytes.json names those
paths, each with the reason it is frozen and the hash it had at the baseline. The hashes are in that file, so the check
needs no git history: it gives the same verdict on a shallow clone and on a repository squashed to one commit.

    python3 ci_checks/check-frozen-bytes.py                # exit 1 when a frozen path differs (the CI job frozen-bytes)
    python3 ci_checks/check-frozen-bytes.py --root <dir>   # the same on another checkout, with this checkout's list:
                                                           # a branch's worktree before it is merged
    python3 ci_checks/check-frozen-bytes.py --write        # record the checkout as the new baseline (see below)
    python3 ci_checks/check-frozen-bytes.py --selftest     # the rules fire, on a small temporary repository

An entry of `frozen` is a `path`, a `reason` (one line: why a maintainer must not edit it) and one of
  bytes   (the default) a file, or a directory (the path ends in `/`): `sha256` over "<path>\\0<mode>\\0<sha256 of the
          file>\\n" of its `files`, sorted by path. The mode is 755 or 644, the executable bit as git records it. The
          files are the checkout's, tracked or untracked and not ignored (git ls-files): a file added to a frozen
          directory counts, a __pycache__ does not. `only` / `except` (patterns on the path below the directory, `*`
          crosses `/`) narrow a directory.
  value   `value` names a module-level constant of the Python file `path` that is assigned a literal: `sha256` of the
          value as JSON. For a file whose code and docstring may change while one list in it is written into every task.
  source  `how` is "tools/release/source_hash.py --check": that script runs. It must pass (SOURCE.json of the checkout
          equals scripts/roboarc/release.json runtime.source) and print the recorded SOURCE.json (`files`, `sha256`).
`generators` lists the code that renders the tasks. It is not hashed (it may change while its output does not): its
output, tasks/, is frozen above, and the CI jobs `generator` and `ikea` render tasks/ again and compare. The three
together are the gate: this check, `python3 scripts/roboarc/mk_tasks.py --check` and `bash ikea/ci/check.sh`.

A failure names the path and its reason. Where the clone holds the baseline commit, it also names the files that differ
from it (the verdict never depends on that). Undo the change. Only a release changes frozen bytes on purpose: its pull
request runs `--write` (with `--baseline <commit> <what>`: the commit that holds the recorded bytes, and what it is),
and the diff of frozen-bytes.json then shows the reviewers which frozen paths the release changed. Python standard
library and git only.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "frozen-bytes.json"
SOURCE_HASH = "tools/release/source_hash.py"   # the one tool an entry may name (`how`: "<this> --check")
ABSENT = {"sha256": "absent"}
NO_HASH = ("absent", "unreadable", "not a literal")   # what `sha256` says when there is nothing to hash


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "--literal-pathspecs", "-C", str(root), *args], check=True, capture_output=True,
                          text=True).stdout


def kind(e: dict) -> str:
    if "how" in e:
        if e["how"] != SOURCE_HASH + " --check":
            raise SystemExit(f"FAIL frozen-bytes.json: {e['path']}: `how` can only be \"{SOURCE_HASH} --check\"")
        return "source"
    return "value" if "value" in e else "bytes"


def selected(e: dict, names: list[str]) -> list[str]:
    """Of the repository paths `names`, those the bytes entry `e` covers."""
    path = e["path"]
    if not path.endswith("/"):
        return [f for f in names if f == path]

    def hit(f: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatchcase(f[len(path):], p) for p in patterns)
    return [f for f in names if f.startswith(path) and (hit(f, e["only"]) if "only" in e else True)
            and not hit(f, e.get("except", []))]


def files_of(root: Path, e: dict) -> list[str]:
    """The checkout's files of a bytes entry: tracked, or untracked and not ignored."""
    out = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", e["path"].rstrip("/"))
    return selected(e, sorted({f for f in out.split("\0") if f}))


def file_record(root: Path, f: str) -> str:
    """`<path>\\0<mode>\\0<sha256>\\n` of one file (a tracked file that is gone: mode and hash `absent`)."""
    p = root / f
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return f"{f}\0absent\0absent\n"
    h = hashlib.sha256()
    if stat.S_ISLNK(st.st_mode):
        h.update(os.readlink(p).encode())
        return f"{f}\0link\0{h.hexdigest()}\n"
    with open(p, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return f"{f}\0{'755' if st.st_mode & 0o100 else '644'}\0{h.hexdigest()}\n"


def value_state(root: Path, e: dict) -> dict:
    try:
        tree = ast.parse((root / e["path"]).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {"sha256": "unreadable"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == [e["value"]]:
            try:
                v = ast.literal_eval(node.value)
            except ValueError:
                return {"sha256": "not a literal"}
            return {"sha256": hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    return ABSENT


def source_state(root: Path) -> tuple[dict, str]:
    """(SOURCE.json as tools/release/source_hash.py --check prints it, its complaint when it does not pass)."""
    r = subprocess.run([sys.executable, str(root / SOURCE_HASH), "--check"], capture_output=True, text=True)
    try:
        got = json.loads(r.stdout.splitlines()[0])
        got = {"files": got["files"], "sha256": got["sha256"]}
    except (IndexError, KeyError, TypeError, ValueError):
        got = dict(ABSENT)
    said = (r.stderr.strip() or r.stdout.strip()).splitlines()
    return got, ("" if r.returncode == 0 else f"{SOURCE_HASH} --check: " + (said[-1] if said else f"exit {r.returncode}"))


def states(root: Path, entries: list[dict]) -> list[tuple[dict, str]]:
    """Per entry: (its state now, in the fields the data file records; what is wrong beyond a difference, or "")."""
    listing = {i: files_of(root, e) for i, e in enumerate(entries) if kind(e) == "bytes"}
    todo = [f for fs in listing.values() for f in fs]
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:   # sha256 releases the GIL
        record = dict(zip(todo, pool.map(lambda f: file_record(root, f), todo)))
    out = []
    for i, e in enumerate(entries):
        if kind(e) == "value":
            out.append((value_state(root, e), ""))
        elif kind(e) == "source":
            out.append(source_state(root))
        else:
            lines = [record[f] for f in listing[i]]
            gone = [f for f, line in zip(listing[i], lines) if line.endswith("\0absent\n")]
            said = f"{len(gone)} tracked file(s) are not in the checkout, the first: {gone[0]}" if gone else ""
            if len(gone) == len(lines):
                out.append((dict(ABSENT, files=0), said or "no such file in the checkout"))
            else:
                out.append(({"files": len(lines) - len(gone), "sha256": hashlib.sha256("".join(lines).encode()).hexdigest()}, said))
    return out


def changed(root: Path, commit: str, e: dict, limit: int = 8) -> list[str] | None:
    """The files of the entry that differ from the baseline commit, when this clone holds it (else None). For the
    message only: the verdict is the hashes'."""
    try:
        git(root, "cat-file", "-e", commit + "^{commit}")
        path = e["path"].rstrip("/")
        diff = git(root, "diff", "--name-status", "--no-renames", "-z", commit, "--", path).split("\0")
        new = [f for f in git(root, "ls-files", "-z", "--others", "--exclude-standard", "--", path).split("\0") if f]
    except (subprocess.CalledProcessError, OSError):
        return None
    names = {f: s for s, f in zip(diff[0::2], diff[1::2])} | {f: "new" for f in new}
    if kind(e) == "bytes":
        keep = set(selected(e, sorted(names)))
    else:
        keep = set(names)
        if kind(e) == "source":   # the hashed files only, by the tool's own rule
            try:
                spec = importlib.util.spec_from_file_location("source_hash", root / SOURCE_HASH)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                keep = {f for f in names if not mod.excluded(f[len("source/"):])}
            except Exception:   # noqa: BLE001  the message goes without the filter
                pass
    lines = [f"{names[f]:>3} {f}" for f in sorted(keep)]
    return lines[:limit] + ([f"... {len(lines) - limit} more"] if len(lines) > limit else [])


def show(s: dict) -> str:
    n = f"{s['files']} file(s), " if s.get("files") is not None else ""
    return n + (s["sha256"] if s["sha256"] in NO_HASH else f"sha256 {s['sha256'][:16]}")


def check(root: Path, data: Path) -> tuple[list[str], str]:
    """(the failures, the summary of what was compared)."""
    doc = json.loads(data.read_text(encoding="utf-8"))
    entries, base = doc["frozen"], doc["baseline"]
    short = base["commit"][:8]
    bad = []
    for e, (now, said) in zip(entries, states(root, entries)):
        want = {k: e.get(k) for k in now}
        if now == want and not said:
            continue
        what = e["path"] + (f" ({e['value']})" if "value" in e else "")
        if now != want:
            was = show(want) if want.get("sha256") else "nothing recorded"
            msg = [f"{what} differs from the baseline {short}: {was} at the baseline, {show(now)} now"]
        else:
            msg = [f"{what}: {said}"]
        msg.append(f"    frozen because: {e['reason']}")
        if now != want and said:
            msg.append(f"    {said}")
        names = changed(root, base["commit"], e)
        if names is None:
            msg.append(f"    which files: not known here (this clone does not hold the commit {short}); in a clone "
                       f"that does: git diff --stat {short} -- {e['path']}")
        elif names:
            msg += [f"    differs from {short}:"] + [f"      {n}" for n in names]
        bad.append("\n".join(msg))
    for g in doc.get("generators", []):
        if not (root / g["path"]).is_file():
            bad.append(f"{g['path']}: a generator frozen-bytes.json names does not exist (moved? correct the list)")
    n = sum(e.get("files", 1) for e in entries)
    return bad, f"{len(entries)} frozen path(s), {n} file(s), baseline {short} ({base['what']})"


def write(root: Path, data: Path, baseline: list[str] | None, quiet: bool = False) -> int:
    doc = json.loads(data.read_text(encoding="utf-8"))
    for e, (now, said) in zip(doc["frozen"], states(root, doc["frozen"])):
        if said or now["sha256"] in NO_HASH:
            print(f"FAIL {e['path']}: nothing to record ({said or show(now)}): correct the path or the checkout")
            return 1
        for k in ("files", "sha256"):
            e.pop(k, None)
        e.update(now)
    if baseline:
        doc["baseline"] = {"commit": git(root, "rev-parse", baseline[0] + "^{commit}").strip(), "what": baseline[1]}
    data.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not quiet:
        print(f"wrote {data}: {len(doc['frozen'])} frozen path(s); the baseline is named {doc['baseline']['commit'][:8]} "
              f"({doc['baseline']['what']})" + ("" if baseline else "; --baseline <commit> <what> renames it"))
    return 0


def selftest() -> int:
    """Each rule fires on a small repository: a baseline is written, then one change at a time must fail the check by
    the path's name (or, for a change outside the frozen bytes, pass), and is undone."""
    with tempfile.TemporaryDirectory() as tmp:
        root, data = Path(tmp) / "repo", Path(tmp) / "frozen-bytes.json"
        for f, text in (("d/a.txt", "a\n"), ("d/sub/b.npz", "b\n"), ("d/run.sh", "#!/bin/sh\n"), ("e/x.npz", "x\n"),
                        ("e/x.py", "x = 1\n"), ("e/README.md", "x\n"), ("one.json", "{}\n"), (".gitignore", "__pycache__/\n"),
                        ("hosts.py", '"""doc"""\nimport re\nHOSTS = ["a.org", "b.org"]   # one\nOTHER = 1\n')):
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            (root / f).write_text(text)
        (root / "d/run.sh").chmod(0o755)
        (root / "d/a.txt").chmod(0o644)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        frozen = [{"path": "d/", "reason": "a directory"}, {"path": "one.json", "reason": "a file"},
                  {"path": "e/", "only": ["*.npz", "*.md"], "except": ["README.md"], "reason": "a narrowed directory"},
                  {"path": "hosts.py", "value": "HOSTS", "reason": "a value"}]
        data.write_text(json.dumps({"baseline": {"commit": "0" * 40, "what": "selftest"}, "frozen": frozen}))
        if write(root, data, None, quiet=True) != 0:
            return 1
        a, hosts = root / "d/a.txt", root / "hosts.py"
        cases = [   # (what, the frozen path that must fail or None, do, undo)
            ("the baseline itself", None, lambda: None, lambda: None),
            ("one byte of a file in a directory", "d/", lambda: a.write_text("b\n"), lambda: a.write_text("a\n")),
            ("the executable bit", "d/", lambda: a.chmod(0o755), lambda: a.chmod(0o644)),
            ("a new file in a directory", "d/", lambda: (root / "d/new").write_text(""), (root / "d/new").unlink),
            ("a file removed", "d/", (root / "d/sub/b.npz").unlink, lambda: (root / "d/sub/b.npz").write_text("b\n")),
            ("an ignored file", None, lambda: ((root / "d/__pycache__").mkdir(), (root / "d/__pycache__/a.pyc").write_text("")),
             lambda: None),
            ("a single file", "one.json", lambda: (root / "one.json").write_text("{ }\n"), lambda: (root / "one.json").write_text("{}\n")),
            ("a file `only` selects", "e/", lambda: (root / "e/x.npz").write_text("y\n"), lambda: (root / "e/x.npz").write_text("x\n")),
            ("a file `only` leaves out", None, lambda: (root / "e/x.py").write_text("x = 2\n"), lambda: None),
            ("a file `except` leaves out", None, lambda: (root / "e/README.md").write_text("y\n"), lambda: None),
            ("the value", "hosts.py", lambda: hosts.write_text('HOSTS = ["a.org", "c.org"]\n'), lambda: None),
            ("the code and comments around the value", None, lambda: hosts.write_text('"""new"""\nHOSTS = ["a.org",\n  "b.org"]\n'),
             lambda: None),
            ("the value gone", "hosts.py", lambda: hosts.write_text("OTHER = 1\n"), lambda: hosts.write_text('HOSTS = ["a.org", "b.org"]\n')),
        ]
        wrong = []
        for what, path, do, undo in cases:
            do()
            bad, _ = check(root, data)
            hit = [b.splitlines()[0].split(" ")[0] for b in bad]
            if hit != ([path] if path else []):
                wrong.append(f"{what}: expected {[path] if path else 'a pass'}, the check reported {hit or 'a pass'}")
            undo()
        for w in wrong:
            print("FAIL check-frozen-bytes.py --selftest:", w)
        if not wrong:
            print(f"ok check-frozen-bytes.py --selftest: {len(cases)} cases, every change of a frozen byte fails by its "
                  f"path and no other change does")
        return 1 if wrong else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="No frozen path differs from the baseline (ci_checks/frozen-bytes.json).")
    ap.add_argument("--root", type=Path, default=HERE.parent, help="the checkout to check (default: this one)")
    ap.add_argument("--data", type=Path, default=DATA, help="the list and its baseline (default: frozen-bytes.json next to this file)")
    ap.add_argument("--write", action="store_true", help="record the checkout's hashes as the baseline")
    ap.add_argument("--baseline", nargs=2, metavar=("COMMIT", "WHAT"), help="with --write: the new baseline's commit and what it is")
    ap.add_argument("--selftest", action="store_true", help="the rules fire, on a temporary repository")
    a = ap.parse_args()
    sys.dont_write_bytecode = True
    if a.selftest:
        return selftest()
    root = a.root.resolve()
    try:
        git(root, "rev-parse", "--show-toplevel")
    except (subprocess.CalledProcessError, OSError):
        print(f"FAIL check-frozen-bytes.py: {root} is not a git checkout (git ls-files names the files to hash)")
        return 1
    if a.write:
        return write(root, a.data, a.baseline)
    t0 = time.monotonic()
    bad, summary = check(root, a.data)
    for b in bad:
        print("FAIL", b)
    took = f"{time.monotonic() - t0:.1f} s"
    if bad:
        print(f"FAIL frozen bytes: {len(bad)} path(s) differ; {summary}; {took}. Undo the change: these bytes change "
              f"only with a release (ci_checks/check-frozen-bytes.py, ci_checks/frozen-bytes.json)")
        return 1
    print(f"ok frozen bytes: {summary}; nothing differs; {took}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
