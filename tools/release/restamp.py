#!/usr/bin/env python3
"""Re-stamp a privileged reference trajectory for another appearance seed of its base (owner decision D3 (a) for 1.2.0;
docs/RUNNING.md "Seeds").

    python3 tools/release/restamp.py <privileged.npz> --sha <birth state_sha of the seed> --out <privileged.s<S>.npz>
    python3 tools/release/restamp.py --check <privileged.npz> <privileged.s<S>.npz> [--sha <expected>]

A trajectory records the state hash of the birth it was planned from (`instance_sha.npy`), and the grader checks it
against the instance it replays on. An appearance seed has the same physics as seed 0 but another birth hash (the
appearance parameters enter `instance_id`), so its reference is seed 0's trajectory with only that one array replaced:
every other member of the archive (the action rows, `scene_seed`, the layout, the metadata) keeps its bytes, and so do
the archive's member order, names, dates and compression. Whether the rows reach the goal on the seed is shown by the
seed's strict double replay (the 1.2.0 evidence), not by this script.

--check: the two archives differ in `instance_sha.npy` only (and, with --sha, the second names that hash).
Standard library only.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

MEMBER = "instance_sha.npy"
SHA_RE = re.compile(r"[0-9a-f]{16}")


def npy_str(raw: bytes) -> tuple[bytes, str]:
    """(header, value) of a 0-d '<U16' .npy member: the header bytes and the string."""
    if not raw.startswith(b"\x93NUMPY"):
        raise ValueError(f"{MEMBER} is not a .npy array")
    major = raw[6]
    hlen = int.from_bytes(raw[8:10], "little") if major == 1 else int.from_bytes(raw[8:12], "little")
    start = (10 if major == 1 else 12) + hlen
    header = raw[:start]
    if b"'descr': '<U16'" not in header or b"'shape': ()" not in header:
        raise ValueError(f"{MEMBER} is not a 0-d '<U16' array: {header!r}")
    return header, raw[start:].decode("utf-32-le")


def read(p: Path) -> list[tuple[zipfile.ZipInfo, bytes]]:
    with zipfile.ZipFile(p) as z:
        return [(i, z.read(i)) for i in z.infolist()]


def restamp(src: Path, sha: str, out: Path) -> str:
    if not SHA_RE.fullmatch(sha):
        raise SystemExit(f"--sha {sha!r}: a state_sha is 16 lower-case hex digits")
    members = read(src)
    names = [i.filename for i, _ in members]
    if names.count(MEMBER) != 1:
        raise SystemExit(f"{src}: no single {MEMBER}")
    old = None
    with zipfile.ZipFile(out, "w") as z:
        for info, data in members:
            if info.filename == MEMBER:
                header, old = npy_str(data)
                data = header + sha.encode("utf-32-le")
            z.writestr(info, data)
    bad = check(src, out, sha)
    if bad:
        out.unlink()
        raise SystemExit("; ".join(bad))
    return old


def check(a: Path, b: Path, sha: str | None = None) -> list[str]:
    ma, mb = read(a), read(b)
    bad = []
    if [i.filename for i, _ in ma] != [i.filename for i, _ in mb]:
        return [f"{b}: not the members of {a} in their order"]
    for (ia, da), (ib, db) in zip(ma, mb):
        same = (ia.compress_type, ia.date_time, ia.external_attr) == (ib.compress_type, ib.date_time, ib.external_attr)
        if not same:
            bad.append(f"{ia.filename}: its zip entry changed (compression, date or attributes)")
        if ia.filename == MEMBER:
            if sha is not None and npy_str(db)[1] != sha:
                bad.append(f"{MEMBER} of {b} is {npy_str(db)[1]}, not {sha}")
            if npy_str(da)[0] != npy_str(db)[0]:
                bad.append(f"{MEMBER}: its .npy header changed")
        elif da != db:
            bad.append(f"{ia.filename}: bytes differ (only {MEMBER} may)")
    return bad


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("other", type=Path, nargs="?")
    ap.add_argument("--sha")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        if a.other is None:
            ap.error("--check takes two archives")
        bad = check(a.src, a.other, a.sha)
        for x in bad:
            print("FAIL", x)
        if not bad:
            print(f"ok {a.other}: {a.src} with {MEMBER} = {npy_str(dict((i.filename, d) for i, d in read(a.other))[MEMBER])[1]}")
        return 1 if bad else 0
    if a.out is None or a.sha is None or a.other is not None:
        ap.error("restamp takes <src> --sha <state_sha> --out <out>")
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    old = restamp(a.src, a.sha, a.out)
    print(f"ok {a.out}: {MEMBER} {old} -> {a.sha}, every other member unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
