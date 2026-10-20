#!/usr/bin/env python3
"""The documentation names only paths that exist: every relative markdown link of docs/**/*.md (docs/history/ is a
record and is left as written) resolves from its file, and every repository path the text names (a word that starts
with one of ROOTS, in prose, backticks or code blocks) exists in the checkout.

A path may hold placeholders: `<base>`, `*`, `...` / `…` match anything, `{a,b}` either. A trailing `:<line>` is
dropped. A path of the development repository that docs/rcb/README.md maps to its present place (a backticked path in
the first column of its tables, placeholders allowed) is accepted as written: the frozen task files and the
development notes cite those. IKEA's pages may name paths relative to `ikea/`. ABSENT lists the paths the docs name on
purpose although this repository does not hold them, RECORDS the pages whose prose paths are the development
repository's (with the reason).

    python3 ci_checks/check-doc-paths.py [files...]     # default: docs/**/*.md except docs/history/; stdlib only
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOTS = ("scripts", "tools", "records", "source", "docs", "ci_checks", "images", "ikea", "tasks", "examples", "roboarc",
         "tests/api", ".github", "LICENSES")
PATH = re.compile(r"(?<![\w./~-])((?:" + "|".join(re.escape(r) for r in ROOTS) + r")/(?:[^\s`'\"()\[\],;|{]|\{[^}\s]*\})*)")
LINK = re.compile(r"\]\(([^)\s]+)\)")
MAP = ROOT / "docs/rcb/README.md"
HOLE = r"<[^>]*>|\*|\.\.\.|…"   # a placeholder in a path
RECORDS = {   # pages that record the development repository: their prose paths are that repository's (links still checked)
    "docs/simulator/TOPOGYM_PROTOCOL.md": "the Genesis-era protocol record the task files cite; its preface says so",
}
ABSENT = {
    "tests/api": "the Python package's tests, once added (CI job api-tests is skipped without them)",
}


def mapped() -> list[re.Pattern]:
    """The old paths docs/rcb/README.md maps (backticked, first column of its tables), as patterns: a placeholder
    (`<x>`, `*`) stands for one or more path characters, `{a,b}` for either, and the path covers what is below it."""
    out = []
    for line in (MAP.read_text(encoding="utf-8").splitlines() if MAP.is_file() else []):
        if line.startswith("| `"):
            for p in re.findall(r"`([^`]+)`", line.split(" | ")[0]):
                for q in braces(p.rstrip("/")):
                    parts = re.split(f"({HOLE})", q)
                    rx = "".join("[^/]+" if re.fullmatch(HOLE, x) else re.escape(x) for x in parts)
                    out.append(re.compile(rx + r"(?:/.*)?$"))
    return out


def braces(p: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", p)
    if not m:
        return [p]
    return [q for alt in m.group(1).split(",") for q in braces(p[:m.start()] + alt + p[m.end():])]


def normal(p: str) -> str:
    return re.sub(HOLE, "*", p)


def exists(p: str, bases: list[Path]) -> bool:
    p = re.sub(r":\d+(?:-\d+)?$", "", p.rstrip(".:"))
    pat = normal(p)
    for q in braces(pat):
        for b in bases:
            if any(Path(x).exists() for x in glob.glob(str(b / q))) or (b / q).exists():
                return True
    return False


def check(f: Path, old: list[re.Pattern]) -> list[str]:
    text = f.read_text(encoding="utf-8")
    bad = []
    bases = [ROOT] + ([ROOT / "ikea"] if "ikea" in f.name else [])
    for n, line in enumerate(text.splitlines(), 1):
        for m in LINK.finditer(line):
            t = m.group(1).split("#")[0]
            if t and not t.startswith(("http://", "https://", "mailto:")) and not (f.parent / t).exists():
                bad.append(f"{f.relative_to(ROOT)}:{n}: link {t}")
        if f == MAP or f.relative_to(ROOT).as_posix() in RECORDS:
            continue
        clean = re.sub(r"\]\([^)]*\)", "]", line)
        for m in PATH.finditer(clean):
            p = m.group(1)
            if m.start(1) > 0 and clean[m.start(1) - 1] == "<":
                p = p.split(">")[0]                         # <images/roboarc/VERSION>: a placeholder naming a file
            p = p.rstrip(".:")
            if any(p.rstrip("/") == a or p.startswith(a + "/") for a in ABSENT):
                continue
            q = normal(p).replace("*", "x").rstrip("/")
            if any(o.match(q) or o.match(q + "/x") for o in old):      # the path, or the directory that holds it
                continue
            if not exists(p, bases):
                bad.append(f"{f.relative_to(ROOT)}:{n}: path {p}")
    return bad


def main(argv: list[str]) -> int:
    files = [Path(a).resolve() for a in argv] or sorted(
        p for p in (ROOT / "docs").rglob("*.md") if "history" not in p.relative_to(ROOT / "docs").parts[:1])
    old = mapped()
    bad = [b for f in files for b in check(f, old)]
    for b in bad:
        print("FAIL", b)
    print(f"{'FAIL' if bad else 'ok'} check-doc-paths: {len(files)} file(s), {len(bad)} missing path(s) or link(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
