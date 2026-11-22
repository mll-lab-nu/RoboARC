"""The owner's exceptions to the 14400 s cutoff cap (scripts/task_conventions.py CUTOFF_EXCEPTIONS) and what the
generator renders for an excepted base (standard library only, about 1 s):
  1. every exception names a RoboARC base (no mode suffix) and is a whole number of seconds above the cap;
  2. cutoff_cap_for gives the exception to both modes of its base, and the cap to every other task;
  3. ci_checks/check-cutoffs.py accepts an excepted base's cutoff and refuses the same number on another base;
  4. mk_tasks: cutoff_of is the base's exception, else CUTOFF_SEC; grading_timeouts leaves every package's
     grading timeouts as they are at CUTOFF_SEC, and raises them for a longer cutoff (privileged verifier
     2 x cutoff + 1800 s, standard collect hook 1.5 x cutoff), never lowering one; verifier_timeout (row_cap) follows.
usage: python3 scripts/roboarc/tests/test_cutoff_exceptions.py"""
import importlib.util, pathlib, re, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import mk_tasks as M  # noqa: E402
TC = M.TC
spec = importlib.util.spec_from_file_location("check_cutoffs", ROOT / "ci_checks/check-cutoffs.py")
CC = importlib.util.module_from_spec(spec); spec.loader.exec_module(CC)

PRIV = r"(?ms)^\[verifier\]\n.*?^timeout_sec = ([0-9.]+)"
COLLECT = r'(?ms)^command = "sh /srv/finalize\.sh".*?\n^timeout_sec = ([0-9.]+)'


def toml_task(d: pathlib.Path, name: str, sec: int) -> pathlib.Path:
    t = d / name; t.mkdir()
    (t / "task.toml").write_text(f"[agent]\ntimeout_sec = {sec}.0\n")
    return t


def main() -> int:
    modes = sorted(TC.MODE_REGISTRY, key=len, reverse=True)
    for base, sec in TC.CUTOFF_EXCEPTIONS.items():
        assert base.startswith(M.BENCH + "-") and not any(base.endswith("-" + m) for m in modes), base
        assert isinstance(sec, int) and sec > TC.CUTOFF_CAP, (base, sec)
    wire, sec = "roboarc-wire-splicing-01-i01", TC.CUTOFF_EXCEPTIONS["roboarc-wire-splicing-01-i01"]
    for m in ("privileged", "standard"):
        assert TC.cutoff_cap_for(pathlib.Path(f"tasks/{wire}-{m}")) == sec
    some = sorted(M.REL["bases"])[0]
    assert TC.cutoff_cap_for(pathlib.Path(f"tasks/{M.BENCH}-{some}-standard")) == TC.CUTOFF_CAP
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        assert CC.number_errors(toml_task(d, f"{wire}-standard", sec)) == []
        errs = CC.number_errors(toml_task(d, f"{M.BENCH}-{some}-standard", sec))
        assert errs and "cap" in errs[0], errs
    n = 0
    for b in sorted(M.REL["bases"]):
        assert M.cutoff_of(b) == TC.CUTOFF_EXCEPTIONS.get(f"{M.BENCH}-{b}", M.CUTOFF_SEC), b
        for mode, pat in (("privileged", PRIV), ("standard", COLLECT)):
            s = (M.PKG / f"{b}-{mode}" / "task.toml").read_text()
            assert M.grading_timeouts(s, mode, M.CUTOFF_SEC, b) == s, (b, mode)
            old = float(re.search(pat, s).group(1))
            for cut in (21600, 28800):
                new = float(re.search(pat, M.grading_timeouts(s, mode, cut, b)).group(1))
                need = 2 * cut + 1800 if mode == "privileged" else int(1.5 * cut)
                assert new == max(old, need), (b, mode, cut, old, new)
            n += 1
    saved = dict(TC.CUTOFF_EXCEPTIONS)
    try:
        TC.CUTOFF_EXCEPTIONS[f"{M.BENCH}-{some}"] = 28800
        assert M.cutoff_of(some) == 28800 and M.verifier_timeout(some) == 2 * 28800 + 1800
    finally:
        TC.CUTOFF_EXCEPTIONS.clear(); TC.CUTOFF_EXCEPTIONS.update(saved)
    print(f"cutoff exceptions: {len(TC.CUTOFF_EXCEPTIONS)} base(s); {n} package tomls unchanged at {M.CUTOFF_SEC} s and "
          "raised as needed above it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
