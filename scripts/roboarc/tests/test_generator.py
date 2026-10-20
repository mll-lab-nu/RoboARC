"""Invariants of the task generator scripts/roboarc/mk_tasks.py (standard library only, Python >= 3.11, about 10 s;
the full proof is `mk_tasks.py --check`, which renders all 202 task dirs):
  1. every package is final (its fields, e.g. {{AGENT_EGRESS}} in task.toml) and holds only the fields mk_tasks fills in
     its mode, and no
     internal dev image tag;
  2. every package's task.toml states a difficulty of TIERS, the one roboarc_names.json gives where it names the base;
  3. fill() refuses an unknown field;
  4. the pinned tooling is the release's (tooling_check), and the families that render with nyx_direct.py are read;
     the interface tools/release/release_rebuild.py drives the generator through (PACKAGE_FORMAT) holds;
  5. two bases (a rigid one, a cloth one with goal images) render to exactly their task dirs in tasks/.
usage: python3 scripts/roboarc/tests/test_generator.py"""
import pathlib, re, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mk_tasks as M  # noqa: E402

FIELDS = {"privileged": {"IMAGE", "AGENT_EGRESS", "TOOLING_VERSION", "ORACLE", "REFERENCE_SOLUTION", "NOP", "PROVENANCE",
                         "CHANGE_LOG", "ROW_CAP", "ROW_CAP_MIN", "VERIFIER_BUDGET_S"},
          "standard": {"IMAGE", "AGENT_EGRESS", "TOOLING_VERSION", "ORACLE", "REFERENCE_SOLUTION", "NOP", "PROVENANCE",
                       "CHANGE_LOG", "TIME_PER_CALL"}}
DEV_TAG = re.compile(r"rcb-topogym:|\b(?:labv1\d\w*|hsbe2|obejw)/")


def main() -> int:
    bases = sorted(M.REL["bases"])
    for b in bases:
        assert "{{AGENT_EGRESS}}" in (M.PKG / f"{b}-privileged" / "task.toml").read_text(), f"packages/{b}: not a final package"
    named = {v["new"].removeprefix(M.BENCH + "-"): v["difficulty"] for v in M.NAMES.values()}
    for b in bases:
        for mode in ("privileged", "standard"):
            up = M.PKG / f"{b}-{mode}"
            assert sorted(set(M.TEXTS) - {"tests/docker-compose.yaml"}) == sorted(n for n in M.TEXTS if (up / n).is_file()
                                                                                 and n != "tests/docker-compose.yaml"), up
            for n in M.TEXTS:
                if (up / n).is_file():
                    s = (up / n).read_text()
                    extra = set(M.FIELD_RX.findall(s)) - FIELDS[mode]
                    assert not extra, f"{up.name}/{n}: fields {extra}"
                    assert not DEV_TAG.search(s), f"{up.name}/{n}: an internal dev image tag"
            d = re.search(r'(?m)^difficulty = "([^"]*)"', (up / "task.toml").read_text())
            assert d and d[1] in M.TIERS, f"{up.name}: no difficulty"
            assert b not in named or named[b] == d[1], f"{up.name}: difficulty {d[1]}, roboarc_names.json {named[b]}"
    try:
        M.fill("{{NOT_A_FIELD}}", {}, "test")
        raise AssertionError("fill() accepted an unknown field")
    except AssertionError as e:
        assert "unknown fields" in str(e), e
    assert M.tooling_check() == [], M.tooling_check()
    assert M.PACKAGE_FORMAT == "final-texts/1", "the interface release_rebuild.py relies on"
    assert M.DIRECT and set(M.DIRECT) <= {r["family"] for r in M.REL["bases"].values()}, M.DIRECT
    with tempfile.TemporaryDirectory() as tmp:
        for b in ("orbits-01-i01", "blanket-fold-02-i01"):
            for t in M.render(b, pathlib.Path(tmp)):
                assert M.snapshot(t) == M.snapshot(M.TASKS / t.name), f"{t.name}: differs from tasks/"
    print(f"ok mk_tasks: {len(bases)} bases' packages final, 4 task dirs equal tasks/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
