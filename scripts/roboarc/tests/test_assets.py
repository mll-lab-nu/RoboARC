"""Unit test of scripts/roboarc/assets.py on a small fake dataset (standard library only, no network, about 1 s):
  1. pin_from writes the committed assets.json layout (keys, order, formatting; instances/ only);
  2. check passes on the pinned copy and names a changed, a missing file and a changed instance.json;
  3. check-instances (an image build context, one directory per base) passes on exactly the pinned files and names an
     extra file, an extra base and a changed file;
  4. the committed assets.json has that layout, and its instances agree with its files;
  5. appearance seeds (instances/<base>/seeds/<S>/): pinned with their own canonical sha256, checked by check and
     check-instances; a seed's files without its pin, or a pinned seed without its files, are named.
usage: python3 scripts/roboarc/tests/test_assets.py"""
import json, pathlib, shutil, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import assets as A  # noqa: E402


def fake(d: pathlib.Path) -> None:
    for base, n in (("alpha-01-i01", 2), ("beta-02-i01", 1)):
        b = d / "instances" / base
        (b / "instance_assets" / "mujoco").mkdir(parents=True)
        (b / "instance.json").write_text(json.dumps({"b": 1, "a": base}, indent=2))
        for i in range(n):
            (b / "instance_assets" / "mujoco" / f"part{i}.xml").write_text(f"<mujoco {i}/>")
    (d / "image").mkdir()
    (d / "image" / "release-abc.tar.gz").write_bytes(b"not pinned")   # the dataset's image/ files are never pinned
    (d / ".cache").mkdir()
    (d / ".cache" / "x").write_text("hf cache")


def main() -> int:
    committed = json.loads(A.PIN.read_text())
    assert A.dumps(committed) == A.PIN.read_text(), "assets.json is not in the layout pin writes"
    assert list(committed) == ["repo", "repo_type", "private", "revision", "files", "instances"], list(committed)
    assert all(k.startswith(A.PREFIX) for k in committed["files"]), "assets.json pins files outside instances/"
    assert {k.split("/")[1] for k in committed["files"]} == set(committed["instances"]), "files and instances disagree"
    assert A.schema(committed) == [], A.schema(committed)
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp) / "ds"
        fake(d)
        pin = A.pin_from(d, "f" * 40, old=committed)
        assert list(pin) == list(committed) and pin["repo"] == committed["repo"] and pin["revision"] == "f" * 40
        assert sorted(pin["files"]) == [
            "instances/alpha-01-i01/instance.json", "instances/alpha-01-i01/instance_assets/mujoco/part0.xml",
            "instances/alpha-01-i01/instance_assets/mujoco/part1.xml", "instances/beta-02-i01/instance.json",
            "instances/beta-02-i01/instance_assets/mujoco/part0.xml"], pin["files"]
        assert set(pin["instances"]) == {"alpha-01-i01", "beta-02-i01"}
        assert A.check(d, pin) == []
        (d / "instances/beta-02-i01/instance_assets/mujoco/part0.xml").write_text("changed")
        (d / "instances/alpha-01-i01/instance_assets/mujoco/part1.xml").unlink()
        (d / "instances/alpha-01-i01/instance.json").write_text(json.dumps({"a": "alpha-01-i01", "b": 2}))
        bad = A.check(d, pin)
        assert any("beta-02-i01/instance_assets/mujoco/part0.xml: sha256" in b for b in bad), bad
        assert any("part1.xml: missing" in b for b in bad), bad
        assert any("alpha-01-i01/instance.json: canonical" in b for b in bad), bad
        # an image build context: one directory per base
        d2 = pathlib.Path(tmp) / "ds2"
        fake(d2)
        pin = A.pin_from(d2, "e" * 40, old=committed)
        ctx = d2 / "instances"
        assert A.check_instances(ctx, pin) == []
        (ctx / "alpha-01-i01" / "instance_assets" / "stray.bin").write_bytes(b"x")
        (ctx / "gamma-03-i01").mkdir()
        shutil.copy(ctx / "beta-02-i01" / "instance.json", ctx / "gamma-03-i01" / "instance.json")
        (ctx / "beta-02-i01" / "instance.json").write_text("{}")
        bad = A.check_instances(ctx, pin)
        assert "alpha-01-i01/instance_assets/stray.bin: not pinned in scripts/roboarc/assets.json" in bad, bad
        assert any(b.startswith("gamma-03-i01") for b in bad), bad
        assert "beta-02-i01/instance.json: sha256 differs from scripts/roboarc/assets.json" in bad, bad
        # appearance seeds
        d3 = pathlib.Path(tmp) / "ds3"
        fake(d3)
        for s in (1, 2):
            sd = d3 / "instances/alpha-01-i01/seeds" / str(s)
            shutil.copytree(d3 / "instances/alpha-01-i01/instance_assets", sd / "instance_assets")
            (sd / "instance.json").write_text(json.dumps({"a": "alpha-01-i01", "b": 1, "seed": s}))
        pin = A.pin_from(d3, "d" * 40, old=committed)
        assert "seeds" not in pin["instances"]["beta-02-i01"]
        assert sorted(pin["instances"]["alpha-01-i01"]["seeds"]) == ["1", "2"]
        assert "instances/alpha-01-i01/seeds/2/instance_assets/mujoco/part1.xml" in pin["files"]
        assert A.check(d3, pin) == [] and A.check_instances(d3 / "instances", pin) == []
        (d3 / "instances/alpha-01-i01/seeds/2/instance.json").write_text(json.dumps({"seed": 3}))
        assert any("seeds/2/instance.json" in b for b in A.check(d3, pin)), A.check(d3, pin)
        assert any("seeds/2/instance.json" in b for b in A.check_instances(d3 / "instances", pin))
        unpinned = json.loads(json.dumps(pin))
        del unpinned["instances"]["alpha-01-i01"]["seeds"]["2"]
        assert any("which instances.alpha-01-i01.seeds does not pin" in b for b in A.schema(unpinned)), A.schema(unpinned)
        nofiles = json.loads(json.dumps(pin))
        nofiles["instances"]["beta-02-i01"]["seeds"] = {"1": {"instance_sha256": "0" * 64}}
        assert any("pinned instance without its file" in b for b in A.schema(nofiles)), A.schema(nofiles)
    print("ok assets.py: pin layout, check and check-instances, appearance seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
