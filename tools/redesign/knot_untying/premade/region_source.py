"""A make_premade.py source whose goal region follows a later region decision (V3, 2026-10-17): a copy of a frozen source
instance with only task_parameters.redesign.regions replaced by a lab birth's regions. The user's decision of 2026-10-13
(_r3, option A: the back strip of easier and easy runs x -0.45..0.42) changed the region only; the _r3 births are the base
births with the region and the note changed (docs/simulator/redesign/knot_untying.md section 6), and the region is read only by
the goal (the same rows give identical world and robot hashes on either instance). knot-untying-07-i01 = inst10/easier_v8_r2
with births/easier_v8_r3's region.

usage: python region_source.py <source instance dir> <births/<name>_r3 dir> <out dir>   (then make_premade.py <out dir> ...)"""
import json
import shutil
import sys
from pathlib import Path

src, birth, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if out.exists():
    shutil.rmtree(out)
shutil.copytree(src, out)
inst = json.loads((out / "instance.json").read_text())
regions = json.loads((birth / "scene_start.json").read_text())["regions"]
old = inst["simulator"]["task_parameters"]["redesign"]["regions"]
if set(old) != set(regions):
    raise SystemExit(f"region names differ: {sorted(old)} vs {sorted(regions)}")
inst["simulator"]["task_parameters"]["redesign"]["regions"] = regions
(out / "instance.json").write_text(json.dumps(inst, indent=1))
print("REGION " + json.dumps(dict(source=str(src), birth=birth.name, old=old, new=regions)))
