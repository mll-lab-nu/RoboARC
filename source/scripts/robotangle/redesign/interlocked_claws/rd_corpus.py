"""A corpus manifest for the redesigned claws instances: the family's corpus streams (manifest cases of the released
interlocked_claws instances) applied to each redesign instance base (same robot, same action layout).
python work/rd_corpus.py <stress corpus dir> <out dir> <base> [<base> ...]"""
import json, sys
from pathlib import Path
src, out, bases = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
m = json.loads((src / "manifest.json").read_text())
cases = [c for c in m["cases"] if c["base"].startswith("interlocked-claws-")]
new = []
for b in bases:
    for c in cases:
        e = dict(c); e["base"] = b; e["key"] = f"{b}:{c['actions_sha256']}"; e["from_base"] = c["base"]
        new.append(e)
out.mkdir(parents=True, exist_ok=True)
link = out / "inputs"
if not link.exists():
    link.symlink_to((src / "inputs").resolve())
(out / "manifest.json").write_text(json.dumps(dict(about="claws corpus applied to the redesign instances (agent claws)",
                                                   families=m["families"], cases=new, skipped=[]), indent=1))
print(len(new), "cases for", bases)
