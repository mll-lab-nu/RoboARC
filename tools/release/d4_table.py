#!/usr/bin/env python3
"""One table of the appearance-seed cheap checks (tools/release/d4_cheap.sh logs): d4_table.py <out dir> (markdown to
stdout; d4.json next to the logs)."""
import json, pathlib, sys
out = pathlib.Path(sys.argv[1]); rows = []; missing = []
for f in sorted(out.glob("*.log")):
    got = [json.loads(l[3:]) for l in f.read_text().splitlines() if l.startswith("D4 ")]
    rows += got
    if not got:
        missing.append(f.stem)
(out / "d4.json").write_text(json.dumps(rows, indent=1))
ok = sum(r["ok"] for r in rows)
print(f"# D4 cheap checks\n\n{ok} / {len(rows)} seed instances pass; bases without a result: {missing or 'none'}\n")
print("| base | seed | birth | a files | a fields | a model | b strict | c birth (parts differing) | d restamp | ok |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['base']} | {r['seed']} | `{r.get('birth')}` | {r['a_files']} | {r['a_fields']} | {r['a_model']} | {r['b_strict']} | "
          f"{r['c_birth']} ({','.join(r['c_birth_parts_differing'])}) | {r['d_restamp']} | {'**ok**' if r['ok'] else '**FAIL**'} |")
