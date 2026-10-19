# hedgehog_in_cage cast_cage reference for `rcb_topogym.py run-oracle --code`: the oracle in
# topogym/planning/hedgehog_in_cage/cast_cage.py (the rod taken, one push through W90, the rod put back, spike U held
# and the star threaded out of W0 along the designed route, carried into the goal), then a 200-row hold tail.
import json
from topogym.planning.hedgehog_in_cage.cast_cage import run_oracle
report = run_oracle(session, log=lambda e: print(json.dumps(e)[:600], flush=True))
print("ORACLE", json.dumps({k: v for k, v in report.items() if k != "stages"}), flush=True)
for _ in range(200):
    session.step(session.hold_action())
