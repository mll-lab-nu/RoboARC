# hedgehog_in_cage window_change reference: run-oracle code. The variant's own oracle (topogym/planning/hedgehog_in_cage/
# window_change.py with window_change_plan.json: three grasps, two set-downs), then hold rows until the episode ends (at
# most 200). Its instance is made in the width gripper space (the oracle commands widths):
#   python scripts/robotangle/shared/rcb_topogym.py make-instance --task hedgehog_in_cage \
#       --variation hedgehog_in_cage_window_change --scene-seed 0 --gripper-space width --out <dir>/instance.json
#   python scripts/robotangle/shared/rcb_topogym.py run-oracle --task hedgehog_in_cage --instance <dir>/instance.json \
#       --code scripts/robotangle/references/hedgehog_in_cage/window_change_code.py --out <dir>/privileged.npz
import json
from topogym.planning.hedgehog_in_cage.window_change import run_oracle

report = run_oracle(session, log=lambda e: print(json.dumps(e)[:600], flush=True))
print("ORACLE", json.dumps({k: v for k, v in report.items() if k != "stages"}), flush=True)
for _ in range(200):
    if session.step(session.hold_action()):
        break
