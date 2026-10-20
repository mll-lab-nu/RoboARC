# hedgehog_in_cage reference: run-oracle code for any tier. The family's oracle (topogym/planning/hedgehog_in_cage/
# public_session.py: free-hand route, approach, contact-weld grasp, test lift, measured held route, regrasp where the tier
# has one, the unchanged Task goal), closed loop on the frozen instance, then hold rows until the episode ends (at most
# 400). The variation comes from the instance. docs/rcb/families/hedgehog_in_cage.md.
#   python scripts/roboarc/shared/rcb_topogym.py run-oracle --task hedgehog_in_cage --instance <instance.json> \
#       --code tools/references/hedgehog_in_cage/oracle_code.py --out <tier>.npz
import json
import tempfile
from topogym.planning.hedgehog_in_cage.public_session import run_feedback_policy

variation = session.env._task.variation_id.removeprefix("hedgehog_in_cage_")
report = run_feedback_policy(session, tempfile.mkdtemp(prefix=f"hedgehog_{variation}_"), variation=variation)
print("ORACLE", json.dumps({k: report.get(k) for k in ("status", "task_success", "test_lift_rise_m")}, default=str),
      flush=True)
for _ in range(400):
    if session.step(session.hold_action()):
        break
