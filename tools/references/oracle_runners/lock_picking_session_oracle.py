# Oracle code for `scripts/roboarc/shared/rcb_topogym.py run-oracle --task lock_picking --code <this file>`:
# the chain lock's certified plan through the given Session (`session` is in scope), which records every executed
# action. Diagnostic files go to $LOCK_ORACLE_OUTPUT (default ./lock_oracle_output).
import os

from topogym.planning.lock_picking.public_session import run_feedback_policy

run_feedback_policy(session, os.environ.get("LOCK_ORACLE_OUTPUT", "lock_oracle_output"),  # noqa: F821
                    variation=session.env._task.variation_id.removeprefix("lock_picking_"))  # noqa: F821
