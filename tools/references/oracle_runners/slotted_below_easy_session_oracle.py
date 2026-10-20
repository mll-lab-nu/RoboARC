# Oracle code for `scripts/roboarc/shared/rcb_topogym.py run-oracle --task slotted_post_ring --code <this file>`
# (slotted very_easy, easier): the below-easy feedback policy (public_session.run_feedback_policy) through the given
# Session (`session` is in scope), which records every executed action. Diagnostic files go to $SLOTTED_ORACLE_OUTPUT
# (default ./slotted_oracle_output).
import os

from topogym.planning.slotted_post_ring.public_session import run_feedback_policy

run_feedback_policy(session, os.environ.get("SLOTTED_ORACLE_OUTPUT", "slotted_oracle_output"),  # noqa: F821
                    variation=session.config["variation_id"].removeprefix("slotted_post_ring_"))  # noqa: F821
