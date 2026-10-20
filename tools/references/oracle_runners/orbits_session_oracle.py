# Oracle code for `scripts/roboarc/shared/rcb_topogym.py run-oracle --task orbits --code <this file>` (orbits
# very_easy; easier has no passing reference): the native feedback policy (native_session.run_feedback_policy) through
# the given Session (`session` is in scope), which records every executed action. Diagnostic files go to
# $ORBITS_ORACLE_OUTPUT (default ./orbits_oracle_output).
import os

from topogym.planning.orbits.native_session import run_feedback_policy

run_feedback_policy(session, os.environ.get("ORBITS_ORACLE_OUTPUT", "orbits_oracle_output"),  # noqa: F821
                    variation=session.config["variation_id"].removeprefix("orbits_"))  # noqa: F821
