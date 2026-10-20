# Oracle code for `scripts/roboarc/shared/rcb_topogym.py run-oracle --task slotted_post_ring --code <this file>`
# (#108's chain settings: easy_chain, medium_chain, hard_chain): run_slotted_chain_oracle.run in the given Session
# (`session` is in scope), which records every executed action. Its report and own trajectory go to
# $SLOTTED_ORACLE_OUTPUT (default ./slotted_oracle_output).
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ.get("TOPOGYM_ROOT", ".")) / "tools" / "references" / "oracle_runners"))
from run_slotted_chain_oracle import run  # noqa: E402

run(session, os.environ.get("SLOTTED_ORACLE_OUTPUT", "slotted_oracle_output"))  # noqa: F821
