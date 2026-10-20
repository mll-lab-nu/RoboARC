# Oracle code for `rcb_topogym.py run-oracle --task horseshoe_ring --code <this file>` with a recipe file of one's own:
# the family's staged feedback oracle (public_session: the recipe's stage factory, the tier's below-easy IK
# scope where it applies) run closed loop in the given Session (`session` is in scope), with the recipe at
# $HS_RECIPE instead of reference_data/horseshoe_ring/<tier>/recipe.json; its planning inputs <name>.json lie next to it
# (hash-bound by the recipe, as the released ones). Diagnostic files go to $HS_ORACLE_OUTPUT. Used to try recipe changes
# before they are written into reference_data; the references themselves run the Harbor callback (hs_oracle.py).
import json
import os
from pathlib import Path

from topogym.planning.harbor_bridge.callbacks import run_stages
from topogym.planning.horseshoe_ring.public_session import FACTORIES
from topogym.planning.oracle_ik import scoped_session_oracle
from topogym.planning.standard_rate import rate_limited_runner


def _run(session, output):
    path = Path(os.environ["HS_RECIPE"])
    recipe = json.loads(path.read_text())
    task = session.env._task
    if (task.model_revision != recipe["model_revision"] or task.terminal_revision != recipe["terminal_revision"]
            or task.resource_hashes != recipe["resource_sha256"]):
        raise ValueError("Frozen Horseshoe model or terminal differs from the recipe")
    selection = recipe["selection"]
    for name, item in selection[0]["parameters"].items():
        if isinstance(item, dict) and "sha256" in item:
            item["path"] = str(path.parent / f"{name}.json")      # as public_session: the inputs next to the recipe
    return run_stages(session, output, FACTORIES[recipe.get("stages", "gravity_release_table_pickup_stages")], selection=selection,
                      runner=rate_limited_runner(session, mode="clamp"))


_result = scoped_session_oracle(_run)(session, os.environ["HS_ORACLE_OUTPUT"])  # noqa: F821
print("ORACLE_RESULT", _result, flush=True)
