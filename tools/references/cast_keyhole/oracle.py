# Oracle code for `rcb_topogym.py run-oracle --task cast_keyhole --code <this file>` (docs/simulator/families/cast_keyhole.md):
# the tier's own Harbor session callback (catalog.session_callback: very_easy / easier -> below_easy ->
# cast_keyhole.public_session, SimplifiedKeyholePolicy; easy / medium / hard -> cast_keyhole.harbor_session ->
# runner.worker, CastKeyholePolicy with its regrasps), closed loop in the given Session (`session` is in scope), which
# records every executed action. If the policy returns before the episode has ended, its last targets are held until
# the episode ends or 20 s pass. Diagnostic files go to $KEYHOLE_ORACLE_OUTPUT.
import importlib
import json
import os
import sys
import time

from topogym.planning.harbor_bridge.catalog import session_callback

_tier = session.env._task.variation_id.removeprefix("cast_keyhole_")  # noqa: F821
_module, _name = session_callback("cast_keyhole", _tier).split(":")
print("keyhole oracle:", _tier, _module, flush=True)
_t0 = time.time()
try:
    _result = getattr(importlib.import_module(_module), _name)(session, os.environ["KEYHOLE_ORACLE_OUTPUT"],  # noqa: F821
                                                               variation=_tier)
finally:
    print(json.dumps({"oracle_n_steps": session.n_steps, "episode_end": session.episode_end,  # noqa: F821
                      "wall_s": round(time.time() - _t0, 1)}), file=sys.stderr, flush=True)
_hold = 0
while not session.episode_done and session.episode_end is None and _hold < 20 * session.control_hz:  # noqa: F821
    session.step(session.hold_action())  # noqa: F821
    _hold += 1
print(json.dumps({"hold_rows": _hold, "episode_end": session.episode_end, "success": session.success()}),  # noqa: F821
      file=sys.stderr, flush=True)
