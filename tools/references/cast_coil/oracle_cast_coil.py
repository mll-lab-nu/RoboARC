# Oracle code for `rcb_topogym.py run-oracle --task cast_coil --code <this file>` on the released tiers
# (docs/simulator/families/cast_coil.md): the family's own closed-loop policy
# (topogym.planning.cast_coil.public_session.run_feedback_policy: CastCoilPolicy / SimplifiedCoilPolicy, measured piece
# poses in, EE_ABS out) in this MuJoCo Session, unchanged. If the policy finishes before the episode has ended, the last
# targets are held (hold rows, as core's re-record does) until the episode ends or 20 s pass.
import json, sys, time
from topogym.planning.cast_coil.public_session import run_feedback_policy

t0 = time.time()
try:
    result = run_feedback_policy(session, None)
finally:
    print(json.dumps({"oracle_n_steps": session.n_steps, "episode_end": session.episode_end,
                      "wall_s": round(time.time() - t0, 1)}), file=sys.stderr, flush=True)
print(json.dumps({"oracle_result": result}, default=str), file=sys.stderr, flush=True)
hold = 0
while not session.episode_done and session.episode_end is None and hold < 20 * session.control_hz:
    session.step(session.hold_action())
    hold += 1
print(json.dumps({"hold_rows": hold, "episode_end": session.episode_end, "success": session.success()}),
      file=sys.stderr, flush=True)
