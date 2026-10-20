# Session oracles (run-oracle code)

The code that `rcb_topogym.py run-oracle --code <file>` runs inside a Session of a frozen instance to record a
reference (`session` is in scope; every executed action is recorded). The released orbits and slotted_post_ring
references were recorded with them through `tools/references/{orbits,slotted_post_ring}/record_reference.sh`, which
puts `source/` on PYTHONPATH (the oracles live in `topogym.planning`, which the image does not carry) and sets
`TOPOGYM_ROOT` to the repository root.

| file | family / settings | oracle |
|---|---|---|
| `orbits_session_oracle.py` | orbits (very_easy, easier) | `topogym.planning.orbits.native_session.run_feedback_policy` |
| `slotted_below_easy_session_oracle.py` | slotted_post_ring very_easy, easier | `topogym.planning.slotted_post_ring.public_session.run_feedback_policy` |
| `slotted_chain_session_oracle.py` + `run_slotted_chain_oracle.py` | slotted_post_ring easy / medium / hard chain settings | `topogym.planning.slotted_post_ring.chain` (stages M1-M4, settle, goal hold) |
| `lock_picking_session_oracle.py` | lock_picking (chain lock) | `topogym.planning.lock_picking.public_session.run_feedback_policy` |

Example, from the repository root in the release image's Python (or an environment with its packages):

    tools/references/orbits/record_reference.sh orbits orbits_very_easy width \
        tools/references/oracle_runners/orbits_session_oracle.py <out dir> <work dir>

Diagnostics go to `$ORBITS_ORACLE_OUTPUT`, `$SLOTTED_ORACLE_OUTPUT` or `$LOCK_ORACLE_OUTPUT`.
These files come from the development history (commit 5cb4114f10), with only their paths updated to this repository.
