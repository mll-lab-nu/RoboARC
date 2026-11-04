#!/usr/bin/env bash
# The evaluation run's seed-0 trials: every task's trial at appearance seed 0, in both modes (docs/COLLABORATOR_SETUP.md,
# section 5; the settings: bash-runners/start_lanes.sh). Then run_seed1.sh and run_seed2.sh, with the same settings and
# batch name.
#   MODEL=openai/<model> EFFORT=high RESULTS=/path/outside/the/repo bash-runners/run_seed0.sh <batch name> [<base> ...]
SEEDS=0 exec "$(dirname "$0")/start_lanes.sh" "$@"
