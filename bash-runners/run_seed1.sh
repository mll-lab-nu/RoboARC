#!/usr/bin/env bash
# The evaluation run's seed-1 trials: every task's trial at appearance seed 1, in both modes (docs/COLLABORATOR_SETUP.md,
# section 5; the settings: bash-runners/start_lanes.sh). Start it once run_seed0.sh's lanes have ended, with the same
# settings and batch name; then run_seed2.sh.
#   MODEL=openai/<model> EFFORT=high RESULTS=/path/outside/the/repo bash-runners/run_seed1.sh <batch name> [<base> ...]
SEEDS=1 exec "$(dirname "$0")/start_lanes.sh" "$@"
