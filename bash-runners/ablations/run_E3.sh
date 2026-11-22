#!/usr/bin/env bash
# Ablation experiment E3 (docs/ABLATIONS.md): the information-access ablation on deformable tasks; its trials are
# listed by `python3 scripts/ablations/experiments.py plan E3`. Starts only after bash-runners/ablations/check_variants.sh E3
# passed. Settings and environment: bash-runners/ablations/run_experiment.sh.
#   MODEL=gpt-6-astra RESULTS=/path/outside/the/repo bash-runners/ablations/run_E3.sh
exec "$(dirname "$0")/run_experiment.sh" E3 "$@"
