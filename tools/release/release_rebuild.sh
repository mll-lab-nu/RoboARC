#!/bin/bash
# The RoboTangle release rebuild in one command: scripts/robotangle/release_rebuild.py (usage and steps there).
#   scripts/robotangle/release_rebuild.sh <head> --version <v> [options]
# Needs docker (where needed: sg docker -c "..."), harbor and a Python >= 3.11 (mk_tasks.py): $PYTHON, or the first of
# python3.12 / python3.11 / python3 that is one.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
for py in "${PYTHON:-}" python3.12 python3.11 python3; do
  if [ -n "$py" ] && command -v "$py" >/dev/null && "$py" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    exec "$py" "$here/release_rebuild.py" "$@"
  fi
done
echo "release_rebuild.sh: needs Python >= 3.11; set PYTHON" >&2
exit 1
