#!/usr/bin/env bash
# Get every image the tasks pin, and check each one against its pin (images/README.md).
#
#   bash images/pull_all.sh [--accept-licences] [--pins <pins.json>]   pull what is not local yet, then check all
#   bash images/pull_all.sh --check [--json] [--pins <pins.json>]      check the local images; pulls nothing
#   bash images/pull_all.sh --list [--pins <pins.json>]                print "<ref> <id>" per image; needs no Docker
#
# images/pins.json lists the images in pull order (a base before the image built FROM it), each with the reference the
# tasks pin (<repo>:<tag>@sha256:<manifest digest>) and its image id (the config digest). An image that is local with
# its pin is not pulled again; the others are pulled by digest (`docker pull <ref>`). The packages are public: a pull
# needs no login and no token, and the script never asks for, reads or prints one. When the registry refuses a pull the
# script says what to do (an old login: `docker logout ghcr.io`; packages that are not public yet: `docker login
# ghcr.io` once, with a token that has the read:packages scope).
# The check is right for both of Docker's image stores: `docker image inspect <ref>` must report the pinned id (the
# classic store) or the pinned manifest digest (the containerd store, a new Docker Engine 29 install).
# Pulling needs your acceptance of the third-party licences in LICENSES/THIRD_PARTY.md: --accept-licences, or
# ROBOARC_ACCEPT_LICENCES=1. Nothing to pull, --check and --list need none.
# Exit 0 only when every image is local with its pin; 1 when one is missing or differs or a pull failed; 2 for a usage
# error, unreadable pins or no usable Docker; 3 when a pull is needed and the licences are not accepted.
# The work is done by images/pins.py (python3 >= 3.8, standard library).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
usage() { echo "usage: bash images/pull_all.sh [--accept-licences] [--check [--json]] [--list] [--pins <pins.json>]" >&2; exit 2; }
cmd=pull args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --pins) [ $# -ge 2 ] || usage; args+=(--pins "$2"); shift 2 ;;
    --list) cmd=list; shift ;;
    --check) cmd=check; shift ;;
    --json) args+=(--json); shift ;;
    --accept-licences|--accept-licenses) export ROBOARC_ACCEPT_LICENCES=1; shift ;;
    *) usage ;;
  esac
done
exec python3 "$ROOT/images/pins.py" "$cmd" ${args[@]+"${args[@]}"}
