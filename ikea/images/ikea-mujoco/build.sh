#!/usr/bin/env bash
# Build the two IKEA MuJoCo images of RoboARC from this checkout (ikea/images/ikea-mujoco/README.md):
#   ghcr.io/embodied-first-exam/roboarc-ikea:<VERSION>       privileged tasks and the physics of the standard ones
#   ghcr.io/embodied-first-exam/roboarc-ikea-sim:<sim/VERSION>  the standard tasks' `sim` sidecar (+ Nyx, EGL, ffmpeg)
#
#   ikea/images/ikea-mujoco/build.sh [--assets <dir>] [--download <dir>]
#
# 1. the scene meshes: --assets <dir> (<task>/assets/<file> per task, a local copy), or the private HF dataset pinned in
#    ikea/scripts/ikea_mujoco/assets.json (fetch_image_context.sh; read access needed: `hf auth login` or HF_TOKEN);
# 2. stage_context.py: the scene code of images/ikea-mujoco/scenes/ plus the meshes -> images/ikea-mujoco/context/,
#    every file checked against its sha256 in assets.json;
# 3. docker build of the base image, then of the sim image FROM it;
# 4. the base image's id must be the one every task is bound to (scripts/ikea_mujoco/bindings.json "image" "id").
# Both builds pip-install pinned wheels and the sim build apt-installs graphics libraries, so a build without a warm
# cache on another host gives the same files but may give another image id for the sim image (the base image's files are
# pinned: wheels by version, the scenes by sha256, libmujoco.so by sha256).
set -euo pipefail
I="$(cd "$(dirname "$0")" && pwd)"; R="$(cd "$I/../.." && pwd)"; H="$R/scripts/ikea_mujoco"
ASSETS="" DL="${RCB_IKEA_ASSETS:-$HOME/.cache/ikea-mujoco-assets}"
while [ $# -gt 0 ]; do
  case "$1" in
    --assets) ASSETS="$2"; shift 2 ;;
    --download) DL="$2"; shift 2 ;;
    *) echo "usage: $0 [--assets <dir>] [--download <dir>]" >&2; exit 2 ;;
  esac
done
if [ -z "$ASSETS" ]; then
  bash "$H/fetch_image_context.sh" "$DL"          # downloads, checks and stages the context
else
  TREE="$(mktemp -d)"; trap 'rm -rf "$TREE"' EXIT
  for d in "$I"/scenes/*/; do
    t="$(basename "$d")"; mkdir -p "$TREE/scenes/$t"
    cp -r "$d." "$TREE/scenes/$t/"; cp -r "$ASSETS/$t/assets" "$TREE/scenes/$t/assets"
  done
  python3 "$H/stage_context.py" "$TREE"           # checks every staged file against assets.json
fi
BASE="ghcr.io/embodied-first-exam/roboarc-ikea:$(cat "$I/VERSION")"
SIM="ghcr.io/embodied-first-exam/roboarc-ikea-sim:$(cat "$I/sim/VERSION")"
grep -q "^FROM $BASE\$" "$I/sim/Dockerfile" || { echo "sim/Dockerfile is not FROM $BASE" >&2; exit 1; }
docker build -t "$BASE" "$I/context" -f "$I/Dockerfile"
docker build -t "$SIM" "$I/sim"
want=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["image"]["id"])' "$H/bindings.json")
have=$(docker image inspect --format '{{.Id}}' "$BASE")
echo "built $BASE = $have"
echo "built $SIM = $(docker image inspect --format '{{.Id}}' "$SIM")"
[ "$have" = "$want" ] && echo "ok: the base image is the one every task is bound to" \
  || { echo "the base image $have is not the bound image $want: the tasks refuse it" >&2; exit 1; }
