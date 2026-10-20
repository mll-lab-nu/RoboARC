#!/usr/bin/env bash
# Build the RoboARC release image from this checkout (images/roboarc/README.md, Build).
#
#   images/roboarc/build.sh [--instances <dir>] [--download <dir>] [--tag <ref>] [--noinst]
#
# The recipe is source/images/roboarc/Dockerfile, built from source/ (its .dockerignore keeps the context to the paths
# the recipe copies), with the frozen instances as the named build context `instances`. This is the build step of
# tools/release/release_rebuild.py:
#   1. the frozen instances: --instances <dir> (one directory per base, instance.json + instance_assets/), or, by default,
#      the private HF dataset pinned in scripts/roboarc/assets.json, downloaded into --download <dir> (default
#      $RCB_ROBOTANGLE_ASSETS or ~/.cache/robotangle-assets). Either way the context must hold exactly the pinned bases
#      and files, each with its sha256 (scripts/roboarc/assets.py check-instances);
#   2. `docker buildx build` of the recipe: the Python base image pinned by digest, Debian packages at exact versions from
#      the Debian snapshot archive, wheels by sha256 (source/images/roboarc/runtime.lock, vbd-requirements.txt);
#   3. the installed-runtime check: the image must install exactly the runtime every task is bound to
#      (scripts/roboarc/release.json `runtime`), or the tag is removed again.
# --noinst builds the image without instances (release_rebuild's `-noinst` image, used to make new instances).
# Needs Docker with BuildKit (the default `docker` driver), ~25 GB of free disk, network access to Docker Hub (the
# Python base image), snapshot.debian.org, pypi.org / download.pytorch.org (the wheels) and dl.polyhaven.org (the Nyx
# studio HDRI and wood maps, sha256-checked), and (for the default instance source) read access to the HF dataset
# (`hf auth login` or HF_TOKEN; never printed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VERSION="$(cat "$ROOT/images/roboarc/VERSION")"
TAG="ghcr.io/embodied-first-exam/roboarc:$VERSION"
INST="" DL="${RCB_ROBOTANGLE_ASSETS:-$HOME/.cache/robotangle-assets}" NOINST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --instances) INST="$2"; shift 2 ;;
    --download) DL="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --noinst) NOINST=1; shift ;;
    *) echo "usage: $0 [--instances <dir>] [--download <dir>] [--tag <ref>] [--noinst]" >&2; exit 2 ;;
  esac
done
WORK="$(mktemp -d "${TMPDIR:-/tmp}/roboarc-build.XXXXXX")"; trap 'rm -rf "$WORK"' EXIT
if [ $NOINST -eq 1 ]; then
  INST="$WORK/empty"; mkdir -p "$INST"; TAG="${TAG}-noinst"
elif [ -z "$INST" ]; then
  python3 "$ROOT/scripts/roboarc/assets.py" fetch "$DL"
  INST="$WORK/instances"; mkdir -p "$INST"
  python3 - "$ROOT/scripts/roboarc/assets.json" "$DL/instances" "$INST" <<'PY'
import json, os, pathlib, sys
pins, src, out = json.load(open(sys.argv[1])), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
for base in sorted(pins["instances"]):         # this release's bases only, as symlinks into the checked download
    os.symlink((src / base).resolve(), out / base)
print(f"instances: {len(pins['instances'])} bases from the pinned dataset")
PY
  # a named build context does not follow symlinked directories: copy them (hard links where the filesystem allows)
  mkdir -p "$WORK/ctx"
  for d in "$INST"/*; do cp -al "$(readlink -f "$d")" "$WORK/ctx/$(basename "$d")" 2>/dev/null \
    || cp -a "$(readlink -f "$d")" "$WORK/ctx/$(basename "$d")"; done
  INST="$WORK/ctx"
fi
INST="$(cd "$INST" && pwd)"   # the build runs from source/: an absolute path for the named context
[ $NOINST -eq 1 ] || python3 "$ROOT/scripts/roboarc/assets.py" check-instances "$INST" \
  || { echo "the instances are not the ones pinned in scripts/roboarc/assets.json: nothing built" >&2; exit 1; }
n=$(find "$INST" -mindepth 2 -maxdepth 2 -name instance.json | wc -l)
echo "build $TAG from source/ with $n instances"
(cd "$ROOT/source" && docker buildx build --load -f images/roboarc/Dockerfile --build-arg "RELEASE_VERSION=$VERSION" \
  --build-context "instances=$INST" -t "$TAG" .)
# the runtime check (images/roboarc/README.md, Check a build)
docker run --rm --network none "$TAG" python -c 'import json; from topogym.simulator.frozen_binding import installed_runtime; print(json.dumps(installed_runtime(), sort_keys=True))' \
  | tail -n 1 > "$WORK/runtime.json"
python3 - "$WORK/runtime.json" "$ROOT/scripts/roboarc/release.json" <<'PY' || { docker rmi "$TAG" >/dev/null; exit 1; }
import json, sys
got, want = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))["runtime"]
got = got.get("runtime", got)
if got != want:
    sys.exit("the image's installed runtime differs from scripts/roboarc/release.json: tag removed, do not run it")
print("ok: the image installs the runtime every task is bound to")
PY
echo "built $TAG = $(docker image inspect --format '{{.Id}}' "$TAG")"
