#!/usr/bin/env bash
# Export an image of the local Docker store into a deterministic OCI archive, the bytes images/push_all.sh copies to
# the registry (images/README.md).
#
#   images/archive.sh <local ref> <out .oci.tar> [<registry ref>]
#
# e.g. images/archive.sh ghcr.io/embodied-first-exam/roboarc:1.1.0 roboarc-1.1.0.oci.tar
# For an image built into the Docker store (images/roboarc/build.sh); the monitor and IKEA builds write their OCI
# archives themselves. `docker save`, then images/oci_archive.py in the pinned client base image (scripts/eai_standard/
# client.env, no network): the config bytes are kept (config digest = the local image id), every layer is checked
# against its diff id and gzip-compressed with fixed settings, and the archive is written with fixed tar headers. The
# same image always gives the same archive and so the same manifest digest: that digest is the one the tasks pin
# (images/pins.json). <registry ref> (default <local ref>) is the name recorded in the archive.
# Needs free disk next to <out> for the saved image and the compressed layers (about 2x the image size).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ $# -ge 2 ] && [ $# -le 3 ] || { echo "usage: images/archive.sh <local ref> <out .oci.tar> [<registry ref>]" >&2; exit 2; }
REF="$1"; OUT="$(realpath -m "$2")"; NAME="${3:-$1}"
PY="$(sed -n 's/^CLIENT_IMAGE=//p' "$ROOT/scripts/eai_standard/client.env")"
ID="$(docker image inspect --format '{{.Id}}' "$REF")"
mkdir -p "$(dirname "$OUT")"
W="$(mktemp -d "$(dirname "$OUT")/.archive.XXXXXX")"; trap 'rm -rf "$W"' EXIT
echo "save $REF ($ID)" >&2
docker save "$REF" -o "$W/save.tar"
docker run --rm --network none --memory 2g --label roboarc-archive=1 ${ARCHIVE_DOCKER_LABEL:+--label "$ARCHIVE_DOCKER_LABEL"} \
  --user "$(id -u):$(id -g)" -v "$ROOT/images/oci_archive.py:/oci_archive.py:ro" -v "$W:/w" \
  -v "$(dirname "$OUT"):/out" "$PY" python3 -I /oci_archive.py /w/save.tar "/out/$(basename "$OUT")" --id "$ID" --name "$NAME"
