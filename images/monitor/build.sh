#!/usr/bin/env bash
# Build the monitor sidecar image reproducibly, as an OCI archive whose manifest digest is the one the tasks pin.
#
#   images/monitor/build.sh [--out <file>] [--load]
#
# - Output: an OCI image layout archive (default ./roboarc-monitor-<VERSION>.oci.tar), the exact bytes images/monitor/
#   push.sh copies to the registry. Its manifest digest is printed and compared with the pin in
#   scripts/rcb_monitor/inject.py (`IMAGE`): exit 1 when they differ.
# - Reproducible: a throwaway BuildKit builder of a pinned version, the base image pinned by digest, no provenance or
#   SBOM attestations, and every timestamp set to SOURCE_DATE_EPOCH below (rewrite-timestamp), so the same recipe gives
#   the same digest on any host.
# - --load also loads the archive into the local Docker image store as ghcr.io/embodied-first-exam/roboarc-monitor:<VERSION>
#   (a local tag only: Docker resolves the tasks' `<ref>@<digest>` pin only after a pull from a registry; README.md).
# Needs Docker with buildx (the docker-container driver) and network access to docker.io (BuildKit, the base image).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VERSION="$(cat "$ROOT/images/monitor/VERSION")"
REF="ghcr.io/embodied-first-exam/roboarc-monitor:$VERSION"
BUILDKIT="docker.io/moby/buildkit:v0.23.2@sha256:ddd1ca44b21eda906e81ab14a3d467fa6c39cd73b9a39df1196210edcb8db59e"
EPOCH=1791625780   # 2026-10-10T09:49:40Z: the build time of the first monitor image (the mtime of its /var/rcb-monitor)
OUT="$PWD/roboarc-monitor-$VERSION.oci.tar" LOAD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$(realpath -m "$2")"; shift 2 ;;
    --load) LOAD=1; shift ;;
    *) echo "usage: $0 [--out <file>] [--load]" >&2; exit 2 ;;
  esac
done
B="roboarc-monitor-build-$$"
trap 'docker buildx rm "$B" >/dev/null 2>&1 || true' EXIT
docker buildx create --name "$B" --driver docker-container --driver-opt "image=$BUILDKIT" --driver-opt memory=2g >/dev/null
docker buildx build --builder "$B" --platform linux/amd64 --no-cache --provenance=false --sbom=false \
  --build-arg "VERSION=$VERSION" --build-arg "SOURCE_DATE_EPOCH=$EPOCH" \
  --output "type=oci,dest=$OUT,name=$REF,rewrite-timestamp=true" "$ROOT/images/monitor"
read -r DIGEST CONFIG < <(python3 - "$OUT" <<'PY'
import json, sys, tarfile
t = tarfile.open(sys.argv[1])
(m,) = json.load(t.extractfile("index.json"))["manifests"]
blob = t.extractfile("blobs/" + m["digest"].replace(":", "/")).read()
print(m["digest"], json.loads(blob)["config"]["digest"])
PY
)
echo "built $OUT"
echo "  manifest digest $DIGEST (the registry's digest after push.sh)"
echo "  config digest   $CONFIG (the Docker image id)"
if [ $LOAD -eq 1 ]; then   # docker load (classic image store) wants a manifest.json next to the OCI layout: add one
  python3 - "$OUT" "$REF" <<'PY' | docker load -q >/dev/null
import io, json, sys, tarfile
t, blob = tarfile.open(sys.argv[1]), lambda d: "blobs/" + d.replace(":", "/")
(m,) = json.load(t.extractfile("index.json"))["manifests"]
man = json.loads(t.extractfile(blob(m["digest"])).read())
data = json.dumps([{"Config": blob(man["config"]["digest"]), "RepoTags": [sys.argv[2]],
                    "Layers": [blob(l["digest"]) for l in man["layers"]]}]).encode()
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as o:
    for mem in t:
        o.addfile(mem, t.extractfile(mem) if mem.isfile() else None)
    info = tarfile.TarInfo("manifest.json"); info.size = len(data); o.addfile(info, io.BytesIO(data))
PY
  [ "$(docker image inspect --format '{{.Id}}' "$REF")" = "$CONFIG" ] || { echo "load: $REF is not $CONFIG" >&2; exit 1; }
  echo "  loaded as $REF"
fi
PIN="$(python3 -c 'import re,sys; print(re.search(r"^IMAGE = \"([^\"]+)\"", open(sys.argv[1]).read(), re.M).group(1))' \
  "$ROOT/scripts/rcb_monitor/inject.py")"
if [ "$PIN" = "$REF@$DIGEST" ]; then echo "ok: the tasks' pin $PIN"
else echo "differs from the tasks' pin $PIN (scripts/rcb_monitor/inject.py IMAGE)" >&2; exit 1; fi
