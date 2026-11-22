#!/usr/bin/env bash
# Build the two IKEA MuJoCo images of RoboARC from this checkout, reproducibly (ikea/images/ikea-mujoco/README.md):
#   ghcr.io/mll-lab-nu/roboarc-ikea:<VERSION>                   privileged tasks and the physics of the standard ones
#   ghcr.io/mll-lab-nu/roboarc-ikea-sim:<sim/VERSION>           the standard tasks' `sim` sidecar (+ Nyx, EGL)
#
#   ikea/images/ikea-mujoco/build.sh [--assets <dir>] [--download <dir>] [--out <dir>] [--no-load]
#
# 1. the scene meshes: --assets <dir> (<scene>/assets/<file> per scene, a local copy), or the private HF dataset pinned
#    in ikea/scripts/ikea_mujoco/assets.json (fetch_image_context.sh; read access needed: `hf auth login` or HF_TOKEN);
# 2. stage_context.py: the scene code of images/ikea-mujoco/scenes/ plus the meshes -> images/ikea-mujoco/context/,
#    every file checked against its sha256 in assets.json, with the EPA patch script and requirements.lock;
# 3. the base image, then the sim image FROM it, each as an OCI archive in --out (default: the current directory),
#    roboarc-ikea-<VERSION>.oci.tar and roboarc-ikea-sim-<sim/VERSION>.oci.tar: the bytes a push copies to the registry;
# 4. the checks: the base archive's manifest digest is the one sim/Dockerfile's FROM line pins (the sim build gets that
#    base as the named context of the pinned reference, so it never pulls it), and, after loading both into Docker
#    (unless --no-load; local tags only), the base image's id is the one every task is bound to
#    (scripts/ikea_mujoco/bindings.json "image" "id").
# Reproducible: a throwaway BuildKit builder of a pinned version, inputs pinned (the Python base image by digest, Debian
# packages by version from the Debian snapshot archive, wheels by sha256, the scenes and the HDRI by sha256), no
# provenance or SBOM attestations, and every timestamp set to SOURCE_DATE_EPOCH (rewrite-timestamp): the same checkout
# gives the same digests and image ids on any host.
# Needs Docker with buildx (the docker-container driver), ~40 GB of free disk, and network access to docker.io (BuildKit,
# the Python base image), snapshot.debian.org, pypi.org and download.pytorch.org (the wheels) and dl.polyhaven.org (the
# HDRI). Running the images needs no network.
set -euo pipefail
I="$(cd "$(dirname "$0")" && pwd)"; R="$(cd "$I/../.." && pwd)"; H="$R/scripts/ikea_mujoco"
ASSETS="" DL="${RCB_IKEA_ASSETS:-$HOME/.cache/ikea-mujoco-assets}" OUT="$PWD" LOAD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --assets) ASSETS="$2"; shift 2 ;;
    --download) DL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --no-load) LOAD=0; shift ;;
    *) echo "usage: $0 [--assets <dir>] [--download <dir>] [--out <dir>] [--no-load]" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
BUILDKIT="docker.io/moby/buildkit:v0.23.2@sha256:ddd1ca44b21eda906e81ab14a3d467fa6c39cd73b9a39df1196210edcb8db59e"
EPOCH=1792396800   # 2026-10-19T08:00:00Z, the day the 1.0.0 images were built (their Debian snapshot's day)
VERSION="$(cat "$I/VERSION")"; SIM_VERSION="$(cat "$I/sim/VERSION")"
BASE="ghcr.io/embodied-first-exam/roboarc-ikea:$VERSION"
SIM="ghcr.io/embodied-first-exam/roboarc-ikea-sim:$SIM_VERSION"
# BASE and SIM: the names the archives record and sim/Dockerfile's FROM pins (an input of the sim image), as the
# 1.1.0 images were built; the registry (images/pins.json) and the local tags of --load are ghcr.io/mll-lab-nu/
REG="ghcr.io/mll-lab-nu"
PIN="$(sed -n 's/^FROM \(\S*\)$/\1/p' "$I/sim/Dockerfile")"
case "$PIN" in "$BASE@sha256:"*) ;; *) echo "sim/Dockerfile is not FROM $BASE@sha256:<digest> ($PIN)" >&2; exit 1 ;; esac
WORK="$(mktemp -d "${TMPDIR:-/tmp}/roboarc-ikea-build.XXXXXX")"
B="roboarc-ikea-build-$$"
trap 'docker buildx rm "$B" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT

if [ -z "$ASSETS" ]; then
  bash "$H/fetch_image_context.sh" "$DL"          # downloads, checks and stages the context
else
  for d in "$I"/scenes/*/; do
    t="$(basename "$d")"; mkdir -p "$WORK/tree/scenes/$t"
    cp -r "$d." "$WORK/tree/scenes/$t/"; cp -r "$ASSETS/$t/assets" "$WORK/tree/scenes/$t/assets"
  done
  python3 "$H/stage_context.py" "$WORK/tree"      # checks every staged file against assets.json
fi

# archive -> "<manifest digest> <config digest>"
digests() { python3 - "$1" <<'PY'
import json, sys, tarfile
t = tarfile.open(sys.argv[1])
(m,) = json.load(t.extractfile("index.json"))["manifests"]
print(m["digest"], json.loads(t.extractfile("blobs/" + m["digest"].replace(":", "/")).read())["config"]["digest"])
PY
}
docker buildx create --name "$B" --driver docker-container --driver-opt "image=$BUILDKIT" --driver-opt memory=8g >/dev/null
build() {   # <ref> <archive> <dockerfile> <context> [extra buildx args]
  local ref="$1" out="$2" file="$3" ctx="$4"; shift 4
  docker buildx build --builder "$B" --platform linux/amd64 --no-cache --provenance=false --sbom=false \
    --build-arg "SOURCE_DATE_EPOCH=$EPOCH" --build-arg "VERSION=${ref##*:}" "$@" -f "$file" \
    --output "type=oci,dest=$out,name=$ref,rewrite-timestamp=true" "$ctx"
}
BASE_TAR="$OUT/roboarc-ikea-$VERSION.oci.tar"; SIM_TAR="$OUT/roboarc-ikea-sim-$SIM_VERSION.oci.tar"
build "$BASE" "$BASE_TAR" "$I/Dockerfile" "$I/context"
read -r BASE_DIGEST BASE_ID < <(digests "$BASE_TAR")
echo "built $BASE_TAR: manifest $BASE_DIGEST, image id $BASE_ID"
[ "$PIN" = "$BASE@$BASE_DIGEST" ] || { echo "the base's digest is not the one sim/Dockerfile pins ($PIN): if the base changed on purpose, pin the new digest there (and in bindings.json)" >&2; exit 1; }
mkdir -p "$WORK/base-oci" && tar -xf "$BASE_TAR" -C "$WORK/base-oci"
build "$SIM" "$SIM_TAR" "$I/sim/Dockerfile" "$I/sim" --build-context "$PIN=oci-layout://$WORK/base-oci@$BASE_DIGEST"
read -r SIM_DIGEST SIM_ID < <(digests "$SIM_TAR")
echo "built $SIM_TAR: manifest $SIM_DIGEST, image id $SIM_ID"
[ $LOAD -eq 1 ] || exit 0

load() {   # docker load (classic image store) wants a manifest.json next to the OCI layout: add one
  python3 - "$1" "$2" <<'PY' | docker load -q >/dev/null
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
  [ "$(docker image inspect --format '{{.Id}}' "$2")" = "$3" ] || { echo "load: $2 is not $3" >&2; exit 1; }
  echo "loaded $2 = $3"
}
load "$BASE_TAR" "$REG/roboarc-ikea:$VERSION" "$BASE_ID"
load "$SIM_TAR" "$REG/roboarc-ikea-sim:$SIM_VERSION" "$SIM_ID"
want=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["image"]["id"])' "$H/bindings.json")
[ "$BASE_ID" = "$want" ] && echo "ok: the base image is the one every task is bound to" \
  || { echo "the base image $BASE_ID is not the bound image $want: the tasks refuse it" >&2; exit 1; }
