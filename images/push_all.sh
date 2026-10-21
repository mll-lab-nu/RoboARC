#!/usr/bin/env bash
# Push RoboARC's images to their registry byte for byte, credential-free (images/README.md, Push).
#
#   <token source> | GHCR_USER=<github user> images/push_all.sh --archives <dir> [--log <dir>] [--registry <host:port>]
#
# images/pins.json lists the images in push order (a base before the image built FROM it), each with the reference the
# tasks pin (<repo>:<tag>@sha256:<manifest digest>), its image id (the config digest) and its archive's file name in
# <dir> (the OCI archives of images/archive.sh, images/monitor/build.sh, ikea/images/ikea-mujoco/build.sh).
# 1. Before logging in, every archive is read in full: exactly one manifest, whose digest is the pin's; its config
#    digest is the pinned image id; every blob it names is there and its sha256 is its name. Any difference: nothing is
#    pushed (exit 1).
# 2. One skopeo container (pinned by digest): `skopeo login --password-stdin` reads the registry token (write:packages)
#    from STDIN; the login file lives only in the container's tmpfs and is gone when the container exits. Per image:
#    `skopeo copy --preserve-digests oci-archive:<archive> docker://<repo>:<tag>` (the registry stores the archive's
#    manifest bytes unchanged, so the pushed digest is the pinned one), then the registry's manifest is fetched by
#    digest and by tag and hashed, and its config hashed: all three must be the pin's. Logout. Nothing is written to
#    ~/.docker; the token is never printed.
# 3. An anonymous fetch of each digest is reported (a new package starts private; the tasks need it public).
# --registry <host:port>: push to that registry instead of ghcr.io, under the same repository paths, e.g. a local test
#   registry (localhost / 127.0.0.1 are used without TLS).
# Log: <dir>/push.log (default: the current directory), skopeo's output in push_raw.log. Exit 0 only when every image was
# pushed and checked. Needs docker and python3 on the host, and free disk in the log dir for the largest archive
# (skopeo unpacks an archive there while it copies it; removed afterwards). PUSH_DOCKER_LABEL=<k=v> adds a label to the
# push container (e.g. to tell it apart on a shared host).
set -u
SKOPEO="quay.io/skopeo/stable:v1.20.0@sha256:47853bb9fb24202af9110531ebd6e43c5f97701254ca290596640290d17942f4"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PINS="$ROOT/images/pins.json"
A="" LOGDIR="." REG=""
usage() { echo "usage: images/push_all.sh --archives <dir> [--log <dir>] [--registry <host:port>]" >&2; cat > /dev/null; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --archives) A="$2"; shift 2 ;;
    --log) LOGDIR="$2"; shift 2 ;;
    --registry) REG="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$A" ] && [ -d "$A" ] || usage
: "${GHCR_USER:?set GHCR_USER to the GitHub user that owns the token}"
A="$(realpath "$A")"; mkdir -p "$LOGDIR"; LOGDIR="$(realpath "$LOGDIR")"; L="$LOGDIR/push.log"
echo "$(date -Is) push_all: $(sha256sum "$PINS" | cut -c1-16) images/pins.json, archives in $A${REG:+, registry $REG}" >> "$L"

# 1. the archives against the pins (no network, no token); prints "<archive> <repo>:<tag> <digest> <id>" per image
LIST="$(python3 - "$PINS" "$A" "$REG" <<'PY'
import hashlib, json, sys, tarfile
pins, d, reg = json.load(open(sys.argv[1]))["images"], sys.argv[2], sys.argv[3]
bad = []
for p in pins:
    name, digest = p["ref"].split("@")
    path = f"{d}/{p['archive']}"
    try:
        blobs, index = {}, None
        with tarfile.open(path, "r|") as t:          # one pass over the whole archive
            for m in t:
                if not m.isfile():
                    continue
                f, h, small = t.extractfile(m), hashlib.sha256(), b""
                while chunk := f.read(1 << 22):
                    h.update(chunk)
                    if m.size <= 1 << 20:
                        small += chunk
                if m.name.lstrip("./") == "index.json":
                    index = json.loads(small)
                elif m.name.lstrip("./").startswith("blobs/sha256/"):
                    blobs[m.name.rsplit("/", 1)[1]] = (h.hexdigest(), small)
        (man,) = index["manifests"]
        wrong = [n for n, (h, _) in blobs.items() if h != n]
        manifest = json.loads(blobs[man["digest"][7:]][1])
        need = [manifest["config"]["digest"]] + [l["digest"] for l in manifest["layers"]]
        missing = [x for x in need if x[7:] not in blobs]
        if man["digest"] != digest:
            bad.append(f"{path}: manifest {man['digest']} is not the pin {digest}")
        elif manifest["config"]["digest"] != p["id"]:
            bad.append(f"{path}: config {manifest['config']['digest']} is not the pinned image id {p['id']}")
        elif wrong or missing:
            bad.append(f"{path}: blobs whose sha256 is not their name {wrong}, missing {missing}")
    except Exception as e:  # noqa: BLE001  (a missing or unreadable archive)
        bad.append(f"{path}: {type(e).__name__}: {e}")
        continue
    if reg:
        name = reg + name[name.index("/"):]
    print(p["archive"], name, digest, p["id"])
for b in bad:
    print("BAD", b)
sys.exit(1 if bad else 0)
PY
)"; ok=$?
echo "$LIST" | grep '^BAD' | tee -a "$L" >&2
if [ $ok -ne 0 ]; then echo "$(date -Is) an archive is not its pin (see above): nothing pushed" | tee -a "$L" >&2; cat > /dev/null; exit 1; fi
echo "$LIST" | sed 's/^/  verified /' >> "$L"
REGISTRY="${REG:-ghcr.io}"
TLS=""; case "$REGISTRY" in localhost:*|127.0.0.1:*) TLS="--tls-verify=false" ;; esac

# 2. login, copy, check: one container; the token arrives on its stdin (ours)
W="$(mktemp -d "$LOGDIR/.push.XXXXXX")"; trap 'rm -rf "$W"' EXIT
out=$(docker run --rm -i --network host --memory 2g --label roboarc-push=1 ${PUSH_DOCKER_LABEL:+--label "$PUSH_DOCKER_LABEL"} \
  --user "$(id -u):$(id -g)" -e HOME=/tmp --tmpfs "/auth:mode=0700,uid=$(id -u),gid=$(id -g)" \
  -e "U=$GHCR_USER" -e "REG=$REGISTRY" -e "TLS=$TLS" -e "LIST=$LIST" \
  -v "$A:/in:ro" -v "$W:/work" --entrypoint sh "$SKOPEO" -c '
    set -u; AF=/auth/auth.json
    skopeo login $TLS --authfile "$AF" -u "$U" --password-stdin "$REG" >/dev/null 2>&1 || { echo "result=login-failed"; exit 1; }
    echo "$LIST" | while read -r archive name digest id; do
      repo="${name%:*}"
      if skopeo inspect $TLS --raw --authfile "$AF" "docker://$name" > /work/before 2>/dev/null; then
        before=sha256:$(sha256sum < /work/before | cut -d" " -f1); else before=none; fi
      if skopeo --tmpdir /work copy $TLS --preserve-digests --authfile "$AF" "oci-archive:/in/$archive" "docker://$name" >&2; then
        r=$(skopeo inspect $TLS --raw --authfile "$AF" "docker://$repo@$digest" | sha256sum | cut -d" " -f1)
        t=$(skopeo inspect $TLS --raw --authfile "$AF" "docker://$name" | sha256sum | cut -d" " -f1)
        c=$(skopeo inspect $TLS --config --raw --authfile "$AF" "docker://$repo@$digest" | sha256sum | cut -d" " -f1)
        if skopeo inspect $TLS --raw --no-creds "docker://$repo@$digest" >/dev/null 2>&1; then anon=yes; else anon=no; fi
        echo "$name remote=sha256:$r remote_tag=sha256:$t config=sha256:$c anonymous=$anon tag_before=$before"
      else
        echo "$name copy-failed"
      fi
    done
    skopeo logout --authfile "$AF" "$REG" >/dev/null 2>&1; true
  ' 2>>"$LOGDIR/push_raw.log")
echo "$out" | sed 's/^/  /' >> "$L"
rc=0
field() { echo "$row" | sed -n "s/.* $1=\([^ ]*\).*/\1/p"; }
while read -r archive name digest id; do
  row=$(echo "$out" | grep -F -- "$name remote=" | head -1)
  if [ -n "$row" ] && [ "$(field remote)" = "$digest" ] && [ "$(field remote_tag)" = "$digest" ] && [ "$(field config)" = "$id" ]; then
    echo "  OK $name@$digest" | tee -a "$L"
    case "$row" in *anonymous=yes*) ;; *) echo "  anonymous pull: no ($name is private; the tasks need it public)" | tee -a "$L" ;; esac
  else
    echo "  FAILED $name@$digest (${row:-no result})" | tee -a "$L"; rc=1
  fi
done <<< "$LIST"
case "$out" in *result=login-failed*) echo "  login failed: nothing pushed" | tee -a "$L"; rc=1 ;; esac
echo "$(date -Is) push_all done rc=$rc" >> "$L"
exit $rc
