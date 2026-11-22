# RoboARC's images: pins, pull, check, archives, push

RoboARC's tasks run four images of its own, all in `ghcr.io/mll-lab-nu/`, and the standard client base
`docker.io/library/python:3.12.14-slim@sha256:…` (`scripts/eai_standard/client.env`). Every task names each image by
its manifest digest, `<repo>:<tag>@sha256:<digest>`, so a pull gets exactly the bytes that were validated. The digests
are in one file, [`pins.json`](pins.json), in push order:

| image | recipe | archive made by | used by |
|---|---|---|---|
| `roboarc` | `source/images/roboarc/Dockerfile` (`images/roboarc/`) | `images/archive.sh` from the local build | every puzzle task: both Dockerfiles (privileged), the `sim` sidecar (standard) |
| `roboarc-ikea` | `ikea/images/ikea-mujoco/Dockerfile` | `ikea/images/ikea-mujoco/build.sh` | IKEA privileged tasks (both Dockerfiles); the base of `roboarc-ikea-sim` |
| `roboarc-ikea-sim` | `ikea/images/ikea-mujoco/sim/Dockerfile` (`FROM` the `roboarc-ikea` pin) | `ikea/images/ikea-mujoco/build.sh` | IKEA standard tasks: the `sim` sidecar |
| `roboarc-monitor` | `images/monitor/Dockerfile` | `images/monitor/build.sh` | every task: the `rcb_monitor` sidecar (`scripts/rcb_monitor/inject.py` `IMAGE`) |

Each entry of `pins.json` has the pinned reference (`ref`), the image id (`id`, the config digest) and the file name
of its OCI archive (`archive`). The generators read it
(`scripts/task_conventions.py` `pinned_image`), and `ci_checks/check-image-pins.py` (CI job `image-pins`) fails a task
that names an image by tag only, by another digest or from another registry, and checks that the monitor's `IMAGE` and
the IKEA sim recipe's `FROM` are pins of `pins.json` (the sim recipe, an input of that image, keeps the
registry name it was built under, `ghcr.io/embodied-first-exam/`, with the same tag and digest).

## Get the images

```bash
bash images/pull_all.sh --accept-licences
```

pulls every pinned image by its digest and checks it against its pin. The images contain software and data of others:
`--accept-licences` (or `ROBOARC_ACCEPT_LICENCES=1` in the environment) states that you accept their licences, which
[LICENSES/THIRD_PARTY.md](../LICENSES/THIRD_PARTY.md) lists. Without it the script pulls nothing and says so in one
sentence (exit 3). The first run prints, for each of the four images,

```
pull ghcr.io/mll-lab-nu/roboarc:2.2.0@sha256:96a0a895c0e4934adca39f7a370e67bf82bfb4db7ef049131a3ac0dc9d0f9057
ok   ghcr.io/mll-lab-nu/roboarc:2.2.0@sha256:96a0a895c0e4934adca39f7a370e67bf82bfb4db7ef049131a3ac0dc9d0f9057 = f8fdee211ec5
```

and ends with `pull_all: every pinned image is local with its pinned id` (exit 0).

- **Size.** About 19 GB on disk: `roboarc` 10.0 GB, `roboarc-ikea` 1.05 GB, `roboarc-ikea-sim` 7.67 GB,
  `roboarc-monitor` 0.13 GB. On a terminal Docker shows its progress; in a log there is one `pull` line per image.
- **Again after every `git pull`.** An image that is local with its pin is not pulled again, and a run that has
  nothing to pull asks for no acceptance. A new release downloads only the images it changed.
- **No login.** The packages are public: Docker pulls them anonymously, and the script never asks for, reads or
  prints a token. If the registry refuses a pull all the same (`FAIL … docker pull failed: … unauthorized`, or
  `denied`), Docker holds an old login for `ghcr.io`, which the registry refuses in place of an anonymous pull:
  `docker logout ghcr.io`, and run the script again.
  If the packages are not public yet (before the release), log in once with a GitHub token that has the
  `read:packages` scope: `echo <token> | docker login ghcr.io -u <github user> --password-stdin`.
- **Exit codes.** 0: every image is local with its pin. 1: one is missing or differs, or a pull failed. 2: a usage
  error, unreadable pins, or Docker cannot be used. 3: a pull is needed and the licences are not accepted.

## Check the local images

```bash
bash images/pull_all.sh --check        # pulls nothing and needs no acceptance; --json prints the same for a program
```

| line | meaning |
|---|---|
| `ok   <ref> = <id>` | Docker resolves the pinned reference, to the pinned image |
| `FAIL <ref>: not local` | the image is not there: pull it |
| `FAIL <ref>: local as <repo>:<tag> only, not under the digest the tasks name` | the pinned image is there, loaded from an archive or built, but Docker does not resolve `<repo>:<tag>@sha256:<digest>` for it, so a task would still ask the registry: pull it, or use a local registry ([below](#use-the-images-without-ghcrio)) |
| `FAIL <ref>: image id …, pinned …` | the pinned reference resolves to another image: `pins.json` was edited, or the store is damaged |

The last lines name Docker's image store and the result. Docker has two image stores, and they report different image
ids for the same image:

| image store | `docker info --format '{{.Driver}} {{.DriverStatus}}'` | `docker image inspect --format '{{.Id}}' <ref>` |
|---|---|---|
| classic | `overlay2 [[Backing Filesystem …` | the config digest: the pin's `id` |
| containerd (a new install of Docker Engine 29) | `overlayfs [[driver-type io.containerd.snapshotter.v1]]` | the manifest digest: the digest in the pin's `ref` |

The check accepts the id of the store it runs on (`images/pins.py`, `judge`). A manifest names its config by digest,
so an image with the pinned manifest digest has the pinned id. On the containerd store an `ok` line shows the manifest
digest, as `docker images` does, and adds the pinned id. Read on Docker 28.0.0 (classic) and Docker 29.1.3
(containerd).

## The archives

An image is pushed from an OCI archive whose manifest digest is known before anything is published:

- `images/monitor/build.sh` and `ikea/images/ikea-mujoco/build.sh` build their images reproducibly straight into OCI
  archives (BuildKit, fixed timestamps).
- The `roboarc` image is built into the local Docker store (`images/roboarc/build.sh`, or `tools/release/release_rebuild.py`)
  and validated there. `images/archive.sh` then exports that image without rebuilding it:

  ```bash
  images/archive.sh ghcr.io/mll-lab-nu/roboarc:1.1.0 roboarc-1.1.0.oci.tar
  ```

  `docker save`, then `images/oci_archive.py` in the pinned client base image (no network): the image's config is kept
  byte for byte, so the archive's config digest is the local image id; every layer is checked against its diff id and
  gzip-compressed with fixed settings; the archive's tar headers are fixed. The same image gives the same archive and
  the same manifest digest every time. It prints both digests; put them into `pins.json`. It takes the image id from
  `docker image inspect` and so needs the classic image store, as the `--load` step of the two build scripts does.

### Check an archive against its pin

```bash
python3 images/pins.py archives --archives <dir>      # no Docker, no network; --json for a program
```

reads the four archives named in `pins.json` from `<dir>`, each in full: `index.json` names exactly one manifest, its
digest is the pinned one, its config is the pinned image id, and every blob it names is in the archive under the sha256
of its bytes. It prints `ok   <archive> = <ref>` per archive and `archives: every archive is its pinned image`
(exit 0), or `FAIL <archive>: <what differs>` (exit 1). The four archives of this release are 10.3 GB and take about
20 s on an NVMe disk. `images/push_all.sh` makes the same check before it pushes.

### Use the images without ghcr.io

Tasks name `ghcr.io/mll-lab-nu/<image>:<tag>@sha256:<digest>`. Which of the two ways below works depends on the image
store (the table above).

**A local registry (either store).** The classic store resolves `<repo>@sha256:<digest>` only for an image it pulled
from a registry: a `docker load` gives the image its tag and its pinned id, and `--check` then reports
`local as <repo>:<tag> only`. (That store also loads only `roboarc-<version>.oci.tar`: the other three archives are
plain OCI layouts, without the `manifest.json` it needs.) So serve the archives from a registry of your own and pull
from it:

```bash
docker run -d --name roboarc-registry -p 127.0.0.1:5000:5000 registry:2
echo unused | GHCR_USER=unused images/push_all.sh --archives <dir> --registry 127.0.0.1:5000
sed 's#ghcr.io/mll-lab-nu/#127.0.0.1:5000/mll-lab-nu/#' images/pins.json > pins.local.json
bash images/pull_all.sh --accept-licences --pins pins.local.json
```

`push_all.sh` checks the archives against the pins and copies them byte for byte, so the registry serves the pinned
digests (this registry has no authentication: the user name and the word on stdin are placeholders). The images are
then local as `127.0.0.1:5000/mll-lab-nu/<image>@sha256:<digest>`, with the pinned digests and ids. The tasks name
`ghcr.io`, so run them from a copy with the registry rewritten:

```bash
cp -a tasks tasks.local
grep -rl 'ghcr.io/mll-lab-nu/' tasks.local | xargs sed -i 's#ghcr.io/mll-lab-nu/#127.0.0.1:5000/mll-lab-nu/#g'
```

The copies run the same image bytes, but their image lines, and so their `DIGEST`s, are not the release's: say so
with any result from them.

**`docker load` (the containerd store only).** The containerd store keeps a loaded archive's manifest, so the digest
reference resolves once the image has the pinned repository name. An archive is loaded under the name it records:
`roboarc-2.2.0.oci.tar` under its pinned name, the other three under `ghcr.io/embodied-first-exam/<image>:<tag>`, the
name they were built under. Give those the pinned name too:

```bash
docker load -i <dir>/roboarc-2.2.0.oci.tar
docker load -i <dir>/roboarc-monitor-0.1.0.oci.tar
docker tag ghcr.io/embodied-first-exam/roboarc-monitor:0.1.0 ghcr.io/mll-lab-nu/roboarc-monitor:0.1.0
# the same two lines for roboarc-ikea 1.1.0 and roboarc-ikea-sim 1.1.0
bash images/pull_all.sh --check
```

`--check` must print `ok` for each image. This way was read on Docker 29.1.3 with another image from an archive of
the same format (the digest reference resolves, also under a second name given with `docker tag`), not yet with these
four archives.

## Push

```bash
<token source> | GHCR_USER=<github user> images/push_all.sh --archives <dir> [--log <dir>]
```

pushes the four archives in `<dir>` (named as in `pins.json`) in the order of `pins.json`, `roboarc-ikea` before
`roboarc-ikea-sim`:

1. Before logging in it reads every archive in full: one manifest, whose digest is the pin; its config digest is the
   pinned image id; every blob is there and its sha256 is its name. Any difference: nothing is pushed.
2. In one skopeo container (pinned by digest) it logs in with the token from stdin (the login file lives only in the
   container's tmpfs), copies each archive with `--preserve-digests` (the registry stores the archive's manifest bytes,
   so the pushed digest is the pinned one; a `docker push` would re-compress the layers and give other digests), then
   fetches the registry's manifest by digest and by tag and the config, and hashes them: all must be the pin's. Nothing
   is written to `~/.docker`; the token is never printed.
3. It reports whether each digest can be pulled anonymously: a new package starts private, and the tasks need it public.

Exit 0 only when every image was pushed and checked; `<dir>/push.log` (default: the current directory) has the record.
`--registry <host:port>` pushes to another registry under the same repository paths, e.g. a local registry
(`registry:2` on 127.0.0.1, no TLS), which is also how pinned tasks run before the push
([above](#use-the-images-without-ghcrio)).

## Changing an image

Build or archive it, put its `ref` and `id` into `pins.json` (and the monitor's into `inject.py` `IMAGE`, the IKEA
base's into `sim/Dockerfile`), re-render every task (both generators; docs/DEVELOPING.md), validate, then push the
archives before the tasks are published.
