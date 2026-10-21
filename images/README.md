# RoboARC's images: pins, archives, push

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

Each entry of `pins.json` has the pinned reference (`ref`), the image id (`id`, the config digest: what `docker image
inspect` shows after a pull) and the file name of its OCI archive (`archive`). The generators read it
(`scripts/task_conventions.py` `pinned_image`), and `ci_checks/check-image-pins.py` (CI job `image-pins`) fails a task
that names an image by tag only, by another digest or from another registry, and checks that the monitor's `IMAGE` and
the IKEA sim recipe's `FROM` are pins of `pins.json` (the sim recipe, an input of that image, keeps the
registry name it was built under, `ghcr.io/embodied-first-exam/`, with the same tag and digest).

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
  the same manifest digest every time. It prints both digests; put them into `pins.json`.

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
`--registry <host:port>` pushes to another registry under the same repository paths, e.g. a local test registry
(`registry:2` on 127.0.0.1, no TLS), which is also how pinned tasks run before the push: Docker resolves
`<name>@sha256:<digest>` only for an image pulled from a registry, so a local test pulls the archives from such a
registry (the test copies of the tasks name it instead of `ghcr.io`).

## Changing an image

Build or archive it, put its `ref` and `id` into `pins.json` (and the monitor's into `inject.py` `IMAGE`, the IKEA
base's into `sim/Dockerfile`), re-render every task (both generators; docs/DEVELOPING.md), validate, then push the
archives before the tasks are published.
