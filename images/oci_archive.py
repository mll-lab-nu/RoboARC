#!/usr/bin/env python3
"""Turn a `docker save` of one local image into a deterministic OCI archive (images/archive.sh runs it).

usage: oci_archive.py <docker save tar> <out .oci.tar> --id <sha256:image id> --name <registry/repo:tag>

The image's config bytes are kept unchanged, so the archive's config digest is the local image id. Each layer is
checked against the config's diff id, then gzip-compressed with fixed settings (level 6, no file name, mtime 0). The
OCI manifest, index.json, a manifest.json for `docker load`, and the archive's tar headers (sorted, mtime 0, owner 0)
are written the same way every time, so the same image gives the same archive bytes and the same manifest digest.
Prints a JSON summary: the manifest digest, the config digest, the layers (diff id, digest, size), the archive's sha256.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile

CHUNK = 1 << 22
MT_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
MT_INDEX = "application/vnd.oci.image.index.v1+json"
MT_CONFIG = "application/vnd.oci.image.config.v1+json"
MT_LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"


def dumps(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()


class Hashing(io.RawIOBase):
    """A write-only file that hashes and counts what passes through it."""

    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def writable(self):
        return True

    def write(self, b):
        self.h.update(b)
        self.n += len(b)
        return self.f.write(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("save")
    ap.add_argument("out")
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    tag = a.name.rsplit(":", 1)[1] if ":" in a.name.rsplit("/", 1)[-1] else "latest"
    work = tempfile.mkdtemp(prefix=".oci-", dir=os.path.dirname(os.path.abspath(a.out)))
    try:
        with tarfile.open(a.save, "r:") as t:
            (entry,) = json.load(t.extractfile("manifest.json"))
            config = t.extractfile(entry["Config"]).read()
            cid = "sha256:" + hashlib.sha256(config).hexdigest()
            if cid != a.id:
                sys.exit(f"the saved config is {cid}, not the image id {a.id}")
            diff_ids = json.loads(config)["rootfs"]["diff_ids"]
            if len(diff_ids) != len(entry["Layers"]):
                sys.exit("the saved layers do not match the config's diff ids")
            blobs, layers = {}, []
            for path, diff_id in zip(entry["Layers"], diff_ids):
                src, raw = t.extractfile(path), hashlib.sha256()
                tmp = os.path.join(work, f"layer{len(layers)}")
                with open(tmp, "wb") as f:
                    hw = Hashing(f)
                    with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=hw, mtime=0) as gz:
                        while chunk := src.read(CHUNK):
                            raw.update(chunk)
                            gz.write(chunk)
                if "sha256:" + raw.hexdigest() != diff_id:
                    sys.exit(f"{path}: content sha256:{raw.hexdigest()} is not the diff id {diff_id}")
                digest = "sha256:" + hw.h.hexdigest()
                blobs[digest] = tmp
                layers.append({"mediaType": MT_LAYER, "digest": digest, "size": hw.n, "diff_id": diff_id})
        manifest = dumps({"schemaVersion": 2, "mediaType": MT_MANIFEST,
                          "config": {"mediaType": MT_CONFIG, "digest": cid, "size": len(config)},
                          "layers": [{k: l[k] for k in ("mediaType", "digest", "size")} for l in layers]})
        mid = "sha256:" + hashlib.sha256(manifest).hexdigest()
        index = dumps({"schemaVersion": 2, "mediaType": MT_INDEX,
                       "manifests": [{"mediaType": MT_MANIFEST, "digest": mid, "size": len(manifest),
                                      "annotations": {"io.containerd.image.name": a.name,
                                                      "org.opencontainers.image.ref.name": tag}}]})
        docker = dumps([{"Config": f"blobs/sha256/{cid[7:]}", "RepoTags": [a.name],
                         "Layers": [f"blobs/sha256/{l['digest'][7:]}" for l in layers]}])
        small = {cid: config, mid: manifest}

        def info(name: str, size: int, d: bool = False) -> tarfile.TarInfo:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = (0 if d else size), 0, 0, 0, "", ""
            ti.mode, ti.type = (0o755, tarfile.DIRTYPE) if d else (0o644, tarfile.REGTYPE)
            return ti

        part = a.out + ".part"
        with open(part, "wb") as f:
            hw = Hashing(f)
            with tarfile.open(fileobj=hw, mode="w|", format=tarfile.PAX_FORMAT) as out:
                out.addfile(info("blobs/", 0, True))
                out.addfile(info("blobs/sha256/", 0, True))
                for digest in sorted(set(blobs) | set(small)):
                    name = f"blobs/sha256/{digest[7:]}"
                    if digest in small:
                        out.addfile(info(name, len(small[digest])), io.BytesIO(small[digest]))
                    else:
                        with open(blobs[digest], "rb") as b:
                            out.addfile(info(name, os.path.getsize(blobs[digest])), b)
                for name, data in (("index.json", index), ("manifest.json", docker),
                                   ("oci-layout", b'{"imageLayoutVersion":"1.0.0"}')):
                    out.addfile(info(name, len(data)), io.BytesIO(data))
        os.replace(part, a.out)
        print(json.dumps({"name": a.name, "manifest_digest": mid, "config_digest": cid, "layers": layers,
                          "archive_sha256": hw.h.hexdigest(), "archive_bytes": hw.n}, indent=1))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
