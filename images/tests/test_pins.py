"""Tests for images/pins.py without Docker (standard library only, about 1 s):

    python3 images/tests/test_pins.py

- judge: a pinned reference that resolves to the pinned id (classic image store) or to the pinned manifest digest
  (containerd image store) is the pin; to anything else, it differs; an image under its tag only is told apart;
- classify: docker's words for a refused pull, an unreachable registry, a full disk and an unreachable daemon;
- archives: an OCI archive is its pin only with the pinned manifest digest, the pinned config and every blob under the
  sha256 of its bytes; a changed blob, another manifest, a missing archive fail;
- licences: the image build scripts stop (exit 3) before they do anything without --accept-licences or
  ROBOARC_ACCEPT_LICENCES=1, with one sentence that names LICENSES/THIRD_PARTY.md.
"""
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("pins", REPO / "images/pins.py")
pins = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pins)

C, D, X = "sha256:" + "c" * 64, "sha256:" + "d" * 64, "sha256:" + "e" * 64
PIN = {"name": "img", "ref": "ghcr.io/example/img:1.0@" + D, "id": C, "repo": "ghcr.io/example/img", "tag": "1.0",
       "digest": D, "archive": "img-1.0.oci.tar"}


def sha(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def archive(path: pathlib.Path, change=None) -> dict:
    """A small OCI archive (one layer) and the pin of its image. change: a function on the {member name: bytes} about
    to be written, to spoil it."""
    config, layer = b'{"rootfs":{"diff_ids":[]}}', b"layer bytes"
    manifest = json.dumps({"schemaVersion": 2, "config": {"digest": sha(config), "size": len(config)},
                           "layers": [{"digest": sha(layer), "size": len(layer)}]}).encode()
    files = {"blobs/sha256/" + sha(b)[7:]: b for b in (config, layer, manifest)}
    files["index.json"] = json.dumps({"schemaVersion": 2, "manifests": [{"digest": sha(manifest)}]}).encode()
    files["oci-layout"] = b'{"imageLayoutVersion":"1.0.0"}'
    if change:
        change(files)
    with tarfile.open(path, "w") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return dict(PIN, id=sha(config), digest=sha(manifest), ref="ghcr.io/example/img:1.0@" + sha(manifest),
                archive=path.name)


class Judge(unittest.TestCase):
    def test_both_stores(self):
        self.assertEqual(pins.judge(PIN, C, None), ("ok", C))               # classic: the config digest
        self.assertEqual(pins.judge(PIN, D, None), ("ok", D))               # containerd: the manifest digest
        self.assertEqual(pins.judge(PIN, X, None), ("differs", X))
        self.assertEqual(pins.judge(PIN, None, C), ("tag-only", C))
        self.assertEqual(pins.judge(PIN, None, X), ("missing", X))
        self.assertEqual(pins.judge(PIN, None, None), ("missing", None))

    def test_lines(self):
        self.assertEqual(pins.line(PIN, "ok", C), f"ok   {PIN['ref']} = cccccccccccc")
        self.assertTrue(pins.line(PIN, "ok", D).startswith(f"ok   {PIN['ref']} = dddddddddddd (the manifest digest"))
        self.assertIn("pinned id cccccccccccc", pins.line(PIN, "ok", D))
        self.assertEqual(pins.line(PIN, "missing", None), f"FAIL {PIN['ref']}: not local")


class Classify(unittest.TestCase):
    def test_dockers_words(self):
        for text, kind in (
                # an anonymous pull of a private ghcr.io package, read on Docker 28.0.0 (classic) and 29.1.3 (containerd)
                ('Error response from daemon: Get "https://ghcr.io/v2/o/i/manifests/sha256:ab": unauthorized', "denied"),
                ("Error response from daemon: error from registry: unauthorized\nunauthorized", "denied"),
                ('Error response from daemon: Head "https://ghcr.io/v2/o/i/manifests/1.0": denied: denied', "denied"),
                ("failed to authorize: failed to fetch anonymous token: unexpected status: 403 Forbidden", "denied"),
                ("pull access denied for x, repository does not exist or may require 'docker login'", "denied"),
                ('Get "https://ghcr.io/v2/": dial tcp: lookup ghcr.io on 127.0.0.53:53: no such host', "network"),
                ('Get "https://ghcr.io/v2/": net/http: TLS handshake timeout', "network"),
                ('Get "https://ghcr.io/v2/": context deadline exceeded (Client.Timeout exceeded)', "network"),
                ("failed to register layer: write /usr/lib/x.so: no space left on device", "disk"),
                ("Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?", "daemon"),
                ("permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock", "daemon"),
                ("Error response from daemon: manifest unknown", "other")):
            self.assertEqual(pins.classify(text), kind, text)


class Archives(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_archive_is_its_pin(self):
        pin = archive(self.d / "img-1.0.oci.tar")
        self.assertEqual(pins.check_archive(self.d / "img-1.0.oci.tar", pin), [])
        bad = pins.check_archive(self.d / "img-1.0.oci.tar", dict(pin, digest=D, id=C))
        self.assertEqual(len(bad), 2, bad)
        self.assertIn(f"is not the pinned {D}", bad[0])
        self.assertIn(f"is not the pinned image id {C}", bad[1])

    def test_spoiled_archives(self):
        def flip(files):        # a layer whose bytes are not its name
            name = next(n for n, b in files.items() if b == b"layer bytes")
            files[name] = b"other bytes"

        def drop(files):
            del files[next(n for n, b in files.items() if b == b"layer bytes")]

        pin = archive(self.d / "a.tar", flip)
        self.assertIn("blobs whose sha256 is not their name", pins.check_archive(self.d / "a.tar", pin)[0])
        pin = archive(self.d / "b.tar", drop)
        self.assertIn("blobs the manifest names are missing", pins.check_archive(self.d / "b.tar", pin)[0])
        pin = archive(self.d / "c.tar", lambda files: files.pop("index.json"))
        self.assertIn("not an OCI archive of one image", pins.check_archive(self.d / "c.tar", pin)[0])
        (self.d / "d.tar").write_bytes(b"not a tar file")
        self.assertIn("cannot be read", pins.check_archive(self.d / "d.tar", pin)[0])

    def test_command(self):
        pin = archive(self.d / "img-1.0.oci.tar")
        other = dict(pin, name="gone", archive="gone-1.0.oci.tar")
        (self.d / "pins.json").write_text(json.dumps({"images": [pin, other]}))
        run = lambda *a: subprocess.run([sys.executable, str(REPO / "images/pins.py"), "archives", "--archives", str(self.d),
                                         "--pins", str(self.d / "pins.json"), *a], capture_output=True, text=True, timeout=60)
        p = run()
        self.assertEqual(p.returncode, 1)
        self.assertIn(f"ok   img-1.0.oci.tar = {pin['ref']}", p.stdout)
        self.assertIn("FAIL gone-1.0.oci.tar: missing", p.stderr)
        got = json.loads(run("--json").stdout)
        self.assertEqual((got["ok"], [a["status"] for a in got["archives"]]), (False, ["ok", "fail"]))
        (self.d / "pins.json").write_text(json.dumps({"images": [pin]}))
        p = run()
        self.assertEqual((p.returncode, p.stdout.splitlines()[-1]), (0, "archives: every archive is its pinned image"))


class Licences(unittest.TestCase):
    def run_it(self, *cmd, accept=None):
        env = {k: v for k, v in os.environ.items() if k != "ROBOARC_ACCEPT_LICENCES"}
        env["PATH"] = self.bin + ":" + env["PATH"]
        if accept is not None:
            env["ROBOARC_ACCEPT_LICENCES"] = accept
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)

    def setUp(self):        # a docker that only records that it was called
        self.tmp = tempfile.TemporaryDirectory()
        self.bin = self.tmp.name
        self.called = pathlib.Path(self.tmp.name) / "called"
        docker = pathlib.Path(self.bin) / "docker"
        docker.write_text(f"#!/bin/sh\necho \"$@\" >> {self.called}\nexit 1\n")
        docker.chmod(0o755)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_rule(self):
        tool = [sys.executable, str(REPO / "images/pins.py"), "licences", "x.sh: building the image"]
        p = self.run_it(*tool)
        self.assertEqual(p.returncode, 3)
        self.assertEqual(p.stderr.strip(), "x.sh: building the image needs your acceptance of the third-party licences "
                         "in LICENSES/THIRD_PARTY.md: read it, then run this again with --accept-licences (or "
                         "ROBOARC_ACCEPT_LICENCES=1).")
        self.assertTrue((REPO / "LICENSES/THIRD_PARTY.md").is_file())
        self.assertEqual(self.run_it(*tool, "--accept-licences").returncode, 0)
        self.assertEqual(self.run_it(*tool, accept="1").returncode, 0)
        self.assertEqual(self.run_it(*tool, accept="yes").returncode, 3)

    def test_the_build_scripts_stop_first(self):
        for script in ("images/roboarc/build.sh", "images/monitor/build.sh"):
            p = self.run_it("bash", str(REPO / script))
            self.assertEqual(p.returncode, 3, script + p.stderr)
            self.assertIn(f"{script}: building the image needs your acceptance", p.stderr)
            self.assertIn("LICENSES/THIRD_PARTY.md", p.stderr)
            self.assertFalse(self.called.exists(), script)
        # accepted: the monitor build goes on to docker (which is the recorder here, so the build stops there)
        p = self.run_it("bash", str(REPO / "images/monitor/build.sh"), "--accept-licences", "--out", str(pathlib.Path(self.bin) / "m.tar"))
        self.assertNotEqual(p.returncode, 3)
        self.assertIn("buildx create", self.called.read_text())


if __name__ == "__main__":
    unittest.main()
