"""Tests for images/pull_all.sh (images/pins.py) with a fake `docker` (no network, no registry), standard library only:

    python3 images/tests/test_pull_all.py

The fake keeps an image store of one of Docker's two kinds and answers as the real ones do (read on Docker 28.0.0 with
the classic store and on Docker 29.1.3 with the containerd store): after a pull by digest, `docker image inspect
--format {{.Id}} <ref>` prints the config digest in the classic store and the manifest digest in the containerd store.
- --list prints "<ref> <id>" for every pin of the repository's images/pins.json, in its order;
- a malformed pin (a tag without its digest, an id that is not a sha256) stops before any pull;
- every missing image is pulled by its pinned reference, in order, and passes the check in both stores; an image that
  is local with its pin is not pulled again; a reference that resolves to another image fails the run;
- a pull needs the licences' acceptance (flag or ROBOARC_ACCEPT_LICENCES=1); nothing to pull, --check and --list do not;
- a refused pull says how to log in (and that an old login can be the cause), an unreachable registry says so;
- --check pulls nothing and tells an image that is local under its tag only from one that is absent.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "images/pull_all.sh"
FAKE_DOCKER = r'''#!{py}
import json, os, sys
a, path = sys.argv[1:], os.environ["FAKE_DOCKER_STATE"]
st = json.load(open(path))
with open(os.environ["FAKE_DOCKER_LOG"], "a") as f:
    f.write(json.dumps(a) + "\n")


def norm(ref):      # Docker looks a <repo>:<tag>@<digest> reference up by its digest: the tag is dropped
    name, at, digest = ref.partition("@")
    return name.rsplit(":", 1)[0] + at + digest if at and ":" in name.rsplit("/", 1)[-1] else ref


if st.get("daemon") == "down":
    sys.exit("permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock")
if a[:2] == ["image", "inspect"]:
    got = st["images"].get(norm(a[-1]))
    if got is None:
        print()
        sys.exit("Error response from daemon: No such image: " + a[-1])
    print(got)
elif a[0] == "info":
    print("overlayfs [[driver-type io.containerd.snapshotter.v1]]" if st["store"] == "containerd"
          else "overlay2 [[Backing Filesystem extfs] [Supports d_type true]]")
elif a[0] == "pull":
    ref = norm(a[-1])
    served = st["registry"].get(ref, "Error response from daemon: manifest unknown")
    if not served.startswith("sha256:"):
        sys.exit(served)        # the registry's refusal, as docker words it
    st["images"][ref] = served if st["store"] == "classic" else ref.split("@")[1]
    json.dump(st, open(path, "w"))
'''
A = "ghcr.io/example/base:1.0@sha256:" + "a" * 64
B = "ghcr.io/example/sim:1.0@sha256:" + "b" * 64
IA, IB, OTHER = "sha256:" + "1" * 64, "sha256:" + "2" * 64, "sha256:" + "9" * 64
PINS = [{"name": "base", "ref": A, "id": IA}, {"name": "sim", "ref": B, "id": IB}]
REG = {"ghcr.io/example/base@sha256:" + "a" * 64: IA, "ghcr.io/example/sim@sha256:" + "b" * 64: IB}
# docker's words for an anonymous pull of a private ghcr.io package, read on 2026-11-21
DENIED = {"classic": 'Error response from daemon: Get "https://ghcr.io/v2/example/base/manifests/sha256:' + "a" * 64
                     + '": unauthorized',
          "containerd": "Error response from daemon: error from registry: unauthorized\nunauthorized"}
SENTENCE = ("pull_all: pulling the images needs your acceptance of the third-party licences in "
            "LICENSES/THIRD_PARTY.md: read it, then run this again with --accept-licences (or ROBOARC_ACCEPT_LICENCES=1).")


class PullAll(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        (self.d / "bin").mkdir()
        docker = self.d / "bin" / "docker"
        docker.write_text(FAKE_DOCKER.replace("{py}", sys.executable))
        docker.chmod(0o755)
        self.log, self.state = self.d / "docker.jsonl", self.d / "state.json"

    def tearDown(self):
        self.tmp.cleanup()

    def pins(self, images=PINS):
        p = self.d / "pins.json"
        p.write_text(json.dumps({"images": images}))
        return p

    def run_it(self, *args, store="classic", images=None, registry=REG, env=None, **state):
        """Run the script on a fake Docker of `store` holding `images` ({reference: image id}), with `registry`
        serving {<repo>@<digest>: config digest, or docker's error text}."""
        self.state.write_text(json.dumps(dict(state, store=store, images=images or {}, registry=registry)))
        if self.log.exists():
            self.log.unlink()
        e = {k: v for k, v in os.environ.items() if k != "ROBOARC_ACCEPT_LICENCES"}
        e.update(PATH=f"{self.d / 'bin'}:{os.environ['PATH']}", FAKE_DOCKER_LOG=str(self.log),
                 FAKE_DOCKER_STATE=str(self.state), **(env or {}))
        return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=e, timeout=60)

    def calls(self, verb=None):
        calls = [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []
        return [c for c in calls if verb is None or c[0] == verb]

    def local(self, store, pins=PINS):
        """The store's images after every pin was pulled from a registry."""
        return {p["ref"].replace(":1.0@", "@"): p["id"] if store == "classic" else p["ref"].split("@")[1] for p in pins}

    def test_list_of_the_repository_pins(self):
        p = self.run_it("--list")
        self.assertEqual(p.returncode, 0, p.stderr)
        want = [f"{x['ref']} {x['id']}" for x in json.loads((REPO / "images/pins.json").read_text())["images"]]
        self.assertEqual(p.stdout.splitlines(), want)
        self.assertEqual(self.calls(), [])

    def test_malformed_pins_stop_before_any_pull(self):
        for bad in ({"name": "x", "ref": "ghcr.io/example/base:1.0", "id": IA},
                    {"name": "x", "ref": A, "id": "1234"}, {"name": "x"}, None):
            p = self.run_it("--accept-licences", "--pins", str(self.pins([{"name": "ok", "ref": B, "id": IB}, bad])))
            self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
            self.assertEqual(self.calls(), [])
        self.assertEqual(self.run_it("--pins", str(self.d / "missing.json")).returncode, 2)
        self.assertEqual(self.run_it("--no-such-option").returncode, 2)

    def test_pull_in_order_and_check_in_both_stores(self):
        for store in ("classic", "containerd"):
            p = self.run_it("--accept-licences", "--pins", str(self.pins()), store=store)
            self.assertEqual(p.returncode, 0, store + p.stdout + p.stderr)
            self.assertEqual([c[-1] for c in self.calls("pull")], [A, B])
            self.assertIn("pull_all: every pinned image is local with its pinned id", p.stdout)
            # the id on each line is the one `docker images` shows in that store
            shown = IA[7:19] if store == "classic" else "a" * 12
            self.assertIn(f"ok   {A} = {shown}", p.stdout)
            self.assertEqual("containerd image store" in p.stdout, store == "containerd")

    def test_a_reference_that_resolves_to_another_image_fails(self):
        p = self.run_it("--accept-licences", "--pins", str(self.pins()), registry=dict(REG, **{B.replace(":1.0@", "@"): OTHER}))
        self.assertEqual(p.returncode, 1)
        self.assertIn(f"FAIL {B}: image id {OTHER}, pinned {IB}", p.stderr)
        self.assertIn(f"ok   {A}", p.stdout)
        # and a pull is not tried again for it: the digest is local already
        p = self.run_it("--pins", str(self.pins()), images={A.replace(":1.0@", "@"): IA, B.replace(":1.0@", "@"): OTHER})
        self.assertEqual((p.returncode, self.calls("pull")), (1, []))

    def test_local_images_are_not_pulled_and_need_no_acceptance(self):
        for store in ("classic", "containerd"):
            p = self.run_it("--pins", str(self.pins()), store=store, images=self.local(store))
            self.assertEqual(p.returncode, 0, store + p.stdout + p.stderr)
            self.assertEqual(self.calls("pull"), [])
            # one missing: only that one is pulled
            p = self.run_it("--accept-licences", "--pins", str(self.pins()), store=store, images=self.local(store, PINS[:1]))
            self.assertEqual(p.returncode, 0, store + p.stdout + p.stderr)
            self.assertEqual([c[-1] for c in self.calls("pull")], [B])

    def test_a_pull_needs_the_licences_accepted(self):
        p = self.run_it("--pins", str(self.pins()))
        self.assertEqual((p.returncode, p.stdout, p.stderr.strip()), (3, "", SENTENCE))
        self.assertEqual(self.calls("pull"), [])
        for args, env in ((["--accept-licences"], {}), (["--accept-licenses"], {}), ([], {"ROBOARC_ACCEPT_LICENCES": "1"})):
            p = self.run_it(*args, "--pins", str(self.pins()), env=env)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(self.run_it("--pins", str(self.pins()), env={"ROBOARC_ACCEPT_LICENCES": "0"}).returncode, 3)

    def test_a_refused_pull_says_how_to_log_in(self):
        for store in ("classic", "containerd"):
            p = self.run_it("--accept-licences", "--pins", str(self.pins()), store=store,
                            registry=dict(REG, **{A.replace(":1.0@", "@"): DENIED[store]}))
            self.assertEqual(p.returncode, 1, store)
            self.assertIn(f"FAIL {A}: docker pull failed: {DENIED[store].splitlines()[0]}", p.stderr)
            self.assertIn("docker login ghcr.io -u <github user> --password-stdin", p.stderr)
            self.assertIn("read:packages", p.stderr)
            self.assertIn("docker logout ghcr.io", p.stderr)
            self.assertIn(f"ok   {B}", p.stdout)        # the other image is still pulled and checked
            self.assertEqual(p.stderr.count("refused the pull"), 1)
        p = self.run_it("--accept-licences", "--pins", str(self.pins()), registry={
            k: 'Error response from daemon: Get "https://ghcr.io/v2/": dial tcp: lookup ghcr.io: no such host' for k in REG})
        self.assertEqual(p.returncode, 1)
        self.assertIn("ghcr.io could not be reached", p.stderr)
        self.assertNotIn("docker login", p.stderr)
        self.assertEqual(p.stderr.count("could not be reached"), 1)     # said once, not per image

    def test_check_pulls_nothing(self):
        p = self.run_it("--check", "--pins", str(self.pins()))
        self.assertEqual(p.returncode, 1)
        self.assertIn(f"FAIL {A}: not local", p.stderr)
        self.assertIn("Docker image store: classic (overlay2)", p.stdout)
        # loaded from an archive or built, in the classic store: there under its tag, not under its digest
        p = self.run_it("--check", "--pins", str(self.pins()), images={"ghcr.io/example/base:1.0": IA, "ghcr.io/example/sim:1.0": OTHER})
        self.assertEqual(p.returncode, 1)
        self.assertIn(f"FAIL {A}: local as ghcr.io/example/base:1.0 only, not under the digest the tasks name", p.stderr)
        self.assertIn(f"FAIL {B}: not local (ghcr.io/example/sim:1.0 is another image, {OTHER})", p.stderr)
        for store in ("classic", "containerd"):
            p = self.run_it("--check", "--json", "--pins", str(self.pins()), store=store, images=self.local(store))
            self.assertEqual(p.returncode, 0, p.stderr)
            got = json.loads(p.stdout)
            self.assertEqual((got["ok"], got["store"], [i["status"] for i in got["images"]]), (True, store, ["ok", "ok"]))
        self.assertEqual(self.calls("pull"), [])

    def test_no_usable_docker(self):
        p = self.run_it("--check", "--pins", str(self.pins()), daemon="down")
        self.assertEqual(p.returncode, 2)
        self.assertIn("Docker cannot be used: permission denied while trying to connect to the Docker daemon", p.stderr)
        self.assertIn("is your user in the `docker` group?", p.stderr)


if __name__ == "__main__":
    unittest.main()
