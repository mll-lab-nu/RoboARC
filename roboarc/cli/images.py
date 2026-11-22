"""roboarc images pull / check / build: the pinned container images, through the scripts under images/. The
command line keeps the licence consent (licences.py: asked once, then remembered) and hands it to the script as
ROBOARC_ACCEPT_LICENCES=1; every check of an image is the script's."""
from __future__ import annotations

import argparse

from . import _common as C
from . import licences

# subcommand -> the script it runs (a repository path) and that script's first arguments
SCRIPTS = {"pull": ("images/pull_all.sh",),
           "check": ("images/pull_all.sh", "--check"),               # the local Docker images against the pins
           "check-archives": ("images/pins.py", "archives")}         # OCI archives against the pins
BUILDS = {"roboarc": "images/roboarc/build.sh", "ikea": "ikea/images/ikea-mujoco/build.sh",
          "monitor": "images/monitor/build.sh"}


def _pins(a) -> list[str]:
    return ["--pins", a.pins] if a.pins else []


def _gated(a, what: str, command: list[str]) -> int:
    """Run an image script (found first: no question about a script this checkout lacks) once the licences are accepted."""
    return C.run(command, licences.require(a.accept_licences, what))


def _pull(a) -> int:
    return _gated(a, "images pull", C.command(*SCRIPTS["pull"], *_pins(a)))


def _check(a) -> int:
    extra = (["--json"] if a.json else []) + _pins(a)
    if a.archives:
        return C.run(C.command(*SCRIPTS["check-archives"], "--archives", a.archives, *extra))
    return C.run(C.command(*SCRIPTS["check"], *extra))


def _build(a) -> int:
    args = a.args[1:] if a.args[:1] == ["--"] else a.args
    return _gated(a, "images build", C.command(BUILDS[a.image], *args))


def _accept(p) -> None:
    p.add_argument(licences.FLAG, action="store_true", help=f"accept the third-party licences in {licences.LICENCES}; "
                   f"{licences.VARIABLE}=1 in the environment says the same. Needed once per user: without it a run "
                   "on a terminal asks, and a non-interactive run stops with one sentence (default: not given)")
    p.add_argument("--accept-licenses", dest="accept_licences", action="store_true", help=argparse.SUPPRESS)


def add_parsers(sub) -> None:
    pins = dict(metavar="FILE", help="the pins file to use instead of this checkout's: a JSON file like "
                "images/pins.json, e.g. a copy that names a local registry (default: images/pins.json)")

    top = sub.add_parser("images", formatter_class=C.Formatter,
                         help="the pinned container images: pull, check, build", description=f"""\
The four container images the tasks run in. images/pins.json lists them, each by its manifest digest:
  roboarc            every puzzle task               roboarc-ikea       the IKEA tasks, privileged
  roboarc-monitor    every task (resource monitor)   roboarc-ikea-sim   the IKEA tasks, standard
About 19 GB on disk. Pulling them is the normal path; building them from this checkout is optional.

Third-party licences. The images contain third-party software and assets under their own licences, listed in
{licences.LICENCES}. pull and build ask you to accept them once: answer the question on the terminal, or
give {licences.FLAG} (or set {licences.VARIABLE}=1). The answer is kept in
~/.config/roboarc/licences-accepted.json (under $XDG_CONFIG_HOME when that is set); delete the file to be asked again.
check brings no image onto the machine and does not ask.""", epilog=f"""\
example:
  roboarc images pull {licences.FLAG} && roboarc images check
"roboarc images SUBCOMMAND --help" has each subcommand's arguments.""")
    s = top.add_subparsers(dest="images_cmd", metavar="SUBCOMMAND", title="subcommands", required=True)

    p = s.add_parser("pull", formatter_class=C.Formatter, help="pull every pinned image and check its image id",
                     description="""\
Pull every image of the pins file from its registry, in the file's order, then check each one against its pinned image
id. Runs images/pull_all.sh. Needs Docker and about 19 GB of free disk; no registry login for the public images.
Exit status: 0 only when every pinned image is local with its pinned id.""", epilog=f"""\
example:
  roboarc images pull {licences.FLAG}""")
    p.add_argument("--pins", **pins)
    _accept(p)
    p.set_defaults(fn=_pull)

    p = s.add_parser("check", formatter_class=C.Formatter,
                     help="check the local images, or OCI archives, against the pins without pulling", description="""\
Check what is on this machine against the pins, without pulling anything: the images in the local Docker store
(images/pull_all.sh --check) or, with --archives, the OCI archives in a directory (images/pins.py archives: each file
the pins name must hold one manifest whose digest is the pin, a config that is the pinned image id, and every blob
under its sha256). images/README.md describes how archives serve a host that cannot reach the registry.
Exit status: 0 only when every pinned image, or archive, is there and is the pinned one.""", epilog="""\
examples:
  roboarc images check
  roboarc images check --archives /data/roboarc-archives""")
    p.add_argument("--pins", **pins)
    p.add_argument("--archives", metavar="DIR", help="check the OCI archives in this directory, named as the pins "
                   "file's `archive` fields, instead of the local Docker store (default: the local Docker store)")
    p.add_argument("--json", action="store_true", help="print the result as one JSON object instead of one line per "
                   "image")
    p.set_defaults(fn=_check)

    p = s.add_parser("build", formatter_class=C.Formatter, help="build an image from this checkout (optional)",
                     description="""\
Build an image from this checkout instead of pulling it: only needed to change the simulator. Runs the image's own
build script with the arguments after `--`:
  roboarc   images/roboarc/build.sh [--instances DIR] [--download DIR] [--tag REF] [--noinst]
  ikea      ikea/images/ikea-mujoco/build.sh [--assets DIR] [--download DIR] [--out DIR] [--no-load]
            (both IKEA images, as OCI archives)
  monitor   images/monitor/build.sh [--out FILE] [--load]
Each script's header describes its arguments, its inputs (the asset datasets among them) and its checks;
images/README.md has the pins and the archives.""",
                     epilog=f"""\
example:
  roboarc images build roboarc {licences.FLAG} -- --instances /data/roboarc-instances""")
    p.add_argument("image", metavar="IMAGE", choices=tuple(BUILDS), help="which image to build: roboarc, ikea or "
                   "monitor")
    _accept(p)
    p.add_argument("args", metavar="-- ARGS", nargs="*", help="arguments for the build script, after `--` (default: "
                   "none, the script's own defaults)")
    p.set_defaults(fn=_build)
