#!/usr/bin/env python3
"""Cut the README gallery GIFs from the studio clips of the reference solutions.

Each GIF is a window of one clip, cropped, played faster and ending on the solved state: 320 x 180 pixels,
10 frames per second, 60 frames (6.0 s), looping forever. The first 52 frames run through the window, the last of
them is the window's final frame, and it is repeated for the remaining 8 frames so that the solved state is seen
before the loop starts again. `GALLERY` below is the source of truth for the windows and crops; the table in
README.md beside this file is printed from it (`--table`).

    python3 docs/assets/gallery/make_gallery.py                       # all twelve, masters from Hugging Face
    python3 docs/assets/gallery/make_gallery.py --masters DIR         # masters already on disk
    python3 docs/assets/gallery/make_gallery.py --only knot-untying-04
    python3 docs/assets/gallery/make_gallery.py --table               # the table of README.md
    python3 docs/assets/gallery/make_gallery.py --check               # the GIFs here, the tasks and README.md agree

The masters are public: https://huggingface.co/datasets/MLL-Lab/embodied-exam-data, `masters/roboarc/<id>/video.mp4`
(2560 x 1440 or 3840 x 2160, 30 frames per second, the last 1.6 s or more a still of the final state). Without
`--masters` they are downloaded once into `--cache`. With `--masters DIR` a clip is read from `DIR/<id>/video.mp4`
or `DIR/roboarc-<base>.mp4`. Either way a clip is checked against the SHA-256 recorded here before it is cut.

Needs ffmpeg and ffprobe (4.4 or newer) on the PATH. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import struct
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HF = "https://huggingface.co/datasets/MLL-Lab/embodied-exam-data/resolve/main/masters/roboarc"
WIDTH, HEIGHT = 320, 180
FPS = 10
FRAMES = 60          # 6.0 s
HOLD = 8             # frames at the end that repeat the window's final frame
MOVING = FRAMES - HOLD
MAX_BYTES = 1_000_000
# A light spatial denoise of the small frame: the studio floor is a smooth grey gradient with render grain, and the
# grain costs a quarter of the file. No temporal part, so nothing trails behind the moving arms.
DENOISE = "hqdn3d=2:1.5:0:0"
# One palette for the whole clip and no dithering: the floor stays a clean gradient, rope and cloth edges stay sharp.
PALETTE = "stats_mode=full:reserve_transparent=0"
QUANTISE = "dither=none:diff_mode=rectangle"


@dataclass(frozen=True)
class Clip:
    name: str        # the GIF's file name without `.gif`: the task base without `roboarc-` and the instance suffix
    title: str
    constraint: str
    robot: str
    master: str      # `<id>` in masters/roboarc/<id>/video.mp4
    base: str        # the task base: tasks/roboarc-<base>-{privileged,standard}
    sha256: str      # of the master
    t0: float        # window start in the master, seconds
    t1: float        # window end, seconds: inside the master's final still, so the last frame is the solved state
    crop: tuple[float, float, float]   # width fraction F and top-left corner (X, Y), fractions of the frame (16:9 kept)
    shows: str
    colors: int = 256                  # palette size; fewer where the file would pass 1 MB


ALOHA, FRANKA, FRANKA2 = "ALOHA 2", "Franka + Robotiq 2F-85", "two Franka arms"

# README order: four rows of three, each constraint type twice. Every crop keeps the whole robot in the frame for the
# whole window (in interlocked-claws-05 and cast-coil-08 the master itself cuts the right arm's base at the bottom).
GALLERY: list[Clip] = [
    Clip("interlocked-claws-05", "Interlocked claws 05", "Interlocking", ALOHA, "interlocked_claws_05",
         "interlocked-claws-05-i01", "4c4f110ba7f7b82eb78a549d5ba5b29f94454bc164f99773a036f777be180cae",
         22.0, 41.3, (0.86, 0.06, 0.14),
         "the last twists, the two claws come apart and are held apart"),
    Clip("hedgehog-in-cage-03", "Hedgehog in cage 03", "Narrow passage", FRANKA, "hedgehog_in_cage_03",
         "hedgehog-in-cage-03-i01", "0729045fc171526003acfe6740265eb6bcc3c6c99e1ec8b7bebf44b805d866f0",
         15.5, 26.7, (0.80, 0.07, 0.09),
         "the spiked ball is turned inside the copper cage, pulled out between the bars and set down"),
    Clip("ikea-gladom", "IKEA GLADOM", "Assembly", FRANKA2, "ikea_gladom",
         "ikea-gladom-i00", "bb8dfd0a747e16e63a1b680d9b13c02fbc7d24fd9d0b46df2ed08bb47517ed39",
         14.0, 21.3, (0.78, 0.10, 0.13),
         "the wire frame is finished and the tray is lifted from the floor onto it"),
    Clip("knot-untying-04", "Knot untying 04", "Untangling", ALOHA, "knot_untying_04",
         "knot-untying-04-i01", "0d400a1178a90efd41ba4e148c30bf38c4318713ffe40bf6d0da2237faed811a",
         1.8, 16.3, (0.76, 0.09, 0.09),
         "the knot is loosened and the rope pulled straight"),
    Clip("rope-threading-08", "Rope threading 08", "Threading", ALOHA, "rope_threading_08",
         "rope-threading-08-i01", "eeaf61ce1c591f13e4befd8ba9f974c007b53143049c261855fc4def7873ad9f",
         12.0, 31.5, (0.76, 0.10, 0.02),
         "the rope is passed through the wall grommets and the rings, the coil on the floor shrinks"),
    Clip("blanket-fold-03", "Blanket fold 03", "Shape control", ALOHA, "blanket_fold_03",
         "blanket-fold-03-i01", "e2297950912ce866570ac0a51792c008cf825c86f352cd0a2eba2cad7934caf9",
         23.0, 31.1, (0.80, 0.11, 0.04),
         "the flat blanket is folded in half"),
    Clip("cast-coil-08", "Cast coil 08", "Interlocking", ALOHA, "cast_coil_08",
         "cast-coil-08-i01", "a8c7a0b6fd379156dc318c8b51d7dcad00bf97cc58aa2dd40e21626f5e3a6b3b",
         22.0, 41.3, (0.84, 0.12, 0.16),
         "the gold and the steel piece slide into each other and the joined puzzle is set down", 192),
    Clip("hedgehog-in-cage-06", "Hedgehog in cage 06", "Narrow passage", FRANKA, "hedgehog_in_cage_06",
         "hedgehog-in-cage-06-i01", "d315fdb6f96f3cd5cfe0dc606a6ca2da48b5e8037884003d8a6308c7451376ec",
         0.5, 15.9, (0.90, 0.05, 0.08),
         "the spiked ball is worked up through the black cast cage, taken out at the top and set down"),
    Clip("slotted-post-ring-05", "Slotted post ring 05", "Untangling", ALOHA, "slotted_post_ring_05",
         "slotted-post-ring-05-i01", "a1a38d50ea07fe4077e7d64ecd9c4732836cc9512e30cf416b3fcab27eec30f3",
         18.0, 30.7, (0.92, 0.03, 0.06),
         "the blue ring is worked off the beaded cord and placed in the goal region"),
    Clip("ikea-vesken", "IKEA VESKEN", "Assembly", FRANKA2, "ikea_vesken",
         "ikea-vesken-i00", "046bc39d3b4c441f3254a7da37207155f9173c228486f81a005695b7052b7a95",
         22.0, 41.3, (0.76, 0.12, 0.10),
         "the shelf unit grows from one shelf to three"),
    Clip("fiber-bundling-08", "Fiber bundling 08", "Threading", FRANKA, "fiber_bundling_08",
         "fiber-bundling-08-i01", "f2929447f8b1d87bdc767a44413e431bcaf5fda67ad4744a92403ec6840acb36",
         6.0, 20.5, (0.80, 0.02, 0.10),
         "the fibers at the cable's end are gathered and pushed through the ring"),
    Clip("blanket-flatten-04", "Blanket flatten 04", "Shape control", ALOHA, "blanket_flatten_04",
         "blanket-flatten-04-i01", "667991c9c154712814b3cf43ce29de38e7126d3153740cdb5c13d3fe14faf598",
         1.0, 16.3, (0.80, 0.09, 0.05),
         "the wrinkled blanket is lifted, spread and laid flat"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def master_path(c: Clip, masters: Path | None, cache: Path) -> Path:
    """The clip on disk: from `--masters`, else from the cache, downloaded on first use."""
    if masters is not None:
        for p in (masters / c.master / "video.mp4", masters / f"roboarc-{c.base}.mp4"):
            if p.is_file():
                return p
        sys.exit(f"{c.name}: no master under {masters} ({c.master}/video.mp4 or roboarc-{c.base}.mp4)")
    p = cache / c.master / "video.mp4"
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF}/{c.master}/video.mp4"
        print(f"{c.name}: downloading {url}")
        tmp = p.with_suffix(".part")
        with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(p)
    return p


def frame_rate(path: Path) -> float:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=r_frame_rate", "-of", "csv=p=0", str(path)])
    num, den = out.decode().strip().rstrip(",").split("/")
    return float(num) / float(den)


def cut(c: Clip, src: Path, out: Path, threads: int = 4) -> None:
    """The window [t0, t1) of `src` as `MOVING` frames whose last one is the window's last frame, then `HOLD` copies."""
    rate = frame_rate(src)
    span = c.t1 - c.t0
    last = (round(span * rate) - 1) / rate                 # time of the window's last frame, from t0
    f, x, y = c.crop
    chain = (f"setpts=(PTS-STARTPTS)*{(MOVING - 1) / FPS}/{last:.6f},fps={FPS},"
             f"crop=iw*{f}:ih*{f}:iw*{x}:ih*{y},scale={WIDTH}:{HEIGHT}:flags=lanczos,{DENOISE},"
             f"tpad=stop_mode=clone:stop_duration={(HOLD + 5) / FPS}")
    graph = (f"[0:v]{chain},split[a][b];[a]palettegen=max_colors={c.colors}:{PALETTE}[p];"
             f"[b][p]paletteuse={QUANTISE}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", str(threads), "-ss", f"{c.t0}", "-t", f"{span:.4f}",
                    "-i", str(src), "-filter_complex", graph, "-frames:v", str(FRAMES), "-loop", "0", str(out)],
                   check=True)


def gif_info(path: Path) -> tuple[int, int, list[int], int | None]:
    """Width, height, every frame's delay in 1/100 s and the loop count (0: forever, None: plays once) of a GIF."""
    b = path.read_bytes()
    if b[:6] not in (b"GIF87a", b"GIF89a"):
        raise ValueError(f"{path}: not a GIF")
    width, height, packed = struct.unpack_from("<HHB", b, 6)
    i = 13 + (3 << ((packed & 7) + 1) if packed & 0x80 else 0)
    delays: list[int] = []
    loop, delay = None, 0

    def skip(i: int) -> int:                               # past a chain of sub-blocks
        while b[i]:
            i += b[i] + 1
        return i + 1

    while b[i] != 0x3B:
        if b[i] == 0x21:                                   # an extension
            if b[i + 1] == 0xF9:
                delay = struct.unpack_from("<H", b, i + 4)[0]
            elif b[i + 1] == 0xFF and b[i + 3:i + 14] == b"NETSCAPE2.0" and b[i + 14] == 3:
                loop = struct.unpack_from("<H", b, i + 16)[0]
            i = skip(i + 2)
        elif b[i] == 0x2C:                                 # an image
            packed = b[i + 9]
            i = skip(i + 11 + (3 << ((packed & 7) + 1) if packed & 0x80 else 0))
            delays.append(delay)
        else:
            raise ValueError(f"{path}: unknown block 0x{b[i]:02x} at byte {i}")
    return width, height, delays, loop


def faults(c: Clip, gif: Path) -> list[str]:
    """What is wrong with the GIF of `c`: empty when it has the gallery's size, length, timing, loop and weight."""
    if not gif.is_file():
        return ["missing"]
    width, height, delays, loop = gif_info(gif)
    out = []
    if (width, height) != (WIDTH, HEIGHT):
        out.append(f"{width} x {height}, expected {WIDTH} x {HEIGHT}")
    if len(delays) != FRAMES:
        out.append(f"{len(delays)} frames, expected {FRAMES}")
    if set(delays) != {100 // FPS}:
        out.append(f"frame delays {sorted(set(delays))} (1/100 s), expected {100 // FPS}")
    if loop != 0:
        out.append("does not loop forever")
    if gif.stat().st_size > MAX_BYTES:
        out.append(f"{gif.stat().st_size / 1e6:.2f} MB, over 1 MB: lower `colors` or crop tighter")
    return out


def check() -> int:
    """The entries, the GIFs beside this file, the task directories and README.md agree. No ffmpeg, no masters."""
    errs = []
    names = [c.name for c in GALLERY]
    errs += [f"{n}: named twice" for n in sorted(set(names)) if names.count(n) > 1]
    for c in GALLERY:
        f, x, y = c.crop
        if not (0 < f <= 1 and x >= 0 and y >= 0 and x + f <= 1 + 1e-9 and y + f <= 1 + 1e-9):
            errs.append(f"{c.name}: crop {c.crop} leaves the frame")
        if c.t1 <= c.t0:
            errs.append(f"{c.name}: empty window {c.t0} to {c.t1}")
        errs += [f"{c.name}: no tasks/roboarc-{c.base}-{mode}" for mode in ("privileged", "standard")
                 if not (ROOT / "tasks" / f"roboarc-{c.base}-{mode}").is_dir()]
        errs += [f"{c.name}.gif: {m}" for m in faults(c, HERE / f"{c.name}.gif")]
    errs += [f"{p.name}: not in GALLERY" for p in sorted(HERE.glob("*.gif")) if p.stem not in names]
    if table() not in (HERE / "README.md").read_text(encoding="utf-8"):
        errs.append("README.md: its table is not the one `--table` prints")
    for e in errs:
        print(f"FAIL {e}")
    print(f"{'FAIL' if errs else 'ok'} make_gallery --check: {len(GALLERY)} GIF(s), {len(errs)} finding(s)")
    return 1 if errs else 0


def speed_up(c: Clip) -> float:
    return (c.t1 - c.t0) / ((MOVING - 1) / FPS)


def table() -> str:
    rows = ["| # | GIF | task | constraint type | robot | master | window (s) | speed-up | crop F, (X, Y) | shows |",
            "|---:|---|---|---|---|---|---|---:|---|---|"]
    for i, c in enumerate(GALLERY, 1):
        f, x, y = c.crop
        rows.append(f"| {i} | `{c.name}.gif` | `roboarc-{c.base}` | {c.constraint} | {c.robot} | "
                    f"`{c.master}/video.mp4` | {c.t0:g} to {c.t1:g} | {speed_up(c):.1f} | "
                    f"{f:g}, ({x:g}, {y:g}) | {c.shows} |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--masters", type=Path, help="directory with the masters (default: download from Hugging Face)")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "roboarc-gallery",
                    help="where downloaded masters are kept (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=HERE, help="output directory (default: this directory)")
    ap.add_argument("--only", action="append", metavar="NAME", help="cut only this GIF (repeatable)")
    ap.add_argument("--threads", type=int, default=4, help="ffmpeg decoding threads (default: %(default)s)")
    ap.add_argument("--table", action="store_true", help="print the table of README.md and exit")
    ap.add_argument("--check", action="store_true",
                    help="check the GIFs in this directory, the task names and README.md against GALLERY and exit")
    a = ap.parse_args()

    if a.table:
        print(table())
        return 0
    if a.check:
        return check()
    names = {c.name for c in GALLERY}
    for n in a.only or []:
        if n not in names:
            sys.exit(f"unknown GIF {n!r}; one of: {', '.join(sorted(names))}")
    a.out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for c in GALLERY:
        if a.only and c.name not in a.only:
            continue
        src = master_path(c, a.masters, a.cache)
        if sha256(src) != c.sha256:
            sys.exit(f"{c.name}: {src} is not the recorded master (SHA-256 differs)")
        out = a.out / f"{c.name}.gif"
        cut(c, src, out, a.threads)
        notes = faults(c, out)
        bad += bool(notes)
        print(f"{out.name:28s} {len(gif_info(out)[2])} frames  {out.stat().st_size / 1e6:.2f} MB  x{speed_up(c):.1f}"
              + "".join(f"  <- {m}" for m in notes))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
