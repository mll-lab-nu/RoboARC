#!/usr/bin/env python3
"""Tests of scripts/ablations/cond_clearance.py (ablation E4) that need neither docker nor the release image: the
relaxations against the repository's simulator source, the clearance measurements on synthetic scenes, and the edits of
a variant directory on copies of released tasks, with a stand-in for the frozen instance. Standard library only.

    python3 scripts/ablations/tests/test_cond_clearance.py"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import pathlib
import re
import shutil
import sys
import tempfile
import tomllib
import unittest
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import cond_clearance as cc  # noqa: E402


def bore_scene(folder: pathlib.Path, bore_r: float, shank_r: float = .005) -> pathlib.Path:
    """A scene.xml with the bed_bolt bore (16 wall boxes round an axis along x) and a round shank prism."""
    (folder / "meshes").mkdir(parents=True, exist_ok=True)
    half = .002
    walls = []
    for i in range(16):
        a = 2 * math.pi * i / 16
        walls.append(f'<geom name="bed_frame::bore_{i}" type="box" pos="0.25 {0.1 + (bore_r + half) * math.cos(a)!r} '
                     f'{0.135 + (bore_r + half) * math.sin(a)!r}" size="0.043 {half} 0.002"/>')
    v = "\n".join(f"v {shank_r * math.cos(2 * math.pi * k / 32)!r} {shank_r * math.sin(2 * math.pi * k / 32)!r} {z}"
                  for z in (-.04, .04) for k in range(32))
    (folder / "meshes" / "shank.obj").write_text(v + "\n")
    xml = ('<mujoco><asset><mesh name="shank" file="meshes/shank.obj"/></asset><worldbody>' + "".join(walls)
           + '<body name="bolt::bolt"><geom name="bolt::bolt_shank" type="mesh" mesh="shank"/></body></worldbody></mujoco>')
    (folder / "scene.xml").write_text(xml)
    return folder / "scene.xml"


def coil_scene(folder: pathlib.Path, cell: float, gap: float) -> pathlib.Path:
    folder.mkdir(parents=True, exist_ok=True)
    boxes = []
    for name in ("coil_gold::coil_gold", "coil_silver::coil_silver"):
        sizes = [(cell, cell, cell - gap), (cell - gap / 2, cell, cell), (cell - gap, cell - gap / 2, cell)]
        geoms = "".join(f'<geom type="box" size="{a / 2!r} {b / 2!r} {c / 2!r}"/>' for a, b, c in sizes)
        boxes.append(f'<body name="{name}">{geoms}<geom type="cylinder" size="0.0135 0.05"/></body>')
    (folder / "scene.xml").write_text("<mujoco><worldbody>" + "".join(boxes) + "</worldbody></mujoco>")
    return folder / "scene.xml"


def lock_scene(folder: pathlib.Path, bore_r: float, shaft_r: float) -> pathlib.Path:
    """A scene.xml with two lock plates (a ring of points at the bore radius about x and the plate's rim) and a key
    whose shaft runs along its x axis."""
    folder.mkdir(parents=True, exist_ok=True)
    ring = lambda r, x: " ".join(f"{x} {r * math.cos(2 * math.pi * k / 24)!r} {r * math.sin(2 * math.pi * k / 24)!r}"
                                 for k in range(24))
    meshes = [f'<mesh name="plate" vertex="{ring(bore_r, -.005)} {ring(bore_r, .005)} {ring(.08, -.005)} {ring(.08, .005)}"/>',
              f'<mesh name="shaft" vertex="0 0 0 {ring(shaft_r, -.4)} {ring(shaft_r, .01)}"/>',
              f'<mesh name="tab" vertex="{ring(.036, 0)} {ring(shaft_r, .002)}"/>']
    bodies = ('<body name="lock_disc_0::disc"><geom type="mesh" mesh="plate"/></body>'
              '<body name="lock_goal::disc"><geom type="mesh" mesh="plate"/></body>'
              '<body name="lock_key::key"><geom type="mesh" mesh="shaft"/><geom type="mesh" mesh="tab"/></body>')
    (folder / "scene.xml").write_text("<mujoco><asset>" + "".join(meshes) + "</asset><worldbody>" + bodies
                                      + "</worldbody></mujoco>")
    return folder / "scene.xml"


class Relaxations(unittest.TestCase):
    def test_bases_two_per_constraint(self):
        """Two bases per rigid constraint type."""
        self.assertEqual(cc.BASES, list(cc.RELAXATIONS))
        kinds = [cc.RELAXATIONS[b].constraint for b in cc.BASES]
        self.assertEqual(sorted(kinds), sorted(["assembly"] * 2 + ["interlocking"] * 2 + ["narrow passage"] * 2))
        for b in cc.BASES:
            self.assertTrue((REPO / "tasks" / f"{b}-privileged").is_dir(), b)
            self.assertTrue((REPO / "tasks" / f"{b}-standard").is_dir(), b)

    def test_edits_apply_to_the_source(self):
        """The repository's copy of each edited file is the image's (its sha256), the edited text occurs once, every
        factor's replacement is a different line, and every stated dimension occurs once in both instructions."""
        for r in cc.RELAXATIONS.values():
            for e in r.edits:
                f = REPO / "source" / e.file
                self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(), e.sha256, r.base)
                text = f.read_text()
                self.assertEqual(text.count(e.old), 1, r.base)
                self.assertEqual(sorted(e.new), list(cc.FACTORS))
                for k, new in e.new.items():
                    self.assertNotEqual(new, e.old)
                    self.assertEqual(text.replace(e.old, new).count(new), 1)
            for k in cc.FACTORS:     # the source says which factor it is
                self.assertTrue(any(f"x{k}" in e.new[k] for e in r.edits), (r.base, k))
            for k, pairs in r.texts.items():
                for mode in ("privileged", "standard"):
                    text = (REPO / "tasks" / f"{r.base}-{mode}" / "instruction.md").read_text()
                    for old, new in pairs:
                        self.assertEqual(text.count(old), 1, (r.base, mode, old))

    def test_variation_is_the_released_instance_s(self):
        """Each base is frozen with the variation of its released task (its README names it)."""
        for r in cc.RELAXATIONS.values():
            readme = (REPO / "tasks" / f"{r.base}-privileged" / "README.md").read_text()
            self.assertIn(f"variation {r.variation},", readme, r.base)

    def test_mujoco_env_is_mk_tasks(self):
        text = (REPO / "scripts" / "roboarc" / "mk_tasks.py").read_text()
        m = re.search(r"^MUJOCO_ENV = (\(.*?\))\s*#", text, re.M | re.S)
        self.assertIsNotNone(m)
        self.assertEqual(ast.literal_eval(m.group(1)), cc.MUJOCO_ENV)


class Measurements(unittest.TestCase):
    def test_bore(self):
        with tempfile.TemporaryDirectory() as t:
            for bore, want in ((.0054, .4), (.0058, .8), (.0066, 1.6)):
                got = cc.measure_bore(bore_scene(pathlib.Path(t) / str(bore), bore))
                self.assertAlmostEqual(got["clearance_mm"], want, places=9)
                self.assertAlmostEqual(got["shank_radius_mm"], 5.0, places=9)

    def test_lock_bore(self):
        with tempfile.TemporaryDirectory() as t:
            for bore, shaft, want in ((.0075, .0065, 1.), (.0105, .0065, 4.), (.00975, .00675, 3.)):
                got = cc.measure_lock_bore(lock_scene(pathlib.Path(t) / f"{bore}-{shaft}", bore, shaft))
                self.assertAlmostEqual(got["clearance_mm"], want, places=9)
                self.assertEqual(got["plates"], 2)

    def test_coil_gap(self):
        with tempfile.TemporaryDirectory() as t:
            for gap in (.0015, .003, .006):
                got = cc.measure_coil_gap(coil_scene(pathlib.Path(t) / str(gap), .035, gap))
                self.assertAlmostEqual(got["clearance_mm"], gap * 1e3, places=9)
                self.assertAlmostEqual(got["cell_mm"], 35.0, places=9)

    def test_coil_texts(self):
        """The coil's stated tolerance follows the factor, in the instruction and in the generated goal text."""
        r = cc.RELAXATIONS["roboarc-cast-coil-07-i01"]
        for k in cc.FACTORS:
            (old, new), = r.texts[k]
            self.assertIn("below 1 mm", old)
            self.assertIn(f"below {k} mm", new)
            self.assertIn(f"< .001 * {k} and", r.edits[1].new[k])

    def test_coil_quoted_source(self):
        """The privileged instruction quotes evaluate_state verbatim: the quote shows the relaxed tolerance line, the
        standard one (no quote) only the restated sentence."""
        r = cc.RELAXATIONS["roboarc-cast-coil-07-i01"]
        with tempfile.TemporaryDirectory() as t:
            for mode in ("privileged", "standard"):
                d = pathlib.Path(t) / mode
                d.mkdir()
                for f in ("instruction.md", "README.md"):
                    shutil.copy(REPO / "tasks" / f"roboarc-cast-coil-07-i01-{mode}" / f, d / f)
                cc._texts(d, r, 4)
                text = (d / "instruction.md").read_text()
                self.assertNotIn("position_error < .001 and", text)
                self.assertEqual(text.count("position_error < .001 * 4 and"), mode == "privileged")
                self.assertEqual(text.count("relative position error below 4 mm"), 1)
                self.assertNotIn("relative position error below 1 mm", text + (d / "README.md").read_text())

    def test_lock_texts(self):
        """The lock texts' numbers: the released ones from the formula, and the relaxed bore in every sentence."""
        for base, released in (("roboarc-lock-picking-04-i01", ("1.01 mm", "0.31 degrees")),
                               ("roboarc-lock-picking-05-i01", ("0.76 mm", "0.18 degrees"))):
            r = cc.RELAXATIONS[base]
            for k in cc.FACTORS:
                (old1, new1), (old2, new2) = r.texts[k]
                self.assertIn(released[0], old2)
                self.assertIn(released[1], old2)
                bore = 7.5 + (k - 1) * r.clearance_mm
                self.assertIn(f"bore radius {bore:g} mm", new1)
                self.assertIn(f"may exceed {bore:g} mm", new2)


class Make(unittest.TestCase):
    """make() on copies of a released task, with a stand-in for the frozen relaxed instance."""
    BASE = "roboarc-bed-bolt-01-i01"

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="test-e4-"))
        r = cc.RELAXATIONS[self.BASE]
        fz = self.tmp / "frozen"
        source = dict(sha256="ab" * 32, files=455)
        for e in r.edits:
            (fz / "topogym" / e.file.split("/", 1)[1]).parent.mkdir(parents=True)
            (fz / "topogym" / e.file.split("/", 1)[1]).write_text("RB = .0058  # " + e.new[2] + "\n")
        (fz / "SOURCE.json").write_text(json.dumps(source, indent=1) + "\n")
        bore_scene(fz / "instance" / "instance_assets" / "mujoco", .0058)
        (fz / "instance" / "instance.json").write_text(json.dumps(dict(state_sha="0123456789abcdef", topogym_source=source)))
        self.frozen_dir = fz
        self._frozen = cc.frozen
        cc.frozen = lambda base, factor: fz

    def tearDown(self):
        cc.frozen = self._frozen
        shutil.rmtree(self.tmp)

    def build(self, mode: str, out: pathlib.Path) -> pathlib.Path:
        src = REPO / "tasks" / f"{self.BASE}-{mode}"
        dst = out / f"{self.BASE}-clr2x-{mode}"
        shutil.copytree(src, dst, symlinks=True)
        cc.make(src, dst, 2)
        return dst

    def test_privileged(self):
        d = self.build("privileged", self.tmp / "a")
        name = cc.RELAXATIONS[self.BASE].instance_name()
        for ctx in ("environment", "tests"):
            df = (d / ctx / "Dockerfile").read_text()
            self.assertNotIn("runtime_binding.s1.json", df)
            for line in (f"COPY {cc.E4}/topogym/ /opt/topogym/topogym/", f"COPY {cc.E4}/SOURCE.json /opt/topogym/SOURCE.json",
                         f"RUN rm -rf /opt/robotangle/instances/{name}",
                         f"COPY {cc.E4}/instance/ /opt/robotangle/instances/{name}/"):
                self.assertIn(line + "\n", df)
            self.assertLess(df.index(f"COPY {cc.E4}/instance/"), df.index("stage_instance.py"))
            self.assertTrue((d / ctx / cc.E4 / "instance" / "instance.json").is_file())
        self.assertEqual(sorted(d.rglob("runtime_binding.s*.json")), [])
        self.assertEqual(sorted((d / "solution").glob("trajectory.s*.npz")), [])
        with zipfile.ZipFile(d / "solution" / "trajectory.npz") as z:
            self.assertIn("0123456789abcdef", z.read("instance_sha.npy").decode("utf-32-le", "ignore"))
        self.check_bindings(d)
        self.check_toml(d)
        self.assertEqual(cc.leak_check(d), [])
        self.assertTrue(cc.check(d, 2)["ok"])

    def test_standard(self):
        d = self.build("standard", self.tmp / "a")
        name = cc.RELAXATIONS[self.BASE].instance_name()
        compose = (d / "environment" / "docker-compose.yaml").read_text()
        rel = cc.RELAXATIONS[self.BASE].edits[0].file.split("/", 1)[1]
        for m in (f"./sim/{cc.E4}/topogym/{rel}:/opt/topogym/topogym/{rel}:ro", f"./sim/{cc.E4}/SOURCE.json:/opt/topogym/SOURCE.json:ro",
                  f"./sim/{cc.E4}/instance:/opt/robotangle/instances/{name}:ro"):
            self.assertIn(m, compose)
        self.assertNotIn("runtime_binding.s1.json", (d / "tests" / "Dockerfile").read_text())
        self.assertEqual((d / "environment" / "Dockerfile").read_text(),
                         (REPO / "tasks" / f"{self.BASE}-standard" / "environment" / "Dockerfile").read_text())
        self.assertEqual((d / "instruction.md").read_text(),
                         (REPO / "tasks" / f"{self.BASE}-standard" / "instruction.md").read_text())
        self.assertEqual(sorted(d.rglob("runtime_binding.s*.json")), [])
        self.check_bindings(d)
        self.check_toml(d)
        self.assertEqual(cc.leak_check(d), [])
        self.assertTrue(cc.check(d, 2)["ok"])

    def test_standard_leak_check_reads_the_released_instruction(self):
        out = self.tmp / "tasks"
        shutil.copytree(REPO / "tasks" / f"{self.BASE}-standard", out / f"{self.BASE}-standard", symlinks=True)
        d = self.build("standard", out)
        self.assertEqual(cc.leak_check(d), [])
        with open(d / "instruction.md", "a") as f:
            f.write("The bolt starts at x = 0.057 m.\n")
        self.assertEqual(cc.leak_check(d), ["the instruction differs from the released task's beyond the restated dimensions"])

    def test_deterministic(self):
        a, b = self.build("privileged", self.tmp / "a"), self.build("privileged", self.tmp / "b")
        self.assertEqual(cc._files(a), cc._files(b))

    def check_bindings(self, d: pathlib.Path):
        inst = json.loads((self.frozen_dir / "instance" / "instance.json").read_text())
        src = json.loads((self.frozen_dir / "SOURCE.json").read_text())
        bindings = sorted(d.rglob("runtime_binding.json"))
        self.assertGreaterEqual(len(bindings), 2)
        for b in bindings:
            lock = json.loads(b.read_text())
            self.assertEqual(b.read_text(), json.dumps(lock, indent=2) + "\n")
            self.assertEqual(lock["instance_sha256"], cc.canonical(inst))
            self.assertEqual(lock["creation_source"], src)
            self.assertEqual(lock["runtime"]["source"], src)
            self.assertEqual(sorted(lock["asset_files"]), ["instance_assets/mujoco/meshes/shank.obj",
                                                           "instance_assets/mujoco/scene.xml"])

    def check_toml(self, d: pathlib.Path):
        meta = tomllib.loads((d / "task.toml").read_text())["metadata"]
        self.assertEqual((meta["seeds"]["default"], meta["seeds"]["allowed"], meta["seeds"]["oracle"]), (0, [0], [0]))
        src = tomllib.loads((REPO / "tasks" / d.name.replace("-clr2x", "") / "task.toml").read_text())["metadata"]
        self.assertEqual({k: v for k, v in meta.items() if k != "seeds"}, {k: v for k, v in src.items() if k != "seeds"})


if __name__ == "__main__":
    unittest.main()
