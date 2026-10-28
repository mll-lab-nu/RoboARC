#!/usr/bin/env python3
"""The image's visual assets (/opt/topogym-visuals, source/images/roboarc/prepare_nyx_studio.py) hold every map the
renderer configuration names: the appearance seeds' finishes (source/topogym/world/look.py TABLE_FINISHES,
FLOOR_FINISHES, FRAME_FINISHES: diffuse_map, normal_map, arm_map) and compact_lab's environment map and table maps;
every texture set has its line in CREDITS.txt and every download a sha256. Needs numpy (look.py imports it):

    python tools/tests/test_visual_assets.py         # in the release image, or anywhere with numpy
"""
import importlib.util
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


studio = load("prepare_nyx_studio", ROOT / "source/images/roboarc/prepare_nyx_studio.py")
look = load("look", ROOT / "source/topogym/world/look.py")      # by path: the topogym package is not imported


class VisualAssets(unittest.TestCase):
    def test_every_named_map_is_prepared(self):
        named = {(table, name, key): finish[key]
                 for table in ("TABLE_FINISHES", "FLOOR_FINISHES", "FRAME_FINISHES")
                 for name, finish in getattr(look, table).items()
                 for key in ("diffuse_map", "normal_map", "arm_map") if key in finish}
        self.assertTrue(named)
        missing = {k: v for k, v in named.items() if v not in studio.FILES}
        self.assertEqual(missing, {}, "maps look.py names that the image does not prepare")
        lab = (ROOT / "source/topogym/config/profiles/presentation_configs/compact_lab.yaml").read_text()
        maps = re.findall(r"(?m)^\s*(?:env_map|diffuse_map|normal_map|arm_map):\s*(\S+)\s*$", lab)
        self.assertTrue(maps)
        self.assertEqual([m for m in maps if m not in studio.FILES], [])
        # no texture set the finishes leave unused
        used = {re.sub(r"_(diff|nor_gl|arm)_1k\.jpg$", "", v) for v in named.values()}
        self.assertEqual(set(studio.TEXTURES) - used, set())

    def test_credits_and_checksums(self):
        for name, (title, authors, _) in studio.TEXTURES.items():
            self.assertIn(f"{title} by {authors} / Poly Haven\nhttps://polyhaven.com/a/{name}\nLicense: CC0\n",
                          studio.CREDITS)
        for name, (url, checksum) in studio.FILES.items():
            self.assertTrue(url.startswith("https://dl.polyhaven.org/") and url.endswith("/" + name), url)
            self.assertRegex(checksum, r"^[0-9a-f]{64}$", name)
        self.assertEqual(len(set(c for _, c in studio.FILES.values())), len(studio.FILES))


if __name__ == "__main__":
    unittest.main()
