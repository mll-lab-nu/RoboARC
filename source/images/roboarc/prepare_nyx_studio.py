"""Download the CC0 studio HDRI and texture maps for TopoGym's runtime Nyx RGB.

Source: https://polyhaven.com/a/studio_small_09 (Sergej Majboroda); the table's wood maps (wood_table_001) and the
appearance seeds' table top, frame and floor finishes (topogym/world/look.py: TEXTURES below), all CC0 Poly Haven
assets, each file checked by sha256 (the bytes dl.polyhaven.org serves; the textures' also matched Poly Haven's API
md5 on 2026-10-27).
"""

import hashlib
import os
from pathlib import Path
import urllib.request

PH = "https://dl.polyhaven.org/file/ph-assets"
# The appearance seeds' textures, the names topogym/world/look.py's finishes use: name -> (title, authors, the sha256 of
# its 1k JPG colour, OpenGL normal and ARM (ambient occlusion, roughness, metalness) maps, MAPS order)
MAPS = ("diff", "nor_gl", "arm")
TEXTURES = {
    "ash_veneer": ("Ash Veneer", "Jenelle van Heerden", (
        "13fd87c8b0a5cc1d4e408fb75b3eb629417ea6b1aaf1672b5adf993640486673",
        "824898e187fb1d701b36a898ecd736a0464f49fa46233a35592e1ed9348e7fc4",
        "b74cdcb4a3383ac6a35b180d24c64a1762aada219dac7da231b9c84db7ac5597")),
    "cherry_veneer": ("Cherry Veneer", "Jenelle van Heerden", (
        "d50fed9f9964721d6b18581cadb6d95717a714b85361b22078e7d913ba5ce4d2",
        "d2c722c80651e7c1074d46f27c18a78de5bce9ced1c8afc82ea3aa555555e57a",
        "24f0a083f3e3f798b096994f4e05e186e6df71faa8f46f046d1ec93bbe45350c")),
    "coated_pine": ("Coated Pine", "Charlotte Baglioni and Rico Cilliers", (
        "afee7cfb2ec1716edc712f929cdb3d3dfbac5a0ba290e25943b4472a8144bbaf",
        "91eb707d7b0dd44133502878f37ede93e6be23c2019e64df72dffe46f6985b3a",
        "1df6001b7920389729415068ffe3ac473570f37be35ec628e895da2daa68d8f0")),
    "terrazzo_tiles": ("Terrazzo Tiles", "Amal Kumar", (
        "462e3e719cda0472a33152523cf5ca2fc660e1c5869348be1609ba3dca12a2ef",
        "56bbe2d960b47ddb2dc538be2953a1ad07083b017cdada1cdc7958421a6cfcd9",
        "f6bc98a79c33010b54759c5dba5f915179a74b78d46f7b0d03ecf4feb6347c0e")),
    "brushed_concrete": ("Brushed Concrete", "Dario Barresi and Dimitrios Savva", (
        "4054ed6eb65cd063356ffb4a88706b7d7b78c5435051c48bf0ee64290f04bc8c",
        "f4953a881f34627cefcc22f780d392314635e31bb41f75cf3888a5af55c21820",
        "374e4408acfb0bb4512f904f6fd3ae1006195ed12f2e258625b5b8b3a91f6528")),
    "granite_tile_03": ("Granite Tile 03", "Charlotte Baglioni", (
        "bdb28bea17413f1a05b0498b43e9f61f253bb72d753d3bf6b555cbf5f3c9b1b4",
        "016a678038f885f58fdb2ab642835dddd5a4d1302dbec8cb656e37ddd17dbbff",
        "13e2e13adc697b9e70a7a1f902ba5111ca518560ebbd5eaed9fd05c0ccc51c28")),
    "diagonal_parquet": ("Diagonal Parquet", "Jenelle van Heerden and Sergej Majboroda", (
        "dd19af5da746c15c68192dea8ad8374111cc5dd04bc8d715c70dc50de0f48022",
        "a6b94b86714547d9dbbc0b040939d541857a200d610144e55ce935635925f193",
        "3964f450336a96a01686013600904d59ab3225c1d265e720f274c18eca700564")),
    "interior_tiles": ("Interior Tiles", "Charlotte Baglioni", (
        "f267c4357e629f5ca2ab86f30da4fb0ae8c655a566cec6d81278df8d78dcca4a",
        "fdfb9fe192d438176508290fe873e181d88c5e6444a43bbab2ec156c9a78aa33",
        "e21a6758bb84ffce7baf0dd5937f77fa9e40c94d4d6b779097b1c5dd6189d81b")),
    "terracotta_floor_tiles": ("Terracotta Floor Tiles", "Dimitrios Savva", (
        "966ff423065e4a4ce9490864fba61fdb491985d4a2c60eb2b85181c7e7edcd5d",
        "7d9a6935951f3b2c1f6ae372dd512969dd5c7c27efbbfe6941ed3ec1b7da9758",
        "e518e200a45a630f5e66be477c21b8694eeb40b6d3dab9c6b13e5401817ed9a6")),
    "poly_wool_herringbone": ("Poly Wool Herringbone", "colormass and Rico Cilliers", (
        "fd889a46f617efd331e522e36b8ad2548e16e1cb018c001872f17537d7666af5",
        "314ac7c1b7c101a48106e05199bfdb24e33671e58a6ccb334249b441d8e8ac98",
        "f565985fe68d42b6638080c52628f13312b178001d758ef89fe76adb773fb227")),
}
# every file: name -> (url, sha256)
FILES = {
    "studio_small_09_2k.hdr": (f"{PH}/HDRIs/hdr/2k/studio_small_09_2k.hdr",
                               "36724313c0fc66dab126ff90f081b85a0b1ef47be65eda1417bd20b033f0573f"),
    "wood_table_001_diff_1k.png": (f"{PH}/Textures/png/1k/wood_table_001/wood_table_001_diff_1k.png",
                                   "aef45ab36d9f258b80edf278d417920b7326ad77c341d6618498d4598c59fd2c"),
    "wood_table_001_nor_gl_1k.png": (f"{PH}/Textures/png/1k/wood_table_001/wood_table_001_nor_gl_1k.png",
                                     "3d1dc2cc14cd9a3d2bc0035560aa715f4f8786902b4b72e78135f1d097428f98"),
    **{f"{name}_{m}_1k.jpg": (f"{PH}/Textures/jpg/1k/{name}/{name}_{m}_1k.jpg", checksum)
       for name, (_, _, sums) in TEXTURES.items() for m, checksum in zip(MAPS, sums)},
}
CREDITS = (
    "Studio Small 09 by Sergej Majboroda / Poly Haven\n"
    "https://polyhaven.com/a/studio_small_09\nLicense: CC0\n\n"
    "Wood Table 001 by Dimitrios Savva and Rico Cilliers / Poly Haven\n"
    "https://polyhaven.com/a/wood_table_001\nLicense: CC0\n"
    + "".join(f"\n{title} by {authors} / Poly Haven\nhttps://polyhaven.com/a/{name}\nLicense: CC0\n"
              for name, (title, authors, _) in TEXTURES.items()))


def main():
    # Keep this installation-independent path contract aligned with Nyx's
    # asset_path(); the download utility does not import the simulator engine.
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser()
    root = Path(os.environ.get("TOPOGYM_VISUAL_ASSET_ROOT", str(cache / "topogym/visuals/nyx")))
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "CREDITS.txt").write_text(CREDITS)
    for name, (url, checksum) in FILES.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != checksum:
                raise RuntimeError(f"checksum mismatch: {name}")
            path.write_bytes(data)
    print(f"Nyx studio and texture assets ready: {root} ({len(FILES)} files)")


if __name__ == "__main__":
    main()
