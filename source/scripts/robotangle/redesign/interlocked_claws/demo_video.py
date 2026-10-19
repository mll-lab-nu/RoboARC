"""Demo video of an oracle: replay its rows in a fresh Session of the tier and render every `every`-th row from two
views (ALOHA overview + close-up on the lock), side by side, with the phase name and the row: python work/demo_video.py
<tier> <privileged.npz> <out.mp4> [--every 6] [--fps 20]"""
import json, sys
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from previews_claws import render, colour_pieces
from topogym.tasks.interlocked_claws.redesign.task import make_session

tier, src, out = sys.argv[1:4]
every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 6
fps = int(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 20
z = np.load(src, allow_pickle=False)
rows = z["actions"]
meta = json.loads(str(z["meta"]))
bounds, n = [], 0
for ph in meta["phases"]:
    n += ph["rows"]; bounds.append((n, ph["name"]))
s = make_session(tier)
m, data = s.env._world.model, s.env._world.data
colour_pieces(m)
st = s.env._world.read_state()
close_at = np.asarray(st["bodies"]["piece_1"]["position_m"], float)
w = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=8)
def frame(k):
    imgs = render(m, data, views=("aloha", "close"), w=640, h=400, close_at=close_at)
    a, b = Image.fromarray(imgs["aloha"]), Image.fromarray(imgs["close"])
    S = Image.new("RGB", (1280, 400))
    S.paste(a, (0, 0)); S.paste(b, (640, 0))
    ph = next((nm for e, nm in bounds if e >= k), bounds[-1][1])
    ImageDraw.Draw(S).text((10, 8), f"{tier}  row {k}/{len(rows)}  phase: {ph}" + ("   SUCCESS" if s.success() else ""),
                           fill=(255, 255, 255))
    w.append_data(np.asarray(S))
frame(0)
for k, r in enumerate(rows, 1):
    s.step(np.asarray(r, np.float32))
    if k % every == 0 or k == len(rows):
        frame(k)
for _ in range(fps):
    frame(len(rows))
w.close()
print("VIDEO", out, s.status()["success"], s.status()["state_sha"])
