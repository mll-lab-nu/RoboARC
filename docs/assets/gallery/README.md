# README gallery

The twelve GIFs at the top of the repository's README, four rows of three. Each one is cut from the studio clip of a
task's reference solution: a path-traced replay of the reference trajectory, played as a time-lapse with a slow camera
orbit. [`make_gallery.py`](make_gallery.py) holds the cut parameters and remakes all twelve.

| property | value |
|---|---|
| size | 320 x 180 pixels (16:9) |
| frames, rate, length | 60 frames, 10 per second, 6.0 s, loops forever |
| end of the loop | frames 0 to 51 run through the window; frames 52 to 59 repeat frame 51, the solved state |
| colours | one palette for the whole clip, 256 colours (`cast-coil-08`: 192), no dithering |
| file size | 0.56 to 0.95 MB each, 9.4 MB for the twelve |

## The twelve

In README order. Each constraint type appears twice. Robots: ALOHA 2 seven times, Franka + Robotiq 2F-85 three times,
two Franka arms twice.

| # | GIF | task | constraint type | robot | master | window (s) | speed-up | crop F, (X, Y) | shows |
|---:|---|---|---|---|---|---|---:|---|---|
| 1 | `interlocked-claws-05.gif` | `roboarc-interlocked-claws-05-i01` | Interlocking | ALOHA 2 | `interlocked_claws_05/video.mp4` | 22 to 41.3 | 3.8 | 0.86, (0.06, 0.14) | the last twists, the two claws come apart and are held apart |
| 2 | `hedgehog-in-cage-03.gif` | `roboarc-hedgehog-in-cage-03-i01` | Narrow passage | Franka + Robotiq 2F-85 | `hedgehog_in_cage_03/video.mp4` | 15.5 to 26.7 | 2.2 | 0.8, (0.07, 0.09) | the spiked ball is turned inside the copper cage, pulled out between the bars and set down |
| 3 | `ikea-gladom.gif` | `roboarc-ikea-gladom-i00` | Assembly | two Franka arms | `ikea_gladom/video.mp4` | 14 to 21.3 | 1.4 | 0.78, (0.1, 0.13) | the wire frame is finished and the tray is lifted from the floor onto it |
| 4 | `knot-untying-04.gif` | `roboarc-knot-untying-04-i01` | Untangling | ALOHA 2 | `knot_untying_04/video.mp4` | 1.8 to 16.3 | 2.8 | 0.76, (0.09, 0.09) | the knot is loosened and the rope pulled straight |
| 5 | `rope-threading-08.gif` | `roboarc-rope-threading-08-i01` | Threading | ALOHA 2 | `rope_threading_08/video.mp4` | 12 to 31.5 | 3.8 | 0.76, (0.1, 0.02) | the rope is passed through the wall grommets and the rings, the coil on the floor shrinks |
| 6 | `blanket-fold-03.gif` | `roboarc-blanket-fold-03-i01` | Shape control | ALOHA 2 | `blanket_fold_03/video.mp4` | 23 to 31.1 | 1.6 | 0.8, (0.11, 0.04) | the flat blanket is folded in half |
| 7 | `cast-coil-08.gif` | `roboarc-cast-coil-08-i01` | Interlocking | ALOHA 2 | `cast_coil_08/video.mp4` | 22 to 41.3 | 3.8 | 0.84, (0.12, 0.16) | the gold and the steel piece slide into each other and the joined puzzle is set down |
| 8 | `hedgehog-in-cage-06.gif` | `roboarc-hedgehog-in-cage-06-i01` | Narrow passage | Franka + Robotiq 2F-85 | `hedgehog_in_cage_06/video.mp4` | 0.5 to 15.9 | 3.0 | 0.9, (0.05, 0.08) | the spiked ball is worked up through the black cast cage, taken out at the top and set down |
| 9 | `slotted-post-ring-05.gif` | `roboarc-slotted-post-ring-05-i01` | Untangling | ALOHA 2 | `slotted_post_ring_05/video.mp4` | 18 to 30.7 | 2.5 | 0.92, (0.03, 0.06) | the blue ring is worked off the beaded cord and placed in the goal region |
| 10 | `ikea-vesken.gif` | `roboarc-ikea-vesken-i00` | Assembly | two Franka arms | `ikea_vesken/video.mp4` | 22 to 41.3 | 3.8 | 0.76, (0.12, 0.1) | the shelf unit grows from one shelf to three |
| 11 | `fiber-bundling-08.gif` | `roboarc-fiber-bundling-08-i01` | Threading | Franka + Robotiq 2F-85 | `fiber_bundling_08/video.mp4` | 6 to 20.5 | 2.8 | 0.8, (0.02, 0.1) | the fibers at the cable's end are gathered and pushed through the ring |
| 12 | `blanket-flatten-04.gif` | `roboarc-blanket-flatten-04-i01` | Shape control | ALOHA 2 | `blanket_flatten_04/video.mp4` | 1 to 16.3 | 3.0 | 0.8, (0.09, 0.05) | the wrinkled blanket is lifted, spread and laid flat |

- **GIF**: the file name is the task base without `roboarc-` and without the instance suffix (`-i01`, `-i00`).
- **task**: the task base; its two tasks are `tasks/roboarc-<base>-{privileged,standard}`. Constraint types and robots
  are those of the repository's README and [TAXONOMY.md](../../TAXONOMY.md).
- **master**: the source video, below
  `https://huggingface.co/datasets/MLL-Lab/embodied-exam-data/resolve/main/masters/roboarc/` (public, no login).
  2560 x 1440 or 3840 x 2160, 30 frames per second, Blender Cycles. The `meta.json` beside each video names the
  reference file it was rendered from (`reference_sha256`, `rows`) and the simulated time the clip covers.
- **window**: start and end in seconds of the master. Every window ends inside the master's final still (its last
  1.6 s or more show the final state), so frame 51 of the GIF is the solved state.
- **speed-up**: of the GIF against the master. The master is already a time-lapse of the simulation.
- **crop**: width fraction `F` and top-left corner `(X, Y)`, in fractions of the master's frame. The aspect stays 16:9
  and the same numbers hold for both master sizes. Every crop keeps the whole robot in the frame for the whole window
  (in `interlocked-claws-05` and `cast-coil-08` the master's own frame ends at the base of the right arm).

## Remaking them

Needs `ffmpeg` and `ffprobe` (4.4 or newer) and Python 3.11 or newer, nothing else.

```bash
python3 docs/assets/gallery/make_gallery.py                        # all twelve; downloads the masters (240 MB) once
python3 docs/assets/gallery/make_gallery.py --masters DIR          # masters already on disk
python3 docs/assets/gallery/make_gallery.py --only knot-untying-04 # one GIF
```

Without `--masters` the videos are kept in `~/.cache/roboarc-gallery/` (`--cache` changes it). With `--masters DIR`
a video is read from `DIR/<id>/video.mp4` or `DIR/roboarc-<base>.mp4`. Each master is checked against the SHA-256
recorded in the script before it is cut. Expected output, one line per GIF (sizes may differ by a few kilobytes with
another ffmpeg version):

```text
interlocked-claws-05.gif     60 frames  0.81 MB  x3.8
hedgehog-in-cage-03.gif      60 frames  0.78 MB  x2.2
...
blanket-flatten-04.gif       60 frames  0.89 MB  x3.0
```

The script exits with 1 if a GIF is not 320 x 180, does not have 60 frames at 10 per second, does not loop forever or
is over 1 MB. The same test on the files already here, without ffmpeg and without the masters, together with the task
names and the table above:

```bash
python3 docs/assets/gallery/make_gallery.py --check
```

```text
ok make_gallery --check: 12 GIF(s), 0 finding(s)
```

## Changing one

1. Look at a frame of the new window and crop first (here 30 s into a master, crop `0.86, (0.06, 0.14)`):

   ```bash
   ffmpeg -ss 30 -i video.mp4 -frames:v 1 \
     -vf "crop=iw*0.86:ih*0.86:iw*0.06:ih*0.14,scale=640:360:flags=lanczos" preview.png
   ```

2. Edit the entry in `GALLERY` in `make_gallery.py` (a new task also needs the master's SHA-256), run the script with
   `--only <name>` and look at the GIF: the whole robot in the frame, the task readable at 320 pixels, the last
   frame the solved state.
3. If the file is over 1 MB, give the entry fewer colours (`colors`, 192 then 128) or a tighter crop.
4. Print the table with `--table` and replace the one above, update the `<img>` line in the repository's README, and
   run `--check`.
