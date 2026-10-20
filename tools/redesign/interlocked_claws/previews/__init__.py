"""previews: CPU-only MuJoCo offscreen previews (EGL on software Mesa / llvmpipe; matplotlib fallback) of a MuJoCo
model + state from fixed views (front, side, top, the ALOHA's cam_high), downscaled PNG <= 800 px wide, plus a contact
sheet. Never touches the GPU (CUDA_VISIBLE_DEVICES empty, the Mesa EGL vendor forced). See previews.render."""
