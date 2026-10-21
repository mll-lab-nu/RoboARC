# Separate verifier sandbox (PROTOCOL §4): the same image with the verifier's own copies of the tooling and the scene's
# binding. Harbor uploads only the declared artifacts (/app/output/), so nothing the agent changed in its container
# reaches the verdict.
FROM ghcr.io/embodied-first-exam/roboarc-ikea:1.1.0@sha256:2c56d9226aaf97a325b610d464a1cd81e87db1a6285e923539397f493e21136b
# Replay video (PROTOCOL §4; evidence only): MuJoCo's renderer on CPU software OpenGL (OSMesa) and ffmpeg, both in the
# base image (pinned there; ikea/images/ikea-mujoco/Dockerfile), so building this image installs nothing.
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
COPY grade.py rcb_ikea.py render_video.py video_view.json test.sh runtime_binding.json /tests/
RUN chmod +x /tests/test.sh && mkdir -p /app/output /logs/verifier
ENV PYTHONPATH=/tests
WORKDIR /tests
# resource limits: the verifier step and its copy of the limits (docs/PROTOCOL.md §2.10) # rcb-monitor
COPY rcb_limits.py rcb_limits.json /tests/
