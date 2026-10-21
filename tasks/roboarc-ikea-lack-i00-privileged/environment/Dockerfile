# Base image = plain MuJoCo 3.14.0 and the IKEA scenes (ikea/images/ikea-mujoco/). Everything task-specific is copied here.
FROM ghcr.io/embodied-first-exam/roboarc-ikea:1.1.0
COPY rcb_ikea.py /opt/tools/rcb_ikea.py
ENV PYTHONPATH=/opt/tools
RUN mkdir -p /app/output
WORKDIR /app
