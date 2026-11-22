# roboarc-monitor: the runtime of the resource-limits monitor sidecar (service `rcb_monitor` in every task;
# docs/PROTOCOL.md §2.10). Python and nothing else: the monitor's code is not in the image, it is mounted read-only from
# the task directory (environment/rcb_monitor/, part of the task digest). NVIDIA_DRIVER_CAPABILITIES=utility makes the
# NVIDIA runtime / CDI inject libnvidia-ml when the task reserves a GPU. Built reproducibly by images/monitor/build.sh
# (README.md here); tag = images/monitor/VERSION; tasks pin the manifest digest (scripts/rcb_monitor/inject.py).
# The base is pinned by digest: Python 3.12.15 on Debian bookworm, the same filesystem layers as the first monitor image.
FROM docker.io/library/python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ARG VERSION
LABEL org.opencontainers.image.title="roboarc-monitor" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/Embodied-First-Exam/RoboARC"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NVIDIA_DRIVER_CAPABILITIES=utility
WORKDIR /var/rcb-monitor
