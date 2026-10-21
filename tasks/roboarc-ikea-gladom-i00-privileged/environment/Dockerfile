# Base image = plain MuJoCo 3.14.0 and the IKEA scenes (ikea/images/ikea-mujoco/). Everything task-specific is copied here.
FROM ghcr.io/mll-lab-nu/roboarc-ikea:1.1.0@sha256:2c56d9226aaf97a325b610d464a1cd81e87db1a6285e923539397f493e21136b
COPY rcb_ikea.py /opt/tools/rcb_ikea.py
ENV PYTHONPATH=/opt/tools
RUN mkdir -p /app/output
WORKDIR /app
