# Base image = the RoboARC release image (images/roboarc/). Everything task-specific is copied here.
FROM ghcr.io/mll-lab-nu/roboarc:1.2.0@sha256:a8561b770908d2e9289390ff31992adeb35d8eee7453560177b57386291174a3
# MuJoCo task: numpy / OpenBLAS SIMD masking, so every CPU class generates the frozen bytes
ENV NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR OPENBLAS_CORETYPE=Haswell
COPY runtime_binding.json runtime_binding.s1.json runtime_binding.s2.json seed /tmp/rcb-seed/
RUN python /opt/robotangle/stage_instance.py knot-untying-01-i01 /app --binding /tmp/rcb-seed/runtime_binding.json --seed-file /tmp/rcb-seed/seed --purge && cp /tmp/rcb-seed/seed /opt/robotangle/seed && rm -rf /tmp/rcb-seed   # the frozen instance, from the image (scripts/roboarc/assets.json), checked against runtime_binding.json
COPY rcb_topogym.py /opt/tools/rcb_topogym.py
COPY render_bridge.py /opt/tools/render_bridge.py
COPY nyx_direct.py /opt/tools/nyx_direct.py
ENV PYTHONPATH=/opt/tools:/opt/topogym
# the longest trajectory the verifier replays within its timeout (scripts/roboarc/mk_tasks.py row_cap)
ENV RCB_MAX_STEPS=13000
RUN mkdir -p /app/output
WORKDIR /app
