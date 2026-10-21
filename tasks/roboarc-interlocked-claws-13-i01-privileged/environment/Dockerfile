# Base image = TopoGym's simulator (see images/topogym/). Everything task-specific is copied here.
FROM ghcr.io/embodied-first-exam/roboarc:1.2.0
# MuJoCo task: numpy / OpenBLAS SIMD masking, so every CPU class generates the frozen bytes
ENV NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR OPENBLAS_CORETYPE=Haswell
# the task's runtime bindings, one per seed, and in a runner's copy for one seed its `seed` file (scripts/task_seed.py
# materialize; `see[d]` matches that file or nothing)
COPY runtime_binding*.json see[d] /opt/robotangle/task/
RUN python /opt/robotangle/stage_instance.py interlocked-claws-13-i01 /app --binding /opt/robotangle/task/runtime_binding.json --seed-file /opt/robotangle/task/seed --purge && rm -rf /opt/robotangle/task   # the frozen instance of the task's seed (no seed file: the default one), from the image (scripts/roboarc/assets.json), checked against that seed's runtime binding
COPY rcb_topogym.py /opt/tools/rcb_topogym.py
COPY render_bridge.py /opt/tools/render_bridge.py
COPY nyx_direct.py /opt/tools/nyx_direct.py
ENV PYTHONPATH=/opt/tools:/opt/topogym
# the longest trajectory the verifier replays within its timeout (scripts/roboarc/mk_tasks.py row_cap)
ENV RCB_MAX_STEPS=200000
RUN mkdir -p /app/output
WORKDIR /app
