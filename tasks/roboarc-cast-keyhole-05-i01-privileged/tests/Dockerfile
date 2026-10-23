# Separate verifier sandbox (PROTOCOL §4): the same simulator image with the verifier's own copies of the
# tooling and the frozen instance baked in. Nothing the agent did in its container reaches this one — Harbor
# uploads only the declared artifacts (/app/output/), so a patched success check or a swapped interpreter in
# the agent container cannot influence the verdict.
FROM ghcr.io/mll-lab-nu/roboarc:1.3.0@sha256:2bcded002899b4be82dcffe5c0d15277379309ff0365522d8189f1f3a7a4873a
# MuJoCo task: numpy / OpenBLAS SIMD masking, so every CPU class generates the frozen bytes
ENV NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR OPENBLAS_CORETYPE=Haswell
COPY runtime_binding.json runtime_binding.s1.json runtime_binding.s2.json seed /tmp/rcb-seed/
COPY grade.py rcb_topogym.py render_bridge.py nyx_direct.py test.sh /tests/
RUN python /opt/robotangle/stage_instance.py cast-keyhole-05-i01 /tests --binding /tmp/rcb-seed/runtime_binding.json --seed-file /tmp/rcb-seed/seed --purge && rm -rf /tmp/rcb-seed   # the frozen instance, from the image (scripts/roboarc/assets.json), checked against runtime_binding.json
RUN chmod +x /tests/test.sh && mkdir -p /app/output /logs/verifier
ENV PYTHONPATH=/tests:/opt/topogym
# the longest trajectory the verifier replays within its timeout (scripts/roboarc/mk_tasks.py row_cap)
ENV RCB_MAX_STEPS=200000 RCB_VERIFIER_BUDGET_S=45000
WORKDIR /tests
# resource limits: the verifier step and its copy of the limits (docs/PROTOCOL.md §2.10) # rcb-monitor
COPY rcb_limits.py rcb_limits.json /tests/
