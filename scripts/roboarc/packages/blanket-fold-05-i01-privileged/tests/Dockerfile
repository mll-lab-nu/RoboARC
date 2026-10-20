# Separate verifier sandbox (PROTOCOL §4): the same simulator image with the verifier's own copies of the
# tooling and the frozen instance baked in. Nothing the agent did in its container reaches this one — Harbor
# uploads only the declared artifacts (/app/output/), so a patched success check or a swapped interpreter in
# the agent container cannot influence the verdict.
FROM {{IMAGE}}
# MuJoCo task: numpy / OpenBLAS SIMD masking, so every CPU class generates the frozen bytes
ENV NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR OPENBLAS_CORETYPE=Haswell
COPY runtime_binding.json /tests/runtime_binding.json
COPY grade.py rcb_topogym.py render_bridge.py nyx_direct.py test.sh /tests/
RUN python /opt/robotangle/stage_instance.py blanket-fold-05-i01 /tests --purge   # the frozen instance, from the image (scripts/roboarc/assets.json), checked against runtime_binding.json
RUN chmod +x /tests/test.sh && mkdir -p /app/output /logs/verifier
ENV PYTHONPATH=/tests:/opt/topogym
# the longest trajectory the verifier replays within its timeout (scripts/roboarc/mk_tasks.py row_cap)
ENV RCB_MAX_STEPS={{ROW_CAP}} RCB_VERIFIER_BUDGET_S={{VERIFIER_BUDGET_S}}
WORKDIR /tests
