# Separate verifier sandbox: grades the sim service's own artifacts; needs nothing but python.
FROM docker.io/library/python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
COPY test.sh /tests/test.sh
COPY grade.py /tests/grade.py
RUN chmod +x /tests/test.sh
# the task's runtime bindings, one per seed, and in a runner's copy for one seed its `seed` file (scripts/task_seed.py
# materialize; `see[d]` matches that file or nothing)
COPY runtime_binding*.json see[d] /tmp/rcb-binding/
RUN s=$(cat /tmp/rcb-binding/seed 2>/dev/null || echo 0) && case "$s" in 0) b=runtime_binding.json ;; *[!0-9]*|"") exit 1 ;; *) b="runtime_binding.s$s.json" ;; esac && cp "/tmp/rcb-binding/$b" /tests/runtime_binding.json && rm -rf /tmp/rcb-binding   # the runtime binding of the task's seed (no seed file: runtime_binding.json), as images/roboarc/stage_instance.py picks it
# resource limits: the verifier step and its copy of the limits (docs/PROTOCOL.md §2.10) # rcb-monitor
COPY rcb_limits.py rcb_limits.json /tests/
