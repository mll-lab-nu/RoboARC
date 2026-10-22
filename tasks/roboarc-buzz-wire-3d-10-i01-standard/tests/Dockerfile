# Separate verifier sandbox: grades the sim service's own artifacts; needs nothing but python.
FROM docker.io/library/python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
COPY test.sh /tests/test.sh
COPY grade.py /tests/grade.py
RUN chmod +x /tests/test.sh
COPY runtime_binding.json runtime_binding.s1.json runtime_binding.s2.json seed /tmp/rcb-seed/
RUN s=$(cat /tmp/rcb-seed/seed) && case "$s" in 0) b=runtime_binding.json ;; *[!0-9]*|"") exit 1 ;; *) b=runtime_binding.s$s.json ;; esac && cp /tmp/rcb-seed/$b /tests/runtime_binding.json && rm -rf /tmp/rcb-seed   # the runtime binding of the seed (docs/RUNNING.md "Seeds"); a missing one fails the build
# resource limits: the verifier step and its copy of the limits (docs/PROTOCOL.md §2.10) # rcb-monitor
COPY rcb_limits.py rcb_limits.json /tests/
