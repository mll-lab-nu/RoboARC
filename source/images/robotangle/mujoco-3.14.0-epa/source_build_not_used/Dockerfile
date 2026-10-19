# libmujoco.so.3.14.0 with the upstream EPA horizon fix (epa_horizon.patch; see README.md).
#
#   docker build -t mujoco-3.14.0-epa images/ikea-mujoco/mujoco-3.14.0-epa        # -> /out/libmujoco.so.3.14.0 + /out/SHA256SUMS
#   consumer:  FROM ... AS app ... RUN pip install mujoco==3.14.0
#              COPY --from=<this stage> /out/libmujoco.so.3.14.0 /usr/local/lib/python3.12/site-packages/mujoco/
#
# Toolchain = the one the PyPI 3.14.0 wheel's libmujoco.so names in its .comment section (clang 20.1.8 + LLD 20.1.8,
# llvm-project 87f0227cb601), with upstream's default Release CMake options (-O3 -flto=thin -mavx, MuJoCo's own AVX
# setting); nothing added. APPLY_PATCH=0 builds the unpatched 3.14.0 with the same toolchain (a diagnostic).
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS mujoco-epa-build

ARG MUJOCO_TAG=3.14.0
ARG MUJOCO_COMMIT=9ecbb9d7b5ee623f54745638d36799ff90e6f7cd
ARG LLVM_DEB=1:20.1.8~++20250809043815+87f0227cb601-1~exp1~20250809163919.3
ARG APPLY_PATCH=1
ARG JOBS=8

RUN apt-get update \
 && apt-get install -y --no-install-recommends wget gnupg ca-certificates git cmake=3.31.6-2 ninja-build=1.12.1-1 \
 && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor > /usr/share/keyrings/llvm.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/trixie/ llvm-toolchain-trixie-20 main" \
      > /etc/apt/sources.list.d/llvm.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends clang-20=${LLVM_DEB} lld-20=${LLVM_DEB} llvm-20=${LLVM_DEB} \
      libllvm20=${LLVM_DEB} libclang-cpp20=${LLVM_DEB} \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${MUJOCO_TAG} https://github.com/google-deepmind/mujoco.git /src \
 && test "$(git -C /src rev-parse HEAD)" = "${MUJOCO_COMMIT}"

COPY epa_horizon.patch /epa_horizon.patch
RUN if [ "${APPLY_PATCH}" = 1 ]; then git -C /src apply /epa_horizon.patch; fi \
 && cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
      -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_BUILD_SIMULATE=OFF \
 && ninja -C /build -j ${JOBS} -v mujoco > /build/ninja.log \
 && mkdir -p /out && cp -L /build/lib/libmujoco.so.3.14.0 /out/ \
 && (cd /out && sha256sum libmujoco.so.3.14.0 > SHA256SUMS) \
 && (echo "mujoco ${MUJOCO_TAG} ${MUJOCO_COMMIT} patch=${APPLY_PATCH}"; clang-20 --version | head -1; ld.lld-20 --version; \
     cmake --version | head -1; grep -m1 -o "clang-20 .*engine_collision_gjk.c" /build/ninja.log) > /out/BUILDINFO \
 && cat /out/SHA256SUMS /out/BUILDINFO
