# source on the build host: the patched mujoco venv + masking + this scene dir. ST_WORK is the work directory that
# holds stenerik/{scenes,tools,refs}; ST_VENV the venv with the EPA-patched mujoco (default $ST_WORK/venv_patched)
export NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${ST_WORK:?set ST_WORK to the work directory that holds stenerik/scenes, stenerik/tools and stenerik/refs}"
export ST_WORK IKEA_SCENES=$ST_WORK/stenerik/scenes
export PYTHONPATH=$ST_WORK/stenerik/tools
PY=${ST_VENV:-$ST_WORK/venv_patched}/bin/python
