# source on a cloud host: the patched mujoco venv + masking + this scene dir
export NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export IKEA_SCENES=/workspace/qn_ikea_b2/stenerik/scenes
export PYTHONPATH=/workspace/qn_ikea_b2/stenerik/tools
PY=/workspace/qn_ikea/venv_patched/bin/python
