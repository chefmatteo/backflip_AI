#!/usr/bin/env bash
set -euo pipefail

export MAMBA_ROOT_PREFIX="$HOME/mamba"
export PATH="$HOME/bin:$PATH"
eval "$("$HOME/bin/micromamba" shell hook -s bash)"

echo "Creating backflip env (this may take a few minutes)..."
micromamba create -y -n backflip -c conda-forge \
  python=3.12 \
  pytorch-cpu \
  numpy \
  cxx-compiler \
  gxx_linux-64 \
  glfw \
  glew \
  glm \
  llvm-openmp \
  libgomp \
  pkg-config \
  libgl-devel \
  mesalib \
  xorg-libx11 \
  xorg-libxcursor \
  xorg-libxrandr \
  xorg-libxi \
  xorg-libxinerama \
  xorg-xorgproto \
  xorg-libxext \
  libxcb

echo ""
echo "=== verifying Python ==="
micromamba run -n backflip python - <<'PY'
import torch, numpy
print("torch", torch.__version__)
print("numpy", numpy.__version__)
print("cuda", torch.cuda.is_available())
PY

echo ""
echo "=== verifying C++ toolchain ==="
micromamba run -n backflip bash -lc '
set -e
echo "g++: $(g++ --version | head -1)"
echo "CONDA_PREFIX=$CONDA_PREFIX"
pkg-config --modversion glfw3 && echo glfw_ok
pkg-config --modversion glew && echo glew_ok
test -f "$CONDA_PREFIX/include/glm/glm.hpp" && echo glm_ok
ls "$CONDA_PREFIX/lib"/libomp* "$CONDA_PREFIX/lib"/libgomp* 2>/dev/null | head
'

echo ""
echo "DONE. Activate later with:"
echo "  eval \"\$(\$HOME/bin/micromamba shell hook -s bash)\""
echo "  micromamba activate backflip"
