#!/usr/bin/env bash
# Build the 3D sim inside the micromamba "backflip" env.
set -euo pipefail
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/mamba}"
export PATH="$HOME/bin:$PATH"
eval "$("$HOME/bin/micromamba" shell hook -s bash)"
micromamba activate backflip

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

P="$CONDA_PREFIX"
export CPATH="${P}/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${P}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${P}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

NUM_ENVS_FLAG="${1:-}"
OUT=src/mma
EXTRA=()
if [[ "$NUM_ENVS_FLAG" == "1" ]]; then
  OUT=src/mma1
  EXTRA=(-DNUM_ENVS=1)
fi

g++ src/mma.cpp -O2 -o "$OUT" -fopenmp "${EXTRA[@]}" \
  -I"$P/include" -L"$P/lib" \
  -lglfw -lGLEW -lGL -lX11 -ldl -lpthread

echo "built $OUT"
