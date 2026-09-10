#!/usr/bin/env bash
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/mamba}"
export PATH="$HOME/bin:$PATH"
eval "$("$HOME/bin/micromamba" shell hook -s bash)"
micromamba activate backflip
ROOT=/mnt/c/Users/matth/OneDrive/Desktop/backflip_AI
cd "$ROOT"
export DISPLAY="${DISPLAY:-:0}"
unset MESA_GL_VERSION_OVERRIDE || true

pkill -f demo_ref.py 2>/dev/null || true
pkill -x mma1 2>/dev/null || true
pkill -x mma 2>/dev/null || true
sleep 0.5

echo "== reconvert in-place + REST =="
bash data/sources/digital_stunts/reconvert.sh

python - <<'PY'
import sys, numpy as np
sys.path.insert(0,'helpers')
import checkclip, dm_common as C
from dm_common import qrot
_, jq, lp, rq = checkclip.load('animations/ski_inplace.csv')
up = np.array([0.,1.,0.])
ru = np.stack([qrot(rq[i], up) for i in range(len(rq))])
feet = np.minimum(lp[:,0,1], lp[:,1,1])
travel = np.linalg.norm(lp[-1,C.ROOT,[0,2]] - lp[0,C.ROOT,[0,2]])
print(f"up·Y={ru[:,1].mean():.3f} pelvis={np.median(lp[:,C.ROOT,1]):.3f} "
      f"feet={np.median(feet):.3f} xz_travel={travel:.3f}m")
print('frame0 y', np.round(lp[0,:,1],3))
PY

python helpers/use.py ski_inplace
test -x src/mma1 || bash build_sim.sh 1

nohup ./src/mma1 > /tmp/mma1_demo.out 2>&1 &
MMA_PID=$!
disown "$MMA_PID" || true
echo "mma1 pid=$MMA_PID"
sleep 2
kill -0 "$MMA_PID" 2>/dev/null || { cat /tmp/mma1_demo.out; exit 1; }

# try a window screenshot if tools exist (not Playwright — native GL)
mkdir -p /tmp/ski_demo_shots
if command -v import >/dev/null 2>&1; then
  sleep 1
  import -window root /tmp/ski_demo_shots/root.png 2>/dev/null || true
  echo "screenshot attempt -> /tmp/ski_demo_shots/"
elif command -v scrot >/dev/null 2>&1; then
  scrot /tmp/ski_demo_shots/root.png 2>/dev/null || true
fi

cd "$ROOT/src"
exec python -u ../helpers/demo_ref.py
