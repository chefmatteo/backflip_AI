# 2D bipedal

The version before the 3D one: a 2D biped learning to stay standing while getting
shoved. Same idea — C++ sim, PPO in Python, UDP between them — except the curriculum
grows the shove instead of moving the start point. Survive the current push reliably
and it gets harder.

```bash
g++ mma2D.cpp -O2 -o mma2d -lglfw -lGLEW -lGL -lX11 -ldl -lpthread
./mma2d &
python train.py     # or test.py to watch IMPULSE_POLICY.pt
```
