# 2D bipedal

The earlier version: a 2D biped learning to stay upright while being shoved. Same
shape as the 3D project — C++ sim, PPO in Python, UDP between them — with a curriculum
on the disturbance instead of on the start phase. Once the policy survives a given
impulse reliably, the impulse grows.

```bash
g++ mma2D.cpp -O2 -o mma2d -lglfw -lGLEW -lGL -lX11 -ldl -lpthread
./mma2d &
python train.py     # or: python test.py to run IMPULSE_POLICY.pt
```
