# backflip_AI

A humanoid learns backflips and other acrobatics by imitating mocap. The physics sim
is C++ written from scratch, the learning is PPO in PyTorch, and they talk over UDP —
25 humanoids train in parallel.

## Running it

You need GLFW, GLEW and OpenGL. Build the sim:

```bash
g++ src/mma.cpp -O2 -o mma -fopenmp -lglfw -lGLEW -lGL -lX11 -ldl -lpthread
```

Start it, then train against whatever clip is live:

```bash
./mma &
cd src && python train.py
```

To watch a trained policy instead, build with one humanoid rather than 25:

```bash
g++ src/mma.cpp -O2 -DNUM_ENVS=1 -o mma1 -fopenmp -lglfw -lGLEW -lGL -lX11 -ldl -lpthread
./mma1 &
cd src && cp ../models/backflip.pt POLICY_3D.pt && python test.py
```

`r` snaps to the start pose, `r` again runs from there, `q` quits. In the sim window
`b` cycles how often it draws — training is much faster when it isn't rendering.

## Switching clips

```bash
cd helpers
python use.py              # list what's available
python use.py backflip     # make that one live
```

It prints a `DPHASE` value — put it in `test.py` when you change clips.

## Reverse RSI

Episodes normally start at a random point in the clip. That doesn't work for a
backflip: the landing is only reachable if the takeoff and flight already went right,
so a policy starting from frame 0 fails at the landing forever.

So the start window runs backwards instead. Episodes begin in the last 10% of the clip
where all that's left is to land, and once that's solid the window widens back toward
the takeoff. Every stage starts from something the previous stage already solved.

`RC_ENABLE` in `src/train.py` turns it on. Leave it off for walking and running, where
every part of the cycle is reachable anyway and plain random starts are fine.

## Layout

```
src/         sim, training, playback, and the keyframe animation editor
2D/          the earlier 2D version
animations/  reference clips
models/      trained policies, one per skill
helpers/     retarget mocap and check clips are physically possible
```

Clips are retargeted from [DeepMimic](https://github.com/xbpeng/DeepMimic), plus a few
animated by hand.
