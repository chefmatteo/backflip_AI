# backflip_AI

A humanoid learns acrobatics — backflips, cartwheels, kicks — by imitating reference
motion in a physics simulator written from scratch.

The simulator is C++ (rigid bodies, joint constraints, PD torque control, contacts,
all hand-rolled on top of OpenGL). The learner is PPO in PyTorch. They talk over UDP:
25 humanoids step in parallel in the sim, Python sends joint targets and gets back
state, reward and contact flags.

Training follows DeepMimic: the reward scores how closely the body tracks a reference
clip (pose, velocity, end effectors, root, centre of mass), and episodes start at a
random point in the clip rather than always at frame 0.

## Reverse RSI

Reference State Initialisation normally spawns episodes uniformly across the clip. A
backflip does not cooperate with that — the landing is only reachable if the takeoff
and the flight already worked, so a policy starting at frame 0 fails at the landing
forever and never sees the reward for getting it right.

So the start window runs backwards. Episodes begin in `[rc_lo, 1.0]`, with `rc_lo`
starting at 0.9 — the last few frames, where all that remains is to land. Once the
policy clears that stage, `rc_lo` steps down and the window widens back toward the
takeoff. Each stage begins from a state the previous stage already solved.

The gate is a binary success rate over a window of iterations: above `RC_RMAX` the
window widens, below `RC_RMIN` it retreats. Success means the episode neither fell nor
drifted off the reference's root orientation. A fraction of spawns (`RC_FRONTIER_P`)
concentrate on the newest frames, the rest spread over already-mastered ground so the
earlier stages don't rot.

Set `RC_ENABLE = True` in `src/train.py` to use it. Leave it off for cyclic gaits
(walk, jog, run), where every phase is reachable from every other and plain uniform
RSI is what you want.

## Layout

```
src/         mma.cpp      the 3D simulator, 25 parallel humanoids
             train.py     PPO + imitation reward + reverse RSI
             test.py      run a trained policy
             animate.cpp  keyframe animation editor, writes reference clips
2D/          the earlier 2D bipedal version, kept as it was
animations/  reference clips (150-column CSV)
models/      trained policies
helpers/     clip pipeline: retarget, physics-correct, select
```

## Build and run

The simulator needs GLFW, GLEW and OpenGL.

```bash
g++ src/mma.cpp -O2 -o mma -fopenmp -lglfw -lGLEW -lGL -lX11 -ldl -lpthread
```

Train — start the sim first, it listens on UDP 5005:

```bash
./mma &
cd src && python train.py
```

Run a trained policy — `test.py` drives a single humanoid, so build the sim with one:

```bash
g++ src/mma.cpp -O2 -DNUM_ENVS=1 -o mma1 -fopenmp -lglfw -lGLEW -lGL -lX11 -ldl -lpthread
./mma1 &
cd src && cp ../models/backflip.pt POLICY_3D.pt && python test.py
```

Press `r` to snap to the handoff phase, `r` again to run from there, `q` to quit.
In the sim window, `b` cycles how often it renders — training is much faster when it
is not drawing every frame.

## Choosing a clip

`helpers/use.py` makes one clip live, writing `src/baked_motion.csv` (what the sim and
editor read) and `src/motion.npz` (what training reads):

```bash
cd helpers
python use.py              # list every clip with a physics report
python use.py backflip     # make that one live
```

A clip has to be physically reachable or the reward is unreachable too: while both feet
are off the floor the centre of mass must follow a parabola, and angular momentum must
be conserved. `helpers/` checks this and corrects clips that break it, adjusting only
root translation and flight timing — joint angles and style are left alone.

```bash
python checkclip.py ../animations/backflip.csv    # is it learnable?
python build.py dm spinkick                       # retarget a DeepMimic clip
python build.py all                               # rebuild the standard set
```

After switching clips, set `DPHASE` in `test.py` to `1/(frames-1)` — `use.py` prints
the value.

## Notes

Reference clips are retargeted from [DeepMimic](https://github.com/xbpeng/DeepMimic)'s
`humanoid3d` mocap, plus some hand-animated in the built-in editor. The two skeletons
have near-identical proportions, so joint rotations transfer almost directly.

Policies are trained per skill — one clip, one checkpoint.
