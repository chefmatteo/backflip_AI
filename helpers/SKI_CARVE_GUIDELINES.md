Ski carve clip — working guidelines
===================================

Goal: before any training, cut flat-5 down to one clean, complete carve and
confirm the figure looks like a skier on the grid. Do not train on the full
8s dump until that clip passes the checks below.


Why not the full take
---------------------
- flat-5 is ~8.7s / 261 frames at 30fps of continuous skiing, not one skill cycle.
- Start and end poses do not match (acyclic). Uniform RSI over the whole take
  mixes initiation, edge change, and exit in one episode.
- The first conversion laid the body on its side: source body-up is ~+Z and
  travel is ~-Y, while mma expects Y-up and +X forward. That is why the sim
  showed tumbling ragdolls, not skiing.
- Fix orientation/scale in `helpers/digital_stunts_to_mma.py` (world align +
  leg-length scale) before judging pose quality. Then cut.


What “one complete carve” means here
------------------------------------
One continuous turn on a single edge, roughly:

  1. Initiation — weight/edge commits, skis start to arc
  2. Fall line — middle of the turn, clearest ski posture
  3. Completion — turn finishes, ready for (but not including) the next edge change

Prefer ~1.0–2.5s (about 30–75 frames at 30fps, ~30–75 at mma 30fps after
resample). Shorter than ~0.8s is usually only a fragment; longer than ~3s
often includes a second turn or a traverse.


How to pick the window
----------------------
1. Watch the reference video(s) in
   `data/sources/digital_stunts/raw/takes/flat-5/`
   (e.g. mocap overlay / GoPro). Pick one turn that is:
   - clearly carved (smooth arc), not a skid or flat traverse
   - fully visible (no big mocap holes)
   - upright enough that hips stay above the feet

2. Note approximate timestamps (seconds) from the video. Convert to source
   frames with fps=30 from the Blender export:
   frame ≈ round(t_seconds * 30)
   The processed npz uses the action frame range from the FBX (261 frames).

3. Optional numeric assist (after world-align exists): plot hip path in the
   horizontal plane and heading rate. A single carve is one smooth arc with
   heading changing mostly one way (one sign of turn rate), bounded by
   near-zero turn-rate or an edge change. Do not trust raw pre-align axes.

4. Export a trial cut with `--max-seconds` only for quick looks; for the real
   clip, cut by explicit frame indices (add `--lo/--hi` to the converter or
   slice after convert). Keep the same window while iterating retarget fixes.


Pipeline order (do not skip)
----------------------------
1. Retarget fix
   - Reconvert npz → csv with world align (body-up → +Y, travel → +X) and
     scale from hip–foot length, then ground clamp.
   - Command sketch:
     bash data/sources/digital_stunts/reconvert.sh

2. Visual sanity in numbers (before the sim)
   Compare to `animations/walk.csv` style stats:
   - root up·Y ≈ 1 (upright), not ~0 or negative
   - root_y around hip height (~0.7–1.0 after clamp), not negative
   - feet_y near ground (~0.05–0.15)
   - head above pelvis above feet in Y
   If these fail, do not cut or train — fix convert/mapping first.

3. Cut one carve
   - Write something like `animations/ski_carve1.csv` (keep flat-5 raw/
     processed sources untouched).
   - Re-run `helpers/checkclip.py` on the cut. Skiing should stay
     “grounded (no flight)”. Peak joint speed should stay under the
     checkclip limit.

4. Activate and look, still no PPO
   - `cd helpers && python use.py ski_carve1`
   - Build/run `src/mma1` (NUM_ENVS=1) and scrub the reference / replay.
   - Pass bar: figure upright, feet on the grid, posture reads as a carve
     (flexed ankles/knees, angled shins), arc moves mostly +X.
   - Fail bar: tumbling, buried pelvis, T-pose limbs, sliding on the head.
     Go back to retarget or pick another window.

5. Only then train
   - Fresh policy (or NEW_SKILL), RC_ENABLE off (grounded locomotion-like).
   - Short episode budget first; judge pose/end/root terms, not just reward.
   - Save as `models/ski_carve1.pt` when something works.


Retarget checklist (known failure modes)
----------------------------------------
- World axes: if body-up is not +Y, the whole clip is garbage in mma.
- Scale: do not scale by median hip Y when it can be negative; use leg length.
- Joint mapping: QuickRig local deltas → 13 mma joints; if upright root but
  limbs look wrong, revisit shoulder/hip REST vs matrix_basis (see DeepMimic
  path in `helpers/retarget.py`).
- Skis/poles are not in the sim — train body imitation only.
- Slope is flattened to the ground plane; “skiing” here means matching the
  body clip on flat ground, not snow physics.


Suggested acceptance criteria for ski_carve1
--------------------------------------------
- Duration 1.0–2.5s, single turn direction dominant
- checkclip PASS, grounded
- root up·Y mean > 0.85 over the cut
- In mma1, feet contact the grid most of the clip; no flip/ragdoll when
  playing the reference
- Training deferred until the above are true


Starting training smoothly (best practices)
-------------------------------------------
Do not dive into hyperparameter tuning until the clip looks right in mma1.
Most “training is broken” failures here were bad reference data.

Order of operations
1. Make the clip honest — align, scale, cut one carve, checkclip PASS.
2. Watch it — `use.py` + `mma1`, scrub the motion. If you would not imitate
   it by eye, the policy will not either.
3. Then train — one skill, one clip, default PPO settings first.

Clip choices that make learning easier
- Short (1–2.5s) beats long. Fewer phases → denser reward on the hard part.
- Single carve, one turn direction. Avoid edge changes in v1.
- For a visual demo, `--in-place` pins root XZ so the figure carves in place
  (lean left/right) instead of gliding across the grid. Training may still
  want real travel later; demo first with `animations/ski_inplace.csv`.
- Grounded only (this ski cut should be). Leave CONTACT_SKILL False unless
  the torso is meant to touch the ground (roll/getup).
- Prefer a window where feet stay near the plane and the root stays upright.
  Huge root yaw errors early → falls → tiny pose reward forever.

train.py switches for a first ski carve
- RC_ENABLE = False — carve is not a backflip; reverse curriculum is for
  skills where the end is unreachable until the start is solved. Uniform RSI
  over a short carve is the right default (same idea as walk/run in README).
- NEW_SKILL = True if warm-starting from another policy (e.g. walk/stance);
  or delete src/POLICY_3D.pt for a cold start. Do not resume a backflip
  checkpoint as if it were the same run.
- FORCE_ACYCLIC — Reference auto-detects loop seams; a carve cut should stay
  acyclic (no phase wrap). That is fine.
- Keep ACTION_STD / EP_LIMIT at defaults for the first run. Change clip
  quality before changing learning rates.

Warm start vs cold start
- Cold start: fine for a short carve; expect very low reward for a while.
- Warm start from a standing/walk policy (NEW_SKILL): often smoother — body
  already balances; it only has to match the carve pose/velocity.
- Do not warm-start from a flip policy into skiing.

What “healthy” early training looks like
- pose / end / root terms move up over iterations; ep_len grows; nofall rises.
- If pose stays ~0 while the viewer still shows tumbling reference, stop and
  fix the clip — more iterations will not help.
- Press `b` in the sim to cut render rate so rollouts are faster.

Practical loop
1. Fix/cut clip → use.py → mma1 eyeball
2. Short train (minutes), watch pose/end/root not only total reward
3. If stuck: shorten the window further (middle of the carve only), or warm
   start from stance/walk
4. Only after one carve tracks: lengthen or add the opposite turn

What not to do first
- Train the full 8s flat-5
- Turn on reverse curriculum “just in case”
- Tune CLIP / network size / ENT_COEF before the figure stands in a carve pose
- Chase snow/ski geometry — body imitation on flat ground is the v1 target


Out of scope for v1
-------------------
- Full run with linked left+right carves
- Terrain / snow / ski meshes
- Reverse curriculum (leave off until you have a reason; not the default here)


Reference paths
---------------
- Source take: data/sources/digital_stunts/raw/takes/flat-5/
- Bone npz:    data/sources/digital_stunts/processed/flat-5_bones.npz
- Converter:   helpers/digital_stunts_to_mma.py
- Activate:    helpers/use.py
- Physics QA:  helpers/checkclip.py
