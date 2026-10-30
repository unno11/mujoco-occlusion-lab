# MuJoCo Occlusion Manipulation Lab

A Franka Panda pushes a red box to a green goal using **only a fixed RGB camera**, and
keeps pushing after the box disappears behind an occluding plate.

The interesting question is not whether a scripted pusher can hit a box — it is what
happens to the closed loop *观察 → 估计 → 规划 → 执行 → 反馈 → 更新* when the object
is not in the image. This repo builds that loop end to end, then measures three
policies against each other under four occlusion conditions, and reports what the
numbers actually say — including the case where memory does not help.

![occlusion](figures/occlusion_ABC.png)
![comparison](figures/comparison.png)

## What is actually implemented

| module | what it does |
| --- | --- |
| `envs/push_env.py` | MuJoCo scene (Panda + table + 10 cm box + goal + slide-mounted occluder), `reset/step/get_observation`, success + timeout, per-step detection |
| `control/ik_controller.py` | 6-DOF damped-least-squares IK at 500 Hz on top of the 50 Hz command interface, tool point = the 12 mm pusher sphere |
| `perception/calibration.py` | one-time homography from 4 magenta fiducials (their table coordinates are known), validated against the goal patch |
| `perception/color_blob.py` | HSV segmentation of the box, area-based visibility/partial/occluded classification, calibrated silhouette-bias correction |
| `state_estimation/object_tracker.py` | measure → correct, hide → predict (tool-velocity when the pusher is in contact, else decaying), uncertainty that grows while hidden, "lost" → retreat |
| `planning/push_policy.py` | GO_ABOVE → DESCEND → ADVANCE push primitive with replanning when the object leaves the line |
| `planning/policies.py` | the three policies below, all sharing the same primitive |
| `evaluation/` + `scripts/30_run_matrix.py` | paired-seed batch runs, per-step logs, CSV/JSON, summary |

## The three policies

| policy | object position comes from | when the object is hidden |
| --- | --- | --- |
| `oracle` | MuJoCo ground truth (upper bound + debug only) | keeps pushing |
| `current-obs` | the current frame only | **stops and waits** — it is not allowed to use the last seen position |
| `history` | tracker estimate | keeps pushing on the prediction, corrects on re-observation |

Only `oracle` ever touches `env.object_truth_xy()`. The other two see
`obs["detection"]` (pixels → table coordinates) and nothing else.

## Occlusion conditions

| id | setup |
| --- | --- |
| A | no occluder |
| B | plate parked at partial height — the box is always partially visible |
| C | plate rises the step the pusher first touches the box, follows the box in x, stays 100 control steps (2 s), then drops |
| D | like C but the plate never drops — the object stays hidden for the rest of the episode |

The plate is mounted on two slide joints and has `contype=0 conaffinity=0`: it blocks
the camera ray and nothing else, so the robot's reachability is identical in all four
conditions.

## Results (10 paired seeds per cell, 120 episodes)

Every condition uses the same seeds, so the start poses are paired across policies.
Success = box centre within 50 mm of the goal.

| condition | oracle | current-obs | history |
| --- | --- | --- | --- |
| A: no occlusion | **10/10** (319 steps) | 9/10 (355) | 9/10 (355) |
| B: partial | **10/10** (319) | 10/10 (286) | 10/10 (287) |
| C: full, 100 steps | **10/10** (328) | 10/10 (318, waits 100) | 9/10 (344) |
| D: full, persistent | **10/10** (328) | 1/10 (570, waits 381) | 7/10 (396) |

Estimate error while the object is hidden (mean over hidden steps, vs ground truth,
computed offline only): **14.8 mm** (B), **14.3 mm** (C), **102.4 mm** (D).
Re-localisation after the object reappears: **1 control step**.

### What these numbers mean

**Oracle is a clean upper bound** — 40/40 across all conditions. Nothing about the
scene or the pushing primitive is broken; every difference below is about perception.

**Temporary occlusion does not need memory.** In C the baseline reaches 10/10 by
simply freezing for the 100 hidden steps and resuming when the plate drops; the
history policy is *slower* (344 vs 318 steps) and drops one seed. This is a null
result and it is the honest one: if the object comes back quickly and the robot can
afford to wait, caching the last position buys nothing.

**Persistent occlusion is where memory is the difference between a task and no task.**
In D the baseline collapses to 1/10 — it waits an average of 381 steps for an object
that never returns. The history policy finishes 7/10.

**Memory is not free, and its failure mode is visible.** In D the estimate drifts to
102 mm mean error: contact-aware prediction keeps it alive but does not stop it, and
3/10 episodes still thrash (retreat → re-approach → retreat). The estimator trusts a
constant-velocity + pusher-motion model, which is wrong as soon as the box slips out
from under the pusher.

## Perception numbers (the part that is easy to fake, so here are the checks)

- Homography from 4 fiducials: marker residual **0.0 mm**; independent check against
  the goal patch: **2.6 mm**.
- Detection error over a 6×5 grid covering the workspace (30 positions, box moved by
  setting the free joint, camera frame rendered, HSV centroid → table):
  **mean 13.4 mm, p95 20.4 mm, max 23.1 mm, 30/30 detected**.
- That 13 mm is a *systematic* artifact, not noise: the silhouette centroid of a 10 cm
  cube seen obliquely is not the projection of its base centre. A constant correction
  vector is fitted **once** during calibration; without it the error is ~73 mm.

## Engineering notes worth reading

Four failures that took real debugging, because they are the ones that recur:

1. **The tool point is not the fingertip.** The finger body origin sits 9 cm above the
   finger tip; the gripper collision mesh (`hand_c`) is a single 23 cm block whose
   lowest point is 2 cm *below* the fingers. Pushing with either one dragged the box
   backwards. Fix: a 12 mm sphere rigidly attached in front of the hand, read back with
   `geom_xpos`, `hand_c` collision disabled.
2. **A QP IK silently did nothing.** `mink` + `daqp` returned ~zero joint velocity with
   no error and the arm parked 12 cm above the target. Replaced with 25 lines of damped
   least squares where every intermediate is printable.
3. **The first IK pose target was geometrically impossible.** Asking for the home
   orientation while descending to the table left the 23 cm palm mesh against the
   surface; orientation error stayed at 47°. Fixed by targeting a reachable
   fingers-down pose and raising the push height.
4. **Calibration looked right and was wrong twice.** With exactly 4 points the
   reprojection residual is *identically zero* — it cannot detect a mismatched point
   ordering. A permutation search then locked onto a 180°-rotated assignment because
   the validation target (the goal patch) sat on the symmetry axis. Fixed by fixing the
   correspondence from image geometry (rows = far/near by image v) and using the goal
   only as a check.

## Limitations

- One object, one colour, HSV thresholds, no depth, no learned perception.
- Flat table only; the homography is invalid off the plane.
- The estimator is constant-velocity + pusher coupling; no object dynamics, no slip model.
- Pushing is a single scripted push; no grasp, no multi-push strategy, no failure recovery beyond retreat-and-retry.
- 10 seeds per cell: differences of one episode (e.g. 9/10 vs 10/10) are not significant.

## Install and run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .            # mujoco, numpy, opencv-python, imageio, matplotlib, pytest
MUJOCO_GL=egl python scripts/00_env_check.py          # scene loads, renders one frame
MUJOCO_GL=egl python scripts/20_calibrate.py          # fits configs/calibration.json
MUJOCO_GL=egl python scripts/30_run_matrix.py --seeds 3 --conditions A,B,C,D
MUJOCO_GL=egl python scripts/31_plots.py --in results/matrix.json --out figures
MUJOCO_GL=egl python scripts/40_demo_video.py --condition D --policy history --seed 0
MUJOCO_GL=egl pytest -q                                # 29 tests
```

The scene is a vendored copy of `mujoco_menagerie/franka_emika_panda` (BSD-3-Clause,
DeepMind) with local modifications documented in `assets/panda/README_mods.md`: an
added pusher sphere, `hand_c` collision disabled, wrist `forcerange` widened to ±40 N·m,
no keyframe. Everything else in `src/`, `scripts/` and `assets/panda/scene_push.xml`
is written for this project.

Headless rendering uses EGL. On this machine a 320×240 frame costs ~55 ms, which is why
batch runs render at half resolution; the demo video uses full resolution.
