# MuJoCo Occlusion Manipulation Lab

A robot that pushes a box to a goal using only a fixed RGB camera, and keeps pushing
when the box disappears behind an occluding plate.

The point of the project is one engineering question: **when vision is interrupted
mid-task, how far does a remembered estimate get you, and when does it hurt?** The
repository contains the simulation, the perception, the estimator, three policies and
the batch evaluation that answers that question — including the case where memory
does *not* help.

![Occluded push demo](results/demo/C_history_seed0.gif)

## What is measured

| | |
| --- | --- |
| Scene | Franka Panda on a table, one 10 cm red cube, green goal patch, fixed 640×480 RGB camera, vertically sliding plate that only blocks the line of sight |
| Task | push the cube into the goal (5 cm radius) |
| Start | cube position uniform in x ∈ [0.36, 0.42], y ∈ [−0.03, 0.03], paired across policies |
| Policies | `oracle` (simulator state), `current-obs` (current frame only), `history` (tracker) |
| Conditions | A none, B partial, C full for 100 steps, D full until the end |
| Budget | 600 control steps = 12 s simulated, 50 Hz control |
| Runs | 10 seeds per (condition, policy) |

<!-- RESULTS -->

## Architecture

```
            fixed RGB camera                 plate (line of sight only)
                    │                                  │
                    ▼                                  ▼
   ┌──────────────────────────┐            ┌───────────────────────┐
   │ perception               │            │ occluder schedule     │
   │  HSV blob → homography   │            │  A/B/C/D conditions   │
   └────────────┬─────────────┘            └───────────┬───────────┘
                │ detection (xy, area, visible)        │
                ▼                                      │
   ┌──────────────────────────┐                        │
   │ state estimation         │                        │
   │  correct / predict /     │                        │
   │  declare lost            │                        │
   └────────────┬─────────────┘                        │
                │ belief (xy, variance, regime)        │
                ▼                                      ▼
   ┌──────────────────────────┐            ┌───────────────────────┐
   │ planning                 │            │ MuJoCo plant          │
   │  go-above → descend →    │───────────▶│  Panda, box, table    │
   │  advance along the line  │  joint     └───────────┬───────────┘
   └────────────┬─────────────┘  targets             │
                │                                     │
                ▼                                     ▼
   ┌──────────────────────────┐            ┌───────────────────────┐
   │ control                  │◀───────────│ 6-dof damped LS IK    │
   │  joint position targets  │  measured  │ on the pusher sphere  │
   └──────────────────────────┘  tip pose  └───────────────────────┘
```

Perception and estimation never read the simulator. Ground truth is used in exactly
three places: scoring, the `oracle` baseline, and the one-time calibration in which a
constant silhouette bias is fitted.

## Quick start

```bash
# WSL2 / Ubuntu, Python 3.12. MuJoCo needs a renderer: EGL works headless here.
cd ~/mujoco-occlusion-lab
python3 -m venv .venv && ./.venv/bin/pip install -e .[dev]

MUJOCO_GL=egl ./.venv/bin/python scripts/20_calibrate.py        # writes configs/calibration.json
MUJOCO_GL=egl ./.venv/bin/python scripts/40_demo_video.py --condition C --policy history --seed 0
MUJOCO_GL=egl ./.venv/bin/python scripts/30_run_matrix.py --seeds 10
./.venv/bin/python scripts/31_plots.py                          # figures/comparison.png
MUJOCO_GL=egl ./.venv/bin/python -m pytest -q                   # 29 tests
```

`scripts/11_batch_oracle.py --seeds 20` is the no-perception sanity check: the plant
and the pushing controller alone, 20/20.

## Perception calibration

| step | result |
| --- | --- |
| Homography from 4 magenta fiducials + goal patch | marker residual 0.0 mm, goal error 2.0 mm |
| Constant silhouette bias (cube seen from 45°) | (−5.7, −73.0) mm, spread 14 mm |
| Detection error over a 6×5 grid of the workspace | mean 13.4 mm, p95 20.4 mm, max 23.1 mm, 30/30 detected |

The bias exists because the centroid of the visible red region of a 10 cm cube is not
the projection of its footprint centre. It is measured once, stored in
`configs/calibration.json`, and applied to every detection.

## What went wrong on the way (the useful part)

Every one of these was found by measuring, not by reading code:

1. **Tool point, three times.** Finger body origins sit 9 cm above the fingertip; the
   gripper collision mesh is a 23 cm blob whose lowest point is 2 cm *below* the
   "fingertip"; support points of that blob are corners. Fixed by mounting an explicit
   12 mm pusher sphere and reading it back with `geom_xpos`.
2. **A QP-based IK froze silently.** `mink` + `daqp` returned near-zero joint
   velocities in closed loop with a 2 cm error and no limit active. Replaced with 25
   lines of damped least squares.
3. **The rotation-error formula was invalid.** Summing column cross products degenerates
   for the 180° flips a naively chosen "fingers down" target asks for; the IK was fed a
   bogus axis. Replaced with an exact axis-angle error, and the downward target now
   keeps the current yaw so the requested rotation is the smallest one.
4. **The actuators are velocity-type.** `force = 4500·ctrl − 4500·q − 450·qd`, so a
   small commanded offset means a slow arm: with a 1 cm Cartesian step the tool
   travelled 1.4 mm per control step and never arrived.
5. **Approach distance is a two-sided constraint.** 17 cm behind the box folds the arm
   until joint 4 hits its limit and freezes; 7 cm and the ball grazes the box side and
   stalls on the way down. 10 cm works.
6. **A white-threshold "calibration" found the robot.** The Panda is white, so four
   white fiducials were "detected" on its links and the homography was 240 mm off.
   Magenta fiducials fixed it.
7. **Four points cannot validate a homography.** They constrain exactly as many
   unknowns as they provide, so a mirrored assignment fits perfectly. Adding the goal
   patch as a fifth point did not help either — the goal sits on the mirror axis. The
   correspondence is now fixed by construction and validated by the goal.
8. **A static occluder stops occluding.** Once the box is pushed past the plate it
   reappears, so "temporary full occlusion" was a no-op until the plate learned to
   follow the box in x.
9. **Constant-velocity prediction runs away.** Extrapolating the object's own last
   velocity overshot by 7 cm in 60 hidden steps and triggered a pointless replan; the
   prediction is now driven by the pusher's velocity while the tool is close.

## Limitations

* The object is a single red cube; detection is an HSV threshold plus a homography,
  not a learned detector. Different object colours would need new thresholds, and the
  constant bias is specific to a 10 cm cube on this table.
* The estimator is constant-velocity with an explicit uncertainty. It has no dynamics
  model of the box, so a fast change of push direction during a long occlusion is not
  predicted.
* The pushing controller is scripted and single-shot: it approaches from behind along
  the line object → goal and has no recovery for a box that rolls away.
* The plate tracks the object, which is convenient for a repeatable experiment and
  slightly generous to the robot: a real occluder would be static.

## Attribution

`assets/panda/` is a copy of `google-deepmind/mujoco_menagerie`'s `franka_emika_panda`
(Apache-2.0); the local modifications are listed in `docs/design.md` and marked inline
with comments. Everything else in this repository — scene, controller, perception,
estimator, policies, evaluation — is written for this project.
