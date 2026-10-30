# Design notes

What the project is made of, and why each piece looks the way it does. Numbers and
the comparison table live in the README; this file is about mechanism and the
decisions behind it.

## Plant

Franka Panda from `mujoco_menagerie` on a table, one 10 cm cube, a fixed RGB camera
and a vertically sliding plate that never collides with anything (`contype=0`), so
occlusion is a pure line-of-sight phenomenon. Fore-aft and lateral motion of the
plate are slide joints, which lets the plate follow the object in x.

Four local modifications to the vendored model, each with an inline comment:
upstream keyframe removed, wrist `forcerange` widened to ±40 N·m, the coarse 23 cm
gripper collision mesh excluded, and a 12 mm pusher sphere added to the hand.

## Control

Position servos, so the command is a joint target. The IK is 6-dof damped least
squares on the pusher sphere:

```
dq = gain * Jᵀ W (J W Jᵀ + λ²I)⁻¹ e6
```

* `J` from `mj_jacGeom` on the pusher sphere, three position rows and three
  orientation rows; orientation weight 0.35.
* Joint-step clamp 0.06 rad per simulation substep, Cartesian clamp 5 cm per control
  step. Both matter: the actuators are velocity-type (`v = kp·offset/kd`), so a small
  commanded offset produces a *slow* arm, and a distant target makes the differential
  IK swing through a wide arc.
* The commanded joint target is re-seeded from the measured joints every substep.
  Integrating a private target instead lets it wind up against a joint limit while
  the robot lags.
* A clamped integrator (gain 0.15, limit 4 cm) removes the ~15 mm gravity sag of the
  position servos.

## Perception

One-time calibration, then nothing but pixels per episode.

1. Four magenta fiducials at known table coordinates and the green goal patch are
   detected in a single frame. The correspondence is fixed by construction (rows by
   image v, columns by u); fitting is validated by the goal patch, which must land
   within 2 cm of its known position.
2. Red box via HSV threshold, centroid to table coordinates through the homography.
3. One constant bias is fitted once (silhouette centroid of a 10 cm cube seen from
   45° is not its footprint centre) and stored in `configs/calibration.json`.

Result on a 6×5 grid over the workspace: mean error 13.4 mm, p95 20.4 mm, max
23.1 mm, 30/30 detected.

## State estimation

Constant-velocity tracker with explicit uncertainty:

* measurement → correct (0.7 measurement + 0.3 prediction), velocity from an EMA of
  consecutive measurements;
* no measurement → predict. If the tool is within 10 cm of the estimate, the
  prediction is driven by the *tool* velocity, because during a push the object moves
  with the pusher. Extrapolating the object's own last velocity instead runs away as
  soon as the pusher slows (measured: 7 cm of drift in 60 hidden steps, enough to
  trigger a pointless replan);
* variance grows with time hidden and with speed; above 60 mm standard deviation, or
  after 120 hidden steps, the estimate is declared *lost* and the policy retreats to
  the last seen position.

The tracker is 120 lines and has no learned component. It is not a world model, and
nothing in the repository claims otherwise.

## Policies

All three share the same three-phase pusher (go above the contact point, descend,
advance along the line object → goal) and differ only in what they believe:

| policy | object source |
| --- | --- |
| `oracle` | simulator state (upper bound, never reported as perception) |
| `current-obs` | current detection only; holds still when the object is not visible |
| `history` | the tracker above |

The policy replans when the estimate leaves the planned line by more than 3.5 cm, and
refuses estimates that jump more than 12 cm in one step.

## Conditions

| id | plate |
| --- | --- |
| A | retracted |
| B | fixed at the height that hides the lower half of the object |
| C | raised when the pusher first touches the object, held 100 steps, retracted |
| D | raised at first contact and never retracted |

The plate tracks the object in x while it is up; a static plate stops hiding the box
as soon as the box is pushed past it.

## Evaluation

`scripts/30_run_matrix.py` runs every (condition, policy, seed) combination with
paired seeds, records per-step truth, belief and visibility, and writes JSON + CSV.
Success is "box centre inside the goal radius (5 cm)"; episodes end on success or at
600 control steps (12 s of simulated time).
