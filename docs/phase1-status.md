# Phase 1 report: plant, IK, scripted pushing

Date: 2026-10-30. **Status: accepted.** Unoccluded pushing works: 20/20 seeds, mean
202 control steps (4.0 s of simulated time), max 289.

## Result

| Metric | Value |
| --- | --- |
| Success | **20 / 20** (`scripts/11_batch_oracle.py --seeds 20`) |
| Steps | mean 202, min 182, max 289 |
| Policy | scripted 3-phase push (go-above → descend → advance), object position from the simulator (oracle) |
| Goal tolerance | 0.05 m, goal at (0.60, 0.00), start box uniform in x ∈ [0.42, 0.47], y ∈ [-0.03, 0.03] |
| Video | `results/phase1/oracle_seed3.mp4` (111 frames, 640x480) |

Termination is "centre of the box inside the goal radius", so a successful episode
always ends just inside the tolerance (0.047-0.050 m). The box is 10 cm, so the
remaining offset is under one box radius.

## What the numbers do NOT say

The pusher is a 12 mm ball on a stiffened Panda wrist, the box is a 10 cm / 0.3 kg cube
with friction 1.0, and the controller is a 6-dof damped-least-squares servo. Nothing
here is a perception result: the object position comes from the simulator.

## Debug chain (each entry is a measured fact)

1. **Tool point was wrong three times.** Finger body origins sit 9 cm above the
   fingertip; the gripper collision mesh `hand_c` is a 23 cm blob whose lowest point is
   2 cm below the "fingertip"; support points of that blob are corners. Fix: an explicit
   `pusher` sphere on the hand, read back with `geom_xpos`.
2. **QP-based IK stalled silently.** `mink` + `daqp` returned near-zero joint velocities
   in closed loop with a 2 cm error and no limit active; the arm froze 12 cm above the
   target. Fix: 25 lines of damped least squares.
3. **Position-only IK tilts the wrist.** The redundant dof rotated the hand until the
   finger axis was horizontal and the 23 cm housing rested on the box. Fix: 6-dof task,
   orientation weight 0.35.
4. **The rotation error formula was invalid.** Summing column cross products is
   degenerate for a 180-degree flip, which is exactly what a naively chosen "fingers
   down" target asks for; the IK was fed a bogus axis and never corrected orientation.
   Fix: exact axis-angle from the relative rotation, and the downward target keeps the
   current yaw so the requested rotation is the smallest one.
5. **Joint-target wind-up.** Integrating a private `q_des` drove joint 2 to its limit
   while the robot lagged, saturating the servo. Fix: re-seed from measured joints every
   substep.
6. **The servo is velocity-type.** `force = 4500·ctrl − 4500·q − 450·qd`, so a small
   commanded offset means a *slow* arm (v = kp·offset/kd). With a 1 cm Cartesian step
   the tool travelled 1.4 mm per control step and never arrived. Fix: 5 cm Cartesian
   step, 0.06 rad joint step.
7. **Steady-state sag ate the push height.** ~15 mm error under gravity put the ball on
   the table instead of the box face. Fix: clamped integrator (gain 0.15, limit 4 cm)
   and push height 5.5 cm.
8. **Approach distance is a two-sided constraint.** 17 cm behind the box folds the arm
   until joint 4 hits its limit (-3.072 rad) and freezes; 7 cm and the ball grazes the
   box side and stalls while descending. 10 cm works (20/20).

## Local modifications to the vendored Panda model

`assets/panda/` copies `google-deepmind/mujoco_menagerie`'s `franka_emika_panda`
(Apache-2.0). Changed locally, each with an inline comment:

- upstream keyframe removed (the scene needs a 17-dof keyframe);
- wrist actuator `forcerange` ±12 → ±40 N·m;
- `hand_c` excluded from collision (coarse 23 cm blob, scrapes table and box);
- `pusher` sphere geom added to the hand.

## Open items before Phase 2

- The success bar is enforced by termination, so the numbers say "reaches the goal
  radius", not "parks at the centre". A stricter criterion (stay inside for N steps)
  would lower the rate; decide before comparing policies.
- No occluder interaction yet in these runs: the plate is retracted for all 20 seeds.
