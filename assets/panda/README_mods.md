# Local modifications to the vendored Panda model

Source: `google-deepmind/mujoco_menagerie`, directory `franka_emika_panda`
(BSD-3-Clause, © Google DeepMind). Copied here unmodified except for the changes
below, so the scene can be loaded without a submodule.

## `panda.xml`

| change | why |
| --- | --- |
| removed the `<keyframe>` | the scene supplies its own reset pose, and a stale keyframe silently competed with it |
| relaxed `<key name="actuator8">` force range from ±12 to ±40 N·m | at ±12 the wrist could not hold the fingers-down pose while pressing down; the hand sagged and the box was never touched |
| disabled collision on the `hand_c` mesh (`contype/conaffinity = 0`) | `hand_c` is one ~23 cm block whose lowest point is ~2 cm *below* the fingers; it scraped the table and shoved the box backwards, producing fake contacts |
| added `<body name="pusher">` with a 12 mm sphere | the tool point needs to be an exact, well-defined point (`geom_xpos`), not "the lowest vertex of a mesh" |

The mesh assets under `assets/` are unchanged.

## `scene_push.xml`

Written for this project (not upstream): table, 10 cm box (`freejoint`, mass 0.30 kg),
painted goal patch, four magenta fiducials, fixed camera, and the occluder on two
slide joints (`contype=conaffinity=0`, so it occludes the camera ray only).

Fiducials are magenta rather than white on purpose: the robot is white, and a white
threshold detects the robot's own links as calibration targets.
