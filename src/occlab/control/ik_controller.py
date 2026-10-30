"""Operational-space controller for the Franka Panda (6-dof damped-least-squares IK).

Design notes, because these were the expensive mistakes:

* The tool point is the centre of the pusher ball mounted on the hand in
  `assets/panda/panda.xml` (12.5 cm along the finger axis). Body origins and mesh
  support points are both wrong references here: the finger bodies sit ~9 cm above
  the fingertip, and the gripper collision mesh (`hand_c`) is a 23 cm blob whose
  support point is just a corner.
* The hand orientation has to be controlled. With a position-only task the redundant
  dof rotates the wrist until the finger axis is horizontal; the 23 cm housing then
  rests on the box while the tool point floats 15 cm above the table.
* The rotation error must be computed as an exact axis-angle from the relative
  rotation. Summing column cross products (the common small-angle trick) is
  degenerate for the 180-degree flips that show up when the target wrist pose is
  chosen naively, and silently feeds the IK a bogus axis.
* The downward orientation is built per episode from the *current* hand yaw
  (`set_downward_orientation`), so the requested rotation is the smallest one that
  points the fingers at the table instead of an arbitrary half-turn.

Actuators are position servos, so the command is a joint target; the outer loop is
closed on the measured tool position every simulation substep.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

log = logging.getLogger(__name__)

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
GRIPPER_CLOSED = 40.0  # actuator8 ctrl in [0, 255]; 0 makes the fingers interpenetrate
DOWN = np.array([0.0, 0.0, -1.0])


class PandaIKController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, control_cfg) -> None:
        self.model = model
        self.data = data
        self.cfg = control_cfg

        self._tool_geom = model.geom("pusher").id
        self._n_arm = len(ARM_JOINTS)
        self._lo = model.jnt_range[: self._n_arm, 0].copy()
        self._hi = model.jnt_range[: self._n_arm, 1].copy()
        self._control_steps = max(1, round(control_cfg.control_dt / model.opt.timestep))

        self.tool_rotation = np.eye(3)
        self.orientation_weight = 0.35
        self.damping = 0.05
        self.gain = 0.6
        self.deadband = 0.004          # m
        self.max_joint_step = 0.06     # rad per simulation substep (servo: v = kp*offset/kd)
        self.max_cart_step = 0.050     # m per control step; the servo is velocity-type,
                                      # so a small step also means a slow arm
        self.wrist_regularisation = 0.02
        self.max_orientation_error = 0.6  # rad, keeps Rotation from dominating Position

        self._q_seed = data.qpos[: self._n_arm].copy()
        # The position servo leaves a ~15 mm steady-state offset under gravity. A
        # small clamped integrator removes it without the oscillation that a fast
        # integrator causes.
        self._integral = np.zeros(3)
        self.integral_gain = 0.15
        self.integral_limit = 0.04

    # ---------- state ----------
    def tool_position(self) -> np.ndarray:
        """Centre of the pusher ball, straight from forward kinematics."""
        return self.data.geom_xpos[self._tool_geom].copy()

    def tool_orientation(self) -> np.ndarray:
        return self.data.geom_xmat[self._tool_geom].reshape(3, 3).copy()

    def set_downward_orientation(self) -> None:
        """Fingers pointing at the table, keeping the current yaw (smallest rotation)."""
        cur = self.tool_orientation()
        x_col = cur[:, 0] - (cur[:, 0] @ DOWN) * DOWN
        n = np.linalg.norm(x_col)
        x_col = x_col / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        y_col = np.cross(DOWN, x_col)
        self.tool_rotation = np.column_stack([x_col, y_col, DOWN])

    def orientation_error(self) -> np.ndarray:
        """Exact rotation vector (axis * angle) taking the current frame to the target."""
        rel = self.tool_rotation @ self.tool_orientation().T
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rel.flatten())
        if quat[0] < 0:  # shortest path
            quat = -quat
        w = float(np.clip(quat[0], -1.0, 1.0))
        v = quat[1:]
        s = float(np.linalg.norm(v))
        if s < 1e-9:
            return np.zeros(3)
        angle = 2.0 * np.arctan2(s, w)
        return v / s * angle

    def reset(self) -> None:
        self._q_seed = self.data.qpos[: self._n_arm].copy()
        self._integral[:] = 0.0
        self.set_downward_orientation()

    # ---------- control ----------
    def command_tool(self, target_tool_xyz: np.ndarray) -> None:
        target = np.asarray(target_tool_xyz, dtype=float)
        if not np.all(np.isfinite(target)):
            raise ValueError(f"non-finite tool target: {target}")

        tip = self.tool_position()
        delta = target - tip
        dist = float(np.linalg.norm(delta))
        if dist > self.max_cart_step:
            target = tip + delta * (self.max_cart_step / dist)

        self._integral += self.integral_gain * (target - self.tool_position())
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        target = target + self._integral

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        weights = np.array([1.0, 1.0, 1.0] + [self.orientation_weight] * 3)

        for _ in range(self._control_steps):
            mujoco.mj_forward(self.model, self.data)
            ori_err = self.orientation_error()
            n_ori = float(np.linalg.norm(ori_err))
            if n_ori > self.max_orientation_error:
                ori_err = ori_err * (self.max_orientation_error / n_ori)
            err = np.concatenate([target - self.tool_position(), ori_err])

            if float(np.linalg.norm(err[:3])) < self.deadband and n_ori < 0.02:
                dq = np.zeros(self._n_arm)
            else:
                jacp[:] = 0.0
                jacr[:] = 0.0
                mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, self._tool_geom)
                jac = np.vstack([jacp[:, : self._n_arm], jacr[:, : self._n_arm]])
                w = np.diag(weights)
                dq = self.gain * (jac.T @ w @ np.linalg.solve(
                    w @ jac @ jac.T @ w.T + (self.damping ** 2) * np.eye(6), err
                ))
                dq[4:] += self.wrist_regularisation * (self._q_seed[4:] - self.data.qpos[4:7])

            biggest = float(np.max(np.abs(dq))) if dq.size else 0.0
            if biggest > self.max_joint_step:
                dq *= self.max_joint_step / biggest
            # Re-seed from the measured joints: integrating a private target lets it
            # wind up against a joint limit while the robot lags, then the servo is
            # saturated and the arm freezes.
            q_des = np.clip(self.data.qpos[: self._n_arm] + dq, self._lo, self._hi)

            self.data.ctrl[: self._n_arm] = q_des
            self.data.ctrl[7] = GRIPPER_CLOSED
            mujoco.mj_step(self.model, self.data)

        if not np.all(np.isfinite(self.data.qpos)):
            raise RuntimeError("simulation diverged to non-finite qpos")
