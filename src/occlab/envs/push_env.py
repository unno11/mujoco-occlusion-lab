"""Pushing environment with a controllable visual occluder and a real camera.

What the robot gets: an RGB image, the detection that perception produces from it,
joint state and tool pose. Object ground truth is exposed through separate methods
and is used only for scoring and for the oracle baseline.

Occlusion conditions
--------------------
``A``  plate fully retracted.
``B``  plate at the height that hides the lower half of the object, for the whole
       episode.
``C``  plate retracted until the pusher first touches the object, then raised to
       full occlusion for ``hold_steps`` control steps, then retracted. The object
       is therefore always seen before it disappears.
"""

from __future__ import annotations

import logging
import pathlib

import mujoco
import numpy as np

from occlab.config import Config
from occlab.control.ik_controller import PandaIKController, GRIPPER_CLOSED
from occlab.perception.calibration import load as load_calibration
from occlab.perception.color_blob import ColorBlobDetector, Detection

log = logging.getLogger(__name__)

CAMERA = "rgb"
CALIBRATION = pathlib.Path(__file__).resolve().parents[3] / "configs" / "calibration.json"


class PushEnv:
    def __init__(self, cfg: Config, render: bool = False, condition: str = "A",
                 hold_steps: int = 100, rise_delay: int = 0, render_scale: float = 1.0) -> None:
        self.cfg = cfg
        self.condition = condition.upper()
        self.hold_steps = hold_steps
        self.rise_delay = rise_delay
        self.render_scale = float(render_scale)

        self.model = mujoco.MjModel.from_xml_path(str(cfg.scene_path))
        self.data = mujoco.MjData(self.model)
        self.control = PandaIKController(self.model, self.data, cfg.control)

        self._box_qadr = int(self.model.jnt_qposadr[self.model.joint("box_free").id])
        self._occ_z_qadr = int(self.model.jnt_qposadr[self.model.joint("occluder_z").id])
        self._occ_x_qadr = int(self.model.jnt_qposadr[self.model.joint("occluder_x").id])
        self._box_geom = self.model.geom("box").id
        self._pusher_geom = self.model.geom("pusher").id
        self._cam_id = self.model.camera(CAMERA).id

        self._renderer: mujoco.Renderer | None = None
        self.render_enabled = render

        self.goal_xy = np.asarray(cfg.task.goal_xy, dtype=float)
        self.step_count = 0
        self.episode_log: list[dict] = []
        self._min_dist = np.inf
        self._occluder_z = cfg.occluder.retracted_z
        self._touch_step: int | None = None
        self._plate_events: list[dict] = []

        # perception (calibrated once; see scripts/20_calibrate.py)
        self.detector: ColorBlobDetector | None = None
        if CALIBRATION.exists():
            H, meta = load_calibration(CALIBRATION)
            self.detector = ColorBlobDetector(
                H,
                nominal_red_area=int(meta.get("nominal_red_area", 1463)),
                center_offset=np.asarray(meta.get("center_offset_m", [0.0, 0.0]), float),
                pixel_scale=1.0 / self.render_scale,
            )
            self.calibration_meta = meta
        else:
            self.calibration_meta = {}
            log.warning("no calibration file at %s -- perception disabled", CALIBRATION)

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None) -> dict:
        rng = np.random.default_rng(seed)
        self.seed = seed

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)

        half = self.cfg.task.box_half_size
        x = rng.uniform(*self.cfg.task.start_x_range)
        y = rng.uniform(*self.cfg.task.start_y_range)
        self.data.qpos[self._box_qadr : self._box_qadr + 3] = (x, y, half)
        self.data.qpos[self._box_qadr + 3 : self._box_qadr + 7] = (1.0, 0.0, 0.0, 0.0)

        home_ctrl = self.model.key("home").ctrl.copy()
        self.data.ctrl[:] = home_ctrl
        for _ in range(400):
            mujoco.mj_step(self.model, self.data)

        self.control.reset()
        self.step_count = 0
        self._min_dist = float(np.linalg.norm(self.object_truth_xy() - self.goal_xy))
        self.episode_log = []
        self._touch_step = None
        self._plate_events = []
        self._apply_condition_start()
        return self.get_observation()

    def _apply_condition_start(self) -> None:
        if self.condition == "B":
            self.set_occluder(self.occlusion_z_for(0.55))
        else:
            self.set_occluder(self.cfg.occluder.retracted_z)

    # ------------------------------------------------------------- occluder
    def set_occluder(self, body_z: float, x_rel: float | None = None) -> None:
        self._occluder_z = float(body_z)
        self.data.qpos[self._occ_z_qadr] = self._occluder_z
        if x_rel is not None:
            self.data.qpos[self._occ_x_qadr] = float(x_rel)
        mujoco.mj_forward(self.model, self.data)

    def occlusion_z_for(self, fraction: float, obj_xy: np.ndarray | None = None, margin: float = 0.0) -> float:
        """Body z that makes the plate hide `fraction` (0..1) of the object's image height.

        Uses the box's eight corners, not just its centre: the box is 10 cm deep and
        the ray to its far top corner crosses the plate plane ~6 cm above the centre
        ray, which is enough for a "fully occluded" object to keep a visible red strip.
        """
        obj = self.object_truth_xy() if obj_xy is None else np.asarray(obj_xy, float)
        half = self.cfg.task.box_half_size
        cam = self.model.cam_pos[self._cam_id].copy()
        plate_y = self.cfg.occluder.y
        corners = np.array(
            [
                [obj[0] + sx * half, obj[1] + sy * half, 0.0 if sz < 0 else 2 * half]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )

        def z_at_plane(p: np.ndarray) -> float:
            t = (plate_y - cam[1]) / (p[1] - cam[1])
            return float(cam[2] + t * (p[2] - cam[2]))

        zs = [z_at_plane(c) for c in corners]
        z_bot, z_top = min(zs), max(zs)
        target_top = z_bot + fraction * (z_top - z_bot) + margin
        plate_h = 2 * self.cfg.occluder.plate_half_height
        return target_top - plate_h + 0.002

    def get_occluder(self) -> dict:
        return {"qpos": self._occluder_z, "y": self.cfg.occluder.y, "condition": self.condition}

    def _touching(self) -> bool:
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if {c.geom1, c.geom2} == {self._pusher_geom, self._box_geom}:
                return True
        return False

    def _update_occluder_schedule(self) -> None:
        """Condition C: raise the plate once the push has started, hold, then retract.

        The plate follows the object in x while it is up. A static plate stops hiding
        the box as soon as the box is pushed past it, which made the "temporary full
        occlusion" condition a no-op in the first version.

        Condition D is the same, but the plate never comes back down: the object is
        hidden for the rest of the episode. That is the case where a policy without
        memory simply cannot finish, and it is reported as a separate condition rather
        than being folded into C.
        """
        if self.condition not in ("C", "D"):
            return
        if self._touch_step is None and self._touching():
            self._touch_step = self.step_count
        if self._touch_step is None:
            return
        since = self.step_count - self._touch_step
        if since < self.rise_delay:
            return
        persisting = self.condition == "D"
        if persisting or since < self.rise_delay + self.hold_steps:
            obj = self.object_truth_xy()
            self.set_occluder(
                self.occlusion_z_for(1.0, obj_xy=obj, margin=0.03),
                x_rel=float(obj[0] - 0.44),
            )
            if not self._plate_events or self._plate_events[-1].get("event") != "occluded":
                self._plate_events.append({"step": self.step_count, "event": "occluded"})
        elif self._plate_events and self._plate_events[-1].get("event") == "occluded":
            self.set_occluder(self.cfg.occluder.retracted_z, x_rel=0.0)
            self._plate_events.append({"step": self.step_count, "event": "revealed"})

    # ------------------------------------------------------------------- api
    def object_truth_xy(self) -> np.ndarray:
        return self.data.geom(self._box_geom).xpos[:2].copy()

    def object_truth_pose(self) -> np.ndarray:
        return self.data.geom(self._box_geom).xpos.copy()

    def tool_position(self) -> np.ndarray:
        return self.control.tool_position()

    def goal(self) -> np.ndarray:
        return self.goal_xy.copy()

    def get_observation(self) -> dict:
        image = None
        det = None
        if self.detector is not None:
            image = self.render()
            det = self.detector.detect_object(image)
        return self._make_obs(image, det)

    def _make_obs(self, image=None, det=None) -> dict:
        obs = {
            "joint_pos": self.data.qpos[:9].copy(),
            "joint_vel": self.data.qvel[:9].copy(),
            "tool_pos": self.tool_position(),
            "step": self.step_count,
            "occluder": self.get_occluder(),
        }
        if image is not None:
            obs["image"] = image
        if det is not None:
            obs["detection"] = det
        return obs

    def detect(self, image: np.ndarray | None = None) -> Detection:
        if self.detector is None:
            return Detection(None, 0, None, visible=False, partial=False)
        img = self.render() if image is None else image
        return self.detector.detect_object(img)

    def render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=max(60, int(self.cfg.camera.height * self.render_scale)),
                width=max(80, int(self.cfg.camera.width * self.render_scale)),
            )
        self._renderer.update_scene(self.data, camera=self._cam_id)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ step
    def step(self, tool_target: np.ndarray):
        if self.step_count >= self.cfg.task.max_control_steps:
            raise RuntimeError("episode already truncated; call reset()")
        tool_target = np.asarray(tool_target, dtype=float)
        self.control.command_tool(tool_target)
        self.step_count += 1
        self._update_occluder_schedule()

        obj = self.object_truth_xy()
        dist = float(np.linalg.norm(obj - self.goal_xy))
        self._min_dist = min(self._min_dist, dist)
        # one render per step: the same frame feeds detection and the observation
        image = self.render() if self.detector is not None else None
        det = self.detector.detect_object(image) if image is not None else None
        self.episode_log.append(
            {
                "step": self.step_count,
                "tool": self.tool_position().round(5).tolist(),
                "truth_xy": obj.round(5).tolist(),
                "det_visible": bool(det.visible) if det else False,
                "det_xy": None if not det or det.xy is None else det.xy.round(5).tolist(),
                "det_area": int(det.area_px) if det else 0,
                "dist": round(dist, 5),
                "occluder_z": round(self._occluder_z, 4),
            }
        )

        success = dist < self.cfg.task.goal_tol
        truncated = self.step_count >= self.cfg.task.max_control_steps
        reward = (10.0 if success else 0.0) - dist
        info = {
            "dist": dist,
            "success": success,
            "occluder_z": self._occluder_z,
            "det_visible": bool(det.visible) if det else False,
            "det_xy": None if not det or det.xy is None else det.xy,
        }
        return self._make_obs(image, det), reward, success or truncated, truncated, info

    def get_metrics(self) -> dict:
        return {
            "seed": self.seed,
            "condition": self.condition,
            "steps": self.step_count,
            "sim_time_s": round(self.step_count * self.cfg.control.control_dt, 3),
            "final_dist": round(float(np.linalg.norm(self.object_truth_xy() - self.goal_xy)), 5),
            "min_dist": round(float(self._min_dist), 5),
            "success": bool(np.linalg.norm(self.object_truth_xy() - self.goal_xy) < self.cfg.task.goal_tol),
            "final_box_xy": self.object_truth_xy().round(4).tolist(),
            "plate_events": self._plate_events,
            "trajectory": self.episode_log,
        }
