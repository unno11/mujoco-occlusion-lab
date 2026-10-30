"""The three policies compared in the experiments.

All of them share the same pushing mechanism (`PushPolicy`) and differ only in what
they believe about the object's position:

* ``oracle``        -- reads the simulator state. Upper bound and debugging aid,
                       never reported as a perception result.
* ``current-obs``   -- only the detection from the current frame. When the object is
                       not visible it holds still and waits; it never uses a stale
                       position.
* ``history``       -- the tracker in `state_estimation`: last measurement, motion
                       prediction, correction on re-observation, and an explicit
                       retreat when the estimate is deemed lost.

Only the oracle touches `truth_xy`; the other two receive it as `None`.
"""

from __future__ import annotations

import logging

import numpy as np

from occlab.planning.push_policy import PushPolicy
from occlab.state_estimation.object_tracker import Belief, ObjectTracker, TrackerConfig

log = logging.getLogger(__name__)


class Agent:
    name = "base"
    uses_truth = False

    def __init__(self, task_cfg, control_dt: float = 0.02) -> None:
        self.cfg = task_cfg
        self.dt = control_dt
        self.push = PushPolicy(task_cfg)
        self.goal = np.zeros(2)
        self.last_target = np.zeros(3)
        self.waits = 0
        self.recoveries = 0
        self._recovery_left = 0
        self._last_seen: np.ndarray | None = None
        self.belief: Belief | None = None

    # ---------- lifecycle ----------
    def reset(self, obs: dict, goal_xy: np.ndarray, truth_xy: np.ndarray | None = None) -> None:
        self.goal = np.asarray(goal_xy, float)
        self.waits = 0
        self.recoveries = 0
        self._recovery_left = 0
        self._last_seen = None
        self.belief = None
        start = self._initial_object(obs, truth_xy)
        if start is None:
            start = self.goal.copy()  # will be replaced as soon as something is seen
        self.push.reset(start, self.goal)
        self.last_target = np.asarray(obs["tool_pos"], float)

    def _initial_object(self, obs: dict, truth_xy: np.ndarray | None) -> np.ndarray | None:
        det = obs.get("detection")
        if det is not None and getattr(det, "visible", False) and det.xy is not None:
            return np.asarray(det.xy, float)
        return None

    # ---------- api ----------
    def act(self, obs: dict, truth_xy: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    def status(self) -> dict:
        b = self.belief
        return {
            "policy": self.name,
            "waits": self.waits,
            "recoveries": self.recoveries,
            "belief_xy": None if b is None or b.xy is None else np.round(b.xy, 4).tolist(),
            "belief_sd_mm": None if b is None else round(float(np.sqrt(b.var)) * 1000, 1),
            "belief_regime": None if b is None else b.regime,
            "invisible_for": None if b is None else b.steps_since_seen,
        }


class OracleAgent(Agent):
    name = "oracle"
    uses_truth = True

    def _initial_object(self, obs, truth_xy):
        return None if truth_xy is None else np.asarray(truth_xy, float)

    def act(self, obs: dict, truth_xy: np.ndarray | None = None) -> np.ndarray:
        if truth_xy is None:
            raise ValueError("oracle policy needs the simulator state")
        self.last_target = self.push.act(np.asarray(truth_xy, float), obs["tool_pos"], self.goal)
        return self.last_target


class CurrentObservationAgent(Agent):
    """Holds still whenever the object is not detected in the current frame."""

    name = "current-obs"

    def act(self, obs: dict, truth_xy: np.ndarray | None = None) -> np.ndarray:
        det = obs.get("detection")
        visible = det is not None and getattr(det, "visible", False) and det.xy is not None
        if visible:
            self.last_target = self.push.act(np.asarray(det.xy, float), obs["tool_pos"], self.goal)
        else:
            self.waits += 1
        return self.last_target


class HistoryAgent(Agent):
    """Keeps an estimate across occlusion and retreats when it is no longer trusted."""

    name = "history"

    def __init__(self, task_cfg, control_dt: float = 0.02,
                 tracker_cfg: TrackerConfig | None = None) -> None:
        super().__init__(task_cfg, control_dt)
        self.tracker = ObjectTracker(tracker_cfg)
        self.recovery_steps = 25
        self._prev_tool: np.ndarray | None = None

    def reset(self, obs, goal_xy, truth_xy=None):
        self.tracker.reset()
        self._prev_tool = None
        super().reset(obs, goal_xy, truth_xy)

    def act(self, obs: dict, truth_xy: np.ndarray | None = None) -> np.ndarray:
        det = obs.get("detection")
        visible = det is not None and getattr(det, "visible", False) and det.xy is not None
        tool_xy = np.asarray(obs["tool_pos"], float)
        tool_vel = None if self._prev_tool is None else (tool_xy - self._prev_tool) / self.dt
        self._prev_tool = tool_xy.copy()

        belief = self.tracker.update(
            np.asarray(det.xy, float) if visible else None,
            visible=visible,
            partial=bool(getattr(det, "partial", False)) if det is not None else False,
            dt=self.dt,
            tool_xy=tool_xy[:2],
            tool_vel=None if tool_vel is None else tool_vel[:2],
        )
        self.belief = belief
        if visible:
            self._last_seen = np.asarray(det.xy, float)

        if belief.regime == "lost" and self._recovery_left == 0:
            # back off to where the object was last seen, then look again
            self._recovery_left = self.recovery_steps
            self.recoveries += 1
            log.info("estimate lost -> retreat for %d steps", self.recovery_steps)

        if self._recovery_left > 0:
            self._recovery_left -= 1
            anchor = self._last_seen if self._last_seen is not None else belief.xy
            if anchor is not None:
                self.last_target = np.array([anchor[0], anchor[1], self.cfg.approach_height])
            return self.last_target

        if belief.usable and belief.xy is not None:
            self.last_target = self.push.act(np.asarray(belief.xy, float), obs["tool_pos"], self.goal)
        else:
            self.waits += 1
        return self.last_target


AGENTS = {
    "oracle": OracleAgent,
    "current-obs": CurrentObservationAgent,
    "history": HistoryAgent,
}


def make_agent(name: str, task_cfg, control_dt: float = 0.02) -> Agent:
    if name not in AGENTS:
        raise KeyError(f"unknown policy {name!r}; available: {sorted(AGENTS)}")
    return AGENTS[name](task_cfg, control_dt)
