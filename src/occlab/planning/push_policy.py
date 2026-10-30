"""Pushing controller: scripted, closed-loop on the object estimate.

Three phases, deliberately boring and explainable:

    GO_ABOVE -> DESCEND -> ADVANCE

The tool first moves to a point *behind* the object (on the line object -> goal,
with a generous stand-off so the fingers never land on top of the box), drops to
push height there, and then advances along that line until the object reaches the
goal or the step budget runs out. If the object estimate drifts off the planned
line, the policy replans a fresh line instead of pushing blindly.
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np

log = logging.getLogger(__name__)


class Phase(str, Enum):
    GO_ABOVE = "go_above"
    DESCEND = "descend"
    ADVANCE = "advance"
    DONE = "done"


class PushPolicy:
    def __init__(self, task_cfg, advance_step: float = 0.004) -> None:
        self.cfg = task_cfg
        self.advance_step = advance_step
        self.phase = Phase.GO_ABOVE
        self.direction = np.array([1.0, 0.0])
        self.line_origin = np.zeros(2)   # contact point on the planned line
        self.contact = np.zeros(2)
        self.advance = 0.0
        self.replans = 0
        self._last_est: np.ndarray | None = None

    # ---------- helpers ----------
    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.array([1.0, 0.0])

    def _contact_for(self, obj_xy: np.ndarray, goal_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self._unit(goal_xy - obj_xy)
        standoff = self.cfg.box_half_size + self.cfg.contact_gap
        return obj_xy - d * standoff, d

    def _sanitize(self, obj_xy: np.ndarray) -> np.ndarray:
        est = np.asarray(obj_xy, dtype=float)
        if self._last_est is not None and np.linalg.norm(est - self._last_est) > 0.12:
            log.warning("estimate jumped %.3f m, keeping previous", float(np.linalg.norm(est - self._last_est)))
            return self._last_est
        self._last_est = est
        return est

    # ---------- api ----------
    def reset(self, obj_xy: np.ndarray, goal_xy: np.ndarray) -> None:
        obj = np.asarray(obj_xy, float)
        self.contact, self.direction = self._contact_for(obj, np.asarray(goal_xy, float))
        self.line_origin = self.contact.copy()
        self.advance = 0.0
        self.phase = Phase.GO_ABOVE
        self.replans = 0
        self._last_est = None

    def _replan(self, obj: np.ndarray, goal: np.ndarray) -> None:
        self.contact, self.direction = self._contact_for(obj, goal)
        self.line_origin = self.contact.copy()
        self.advance = 0.0
        self.phase = Phase.GO_ABOVE
        self.replans += 1

    def act(self, obj_xy_est: np.ndarray, tip_xyz: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
        if self.phase is Phase.DONE:
            return np.asarray(tip_xyz, float)

        obj = self._sanitize(obj_xy_est)
        goal = np.asarray(goal_xy, float)
        tip = np.asarray(tip_xyz, float)

        # drift check: if the object left the planned push line, replan
        if self.phase in (Phase.DESCEND, Phase.ADVANCE):
            lateral = abs(
                float(self.direction[0] * (obj - self.line_origin)[1]
                      - self.direction[1] * (obj - self.line_origin)[0])
            )
            if lateral > 0.035:
                log.info("object off the push line by %.3f m -> replan", lateral)
                self._replan(obj, goal)

        if self.phase is Phase.GO_ABOVE:
            target = np.array([self.line_origin[0], self.line_origin[1], self.cfg.approach_height])
            ready = np.linalg.norm(tip[:2] - self.line_origin) < 0.015 and abs(tip[2] - self.cfg.approach_height) < 0.02
            if ready:
                self.phase = Phase.DESCEND
            return target

        if self.phase is Phase.DESCEND:
            target = np.array([self.line_origin[0], self.line_origin[1], self.cfg.push_height])
            if tip[2] < self.cfg.push_height + 0.008:
                self.phase = Phase.ADVANCE
                self.advance = 0.0
            return target

        # ADVANCE: creep along the frozen line; the box is in the way, that is the point
        self.advance += self.advance_step
        point = self.line_origin + self.direction * self.advance
        return np.array([point[0], point[1], self.cfg.push_height])

    def finish(self) -> None:
        self.phase = Phase.DONE
