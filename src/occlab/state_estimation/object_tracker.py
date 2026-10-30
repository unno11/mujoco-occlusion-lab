"""Object state estimation across occlusion.

Deliberately small and explainable: a constant-velocity predictor corrected by
measurements, with an explicit uncertainty that grows while the object is not
visible. There is no learned world model here and none is claimed.

Three regimes, all reported to the caller so the policy can react:

* ``measured``  -- a detection arrived this step; the estimate is corrected.
* ``predicted`` -- no detection, but the object was seen recently; the estimate is
  propagated with the last reliable velocity.
* ``lost``      -- no detection and uncertainty above the threshold; the estimate is
  held but flagged, and the policy is expected to do something cautious.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Belief:
    xy: np.ndarray | None
    var: float                 # position variance proxy, m^2
    regime: str                # measured | predicted | lost | unknown
    steps_since_seen: int
    visible: bool
    partial: bool

    @property
    def usable(self) -> bool:
        return self.xy is not None and self.regime != "unknown"

    def __str__(self) -> str:
        if self.xy is None:
            return f"<unknown {self.regime}>"
        return (f"<{self.regime} xy={np.round(self.xy, 3)} sd={np.sqrt(self.var) * 1000:.0f}mm "
                f"invisible_for={self.steps_since_seen}>")


@dataclass
class TrackerConfig:
    measurement_var: float = 0.02 ** 2   # ~20 mm p95 detection error
    process_var: float = 0.02 ** 2       # unmodelled object motion per second
    velocity_alpha: float = 0.5          # EMA on the measured velocity
    max_coast_steps: int = 120           # beyond this the estimate is "lost"
    lost_var: float = 0.06 ** 2          # sd > 60 mm -> lost
    max_speed: float = 0.4               # m/s, sanity clamp on the predictor
    contact_radius: float = 0.10         # m, tool within this distance is "pushing"
    velocity_decay: float = 0.85         # per hidden step, when the tool is far


class ObjectTracker:
    """Constant-velocity tracker with visibility bookkeeping."""

    def __init__(self, cfg: TrackerConfig | None = None) -> None:
        self.cfg = cfg or TrackerConfig()
        self.reset()

    def reset(self) -> None:
        self.xy: np.ndarray | None = None
        self.vel = np.zeros(2)
        self.var = self.cfg.measurement_var
        self.steps_since_seen = 0
        self.regime = "unknown"
        self._prev_meas: np.ndarray | None = None

    def update(self, det_xy: np.ndarray | None, visible: bool, partial: bool, dt: float,
               tool_xy: np.ndarray | None = None, tool_vel: np.ndarray | None = None) -> Belief:
        """Fuse one frame.

        While the object is invisible the prediction is driven by the *tool* velocity
        whenever the tool is close to the estimate: during a push the object moves with
        the pusher, and extrapolating the object's own last velocity instead makes the
        estimate run away as soon as the pusher slows down or stops (measured: 7 cm of
        drift over 60 hidden steps, which then triggered a pointless replan).
        """
        cfg = self.cfg
        if visible and det_xy is not None:
            meas = np.asarray(det_xy, float)
            if self.xy is None:
                self.xy = meas.copy()
                self.vel[:] = 0.0
            else:
                pred = self.xy + self.vel * dt
                # correction: trust a fresh measurement, keep a little of the prediction
                self.xy = 0.7 * meas + 0.3 * pred
                if self._prev_meas is not None and dt > 0:
                    raw = (meas - self._prev_meas) / dt
                    speed = float(np.linalg.norm(raw))
                    if speed > cfg.max_speed:
                        raw *= cfg.max_speed / speed
                    self.vel = (1 - cfg.velocity_alpha) * self.vel + cfg.velocity_alpha * raw
            self._prev_meas = meas.copy()
            self.var = cfg.measurement_var
            self.steps_since_seen = 0
            self.regime = "measured"
        else:
            self.steps_since_seen += 1
            if self.xy is None:
                self.regime = "unknown"
            else:
                near_tool = (
                    tool_xy is not None
                    and tool_vel is not None
                    and float(np.linalg.norm(np.asarray(tool_xy, float) - self.xy)) < cfg.contact_radius
                )
                if near_tool:
                    # object is being pushed: follow the pusher
                    self.vel = 0.5 * self.vel + 0.5 * np.asarray(tool_vel, float)
                else:
                    self.vel = self.vel * cfg.velocity_decay
                speed = float(np.linalg.norm(self.vel))
                if speed > cfg.max_speed:
                    self.vel *= cfg.max_speed / speed
                self.xy = self.xy + self.vel * dt
                self.var += cfg.process_var * dt + (speed * dt) ** 2 * 0.1
                self._prev_meas = None
                if self.steps_since_seen > cfg.max_coast_steps or self.var > cfg.lost_var:
                    self.regime = "lost"
                else:
                    self.regime = "predicted"
        return Belief(
            xy=None if self.xy is None else self.xy.copy(),
            var=float(self.var),
            regime=self.regime,
            steps_since_seen=self.steps_since_seen,
            visible=visible,
            partial=partial,
        )
