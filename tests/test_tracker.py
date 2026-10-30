"""Tests for the tracker: the three regimes, prediction, correction, uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from occlab.state_estimation.object_tracker import ObjectTracker, TrackerConfig

DT = 0.02


def feed(tracker, meas_list, visible=True):
    belief = None
    for m in meas_list:
        belief = tracker.update(np.array(m, float) if m is not None else None,
                                visible=visible, partial=False, dt=DT)
    return belief


def test_first_measurement_initialises():
    t = ObjectTracker()
    b = t.update(np.array([0.4, 0.1]), visible=True, partial=False, dt=DT)
    assert b.regime == "measured"
    assert np.allclose(b.xy, [0.4, 0.1])
    assert b.visible


def test_invisible_without_history_is_unknown():
    t = ObjectTracker()
    b = t.update(None, visible=False, partial=False, dt=DT)
    assert b.regime == "unknown"
    assert b.xy is None
    assert not b.usable


def test_coast_keeps_a_usable_estimate_and_grows_uncertainty():
    t = ObjectTracker()
    feed(t, [[0.40, 0.0], [0.402, 0.0], [0.404, 0.0]])
    var_before = t.var
    b = feed(t, [None] * 5, visible=False)
    assert b.regime == "predicted"
    assert b.xy is not None
    assert b.var > var_before
    assert b.steps_since_seen == 5


def test_prediction_follows_the_tool_when_pushing():
    t = ObjectTracker()
    feed(t, [[0.40, 0.0], [0.402, 0.0]])
    b = t.update(None, visible=False, partial=False, dt=DT,
                 tool_xy=np.array([0.41, 0.0]), tool_vel=np.array([0.10, 0.0]))
    assert b.xy is not None and b.xy[0] > 0.402, "must move along the pusher"


def test_prediction_decays_when_tool_is_far():
    t = ObjectTracker()
    feed(t, [[0.40, 0.0], [0.402, 0.0]])
    speed_before = np.linalg.norm(t.vel)
    for _ in range(3):
        t.update(None, visible=False, partial=False, dt=DT,
                 tool_xy=np.array([0.90, 0.30]), tool_vel=np.array([0.0, 0.0]))
    assert np.linalg.norm(t.vel) <= speed_before + 1e-9


def test_reobservation_corrects_the_estimate():
    t = ObjectTracker()
    feed(t, [[0.40, 0.0], [0.402, 0.0], [0.404, 0.0]])
    feed(t, [None] * 10, visible=False)
    assert np.linalg.norm(t.xy - np.array([0.404, 0.0])) > 0.0
    b = t.update(np.array([0.50, 0.02]), visible=True, partial=False, dt=DT)
    assert b.regime == "measured"
    assert abs(b.xy[0] - 0.50) < 0.03, "fresh measurement must pull the estimate back"


def test_long_occlusion_becomes_lost():
    t = ObjectTracker(TrackerConfig(max_coast_steps=5))
    feed(t, [[0.40, 0.0], [0.402, 0.0]])
    b = feed(t, [None] * 8, visible=False)
    assert b.regime == "lost"


def test_reset_clears_state():
    t = ObjectTracker()
    feed(t, [[0.4, 0.0], [0.42, 0.0]])
    t.reset()
    assert t.xy is None and t.regime == "unknown"
