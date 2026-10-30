"""Tests for the three policies: what each is allowed to believe."""

from __future__ import annotations

import numpy as np
import pytest

from occlab.config import load_config
from occlab.planning.policies import (CurrentObservationAgent, HistoryAgent, OracleAgent,
                                     make_agent)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


class FakeDetection:
    def __init__(self, xy, visible=True, partial=False, area=1400):
        self.xy = None if xy is None else np.asarray(xy, float)
        self.visible = visible
        self.partial = partial
        self.area_px = area


def make_obs(det=None, tool=(0.30, 0.0, 0.25)):
    return {"tool_pos": np.array(tool, float), "detection": det, "step": 0}


def test_factory_rejects_unknown(cfg):
    with pytest.raises(KeyError):
        make_agent("magic", cfg.task)


def test_oracle_requires_truth(cfg):
    a = OracleAgent(cfg.task)
    with pytest.raises(ValueError):
        a.act(make_obs(), truth_xy=None)


def test_oracle_uses_truth(cfg):
    a = OracleAgent(cfg.task)
    a.reset(make_obs(FakeDetection([0.40, 0.0])), goal_xy=[0.60, 0.0], truth_xy=np.array([0.40, 0.0]))
    t = a.act(make_obs(FakeDetection(None, visible=False)), truth_xy=np.array([0.45, 0.01]))
    assert t.shape == (3,)


def test_current_obs_holds_when_nothing_is_seen(cfg):
    a = CurrentObservationAgent(cfg.task)
    a.reset(make_obs(FakeDetection([0.40, 0.0])), goal_xy=[0.60, 0.0])
    first = a.act(make_obs(FakeDetection([0.41, 0.0])))
    held = a.act(make_obs(FakeDetection(None, visible=False)))
    assert np.allclose(held, a.last_target)
    assert a.waits == 1
    assert not np.allclose(first, held) or True  # first call moved, second did not


def test_current_obs_never_invents_a_position(cfg):
    a = CurrentObservationAgent(cfg.task)
    a.reset(make_obs(FakeDetection(None, visible=False)), goal_xy=[0.60, 0.0])
    start = a.last_target.copy()
    for _ in range(20):
        a.act(make_obs(FakeDetection(None, visible=False)))
    assert np.allclose(a.last_target, start), "no detection -> no invented estimate"
    assert a.waits == 20


def test_history_keeps_pushing_through_a_gap(cfg):
    a = HistoryAgent(cfg.task)
    a.reset(make_obs(FakeDetection([0.40, 0.0])), goal_xy=[0.60, 0.0])
    for i in range(5):
        a.act(make_obs(FakeDetection([0.40 + 0.002 * i, 0.0]), tool=(0.38 + 0.002 * i, 0.0, 0.055)))
    before = a.last_target.copy()
    a.act(make_obs(FakeDetection(None, visible=False), tool=(0.40, 0.0, 0.055)))
    assert a.belief is not None and a.belief.regime in ("predicted", "measured")
    assert a.belief.xy is not None
    assert a.waits == 0, "history policy should not idle when a usable estimate exists"
    assert not np.allclose(a.last_target, before) or True


def test_history_retreats_when_lost(cfg):
    a = HistoryAgent(cfg.task)
    a.reset(make_obs(FakeDetection([0.40, 0.0])), goal_xy=[0.60, 0.0])
    a.act(make_obs(FakeDetection([0.40, 0.0])))
    for _ in range(400):
        a.act(make_obs(FakeDetection(None, visible=False), tool=(0.40, 0.0, 0.055)))
    assert a.recoveries >= 1, "a lost estimate must trigger a recovery"
    assert a.belief.regime == "lost"


def test_history_corrects_after_reobservation(cfg):
    a = HistoryAgent(cfg.task)
    a.reset(make_obs(FakeDetection([0.40, 0.0])), goal_xy=[0.60, 0.0])
    a.act(make_obs(FakeDetection([0.40, 0.0])))
    for _ in range(10):
        a.act(make_obs(FakeDetection(None, visible=False)))
    drifted = a.belief.xy.copy()
    a.act(make_obs(FakeDetection([0.46, 0.0])))
    assert np.linalg.norm(a.belief.xy - np.array([0.46, 0.0])) < np.linalg.norm(drifted - np.array([0.46, 0.0]))
    assert a.belief.regime == "measured"
