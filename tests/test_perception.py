"""Tests for perception: calibration validity and detection on a rendered frame."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from occlab.config import load_config
from occlab.envs.push_env import PushEnv
from occlab.perception import calibration
from occlab.perception.color_blob import ColorBlobDetector

ROOT = pathlib.Path(__file__).resolve().parents[1]
CALIB = ROOT / "configs" / "calibration.json"

pytestmark = pytest.mark.skipif(not CALIB.exists(), reason="run scripts/20_calibrate.py first")


def test_calibration_maps_the_goal(cfg):
    H, meta = calibration.load(CALIB)
    assert meta["goal_error_mm"] < 25, "homography must place the goal patch correctly"
    assert meta["grid_error_mean_mm"] < 40, "detection bias must be under the goal tolerance"


def test_detector_finds_the_box_near_truth(cfg):
    env = PushEnv(cfg, render=True, condition="A")
    env.reset(seed=4)
    det = env.detect()
    truth = env.object_truth_xy()
    env.close()
    assert det.visible and det.xy is not None
    assert np.linalg.norm(det.xy - truth) < 0.05


def test_detector_reports_partial_occlusion(cfg):
    env = PushEnv(cfg, render=True, condition="A")
    env.reset(seed=4)
    full = env.detect()
    env.set_occluder(env.occlusion_z_for(0.55))
    part = env.detect()
    env.close()
    assert full.area_px > 0
    assert part.area_px < full.area_px, "partial occlusion must shrink the blob"


def test_detector_reports_full_occlusion(cfg):
    env = PushEnv(cfg, render=True, condition="A")
    env.reset(seed=4)
    env.set_occluder(env.occlusion_z_for(1.0, margin=0.03))
    det = env.detect()
    env.close()
    assert not det.visible, f"expected no detection, got {det}"
    assert det.xy is None


def test_goal_detection_is_stable(cfg):
    env = PushEnv(cfg, render=True, condition="A")
    env.reset(seed=0)
    H, _ = calibration.load(CALIB)
    goal_px = calibration.detect_goal_pixel(env.render())
    env.close()
    est = calibration.pixel_to_world(H, goal_px)
    assert np.linalg.norm(est - cfg.task.goal_xy) < 0.03
