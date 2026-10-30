"""Tests for the environment: determinism, success logic, occlusion geometry."""

from __future__ import annotations

import numpy as np
import pytest

from occlab.config import load_config
from occlab.envs.push_env import PushEnv


def test_reset_is_deterministic(cfg):
    env = PushEnv(cfg, render=False)
    a = env.reset(seed=7)
    box_a = env.object_truth_xy().copy()
    b = env.reset(seed=7)
    box_b = env.object_truth_xy().copy()
    env.close()
    assert np.allclose(box_a, box_b), "same seed must give the same start"
    assert a["tool_pos"].shape == (3,)


def test_different_seeds_differ(cfg):
    env = PushEnv(cfg, render=False)
    env.reset(seed=0)
    a = env.object_truth_xy().copy()
    env.reset(seed=1)
    b = env.object_truth_xy().copy()
    env.close()
    assert not np.allclose(a, b)


def test_box_starts_on_the_table(cfg):
    env = PushEnv(cfg, render=False)
    env.reset(seed=3)
    pose = env.object_truth_pose()
    env.close()
    assert abs(pose[2] - cfg.task.box_half_size) < 0.01, "box should rest on the table"


def test_success_terminates_and_metrics_agree(cfg):
    env = PushEnv(cfg, render=False)
    env.reset(seed=5)
    # teleport the box into the goal to check the success path end to end
    adr = int(env.model.jnt_qposadr[env.model.joint("box_free").id])
    env.data.qpos[adr : adr + 2] = env.goal() + 0.001
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    obs, reward, done, truncated, info = env.step(env.tool_position())
    m = env.get_metrics()
    env.close()
    assert done and info["success"] and m["success"]
    assert m["final_dist"] < cfg.task.goal_tol


def test_budget_truncates(cfg):
    env = PushEnv(cfg, render=False)
    env.reset(seed=0)
    done = False
    steps = 0
    while not done:
        obs, _, done, truncated, info = env.step(np.array([0.30, 0.20, 0.25]))
        steps += 1
    env.close()
    assert truncated and steps == cfg.task.max_control_steps


def test_occlusion_geometry_is_monotonic(cfg):
    """A higher plate must hide more of the object (smaller z means lower)."""
    env = PushEnv(cfg, render=False, condition="A")
    env.reset(seed=0)
    z_none = env.cfg.occluder.retracted_z
    z_partial = env.occlusion_z_for(0.5)
    z_full = env.occlusion_z_for(1.0, margin=0.02)
    env.close()
    assert z_none < z_partial < z_full


def test_condition_b_raises_the_plate_at_reset(cfg):
    env = PushEnv(cfg, render=False, condition="B")
    env.reset(seed=0)
    z = env.get_occluder()["qpos"]
    env.close()
    assert z > cfg.occluder.retracted_z, "condition B must start partially occluded"


def test_condition_a_keeps_the_plate_down(cfg):
    env = PushEnv(cfg, render=False, condition="A")
    env.reset(seed=0)
    z = env.get_occluder()["qpos"]
    env.close()
    assert z == pytest.approx(cfg.occluder.retracted_z)
