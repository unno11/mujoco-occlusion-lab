"""Typed access to configs/*.json. Kept small on purpose: no config framework."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass
class ControlCfg:
    control_dt: float = 0.02
    ik_dt: float = 0.01
    position_cost: float = 1.0
    orientation_cost: float = 0.15
    posture_cost: float = 0.002
    solver: str = "daqp"


@dataclass
class TaskCfg:
    box_half_size: float = 0.028
    goal_xy: tuple[float, float] = (0.60, 0.0)
    goal_tol: float = 0.035
    start_x_range: tuple[float, float] = (0.36, 0.42)
    start_y_range: tuple[float, float] = (-0.09, 0.09)
    max_control_steps: int = 400
    push_height: float = 0.012
    approach_height: float = 0.110
    contact_gap: float = 0.072


@dataclass
class CameraCfg:
    name: str = "rgb"
    width: int = 640
    height: int = 480


@dataclass
class OccluderCfg:
    y: float = -0.12
    plate_half_width: float = 0.10
    plate_half_height: float = 0.11
    retracted_z: float = -0.30


@dataclass
class Config:
    scene: str = "assets/panda/scene_push.xml"
    control: ControlCfg = field(default_factory=ControlCfg)
    task: TaskCfg = field(default_factory=TaskCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    occluder: OccluderCfg = field(default_factory=OccluderCfg)

    @property
    def scene_path(self) -> pathlib.Path:
        p = pathlib.Path(self.scene)
        return p if p.is_absolute() else ROOT / p


def _fill(obj: Any, data: dict) -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key {key!r} for {type(obj).__name__}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _fill(current, value)
        elif isinstance(current, tuple):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)


def load_config(path: str | pathlib.Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        path = ROOT / "configs" / "default.json"
    path = pathlib.Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        _fill(cfg, json.load(f))
    return cfg
