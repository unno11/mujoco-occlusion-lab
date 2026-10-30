"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from occlab.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()
