# -*- coding: utf-8 -*-
"""测试公共fixture。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def env_cfg():
    return load_config("environment")["environment"]


@pytest.fixture(scope="session")
def l2cfg():
    return load_config("environment")["level2"]


@pytest.fixture(scope="session")
def muzero_cfg():
    from utils.config import load_section
    return load_section("muzero", smoke=True)


@pytest.fixture(scope="session")
def t_max():
    return 20
