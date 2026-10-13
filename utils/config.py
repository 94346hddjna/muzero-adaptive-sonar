# -*- coding: utf-8 -*-
"""配置加载：YAML配置 + smoke模式覆盖。

约定：每个yaml的主节与文件同名（training.yaml→training节）；smoke模式覆盖
写在顶层 `smoke:` 节（映射：节名→覆盖dict），仅作用于load_section。
"""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

_CACHE = {}


def set_config_dir(path):
    """切换配置目录（实验轮次v2用configs_v2；切换后清空缓存）。"""
    global CONFIG_DIR
    CONFIG_DIR = Path(path)
    _CACHE.clear()


def load_config(name: str, smoke: bool = False) -> dict:
    """加载完整yaml文档（含smoke节）。smoke参数仅为缓存键区分，不再改写文档。"""
    key = (name, smoke)
    if key in _CACHE:
        return copy.deepcopy(_CACHE[key])
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _CACHE[key] = cfg
    return copy.deepcopy(cfg)


def load_section(name: str, smoke: bool = False) -> dict:
    """加载 configs/<name>.yaml 的主节（与文件同名的顶层键）。

    smoke=True时用顶层smoke节中<name>的dict做深度覆盖。
    """
    cfg = load_config(name, smoke=False)
    section = cfg.get(name)
    if not isinstance(section, dict):
        section = {k: v for k, v in cfg.items() if k != "smoke"}
    if smoke:
        ov = (cfg.get("smoke") or {}).get(name) or {}
        section = merge_config(section, ov)
    return copy.deepcopy(section)


def merge_config(base: dict, override: dict) -> dict:
    """深度合并（override优先），用于实验期参数覆盖（OOD/消融等）。"""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
