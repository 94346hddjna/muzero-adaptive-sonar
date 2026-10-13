# -*- coding: utf-8 -*-
"""硬件资源自适应检测（第30章）：CPU/RAM/GPU/CUDA/显存/PyTorch状态。"""
import json
import platform
from pathlib import Path

import torch


def detect_hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": __import__("os").cpu_count(),
        "machine": platform.machine(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / 1e9, 2)
    except ImportError:
        info["ram_total_gb"] = None
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu_name"] = None
        info["gpu_memory_total_gb"] = None
        info["cuda_version"] = torch.version.cuda
    return info


def resolve_device(pref: str = "auto") -> str:
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def adapt_workers(hardware: dict) -> dict:
    """根据硬件自动选择并行度（不改变算法/环境/指标定义）。"""
    cpu = hardware.get("cpu_count") or 4
    has_gpu = hardware.get("cuda_available", False)
    return {
        "num_parallel_envs_suggested": 64 if has_gpu else 32,
        "eval_workers_suggested": max(1, min(4, cpu // 4)),
        "note": "仅并行度自适应；算法定义、环境定义、评价指标与公平比较协议不变",
    }


def save_hardware_report(path: Path):
    info = detect_hardware()
    info["worker_adaptation"] = adapt_workers(info)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info
