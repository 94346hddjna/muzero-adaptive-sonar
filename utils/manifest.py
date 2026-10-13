# -*- coding: utf-8 -*-
"""实验Manifest（第32章）：每次实验的可追溯记录。"""
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Dict, Optional

from utils import hardware as hw
from utils.io_utils import write_json


def _config_hash(cfg: Dict) -> str:
    s = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def make_manifest(
    experiment_id: str,
    algorithm: str,
    seed: Optional[int],
    difficulty: str,
    config: Dict,
    training_steps: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    stage: str = "main",
    extra: Optional[Dict] = None,
) -> Dict:
    hwd = hw.detect_hardware()
    manifest = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "algorithm": algorithm,
        "seed": seed,
        "difficulty": difficulty,
        "hyperparameters": config,
        "config_sha256_16": _config_hash(config),
        "training_steps": training_steps,
        "checkpoint_path": checkpoint_path,
        "software": {
            "python": platform.python_version(),
            "pytorch": hwd["pytorch_version"],
            "numpy": __import__("numpy").__version__,
            "scipy": __import__("scipy").__version__,
            "os": platform.platform(),
        },
        "hardware": {
            "cpu_count": hwd.get("cpu_count"),
            "ram_total_gb": hwd.get("ram_total_gb"),
            "gpu_name": hwd.get("gpu_name"),
            "gpu_memory_total_gb": hwd.get("gpu_memory_total_gb"),
            "cuda_version": hwd.get("cuda_version"),
            "cuda_available": hwd.get("cuda_available"),
        },
        "git_commit": _git_commit(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def _git_commit() -> Optional[str]:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def save_manifest(manifest: Dict, results_dir: Path, subdir: str = "raw") -> Path:
    out = results_dir / subdir / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{manifest['experiment_id']}.json"
    write_json(path, manifest)
    return path
