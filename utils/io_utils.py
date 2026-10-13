# -*- coding: utf-8 -*-
"""IO工具：JSONL逐行写入/读取、CSV追加、目录骨架。"""
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RESULT_SUBDIRS = ["raw", "processed", "statistics", "figures", "tables",
                  "logs", "checkpoints", "reports"]


def ensure_results_tree(results_dir: Path = RESULTS_DIR):
    for sub in RESULT_SUBDIRS:
        (results_dir / sub).mkdir(parents=True, exist_ok=True)


class JsonlWriter:
    """逐行JSONL追加写（原始数据格式，results/raw/）。"""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._n = 0
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, record: Dict):
        self._fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        self._n += 1
        if self._n % 500 == 0:
            self._fh.flush()

    def flush(self):
        self._fh.flush()

    def close(self):
        self._fh.flush()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _json_default(o):
    """numpy类型→Python原生（含0维/多维数组）。"""
    import numpy as _np
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    if isinstance(o, torch_tensor_check()):
        return o.detach().cpu().tolist()
    return str(o)


def torch_tensor_check():
    try:
        import torch
        return torch.Tensor
    except ImportError:
        class _Never:  # 永不匹配
            pass
        return _Never


def write_json(path: Path, data, indent: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=_json_default)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def done_marker(dir_or_path: Path, name: str, payload: Optional[Dict] = None):
    """断点续训完成标记：已完成实验默认skip（第28章）。"""
    if dir_or_path.suffix:
        marker_dir = dir_or_path.parent
    else:
        marker_dir = dir_or_path
    marker_dir.mkdir(parents=True, exist_ok=True)
    p = marker_dir / f".done_{name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload or {"done": True, "os_pid": os.getpid()}, f, ensure_ascii=False, default=float)
    return p


def is_done(dir_or_path: Path, name: str) -> bool:
    if dir_or_path.suffix:
        marker_dir = dir_or_path.parent
    else:
        marker_dir = dir_or_path
    return (marker_dir / f".done_{name}.json").exists()


def read_done_payload(dir_or_path: Path, name: str) -> Optional[Dict]:
    if dir_or_path.suffix:
        marker_dir = dir_or_path.parent
    else:
        marker_dir = dir_or_path
    p = marker_dir / f".done_{name}.json"
    if not p.exists():
        return None
    try:
        return read_json(p)
    except Exception:
        return {"done": True}
