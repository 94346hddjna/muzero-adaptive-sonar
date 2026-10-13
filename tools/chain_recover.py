# -*- coding: utf-8 -*-
"""恢复编排器：v1 sweep/OOD重评测 → v1最终分析 → v2超参sweep → v2 FULL。

背景：修复了sweep场景种子流的跨条件共享问题后，需重新生成并重评测受影响的
robustness/OOD数据（各条件内部配对评测原本有效，此为根因修复后的数据卫生重跑）。
每步可安全重入；中途死亡后重新运行本脚本即续跑。
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "logs" / "chain_recover.log"


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_step(name, args, timeout_s=24 * 3600):
    log(f"开始 {name}: {' '.join(str(a) for a in args)}")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u"] + [str(a) for a in args],
                       cwd=str(ROOT), timeout=timeout_s,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0
    tail = "\n".join((r.stdout or "").split("\n")[-8:])
    log(f"{name}: {'OK' if ok else 'FAILED'} ({(time.time()-t0)/60:.0f}分钟)\n{tail}")
    if not ok:
        err = "\n".join((r.stderr or "").split("\n")[-15:])
        log(f"{name} 错误尾部:\n{err}")
    return ok


def main():
    log("恢复编排器启动")
    # 1) v1 sweep/OOD重评测（种子流修复后）
    if not run_step("v1-robustness-reeval",
                    [ROOT / "run_all_experiments.py", "--mode", "full",
                     "--stage", "robustness"]):
        return
    # 2) v1最终分析（干净的质量检查+报告）
    if not run_step("v1-analysis",
                    [ROOT / "run_all_experiments.py", "--mode", "full",
                     "--stage", "analysis"]):
        return
    # 3) v2超参sweep（验证集驱动）
    if not run_step("v2-hp-sweep",
                    [ROOT / "experiments" / "hp_sweep.py",
                     "--config-dir", "configs_v2", "--results-root", "results_v2"]):
        return
    # 4) v2 FULL（脱离启动）
    (ROOT / "results_v2").mkdir(parents=True, exist_ok=True)
    out_log = str((ROOT / "results_v2" / "full_run.log")).replace("\\", "/")
    err_log = str((ROOT / "results_v2" / "full_run_err.log")).replace("\\", "/")
    r = subprocess.run(
        ["powershell", "-Command",
         f"$p = Start-Process -FilePath 'python' -ArgumentList "
         f"'-u','{ROOT / 'run_all_experiments.py'}','--mode','full',"
         f"'--config-dir','configs_v2','--results-root','{ROOT / 'results_v2'}' "
         f"-WindowStyle Hidden -WorkingDirectory '{ROOT}' "
         f"-RedirectStandardOutput '{out_log}' "
         f"-RedirectStandardError '{err_log}' -PassThru; Write-Output $p.Id"],
        capture_output=True, text=True, cwd=str(ROOT))
    log(f"v2 FULL已启动: PID={r.stdout.strip()} (stderr={r.stderr.strip()[:200]})")


if __name__ == "__main__":
    main()
