# -*- coding: utf-8 -*-
"""强化轮(v2)自动接续启动器。

等待当前FULL实验（v1）进程退出 → v2 smoke验证 → 验证集超参sweep → 启动v2 FULL。
全程无人值守；v2结果写入results_v2/，与v1完全隔离（公平协议不变：同预算/同场景/同评测）。
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = ROOT / "results" / "logs" / "chain_round2.log"


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def pid_alive(pid):
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
    return "python.exe" in r.stdout


def wait_for_exit(pid, timeout_s=48 * 3600):
    log(f"等待v1实验进程 PID={pid} 退出...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not pid_alive(pid):
            log(f"v1进程已退出（等待了{(time.time()-t0)/60:.0f}分钟）")
            return True
        time.sleep(120)
    log("等待超时（48h），放弃接续")
    return False


def main(v1_pid=22632):
    log("接续启动器开始运行")
    if not wait_for_exit(v1_pid):
        return
    # v1结束后：确认其analysis阶段完整（final summary存在）
    for _ in range(180):  # 最多等30分钟让v1收尾写报告
        if (ROOT / "results" / "FINAL_SUMMARY.md").exists():
            break
        time.sleep(10)
    log("v1已收尾，开始v2流程")
    # 1) v2 smoke（CPU跑避免干扰；验证端到端）
    r = subprocess.run([sys.executable, "-u", str(ROOT / "run_all_experiments.py"),
                        "--mode", "smoke", "--config-dir", "configs_v2",
                        "--results-root", "results_v2_smoke"],
                       capture_output=True, text=True, timeout=3600,
                       cwd=str(ROOT))
    ok = r.returncode == 0
    log(f"v2 smoke: {'OK' if ok else 'FAILED'}")
    if not ok:
        log("v2 smoke失败，输出尾部:\n" + "\n".join((r.stdout or "").split("\n")[-20:]))
        return
    # 2) 验证集超参sweep
    log("开始验证集超参sweep（4配置×300k步，约2-3小时）")
    r = subprocess.run([sys.executable, "-u", str(ROOT / "experiments" / "hp_sweep.py"),
                        "--config-dir", "configs_v2", "--results-root", "results_v2"],
                       capture_output=True, text=True, timeout=8 * 3600, cwd=str(ROOT))
    log(f"hp_sweep: {'OK' if r.returncode == 0 else 'FAILED'}\n" +
        "\n".join((r.stdout or "").split("\n")[-10:]))
    if r.returncode != 0:
        return
    # 3) v2 FULL（完全脱离启动，日志绝对路径）
    (ROOT / "results_v2").mkdir(parents=True, exist_ok=True)
    out_log = str(ROOT / "results_v2" / "full_run.log").replace("\\", "/")
    err_log = str(ROOT / "results_v2" / "full_run_err.log").replace("\\", "/")
    r = subprocess.run(
        ["powershell", "-Command",
         f"$p = Start-Process -FilePath 'python' -ArgumentList "
         f"'-u','{ROOT / 'run_all_experiments.py'}','--mode','full',"
         f"'--config-dir','configs_v2','--results-root','{ROOT / 'results_v2'}' "
         f"-WindowStyle Hidden -WorkingDirectory '{ROOT}' "
         f"-RedirectStandardOutput '{out_log}' "
         f"-RedirectStandardError '{err_log}' -PassThru; Write-Output $p.Id"],
        capture_output=True, text=True, cwd=str(ROOT))
    log(f"v2 FULL已启动: {r.stdout.strip()} (stderr={r.stderr.strip()[:200]})")


if __name__ == "__main__":
    main()
