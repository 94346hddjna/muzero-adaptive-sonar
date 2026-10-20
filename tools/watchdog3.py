# -*- coding: utf-8 -*-
"""防弹看门狗v3：守护results_v3实验进程。

- 字节模式捕获所有子进程输出（GBK/UTF-8混排安全）
- 每轮全部try/except，任何异常只记录不退出
- 心跳文件确认看门狗自身存活
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R3 = ROOT / "results_v3"
LOG = R3 / "watchdog.log"
HEARTBEAT = R3 / "watchdog_heartbeat.txt"


def log(msg):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def ps(args):
    """PowerShell调用，字节捕获安全解码。"""
    r = subprocess.run(["powershell", "-Command"] + args,
                       capture_output=True, cwd=str(ROOT))
    return (r.stdout + r.stderr).decode("utf-8", errors="replace")


def experiment_running():
    try:
        out = ps(["Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                  "Where-Object {$_.CommandLine -like '*configs_v3*'} | "
                  "Select-Object -ExpandProperty ProcessId"])
        return bool(out.strip())
    except Exception as e:
        log(f"experiment_running异常: {e}")
        return True  # 检查失败时宁可不动


def is_complete():
    # 当前处于"新基线补跑"阶段：以新基线训练全部完成为准
    # dqn_gru×5 + ppo_stack×5 的summary都存在才算完
    from pathlib import Path
    for algo in ["dqn_gru", "ppo_stack"]:
        for seed in [101, 102, 103, 104, 105]:
            p = R3 / "checkpoints" / algo / f"main_seed{seed}" / f"{algo}_main_seed{seed}_summary.json"
            if not p.exists():
                return False
    return True


def launch():
    try:
        R3.mkdir(parents=True, exist_ok=True)
        out = str((R3 / "full_run.log")).replace("\\", "/")
        err = str((R3 / "full_run_err.log")).replace("\\", "/")
        pid = ps([f"$p = Start-Process -FilePath 'python' -ArgumentList "
                  f"'-u','{ROOT / 'run_all_experiments.py'}','--mode','full',"
                  f"'--config-dir','configs_v3','--results-root','{R3}' "
                  f"-WindowStyle Hidden -WorkingDirectory '{ROOT}' "
                  f"-RedirectStandardOutput '{out}' "
                  f"-RedirectStandardError '{err}' -PassThru; Write-Output $p.Id"])
        log(f"已重新拉起实验进程 PID={pid.strip()}")
    except Exception as e:
        log(f"launch异常: {e}")


def _heartbeat_fresh():
    """心跳文件修改时间在6分钟内→另一看门狗存活。"""
    try:
        age = time.time() - HEARTBEAT.stat().st_mtime
        return age < 360
    except Exception:
        return False


def main():
    if _heartbeat_fresh():
        return  # 已有实例在运行（计划任务多触发保护）
    log("看门狗v3启动（每5分钟检查，心跳每轮写入）")
    n = 0
    while True:
        try:
            with open(HEARTBEAT, "w", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
            if is_complete():
                log("v3已完成（FINAL_SUMMARY存在），看门狗退出")
                return
            if not experiment_running():
                log("实验进程不存在，重启...")
                launch()
                time.sleep(600)
                continue
        except Exception as e:
            log(f"主循环异常(继续): {e}")
        n += 1
        time.sleep(300)


if __name__ == "__main__":
    main()
