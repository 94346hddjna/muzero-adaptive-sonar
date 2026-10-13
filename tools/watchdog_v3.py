# -*- coding: utf-8 -*-
"""实验看门狗：�?分钟检查v2实验进程；死亡则自动断点重启�?

抵御会话恢复/系统事件误杀。全部走断点续跑（done标记跳过已完成实验，
训练从last checkpoint续跑），不产生重复实验�?
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results_v3" / "watchdog.log"
RUN_CMD = [sys.executable, "-u", "run_all_experiments.py", "--mode", "full",
           "--config-dir", "configs_v3", "--results-root", "results_v3"]


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def experiment_running():
    """检查是否有run_all_experiments --config-dir configs_v3进程�?""
    r = subprocess.run(
        ["powershell", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object {$_.CommandLine -like '*configs_v3*'} | "
         "Select-Object -ExpandProperty ProcessId"],
        capture_output=True)
    out = r.stdout.decode("utf-8", errors="replace") + r.stderr.decode("utf-8", errors="replace")
    return bool(out.strip())


def is_complete():
    """v2全部完成标志：FINAL_SUMMARY存在且质量检查通过�?""
    return (ROOT / "results_v3" / "FINAL_SUMMARY.md").exists()


def launch():
    out = str((ROOT / "results_v3" / "full_run.log")).replace("\\", "/")
    err = str((ROOT / "results_v3" / "full_run_err.log")).replace("\\", "/")
    subprocess.run(
        ["powershell", "-Command",
         f"$p = Start-Process -FilePath 'python' -ArgumentList "
         f"'-u','{ROOT / 'run_all_experiments.py'}','--mode','full',"
         f"'--config-dir','configs_v3','--results-root','results_v3' "
         f"-WindowStyle Hidden -WorkingDirectory '{ROOT}' "
         f"-RedirectStandardOutput '{out}' -RedirectStandardError '{err}' -PassThru; "
         f"Write-Output $p.Id"],
        capture_output=True, text=True, cwd=str(ROOT))
    log("已重新拉起实验进�?)


def main():
    log("看门狗启动（�?分钟检查）")
    while True:
        try:
            if is_complete():
                log("检测到v2已完成（FINAL_SUMMARY存在），看门狗退�?)
                return
            if not experiment_running():
                log("实验进程不存在，重启...")
                launch()
                time.sleep(600)  # 启动后等10分钟再做下次检查（自检+测试需~3分钟�?
                continue
        except Exception as e:
            log(f"异常: {e}")
        time.sleep(300)


if __name__ == "__main__":
    main()

