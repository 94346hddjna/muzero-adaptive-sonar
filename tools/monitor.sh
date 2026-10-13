#!/bin/bash
# 持续监控：每20分钟记录训练进度与进程状态
cd "$(dirname "$0")/.."
while true; do
  {
    echo "=== $(date '+%m-%d %H:%M:%S') ==="
    python -c "
import json, glob
lines = []
for f in sorted(glob.glob('results/checkpoints/*/*/*_progress.json')):
    d = json.load(open(f))
    lines.append(f\"{d['exp_id']}: {d['env_steps']//1000}k, best_val={d['best_val_sr']:.3f}\")
print('\n'.join(lines))
" 2>/dev/null
    tasklist //FI "PID eq 1524" //FO CSV 2>/dev/null | tail -1
    tail -2 results/logs/full_run.log 2>/dev/null
  } >> results/logs/monitor.log 2>&1
  sleep 1200
done
