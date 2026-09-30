#!/usr/bin/bash
# large-384 / cfg=0.5 / N=2 — step gate 左端 (START_STEP) 的 50k FID/IS 扫描。
#
#   START_STEP = 0, 1, 2, 3, 4   (5 = 既有默认档, 已有结果 FID 2.6951 / IS 252.3132,
#                                 不重跑 —— 见 eval_fid_large384_lazycache_gatestart.sh
#                                 里"逐项一致"的说明)
#   共 5 个 50k eval, 单个约 2.5~2.7h, 串行合计约 13h。
#
# 只测 FID/IS —— 本次不做 latency / 理论 FLOPs。
#
# 用法:
#   setsid nohup bash launch/run_large384_gatestart_sweep.sh > /tmp/gatestart_sweep.log 2>&1 &
#   STARTS="0 2 4" bash launch/run_large384_gatestart_sweep.sh    # 只跑一部分
#   STARTS="6 7 8" PROGRESS_FILE=results/.foo.txt bash ...        # 换个进度文件, 别覆盖上一轮的
set -uo pipefail

export PATH="/artic/q-li/miniconda3/envs/maskgit/bin:$PATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PROGRESS="${PROGRESS_FILE:-$REPO/results/.gatestart_sweep_progress.txt}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROGRESS"; }

: > "$PROGRESS"
log "队列: START_STEP = ${STARTS:-0 1 2 3 4}  (large-384, cfg=0.5, N=2, 50k each)"

for S in ${STARTS:-0 1 2 3 4}; do
    log "===== START_STEP=$S begin ====="
    t0=$(date +%s)
    START_STEP="$S" REFRESH_N=2 bash launch/eval_fid_large384_lazycache_gatestart.sh 2>&1 | tee -a "$PROGRESS"
    rc=${PIPESTATUS[0]}
    el=$(( $(date +%s) - t0 ))
    log "===== START_STEP=$S done rc=$rc elapsed=$((el/3600))h$(( (el%3600)/60 ))m ====="
done

log "ALL DONE"
