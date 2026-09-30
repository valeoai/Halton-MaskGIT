#!/usr/bin/bash
# lazy r=1 token cache 补测 N=5,6 —— 三个尺寸全套 (FID / IS / 实测 latency / 理论 FLOPs)
#
#   1) 理论 FLOPs : python flops_lazy_refresh.py {large,base,small}  (REFRESH_NS=5,6)
#   2) 实测 latency: python bench_latency_refresh.py 16 1 5 {size}   —— 单卡独占 (GPU 0)
#   3) 50k FID/IS : launch/eval_fid_{size}384_lazycache.sh           —— 4 卡 DDP
#
# 配置与既有 sweep 逐项对齐: LAZY_RATIO=1.0 / lazy 覆盖全部层 / lazy head /
# seed=42 / step=32 / global_bsize=32 / fp32。cfg_w 各尺寸沿用既有设定
# (large 0.5 / base 0.7 / small 1.0), 由各自的 eval 脚本写死。
#
# 用法:
#   setsid nohup bash launch/run_lazycache_N5N6_all.sh > /tmp/lazycache_N5N6.log 2>&1 &
set -uo pipefail

export PATH="/artic/q-li/miniconda3/envs/maskgit/bin:$PATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PROGRESS="$REPO/results/.lazycache_N5N6_progress.txt"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROGRESS"; }

: > "$PROGRESS"
log "队列: FLOPs -> latency(单卡 x3) -> 50k FID (large/base/small x N=5,6, 共 6 个 eval)"

mkdir -p results/latency

# ── 1) 理论 FLOPs ────────────────────────────────────────────────────────────
log "===== 理论 FLOPs (N=5,6) ====="
REFRESH_NS=2,3,4,5,6,8,13 python flops_lazy_refresh.py large base small \
  2>&1 | tee results/latency/flops_lazycache_N5N6.txt | tee -a "$PROGRESS"

# ── 2) 实测 latency (单卡 GPU 0 独占, 与既有 bench 同方法 batch=16 warmup=1 timed=5) ──
for SIZE in large base small; do
    log "===== latency $SIZE (N=5,6) start ====="
    CUDA_VISIBLE_DEVICES=0 REFRESH_NS=5,6 \
      python bench_latency_refresh.py 16 1 5 "$SIZE" \
      2>&1 | tee "results/latency/bench_refresh_${SIZE}_N5N6.txt" | tee -a "$PROGRESS"
    log "===== latency $SIZE done rc=${PIPESTATUS[0]} ====="
done

# ── 3) 50k FID / IS (4 卡 DDP) ───────────────────────────────────────────────
run_one() {   # run_one <script> <N> <label>
    local script="$1" n="$2" label="$3"
    log "===== $label start ====="
    local start; start=$(date +%s)
    LAZY_RATIO=1.0 REFRESH_N="$n" bash "$script" 2>&1 | tee -a "$PROGRESS"
    local rc=${PIPESTATUS[0]}
    local el=$(( $(date +%s) - start ))
    log "===== $label done rc=$rc elapsed=$((el/3600))h$(( (el%3600)/60 ))m ====="
}

for SIZE in large base small; do
    for N in 5 6; do
        run_one "launch/eval_fid_${SIZE}384_lazycache.sh" "$N" "$SIZE N=$N"
    done
done

log "ALL DONE"
