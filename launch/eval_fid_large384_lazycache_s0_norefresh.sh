#!/usr/bin/bash
# Large-384 / cfg=0.5 — lazycache **从 step 0 开始 + 不 refresh**
#
# 要测的设定 (本次请求): "从第 0 step 开始使用 cache, 然后不使用 refresh"。
# 这在采样器里是 step gate 的左端 = 0, 且门内没有周期性全量刷新。
#
# ── 一个必须说清楚的坑 ──────────────────────────────────────────────────
# 直接写 START_STEP=0 + REFRESH_N=0 是**不成立**的组合:
#   partial 步的 inactive 行全部取自缓存 (per-layer K/V、restore、head logit),
#   而缓存只在**全量步**被写。gate 从 step 0 就开且从不 refresh 的话, 整条轨迹
#   上一个全量步都没有 (step 31 除外, 在最后), step 0 读到的是空缓存 ——
#   实装里退化成 kv_miss / restore_miss / head_miss, inactive 行拿到的是 0 而
#   不是"上一步的值", 且没有 refresh 去修它。已在小模型上核实:
#       START=0 N=0  -> 模式 P×31 F, kv_miss=12 restore_miss=1 head_miss=1
#       START=0 N=31 -> 模式 F P×30 F, 全部 miss = 0
# 所以本脚本用 REFRESH_N=31 表达该设定: 门内 31 个 step (0..30) 的
# gated_step_counter = 0..30, counter%31==0 只在 counter=0 命中 ->
#   step 0    全量, 唯一作用是把缓存写满 (seed);
#   step 1..30 全部 partial, 期间**一次 refresh 都没有**;
#   step 31   全量 (门外, 最后一步一律全量, 与所有既有档一致)。
# 即 30/32 步走缓存 —— 比既有最激进的档 (START=5 N=none, 26/32) 还多 4 步。
#
# MODE=literal 可以跑字面意义的 START=0 + REFRESH_N=0 那一档 (空缓存), 用来
# 量化"没有 seed"到底赔多少 —— 它不是一个可用配置, 只是对照。
#
# 其余全部与 launch/eval_fid_large384_lazycache.sh 逐项一致 (r=1 / 全部 24 层 /
# register 保持 active / halton step=32 / cfg=0.5 / fp32 / seed=42 /
# global_bsize=32 / 4 卡 DDP), 故可与既有各表直接并列。
#
# 对照 (同 sampler / 同 seed):
#   baseline (无 cache)              FID 2.5410  / IS 256.15
#   START=5 N=2   (默认档)           FID 2.6951  / IS 252.31   理论 1.618x
#   START=5 N=none(26/32 partial)    FID 10.8181 / IS 153.50   理论 4.268x
#   START=0 N=2   (15/32 partial)    FID 3.9676  / IS 228.43
#
# 理论 FLOPs (python flops_lazy_refresh.py, HALTON_PARTIAL_START_STEP=0):
#   seed@0 + no refresh : 1495.5 GFLOPs/img  0.1145x  -> 8.733x
#   字面 N=0 (空缓存)   : 1096.2 GFLOPs/img  0.0839x  -> 11.914x
#   baseline            : 13059.8 GFLOPs/img
#
# 用法:
#   bash launch/eval_fid_large384_lazycache_s0_norefresh.sh              # seed@0
#   MODE=literal bash launch/eval_fid_large384_lazycache_s0_norefresh.sh # 空缓存对照
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODE="${MODE:-seeded}"
START_STEP=0
case "$MODE" in
  seeded)  REFRESH_N=31 ; TAG="lazycache_r1.0_S0_norefresh"         ;;  # step0 全量 seed
  literal) REFRESH_N=0  ; TAG="lazycache_r1.0_S0_norefresh_noseed"  ;;  # 空缓存对照
  *) echo "MODE 只能是 seeded / literal" >&2; exit 1 ;;
esac
LAZY_RATIO="${LAZY_RATIO:-1.0}"
LAZY_START="${LAZY_START:-0}"                    # 层区间: 全部 24 层
LAZY_END="${LAZY_END:-23}"
CFG_W="${CFG_W:-0.5}"

export HALTON_PARTIAL_UPDATE=1
export HALTON_LAZY_CACHE=1                       # LazyMAR Token Cache
export HALTON_LAZY_CACHE_RATIO="$LAZY_RATIO"
export HALTON_LAZY_START_LAYER="$LAZY_START"
export HALTON_LAZY_END_LAYER="$LAZY_END"
export HALTON_CACHE_REFRESH_N="$REFRESH_N"
export HALTON_PARTIAL_START_STEP="$START_STEP"
unset HALTON_LAZY_REGISTER_ACTIVE 2>/dev/null || true   # register 保持默认 (active)
unset HALTON_PARTIAL_START_LAYER HALTON_PARTIAL_END_LAYER 2>/dev/null || true
unset HALTON_LAYER_CACHE HALTON_ATTN_CACHE 2>/dev/null || true
unset HALTON_LAZY_VSIM HALTON_LAZY_VSIM_SCHED 2>/dev/null || true
unset HALTON_CFG_DELTA HALTON_CFG_DELTA_N HALTON_CFG_DELTA_START 2>/dev/null || true

LOG="/tmp/eval_fid_large384_${TAG}.log"
echo "[$(date '+%F %T')] start large384 ${TAG} (MODE=$MODE, REFRESH_N=$REFRESH_N) -> $LOG"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  main.py \
  --mode cls-to-img --test-only --resume \
  --data-folder "" \
  --eval-folder "/work/Shared/Datasets/ILSVRC/ILSVRC2012/" \
  --vit-folder "./saved_networks/ImageNet_384_large.pth" \
  --vqgan-folder "./saved_networks/vq_ds16_c2i.pt" \
  --writer-log "./logs/" \
  --data imagenet --dtype float32 \
  --vit-size large --img-size 384 \
  --f-factor 16 --codebook-size 16384 --mask-value 16384 \
  --register 1 --proj 1 --dropout 0.1 \
  --nb-class 1000 --num-workers 8 --global-bsize 32 \
  --sampler halton --step 32 --cfg-w "$CFG_W" \
  --sm-temp 1.0 --sm-temp-min 1.0 --temp-warmup 1 \
  --sched-pow 2 --top-k -1 \
  --seed 42 > "$LOG" 2>&1

FID=$(grep -oE "'FID': [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
IS=$(grep -oE "'IS': [0-9.]+"   "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
RT=$(grep -oE "[0-9]+:[0-9]+(:[0-9]+)?<00:00"  "$LOG" | tail -1 | cut -d'<' -f1 || echo "NA")

if [ "$MODE" = "seeded" ]; then
  NOTE="gate=0..30 no-refresh (step0 seed) partial=30/32 ratio=${LAZY_RATIO} layers=${LAZY_START}..${LAZY_END} cfg=${CFG_W}"
else
  NOTE="gate=0..30 no-refresh NO-SEED (空缓存) partial=31/32 ratio=${LAZY_RATIO} layers=${LAZY_START}..${LAZY_END} cfg=${CFG_W}"
fi

mkdir -p results
printf "%-46s %-8s %-9s %-9s %s\n" "$TAG" "$FID" "$IS" "$RT" "$NOTE" \
  >> results/halton_large384_lazycache_s0_norefresh.txt

echo "[$(date '+%F %T')] done  large384 ${TAG}  FID=$FID  IS=$IS  rt=$RT"
