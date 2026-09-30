#!/usr/bin/bash
# Large-384 / cfg=0.5 / N=2 — lazycache 的 **register token 消融**
#
# 被消融的东西 (Network/transformer.py:1108-1114):
#   lazycache (HALTON_LAZY_CACHE=1, r=1) 的默认 active 集合是
#       active = U_t ∪ U_{t-1} ∪ register
#   register 被无条件强制重算, 理由写在那几行注释里 —— "register 是全局
#   attention sink, 冻结它的代价远大于重算 (只有 1 个 token)"。
#   本实验把这一条去掉:
#       active = U_t ∪ U_{t-1}                    (HALTON_LAZY_REGISTER_ACTIVE=0)
#   于是 partial 步里 register 行完全走缓存: 残差流取 lazy_restore_cache 里上一步
#   的值, 每层 K/V 取 per-layer 缓存里上一步的值, head 也不再为它算 logit
#   (它本来就不产 code)。register 仍然是全部 577 个 token 的 attention 上下文的
#   一部分 —— 变的只是"它自己的表示每步更不更新"。
#
# 省的算力可以忽略 (577 -> 576 个 token 里少算 1 个, < 0.2%), 所以这一档要看的
# 不是速度而是 **质量**: 那句"冻结代价远大于重算"到底成不成立。
#
# 对照 (同脚本同参数, 唯一差别就是这个开关):
#   results/halton_large384_lazycache_sweep.txt
#     lazycache_r1.0_N2_L0-23    FID 2.6951 / IS 252.3132 / 2:41:29
#   baseline (无 cache)          FID 2.5410 / IS 256.15
#
# 正确性: python test_lazy_register_ablation.py (6/6) —— 含"active 集合恰好
#   等于 U_t ∪ U_{t-1} 且 register 一列都不在里面"、"默认设定下 register 必被
#   强制 active (反证)"、"全量步逐位仍等于 baseline"、"logits 确实变了 (消融
#   非空)"、"head 的 active 下标不越界且 head_miss=0"。
#   既有方案未受影响: test_lazy_token_cache.py (16/16)。
#
# 用法:
#   REFRESH_N=2 bash launch/eval_fid_large384_lazycache_noreg.sh
# 参数:
#   REG_ACTIVE  1 = 保留 register (等于原 lazycache); 0 = 消融掉 (默认)
#   REFRESH_N   每 N 个 gated step 全量刷新一次 (默认 2)
#   LAZY_RATIO / LAZY_START / LAZY_END  与 eval_fid_large384_lazycache.sh 同义
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

REG_ACTIVE="${REG_ACTIVE:-0}"                    # <<< 本实验的消融开关
LAZY_RATIO="${LAZY_RATIO:-1.0}"
REFRESH_N="${REFRESH_N:-2}"
LAZY_START="${LAZY_START:-0}"                    # 默认全部 24 层都走 lazy
LAZY_END="${LAZY_END:-23}"
CFG_W="${CFG_W:-0.5}"

export HALTON_PARTIAL_UPDATE=1
export HALTON_LAZY_CACHE=1                       # LazyMAR Token Cache
export HALTON_LAZY_CACHE_RATIO="$LAZY_RATIO"
export HALTON_LAZY_REGISTER_ACTIVE="$REG_ACTIVE" # <<< 0 = 每步的 token 集合里去掉 register
export HALTON_LAZY_START_LAYER="$LAZY_START"
export HALTON_LAZY_END_LAYER="$LAZY_END"
export HALTON_CACHE_REFRESH_N="$REFRESH_N"
unset HALTON_PARTIAL_START_LAYER HALTON_PARTIAL_END_LAYER 2>/dev/null || true
unset HALTON_LAYER_CACHE HALTON_ATTN_CACHE 2>/dev/null || true
unset HALTON_LAZY_VSIM HALTON_LAZY_VSIM_SCHED 2>/dev/null || true
unset HALTON_CFG_DELTA HALTON_CFG_DELTA_N HALTON_CFG_DELTA_START 2>/dev/null || true

SUFFIX=$([ "$REG_ACTIVE" = "0" ] && echo "_noreg" || echo "_reg")
TAG="lazycache_r${LAZY_RATIO}_N${REFRESH_N}_L${LAZY_START}-${LAZY_END}${SUFFIX}"
LOG="/tmp/eval_fid_large384_${TAG}.log"
echo "[$(date '+%F %T')] start large384 ${TAG} -> $LOG"

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

mkdir -p results
printf "%-46s %-8s %-9s %-9s %s\n" \
  "$TAG" "$FID" "$IS" "$RT" \
  "register_active=${REG_ACTIVE} ratio=${LAZY_RATIO} refresh_every_${REFRESH_N} layers=${LAZY_START}..${LAZY_END} cfg=${CFG_W}" \
  >> results/halton_large384_lazycache_register_ablation.txt

echo "[$(date '+%F %T')] done  large384 ${TAG}  FID=$FID  IS=$IS  rt=$RT"
