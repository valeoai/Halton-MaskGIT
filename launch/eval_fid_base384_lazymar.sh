#!/usr/bin/bash
# Base-384 / cfg=0.7 — LazyMAR 复现 (Token Cache + CFG 残差 Cache)
#
# 两个方案正交, 省的是不同的东西:
#   lazyvsim (HALTON_LAZY_VSIM)  省 **token**: partial 步只重算 k_t=ceil(rho_t*N) 个
#                                token 的 Q/FFN, K/V 走全长 per-layer 缓存。
#                                选点 = TopK(1 - cos(V_l3(t), V_l3(t-1))), 无强制集合。
#   cfgdelta (HALTON_CFG_DELTA)  省 **batch**: lazy 步只把 [cond ; uncond] 的 cond 半个
#                                batch 送进主干, uncond = cond + 上一个整批步记下的
#                                (uncond - cond) 残差 —— LazyMAR forward_mae_decoder
#                                的 `x = cat([x, x + diff])`。残差缓存点在 transformer
#                                出口 / last_norm 之前, 所以 uncond 分支仍然用自己的 y
#                                过 AdaNorm 和 head, 被跳过的只有主干的 12 层。
#
# **两个方案的 N 取同一个值** (本次实验设定): HALTON_CACHE_REFRESH_N = HALTON_CFG_DELTA_N。
#   token cache 的 partial 步一律 CFG-reuse (batch 必须保持半份);
#   只有在 token cache 的全量步上, cfgdelta 才按自己的 N 决定刷不刷新残差。
#
# 采样调度 (Halton 顺序、step gate 5..30、CFG、温度、seed、global_bsize) 与
# launch/eval_fid_base384_lazyvsim.sh 逐项一致 —— 与只开 vsim 的那一档严格可比。
#
# 正确性: python test_cfg_delta.py (10/10) —— 含"输入不变时残差重构精确到浮点噪声
#         floor"、"N=1 逐位等于 baseline"、"与 lazyvsim 同开时逐步预算完全不变、
#         所有缓存 miss 为 0"。既有方案未受影响: test_lazy_token_cache.py (16/16),
#         test_lazy_vsim.py (9/9)。
#
# 用法:
#   REFRESH_N=2 bash launch/eval_fid_base384_lazymar.sh
# 参数:
#   REFRESH_N   每 N 个 gated step 全量刷新一次 (0 = 不刷新); 同时用作 CFG_DELTA_N
#   CFGD_N      单独覆盖 cfgdelta 的 N (默认 = REFRESH_N)
#   LAZY_START / LAZY_END   lazy 层区间, 默认 3..11
#   VSIM_SCHED  重算比例表: "lazymar" (默认) / 常数 "0.3" / 逐步 "1,1,0.5,..."
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

REFRESH_N="${REFRESH_N:-2}"
CFGD_N="${CFGD_N:-$REFRESH_N}"                    # 默认与 token cache 的 N 一致
LAZY_START="${LAZY_START:-3}"                     # LazyMAR 的 decoder layer 3
LAZY_END="${LAZY_END:-11}"                       # 一直用到最后一层
VSIM_SCHED="${VSIM_SCHED:-lazymar}"
CFG_W="${CFG_W:-0.7}"

export HALTON_PARTIAL_UPDATE=1
export HALTON_LAZY_CACHE=1                       # 执行机制沿用 LazyMAR Token Cache
export HALTON_LAZY_VSIM=1                        # 选点: 纯 V 相似度, 无强制集合
export HALTON_LAZY_VSIM_SCHED="$VSIM_SCHED"
export HALTON_LAZY_START_LAYER="$LAZY_START"
export HALTON_LAZY_END_LAYER="$LAZY_END"
export HALTON_CACHE_REFRESH_N="$REFRESH_N"
export HALTON_CFG_DELTA=1                        # <<< CFG 残差缓存
export HALTON_CFG_DELTA_N="$CFGD_N"
export HALTON_CFG_DELTA_START=5                  # 与 token gate 的起点一致
unset HALTON_LAZY_CACHE_RATIO 2>/dev/null || true
unset HALTON_PARTIAL_START_LAYER HALTON_PARTIAL_END_LAYER 2>/dev/null || true
unset HALTON_LAYER_CACHE HALTON_ATTN_CACHE 2>/dev/null || true

TAG="lazymar_${VSIM_SCHED}_N${REFRESH_N}_L${LAZY_START}-${LAZY_END}_cfg${CFG_W}"
LOG="/tmp/eval_fid_base384_${TAG}.log"
echo "[$(date '+%F %T')] start base384 ${TAG} -> $LOG"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  main.py \
  --mode cls-to-img --test-only --resume \
  --data-folder "" \
  --eval-folder "/work/Shared/Datasets/ILSVRC/ILSVRC2012/" \
  --vit-folder "./saved_networks/ImageNet_384_base.pth" \
  --vqgan-folder "./saved_networks/vq_ds16_c2i.pt" \
  --writer-log "./logs/" \
  --data imagenet --dtype float32 \
  --vit-size base --img-size 384 \
  --f-factor 16 --codebook-size 16384 --mask-value 16384 \
  --register 1 --proj 1 --dropout 0.1 \
  --nb-class 1000 --num-workers 8 --global-bsize 32 \
  --sampler halton --step 32 --cfg-w "$CFG_W" \
  --sm-temp 1.0 --sm-temp-min 1.0 --temp-warmup 1 \
  --sched-pow 2 --top-k -1 \
  --seed 42 > "$LOG" 2>&1

FID=$(grep -oE "'FID': [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
IS=$(grep -oE "'IS': [0-9.]+"   "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
# tqdm 不到 1 小时只打 MM:SS, 超过才打 H:MM:SS —— 两种都要认。
RT=$(grep -oE "[0-9]+:[0-9]+(:[0-9]+)?<00:00"  "$LOG" | tail -1 | cut -d'<' -f1 || echo "NA")

mkdir -p results
printf "%-52s %-8s %-9s %-9s %s\n" \
  "$TAG" "$FID" "$IS" "$RT" \
  "sched=${VSIM_SCHED} refresh_every_${REFRESH_N} cfgd_N=${CFGD_N} layers=${LAZY_START}..${LAZY_END} cfg=${CFG_W} no-forced-token +cfg-residual-cache" \
  >> results/halton_base384_lazymar_sweep.txt

echo "[$(date '+%F %T')] done  base384 ${TAG}  FID=$FID  IS=$IS  rt=$RT"
