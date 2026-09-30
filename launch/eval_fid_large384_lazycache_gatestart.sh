#!/usr/bin/bash
# Large-384 / cfg=0.5 / N=2 — lazycache 的 **step gate 左端消融**
#
# 被消融的东西 (Sampler/halton_sampler.py 的 partial-update 门):
#       partial_start_step <= index < 31
#   左端原本写死成 5, 出处是 analyze_ffn_delta_stability —— "前几步整幅图还在
#   剧烈变化, 缓存里的东西陈旧得没法用"。这个 5 从来没有被 50k FID 扫过。
#   本实验把左端扫一遍: START_STEP = 0, 1, 2, 3, 4 (5 = 既有默认档, 已有结果)。
#   右端 31 不动 —— 最后一步必须全量。
#
# 注意 gate 左端同时改变的是**两件事**:
#   (1) 多了几个受门管辖的 step;
#   (2) refresh 的相位 —— gated_step_counter 从门一开就计数, counter%N==0 是全量
#       步。所以 START=0 时 step0 全量 / step1 partial / step2 全量 ...;
#       START=5 时 step5 全量 / step6 partial ...。
#   N=2 下各档的 partial 步数 (32 步): START=0:15  1:15  2:14  3:14  4:13  5:13
#   —— 由 test_step_gate_start.py 的 G2/G3 逐步核对过。
#
# 其余全部与 launch/eval_fid_large384_lazycache.sh 逐项一致 (r=1 / 全部 24 层 /
# register 保持 active / halton step=32 / cfg=0.5 / fp32 / seed=42 /
# global_bsize=32 / 4 卡 DDP), 所以 START=5 这一档就是既有的
# results/halton_large384_lazycache_sweep.txt 里的 lazycache_r1.0_N2_L0-23
# (FID 2.6951 / IS 252.3132), 不必重跑。
#
# 正确性: python test_step_gate_start.py (7/7) —— 含"不设 env 时与旧代码逐位
#   一致"、"start=0..5 的 partial 步集合与门的定义逐步吻合"、"门关死时逐位等于
#   baseline"、"各档 kv/restore/head miss 全 0"。
#   既有方案未受影响: test_lazy_token_cache.py (16/16) / test_lazy_vsim.py (9/9)
#   / test_cfg_delta.py (10/10) / test_lazy_register_ablation.py (6/6)。
#
# 用法:
#   START_STEP=0 bash launch/eval_fid_large384_lazycache_gatestart.sh
# 参数:
#   START_STEP  partial-update 门的左端 (默认 5 = 原行为)
#   REFRESH_N   每 N 个 gated step 全量刷新一次 (默认 2)
#   LAZY_RATIO / LAZY_START / LAZY_END  与 eval_fid_large384_lazycache.sh 同义
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

START_STEP="${START_STEP:-5}"                    # <<< 本实验的消融维度
LAZY_RATIO="${LAZY_RATIO:-1.0}"
REFRESH_N="${REFRESH_N:-2}"
LAZY_START="${LAZY_START:-0}"                    # 层区间: 全部 24 层
LAZY_END="${LAZY_END:-23}"
CFG_W="${CFG_W:-0.5}"

export HALTON_PARTIAL_UPDATE=1
export HALTON_LAZY_CACHE=1                       # LazyMAR Token Cache
export HALTON_LAZY_CACHE_RATIO="$LAZY_RATIO"
export HALTON_LAZY_START_LAYER="$LAZY_START"
export HALTON_LAZY_END_LAYER="$LAZY_END"
export HALTON_CACHE_REFRESH_N="$REFRESH_N"
export HALTON_PARTIAL_START_STEP="$START_STEP"   # <<< step gate 的左端
unset HALTON_LAZY_REGISTER_ACTIVE 2>/dev/null || true   # register 保持默认 (active)
unset HALTON_PARTIAL_START_LAYER HALTON_PARTIAL_END_LAYER 2>/dev/null || true
unset HALTON_LAYER_CACHE HALTON_ATTN_CACHE 2>/dev/null || true
unset HALTON_LAZY_VSIM HALTON_LAZY_VSIM_SCHED 2>/dev/null || true
unset HALTON_CFG_DELTA HALTON_CFG_DELTA_N HALTON_CFG_DELTA_START 2>/dev/null || true

TAG="lazycache_r${LAZY_RATIO}_N${REFRESH_N}_L${LAZY_START}-${LAZY_END}_S${START_STEP}"
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
  "gate=${START_STEP}..30 ratio=${LAZY_RATIO} refresh_every_${REFRESH_N} layers=${LAZY_START}..${LAZY_END} cfg=${CFG_W}" \
  >> results/halton_large384_lazycache_gatestart_ablation.txt

echo "[$(date '+%F %T')] done  large384 ${TAG}  FID=$FID  IS=$IS  rt=$RT"
