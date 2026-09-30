"""lazy r=1 / 全部层 / lazy head 在不同 refresh 周期 N 下的理论 FLOPs 与加速比。

FLOPs 模型与 flops_lazy_token_cache.py 完全一致 (FLOPs = 2 x MACs), 唯一新增的是
refresh 步的处理 —— 语义严格照抄 Sampler/halton_sampler.py:225-231:

    is_refresh = (cache_refresh_n >= 1 and gated_step_counter % cache_refresh_n == 0)
    gated_step_counter += 1          # <<< 先判断后自增

即 gated_step_counter 从 0 开始, refresh 落在 counter = 0, N, 2N, ...
N=2 -> 13/26 步刷新, N=4 -> 7/26, 与既有 layercache sweep 的 notes 一致。
refresh 步走 active_mask=None 的全量分支, 与 baseline 单步同价。

模型尺寸取自 Trainer/abstract_trainer.py:transformer_size:
    small = 512 dim / 12 层 / 6 head   (77.9M)
    base  = 768 dim / 12 层 / 12 head  (155.1M)
    large = 1024 dim / 24 层 / 16 head (479.6M)
lazy 覆盖全部层, 故 depth 只影响总量而不影响 partial/full 的比例结构;
不同尺寸之间的理论加速差异只来自 head (D x 16385) 与 attention 在总量中的占比。

用法:
    python flops_lazy_refresh.py              # 三个尺寸都打印
    python flops_lazy_refresh.py base         # 只打印 base
被 bench_latency_refresh.py import 时, 先调用 set_size(size) 再用 run()/BASE。
"""
import math
import os
import sys

# ── 与尺寸无关的常量 ──────────────────────────────────────────────────────
GRID     = 384 // 16
TOK      = GRID * GRID                # 576
REGISTER = 1
SEQ      = TOK + REGISTER             # 577
CODEBOOK = 16384
STEPS    = 32
# partial 步: GATE_LO <= t < GATE_HI。
# 左端跟采样器一样可由 HALTON_PARTIAL_START_STEP 覆盖 (默认 5), 这样本文件和
# Sampler/halton_sampler.py 读的是同一个 env, 不会两边对不上。右端 31 固定。
GATE_LO = max(0, int(os.environ.get("HALTON_PARTIAL_START_STEP", "5")))
GATE_HI = 31

SIZES = {                             # name -> (depth, hidden_dim)
    "tiny":  (6, 384),
    "small": (12, 512),
    "base":  (12, 768),
    "large": (24, 1024),
    "xlarge": (28, 1152),
}

REFRESH_NS = [0, 2, 3, 4, 8, 13]      # 0 = no refresh
if os.environ.get("REFRESH_NS"):      # 逗号分隔, 用来只跑某几档
    REFRESH_NS = [int(v) for v in os.environ["REFRESH_NS"].split(",") if v.strip()]


def swiglu_hidden(mlp_dim, multiple_of=256):
    h = int(2 * mlp_dim / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


def lin(n_in, n_out, tokens):
    return 2 * tokens * n_in * n_out


def block_flops(q_tok, kv_tok, ctx_len):
    attn = (lin(D, D, q_tok)                  # wq
            + lin(D, D, kv_tok) * 2           # wk, wv
            + lin(D, D, q_tok)                # wo
            + 2 * (2 * q_tok * ctx_len * D))  # QK^T + attn@V
    ffn = 2 * lin(D, H, q_tok) + lin(H, D, q_tok)
    return attn + ffn


def set_size(size):
    """把模块级 globals 切到指定尺寸。返回 (depth, dim)。"""
    global SIZE, DEPTH, D, MLP_DIM, H, HEAD_FULL, ADALN, FULL_BLOCK, FULL_STEP, BASE
    if size not in SIZES:
        raise ValueError(f"unknown vit size {size!r}, expect one of {list(SIZES)}")
    SIZE = size
    DEPTH, D = SIZES[size]
    MLP_DIM = 4 * D
    H = swiglu_hidden(MLP_DIM)
    HEAD_FULL = lin(D, CODEBOOK + 1, TOK)
    ADALN = DEPTH * lin(D, 6 * D, 1) + lin(D, 2 * D, 1)
    FULL_BLOCK = block_flops(SEQ, SEQ, SEQ)
    FULL_STEP = FULL_BLOCK * DEPTH + HEAD_FULL + ADALN
    BASE = FULL_STEP * STEPS
    return DEPTH, D


def halton_r(t):
    r = 1 - (math.acos(min(1.0, (t + 1) / STEPS)) / (math.pi * 0.5))
    return max(t + 1, int(r * TOK))


R = [halton_r(t) for t in range(STEPS)]


def active_count(t):
    """|U_t ∪ U_{t-1}| + register."""
    r_t  = R[t]
    r_p1 = R[t - 1] if t >= 1 else 0
    r_p2 = R[t - 2] if t >= 2 else 0
    return (r_t - r_p1) + (r_p1 - r_p2) + REGISTER


def seeded(refresh_n, gate_lo=None):
    """门内的第一个步是不是全量步 —— 即缓存到底有没有被写满过一次。

    partial 步的 inactive 行全部取自缓存 (K/V / restore / head logit)。缓存只在
    全量步被写。gate_lo=0 且 refresh_n<=0 时, step 0 就是 partial 步而此前没有
    任何全量步 —— 那一步读到的是空缓存, 实装里退化成 kv_miss / restore_miss /
    head_miss, inactive 行拿到的是 0 而不是"上一步的值"。这不是"缓存变陈旧",
    是缓存里根本没有东西, 而且没有 refresh 去修它。
    此时下面的 FLOPs 数学上仍成立, 但它对应的是一次数值上无意义的采样。
    """
    lo = GATE_LO if gate_lo is None else gate_lo
    return lo > 0 or refresh_n >= 1


def run(refresh_n):
    """返回 (总 FLOPs, refresh 步数, partial 步数, partial 平均 k/N)。"""
    total = 0.0
    counter = 0
    n_refresh = n_partial = 0
    ratios = []
    for t in range(STEPS):
        if GATE_LO <= t < GATE_HI:
            is_refresh = refresh_n >= 1 and counter % refresh_n == 0
            counter += 1
            if is_refresh:
                total += FULL_STEP
                n_refresh += 1
                continue
            k = active_count(t)
            total += (block_flops(k, k, SEQ) * DEPTH
                      + lin(D, CODEBOOK + 1, k)      # lazy head
                      + ADALN)
            n_partial += 1
            ratios.append(k / SEQ)
        else:
            total += FULL_STEP
    return total, n_refresh, n_partial, (sum(ratios) / len(ratios) if ratios else 0.0)


def report(size):
    set_size(size)
    print(f"== 理论 FLOPs | lazy r=1 | {size}-384 | {DEPTH}/{DEPTH} 层 | lazy head | 32 step ==")
    print(f"   depth={DEPTH} d={D} swiglu_h={H} seq={SEQ} | gate = [{GATE_LO}, {GATE_HI})")
    print(f"   baseline = {BASE/1e9:.1f} GFLOPs/image\n")
    print(f"{'refresh N':>10s} {'refresh步':>9s} {'partial步':>9s} {'全量步合计':>11s} "
          f"{'平均k/N':>8s} {'GFLOPs':>9s} {'FLOPs比':>8s} {'理论加速':>9s}")
    for n in REFRESH_NS:
        tot, nr, npart, kr = run(n)
        n_full = STEPS - (GATE_HI - GATE_LO) + nr
        label = "none" if n == 0 else str(n)
        flag = "" if seeded(n) else "   <- 缓存从未被写满 (空缓存, 数值无意义)"
        print(f"{label:>10s} {nr:>9d} {npart:>9d} {n_full:>11d} "
              f"{kr:>8.4f} {tot/1e9:>9.1f} {tot/BASE:>8.4f} {BASE/tot:>8.3f}x{flag}")
    print()


# import 时的默认尺寸 (向后兼容: 老脚本 import 后直接用 run()/BASE 得到 large)
set_size(os.environ.get("LAZY_VIT_SIZE", "large"))

if __name__ == "__main__":
    targets = sys.argv[1:] or ["large", "base", "small"]
    for s in targets:
        report(s)
