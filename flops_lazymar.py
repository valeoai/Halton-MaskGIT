"""LazyMAR 在 Halton-MaskGIT 上的复现 —— 理论 FLOPs 与加速比。

LazyMAR (ICCV'25) 的两个缓存一起上, 它们省的是不同的维度:

  Token Cache     省 **token**: partial 步只重算 k_t = ceil(rho_t*N) 个 token 的
                  Q/FFN, K/V 走全长缓存; 选点 = 入口层 V 的余弦变化量 TopK。
                  (LazyMAR/models/basic.py 的 _prune_tokens;
                   本仓库 = HALTON_LAZY_CACHE + HALTON_LAZY_VSIM)
  CFG 残差 Cache  省 **batch**: lazy 步只把 [cond ; uncond] 的 cond 半个 batch 送进
                  主干, uncond = cond + 上一个整批步记下的 (uncond - cond) 残差。
                  (LazyMAR/models/mar_cache.py forward_mae_decoder 的
                   x = cat([x, x + diff]); 本仓库 = HALTON_CFG_DELTA)

本文件与 flops_lazy_vsim.py (只有 Token Cache 那一半) 的唯一结构差别是:
**CFG 的 2x 必须显式建模**。那边 baseline 和被测档都是整齐的 2x, 约掉了;
这里 lazy 步的主干是 1x 而 head 仍是 2x, 不能再约。

每步的 per-image 成本 (CFG 下一张图要跑两个前向):

    整批步 (CFG refresh)        2 * trunk(全量或 partial) + 2 * head
    lazy 步 (CFG reuse)         1 * trunk(全量或 partial) + 2 * head
                                └ 主干只跑 cond 半个 batch; last_norm + head 仍对
                                  两半各算一次 —— 缓存点在 transformer 出口,
                                  uncond 分支仍用自己的 y 过 AdaNorm 和 head
                                  (与 Network/transformer.py 的实装一致)。

门控严格照抄 Sampler/halton_sampler.py:
    token cache : GATE_LO <= t < GATE_HI, counter % N == 0 -> 全量刷新
    cfgdelta    : 同一个窗口; token 的 partial 步一律 reuse (batch 必须保持半份),
                  只有在 token 的全量步上才按 HALTON_CFG_DELTA_N 自己决定。
本文件按"两个 N 取同一个值"来跑 (实验设定), 可用 CFGD_N 单独覆盖。

用法:
    python flops_lazymar.py                 # large / base / small 都打印
    python flops_lazymar.py base
被 bench_latency_lazymar_sweep.py import 时, 先 set_size(size) 再用 run()/BASE。
"""
import os
import sys

import flops_lazy_vsim as FV

# 与 flops_lazy_vsim 共用全部常量与预算模型 —— 保证两张表可以逐行对照。
STEPS = FV.STEPS
GATE_LO, GATE_HI = FV.GATE_LO, FV.GATE_HI
SEQ, TOK, CODEBOOK = FV.SEQ, FV.TOK, FV.CODEBOOK
REFRESH_NS = FV.REFRESH_NS
CFG_START = int(os.environ.get("CFGD_START", GATE_LO))


def set_size(size, start=None, end=None):
    """切换尺寸 (转发到 flops_lazy_vsim), 并重算本文件要用的 trunk/head 拆分。"""
    global DEPTH, D, H, L_START, L_END
    global TRUNK_FULL, HEAD_FULL_ROWS, ADALN_TRUNK, ADALN_HEAD, BASE
    DEPTH, D = FV.set_size(size, start, end)
    H, L_START, L_END = FV.H, FV.L_START, FV.L_END
    # ADALN 拆成"主干里的 per-block modulation"和"last_norm 的那一支":
    # 前者随主干一起被 cfgdelta 省掉, 后者跟着 head 走。
    ADALN_TRUNK = DEPTH * FV.lin(D, 6 * D, 1)
    ADALN_HEAD = FV.lin(D, 2 * D, 1)
    TRUNK_FULL = FV.FULL_BLOCK * DEPTH + ADALN_TRUNK
    HEAD_FULL_ROWS = FV.lin(D, CODEBOOK + 1, TOK) + ADALN_HEAD
    BASE = 2 * (TRUNK_FULL + HEAD_FULL_ROWS) * STEPS      # baseline: 每步整批
    return DEPTH, D


def head_flops(rows):
    return FV.lin(D, CODEBOOK + 1, rows) + ADALN_HEAD


def trunk_partial(k):
    """一个 vsim partial 步的主干 FLOPs (不含 head), k = 本步重算的 token 数。"""
    if k >= SEQ:
        return TRUNK_FULL                              # rho_t = 1: 退化成全量
    total = FV.FULL_BLOCK * L_START                    # 层 0..start-1: 全量
    total += FV.block_flops(k, SEQ, SEQ)               # 入口层: 全长 K/V + active Q/FFN
    total += FV.block_flops(k, k, SEQ) * (L_END - L_START)
    total += FV.FULL_BLOCK * (DEPTH - 1 - L_END)       # 区间以上: 全量
    return total + ADALN_TRUNK


def gate(refresh_n, cfgd_n, cfgdelta_on=True):
    """逐 step 还原采样器的门控, 返回 [(t, token_partial, k, cfg_reuse), ...]。"""
    out = []
    counter = cfg_counter = 0
    for t in range(STEPS):
        token_partial = False
        if GATE_LO <= t < GATE_HI:
            is_refresh = refresh_n >= 1 and counter % refresh_n == 0
            counter += 1
            token_partial = not is_refresh
        k = FV.budget(t) if token_partial else SEQ
        cfg_reuse = False
        if cfgdelta_on and CFG_START <= t < GATE_HI:
            if token_partial:
                cfg_reuse = True                        # batch 必须保持半份
            else:
                is_cfg_refresh = cfgd_n >= 1 and cfg_counter % cfgd_n == 0
                cfg_counter += 1
                cfg_reuse = not is_cfg_refresh
        out.append((t, token_partial, k, cfg_reuse))
    return out


def run(refresh_n, cfgd_n=None, cfgdelta_on=True):
    """返回 (总 FLOPs, dict of 计数)。cfgd_n 默认与 refresh_n 相同 (实验设定)。"""
    if cfgd_n is None:
        cfgd_n = refresh_n
    total = 0.0
    n_partial = n_degen = n_reuse = 0
    ratios = []
    for t, token_partial, k, cfg_reuse in gate(refresh_n, cfgd_n, cfgdelta_on):
        trunk = trunk_partial(k) if token_partial else TRUNK_FULL
        rows = (k if token_partial and k < SEQ else
                (SEQ if token_partial else TOK))
        head = head_flops(rows)
        total += (1 if cfg_reuse else 2) * trunk + 2 * head
        if token_partial:
            n_partial += 1
            n_degen += (k >= SEQ)
            ratios.append(k / SEQ)
        n_reuse += cfg_reuse
    return total, dict(
        n_partial=n_partial, n_full=STEPS - n_partial, n_degen=n_degen,
        n_reuse=n_reuse, n_batch2=STEPS - n_reuse,
        mean_k=(sum(ratios) / len(ratios) if ratios else 0.0))


def report(size):
    set_size(size)
    print(f"== 理论 FLOPs | LazyMAR 复现 (Token Cache + CFG 残差 Cache) | {size}-384 | "
          f"lazy 层 L{L_START}-{L_END}/{DEPTH-1} | lazy head | 32 step ==")
    print(f"   depth={DEPTH} d={D} swiglu_h={H} seq={SEQ} | gate = [{GATE_LO}, {GATE_HI}) | "
          f"CFG 2x 已显式建模")
    print(f"   baseline = {BASE/1e9:.1f} GFLOPs/image  (= 2 x 单前向 x 32 步)\n")
    hdr = (f"{'N':>6s} {'partial步':>9s} {'其中rho=1':>9s} {'半batch步':>10s} "
           f"{'平均k/N':>8s} {'GFLOPs':>9s} {'FLOPs比':>8s} {'理论加速':>9s}")
    print(hdr)
    for n in REFRESH_NS:
        tot, c = run(n)
        label = "none" if n == 0 else str(n)
        print(f"{label:>6s} {c['n_partial']:>9d} {c['n_degen']:>9d} {c['n_reuse']:>10d} "
              f"{c['mean_k']:>8.4f} {tot/1e9:>9.1f} {tot/BASE:>8.4f} {BASE/tot:>8.3f}x")
    print()
    print("   'partial步' = Token Cache 生效 (只重算 k_t 个 token) 的 step 数;")
    print("   '其中rho=1' = LazyMAR 的比例表取 1.0 而退化成全量的 partial 步, 完全不省;")
    print("   '半batch步' = CFG 残差 Cache 生效 (主干只跑 cond 半个 batch) 的 step 数,")
    print("                 总是 >= partial 步 —— Token Cache 的全量步上它仍可复用残差。")
    print()


set_size(os.environ.get("LAZY_VIT_SIZE", "large"))

if __name__ == "__main__":
    for s in (sys.argv[1:] or ["large", "base", "small"]):
        report(s)
