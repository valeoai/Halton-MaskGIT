"""LazyMAR CFG 残差缓存 (HALTON_CFG_DELTA) 的正确性测试。

方案: lazy 步只把 [cond ; uncond] 的 **cond 半个 batch** 送进 transformer,
uncond 半由 "cond + 上一个整批步记下的 (uncond - cond) 残差" 补出来 —— 即
LazyMAR forward_mae_decoder 里 `x = cat([x, x + diff])` 那一手。缓存点取在
transformer 出口 / last_norm 之前, 所以 uncond 分支仍然用它自己的 y 过 AdaNorm
和 head, 被跳过的只有主干的 depth 层。

不需要预训练权重: 随机初始化的 Transformer + 真正的 HaltonSampler 跑完整 decoding。

断言:
  D1  重构是精确的: 输入不变时, "cond + 残差" 与真跑一遍 uncond 数值一致
  D2  N=1 (每个门内整批步都刷新) → 与 baseline 逐位一致 (方案等价于关闭)
  D3  端到端跑得通: 有 reuse 步, 没有 miss, lazy 步 transformer 只见半个 batch
  D4  近似误差可量化: 单步受控实验下, uncond 的相对误差远小于 CFG 项本身
  D5  跨 generation 缓存清理生效 (两次 generation 逐位一致)
  D6  接口保护: w=0 / 与旧 delta 缓存方案组合时明确报错
  D7  与 HALTON_LAZY_CACHE 同开: 所有跨 step 缓存零 miss, 且 batch 维正确收缩
  D8  与 HALTON_LAZY_VSIM 同开: 零 miss; 逐步预算与只开 vsim 时完全相同
  D9  与 vsim 同开时 lazy 步的选点在半 batch 上做, 且不再做 cfg_pair 分数平均
  D10 只开 CFG 残差缓存 (不开 token cache) 也能独立工作

用法:  python test_cfg_delta.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Network.transformer as T
from Network.transformer import Transformer
from test_lazy_token_cache import (env, make_trainer, run, check, RESULTS,
                                   assert_bitwise, DEVICE)

# 三种组合, 后面反复用
CFG_ONLY = dict(HALTON_CFG_DELTA=1)
CFG_LAZY = dict(HALTON_CFG_DELTA=1, HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1)
CFG_VSIM = dict(HALTON_CFG_DELTA=1, HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1,
                HALTON_LAZY_VSIM=1, HALTON_LAZY_START_LAYER=1, HALTON_LAZY_END_LAYER=5)
VSIM_ONLY = dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_VSIM=1,
                 HALTON_LAZY_START_LAYER=1, HALTON_LAZY_END_LAYER=5)


def spy_encoder_batches():
    """记录每次 TransformerEncoder.forward 见到的 (batch, cfg_pair)。"""
    seen = []
    orig = T.TransformerEncoder.forward

    def spy(self, x, cond, *a, **kw):
        seen.append((int(x.size(0)), bool(kw.get("cfg_pair", False))))
        return orig(self, x, cond, *a, **kw)

    return seen, orig, spy


def stats_of(trainer):
    return trainer.vit.get_lazy_stats()


# ---------------------------------------------------------------------------
def d1_reconstruction_is_exact_when_input_unchanged():
    """同一份输入连跑两次: 第二次用残差补出来的 uncond 必须等于第一次真算的。

    这是本方案唯一"没有近似"的情形, 也因此是实装是否正确的判据: 残差写入 /
    读取 / 半 batch 切分 / AdaNorm 与 head 的 y 对齐, 任何一处错位都会在这里暴露。

    对照物 (floor): 同一份 cond 输入分别按 batch=2B 和 batch=B 跑一遍的差。
    matmul kernel 会随 batch 换 tile / 换归约顺序, 所以"半 batch 跑"这件事本身
    就带 ~1e-7 的噪声, 与本方案无关。判据因此是: reuse 的误差 <= 这个 floor 的
    几倍, 即残差重构没有引入 floor 之外的任何误差。
    """
    trainer = make_trainer(seed=11, input_size=8, register=1, depth=6)
    b = 4
    torch.manual_seed(0)
    code = torch.randint(0, 64, (2 * b, 8, 8), device=DEVICE)
    lab = torch.arange(2 * b, device=DEVICE) % 10
    drop = torch.cat([torch.zeros(b, dtype=torch.bool), torch.ones(b, dtype=torch.bool)]).to(DEVICE)

    with env(**CFG_ONLY), torch.no_grad():
        trainer.vit.clear_ffn_cache()
        full = trainer.vit(code, lab, drop, cfg_pair=True, cfg_delta=False)
        reused = trainer.vit(code, lab, drop, cfg_pair=True, cfg_delta=True)
        st = stats_of(trainer)
    with env(), torch.no_grad():                     # floor: 纯 batch 大小的影响
        trainer.vit.clear_ffn_cache()
        half_only = trainer.vit(code[:b], lab[:b], drop[:b], cfg_pair=False)
    floor = (full[:b] - half_only).abs().max().item()

    assert st["cfg_delta_full_calls"] == 1 and st["cfg_delta_reuse_calls"] == 1, st
    assert st["cfg_delta_miss"] == 0, st
    dc = (full[:b] - reused[:b]).abs().max().item()
    du = (full[b:] - reused[b:]).abs().max().item()
    scale = full.abs().max().item()
    tol = max(4 * floor, 1e-6 * scale)
    assert dc <= tol, f"cond 半边偏离 floor: max|Δ|={dc:.3e} > {tol:.3e}"
    assert du <= tol, f"uncond 重构偏离 floor: max|Δ|={du:.3e} > {tol:.3e}"
    return (f"cond Δ={dc:.2e} / uncond Δ={du:.2e}, 半 batch 噪声 floor={floor:.2e} "
            f"(logit scale {scale:.2e})")


def d2_refresh_every_step_equals_baseline():
    """N=1 → 门内每个整批步都刷新 → 从不 reuse → 必须与 baseline 逐位一致。"""
    trainer = make_trainer(seed=12, input_size=8, register=1, depth=6)
    with env():
        base = run(trainer, nb_sample=4, steps=12, cfg_w=1.5, seed=7)
    with env(HALTON_CFG_DELTA=1, HALTON_CFG_DELTA_N=1):
        got = run(trainer, nb_sample=4, steps=12, cfg_w=1.5, seed=7)
        st = stats_of(trainer)
    assert st["cfg_delta_reuse_calls"] == 0, st
    assert st["cfg_delta_full_calls"] > 0, st
    assert_bitwise(base[2], got[2], "logits(N=1)")
    assert_bitwise(base[1], got[1], "codes(N=1)")
    return f"{st['cfg_delta_full_calls']} 个整批步, 0 reuse, 逐位一致"


def d3_end_to_end_half_batch():
    """默认门控下: 有 reuse 步; reuse 步 transformer 只见半个 batch 且 cfg_pair=False。"""
    trainer = make_trainer(seed=13, input_size=8, register=1, depth=6)
    seen, orig, spy = spy_encoder_batches()
    T.TransformerEncoder.forward = spy
    try:
        with env(**CFG_ONLY):
            out = run(trainer, nb_sample=4, steps=12, cfg_w=1.5, seed=7)
            st = stats_of(trainer)
    finally:
        T.TransformerEncoder.forward = orig

    full = [s for s in seen if s[0] == 8]
    halfb = [s for s in seen if s[0] == 4]
    assert len(seen) == 12, f"{len(seen)} 次 forward, 期望 12"
    assert len(halfb) == st["cfg_delta_reuse_calls"] > 0, (len(halfb), st)
    assert len(full) == st["cfg_delta_full_calls"], (len(full), st)
    assert st["cfg_delta_miss"] == 0, st
    assert all(cp for _, cp in full), "整批步必须 cfg_pair=True"
    assert not any(cp for _, cp in halfb), "半 batch 步必须 cfg_pair=False"
    assert torch.isfinite(out[0]).all(), "输出出现 NaN/Inf"
    return f"{len(full)} 整批 + {len(halfb)} 半 batch, miss=0"


def d4_error_grows_with_input_drift():
    """受控单步实验: 误差必须随"输入相对存残差那一刻漂移了多少"单调增长。

    做法: 用 code_0 跑一个整批步存下残差, 再改掉 m 个 token (模拟往前走了几步),
    用 reuse 和真算各跑一次, 比较最终 CFG 合成 logit。m=0 时必须落回 D1 的 floor,
    m 增大时误差单调上升 —— 这正是"残差在多少步内还能用"这条曲线的单点版本。

    注意: 这里是**随机初始化**的权重, 数值本身不代表训练好的模型上的近似质量;
    要那条曲线得拿 checkpoint 跑 bench (见 bench_latency_cfgdelta.py)。
    """
    trainer = make_trainer(seed=14, input_size=8, register=1, depth=6)
    b, w = 4, 3.0
    lab = torch.arange(2 * b, device=DEVICE) % 10
    drop = torch.cat([torch.zeros(b, dtype=torch.bool), torch.ones(b, dtype=torch.bool)]).to(DEVICE)
    torch.manual_seed(1)
    code0 = torch.randint(0, 64, (2 * b, 8, 8), device=DEVICE)
    flip = torch.randint(0, 64, (2 * b, 8, 8), device=DEVICE)
    order = torch.randperm(64, device=DEVICE)

    def cfg_logit(lg):
        return (1 + w) * lg[:b] - w * lg[b:]

    errs = []
    for m in (0, 4, 16, 48):
        code1 = code0.clone()
        if m:
            sel = torch.zeros(64, dtype=torch.bool, device=DEVICE)
            sel[order[:m]] = True
            code1.view(2 * b, 64)[:, sel] = flip.view(2 * b, 64)[:, sel]
        with env(**CFG_ONLY), torch.no_grad():
            trainer.vit.clear_ffn_cache()
            trainer.vit(code0, lab, drop, cfg_pair=True, cfg_delta=False)   # 存残差
            approx = trainer.vit(code1, lab, drop, cfg_pair=True, cfg_delta=True)
        with env(), torch.no_grad():
            trainer.vit.clear_ffn_cache()
            exact = trainer.vit(code1, lab, drop, cfg_pair=True, cfg_delta=False)
        errs.append(((cfg_logit(approx) - cfg_logit(exact)).norm()
                     / cfg_logit(exact).norm()).item())

    assert errs[0] < 1e-4, f"m=0 时误差应落回数值噪声, 实际 {errs[0]:.3e}"
    assert all(errs[i] < errs[i + 1] for i in range(len(errs) - 1)), \
        f"误差没有随输入漂移单调增长: {errs}"
    return "CFG logit 相对误差 @ 改动 0/4/16/48 个 token = " + \
        ", ".join(f"{e:.2e}" for e in errs)


def d5_cache_cleared_between_generations():
    """两次 generation 必须逐位一致 —— 残差不能跨 generation 串台。"""
    trainer = make_trainer(seed=15, input_size=8, register=1, depth=6)
    with env(**CFG_VSIM):
        a = run(trainer, nb_sample=4, steps=12, cfg_w=1.5, seed=7)
        b = run(trainer, nb_sample=4, steps=12, cfg_w=1.5, seed=7)
    assert_bitwise(a[2], b[2], "logits(两次 generation)")
    assert_bitwise(a[1], b[1], "codes(两次 generation)")
    return "两次 generation 逐位一致"


def d6_guards():
    """w=0 与"和旧 delta 缓存方案组合"必须报错, 而不是静默算错。"""
    trainer = make_trainer(seed=16, input_size=8, register=1, depth=6)
    hit = []
    with env(**CFG_ONLY):
        try:
            run(trainer, nb_sample=4, steps=8, cfg_w=0.0, seed=7)
        except RuntimeError as e:
            hit.append("w=0")
        else:
            raise AssertionError("w=0 时应当报错")
    # HALTON_PARTIAL_UPDATE 但不开 LAZY_CACHE = 旧的 FFN-delta 方案
    with env(HALTON_CFG_DELTA=1, HALTON_PARTIAL_UPDATE=1):
        try:
            run(trainer, nb_sample=4, steps=12, cfg_w=1.5, seed=7)
        except RuntimeError as e:
            hit.append("legacy-delta")
        else:
            raise AssertionError("与旧 FFN-delta 方案组合时应当报错")
    return " / ".join(hit) + " 均已拦截"


def d7_with_lazy_token_cache_no_miss():
    """与 HALTON_LAZY_CACHE 同开: 跨 step 缓存零 miss, 且 batch 维正确收缩到 cond 半。"""
    trainer = make_trainer(seed=17, input_size=8, register=1, depth=6)
    with env(**CFG_LAZY):
        out = run(trainer, nb_sample=4, steps=12, cfg_w=1.5, seed=7)
        st = stats_of(trainer)
    for k in ("kv_miss", "restore_miss", "score_miss", "head_miss", "cfg_delta_miss"):
        assert st[k] == 0, f"{k}={st[k]} (应为 0): {st}"
    assert st["cfg_delta_reuse_calls"] > 0 and st["partial_calls"] > 0, st
    # 最后一步是 reuse 步 → 所有跨 step 缓存应停在半 batch 上
    enc = trainer.vit.transformer
    assert enc.lazy_restore_cache.size(0) == 4, enc.lazy_restore_cache.shape
    for i, blk in enumerate(enc.layers):
        if blk.attn.cached_k is not None:
            assert blk.attn.cached_k.size(0) == 4, (i, blk.attn.cached_k.shape)
    assert torch.isfinite(out[0]).all()
    return (f"partial={st['partial_calls']} reuse={st['cfg_delta_reuse_calls']} "
            f"active_ratio={st['partial_active_ratio']:.3f}, 全部 miss=0")


def d8_with_vsim_budget_unchanged():
    """与 HALTON_LAZY_VSIM 同开: 零 miss, 且逐步预算与只开 vsim 时逐个相同。

    预算 = ceil(rho_t * N) 只取决于 step, 与 CFG 残差缓存无关 —— 若开了残差缓存
    之后预算变了, 说明两者互相干扰 (例如 batch 收缩把 vsim 的分支带偏)。
    """
    def budgets(cfg):
        trainer = make_trainer(seed=18, input_size=8, register=1, depth=6)
        seen = []
        orig = T.TransformerEncoder._forward_lazy

        def spy(self, x, cond, mask, forced_mask, start, end, cfg_pair, **kw):
            out = orig(self, x, cond, mask, forced_mask, start, end, cfg_pair, **kw)
            seen.append((kw.get("step"), int(x.size(0)),
                         int(self.lazy_last_active_idx.size(1))
                         if self.lazy_last_active_idx is not None else int(x.size(1))))
            return out

        T.TransformerEncoder._forward_lazy = spy
        try:
            with env(**cfg):
                trainer_out = run(trainer, nb_sample=4, steps=12, cfg_w=1.5, seed=7)
        finally:
            T.TransformerEncoder._forward_lazy = orig
        return seen, stats_of(trainer), trainer_out

    ref, st_ref, _ = budgets(VSIM_ONLY)
    got, st_got, _ = budgets(CFG_VSIM)

    for k in ("kv_miss", "restore_miss", "score_miss", "head_miss", "cfg_delta_miss"):
        assert st_got[k] == 0, f"{k}={st_got[k]} (应为 0): {st_got}"
    assert [(s, k) for s, _, k in ref] == [(s, k) for s, _, k in got], \
        f"逐步预算被改变:\n  vsim only = {[(s, k) for s, _, k in ref]}\n  + cfgdelta = {[(s, k) for s, _, k in got]}"
    halves = [b for _, b, _ in got if b == 4]
    assert len(halves) == st_got["cfg_delta_reuse_calls"] > 0, (len(halves), st_got)
    return (f"{len(ref)} 步预算逐个相同; 其中 {len(halves)} 步在半 batch 上跑; "
            f"reuse={st_got['cfg_delta_reuse_calls']}, 全部 miss=0")


def d9_vsim_selection_on_half_batch():
    """lazy 步的 V 打分必须在半 batch 上做, 且不再做 cfg_pair 的两半分数平均。"""
    trainer = make_trainer(seed=19, input_size=8, register=1, depth=6)
    seen = []
    orig = T.TransformerEncoder._lazy_select_vsim

    def spy(self, xv, prev_v, budget, cfg_pair):
        seen.append((int(xv.size(0)), bool(cfg_pair),
                     None if prev_v is None else int(prev_v.size(0))))
        return orig(self, xv, prev_v, budget, cfg_pair)

    T.TransformerEncoder._lazy_select_vsim = spy
    try:
        with env(**CFG_VSIM):
            run(trainer, nb_sample=4, steps=12, cfg_w=1.5, seed=7)
    finally:
        T.TransformerEncoder._lazy_select_vsim = orig

    assert seen, "vsim 打分一次都没跑到"
    for b_xv, cp, b_prev in seen:
        assert b_xv == 4, f"打分时 batch={b_xv}, 期望 4 (只有 cond 半)"
        assert not cp, "半 batch 上不应再做 cfg_pair 分数平均"
        assert b_prev == 4, f"prev_v batch={b_prev}, 期望 4 —— 缓存没有正确切到 cond 半"
    return f"{len(seen)} 次打分全部在 batch=4 上, cfg_pair=False, prev_v 已对齐"


def d10_standalone_speed_shape():
    """只开 CFG 残差缓存 (不开 token cache): 主干算力应当按 reuse 步数成比例下降。"""
    trainer = make_trainer(seed=20, input_size=8, register=1, depth=6)
    seen, orig, spy = spy_encoder_batches()
    T.TransformerEncoder.forward = spy
    try:
        with env(HALTON_CFG_DELTA=1, HALTON_CFG_DELTA_N=3):
            run(trainer, nb_sample=4, steps=12, cfg_w=1.5, seed=7)
            st = stats_of(trainer)
    finally:
        T.TransformerEncoder.forward = orig
    tok = sum(b for b, _ in seen)
    full_tok = 8 * len(seen)
    assert st["cfg_delta_miss"] == 0, st
    assert 0 < st["cfg_delta_reuse_calls"] < len(seen), st
    return (f"N=3: {st['cfg_delta_full_calls']} 整批 / {st['cfg_delta_reuse_calls']} reuse, "
            f"主干 batch·step 降到 {tok / full_tok:.0%}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"device = {DEVICE}")
    print("HALTON_CFG_DELTA (LazyMAR CFG 残差缓存)")
    check("D1  输入不变时重构精确", d1_reconstruction_is_exact_when_input_unchanged)
    check("D2  N=1 等价于关闭 (逐位一致)", d2_refresh_every_step_equals_baseline)
    check("D3  端到端: lazy 步只跑半个 batch", d3_end_to_end_half_batch)
    check("D4  误差随输入漂移增长", d4_error_grows_with_input_drift)
    check("D5  跨 generation 缓存清理", d5_cache_cleared_between_generations)
    check("D6  接口保护", d6_guards)
    check("D7  + LAZY_CACHE 零 miss", d7_with_lazy_token_cache_no_miss)
    check("D8  + LAZY_VSIM 预算不受影响", d8_with_vsim_budget_unchanged)
    check("D9  + LAZY_VSIM 打分在半 batch 上", d9_vsim_selection_on_half_batch)
    check("D10 单独开启也能省算力", d10_standalone_speed_shape)

    ok = sum(1 for r in RESULTS if r[0] is True)
    bad = [r for r in RESULTS if r[0] is False]
    print(f"\n{ok}/{len(RESULTS)} passed")
    for _, name, info in bad:
        print(f"  FAILED: {name}: {info}")
    sys.exit(1 if bad else 0)
