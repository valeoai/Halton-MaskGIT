"""LazyMAR Token Cache (HALTON_LAZY_CACHE) 的正确性测试。

不需要预训练权重: 用随机初始化的 Transformer + 真正的 HaltonSampler 跑完整的
decoding 循环, 逐位比较 logits / code / image。

核心断言:
  T1  cache ratio = 0  →  与 baseline (不开任何 cache) 逐位一致
  T2  lazy 开启但采样器不给 active_mask (全量步) → 与 baseline 逐位一致
  T3  cache ratio > 0  →  能跑通; active 数 = 预期预算; 无缓存 miss
  T4  强制集合 (U_t ∪ U_{t-1} ∪ register) 必被选中 —— 否则会用陈旧 logit commit
  T5  CFG 两半选中同一批 token (cfg_pair)
  T6  不依赖 depth / token 数 / hidden / batch / register / proj 的硬编码
  T7  与既有 cache 方案互斥的保护生效
  T8  跨 generation 的缓存清理 (clear_ffn_cache) 生效
  T10 r=1 (默认) → active 集合 *恰好* 等于 U_t ∪ U_{t-1} ∪ register
  T11 r=1 时一次 V 余弦都不算
  T12 lazy 区间默认覆盖全部层, 且入口层不再算全长 K/V
  T13 layer 0 的输入对 inactive token 逐位不变 → 缓存的 K/V 精确有效 (start=0 的前提)
  T14 head 只算 active 行 —— 与全量算 head 数值等价 (不是新的近似)

用法:  python test_lazy_token_cache.py
"""
import os
import sys
import contextlib
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Network.transformer import Transformer
from Sampler.halton_sampler import HaltonSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAZY_ENVS = ["HALTON_LAZY_CACHE", "HALTON_LAZY_CACHE_RATIO", "HALTON_LAZY_REGISTER_ACTIVE",
             "HALTON_PARTIAL_UPDATE", "HALTON_CACHE_REFRESH_N",
             "HALTON_PARTIAL_START_LAYER", "HALTON_PARTIAL_END_LAYER",
             "HALTON_LAZY_START_LAYER", "HALTON_LAZY_END_LAYER", "HALTON_LAZY_HEAD",
             "HALTON_LAYER_CACHE", "HALTON_ATTN_CACHE",
             "HALTON_LAZY_VSIM", "HALTON_LAZY_VSIM_SCHED",
             "HALTON_CFG_DELTA", "HALTON_CFG_DELTA_N", "HALTON_CFG_DELTA_START"]


@contextlib.contextmanager
def env(**kw):
    """临时设置 env, 退出时恢复 (未指定的开关一律清空, 保证互不污染)。"""
    old = {k: os.environ.get(k) for k in LAZY_ENVS}
    for k in LAZY_ENVS:
        os.environ.pop(k, None)
    for k, v in kw.items():
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class StubAE:
    """decode_code: code 的确定性函数, 让 image 的逐位比较等价于 code 的比较。"""
    def decode_code(self, code):
        return (code.float().unsqueeze(1).repeat(1, 3, 1, 1) / 16384.0) * 2 - 1


def make_trainer(input_size=8, hidden=64, depth=6, heads=4, register=1, proj=1,
                 codebook_size=64, seed=0):
    torch.manual_seed(seed)
    vit = Transformer(input_size=input_size, nclass=10, hidden_dim=hidden,
                      codebook_size=codebook_size, depth=depth, heads=heads,
                      mlp_dim=hidden * 4, dropout=0., register=register, proj=proj)

    # !! 关键: Transformer.initialize_weights 是 DiT 式零初始化 ——
    #    nn.init.constant_(block.mlp[1].weight, 0) 让随机模型的 AdaLN 门
    #    alpha1/alpha2 全为 0, 于是 x = x + alpha*(...) 退化成恒等映射, 整个
    #    transformer stack 什么都不做, 任何 cache 测试都会平凡通过。
    #    训练好的 checkpoint 这些门是非零的, 所以这里必须重新随机化来模拟。
    for blk in vit.transformer.layers:
        torch.nn.init.normal_(blk.mlp[1].weight, std=0.02)
        torch.nn.init.normal_(blk.mlp[1].bias, std=0.5)

    vit = vit.to(DEVICE).eval()
    trainer = types.SimpleNamespace()
    trainer.vit = vit
    trainer.input_size = input_size
    trainer.ae = StubAE()
    trainer.autocast = contextlib.nullcontext()
    trainer.args = types.SimpleNamespace(device=DEVICE, mask_value=codebook_size,
                                         codebook_size=codebook_size)
    return trainer


def run(trainer, nb_sample=4, steps=8, cfg_w=1.5, seed=1234, randomize=False):
    """跑一次完整采样, 返回 (image, codes, 每步 logits)。"""
    sampler = HaltonSampler(sm_temp_min=1.0, sm_temp_max=1.0, temp_pow=1, w=cfg_w,
                            sched_pow=2, step=steps, randomize=randomize, top_k=-1)
    logits = []
    h = trainer.vit.register_forward_hook(lambda m, i, o: logits.append(o.detach().clone()))
    try:
        torch.manual_seed(seed)
        labels = torch.arange(nb_sample, device=DEVICE) % 10
        x, l_codes, l_U, l_M = sampler(trainer, nb_sample=nb_sample, labels=labels, verbose=False)
    finally:
        h.remove()
    return x, l_codes, logits, l_U


def assert_bitwise(a, b, what):
    assert len(a) == len(b), f"{what}: 长度不同 {len(a)} vs {len(b)}"
    for i, (u, v) in enumerate(zip(a, b)):
        assert u.shape == v.shape, f"{what}[{i}]: shape {tuple(u.shape)} vs {tuple(v.shape)}"
        if not torch.equal(u, v):
            d = (u.float() - v.float()).abs()
            raise AssertionError(
                f"{what}[{i}] 不是逐位一致: max|Δ|={d.max().item():.3e} "
                f"mean|Δ|={d.mean().item():.3e} 不同元素={int((u != v).sum())}/{u.numel()}")


# ---------------------------------------------------------------------------
RESULTS = []


class Skip(Exception):
    pass


def check(name, fn):
    try:
        info = fn() or ""
        RESULTS.append((True, name, info))
        print(f"  PASS  {name}  {info}")
    except Skip as e:
        RESULTS.append((None, name, str(e)))
        print(f"  SKIP  {name}  {e}")
    except Exception as e:
        RESULTS.append((False, name, str(e)))
        print(f"  FAIL  {name}\n        {e}")


def baseline_supports(**cfg):
    """proj>1 在未改动的 baseline 里就跑不通 (Transformer.forward 给 rearrange 传了
    模式里没有的 proj_h)。所有实际配置都是 proj=1, 属于死代码, 不在本次改动范围内,
    因此这里探测后跳过而不是当作回归。"""
    trainer = make_trainer(**cfg)
    code = torch.randint(0, 64, (1, cfg.get("input_size", 8), cfg.get("input_size", 8)),
                         device=DEVICE)
    try:
        with env(), torch.no_grad():
            trainer.vit(code, torch.zeros(1, dtype=torch.long, device=DEVICE),
                        torch.zeros(1, dtype=torch.bool, device=DEVICE))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
def t0_harness_is_sensitive():
    """先证明测试本身有鉴别力, 否则后面的 "逐位一致" 全是空洞的。

    两件事:
      (a) AdaLN 门非零 —— 否则 x = x + alpha*(...) 是恒等映射, transformer 什么
          都不做 (Transformer.initialize_weights 的 DiT 零初始化就是这个效果);
      (b) ratio>0 必须让输出 *变化* —— 缓存若对结果毫无影响, T1 的一致性无意义。
    """
    trainer = make_trainer(seed=101)
    gates = [blk.mlp[1].weight.abs().max().item() for blk in trainer.vit.transformer.layers]
    assert min(gates) > 0, f"AdaLN 门为零 → transformer 退化成恒等映射: {gates}"

    with env():
        base = run(trainer, seed=99)
    with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=0.9,
             HALTON_PARTIAL_START_LAYER=1, HALTON_PARTIAL_END_LAYER=4):
        lazy = run(trainer, seed=99)
    diffs = [(u.float() - v.float()).abs().max().item() for u, v in zip(base[2], lazy[2])]
    assert max(diffs) > 0, "ratio=0.9 与 baseline 完全相同 → 缓存没起作用, 测试无鉴别力"
    return f"AdaLN 门 max={max(gates):.3f}; ratio=0.9 使 logits 最大变化 {max(diffs):.3f}"


def t1_ratio_zero_equals_baseline():
    trainer = make_trainer()
    with env():
        base = run(trainer)
    with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=0,
             HALTON_PARTIAL_START_LAYER=1, HALTON_PARTIAL_END_LAYER=4):
        lazy = run(trainer)
        stats = trainer.vit.get_lazy_stats()
    assert_bitwise(base[2], lazy[2], "logits")
    assert torch.equal(base[0], lazy[0]), "image 不一致"
    assert_bitwise(base[1], lazy[1], "codes")
    assert stats["partial_calls"] > 0, "没有走到 partial 分支, 测试无效"
    assert stats["kv_miss"] == 0 and stats["restore_miss"] == 0, f"缓存 miss: {stats}"
    assert abs(stats["mean_active_ratio"] - 1.0) < 1e-9, stats
    return (f"partial_calls={stats['partial_calls']} full_calls={stats['full_calls']} "
            f"active_ratio={stats['mean_active_ratio']:.3f}")


def t2_full_steps_equal_baseline():
    """lazy 开着但采样器从不给 active_mask → 每步都是全量步, 仍需逐位一致。"""
    trainer = make_trainer(seed=3)
    with env():
        base = run(trainer)
    with env(HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=0.9):  # ratio 无效, 因为没有 partial 步
        lazy = run(trainer)
        stats = trainer.vit.get_lazy_stats()
    assert_bitwise(base[2], lazy[2], "logits")
    assert stats["partial_calls"] == 0 and stats["full_calls"] > 0, stats
    return f"full_calls={stats['full_calls']}"


def t3_ratio_positive_runs():
    trainer = make_trainer(seed=5)
    out = {}
    for r in (0.25, 0.5, 0.75):
        with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=r,
                 HALTON_PARTIAL_START_LAYER=1, HALTON_PARTIAL_END_LAYER=4):
            x, codes, logits, _ = run(trainer)
            st = trainer.vit.get_lazy_stats()
        assert torch.isfinite(x).all(), f"ratio={r}: 输出含 NaN/Inf"
        assert st["kv_miss"] == 0, f"ratio={r}: kv_miss={st['kv_miss']}"
        assert st["restore_miss"] == 0, f"ratio={r}: restore_miss={st['restore_miss']}"
        assert st["score_miss"] == 0, f"ratio={r}: score_miss={st['score_miss']}"
        out[r] = round(st["mean_active_ratio"], 3)
    # ratio 越大 → 重算的 token 越少
    assert out[0.25] > out[0.5] > out[0.75], out
    return f"mean_active_ratio {out}"


def t4_forced_tokens_always_recomputed():
    """U_t ∪ U_{t-1} ∪ register 必须落在 active 集合里。"""
    import Network.transformer as T
    trainer = make_trainer(seed=7)
    seen = []
    orig = T.TransformerEncoder._lazy_select

    def spy(self, xv, prev_v, forced_mask, budget, cfg_pair):
        idx = orig(self, xv, prev_v, forced_mask, budget, cfg_pair)
        if forced_mask is not None:
            b, n = forced_mask.shape
            sel = torch.zeros(b, n, dtype=torch.bool, device=idx.device)
            sel.scatter_(1, idx, True)
            missing = int((forced_mask & ~sel).sum())
            seen.append((missing, int(idx.size(1)), n, forced_mask, sel))
        return idx

    T.TransformerEncoder._lazy_select = spy
    try:
        with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=0.9,
                 HALTON_PARTIAL_START_LAYER=1, HALTON_PARTIAL_END_LAYER=4):
            run(trainer)
    finally:
        T.TransformerEncoder._lazy_select = orig
    assert seen, "没有进入 partial 分支"
    for missing, k, n, forced, sel in seen:
        assert missing == 0, f"有 {missing} 个强制 token 没被选中"
    # register (最后一列) 必须 active
    for _, _, n, forced, sel in seen:
        assert bool(sel[:, -1].all()), "register token 未被强制 active"
    ks = {k for _, k, _, _, _ in seen}
    return f"partial 步数={len(seen)} k={sorted(ks)}"


def t5_cfg_halves_share_selection():
    import Network.transformer as T
    trainer = make_trainer(seed=11)
    same = []
    orig = T.TransformerEncoder._lazy_select

    def spy(self, xv, prev_v, forced_mask, budget, cfg_pair):
        idx = orig(self, xv, prev_v, forced_mask, budget, cfg_pair)
        if forced_mask is not None and idx.size(0) % 2 == 0 and cfg_pair:
            h = idx.size(0) // 2
            same.append(bool(torch.equal(idx[:h], idx[h:])))
        return idx

    T.TransformerEncoder._lazy_select = spy
    try:
        with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=0.8,
                 HALTON_PARTIAL_START_LAYER=1, HALTON_PARTIAL_END_LAYER=4):
            run(trainer, cfg_w=1.5)
    finally:
        T.TransformerEncoder._lazy_select = orig
    assert same, "没有观察到 CFG partial 步"
    assert all(same), f"cond/uncond 两半选了不同 token: {same}"
    return f"{len(same)} 个 partial 步两半选择一致"


def t6_no_hardcoded_shapes():
    """不同 depth / token 数 / hidden / register / proj / batch / step 都要 ratio=0 == baseline。"""
    cfgs = [
        dict(input_size=8,  hidden=64,  depth=6,  heads=4,  register=1, proj=1),
        dict(input_size=6,  hidden=96,  depth=3,  heads=6,  register=0, proj=1),   # 36 token, 无 register
        dict(input_size=12, hidden=48,  depth=8,  heads=8,  register=3, proj=1),   # 144 token, 3 register
        dict(input_size=8,  hidden=64,  depth=4,  heads=4,  register=1, proj=2),   # proj=2
    ]
    cfgs = [c for c in cfgs if baseline_supports(**c)]
    runs = [dict(nb_sample=1, steps=6, cfg_w=0.0),     # 无 CFG, batch=1
            dict(nb_sample=3, steps=12, cfg_w=2.0),
            dict(nb_sample=2, steps=7,  cfg_w=1.0, randomize=True)]
    assert cfgs, "没有可用配置"
    n = 0
    for ci, c in enumerate(cfgs):
        trainer = make_trainer(seed=20 + ci, **c)
        for ri, r in enumerate(runs):
            with env():
                base = run(trainer, seed=100 + ri, **r)
            with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=0,
                     HALTON_PARTIAL_START_LAYER=1, HALTON_PARTIAL_END_LAYER=99):
                lazy = run(trainer, seed=100 + ri, **r)
                st = trainer.vit.get_lazy_stats()
            assert_bitwise(base[2], lazy[2], f"logits(cfg{ci},run{ri})")
            assert st["kv_miss"] == 0 and st["restore_miss"] == 0, (ci, ri, st)
            n += 1
    note = "" if len(cfgs) == 4 else "  (proj>1 在 baseline 即不可用, 已跳过)"
    return f"{len(cfgs)} 个模型 x {len(runs)} 组采样参数 = {n} 次逐位一致{note}"


def t6b_no_hardcoded_shapes_lazy_active():
    """同上, 但 ratio>0 (真正走缓存) —— 只验证能跑通且统计正常。"""
    cfgs = [
        dict(input_size=6,  hidden=96, depth=3, heads=6, register=0, proj=1),
        dict(input_size=12, hidden=48, depth=8, heads=8, register=3, proj=1),
        dict(input_size=8,  hidden=64, depth=4, heads=4, register=1, proj=2),
    ]
    cfgs = [c for c in cfgs if baseline_supports(**c)]
    got = []
    for ci, c in enumerate(cfgs):
        trainer = make_trainer(seed=40 + ci, **c)
        with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=0.7,
                 HALTON_PARTIAL_START_LAYER=1, HALTON_PARTIAL_END_LAYER=99):
            x, codes, logits, _ = run(trainer, nb_sample=2, steps=10, cfg_w=1.5, randomize=True)
            st = trainer.vit.get_lazy_stats()
        assert torch.isfinite(x).all(), f"cfg{ci}: NaN/Inf"
        assert st["kv_miss"] == 0 and st["restore_miss"] == 0 and st["score_miss"] == 0, (ci, st)
        got.append(round(st["mean_active_ratio"], 3))
    note = "proj=2 / " if len(cfgs) == 3 else ""
    return f"mean_active_ratio={got} (含 randomize=True / {note}register=0)"


def t7_mutual_exclusion():
    trainer = make_trainer(seed=13)
    for other in ("HALTON_LAYER_CACHE", "HALTON_ATTN_CACHE"):
        raised = False
        try:
            with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, **{other: 1}):
                run(trainer, steps=8)
        except RuntimeError as e:
            raised = "互斥" in str(e)
        assert raised, f"{other} 与 HALTON_LAZY_CACHE 同开时没有报错"
    return "HALTON_LAYER_CACHE / HALTON_ATTN_CACHE 冲突均被拦截"


def t8_cache_cleared_between_generations():
    """连续两次 generation 必须给出相同结果 —— 证明缓存没有跨 __call__ 串台。"""
    trainer = make_trainer(seed=17)
    with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=0.8,
             HALTON_PARTIAL_START_LAYER=1, HALTON_PARTIAL_END_LAYER=4):
        a = run(trainer, seed=555)
        b = run(trainer, seed=555)
    assert_bitwise(a[2], b[2], "logits(两次 generation)")
    # 缓存确实被清空了
    blk = trainer.vit.transformer.layers[1]
    with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=0.8):
        trainer.vit.clear_ffn_cache()
    assert blk.attn.cached_k is None and blk.attn.cached_v is None, "clear 后 K/V 缓存仍在"
    assert trainer.vit.transformer.lazy_restore_cache is None, "clear 后 restore 缓存仍在"
    return "两次 generation 逐位一致, clear_ffn_cache 清空所有 lazy 状态"


def t9_ddp_clear_hook():
    """DDP 包装下 clear 钩子仍能触发 (getattr(vit,'module',vit))。"""
    trainer = make_trainer(seed=19)
    real = trainer.vit
    called = {"n": 0}
    orig_clear = real.clear_ffn_cache

    def counting_clear():
        called["n"] += 1
        orig_clear()
    real.clear_ffn_cache = counting_clear

    class FakeDDP(torch.nn.Module):
        """模拟 DDP: 不转发属性访问。"""
        def __init__(self, m):
            super().__init__()
            self.module = m
        def forward(self, *a, **kw):
            return self.module(*a, **kw)
        def __getattr__(self, name):
            if name in ("clear_ffn_cache", "get_lazy_stats"):
                raise AttributeError(name)
            return super().__getattr__(name)

    trainer.vit = FakeDDP(real).to(DEVICE)
    with env():
        run(trainer, steps=4)
    assert called["n"] == 1, f"DDP 包装下 clear_ffn_cache 被调用 {called['n']} 次 (应为 1)"
    return "DDP 包装下缓存清理正常触发"


# ---------------------------------------------------------------------------
# forced-only 模式 (r = 1, 默认): 不打分, active = U_t ∪ U_{t-1} ∪ register
# ---------------------------------------------------------------------------
def t10_forced_only_selection_is_exact():
    """默认 (不设 RATIO) 就该是 r=1: 选中的 token 集合 *恰好* 是强制集合。"""
    import Network.transformer as T
    trainer = make_trainer(seed=17)
    seen = []
    orig = T.TransformerEncoder._lazy_select

    def spy(self, xv, prev_v, forced_mask, budget, cfg_pair):
        idx = orig(self, xv, prev_v, forced_mask, budget, cfg_pair)
        if forced_mask is not None:
            b, n = forced_mask.shape
            sel = torch.zeros(b, n, dtype=torch.bool, device=idx.device)
            sel.scatter_(1, idx, True)
            seen.append((bool(torch.equal(sel, forced_mask)), xv is None,
                         int(idx.size(1)), int(forced_mask[0].sum())))
        return idx

    T.TransformerEncoder._lazy_select = spy
    try:
        with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1):   # 不给 RATIO → 默认 1.0
            x, _, _, l_U = run(trainer, nb_sample=2, steps=10, cfg_w=1.5)
            st = trainer.vit.get_lazy_stats()
    finally:
        T.TransformerEncoder._lazy_select = orig

    assert seen, "没有进入 partial 分支"
    assert all(eq for eq, _, _, _ in seen), "active 集合与强制集合不相等"
    assert all(no_xv for _, no_xv, _, _ in seen), "forced-only 路径仍然拿到了全长 V"
    assert torch.isfinite(x).all(), "输出含 NaN/Inf"
    assert st["scored_calls"] == 0, f"仍走了打分路径: {st}"
    assert st["forced_only_calls"] == st["lazy_calls"], st
    assert st["kv_miss"] == 0 and st["restore_miss"] == 0 and st["forced_nonuniform"] == 0, st
    ks = [k for _, _, k, _ in seen]
    return (f"partial 步 {len(seen)} 个, k={ks}, "
            f"partial_active_ratio={st['partial_active_ratio']:.3f}")


def t11_no_similarity_computed():
    """r=1 时不能再调用 F.cosine_similarity —— 打分本身也是要算力的。"""
    import Network.transformer as T
    trainer = make_trainer(seed=19)
    calls = {"n": 0}
    orig = T.F.cosine_similarity

    def boom(*a, **kw):
        calls["n"] += 1
        raise AssertionError("r=1 下不应计算 V 的余弦相似度")

    T.F.cosine_similarity = boom
    try:
        with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=1.0):
            run(trainer, nb_sample=2, steps=10, cfg_w=1.5)
        # 反证: ratio<1 时它必须被调用, 否则这个测试是空的
        with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=0.5):
            raised = False
            try:
                run(trainer, nb_sample=2, steps=10, cfg_w=1.5)
            except AssertionError:
                raised = True
    finally:
        T.F.cosine_similarity = orig
    assert calls["n"] == 1 and raised, f"打分路径没被触发, 测试无鉴别力 (calls={calls['n']})"
    return "r=1 调用 0 次; r=0.5 调用 1 次 (反证测试有效)"


def t12_default_range_is_all_layers():
    """不设层范围时, lazy 应覆盖 0..depth-1, 且没有任何层走"全长 K/V"入口。"""
    import Network.transformer as T
    depth = 6
    trainer = make_trainer(seed=21, depth=depth)
    idx_of = {id(b): i for i, b in enumerate(trainer.vit.transformer.layers)}
    lazy_layers, full_kv_layers, plain_layers = set(), set(), set()
    orig_lazy = T.Block.forward_lazy
    orig_fwd = T.Block.forward

    def spy_lazy(self, x, cond, active_idx, seq_len, mask=None, select_fn=None, gather_x=False):
        lazy_layers.add(idx_of[id(self)])
        if active_idx is None:
            full_kv_layers.add(idx_of[id(self)])
        return orig_lazy(self, x, cond, active_idx, seq_len, mask=mask,
                         select_fn=select_fn, gather_x=gather_x)

    def spy_fwd(self, x, cond, mask=None, active_mask=None):
        plain_layers.add(idx_of[id(self)])
        return orig_fwd(self, x, cond, mask=mask, active_mask=active_mask)

    T.Block.forward_lazy, T.Block.forward = spy_lazy, spy_fwd
    try:
        with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1):
            run(trainer, nb_sample=2, steps=10, cfg_w=1.5)
    finally:
        T.Block.forward_lazy, T.Block.forward = orig_lazy, orig_fwd

    assert lazy_layers == set(range(depth)), f"lazy 只覆盖了 {sorted(lazy_layers)}"
    assert not plain_layers, f"层 {sorted(plain_layers)} 仍走了全量 Block.forward"
    assert not full_kv_layers, f"层 {sorted(full_kv_layers)} 仍在算全长 K/V"

    # 显式设层范围时仍要被尊重 (向后兼容)
    lazy_layers.clear(); plain_layers.clear()
    T.Block.forward_lazy, T.Block.forward = spy_lazy, spy_fwd
    try:
        with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1,
                 HALTON_LAZY_START_LAYER=2, HALTON_LAZY_END_LAYER=4):
            run(trainer, nb_sample=2, steps=10, cfg_w=1.5)
    finally:
        T.Block.forward_lazy, T.Block.forward = orig_lazy, orig_fwd
    assert lazy_layers == {2, 3, 4}, f"层范围没被尊重: {sorted(lazy_layers)}"
    assert plain_layers == {0, 1, 5}, f"区间外的层不对: {sorted(plain_layers)}"
    return f"默认 lazy 层 = {sorted(lazy_layers | {0,1,5}) and list(range(depth))}; 显式 [2,4] 也生效"


def t13_layer0_input_frozen_for_inactive():
    """start=0 的正当性: layer 0 的输入 = tok_emb(code)+pos, 与其它 token 无关;
    code 只在 U_{t-1} 处被改写, 而 U_{t-1} 已在强制集合里 —— 所以 inactive token
    的 layer-0 输入逐位不变, 缓存里的 K/V 就是精确值 (不是近似)。"""
    import Network.transformer as T
    trainer = make_trainer(seed=23)
    seen = []
    orig = T.TransformerEncoder._forward_lazy

    def spy(self, x, cond, mask, forced_mask, start, end, cfg_pair, **kw):
        seen.append((x.detach().clone(),
                     None if forced_mask is None else forced_mask.clone()))
        return orig(self, x, cond, mask, forced_mask, start, end, cfg_pair, **kw)

    T.TransformerEncoder._forward_lazy = spy
    try:
        with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1):
            run(trainer, nb_sample=2, steps=10, cfg_w=1.5)
    finally:
        T.TransformerEncoder._forward_lazy = orig

    checked = 0
    for (x_prev, _), (x_cur, forced) in zip(seen, seen[1:]):
        if forced is None:
            continue
        inactive = ~forced                                   # (b, N)
        d = (x_cur - x_prev)[inactive]
        assert torch.equal(d, torch.zeros_like(d)), (
            f"inactive token 的 layer-0 输入变了: max|Δ|={d.abs().max().item():.3e}")
        checked += 1
    assert checked, "没有可比较的连续 partial 步"
    return f"{checked} 个 partial 步: inactive 行的 layer-0 输入逐位不变"


def _two_step(trainer, head_lazy, am, code, labels, drop):
    """跑"一个全量步 + 一个 partial 步", 返回 partial 步的 logits 与 stats。

    不经过采样器, 所以两次调用之间没有任何随机性 —— 唯一的变量就是
    HALTON_LAZY_HEAD。
    """
    vit = trainer.vit
    with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_HEAD=head_lazy):
        vit.clear_ffn_cache()
        with torch.no_grad():
            vit(code, labels, drop)                       # 全量步: 写满所有缓存
            logit = vit(code, labels, drop, active_mask=am, cfg_pair=False)
        return logit.clone(), vit.get_lazy_stats()


def t14_lazy_head_matches_full_head():
    """head 只算 active 行 vs 全量算 head: 每一行都该是同一个数。

    last_norm + head 逐 token, inactive 行的 hidden state 又本来就是缓存值,
    所以这不是新的近似 —— 只是不再重复算注定相同的结果。允许的差异只有
    GEMM 形状变化带来的浮点噪声。
    """
    isz = 8
    trainer = make_trainer(seed=31, input_size=isz)
    torch.manual_seed(0)
    code = torch.randint(0, 64, (2, isz, isz), device=DEVICE)
    labels = torch.arange(2, device=DEVICE) % 10
    drop = torch.zeros(2, dtype=torch.bool, device=DEVICE)
    am = torch.zeros(2, isz, isz, dtype=torch.bool, device=DEVICE)
    am.view(2, -1)[:, ::7] = True                          # 每行 active 数一致

    ref, st_ref = _two_step(trainer, 0, am, code, labels, drop)
    lazy, st_lazy = _two_step(trainer, 1, am, code, labels, drop)

    assert st_ref["head_partial_calls"] == 0 and st_ref["head_full_calls"] == 2, st_ref
    assert st_lazy["head_partial_calls"] == 1, st_lazy
    assert st_lazy["head_miss"] == 0, st_lazy

    d = (ref.float() - lazy.float()).abs()
    tol = 1e-4 * ref.float().abs().max().item()
    assert d.max().item() <= max(tol, 1e-5), (
        f"head 只算 active 后 logit 变了: max|Δ|={d.max().item():.3e} (tol={tol:.3e})")

    # inactive 行必须逐位来自缓存 (没有被重算过, 所以连浮点噪声都不该有)
    flat = am.view(2, -1)
    inact = (lazy - ref)[~flat]
    assert torch.equal(inact, torch.zeros_like(inact)), "inactive 行不是逐位取自缓存"
    return (f"active 行 max|Δ|={d.max().item():.2e} (浮点噪声), "
            f"inactive 行逐位相同; head_partial_calls={st_lazy['head_partial_calls']}")


def main():
    print(f"device = {DEVICE}, torch = {torch.__version__}\n")
    print("LazyMAR Token Cache 测试")
    check("T0  测试有鉴别力 (非恒等映射 / 缓存确有影响)", t0_harness_is_sensitive)
    check("T1  cache ratio=0 逐位等于 baseline",        t1_ratio_zero_equals_baseline)
    check("T2  全量步路径逐位等于 baseline",             t2_full_steps_equal_baseline)
    check("T3  cache ratio>0 可运行且预算单调",          t3_ratio_positive_runs)
    check("T4  强制集合 (U_t∪U_{t-1}∪reg) 必被重算",     t4_forced_tokens_always_recomputed)
    check("T5  CFG 两半共享同一选择",                    t5_cfg_halves_share_selection)
    check("T6  无形状硬编码 (ratio=0 逐位一致)",         t6_no_hardcoded_shapes)
    check("T6b 无形状硬编码 (ratio>0 可运行)",           t6b_no_hardcoded_shapes_lazy_active)
    check("T7  与既有 cache 方案互斥",                   t7_mutual_exclusion)
    check("T8  跨 generation 缓存清理",                  t8_cache_cleared_between_generations)
    check("T9  DDP 包装下清理钩子生效",                  t9_ddp_clear_hook)
    check("T10 r=1 时 active 集合恰好等于强制集合",       t10_forced_only_selection_is_exact)
    check("T11 r=1 时不计算 V 余弦相似度",                t11_no_similarity_computed)
    check("T12 lazy 默认覆盖全部层且不算全长 K/V",        t12_default_range_is_all_layers)
    check("T13 inactive 的 layer-0 输入逐位不变",         t13_layer0_input_frozen_for_inactive)
    check("T14 head 只算 active 与全量算数值等价",        t14_lazy_head_matches_full_head)

    n_fail = sum(1 for ok, _, _ in RESULTS if ok is False)
    n_skip = sum(1 for ok, _, _ in RESULTS if ok is None)
    print(f"\n{len(RESULTS) - n_fail - n_skip}/{len(RESULTS)} 通过"
          + (f", {n_skip} 跳过" if n_skip else "")
          + (f", {n_fail} 失败" if n_fail else ""))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
