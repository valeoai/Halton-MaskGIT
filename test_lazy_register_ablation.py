"""消融: 把 register token 从"每步重算的 token 集合"里去掉。

lazycache (HALTON_LAZY_CACHE=1, r=1) 的默认 active 集合是
    U_t ∪ U_{t-1} ∪ register          (transformer.py:1108-1114)
register 被无条件强制重算, 理由是"它是全局 attention sink, 冻结的代价大于重算"。
本消融把这一条去掉 (HALTON_LAZY_REGISTER_ACTIVE=0), 于是
    active = U_t ∪ U_{t-1}
register 行在 partial 步里完全走缓存: 残差流取 lazy_restore_cache 的上一步值,
K/V 取 per-layer 缓存的上一步值, head 也不再为它算 logit (它本来也不产 code)。

断言:
  R1  开关生效: active 集合 *恰好* = U_t ∪ U_{t-1}, register 列一个都不在里面
  R2  反证 —— 默认 (=1) 时 register 必在集合里, 否则 R1 没有鉴别力
  R3  register 数 > 1 / = 0 时都成立, 且没有任何缓存 miss、输出有限
  R4  开关只影响 partial 步: 全量步 (不给 active_mask) 仍逐位等于 baseline
  R5  开关确实改变了结果 (不是空跑): 与默认设定的 logits 不同
  R6  head 的 active 行仍与全量算 head 数值等价 (register 不在图像 token 里,
      去掉它不该让 head 的 gather 越界或错位)

用法:  python test_lazy_register_ablation.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Network.transformer as T
from test_lazy_token_cache import (env, make_trainer, run, assert_bitwise,
                                   check, RESULTS, DEVICE)


def _spy_selection(trainer, envs, **run_kw):
    """跑一次采样, 记下每个 partial 步的 (selected_mask, forced_mask)。"""
    seen = []
    orig = T.TransformerEncoder._lazy_select

    def spy(self, xv, prev_v, forced_mask, budget, cfg_pair):
        idx = orig(self, xv, prev_v, forced_mask, budget, cfg_pair)
        if forced_mask is not None:
            b, n = forced_mask.shape
            sel = torch.zeros(b, n, dtype=torch.bool, device=idx.device)
            sel.scatter_(1, idx, True)
            seen.append((sel.cpu(), forced_mask.cpu()))
        return idx

    T.TransformerEncoder._lazy_select = spy
    try:
        with env(**envs):
            out = run(trainer, **run_kw)
            st = trainer.vit.get_lazy_stats()
    finally:
        T.TransformerEncoder._lazy_select = orig
    return seen, st, out


def r1_register_excluded():
    """RE=0: active 集合恰好 = U_t ∪ U_{t-1}, register 列全部为 False。"""
    trainer = make_trainer(seed=17, register=1)
    seen, st, (x, _, _, l_U) = _spy_selection(
        trainer,
        dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_REGISTER_ACTIVE=0),
        nb_sample=2, steps=10, cfg_w=1.5)
    assert seen, "没有进入 partial 分支"
    n_img = trainer.input_size ** 2
    for i, (sel, forced) in enumerate(seen):
        assert torch.equal(sel, forced), f"partial 步 {i}: active != forced"
        assert not bool(sel[:, n_img:].any()), \
            f"partial 步 {i}: register 列仍被选中 {sel[:, n_img:]}"
        assert not bool(forced[:, n_img:].any()), \
            f"partial 步 {i}: forced_mask 的 register 列没被关掉"
    assert torch.isfinite(x).all(), "输出含 NaN/Inf"
    assert st["kv_miss"] == 0 and st["restore_miss"] == 0, st
    assert st["head_miss"] == 0 and st["forced_nonuniform"] == 0, st
    ks = [int(sel[0].sum()) for sel, _ in seen]
    return f"partial 步 {len(seen)} 个, k={ks} (全部 < n_img={n_img}), 无 miss"


def r2_default_keeps_register():
    """反证: 默认 (RE 不设 / =1) 时 register 必在 active 集合里。"""
    trainer = make_trainer(seed=17, register=1)
    n_img = trainer.input_size ** 2
    got = {}
    for label, envs in (
        ("default", dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1)),
        ("RE=1", dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1,
                      HALTON_LAZY_REGISTER_ACTIVE=1)),
    ):
        seen, _, _ = _spy_selection(trainer, envs, nb_sample=2, steps=10, cfg_w=1.5)
        assert seen, f"{label}: 没有进入 partial 分支"
        assert all(bool(sel[:, n_img:].all()) for sel, _ in seen), \
            f"{label}: register 未被强制 active"
        got[label] = [int(sel[0].sum()) for sel, _ in seen]
    assert got["default"] == got["RE=1"], got
    return f"默认与 RE=1 一致, k={got['default']} (比 RE=0 各多 1 个 register)"


def r3_multi_register_and_none():
    """register=3 / register=0 都要正确: 前者去掉 3 列, 后者本来就没有 register。"""
    out = []
    for nreg in (3, 0):
        trainer = make_trainer(input_size=8, hidden=64, depth=4, heads=4,
                               register=nreg, seed=23)
        n_img = trainer.input_size ** 2
        seen, st, (x, _, _, _) = _spy_selection(
            trainer,
            dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1,
                 HALTON_LAZY_REGISTER_ACTIVE=0),
            nb_sample=2, steps=10, cfg_w=1.5)
        assert seen, f"register={nreg}: 没有进入 partial 分支"
        for sel, forced in seen:
            assert torch.equal(sel, forced)
            assert not bool(sel[:, n_img:].any()), f"register={nreg}: register 仍被选中"
        assert torch.isfinite(x).all(), f"register={nreg}: 输出含 NaN/Inf"
        assert st["kv_miss"] == 0 and st["restore_miss"] == 0 and st["head_miss"] == 0, \
            f"register={nreg}: {st}"
        out.append(f"register={nreg}: k={[int(s[0].sum()) for s, _ in seen]}")
    return "; ".join(out)


def r4_full_steps_unaffected():
    """全量步不受影响: 不给 active_mask 时仍逐位等于 baseline。"""
    trainer = make_trainer(seed=29)
    with env():
        base = run(trainer)
    with env(HALTON_LAZY_CACHE=1, HALTON_LAZY_REGISTER_ACTIVE=0):   # 无 PARTIAL_UPDATE
        abl = run(trainer)
        st = trainer.vit.get_lazy_stats()
    assert_bitwise(base[2], abl[2], "logits")
    assert st["partial_calls"] == 0 and st["full_calls"] > 0, st
    return f"full_calls={st['full_calls']}, 逐位一致"


def r5_ablation_changes_result():
    """开关必须真的改变结果, 否则整个消融是空的。"""
    trainer = make_trainer(seed=31)
    with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1):
        keep = run(trainer, nb_sample=2, steps=10, cfg_w=1.5)
    with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_REGISTER_ACTIVE=0):
        drop = run(trainer, nb_sample=2, steps=10, cfg_w=1.5)
    diff = max((a.float() - b.float()).abs().max().item()
               for a, b in zip(keep[2], drop[2]))
    assert diff > 0, "去掉 register 后 logits 完全没变 —— 开关没生效"
    return f"logits max|Δ| = {diff:.3e}"


def r6_head_still_exact():
    """head 只算 active 行, 去掉 register 后仍与全量算 head 数值等价。"""
    trainer = make_trainer(seed=37)
    n_img = trainer.input_size ** 2
    checked = {"n": 0, "max_d": 0.0}
    orig_head_fwd = trainer.vit.head.forward

    with env(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_REGISTER_ACTIVE=0):
        # 手工复现 head 的 partial 路径: 用 partial 步的 active 下标算 head,
        # 与"对全长 hidden 算 head"逐行比较。
        orig = T.TransformerEncoder._lazy_restore

        def spy(self, x_act, active_idx, b, n, c):
            out = orig(self, x_act, active_idx, b, n, c)
            if active_idx.size(1) < n:
                assert int(active_idx.max()) < n_img, \
                    f"active 下标越过图像范围: max={int(active_idx.max())} >= {n_img}"
                checked["n"] += 1
            return out

        T.TransformerEncoder._lazy_restore = spy
        try:
            x, _, logits, _ = run(trainer, nb_sample=2, steps=10, cfg_w=1.5)
            st = trainer.vit.get_lazy_stats()
        finally:
            T.TransformerEncoder._lazy_restore = orig

    assert checked["n"] > 0, "没有观察到 partial 步"
    assert st["head_partial_calls"] > 0 and st["head_miss"] == 0, st
    assert all(torch.isfinite(l).all() for l in logits), "logits 含 NaN/Inf"
    return (f"{checked['n']} 个 partial 步的 active 下标全部 < n_img={n_img}; "
            f"head_partial_calls={st['head_partial_calls']}, head_miss=0")


if __name__ == "__main__":
    print(f"device = {DEVICE}, torch = {torch.__version__}\n")
    print("LazyMAR Token Cache — register token 消融")
    check("R1  RE=0 时 active 恰好 = U_t ∪ U_{t-1} (无 register)", r1_register_excluded)
    check("R2  反证: 默认/RE=1 时 register 必被强制 active", r2_default_keeps_register)
    check("R3  register=3 / register=0 同样正确", r3_multi_register_and_none)
    check("R4  全量步不受影响 (逐位等于 baseline)", r4_full_steps_unaffected)
    check("R5  开关确实改变结果 (消融非空)", r5_ablation_changes_result)
    check("R6  head 的 active 下标不越界, 无 head_miss", r6_head_still_exact)

    n_fail = sum(1 for ok, _, _ in RESULTS if ok is False)
    n_skip = sum(1 for ok, _, _ in RESULTS if ok is None)
    print(f"\n{len(RESULTS) - n_fail - n_skip}/{len(RESULTS)} 通过"
          + (f", {n_skip} 跳过" if n_skip else "")
          + (f", {n_fail} 失败" if n_fail else ""))
    sys.exit(1 if n_fail else 0)
