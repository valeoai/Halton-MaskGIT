"""step gate 的**左端**消融 (HALTON_PARTIAL_START_STEP) 的正确性测试。

采样器原本把 partial-update 的门写死成  5 <= index < 31  (Sampler/halton_sampler.py):
  * 左端 5  —— 出自 analyze_ffn_delta_stability, "前几步整幅图还在剧烈变化,
    缓存里的东西陈旧得没法用"。这个 5 从来没有被 50k FID 扫过。
  * 右端 31 —— 最后一步必须全量, 本次不动。

本 env 让左端可调 (0 = 从第一步就开始用缓存), 用来扫 FID/IS。

断言:
  G1  不设 env 时行为与旧代码逐位一致 (默认仍是 5)
  G2  start=0..5 的 partial 步集合与门的定义逐步吻合 (含 refresh 的相位)
  G3  partial 步数随 start 减小而单调增加; index >= 31 永远是全量步
  G4  start 大到门关死 → 一个 partial 步都没有 → 逐位等于 baseline
  G5  消融非空: start=0 与 start=5 的 logits 确实不同
  G6  保护: CFG 残差窗口 (HALTON_CFG_DELTA_START) 比 partial 门开得晚要报错
  G7  无缓存 miss (kv / restore / head / forced_nonuniform 全 0), 输出有限

用法:  python test_step_gate_start.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_lazy_token_cache import (env, make_trainer, run, assert_bitwise,
                                   check, RESULTS, DEVICE)

STEPS = 32
GATE_END = 31          # 右端, 与采样器里的常数一致


def expected_partial_steps(start, n, steps=STEPS, gate_end=GATE_END):
    """按门的定义算出哪些 step 应该是 partial 步 —— 与采样器实现无关的参照。"""
    out, counter = [], 0
    for i in range(steps):
        if start <= i < gate_end:
            is_refresh = (n >= 1 and counter % n == 0)
            counter += 1
            if not is_refresh:
                out.append(i)
    return out


def observed_partial_steps(trainer, envs, steps=STEPS, **run_kw):
    """跑一次采样, 记录每个 step 传给 vit 的 active_mask 是不是 None。"""
    seen = []

    def pre_hook(module, args, kwargs):
        seen.append(kwargs.get("active_mask") is not None)
        return None

    h = trainer.vit.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with env(**envs):
            out = run(trainer, steps=steps, **run_kw)
            st = trainer.vit.get_lazy_stats()
    finally:
        h.remove()
    assert len(seen) == steps, f"vit 被调用 {len(seen)} 次, 期望 {steps} 次"
    return [i for i, is_partial in enumerate(seen) if is_partial], st, out


LAZY_N2 = dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_CACHE_REFRESH_N=2)


def g1_default_is_five():
    """不设 env → 门仍是 5..30, 且与显式 start=5 逐位一致。"""
    trainer = make_trainer(seed=41)
    got, st, out_default = observed_partial_steps(trainer, dict(LAZY_N2),
                                                  nb_sample=2, cfg_w=1.5)
    exp = expected_partial_steps(5, 2)
    assert got == exp, f"默认门 = {got}, 期望 {exp}"

    _, _, out_explicit = observed_partial_steps(
        trainer, dict(LAZY_N2, HALTON_PARTIAL_START_STEP=5), nb_sample=2, cfg_w=1.5)
    assert_bitwise(out_default[2], out_explicit[2], "logits (默认 vs 显式 start=5)")
    return f"默认 partial 步 = {got} ({len(got)} 个), 与显式 start=5 逐位一致"


def g2_gate_matches_definition():
    """start=0..5 每一档的 partial 步集合都要与门的定义吻合。"""
    trainer = make_trainer(seed=43)
    rows = []
    for start in range(6):
        got, st, _ = observed_partial_steps(
            trainer, dict(LAZY_N2, HALTON_PARTIAL_START_STEP=start),
            nb_sample=2, cfg_w=1.5)
        exp = expected_partial_steps(start, 2)
        assert got == exp, f"start={start}: partial 步 = {got}, 期望 {exp}"
        assert st["partial_calls"] == len(exp), \
            f"start={start}: partial_calls={st['partial_calls']} != {len(exp)}"
        rows.append(f"start={start}:{len(exp)}")
    return "partial 步数 " + " ".join(rows)


def g3_monotone_and_last_step_full():
    """start 越小 partial 步越多; index >= 31 永远全量。"""
    trainer = make_trainer(seed=47)
    counts = {}
    for start in range(6):
        got, _, _ = observed_partial_steps(
            trainer, dict(LAZY_N2, HALTON_PARTIAL_START_STEP=start),
            nb_sample=2, cfg_w=1.5)
        assert all(i < GATE_END for i in got), f"start={start}: 门右端漏了 {got}"
        counts[start] = len(got)
    order = [counts[s] for s in range(6)]
    assert all(a >= b for a, b in zip(order, order[1:])), counts
    assert counts[0] > counts[5], counts
    return f"partial 步数 {order} (start=0..5), 单调不增, 且全部 < {GATE_END}"


def g4_gate_closed_equals_baseline():
    """start 大到门关死 → 没有 partial 步 → 逐位等于 baseline。"""
    trainer = make_trainer(seed=53)
    with env():
        base = run(trainer, steps=STEPS, nb_sample=2, cfg_w=1.5)
    got, st, lazy = observed_partial_steps(
        trainer, dict(LAZY_N2, HALTON_PARTIAL_START_STEP=STEPS),
        nb_sample=2, cfg_w=1.5)
    assert got == [], f"门应该关死, 却有 partial 步 {got}"
    assert st["partial_calls"] == 0 and st["full_calls"] == STEPS, st
    assert_bitwise(base[2], lazy[2], "logits")
    return f"start={STEPS}: partial 步 0 个, full_calls={st['full_calls']}, 逐位等于 baseline"


def g5_ablation_changes_result():
    """消融非空: start=0 与 start=5 的结果必须不同。"""
    trainer = make_trainer(seed=59)
    _, _, a = observed_partial_steps(
        trainer, dict(LAZY_N2, HALTON_PARTIAL_START_STEP=5), nb_sample=2, cfg_w=1.5)
    _, _, b = observed_partial_steps(
        trainer, dict(LAZY_N2, HALTON_PARTIAL_START_STEP=0), nb_sample=2, cfg_w=1.5)
    diff = max((u.float() - v.float()).abs().max().item() for u, v in zip(a[2], b[2]))
    assert diff > 0, "start=0 与 start=5 的 logits 完全相同 —— 开关没生效"
    return f"logits max|Δ| = {diff:.3e}"


def g6_cfg_delta_window_guard():
    """CFG 残差窗口比 partial 门开得晚 → 必须报错而不是静默退化。"""
    trainer = make_trainer(seed=61)
    bad = dict(LAZY_N2, HALTON_PARTIAL_START_STEP=0,
               HALTON_CFG_DELTA=1, HALTON_CFG_DELTA_N=2, HALTON_CFG_DELTA_START=5)
    raised = ""
    try:
        with env(**bad):
            run(trainer, steps=STEPS, nb_sample=2, cfg_w=1.5)
    except RuntimeError as e:
        raised = str(e)
    assert "HALTON_CFG_DELTA_START" in raised, f"没有拦住不合法组合 (raised={raised!r})"

    # 反证: 两端对齐时必须能跑通
    ok = dict(LAZY_N2, HALTON_PARTIAL_START_STEP=0,
              HALTON_CFG_DELTA=1, HALTON_CFG_DELTA_N=2, HALTON_CFG_DELTA_START=0)
    with env(**ok):
        x, _, _, _ = run(trainer, steps=STEPS, nb_sample=2, cfg_w=1.5)
        st = trainer.vit.get_lazy_stats()
    assert torch.isfinite(x).all(), "对齐后仍然跑不通"
    assert st["cfg_delta_miss"] == 0 and st["kv_miss"] == 0 and st["restore_miss"] == 0, st
    return "不合法组合被拦截; START 对齐 (0/0) 时可跑通且 miss 全 0"


def g7_no_cache_miss():
    """每一档都不能有缓存 miss, 输出不能有 NaN/Inf。"""
    trainer = make_trainer(seed=67)
    info = []
    for start in range(6):
        _, st, (x, _, _, _) = observed_partial_steps(
            trainer, dict(LAZY_N2, HALTON_PARTIAL_START_STEP=start),
            nb_sample=2, cfg_w=1.5)
        assert torch.isfinite(x).all(), f"start={start}: 输出含 NaN/Inf"
        for k in ("kv_miss", "restore_miss", "head_miss", "forced_nonuniform"):
            assert st[k] == 0, f"start={start}: {k}={st[k]}"
        info.append(round(st["partial_active_ratio"], 3))
    return f"start=0..5 全部无 miss; partial_active_ratio={info}"


if __name__ == "__main__":
    print(f"device = {DEVICE}, torch = {torch.__version__}\n")
    print("step gate 左端 (HALTON_PARTIAL_START_STEP) 测试")
    check("G1  默认仍是 start=5 (逐位一致)",              g1_default_is_five)
    check("G2  start=0..5 的 partial 步与定义吻合",        g2_gate_matches_definition)
    check("G3  partial 步数单调 & 右端 31 不受影响",       g3_monotone_and_last_step_full)
    check("G4  门关死时逐位等于 baseline",                 g4_gate_closed_equals_baseline)
    check("G5  消融非空 (start=0 != start=5)",             g5_ablation_changes_result)
    check("G6  CFG 残差窗口不对齐时报错",                  g6_cfg_delta_window_guard)
    check("G7  各档均无缓存 miss",                         g7_no_cache_miss)

    n_fail = sum(1 for ok, _, _ in RESULTS if ok is False)
    n_skip = sum(1 for ok, _, _ in RESULTS if ok is None)
    print(f"\n{len(RESULTS) - n_fail - n_skip}/{len(RESULTS)} 通过"
          + (f", {n_skip} 跳过" if n_skip else "")
          + (f", {n_fail} 失败" if n_fail else ""))
    sys.exit(1 if n_fail else 0)
