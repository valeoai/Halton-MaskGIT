# Transformer Encoder architecture
# some part have been borrowed from:
#   - NanoGPT: https://github.com/karpathy/nanoGPT
#   - DiT: https://github.com/facebookresearch/DiT

import math
import os

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange


def param_count(archi, model):
    print(f"Size of model {archi}: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Env helpers for the inference-time cache switches.
# All new behaviour is opt-in: with nothing exported the module behaves exactly
# like the original baseline.
# ---------------------------------------------------------------------------
def _env_flag(name, default="0"):
    return os.environ.get(name, default) == "1"


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _fit_cond_half(t, b):
    """把上一步存的缓存对齐到本步的 batch。

    CFG 残差缓存 (HALTON_CFG_DELTA=1) 打开后, 同一次 generation 里 batch 会在
    2B ([cond ; uncond]) 和 B (只有 cond) 之间切换。所有跨 step 的缓存 (K/V、
    restore) 都是按"上一步的 batch"存下来的, 于是 2B -> B 的那一步必须把前半
    (= cond) 切出来 —— 这与 LazyMAR 把 cache_dic 直接按 bsz 而不是 2*bsz 开、
    refresh 步只写 k[:B/2] 是同一件事, 只是改成在读侧做, 这样"整批步之后到底
    是不是 lazy 步"不需要被跨 step 地预知。
    dim0 上的切片是连续视图, 不复制。

    没开 HALTON_CFG_DELTA 时直接原样返回 —— 保证既有的 lazycache / lazyvsim
    路径逐位不受本改动影响 (batch 恒定, 本来也走不到切片分支)。
    """
    if not _env_flag("HALTON_CFG_DELTA"):
        return t
    if t is not None and t.size(0) == 2 * b:
        return t[:b]
    return t


# ---------------------------------------------------------------------------
# LazyMAR V-similarity selection (HALTON_LAZY_VSIM=1)
# ---------------------------------------------------------------------------
# 逐位抄自 LazyMAR/models/basic.py 的 RETAIN_RATIO_SCHEDULE (ICCV'25)。
# 语义 = 该 decoding step 要"重算 (retain)"的 token 占比, 随 step 单调衰减:
# 生成早期整幅图还在剧烈变化 -> 全部重算; 后期只剩局部细节 -> 5% 就够。
# LazyMAR 的 MAR 走 64 个 AR step, Halton 这里默认 32 步, 故按"生成进度"
# 重采样 (见 _lazymar_ratio)。
LAZYMAR_RETAIN_RATIO_SCHEDULE = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.6, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4,
    0.15, 0.15, 0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.12, 0.12,
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    0.05, 0.05, 0.05, 0.05,
]


def _lazymar_ratio(step, total_steps):
    """LazyMAR 的 64 步衰减表按生成进度重采样到 total_steps 步。

    idx = floor(step * 64 / total_steps) —— total_steps=32 时就是 idx = 2*step,
    即两张表在"已生成比例"这条轴上对齐。
    """
    n = len(LAZYMAR_RETAIN_RATIO_SCHEDULE)
    if step is None:
        return 1.0
    t = int(total_steps) if total_steps else n
    idx = int(int(step) * n / max(1, t))
    return LAZYMAR_RETAIN_RATIO_SCHEDULE[min(max(idx, 0), n - 1)]


def _lazy_vsim_ratio(step, total_steps):
    """本 step 的重算比例 rho_t。HALTON_LAZY_VSIM_SCHED 可覆盖:

        "lazymar" (默认)   LazyMAR 的衰减表, 按进度重采样
        "0.3"              常数比例
        "1,1,0.5,0.2,..."  逐 step 给定 (超出长度时沿用最后一个值)
    """
    sched = os.environ.get("HALTON_LAZY_VSIM_SCHED", "lazymar").strip()
    if sched in ("", "lazymar"):
        return _lazymar_ratio(step, total_steps)
    try:
        vals = [float(v) for v in sched.split(",") if v.strip() != ""]
    except ValueError:
        return _lazymar_ratio(step, total_steps)
    if not vals:
        return _lazymar_ratio(step, total_steps)
    if step is None:
        return vals[0]
    return vals[min(int(step), len(vals) - 1)]


class FeedForward(nn.Module):
    def __init__(self, dim, h_dim, multiple_of=256, bias=False, dropout=0.):
        super().__init__()
        self.dropout = dropout
        # swinGLU
        h_dim = int(2 * h_dim / 3)
        # make sure it is a power of 256
        h_dim = multiple_of * ((h_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, h_dim, bias=bias)
        self.w2 = nn.Linear(h_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, h_dim, bias=bias)

    def forward(self, x):
        # SwiGLU activation
        x = F.silu(self.w1(x)) * self.w3(x)
        if self.dropout > 0. and self.training:
            x = F.dropout(x, self.dropout)
        return self.w2(x)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim, linear=False, bias=False)
        self.key_norm = RMSNorm(dim, linear=False, bias=False)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., use_flash=True, bias=False):
        super().__init__()
        self.flash = use_flash
        self.n_local_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)
        self.qk_norm = QKNorm(num_heads * self.head_dim)
        self.cache = None
        # 上一步本 attn 算出的完整 (pre-gate) 注意力输出 delta, 形状 (b, h_w, d)。
        # 仅在 HALTON_ATTN_CACHE=1 时使用: partial 步里 inactive 位置用它代替 0
        # (与 Block.cached_ffn_delta 对称)。每个 full 步刷新, 跨 generation 由
        # Block.clear_ffn_cache 一并清空。
        self.cached_attn_delta = None
        # ── LazyMAR Token Cache (HALTON_LAZY_CACHE=1) ─────────────────────
        # 本层上一次"新鲜计算"得到的 K / V, 形状 (b, heads, seq, head_dim)。
        # 对应 LazyMAR 的 cache_dic['cache'][layer]['k'/'v']: partial step 里只
        # 为 active token 重算 K/V 并 scatter 进这两个 buffer, attention 仍然对
        # 全长 K/V 做 —— 双向 self-attention 的上下文因此保持完整。
        self.cached_k = None
        self.cached_v = None
        # 缓存不可用 (首次 / 形状-dtype 不匹配) 的次数, 供测试断言为 0。
        self.lazy_kv_miss = 0

    def forward(self, x, mask=None, active_mask=None):
        """
        active_mask: (b, seq_len) bool — when provided, Q is computed only for
        active (newly-released) positions; K/V use all positions.
        Inactive positions receive a zero attention delta so the residual stream
        is not updated via attention.  active_mask must have the same True-count
        in every row (guaranteed by HaltonSampler's uniform step schedule).
        """
        b, h_w, _ = x.shape

        if active_mask is None:
            # ── Full update (original behaviour) ──────────────────────────
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            # normalize queries and keys
            xq, xk = self.qk_norm(xq, xk, xv)
            xq = xq.view(b, h_w, self.n_local_heads, self.head_dim)
            xk = xk.view(b, h_w, self.n_local_heads, self.head_dim)
            xv = xv.view(b, h_w, self.n_local_heads, self.head_dim)

            # make heads be a batch dim
            xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
            # attention
            if self.flash:
                if mask is not None:
                    mask = mask.view(b, 1, 1, h_w)
                output = F.scaled_dot_product_attention(
                    xq, xk, xv, mask,
                    dropout_p=self.dropout if self.training else 0.
                )
            else:
                scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
                if mask is not None:
                    scores = scores + mask  # (bs, heads, seqlen, cache_len + seqlen)
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
            # concatenate all the heads
            output = output.transpose(1, 2).contiguous().view(b, h_w, -1)
            # output projection
            proj = self.wo(output)
            if self.dropout > 0. and self.training:
                proj = F.dropout(proj, self.dropout)
            # 刷新 attn-delta 缓存: 让随后的 partial 步 / refresh 步在 inactive
            # 位置有一份全 token 的基线可回退 (对称于 cached_ffn_delta)。
            if os.environ.get("HALTON_ATTN_CACHE", "0") == "1":
                self.cached_attn_delta = proj.detach()
            return proj

        else:
            # ── Q-only-active: Q from U_t tokens, K/V from all tokens ─────
            # active_mask: (b, h_w) bool, uniform True-count across rows
            n_active = int(active_mask[0].sum().item())
            x_active = x[active_mask].view(b, n_active, -1)   # (b, n_active, d)

            xq = self.wq(x_active)   # (b, n_active, d)
            xk = self.wk(x)          # (b, h_w,      d)
            xv = self.wv(x)          # (b, h_w,      d)

            # QK norm applied independently — different seq lengths are fine
            xq = self.qk_norm.query_norm(xq).to(xv)
            xk = self.qk_norm.key_norm(xk).to(xv)

            xq = xq.view(b, n_active, self.n_local_heads, self.head_dim).transpose(1, 2)
            xk = xk.view(b, h_w,      self.n_local_heads, self.head_dim).transpose(1, 2)
            xv = xv.view(b, h_w,      self.n_local_heads, self.head_dim).transpose(1, 2)

            # Cross-length attention: Q(n_active) attends to KV(h_w)
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.
            )  # (b, heads, n_active, head_dim)

            output = output.transpose(1, 2).contiguous().view(b, n_active, -1)
            proj = self.wo(output)   # (b, n_active, d)
            if self.dropout > 0. and self.training:
                proj = F.dropout(proj, self.dropout)

            # Inactive 位置的填充: (HALTON_ATTN_CACHE 方案)
            #   有可用缓存 → 取上一步本层的完整 attn delta (与 cached_ffn_delta 对称);
            #   否则 → 退化为 0 (即原始行为, attn 不更新 inactive 残差)。
            if (
                self.cached_attn_delta is not None
                and self.cached_attn_delta.shape == (b, h_w, proj.shape[-1])
                and self.cached_attn_delta.dtype == x.dtype
            ):
                out_full = self.cached_attn_delta.clone()
            else:
                out_full = torch.zeros(b, h_w, proj.shape[-1], device=x.device, dtype=x.dtype)
            out_full[active_mask] = proj.reshape(b * n_active, -1)
            # 存回合并后的 delta, 供下一步 inactive 位置复用。
            self.cached_attn_delta = out_full.detach()
            return out_full

    def forward_active(self, x, active_mask, mask=None):
        """Layer-output-cache 方案专用的 attention。

        与上面 forward 的 active 分支不同:
          - Q 只对 active token 计算, K/V 对全部 token 计算 (inactive 的 K/V 来自
            上一 step 缓存的本层输入, 已经承载在 x 里);
          - 只返回 active token 的 attention 输出 (b, n_active, d), 不做 inactive
            位置的填充, 也不触碰 cached_attn_delta (那是 HALTON_ATTN_CACHE 方案)。
        active_mask: (b, h_w) bool, 每行 True 数量一致。
        """
        b, h_w, _ = x.shape
        n_active = int(active_mask[0].sum().item())
        x_active = x[active_mask].view(b, n_active, -1)   # (b, n_active, d)

        xq = self.wq(x_active)   # (b, n_active, d)
        xk = self.wk(x)          # (b, h_w,      d)
        xv = self.wv(x)          # (b, h_w,      d)

        xq = self.qk_norm.query_norm(xq).to(xv)
        xk = self.qk_norm.key_norm(xk).to(xv)

        xq = xq.view(b, n_active, self.n_local_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(b, h_w,      self.n_local_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(b, h_w,      self.n_local_heads, self.head_dim).transpose(1, 2)

        attn_mask = mask.view(b, 1, 1, h_w) if mask is not None else None
        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask,
            dropout_p=self.dropout if self.training else 0.
        )  # (b, heads, n_active, head_dim)

        output = output.transpose(1, 2).contiguous().view(b, n_active, -1)
        proj = self.wo(output)   # (b, n_active, d)
        if self.dropout > 0. and self.training:
            proj = F.dropout(proj, self.dropout)
        return proj

    # ------------------------------------------------------------------
    # LazyMAR Token Cache primitives
    # ------------------------------------------------------------------
    def lazy_kv_full(self, h):
        """为全部 token 计算 K/V 并整体刷新缓存。

        用于 lazy 区间的入口层 —— 该层的输入来自"全量更新"的下层, 所有 token 都
        变了, 缓存里的 K/V 对谁都不再有效, 必须全量重算。

        h -> (b, N, d): 已经过 ln1 + modulate 的输入。
        返回 (xk, xv), 形状均为 (b, heads, N, head_dim)。
        """
        b, n, _ = h.shape
        xk = self.qk_norm.key_norm(self.wk(h))
        xv = self.wv(h)
        xk = xk.to(xv)
        xk = xk.view(b, n, self.n_local_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(b, n, self.n_local_heads, self.head_dim).transpose(1, 2)
        self.cached_k, self.cached_v = xk, xv
        return xk, xv

    def lazy_kv_partial(self, h_act, active_idx, seq_len):
        """只为 active token 计算 K/V, scatter 进全长缓存, 返回全长 K/V。

        对 inactive token 而言, 本层的输入自上次重算以来没有变过 (它的残差流被
        冻结在缓存里), 所以缓存中的 K/V 就是"它此刻应有的 K/V"在该近似下的定义
        值 —— 这正是 LazyMAR 的 masked_scatter_ 语义。

        h_act      -> (b, k, d)  已 ln1+modulate 的 active 行
        active_idx -> (b, k)     升序位置下标
        返回 (b, heads, seq_len, head_dim) 的全长 K/V。
        """
        b, k, _ = h_act.shape
        xk = self.qk_norm.key_norm(self.wk(h_act))
        xv = self.wv(h_act)
        xk = xk.to(xv)
        xk = xk.view(b, k, self.n_local_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(b, k, self.n_local_heads, self.head_dim).transpose(1, 2)

        if k == seq_len:
            # 全量步: 等价于整体刷新, 且与 baseline 逐位一致。
            self.cached_k, self.cached_v = xk, xv
            return xk, xv

        shape = (b, self.n_local_heads, seq_len, self.head_dim)
        for name, ref in (("cached_k", xk), ("cached_v", xv)):
            # CFG 残差缓存: 上一步是整批 [cond ; uncond], 本步只跑 cond 半个
            # batch —— 取前半即可, 不是近似。
            cur = _fit_cond_half(getattr(self, name), b)
            setattr(self, name, cur)
            if (cur is None or tuple(cur.shape) != shape
                    or cur.dtype != ref.dtype or cur.device != ref.device):
                setattr(self, name, torch.zeros(shape, dtype=ref.dtype, device=ref.device))
                self.lazy_kv_miss += 1

        idx = active_idx[:, None, :, None].expand(b, self.n_local_heads, k, self.head_dim)
        self.cached_k.scatter_(2, idx, xk)
        self.cached_v.scatter_(2, idx, xv)
        return self.cached_k, self.cached_v

    def lazy_attend(self, h_act, xk, xv, mask=None):
        """Q 只取 active token, K/V 用全长 —— active token 的注意力上下文完整。

        h_act -> (b, k, d) 已 ln1+modulate 的 active 行
        xk/xv -> (b, heads, N, head_dim)
        返回 (b, k, d) 的 attention 输出 (已过 wo)。
        """
        b, k, _ = h_act.shape
        xq = self.qk_norm.query_norm(self.wq(h_act)).to(xv)
        xq = xq.view(b, k, self.n_local_heads, self.head_dim).transpose(1, 2)

        attn_mask = mask.view(b, 1, 1, xk.size(2)) if mask is not None else None
        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask,
            dropout_p=self.dropout if self.training else 0.
        )
        output = output.transpose(1, 2).contiguous().view(b, k, -1)
        proj = self.wo(output)
        if self.dropout > 0. and self.training:
            proj = F.dropout(proj, self.dropout)
        return proj


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, linear=True, bias=True):
        super().__init__()
        self.eps = eps
        self.linear = linear
        self.add_bias = bias
        if self.linear:
            self.weight = nn.Parameter(torch.ones(dim))
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.linear:
            output = self.weight * output
        if self.add_bias:
            output = output + self.bias
        return output


class AdaNorm(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.norm_final = RMSNorm(x_dim, linear=True, bias=True, eps=1e-5)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(y_dim, x_dim * 2))

    def forward(self, x, y):
        shift, scale = self.mlp(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = Attention(dim, heads, dropout=dropout)
        self.ln2 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        # 上一步本 layer 计算出的 (gated) FFN delta, 用于在 inactive 位置上代替 0。
        # 形状 (b, h_w, d); shape mismatch 时自动重置。
        self.cached_ffn_delta = None
        # 上一步本 layer 的完整输出 (residual stream), 形状 (b, h_w, d)。
        # 仅在 HALTON_LAYER_CACHE=1 时使用: partial 步里 inactive 位置直接沿用它,
        # active 位置用本步重算的输出覆盖 (与 cached_ffn_delta 的 delta 缓存不同,
        # 这里缓存的是整层输出本身)。
        self.cached_layer_output = None

    def clear_ffn_cache(self):
        """采样新一轮生成前调用, 避免跨 generation 串台。
        同时清空 attention-delta 缓存 (HALTON_ATTN_CACHE 模式) 与 layer-output
        缓存 (HALTON_LAYER_CACHE 模式): 采样器只调用 clear_ffn_cache 这一个钩子,
        故三类缓存都在这里一起清。"""
        self.cached_ffn_delta = None
        self.attn.cached_attn_delta = None
        self.cached_layer_output = None
        # LazyMAR Token Cache 的 per-layer K/V
        self.attn.cached_k = None
        self.attn.cached_v = None
        self.attn.lazy_kv_miss = 0

    def forward(self, x, cond, mask=None, active_mask=None):
        """
        active_mask: (b, seq_len) bool —
        当前配置:
          - Attention: 当 active_mask 不为 None 时进入 active-only 模式
                       (Q 仅取 active 位置, K/V 取全部 token; 输出仅写回 active 位置)。
          - FFN: 当 active_mask 不为 None 时只对 active 位置算 FFN;
                  inactive 位置不算 FFN, 改为加上上一步同层缓存的 FFN delta
                  (无缓存时退化为 0)。每次 forward 都把本步的完整 (gated)
                  delta 存回 self.cached_ffn_delta 供下一步使用。
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)

        # ── Layer-output-cache 方案 (HALTON_LAYER_CACHE=1) ──────────────────
        # 与 ffn/attn cache 互斥: 开启后 partial 步走这里, 上面两类 cache 不参与。
        # 机制: 缓存整层输出, 下一 step 只重算 active token
        #   (Q 只取 active, K/V 取全部 —— inactive 的 K/V 来自 x 里承载的上一步
        #    本层输入), FFN 也只算 active; inactive 位置直接沿用缓存的整层输出,
        #   最后把缓存里 active 位置更新为本步新算的输出。
        if active_mask is not None and os.environ.get("HALTON_LAYER_CACHE", "0") == "1":
            return self._forward_layer_cache(
                x, active_mask, mask,
                gamma1, beta1, alpha1, gamma2, beta2, alpha2,
            )

        # Attention: 仅在 HALTON_ATTN_CACHE=1 时进入 active-only + cached-delta 模式
        # (active token 的 Q 对全 token K/V 做 attention, inactive 用上一步缓存);
        # 默认 (开关关闭) 保持 full update —— baseline 完全不变。
        attn_active = (
            active_mask
            if (active_mask is not None
                and os.environ.get("HALTON_ATTN_CACHE", "0") == "1")
            else None
        )
        x = x + alpha1.unsqueeze(1) * self.attn(
            modulate(self.ln1(x), gamma1, beta1),
            mask=mask,
            active_mask=attn_active,
        )
        # FFN: active-only when active_mask is provided, with cached-delta fill-in
        if active_mask is None:
            # full-token FFN; 同时刷新缓存
            ff_delta = alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
            x = x + ff_delta
            self.cached_ffn_delta = ff_delta.detach()
        else:
            b, h_w, d = x.shape
            n_active = int(active_mask[0].sum().item())
            x_active = x[active_mask].view(b, n_active, d)
            ff_out = self.ff(modulate(self.ln2(x_active), gamma2, beta2))  # (b, n_active, d)
            active_delta = (alpha2.unsqueeze(1) * ff_out)                  # (b, n_active, d)

            # 起点: inactive 位置取上一步缓存; 没有缓存或形状不匹配则用 0
            if (
                self.cached_ffn_delta is not None
                and self.cached_ffn_delta.shape == x.shape
                and self.cached_ffn_delta.dtype == x.dtype
            ):
                delta = self.cached_ffn_delta.clone()
            else:
                delta = torch.zeros_like(x)
            # 覆盖 active 位置为本步新算的 delta
            delta[active_mask] = active_delta.reshape(b * n_active, d)

            x = x + delta
            self.cached_ffn_delta = delta.detach()

        # full-update 步 (active_mask is None) 顺带刷新 layer-output 缓存, 让随后的
        # partial 步 / refresh 步在 inactive 位置有一份全 token 的整层输出可沿用。
        if active_mask is None and os.environ.get("HALTON_LAYER_CACHE", "0") == "1":
            self.cached_layer_output = x.detach()
        return x

    def _forward_layer_cache(self, x, active_mask, mask,
                             gamma1, beta1, alpha1, gamma2, beta2, alpha2):
        """HALTON_LAYER_CACHE 方案的 partial forward。

        输入 x: active 位置承载本步重算的上一层输出, inactive 位置承载上一步缓存
        的上一层输出 (由上层 _forward_layer_cache 构造)。K/V 用整个 x, 因此 inactive
        的 K/V 天然来自缓存。
        """
        b, h_w, d = x.shape
        n_active = int(active_mask[0].sum().item())

        # ── Attention: Q 仅 active, K/V 全部; 只拿 active 的注意力输出 ──
        h_norm = modulate(self.ln1(x), gamma1, beta1)                  # (b, h_w, d) 供 K/V
        attn_active = self.attn.forward_active(h_norm, active_mask, mask)  # (b, n_active, d)

        # active 位置做残差; inactive 位置不动 (稍后整块用缓存覆盖)
        x_active = x[active_mask].view(b, n_active, d)                 # (b, n_active, d)
        x_active = x_active + alpha1.unsqueeze(1) * attn_active

        # ── FFN 只算 active token ──
        ff_out = self.ff(modulate(self.ln2(x_active), gamma2, beta2))  # (b, n_active, d)
        x_active = x_active + alpha2.unsqueeze(1) * ff_out             # 本步本层的新输出

        # ── 拼整层输出: inactive 用缓存, active 用新算的; 再写回缓存 ──
        if (
            self.cached_layer_output is not None
            and self.cached_layer_output.shape == x.shape
            and self.cached_layer_output.dtype == x.dtype
        ):
            out = self.cached_layer_output.clone()
        else:
            # 首个 partial 步还没缓存: inactive 退化为沿用当前输入 x (即上一层此步的
            # 输出), 相当于本层对 inactive 不更新。
            out = x.clone()
        out[active_mask] = x_active.reshape(b * n_active, d)
        self.cached_layer_output = out.detach()
        return out

    # ------------------------------------------------------------------
    # LazyMAR Token Cache
    # ------------------------------------------------------------------
    def forward_lazy(self, x, cond, active_idx, seq_len, mask=None, select_fn=None,
                     gather_x=False):
        """LazyMAR 风格的 block forward: 残差流只保留 active token。

        三种模式:
          active_idx is None  (V-打分模式下 lazy 区间的入口层)
              x 是全长 (b, N, C)。K/V 为全部 token 新鲜计算并整体刷新缓存,
              然后调用 select_fn(xv) 决定本步的 active 集合 —— 对应 LazyMAR 在
              decoder layer 3 用 V 的余弦相似度做 _prune_tokens。
          active_idx 已给定 + gather_x=True  (forced-only 模式的入口层)
              active 集合在进任何层之前就定死了 (= U_t ∪ U_{t-1} ∪ register),
              不需要打分, 于是入口层也不必算全长 K/V: 这里先把 active 行从全长
              x 里 gather 出来, 之后与区间内其它层完全一样。
          active_idx 已给定 + gather_x=False  (区间内的后续层)
              x 已经是 (b, k, C) 的 active 行。K/V 只为 active 行重算并 scatter
              进全长缓存, attention 仍对全长 K/V 做。

        返回 (x_act, active_idx), x_act 形状 (b, k, C)。
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)

        if active_idx is None:
            assert select_fn is not None, "entry layer needs a select_fn"
            h = modulate(self.ln1(x), gamma1, beta1)            # (b, N, C)
            xk, xv = self.attn.lazy_kv_full(h)                  # (b, H, N, dh)
            active_idx = select_fn(xv)                          # (b, k)
            idx_c = active_idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
            h_act = torch.gather(h, 1, idx_c)                   # (b, k, C)
            x_act = torch.gather(x, 1, idx_c)                   # (b, k, C)
        else:
            if gather_x:
                idx_c = active_idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
                x = torch.gather(x, 1, idx_c)                   # (b, N, C) -> (b, k, C)
            x_act = x                                           # (b, k, C)
            h_act = modulate(self.ln1(x_act), gamma1, beta1)
            xk, xv = self.attn.lazy_kv_partial(h_act, active_idx, seq_len)

        x_act = x_act + alpha1.unsqueeze(1) * self.attn.lazy_attend(h_act, xk, xv, mask=mask)
        x_act = x_act + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x_act), gamma2, beta2))
        return x_act, active_idx


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim, heads, mlp_dim, dropout=dropout))
        # ── LazyMAR Token Cache 的 encoder 级状态 ──────────────────────────
        # lazy 区间出口层的完整输出 (b, N, C): partial 步结束时 inactive 行从这里
        # 取回上一步的值, active 行写入本步新算的值 (对应 LazyMAR 的
        # _unprune_tokens, 但用缓存回填而不是填 0)。
        self.lazy_restore_cache = None
        # 本步 lazy partial 前向选中的 active 下标 (b, k), 仅 partial 步非 None。
        # Transformer.forward 的 head 靠它决定只算哪些行 —— 放在这里而不是重新
        # 由 active_mask 推一遍, 是因为打分模式 (r<1) 的 active 集合是在 encoder
        # 内部选出来的, 外面看不到。
        self.lazy_last_active_idx = None
        self.lazy_stats = self._new_lazy_stats()

    @staticmethod
    def _new_lazy_stats():
        return {"lazy_calls": 0, "full_calls": 0, "partial_calls": 0,
                "active_sum": 0, "token_sum": 0,
                "partial_active_sum": 0, "partial_token_sum": 0,
                "score_miss": 0, "restore_miss": 0, "forced_nonuniform": 0,
                "scored_calls": 0, "forced_only_calls": 0, "vsim_calls": 0,
                "head_partial_calls": 0, "head_full_calls": 0, "head_miss": 0,
                "cfg_delta_full_calls": 0, "cfg_delta_reuse_calls": 0,
                "cfg_delta_miss": 0}

    def clear_ffn_cache(self):
        for blk in self.layers:
            blk.clear_ffn_cache()
        self.lazy_restore_cache = None
        self.lazy_last_active_idx = None
        self.lazy_stats = self._new_lazy_stats()

    def get_lazy_stats(self):
        """返回本次 generation 的 LazyMAR Token Cache 统计 (供测试 / 消融读取)。"""
        stats = dict(self.lazy_stats)
        stats["kv_miss"] = sum(int(blk.attn.lazy_kv_miss) for blk in self.layers)
        n = stats["token_sum"]
        # 全部 step 平均 (含被 gate 排除的全量步) —— 反映整体算力占比
        stats["mean_active_ratio"] = (stats["active_sum"] / n) if n else float("nan")
        # 只统计 partial 步 —— 直接对应 HALTON_LAZY_CACHE_RATIO 的预算
        pn = stats["partial_token_sum"]
        stats["partial_active_ratio"] = (stats["partial_active_sum"] / pn) if pn else float("nan")
        return stats

    def forward(self, x, cond, mask=None, active_mask=None,
                partial_update_start_layer=3, partial_update_end_layer=21,
                cfg_pair=False, step=None, total_steps=None):
        self.lazy_last_active_idx = None    # 只对本次 forward 有效, 先清掉

        # Layer-level gating (from analyze_ffn_delta_stability):
        #   exclude layer 0 (no prior context) and very top layers — use
        #   partial_update only in the stable mid-stack: start ≤ i ≤ end.
        # Env overrides let small/base runs preserve the "top-K layers full"
        # tail without touching the large defaults.
        _s = os.environ.get("HALTON_PARTIAL_START_LAYER")
        if _s:
            try:
                partial_update_start_layer = int(_s)
            except ValueError:
                pass
        _e = os.environ.get("HALTON_PARTIAL_END_LAYER")
        if _e:
            try:
                partial_update_end_layer = int(_e)
            except ValueError:
                pass

        if _env_flag("HALTON_LAZY_VSIM") and not _env_flag("HALTON_LAZY_CACHE"):
            raise RuntimeError(
                "HALTON_LAZY_VSIM 是 LazyMAR Token Cache 的一个选点模式, "
                "必须同时设置 HALTON_LAZY_CACHE=1。"
            )

        # ── LazyMAR Token Cache (HALTON_LAZY_CACHE=1) ─────────────────────
        # 与既有的 ffn/attn/layer cache 三个方案互斥, 便于做公平对比。
        if _env_flag("HALTON_LAZY_CACHE"):
            if _env_flag("HALTON_LAYER_CACHE") or _env_flag("HALTON_ATTN_CACHE"):
                raise RuntimeError(
                    "HALTON_LAZY_CACHE 与 HALTON_LAYER_CACHE / HALTON_ATTN_CACHE 互斥, "
                    "请只开启其中一个方案。"
                )
            # lazy 区间默认覆盖 *全部* 层: forced-only 模式不再需要靠底层几层去
            # 算 V 的相似度, 而 layer 0 的输入对 inactive token 而言逐位没变
            # (code 只在 U_{t-1} 处被改写, 已在强制集合里), 缓存的 K/V 精确有效。
            # 顺序: HALTON_LAZY_*_LAYER > HALTON_PARTIAL_*_LAYER > 全部层。
            l_start = _env_int("HALTON_LAZY_START_LAYER",
                               _env_int("HALTON_PARTIAL_START_LAYER", 0))
            l_end = _env_int("HALTON_LAZY_END_LAYER",
                             _env_int("HALTON_PARTIAL_END_LAYER", len(self.layers) - 1))
            if _env_flag("HALTON_LAZY_VSIM") and l_start < 1:
                raise RuntimeError(
                    "HALTON_LAZY_VSIM 需要 HALTON_LAZY_START_LAYER >= 1: 打分要用入口层"
                    "相对上一步的 V 变化量, 而 layer 0 的输入只在 U_{t-1} 处变过, "
                    "在那里打分等于退化回调度选点。LazyMAR 用的是 decoder layer 3。"
                )
            return self._forward_lazy(
                x, cond, mask, active_mask, l_start, l_end, cfg_pair,
                step=step, total_steps=total_steps,
            )

        for i, block in enumerate(self.layers):
            use_partial = partial_update_start_layer <= i <= partial_update_end_layer #use_partial为True时，表示在第3到第21层之间使用partial_update，即FFN只更新active_mask指定的位置；否则在其他层使用full update，即FFN更新所有位置。
            x = block(x, cond, mask=mask, active_mask=active_mask if use_partial else None) #active_maskはTransformerEncoderの引数で、Blockのforwardに渡される。use_partialがTrueのとき、active_maskがBlockのforwardに渡され、FFNはactive_maskで指定された位置のみを更新する。use_partialがFalseのとき、active_maskはNoneとしてBlockのforwardに渡され、FFNは全ての位置を更新する。
        return x

    # ------------------------------------------------------------------
    # LazyMAR Token Cache
    # ------------------------------------------------------------------
    def _lazy_budget(self, seq_len, forced_mask):
        """本步要重算的 token 数 k。

        HALTON_LAZY_CACHE_RATIO = 被"缓存复用"的 token 比例 r ∈ [0, 1]:
            k = ceil((1 - r) * N)
        r = 1 (默认) → k = 0 → 预算被强制集合撑到 n_forced, 即"只重算当前步和
            上一步解码的 token (+register)", 不需要任何 V 相似度打分。
        r = 0  → k = N → 全部重算 → 与 baseline 逐位一致 (方案关闭时的语义)。
        0 < r < 1 → 强制集合之外还有名额, 由 _lazy_select 的 V 余弦打分来填。

        forced_mask is None 表示这是一个全量步 (采样器没给 active_mask), 无条件
        k = N, 借此刷新所有缓存 —— 对应 LazyMAR 的 global_force_fresh。
        """
        if forced_mask is None:
            return seq_len
        ratio = _env_float("HALTON_LAZY_CACHE_RATIO", 1.0)
        ratio = min(max(ratio, 0.0), 1.0)
        k = int(math.ceil((1.0 - ratio) * seq_len))
        # 必须装得下强制重算集合 (U_t ∪ U_{t-1} ∪ register), 否则会用陈旧 logit
        # 去 commit 本步的 token。
        n_forced = int(forced_mask.sum(dim=1).max().item())
        return max(1, min(seq_len, max(k, n_forced)))

    def _lazy_select(self, xv, prev_v, forced_mask, budget, cfg_pair):
        """挑出本步要重算的 token, 返回升序下标 (b, k)。

        两条路径:
          budget == |forced|  (r = 1, 默认)
              名额刚好等于强制集合 —— 打分再怎么排也只能选出这同一批 token,
              所以完全跳过 V 余弦, 直接由 forced_mask 取下标。此时 xv 可以是
              None: 调用方连入口层的全长 K/V 都不必算。
          budget >  |forced|  (r < 1)
              判据 = 入口层 V 相对上一 decoding step 的余弦变化量 (LazyMAR 的
              做法): 变化越大越该重算。强制集合直接置 +inf 排在最前, 其余名额
              按变化量从大到小填满 —— 固定名额而非阈值, 保证每行 active 数一致。
        """
        if xv is not None:
            b, _, n, _ = xv.shape
            device = xv.device
        else:
            b, n = forced_mask.shape
            device = forced_mask.device

        def _all():
            return torch.arange(n, device=device).unsqueeze(0).expand(b, n)

        if forced_mask is None:
            # 全量步: 不需要打分, 直接全选。
            return _all()

        # ── forced-only: 名额被强制集合占满, 无需打分 ────────────────────
        counts = forced_mask.sum(dim=1)
        n_forced = int(counts.max().item())
        if budget <= n_forced:
            if int(counts.min().item()) != n_forced:
                # 每行强制数不一致 → 没法拼成规整的 (b, k)。Halton 构造下不会
                # 发生 (U_t / U_{t-1} 的大小与行无关), 计数并退化成全量重算。
                self.lazy_stats["forced_nonuniform"] += 1
                return _all()
            # nonzero 按行优先、列升序返回 → 每行天然是升序下标。
            return forced_mask.nonzero(as_tuple=False)[:, 1].view(b, n_forced)

        if prev_v is None or prev_v.shape != xv.shape or prev_v.device != xv.device:
            # 没有可比的上一步 V (generation 的第一个 lazy 步): 退化成全量重算,
            # 安全但不省算力。gate 正常时不该发生, 计数供测试断言。
            self.lazy_stats["score_miss"] += 1
            return _all()

        # fp32 打分: bf16 下近乎相同的两个 1024 维向量做余弦会丢掉判别位。
        cos = F.cosine_similarity(xv.float(), prev_v.float(), dim=-1).mean(dim=1)  # (b, N)
        score = 1.0 - cos

        if cfg_pair and b % 2 == 0:
            # batch 布局是 [cond ; uncond], 两半 token 输入完全相同, 只有类别条件
            # 不同。让两半选同一批 token, 否则 CFG 相减时会引入两套不同的近似误差。
            half = b // 2
            shared = 0.5 * (score[:half] + score[half:])
            score = torch.cat([shared, shared], dim=0)

        score = score.masked_fill(forced_mask, float("inf"))
        idx = score.topk(budget, dim=1).indices
        idx, _ = torch.sort(idx, dim=1)      # 升序: 与 boolean-index 的顺序一致
        return idx

    def _lazy_select_vsim(self, xv, prev_v, budget, cfg_pair):
        """纯 V-相似度选点 (HALTON_LAZY_VSIM=1) —— LazyMAR _prune_tokens 的复刻。

        与 _lazy_select 的区别只有一个, 但是本质的: **不存在强制集合**。
        LazyMAR 会把 mask_to_pred / prev_mask_to_pred 的分数按住不放 (score=0,
        升序排在最前) 从而无条件保留; 这里按要求把这条去掉, 名额全部由"入口层 V
        相对上一 decoding step 的余弦变化量"决定 —— 变化越大越该重算。
        于是本步要 commit 的 U_t 也可能落选而用上一步的 logit, 这正是本接口要
        测的东西。

        budget = ceil(rho_t * N), rho_t 由 _lazy_vsim_ratio 给出 (随 step 衰减)。
        固定名额而非阈值: 保证每行 active 数一致, (b, k) 才拼得出来。
        返回升序下标 (b, budget)。
        """
        b, _, n, _ = xv.shape
        if (prev_v is None or prev_v.shape != xv.shape
                or prev_v.device != xv.device or prev_v.dtype != xv.dtype):
            # 没有可比的上一步 V: 退化成全量重算, 安全但不省算力。
            self.lazy_stats["score_miss"] += 1
            return torch.arange(n, device=xv.device).unsqueeze(0).expand(b, n)

        # fp32 打分: bf16 下近乎相同的两个高维向量做余弦会丢掉判别位。
        cos = F.cosine_similarity(xv.float(), prev_v.float(), dim=-1).mean(dim=1)
        score = 1.0 - cos                                   # (b, N) 变化量

        if cfg_pair and b % 2 == 0:
            # [cond ; uncond] 两半必须选同一批 token, 否则 CFG 相减时会引入两套
            # 不同的近似误差。
            half = b // 2
            shared = 0.5 * (score[:half] + score[half:])
            score = torch.cat([shared, shared], dim=0)

        idx = score.topk(budget, dim=1).indices
        idx, _ = torch.sort(idx, dim=1)      # 升序: 与 boolean-index 的顺序一致
        return idx

    def _lazy_restore(self, x_act, active_idx, b, n, c):
        """把 active 行写回全长残差流, inactive 行取缓存中上一步的值。

        对应 LazyMAR 的 _unprune_tokens —— 区别是那里把 inactive 位置填 0 (因为
        MAR 只在 mask_to_pred 处读 z), 这里必须回填缓存, 因为 Halton 采样器会对
        全部 576 个位置采样并记入 l_codes。
        """
        cache = _fit_cond_half(self.lazy_restore_cache, b)   # 见 _fit_cond_half
        valid = (cache is not None and tuple(cache.shape) == (b, n, c)
                 and cache.dtype == x_act.dtype and cache.device == x_act.device)
        if valid:
            out = cache.clone()
        else:
            if active_idx.size(1) < n:
                self.lazy_stats["restore_miss"] += 1
            out = torch.zeros(b, n, c, dtype=x_act.dtype, device=x_act.device)
        out.scatter_(1, active_idx.unsqueeze(-1).expand(-1, -1, c), x_act)
        self.lazy_restore_cache = out.detach()
        return out

    def _forward_lazy(self, x, cond, mask, forced_mask, start, end, cfg_pair,
                      step=None, total_steps=None):
        """LazyMAR Token Cache 的完整前向。

        forced_mask is None → 全量步: k = N, 所有缓存被整体刷新, 且逐位等于
        baseline。partial 步 → 只有 [start, end] 区间的 k 个 token 走 Q/FFN,
        K/V 走全长缓存, 区间外的层照常全量。

        两条选点路径 (见 _lazy_select):
          budget >  |forced|  (r < 1)  入口层算全长 K/V, 用 V 的余弦变化量补名额;
                                       start 必须 > 0, 底下几层给它攒材料。
          budget == |forced|  (r = 1)  不打分, active 直接就是 U_t ∪ U_{t-1} ∪
                                       register; 入口层不必算全长 K/V, start 可
                                       以放到 0 —— layer 0 的输入 = tok_emb(code)
                                       + pos, 与别的 token 无关, 而 code 只在
                                       U_{t-1} 处变过 (已在强制集合里), 所以
                                       inactive 行缓存的 K/V 是精确值而非近似。
        """
        depth = len(self.layers)
        end = max(0, min(int(end), depth - 1))
        start = max(0, min(int(start), end))

        # 1) lazy 区间以下: 永远全量 (它们的输入每步都变, 没有可复用的东西)
        for i in range(start):
            x = self.layers[i](x, cond, mask=mask, active_mask=None)

        b, n, c = x.shape

        # ── V-打分模式 (HALTON_LAZY_VSIM=1): 无强制集合, 名额随 step 衰减 ──
        # forced_mask 在这个模式里只剩一个用途: 标记"这是不是一个 partial 步"
        # (采样器的 step gate 5..30 + REFRESH_N 决定), 与 lazy cache 完全一致。
        vsim = _env_flag("HALTON_LAZY_VSIM") and forced_mask is not None
        if vsim:
            rho = min(max(_lazy_vsim_ratio(step, total_steps), 0.0), 1.0)
            budget = max(1, min(n, int(math.ceil(rho * n))))
        else:
            budget = self._lazy_budget(n, forced_mask)
        n_forced = n if forced_mask is None else int(forced_mask.sum(dim=1).max().item())

        if vsim and budget < n:
            # 入口层要全长 K/V 才能算 V 的余弦变化量 (LazyMAR 在 decoder layer 3
            # 做这件事; 这里 start 同样应 >= 1, 让下面几层每步全量给它攒材料)。
            prev_v = _fit_cond_half(self.layers[start].attn.cached_v, b)  # 上一步入口层的 V
            self.lazy_stats["vsim_calls"] += 1
            self.lazy_stats["scored_calls"] += 1
            x_act, active_idx = self.layers[start].forward_lazy(
                x, cond, active_idx=None, seq_len=n, mask=mask,
                select_fn=lambda xv: self._lazy_select_vsim(xv, prev_v, budget, cfg_pair),
            )
            first = start + 1
        elif vsim:
            # rho_t = 1 (LazyMAR 表的前 20/64 步): 本步全部重算 —— 不必先算全长
            # K/V 去打分, 直接走便宜的那条路, 顺带把所有层的 K/V 缓存刷新。
            self.lazy_stats["vsim_calls"] += 1
            self.lazy_stats["forced_only_calls"] += 1
            active_idx = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)
            x_act = x
            first = start
        elif budget > n_forced:
            # ── 打分模式 (r < 1): 入口层要全长 K/V 才能算 V 的余弦变化量 ──
            prev_v = _fit_cond_half(self.layers[start].attn.cached_v, b)  # 上一步入口层的 V
            self.lazy_stats["scored_calls"] += 1
            x_act, active_idx = self.layers[start].forward_lazy(
                x, cond, active_idx=None, seq_len=n, mask=mask,
                select_fn=lambda xv: self._lazy_select(xv, prev_v, forced_mask, budget, cfg_pair),
            )
            first = start + 1
        else:
            # ── forced-only (r = 1, 默认): active 集合在进层之前就定死了 ──
            # 不打分 → 入口层不需要全长 K/V → 入口层本身也能只算 active 行,
            # 这也是 start 可以一路放到 0 的原因。
            self.lazy_stats["forced_only_calls"] += 1
            active_idx = self._lazy_select(None, None, forced_mask, budget, cfg_pair) \
                if forced_mask is not None else \
                torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)
            x_act = x
            first = start

        # 3) 区间内的层: 残差流只带 active 行, K/V 走全长缓存
        for i in range(first, end + 1):
            x_act, _ = self.layers[i].forward_lazy(
                x_act, cond, active_idx=active_idx, seq_len=n, mask=mask,
                gather_x=(i == first and first == start),
            )

        # 4) 回填全长
        x = self._lazy_restore(x_act, active_idx, b, n, c)

        # 5) lazy 区间以上: 全量
        for i in range(end + 1, depth):
            x = self.layers[i](x, cond, mask=mask, active_mask=None)

        # partial 步才把 active 下标交给 head; 全量步 head 本来就该全算。
        self.lazy_last_active_idx = active_idx if forced_mask is not None else None

        k = int(active_idx.size(1))
        self.lazy_stats["lazy_calls"] += 1
        self.lazy_stats["full_calls" if forced_mask is None else "partial_calls"] += 1
        self.lazy_stats["active_sum"] += k
        self.lazy_stats["token_sum"] += n
        if forced_mask is not None:
            self.lazy_stats["partial_active_sum"] += k
            self.lazy_stats["partial_token_sum"] += n
        return x


class Transformer(nn.Module):
    """ DiT-like transformer with adaLayerNorm with zero initializations """

    def clear_ffn_cache(self):
        """转发到 TransformerEncoder, 清空每层 Block 的 FFN delta 缓存。"""
        self.lazy_logit_cache = None
        self.cfg_delta_cache = None
        self.transformer.clear_ffn_cache()

    def get_lazy_stats(self):
        """转发到 TransformerEncoder: LazyMAR Token Cache 的运行统计。"""
        return self.transformer.get_lazy_stats()

    def __init__(self, input_size=16, hidden_dim=768, codebook_size=1024,
                 depth=12, heads=16, mlp_dim=3072, dropout=0., nclass=1000,
                 register=1, proj=1, **kwargs):
        super().__init__()

        self.nclass = nclass
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.proj = proj

        self.cls_emb = nn.Embedding(nclass + 1, hidden_dim)
        self.tok_emb = nn.Embedding(codebook_size + 1, hidden_dim)
        self.pos_emb = nn.Embedding(input_size ** 2, hidden_dim)

        if self.proj > 1:
            self.in_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, bias=False)
            self.out_proj = nn.Conv2d(
                hidden_dim, hidden_dim * 4, kernel_size=1, stride=1, padding=0, bias=False
            ).to(memory_format=torch.channels_last)

        self.transformer = TransformerEncoder(
            dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout
        )

        self.last_norm = AdaNorm(x_dim=hidden_dim, y_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, codebook_size + 1)
        # LazyMAR Token Cache: 上一次算过的全长 logit (b, h*w, codebook+1)。
        # partial 步只为 active 行重算, 其余行从这里取。
        self.lazy_logit_cache = None
        # LazyMAR CFG 残差缓存 (HALTON_CFG_DELTA=1): 上一个整批步在 transformer
        # 出口 (last_norm 之前) 记下的 uncond - cond, 形状 (b/2, N, C)。
        self.cfg_delta_cache = None
        self.head.weight = self.tok_emb.weight

        self.register = register
        if self.register > 0:
            self.reg_tokens = nn.Embedding(self.register, hidden_dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.cls_emb.weight, std=0.02)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        for block in self.transformer.layers:
            nn.init.constant_(block.mlp[1].weight, 0)
            nn.init.constant_(block.mlp[1].bias, 0)

        if self.proj > 1:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

        if self.register > 0:
            nn.init.normal_(self.reg_tokens.weight, std=0.02)

    def forward(self, x, y, drop_label, mask=None, active_mask=None, cfg_pair=False,
                step=None, total_steps=None, cfg_delta=False):
        """
        active_mask: (b, h, w) bool — newly-active token positions for
        partial-update mode (Q-only-active attention).
        Pass None for the standard full-update forward pass.
        cfg_pair:    True when the batch is laid out as [cond ; uncond] (the
        sampler's CFG call).  Only used by the LazyMAR Token Cache, so that both
        halves select the same token set.
        step / total_steps: current decoding step index and the total number of
        steps.  Only used by the V-similarity selection (HALTON_LAZY_VSIM=1),
        whose recompute ratio decays with the step (LazyMAR's
        RETAIN_RATIO_SCHEDULE).  Everything else ignores them.
        cfg_delta:   True asks for a "lazy" CFG step (HALTON_CFG_DELTA=1): only
        the conditional half of the [cond ; uncond] batch goes through the
        transformer, and the unconditional half is rebuilt from the cached
        residual.  Ignored unless HALTON_CFG_DELTA=1 and cfg_pair is True.
        """
        b, h, w = x.size()
        h0, w0 = h, w   # original spatial dims before any proj
        x = x.reshape(b, h * w)

        y = torch.where(drop_label, torch.full_like(y, self.nclass), y)
        y = self.cls_emb(y)

        pos = torch.arange(0, w * h, dtype=torch.long, device=x.device)
        pos = self.pos_emb(pos)

        x = self.tok_emb(x) + pos

        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w, b=b, c=self.hidden_dim).contiguous()
            x = self.in_proj(x)
            _, _, h, w = x.shape   # h, w updated to projected resolution
            x = rearrange(x, 'b c h proj_w -> b (h proj_w) c', proj_h=h, proj_w=w, b=b, c=self.hidden_dim).contiguous()

        # Build sequence-level active_mask aligned with current seq length.
        seq_active_mask = None
        if active_mask is not None:
            if self.proj > 1:
                # Pool (b, h0, w0) → (b, h, w) by OR over each proj×proj patch.
                # view(b, h, proj, w, proj) groups pixels by conv-patch correctly
                # because PyTorch row-major layout maps [ph*proj+sh, pw*proj+sw]
                # to indices [ph, sh, pw, sw] under this reshape.
                am = active_mask.view(b, h, self.proj, w, self.proj)
                am = am.any(2).any(3)          # (b, h, w) — active if any sub-token is
                seq_active_mask = am.view(b, h * w)
            else:
                seq_active_mask = active_mask.view(b, h0 * w0)

            if self.register > 0:
                # Register tokens are never in U_t; they always attend fully.
                # LazyMAR Token Cache 例外: register 是全局 attention sink, 冻结它
                # 的代价远大于重算 (只有 self.register 个 token), 故强制 active。
                reg_val = _env_flag("HALTON_LAZY_CACHE") and _env_flag("HALTON_LAZY_REGISTER_ACTIVE", "1")
                reg_col = torch.full((b, self.register), reg_val, dtype=torch.bool, device=x.device)
                seq_active_mask = torch.cat([seq_active_mask, reg_col], dim=1)

        if self.register > 0:
            reg = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            x = torch.cat([x, self.reg_tokens(reg).expand(b, self.register, self.hidden_dim)], dim=1)

        # ── LazyMAR CFG 残差缓存 (HALTON_CFG_DELTA=1) ─────────────────────
        # LazyMAR forward_mae_decoder 的做法: 整批步跑完整的 [cond ; uncond] 并把
        # uncond - cond 的残差存下来 (它存在 decoder 最后一层输出上); lazy 步只跑
        # cond 半个 batch, uncond 用 "cond + 上次的残差" 补出来。
        # 这里的缓存点取在 transformer 出口 / last_norm 之前, 与 LazyMAR 对应 ——
        # 于是 uncond 分支仍然用它自己的 y 过 AdaNorm 和 head, 被跳过的只有主干。
        # 切前半是精确的: 两半的 token embedding 逐位相同 (采样器传的是
        # cat([code, code])), cond/uncond 的差别只经由 y 进入 AdaLN。
        cfg_delta_on = _env_flag("HALTON_CFG_DELTA") and cfg_pair and b % 2 == 0
        if cfg_delta_on and seq_active_mask is not None and not _env_flag("HALTON_LAZY_CACHE"):
            raise RuntimeError(
                "HALTON_CFG_DELTA 只支持两种组合: 与 HALTON_LAZY_CACHE 一起用, 或者在"
                "完全不给 active_mask 的全量模式下单独用。旧的 FFN/ATTN/LAYER delta "
                "缓存在 batch 减半时会静默退化成 0-delta, 故拒绝。"
            )
        half = b // 2
        stats = self.transformer.lazy_stats
        reuse = False
        if cfg_delta_on and cfg_delta:
            cache = self.cfg_delta_cache
            # dtype 不参与判定: 缓存来自 transformer 的输出 (autocast 下可能是
            # bf16), 而此处的 x 还是 embedding 的 dtype, 两者本就可以不同。
            reuse = (cache is not None
                     and tuple(cache.shape) == (half, x.size(1), self.hidden_dim)
                     and cache.device == x.device)
            if not reuse:
                # 没有可用残差。gate 正常时只可能发生在本次 generation 的第一个
                # lazy 步之前没有任何整批步的情况下 —— 退回整批前向, 安全但不省。
                stats["cfg_delta_miss"] += 1

        if reuse:
            x_in, y_in = x[:half], y[:half]
            am_in = seq_active_mask[:half] if seq_active_mask is not None else None
            mask_in = mask[:half] if (mask is not None and mask.size(0) == b) else mask
        else:
            x_in, y_in, am_in, mask_in = x, y, seq_active_mask, mask

        x = self.transformer(x_in, y_in, mask=mask_in, active_mask=am_in,
                             # lazy 步的 batch 里只剩 cond 一半, 不再是 [cond;uncond]:
                             # 此时若还按 cfg_pair 把前后半的选点分数平均, 等于把两个
                             # 不相干的样本混在一起。
                             cfg_pair=(cfg_pair and not reuse),
                             step=step, total_steps=total_steps)

        if reuse:
            x = torch.cat([x, x + self.cfg_delta_cache.to(x.dtype)], dim=0)
            stats["cfg_delta_reuse_calls"] += 1
        elif cfg_delta_on:
            self.cfg_delta_cache = (x[half:] - x[:half]).detach()
            stats["cfg_delta_full_calls"] += 1

        x = x[:, :h * w].contiguous()   # drop register tokens

        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w, b=b, c=self.hidden_dim).contiguous()
            x = self.out_proj(x)
            x = rearrange(
                x, 'b (c s1 s2) h w -> b (h s1 w s2) c',
                s1=self.proj, s2=self.proj, b=b, h=h, w=w, c=self.hidden_dim
            ).contiguous()

        # ── LazyMAR Token Cache: head 也只算 active token ─────────────────
        # last_norm + head 都是逐 token 的 (RMSNorm 沿最后一维, shift/scale 来自
        # per-sample 的 y), 所以对行的子集算与对全部行算, 每一行的结果相同。
        # inactive 行此刻的 hidden state 本来就是缓存里上一步的值, 而 y 在整个
        # 采样循环里不变 —— 于是"拿旧 hidden 现算一遍 logit"和"直接用上一步算好
        # 的 logit"是同一个数。这一步因此不引入任何新的近似, 只是把注定重复的
        # 19.3 GFLOPs/step (large-384) 省掉。
        # 采样器也只在 U_t (⊆ active) 处 commit, inactive 行的 logit 除了写进
        # l_codes 供可视化外没有别的去处。
        act_idx = None
        if (self.proj == 1 and _env_flag("HALTON_LAZY_CACHE")
                and _env_flag("HALTON_LAZY_HEAD", "1")):
            act_idx = self.transformer.lazy_last_active_idx

        if act_idx is not None and act_idx.size(0) != b:
            # CFG 残差缓存的 lazy 步: transformer 只跑了 cond 半个 batch, 两半共用
            # 同一批 active 下标 (uncond 就是 cond 加残差补出来的)。
            act_idx = act_idx.repeat(2, 1) if act_idx.size(0) * 2 == b else None

        if act_idx is not None:
            n_img = x.size(1)
            if _env_flag("HALTON_LAZY_VSIM"):
                # V-打分模式下 active 集合是内容自适应的: 第 i 行可能选中 register
                # token 而第 j 行没有, "图像 token 数"因此逐行不同, 不能拿行 0 的
                # 前缀长度去 gather。把下标 clamp 进图像范围即可 —— 被 clamp 的条目
                # 只是对某个 inactive 图像 token 多算一次 head, 而它此刻的 hidden
                # state 正是缓存里那份 logit 的来源, 写回去是恒等操作。
                idx = act_idx.clamp(max=n_img - 1)
                k_img = int(idx.size(1))
            else:
                # active_idx 升序 → 图像 token 的下标全部排在 register 之前
                k_img = int((act_idx[0] < n_img).sum().item())
                idx = act_idx[:, :k_img]
            if k_img > 0:
                x_act = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
                logit_act = self.head(self.last_norm(x_act, y))
                cache = self.lazy_logit_cache
                if (cache is not None
                        and tuple(cache.shape) == (b, n_img, logit_act.size(-1))
                        and cache.dtype == logit_act.dtype
                        and cache.device == logit_act.device):
                    logit = cache.clone()
                    logit.scatter_(
                        1, idx.unsqueeze(-1).expand(-1, -1, logit.size(-1)), logit_act)
                    self.lazy_logit_cache = logit.detach()
                    self.transformer.lazy_stats["head_partial_calls"] += 1
                    return logit
                # 缓存不可用 (首次 / 形状-dtype 不匹配): 退回全量, 安全但不省算力。
                # gate 正常时不该发生 —— partial 步之前必有全量步把缓存写满。
                self.transformer.lazy_stats["head_miss"] += 1

        x = self.last_norm(x, y)
        logit = self.head(x)
        if _env_flag("HALTON_LAZY_CACHE"):
            self.lazy_logit_cache = logit.detach()
            self.transformer.lazy_stats["head_full_calls"] += 1

        return logit


if __name__ == "__main__":
    from thop import profile

    for size in ["tiny", "small", "base"]:
        print(size)
        if size == "tiny":
            hidden_dim, depth, heads = 384, 6, 6
        elif size == "small":
            hidden_dim, depth, heads = 512, 12, 6
        elif size == "base":
            hidden_dim, depth, heads = 768, 12, 12
        elif size == "large":
            hidden_dim, depth, heads = 1024, 24, 16
        elif size == "xlarge":
            hidden_dim, depth, heads = 1152, 28, 16
        else:
            hidden_dim, depth, heads = 768, 12, 12

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = 16
        model = Transformer(
            input_size=input_size, nclass=1000, hidden_dim=hidden_dim, codebook_size=16834,
            depth=depth, heads=heads, mlp_dim=hidden_dim * 4, dropout=0.1
        ).to(device)
        code = torch.randint(0, 16384, size=(1, input_size, input_size)).to(device)
        cls = torch.randint(0, 1000, size=(1,)).to(device)
        d_label = (torch.rand(1) < 0.1).to(device)

        flops, params = profile(model, inputs=(code, cls, d_label))
        print(f"FLOPs: {flops//1e9:.2f}G, Params: {params/1e6:.2f}M")
