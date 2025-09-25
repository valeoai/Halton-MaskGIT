# Transformer Encoder architecture
# some part have been borrowed from:
#   - NanoGPT: https://github.com/karpathy/nanoGPT
#   - DiT: https://github.com/facebookresearch/DiT

import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from Sampler.halton_sampler import HaltonSampler  # Replace with the correct module if different
from Sampler.halton_sampler import _halton_centers_and_assignments_sq
from Sampler.halton_sampler import _merge_tokens_spatial
from Sampler.halton_sampler import _unmerge_tokens_spatial


def param_count(archi, model):
    print(f"Size of model {archi}: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
        self.flash = use_flash # use flash attention?
        self.n_local_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)

        self.qk_norm = QKNorm(num_heads * self.head_dim)

        # will be KVCache object managed by inference context manager
        self.cache = None

    def forward(self, x, mask=None):
        b, h_w, _ = x.shape
        # calculate query, key, value and split out heads
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
            output = F.scaled_dot_product_attention(xq, xk, xv, mask, dropout_p=self.dropout if self.training else 0.)
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

    def forward(self, x, cond, mask=None):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)
        x = x + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x), gamma1, beta1), mask=mask)
        x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim, heads, mlp_dim, dropout=dropout))

    def forward(self, x, cond, mask=None):
        for block in self.layers:
            x = block(x, cond, mask=mask)
        return x


class Transformer(nn.Module):
    """ DiT-like transformer with adaLayerNorm with zero initializations """
    def __init__(self, input_size=16, hidden_dim=768, codebook_size=1024,
                 depth=12, heads=16, mlp_dim=3072, dropout=0., nclass=1000,
                 register=1, proj=1, **kwargs):
        super().__init__()

        self.nclass = nclass                                             # Number of classes
        self.input_size = input_size                                     # Number of tokens as input
        self.hidden_dim = hidden_dim                                     # Hidden dimension of the transformer
        self.codebook_size = codebook_size                               # Amount of code in the codebook
        self.proj = proj                                                 # Projection

        self.cls_emb = nn.Embedding(nclass + 1, hidden_dim)              # Embedding layer for the class token
        self.tok_emb = nn.Embedding(codebook_size + 1, hidden_dim)       # Embedding layer for the 'visual' token
        self.pos_emb = nn.Embedding(input_size ** 2, hidden_dim)         # Learnable Positional Embedding

        if self.proj > 1:
            self.in_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, bias=False)
            self.out_proj = nn.Conv2d(
                hidden_dim, hidden_dim*4, kernel_size=1, stride=1, padding=0, bias=False
            ).to(memory_format=torch.channels_last)

        # The Transformer Encoder a la BERT :)
        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        self.last_norm = AdaNorm(x_dim=hidden_dim, y_dim=hidden_dim)   # Last Norm

        self.head = nn.Linear(hidden_dim, codebook_size + 1)
        self.head.weight = self.tok_emb.weight  # weight tied with the tok_emb layer

        self.register = register
        if self.register > 0:
            self.reg_tokens = nn.Embedding(self.register, hidden_dim)

        self.initialize_weights()  # Init weight
        # ---- Halton-Token-Merge 配置 ----
        self.tome_keep_ratio = kwargs.get("tome_keep_ratio", 1.0)  # 1.0=不合并；比如 0.5=保留一半 token
        self.tome_merge_layer_idx = kwargs.get("tome_merge_layer_idx", 0)  # 第 1 层前合并
        self.tome_unmerge_before_idx = kwargs.get("tome_unmerge_before_idx", -1)  # 最后一层前反合并
        self.tome_random_roll = kwargs.get("tome_random_roll", False)  # 如需随机 roll Halton 序列可用
        self._tome_cache = {}  # key=(h,w,M,device) -> {"centers":..., "ids":..., "counts":...}

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Init embedding
        nn.init.normal_(self.cls_emb.weight, std=0.02)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # Zero-out adaNorm modulation layers in blocks:
        for block in self.transformer.layers:
            nn.init.constant_(block.mlp[1].weight, 0)
            nn.init.constant_(block.mlp[1].bias, 0)

        # Init proj layer
        if self.proj > 1:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

        # Init embedding
        if self.register > 0:
            nn.init.normal_(self.reg_tokens.weight, std=0.02)

    def forward(self, x, y, drop_label, mask=None):
        b, h, w = x.size()
        x = x.reshape(b, h * w)

        # Drop the label if drop_label
        y = torch.where(drop_label, torch.full_like(y, self.nclass), y)
        y = self.cls_emb(y)

        # 位置编码（按原样）
        pos = torch.arange(0, w * h, dtype=torch.long, device=x.device)
        pos = self.pos_emb(pos)

        x = self.tok_emb(x) + pos  # [B, N, C], N=h*w

        # optional 下采样 patchify
        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w, b=b, c=self.hidden_dim).contiguous()
            x = self.in_proj(x)  # stride=2
            _, _, h, w = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c', b=b, c=self.hidden_dim, h=h, w=w).contiguous()

        # ===== 这里开始：准备 register token，并在进入 blocks 前做 ToMe 合并 =====
        reg_tok = None
        if self.register > 0:
            reg = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            reg_tok = self.reg_tokens(reg).expand(b, self.register, self.hidden_dim)  # [B,R,C]
            # 注意：合并只对空间 token 生效，register 不参与合并

        N = h * w  # 空间 token 数
        use_tome = (self.tome_keep_ratio is not None) and (self.tome_keep_ratio < 1.0) and (N > 0)

        # 构建合并后的输入序列 x_for_blocks，以及为反合并准备的映射
        tome_ctx = None
        if use_tome:
            # 计算 M（合并后保留的 token 数）
            M = max(1, int(round(N * self.tome_keep_ratio)))

            # 取/建缓存
            key = (h, w, M, x.device.type)
            if key not in self._tome_cache:
                # 仅支持正方形网格；如果将来 h!=w，需要相应修改为长方形 Halton
                assert h == w, f"Halton-ToMe 当前实现仅支持正方形网格，got h={h}, w={w}"
                builder = _halton_centers_and_assignments_sq(h, M, x.device)
                centers, cluster_ids, counts = builder(M)
                if self.tome_random_roll:
                    # 可选：对 halton 序列整体 roll，做一点随机性
                    shift = torch.randint(0, N, (1,), device=x.device).item()
                    order = torch.arange(N, device=x.device)
                    cluster_ids = cluster_ids[order.roll(shifts=shift)]
                self._tome_cache[key] = {
                    "centers": centers,            # [M,2] 目前没直接用到，但保留
                    "ids": cluster_ids,            # [N]
                    "counts": counts               # [M]
                }
            ids = self._tome_cache[key]["ids"]
            counts = self._tome_cache[key]["counts"]

            # 只合并空间 token
            x_sp = x[:, :N, :]                                # [B,N,C]
            x_sp_m = _merge_tokens_spatial(x_sp, ids, counts) # [B,M,C]

            # 与 register token 重新拼接得到进入 blocks 的序列
            if reg_tok is not None:
                x_for_blocks = torch.cat([x_sp_m, reg_tok], dim=1)   # [B, M+R, C]
            else:
                x_for_blocks = x_sp_m                                 # [B, M, C]

            tome_ctx = {
                "ids": ids, "counts": counts, "N": N, "M": M, "has_reg": reg_tok is not None
            }
        else:
            # 不启用合并：保持原样
            x_for_blocks = torch.cat([x, reg_tok], dim=1) if reg_tok is not None else x

        # ======= 进入 Transformer blocks：第 1 层前已合并；最后一层前反合并 =======
        L = len(self.transformer.layers)
        unmerge_before = self.tome_unmerge_before_idx if self.tome_unmerge_before_idx >= 0 else (L - 1)

        for i, block in enumerate(self.transformer.layers):
            # 在最后一层 block 之前把 token 反合并回原始长度（仅当启用 ToMe）
            if use_tome and i == unmerge_before:
                if tome_ctx["has_reg"]:
                    x_sp_m, x_reg = x_for_blocks[:, :tome_ctx["M"], :], x_for_blocks[:, tome_ctx["M"]:, :]
                else:
                    x_sp_m, x_reg = x_for_blocks, None

                x_sp_full = _unmerge_tokens_spatial(x_sp_m, tome_ctx["ids"])  # [B,N,C]

                x_for_blocks = torch.cat([x_sp_full, x_reg], dim=1) if x_reg is not None else x_sp_full
                # 注意：反合并后，后续 block（也就是最后一个 block）将对“完整长度”的序列计算

            # 关于 mask：
            #   合并阶段（前 L-1 层）序列长度变化，mask 需要同步聚合；若你的训练里不依赖 mask，推荐传 None。
            #   最后一层已反合并，我们把原 mask 直接传给最后一层（如果你需要）。
            if (use_tome and i < unmerge_before):
                x_for_blocks = block(x_for_blocks, y, mask=None)
            else:
                x_for_blocks = block(x_for_blocks, y, mask=mask)

        x = x_for_blocks  # [B, (N或M)+R, C]，取决于最后是否已反合并

        # 去掉 register，仅保留空间 token（此时若启用 ToMe，已反合并回 N）
        x = x[:, :h * w].contiguous()

        # optional 上采样还原 token 网格（按你原逻辑）
        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w, b=b, c=self.hidden_dim).contiguous()
            x = self.out_proj(x)
            x = rearrange(x, 'b (c s1 s2) h w -> b (h s1 w s2) c',
                          s1=self.proj, s2=self.proj, b=b, h=h, w=w, c=self.hidden_dim).contiguous()

        x = self.last_norm(x, y)
        logit = self.head(x)
        return logit



if __name__ == "__main__":
    from thop import profile
    # 测试不同规模的transformer的FLOPs和参数量
    for size in ["tiny", "small", "base"]: # "large", "xlarge"]:
        # size = "tiny"
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
        model = Transformer(input_size=input_size, nclass=1000, hidden_dim=hidden_dim, codebook_size=16834,
                            depth=depth, heads=heads, mlp_dim=hidden_dim * 4, dropout=0.1).to(device)
        # model = torch.compile(model)
        code = torch.randint(0, 16384, size=(1, input_size, input_size)).to(device)
        cls = torch.randint(0, 1000, size=(1,)).to(device)
        d_label = (torch.rand(1) < 0.1).to(device)

        flops, params = profile(model, inputs=(code, cls, d_label))
        print(f"FLOPs: {flops//1e9:.2f}G, Params: {params/1e6:.2f}M")

