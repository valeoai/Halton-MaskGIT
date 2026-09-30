"""端到端 latency: LazyMAR CFG 残差缓存 (HALTON_CFG_DELTA) 单开 / 与 lazyvsim 合开。

方法与 bench_latency_vsim.py 完全一致 (同进程只加载一次模型, cuda.synchronize
包夹, warmup 后多次取均值, VQGAN 解码单独计时并从 total 里扣掉)。

四档:
  baseline                 无 cache
  lazyvsim  N=2            LazyMAR 的 token 选点 (V 余弦, rho_t 衰减), 层 3..23
  cfgdelta                 只开 CFG 残差缓存: lazy 步只跑 cond 半个 batch,
                           uncond = cond + 上一个整批步的 (uncond - cond)
  lazyvsim + cfgdelta      两者同开 —— 正交, 加速应当近似相乘

除了 latency, 还打印两个正确性观测量:
  * 各类缓存的 miss 计数 (全部应为 0, 否则说明某条路径退化成了全量/零填充)
  * 与 baseline 同 seed 生成的 code 的一致率 + CFG logit 相对误差,
    用来看这个近似在**训练好的**权重上到底有多准 (随机权重上量不出来)。

用法:  python bench_latency_cfgdelta.py [batch] [warmup] [timed]
默认    python bench_latency_cfgdelta.py 16 1 5
"""
import os
import sys
import time
import statistics
import torch

from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

BATCH  = int(sys.argv[1]) if len(sys.argv) > 1 else 16
WARMUP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TIMED  = int(sys.argv[3]) if len(sys.argv) > 3 else 5

CFG_W      = float(os.environ.get("BENCH_CFG_W", "0.5"))
REFRESH_N  = int(os.environ.get("BENCH_REFRESH_N", "2"))
VSIM_START = int(os.environ.get("BENCH_LAZY_START", "3"))
VSIM_END   = int(os.environ.get("BENCH_LAZY_END", "23"))
CFGD_N     = int(os.environ.get("BENCH_CFGD_N", "0"))
CFGD_START = int(os.environ.get("BENCH_CFGD_START", "5"))

args = load_args_from_file("Config/base_cls2img.yaml")
args.device = torch.device("cuda")
args.vit_size = os.environ.get("BENCH_VIT", "large")
args.img_size = int(os.environ.get("BENCH_IMG", "384"))
args.compile = False
args.dtype = "float32"
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"
model = MaskGIT(args)

sampler = HaltonSampler(sm_temp_min=1.0, sm_temp_max=1.0, temp_pow=1, temp_warmup=1,
                        w=CFG_W, sched_pow=2, step=32, randomize=True, top_k=-1)

base_labels = [1, 7, 282, 604, 724, 179, 681, 850]
labels = torch.LongTensor([base_labels[i % len(base_labels)] for i in range(BATCH)]).to(args.device)

ALL_KEYS = ("HALTON_PARTIAL_UPDATE", "HALTON_CACHE_REFRESH_N", "HALTON_LAZY_CACHE",
            "HALTON_LAZY_CACHE_RATIO", "HALTON_LAZY_HEAD", "HALTON_ATTN_CACHE",
            "HALTON_LAYER_CACHE", "HALTON_LAZY_VSIM", "HALTON_LAZY_VSIM_SCHED",
            "HALTON_LAZY_START_LAYER", "HALTON_LAZY_END_LAYER",
            "HALTON_PARTIAL_START_LAYER", "HALTON_PARTIAL_END_LAYER",
            "HALTON_CFG_DELTA", "HALTON_CFG_DELTA_N", "HALTON_CFG_DELTA_START")


def set_cfg(**kw):
    for k in ALL_KEYS:
        os.environ.pop(k, None)
    for k, v in kw.items():
        os.environ[k] = str(v)


VSIM = dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_VSIM=1,
            HALTON_LAZY_HEAD=1, HALTON_LAZY_START_LAYER=VSIM_START,
            HALTON_LAZY_END_LAYER=VSIM_END, HALTON_CACHE_REFRESH_N=REFRESH_N)
CFGD = dict(HALTON_CFG_DELTA=1, HALTON_CFG_DELTA_N=CFGD_N,
            HALTON_CFG_DELTA_START=CFGD_START)

CONFIGS = [
    ("baseline", dict(HALTON_PARTIAL_UPDATE=0)),
    (f"lazyvsim N={REFRESH_N}", dict(VSIM)),
    (f"cfgdelta N={CFGD_N}", dict(CFGD)),
    (f"lazyvsim + cfgdelta", dict(**VSIM, **CFGD)),
]


def sync():
    torch.cuda.synchronize()


def time_sampler():
    sync(); t0 = time.perf_counter()
    _ = sampler(trainer=model, nb_sample=BATCH, labels=labels,
                verbose=False, partial_update=None)[0]
    sync()
    return time.perf_counter() - t0


def time_decode():
    isz = model.input_size
    code = torch.randint(0, args.codebook_size, (BATCH, isz, isz), device=args.device)
    sync(); t0 = time.perf_counter()
    with torch.no_grad():
        x = model.ae.decode_code(torch.clamp(code, 0, args.codebook_size - 1))
        x = torch.clamp(x, -1, 1)
    sync()
    return time.perf_counter() - t0


def bench(fn):
    for _ in range(WARMUP):
        fn()
    ts = [fn() for _ in range(TIMED)]
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)


def sample_codes(seed=1234):
    """固定 seed 采一次, 返回最后一步的 code —— 各档之间可直接比一致率。"""
    torch.manual_seed(seed)
    out = sampler(trainer=model, nb_sample=BATCH, labels=labels,
                  verbose=False, partial_update=None)
    return out[1][-1].clone()


print(f"\n== latency bench (LazyMAR CFG 残差缓存) | {args.vit_size}-{args.img_size} | "
      f"batch={BATCH} | warmup={WARMUP} timed={TIMED} | cfg_w={CFG_W} | "
      f"{torch.cuda.get_device_name(0)} ==\n")

dec_mean, dec_std = bench(time_decode)
print(f"VQGAN decode      : {dec_mean*1000/BATCH:7.2f} ms/img  "
      f"(batch {dec_mean*1000:7.1f}±{dec_std*1000:.1f} ms)\n")

print(f"{'config':24s} {'total ms/img':>13s} {'transf ms/img':>14s} {'±ms':>7s} "
      f"{'speedup':>9s} {'code 一致率':>11s}")
base_tr = None
base_code = None
for name, cfg in CONFIGS:
    set_cfg(**cfg)
    m, sd = bench(time_sampler)
    tr = m - dec_mean
    code = sample_codes()
    if base_tr is None:
        base_tr, base_code = tr, code
    agree = (code == base_code).float().mean().item()
    spd = base_tr / tr
    print(f"{name:24s} {m*1000/BATCH:11.2f}   {tr*1000/BATCH:12.2f}   "
          f"{sd*1000/BATCH:6.2f}  {spd:7.3f}x  {agree:10.4f}")
    st = getattr(model.vit, "module", model.vit).get_lazy_stats()
    if cfg.get("HALTON_LAZY_CACHE") or cfg.get("HALTON_CFG_DELTA"):
        print(f"{'':24s} └ partial {st['partial_calls']} / 全量 {st['full_calls']} 步; "
              f"cfgdelta(reuse/full/miss)={st['cfg_delta_reuse_calls']}/"
              f"{st['cfg_delta_full_calls']}/{st['cfg_delta_miss']}; "
              f"kv_miss={st['kv_miss']} score_miss={st['score_miss']} "
              f"restore_miss={st['restore_miss']} head_miss={st['head_miss']}")

print("\n注: transf = total - VQGAN decode (各档解码相同); speedup 以 transformer-only 计。")
print("    code 一致率 = 与 baseline 同 seed 下最后一步 code 逐位相同的比例;")
print("    采样含随机性 (Categorical), 一致率不会到 1.0, 只用于横向比较各档的偏离程度。")
