"""端到端 latency: LazyMAR 复现 (Token Cache + CFG 残差 Cache) 在 N ∈ {none,2,3,4,8,13} 下的实测。

方法与 bench_latency_vsim_sweep.py 完全一致 (同进程只加载一次模型, cuda.synchronize
包夹, warmup 后多次取均值, VQGAN 解码单独计时并从 total 里扣掉), 被测配置换成:

    baseline          无 cache
    lazymar N=...     Token Cache (V 相似度选点, 无强制集合) + CFG 残差 Cache 同开,
                      两个缓存的 refresh N 取同一个值 (实验设定)

理论列取自 flops_lazymar.py, 与 FID sweep 记的是同一个 FLOPs 模型。

用法:  python bench_latency_lazymar_sweep.py [batch] [warmup] [timed] [size]
默认    python bench_latency_lazymar_sweep.py 16 1 5 large
size ∈ {small, base, large}; cfg_w 默认按尺寸取 FID sweep 用的值
(large 0.5 / base 0.7 / small 1.0), 可用 BENCH_CFG_W 覆盖。
BENCH_WITH_VSIM=1 可以额外加测"只开 Token Cache"的分解档 (默认不测)。
"""
import os
import sys
import time
import statistics
import torch

from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler
import flops_lazy_vsim as FV
import flops_lazymar as FC

BATCH  = int(sys.argv[1]) if len(sys.argv) > 1 else 16
WARMUP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TIMED  = int(sys.argv[3]) if len(sys.argv) > 3 else 5
SIZE   = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("LAZY_VIT_SIZE", "large")

CFG_BY_SIZE = {"large": 0.5, "base": 0.7, "small": 1.0}
CFG_W = float(os.environ.get("BENCH_CFG_W", CFG_BY_SIZE.get(SIZE, 0.5)))
WITH_VSIM = os.environ.get("BENCH_WITH_VSIM", "0") == "1"

FV.set_size(SIZE)
FC.set_size(SIZE)
DEPTH = FV.DEPTH
L_START, L_END = FV.L_START, FV.L_END
VSIM_SCHED = os.environ.get("BENCH_VSIM_SCHED", "lazymar")

args = load_args_from_file("Config/base_cls2img.yaml")
args.device = torch.device("cuda")
args.vit_size = SIZE
args.img_size = 384
args.compile = False
args.dtype = "float32"
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"
model = MaskGIT(args)

# 采样调度与 launch/eval_fid_*_lazymar.sh 逐项一致
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


def vsim_cfg(n):
    return dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_VSIM=1,
                HALTON_LAZY_VSIM_SCHED=VSIM_SCHED, HALTON_LAZY_HEAD=1,
                HALTON_LAZY_START_LAYER=L_START, HALTON_LAZY_END_LAYER=L_END,
                HALTON_CACHE_REFRESH_N=n)


def lazymar_cfg(n):
    """两个缓存同开, refresh N 保持一致。"""
    return dict(vsim_cfg(n), HALTON_CFG_DELTA=1, HALTON_CFG_DELTA_N=n,
                HALTON_CFG_DELTA_START=5)


# (name, env, 理论加速)
CONFIGS = [("baseline", dict(HALTON_PARTIAL_UPDATE=0), None)]
if WITH_VSIM:      # 可选的分解档: 只开 Token Cache
    for n in FV.REFRESH_NS:
        label = "none" if n == 0 else str(n)
        CONFIGS.append((f"token-cache N={label}", vsim_cfg(n), FV.BASE / FV.run(n)[0]))
for n in FV.REFRESH_NS:
    label = "none" if n == 0 else str(n)
    CONFIGS.append((f"lazymar N={label}", lazymar_cfg(n), FC.BASE / FC.run(n)[0]))


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


print(f"\n== latency bench (LazyMAR 复现 sweep) | {SIZE}-384 | batch={BATCH} | "
      f"warmup={WARMUP} timed={TIMED} | cfg_w={CFG_W} | lazy L{L_START}-{L_END} | "
      f"{torch.cuda.get_device_name(0)} ==\n")

dec_mean, dec_std = bench(time_decode)
print(f"VQGAN decode      : {dec_mean*1000/BATCH:7.2f} ms/img  "
      f"(batch {dec_mean*1000:7.1f}±{dec_std*1000:.1f} ms)\n")

print(f"{'config':22s} {'total ms/img':>13s} {'transf ms/img':>14s} {'±ms':>7s} "
      f"{'实测':>8s} {'理论':>7s} {'达成率':>7s}")
base_tr = None
for name, cfg, theo in CONFIGS:
    set_cfg(**cfg)
    m, sd = bench(time_sampler)
    tr = m - dec_mean
    if base_tr is None:
        base_tr = tr
        theo = 1.000
    spd = base_tr / tr
    print(f"{name:22s} {m*1000/BATCH:11.2f}   {tr*1000/BATCH:12.2f}   "
          f"{sd*1000/BATCH:6.2f}  {spd:6.3f}x  {theo:6.3f}x  {spd/theo*100:5.1f}%")
    if cfg.get("HALTON_LAZY_CACHE"):
        st = getattr(model.vit, "module", model.vit).get_lazy_stats()
        print(f"{'':22s} └ partial k/N={st['partial_active_ratio']:.4f} "
              f"({st['partial_calls']} 步) / 全量 {st['full_calls']} 步; "
              f"vsim={st['vsim_calls']} scored={st['scored_calls']} "
              f"cfgd(reuse/full/miss)={st['cfg_delta_reuse_calls']}/"
              f"{st['cfg_delta_full_calls']}/{st['cfg_delta_miss']} "
              f"kv_miss={st['kv_miss']} score_miss={st['score_miss']} "
              f"restore_miss={st['restore_miss']} head_miss={st['head_miss']}")

print("\n注: transf = total - VQGAN decode (各档解码相同); speedup 以 transformer-only 计。")
print("    理论列取自 flops_lazymar.py (CFG 的 2x 显式建模: lazy 步主干 1x + head 2x)。")
