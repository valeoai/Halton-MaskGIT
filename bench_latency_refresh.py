"""端到端 latency: lazy r=1 / 全部层 / lazy head 在 refresh N ∈ {0,2,3,4,8,13} 下的实测。

与 bench_latency.py 同一套方法 (同进程只加载一次模型, cuda.synchronize 包夹,
warmup 后多次取均值, VQGAN 解码单独计时并从 total 里扣掉), 只是把配置换成
refresh 周期扫描。理论列直接从 flops_lazy_refresh.py 取, 保证两边同一个 FLOPs 模型。

用法:  python bench_latency_refresh.py [batch] [warmup] [timed] [size]
默认    python bench_latency_refresh.py 16 1 5 large
size ∈ {small, base, large} —— 对应 saved_networks/ImageNet_384_{size}.pth,
理论列会同步切到同一尺寸 (flops_lazy_refresh.set_size)。
"""
import os
import sys
import time
import statistics
import torch

from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler
import flops_lazy_refresh as FL

BATCH  = int(sys.argv[1]) if len(sys.argv) > 1 else 16
WARMUP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TIMED  = int(sys.argv[3]) if len(sys.argv) > 3 else 5
SIZE   = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("LAZY_VIT_SIZE", "large")

# step gate 的左端 —— 与 Sampler/halton_sampler.py 读同一个 env (默认 5)。
# flops_lazy_refresh 在 import 时已经读过同一个 env, 所以理论列与被测配置同门。
GATE_START = FL.GATE_LO

FL.set_size(SIZE)                    # 理论列与被测模型同尺寸

args = load_args_from_file("Config/base_cls2img.yaml")
args.device = torch.device("cuda")
args.vit_size = SIZE
args.img_size = 384
args.compile = False
args.dtype = "float32"
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"
model = MaskGIT(args)

sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0,
                        w=2, sched_pow=2, step=32, randomize=True, top_k=-1)

base_labels = [1, 7, 282, 604, 724, 179, 681, 850]
labels = torch.LongTensor([base_labels[i % len(base_labels)] for i in range(BATCH)]).to(args.device)


def set_cfg(lazy, refresh_n):
    """与 launch/eval_fid_large384_lazycache.sh 的 env 设置保持一致。"""
    os.environ["HALTON_PARTIAL_UPDATE"] = "1" if lazy else "0"
    os.environ["HALTON_PARTIAL_START_STEP"] = str(GATE_START)
    os.environ["HALTON_CACHE_REFRESH_N"] = str(refresh_n)
    os.environ["HALTON_LAZY_CACHE"] = "1" if lazy else "0"
    os.environ["HALTON_LAZY_CACHE_RATIO"] = "1.0"
    os.environ["HALTON_LAZY_HEAD"] = "1"
    os.environ["HALTON_ATTN_CACHE"] = "0"
    for k in ("HALTON_LAYER_CACHE", "HALTON_LAZY_START_LAYER", "HALTON_LAZY_END_LAYER",
              "HALTON_PARTIAL_START_LAYER", "HALTON_PARTIAL_END_LAYER"):
        os.environ.pop(k, None)


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


CONFIGS = [("baseline", None)] + [(f"lazy r=1 N={n if n else 'none'}", n)
                                  for n in FL.REFRESH_NS]

print(f"\n== latency bench (refresh sweep) | {SIZE}-384 | batch={BATCH} | "
      f"warmup={WARMUP} timed={TIMED} | {torch.cuda.get_device_name(0)} ==")
print(f"   step gate = [{GATE_START}, {FL.GATE_HI})  (HALTON_PARTIAL_START_STEP)\n")

dec_mean, dec_std = bench(time_decode)
print(f"VQGAN decode      : {dec_mean*1000/BATCH:7.2f} ms/img  "
      f"(batch {dec_mean*1000:7.1f}±{dec_std*1000:.1f} ms)\n")

print(f"{'config':20s} {'total ms/img':>13s} {'transf ms/img':>14s} "
      f"{'speedup':>8s} {'theo':>7s} {'达成率':>7s}")
base_tr = None
rows = []
for name, n in CONFIGS:
    set_cfg(lazy=(n is not None), refresh_n=(n if n is not None else 0))
    m, sd = bench(time_sampler)
    tr = m - dec_mean
    if base_tr is None:
        base_tr = tr
        theo = 1.000
    else:
        tot, _, _, _ = FL.run(n)
        theo = FL.BASE / tot
    spd = base_tr / tr
    print(f"{name:20s} {m*1000/BATCH:11.2f}   {tr*1000/BATCH:12.2f}   "
          f"{spd:6.3f}x  {theo:6.3f}x  {spd/theo*100:5.1f}%")
    rows.append((name, m*1000/BATCH, tr*1000/BATCH, spd, theo))
    if n is not None and not FL.seeded(n, GATE_START):
        print(f"{'':20s} └ !! 门从 step0 就开且不 refresh: 没有任何全量步先把缓存"
              f"写满, partial 步读到的是空缓存 (见下面的 miss 计数)。这一行的"
              f"速度是真的, 但它对应的采样在数值上无意义。")
    if n is not None:
        st = getattr(model.vit, "module", model.vit).get_lazy_stats()
        print(f"{'':20s} └ partial k/N={st['partial_active_ratio']:.4f} "
              f"({st['partial_calls']} 步) / 全量 {st['full_calls']} 步; "
              f"kv_miss={st['kv_miss']} head(p/f/miss)="
              f"{st['head_partial_calls']}/{st['head_full_calls']}/{st['head_miss']}")

print("\n注: transf = total - VQGAN decode (各档解码相同); speedup 以 transformer-only 计。")
print("    达成率 = 实测加速 / 理论加速。")
