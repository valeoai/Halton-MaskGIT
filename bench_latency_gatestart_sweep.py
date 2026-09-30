"""端到端 latency: lazycache r=1 / L0-23 / N=2 在 step gate 左端 S ∈ {0..8} 下的实测。

方法与 bench_latency_refresh.py 完全一致 (同进程只加载一次模型, cuda.synchronize
包夹, warmup 后多次取均值, VQGAN 解码单独计时并从 total 里扣掉), 唯一的区别是
扫的维度换成 HALTON_PARTIAL_START_STEP (门的左端 S), refresh N 固定 2。

被测配置逐项对应 results/halton_large384_lazycache_gatestart_ablation.txt 里的
lazycache_r1.0_N2_L0-23_S{S} —— 即 gate=[S,31) / ratio=1.0 / refresh_every_2 /
layers=0..23 / lazy head。理论列取自 flops_lazy_refresh.py (同一个 FLOPs 模型),
每档把 FL.GATE_LO 切到对应的 S 再 run(2)。

采样器超参沿用 bench_latency_refresh.py 的设定, 这样新表与既有的
results/latency/bench_lazycache_gatestart_s0_large.txt (S=0 / S=5) 可直接并列。
latency 与 cfg_w / temperature 无关, 只与门/刷新的步结构有关。

用法:  python bench_latency_gatestart_sweep.py [batch] [warmup] [timed] [size]
默认    python bench_latency_gatestart_sweep.py 16 1 5 large
       GATE_STARTS="0,1,2,3,4,5,6,7,8" 可改扫描点。
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
REFRESH_N = int(os.environ.get("REFRESH_N", "2"))
STARTS = [int(v) for v in os.environ.get("GATE_STARTS", "0,1,2,3,4,5,6,7,8").split(",") if v.strip()]

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


def set_cfg(lazy, start_step):
    """与 launch/eval_fid_large384_lazycache_gatestart.sh 的 env 设置保持一致。"""
    os.environ["HALTON_PARTIAL_UPDATE"] = "1" if lazy else "0"
    os.environ["HALTON_PARTIAL_START_STEP"] = str(start_step)
    os.environ["HALTON_CACHE_REFRESH_N"] = str(REFRESH_N if lazy else 0)
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


def theo(start_step):
    """理论加速比 —— 把 flops_lazy_refresh 的门左端切到 start_step 再算。"""
    old = FL.GATE_LO
    FL.GATE_LO = start_step
    try:
        tot, nr, npart, kr = FL.run(REFRESH_N)
    finally:
        FL.GATE_LO = old
    return FL.BASE / tot, tot, nr, npart, kr


print(f"\n== latency bench (step-gate start sweep) | {SIZE}-384 | batch={BATCH} | "
      f"warmup={WARMUP} timed={TIMED} | {torch.cuda.get_device_name(0)} ==")
print(f"   lazy r=1.0 / L0-{FL.DEPTH-1} / lazy head / refresh N={REFRESH_N} / "
      f"gate = [S, {FL.GATE_HI}),  S ∈ {STARTS}")
print(f"   baseline = {FL.BASE/1e9:.1f} GFLOPs/image\n")

dec_mean, dec_std = bench(time_decode)
print(f"VQGAN decode      : {dec_mean*1000/BATCH:7.2f} ms/img  "
      f"(batch {dec_mean*1000:7.1f}±{dec_std*1000:.1f} ms)\n")

print(f"{'config':16s} {'total ms/img':>13s} {'transf ms/img':>14s} {'±sd':>7s} "
      f"{'speedup':>8s} {'theo':>7s} {'达成率':>7s} {'GFLOPs':>9s} {'refresh/partial':>16s}")

set_cfg(lazy=False, start_step=STARTS[0])
m, sd = bench(time_sampler)
base_tr = m - dec_mean
print(f"{'baseline':16s} {m*1000/BATCH:11.2f}   {base_tr*1000/BATCH:12.2f}   "
      f"{sd*1000/BATCH:6.2f}  {1.0:6.3f}x  {1.0:6.3f}x  {100.0:5.1f}%  "
      f"{FL.BASE/1e9:8.1f}  {'0/0 (32 full)':>16s}")

rows = []
for s in STARTS:
    set_cfg(lazy=True, start_step=s)
    m, sd = bench(time_sampler)
    tr = m - dec_mean
    th, tot, nr, npart, kr = theo(s)
    spd = base_tr / tr
    print(f"{'S=%d N=%d' % (s, REFRESH_N):16s} {m*1000/BATCH:11.2f}   {tr*1000/BATCH:12.2f}   "
          f"{sd*1000/BATCH:6.2f}  {spd:6.3f}x  {th:6.3f}x  {spd/th*100:5.1f}%  "
          f"{tot/1e9:8.1f}  {'%d/%d' % (nr, npart):>16s}")
    st = getattr(model.vit, "module", model.vit).get_lazy_stats()
    print(f"{'':16s} └ 实测 partial k/N={st['partial_active_ratio']:.4f} "
          f"({st['partial_calls']} 步) / 全量 {st['full_calls']} 步; 理论 k/N={kr:.4f} "
          f"({npart} 步); kv_miss={st['kv_miss']} head(p/f/miss)="
          f"{st['head_partial_calls']}/{st['head_full_calls']}/{st['head_miss']}")
    rows.append((s, m*1000/BATCH, tr*1000/BATCH, spd, th, tot/1e9))

print("\n注: transf = total - VQGAN decode (各档解码相同); speedup 以 transformer-only 计。")
print("    达成率 = 实测加速 / 理论加速。")
