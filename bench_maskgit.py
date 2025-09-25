# ==== Benchmark helpers for MaskGIT + Halton + Token Merge ====
import time, math, os, gc
import torch
from contextlib import contextmanager
import torch
from Utils.utils import load_args_from_file
from Utils.viz import show_images_grid
from huggingface_hub import hf_hub_download
from contextlib import nullcontext
import torch.nn as nn

from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler


try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

@contextmanager
def cuda_timer():
    """GPU 计时（毫秒）+ CPU 墙钟时间（秒）。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)
        start_event.record()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            gpu_ms = start_event.elapsed_time(end_event)
        else:
            gpu_ms = float('nan')
        t1 = time.perf_counter()
        cuda_timer.last = {"gpu_ms": gpu_ms, "wall_s": t1 - t0}

def _bytes_to_mib(x):
    return x / (1024**2)

def _set_trainer_modules_eval(trainer):
    """尽量把 trainer 中真正的 nn.Module 子模块切到 eval。找不到就跳过。"""
    possible_attrs = ("vit", "vq", "vqgan", "vq_ae", "net", "model", "encoder", "decoder")
    for name in possible_attrs:
        m = getattr(trainer, name, None)
        if isinstance(m, nn.Module):
            m.eval()

def measure_once(model, sampler, labels, nb_sample, dtype="float32"):
    device = labels.device

    # ✅ 不调用 MaskGIT 自己的 eval/train；只把内部 nn.Module 子网切到 eval（若存在）
    _set_trainer_modules_eval(model)

    # 确保无梯度 +（可选）混合精度
    ctx_infer = torch.inference_mode()
    ctx_autocast = (
        torch.autocast(device_type="cuda", dtype=getattr(torch, dtype))
        if (device.type == "cuda" and dtype in ["float16", "bfloat16"])
        else nullcontext()
    )

    # 清理缓存 & 重置峰值显存统计
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)

    cpu_rss_before = psutil.Process(os.getpid()).memory_info().rss if _HAS_PSUTIL else None

    with ctx_infer, ctx_autocast:
        with cuda_timer():
            # === 生成一次 ===
            imgs = sampler(trainer=model, nb_sample=nb_sample, labels=labels, verbose=False)[0]

    # GPU 显存峰值
    if device.type == "cuda":
        peak_alloc_mib = _bytes_to_mib(torch.cuda.max_memory_allocated(device))
        peak_resvd_mib = _bytes_to_mib(torch.cuda.max_memory_reserved(device))
    else:
        peak_alloc_mib = peak_resvd_mib = float('nan')

    cpu_rss_after = psutil.Process(os.getpid()).memory_info().rss if _HAS_PSUTIL else None
    cpu_delta_mib = _bytes_to_mib((cpu_rss_after - cpu_rss_before)) if (_HAS_PSUTIL and cpu_rss_after and cpu_rss_before) else None

    gpu_ms  = cuda_timer.last["gpu_ms"]
    wall_s  = cuda_timer.last["wall_s"]
    ips     = nb_sample / wall_s if wall_s > 0 else float('inf')
    per_img = (wall_s / nb_sample) if nb_sample > 0 else float('nan')

    metrics = {
        "nb_sample": nb_sample,
        "gpu_ms_total": gpu_ms,
        "wall_s_total": wall_s,
        "img_per_sec": ips,
        "latency_per_img_s": per_img,
        "gpu_peak_alloc_MiB": peak_alloc_mib,
        "gpu_peak_reserved_MiB": peak_resvd_mib,
        "cpu_mem_delta_MiB": cpu_delta_mib,
    }
    return metrics, imgs

def benchmark_grid(
    model_ctor,        # 一个返回已初始化好且加载权重的 MaskGIT 实例的函数：lambda args: MaskGIT(args)
    sampler_ctor,      # 一个返回已配置好的 HaltonSampler 的函数：lambda: HaltonSampler(...)
    labels,            # LongTensor [B]
    base_args,         # args 对象（会被复制浅拷贝后改字段）
    trials=5,          # 重复次数（会丢弃第1次预热）
    nb_sample_list=(8,),
    tome_keep_ratio_list=(1.0, 0.75, 0.5, 0.33),
    vit_size_list=None,    # 例如 ("tiny","small","base","large")
    img_size_list=None,    # 例如 (256, 384)
    compile_list=(False,), # 也可以试 (False, True) （注意：compile 首次会有额外编译时延，不计入正式统计）
    dtype_list=("float32",),   # 可试 ("float32","bfloat16","float16")
    extra_notes_fn=None,   # 可选：传入函数把 sampler 的 step/top_k 等附加写到结果里
):
    results = []
    # 随机性控制（更公平的对比）
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    for nb_sample in nb_sample_list:
        for keep in tome_keep_ratio_list:
            for dtype in dtype_list:
                for do_compile in compile_list:
                    vit_sizes = vit_size_list or (base_args.vit_size,)
                    img_sizes = img_size_list or (base_args.img_size,)
                    for vit in vit_sizes:
                        for imsz in img_sizes:
                            # --- 准备 args 的拷贝 ---
                            args = base_args
                            # 建议：把 Token Merge 开关放在 args 里（你代码中已有相应字段）
                            args.tome_keep_ratio = keep
                            # 如需定位层，可设置：
                            # args.tome_merge_layer_idx = 0
                            # args.tome_unmerge_before_idx = -1
                            # args.tome_random_roll = False
                            args.vit_size = vit
                            args.img_size = imsz
                            args.dtype = dtype
                            args.compile = bool(do_compile)

                            # --- 初始化模型 & 采样器 ---
                            model = model_ctor(args)
                            sampler = sampler_ctor()

                            # 若要基准公平，建议固定采样器随机性
                            if hasattr(sampler, "randomize"):
                                sampler.randomize = False

                            # 预热（不计入统计；也触发 compile 的图捕获）
                            _ = measure_once(model, sampler, labels, nb_sample, dtype=dtype)

                            trial_metrics = []
                            for t in range(trials):
                                m, _imgs = measure_once(model, sampler, labels, nb_sample, dtype=dtype)
                                trial_metrics.append(m)

                            # 丢弃第1次（cache 效应最大），汇总平均与方差
                            use = trial_metrics[1:] if len(trial_metrics) > 1 else trial_metrics
                            def avg(key):
                                vals = [d[key] for d in use if not (isinstance(d[key], float) and math.isnan(d[key]))]
                                return sum(vals)/len(vals) if vals else float('nan')

                            row = {
                                "nb_sample": nb_sample,
                                "tome_keep_ratio": keep,
                                "dtype": dtype,
                                "compile": bool(do_compile),
                                "vit_size": vit,
                                "img_size": imsz,
                                "gpu_ms_total_avg": avg("gpu_ms_total"),
                                "wall_s_total_avg": avg("wall_s_total"),
                                "img_per_sec_avg": avg("img_per_sec"),
                                "latency_per_img_s_avg": avg("latency_per_img_s"),
                                "gpu_peak_alloc_MiB_avg": avg("gpu_peak_alloc_MiB"),
                                "gpu_peak_reserved_MiB_avg": avg("gpu_peak_reserved_MiB"),
                                "cpu_mem_delta_MiB_avg": avg("cpu_mem_delta_MiB"),
                            }
                            if extra_notes_fn is not None:
                                row.update(extra_notes_fn(model, sampler))
                            results.append(row)

                            # 回收显存/内存，减少不同组合之间的干扰
                            del model, sampler, _imgs
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
    return results

def pretty_print(results):
    # 简洁打印为表格
    if not results:
        print("No results.")
        return
    keys = ["vit_size","img_size","nb_sample","tome_keep_ratio","dtype","compile",
            "img_per_sec_avg","latency_per_img_s_avg","gpu_peak_alloc_MiB_avg","gpu_peak_reserved_MiB_avg","wall_s_total_avg"]
    widths = {k:max(len(k), max(len(f"{r.get(k)}") for r in results)) for k in keys}
    sep = " | "
    print(sep.join(k.ljust(widths[k]) for k in keys))
    print("-" * (sum(widths.values()) + len(sep)*(len(keys)-1)))
    for r in results:
        print(sep.join(str(r.get(k)).ljust(widths[k]) for k in keys))

# ====== 示例：在你已有初始化代码之后追加 ======
config_path = "Config/base_cls2img.yaml"        # Path to your config file
args = load_args_from_file(config_path)

# Update arguments
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select Network (Large 384 is the best, but the slowest)
args.vit_size = "large"  # "tiny", "small", "base", "large"
args.img_size = 384  # 256 or 384
args.compile = False
args.dtype = "float32"
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"

# Download the MaskGIT
hf_hub_download(repo_id="llvictorll/Halton-Maskgit",
                filename=f"ImageNet_{args.img_size}_{args.vit_size}.pth",
                local_dir="./saved_networks")

# Download VQGAN
hf_hub_download(repo_id="FoundationVision/LlamaGen",
                filename="vq_ds16_c2i.pt",
                local_dir="./saved_networks")



# [goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear]
labels = torch.LongTensor([1, 7, 282, 604, 724, 179, 681, 850]).to(args.device)


if __name__ == "__main__":
    # 你上面已有：
    # args = load_args_from_file(config_path)
    # ... 下载权重略 ...
    # 注意：labels 的 batch 大小就是 nb_sample
    base_args = args  # 重用

    # 构造器封装（确保每次新建干净实例）
    def _model_ctor(_args):
        m = MaskGIT(_args)
        # 有些工程将 device 放在 args 里自动 to(device)，若没有可手动：
        # m.to(_args.device)
        return m

    def _sampler_ctor():
        # 与你示例保持一致，可改 step/top_k 以实验不同调度
        return HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0, w=2,
                             sched_pow=2, step=32, randomize=True, top_k=-1)

    # 可把 sampler 的关键参数也记录进结果（可选）
    def _extra(model, sampler):
        out = {}
        for k in ["step", "top_k", "w", "sched_pow"]:
            if hasattr(sampler, k):
                out[f"sampler_{k}"] = getattr(sampler, k)
        return out

    # ==== 配置你的实验网格 ====
    trials = 5  # 每个组合跑5次（自动丢弃第1次）
    nb_sample_list = (len(labels),)  # 或者试 (4, 8, 16)
    tome_keep_ratio_list = (1.0, 0.75, 0.5, 0.33)
    vit_size_list = (base_args.vit_size,)  # 先固定 ViT 尺寸；也可试 ("tiny","small","base","large")
    img_size_list = (base_args.img_size,)  # 先固定分辨率；也可试 (256, 384)
    compile_list = (False,)                # 也可试 (False, True)
    dtype_list = (base_args.dtype,)        # 也可试 ("float32","bfloat16","float16")

    results = benchmark_grid(
        model_ctor=_model_ctor,
        sampler_ctor=_sampler_ctor,
        labels=labels,
        base_args=base_args,
        trials=trials,
        nb_sample_list=nb_sample_list,
        tome_keep_ratio_list=tome_keep_ratio_list,
        vit_size_list=vit_size_list,
        img_size_list=img_size_list,
        compile_list=compile_list,
        dtype_list=dtype_list,
        extra_notes_fn=_extra
    )

    pretty_print(results)
    # 如需保存：
    # import json; json.dump(results, open("benchmark_results.json","w"), ensure_ascii=False, indent=2)
