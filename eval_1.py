# filepath: eval.py
import os
import torch
import random
import numpy as np

from Trainer.cls_trainer import MaskGIT
from Utils.utils import load_args_from_file

# ==== 你要用到的外部统计文件与评测设置 ====
REAL_STATS_PATH = "/work/q-li/Halton-MaskGIT/saved_networks/ImageNet_256_val_stats.pt"  # <- 改成你的绝对路径
IMG_SIZE_FOR_FID = 384       # 用 256x256 的统计
NUM_IMAGES_FOR_FID = 50000    # FID 通常用 50k，跑 10k 也行
EVAL_IMAGENET_ROOT = "../datasets/imagenet"  # 你的 ImageNet 根目录（包含 train/val）

def try_inject_real_stats(trainer, stats_path: str) -> dict:
    """
    从 stats_path 读取 (mu, sigma, n_real)，优先以 kwargs 形式传入 eval；
    如 eval 不支持，则尝试注入到 trainer 的度量器中。
    """
    import os, torch
    if not os.path.isfile(stats_path):
        print(f"[WARN] REAL_STATS_PATH 不存在：{stats_path}。回退为在线提取真实分布。")
        return {}

    obj = torch.load(stats_path, map_location="cpu")

    # ---- 解析 (mu, sigma, n_real) ----
    mu = sigma = n_real = None

    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2:
            mu, sigma = obj[0], obj[1]
            n_real = obj[2] if len(obj) > 2 else None
    elif isinstance(obj, dict):
        def first(d, names):
            for k in names:
                if k in d:
                    return d[k]
            return None
        mu    = first(obj, ["mu", "mean", "m"])
        sigma = first(obj, ["sigma", "cov", "s", "S", "Sigma"])
        n_real = first(obj, ["n_real", "n", "N", "num", "count"])
    else:
        raise TypeError(f"[ERROR] 未知的统计文件结构：{type(obj)}")

    if mu is None or sigma is None:
        raise ValueError(f"[ERROR] 未找到 mu/sigma。可用 keys: {list(obj.keys()) if isinstance(obj, dict) else 'N/A'}")

    if isinstance(n_real, torch.Tensor):
        n_real = int(n_real.item())
    if n_real is None:
        n_real = 50000  # 合理缺省

    # 简单形状校验（不通过也不中断，只提示）
    try:
        print(f"[INFO] stats: mu={tuple(mu.shape)} sigma={tuple(sigma.shape)} n_real={n_real}")
    except Exception:
        pass

    # 1) 首选：以 kwargs 传给 eval（若支持）
    payload = {"fid_real_stats": (mu, sigma, int(n_real))}

    # 2) 兜底：尝试注入到 trainer 的度量器
    injected = False
    for path in [
        ("metrics", "set_real_stats_from_tensors"),
        ("fid", "set_reference"),
        ("fid_evaluator", "set_stats"),
        ("metrics", "fid", "set_reference"),
    ]:
        try:
            obj_ = trainer
            for attr in path[:-1]:
                obj_ = getattr(obj_, attr)
            setter = getattr(obj_, path[-1])
            try:
                setter(mu, sigma, int(n_real))
            except TypeError:
                setter(mu, sigma)
            print(f"[INFO] 已通过 {'.'.join(path)} 注入真实分布统计。")
            injected = True
            break
        except Exception:
            pass

    if not injected:
        print("[INFO] 未发现可注入接口；若 eval(...) 支持 fid_real_stats，将以参数形式传入。")

    return payload


def main():
    # 1) 加载配置
    config_path = "Config/base_cls2img.yaml"
    args = load_args_from_file(config_path)

    # 2) 指定模型与评测尺寸（重要：FID-256 用 256 尺寸）
    args.resume = True
    args.vit_folder = "./saved_networks/ImageNet_384_large.pth"  # 评估的 ckpt
    args.tome_keep_ratio = 1.0
    args.img_size = IMG_SIZE_FOR_FID   # <- 关键：生成也用 256，避免和真实分布分辨率不一致

    # 3) 单卡设备设置
    if torch.cuda.is_available():
        args.gpu = getattr(args, "gpu", 0)   # 你之前用 3，这里按需改
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.gpu = -1
        args.device = torch.device("cpu")

    args.world_size = getattr(args, "world_size", 1)
    args.rank = getattr(args, "rank", 0)
    args.local_rank = getattr(args, "local_rank", 0)
    args.is_master = True
    args.is_multi_gpus = False
    args.nb_gpus = 1

    # 4) 可复现
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 5) 评测数据根目录（很多仓库用 args.eval_folder / args.data_folder）
    #    确保 ../datasets/imagenet 下有 val/ 子目录
    args.eval_folder = EVAL_IMAGENET_ROOT
    # 有的实现用 args.data = "imagenet_256"；有的用 "imagenet"
    # 这里保持原仓库风格：后续 eval(data=args.data.split('_')[0]) 会得到 "imagenet"
    args.data = "imagenet_256"

    # 6) 初始化训练器
    trainer = MaskGIT(args)

    # 7) 构建采样器（与训练同款）
    if args.sampler == "halton":
        from Sampler.halton_sampler import HaltonSampler
        eval_sampler = HaltonSampler(
            sm_temp_min=1., sm_temp_max=1., temp_warmup=0, w=0,
            sched_pow=args.sched_pow, step=args.step,
            randomize=args.randomize, top_k=-1
        )
    else:
        from Sampler.confidence_sampler import ConfidenceSampler
        eval_sampler = ConfidenceSampler(
            sm_temp=1., w=0, r_temp=args.r_temp,
            sched_mode=args.sched_mode, step=args.step
        )

    # 8) 尝试把真实分布（mu/sigma）传递/注入
    fid_kwargs = try_inject_real_stats(trainer, REAL_STATS_PATH)

    # 9) 跑评测
    print(f"Starting evaluation for model in {args.vit_folder}...")
    # 注意几个点：
    # - split: 用 "val"（大多数实现里 FID 对 real 是 val 集）
    # - data: 传 "imagenet"（很多仓库内部用这个 key 去选数据集）
    # - num_images: 50k（可改 10k 快速看趋势）
    # - save_exemple/compute_pr: 关掉可加速
    eval_call_kwargs = dict(
        sampler=eval_sampler,
        num_images=NUM_IMAGES_FOR_FID,
        save_exemple=False,
        compute_pr=False,
        split="val",
        mode="c2i",
        data=args.data.split("_")[0],  # -> "imagenet"
    )
    # 如果 eval 支持 fid_real_stats，合并传进去；不支持也不会报错（Python 会忽略未知 kw 的写法不成立，
    # 所以我们 try 一下；不支持时用不带 fid_real_stats 的版本再调用一次）
    try:
        metrics = trainer.eval(**eval_call_kwargs, **fid_kwargs)
    except TypeError:
        # 老实现不接受 fid_real_stats 关键字参数，退回不带它的调用（真实分布已尝试注入）
        metrics = trainer.eval(**eval_call_kwargs)

    # 10) 打印结果
    print("Evaluation finished.")
    print(
        "FID: {fid} | IS: {iscore} | num_fake_seen: {nfake} | num_real_seen: {nreal}".format(
            fid=metrics.get("FID", "N/A"),
            iscore=metrics.get("IS", "N/A"),
            nfake=metrics.get("num_fake_seen", "N/A"),
            nreal=metrics.get("num_real_seen", "N/A"),
        )
    )

if __name__ == '__main__':
    main()
