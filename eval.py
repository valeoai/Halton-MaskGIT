# filepath: eval.py
import torch
from Trainer.cls_trainer import MaskGIT
from Utils.utils import load_args_from_file

def main():
    # 1. 加载参数
    # 确保这里的参数（特别是模型尺寸和数据集）与您训练时使用的参数一致
    config_path = "Config/base_cls2img.yaml"        # Path to your config file
    args = load_args_from_file(config_path)

    # 2. 指定要评估的模型路径
    #    - args.vit_folder 应该是包含模型文件的目录
    #    - args.resume 必须为 True 才能加载模型
    args.resume = True
    # 例如，评估 EMA 模型
    args.vit_folder = "./saved_networks/ImageNet_384_large.pth"   
    args.tome_keep_ratio = 1.0
    
    # 如果要评估非 EMA 模型，请确保 vit_folder 指向正确的 current.pth 或 epoch_xxx.pth
    # 如果要评估 EMA 模型，代码会自动寻找目录下的 ema.pth

    # 3. 初始化训练器
    #    - 设置 is_master=True 以确保评估和日志记录被执行
    #    - 设置 data_folder 以加载评估所需的数据集
    args.is_master = True
    args.is_multi_gpus = False # 在单卡上评估
    
    # ---- 设备与分布式占位（单卡评估）----
    if torch.cuda.is_available():
        # 若你的 config 里没有 gpu/local_rank，就直接写死 0
        args.gpu = getattr(args, "gpu", 3)
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.gpu = -1
        args.device = torch.device("cpu")

    # 单机单卡时，把这些值都设好，避免内部取用
    args.world_size = getattr(args, "world_size", 1)
    args.rank = getattr(args, "rank", 0)
    args.local_rank = getattr(args, "local_rank", 0)   # 有些代码会读这个
    args.is_master = True
    args.is_multi_gpus = False
    args.nb_gpus = 1

    # 可复现和 cudnn 设置（可选，但建议）
    import random, numpy as np
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    trainer = MaskGIT(args)

    # 4. 运行评估
    print(f"Starting evaluation for model in {args.vit_folder}...")
    
    # 从 fit 方法中借鉴 sampler 的创建方式
    if args.sampler == "halton":
        from Sampler.halton_sampler import HaltonSampler
        eval_sampler = HaltonSampler(
            sm_temp_min=1., sm_temp_max=1., temp_warmup=0, w=0, sched_pow=args.sched_pow,
            step=args.step, randomize=args.randomize, top_k=-1
        )
    else: # confidence sampler
        from Sampler.confidence_sampler import ConfidenceSampler
        eval_sampler = ConfidenceSampler(
            sm_temp=1., w=0, r_temp=args.r_temp, sched_mode=args.sched_mode, step=args.step
        )

    # 调用基类 Trainer 中的 eval 方法
    metrics = trainer.eval(
        sampler=eval_sampler, 
        num_images=50000,  # 通常在 ImageNet 上使用 10k 或 50k 张图片计算 FID
        save_exemple=False, 
        compute_pr=False,
        split="Test", 
        mode="c2i", 
        data=args.data.split("_")[0]
    )

    # 5. 打印结果
    print("Evaluation finished.")
    print(
        "FID: {fid} | IS: {iscore} | "
        "num_fake_seen: {nfake} | num_real_seen: {nreal}".format(
            fid=metrics.get("FID", "N/A"),
            iscore=metrics.get("IS", "N/A"),
            nfake=metrics.get("num_fake_seen", "N/A"),
            nreal=metrics.get("num_real_seen", "N/A"),
        )
    )

if __name__ == '__main__':
    main()