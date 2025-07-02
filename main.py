# Main file to launch training or evaluation
import os
import random

import numpy as np
import argparse

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from torch.distributed import init_process_group, destroy_process_group


def ddp_setup():
    """ Initialization of the multi_gpus training"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def launch_multi_main(args):
    """ Launch multi training"""

    ddp_setup()
    args.device = int(os.environ["LOCAL_RANK"])
    args.global_rank = int(os.environ["RANK"])
    args.is_master = args.global_rank == 0
    args.nb_gpus = torch.distributed.get_world_size()
    args.bsize = args.global_bsize // args.nb_gpus
    if args.is_master:
        print(f"{args.nb_gpus} GPU(s) found, launch DDP")
    args.num_nodes = torch.distributed.get_world_size() // 8
    main(args)
    destroy_process_group()


def main(args):
    """ Main function: Train or eval MaskGIT """
    if args.mode == "cls-to-img":
        from Trainer.cls_trainer import MaskGIT
    elif args.mode == "txt-to-img":
        from Trainer.txt_trainer import MaskGIT
    else:
        raise "What is this mode ?????"

    maskgit = MaskGIT(args)

    if not args.test_only:
        maskgit.fit()

    eval_sampler = maskgit.sampler
    m = maskgit.eval(sampler=eval_sampler, num_images=50_000, save_exemple=False, compute_pr=False,
                     split="Test", mode="c2i", data=args.data.split("_")[0])
    
    if maskgit.args.is_master:
        selected_hparams = {
            'Vit size': maskgit.args.vit_size,
            'img-size': maskgit.args.img_size,
            'sampler': maskgit.args.sampler,
            'cfg_w': maskgit.args.cfg_w,
            'step': int(maskgit.args.step),
            'iter': int(maskgit.args.iter)
        }

        selected_metrics = {
            'FID': float(m['FID']),
            'IS': float(m['IS'])
        }
        version = sum(1 for name in os.listdir(maskgit.args.writer_log) if os.path.isdir(os.path.join(maskgit.args.writer_log, name)) and name.startswith("version_"))
        maskgit.writer.add_hparams(
            hparam_dict=selected_hparams,
            metric_dict=selected_metrics,
            run_name=f"version_{version}"
        )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument("--vqgan-folder",  type=str,   default="",           help="folder of the pretrained VQGAN")
    parser.add_argument("--vit-folder",    type=str,   default="",           help="folder where to save the Transformer")
    parser.add_argument("--writer-log",    type=str,   default="",           help="folder where to store the logs")
    parser.add_argument("--data-folder",   type=str,   default="",           help="folder containing the dataset")
    parser.add_argument("--eval-folder",   type=str,   default="",           help="folder containing data for evaluation")
    # Mode
    parser.add_argument("--mode",          type=str,   default="cls-to-img", help="cls-to-img|txt-to-img")
    parser.add_argument("--dtype",         type=str,   default="bfloat16",   help="precision")
    parser.add_argument("--test-only",     action='store_true',              help="only evaluate the model")
    parser.add_argument("--debug",         action='store_true',              help="")
    parser.add_argument("--resume",        action='store_true',              help="resume training of the model")
    parser.add_argument("--compile",       action='store_true',              help="compile the network pytorch 2.0")
    parser.add_argument("--use-ema",       action='store_true',              help="use an ema or not")
    # Model and Flop
    parser.add_argument("--vit-size",      type=str,   default="base",       help="size of the vit")
    parser.add_argument('--f-factor',      type=int,   default=16,            help="image size")
    parser.add_argument("--codebook-size", type=int,   default=1024,         help="codebook size")
    parser.add_argument("--mask-value",    type=int,   default=1024,         help="number of epoch")
    parser.add_argument("--register",      type=int,   default=1,            help="number of register")
    parser.add_argument("--dropout",       type=float, default=0.,           help="dropout in the transformer")
    parser.add_argument("--proj",          type=float, default=1,            help="dropout in the transformer")
    # Data
    parser.add_argument("--data",          type=str,   default="imagenet",   help="dataset on which dataset to train")
    parser.add_argument("--nb-class",      type=int,   default=1_000,        help="number of classes")
    parser.add_argument("--num-workers",   type=int,   default=8,            help="number of workers")
    parser.add_argument('--img-size',      type=int,   default=256,          help="image size")
    parser.add_argument('--seed',          type=int,   default=-1,           help="fix seed")
    parser.add_argument("--global-bsize",  type=int,   default=256,          help="batch size")
    # Learning
    parser.add_argument("--epoch",         type=int,   default=10_000,         help="number of epoch")
    parser.add_argument("--drop-label",    type=float, default=0.1,          help="drop rate for cfg")
    parser.add_argument("--grad-cum",      type=int,   default=1,            help="accumulate gradient")
    parser.add_argument("--sched-mode",    type=str,   default="arccos",     help="scheduler mode when sampling")
    parser.add_argument("--warm-up",       type=int,   default=2_500,        help="lr warmup")
    parser.add_argument("--max-iter",      type=int,   default=750_000,      help="max iteration")
    parser.add_argument("--lr",            type=float, default=1e-4,         help="learning rate to train the transformer")
    parser.add_argument("--grad-clip",     type=float, default=1,            help="drop rate for cfg")
    # Sampler
    parser.add_argument("--sampler",       type=str,   default="confidence", help="type of sampler")
    parser.add_argument("--temp-warmup",   type=int,   default=0,            help="decrease the temperature of x iter")
    parser.add_argument("--step",          type=int,   default=12,           help="number of step for sampling")
    parser.add_argument('--top-k',         type=int,   default=-1,           help="top_k")
    parser.add_argument('--sched-pow',     type=float,   default=3,          help="scheduler incrementation")
    parser.add_argument("--cfg-w",         type=float, default=3,            help="classifier free guidance wight")
    parser.add_argument("--r-temp",        type=float, default=4.5,          help="Gumbel noise temperature when sampling")
    parser.add_argument("--sm-temp",       type=float, default=1.,           help="temperature before softmax when sampling")
    parser.add_argument("--sm-temp-min",   type=float, default=1.,           help="temperature before softmax when sampling")
    parser.add_argument("--randomize",     action='store_true',              help="only evaluate the model")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.iter = 0
    args.global_epoch = 0

    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True

    world_size = torch.cuda.device_count()
    if world_size > 1:
        args.is_multi_gpus = True
        launch_multi_main(args)
    else:
        print(f"{world_size} GPU found")
        args.global_rank = 0
        args.num_nodes = 1
        args.is_master = True
        args.is_multi_gpus = False
        args.nb_gpus = 1
        args.bsize = args.global_bsize
        main(args)
