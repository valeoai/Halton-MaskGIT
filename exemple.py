import torch
from Utils.utils import load_args_from_file
from Utils.viz import show_images_grid
from huggingface_hub import hf_hub_download

from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

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


# Initialisation of the model
model = MaskGIT(args)

# select your scheduler (Halton is better)
sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0, w=2,
                        sched_pow=2, step=32, randomize=True, top_k=-1)

# [goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear]
labels = torch.LongTensor([1, 7, 282, 604, 724, 179, 681, 850]).to(args.device)

gen_images = sampler(trainer=model, nb_sample=8, labels=labels, verbose=True)[0]
show_images_grid(gen_images)
