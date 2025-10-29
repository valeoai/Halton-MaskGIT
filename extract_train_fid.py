import torch
import argparse
from tqdm import tqdm
from Dataset.dataloader import get_data
from Metrics.inception_metrics import MultiInceptionMetrics
import os


class SampleAndEval:
    def __init__(self, device):
        super().__init__()
        self.inception_metrics = MultiInceptionMetrics(
            device=device, compute_manifold=False, num_classes=1000,
            num_inception_chunks=10, manifold_k=3, model="inception",
        )

        self.device = device

    def compute_images_features_and_save(self, dataloader):
        bar = tqdm(dataloader, leave=False, desc="Computing images features")
        for images, labels in bar:
            self.inception_metrics.update(images, image_type="real")

        real_features = torch.cat(self.inception_metrics.real_features, dim=0)
        real_features_mean = real_features.mean(dim=0)
        real_features_cov = self.inception_metrics.cov(real_features, real_features_mean)

        data = {
            "mu": real_features_mean.cpu(),
            "cov": real_features_cov.cpu(),
        }

        # Save the dictionary to a file
        torch.save(data, "./saved_networks/ImageNet_384_val_stats.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder",  type=str, default="",  help="data source")
    parser.add_argument("--img-size",     type=int, default=256, help="image size")
    parser.add_argument("--bsize",        type=int, default=128, help="batch size")
    parser.add_argument("--num-workers",   type=int, default=8, help="batch size")
    args = parser.parse_args()

    args.data_folder = "../datasets/imagenet"
    args.img_size = 384
    data_loader = get_data(
        "imagenet", img_size=args.img_size, data_folder=args.data_folder, bsize=args.bsize,
        num_workers=args.num_workers, is_multi_gpus=False, seed=-1,split="val" 
    )[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = SampleAndEval(device)
    sae.compute_images_features_and_save(data_loader)
