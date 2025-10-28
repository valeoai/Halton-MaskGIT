import os
from typing import Any

import numpy as np
from cleanfid import resize as clean_resize
import torch.distributed as dist

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE

from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

if _SCIPY_AVAILABLE:
    import scipy

import sklearn.metrics


def custom_resize_norm(img_batch, norm=True, size=(299, 299)):
    l_resized_batch = []
    if norm:
        img_batch = (img_batch + 1) / 2
        img_batch = torch.clip(img_batch * 255, 0, 255)

    for idx in range(len(img_batch)):
        curr_img = img_batch[idx]
        img_np = curr_img.cpu().numpy().transpose((1, 2, 0))
        img_resize = clean_resize.make_resizer("PIL", False, "bicubic", size)(img_np)
        l_resized_batch.append(torch.tensor(img_resize.transpose((2, 0, 1))).unsqueeze(0))
    return torch.cat(l_resized_batch, dim=0)


class MatrixSquareRoot(Function):
    @staticmethod
    def forward(ctx: Any, input_data: Tensor) -> Tensor:
        m = input_data.detach().cpu().numpy().astype(np.float64)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
        sqrtm = torch.from_numpy(scipy_res.real).to(input_data)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


class MultiInceptionMetrics(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    def __init__(self, device, compute_manifold=False, num_classes=1000, num_inception_chunks=10, manifold_k=3,
                 model="inception", **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.num_inception_chunks = num_inception_chunks
        self.manifold_k = manifold_k
        self.compute_manifold = compute_manifold

        if model == "inception":
            class NoTrainInceptionV3(FeatureExtractorInceptionV3):
                def __init__(self, name, features_list, feature_extractor_weights_path=None):
                    super().__init__(name, features_list, feature_extractor_weights_path)

                @staticmethod
                def preprocess(image):
                    """ convert from {(size, size), [-1, 1], float32} --> {(299, 299), [0, 255], int8} """
                    image = custom_resize_norm(image).to(torch.uint8).to(device)
                    return image

                def forward(self, x: Tensor) -> Tensor:
                    out = super().forward(self.preprocess(x))
                    return out

            self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048", "logits_unbiased"])
            self.inception = self.inception.to(device)
            self.inception.eval()

        elif model == "dinov2":
            class _Dinov2(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(device)
                    self.model.eval()
                    self.model.linear_head = nn.Identity()

                @staticmethod
                def preprocess(images):
                    """ convert from {(size, size), [-1, 1], float32} --> {(224, 224), [-1, 1], float32} """
                    new_size = ((images.size(-2) // 14) * 14, (images.size(-1) // 14) * 14)
                    images = torch.nn.functional.interpolate(images, new_size, mode='bilinear')
                    return images

                def forward(self, x) -> Any:
                    out = self.model(self.preprocess(x))
                    return out, torch.zeros_like(out)

            self.inception = _Dinov2().to(device)

        elif model == "clip":
            from transformers import CLIPModel

            class _Clip(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                    self.model.eval()

                @staticmethod
                def preprocess(images):
                    """ convert from {(size, size), [-1, 1], float32} --> {(224, 224), [-1, 1], float32} """
                    images = torch.nn.functional.interpolate(images, (224, 224))
                    return images

                def forward(self, x) -> Any:
                    out = self.model.get_image_features(self.preprocess(x))
                    return out, torch.zeros_like(out)

            self.inception = _Clip().to(device)

        else:
            print("feature extractor does not exist")
            exit()

        # variable to stock the features
        self.real_features = []
        self.fake_features = []
        self.fake_logits = []
        self.add_state("count", torch.tensor(0.0, device=device), dist_reduce_fx="sum")

    def update(self, images, image_type="fake") -> None:
        # extract the features
        features, logits = self.inception(images)

        features = features.view(features.size(0), -1)

        if features.dim() == 1:
            features = features.unsqueeze(0)
            logits = logits.unsqueeze(0)

        self.count += features.size(0) 

        if image_type == "real":
            self.real_features.append(features)
        elif image_type == "fake":
            self.fake_features.append(features)
            self.fake_logits.append(logits)

    def fid(self, real_features, fake_features):
        if not self.real_features:
            if os.path.exists("./saved_networks/ImageNet_256_train_stats.pt"):
                loaded_data = torch.load("./saved_networks/ImageNet_256_train_stats.pt", weights_only=True)
                real_features_mean = loaded_data["mu"].to(fake_features.device)
                real_features_cov = loaded_data["cov"].to(fake_features.device)
            else:
                print("Pre-computed stats does not exist")
                exit()
        else:
            real_features_mean = real_features.mean(dim=0)
            real_features_cov = self.cov(real_features, real_features_mean)
        fake_features_mean = fake_features.mean(dim=0)
        fake_features_cov = self.cov(fake_features, fake_features_mean)
        return self._compute_fid(real_features_mean, real_features_cov, fake_features_mean, fake_features_cov).item()

    def cov(self, features, features_mean):
        features = features - features_mean
        return torch.mm(features.t(), features) / (features.size(0) - 1)

    def _compute_fid(self, mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = 1e-6) -> Tensor:
        diff = mu1 - mu2

        covmean = sqrtm(sigma1.mm(sigma2))
        # Product might be almost singular
        if not torch.isfinite(covmean).all():
            rank_zero_info(f"FID calculation produces singular product")
            offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
            covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

        tr_covmean = torch.trace(covmean)
        return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean

    def inception_score(self, logits):
        idx = torch.randperm(logits.size(0))
        logits = logits[idx]

        prob = logits.softmax(dim=1)
        log_prob = logits.log_softmax(dim=1)

        prob = prob.chunk(self.num_inception_chunks, dim=0)
        log_prob = log_prob.chunk(self.num_inception_chunks, dim=0)
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]

        kl_ = [p * (log_p - torch.log(m_p)) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = torch.stack(kl_).mean()
        return kl.item()

    def compute_pairwise_distance(self, data_x, data_y=None):
        if data_y is None:
            data_y = data_x
        dists = sklearn.metrics.pairwise_distances(data_x, data_y, metric="euclidean", n_jobs=8)
        return dists

    def get_kth_value(self, unsorted, k, axis=-1):
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values

    def compute_nearest_neighbour_distances(self, input_features, nearest_k):
        distances = self.compute_pairwise_distance(input_features)
        radii = self.get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii

    def compute_prdc(self, real_features, fake_features, nearest_k):
        real_nearest_neighbour_distances = self.compute_nearest_neighbour_distances(real_features, nearest_k)
        fake_nearest_neighbour_distances = self.compute_nearest_neighbour_distances(fake_features, nearest_k)
        distance_real_fake = self.compute_pairwise_distance(real_features, fake_features)

        precision = (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0).mean()

        recall = (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1).mean()

        density = (1.0 / float(nearest_k)) * (
                    distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1))
        density = density.sum(axis=0).mean()

        coverage = (distance_real_fake.min(axis=1) < real_nearest_neighbour_distances).mean()

        return precision, recall, density, coverage

    def manifold_metrics(self, real_features, fake_features, nearest_k, num_splits=5):
        real_features = real_features.chunk(num_splits, dim=0)
        fake_features = fake_features.chunk(num_splits, dim=0)
        precision, recall, density, coverage = [], [], [], []
        for real, fake in zip(real_features, fake_features):
            p, r, d, c = self.compute_prdc(real.cpu().numpy(), fake.cpu().numpy(), nearest_k=nearest_k)
            precision.append(torch.tensor(p, device=real.device))
            recall.append(torch.tensor(r, device=real.device))
            density.append(torch.tensor(d, device=real.device))
            coverage.append(torch.tensor(c, device=real.device))
        return (
            torch.stack(precision).mean().item(),
            torch.stack(recall).mean().item(),
            torch.stack(density).mean().item(),
            torch.stack(coverage).mean().item(),
        )

    def compute(self) -> dict:
        # remove network from cuda
        self.free()

        # Compute the actual score
        output_metrics = {}
        fake_features = torch.cat(self.fake_features, dim=0)
        fake_features = self.gather_and_concat(fake_features)

        fake_logits = torch.cat(self.fake_logits, dim=0)
        fake_logits = self.gather_and_concat(fake_logits)

        idx = torch.randperm(fake_features.size(0))  # shuffle the image!
        fake_features = fake_features[idx]
        fake_logits = fake_logits[idx]

        if self.real_features:
            real_features = torch.cat(self.real_features, dim=0)
            real_features = self.gather_and_concat(real_features)
            real_features = real_features[idx]
        else:
            real_features = None

        output_metrics["FID"] = self.fid(real_features, fake_features)
        output_metrics["IS"] = self.inception_score(fake_logits)
        if self.compute_manifold:
            output_metrics["Prec"], output_metrics["Recall"], output_metrics["Density"], output_metrics["Cov"] = \
                self.manifold_metrics(real_features, fake_features, self.manifold_k)

        return output_metrics

    def free(self):
        self.inception.to('cpu')
        del self.inception  # delete the model

    @staticmethod
    def gather_and_concat(tensor):
        """
        Gather a tensor from all devices and concatenate the results.

        Args:
            tensor (torch.Tensor): The tensor to gather and concatenate across devices.

        Returns:
            torch.Tensor: The concatenated tensor containing data from all devices.
        """
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # List to hold gathered tensors from each device
            gathered_tensors = [
                torch.empty_like(tensor, device=tensor.device).clone() for _ in range(dist.get_world_size())
            ]
            # Gather tensors from all devices
            dist.all_gather(gathered_tensors, tensor.contiguous())
            # Concatenate along the first dimension
            concatenated_tensor = torch.cat(gathered_tensors, dim=0)
            return concatenated_tensor
        else:
            return tensor
