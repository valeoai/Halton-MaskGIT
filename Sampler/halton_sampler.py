import torch
import random
import math
import numpy as np
from tqdm import tqdm


class HaltonSampler(object):
    """
    Halton Sampler is a sampling strategy for iterative masked token prediction in image generation models.

    It follows a Halton-based scheduling approach to determine which tokens to predict at each step.
    """

    def __init__(self, sm_temp_min=1, sm_temp_max=1, temp_pow=1, w=4, sched_pow=2.5, step=64, randomize=False, top_k=-1, temp_warmup=0):
        """
        Initializes the HaltonSampler with configurable parameters.

        params:
            sm_temp_min  -> float: Minimum softmax temperature.
            sm_temp_max  -> float: Maximum softmax temperature.
            temp_pow     -> float: Exponent for temperature scheduling.
            w            -> float: Weight parameter for the CFG.
            sched_pow    -> float: Exponent for mask scheduling.
            step         -> int: Number of steps in the sampling process.
            randomize    -> bool: Whether to randomize the Halton sequence for the generation.
            top_k        -> int: If > 0, applies top-k sampling for token selection.
            temp_warmup  -> int: Number of initial steps where temperature is reduced.
        """
        super().__init__()
        self.sm_temp_min = sm_temp_min
        self.sm_temp_max = sm_temp_max
        self.temp_pow = temp_pow
        self.w = w
        self.sched_pow = sched_pow
        self.step = step
        self.randomize = randomize
        self.top_k = top_k
        self.basic_halton_mask = None  # Placeholder for the Halton-based mask
        self.temp_warmup = temp_warmup
        # Linearly interpolate the temperature over the sampling steps
        self.temperature = torch.linspace(self.sm_temp_min, self.sm_temp_max, self.step)

    def __str__(self):
        """Returns a string representation of the sampler configuration."""
        return f"Scheduler: halton, Steps: {self.step}, " \
               f"sm_temp_min: {self.sm_temp_min}, sm_temp_max: {self.sm_temp_max}, w: {self.w}, " \
               f"Top_k: {self.top_k}, temp_warmup: {self.temp_warmup}"

    def __call__(self, trainer, init_code=None, nb_sample=50, labels=None, verbose=True):
        """
        Runs the Halton-based sampling process.

        Args:
            trainer    -> MaskGIT: The model trainer.
            init_code  -> torch.Tensor: Pre-initialized latent code.
            nb_sample  -> int: Number of images to generate.
            labels     -> torch.Tensor: Class labels for conditional generation.
            verbose    -> bool: Whether to display progress.

        Returns:
            Tuple: Generated images, list of intermediate codes, list of masks used during generation.
        """

        # Build the Halton mask if not already created
        if self.basic_halton_mask is None:
            self.basic_halton_mask = self.build_halton_mask(trainer.input_size)

        trainer.vit.eval()
        l_codes = []  # List to store intermediate latent codes
        l_mask = []  # Save the intermediate masks
        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850] + [random.randint(0, 999) for _ in range(nb_sample - 9)]
                labels = torch.LongTensor(labels[:nb_sample]).to(trainer.args.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(trainer.args.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code
            else:  # Initialize a code
                code = torch.full((nb_sample, trainer.input_size, trainer.input_size),
                                  trainer.args.mask_value).to(trainer.args.device)

            # Randomizing the mask sequence if enabled
            if self.randomize:
                randomize_mask = torch.randint(0, trainer.input_size ** 2, (nb_sample,))
                halton_mask = torch.zeros(nb_sample, trainer.input_size ** 2, 2, dtype=torch.long)
                for i_h in range(nb_sample):
                    rand_halton = torch.roll(self.basic_halton_mask.clone(), randomize_mask[i_h].item(), 0)
                    halton_mask[i_h] = rand_halton
            else:
                halton_mask = self.basic_halton_mask.clone().unsqueeze(0).expand(nb_sample, trainer.input_size ** 2, 2)

            # Softmax temperature
            bar = tqdm(range(self.step), leave=False) if verbose else range(self.step)
            prev_r = 0
            for index in bar:
                # Compute the number of tokens to predict
                ratio = ((index + 1) / self.step)
                r = 1 - (torch.arccos(torch.tensor(ratio)) / (math.pi * 0.5))
                r = int(r * (trainer.input_size ** 2))
                r = max(index + 1, r)

                # Construct the mask for the current step
                _mask = halton_mask.clone()[:, prev_r:r]
                mask = torch.zeros(nb_sample, trainer.input_size, trainer.input_size, dtype=torch.long)
                for i_mask in range(nb_sample):
                    mask[i_mask, _mask[i_mask, :, 0], _mask[i_mask, :, 1]] = 1
                mask = mask.bool()

                # Choose softmax temperature
                _temp = self.temperature[index] ** self.temp_pow
                if index < self.temp_warmup:
                    _temp *= 0.5  # Reduce temperature during warmup

                if self.w != 0: # Model Prediction with cfg
                    with trainer.autocast:
                        logit = trainer.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                            torch.cat([labels, labels], dim=0),
                                            torch.cat([~drop, drop], dim=0))
                    logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                    logit = (1 + self.w) * logit_c - self.w * logit_u
                else:
                    with trainer.autocast:
                        logit = trainer.vit(code.clone(), labels, ~drop)

                # Compute probabilities using softmax
                prob = torch.softmax(logit * _temp, -1)
                if self.top_k > 0:# Apply top-k filtering
                    top_k_probs, top_k_indices = torch.topk(prob, self.top_k)
                    top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)
                    next_token_index = torch.multinomial(top_k_probs.view(-1, self.top_k), num_samples=1)
                    pred_code = top_k_indices.gather(-1, next_token_index.view(nb_sample, trainer.input_size ** 2, 1))
                else:
                    # Sample from the categorical distribution
                    pred_code = torch.distributions.Categorical(probs=prob).sample()

                # Update code with new predictions
                code[mask] = pred_code.view(nb_sample, trainer.input_size, trainer.input_size)[mask]

                l_codes.append(pred_code.view(nb_sample, trainer.input_size, trainer.input_size).clone())
                l_mask.append(mask.view(nb_sample, trainer.input_size, trainer.input_size).clone().float())
                prev_r = r
            # Decode the final prediction
            code = torch.clamp(code, 0, trainer.args.codebook_size - 1)
            x = trainer.ae.decode_code(code)
            x = torch.clamp(x, -1, 1)

        trainer.vit.train()  # Restore training mode
        return x, l_codes, l_mask

    @staticmethod
    def build_halton_mask(input_size, nb_point=10_000):
        """ Generate a halton 'quasi-random' sequence in 2D.
          :param
            input_size -> int: size of the mask, (input_size x input_size).
            nb_point   -> int: number of points to be sample, it should be high to cover the full space.
            h_base     -> torch.LongTensor: seed for the sampling.
          :return:
            mask -> Torch.LongTensor: (input_size x input_size) the mask where each value corresponds to the order of sampling.
        """

        def halton(b, n_sample):
            """Naive Generator function for Halton sequence."""
            n, d = 0, 1
            res = []
            for index in range(n_sample):
                x = d - n
                if x == 1:
                    n = 1
                    d *= b
                else:
                    y = d // b
                    while x <= y:
                        y //= b
                    n = (b + 1) * y - x
                res.append(n / d)
            return res

        # Sample 2D mask
        data_x = torch.asarray(halton(2, nb_point)).view(-1, 1)
        data_y = torch.asarray(halton(3, nb_point)).view(-1, 1)
        mask = torch.cat([data_x, data_y], dim=1) * input_size
        mask = torch.floor(mask)

        # remove duplicate
        indexes = np.unique(mask.numpy(), return_index=True, axis=0)[1]
        mask = [mask[index].numpy().tolist() for index in sorted(indexes)]
        return torch.LongTensor(np.array(mask))

# ========= Halton-ToMe helpers =========
def _halton_centers_and_assignments_sq(h_or_w: int, num_centers: int, device):
    """
    使用 HaltonSampler.build_halton_mask(h) 生成 [N,2] (y,x) 样本点，
    前 num_centers 个作为中心，其余 token 依据 2D 欧氏距离分配到最近中心。
    仅支持正方形网格 (h==w) 的场景；N = h*h。
    返回:
      centers_yx: [M,2]
      cluster_ids: [N] 中每个 token 的簇 id (0..M-1)
      counts: [M] 每个簇的成员数
    """
    N = h_or_w * h_or_w
    # 总的 token 数
    M = max(1, int(round(N * 1.0)))  # 默认先占位，实际由外层控制 keep_ratio 决定
    # 这里仅生成 Halton mask，真正的 M 在外部生效
    # 确保至少是 1，避免出现 M = 0 的情况
    halton_mask = HaltonSampler.build_halton_mask(h_or_w).to(device)  # [N,2] (y,x)

    def _build_with_M(M_): # M_ 保留的中心数
        centers_yx = halton_mask[:M_].to(device)               # [M,2]
        # 取前 M 个作为中心
        yy, xx = torch.meshgrid(
            torch.arange(h_or_w, device=device),
            torch.arange(h_or_w, device=device),
            indexing="ij"
        )
        coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2).float()  # [N,2]
        # 计算 N×M 距离并找最近中心
        dist = torch.cdist(coords, centers_yx.float(), p=2)             # [N,M]
        # torch.cdist 计算两两之间的距离。
        # 得到一个 [N, M] 的矩阵：第 i 行表示第 i 个 token 到所有 M 个中心的距离。
        cluster_ids = dist.argmin(dim=1)                                 # [N]
        # cluster_ids 的形状是 [N]，每个元素是 0..M_-1 之间的整数，代表 token 属于哪个簇
        counts = torch.bincount(cluster_ids, minlength=M_).clamp_min(1) # [M]
        # counts 的形状是 [M]，每个元素表示对应簇的成员数量，确保最少为 1
        return centers_yx, cluster_ids, counts
        # centers_yx：中心坐标（形状 [M, 2]，比如 (y, x)）。
        # cluster_ids：每个 token 的簇 ID（形状 [N]）。
        # counts：每个簇的 token 数（形状 [M]，不会小于 1）。

    return _build_with_M
    # 用法：build = _halton_centers_and_assignments_sq(h, None, device); centers, ids, cnt = build(M)
    

def _merge_tokens_spatial(x_spatial, cluster_ids, counts):
    """
    均值合并同簇 token。
    x_spatial: [B,N,C]; cluster_ids: [N]; counts: [M]
    返回 xm: [B,M,C]
    """
    B, N, C = x_spatial.shape
    M = counts.numel()
    xm = x_spatial.new_zeros(B, M, C)
    idx = cluster_ids.view(1, N, 1).expand(B, N, C)           # [B,N,C]
    xm.scatter_add_(1, idx, x_spatial)                        # 同簇求和
    # xm: [B,M,C]，每个中心的特征是其簇内所有 token 特征的和
    xm = xm / counts.view(1, M, 1)                            # 均值
    # 得到均值特征
    return xm


def _unmerge_tokens_spatial(xm_spatial, cluster_ids):
    """
    反合并：把中心特征复制回各成员。
    xm_spatial: [B,M,C]; cluster_ids: [N]
    返回 x: [B,N,C]
    """
    B, M, C = xm_spatial.shape
    N = cluster_ids.numel()
    idx = cluster_ids.view(1, N, 1).expand(B, N, C)           # [B,N,C] 值域 0..M-1
    x = xm_spatial.gather(dim=1, index=idx)                   # 复制中心到成员
    return x
# ======================================
