import os
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

    # ------------------------------------------------------------------
    # Schedule pre-computation
    # ------------------------------------------------------------------

    def compute_schedule(self, input_size, nb_sample=1):
        """Pre-compute U_t and M_t masks for every decoding step.

        U_t  — newly unmasked (released) tokens at step t:  bool (nb_sample, h, w)
        M_t  — all tokens unmasked up to and including step t: bool (nb_sample, h, w)

        This uses the non-randomized Halton order.  Call before __call__ when
        you need the full schedule for e.g. partial-update planning.

        Returns:
            l_U_t: list[Tensor] length step, each (nb_sample, input_size, input_size)
            l_M_t: list[Tensor] length step, each (nb_sample, input_size, input_size)
        """
        if self.basic_halton_mask is None:
            self.basic_halton_mask = self.build_halton_mask(input_size)

        # shape (nb_sample, n_tokens, 2)  — same order for every sample
        halton_mask = (
            self.basic_halton_mask.clone()
            .unsqueeze(0)
            .expand(nb_sample, input_size ** 2, 2)
        )

        l_U_t, l_M_t = [], []
        prev_r = 0
        for index in range(self.step):
            ratio = (index + 1) / self.step
            r = 1 - (torch.arccos(torch.tensor(ratio)) / (math.pi * 0.5))
            r = int(r * (input_size ** 2))
            r = max(index + 1, r)

            # U_t: token coordinates newly released at this step
            _u = halton_mask[:, prev_r:r]   # (nb_sample, n_new, 2)
            U_t = torch.zeros(nb_sample, input_size, input_size, dtype=torch.bool)
            for i in range(nb_sample):
                U_t[i, _u[i, :, 0], _u[i, :, 1]] = True

            # M_t: all released token coordinates up to this step
            _m = halton_mask[:, :r]         # (nb_sample, r, 2)
            M_t = torch.zeros(nb_sample, input_size, input_size, dtype=torch.bool)
            for i in range(nb_sample):
                M_t[i, _m[i, :, 0], _m[i, :, 1]] = True

            l_U_t.append(U_t)
            l_M_t.append(M_t)
            prev_r = r

        return l_U_t, l_M_t

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------

    def __call__(self, trainer, init_code=None, nb_sample=50, labels=None,
                 verbose=True, partial_update=None):
        """
        Runs the Halton-based sampling process.

        Args:
            trainer        -> MaskGIT: The model trainer.
            init_code      -> torch.Tensor: Pre-initialized latent code.
            nb_sample      -> int: Number of images to generate.
            labels         -> torch.Tensor: Class labels for conditional generation.
            verbose        -> bool: Whether to display progress.
            partial_update -> bool|None: If True, pass active_mask (U_t) to the
                             transformer so attention uses Q-only-active mode.
                             If None (default), fall back to the env var
                             HALTON_PARTIAL_UPDATE ("1" -> True, else False).
                             This lets the fixed eval pipeline
                             (Metrics/sample_and_eval.py, which does not pass
                             this arg) be toggled without code edits, while
                             explicit callers (e.g. eval.py) keep full control.

        Returns:
            Tuple: (generated images,
                    list of per-step predicted codes,
                    l_U_t — list of per-step newly-released masks,
                    l_M_t — list of per-step cumulative released masks)
        """

        # Resolve partial_update from env when the caller didn't specify it.
        if partial_update is None:
            partial_update = os.environ.get("HALTON_PARTIAL_UPDATE", "0") == "1"

        # Cache-refresh cadence within the partial-update gate (steps 5..30).
        # HALTON_CACHE_REFRESH_N semantics:
        #   N missing / "" / <=0 : no periodic refresh (pure partial mode)
        #   N = 1                : every gated step refreshes -> equivalent to
        #                          full-update baseline (sanity check)
        #   N >= 2               : every N-th gated step forces vit_active_mask=None
        #                          so all blocks run full FFN and rewrite the cache;
        #                          remaining N-1 steps stay in partial+cached mode.
        try:
            cache_refresh_n = int(os.environ.get("HALTON_CACHE_REFRESH_N", "0"))
        except ValueError:
            cache_refresh_n = 0

        # Left edge of the partial-update gate (HALTON_PARTIAL_START_STEP).
        # 默认 5 —— 出自 analyze_ffn_delta_stability: 前几步整幅图还在剧烈变化,
        # 缓存里的东西陈旧得没法用。这个 5 从来没有被 50k FID 扫过, 本 env 就是
        # 为了扫它 (0 = 从第一步就开始缓存)。右端 31 不动: 最后一步必须全量。
        try:
            partial_start_step = int(os.environ.get("HALTON_PARTIAL_START_STEP", "5"))
        except ValueError:
            partial_start_step = 5
        partial_start_step = max(0, partial_start_step)

        # V-similarity token selection (HALTON_LAZY_VSIM=1) is the only consumer
        # of the step index: its recompute ratio decays with the decoding step
        # (LazyMAR's RETAIN_RATIO_SCHEDULE).  Pass the step only in that mode so
        # every other model / sampler path keeps its original call signature.
        lazy_vsim = os.environ.get("HALTON_LAZY_VSIM", "0") == "1"

        # ── LazyMAR CFG 残差缓存 (HALTON_CFG_DELTA=1) ──────────────────────
        # lazy 步只把 cond 半个 batch 送进 transformer, uncond 用上一整批步记下的
        # (uncond - cond) 残差补出来 —— LazyMAR forward_mae_decoder 的做法。
        #   HALTON_CFG_DELTA_START (默认 5)  之前的步一律整批 —— 残差就是在这些
        #       步里写下的 (每个整批步都无条件刷新残差), 所以门一开就有得用。
        #   HALTON_CFG_DELTA_N   门开之后, 每 N 个"CFG 自己说了算"的步刷新一次:
        #       <=0 (默认) 门开之后不再主动刷新, 一直沿用门开前最后一个整批步的残差
        #       N=1        每个这样的步都刷新 -> 从不 reuse, 等价于关掉本方案 (sanity check)
        #       N>=2       每 N 个刷新一次
        #       注: token cache 开着时, "CFG 自己说了算"的步 = 它的全量步; token
        #       cache 关着时 = 门内的每一步。
        # 不变式: **跑整批 [cond;uncond] 的步必须是 token cache 的全量步**。否则
        # 半 batch 的 K/V / restore 缓存会跟 2B 的输入对不上。所以 token cache 的
        # partial 步一律 reuse, CFG 自己的节奏只在它的全量步上生效。
        cfg_delta_on = os.environ.get("HALTON_CFG_DELTA", "0") == "1"
        if cfg_delta_on and self.w == 0:
            raise RuntimeError(
                "HALTON_CFG_DELTA 需要 CFG 打开 (w != 0): w=0 时 batch 里根本没有 "
                "uncond 分支, 残差缓存无从谈起。"
            )
        try:
            cfg_delta_n = int(os.environ.get("HALTON_CFG_DELTA_N", "0"))
        except ValueError:
            cfg_delta_n = 0
        try:
            cfg_delta_start = int(os.environ.get("HALTON_CFG_DELTA_START", "5"))
        except ValueError:
            cfg_delta_start = 5
        if cfg_delta_on and partial_update and cfg_delta_start > partial_start_step:
            # 不变式: 跑整批 [cond;uncond] 的步必须是 token cache 的全量步。
            # cfg 窗口比 partial 门开得晚 -> 中间那几步是 "token cache 的 partial
            # 步 + CFG 的整批步", 那一步的 K/V / restore 缓存 (半 batch) 跟 2B 的
            # 输入对不上, 会静默退化成缓存 miss。宁可报错。
            raise RuntimeError(
                f"HALTON_CFG_DELTA_START ({cfg_delta_start}) 必须 <= "
                f"HALTON_PARTIAL_START_STEP ({partial_start_step}): CFG 残差窗口"
                "要覆盖 token cache 的每一个 partial 步。"
            )

        # Build the Halton mask if not already created
        if self.basic_halton_mask is None:
            self.basic_halton_mask = self.build_halton_mask(trainer.input_size)

        trainer.vit.eval()
        # 清空各 Block 的缓存 (FFN delta / attn delta / layer output / LazyMAR
        # K-V), 避免上一次 __call__ 的残留影响。
        # 注意: DDP 不转发属性访问, 直接 hasattr(trainer.vit, ...) 在多卡 eval 下
        # 恒为 False —— 缓存会跨 batch 串台。必须先取出 .module。
        _vit = getattr(trainer.vit, "module", trainer.vit)
        if hasattr(_vit, "clear_ffn_cache"):
            _vit.clear_ffn_cache()
        l_codes = []   # intermediate predicted codes
        l_U_t = []     # per-step newly-released token mask  (U_t)
        l_M_t = []     # per-step cumulative released mask   (M_t)

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

            bar = tqdm(range(self.step), leave=False) if verbose else range(self.step)
            prev_r = 0
            prev_U_t = None  # U_t from the previous step (for active-mask union)
            gated_step_counter = 0  # counts steps that fall inside the partial-update gate
            cfg_gated_counter = 0   # 同上, 但只数 CFG 残差缓存自己决策的那些步
            for index in bar:
                # Compute the number of tokens to predict
                ratio = ((index + 1) / self.step)
                r = 1 - (torch.arccos(torch.tensor(ratio)) / (math.pi * 0.5))
                r = int(r * (trainer.input_size ** 2))
                r = max(index + 1, r)

                # U_t: newly released token positions at this step
                _u = halton_mask.clone()[:, prev_r:r]
                U_t = torch.zeros(nb_sample, trainer.input_size, trainer.input_size, dtype=torch.bool)
                for i_mask in range(nb_sample):
                    U_t[i_mask, _u[i_mask, :, 0], _u[i_mask, :, 1]] = True

                # M_t: all released token positions up to this step
                _m = halton_mask.clone()[:, :r]
                M_t = torch.zeros(nb_sample, trainer.input_size, trainer.input_size, dtype=torch.bool)
                for i_mask in range(nb_sample):
                    M_t[i_mask, _m[i_mask, :, 0], _m[i_mask, :, 1]] = True

                # active_mask sent to the transformer only in partial_update mode.
                # Active set = previous-step U_{t-1} ∪ current-step U_t
                # (consecutive U_t / U_{t-1} are disjoint by Halton construction,
                #  so per-row True-count stays uniform across the batch).
                # Step-level gating (from analyze_ffn_delta_stability):
                #   t >= partial_start_step : skip early unstable steps
                #                             (默认 5, HALTON_PARTIAL_START_STEP 可改)
                #   t < 31  : last step is computed in full
                vit_active_mask = None
                if partial_update and partial_start_step <= index < 31:
                    # Decide if this gated step is a "refresh" step that runs
                    # full FFN to rewrite cached_ffn_delta for all positions.
                    is_refresh = (
                        cache_refresh_n >= 1
                        and (gated_step_counter % cache_refresh_n == 0)
                    )
                    gated_step_counter += 1
                    if not is_refresh:
                        if prev_U_t is not None:
                            active = prev_U_t | U_t
                        else:
                            active = U_t
                        vit_active_mask = active.to(trainer.args.device)
                    # else: leave vit_active_mask=None so all blocks run the
                    # full-FFN branch (active_mask=None) and refresh the cache.

                # CFG 残差缓存的 refresh / reuse 门控 (见上面的不变式说明)。
                # 窗口的右端必须与上面 partial-update 的门 (5 <= index < 31) 完全
                # 一致, 否则会出现"token cache 的 partial 步 + CFG 的整批步"这种
                # 组合 —— 那一步的 K/V 缓存 (半 batch) 跟输入 (2B) 对不上。
                cfg_delta_reuse = False
                if cfg_delta_on and cfg_delta_start <= index < 31:
                    if vit_active_mask is not None:
                        # token cache 的 partial 步: batch 必须保持半份。
                        cfg_delta_reuse = True
                    else:
                        is_cfg_refresh = (cfg_delta_n >= 1
                                          and cfg_gated_counter % cfg_delta_n == 0)
                        cfg_gated_counter += 1
                        cfg_delta_reuse = not is_cfg_refresh

                # Choose softmax temperature
                _temp = self.temperature[index] ** self.temp_pow
                if index < self.temp_warmup:
                    _temp *= 0.5  # Reduce temperature during warmup

                step_kw = {"step": index, "total_steps": self.step} if lazy_vsim else {}

                if self.w != 0:  # Model prediction with CFG
                    am_cat = (
                        torch.cat([vit_active_mask, vit_active_mask], dim=0)
                        if vit_active_mask is not None else None
                    )
                    with trainer.autocast:
                        logit = trainer.vit(
                            torch.cat([code.clone(), code.clone()], dim=0),
                            torch.cat([labels, labels], dim=0),
                            torch.cat([~drop, drop], dim=0),
                            active_mask=am_cat,
                            cfg_pair=True,
                            cfg_delta=cfg_delta_reuse,
                            **step_kw,
                        )
                    logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                    logit = (1 + self.w) * logit_c - self.w * logit_u
                else:
                    with trainer.autocast:
                        logit = trainer.vit(
                            code.clone(), labels, ~drop,
                            active_mask=vit_active_mask,
                            **step_kw,
                        )

                # Compute probabilities using softmax
                prob = torch.softmax(logit * _temp, -1)
                if self.top_k > 0:  # Apply top-k filtering
                    top_k_probs, top_k_indices = torch.topk(prob, self.top_k)
                    top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)
                    next_token_index = torch.multinomial(top_k_probs.view(-1, self.top_k), num_samples=1)
                    pred_code = top_k_indices.gather(-1, next_token_index.view(nb_sample, trainer.input_size ** 2, 1))
                else:
                    # Sample from the categorical distribution.
                    # Pass logits (not probs): PyTorch then applies a
                    # numerically stable log_softmax internally. Passing the
                    # softmax probs instead can leave a row summing to <1 due to
                    # accumulation rounding (notably in bf16 over 16k classes),
                    # which fails Categorical's Simplex() check and, when it
                    # doesn't hard-fail, samples garbage tokens that pollute FID.
                    pred_code = torch.distributions.Categorical(logits=logit * _temp).sample()

                # Update code with new predictions at U_t positions
                code[U_t] = pred_code.view(nb_sample, trainer.input_size, trainer.input_size)[U_t]

                l_codes.append(pred_code.view(nb_sample, trainer.input_size, trainer.input_size).clone())
                l_U_t.append(U_t.clone().float())
                l_M_t.append(M_t.clone().float())
                prev_r = r
                prev_U_t = U_t  # remember for next step's active-mask union

            # Decode the final prediction
            code = torch.clamp(code, 0, trainer.args.codebook_size - 1)
            x = trainer.ae.decode_code(code)
            x = torch.clamp(x, -1, 1)

        trainer.vit.train()  # Restore training mode
        return x, l_codes, l_U_t, l_M_t

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
