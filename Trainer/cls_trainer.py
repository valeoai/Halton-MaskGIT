# Trainer for Cls-to-Img MaskGIT
import os
import time
from datetime import datetime

from tqdm import tqdm
from collections import deque
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from Trainer.abstract_trainer import Trainer

from Network.vq_model import VQ_models
from Network.transformer import Transformer
from Sampler.confidence_sampler import ConfidenceSampler
from Sampler.halton_sampler import HaltonSampler
from Dataset.dataloader import get_data
from Utils.viz import reconstruction
from Utils.masking_scheduler import get_mask_code


class MaskGIT(Trainer):

    def __init__(self, args):
        """ Initialize model, optimizer, loss function, and data loaders."""
        super().__init__(args)
        print(f"Init Cls-to-Img Maskgit on [GPU{args.global_rank}]")

        self.args = args                                                        # Main argument see main.py
        self.input_size = self.args.img_size // self.args.f_factor              # Define input size for transformer

        # Load transformer (Masked Bidirectional Transformer) and VQGAN models
        self.vit = self.get_network("vit")                                      # Load Masked Bidirectional Transformer
        self.ae = self.get_network("vqgan-llama")                               # Load VQGAN

        # Initialize Exponential Moving Average (EMA) model if enabled
        if self.args.use_ema:
            self.ema = self.get_network("ema")

        # Define loss function and optimizer
        self.criterion = self.get_loss("cross_entropy", ignore_index=-100)      # Get cross-entropy loss
        self.optim = self.get_optim(self.vit, self.args.lr, betas=(0.9, 0.999), weight_decay=0.03)  # Get AdamW Optimizer

        # Set up automatic mixed precision for training efficiency
        if self.args.device != 'cpu' and self.args.dtype == "bfloat16":
            self.autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            self.autocast = nullcontext()

        # Load training and testing data if specified
        if args.data_folder != "":
            self.train_data, self.test_data = get_data(
                args.data, args.img_size, args.data_folder, args.bsize,
                args.num_workers, args.is_multi_gpus, args.seed
            )

        # Select the appropriate sampling method (Halton or Confidence-based)
        if args.sampler == "halton":
            self.sampler = HaltonSampler(sm_temp_min=self.args.sm_temp_min, sm_temp_max=self.args.sm_temp,
                                         temp_pow=1, temp_warmup=self.args.temp_warmup, w=self.args.cfg_w,
                                         sched_pow=self.args.sched_pow, step=self.args.step,
                                         randomize=self.args.randomize, top_k=self.args.top_k)

        elif args.sampler == "confidence":
            self.sampler = ConfidenceSampler(sm_temp=self.args.sm_temp, w=self.args.cfg_w, r_temp=self.args.r_temp,
                                             sched_mode=self.args.sched_mode, step=self.args.step)
        else:
            self.sampler = None

        if self.args.is_master:
            print("Parameters: ", self.args)
            args_str = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
            self.log_add_txt("Parameters", args_str, self.args.iter)

    def get_network(self, archi):
        """ return the network, load checkpoint if self.args.resume == True
            :param
                archi -> str: vit|autoencoder|ema, the architecture to load
            :return
                model -> nn.Module: the network
        """
        if archi == "vit":
            if self.args.is_master:
                if not os.path.exists(self.args.vit_folder):
                    os.makedirs(self.args.vit_folder)
                    print(f"Folder created: {self.args.vit_folder}")

            # Define transformer architecture parameters
            hidden_dim, depth, heads = self.transformer_size(self.args.vit_size)
            # large: 1024, 24, 16 | base: 768, 12, 12 | small: 512, 8, 8 | tiny: 256, 6, 4
            model = Transformer(
                input_size=self.input_size, nclass=self.args.nb_class, c=hidden_dim,
                hidden_dim=hidden_dim, codebook_size=self.args.codebook_size,
                depth=depth, heads=heads, mlp_dim=hidden_dim * 4, dropout=self.args.dropout,
                register=self.args.register, proj=self.args.proj, 
                    # === 新增：Halton-Token-Merge 开关与配置 ===
                # 1.0 = 关闭合并；例如 0.5 表示只保留一半 token（第1层前合并、最后一层前反合并）
                tome_keep_ratio=self.args.tome_keep_ratio,
                # 在第几层前合并；按你实现，0 就是“第一层前”
                tome_merge_layer_idx=self.args.tome_merge_layer_idx,
                # 在倒数第1层前反合并；-1 就是“最后一层前”
                tome_unmerge_before_idx=self.args.tome_unmerge_before_idx,
                # 如需为每个前向随机 roll 一下 Halton 次序（增强/去同质化），就传 True
                tome_random_roll=self.args.tome_random_roll,)

            # Load model checkpoint if resuming training
            if self.args.resume:
                if os.path.isfile(self.args.vit_folder + "current.pth"):
                    ckpt = self.args.vit_folder + "current.pth"
                elif os.path.isfile(self.args.vit_folder):
                    ckpt = self.args.vit_folder
                else:
                    ckpt = None

                if ckpt is not None:
                    checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
                    state_dict = checkpoint['model_state_dict']
                    new_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                    model.load_state_dict(new_state_dict, strict=True)

                    # Update the current epoch and iteration
                    self.args.iter = checkpoint['iter']
                    self.args.global_epoch = checkpoint['global_epoch']
                    if self.args.is_master:
                        print("Load ckpt from:", ckpt)
                        print("Number of iteration(s):", self.args.iter)

        elif archi == "vqgan-llama":
            # Initialize and load VQGAN model
            model = VQ_models[f"VQ-{self.args.f_factor}"](codebook_size=16384, codebook_embed_dim=8)
            checkpoint = torch.load(self.args.vqgan_folder, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            model = model.eval()

        elif archi == "ema":
            # Load Exponential Moving Average (EMA) model
            model = self.get_ema(self.vit, device=self.args.device)
            if self.args.resume and os.path.isfile(self.args.vit_folder + "ema.pth"):
                ckpt = self.args.vit_folder + "ema.pth"
                checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
                state_dict = checkpoint['model_state_dict']
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = "_orig_mod." + k if self.args.compile else k
                    k = "module." + k if self.args.is_multi_gpus else k
                    new_state_dict[k] = v
                model.module.load_state_dict(new_state_dict, strict=True)
            return model

        else:
            model = None

        model = model.to(self.args.device)

        if self.args.compile: # Enable model compilation if using PyTorch 2.0+
            model = torch.compile(model)

        if self.args.is_multi_gpus:  # Enable multi-GPU training if available
            model = DDP(model, device_ids=[self.args.device])
            if archi == "vqgan-llama":
                model = model.module

        if self.args.is_master:
            print(f"Size of model {archi}: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

        return model

    def train_one_epoch(self, log_iter=2_500, ema_update=1):
        """
        Train the model for one epoch
            :param
                log_iter -> int: Frequency of logging and saving the model.
                ema_update --> int: Frequency of updating the EMA model.
            :return
                (avg_loss, avg_acc) -> tuple: (average loss, average accuracy) over the train dataset.
        """

        self.vit.train()
        cum_loss, cum_acc = 0., 0.
        # Deque to store loss and accuracy over a moving window
        window_loss = deque(maxlen=self.args.grad_cum)
        window_acc = deque(maxlen=self.args.grad_cum)

        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        for i_bar, data in enumerate(bar):
            # Determine whether to update gradients based on gradient accumulation steps
            update_grad = (i_bar % self.args.grad_cum) == self.args.grad_cum - 1
            # Adjust the learning rate with warmup and cosine decay for the 10% last iter
            self.adapt_learning_rate()

            code = data["code"].long().to(self.args.device)
            y = data["y"].to(self.args.device)
            b, h, w = code.size()

            # Randomly drop conditions for conditional generation (CFG)
            drop_label = (torch.rand(b) < self.args.drop_label).to(self.args.device)
            # Apply masking to encoded codes
            masked_code, mask, loss_ign = get_mask_code(
                code, value=self.args.mask_value, codebook_size=self.args.codebook_size,
                mode=self.args.sched_mode, sched_pow=self.args.sched_pow)

            # Create target code, replacing ignored tokens with -100
            target_code = torch.where(loss_ign, code.detach().clone(), -100)

            with self.autocast:  # Perform forward pass using mixed precision (if available)
                # Model prediction
                pred = self.vit(x=masked_code, y=y, drop_label=drop_label)
                # Compute loss using cross-entropy over masked tokens
                loss = self.criterion(pred.reshape(b*h*w, self.args.codebook_size + 1), target_code.reshape(b*h*w)).contiguous()
                # Normalize loss for gradient accumulation
                loss = loss / self.args.grad_cum

            loss.backward()

            # Perform gradient update if gradient accumulation step is reached
            if update_grad:
                nn.utils.clip_grad_norm_(self.vit.parameters(), self.args.grad_clip)  # Clip gradient
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)

                # Update EMA model periodically
                if self.args.use_ema and i_bar % ema_update == 0:
                    self.ema.update(self.vit)

            # Compute running loss and accuracy
            cum_loss += loss.cpu().item() * self.args.grad_cum
            acc = torch.max(pred.reshape(b*h*w, self.args.codebook_size + 1).data, 1)[1]
            acc = (acc.view(b*h*w) == code.view(b*h*w)).float()[mask.view(b*h*w)]
            cum_acc += acc.mean().cpu().item()
            window_loss.append(loss.cpu().item() * self.args.grad_cum)
            window_acc.append(acc.mean().item())

            # Logging and visualization
            if update_grad:
                if self.args.is_multi_gpus: # Synchronize logs across multiple GPUs if applicable
                    mini_batch_loss = self.all_gather(torch.tensor(window_loss).mean())
                    mini_batch_acc = self.all_gather(torch.tensor(window_acc).mean())
                else:
                    mini_batch_loss = torch.tensor(window_loss).mean()
                    mini_batch_acc = torch.tensor(window_acc).mean()

                if self.args.is_master: # Master process logs metrics
                    self.log_add_scalar('Train/MiniBatchLoss', mini_batch_loss, self.args.iter)
                    self.log_add_scalar('Train/MiniBatchAcc', mini_batch_acc, self.args.iter)
                    self.log_add_scalar('Train/LearningRate', self.optim.param_groups[0]['lr'], self.args.iter)

                # Save model and visualize samples periodically
                if self.args.iter % log_iter == 0 and self.args.is_master:
                    gen_sample = self.sampler(self, nb_sample=b//2)[0][:10]
                    reco_sample = reconstruction(self, code=code[:10], unmasked_code=pred.argmax(dim=-1)[:10], mask=mask[:10])
                    self.log_add_img("Images/Reconstruction", reco_sample, self.args.iter)
                    self.log_add_img("Images/Sampling", gen_sample, self.args.iter)

                    # Save the current model state
                    self.save_network(model=self.vit, path=os.path.join(self.args.vit_folder, "current.pth"),
                                      optimizer=self.optim, iter=self.args.iter, global_epoch=self.args.global_epoch)

                    if self.args.use_ema:  # Save the EMA model if enabled
                        self.save_network(model=self.ema.module, path=os.path.join(self.args.vit_folder, "ema.pth"),
                                          iter=self.args.iter, global_epoch=self.args.global_epoch)

                # Increment global iteration counter
                self.args.iter += 1

            # Stop training if max iterations are reached
            if self.args.iter > self.args.max_iter:
                if self.args.is_master:
                    print("End of training: reached max iterations")
                return cum_loss / i_bar, cum_acc / i_bar

        # Return average loss for the epoch
        return cum_loss / len(self.train_data), cum_acc / len(self.train_data)

    @torch.no_grad()
    def eval_one_epoch(self):
        """
        Eval the model for one epoch
            :return
                (avg_loss, avg_acc) -> tuple: (average loss, average accuracy) over the test dataset.
        """
        # Use the Exponential Moving Average (EMA) model if enabled; otherwise, use the main model
        model = self.ema if self.args.use_ema else self.vit
        model.eval()
        cum_loss, cum_acc = 0., 0.

        bar = tqdm(self.test_data, leave=False) if self.args.is_master else self.test_data
        for i_bar, (data) in enumerate(bar):
            code = data["code"].long().to(self.args.device)
            y = data["y"].to(self.args.device)
            b, h, w = code.size()

            # Randomly drop conditions for conditional generation (CFG)
            drop_label = (torch.rand(b) < self.args.drop_label).to(self.args.device)

            # Apply masking to encoded codes
            masked_code, mask, loss_ign = get_mask_code(
                code, value=self.args.mask_value, codebook_size=self.args.codebook_size,
                mode=self.args.sched_mode, sched_pow=self.args.sched_pow)

            # Create target code, replacing ignored tokens with -100 (ignored in loss computation)
            target_code = torch.where(loss_ign, code.detach().clone(), -100)

            with self.autocast:  # Perform forward pass using mixed precision (if available)
                # Model prediction
                pred = self.vit(x=masked_code, y=y, drop_label=drop_label)
                loss = self.criterion(pred.reshape(b*h*w, self.args.codebook_size + 1), target_code.view(b*h*w))

            # Compute running loss and accuracy
            acc = torch.max(pred.reshape(b*h*w, self.args.codebook_size + 1).data, 1)[1]
            acc = (acc.view(b*h*w) == code.view(b*h*w)).float()[mask.view(b*h*w)]
            cum_acc += acc.mean().cpu().item()
            cum_loss += loss.cpu().item()
        model.train()

        # Return average loss for the epoch
        return cum_loss / len(self.test_data), cum_acc / len(self.test_data)

    def fit(self, metrics_eval=10):
        """
        Train the model for multiple epochs.

        Args:
            metrics_eval (int): Frequency (in epochs) at which FID/IS is performed.

        This function handles:
            - Training and evaluation for each epoch
            - Synchronization across multiple GPUs if applicable
            - Model checkpointing at specific epochs
            - Logging training progress and metrics
        """

        if self.args.is_master:
            print("Start training:")

        start = time.time()
        # This is the sampler used for evaluation (not for visualization)
        if self.args.sampler == "halton":
            eval_sampler = HaltonSampler(
                sm_temp_min=1., sm_temp_max=1., temp_warmup=0, w=0, sched_pow=self.args.sched_pow,
                step=self.args.step, randomize=self.args.randomize, top_k=-1
            )
        elif self.args.sampler == "confidence":
            eval_sampler = ConfidenceSampler(
                sm_temp=1., w=0, r_temp=self.args.r_temp, sched_mode=self.args.sched_mode, step=self.args.step
            )
        else:
            eval_sampler = None
        # Main training loop across epochs
        for e in range(self.args.global_epoch, self.args.epoch+1):
            # Synchronize dataset shuffling across multiple GPUs if applicable
            if self.args.is_multi_gpus:
                self.train_data.sampler.set_epoch(e)
                self.test_data.sampler.set_epoch(e)

            # Train/Eval the model and get test loss and accuracy
            train_loss, train_acc = self.train_one_epoch()
            test_loss, test_acc = self.eval_one_epoch()

            # Synchronize loss and accuracy across GPUs if applicable
            if self.args.is_multi_gpus:
                train_loss = self.all_gather(train_loss)
                train_acc = self.all_gather(train_acc)
                test_loss = self.all_gather(test_loss)
                test_acc = self.all_gather(test_acc)

            # Save model checkpoints at specific epochs
            if self.args.global_epoch % 10 == 0 and self.args.global_epoch > 0 and self.args.is_master:
                self.save_network(model=self.vit,
                                  path=self.args.vit_folder + f"epoch_{self.args.global_epoch:03d}.pth",
                                  optimizer=self.optim, iter=self.args.iter,
                                  global_epoch=self.args.global_epoch)

            # Logging and printing progress
            if self.args.is_master:
                clock_time = (time.time() - start)
                self.log_add_scalar('Epoch/Loss', {"Train": train_loss, "Eval": test_loss}, self.args.global_epoch)
                self.log_add_scalar('Epoch/Accuracy', {"Train": train_acc, "Eval": test_acc}, self.args.global_epoch)
                now = datetime.now()
                print(f"\rEpoch {self.args.global_epoch},"
                      f" Iter {self.args.iter},"
                      f" Train: {train_loss:.4f}, Eval: {test_loss:.4f},"
                      f" Time: {int(clock_time // 3600):.0f}:{int((clock_time % 3600) // 60):02d}:{int(clock_time % 60):02d},"
                      f" Date: {now.date()} {now.hour:02}:{now.minute:02}")

            # Perform FID/IS at specified intervals
            if e % metrics_eval == 0 and e > 0:
                m = self.eval(sampler=eval_sampler, num_images=10_000, save_exemple=False, compute_pr=False,
                              split="Test", mode="c2i", data=self.args.data.split("_")[0])

                if self.args.is_master:
                    self.log_add_scalar('Metrics/FID', m["FID"], self.args.global_epoch)
                    self.log_add_scalar('Metrics/IS', m["IS"], self.args.global_epoch)

            self.args.global_epoch += 1

            # Stop training if the maximum number of iterations is reached
            if self.args.iter > self.args.max_iter:
                break

        # Final model saving when training is completed
        if self.args.is_master:
            print(f"Training is done, saving last check point at {self.args.vit_folder}")
            self.save_network(model=self.vit, path=self.args.vit_folder + f"last.pth",
                              optimizer=self.optim, iter=self.args.iter,
                              global_epoch=self.args.global_epoch)
            if self.args.use_ema:
                self.save_network(model=self.ema.module, path=os.path.join(self.args.vit_folder, "last_ema.pth"),
                                  iter=self.args.iter, global_epoch=self.args.global_epoch)
        return

    def train(self, mode: bool = True):
        # 把内部所有 nn.Module 的 train/eval 状态一起切换
        def _set_mode(obj):
            if isinstance(obj, torch.nn.Module):
                obj.train(mode)
                return
            if isinstance(obj, dict):
                for v in obj.values(): _set_mode(v)
            elif isinstance(obj, (list, tuple, set)):
                for v in obj: _set_mode(v)
            else:
                for name in dir(obj):
                    if name.startswith('_'): continue
                    try:
                        v = getattr(obj, name)
                    except Exception:
                        continue
                    if callable(v): continue
                    _set_mode(v)
        _set_mode(self)
        return self

    def eval(self):
        return self.train(False)