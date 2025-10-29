import os
import torch
from tqdm import tqdm

from torchmetrics.multimodal.clip_score import CLIPScore

from Metrics.inception_metrics import MultiInceptionMetrics


class SampleAndEval:
    def __init__(self, device, is_master, nb_gpus, num_images=50_000, num_classes=1_000, compute_manifold=True, mode="c2i"):
        super().__init__()
        self.inception_metrics = MultiInceptionMetrics(
            device=device, compute_manifold=compute_manifold, num_classes=num_classes,
            num_inception_chunks=10, manifold_k=3, model="inception")

        self.num_images = num_images
        self.device = device
        self.is_master = is_master
        self.nb_gpus = nb_gpus
        self.mode = mode

        if mode == "t2i":
            self.clip_score = CLIPScore("openai/clip-vit-large-patch14").to(device)

    @torch.no_grad()
    def compute_images_features_from_model(self, trainer, sampler, data_loader):
        import os, torch, time
        # ---------- ① 预加载已缓存的 real 统计 ----------
        stats_path = f"./saved_networks/imagenet_val_stats_{getattr(trainer.args, 'img_size', 256)}.pt"
        use_cached_real = False
        if trainer.args.data.startswith("imagenet") and os.path.exists(stats_path):
            ckpt = torch.load(stats_path, map_location="cpu")
            # MultiInceptionMetrics 若无专门接口，可以直接赋值（按你项目类实现而定）
            self.inception_metrics.real_mu = ckpt["mu"]
            self.inception_metrics.real_cov = ckpt["cov"]
            self.inception_metrics.real_stats_ready = True  # 若类中没有这个属性也没关系
            use_cached_real = True
            if self.is_master:
                print(f"[EVAL] Use pre-computed ImageNet val stats: {stats_path}")

        bar = tqdm(data_loader, leave=False, desc="Computing images features") if self.is_master else data_loader

        # 记录吞吐、计数
        fake_seen = 0
        real_seen = 0
        start = time.time()

        for images, labels in bar:
            labels = labels.to(self.device)

            # 生成 fake
            if self.mode == "t2i":
                labels_txt = labels[0]  # coco: 每图5条caption
                gen_images = sampler(trainer=trainer, txt_promt=labels_txt)[0]
                self.clip_score.update(images, labels_txt)
            elif self.mode == "c2i":
                gen_images = sampler(trainer=trainer, nb_sample=images.size(0), labels=labels, verbose=False)[0]
            elif self.mode == "vq":
                code = trainer.ae.encode(images.to(self.device)).to(self.device)
                code = code.view(code.size(0), trainer.input_size, trainer.input_size)
                gen_images = trainer.ae.decode_code(torch.clamp(code, 0, trainer.args.codebook_size - 1))

            self.inception_metrics.update(gen_images, image_type="fake")
            fake_seen += gen_images.size(0)

            # 只在没有缓存 real 统计时，才遍历 real
            if (not use_cached_real) and trainer.args.data.startswith("imagenet"):
                self.inception_metrics.update(images, image_type="real")
                real_seen += images.size(0)

            # 进度/ETA（可选）
            if self.is_master and (fake_seen % 1024 == 0 or fake_seen >= self.num_images):
                elapsed = max(time.time() - start, 1e-6)
                ips = fake_seen / elapsed
                remain = max(self.num_images - fake_seen, 0)
                eta = remain / max(ips, 1e-6)
                print(f"[GEN] {fake_seen}/{self.num_images} imgs | {ips:.1f} img/s | ETA ~ {int(eta)} s")

            if self.inception_metrics.count >= self.num_images:
                break

        # ---------- ② 如未缓存且遍历了 real，这里保存 mu/cov ----------
        if (not use_cached_real) and trainer.args.data.startswith("imagenet"):
            try:
                # 你的 MultiInceptionMetrics 里通常会存 real_features 的 list
                real_features = torch.cat(self.inception_metrics.real_features, dim=0)
                real_mu = real_features.mean(dim=0)
                real_cov = self.inception_metrics.cov(real_features, real_mu)
                os.makedirs(os.path.dirname(stats_path), exist_ok=True)
                torch.save({"mu": real_mu.cpu(), "cov": real_cov.cpu()}, stats_path)
                if self.is_master:
                    print(f"[EVAL] Saved real stats to {stats_path} | real_seen={real_seen}")
            except Exception as e:
                if self.is_master:
                    print(f"[EVAL][WARN] failed to save real stats: {e}")

        # ---------- ③ 计算指标，并把计数写进 metrics ----------
        metrics = self.inception_metrics.compute()

        if self.mode == "t2i":
            metrics["clip_score"] = round(self.clip_score.compute().item(), 4)

        # 记录实际看的数量（便于你核对是否 = 50,000）
        metrics["num_fake_seen"] = int(fake_seen)
        metrics["num_real_seen"] = int(real_seen) if not use_cached_real else int(getattr(self.inception_metrics, "real_count", 0) or 50000)

        metrics = {k: (round(v, 4) if isinstance(v, (float, int)) else v) for k, v in metrics.items()}
        return metrics

