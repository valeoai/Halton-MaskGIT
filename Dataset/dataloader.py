import os
import glob
import webdataset as wds

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.stl10 import STL10
from torchvision.datasets.coco import CocoCaptions
from torchvision.datasets import ImageFolder

from Dataset.dataset import ImageNetKaggle
try:
    from Dataset.dataset import CodeDataset
except Exception:
    CodeDataset = None



def get_data(data, img_size, data_folder, bsize, num_workers, is_multi_gpus, seed,
             split="both",eval_only=False, **kwargs):
    """
    加了 split 参数：
      - split="val"   只构建并返回 (None, val_loader)
      - split="train" 只构建并返回 (train_loader, None)
      - split="both"  同时构建并返回 (train_loader, val_loader)

    注意：当 split 不是 "both" 时，我们完全不会去尝试构建另一侧的数据集，
         从而避免你遇到的 train 不存在导致的报错。
    """
    _ = eval_only
    data_train = None
    data_test  = None

    if data == "mnist":
        if split in ("both", "train"):
            data_train = MNIST('./dataset_storage/mnist/', download=False,
                               transform=transforms.Compose([transforms.Resize(img_size),
                                                             transforms.ToTensor()]))
        if split in ("both", "val"):
            data_test = MNIST('./dataset_storage/mnist/', train=False, download=False,
                              transform=transforms.Compose([transforms.Resize(img_size),
                                                            transforms.ToTensor()]))

    elif data == "cifar10":
        if split in ("both", "train"):
            data_train = CIFAR10(data_folder, train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]))
        if split in ("both", "val"):
            data_test = CIFAR10(data_folder, train=False, download=False,
                                transform=transforms.Compose([
                                    transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))

    elif data == "stl10":
        if split in ("both", "train"):
            data_train = STL10('./Dataset/stl10', split="train+unlabeled",
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
        if split in ("both", "val"):
            data_test = STL10('./Dataset/stl10', split="test",
                              transform=transforms.Compose([
                                  transforms.Resize(img_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))

    elif data == "imagenet":
        t_train = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop((img_size, img_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        t_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        # ---- 只构建 train ----
        if split == "train":
            try:
                data_train = ImageFolder(os.path.join(data_folder, "train"), transform=t_train)
            except Exception:
                data_train = ImageNetKaggle(data_folder, split="train", img_size=img_size, transform=t_train)

        # ---- 只构建 val ----
        elif split == "val":
            try:
                data_test = ImageFolder(os.path.join(data_folder, "val"), transform=t_test)
            except Exception:
                data_test = ImageNetKaggle(data_folder, split="val", img_size=img_size, transform=t_test)

        # ---- 同时构建 train + val ----
            # ---- 同时构建 train + val ----
        else:  # split == "both"
            # train（单独 try）
            try:
                data_train = ImageFolder(os.path.join(data_folder, "train"), transform=t_train)
            except Exception:
                try:
                    data_train = ImageNetKaggle(
                        data_folder, split="train", img_size=img_size, transform=t_train
                    )
                except Exception:
                    data_train = None  # 没有 train 也允许继续

            # val（单独 try）
            try:
                data_test = ImageFolder(os.path.join(data_folder, "val"), transform=t_test)
            except Exception:
                data_test = ImageNetKaggle(
                    data_folder, split="val", img_size=img_size, transform=t_test
                )


    elif data == "imagenet_feat":
        if CodeDataset is None:
            raise ImportError("CodeDataset 未找到：请确认 Dataset/dataset.py 中已实现并可导入 CodeDataset")
        if split in ("both", "train"):
            data_train = CodeDataset(os.path.join(data_folder, "Train"))
        if split in ("both", "val"):
            data_test  = CodeDataset(os.path.join(data_folder, "Eval"))


    elif data == "mscoco":
        if split in ("both", "val"):
            data_test = CocoCaptions(
                root=os.path.join(data_folder, 'images/val2017/'),
                annFile=os.path.join(data_folder, 'annotations/captions_val2017.json'),
                transform=transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
                ]),
                target_transform=lambda x: x[:5]
            )
            test_sampler = DistributedSampler(data_test, shuffle=False, seed=seed) if is_multi_gpus else None
            test_loader = DataLoader(data_test, batch_size=bsize, shuffle=False,
                                     num_workers=num_workers, pin_memory=True,
                                     drop_last=False, sampler=test_sampler)
            return None, test_loader

    elif data == "webdata":
        # 这个分支跟 split 无关，保持原样
        sa_feat = glob.glob(os.path.join(data_folder, "sa_feat/*.tar"))
        cc12m = glob.glob(os.path.join(data_folder, "cc12m_feat/*.tar"))
        diffusiondb = glob.glob(os.path.join(data_folder, "diffusiondb_feat/*.tar"))
        midjourneydb = glob.glob(os.path.join(data_folder, "midjourney_feat/*.tar"))

        urls = list(sa_feat) + list(cc12m) + list(diffusiondb) + list(midjourneydb)
        print(f"number of shard: {len(urls)}")

        def preprocess(sample):
            keys = sample.keys()
            txt = sample['png.txt'] if 'png.txt' in keys else sample['txt']
            if 'vq_feat.npy' in keys:
                vq_feat = sample['vq_feat.npy']
            elif 'png.vq_feat.npy' in keys:
                vq_feat = sample['png.vq_feat.npy']
            else:
                vq_feat = sample['vq_feat']

            if 'txt_feat.npy' in keys:
                txt_feat = sample['txt_feat.npy']
            elif 'png.txt_feat.npy' in keys:
                txt_feat = sample['png.txt_feat.npy']
            else:
                txt_feat = sample['txt_feat']

            return txt, torch.LongTensor(vq_feat), torch.FloatTensor(txt_feat)

        dataset = (
            wds.WebDataset(urls, resampled=True, nodesplitter=wds.split_by_node)
            .shuffle(1_000)
            .decode("rgb")
            .map(preprocess)
            .batched(bsize)
        )

        train_loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
        train_loader = train_loader.unbatched().shuffle(1000).batched(bsize)
        train_loader = train_loader.with_epoch(16_884_356 // (16*8*4))
        return train_loader, None

    # ------- 根据是否构建到对应数据集，安全地创建 DataLoader -------
    train_loader = None
    test_loader  = None

    if data_train is not None:
        train_sampler = DistributedSampler(data_train, shuffle=True, seed=seed) if is_multi_gpus else None
        train_loader = DataLoader(
            data_train, batch_size=bsize,
            shuffle=False if is_multi_gpus else True,
            num_workers=num_workers, pin_memory=True,
            drop_last=True, sampler=train_sampler
        )

    if data_test is not None:
        test_sampler = DistributedSampler(data_test, shuffle=False, seed=seed) if is_multi_gpus else None
        test_loader = DataLoader(
            data_test, batch_size=bsize,
            shuffle=False,
            num_workers=num_workers, pin_memory=True,
            drop_last=False, sampler=test_sampler
        )

    return train_loader, test_loader
