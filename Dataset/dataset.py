import os, json
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _list_images(folder, exts={".jpg", ".jpeg", ".png", ".JPEG", ".JPG"}):
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1] in exts]

class ImageNetKaggle(Dataset):
    """
    支持三种布局：
    A) 官方展开后的常见结构（推荐）：
       <root>/train/<wnid>/*.JPEG
       <root>/val/<wnid>/*.JPEG

    B) Kaggle 的 val 平铺 + 标签文件（文件名→wnid），需要：
       <root>/ILSVRC2012_val_labels.json
       布局：
       <root>/train/<wnid>/*.JPEG     (可选)
       <root>/val/*.JPEG              (flat)

    C) 原始 devkit 目录（不常见）：
       <root>/ILSVRC/Data/CLS-LOC/<split>/...
    """
    def __init__(self, root, split, img_size=512, transform=None):
        assert split in {"train", "val"}
        self.root = root
        self.split = split
        self.transform = transform

        # 1) 准备 idx↔wnid 映射（若有 json 用之；没有也可通过 ImageFolder-like 模式绕过）
        self.syn_to_class = {}
        self.class_to_syn = {}
        idx_json = os.path.join(root, "imagenet_class_index.json")
        if os.path.isfile(idx_json):
            with open(idx_json, "r") as f:
                idxmap = json.load(f)  # {"0": ["n01440764","tench"], ...}
            for k, v in idxmap.items():
                wnid = v[0]
                idx = int(k)
                self.syn_to_class[wnid] = idx
                self.class_to_syn[idx] = wnid

        self.samples, self.targets, self.paths = [], [], []

        # 2) 优先支持常见结构 <root>/<split>/<wnid>/*.JPEG
        split_dir_A = os.path.join(root, split)
        if os.path.isdir(split_dir_A):
            # 有子目录就是每类；没有子目录但有图片=flat
            subdirs = sorted([d for d in os.listdir(split_dir_A)
                              if os.path.isdir(os.path.join(split_dir_A, d))])

            if subdirs:  # class folders 模式
                for wnid in subdirs:
                    cls_dir = os.path.join(split_dir_A, wnid)
                    imgs = sorted(_list_images(cls_dir))
                    if not imgs:
                        continue
                    if self.syn_to_class:
                        target = self.syn_to_class.get(wnid, None)
                        if target is None:
                            # 若 json 不覆盖这个 wnid，则跳过或按需动态分配
                            continue
                    else:
                        # 没有 idx json，就按子目录顺序动态编号
                        target = subdirs.index(wnid)

                    for name in imgs:
                        p = os.path.join(cls_dir, name)
                        self.samples.append(p)
                        self.targets.append(target)
                        self.paths.append(p)

                if self.samples:
                    return  # A模式构建完成

            else:
                # flat val：<root>/val/*.JPEG 需要 ILSVRC2012_val_labels.json
                if split == "val":
                    val_json = os.path.join(root, "ILSVRC2012_val_labels.json")
                    if os.path.isfile(val_json):
                        with open(val_json, "r") as f:
                            fname2wnid = json.load(f)  # {"ILSVRC2012_val_00000001.JPEG": "n01440764", ...}
                        imgs = sorted(_list_images(split_dir_A))
                        for name in imgs:
                            wnid = fname2wnid.get(name, None)
                            if wnid is None:
                                continue
                            if self.syn_to_class:
                                target = self.syn_to_class.get(wnid, None)
                                if target is None:
                                    continue
                            else:
                                # 动态建立出现过的 wnid 的索引
                                if wnid not in self.syn_to_class:
                                    self.syn_to_class[wnid] = len(self.syn_to_class)
                                target = self.syn_to_class[wnid]
                            p = os.path.join(split_dir_A, name)
                            self.samples.append(p)
                            self.targets.append(target)
                            self.paths.append(p)
                        if self.samples:
                            return  # B模式(flat val)构建完成

        # 3) 尝试 devkit 原始路径 C) <root>/ILSVRC/Data/CLS-LOC/<split>
        split_dir_C = os.path.join(root, "ILSVRC", "Data", "CLS-LOC", split)
        if os.path.isdir(split_dir_C):
            if split == "train":
                wnids = sorted([d for d in os.listdir(split_dir_C)
                                if os.path.isdir(os.path.join(split_dir_C, d))])
                for wnid in wnids:
                    cls_dir = os.path.join(split_dir_C, wnid)
                    imgs = sorted(_list_images(cls_dir))
                    if self.syn_to_class:
                        target = self.syn_to_class.get(wnid, None)
                        if target is None:
                            continue
                    else:
                        target = wnids.index(wnid)
                    for name in imgs:
                        p = os.path.join(cls_dir, name)
                        self.samples.append(p)
                        self.targets.append(target)
                        self.paths.append(p)
                if self.samples:
                    return
            else:  # val
                # 有些人把 val 平铺在 CLS-LOC/val 下，并提供 ILSVRC2012_val_labels.json
                imgs = sorted(_list_images(split_dir_C))
                if imgs:
                    val_json = os.path.join(root, "ILSVRC2012_val_labels.json")
                    if os.path.isfile(val_json):
                        with open(val_json, "r") as f:
                            fname2wnid = json.load(f)
                        for name in imgs:
                            wnid = fname2wnid.get(name, None)
                            if wnid is None:
                                continue
                            if self.syn_to_class:
                                target = self.syn_to_class.get(wnid, None)
                                if target is None:
                                    continue
                            else:
                                if wnid not in self.syn_to_class:
                                    self.syn_to_class[wnid] = len(self.syn_to_class)
                                target = self.syn_to_class[wnid]
                            p = os.path.join(split_dir_C, name)
                            self.samples.append(p)
                            self.targets.append(target)
                            self.paths.append(p)
                        if self.samples:
                            return

        # 4) 仍然为空：报清晰错误，提示如何修复
        raise FileNotFoundError(
            f"ImageNetKaggle: cannot locate {split} set.\n"
            f"Tried:\n"
            f"  A) {os.path.join(root, split, '<wnid>/*.JPEG')} (class folders) or flat with ILSVRC2012_val_labels.json\n"
            f"  C) {os.path.join(root, 'ILSVRC', 'Data', 'CLS-LOC', split)}\n"
            f"Also optional: {idx_json} for wnid->index mapping.\n"
            f"Make sure your directory and mapping files are in place."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]