import os

import time
import pandas as pd
import torch.utils.data as data

from src.utils.dataset_utils import pil_loader

from .dataloader_mode import DataloaderMode


class EvalMuse10k_Dataset(data.Dataset):
    def __init__(self, cfg, index, transform, mode):
        self.root = cfg.data.root
        meta_info = pd.read_csv(cfg.data.meta_info_file)

        if mode is DataloaderMode.train:
            patch_num = cfg.train.patch_num
        elif mode is DataloaderMode.val:
            patch_num = cfg.val.patch_num
        elif mode is DataloaderMode.test:
            patch_num = cfg.test.patch_num
        else:
            raise ValueError(f"invalid dataloader mode {mode}")

        sample = []
        for idx in index:
            img_name = meta_info.loc[idx]["img_name"]
            img_path = os.path.join("train", "images", img_name)
            label = meta_info.loc[idx]["mos"]
            for _ in range(patch_num):
                sample.append((img_path, label))

        #################################
        # 提交测试集推理结果
        if cfg.name == "loda_evalmuse10k_eval_split":
            print("\033[1;91m### WARNING: Load test images for inference ...\033[0m")
            time.sleep(5)
            sample = []
            self.root = "data/evalmuse10k/test"
            for image_name in sorted(os.listdir(self.root)):
                if image_name.endswith(".jpg"):
                    sample.append((image_name, 3.0))
        #################################

        self.samples = sample
        self.transform = transform
        print(mode, len(self.samples))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = pil_loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)

        # if there are more than one image or more than one target
        # can organize it as
        # return [img1, img2], [targe1, target2]
        return img, target

    def __len__(self):
        length = len(self.samples)
        return length
