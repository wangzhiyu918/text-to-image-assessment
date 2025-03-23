"""
based on IQA-PyTorch: https://github.com/chaofengc/IQA-PyTorch
"""
import csv
import os
import json
import pickle
import random

import pandas as pd
import pyrootutils
import torchvision
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.dataset_utils import pil_loader


def get_meta_info():
    """
    Train/Val/Test split file from official github:
        https://github.com/subpic/koniq/blob/master/metadata/koniq10k_distributions_sets.csv
    """
    info_file = "../data/EvalMuse-Structure/train_info.json"

    save_meta_path = "data/meta_info/meta_info_EvalMuse10kDataset.csv"
    with open(info_file, "r") as f:
        info = json.load(f)

    with open(save_meta_path, "w+") as sf:
        csvwriter = csv.writer(sf)
        new_head = [
            "img_name",
            "mos",
        ]
        csvwriter.writerow(new_head)
        for k, v in info.items():
            image_name = k + ".jpg"
            mos = float(v["mos"])
            new_row = [image_name, mos]
            csvwriter.writerow(new_row)


def get_random_splits(seed=3407):
    """
    Use 10 splits as most paper
    """
    random.seed(seed)
    total_num = 10000
    all_img_index = list(range(total_num))
    num_splits = 5

    # ratio = [0.8, 0.2]  # train/test
    train_index = int(round(0.9 * total_num))

    save_path = f"./data/train_split_info/evalmuse10k_91_seed{seed}.pkl"
    split_info = {}
    for i in range(num_splits):
        random.shuffle(all_img_index)
        split_info[i] = {
            "train": all_img_index[:train_index],
            "val": [],
            "test": all_img_index[train_index:],
        }
        print(
            "train num: {} | val num: {} | test num: {}".format(
                len(split_info[i]["train"]),
                len(split_info[i]["val"]),
                len(split_info[i]["test"]),
            )
        )
    with open(save_path, "wb") as sf:
        pickle.dump(split_info, sf)


if __name__ == "__main__":
    get_meta_info()
    get_random_splits()
