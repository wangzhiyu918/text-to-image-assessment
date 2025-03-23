import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
import os

def load_gt(gt_path):
    with open(gt_path, "r") as f:
        gt_data = json.load(f)
    
    # 获取 image_id 到文件路径的映射
    image_id_to_path = {img["id"]: img["file_name"] for img in gt_data["images"]}
    
    # 获取 category_id 到类别名称的映射
    category_map = {cat["id"]: cat["name"] for cat in gt_data["categories"]}
    
    return image_id_to_path, category_map

def load_results(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)
    return results

def convert_results(image_path, annotations, output_dir):
    # 读取原始图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = np.zeros(image.shape[:2], dtype=np.uint8)

    for ann in annotations:
        if ann["score"] < 0.3:
            continue

        if ann["category_id"] == 1:
            continue

        assert ann["category_id"] == 1 or ann["category_id"] == 2
        print(image_path)

        # 解析 RLE 掩码
        seg_mask = mask_utils.decode(ann["segmentation"])

        mask_indices = seg_mask > 0
        result[mask_indices] = 1

    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(result).save(
        os.path.join(output_dir, os.path.basename(image_path)[:-4] + ".png")
    )

def main(gt_json, results_json, image_dir, output_dir):
    # 读取数据
    image_id_to_path, category_map = load_gt(gt_json)
    results = load_results(results_json)

    # 按照 image_id 分组
    image_to_annotations = {}
    for ann in results:
        image_id = ann["image_id"]

        if ann["category_id"] != 1 and ann["category_id"] != 2:
            assert ann["score"] < 0.1

        if image_id in image_to_annotations:
            image_to_annotations[image_id].append(ann)
        else:
            image_to_annotations[image_id] = [ann]

    # 可视化每张图像的分割结果
    for image_id, annotations in image_to_annotations.items():
        if image_id in image_id_to_path:
            image_path = os.path.join(image_dir, image_id_to_path[image_id])
            convert_results(image_path, annotations, output_dir)
        else:
            print(f"Warning: Image ID {image_id} not found in GT.")

# 5. 运行代码
if __name__ == "__main__":
    gt_json = "./data/evalmuse/test/evalmuse_test999_coco_instances.json"
    image_dir = "./data/evalmuse/test/images/"
    
    results_json = f"./co_dino_5scale_vit_large_evalmuse_instance.segm.json"
    output_dir = f"./co_dino_5scale_vit_large_evalmuse_instance"
    main(gt_json, results_json, image_dir, output_dir)
