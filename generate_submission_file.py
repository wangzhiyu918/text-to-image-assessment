import os
import time
import cv2
import random
import zipfile
import pickle
import numpy as np
from PIL import Image

test_image_dir = "./Co-DETR/data/evalmuse/test/images"
test_image_list = sorted(os.listdir(test_image_dir))

mos_result_file = "./LoDa/loda_results.npy"
mask_result_dir = "./Co-DETR/co_dino_5scale_vit_large_evalmuse_instance/"

mos_results = np.load(mos_result_file)
assert len(mos_results) == len(test_image_list) == 999

image2mos = dict()
for image_name, mos_score in zip(test_image_list, mos_results):
    image2mos[image_name] = mos_score

output = dict()
for image_name in test_image_list:
    k = image_name[:-4]
    pred_map_path = os.path.join(mask_result_dir, k + ".png")
    pred_map = np.array(Image.open(pred_map_path), dtype=np.uint8)

    output[k] = {
        "score": image2mos[image_name],
        "pred_area": pred_map
    }
    
with open(f'./output.pkl', 'wb') as f:
    pickle.dump(output, f)

with zipfile.ZipFile(f"results.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("output.pkl")
    zipf.write("readme.txt")
