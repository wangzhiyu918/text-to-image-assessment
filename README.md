# Learning Local Distortion Features for Quality Assessment of Text-to-Image Generation Models


## Team

Team name: HNU-VPAI

Colab Account: wangzhiyu918

Concat Email: wangzhiyu.wzy1@gmail.com

## Introduction

We adopt a separate model for each sub-task, addressing them independently to optimize performance.

To be specific, we use an Image Quality Assessment (IQA) model (see `LoDa`) to predict quality scores and an instance
segmentation model (see `Co-DETR`) to identify structural issue regions in the image.

## Usage

### Step1. Clone

```bash
# clone this repo
git clone git@github.com:wangzhiyu918/text-to-image-assessment.git
```

### Step2. Image Quality Assessment

1. Enter the specified directory

```
cd LoDa
```

2. Prepare environment

```
# Using conda (Recommend)
conda env create -f environment.yaml
conda activate loda

# Using pip
pip install -r requirements.txt
pip install timm==1.0.15
```

3. Data Preparation

You need to place the data under the `./data` directory as follows.

```
├── data
|    ├── evalmuse10k
|    |    ├── train
|    |    ├── test
|    |    ├── ...
|    ├── meta_info
|    |    ├── meta_info_EvalMuse10kDataset.csv
|    |    ├── ...
|    ├── train_split_info
|    |    ├── evalmuse10k_91_seed3407.pkl
|    |    ├── ...
```

4. Training 

```
# We use 8xA100 for training
bash scripts/benchmark/benchmark_loda_evalmuse10k.sh
```

5. Inference

```
# Download the pretrained model from:
# https://huggingface.co/datasets/wangzhiyu918/results/blob/main/loda_evalmuse10k_train_split0_1120.pt
#
# Place the model in the following directory:
# runs/loda_benchmark_evalmuse10k/loda_evalmuse10k_train_split0/chkpt_dir
#
# Then, execute the following command to generate results, which will be saved to ./loda_results.npy.
bash scripts/benchmark/benchmark_loda_evalmuse10k_infer.sh
```

### Step3. Structure Distortion Detection

1. Enter the specified directory

```
cd Co-DETR
```

2. Prepare environment

```
# Please follow the README to prepare environment
# https://github.com/Sense-X/Co-DETR/blob/main/README.md

# We strongly recommend creating a new Conda environment for Co-DETR
```

3. Data Preparation

You need to place the data under the `./data` directory as follows.


```
# evalmuse_trainval10000_coco_instances_v3.json and evalmuse_test999_coco_instances.json
# can be downloaded from: https://huggingface.co/datasets/wangzhiyu918/results/

├── data
|    ├── evalmuse10k
|    |    ├── train
|    |    |   ├──images
|    |    |   |--evalmuse_trainval10000_coco_instances_v3.json
|    |    ├── test
|    |    |   ├──images
|    |    |   |--evalmuse_test999_coco_instances.json
```

4. Training 

```
# Download CO-DETR COCO pretrained weights from:
# https://huggingface.co/zongzhuofan/co-detr-vit-large-coco-instance/blob/main/pytorch_model.pth
# 
# Place the model in the following directory: ./pretrained_weights/
# 
# We use 8xA100 for training
bash tools/dist_train.sh projects/configs/co_dino_vit/co_dino_5scale_vit_large_evalmuse_instance.py 8 work_dirs/co_dino_5scale_vit_large_evalmuse_instance
```

5. Inference

```
# Download the pretrained model from:
# https://huggingface.co/datasets/wangzhiyu918/results/blob/main/codetr-evalmuse.pth
#
# Place the model in the following directory: ./pretrained_weights/
#
# Then, execute the following command to generate results, which will be saved to:
# ./co_dino_5scale_vit_large_evalmuse_instance.segm.json

bash tools/dist_test.sh projects/configs/co_dino_vit/co_dino_5scale_vit_large_evalmuse_instance.py pretrained_weights/codetr-evalmuse.pth 8 --format-only --eval-options "jsonfile_prefix=co_dino_5scale_vit_large_evalmuse_instance"

# Convert instance into mask
# The results will be saved to: ./co_dino_5scale_vit_large_evalmuse_instance
python convert_instance_to_mask.py
```

### Step4. Prepare Submission File

```bash
# The submission file will be saved to: ./results.zip
python generate_submission_file.py
```

