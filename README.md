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
```

3. Data Preparation

You need to place the data under the `./data` directory as follows.

```
├── data
|    ├── evalmuse10k
|    |    ├── train
|    |    ├── test
|    |    ├── ...
```

4. Training 

```
# We use 8xA100 for training (takes about 10h)
bash tools/dist_train.sh projects/configs/co_dino_vit/co_dino_5scale_vit_large_evalmuse_instance.py 8 work_dirs/co_dino_5scale_vit_large_evalmuse_instance

```

5. Inference

```
# The pretrained model is located at 
# runs/loda_benchmark_evalmuse10k/loda_evalmuse10k_train_split0/chkpt_dir
# And the results will dump into ./loda_results.npy
bash tools/dist_test.sh projects/configs/co_dino_vit/co_dino_5scale_vit_large_evalmuse_instance.py work_dirs/co_dino_5scale_vit_large_evalmuse_instance/epoch_10.pth 8 --format-only --eval-options "jsonfile_prefix=co_dino_5scale_vit_large_evalmuse_instance"

```

### Step4. 

```bash
# download test data (i.e., query_drone_160k_wx_24 and gallery_satellite_160k) to the ./data
# change the checkpoint path in eval_mm_2024.py (line54)
python eval_mm_2024.py
```

