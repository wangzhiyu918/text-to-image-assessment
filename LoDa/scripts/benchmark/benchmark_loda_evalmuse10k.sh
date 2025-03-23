#!/bin/bash

split_idx=0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/trainer.py \
    job=train_loda_evalmuse10k \
    split_index="${split_idx}"
