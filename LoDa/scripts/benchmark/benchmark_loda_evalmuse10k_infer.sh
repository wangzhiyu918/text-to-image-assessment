#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    job=eval \
    run_group=loda_evalmuse10k_eval \
    name=loda_evalmuse10k_eval_split"${split_index}" \
    split_index="${split_idx}" \
    data=evalmuse10k \
    load.network_chkpt_path=runs/loda_benchmark_evalmuse10k/loda_evalmuse10k_train_split0/chkpt_dir/loda_evalmuse10k_train_split0_1120.pt
