#/bin/bash

export CUDA_VISIBLE_DEVICES=3
nohup python train.py --save_dir=save_w2v &
