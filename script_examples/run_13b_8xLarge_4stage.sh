#!/bin/bash

python train.py \
  --model-name-or-path [YOUT_PATH_TO_MODEL] \
  --batch-size 1024 \
  --lr 1e-4 \
  --use-pipeline \
  --split-layers 9 19 29 \
  --num-micro-batches 64 \
  --bfloat16 \
  --block-size 4096 \
  --distribute-parameter \
  --enable-activation-recomputation
  
