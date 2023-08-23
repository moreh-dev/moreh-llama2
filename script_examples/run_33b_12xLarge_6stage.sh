#!/bin/bash

python train.py \
  --model-name-or-path [YOUT_PATH_TO_MODEL] \
  --batch-size 1024 \
  --lr 1e-4 \
  --use-pipeline \
  --split-layers 9 19 29 39 49 \
  --num-micro-batches 32 \
  --bfloat16 \
  --block-size 2048 \
  --distribute-parameter \
  --enable-activation-recomputation
