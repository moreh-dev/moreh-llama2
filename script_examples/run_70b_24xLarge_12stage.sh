#!/bin/bash

python train.py \
  --model-name-or-path [YOUT_PATH_TO_MODEL] \
  --batch-size 1024 \
  --lr 1e-4 \
  --use-pipeline \
  --split-layers 6 13 20 27 34 41 48 55 61 67 73 \
  --num-micro-batches 32 \
  --bfloat16 \
  --block-size 2048 \
  --distribute-parameter \
  --enable-activation-recomputation
