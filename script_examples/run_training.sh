#!/bin/bash

MODEL_DIR=$1

python train.py \
  --model-name-or-path $MODEL_DIR \
  --batch-size 1024 \
  --lr 1e-4 \
  --block-size 2048
