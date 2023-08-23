# Moreh LLaMA2 Alpaca fine-tuning example
## Prerequisite

Hugging Face format 으로 변환된 모델 & tokenizer 가 필요합니다.

## Install Dependeicies

```
pip install -r requirements.txt
```

## How To Run

### 7B 4node 2 Pipeline Parallel stage example 

```
#!/bin/bash

python train.py \
  --model-name-or-path [YOUT_PATH_TO_MODEL] \
  --batch-size 1024 \
  --lr 1e-4 \
  --use-pipeline \
  --split-layers 15 \
  --num-micro-batches 32 \
  --bfloat16 \
  --block-size 2048 \
  --distribute-parameter
```


### 33B 12node 6 Pipeline Parallel stage example 

```
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
```
### 70B 24node 12 Pipeline Parallel stage example 

```
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
```

## Arguments
| Parameter|Type| Default| Description|	
| :------------ | :-------------: | :------------: | :------------- |
| --model-name-or-path | str  | | model name or path |
| --batch-size | int  | 8 | number of examples for each training iteration |
| --lr  | float  | 0.00001 | learning rate |
| --bfloat16 | bool  | false | whether to use bfloat16 |
| --distribute-parameter | bool  | false | whether to distribute fp32 master parameters |
| --num-micro-batches | int  | 1 | split batch to N steps (micro batches) |
| --log-interval | int  | 10 |logging interval|
| --save-model-dir | str  | ./ | path to save model at the end of training |
