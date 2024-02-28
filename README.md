# Moreh LLaMA2 Alpaca fine-tuning example
## 준비 사항

Moreh LLaMA2를 사용해 학습/추론하기 위해서 모델 체크포인트 및 토크나이저를 준비해야합니다.     
이 때 체크포인트는 Hugging Face 형식으로 변환된 것이어야합니다.       
이에 대한 자세한 내용은 Hugging Face의 [Docs](https://huggingface.co/docs/transformers/en/model_doc/llama2)를 참고하시면 좋습니다.     


## Dependencies 설치

Moreh LLaMA2를 사용하기 위해 환경 세팅을 합니다.


```shell
pip install -r requirements.txt
```

## 학습 실행

### LLaMA2 7B 모델 학습 예제

아래의 스크립트를 사용해 모델을 학습시킬 수 있습니다.     
train.py 코드에는 모델을 병렬화하는 Moreh의 Advanced Parallelization 기법이 적용되어있어,      
사용자는 사용할 노드 수(SDA 모델)만 미리 설정해두면 별도의 병렬화 필요없이 LLaMA2 모델을 학습시킬 수 있습니다. 

```
#!/bin/bash

python train.py \
  --model-name-or-path [YOUT_PATH_TO_MODEL] \
  --batch-size 1024 \
  --lr 1e-4 \
  --block-size 2048 
```


## Arguments
| Parameter|Type| Default| Description|	
| :------------ | :-------------: | :------------: | :------------- |
| --model-name-or-path | str  | | model name or path |
| --batch-size | int  | 8 | number of examples for each training iteration |
| --lr  | float  | 0.00001 | learning rate |
| --log-interval | int  | 10 |logging interval|
| --save-model-dir | str  | ./ | path to save model at the end of training |
