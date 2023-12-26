# Semi-Supervised Learning

## 모델 설명
Pi-Model은 레이블이 지정된(labeled) 및 지정되지 않은(unlabeled) 데이터에 대한 consistency을 활용하여 모델을 학습시키는 방법입니다. 이 모델은 크게 두 가지 단계로 나눌 수 있습니다.

### 레이블이 지정된 데이터 학습
- 레이블이 지정된 데이터에 대해 기본 모델을 사용하여 예측을 수행하고, Cross-Entropy Loss를 계산합니다.

### 레이블이 지정되지 않은 데이터 학습
- 레이블이 지정되지 않은 데이터에 대해서는 예측 및 consistency 손실 계산이 이루어집니다.
- 예측된 가짜 레이블(pseudo-labels)을 생성하고, 이를 사용하여 일관성 손실을 계산합니다.
- 모델의 예측과 Teacher Model의 예측 간의 일관성을 유지하기 위해 EMA(Exponential Moving Average)를 사용합니다.
- 이 일관성 손실은 모델이 레이블이 지정되지 않은 데이터에 대해서도 안정적으로 학습하도록 도와줍니다.

### 결과
- 실험 결과, overfitting이 심각하게 발생하며 validation accuracy가 46-7%부근에서 잘 올라가지 않는 현상을 발견했습니다.
- 여러 시도를 했지만 모든 경우에 동일하게 accuracy가 상승하지 않았습니다.
- 모델의 복잡성 문제로 mobileNetV2로 실험을 진행하였으나, resnet18을 사용할 때보다 성능이 낮았습니다.
- 데이터 변환과 관련하여 여러 실험을 진행하였으나 특별한 성능 향상을 관측하지 못했습니다.
- 추가적인 random augmentation 없이 normalize 수치만 변경한 것이 가장 좋은 성능을 보였습니다.

## Self-Supervised Learning

### 모델 설명
- simCLR을 활용했습니다. backbone은 semi-supervised와 마찬가지로 resnet18을 사용했습니다.
- 입력 데이터에 대한 두 가지 다른 변환(Transform)을 정의하고, 이를 DoubleTransform 클래스를 통해 구현했습니다.
- SimCLR 클래스는 핵심 모델을 정의하며, Noise-Contrastive Loss를 사용하여 모델을 학습합니다.
- self-supervised 학습 방법으로 학습된 SimCLR 모델의 인코더를 사용하여 선형 분류기를 정의한 LinModel 클래스를 구현했습니다.

### 결과
- Semi-supervised에 비해 좀 더 성능이 개선되었음을 확인할 수 있습니다.
- overfitting이 epoch 12 정도를 넘어가는 과정에서 관찰되었으며, best model은 validation loss가 최저일 때로 설정되었습니다.
- 추가적인 color jittering을 포함하여 데이터에 대한 심화된 학습을 진행했지만 큰 성능 향상은 없었습니다.
- 고도화된 모델을 사용했을 때도 큰 성능 향상을 보이지 않았습니다.

---

더 자세한 내용은 코드와 함께 확인해주세요!
