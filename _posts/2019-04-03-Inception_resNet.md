---

published: true
layout: single
title: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
category: post
toc: true
tags: [Inception v4, googleNet, Inception-ResNet]
comments: true
use_math : true

---

## Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning

핵심 선행 논문

[1] Inception-V1
[Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf) (GoogleNet)

[2] Inception-V2, V3 [Rethinking
 the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)

[3] resNet
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

## 논문에서 말하고 싶은 것

1. Inception 모듈의 변화(v4)

2. 2015년 우승한 ResNet을 inception에 접목시킨 inception-resNet의 성능

3. residual connection이 학습에 좋은건 알겠지만 이미지 인식 분야에서는 써야하는 지 잘 모르겠다.

### Incetpion

저자는 기존에는 layer들을 나누어서 학습을 진행시켰지만, Tensorflow를 쓰게 되면서 학습을 제어하기 쉬워졌고, 그에 따라 실험들을 여럿 진행해본듯 하다.
네트워크마다 성능이나 메모리 사용량을 부분적으로 최적화(optimization)시키기 위해 시도하고 개선한 모델이 inception-v4이다.
그리고 이 v4를 resNet구조와 접목시켜 성능이 어떻게 나오는지 실험하였는데,
이렇게 적용한 모델이 Inception-resNet-v1, v2이다.

Inception-resNet-v1, v2는 각각 Inception-v3, v4와
계산량이 비슷하다고 한다. 하지만 시간을 놓고 보았을 때 v4가 layer가 조금 더 깊어 Inception-resNet-v2보다 느리다고 하였다.

(외관상으로 보는 layer 수는 Inception-v4가 더 작지만 Inception 모듈 안에서
네트워크 깊이는 v4가 더 깊다는 것을 말하는 것 같다.)

개선된 모델들에 대해 그림을 보여주며 설명을 하는게 좋을 것 같다.

논문의 그림 안에 있는 V는 same-padding이라는 의미로, 입력과 출력의 크기가 같다.

[ppt 내용 넣기...]

### 논문에서 시도해본 것

1. Residual Inception Blocks

    논문에서는 Inception 모듈을 기준으로 residual을 쓴 것과 안 쓴것의 차이를 말하고 싶어한 듯 하다. 하지만 조금 차이를 둔 것은 Inception-resNet에서 Batch-Normalization(BN이라 하겠음)을 오직 한 레이어가 끝날 때에만 사용했다고 한다. 그 이유로는 다음과 같다.

    * BN을 사용할수록 유리한 건 알겠지만, single-GPU에서도 학습이 가능한 모델을 만들고 싶었기 때문이라고 한다.

    * 또 하나로 큰 activation을 가지는 layer일수록 부적합한 메모리 사용량이 많다는 것을 발견하였기 때문이다. 따라서 BN을 생략하게 되면서 Inception 블록의 수를 늘릴 수 있었다고 한다.

2. Scaling of Residual

    논문에서 발견한 것 중 하나는 필터의 개수가 1000개를 넘어가게 되면 residual이 불안정해지기 시작하고,
    학습하는 중에서도 average pooling 하기 이전 마지막 layer에서 0을 만들어 네트워크가 '**죽게**' 된다고 한다.
    이 것은 learning-rate를 줄이거나 BN을 추가로 써도 해결할 수 없었지만,
    (구체적으로는 필터 개수가 높은 상황에서 learning_rate를 1e-5(0.00001)로 아주 낮게 주었는데도 불안정했다고 한다)
    activation되기 전 단계에서 residual을 0.1~0.3 사이로 scaling하는 것이 안정적이라는 것을 발견하여 사용했다고 한다.

    [Figure 20 사진 넣기]

    비슷하게 resNet에서도 비슷한 불안함이 있어서 2단계로 학습을 진행시켰는데,

    1번째 단계로 **낮은** learning rate로 학습시키고,

    그다음에 **높은** learning rate로 학습시켰다고 한다.

### 결과

논문에서 학습을 시킨 결과는 다음 사진과 같다.

![Imgur](https://i.imgur.com/OrPLSqc.png)
![Imgur](https://i.imgur.com/rr4A1hl.png)

![Imgur](https://i.imgur.com/41Z5x6x.png)

위의 그래프를 보면 마지막 부분은 비슷하지만, resNet을 이용한 모델은 학습이 조금 더 빠르게 진행되는 것을 알 수 있지만, 결과적으로 에러율은 비슷하다고 볼 수 있다.

논문에서도 낸 결론은 다음과 같다.

[Conclusion 사진]

* Inception-resNet-v1 : Inception-v3과 계산량은 비슷한 hybrid Inception 버전

* Inception-resNet-v2 : 성능이 개선되었지만, 계산량도 더 상승한 hybrid Inception 버전

* Inception-v4 : pure Inception이지만 성능은 Inception-resNet-v2와 비슷하다.