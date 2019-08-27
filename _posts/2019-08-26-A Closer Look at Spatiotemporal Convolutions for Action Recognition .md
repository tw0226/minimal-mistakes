---
layout: post
title: A Closer Look at Spatiotemporal Convolutions for Action Recognition
tags: [Deep learning, Action Recognition]
excerpt_separator: <!--more-->
use_math : true
---

논문 참고는 [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/pdf/1711.11248.pdf) 에서 할 수 있습니다.

해당 논문은 Action Recognition을 위한 네트워크 구조인 3D Convolution에 대한 단순한 아이디어로 시작합니다.

이미지는 W x H x C 구조이기에 공간적인 특징을 잡아내는 2D Convolution이 쓰이지만

비디오의 경우는 W x H x C x T(video Length)라는 3차원 구조가 되고, 각 프레임은 이전, 이후 프레임과 연관되어 있기 때문에 공간적인 특징을 포함해 시간적인 특징도 잡아야 합니다.

해당 논문은 3D Convolution을 발전시켜 R(2+1)D 라는 구조를 제안합니다.

## Convolutional residual blocks for video

논문에서는 다음과 같은 실험을 진행하였습니다.

![Imgur](https://imgur.com/oEjXnFr.png)

### 3. Convolutional Residual Blocks for Video

#### 3.1 R2D

(a) 실험 네트워크 이름을 R2D로 붙였는데 여기서 R은 Residual Module을 의미하며, 비디오의 W x H x 3 x L 의 4D 형태를 W x H x 3L 의 3D형태로 바꾸어 2D Convolution으로 실험을 진행하였습니다.

2D Convolution의 특징으로 각 채널에 독립적으로 Convolution 연산이 적용되기 때문에 R2D 네트워크의 첫 layer는 시간적인 정보를 무너뜨리게 되고, 이후의 layer에서도 시간적인 추론이 불가능하다고 합니다.

#### 3.2 f-R2D

두 번째 실험은 f-R2D 라고 붙였는데 frame-based R2D 의 약자로 R2D를 변형시켜서 만들었다고 합니다.

방금 말한 것처럼 2D Convolution 시에 채널이 독립적으로 연산이 됩니다.

따라서 생기는 시간적 정보 붕괴를 막기 위해 3D Convolution 구조를 앞에 섞어서 진행해봤다고 합니다.

#### 3.3 R3D

3D Covolution은 시간적 정보를 보존하면서 이후 layer에 전달할 수 있는 구조로,

3D Convolution을 이용 시에 생기는 결과물은 N x L x W x H 의 구조로 N은 필터의 개수이고, L은 동영상 길이입니다.

따라서 3D Convolution은 N x T x d x d 의 구조를 가지게 되며, 논문에서는 보통 시간 정보량인 T를 다른 논문에서 쓰인 것과 같이 3으로 정했고, Residual Module을 이용하여 실험을 진행합니다.

위의 (d) 이미지를 참고하면 됩니다.

#### 3.4 MCx and rMCx

MCx의 뜻은 Mixed Convolution으로 여기서 저자의 가정이 하나 들어가는데,

시간적 정보를 담는 모션 모델링(ex:3D Conv)은 꼭 필요하지 않아서 초기 네트워크나 아니면 high-level의 정보를 뽑아내는 특정 부분에만 유효할 것이라고 생각하고, (b), (c)와 같은 구조를 제안합니다.

그리고 위의 사진과 같이 네트워크 별로 비교를 진행하였습니다.

논문에서는 MC5, MC4, MC3 (MCx) 순으로 네트워크의 마지막 단계부터 바꾸면서, 혹은 거꾸로 r(reversed)로 진행합니다.

위 그림의 (b), (c) 이미지를 참고하면 됩니다.

### R(2+1)D

논문에서 제일 중요한 2+1D 네트워크입니다.

![Imgur](https://imgur.com/3SOCv5b.png)

저자는 Separate Convolution 과 같이 3D Convolution의 T x d x d 구조를 T x 1 x 1, 1 x d x d 라는

두 개의 필터로 나누어서 적용하였습니다. (여기서 두 필터에 적용되는 채널의 개수는 동일합니다)

여기서 기존의 3D Convolution 과 차이가 생기는 것이 3D Convolution 의 경우 W, H, T 를 동일하게 할 시에

$N^3$이 됩니다. 같은 경우의 2+1D는 $N^2 x 1 + N x 1 x 1$ 이 되는데,

보통의 Convolution Filter 크기인 3으로 비교해도 C3D는 파라미터 개수가 27개, 2+1D는 12개가 됩니다.

논문에서는 2+1D를 3D Convolution의 파라미터 개수에 근사시키기 위해

필터 채널 개수$M_i$ 에 다음의 식을 적용하였습니다.

$$M_i =  \frac{(td^2N_{i-1}N_i)}{d^2N_{i-1}+tN_i} $$

분자는 layer 단계에서 일반적인 3D Convolution의 파라미터 개수이고, 분모는 2+1D의 파라미터 개수입니다.

3D Convolution과 비교했을 때 2+1D가 갖는 장점은 다음과 같습니다.

1. 파라미터 수가 변화가 없음에도 불구하고, non-linearity가 두배로 증가한다. 2D와 1D에 나눈 블록에 각각 ReLU와 같은 활성화 함수를 적용할 수 있기 때문에 비선형성이 두 배로 증가합니다. 비선형성이 증가하는 것은 함수의 복잡성 또한 표현될 수도 있다. 이는 VGGNet이 3x3 필터를 여러 개 씌우면서 더 큰 필터보다 더 좋은 성능을 낸 것과 비슷한 이유입니다.

2. 3D Convolution을 공간적, 시간적 요소로 구분 할 수 있어지면서 최적화가 더 쉬워진다. 같은 파라미터의 수로 학습시켰을 때, training loss가 더 낮았다고 합니다. (아래 그림 참조)

![Imgur](https://imgur.com/khkN3HR.png)

논문에서는 2+1D의 구조가 [Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf)라는 논문의 P3D 네트워크 구조와 많이 비슷하다고 합니다.

![Imgur](https://imgur.com/jgctYzm.png)

### 4. Experiments

논문에서는 성능을 검증하기 위해 Sports-1M, Kinetics, UCF101, HDMB51 네 개의 데이터셋에 적용시켰습니다.
실험을 진행하는 환경은 데이터로더 부분은 동영상을

* 128 x 171 사이즈로 변환 후, 그 중 112 x 112 크기를 크롭하여 사용하였고,

* GPU당 mini-batch는 32,

* optimizer는 SGD,

* 실험한 딥러닝 툴은 caffe2,

* 처음 lr는 0.01로 시작해 10 epoch 마다 1/10씩 감소시켜 45 epoch을 학습시켰습니다.

위를 조금 더 설명하면 네트워크의 입력은 112 x 112 x 3 x L인데, 여기서 L은 8, 16 두개로 실험하였다고 합니다.
이후에는 아래의 그림과 같은 네트워크 구조의 각 Convolution마다 Batch Normalization을 사용해 실험을 진행하였다고 합니다.

![Imgur](https://imgur.com/KlPOomK.png)

그리고 각 네트워크들과 비교한 결과는 다음과 같습니다.

![Imgur](https://imgur.com/AwhYiBs.png)

#### 4.3 Revisiting practices for video-level prediction

[Varol 의 논문](https://arxiv.org/pdf/1604.04494.pdf)에 의하면 네트워크의 입력 프레임 길이가 길수록, long-term convolution(LTC)를 사용하면 정확도가 올라간다고 합니다.

여기서도 마찬가지로 해당 실험을 진행하였는데, R(2+1)D 18layer 네트워크에 입력을 각각 8, 16, 24, 32, 40, 48으로 바꿔보며 진행시켰습니다.

이렇게 진행했을 때 마지막 layer의 Temporal 크기가 각각 바뀌는 것에 대해서 global average pooling을 이용했는데, 그 결과는 다음과 같습니다.

![Imgur](https://imgur.com/wURPwBG.png)

'여기서 모델의 파라미터 수는 분명 같은데, 왜 정확도의 차이가 발생하는걸까?' 라는 질문이 생겨서 또 다른 실험을 진행하였습니다.

실험의 내용은 아래의 그림과 같습니다.

![Imgur](https://imgur.com/xPligH6.png)

첫번째 8프레임에서 학습시킨 후, 8프레임, 32프레임에서 테스트시켰을 때 32프레임에선 정확도가 각각 1.2%, 5.8% 떨어진 것을 볼 수 있었습니다.

두번째로는 8프레임에서 학습시킨 후 32프레임으로 fine-tuning시킨 것과, 32프레임에서 학습시킨 것과 32프레임에서 테스트를 진행했는데요,

8/8 일때보다 8/32가 7%의 성능 향상이 있고, 32frame에서 학습시킨 것과 차이가 거의 없었습니다.

이 두 실험을 통해 프레임이 짧은 네트워크를 학습한 후 긴 프레임 네트워크로 fine-tuning하는 것이 훈련시간을 상당히 단축시킨다는 것을 확인할 수 있었고, 컨볼루션 필터가 더 긴 시간의 패턴을 학습이 가능하다는 것을 확인할 수 있었습니다.

또, 위 그림의 (b)는 실제 동영상에서 얼만큼의 frame을 clip하는 것이 정확도 측정에 도움이 되는지를 실험한 것으로, 20 crop이 100crop보다 0.5% 안 좋지만, 5배는 빠른 것을 확인할 수 있었다고 합니다.

마지막으로 각 데이터셋에 적용시킨 결과입니다.

![Imgur](https://imgur.com/CmniE8h.png)

![Imgur](https://imgur.com/De8j220.png)
