---
published: true
layout: single
title: Quo Vadis, Action Recognition
category: [Action Recognition, Deep learning]
toc: true
tags: [Action Recognition, Deep learning]
comments: true
toc_sticky: true
---

## 읽은 논문

[Quo Vadis, Action Recognition](https://arxiv.org/pdf/1705.07750.pdf)

논문에서 Contribution은 다음과 같다

### Contirbutions

1. Kinetics라는 새로운 비디오 데이터셋 제안

2. pre-trained 2D Conv network를 이용한 I3D Network

3. transfer-learning의 성능 검증

4. action recognition에 쓰이는 네트워크 비교

### Introduction

기존에는 ImageNet같은 데이터로 이미지를 기반으로 train시키고,
마지막 FC layer를 제외한 pre-trained된 네트워크를 다른 도메인에 사용하는 Transfer learning을 사용해왔다.

하지만 비디오라는 도메인에서는 (W, H, L) 라는 3차원 구조와 시간적인 특징을 잡아야 하므로 2D 구조에서 이루어진 Network들을 사용할 수 없었다.

    이 논문은 3D 구조를 이용하게 되는 비디오 도메인에서 pre-trained된 Network 구조를 이용할 수 없을까?

라는 질문을 시작으로 'Two-Stream Inflated 3D Convolution(I3D)' 구조를 제안한다.

결과적으로 말하면 Kinetics 데이터셋으로 pre-train 시킨 후, HDMB-51이나 UCF-101 같은 데이터셋에 대해 fine-tuning을 진행했는데, 성능이 향상되었다고 한다.

I3D 모델은 Inception-v1 구조를 기반으로 한다.

### Action Classification Architectures - 네트워크 비교

행동 분류 구조에 있어서 네트워크 입력으로 RGB 비디오인지, Optical flow를 포함했는지에 따라 2D 구조와 3D 구조의 커널을 사용하는지로 갈라진다.

아래는 Video Classification에 사용되는 대략적인 네트워크 구조이다.

![Imgur](https://i.imgur.com/lNy3HTF.png)

예를 들어 2D의 경우는 LSTM과 같은 시계열 Network 구조를 이용해 시간적인 특징을 잡으려는 경우도 있다.

3D ConvNet의 경우, 높은 차원에 비례한 파라미터 숫자로  8-layer까지 사용하는데 상대적으로 layer의 깊이가 얕다.

이 논문에서는 2D Conv+LSTM, Two-Stream Network, 3D Conv 구조에 대한 비교를 진행하고, I3D 네트워크를 소개한 후,

VGG, ResNet, Inception과 같은 image Classification Network를 팽창시켜 비디오 도메인에 맞는 '시공간 특징추출'을 할 수 있다는 것을 발견하고, two-stream 구조가 유용하다는 것을 발견했다고 한다.

#### ConvNet + LSTM

Image Classification Network의 높은 성능은 비디오에서도 가능한 최소의 변화를 주면서 다시 사용할 수 있지 않을까? 라고 시도한 논문들이 있었다. 이론적으로 더 만족스러운 접근은 LSTM이라는 시간적인 정보를 잡을 수 있는 RNN계열을 붙이는 것이다.

하지만 논문에서 주장하는 바에 의하면 실제로는 전반적인 시간적 정보를 무시한다는 이슈가 있었다고 한다.

논문에서는 이 구조로 25 FPS의 동영상을 5초동안 Input으로 넣어, Layer의 마지막 Output을 확인하는 구조로 실험을 진행해보았다.

#### 3D ConvNet

3D ConvNet은 비디오 모델링에 있어서 전형적인 CNN같아 보이지만, 시공간적인 필터를 가진다. 그리고 [3D ConvNet](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf)
에 있어서 문제점은
2D ConvNet에 비해 차원수가 추가되는 만큼 더 많은 파라미터를 가지고, 그만큼 학습시키기 어렵다는 단점이 있다.

(2D Conv는 NxN = $N^2$ 이지만,
3D Conv는 N x N 가 N개 있으므로 N x N x N = $N^3$ 이다.
흔히 필터 크기로 쓰이는 3만 넣어도 알 수 있다)

이 경우에서는 이미지넷 데이터를 활용하는 것이 어렵기 때문에 네트워크를 처음부터 만들어서 학습시켜야 했고, 파라미터가 더 많은 만큼 학습이 더디다.

(참고로 C3D Network에서는 8개의 3D Conv layer와 5개의 pooling layer로 이루어져 있다.)

#### Two-Stream Networks

LSTM이 CNN의 결과로 나온 high-level의 feature를 이용해 잡는다지만,
많은 경우에는 low-level의 모션이 중요한데 이를 잡지 못할 때가 있을지도 모른다.

또, 동영상의 긴 frame동안 역전파를 계산하기 위해 네트워크를 풀어나가는 과정은 학습시키기 위해선 비싼(expensive) 일이다.

[Two-Stream](https://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf) 논문은 다른 접근으로 optical flow를 이용해 시간적 스냅샷을 만들고, 이를 10개를 쌓아 네트워크를 진행하였다.

최근(당시의) 논문은 마지막 Conv layer에 Optical flow stream과 Spatial stream 두 개를 융합하여 사용하는 논문이였는데,
이 논문에서는 추가로 Inception-V1을 이용한다. 입력으로는 10프레임 간격으로 된 5개의 RGB 이미지와 Optical Flow 이미지로,
Inception-V1(5 x 7 x 7)에서 마지막 average pooling을 거치기 이전에
512 x 3 x 3 x 3 (Channel x frame x $x$ x $y$) 모양의 3D convolution과 3 x 3 x 3 크기의 3D max pooling과 FC를 한다.
(이 때 layer의 초기값은 Gaussian을 따른다고 한다)

#### Two-Stream Inflated 3D ConvNets

Two-Stream Inflated 3D ConvNet은 해당 논문에서 제안하는 네트워크 구조로 이미지(2D) 분류 모델을 3D ConvNet으로 변환할 수 있다고 하는데, 2D 네트워크 구조에 필터를 부풀려(Inflate) N x N(2D) 에서 N x N x N (3D) 으로 만든다고 한다.

##### Bootstrapping 3D Filters from 2D Filters

논문에 따르면 ImageNet으로부터 pre-trained 된 파라미터들을 3D 모델에도 적용이 가능하다고 한다.

논문에서는 방법으로 이미지를 반복적으로 쌓아 비디오 시퀀스처럼 만드는데 이를 'boring-video'이라 한다.

![Imgur](https://i.imgur.com/Uak62hy.png)

위의 사진과 같이 동영상의 구조를 지녔지만, 움직임이 없는 지루한 (boring) 동영상(video)이 되는 것이다.

boring-video을 3D Model에 pooling, activation 등의 네트워크 결과를 거친 것이 2D 이미지 입력과 같도록 만드는데, 이를 boring-video fixed point 라고 한다.

논문에 따르면 2D 필터를 반복적으로 N번 쌓았다가, 다시 N으로 나누어 재구축하면 만들어진다고 한다.

boring-video는 시간이 변함없이 일정하므로, 네트워크의 결과가 2D에서 나온 결과와 같아야 한다. 따라서 전반적인 네트워크의 반응은 boring-video fixed point를 준수한다고 볼 수 있다.

##### Pacing receptive field growth in space, time and network depth

논문에 따르면 boring-video fixed-point는 충분한 시간축이나 컨볼루션/풀링 레이어를 설정하는데 있어서 자유도를 보장한다고 한다.

Convolution이나 pooling을 통해 나오는 원본에 대한 수용영역은 보통 공간적인 차원에 대해서 진행되고, 네트워크가 진행될수록 수용영역이 넓어지는 것은 자연스러운 일이지만,

하지만 시간을 고려했을 때 수용영역이 대칭적인 것은 꼭 필요한 일은 아니라, fps나 이미지의 차원에 비례해야 한다고 한다.
만약 공간에 비해 시간적인 부분이 빨리 학습될 경우, 초기 특징 검출할 때 edge를 혼동할 수도 있고, 반대로 너무 느리게 학습될 경우, 움직임을 잘 잡지 못할 수도 있다.

실험에 따르면 처음 max-pooling layer일 때 temporal pooling을 진행하지 않는 것이 도움이 된다고 한다.
(처음 2 layer에서는 1 x 3 x 3 kernel, stride = 1로 했다고 한다)
마지막 average pooling layer에서는 2 x 7 x 7 사이즈의 커널을 사용해 64-frame의 입력을 갖는 (25fps의 동영상에서) 모델로 진행했다.

말이 길었지만 temporal pooling을 진행하는데 있어서 실험해본 결과,
초반 layer에서 temporal pooling을 진행하지 않는 것이 더 좋다는 것을 깨닫고 2D Kernel 크기와 다르게 설정해서 진행했다는 이야기다.

##### Two 3D Streams

3D ConvNet은 RGB입력에서 모션 피쳐를 직접 학습할 수 있어야 하지만, feedforward 계산만 수행하는 반면, 

optical flow는 어떤 의미에서는 반복적이다. RGB입력은 이런 반복성이 부족하기 때문에, two-stream 구조로 넣어 원활한 흐름을 갖게 하는 것이 중요하다고 한다.

논문에서는 두 네트워크를 별도로 훈련하고, test-time 때 예측을 평균화시켰다고 한다.

#### Implementation Details

C3D 같아 보이면서도 pre-trained Inception-V1을 베이스로 하는 3D ConvNet이다. Batch Normalization과 활성화 함수로 ReLU를 사용했고, 학습은 SGD로 momentum=0.9로 사용했다고 한다.

입력 이미지는 256 x 256을 랜덤으로 짤라 224 x 224 로 진행했다고 한다.

아래는 네트워크의 전반적인 구조이다.

![Imgur](https://i.imgur.com/rXvUof5.png)

마지막으로 논문에서 말한 네트워크 간 비교이다.

![Imgur](https://i.imgur.com/96w6COr.png)

위의 환경으로 설정하고 실험을 진행한 결과는 아래와 같다.

![Imgur](https://i.imgur.com/pdrzHhG.png)

동영상이 진행되는동안 네트워크를 25frame으로 균등하게 적용시킨 후, 평균값을 낸 것이다.
아마 논문의 흐름상 당연하겠지만 논문에서 제안하는
(e) 부분의 네트워크의 성능이 제일 좋다.

![Imgur](https://i.imgur.com/GSuku6D.png)

논문에서 말했던 Contribution중 하나로 Kinetics 데이터셋에 ImageNet을 통한 학습 결과와 ImageNet을 사용하지 않았을 때 결과이다.

왼쪽을 보면 ImageNet을 학습에 이용한 경우 성능이 대략 3%정도 향상된 것을 볼 수 있다.

### Comparison with the State of the Art

![Imgur](https://i.imgur.com/brooELo.png)

마찬가지로 다른 데이터셋인 UCF-101과 HMDB-51에 적용했을 때로, 
한 cell 기준으로 왼쪽이 ImageNet을 pre-training 시킨 것, 오른쪽은 학습시키지 않았을 때 결과이다.
Original은 같은 데이터셋에서 학습과 테스트를 진행했을 때,

Fixed는 Kinetics에서 pre-training 후 마지막 layer만 바꿔 학습을 진행시키는 transfer-learning을 진행시켰을 때,

Full-FT는 Kinetics Dataset에서 pre-training 후, UCF-101, HMDB-51 데이터셋에 맞춰 fine-tuning을 진행했을 때이다.

![Imgur](https://i.imgur.com/4PANSao.png)

위는 마지막으로 다른 논문들과의 성능을 비교한 것이다.

확실하게 알 수 있는 것은 C3D Network를 앙상블한 것보다 I3D Network가 성능이 확연하게 차이가 난다는 것을 알 수 있었다.