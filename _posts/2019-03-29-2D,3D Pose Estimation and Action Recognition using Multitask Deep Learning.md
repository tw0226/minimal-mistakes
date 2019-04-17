---
published: true
layout: single
title: 2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning
category: [Action Recognition, Deep learning, Pose Estimation]
toc: true
tags: [Action Recognition, Deep learning, Pose Estimation]
comments: true
use_math : true

---

읽은 논문

# 2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning

핵심 선행 논문

[1] [Human Pose Regression by Combining Indirect Part Detection and Contextual Information](https://arxiv.org/pdf/1710.02322.pdf)

pose estimation 네트워크 단계까지 end-to-end 학습이 가능한지 증명한 논문

[2] [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/pdf/1603.06937.pdf)

현재 논문의 Pose Estimation에서 쓰이는 Network 구조를 제안한 논문

[3] [3D Convolutional Neural Networks for Human Action Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6165309)

3D convolution을 통해 행동 인식이 가능하다고 제안한 논문

논문의 Introduction 마지막에 보면 다음과 같은 내용이 있다.

## 논문에서 말하고 싶은 것

1. 3D pose estimation에서 SOTA급 성능을 달성하고, 2D pose estimation에서 regression model을 사용한 것들 중 가장 높은 정확도이다.

2. 일반 영상(still image)에서 기반되었기 떄문에, 일반(in the wild) 영상에서도 성능이 좋다.

3. 논문의 행동 인식(action recognition) 모델은 RGB 이미지로부터 나온 포즈와 시각 정보를 기반으로 했다.

4. 자세 추정(pose estimation) 모델은 2D 정답 데이터로부터
    3D 예측까지 일반화 시킬 수 있는 다양한 데이터셋에서 동시에 학습이 가능하다.

이렇게 적혀있으나, 내 생각에 중점을 둬야할 부분은 자세 추정까지 end-to-end 학습이 가능한 [[1]](https://arxiv.org/pdf/1710.02322.pdf)번 논문을 가지고
행동 인식까지도 end-to-end 학습이 가능한 지 시도를 해보았다는 것 같다.

Pose estimation을 해결 하는 데에는 보통 두가지의 접근 방식이 있는데, Detection 기반 방식과 Regression 기반 방식이 있다.
Regression 방식은 초기에 많이 사용되었지만, Detection 기반이 성능이 더 좋게 나오자 상대적으로 덜 사용되고 있는 추세라고 한다. ([1] Introduction 참고)
하지만 이 논문에서는 Regression 으로 접근한다.
(Detection으로 하면 Argmax로 가야한다는데 흠.. 기억이 잘..)

보통 행동 인식까지는 네트워크 구조가 4단계로 되어있는데,

1. 특징 추출을 위한 CNN 네트워크

2. 추출된 특징들로 관절들을 찾아내는 네트워크

3. 탐지된 관절들을 가지고 자세를 추정하는 Pose 네트워크

4. 자세들(t Frames)을 가지고 행동을 인식하는 Action 네트워크로 되어있다.

## 네트워크 구조

제안하는 구조는 3가지 네트워크 구조로 이루어져 있는데, 다음과 같다.

1. Multi-Task CNN

2. Appearance recognition

3. Pose recognition

### Multi-task CNN

Multi-task CNN의 내용은 CNN을 거쳐 나온 Feature들로 포즈까지 추정하는 부분이 담겨있는데,
이 부분은 [1]번 논문을 참고하면 될 듯 하다.

조금 덧붙여 설명하면 [2]번에서 쓰이는 Hour-Glass 네트워크 구조를 사용하였고,

네트워크 마지막 단계에서 예측된 관절 부분과 정답값의 관절 loss function을 구하기 위해선 argmax를 사용하게 되는데,
argmax는 미분이 불가능하여 네트워크 전체를 학습시킬 수 없다.
(논문 Figure 3. 참조)

(10x10 heatmap을 생각해보자. 제일 높은 값이 있는 부분을 관절이라 예측할 때 argmax를 통해 index값을 구하여
loss function을 구성하게 된다)

( ex) GT(다리) : (8,8) Y_hat(다리) : (6,7) )

### Soft-Argmax

[1]번 논문을 참고해 soft-argmax라는 개념을 도입하면 argmax를 쓰지 않고도 관절의 위치를 예측할 수 있게 된다고 한다.

그렇다면 어떻게 soft-argmax가 argmax의 역할을 대신하게 되는걸까?

이 전에 선행해야 할 것이 하나 있는데 GT 값을 0~1 사이값으로 바꿔주는 것이다.

soft-argmax의 첫번째 단계로 네트워크의 입력 HeatMap을 softmax를 통해 0~1 사이 확률값으로 변환시킨다.

그리고 HeatMap의 각 픽셀에 x,y 값은 $\frac{i}{W}$, $\frac{j}{H}$를 곱한 후 Sum을 취한다.

내 생각으로 YOLO에서 IOU를 계산하기 위해 grid의 인덱스를 offset으로 준것과 비슷한 효과라고 볼 수 있는데,

각 픽셀 입장에서 보면 $\frac{i}{Width}$ 를 곱하게 되면 index값을 주는것과 같은 효과를 볼 수 있게 되고,

이는 역전파가 가능해져 argmax와 비교한다면 별도의 처리가 필요없게 된다.

[1]논문에서 주장하는 바로 이 soft-argmax를 사용하게 되면 sub-pixel 정확도를 예측할 수 있다는 점,
추가로 인위적으로 GT를 만들 필요가 없어진다고 한다.

(이를 통한 예시는 Figure 5.를 참조하면 된다)
그림 2개 첨부하자(GT Heatmap, Soft-argmax Map)

### Pose recognition

다음으로 Pose Recognition 부분인데 Pose Recognition 분야를 처음 해봐서 그런지 이 부분이 아주 흥미롭다.
Pose estimation의 예측 값으로 각 관절들마다 값이 나오게 되는데, 이를 시간 축으로 쌓아서 Convolution을 한다.
구체적인 예를 들어서 말하면,

$\hat{X}$ = [$\hat{X}_{0}$, $\hat{X}_{1}$,$\hat{X}_{2}$, ... , $\hat{X}_{n}$]

$\hat{Y}$ = [$\hat{Y}_{0}$, $\hat{Y}_{1}$,$\hat{Y}_{2}$, ... , $\hat{Y}_{n}$]

N은 관절의 개수고, 각 index는 관절들의 예측값을 말한다고 가정하자.

이런 $\hat{X}$ 내부의 관절 개수(N)를 가로(X)로 쌓고,이 T Frame개 존재하게 되면 T를 세로축(Y)으로,

마지막으로 이미지의 채널(2D, 3D) 개수를 Z축으로 (X,Y,Z)를 만들면

$N$(관절 개수) x $T$(frame) x $C$(Channel) 형식의 행렬 구조를 가지게 되는데,

여기에 Convolution 을 적용시켜서 action HeatMap을 만들고,

K개의 Block을 가진 새로운 [2]Hour-Glass 네트워크 구조에 입력으로 넣는다.

### Appearance recognition

Appearance recognition 부분도 Pose recognition과 많이 다를 게 없다.
CNN을 거쳐 나온 Visual Feature들을 관절들과 관련된 특징으로 뽑아내기 위해 Probability Map과 외적을 하고,
후처리를 통해 Apprearance Feature(V)를 얻는다.

Visual Feature의 차원은 $W$ x $H$ x $F$(Feature Map 개수) 이고 ,

Probability Map의 차원은 $W$ x $H$ x $N$(관절 개수)이다.

단순하게 말하면 1개의 Feature를 N개 사이즈만큼 곱한 것이고, 그게 F개 있다면
그 사이즈는 $W$ x $H$ x $N$ x $F$ 의 크기를 갖게 된다.
$W$ x $H$는 2차원 Map을 1개의 값으로 합하게(∑) 되면서 차원이 사라지고 1개의 값이 된다.

이를 행렬로 표현하면 $N$ x $F$ 이라는 행렬이 되는데,
$N$ x $F$ 행렬에 대해 다시 time축으로 쌓게 되면 $N$ x $F$ x $T$(time) 이라는 3차원 행렬이 만들어진다.

논문에선 이렇게 만들어진 3차원의 Apprearance Featrues(V)를 위의 Pose recognition과 같이 Hour-Glass 네트워크 구조에
입력으로 넣는다고 한다.

### Action aggregation

마지막으로 Action Aggregation을 진행하는데, 논문에서는 다음과 같이 말한다.

'물을 마시는 것' 과 '전화를 하는' 행동은 관절 움직임이 너무나 비슷하지만,
'컵'이나 '폰'이라는 객체 정보가 주어진다면 쉽게 분류할 수 있다.

반대로 '가슴에 손을 얹는 경례' 나 '가슴을 만지는' 행동은 외관적으로 쉽게 구분하기 어렵지만,
이런 경우 포즈 정보는 보완적으로 쉽게 정보를 제공 할 수 있다.

이런 이유로 Action aggregation 이라는 작업을 통해 pose 정보와 appearance 정보를 합한다.

Convolution으로 진행한 이유는 '악수' 를 예를 들었을 때 Fully-Connection으로 사용하면
별로 관련 없는 관절들이 필요해지기 되고, 이렇게 되면 학습이 어려워진다고 설명하였다.

반대로 2D Convolution으로 사용하게 되면 이런 지역적 특징들이 학습이 쉬워진다고 한다.

내 생각으로는 convolution 특성상 지역적이다 보니, 관절간의 거리가 먼 행동들은 제약이 있을것 같고,
그리고 프레임이 긴 행동에 대해서는 값이 올바르게 나오기 힘들지 않을까 싶다.