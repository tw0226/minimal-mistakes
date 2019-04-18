---
published: true
layout: single
title: Stacked Hourglass Networks for Human Pose Estimation
category: [Deep learning, Pose Estimation]
toc: true
tags: [Deep learning, Pose Estimation]
comments: true
toc_sticky: true
read_time: true
use_math : true

---

읽은 논문

## Stacked Hourglass Networks for Human Pose Estimation

핵심 선행 논문

[1] [DeepPose: Human Pose Estimation via Deep Neural Networks](http://openaccess.thecvf.com/content_cvpr_2014/papers/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.pdf)

## 관련 연구

DNN을 이용한 Pose Estimation 연구는 DeepPose 라는 논문에서 시작되었는데,
 논문에서는 x,y 좌표를 regression으로 접근해왔다.

(아직 읽진 못했지만) 그 이후로 다양한 종류의 feature를 얻기 위해 다양한 해상도에서 병렬로 실행하여  heatmap을 생성하는 논문[[2]](http://www.robots.ox.ac.uk/~vgg/rg/papers/tompson2014.pdf)도 있었고, 그래픽 모델로 생성하게 되면 관절간의 공간적인 연관성이 생긴다고 한다.

또 다른 논문[[3]](https://papers.nips.cc/paper/5291-articulated-pose-estimation-by-a-graphical-model-with-image-dependent-pairwise-relations.pdf)은 detection된 정보들을 합쳐서(cluster), 인접 관절을 예측할 때 분류기(classifier)가 추가적인 정보를 제공할 수 있다고 제안했다.

이외에도 Multi-stage 구조로 반복학습을 하는 등의 방식들이 있어왔다고 하지만, 해당 논문에서는 어떻게 진행했는지 살펴보자.

## 네트워크 구조

포즈를 추정하는 데 있어서 중간 단계에서는 얼굴이나 손같은 특징들을 구분하는 것이 필요하지만, 마지막 단계에서는 몸 전체를 이해하는 것이 필요하다.
저자는 다양한 스케일에서 특징을 뽑을 게 필요하다고 생각했던 게 네트워크 구조의 동기부여가 되었다고 한다.

사실 네트워크 구조는 Segmentation에서 쓰이는 FCN의 conv-deconv, Auto-Encoder 에서 encoder-decoder와 비슷하다.

    차이점은
    1. skip-connection을 이용한다는 점과,
    2. 네트워크를 1번 거치는 것이 아니라 K번 쌓는다는 점,
    3. 네트워크를 거칠 때마다 중간 지도(intermediate supervision)를 진행시켰다는 점이다.

중간 학습은 googleNet에서 보조분류기를 둔 것과 비슷하게 학습이 잘 진행되도록 도와준다.

논문에서는 [2]와는 다르게 다양한 해상도의 이미지에서 특징을 뽑는게 아닌 single 이미지에서 입력을 256x256으로 고정시켜 진행한다.

마지막으로 CNN과 pooling 연산을 거쳐 제일 작아지는 크기는 4x4가 되고, 이후로 다시 up-sampling을 진행한다.

이 구조에 대해 많은 실험들을 진행하였는데, 아래의 그림과 같다.

[그림1 : Figure 8]

첫 번째로는 구조적인 실험으로 네트워크를 1개를 깊게 만드는 작업과,

네트워크를 얕게 여러번 만드는 것, 그리고 중간 지도(intermediate supervision)을 어느 때에 주는 것이 좋은지를 비교한 그림이다.

그림에서 보라색을 보면 네트워크를 쌓지 않고, 정답값을 네트워크 마지막에 한번 주었을 때가 성능이 제일 낮다.

파란색을 보면 1개의 Hourglass 구조를 네트워크를 깊게 짠 경우가 그 다음으로 낮고,

초록색은 네트워크를 1개로 깊게 짜고, 중간 지도를 한 경우,

빨간 색은 네트워크를 얕지만 여러 개를 짜고 마지막에 정답값을 준 경우,

마지막으로 하늘색은 네트워크를 여러 개 쌓고, 중간에 정답값을 준 경우이다.

두 번째 실험으로는 네트워크를 구성하는 Hourglass 개수에 따라 각각의 Hourglass가 얼만큼의 성능을 지니는지를 진행하였다.

저자의 의도는 네트워크가 단순히 크고, 깊어질수록 성능이 증가한 것이 아니라

네트워크 '구조'에서 기인한다는 것을 보여주려고 했다고 한다.

아래 그림은 각 Hourglass의 성능과 예측을 보여준다.

[그림 2 : Figure 9]

오른쪽 그림을 보면 점이 hourglass의 개수만큼 존재하는데

이는 각각 Hourglass 마지막에서 성능을 측정한 결과로,

Hourglass 개수가 감소할수록, 각각의 hourglass는 성능이 증가한다.

하지만, 개수를 쌓을수록 정확도가 조금씩 오르는 것도 확인할 수 있다.

마지막 : (87.4%, 87.8%, 88.1%)

중간 : (84.6%, 86.5%, 87.1%)

다른 말로 1개의 네트워크를 깊게 쌓기보단, 여러 개를 쌓는 게 더 학습이 잘 진행되었다고 볼 수 있다.

마지막으로 논문에서는 한 명을 잘 잡으려는 의도로 만든 네트워크이기 때문에
여러 명의 사람들에 대해서는 잘 되지 않는다고 말한다.

그러면서 데이터셋이 정확하지 않다는 지적을 했는데,

당시의 MPII 데이터셋이 어땠는진 모르겠지만 720x1280 이미지 내에서 무용수 두명에 대한 Annotation이 26픽셀밖에 되지 않는다고 말하였고,

한 이미지 내에 여러 명이 존재해서 사람이 겹쳐 있는 경우에 보이지 않는 관절에 대해서도 Annotation이 충분히 되어있지 않다고 하였다.
논문에 따르면 Annotation된 손목과 팔꿈치 중 75%만 보이고 나머지는 보이지 않는다고 한다. 보이는 75%의 관절에 대해서만 진행하였을 경우 85.5%에서 93.6%까지 나왔다고 한다.

## 정리

Stacked-hourglass 논문은 네트워크 구조(Architecture)적으로는 Auto-Encoder, FCN등과 비슷해보이지만,

* Hourglass 구조를 여러 개 쌓음

* Skip-connection을 이용

* 중간지도(Intermediate Supervision)을 이용

위의 점들이 다르고, Pose Estimation 분야에서 많이 인용되는 논문이 되었다고 한다.