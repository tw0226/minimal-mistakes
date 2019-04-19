---
published: true
layout: single
title: Learning Spatiotemporal Features with 3D Convolutional Networks
category: [Deep learning, Action Recognition]
toc: true
tags: [Deep learning, Pose Estimation]
comments: true
toc_sticky: true
read_time: true
use_math : true

---

읽은 논문

## Learning Spatiotemporal Features with 3D Convolutional Networks

## 관련 연구

- Hand-crafted Features

  - HOG, SIFT, iDT 등을 이용한 연구
  
- Two-stream convolutional networks for action recognition in videos

  - 행동인식을 딥러닝으로 Spatial Stream과 Temporal Stream 으로 나누어 제안함

- LRCN(Long Recurrent Convolution Neural Network)

  - 공간적인 특징을 잡아내는 CNN과 시간적인 특징을 잡아내는 RNN을 합쳐서 이용한 논문

## Learning Features with 3D ConvNets

일반적으로 이미지에 대해 Spatial Feature를 뽑기 위해 CNN을 진행한다면 2D Conv와 2D Pooling 이 필요하다.
하지만 영상이라는 이미지가 l(Length) 개가 연속적으로 이어진 구조에서는 시간적인 특징이 들어가게 된다.
3D Convolution은 공간적인 특징을 포함해 '시간적인 특징'까지도 뽑아내려고 시도해보았다.
동영상과 이미지에 대해서 차이를 적자면 아래와 같다.

- Image : Channel x Height x Width
  - 2D Convolution
  - 2D Pooling

- Movie : Channel x Length x Height x Width
  - 3D Convolution
  - 3D Pooling

그리고 어떻게 하면 좋을지 논문에서는 실험을 진행하였다.

실험 조건은 다음과 같다.

- 데이터셋 : UCF-101, Sports-1M
- 5 Layer Network
  - ![Imgur](https://i.imgur.com/Hr0kydM.png)
- Input resize
  - 3 x 16 x 128 x 171 -> 3 x 16 x 112 x 112

### Exploring kernel temporal depth

첫번째로는 depth 에 대한 실험을 진행하였는데, 여기서 depth는 시간(Frame)이다.

![Imgur](https://i.imgur.com/W3nxQc2.png)

일단, depth를 1부터 7까지 홀수 단위로 실험도 진행하였는데 결과는 좌측 그림과 같이 3이 제일 높게 나왔다고 한다.

다음으로 이미지에서 2D Convolution을 진행할 때 Feature Map(Convolution) 개수를 점점 늘리듯이,

마찬가지로 3D Convolution을 진행하는데 시간축에 대해 변경해보았다.

depth를 3-3-5-5-7, 7-5-5-3-3, 그리고 3-3-3-3-3 으로 이렇게 3가지에 대해 진행하였는데,

결과적으로 depth가 3으로 일정한 것이 성능이 더 좋게 나왔다.

자세히 보기 어렵지만 아래 사진을 보면 첫번째로 사람을 찾으려고 하고, 그 다음으로는 움직임을 찾는다고 한다.

![Imgur](https://i.imgur.com/1Vd0a9G.png)

### SpatioTemporal feature learning

3D Convolution을 다른 논문들에서 나온 성능들과 비교를 진행하였는데, 그 결과는 다음과 같다.

![Imgur](https://i.imgur.com/SFMMro9.png)

결과적으로 성능은 Convolution Pooling보다 성능이 낮은데, 이에 대해 저자는
 Convolution Pooling의 경우 120Frame 크기의 정보를 입력으로 사용하고, Network Aggregation을 했기 때문이라고 한다.
(본 논문은 16Frame을 사용한다)

## Action Recognition

논문에서는 3D Convolution을 통해 Action Recognition이 가능한지 UCF-101 데이터를 가지고 실험해았는데, 그 결과는 아래와 같다.

![Imgur](https://i.imgur.com/JRT8hmg.png)

위의 표에 따르면 **Optical Flow를 이용하는 two stream network, RNN 계열을 사용한 LRCN보다 성능이 좋다.**

### Compact Feature

논문에서 3D Convolution의 밀도를 평가하기 위해 PCA를 사용했고, iDT와 비교를 진행하였다.

아래는 그에 대한 결과로 저차원으로 PCA를 시켰을 때 3D Convolution의 성능이 더 좋았고, 3D Convolution이

일반화 기능을 보여주는 Feature가 있는지 체크하고자 t-SNE를 사용해 2차원으로 투영시켰다.

![Imgur](https://i.imgur.com/DGYtSVP.png) ![Imgur](https://i.imgur.com/yyBtCsD.png)

위 사진은 3D convolution이 얼마나 feature embedding이 잘 되었는지 보여준다.

## Action Similarity Labeling

아래는 ASLAN Dataset에 대해서 진행해본 결과로 아래 다음과 같다.

![Imgur](https://i.imgur.com/yX9IG3j.png)
![Imgur](https://i.imgur.com/PsFiyTb.png)

## Scene and Object Recognition

마찬가지로 YUPENN, Maryland Dataset에 대해서도 진행해본 결과로 아래 사진과 같다.

![Imgur](https://i.imgur.com/WkolacM.png)

## Runtime Analysis

3D Convolution 과 iDT, Temporal stream Network에 대해 실행시간을 비교해보는데, 결과는 아래 사진과 같다.

![Imgur](https://i.imgur.com/Jd72mQu.png)

이 부분에 대해선 GPU 사용 여부에 따라 여지가 많겠지만, 결과적으로는 C3D convolution이 제일 성능이 빠르다고 한다.

## Conclusion

이 논문에서는 3D Convolution을 이용해 spatio-temporal feature를 학습할 수 있는지 다루고자 하였다.

3D Convolution의 시간 길이에 따르는 결과를 확인할 수 있었고,

시공간 정보를 동시에 얻을 수 있게 되어 2D convolution에 비해 성능이 뛰어나다는 것을 확인할 수 있었다.
