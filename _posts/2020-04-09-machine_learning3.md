---

title: 머신러닝 수업 3강
excerpt: 머신러닝 수업 4월 9일 멀티캠퍼스 강의
toc: true
toc_sticky: true

categories:
  - multicampus_class

tags:
- machine learning
- 
- 
- 
use_math: true
last_modified_at: 2020-04-09
---



오늘은 3번째  수업입니다. 



## 1.데이터 분석 실습(MNIST, WEATHER데이터)

**가. MNIST 데이터 분석**

- 보통 학습률을 0.1~0.3정도를 사용함

Mnist데이터에 대한 파이썬으로 직접 날 코드로 짠 코드에 대해서 러닝레이트를 조절해서 정확도를 계산함

- 1 epoch 일때 : lr이 0.3일 때 94.7 /  lr이 0.1일때 0.9506 /  lr이 0.2 일때 0.9496 (이 수치는 사람마다 달라질 수 있음)

- 5 epoch 일때 :  lr이 0.3일 때 0.9521 



**quiz :  위 내용을 행렬의 수식으로 표현하기**

![](https://i.imgur.com/74up5nU.png)

w(3x2행렬) input(3x1행렬) 인데, 연산을 위해서 w(2x3 행렬) input(3x1행렬) 로 변경함



**나. WEATHER 데이터 분석**

사용 데이터 : temp10years.csv, winequality-white.csv 

앞의 





## 2. CNN

**가. CNN을 사용하는 이유? **

기본적으로 신경망학습을 하게 되면 이미지 위치, 정확한 색깔 등을 학습하게 됨

이미지 위치가 달라지거나 왜곡된 경우에는 올바르게 작동하지 못함

학습 내용과 정확한 크기, 색깔이 적용되야 예측할 수 있음 -> 실사용에 제한적임

학습을 할 경우 **전반적인 특징(패턴)**을 학습하기 위해서 filter을 사용하여 학습함-> CNN방식 

합성곱 신경망 => 필터를 사용해서 이미지의 특징을 추출

![](https://i.imgur.com/UQHKrQm.png)

**나. CNN 방식의 과정**

![](https://i.imgur.com/7v5BRez.png)

Pooling은 특징을 뽑아 내는 과정 -> 데이터가 너무 많아서 특징에 집중해서 데이터를 축약합니다.



**다. CNN 필터의 원리** 

이미지 분석에서 kernal이라고 하면 os의  kernal이 아니라 이미지 추출하는 요소, 일종의 필터

아래를 보면 3x3의 필터(커널)을 통해  1x1형태로 변경되는 것으로 볼 수 있습니다.  

이미지 합성을 위해 필터를 통한 합성곱의 값을 이용하게 됩니다. 



[cnn의 원리인 필터의 원리를 이해하기 쉬운 사이트](https://setosa.io/ev/image-kernels/)

![](https://i.imgur.com/8fBdCJN.png)



![](https://i.imgur.com/GYqRkNt.png)



**다. Pooling의 원리** 

필터 2x2로 pooling을 하게 되면 원래 이미지의 25%로 변경됨

특징을 추출하기 위해서 필터를 사용함



![](https://i.imgur.com/AfcZ0hW.png)

**Quiz 연습문제 핸즈온 머신러닝 p152 연습문제 풀기( 03_classification.ipynb)**



[이미지 CNN학습 과정을 볼 수 있는 사이트](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html)

아래 자동차에 대한 학습과정은 필터의 종류가 16개로 CNN 작업을 수행하고 있습니다. 

![](https://i.imgur.com/zfmwhop.png)

왜 풀링을 하는가?

Why convolution followed by pooling?
- The input image shrinks

- Since filters stay the same size, they find increasingly large patterns
  (relative to the image)

- This is why CNNs learn hierarchical features

  

![](https://i.imgur.com/cF6EPyi.png)



![](https://i.imgur.com/oe6oLHH.png)

대각선 필터일 때 풀링한 값이 가장 크다. -> 이미지의 대각선적인 특징적 요소가 강함



**라. 색상이 포함된 CNN**

이미지 각 색상에 포함된 3개의 이미지 층과 각 3개의 필터가 연산된 뒤에 2차원 평면으로 값이 구성됨

![](https://i.imgur.com/wK6nErJ.png)

![](https://i.imgur.com/aLjR9s4.png)

민약 컬러 필터의 갯수가 여러개면 여러개의  2차원 평면에 대한 필터수 만큼의 결과가 형성됨



