---

title: 멀티캠퍼스 머신러닝 수업 2강
excerpt: 머신러닝 수업 4월 8일
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
last_modified_at: 2020-04-08
---



오늘은 2번째  수업입니다. 



## 1. 분류학습

- 가설을 통계적 검증으로 정의하면 모델이라고 함



**y(결과)=w(가중치) * X(입력 데이터) +b(편차)**

X : 수식으로는 하나지만 입력데이터가 벡터도 여러 개 들어갈 수 있음 

b :최종 결과값에는 영향을 미치지만 미분하면 0이 됨



<br>

**가. Machine Learning 순서**

학습단계 (Training)

데이터 수집 => 피쳐정리 => 가설정의 =>  비용 함수의 정의 => 학습  =====> 예측단계(prediction)



![머신러닝기본과정](https://i.imgur.com/fuo4Kk3.png)



<br>

## 2.분류함수의 예로 구분석 긋기

학습률을 1.0으로 했을 때, 학습률을0.5로 했을 때, 가중치의 업데이트 결과 비교하기



애벌레와 무당벌레를 구분하는 예제

| 예제 |  폭  | 길이 |   곤충   |
| :--: | :--: | :--: | :------: |
|  1   | 3.0  | 1.1  | 무당벌레 |
|  2   | 1.0  | 2.9  |  애벌레  |

<br>

y=Ax로 b를 생략을 가정함 

**가. 학습률1.0을 기준으로 가중치 업데이트**

**1) 첫번째 학습** 

w에 임의로 0.25를 넣었을 때 입력값이 폭 3.0이면  3.0*0.25=0.75가 되어 출력값이 0.75가 됨

목표값이 1.1일 때 0.35의 오차가 발생함

![1차업데이트](https://i.imgur.com/8h0FXlz.png)



오차 0.35을 통해 델타A의 값을 산출함

![2차기울기계산](https://i.imgur.com/k2zYzN2.png)



![2차기울기계산2](https://i.imgur.com/SkWjRZW.png)

새로운 A 값은 A + △A = 0.25 + 0.1167 = 0.3667

새로운 예측치는 1.1 (y = 0.3667 x 3 = 1.1)

-----

<br>

**2) 두번째 학습**

![두번째업데이트](https://i.imgur.com/fHzptCG.png)

△A = E / x = 2.5333 /1.0 = 2.5333

A  ->  A + △A = 0.3667 + 2.5333 = 2.9로 업데이트

x = 1.0일 때 이 함수는 목표 값인 2.9를 출력



------------

<br>

3) 학습률 1.0일 때 결과**

![머신러닝업데이트과정](https://i.imgur.com/K2NHxFW.png)





<br>

**나. 학습률0.5을 기준으로 가중치 업데이트**

**1) 첫번째 학습**

![학습률0.5반영](https://i.imgur.com/mLifj2G.png)

<br>

**2) 두번째 학습**

![학습률0.5일때](https://i.imgur.com/8dIXwkG.png)

**3) 학습률 0.5때 결과**

결과적으로 학습률이 1.0일때보다 더 빠르게 빨간점과 파란점을 나누는 선을 학습하는 것을 경험할 수 있습니다. 

![가중치0.5일때3](https://i.imgur.com/KggjvqL.png)





## 3.활성화 함수

![활성화함수](https://i.imgur.com/8dXdsZB.png)



**1) 계단함수** 

최근에 거의 쓰지 않음

초기 인간의 신경망에 아이디어를 받아서 도입한 함수모형



**2) 시그모이드 함수**

 결과가 0과 1사이를 s자로 서서히 연결됨

시그모이드 함수 실습

**quiz 시그모이드 함수를 파이썬으로 직접 정의하고, 몇개의 x값에 대응하는 y의 값 찾기** 



**3) 하이퍼볼릭 탄젠트Hyperbolic tangent**

0을 지나는 시그모이드와 유사한 함수



**4) ReLU함수**

- 0보다 작은 값 -> 0

- 0보다 큰값 -> 그 값 그대로 반영



**5) softmax**

- 입력 받은 값을 0~1사이의 출력 값으로 정규화
- 출력 값들의 총합은 항상 1
- 보통 3개 이상의 집단을 구분하는 신경망 모형의 출력층에 사용함

- 결과 값을 One hot encoder 의 입력으로 사용하면, 가장 큰 값만 True

<br>

**강사님 과제 ** :

1. 강사님 문제  PANDAS를 통해 SAL(임금) 데이터 구하기

2. 인터페이스, API, Libary, Framework, Platform 의 정확한 개념과 차이점 알아두기





<br>

## 3. 신경망 계층망 연산 과정(강사님 자료)

**가. 전체 구조**

- 입력층의 노드 개수는 입력 데이터 셋의 특성(Attriubte)수와 동일함

![](https://i.imgur.com/uZqPliG.png)



<br>

**나. 1개의 히든층 노드에서 발생하는 사건 **

각 노드 안에서 가중치를 입력값과 곱해서 더하고 바이어스를 더한 값을 활성화함수(시그모이드 함수)에 넣어서 최종 노드의 값을 출력함

![](https://i.imgur.com/M66DrQw.png)



<br>

**다. 은닉층의 입력값과 출력값 **

각 층을 거치면서 노드별로 출력된 값이 다음층의 입력값이 되어 신경망의 연산이 이뤄지고, 출력층의 결과을 실제 값(Target)과 비교한 뒤에 오차와의 차이를 계산하여 각각의 웨이트값을 수정하는 작업을 수행함(오차역전파)

![피드포워드과정1](https://i.imgur.com/mBSkOYH.png)

![피드포워드과정2](https://i.imgur.com/8QTWuR6.png)

![피드포워드과정3](https://i.imgur.com/P82GsJY.png)



<br>

## 4. 오차함수

**가. 평균제곱오차**

- 오차는 학습 목표 값과 실제 값 간의 차이
- 평균 제곱된 오차와 교차 엔트로피 오차 사용

![](https://i.imgur.com/NnMAxxc.png)

(실제값)t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (3번째가 정답)

1) 3번째가 정답이라고 예측

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] 
mean_suared_error(10, np.array(y), np.array(t))
0.019500000000000007



2) 8번째가 정답이라고 예측

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mean_suared_error(10, np.array(y), np.array(t))
0.11950000000000001

ex1) 정답일 경우 오차 함수의 값이 작게 나옴 (오차 적음)



 **나.교차엔트로피**



![](https://i.imgur.com/h2e0fOI.png)

평균제곱 오차보다 정교하게 오차를 산출할 수 있지만, 계산량이 복잡해서 학습시 느려질 수 있음



1) 3번째가 정답이라고 예측

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

cross_entropy_error(np.array(y), np.array(t))
0.51082545709933802



2) 8번째가 정답이라고 예측

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
cross_entropy_error(np.array(y), np.array(t))
2.3025840929945458

ex1) 정답일 경우 오차 함수 쪽 출력이 작음 (오차 적음)



**다. 가중치의 값에 비례해서 오차를 수정함**

![](https://i.imgur.com/ZCuRSNQ.png)



<br>

## 5.오차역전파



**가. 오차값 업데이트(출력값을 Target과 비교한 결과를 통해 각 층 노드의 오차값 계산)** 

![오차역전파1](https://i.imgur.com/BeHtpO6.png)

![오차역전파2](https://i.imgur.com/vyxjOd4.png)

원래 히든층의 에러를 위와 같은 식으로 계산을 하는데, 가중치 행렬의 트랜스포즈를 취해준 행렬값과 결과적으로 같은 값이 도출됩니다.(이유은 잘 이해가 안되지만...). 계산의 편의성을 위해서 W행렬의 트랜스포즈를 취한 값을 이용합니다. 

![오차역전파3](https://i.imgur.com/AUfB8CZ.png)



<br>

**나. 가중치 업데이트(경사하강법을 이용하여 각 노드의 오차가 최소가 되는 w계산)** 

우리가 알고 싶은 것은 오차가 최소가 되는 w의 값을 알아내는 것입니다.

오차가 최소가 되는 w를 알아내기 위해서 각 오차함수를 각 w로 미분을 합니다.

우리는 e를 알고 있으니 반대로 w의 값에 변화를 주면서 오차가 얼마나 변하는지를 계산하는 방식을 취합니다. 

그런데 실제 히든층에서 결과층으로 연산이 이뤄질 때는 가중치의 곱을 합한  wx+b의 시그마 연산 뿐만 아니라 활성화함수(시그모이드)도 함께 이뤄지기 때문에 아래 수식처럼  미분한값을 구하는 과정이 복잡합니다. 

최종적으로 오차함수를 w로 미분값에 학습률을 곱한 값을 기존 가중치와 더해서 새롭게 업데이트된 가중치 값이 됩니다.  

<br>

![](https://i.imgur.com/x8r0PE7.png)





<br>

## 6. 실습(MINIST 손글씨 데이터 인식)

**vs코드 패키지 설치(수업준비사항)**

VS Code jupyter notebook

Rainbow csv 

edit.csv   



톱니바퀴(setting) word warp on세팅

![](https://i.imgur.com/pvJXXqR.png)

<br>

Rainbow csv 설치하면 이렇게 변경됩니다.

![](https://i.imgur.com/62G9ojM.png)

mninst_train100_100.csv 맨 처음 개행을 수정함(강사님 파일에 오류가 있음)

불필요하게 [] 가 포함되어 있습니다.  



```python
import os
import numpy as np
import matplotlib.pyplot as plt # pip install matplotlib

print(os.getcwd())
data_file = open("../dataset/mnist_train_100.csv", "r")
data_list = data_file.readlines()
data_file.close()

all_values = data_list[0].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()
scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # 0을 만들지 않기 위해서
print(scaled_input)
```

![](https://i.imgur.com/MTpavy7.png)

Anaconda에서 직접 실행함

