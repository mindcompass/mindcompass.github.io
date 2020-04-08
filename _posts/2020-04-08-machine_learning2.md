---

title: 머신러닝 수업 2강
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



dhsmf 







오늘은 2번째  수업입니다. 



## 1. 머신러닝의 사례

- 가설을 통계적 검증으로 정의하면 모델이라고 함



y(결과, 정답)=w(가중치) * X(입력 데이터) +b(편차)**

X : 수식으로는 하나지만 입력데이터가 벡터도 여러 개 들어갈 수 있음 

b :최종 결과값에는 영향을 미치지만 미분하면 0이 됨



**가. Machine Learning 순서**

학습단계 (Training)

데이터 수집 => 피쳐정리 => 가설정의 정의=>  비용 함수의 정의 => 학습  =====> 예측단계(prediction)



![](https://i.imgur.com/fuo4Kk3.png)



**나. 분류함수의 예로 구분석 긋기**



애벌레와 무당벌레를 구분하는 예제

| 예제 |  폭  | 길이 |   곤충   |
| :--: | :--: | :--: | :------: |
|  1   | 3.0  | 1.1  | 무당벌레 |
|  2   | 1.0  | 2.9  |  애벌레  |





**다. 활성화 함수**

![활성화함수](https://i.imgur.com/8dXdsZB.png)



**1) 계단함수** 

최근에 거의 쓰지 않음



**2) 시그모이드 함수**

 결과가 0과 1사이를 s자로 서서히 연결됨



시그모이드 함수 실습

![image-20200408105411846](2020-04-08-machine_learning2.assets/image-20200408105411846.png)



**3) 하이퍼볼릭 탄젠트Hyperbolic tangent**

0을 지나는 시그모이드와 유사한 함수



**4) ReLU함수**

0보다 작은 값 -> 0

0보다 큰값 -> 그 값 그대로 반영



**5) softmax**

- 입력 받은 값을 0~1사이의 출력 값으로 정규화

- 출력 값들의 총합은 항상 1





**강사님 과제 ** :

1. 강사님 문제  PANDAS를 통해 SAL(임금) 데이터 구하기

2. 인터페이스, API, Libary, Framework, Platform 의 정확한 개념과 차이점 알아두기





## 3. 신경망 계층망 연산 과정(강사님 자료)

**가. 전체 구조**

- 입력층의 개수는 입력 데이터 셋의 특성(Attriubte)수와 동일함

![](https://i.imgur.com/uZqPliG.png)



**나. 1개의 히든층 노드에서 발생하는 사건 **

각 노드 안에서 가중치를 입력값과 곱해서 더하고 바이어스를 더한 값을 시그모이드 함수에 넣어서 최종 노드의 값을 출력함

![](https://i.imgur.com/M66DrQw.png)



다. 은닉층의 입력값과 출력값 **

각 층을 거치면서 노드별로 출력된 값이 다음층의 입력값이 되어 신경망가 이뤄지고 실제 값(Target)과 비교한 뒤에 오차와의 차이를 계산하여 각각의 웨이트값을 수정하는 작업을 수행함(오차역전파)

![](https://i.imgur.com/P82GsJY.png)



## 4. 오차함수

**가. 평균제곱오차**

- 오차는 학습 목표 값과 실제 값 간의 차이
- 평균 제곱 오차와 교차 엔트로피 오차 사용

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

ex1) 정답일 경우 오차 함수 쪽 출력이 작음 (오차 적음)



 **나.교차엔트로피**



![](https://i.imgur.com/h2e0fOI.png)

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



## 5.오차역전파



vs코드 

VS Code jupyter notebook

Rainbow csv

eidit.csv   



톱니바퀴(setting) word warp 세팅

![](https://i.imgur.com/pvJXXqR.png)





Rainbow csv 설치하면 이렇게 변경됩니다.

![](https://i.imgur.com/62G9ojM.png)

mninst_train100_100.csv 맨 처음 개행을 수정함(강사님 파일에 오류가 있음)

