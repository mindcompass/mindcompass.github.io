---

title: 머신러닝 수업 1강
excerpt: 머신러닝 수업 4월 6일
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
last_modified_at: 2020-04-06
---

<br>



오늘부터 새로운 강사님과 수업을 진행합니다. 이동원 강사님입니다. 

경영학 전공을 해서 it로 취업하셨고, 지금까지 JP모건,삼성SDS 경력직으로 근무하셨다고 합니다. 





<br>

## 1. 머신러닝의 사례

**가. 훼밍웨이 소설의 학습 예제(텍스트데이터)**

![](https://i.imgur.com/8FCqIqq.png)



이전에 여러 소설에 대한 데이터를 입력하여 모델을 만든 다음, "Robert Cohn WAS once middleweight boxi까지 입력하면 그 다음에 어떤 말이 나올지에 대해서 신경망 모델에서 현재 맥락에 맞게 형성해줍니다. 가령 boxiing player~~처럼 말이죠. 마치 우리가 포탈에 입력하려고 글자를 입력하면 모두 다 입력하지 않아도 자동으로 나머지 단어나 문장을 완성해주는 기능과 유사합니다. 



Andrej Karpathy라는 사람에 의해서 헤밍웨이의 소설 The Sun Also Rises을 RNN으로 학습했다고 합니다. 

영어 소문자, 대문자, 마침표 및 특수문자를 포함한 84개의 고유한 문자였고, 362,239 글자로 구성되어 있었다고 합니다. 100번-> 1000번 -> 1000번 x n번 학습이 이뤄지면서 글의 문장이 실제 그럴 듯하게 변경되는 것을 볼 수 있습니다. 



![헤밍웨이소설](https://i.imgur.com/cHWUMaP.png)

![Imgur](https://i.imgur.com/boctQxd.png)

점차 원작 소설과 유사해짐(단, 일정 수준에 도달하면 더 이상 학습이 이뤄지지 않음)





<br>

**나. Super Mario 게임 배경을 머신러닝을 통해 만드는 사례(이미지 데이터)** 

기계가 슈퍼마리오의 배경 32개의 배경을 학습하여 새로운 배경을 만들어 냄 

32종류, 80%가 아웃도워 배경

훈련을 위한 데이터 세트 -> 1985년 슈퍼 마리오 브라더스

문서 데이터보다 분석이 어려움 -> 배경의 픽셀을 n차원 배열(행렬)으로 만들어 처리함

![훈련데이터 세트](https://i.imgur.com/RMd636g.png)



보통 영상은 3차원 

그림을 "#","="을 대신넣어서 맵을 형성함



![이미지 데이터에 대한 머신러닝 사례](https://i.imgur.com/bmH65jl.png)

파일을 90도 회전시켜서 작업을 분석함

학습시켜서 이미지를 행성함

최종적으로 파일을 원래대로 90도 변경함

![최종 결과물](https://i.imgur.com/zlzCg4I.png)





<br>

**다. 챗봇 서비스**

오늘 강남가는 셔틀버스 서비스 

과거 버전 챗봇(1단계) : 강남, 셔틀, 통근 -> 7시,7시 30분에 차가 있어요. 단순 사실을 전달함

2단계 : 머신러닝 기법을 도입해서 현재 교통량을 파악해서 -> 교통량을 고려해서 7시 차를 타는게 좋다는 내용을 전달





<br>

**라.백설공주와 Machine Learning(책)을 통한 머신러닝 소개**

여왕이 가장 아름다운 사람을 묻자 거울(기계)가 대답하는 형태로 진행됨

세상에서 가장 아름다운 사람을 찾기 위해서는 아름다움과 관련된 특징과 그 특징이 아름다움에 미치는 가중치가 필요함

아름다운 사람을 찾기 위해서 특징이 필요함(예, 머리길이, 나이)

그 중에 무엇이 더 중요한지를 알아야 함(가중치)

![백설공주 거울과 인공지능 이야기 : 볼츠만 머신러닝에서 딥러닝까지 ...](https://t1.daumcdn.net/cfile/tistory/999D27485BC9518532)

[출처] 백설공주 거울과 인공지능 이야기





<br>

## 2. 머신러닝 기본개념

**가. 지도학습과 비지도학습**

비지도 학습 : 헤밍웨이와 슈퍼마리오의 사례처럼 정답이 없는 경우

지도학습 : 정답이 있는 경우

지도학습의 경우, 입력 값과 정답을 아는 상태에서 그 함수를 구하는 것

![머신러닝개념](https://i.imgur.com/y8jci4a.png)

예제1) 3 4 2 =10이 주어졌을 때, * 하고 다음에 -를 해야 답 10이 나온다는 function을 구하는 것 

예제2) km -> 연산 -> mile

선형(linear) -> mile = km x c(c는 상수)





<br>

**나.머신러닝 내부동작 원리**

![km와 mile변환 사례](https://i.imgur.com/oN0Fdeh.png)

1단계 : 100km -> 임으로 c의 값을 0.5으로 넣음 -> 50mile (오차발생, 실제값 62.136이므로 12.137 오차 발생)

2단계 : c의 값을 0.60을 넣음 -> 60mile(근사해졌지만 오차발생, 2.137 오차발생)

3단계 : c의 값을 0.70을 넣음 -> 70mile(오버슈팅, -7.863 오차 발생)

4단계 : c의 값을 0.61을 넣음 -> 62mile(1.137 오차발생)

 

계속할 수도 있지만, 오버피팅이 발생하기  때문에 적당한 단계에서 멈추게 됨





<br>

**다. 대표적 알고리즘**

- 분류

- 회귀
- 군집화
- 차원축소
- 추천
- 이상탐지
- 강화학습





<br>

**라.Scikit-learn 분석 알고리즘 가이드**

어떤 상황에서 어떤 머신러닝을 수행해야 하는지에 대한 명확한 정답은 없지만, 대략적인 가이드는 존재합니다. 

아래는 scikit-learn에서 제공하는 가이드입니다. 

![scikit-learn분석알고리즘가이드](https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F9950844F5C986AD827)





<br>

**quiz it상식**

데브옵스는 소프트웨어의 개발과 운영의 합성어로서, 소프트웨어 개발자와 정보기술 전문가 간의 소통, 협업 및 통합을 강조하는 개발 환경이나 문화를 말한다. 



**IaaS(Infrastructure as a Service)**  서비스를 운영하기 위한 인프라를 제공함(컴퓨터만 빌겨줌)



**Paas(Platform as a Service)**  플랫폼까지 제공(예, 컴퓨터에 윈도우를 깔아줌) 

서비스를 개발 할 수 있는 안정적인 환경(Platform)과 그 환경을 이용하는 응용 프로그램을 개발 할 수 있는 API까지 제공하는 형태



**SaaS(Software as a Service)** 소프트웨어까지 제공(예, 컴퓨터+ 윈도우+ 각종 소프트웨어까지 제공함)

Cloud환경에서 동작하는 응용프로그램을 서비스 형태로 제공하는 것

FaaS  함수까지 제공함(컴퓨터+ 윈도우+ 각종 소프트웨어 + 각종 편의 기능까지)

![](https://img1.daumcdn.net/thumb/R720x0.q80/?scode=mtistory2&fname=http%3A%2F%2Fcfile22.uf.tistory.com%2Fimage%2F9997B33E5D87252D1E1A5B)

[출처] https://shlee0882.tistory.com/259





<br>

## 3.머신러닝 실습하기

[강사님깃허브 주소](https://github.com/joneconsulting/ml)



**가. _AND 알고리즘을 Linear SVC로 수행**

Day1-> and_xor파일 

**시도1 AND알고리즘을 Linear SVC로 수행**

```python
# 라이브러리 읽어 들이기 --- (*1)
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 학습 전용 데이터와 결과 준비하기 --- (*2)
# X , Y
learn_data = [[0,0], [1,0], [0,1], [1,1]]
# X and Y
learn_label = [0, 0, 0, 1]

# 알고리즘 지정하기(LinierSVC) --- (*3)
clf = LinearSVC() #fitting이 끝나면 clf객체에 웨이트값을 포함한 다양한 결과값이 포함됨

# 학습전용 데이터와 결과 학습하기  --- (*4)
clf.fit(learn_data, learn_label)

# 테스트 데이터로 예측하기 --- (*5)
test_data = [[0,0], [1,0], [0,1], [1,1]]
test_label = clf.predict(test_data)

# 예측 결과 평가하기 --- (*6)
print(test_data , "의 예측 결과: " ,  test_label)
print("정답률 = " , accuracy_score([0, 0, 0, 1], test_label))
```

```
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과:  [0 0 0 1]
정답률 =  1.0
```





<br>

**시도2_XOR 알고리즘을 Linear SVC로 수행**

```python

# 시도1 AND알고리즘을 Linear SVC로 수행
# 학습 전용 데이터와 결과 준비하기
# X , Y
learn_data = [[0,0], [1,0], [0,1], [1,1]]
# X xor Y
learn_label = [0, 1, 1, 0]  #(*) xor 전용 레이블로 변경

# 알고리즘 지정하기(LinierSVC)
clf = LinearSVC()

# 학습전용 데이터와 결과 학습하기 
clf.fit(learn_data, learn_label)

# 테스트 데이터로 예측하기
test_data = [[0,0], [1,0], [0,1], [1,1]]
test_label = clf.predict(test_data)

# 테스트 결과 평가하기
print(test_data , "의 예측 결과: " ,  test_label)
print("정답률 = " , accuracy_score([0, 1, 1, 0], test_label))  #(*) xor 전용 레이블로 변경
```

```
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과:  [0 0 0 0]
정답률 =  0.5
```





<br>

**시도3 XOR  알고리즘을 KNeighborsClassifier(군집분석)로 수행**

```python
from sklearn.neighbors import KNeighborsClassifier

# 알고리즘 지정하기(KNeighborsClassifier) --- (*2)
clf = KNeighborsClassifier(n_neighbors = 1)

# 학습전용 데이터와 결과 학습하기 
clf.fit(learn_data, learn_label)

# 테스트 데이터로 예측하기
test_data = [[0,0], [1,0], [0,1], [1,1]]
test_label = clf.predict(test_data)

# 테스트 결과 평가하기
print(test_data , "의 예측 결과: " ,  test_label)
print("정답률 = " , accuracy_score([0, 1, 1, 0], test_label))  #(*) xor 전용 레이블로 변경
```

```
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과:  [0 1 1 0]
정답률 =  1.0
```

**결론은 결과가 좋지 않으면 분석 방법을 바꾸거(옵션1)나 데이터를 변경(옵션2)해야 함**





<br>

**나. IRIS데이터 분석**

[IRIS데이터](https://archive.ics.uci.edu/ml/datasets/Iris)

```python
#jupyter notebook 실행

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 붓꽃 데이터 읽어 들이기 --- (*1)
iris_data = pd.read_csv("iris.csv", encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기 --- (*2)
y = iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

# 학습 전용과 테스트 전용 분리하기 --- (*3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

# 학습하기 --- (*4)
clf = SVC() #clf에는 가설과 모델이 들어있는 객체임 
clf.fit(x_train, y_train)

# 평가하기 --- (*5)
y_pred = clf.predict(x_test)
print("정답률 = " , accuracy_score(y_test, y_pred))
```

```python
정답률 =  0.9333333333333333
```





<br>

**다. wine 데이터분석**

```python
import pandas as pd
df = pd.read_csv("../dataset/winequality-white.csv", sep=";", encoding="utf-8")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

wine = pd.read_csv("../dataset/winequality-white.csv", sep=";", encoding="utf-8")

y = wine["quality"]
x = wine.drop("quality", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))

```

```python
정답률= 0.6724489795918367
```

![](https://i.imgur.com/IGUxbng.png)

위와 같은 워닝은 label데이터가 명확히 구분되지 않을 때 출력된다고 합니다. 

```python
import pandas as pd
import matplotlib.pyplot as plt

wine = pd.read_csv("../dataset/winequality-white.csv", sep=";", encoding="utf-8")

count_data = wine.groupby('quality')["quality"].count()
print(count_data)

count_data.plot()
plt.savefig("wine-count-plt.png")
plt.show()
```



```
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
Name: quality, dtype: int64
```

![](https://i.imgur.com/sdadcKl.png)





<br>

**quality에 5,6,7에 너무 밀집되어 있고, 전체 가지의 수도 너무 많음-> 상중하로 단순화 시켜보자!**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

wine = pd.read_csv("../dataset/winequality-white.csv", sep=";", encoding="utf-8")

y = wine["quality"]
x = wine.drop("quality", axis=1)

# modify y label
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))


```

```python
 precision    recall  f1-score   support

           0       1.00      0.23      0.38        30
           1       0.96      1.00      0.98       919
           2       0.83      0.48      0.61        31

    accuracy                           0.96       980
   macro avg       0.93      0.57      0.66       980
weighted avg       0.96      0.96      0.95       980

정답률= 0.9571428571428572
```

예측결과 값을 3가지(0,1,2,)로 구분했더니 예측 정확성이 높아짐

5,6,7 사이에 값들이 특성이 비슷한데, 서로 다르게 구분할 경우, 오답으로 인식하기 때문에 정확도에서 많은 손실이 발생함





<br>

## 4. Perceptron

다수의 신호를 입력으로 받아 하나의 신호를 출력

Layer fuction : 입력 신호가 뉴런에 보내질 때 각각 고유한 고유치(가중치)가 곱해짐

Activation fuction : 뉴런에서 보내온 신호의 총합이 정해진 한계를 넘어설 때만 1 출력(활성화함수) -> 출력화

![](https://i.imgur.com/3kuRmxc.png)



**가. Perceptron의 문제**

**1) 단측 퍼셉트론으로는 XOR문제를 구현할 수 없음**

멀티 퍼셉트론으로 문제 해결 OR, NAND 퍼셉트론 결과를 다시 AND로 결합하여 XOR 문제를 해결함



**2) 오차역전파**

최종 output 결과를 반영하여 각 hidden layer의 가중치을 수정하게 되하게 됨(backpropagation)

히든레이어가 많으면 원래 output에서 비롯되었던 weight 수정이 거의 이뤄지지 않아서 학습이 되지 않는 문제가 발생함 -> 적절한 활성화 함수를 통해 문제를 해결(ReLU)



