---
title:  "python numpy 정리"
excerpt: "numpy의 각종 함수 정리"
toc: true
toc_sticky: true

categories:
  - programming
tags:
- Python
- numpy
- 함수
- 선형대수
use_math: true
last_modified_at: 2019-12-22
---

## 1.np.arange

np.arange(5) => array([0, 1, 2, 3, 4])

np.arange(1,11) => array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

np.arange(1,11,2) =>array([1, 3, 5, 7, 9])



## 2.np.ones/ np.zeros

```py
np.ones((3,2))
array([[1., 1.],
       [1., 1.],
       [1., 1.]])

np.zeros((3,2))
array([[0., 0.],
       [0., 0.],
       [0., 0.]])
```



## 3. np.empty/ np.full

np.empty =>형태의 n차원 nd.array를 생성하는데 초기 값으로 메모리를 할당해줌(값은 의미없음)

```python
np.full((3,2),5)
array([[5, 5],
       [5, 5],
       [5, 5]])
```



## 4. np.eye

단위행렬 생성

```
np.eye(3)
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
```

## 5.np.linspace

```
np.linspace(1,10,5)
array([ 1.  ,  3.25,  5.5 ,  7.75, 10.  ])
```

## 6.reshape(객체의 메소드함수)

```py
x=np.arange(1,16)
x.reshape(3,5)
x.reshape(3,-1)
# -1의 값은 어차피 5라고 유추가능함
```

## 7.random

```python
np.random.rand(3,2)
# 0,1사이의 분포로 랜덤한 ndarray생성
array([[0.95499599, 0.6761246 ],
       [0.42160889, 0.69337732],
       [0.47178414, 0.57370868]])

np.random.randn(3,2)
# 정규분포로 샘플림된 ndarray생성(평균0,표준편차1)
array([[-0.46274776,  0.82855333],
       [-0.78213146,  1.15499136],
       [-0.46259909,  0.31783189]])

np.random.randint(1,10,size=(3,2))
# 범위 사이에 size에 맞는 정수 생성
array([[7, 5],
       [3, 4],
       [1, 2]])

np.random.seed(100)
# 랜덤함수를 생성하기 전에 수행하면 동일한 랜덤 결과를 생성함

np.random.choice(10, size=(2,4))
# 주어진 1차원 ndarray로부터 랜덤으로 샘플링을 수행
# 정수가 주어진 경우, np.arrange(숫자)로 간주
array([[1, 8, 7, 7],
       [0, 6, 9, 6]])
array1=np.array([1,2,3,4,5,6,7,8,9,10])
np.random.choice(array1, size=(2,4),replace=False)
array([[4, 3, 2, 6],
       [5, 1, 8, 9]])
```

## 8.np.ravel

ndarray의 멤버함수로도 있고, numpy의 일반함수로도 존재함

``````python
x= np.arange(15).reshape(3,5)
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
'order'파라미터
C=row우선(디폴트)
F=column 우선

np.ravel(x, order='C')
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])

np.ravel(x ,order='F')
array([ 0,  5, 10,  1,  6, 11,  2,  7, 12,  3,  8, 13,  4,  9, 14])

#flatten과 결과값이 동일하지만, 원래 자료를 유지(view)(단 order가 c일 경우에 한함)
``````

## 9.np.flatten

```python
y=np.arange(15).reshape(3,5)
y.flatten()
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
#ravel 결과값이 동일하지만, 원래 자료를 복사본을 생성함
```

## 10.(ndadrray).ndim/ (ndadrray).shape

```python
x = np.arange(15).reshape(3, 5)
x.ndim
2
x.shape
(3, 5)
```

## 11. add/substract/multiply/divide

```python
#행과 열이 같은 형식일 경우 연산가능(브로드캐스팅 제외)
# 대부분 함수식보다 x + y 같은 수식으로 사용함
```

## 12. mean/ max/ argmax/ var/median/std (통계관련)

```python
x =np.random.rand(3,5)
array([[0.99027278, 0.31019685, 0.70196529, 0.94970636, 0.88460239],
       [0.04228368, 0.00141197, 0.66206003, 0.67630983, 0.52483855],
       [0.78959036, 0.9211178 , 0.76781088, 0.24881523, 0.76638479]])

np.mean(x) /x.mean()
0.6158244523271563

np.max(x)
0.9902727818658551

np.argmax(x) #가장 큰수의 index
0
np.var(x)
0.09660985107112016

np.median(x)
0.7019652866242324

np.std(x)
0.31082125260528787
```



## 13. sum/ cumsum/any / all/ where 함수

```python
x =np.random.rand(3,5)
array([[0.99027278, 0.31019685, 0.70196529, 0.94970636, 0.88460239],
       [0.04228368, 0.00141197, 0.66206003, 0.67630983, 0.52483855],
       [0.78959036, 0.9211178 , 0.76781088, 0.24881523, 0.76638479]])

np.sum(x)
9.237366784907344 # 요소의 전체 합

np.sum(x, axis=0)
array([1.82214682, 1.23272662, 2.1318362 , 1.87483142, 2.17582573]) # 행 중심의 합계

np.sum(x, axis=1)
array([3.83674367, 1.90690407, 3.49371905]) # 열 축으로 합계

np.cumsum(x)
array([0.99027278, 1.30046963, 2.00243492, 2.95214128, 3.83674367,
       3.87902735, 3.88043932, 4.54249935, 5.21880919, 5.74364774,
       6.53323809, 7.45435589, 8.22216677, 8.470982  , 9.23736678])

np.any(x >0.5) #x의 각 원소 중 하나 이상의 원소가 0.5이상이다
True
np.all(x >0.5) #x의 모든 원소가 0.5이상이다
False

np.where(x>0.5,x,0) #인자가 3가지(조건, 참일 때 x값, 거짓일 때 0)
array([[0.99027278, 0.        , 0.70196529, 0.94970636, 0.88460239],
       [0.        , 0.        , 0.66206003, 0.67630983, 0.52483855],
       [0.78959036, 0.9211178 , 0.76781088, 0.        , 0.76638479]])
#numpy, pandas에서는 속도를 고려해서 루프문을 가급적 사용하지 않음
```



## 14. boolean index

```python
x = np.random.randint(1,10,size=5)
array([6, 9, 4, 8, 9])

x % 2==0
array([ True, False,  True,  True, False])

even_mask = x %2==0
even_mask
array([ True, False,  True,  True, False])

x[even_mask] 또는 x[x % 2 ==0]
array([6, 4, 8])

x[(x % 2 ==0)&(x>5)]
array([6, 8])
```



## 15.linalg 서브모듈

linear algebra : 선형대수

numpy를 통해서 직접 선형대수를 사용해서 연산을 할 수 있음(실제 많이 사용하지는 않음)

```python
x= np.random.rand(3,3)
[[0.06404221 0.42490009 0.16750449]
 [0.66387841 0.01678641 0.69762346]
 [0.3594093  0.47797227 0.38890372]]
x1=np.linalg.inv(x) #x의 역행렬 x1을 생성함

x @ x1

array([[1.00000000e+00, 1.11022302e-16, 0.00000000e+00],
       [8.88178420e-16, 1.00000000e+00, 0.00000000e+00],
       [0.00000000e+00, 2.22044605e-16, 1.00000000e+00]])

행렬의 곱(@)을 사용하지 않으려면 아래처럼 np.matmul을 사용함
np.matmul(x,np.linalg.inv(x)) #x @ x와 동일한 연산

np.linalg.solve
x+y=30
2x+4y=70

A=np.array([[1,1],[2,4]])
B=np.array([30,70])
C=np.linalg.solve(A,B)
array([25.,  5.])

np.allclose(A@C,B) #요소가 모두 동일한가?
True
```

$$
\begin{pmatrix} 1 & 1 \\ 2 & 4 \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix}= \begin{pmatrix} 30 \\ 70 \end{pmatrix}
$$
