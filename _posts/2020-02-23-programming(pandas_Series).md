---
title:  "python pandas 정리1_Series"
excerpt: "pandas의 각종 함수 정리"
toc: true
toc_sticky: true

categories:
  - programming
tags:
- Python
- pandas
- Series
- numpy
use_math: true
last_modified_at: 2019-12-23
---

## 1.Series함수

value_counts: NaN을 제외하고 각 값들의 빈도를 반환

```python
a= np.array([2,1,3,3,4,4,np.NaN])
a.mean()
nan

b=pd.Series(a)
b.mean()
2.8333333333333335

b.size
7

b.count()
6 #NaN은 포함되지 않음

b.shape
(7,)

b.unique()
array([ 2.,  1.,  3.,  4., nan])

b.value_counts()
4.0    2
3.0    2
1.0    1
2.0    1
dtype: int64

b[[1,3,5]] #2개 이상 전달할 때는 리스트로 전달해야 함
1    1.0
3    3.0
5    4.0
```



## 2.(Series).drop()

```py
j = pd.Series(np.arange(105, 110), ['a', 'b', 'c', 'd', 'e'])
j
a    105
b    106
c    107
d    108
e    109

j.drop('e', inplace=True)
j
a    105
b    106
c    107
d    108
dtype: int32



```

## 3. Series indexing

```python
j = pd.Series(np.arange(105, 110), ['a', 'b', 'c', 'd', 'e'])

j['a'] #문자로된 인덱스로 값을 주출
105

j[['a','b']]=[300,400]
j
a    300
b    400
c    107
d    108
e    109

j[1:3] # 문자로된 인덱싱이라도 인데스의 순서로 인덱싱을 할 수 있음
b    400
c    107
dtype: int32

j['c':'d'] #문자 인덱싱을 slicing하면 마지막 문자 index값도 포함됨
c    107
d    108
dtype: int32
```

