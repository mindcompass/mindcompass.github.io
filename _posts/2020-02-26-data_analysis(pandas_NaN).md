---
title:  "python pandas를 통해 결측치 처리하기"
excerpt: "결측치 처리하기"
toc: true
toc_sticky: true

categories:
  - data_analysis
tags:
- 결측치 처리
- pandas
- Dataframe
- NaN
use_math: true
last_modified_at: 2020-02-26
---

## 1. 결측치(누락값, NaN) 확인하기

R에서는 NA는 잘못된 값, Null은 아직 정해지지 않은 값으로, 서로 다른 의미를 가지고 있습니다. 

하지만 python에서는 NaN이라는 한가지 용어로 NaN(Na)와 Null 을 모두 표현합니다.   

```python
#결측지 확인 방법1
num_rows = DF.shape[0] # 데이터프레임의 행의 수를 저장함
num_missing = num_rows-DF.count() # brodcasting이 발생하면서 연산됨
#각 열별 결측자료의 수를 확인할 수 있음

#결측지 확인 방법2
import numpy as np
np.count_nonzero(DF.isnull()) # 기존 데이터 프레임의 데이터를 null을 기준으로 True, False를 기준으로 값을 바꾸고, True의 전체 수를 구함 
np.count_nonzero(DF['특정 컬럼명'].isnull()) #해당 열별 결측치를 확인함

#결측지 확인 방법3
DF['특정 컬럼명'].value_counts(dropna=False) 
#NaN값을 포함시켜서 컬럼별 values값들의 빈도를 계산함

이 외에도 다양한 결측치 확인 방법이 존재합니다. 
```



## 2. 결측치 처리하기(fillna,interpolate)

```python
DF.fillna() #fillna은 특정 값으로 NaN 값을 대체할 수 있음

Signature:
ebola.fillna(
    value=None,
    method=None,
    axis=None,
    inplace=False,
    limit=None,
    downcast=None,
    **kwargs,
)

DF.fillna(method='ffill') #누락값이 나타나기 전의 값으로 누락값이 변경됨
#이 방법의 경우 맨 처음 누락값이 있을 경우 참고할 자료가 없어서(이전 자료가 없어서) NaN으로 남아있음

DF.fillna(method='bfill') #누락값있고 바로 다음 값으로 누락값을 변경함
#이 방법의 경우 맨 마지막에 누락값이 있을 경우 참고할 자료가 없어서 NaN으로 남아있음

DF.interpolate() # 결측치가 있는 열에서 결측치 이전값과 다음값의 중간값을 구하여 누락값을 수정함 
# 시간에따른 누적 데이터일 때, 주로 사용함

열에 지나치게 많은 결측 데이터가 존재할 경우 dropna()명령어를 사용해서 해당 열을 삭제할 수 있음(신중하게 사용해야 함)
DF.dropna()
```



## 3. 결측치를 제외하고 연산하기

NaN이 포함된 값을 연산하게 되면 그 결과값은 NaN이 나타납니다. 

- DF의 column1이라는 컬럼명을 가진 컬럼에 NaN이 존재할 경우

**`DF.column1.sum(skipna=True)`** 

열에서 NaN이 아닌 값들만 모두 더한 값이 리턴됩니다. 



**`DF.column1.sum(skipna=False)`**

NaN이 리턴됩니다. 