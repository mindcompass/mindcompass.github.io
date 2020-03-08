---
title:  "데이터분석 실전1_1(시간데이터 전처리)"
excerpt: "pandas를 통해 여러가지 시간형식을 하나의 타입으로 변경하고 분석테이블을 만드는 연습하기"
toc: true
toc_sticky: true

categories:
  - data_analysis
tags:
- datetime
- pandas
- 
- 
use_math: true
last_modified_at: 2020-03-04
---



 

데이터분석을 하는 과정에서 다양한 시간 데이터를 마주하게 됩니다. 

여러 테이블 사이에 시간 데이터의 형식이 다르면 나도모르게 멘붕에 빠지게 됩니다. 

이번에는 서로 다른 시간 데이터 형식이 있는 테이블을 하나의 테이블로 정리하는 연습을 해보려고 합니다.    

---------



## 1. 유닉스 타임 변경

**유닉스 시간(Unix time)**이란 1970년 1월 1일 00:00:00 협정 세계시 부터의 경과 시간을 초로 환산하여 정수로 나타낸 것이다. 1초에 지날때마다 정수가 1씩 증가합니다. 

### 1) 유닉스타임을 datetime형식으로 변경

만약 주어진 자료가 유닉스 타임이라고 가정하면 아래처럼 날짜 데이터로 변경할 수 있습니다. 

```python
import pandas as pd
from datetime import datetime
import numpy as np

data1=pd.DataFrame(data={
'datetime': [1583020800,1583107200,1583193600,1583280000]    
})

	datetime
0	1583020800
1	1583107200
2	1583193600
3	1583280000

data1['datetime'] = pd.to_datetime(data1['datetime'],unit='s')
# UNIX타입의 변경하는 단위가 초 단위이기 때문에 unit='s'로 설정합니다. 

	datetime
0	2020-03-01
1	2020-03-02
2	2020-03-03
3	2020-03-04

data1.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 1 columns):
datetime    4 non-null datetime64[ns]
dtypes: datetime64[ns](1)
memory usage: 160.0 bytes
    
#데이터 타입이 datetime64[ns]인 것을 확인할 수 있습니다. 
```



### 2) datetime형식을 유닉스타임으로  변경하기

```python
data2=pd.DataFrame(data={'datetime': [datetime(2020,3,1),datetime(2020,3,2),datetime(2020,3,3),datetime(2020,3,4)]    
})

data2['datetime'] = pd.DatetimeIndex(data2['datetime']).astype (np.int32 )
#datatime자료를 정수형태로 데이터를 변경하면 유닉스타임 형식으로 변경됩니다. 

data2
	datetime
0	1583020800
1	1583107200
2	1583193600
3	1583280000
```



## 2. 시간별 데이터를 날짜별 데이터로 변경하기

관련 데이터를 만들기 어려워서 데이콘에서 외부 데이터를 가져왔습니다.

[데이터 출처](https://dacon.io/competitions/official/196878/overview/)

### 1) 사용 데이터 만들기

```python
data3 = pd.read_csv('./data_set/전력사용량_데이콘/test.csv') #각자의 파일경로
# 테스트로 하는 것이여서 데이터가 적은 test데이터를 가져왔습니다. 
# 열은 각각 200개의 지역을 의미합니다. 

num_rows = data3.shape[0] # 총 행의 수가 8760입니다.
num_rows
8760

num_missing = num_rows-data3.count() #각 열별 결측치 값을 계산함
num_missing.sort_values(ascending=True).head(6) # 결측치가 없는 5개 지역만 활용하겠습니다. 
Time      0
NX1343    0
NX1440    0
NX1441    0
NX1332    0
NX1445    0
dtype: int64


data4=data3[['Time','NX1343','NX1440','NX1441','NX1332','NX1445']]
data4.head()
	Time		NX1343	NX1440	NX1441	NX1332	NX1445
0	2017.7.1 0:00	0.109	0.243	2.181	0.093	0.145
1	2017.7.1 1:00	0.143	0.299	2.121	0.093	0.144
2	2017.7.1 2:00	0.130	0.238	1.584	0.092	0.143
3	2017.7.1 3:00	0.114	0.228	2.252	0.092	0.144
4	2017.7.1 4:00	0.133	0.170	2.341	0.093	0.145


data4.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8760 entries, 0 to 8759
Data columns (total 6 columns):
Time      8760 non-null object
NX1343    8760 non-null float64
NX1440    8760 non-null float64
NX1441    8760 non-null float64
NX1332    8760 non-null float64
NX1445    8760 non-null float64
dtypes: float64(5), object(1)
# 첫컬럼이 Time인데 데이터 형이 문자로 되어있고, 나머지 전력량은 숫자로 되어 있습니다. 
```



### 2) 문자형 데이터를 시간데이터로 변경하기

data4데이터를 확인해보면 행은 2017년 7월 1일부터~2018년 6월 30일까지 1년의 데이터입니다. 

열을 보면 Time이라는 열에는 날짜별로 1시간마다 데이터가 있고, 지역은 5곳(NX1343, NX1440, NX1441, NX1332, NX1445)입니다. 

결과값에 NaN 값이 상대적으로 많이 포함된 상태의 데이터라는 것을 알 수 있습니다. 

```python
위 자료는 1시간마다의 전력사용량을 나타내는 자료이기 때문에 이를 1일자료로 변경해야 합니다. 
즉 현재 자료는 8760행(365일 * 24시간)을 365행으로 변경해야 합니다. 

data4['Time']= pd.to_datetime(data4['Time'],format='%Y-%m-%d') 
#Time열이 문자로 되어 있기 떄문에 datetime형식으로 맞추기 위해서 데이터 형태를 바꿔줍니다.

	Time			NX1343	NX1440	NX1441	NX1332	NX1445
0	2017-07-01 00:00:00	0.109	0.243	2.181	0.093	0.145
1	2017-07-01 01:00:00	0.143	0.299	2.121	0.093	0.144
2	2017-07-01 02:00:00	0.130	0.238	1.584	0.092	0.143
3	2017-07-01 03:00:00	0.114	0.228	2.252	0.092	0.144
4	2017-07-01 04:00:00	0.133	0.170	2.341	0.093	0.145

#그렇지만 우리가 원하는 일별 데이터가 아닙니다(뒤에 시간, 분, 초가 붙어 있습니다.).

data4['Time']=data4['Time'].dt.strftime('%Y-%m-%d')
data4['Time']=pd.to_datetime(data4['Time'],format='%Y-%m-%d')
# 번거롭지만 data4['Time']을 우리가 원하는 형태로 바꾸기 위해서 연도, 월, 일만 있는 문자형으로 변경합니다.
# 그 결과를 다시 datetime형식으로 변경합니다. 

	Time		NX1343	NX1440	NX1441	NX1332	NX1445
0	2017-07-01	0.109	0.243	2.181	0.093	0.145
1	2017-07-01	0.143	0.299	2.121	0.093	0.144
2	2017-07-01	0.130	0.238	1.584	0.092	0.143
3	2017-07-01	0.114	0.228	2.252	0.092	0.144
4	2017-07-01	0.133	0.170	2.341	0.093	0.145

data4.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8760 entries, 0 to 8759
Data columns (total 6 columns):
Time      8760 non-null datetime64[ns]
NX1343    8760 non-null float64
NX1440    8760 non-null float64
NX1441    8760 non-null float64
NX1332    8760 non-null float64
NX1445    8760 non-null float64
dtypes: datetime64[ns](1), float64(5)
memory usage: 410.8 KB
#그럼 Time 열의 데이터들이 우리가 원하는 datetime형태로 되어 있는 것을 확인할 수 있습니다.

```



### 3) 시간별 자료를 일별 자료로 변경하기 

```python
data5=data4.groupby('Time').sum() #각 시간대별 합을 그날 전체의 전기사용량임으로 합치겠습니다. 
data5.head()
		NX1343	NX1440	NX1441	NX1332	NX1445
Time					
2017-07-01	3.109	5.582	151.515	2.904	24.844
2017-07-02	3.148	5.634	149.008	3.304	14.055
2017-07-03	23.226	29.651	150.318	9.062	44.102
2017-07-04	23.415	23.772	150.511	2.134	48.467
2017-07-05	30.266	23.145	164.994	13.139	51.013

# 5개의 지역별로 전력사용량을 정리한 데이터가 완성되었습니다. 
```

