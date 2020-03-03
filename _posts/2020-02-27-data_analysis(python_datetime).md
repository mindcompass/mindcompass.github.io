---
title:  "판다스를 통한 시계열데이터 다루기"
excerpt: "판다스 datetime 오브젝트를 통한 시계열 데이터 전처리하기"
toc: true
toc_sticky: true

categories:
  - data_analysis
tags:
- datetime
- 시계열데이터
- pandas
- 
use_math: true
last_modified_at: 2020-02-28
---

datetime 라이브러리는 날짜를 처리하는 time 오브젝트, 시간을 처리하는 time 오브젝트, 날짜와 시간을 모두 처리하는 datetime오브젝트 등을 포함하고 있습니다. 



## 1. now(), today(),datetime(),timedelta

```python
from datetime import datetime

datetime.now() #두 메소드 모두 현재 시간을 표현해줍니다.
datetime.today()
2020-02-27 09:58:11.262434
        
t1 = datetime(2021, 2, 27) #입력 시간을 바탕으로 datetime 오브젝트를 생성합니다.
t2 = datetime(2021, 2, 21, 11, 11, 11)        
        
  
diff1 = t1 - t2 #datetime 오브젝트는 시간계산을 할 수 있습니다.

print(diff1)
print(type(diff1))
5 days, 12:48:49
<class 'datetime.timedelta'> # datetime.timedelta라는 독특한 객체를 생성합니다.
```



## 2. to_datetime() (str데이터를 datetime object로 변환)

```python
time_date = pd.DataFrame( {
    'time' : ['2020-02-01','2020-02-02','2020-02-03','2020-02-04','2020-02-05','2020-02-06']
})

time_date
	time
0	2020-02-01
1	2020-02-02
2	2020-02-03
3	2020-02-04
4	2020-02-05
5	2020-02-06

# 문자열이 time 칼럼을 datetime 객체로 변환한 'time_datetime'컬럼을 생성합니다.
time_date['time_datetime']= pd.to_datetime(time_date['time'])

	time	time_datetime
0	2020-02-01	2020-02-01
1	2020-02-02	2020-02-02
2	2020-02-03	2020-02-03
3	2020-02-04	2020-02-04
4	2020-02-05	2020-02-05
5	2020-02-06	2020-02-06

# 두 칼럼은 동일하게 생긴거 같아도 데이터 형태가 다릅니다. 
time_date.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6 entries, 0 to 5
Data columns (total 2 columns):
time             6 non-null object
time_datetime    6 non-null datetime64[ns]
dtypes: datetime64[ns](1), object(1)
memory usage: 224.0+ bytes
    
# 만약 문자열의 형태가 익숙한 형태가 아니라면, to_datetime의 format속성을 사용하면 됩니다. 
time_date2 = pd.DataFrame( {
    'time2' : ['01/02/20','20/02/02','20/02/03']
})

time_date2
	time2
0	01/02/20
1	20/02:02
2	20-02-03

time_date2['date_1'] = pd.to_datetime(time_date2['time2'], format='%d/%m/%y')
time_date2['date_2'] = pd.to_datetime(time_date2['time2'], format='%m/%d/%y')
time_date2['date_3'] = pd.to_datetime(time_date2['time2'], format='%y/%m/%d')

time_date2 #format의 형식에 따라 데이터가 다르게 입력된 것을 확인하실 수 있습니다. 
	time2		date_1		date_2		date_3
0	01/02/05	2005-02-01	2005-01-02	2001-02-05
1	05/02/02	2002-02-05	2002-05-02	2005-02-02
2	05/02/03	2003-02-05	2003-05-02	2005-02-03 

```



## 3. read_csv에서 파일을 읽을 때. datetime object로 변환

```python
pd.read.csv('[파일 경로]',parse_dates=['바꿀 컬럼명'])
ebola = ebola=pd.read_csv('https://raw.githubusercontent.com/mindcompass/data_sets/master/country_timeseries.csv' ,parse_dates=['Date'])

ebola.info(0)
Date                   122 non-null datetime64[ns] 
# 'Date을 읽을 때 datetime객체로 읽은 것을 알 수 있습니다.'
```



## 4. 날짜 정보 추출하기

```python
# 시계열 데이터를 구분해서 추출하고 싶으면 strftime()를 사용하면 됩니다. 
now = datetime.now()
now
2020-02-27 11:03:21.907462
        
nowDate = now.strftime('%Y-%m-%d')
nowDate
2018-08-27

nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
nowDatetime
2018-08-27 17:10:51
        
# serises에서 날짜를 가지고 오려면 year, month, day속성을 사용합니다.
date_series = pd.Series(['2020-02-01', '2020-02-02', '2020-02-03'])
date1 = pd.to_datetime(date_series) 
date1
0   2020-02-01
1   2020-02-02
2   2020-02-03
dtype: datetime64[ns]
    
date1[0].year
2020

date1[1].month
2

date1[2].day
3

# 위 방법은 데이터프레임에서도 사용 가능합니다.
ebola=pd.read_csv('https://raw.githubusercontent.com/mindcompass/data_sets/master/country_timeseries.csv' ,parse_dates=['Date'])
ebola['Date'][0].year 
2015 #0번째 행의 Date 자료의 year을 추출함

# 하지만 위 방법은 인덱스로 접근하기 때문에 전체 열에 적용하려면 for문 같은 반복문을 사용해야 합니다.
# 동시에 열에 모두 적용시키기 위해서는 dt접근자로 작업합니다. 

ebola['year'] = ebola['Date'].dt.year
ebola['month'] = ebola['Date'].dt.month
ebola['day'] = ebola['Date'].dt.day

ebola[['Date', 'year','month','day']].head()
	Date	year	month	day
0	2015-01-05	2015	1	5
1	2015-01-04	2015	1	4
2	2015-01-03	2015	1	3
3	2015-01-02	2015	1	2
4	2014-12-31	2014	12	31
```


