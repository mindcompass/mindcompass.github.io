---
title:  "데이터분석 실전1_2(통합 분석데이터 정리 )"
excerpt: "2개 이상의 데이터프레임을 결합하여 분석하기 직전의 데이터셋을 만들기"
toc: true
toc_sticky: true

categories:
  - data_analysis
tags:
- merge
- pandas
- 
- 
use_math: true
last_modified_at: 2020-03-05
---



 

지난 번에 만들었던 인천의 5개 지역에 대한 일별 데이터를 기온 데이터와 결합해서 간단한 분석을 하려고 합니다. 이번에는 기상청 데이터를 가져와서 이전에 작업한 데이터프레임과 Merge하는 작업을 진행해보겠습니다. 



## 1. 기상청 데이터 가져와서 정리하기

기상청 기상자료개방포털에 들어가시면 이전 작업으로 진행했던 날짜에 맞는 평균기온 데이터를 다운받을 수 있습니다. 

[기상청 데이터 다운로드 페이지](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36)

```python
weather = pd.read_csv('./data_set/전력사용량_데이콘/weather.csv',encoding='CP949')
#encoding 에러가 발생해서 encoding='CP949'을 추가했습니다. 
weather.head()
	지점	지점명	일시	평균기온(°C)
0	112	인천	2017-07-01	24.1
1	112	인천	2017-07-02	23.6
2	112	인천	2017-07-03	23.2
3	112	인천	2017-07-04	24.8
4	112	인천	2017-07-05	26.0

weather.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 365 entries, 0 to 364
Data columns (total 4 columns):
지점          365 non-null int64
지점명         365 non-null object
일시          365 non-null object
평균기온(°C)    365 non-null float64
dtypes: float64(1), int64(1), object(2)
memory usage: 11.5+ KB
    
일시가 문자형태이긴 하지만 우리가 원하는 형태의 데이터로 정리되어 있습니다. 

weather_modi=weather[['일시','평균기온(°C)']]

weather_modi.columns=['time','temp_aver']

weather_modi['time']= pd.to_datetime(weather_modi['time'],format='%Y-%m-%d')

weather_modi.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 365 entries, 0 to 364
Data columns (total 2 columns):
time         365 non-null datetime64[ns]
temp_aver    365 non-null float64
dtypes: datetime64[ns](1), float64(1)
memory usage: 5.8 KB
    

```

## 2. 기존 전력사용 데이터와 Merge하기

```python
data5.reset_index(inplace=True) #이전 data5데이터가 groupby로 작업하면서 Time변수가 index로 변경되어 다시 칼럼으로 변경해줍니다.
#사실 이 작업이 없더고 Merge에는 문제가 없습니다.

data5
	Time		NX1343	NX1440	NX1441	NX1332	NX1445
0	2017-07-01	3.109	5.582	151.515	2.904	24.844
1	2017-07-02	3.148	5.634	149.008	3.304	14.055
2	2017-07-03	23.226	29.651	150.318	9.062	44.102
3	2017-07-04	23.415	23.772	150.511	2.134	48.467
4	2017-07-05	30.266	23.145	164.994	13.139	51.013

data6=pd.merge(data5,weather_modi,left_on='Time',right_on='time')

data6
	Time		NX1343	NX1440	NX1441	NX1332	NX1445	time	temp_aver
0	2017-07-01	3.109	5.582	151.515	2.904	24.844	2017-07-01	24.1
1	2017-07-02	3.148	5.634	149.008	3.304	14.055	2017-07-02	23.6
2	2017-07-03	23.226	29.651	150.318	9.062	44.102	2017-07-03	23.2
3	2017-07-04	23.415	23.772	150.511	2.134	48.467	2017-07-04	24.8
4	2017-07-05	30.266	23.145	164.994	13.139	51.013	2017-07-05	26.0
#time 열이 중복되기 때문에 삭제해줍니다.

data6.drop('time', axis=1,inplace=True)

data6.head()
	Time		NX1343	NX1440	NX1441	NX1332	NX1445	temp_aver
0	2017-07-01	3.109	5.582	151.515	2.904	24.844	24.1
1	2017-07-02	3.148	5.634	149.008	3.304	14.055	23.6
2	2017-07-03	23.226	29.651	150.318	9.062	44.102	23.2
3	2017-07-04	23.415	23.772	150.511	2.134	48.467	24.8
4	2017-07-05	30.266	23.145	164.994	13.139	51.013	26.0

```

