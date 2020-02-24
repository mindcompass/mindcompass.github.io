---
title:  "python pandas 정리3_Dataframe2"
excerpt: "pandas의 Dataframe Groupby pivot,stack 등
toc: true
toc_sticky: true

categories:
  - programming
tags:
- Python
- pandas
- Dataframe
- numpy
use_math: true
last_modified_at: 2020-02-24
---

## 1.grouping 함수(NaN은 제외하고 연산), groupby함수

grouping 함수(NaN은 제외하고 연산)

count -데이터의 개수

sum - 데이터의 합

mean,std,var -평균, 표준편차, 분산

min, max -최소, 최대값



```python
class_group =df.groupby('Pclass')
class_group #리턴 결과는 데이터프레임 그룹바이 객체
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x0000015863B43C18>

class_group.groups # 딕셔너리 형태로 key에는 컬럼의 값이 들어가고 values에는 인덱스가 포함된 리스트가 출력됨
{1: Int64Index([  1,   3,   6,  11,  23,  27,  30,  31,  34,  35,
             ...
             853, 856, 857, 862, 867, 871, 872, 879, 887, 889],
            dtype='int64', length=216),
 2: Int64Index([  9,  15,  17,  20,  21,  33,  41,  43,  53,  56,
             ...
             848, 854, 861, 864, 865, 866, 874, 880, 883, 886],
            dtype='int64', length=184),
 3: Int64Index([  0,   2,   4,   5,   7,   8,  10,  12,  13,  14,
             ...
             875, 876, 877, 878, 881, 882, 884, 885, 888, 890],
            dtype='int64', length=491)}

class_group.count()
PassengerId	Survived	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
Pclass											
1	216	216	216	216	186	216	216	216	216	176	214
2	184	184	184	184	173	184	184	184	184	16	184
3	491	491	491	491	355	491	491	491	491	12	491

class_group.mean()['Age']
Pclass
1    38.233441
2    29.877630
3    25.140620
Name: Age, dtype: float64
        
class_group.mean()['Survived']
Pclass
1    0.629630
2    0.472826
3    0.242363
Name: Survived, dtype: float64
        
class_group.min() # 각 컬럼별 최소값 출력
class_group.max() # 각 컬럼별 최대값 출력

#성별에 따른 생존율 구하기
class_sex.mean()['Survived']
Sex
female    0.742038
male      0.188908
Name: Survived, dtype: float64
 
#클래스와 성별에 다른 생존률 구해보기
dual_class=df. groupby(['Pclass','Sex'])
dual_class.mean().loc[:,'Survived']
Pclass  Sex   
1       female    0.968085
        male      0.368852
2       female    0.921053
        male      0.157407
3       female    0.500000
        male      0.135447
Name: Survived, dtype: float64

dual_class.mean().index #MultiIndex구조
MultiIndex([(1, 'female'),
            (1,   'male'),
            (2, 'female'),
            (2,   'male'),
            (3, 'female'),
            (3,   'male')],
           names=['Pclass', 'Sex'])
```



## 2. index를 이용한 group by, set_index, reset_index,aggregate

index가 있는 경우, groupby함수에 level 사용 가능

- level은 index의 depth를 의미하며, 가장 왼쪽부터 0부터 증가



set_index 함수

- column 데이터를 index레벨로 변경

reset_index 함수

- 인덱스 초기화

```python
df.set_index(['Pclass','Name'])
'Pclass'와 'Name'이 column명에서 index차원으로 내려감

df.set_index(['Pclass','Name']).reset_index()
'Pclass'와 'Name'이 다시 column명으로 올라감

df.set_index('Age').groupby(level=0).mean() #level=0은 인덱스의 첫번째



import math
def age_categorize(age):
    if math.isnan(age):
        return -1
    return math.floor(age/10)*10

df.set_index('Age').groupby(age_categorize).mean()['Survived']
#groupby를 수행할 때 index가 설정되어 있으면, groupby의 인자에 함수를 넣어 모든 사항에 함수가 적용되도록 할 수 있음

df.set_index(['Pclass','Age']).groupby(level=[0,1]).mean()
# 멀티인덱스를 set_index로 설정하고 이에 대한 groupby 설정은 level을 통해 설정함

df.set_index(['Pclass','Age']).groupby(level=[0,1]).aggregate([np.mean,np.sum, np.max ])
# 멀티인덱스를 set_index로 설정하고 이에 대한 aggregate함수를 설정하면, 복수개의 함수의 결과를 확인할 수 있음
```



## 3. tansform 함수 (groupby)

- groupby 후 transform 함수를 사용하면 `원래의 index`를 유지한 상태로 통계함수를 적용
- 전체 데이터의 집계가 아닌 각 그룹에서의 집계를 계산
- 따라서 새로 생성된 데이터를 원본 dataframe과 합치기 쉬움

```python
df.groupby('Pclass').mean()
	PassengerId	Survived	Age			SibSp		Parch		Fare
Pclass						
1	461.597222	0.629630	38.233441	0.416667	0.356481	84.154687
2	445.956522	0.472826	29.877630	0.402174	0.380435	20.662183
3	439.154786	0.242363	25.140620	0.615071	0.393075	13.675550

df.groupby('Pclass').transform(np.mean) #원래 row의 수를 갖은 데이터프레임 생성
	PassengerId	Survived	Age			SibSp		Parch		Fare
0	439.154786	0.242363	25.140620	0.615071	0.393075	13.675550
1	461.597222	0.629630	38.233441	0.416667	0.356481	84.154687
2	439.154786	0.242363	25.140620	0.615071	0.393075	13.675550
3	461.597222	0.629630	38.233441	0.416667	0.356481	84.154687
4	439.154786	0.242363	25.140620	0.615071	0.393075	13.675550

#예1    
df['Age2'] = df.groupby('Pclass').transform(np.mean)['Age']

#예2    
def saled(cost):
    return cost*0.9    
df['saled_Fare']= df.groupby('Pclass').transform(saled)['Fare']

#예3
df.groupby(['Pclass','Sex']).mean()
df['Age2'] = df.groupby(['Pclass', 'Sex']).transform(np.mean)['Age'] 
# 성별 x 클래스별 각각 평균(6개)에 대응하는 Age2가 생성됨

```



## 4. pivot, pd.pivot_table

#### pivot

- dataframe의 형태를 변경
- 인덱스, 컬럼, 데이터로 사용할 컬럼을 명시

#### pivot_table

- 기능적으로 pivot과 동일
- pivot과의 차이점
  - 중복되는 모호한 값이 있을 경우, aggregation 함수 사용하여 값을 채움

```python
pivot 활용
#출처 패스트캠퍼스 강의노트 
	지역	요일	강수량	강수확률
0	서울	월요일	100	80
1	서울	화요일	80	70
2	서울	수요일	1000	90
3	경기	월요일	200	10
4	경기	화요일	200	20
5	부산	월요일	100	30
6	서울	목요일	50	50
7	서울	금요일	100	90
8	부산	화요일	200	20
9	경기	수요일	100	80
10	경기	목요일	50	50
11	경기	금요일	100	10

df.pivot(index='요일',columns='지역') #나머지는 values로 
		강수량					강수확률
지역	경기		부산	   서울	 경기		부산	  서울
요일						
금요일	100.0	NaN		100.0	10.0	NaN		90.0
목요일	50.0	NaN		50.0	50.0	NaN		50.0
수요일	100.0	NaN		1000.0	80.0	NaN		90.0
월요일	200.0	100.0	100.0	10.0	30.0	80.0
화요일	200.0	200.0	80.0	20.0	20.0	70.0

# 하지만 pivot은 단순히 기존 데이터의 형태만 변경하는 것이기 때문에 중복값이 일을 경우 오류를 발생함
# 예를 들어 1번 인덱스의 요일을 월요일로 바꿀 경우, index 중복이 생겨서 오류가 발생함
# 따라서 값들에 함수를 적용시킬 수 있는 pd.pivot_table을 활용하는 것이 유용함

pd.pivot_table(df,index='요일', columns="지역",aggfunc='mean')

			강수량						강수확률
지역	경기		부산		서울		경기		부산		서울
요일						
금요일	100.0	NaN		100.0		10.0	NaN		90.0
목요일	50.0	NaN		50.0		50.0	NaN		50.0
수요일	100.0	NaN		1000.0		80.0	NaN		90.0
월요일	200.0	100.0	90.0		10.0	30.0	75.0
화요일	200.0	200.0	NaN			20.0	20.0	NaN

# 연습(타이타닉 데이터)
pd.pivot_table(df2, index='Sex',columns='Pclass',values=['Fare','Age'])
					Age										Fare
Pclass		1			2			3			1			2			3
Sex						
female	34.611765	28.722973	21.750000	106.125798	21.970121	16.118810
male	41.281386	30.740707	26.507589	67.226127	19.741782	12.661633
```



### 5. stack & unstack

실제 데이터분석에서 거의 사용하지 않음 

```python
new_df=df.set_index(['지역','요일'])
# 기존 df에서 지역과 요일을 index로 만들어 새로움 데이터프레임을 만듦

		  강수량 강수확률
지역	요일		
서울	월요일	100	 80
	 화요일  80  70
     수요일  1000 90
경기	월요일	200	10
	 화요일 200 20
부산	월요일	100	30
서울	목요일	50	50
	 금요일 100 90
부산	화요일	200	20
경기	수요일	100	80
	 목요일 50	 50
	 금요일 100 10

new_df.unstack(0) #unstack은 index에 쌓인 것을 컬럼으로 변경함
new_df.unstack(1) #1번 인덱스틑 요일
new_df.unstack(1).stack(0)
new_df.unstack(1).stack(1) #stack은 컬럼이 있는 요인을 index로 쌓음
#수행 결과는 아래 그림 참고
```

![new_df](https://i.imgur.com/ZU8pYcU.png)

![](https://i.imgur.com/CIZdpPI.png)

![](https://i.imgur.com/tzbW8aB.png)

![](https://i.imgur.com/mTvc37V.png)

![](https://i.imgur.com/LLXL3Hi.png)

