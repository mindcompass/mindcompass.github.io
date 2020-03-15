---
title:  "pandas를 통한 데이터 전처리 연습하기"
excerpt: "pandas를 통한 데이터 전처리 연습"
toc: true
toc_sticky: true

categories:
  - data_analysis
tags:
- pandas
- EDA
- 
- 

use_math: true
last_modified_at: 2020-03-14
---



## 1. 갭마인더 데이터를 통한 데이터 전처리 연습

### 1. 데이터 탐색
```python

import pandas


df=pd.read_csv('https://raw.githubusercontent.com/jennybc/gapminder/master/inst/extdata/gapminder.tsv', sep='\t')
#파일을 로드합니다.


df.head() #처음 5행을 확인합니다.
	country		continentyear	lifeExp	pop		gdpPercap
0	Afghanistan	Asia	1952	28.801	8425333		779.445314
1	Afghanistan	Asia	1957	30.332	9240934		820.853030
2	Afghanistan	Asia	1962	31.997	10267083	853.100710
3	Afghanistan	Asia	1967	34.020	11537966	836.197138
4	Afghanistan	Asia	1972	36.088	13079460	739.981106


df.info() # 데이터프레임의 각 컬럼별 특성을 파악합니다. 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1704 entries, 0 to 1703
Data columns (total 6 columns):
country      1704 non-null object
continent    1704 non-null object
year         1704 non-null int64
lifeExp      1704 non-null float64
pop          1704 non-null int64
gdpPercap    1704 non-null float64
dtypes: float64(2), int64(2), object(2)
memory usage: 80.0+ KB

```

```python
type(df) #전체 파일의 형태를 파악합니다.  
pandas.core.frame.DataFrame


df.shape #데이터 프레임의 행의 수와 컬럼의 수를 확인합니다.  
(1704, 6)


df.describe() #연속변수에 해당하는 커럼의 특성을 파악합니다.

	year		lifeExp		pop		gdpPercap
count	1704.00000	1704.000000	1.704000e+03	1704.000000
mean	1979.50000	59.474439	2.960121e+07	7215.327081
std	17.26533	12.917107	1.061579e+08	9857.454543
min	1952.00000	23.599000	6.001100e+04	241.165877
25%	1965.75000	48.198000	2.793664e+06	1202.060309
50%	1979.50000	60.712500	7.023596e+06	3531.846989
75%	1993.25000	70.845500	1.958522e+07	9325.462346
max	2007.00000	82.603000	1.318683e+09	113523.132900
```





### 2. 연습문제1

그럼 문제를 풀면서 하나씩 전처리 연습을 해봅시다.



문제1. [Quiz] pop 평균보다 인구가 높은 국가의 1970년대 데이터를 필터링하고,continent, country, pop, gdpPercap 열만 출력하라.

```python
df['year'].unique() #우선 year데이터에 어떤 값이 있는지 확인합니다. 

array([1952, 1957, 1962, 1967, 1972, 1977, 1982, 1987, 1992, 1997, 2002,
       2007], dtype=int64)

#1970년대 데이터는 1972년과 1977년이 있네요.

df1=df[(df['pop']>=df['pop'].mean())&(df['year'].isin([1972,1977]))]

	country		continent	year	lifeExp		pop		gdpPercap
100	Bangladesh	Asia		1972	45.25200	70759295	630.233627
101	Bangladesh	Asia		1977	46.92300	80428306	659.877232
172	Brazil		Americas	1972	59.50400	100840058	4985.711467
173	Brazil		Americas	1977	61.48900	114313951	6660.118654
292	China		Asia		1972	63.11888	862030000	676.900092


df1[['continent','country', 'pop', 'gdpPercap']]

	continent	country		pop		gdpPercap
100	Asia		Bangladesh	70759295	630.233627
101	Asia		Bangladesh	80428306	659.877232
172	Americas	Brazil		100840058	4985.711467
173	Americas	Brazil		114313951	6660.118654
292	Asia		China		862030000	676.900092
```

문제2. 문제1에서 추출한 데이터프레임에서 한국, 중국, 일본의 데이터를 추출하고, 국가별로 그룹화해서 인구수의 평균을 구하라.

```python
df2=df1[['continent','country', 'pop', 'gdpPercap']]

df2.country.unique()
array(['Bangladesh', 'Brazil', 'China', 'Egypt', 'Ethiopia', 'France',
       'Germany', 'India', 'Indonesia', 'Iran', 'Italy', 'Japan',
       'Korea, Rep.', 'Mexico', 'Myanmar', 'Nigeria', 'Pakistan',
       'Philippines', 'Poland', 'Spain', 'Thailand', 'Turkey',
       'United Kingdom', 'United States', 'Vietnam'], dtype=object)

#값을 확인해서 east_asia에 넣어줍니다.
east_asia=['Korea, Rep.','Japan','China']

df2[df2['country'].isin(east_asia)]
	continent	country		pop		gdpPercap
292	Asia		China		862030000	676.900092
293	Asia		China		943455000	741.237470
796	Asia		Japan		107188273	14778.786360
797	Asia		Japan		113872473	16610.377010
844	Asia		Korea, Rep.	33505000	3030.876650
845	Asia		Korea, Rep.	36436000	4657.221020


df2[df2['country'].isin(east_asia)].groupby(['country']).mean()['pop']

		pop	
country		
China		902742500	
Japan		110530373	
Korea, Rep.	34970500	
```

문제3. 문제2에서 작성한 데이터프레임으로 3개국의 1960년대 인구수를 막대그래프로 작성하라(plot()을 사용할 것)

```python
df3=df2[df2['country'].isin(east_asia)].groupby(['country']).mean()['pop']

df3.plot(kind='bar',title="1960's population in East_asia", rot=1, color=['red', 'blue', 'green'])
```





![그래프](https://i.imgur.com/5C5mTpC.png)



## 2.연습문제2(melt, pivot_table 연습)

문제1. 데이터 셋 생성

1. 각각 10명의 학생수를 갖는 두 개의 class A, B를 생성한다.

2. 각 클래스의 학생은 id : 1 ~ 10으로 구분된다.

3. 각 학생들에 대해 math, eng, sci 과목에 대한 30 ~ 100 사이의 random 점수를 생성하라.

   최종 scores 데이터 프레임은 아래와 같다.

   

| class | stu  | math | eng  | sci  |
| :---: | :--: | :--: | :--: | :--: |
|   A   |  1   |  85  |  92  |  51  |
|   A   |  2   |  54  |  52  |  42  |
|   A   |  3   |  89  |  43  |  35  |
|  ...  | ...  | ...  | ...  | ...  |
|   B   |  9   |  64  |  67  |  62  |
|   B   |  10  |  85  |  89  |  42  |

```python
import pandas as pd
import numpy as np
import random

classA=pd.DataFrame({
'class' :['A']*10,
'stu' :np.arange(1,11),
'math' : np.random.randint(30,100,10),
'eng' : np.random.randint(30,100,10),
'sci' : np.random.randint(30,100,10),
}
)

classB=pd.DataFrame({
'class' :['B']*10,
'stu' :np.arange(1,11),
'math' : np.random.randint(30,100,10),
'eng' : np.random.randint(30,100,10),
'sci' : np.random.randint(30,100,10),
}
)

data_total=pd.concat([classA,classB])
```



문제2. melt 함수를 이용해서 subj 변수를 새롭게 만들고, 그 변수의 값에 'math', 'sci', 'eng'를 갖게 하고, score라는 변수를 만들어 3개의 과목에 대한 각각의 점수를 갖도록 하라(학생별로 3과목이 연달아 오도록 할 것)

| class | stu  | **subj** | score |
| :---: | :--: | :------: | :---: |
|   A   |  1   |   math   |  92   |
|   A   |  1   |   eng    |  52   |
|   A   |  1   |   sci    |  43   |
|  ...  | ...  |   ...    |  ...  |
|   B   |  10  |   math   |  67   |
|   B   |  10  |   eng    |  65   |
|   B   |  10  |   sci    |  84   |



```python
data_total_melt=pd.melt(data_total, id_vars=['class','stu'],var_name='subj',value_name='score').sort_values('stu')
```



문제3.  문제2에서 만든 데이터 프레임을 사용해서 pivot_table을 적용해서 문제1의 데이터 형태로 만들고, 학생별 math, eng, sci의 변수의 평균의 값으로 변수 mean을 생성하라

| class | stu  | math | eng  | sci  | mean |
| :---: | :--: | :--: | :--: | :--: | :--: |
|   A   |  1   |  85  |  92  |  51  | 76.0 |
|   A   |  2   |  54  |  52  |  42  | 49.3 |
|   A   |  3   |  89  |  43  |  35  | 55.7 |
|  ...  | ...  | ...  | ...  | ...  | ...  |
|   B   |  9   |  64  |  67  |  62  | 64.3 |
|   B   |  10  |  85  |  89  |  42  | 72.0 |



```python
data_total2=data_total_melt.pivot_table(index=['class','stu'],values='score',columns='subj')

data_total2=data_total2.reset_index() # pivot_table을 하면서 생성되었던 index(class,stu)을 변수로 변환함

data_total2['mean']=(data_total2['eng']+data_total2['math']+data_total2['sci'])/3

data_total2.drop('index',axis=1,inplace=True)
```



문제4.  반별로,상위 5명의 학생에 대해 내림차순으로 나타내라, 아래와 같은 표를 작성하라(평균점수별로 내림차순, class에는 A반 다음,B반이 위치할 것 )

| class | stu  | mean |
| :---: | :--: | :--: |
|   A   |  5   |  87  |
|  ...  | ...  | ...  |
|   A   |  2   |  42  |
|   B   |  9   |  81  |
|  ...  | ...  | ...  |
|   B   |  7   |  53  |


```python
data_total2.sort_values(by=['class','mean'],ascending=[True,False]).groupby('class').head(5)
```

