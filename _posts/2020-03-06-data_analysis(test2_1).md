---
title:  "데이터분석 실전2_1(데이터 전처리)"
excerpt: "kaggle의 제철소 데이터를 활용한 간단한 머신러닝 적용"
toc: true
toc_sticky: true

categories:
  - data_analysis
tags:
- 전처리
- pandas
- 
- 
use_math: true
last_modified_at: 2020-03-06
---



 

이번에는 케글에 올라온 데이터를 바탕으로 간단한 데이터분석을 진행하려고 합니다.    

이 데이터는 커뮤니케이션 과학 연구센터인 Semeion의 연구에서 나온 것이고, 스테인리스 강판의 표면 결합 유형을 올바르게 분류하는 것이 었습니다.  6개의 결함 유형에 대한 추정을 할 수 있는 27가지 지표가 있는데, 아쉽게도 여기에 대한 세부적인 사항은 나와있지 않습니다. 



사이트에 나와있는 데이터셋의 설명은 아래와 같습니다. 

There are 34 fields. The first 27 fields describe some kind of steel plate faults seen in images. Unfortunately, there is no other information that I know of to describe these columns.

- X_Minimum
- X_Maximum
- Y_Minimum
- Y_Maximum
- Pixels_Areas
- X_Perimeter 둘레
- Y_Perimeter 
- Sum*of*Luminosity
- Minimum*of*Luminosity 광도
- Maximum*of*Luminosity 광도
- Length*of*Conveyer 컨베이어벨트 길이
- TypeOfSteel_A300 
- TypeOfSteel_A400
- Steel*Plate*Thickness 두께
- Edges_Index 
- Empty_Index 
- Square_Index 얼마나 정사각형인지?
- Outside*X*Index
- Edges*X*Index
- Edges*Y*Index
- Outside*Global*Index
- LogOfAreas 
- Log*X*Index
- Log*Y*Index
- Orientation_Index
- Luminosity_Index
- SigmoidOfAreas

The last seven columns are one hot encoded classes, i.e. if the plate fault is classified as "Stains" there will be a 1 in that column and 0's in the other columns. If you are unfamiliar with one hot encoding, just know that the last seven columns are your class labels.

- Pastry
- Z_Scratch
- K_Scatch
- Stains
- Dirtiness
- Bumps
- Other_Faults

---------



## 1. 분석 데이터 탐색

### 1) 변수별 히스토그램 그리기

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data1= pd.read_csv('./data_set/steel/faults.csv') #파일에 맞는 데이터 경로 설정

data1.iloc[0:5,0:5] # 실제로는 data1.head()를 수행하세요 
	X_Minimum	X_Maximum	Y_Minimum	Y_Maximum	Pixels_Areas
0	42		50		270900		270944		267
1	645		651		2538079		2538108		108
2	829		835		1553913		1553931		71
3	853		860		369370		369415		176
4	1289		1306		498078		498335		2409

pd.options.display.float_format = '{:.1f}'.format # 소숫점 1자리로만 보여주려고 합니다.

index1=data1.columns
index1
Index(['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
       'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity',
       'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer',
       'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness',
       'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
       'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas',
       'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index',
       'SigmoidOfAreas', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
       'Dirtiness', 'Bumps', 'Other_Faults'],
      dtype='object')

#넘파이에서는 for문을 사용하지 않는게 정석이지만 간편하게 그리기 위해서 사용했습니다. 
plt.figure(figsize=(20,40)) 
for col in range(len(index1)):
    plt.subplot(10,4,col+1)
    plt.hist(data1[index1[col]])
    plt.title(index1[col],size=20)
plt.tight_layout(pad=1.5)
plt.show()
```



![변수_히스토그램](https://i.imgur.com/y9teIex.png)



### 2) 각 변수에 대한 기술적 통계

```python
desc=data1.describe()
desc.T #표가 세로로 길기 때문에 데이터프레임을 전치했습니다.
```



| **변수명**                | **count** | **mean**  | **std**   | **min** | **25%**  | **50%**   | **75%**   | **max**    |
| ------------------------- | ---------- | --------- | --------- | ------- | -------- | --------- | --------- | ---------- |
| **X_Minimum**             | 1941.0    | 571.1     | 520.7     | 0.0     | 51.0     | 435.0     | 1053.0    | 1705.0     |
| **X_Maximum**             | 1941.0    | 618.0     | 497.6     | 4.0     | 192.0    | 467.0     | 1072.0    | 1713.0     |
| **Y_Minimum**             | 1941.0    | 1650684.9 | 1774578.4 | 6712.0  | 471253.0 | 1204128.0 | 2183073.0 | 12987661.0 |
| **Y_Maximum**             | 1941.0    | 1650738.7 | 1774590.1 | 6724.0  | 471281.0 | 1204136.0 | 2183084.0 | 12987692.0 |
| **Pixels_Areas**          | 1941.0    | 1893.9    | 5168.5    | 2.0     | 84.0     | 174.0     | 822.0     | 152655.0   |
| **X_Perimeter**           | 1941.0    | 111.9     | 301.2     | 2.0     | 15.0     | 26.0      | 84.0      | 10449.0    |
| **Y_Perimeter**           | 1941.0    | 83.0      | 426.5     | 1.0     | 13.0     | 25.0      | 83.0      | 18152.0    |
| **Sum_of_Luminosity**     | 1941.0    | 206312.1  | 512293.6  | 250.0   | 9522.0   | 19202.0   | 83011.0   | 11591414.0 |
| **Minimum_of_Luminosity** | 1941.0    | 84.5      | 32.1      | 0.0     | 63.0     | 90.0      | 106.0     | 203.0      |
| **Maximum_of_Luminosity** | 1941.0    | 130.2     | 18.7      | 37.0    | 124.0    | 127.0     | 140.0     | 253.0      |
| **Length_of_Conveyer**    | 1941.0    | 1459.2    | 144.6     | 1227.0  | 1358.0   | 1364.0    | 1650.0    | 1794.0     |
| **TypeOfSteel_A300**      | 1941.0    | 0.4       | 0.5       | 0.0     | 0.0      | 0.0       | 1.0       | 1.0        |
| **TypeOfSteel_A400**      | 1941.0    | 0.6       | 0.5       | 0.0     | 0.0      | 1.0       | 1.0       | 1.0        |
| **Steel_Plate_Thickness** | 1941.0    | 78.7      | 55.1      | 40.0    | 40.0     | 70.0      | 80.0      | 300.0      |
| **Edges_Index**           | 1941.0    | 0.3       | 0.3       | 0.0     | 0.1      | 0.2       | 0.6       | 1.0        |
| **Empty_Index**           | 1941.0    | 0.4       | 0.1       | 0.0     | 0.3      | 0.4       | 0.5       | 0.9        |
| **Square_Index**          | 1941.0    | 0.6       | 0.3       | 0.0     | 0.4      | 0.6       | 0.8       | 1.0        |
| **Outside_X_Index**       | 1941.0    | 0.0       | 0.1       | 0.0     | 0.0      | 0.0       | 0.0       | 0.9        |
| **Edges_X_Index**         | 1941.0    | 0.6       | 0.2       | 0.0     | 0.4      | 0.6       | 0.8       | 1.0        |
| **Edges_Y_Index**         | 1941.0    | 0.8       | 0.2       | 0.0     | 0.6      | 0.9       | 1.0       | 1.0        |
| **Outside_Global_Index**  | 1941.0    | 0.6       | 0.5       | 0.0     | 0.0      | 1.0       | 1.0       | 1.0        |
| **LogOfAreas**            | 1941.0    | 2.5       | 0.8       | 0.3     | 1.9      | 2.2       | 2.9       | 5.2        |
| **Log_X_Index**           | 1941.0    | 1.3       | 0.5       | 0.3     | 1.0      | 1.2       | 1.5       | 3.1        |
| **Log_Y_Index**           | 1941.0    | 1.4       | 0.5       | 0.0     | 1.1      | 1.3       | 1.7       | 4.3        |
| **Orientation_Index**     | 1941.0    | 0.1       | 0.5       | -1.0    | -0.3     | 0.1       | 0.5       | 1.0        |
| **Luminosity_Index**      | 1941.0    | -0.1      | 0.1       | -1.0    | -0.2     | -0.1      | -0.1      | 0.6        |
| **SigmoidOfAreas**        | 1941.0    | 0.6       | 0.3       | 0.1     | 0.2      | 0.5       | 1.0       | 1.0        |
| **Pastry**                | 1941.0    | 0.1       | 0.3       | 0.0     | 0.0      | 0.0       | 0.0       | 1.0        |
| **Z_Scratch**             | 1941.0    | 0.1       | 0.3       | 0.0     | 0.0      | 0.0       | 0.0       | 1.0        |
| **K_Scatch**              | 1941.0    | 0.2       | 0.4       | 0.0     | 0.0      | 0.0       | 0.0       | 1.0        |
| **Stains**                | 1941.0    | 0.0       | 0.2       | 0.0     | 0.0      | 0.0       | 0.0       | 1.0        |
| **Dirtiness**             | 1941.0    | 0.0       | 0.2       | 0.0     | 0.0      | 0.0       | 0.0       | 1.0        |
| **Bumps**                 | 1941.0    | 0.2       | 0.4       | 0.0     | 0.0      | 0.0       | 0.0       | 1.0        |
| **Other_Faults**          | 1941.0    | 0.3       | 0.5       | 0.0     | 0.0      | 0.0       | 1.0       | 1.0        |

### 3) 예측할 종속변수 설정하기

예측하는 변수가 7개나 되기 때문에 단순화를 위해 종속변수를 정리하고자 합니다. 

변수를 보면 6개의 결함지수가 ONE HOT 인코딩 방식으로 되어 있고, Other_Faults에서는 해당 6가지 결함이 없으면 1, 1개라도 있으면 0로 코딩되어 있습니다. 정확히는 알수 없지만 주요 결함은 아닌 것으로 주요 결함보다는 사소한 것 같습니다. 

저는 총 2가지 종속변수를 정리할 생각입니다. 6개의 주요 faults가 있는 경우를 1로 아닌 경우를 0으로 만들어서 해당 변수만을 분석하고 싶습니다. 그래서 Other_Faults의 변수를 원래 값이 1인 경우를 0으로 만들고, 원래 값이 0인 경우를 1로 변경된 새로운 변수(main_fautls)를 만들려고 합니다. 

또한 결함의 종류가 포함된 내용 두번째 변수(kind_of_faults)를 만들겠습니다.

```python
data1.iloc[:,-7:].head(5)
	Pastry	Z_Scratch	K_Scatch	Stains	Dirtiness	Bumps	Other_Faults
0	1	0		0		0	0		0	0
1	1	0		0		0	0		0	0
2	1	0		0		0	0		0	0
3	1	0		0		0	0		0	0
4	1	0		0		0	0		0	0


# 혹시 몰라서 원자료의 코딩이 제대로 되었는지 확인해봅니다. 
# boolean값으로된 faults라는 칼럼을 만들어 합계를 만들어 보았습니다.
data1['faults']=data1.iloc[:,-7:].sum(axis=1) ==1
data1['faults'].sum() #즉 7개 열에서 1줄에 모두 1개의 1이 존재하는 상황입니다.
1941

#Other_Faults와 값이 반대인 새로운 변수인 main_faults를 만들었습니다.
data1['main_faults']=np.where(data1['Other_Faults']==1, 0, 1)

	Other_Faults	faults	main_faults
0	0		True	1
1	0		True	1
2	0		True	1
3	0		True	1
4	0		True	1


one_hot=data1.iloc[:,-9:-2] #원핫인코딩 된 자료만을 선택해서 별도의 데이터프레임을 만듭니다.

#argmax()로 변경하는 방법이 있다고 하는데, 잘 안되서 idxmax을 사용했습니다. 

one_hot.columns=["1","2","3","4","5","6","7"] #데이터프레임 열이름을 번호로 바꿈니다.
# 즉 Pastry열 이름은 1로, Z_Scratch는 2로... 변경합니다.

one_hot_decoding=one_hot.idxmax(axis=1)

one_hot_decoding.head()
0       1
1       1
2       1
3       1
4       1

data2=data1.iloc[:,:27]

data2=pd.concat([data2,data1['main_faults'],one_hot_decoding],  axis=1)

data2=data2.rename(columns={0:"kind_of_faults"})
#칼럼명을 마지막으로 kind_of_faults으로 변경합니다.
```



## 3. 변수간 상관분석 및 변수 선택

이제 분석할 변수들 간의 관계를 분석하려고 합니다 .

```python
data2.iloc[:,-5:].tail()
	Orientation_Index	Luminosity_Index	SigmoidOfAreas	main_faults	kind_of_faults
1936	-0.4			0.0				0.7		0		7
1937	-0.5			-0.1				0.8		0		7
1938	-0.5			0.0				0.7		0		7
1939	-0.1			-0.0				1.0		0		7
1940	-0.2			-0.1				0.5		0		7


data2['kind_of_faults']=data2['kind_of_faults'].astype(int)
#자료형이 문자이기 때문에 정수형으로 변경해줍니다.

corr=data2.corr(method='pearson')
#변수가 너무 많아서 그냥 표로 보려면 보기 힘들기 때문에 편하게 비교할 수 있는 seaborn 라이브러리를 활용해서 이미지로 작성하겠습니다. 
# 그림 사이즈 지정

fig, ax = plt.subplots( figsize=(24,20) )

# 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# 히트맵을 그린다
sns.heatmap(corr, 
            cmap = 'RdYlBu_r', # Red, Yellow, Blue 색상으로 표시
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
sns.set(font_scale=4)
plt.show()

```

<img src="https://i.imgur.com/C8eGiDc.png" alt="이미지2" style="width:150%;" />

상관이 높은 독립변수들이 상당히 많이 포함되어 있는 것을 볼 수 있습니다. 상관관계 .8이상이 되는 것은 지나치게 상관관계가 높기 때문에 회귀분석을 수행할 때, 다중공선성의 문제가 발생할 수 있기 때문에 제외하려고 합니다. 

독립변수들 간에 강한 상관관계가 나타날 경우, 추정한 회귀계수의 분산이 매우 커져서 추정한 회귀계수를 신뢰할 수 없게 됩니다. 

상관관계가 .8이상한데이터 쌍에서 변수를 1개씩 삭제하겠습니다. 1개 이상과 .8 이상으로 상관관계가 높게 나온 변수를 삭제할 예정이고, 해당 변수는 아래와 같습니다. 

 'X_Minimum',

'Y_Maximum',

'Pixels_Areas',

'X_Perimeter',

'TypeOfSteel_A400',

'Luminosity_Index',

'Outside_X_Index',

'Outside_Global_Index',

'Edges_Y_Index',

'LogOfAreas',

'SigmoidOfAreas'



## 4. 독립변수를 표준화 점수(Z)로 변환

```python
data3=data2.drop(['X_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter','TypeOfSteel_A400','Luminosity_Index','Outside_X_Index','Outside_Global_Index','Edges_Y_Index','LogOfAreas','SigmoidOfAreas'] ,axis=1)

corr=data3.corr(method='pearson')

# 그림 사이즈 지정
fig, ax = plt.subplots( figsize=(40,40) )

# 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# 히트맵을 그린다
sns.heatmap(corr, 
            fmt='1.1f',
            cmap = 'RdYlBu_r', # Red, Yellow, Blue 색상으로 표시
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
sns.set(font_scale=2.5)
plt.show()
```

VIF(분산팽창)지수를 이용해서 하나씩 변수를 삭제하면서 구할 수도 있지만, 시간관계상 0.8이상 같은 변수쌍이 있을 경우, 1개씩 삭제해서 11개의 변수를 삭제 했습니다.  특히 여러 변수와 상관관계가 높은 변수를 중심으로 삭제했습니다.

![상관관계이미지](https://i.imgur.com/LtzWkMf.png)

각 변수간의 상관관계가 높은 변수들을 삭제했더니 .8이상 상관관계가 모두 사라졌습니다. 

추가적으로 아직도 변수들간의 독립변수들 간의 상관관계가 높기 때문에  독립변수들을 표준점수(Z)로 변경해줍니다.

```python
import scipy.stats as ss
data_standadized_ss = ss.zscore(data3)
data4= pd.DataFrame(data_standadized_ss,columns=data3.columns)

data5=pd.concat([data4.iloc[:,:-2],data3[['main_faults','kind_of_faults']]], axis=1)
#'main_faults','kind_of_faults'값은 의도치 않게 표준화되었기 때문에 원래 값으로 대체해줬습니다.

index2=data5.columns

plt.figure(figsize=(20,40))
for col in range(len(index2)):
    plt.subplot(10,4,col+1)
    plt.hist(data5[index2[col]])
    plt.title(index2[col],size=20)
plt.tight_layout(pad=1.5)
plt.show()
```

![표준화된 변수](https://i.imgur.com/9dUnWBh.png)

