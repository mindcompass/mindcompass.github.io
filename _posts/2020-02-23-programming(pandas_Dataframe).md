---
title:  "python pandas 정리2_Dataframe"
excerpt: "pandas의 Dataframe 함수 정리"
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
last_modified_at: 2019-12-23
---

## 1.데이터프레임 탐색하는 함수

```python
train_data =pd.read_csv("./train.csv") #타이타닉 데이터 읽어오기
(df명).head() #앞의 5행을 보여줌
(df명).tail() #마지막 5행을 보여줌

train_data.shape #데이터프레임의 형태를 보여줌 trainging데이터는 891행 12열로 구성됨
(891, 12)

train_data.describe()
	PassengerId	Survived	Pclass		Age		SibSp		Parch		Fare
count 	891.0000	891.00000	891.00000	714.00000	891.00000	891.00000	891.00000
mean	446.0000	0.383838	2.308642	29.699118	0.523008	0.381594	32.204208
std	257.353842	0.486592	0.836071	14.526497	1.102743	0.806057	49.693429
min	1.000000	0.000000	1.000000	0.420000	0.000000	0.000000	0.000000
25%	223.500000	0.000000	2.000000	20.125000	0.000000	0.000000	7.910400
50%	446.000000	0.000000	3.000000	28.000000	0.000000	0.000000	14.454200
75%	668.500000	1.000000	3.000000	38.000000	1.000000	0.000000	31.000000
max	891.000000	1.000000	3.000000	80.000000	8.000000	6.000000	512.329200

train_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
    
train_data.index
RangeIndex(start=0, stop=891, step=1)

train_data.columns
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')

type(train_data)
pandas.core.frame.DataFrame
```



## 2.데이터프레임 생성하기

실제 데이터분석에서 거의 사용하지 않음 

```py
data1={'a':10, 'b':20, 'c':30}
pd.DataFrame(data1,index=['x','y','z']) # dict의 각 성분이 열 형태로 구성됨 
	a	b	c
x	10	20	30
y	10	20	30
z	10	20	30

data2={'a':[10,20,30], 'b':[40,50,60], 'c':[70,80,90]}
pd.DataFrame(data2,index=['x','y','z'])
	a	b	c
x	10	40	70
y	20	50	80
z	30	60	90

a=pd.Series([1,2,3],['a','b','c']) 
b=pd.Series([10,20,30],['a','b','c'])
c=pd.Series([100,200,300],['a','b','c'])
#각 행의 인덱스(['a','b','c'])가 결합하면 데이터프레임의 컬럼 네임이 됨
   a	b	c
0	1	2	3
1	10	20	30
2	100	200	300


```

## 3. 데이터프레임 특정 row, column선택하기

```python
train_data = pd.read_csv('./train.csv')
train_data[0] # 데이터 프레임에서 특정 숫자로 인덱스하려고 하면 오류가 남(df에서는 열로 인덱싱하는데 d열 이름이 0인 column이 없기 때문에)

train_data['Pclass'] # 리턴되는 값이 시리즈객체
0      3
1      1
2      3
3      1
4      3
      ..
886    2
887    1
888    3
889    1
890    3
Name: Pclass, Length: 891, dtype: int64

train_data[['Pclass']] #리턴되는 값이 데이터 프레임
train_data[['Pclass','Age']] #2개 이상 칼럼을 사용하려면 [[ ]]을 통해 인덱싱함
            
        
```

train_data[5:10] [["PassengerId","Survived","Pclass"]]

`슬라이싱`은 예외로 행을 선택할 수 있음

|      | PassengerIdSurvived | SurvivedPclass | Pclass |
| :--: | :-----------------: | :------------: | :----: |
|  5   |          6          |       0        |   3    |
|  6   |          7          |       0        |   1    |
|  7   |          8          |       0        |   3    |
|  8   |          9          |       1        |   3    |
|  9   |         10          |       1        |   2    |



## 4.행을 포함한 인덱싱을 위한 함수 loc 과 iloc

```python
#loc은 데이터프레임에 존재하는 index를 활용하는 함수
#iloc은 존재하는 index와 상관없이 0베이스 기본 index를 사용함

train_data.loc[[100,200,300,400],['PassengerId','Survived','Pclass']]

		PassengerId	Survived  Pclass
    100		101			0		3
    200		201			0		3
    300		301			1		3
    400		401			1		3
    
train_data.iloc[[100,200,300,400],[0,1,2]] 리턴 결과는 위 loc과 동일
#단 이런 방식으로는 슬라이싱이 통하지 않음/ slicing을 사용하기 위해서는 아래처럼 구성해야 함
train_data.iloc[0:10,0:5]
train_data.loc[0:10,"PassengerId":"Name"]
# slicing을 하는 경우, []을 사용하지 않음

#슬라이싱을 위해서는 다음처럼 구성해야 함
train_data.loc[100:200][["Name","Pclass"]]
이경우는 loc을 빼고 사용해도 동일함
train_data[100:200][["Name","Pclass"]]
```

## 5. boolean selection으로 row선택

numpy와 동일한 방식으로 해당 조건에 맞는 row만 선택

```python
#타이타닉 데이터에서 1등석에타면서 30대인 사람의 수를 구해보자

age30 = (train_data['Age']>=30 )&(train_data['Age']<40 ) #True, False로 구성된 시리즈 리턴
class1 =train_data['Pclass']==1

train_data[age30&class1].shape
(50, 12)

```

##  6. column 추가 & 삭제하기

```python
#타이타닉 데이터에서 10% 할인된 운행료 컬럼 추가

train_data['Saled Fare']= train_data['Fare']*0.9
#맨 마지막 열에 추가됨

train_data.drop('Saled Fare',axis=1,inplace=True)
train_data.insert(10,'Saled Fare',train_data['Fare']*0.9)
#insert함수를 통해서 원하는 위치에 특정컬럼을 넣을 수 있음
```

## 7. 각 변수에 대한 상관관계 구하기

```python
train_data.corr()
train_data.iloc[:,[1,2,5,6]].corr()
			Survived	Pclass			Age		  SibSp
Survived	1.000000	-0.338481	-0.077221	-0.035322
Pclass		-0.338481	1.000000	-0.369226	0.083081
Age			-0.077221	-0.369226	1.000000	-0.308247
SibSp		-0.035322	0.083081	-0.308247	1.000000

plt.matshow(train_data.iloc[:,[1,2,5,6]].corr()) # 상관관계에 따른 시각화가 가능함
```



## 8. NaN데이터 처리하기(dropna, isna, fillna)

1. NaN데이터가 있는 row/column 삭제

2. NaN데이터를 특정 값으로 대체함

```python
train_data.info()
train_data.isna()
train_data['Embarked'].isna()
train_data['Embarked'].isna().sum()
2

train_data.dropna() #NaN이 하나라도 있으면 해당 row삭제
train_data.dropna(subset=['Embarked']) 
다른 열에 NaN이 있다고 해도 'Embarked'의 값이 존재하면 해당 ROW는 유지함

train_data.dropna(axis=1) #하나라도 컬럼에 NaN이 있으면 해당 컬럼을 삭제하라
train_data['Age'].fillna(train_data['Age'].mean())

#1)생존자 나이 평균, 사망자 나이 평균을 각각 구하고, 
#2)생존자 중 Age가 결측치인 값을 생존자 나이 평균값으로 대치하고, 
#3)사망자 중 Age가 결측치인 값을 사망자 평균 나이 값으로 대체하라
mean1=train_data[train_data['Survived']==1]['Age'].mean()
mean2=train_data[train_data['Survived']==0]['Age'].mean()
print(mean1,mean2)
28.54977812177503 30.415099646415896

train_data[train_data['Survived']==1]['Age'].isna().sum()
52
train_data[train_data['Survived']==0]['Age'].isna().sum()
125

train_data[train_data['Survived']==0]['Age'].fillna(mean1,inplace=True)
# survived가 0인 집단에 대한 age를 선별했기 때문에 메모리 구조가 변경이 일어나 데이터 복제가 일어남
#(따라서 inplace가 작동하지 않음)

대체하기 위해서는 
train_data.loc[train_data['Survived']==1,'Age']=train_data[train_data['Survived']==1]['Age'].fillna(mean1)
train_data.loc[train_data['Survived']==0,'Age']=train_data[train_data['Survived']==0]['Age'].fillna(mean2)

loc을 안쓰고 아래처럼 하게 되면 loc을 사용해서 처리하라고 error를 발생시킨다.
train_data[train_data['Survived']==1]['Age']=train_data[train_data['Survived']==1]['Age'].fillna(mean1)
```



## 9.데이터형 변형(숫자데이터 -> 범주형 데이터)

```python
train_data.info()
Survived  891 non-null int64 # Survived 의 자료형이 int64인것을 확인할 수 있음

train_data['Pclass']=train_data['Pclass'].astype(str) # Survived 의 자료형을 문자형으로 변형

train_data.info()
Pclass  891 non-null object # Survived 의 자료형이 object(문자형)인것을 확인할 수 있음


def age_change(age):
    return math.floor(age/10)*10

train_data['Age'].apply(age_change)
=>ValueError: cannot convert float NaN to integer
    
#함수를 변경함
def age_change(age):
    if math.isnan(age):
        return -1
    return math.floor(age/10)*10

train_data['Age'].apply(age_change)
```



## 10. 범주형 데이터 전처리(one-hot endcoding)

기본적으로 범주형 데이터의 경우 연산이 불가능하기 때문에 연산이 가능하도록 숫자형으로 변경할 필요가 있음

범주의 종류만큼 컬럼을 생성하고(예를 들어 나이가 10대, 20대, 30대, 40대, 50대 이상으로 구분된 범주형 데이터를 각각 5개의 컬럼을 생성하고, 20대의 경우 20대에만 1을 표시하고 나머지의 경우 0으로 채움) 

회귀에서 번주형 데이터를 회기분석의 독립변수로 사용하기 위해 더미코드를 생성하여 연산하는 것과 유사함

```python
pd.get_dummies(train_data,columns=['Pclass','Sex'])
```

![](https://i.imgur.com/PZQTof6.png)

get_dumy함수에서 drop_fisrt를 true를 줄 경우, 첫번째 범주형 데이터컬럼이 사라짐

범주형 컬럼들이 더미변수화가 되었을 때, 한가지 컬럼이 사라지더라도 다른 것과의 관계를 통해 구분할 수 있음

남자가 1이면, 무조건 여자는 0이됨

```python
pd.get_dummies(train_data,columns=['Pclass','Sex'], drop_first=True)
```

![](https://i.imgur.com/iewwwaH.png)

