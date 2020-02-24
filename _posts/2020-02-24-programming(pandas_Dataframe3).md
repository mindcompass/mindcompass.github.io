---
title:  "python pandas 정리4_Dataframe3"
excerpt: "concat, merge'
toc: true
toc_sticky: true

categories:
  - programming
tags:
- Python
- pandas
- Dataframe
- merge
use_math: true
last_modified_at: 2020-02-24
---

## 1. Concat 함수 데이터프레임 결합

단순하게 두개의 데이터프레임에서 컬럼의 종류나 행의 개수가 같은 것을 합침  

```python
df1 = pd.DataFrame({'key1' : np.arange(5), 'value1' : np.random.randn(5)})
df2 = pd.DataFrame({'key1' : np.arange(5), 'value1' : np.random.randn(5)})
#칼럼명이 같고 행의 개수가 5개로 같은 데이터프레임 생성

pd.concat([df1,df2])
	key1	value1
0	0	0.843955
1	1	0.991502
2	2	-0.637123
3	3	0.370595
4	4	2.349160
0	0	2.169253
1	1	-0.911121
2	2	1.887487
3	3	-0.599125
4	4	-1.368194

pd.concat([df1,df2], ignore_index=True) #기존 index를 무시하고 다시 0베이스 index를 생성함 
	key1	value1
0	0	0.843955
1	1	0.991502
2	2	-0.637123
3	3	0.370595
4	4	2.349160
5	0	2.169253
6	1	-0.911121
7	2	1.887487
8	3	-0.599125
9	4	-1.368194


#만약 행으로 concat를 수행할 때, column명이 다르면 없는 컬럼명에 NaN이 발생함 

# 단순한 수평 결합
pd.concat([df1,df2],  axis=1)
	key1	value1	key1	value1
0	0	0.843955	0	2.169253
1	1	0.991502	1	-0.911121
2	2	-0.637123	2	1.887487
3	3	0.370595	3	-0.599125
4	4	2.349160	4	-1.368194
```



## 2. Merge & join함수로 데이터 프레임 결합

출처 : 온라인 패스트캠퍼스 온라인 강의 노트

#### dataframe merge

- SQL의 join처럼 특정한 column을 기준으로 병합

  - join 방식: how 파라미터를 통해 명시
    - inner: 기본값, 일치하는 값이 있는 경우
    - left: left outer join
    - right: right outer join
    - outer: full outer join

- pandas.merge 함수가 사용됨

  

```python
자료 출처: 온라인 패스트캠퍼스 온라인 강의자료

customer = pd.DataFrame({'customer_id' : np.arange(6), 
                    'name' : ['철수'"", '영희', '길동', '영수', '수민', '동건'], 
                    '나이' : [40, 20, 21, 30, 31, 18]})

customer
	customer_id	name	나이
0		0		철수		40
1		1		영희		20
2		2		길동		21
3		3		영수		30
4		4		수민		31
5		5		동건		18    
    
orders = pd.DataFrame({'customer_id' : [1, 1, 2, 2, 2, 3, 3, 1, 4, 9], 'item' : ['치약', '칫솔', '이어폰', '헤드셋', '수건', '생수', '수건', '치약', '생수', '케이스'], 'quantity' : [1, 2, 1, 1, 3, 2, 2, 3, 2, 1]})
orders
	customer_id	item	quantity
0		1		치약		1
1		1		칫솔		2
2		2		이어폰		1
3		2		헤드셋		1
4		2		수건		3
5		3		생수		2
6		3		수건		2
7		1		치약		3
8		4		생수		2
9		9		케이스		1


pd.merge(customer,orders, on='customer_id')
	customer_id	name	나이	item	quantity
0		1		영희	20	치약	1
1		1		영희	20	칫솔	2
2		1		영희	20	치약	3
3		2		길동	21	이어폰	1
4		2		길동	21	헤드셋	1
5		2		길동	21	수건	3
6		3		영수	30	생수	2
7		3		영수	30	수건	2
8		4		수민	31	생수	2
#inner조인이기 때문에 주문 내역이 없는 동건을 join에서 제외됨, 또한 고객테이블에 없는 9번 고객도 제외됨

pd.merge(customer,orders, on='customer_id',how='left')
	customer_id	name	나이	item	quantity
0		0		철수		40	NaN		NaN
1		1		영희		20	치약		1.0
2		1		영희		20	칫솔		2.0
3		1		영희		20	치약		3.0
4		2		길동		21	이어폰		1.0
5		2		길동		21	헤드셋		1.0
6		2		길동		21	수건		3.0
7		3		영수		30	생수		2.0
8		3		영수		30	수건		2.0
9		4		수민		31	생수		2.0
10		5		동건		18	NaN		NaN

pd.merge(customer,orders, on='customer_id',how='right')
pd.merge(customer,orders, on='customer_id',how='outer')

customer1=customer.set_index('customer_id')
orders1=orders.set_index('customer_id')
pd.merge(customer1,orders1,left_index=True,right_index=True)
pd.merge(customer1,orders,left_index=True,right_on="customer_id")
#꼭 on이 아니더라도 left_index, right_on같은 속성으로 merge가능


#연습문제
#1.가장 많이 팔린 아이템은?
df3=pd.merge(customer,orders, on='customer_id',how='left')
df3.groupby("item").sum().sort_values(by="c", ascending=False)
	나이	customer_id	quantity
item			
수건	51		5		5
생수	61		7		4
치약	40		2		4
칫솔	20		1		2
이어폰	21		2		1
헤드셋	21		2		1

#2.영희가 가장 많이 구매한 아이템은?
df3.groupby(["name","item"]).sum().loc['영희','quantity']
item
치약    4
칫솔    2
Name: quantity, dtype: int64
        
```



#### join 함수

- 내부적으로 pandas.merge 함수 사용
- 기본적으로 index를 사용하여 left join

customer1.join(orders1,how='inner') 

# 인덱스를 사용하기 때문에 인덱스를 만들고 join실시