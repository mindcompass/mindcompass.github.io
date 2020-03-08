---
title:  "데이터분석 실전1_4(데이터분석, 분산분석, 회귀분석)"
excerpt: "데이터를 바탕으로 분산분석과 회귀분석 실시"
toc: true
toc_sticky: true

categories:
  - data_analysis
tags:
- ANOVA
- regression
- 
- 
use_math: true
last_modified_at: 2020-03-05
---



## 1. 요일별 전체 전력 사용량 차이검정(ANOVA)

그래프만 보더라도 요일별 차이는 확연합니다. 

그냥 눈으로 보이는 차이가 얼마나 되는지 일원분산 분석을 실시해보고, 그 이유가 어떤 요일간의 차이로 나타났는지 사후검정으로 확인해보고자 합니다.

```python
#요일별 전체 전력사용량을 비교해야 하기 떄문에 total_usage를 새롭게 생성합니다.
data6['total_usage']=data6.iloc[:,1:6].sum(axis=1)
data6
	Time	NX1343	NX1440	NX1441	NX1332	NX1445	temp_aver	Day	total_usage
0	2017-07-01	3.1	5.6	151.5	2.9	24.8	24.1	Sat	188.0
1	2017-07-02	3.1	5.6	149.0	3.3	14.1	23.6	Sun	175.1
2	2017-07-03	23.2	29.7	150.3	9.1	44.1	23.2	Mon	256.4
3	2017-07-04	23.4	23.8	150.5	2.1	48.5	24.8	Tue	248.3
4	2017-07-05	30.3	23.1	165.0	13.1	51.0	26.0	Wed	282.6

import statsmodels.api as sm
from statsmodels.formula.api import ols

results = ols('total_usage~Day',data=data6).fit()

results.summary()
Dep. Variable:	total_usage	R-squared:	0.141
Model:	OLS	Adj. R-squared:	0.126
Method:	Least Squares	F-statistic:	9.777
Date:	Thu, 05 Mar 2020	Prob (F-statistic):	5.52e-10
Time:	17:49:03	Log-Likelihood:	-2084.4
No. Observations:	365	AIC:	4183.
Df Residuals:	358	BIC:	4210.
Df Model:	6		
Covariance Type:	nonrobust
 #F검정 결과 f통계량은 9.777이며, 유의수준은 5.52e-10으로 유의수준 .05에서 유의한 것으로 나타났습니다. 
#사실 더 중요한 것이 어떤 요일에서 가장 큰 차이가 발생했는지 이기 때문에 사후검정을 실시합니다. 

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
mc=MultiComparison(data6['total_usage'],data6['Day'])

print(mc.tukeyhsd())
Multiple Comparison of Means - Tukey HSD, FWER=0.05  
=======================================================
group1 group2 meandiff p-adj    lower    upper   reject
-------------------------------------------------------
   Fri    Mon -56.6098 0.0021  -99.5234 -13.6962   True
   Fri    Sat -61.4365  0.001 -104.1472 -18.7258   True
   Fri    Sun -70.3958  0.001 -113.3094 -27.4822   True
   Fri    Thu  -2.8987    0.9  -45.8123  40.0149  False
   Fri    Tue    -7.03    0.9  -49.9436  35.8836  False
   Fri    Wed  -4.6975    0.9  -47.6111  38.2161  False
   Mon    Sat  -4.8268    0.9  -47.5375  37.8839  False
   Mon    Sun -13.7861    0.9  -56.6997  29.1275  False
   Mon    Thu  53.7111 0.0044   10.7975  96.6247   True
   Mon    Tue  49.5798 0.0121    6.6662  92.4934   True
   Mon    Wed  51.9123 0.0069    8.9987  94.8259   True
   Sat    Sun  -8.9593    0.9    -51.67  33.7514  False
   Sat    Thu  58.5378 0.0012   15.8271 101.2485   True
   Sat    Tue  54.4066 0.0035   11.6959  97.1173   True
   Sat    Wed   56.739 0.0019   14.0283  99.4497   True
   Sun    Thu  67.4971  0.001   24.5835 110.4107   True
   Sun    Tue  63.3659  0.001   20.4523 106.2795   True
   Sun    Wed  65.6983  0.001   22.7847 108.6119   True
   Thu    Tue  -4.1313    0.9  -47.0449  38.7823  False
   Thu    Wed  -1.7988    0.9  -44.7124  41.1148  False
   Tue    Wed   2.3325    0.9  -40.5811  45.2461  False
-------------------------------------------------------

```

금요일과 일요일이 평균의 차이가 가장 크게 나왔고, 전반적으로 일요일이 다른 요일(화요일, 수요일, 목요일, 금요일) 에 비해서 전력사용량이 적게 나타나서 이런 차이가 나타난 것으로 보입니다. 



## 2. 회귀분석 실시

이번에는 회귀분석을 실시합니다. 

종속변수를 평균기온으로 하고,  영향을 미친다고 가정한 요인을 5개 지역의 전력사용량 변인으로 설정하려고 합니다. 

사실 논리적으로 5개 지역의 전력사용량으로 기온을 예측하는 모형을 만든다는게 논리적으로 말이 안됩니다. 그냥 연습한다는 생각으로 해당 모형을 검증해보겠습니다.

```python
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

model = smf.ols(formula = 'temp_aver ~ NX1343 + NX1440 + NX1441 + NX1332 + NX1445', data = data6)
result = model.fit()
result.summary()

Dep. Variable:	temp_aver	R-squared:	0.486
Model:	OLS	Adj. R-squared:	0.479
Method:	Least Squares	F-statistic:	67.90
Date:	Thu, 05 Mar 2020	Prob (F-statistic):	8.11e-50
Time:	18:34:15	Log-Likelihood:	-1266.6
No. Observations:	365	AIC:	2545.
Df Residuals:	359	BIC:	2569.
Df Model:	5		
Covariance Type:	nonrobust	

# 회귀모형의 타당도를 의미하는 f통계량은 67.90이며, 
# 유의확률은 8.11e-50으로 유의수준.01에서 유의하게 나타났습니다. 
# 해당 5개의 변수로 종속변수인 기온을 약 48~49%로 설명할 수 있는 것으로 나타났습니다. 

		coef	std err	 t	P>|t|	[0.025	0.975]
Intercept	11.3003	1.133	9.970	0.000	9.071	13.529
NX1343		0.9945	0.073	13.564	0.000	0.850	1.139
NX1440		-0.2926	0.048	-6.100	0.000	-0.387	-0.198
NX1441		0.0015	0.009	0.163	0.870	-0.017	0.020
NX1332		-0.0108	0.062	-0.174	0.862	-0.133	0.111
NX1445		-0.0994	0.037	-2.675	0.008	-0.172	-0.026
Omnibus:	9.259	Durbin-Watson:	0.682
Prob(Omnibus):	0.010	Jarque-Bera (JB):	15.325
Skew:	0.108	Prob(JB):	0.000470
Kurtosis:	3.980	Cond. No.	332.

# 종속변수에 영향을 미치는 변수는 NX1343,NX1440,NX1445으로 각각 유의수준 0.01에서 유의하게 나타났습니다. 
# NX1441,NX1332 영향을 미치지 않는 것으로 나타났습니다.
```



