---
title:  "데이터분석 실전2_2(데이터분석)"
excerpt: "kaggle의 제철소 데이터를 통한 머신러닝 수행하기"
toc: true
toc_sticky: true

categories:
  - data_analysis
tags:
- decision tree
- 로지스틱
- svm
- random forest

use_math: true
last_modified_at: 2020-03-08
---





이전에는 데이터분석을 하기위한 전처리를 주로 했습니다. 

이번에는 지난번 작성한 데이터를 바탕으로 본격적으로 여러가지 분석을 시행해보겠습니다.

## 1. 로지스틱 회귀 분석 회귀계수

먼저 우리가 종속변수 중 하나로 선정했던 'main_faults'를 종속변수로 하고, 16개를 독립변수로 하는 로지스틱 회귀분석을 실시하도록 하겠습니다.

```python
# 먼저 분석하기 전에 데이터를 train 60%,validate 20%, test 20%으로 구분하겠습니다.  
np.random.seed(20200308) # 매번 동일한 값을 얻고 싶으면 random.seed를 넣으면 됩니다. 
train, validate, test = np.split(data5.sample(frac=1), [int(.6*len(data5)), int(.8*len(data5))])

# data5.sample(frac=1)은 data5의 모든 샘플을 활용해서 랜덤하게 섞이는 것을 의미합니다. 
# 전체 데이터에서 먼저있는 60%, 그다음 80%에 있는 지점, 그리고 나머지를 각각 train, validate, test set으로 넣습니다.

train_y=train['main_faults']
train_x=train.drop(['main_faults','kind_of_faults'], axis=1)

validate_y=validate['main_faults']
validate_x=validate.drop(['main_faults','kind_of_faults'], axis=1)

test_y=test['main_faults']
test_x=test.drop(['main_faults','kind_of_faults'], axis=1)


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#로지스틱 회귀분석을 수행하기 위해 사이킷런의 선형모델과 로지스틱 회귀를 임포트합니다.

logreg = LogisticRegression()
logreg.fit(train_x, train_y)
#훈련용 데이터로 모델을 fitting합니다.

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

import statsmodels.api as sm
logit = sm.Logit(train_y,train_x) #로지스틱 회귀분석 시행
result = logit.fit()
result.summary()

Dep. Variable:	main_faults	No. Observations:	1164
Model:	Logit	Df Residuals:	1148
Method:	MLE	Df Model:	15
Date:	Sun, 08 Mar 2020	Pseudo R-squ.:	0.1081
Time:	15:34:01	Log-Likelihood:	-658.29
converged:	True	LL-Null:	-738.10
Covariance Type:	nonrobust	LLR p-value:	2.940e-26
			coef	std err	z	P>|z|	[0.025	0.975]
X_Maximum	-0.0836	0.074	-1.125	0.261	-0.229	0.062
Y_Minimum	0.0435	0.068	0.638	0.524	-0.090	0.177
Y_Perimeter	-0.3024	0.141	-2.147	0.032	-0.578	-0.026
Sum_of_Luminosity	0.3743	0.220	1.704	0.088	-0.056	0.805
Minimum_of_Luminosity	-0.5681	0.144	-3.950	0.000	-0.850	-0.286
Maximum_of_Luminosity	0.2757	0.107	2.571	0.010	0.066	0.486
Length_of_Conveyer	-0.3150	0.080	-3.962	0.000	-0.471	-0.159
TypeOfSteel_A300	0.3709	0.084	4.439	0.000	0.207	0.535
Steel_Plate_Thickness	-0.7863	0.100	-7.854	0.000	-0.983	-0.590
Edges_Index	0.0656	0.073	0.893	0.372	-0.078	0.210
Empty_Index	-0.2122	0.092	-2.300	0.021	-0.393	-0.031
Square_Index	0.1244	0.077	1.619	0.105	-0.026	0.275
Edges_X_Index	-0.0799	0.127	-0.627	0.530	-0.329	0.170
Log_X_Index	-1.9127	0.359	-5.321	0.000	-2.617	-1.208
Log_Y_Index	1.6889	0.336	5.028	0.000	1.031	2.347
Orientation_Index	-1.2800	0.278	-4.604	0.000	-1.825	-0.735

#logit = sm.Logit.from_formula('main_faults ~ X_Maximum+Y_Minimum+Y_Perimeter+Sum_of_Luminosity+Minimum_of_Luminosity+Maximum_of_Luminosity+Length_of_Conveyer+TypeOfSteel_A300+Steel_Plate_Thickness+Edges_Index+Empty_Index+Square_Index+Edges_X_Index+Log_X_Index+Log_Y_Index+Orientation_Index' ,train) 
# result = logit.fit()
# 절편을 포함시키려면 아래처럼 로지스틱 회귀분석 시행

```



## 2. 로지스틱 회귀분석 정확도, ConfusionMatrix

유의하지 않은 변수를 제외하고 모델을 작성할 수 있지만, 여기에서는 모든 변수를 활용하도록 하겠습니다.

```python
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression()
logreg.fit(train_x, train_y)

y_pred = logreg.predict(validate_x) #fitting한 모델에 validate_x 데이터를 넣고, 값을 추정했습니다.
# y_pred은 0 또는 1의 값을 값은 array로 출력됩니다.

pd.options.display.float_format = '{:.2f}'.format
pd.DataFrame(y_pred)
#보기 쉽게 array값을 데이터 프레임 행태로 변경했습니다.

	0	1
0	0.47	0.53
1	0.40	0.60
2	0.24	0.76
3	0.34	0.66
4	0.67	0.33

result1=pd.DataFrame(y_pred)
result1.to_csv('result1.csv')
# 해당 결과를 저장하고 싶으면 위와 같이 해주시면 됩니다.

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(validate_x, validate_y)))

Accuracy of logistic regression classifier on test set:  0.72
# 정확성은 73%로 나타났습니다.

# 원래 validate 자료를 바탕으로 모델을 튜닝하여 성능을 개선할 수 있지만, 편의상 해당 과정은 생략하도록 하겠습니ㅏ.

y_pred2 = logreg.predict(test_x)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(test_x, test_y)))
Accuracy of logistic regression classifier on test set: 0.76
#테스트 데이터로 했더니 76%로 정확성이 증가합니다. 샘플에 따른 약간의 변화인거 같습니다.    

from sklearn.metrics import confusion_matrix
confusion_matrix2 = confusion_matrix(test_y, y_pred2)
print(confusion_matrix2)
[[ 59  65]
 [ 30 235]]



```

|                     | y_pred2에 의해 0으로 예측 | y_pred2에 의해 1으로 예측 |
| ------------------- | ------------------------- | ------------------------- |
| 실제(test 데이터) 0 | 59                        | 65                        |
| 실제(test 데이터) 1 | 30                        | 235                       |

 

## 3. SVM을 통한 데이터 분석

다음은 0,1이 아닌 1~7까지 구분되는 'kind_of_faults'를 구분하는 모델을 만들겠습니다. 

종속변인이 달라지기 때문에 내용을 변경합니다. 

```python
train_y=train['kind_of_faults']
train_x=train.drop(['main_faults','kind_of_faults'], axis=1)

validate_y=validate['kind_of_faults']
validate_x=validate.drop(['main_faults','kind_of_faults'], axis=1)

test_y=test['kind_of_faults']
test_x=test.drop(['main_faults','kind_of_faults'], axis=1)

from sklearn.svm import SVC
model = SVC(kernel='rbf', probability=True)
model.fit(train_x,train_y)

SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)

y_SVM = model.predict(validate_x)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(validate_x, validate_y)))
Accuracy of logistic regression classifier on test set: 0.72
    
y_SVM2 = model.predict(test_x)    
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(test_x, test_y)))
Accuracy of logistic regression classifier on test set: 0.72
    
## validate, test 셋 모두 72%로 나타났습니다. 

from sklearn.metrics import confusion_matrix
confusion_matrix3 = confusion_matrix(test_y, y_SVM2)
print(confusion_matrix3)

[[ 9  1  0  0  0  1 15]
 [ 0 38  0  0  0  4  3]
 [ 0  0 78  0  0  3  5]
 [ 0  0  0 14  0  1  2]
 [ 1  0  0  0  4  0  6]
 [ 0  0  0  0  0 51 29]
 [ 6  3  0  2  0 25 88]]


# accuracy 및 각 요소별 precision, recall, f1-score을 구하려면 아래와 같이 하면 됩니다.
from sklearn.metrics import classification_report
print(classification_report(test_y, y_SVM2))

 		precision    recall  f1-score   support

           1       0.56      0.35      0.43        26
           2       0.90      0.84      0.87        45
           3       1.00      0.91      0.95        86
           4       0.88      0.82      0.85        17
           5       1.00      0.36      0.53        11
           6       0.60      0.64      0.62        80
           7       0.59      0.71      0.65       124

    accuracy                           0.72       389
   macro avg       0.79      0.66      0.70       389
weighted avg       0.74      0.72      0.73       389

```



## 4.  Random Forest을 통한 데이터 분석

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(train_x, train_y)

y_RF = rf.predict(validate_x)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(rf.score(validate_x, validate_y)))
Accuracy of logistic regression classifier on test set: 0.79

y_RF2 = rf.predict(test_x)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(rf.score(test_x, test_y)))
Accuracy of logistic regression classifier on test set: 0.78
    
confusion_matrix4 = confusion_matrix(test_y, y_RF2 )
print(confusion_matrix4)
[[ 14   2   0   0   0   0  10]
 [  0  40   0   0   0   1   4]
 [  0   0  79   0   0   1   6]
 [  0   0   0  14   0   2   1]
 [  1   0   0   0   6   0   4]
 [  0   0   0   0   0  48  32]
 [  4   1   0   0   0  18 101]]


# accuracy 및 각 요소별 precision, recall, f1-score을 구하려면 아래와 같이 하면 됩니다.
     	precision    recall  f1-score   support

           1       0.74      0.54      0.62        26
           2       0.93      0.89      0.91        45
           3       1.00      0.92      0.96        86
           4       1.00      0.82      0.90        17
           5       1.00      0.55      0.71        11
           6       0.69      0.60      0.64        80
           7       0.64      0.81      0.72       124

    accuracy                           0.78       389
   macro avg       0.86      0.73      0.78       389
weighted avg       0.79      0.78      0.78       389


# svm보다 조금 더 좋은 성능을 보입니다.


```



## 5. Decision Tree을 통한 데이터 분석

```python
#마지막으로 의사결정나무 모델로 분석해보겠습니다.

from sklearn.tree import DecisionTreeClassifier
Tree=DecisionTreeClassifier(max_depth=5)
Tree.fit(train_x, train_y)

DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=5, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')

y_Tree = Tree.predict(validate_x)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(Tree.score(validate_x, validate_y)))
Accuracy of logistic regression classifier on test set: 0.62


y_Tree2 = Tree.predict(test_x)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(Tree.score(test_x, test_y)))
Accuracy of logistic regression classifier on test set: 0.64
# 정확도가 64%정도로 매우 낮습니다.


confusion_matrix5 = confusion_matrix(test_y, y_Tree2 )
print(confusion_matrix5)
[[  4   2   0   0   0   0  20]
 [  0  42   0   0   0   0   3]
 [  0   0  73   0   0   0  13]
 [  0   0   0  13   0   0   4]
 [  2   0   0   0   0   0   9]
 [  0   0   1   0   0   9  70]
 [  9   4   0   0   0   4 107]]
# 전반적으로 7번으로 쏠린 결과가 나타납니다. 5번은 정답률이 0% 입니다.

print(classification_report(test_y, y_Tree2))
  		 precision    recall  f1-score   support

           1       0.27      0.15      0.20        26
           2       0.88      0.93      0.90        45
           3       0.99      0.85      0.91        86
           4       1.00      0.76      0.87        17
           5       0.00      0.00      0.00        11
           6       0.69      0.11      0.19        80
           7       0.47      0.86      0.61       124

    accuracy                           0.64       389
   macro avg       0.61      0.53      0.53       389
weighted avg       0.67      0.64      0.59       389


```



## 6. 군집분석 결과를 통한 모형의 성능 향상
모델의 성능이 가장 좋았던 랜덤포레스트에 군집분석 결과를 추가해서 모델의 성능을 향상시켜보겠습니다. 

```python
data6=data5 #기존 사용했던 data5를 군집분석 결과를 추가해서 새로운 열을 만들려고 합니다.

from sklearn.cluster import KMeans
#클러스터의 개수 지정(n개)
num_clusters = 4 #임의로 4개로 선정했습니다. 
km = KMeans(n_clusters=num_clusters)
km.fit(data6)

data6['cluster']=km.labels_
#마지막 행에 cluster가 추가된 것을 보실 수 있습니다. 

	main_faults	kind_of_faults	cluster
0	1		1		2
1	1		1		2
2	1		1		2
3	1		1		2
4	1		1		2

# 새로운 열(군집분석 결과)이 추가되었기 때문에 새롭게 데이터 set를 분류해줍니다. 
np.random.seed(20200308)
train, validate, test = np.split(data6.sample(frac=1), [int(.6*len(data6)), int(.8*len(data6))])

train_y=train['kind_of_faults']
train_x=train.drop(['main_faults','kind_of_faults'], axis=1)

validate_y=validate['kind_of_faults']
validate_x=validate.drop(['main_faults','kind_of_faults'], axis=1)

test_y=test['kind_of_faults']
test_x=test.drop(['main_faults','kind_of_faults'], axis=1)
#이 부분은 위 과정과 동일합니다. 

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(train_x, train_y)

y_RF = rf.predict(validate_x)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(rf.score(validate_x, validate_y)))
Accuracy of logistic regression classifier on test set: 0.86
    
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(rf.score(test_x, test_y)))    
Accuracy of logistic regression classifier on test set: 0.83 
# 이전 정확도가 79%였던 것에 비해서 4% 향상된 결과가 나타났습니다. 

y_RF2 = rf.predict(test_x)

confusion_matrix6 = confusion_matrix(test_y, y_RF2 )
print(confusion_matrix6)

[[ 23   2   0   0   0   0   1]
 [  1  44   0   0   0   0   0]
 [  3   2  81   0   0   0   0]
 [  0   0   0  14   0   2   1]
 [  0   0   0   0   6   0   5]
 [  0   0   0   0   0  48  32]
 [  0   0   0   0   0  17 107]]

print(classification_report(test_y, y_RF2))

  	 	precision    recall  f1-score   support

           1       0.85      0.88      0.87        26
           2       0.92      0.98      0.95        45
           3       1.00      0.94      0.97        86
           4       1.00      0.82      0.90        17
           5       1.00      0.55      0.71        11
           6       0.72      0.60      0.65        80
           7       0.73      0.86      0.79       124

    accuracy                           0.83       389
   macro avg       0.89      0.81      0.83       389
weighted avg       0.84      0.83      0.83       389
```

군집분석을 한 결과를 추가한 것과 하기 전의 데이터를 비교 분석해보겠습니다.

```python
#군집분석 '수행 전'의 랜덤포레스트 분석 결과
		precision    recall  f1-score   support

           1       0.74      0.54      0.62        26
           2       0.93      0.89      0.91        45
           3       1.00      0.92      0.96        86
           4       1.00      0.82      0.90        17
           5       1.00      0.55      0.71        11
           6       0.69      0.60      0.64        80
           7       0.64      0.81      0.72       124
        
    accuracy                           0.78       389
   macro avg       0.86      0.73      0.78       389
weighted avg       0.79      0.78      0.78       389
```



