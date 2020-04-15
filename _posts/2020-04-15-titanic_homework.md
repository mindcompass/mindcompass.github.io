---
title: 멀티캠퍼스_Titanic_Keras_과제
excerpt: 머신러닝 수업 과제
toc: true
toc_sticky: true

categories:
  - multicampus_class

tags:
- keras
- homework
- 
- 
use_math: true
last_modified_at: 2020-04-15
---



이번에는 Tensorflow를 함수 API형태로 모델을 정의해서 사용합니다. 

<br>

강사님이 제공해주신 코드를 최대한 활용해서 간단하게 과제를 하보겠습니다.

코랩으로 수행하겠습니다. 

<br>

## 1.데이터전처리

```python
from google.colab import drive
drive.mount('/content/gdrive/') 

%cd gdrive/'My Drive'

import numpy
import pandas
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

############################################################
# SibSp　-> one hot enconding
# One hot encoding SibSp
############################################################
def get_dummies_sibSp(df_all, df, df_test) :

    categories = set(df_all['SibSp'].unique())
    df['SibSp'] = pandas.Categorical(df['SibSp'], categories=categories)
    df_test['SibSp'] = pandas.Categorical(df_test['SibSp'], categories=categories)

    df = pandas.get_dummies(df, columns=['SibSp'])
    df_test = pandas.get_dummies(df_test, columns=['SibSp'])

    return df, df_test

############################################################
# Parch　-> one hot enconding
# One hot encoding SibSp
############################################################
def get_dummies_parch(df_all, df, df_test) :

    categories = set(df_all['Parch'].unique())
    df['Parch'] = pandas.Categorical(df['Parch'], categories=categories)
    df_test['Parch'] = pandas.Categorical(df_test['Parch'], categories=categories)

    df = pandas.get_dummies(df, columns=['Parch'])
    df_test = pandas.get_dummies(df_test, columns=['Parch'])

    return df, df_test

############################################################
# Ticket　-> one hot enconding
# One hot encoding Ticket
############################################################
def get_dummies_ticket(df_all, df, df_test) :

    ticket_values = df_all['Ticket'].value_counts()
    ticket_values = ticket_values[ticket_values > 1]
    ticket_values = pandas.Series(ticket_values.index, name='Ticket')
    categories = set(ticket_values.tolist())
    df['Ticket'] = pandas.Categorical(df['Ticket'], categories=categories)
    df_test['Ticket'] = pandas.Categorical(df_test['Ticket'], categories=categories)

    df = pandas.get_dummies(df, columns=['Ticket'])
    df_test = pandas.get_dummies(df_test, columns=['Ticket'])

    return df, df_test

############################################################
# Standardization
############################################################
def standardization(df, df_test) :

    standard = StandardScaler()
    df_std = pandas.DataFrame(standard.fit_transform(df[['Pclass', 'Fare']].values), columns=['Pclass', 'Fare'])
    df.loc[:,'Pclass'] = df_std['Pclass']
    df.loc[:,'Fare'] = df_std['Fare']

    df_test_std = pandas.DataFrame(standard.transform(df_test[['Pclass', 'Fare']].values), columns=['Pclass', 'Fare'])
    df_test.loc[:,'Pclass'] = df_test_std['Pclass']
    df_test.loc[:,'Fare'] = df_test_std['Fare']

    return df, df_test

############################################################
# prepare Data
############################################################
def prepareData() :

    ##############################
    # Data preprocessing
    # Extract necessary items
    ##############################
    # Load gender_submission.csv
    df = pandas.read_csv('./titanic/train.csv')
    df_test = pandas.read_csv('./titanic/test.csv')

    df_all = pandas.concat([df, df_test], sort=False)

    df_test_index = df_test[['PassengerId']]

    df = df[['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare']]
    df_test = df_test[['Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare']]

    ##############################
    # Data preprocessing
    # Fill or remove missing values
    ##############################
    df = df[df['Fare'] != 5].reset_index(drop=True)
    df = df[df['Fare'] != 0].reset_index(drop=True)

    ##############################
    # Data preprocessing
    # Digitize labels
    ##############################
    # Gender
    ##############################
    encoder_sex = LabelEncoder()
    df['Sex'] = encoder_sex.fit_transform(df['Sex'].values)
    df_test['Sex'] = encoder_sex.transform(df_test['Sex'].values)

    ##############################
    # Data preprocessing
    # One-Hot Encoding
    ##############################
    ##############################
    # SibSp
    ##############################
    df, df_test = get_dummies_sibSp(df_all, df, df_test)

    ##############################
    # Parch
    ##############################
    df, df_test = get_dummies_parch(df_all, df, df_test)

    ##############################
    # Ticket
    ##############################
    df, df_test = get_dummies_ticket(df_all, df, df_test)

    ##############################
    ##############################
    df, df_test = standardization(df, df_test)

    ##############################
    # Data preprocessing
    # Fill or remove missing values
    ##############################
    df.fillna({'Fare':0}, inplace=True)
    df_test.fillna({'Fare':0}, inplace=True)

    ##############################
    # Split training data and test data
    ##############################
    x = df.drop(columns='Survived')
    y = df[['Survived']]

    return x, y, df_test, df_test_index

x_train, y_train, x_test, y_test_index = prepareData()
```

<br>

## 2.분류모형 구성하기

적당히 히든층 3개로 모델을 만들어 보겠습니다.

**모델1**

<span style="color:red">**val_loss: 0.4692 /val_accuracy: 0.8457**</span>

```python
#모델1
model = Sequential()
model.add(Dense(512, input_dim=len(x_train.columns), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Epoch 20/30
#700/700 [==============================] - 0s 317us/step - loss: 0.2550 - #accuracy: 0.8943 - val_loss: 0.4692 - val_accuracy: 0.8457


result=model.fit(x_train, y_train, epochs=30, batch_size=10, validation_split=0.2)

plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

![그래프](https://i.imgur.com/91Vasbb.png)

<br>

그래프를 보니 과적합이 심한거 같아서 히든층 사이에  drop아웃을 수행하겠습니다. 

**모델2**

<span style="color:red">**val_loss: 0.3793 - val_accuracy: 0.8571**</span>

```python
model = Sequential()
model.add(Dense(512, input_dim=len(x_train.columns), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

result=model.fit(x_train, y_train, epochs=30, batch_size=10, validation_split=0.2)

#Epoch 8/30
#700/700 [==============================] - 0s 369us/step - loss: 0.2855 - #accuracy: 0.8857 - val_loss: 0.3793 - val_accuracy: 0.8571

#<생략: 앞 코드 참조>
```

![그래프2](https://i.imgur.com/1HKyjoz.png)

그래프를 봐도 약간의 과적합이 감소했고, val_accuracy도 상승했습니다. 

<br>

**모델3**

<span style="color:red">**val_loss: 0.3778 - val_accuracy: 0.8629**</span>

신경망을 더 쌓아서 좀 더 정교한 모델을 만들어 보겠습니다.

```python
model = Sequential()
model.add(Dense(512, input_dim=len(x_train.columns), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

result=model.fit(x_train, y_train, epochs=30, batch_size=10, validation_split=0.2)

#Epoch 8/30
#700/700 [==============================] - 0s 496us/step - loss: 0.3205 - accuracy: 0.8743 - val_loss: 0.3778 - val_accuracy: 0.8629

#<생략: 앞 코드 참조>
```

![그래프3](https://i.imgur.com/A3Z34DO.png)