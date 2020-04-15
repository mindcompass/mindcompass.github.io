---
title: 멀티캠퍼스 머신러닝 수업 6(MNIST,CIFAR)
excerpt: 머신러닝 수업 4월 14일 멀티캠퍼스 강의
toc: true
toc_sticky: true

categories:
  - multicampus_class

tags:
- keras
- 
- 
- 
use_math: true
last_modified_at: 2020-04-14
---



이번에는 Tensorflow를 함수 API형태로 모델을 정의해서 사용합니다. 

<br>

## 1.실습(01.TF2_Fashion_MNIST)

유의점 CNN은 3차원의 데이터를 기본적으로 분석하기 때문에 MNIST데이터에 차원을 1개 늘려서 작업합니다.

```python
#이녀석은 3차원을 4차원으로 변경함
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout 
#여기서 Input, Conv2D는 클래스
from tensorflow.keras.models import Model

# Load in the data
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape:", x_train.shape)


# the x_train data is only 2D
# convolution expects height x width x color
x_train = np.expand_dims(x_train, -1) #마지막에 차원을 하개 늘려라.
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)

# number of classes
K=len(set(y_train)) #K는 10개
print("number of classes:", K)

# Build the model using the functional API
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3,3), strides=2, activation='relu')(i) #x= func(i) 
x = Conv2D(64, (3,3), strides=2, activation='relu')(x) #x= func(i) 
x = Conv2D(128, (3,3), strides=2, activation='relu')(x) #x= func(i) 

x= Flatten()(x)
x= Dropout(0.2)(x)
x= Dense(512, activation='relu')(x)
#x= Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model= Model(i,x)

model.compile(
    loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']
)

#categorical_crossentropy 실수형으로 loss값
result = model.fit(x_train, y_train, 
    batch_size=32, epochs=15, #epochs=50
    verbose=1,
    validation_data=(x_test, y_test)
)


#Epoch 15/15
#1875/1875 [==============================] - 10s 5ms/step - loss: 0.1115 - #accuracy: 0.9567 - val_loss: 0.3718 - val_accuracy: 0.9013

# Plot loss per iteration
plt.plot(result.history['loss'], label='loss')
plt.plot(result.history['val_loss'], label='val_loss')
plt.legend()
# Plot accuracy per iteration

plt.plot(result.history['accuracy'], label='accuracy')
plt.plot(result.history['val_accuracy'], label='val_accuracy')
plt.legend()
```







```python
#confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
def plt_confusion_matrix(cm, classes, 
                          normalize=False, 
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  '''
  This function prints and plots the confusion matrix.
  Normalization can be appled by setting `normalize=True`.
  '''
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('normalized confusion matrix')
  else:
    print('Consusion matrix, without normalization')
  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
  
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plt_confusion_matrix(cm, list(range(10)))
```



![confusion_matrix](https://i.imgur.com/1kwOOnK.png)







```python
# Label mapping

labels='''T-shirt/top
Trouser/pants
Pullover_shirt
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle_boot'''.split()
print(labels)

# Show some misclassified examples
misclassified_idx = np.where(p_test !=y_test)[0]
print(misclassified_idx)

temp_idx = np.random.choice(misclassified_idx)
print(temp_idx)

plt.imshow(x_test[temp_idx].reshape(28,28), cmap='gray')
plt.title("True label: %s, Predicted: %s" % (labels[y_test[temp_idx]], labels[p_test[temp_idx]]) )
```

![Imgur](https://i.imgur.com/vHGCOjS.png)



**참고.confusionmatrix**

어디가 어떻게 틀렸는지 파악할 수 있음

```python
from sklearn.metrics import confusion_matrix
y_true = [2,0,2,2,0,1]
y_predict =[0,0,2,2,0,2]
confusion_matrix(y_true,y_predict)
#array([[2, 0, 0],
#       [0, 0, 1],
#       [1, 0, 2]], dtype=int64) 이게 무슨 의미인지 알아맞혀 보세요?
```



<br>

## 2. 실습2[02.TF2_CIFAR]

위와 같은 방식으로 CIFAR10데이터를 분석함

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model

# Load in the data
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()
print("x_train.shape:", x_train.shape)
print("x_train.shape[0]:", x_train.shape[0])
print("x_train.shape[0]/32:", x_train.shape[0]/32)
print("y_train.shape:", y_train.shape)

#x_train.shape: (50000, 32, 32, 3)
#x_train.shape[0]: 50000
#x_train.shape[0]/32: 1562.5
#y_train.shape: (50000,)


# number of classes
K=len(set(y_train))
print("number of classes:",K)

print(x_train[0].shape) #3차원이여서 별도의 변경이 필요 없음

# Build the model using the functional API
i= Input(shape=x_train[0].shape)
x= Conv2D(32,(3,3), strides=2, activation='relu')(i)
x= Conv2D(64,(3,3), strides=2, activation='relu')(x)
x = Conv2D(128, (3,3), strides=2, activation='relu')(x)

x= Flatten()(x)
x= Dropout(0.5)(x)
x= Dense(1024, activation='relu')(x)
x= Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)
model= Model(i,x)
model.summary()

# Compile and fit
# Note: make sure you are using hte GPU for this!
model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

result = model.fit(x_train, y_train, 
    batch_size=32, epochs=15, #epochs=50
    verbose=1,
    validation_data=(x_test, y_test)
)

#Epoch 15/15
#50000/50000 [==============================] - 25s 506us/sample - loss: 0.6588 #- accuracy: 0.7654 - val_loss: 0.8159 - val_accuracy: 0.7177


# Plot loss per iteration
plt.plot(result.history['loss'], label='loss')
plt.plot(result.history['val_loss'], label='val_loss')
plt.legend()

# Plot accuracy per iteration
plt.plot(result.history['accuracy'], label='accuracy')
plt.plot(result.history['val_accuracy'], label='val_accuracy')
plt.legend()
```



```python
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes, 
                          normalize=False, 
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be appled by setting `normalize=True`.
  """
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("normalized confusion matrix")
  else:
    print('Consusion matrix, without normalization')
  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
  
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))

```



```python
# Label mapping
labels='''airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck'''.split()
print(labels)

# Show some misclassified examples
misclassified_idx = np.where(p_test !=y_test)[0]
print(misclassified_idx)

temp_idx = np.random.choice(misclassified_idx)
print(temp_idx)

temp_idx = np.random.choice(misclassified_idx)
print(temp_idx)
plt.imshow(x_test[temp_idx])
plt.title("True label: %s, Predicted: %s" % (labels[y_test[temp_idx]], labels[p_test[temp_idx]]) )
```



![img_horse](https://i.imgur.com/FcU0J2q.png)



<br>

## 3. 실습3[TF2_CIFAR_Improved]_Data Augmentation

위 실습2[02.TF2_CIFAR]에서 수행했던 방식에 <span style="color:red">**ImageDataGenerator**</span>를 수행하여 약간의 변형된 이미지의 갯수를 증가시킨 뒤에 학습하여 모델의 정확도를 향상시키는 방법입니다.

 주로 학습 데이터가 부족할 때 기존의 이미지를 회전이동, 평행이동, 밝기조절 등을 하여 이미지 수를 증폭시키는 방법인데요. CIFAR데이터에 적용시켜보았습니다. 

![Data Augmentation](https://i.imgur.com/sKioga1.png)

이미지를 물리적으로 복사하지 않고도, 자동적으로 메모리상에서 복사하여 사용하고, 작업이 끝나면 사라집니다. 

실습2에서 15번 epochs에 val_accuracy는  0.7177이였는데, 해당 방법을 적용했을 때는  val_accuracy가 val_accuracy: 0.7003로 오히려 떨어졌습니다. 

아마 데이터가 충분해서 전이학습의 장점을 충분히 못살린거 같습니다. 아니면 파라미터를 제대로 변형시키지 못한 것 같기도 합니다. 



```python
#<생략: 위 코드 참조>

# Build the model using the functional API

i= Input(shape=x_train[0].shape)
x= Conv2D(32,(3,3), strides=2, activation='relu')(i)
x= Conv2D(64,(3,3), strides=2, activation='relu')(x)
x = Conv2D(128, (3,3), strides=2, activation='relu')(x)

x= Flatten()(x)
x= Dropout(0.5)(x)
x= Dense(1024, activation='relu')(x)
x= Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model= Model(i,x)
model.summary()

# Compile and fit
# Note: make sure you are using hte GPU for this!
model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Fit with data augmentation
# Node: if you run this AFTER calling the previous model.fit(), it will CONTINUE training where it left off
# data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
batch_size= 32
data_generator=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
print(x_train.shape[0])
print(x_train.shape[0]//batch_size)


train_generator = data_generator.flow(x_train, y_train, batch_size) #flow 함수에서 yeild작업이 일어남
steps_per_epochs = x_train.shape[0] // batch_size
result = model.fit_generator(train_generator,
                            validation_data=(x_test, y_test),
                            steps_per_epoch=steps_per_epochs,
                            epochs=15)


#<생략: 위 코드 참조>
```



**추가설명**

최근에 functional 프로그래밍이 새롭게 주목받고 있습니다. 

파이썬, node.js이 대표적인 Functional Programming이며, 최근 OOP기반이었던 자바가 최근 Functional, Reactive 등의 프로그래밍 방법론을 감이한 rxjava가 도입되고 있습니다. 

데이터셋 The CIFAR-100 dataset 안내

**참고 : 챗봇 만들 수 있는 플랫폼**

mattermost : https://docs.mattermost.com/deployment/bots.html



<br>

## 4. 자동으로 하이퍼파라미터 조정[GridSearchCV]_실습4[iris_Keras_GridSearchCV]

아이리스 데이터를 바탕으로 GridSearchCV을 활용해서 최적의 하이퍼파라미터를 찾는 GridSearchCV

원래  scikit-learn의 API인데 Keras에서 사용하기 위해 별도의 wrapping을 수행해야 합니다. 

[Wrappers for the Scikit-Learn API](https://keras.io/scikit-learn-api/)

편의상 조건을 간단히 아래와 같이 설정했습니다. 

activation=["relu","sigmoid"]

optimizer=["adam","adagrad"]

out_dim= [100,200]

nb_epoch =[10,25]

batch_size =[5,10]



결과는{'activation': 'relu', 'batch_size': 5, 'nb_epoch': 10, 'optimizer': 'adagrad', #'out_dim': 200} 일 때 최적이고  ACC= 0.8857142925262451으로 나타났습니다. 아무래도 epoch 10,25가 얼마 차이가 안나서 이런 결과가 나온거 같습니다. 

```python
import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# from tensorflow.keras.layers import Input,Dense 함수버전
# from tensorflow.keras.models import Model 함수버전

from keras.utils import np_utils
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier

iris = datasets.load_iris()
x = preprocessing.scale(iris.data)
y = np_utils.to_categorical(iris.target)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size  = 0.7)
num_classes = y_test.shape[1]


def iris_model(activation='relu', optimizer='adam', out_dim='100'):
    model = Sequential()
    model.add(Dense(out_dim, input_dim=4, activation=activation))
    model.add(Dense(out_dim, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

#train
activation=["relu","sigmoid"]
optimizer=["adam","adagrad"]
out_dim= [100,200]
nb_epoch =[10,25]
batch_size =[5,10]

model =KerasClassifier(build_fn=iris_model, verbose=0)
param_grid = dict(activation= activation,
                 optimizer=optimizer,
                 out_dim=out_dim,
                  nb_epoch= nb_epoch,
                  batch_size=batch_size
                 )

grid = GridSearchCV(estimator=model, param_grid=param_grid)

grid_result= grid.fit(x_train, y_train)

print(grid_result.best_score_)
print(grid_result.best_params_)

#0.8857142925262451
#{'activation': 'relu', 'batch_size': 5, 'nb_epoch': 10, 'optimizer': 'adagrad', #'out_dim': 200}
#단, 해당 방법을 사용할 경우, x_test, y_test데이터가 포함되어 검증되지 않았기 때문에 과적합되었을 수 있습니다. 



#최적의 솔루션을 선택했으면 해당 모델로 세팅해줍니다. 
new_model = iris_model(activation=grid_result.best_params_['activation'],
                      optimizer=grid_result.best_params_['optimizer'],
                      out_dim= grid_result.best_params_['out_dim']
                      )
```

<br>

**함수형API로 변경**

위에는 Keras를 직접 import해서 모델을 구성했다면, 이번에는 tensorflow안에 있는 keras를 사용하여 동일한 내용을 함수형 API로 표현해보겠습니다.  

```python
#함수화로 변경

import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential 
#from keras.layers.core import Dense, Activation

from tensorflow.keras.layers import Input,Dense 
from tensorflow.keras.models import Model 

from keras.utils import np_utils
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier

iris = datasets.load_iris()
x = preprocessing.scale(iris.data)
y = np_utils.to_categorical(iris.target)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size  = 0.7)
num_classes = y_test.shape[1]

def iris_model(activation = 'relu', optimizer = 'adam', out_dim = '100') :
    i = Input(4)
    x = Dense(out_dim, activation=activation)(i)
    x = Dense(out_dim, activation=activation)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(i, x)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

#<생략: 위 코드 참조>
```



<br>

## 5. 실습5_타이타닉 데이터분석

강사님이 과제로 내주신 내용인데, 오늘 한번 쭉 정리해주셨습니다. 

캐글에 있는 타이타닉 데이터를 사용했고, 이 데이터의 특징은 test데이터의 ['Survived']변수가 빠져 있습니다. 그래서 아래서 다루는 부분은  **train데이터의 일부를 쪼개서 validation set**으로 활용한 accuracy입니다. 



[innat/Kaggle-Play](https://github.com/innat/Kaggle-Play)

타이타닉 분석 참고 페이지(세부적으로 분석과정을 설명하고 있는 페이지)

[https://github.com/innat/Kaggle-Play/blob/gh-pages/Titanic%20Competition/README.md](https://github.com/innat/Kaggle-Play/blob/gh-pages/Titanic Competition/README.md)

```python
import numpy as np
import pandas as pd
df= pd.read_csv('./titanic/train.csv') #각자의 경로를 지정하세요.

df.head()
df.columns
#Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],dtype='object')
        
df=df[['Survived','Pclass','Sex', 'Fare']] #간편하게 계산하기 위해 NULL이 없은 형태의 열만 선택함

#성별의 데이터 형태가 male, female이라는 string행태로 되어 있어, 0,1의 factor형태로 변경해주기 위해서 수행합니다. 
from sklearn.preprocessing import LabelEncoder
encoder_gender =LabelEncoder()
df['Sex'] = encoder_gender.fit_transform(df['Sex'].values)
df.head()

#	Survived	Pclass	Sex	Fare
#	0	0	3	1	7.2500
#	1	1	1	0	71.2833
#	2	1	3	0	7.9250

#학습효과를 높이기 위해서 'Pclass','Fare'변수를 평균을 0,표준편차1인 수치로 표준화합니다.  

from sklearn.preprocessing import StandardScaler
standard = StandardScaler()
df_std = pd.DataFrame(
    standard.fit_transform(df[['Pclass','Fare']]),
    columns=['Pclass','Fare'])
df['Pclass']= df_std['Pclass']
df['Fare']= df_std['Fare']
df.head()

#	Survived	Pclass	Sex	Fare
#	0	0	0.827377	1	-0.502445
#	1	1	-1.566107	0	0.786845
#	2	1	0.827377	0	-0.488854



```

![labelEncoder](https://i.imgur.com/cRxn5Uw.png)

```python
#훈련 데이터, 정답 데이터 분리
from sklearn.model_selection import train_test_split
x= df.drop(columns='Survived')
y=df[['Survived']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                   random_state=1,
                                                   shuffle=True)
#y_train,y_test모두 데이터프레임 형태이기 때문에 softmax를 사용하기 위해 array형태로 변경해줘야 합니다. 
y_train= np.ravel(y_train)
y_test = np.ravel(y_test)


# LinearSVC로 1차 계산을 합니다. 
from sklearn.svm import LinearSVC
model = LinearSVC(random_state=1)
model.fit(x_train, y_train)
score = model.score(x_test,y_test)
score
#0.776536312849162

# 이번에는 분석 방법을 랜덤포레스트롤 변경하고, 앞에서 배웠던 GridSearchCV 방법으로 여러 옵션을 설정해서 수행하도록 하겠습니다. 

from sklearn.ensemble  import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rdc = RandomForestClassifier(random_state=1)
param_grid = {
    'criterion' :['gini', 'entropy'],
    'n_estimators' :[25, 100, 500, 1000, 2000],
    'min_samples_split' : [0.5, 2,4,10],
    'min_samples_leaf': [1,2,4,10],
    'bootstrap':[True, False]
}
grid =GridSearchCV(estimator=rdc,
                  param_grid= param_grid)
grid = grid.fit(x_train, y_train)
print(grid.best_score_)
print(grid.best_params_)

#0.8189303654092386 정확도가 약 82프로입니다. 
#{'bootstrap': True, 'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 25}
#colab GPU설정으로 30~40분 소요된거 같습니다. 
#단, 위의 svm과 다르게 train셋의 80%로만 검증이 이뤄졌기 때문에 accuracy는 과적합되었을 수 있습니다. 

model = RandomForestClassifier(n_estimators= 500,
                              criterion ='entropy',
                              min_samples_split=2,
                              min_samples_leaf=4,
                              bootstrap= True,
                              random_state=1)
model.fit(x_train, y_train)
score = model.score(x_test,y_test)
score
#0.7932960893854749
```

강사님 모델 비교 자료

![model비교](https://i.imgur.com/oVUxEZH.png)



과제:  TITANIC데이터를 KERAS를 활용하여 딥러닝으로 모델링 하기(조별정리)

<br>

**가. 라이브러리 임포트**

```python
import numpy
import pandas
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
```

<br>

**나.데이터 전처리를 함수로 수행**

```python
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

```

<br>

**다. 모델함수 설정**

```python
##############################
# Model -> 5perceptron
##############################input_dim#입력되는 열의 개수
def create_model_5dim_layer_perceptron(input_dim=234, \ 
                                       activation="relu", \
                                       optimizer="adam", \
                                       out_dim=100, \
                                       dropout=0.5):

    model = Sequential()

    # Input - Hidden1
    model.add(Dense(input_dim=input_dim, units=out_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # Hidden1 - Hidden2
    model.add(Dense(units=out_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # Hidden2 - Hidden3
    model.add(Dense(units=out_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # Hidden3 - Output
    model.add(Dense(units=1))
    model.add(Activation("sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

```

<br>

**라.딥러닝 모델 설계 강사님 풀이1_파라미터 대입**

```python
x_train, y_train, x_test, y_test_index = prepareData()

model = create_model_5dim_layer_perceptron(len(x_train.columns), \
                                           activation="relu", \
                                           optimizer="adam", \
                                           out_dim=702, \
                                           dropout=0.5)
model.summary()   

# Training
fit = model.fit(x_train, y_train, epochs=25, batch_size=16, verbose=2)

#Epoch 25/25 - 0s - loss: 0.3446 - accuracy: 0.8651

# Predict
y_test_proba = model.predict(x_test)
y_test = numpy.round(y_test_proba).astype(int)

# Combine the data frame of PassengerId and the result
df_output = pandas.concat([y_test_index, pandas.DataFrame(y_test, columns=['Survived'])], axis=1)

# Write result.csv to the current directory
df_output.to_csv('result.csv', index=False)


```

<br>

**강사님은 기존 train데이터를 모두 사용해서 accuracy를 구했기 때문에 검증 accuracy가 아닙니다. 따라서 위의 기계학습 결과(SVM,랜덤포레스트)와 같이 validation_data를 20%를 빼서 따로 검증해야 합니다.**

```python
from sklearn.model_selection import train_test_split

x_train, y_train, x_test, y_test_index = prepareData() #train데이터만 따로 사용해야 합니다. 
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2,random_state=1, shuffle=True) # 20% 데이터를 검증데이터로 빼서 학습합니다. 

model = create_model_5dim_layer_perceptron(len(x_train.columns), \
                                           activation="relu", \
                                           optimizer="adam", \
                                           out_dim=702, \
                                           dropout=0.5)
model.summary()   


fit = model.fit(x_train, y_train, epochs=25, batch_size=16, verbose=2,validation_data=(x_test, y_test))

#Epoch 13/25 
# - 0s - loss: 0.3659 - accuracy: 0.8657 - val_loss: 0.5110 - val_accuracy: 0.8114
#accuracy는 약 81%로 랜덤포레스트 방법보다 1% 떨어지는 결과였습니다. 
```





<br>

**마.딥러닝 모델 설계_강사님 풀이2_GridSearchCV**

조건이 제한적이다보니 정확도가 71%정도 밖에 나오지 않았습니다. 

```python
#train
activation=["relu","sigmoid"]
optimizer=["adam","adagrad"]
out_dim= [100,200]
nb_epoch =[10,25]
batch_size =[5,10]

model =KerasClassifier(build_fn=create_model_5dim_layer_perceptron, verbose=0)
param_grid = dict(activation= activation,
                 optimizer=optimizer,
                 out_dim=out_dim,
                  nb_epoch= nb_epoch,
                  batch_size=batch_size
                 )

grid = GridSearchCV(estimator=model, param_grid=param_grid)

grid_result= grid.fit(x_train, y_train)

print(grid_result.best_score_)
print(grid_result.best_params_)

#0.7188571453094482
#{'activation': 'relu', 'batch_size': 5, 'nb_epoch': 25, 'optimizer': 'adagrad', 'out_dim': 200}

#정확도가 떨어지기 때문에 validation_data에 대한 검증은 생략합니다. 
```

