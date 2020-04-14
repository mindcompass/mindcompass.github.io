---
title: 멀티캠퍼스 머신러닝 수업 6
excerpt: 머신러닝 수업 4월 14일 멀티캠퍼스 강의
toc: true
toc_sticky: true

categories:
  - multicampus_class

tags:
- project
- chatbot
- 
- 
use_math: true
last_modified_at: 2020-04-14
---









어제 학습시켰던 CNN모형 결과  확인하기

말, 새 사진 구분하기



CNN 구현 with Tensorflow

최근에 functional 프로그래밍이 새롭게 주목받고 싶음 

자바, 파이썬의 함수형 프로그래밍? rx자바



비동기화 프로그램



**실습(01.TF2_Fashion_MNIST)**

CNN은 컬러값이 포함된 

```python
#이녀석은 3차원을 4차원으로 변경함

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

# the data is only 2D
# convolution expects height x width x color
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)

# number of classes
K=len(set(y_train))
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
# Plot loss per iteration
plt.plot(result.history['loss'], label='loss')
plt.plot(result.history['val_loss'], label='val_loss')
plt.legend()
# Plot accuracy per iteration

plt.plot(result.history['accuracy'], label='accuracy')
plt.plot(result.history['val_accuracy'], label='val_accuracy')
plt.legend()



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

sycitlearn -> grisearchcv 자동으로 여러가지 하이퍼파라미터를 변경해서 최적의 값을 알려줌

confusion Matrix

어디가 어떻게 틀렸는지 파악할 수 있음

```python
from sklearn.metrics import confusion_matrix
y_true = [2,0,2,2,0,1]
y_predict =[0,0,2,2,0,2]
confusion_matrix(y_true,y_predict)
#array([[2, 0, 0],
#       [0, 0, 1],
#       [1, 0, 2]], dtype=int64)
```

**실습2[02.TF2_CIFAR]**



**참고 : 챗봇 만들 수 있는 플랫폼**

mattermost

https://docs.mattermost.com/deployment/bots.html



학습데이터

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

# number of classes
K=len(set(y_train))
print("number of classes:",K)

print(x_train[0])
print(x_train[0].shape)

# Build the model using the functional API
#input
#Con2D
#Con2D
#Con2D
#Flatten
#Dense(relu)
#Dense(softmax)
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

# Plot loss per iteration
plt.plot(result.history['loss'], label='loss')
plt.plot(result.history['val_loss'], label='val_loss')
plt.legend()

# Plot accuracy per iteration
plt.plot(result.history['accuracy'], label='accuracy')
plt.plot(result.history['val_accuracy'], label='val_accuracy')
plt.legend()

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

'The CIFAR-100 dataset' 안내



<점심먹고>

data augmentation

직접 사진을 수정하지 않고, 컴퓨터 메모리상에서 이미지를 수정해서 학습하면 스토리지 공간을 절약할 수 있음

**[실습 ]03.TF2_CIFAR_Improved**

```python
result = model.fit(x_train, y_train, 
    batch_size=32, epochs=3, #epochs=50
    verbose=1,
    validation_data=(x_test, y_test)
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



train_generator = data_generator.flow(x_train, y_train, batch_size) #flow에서 yeid작업이 일어남
steps_per_epochs = x_train.shape[0] // batch_size
result = model.fit_generator(train_generator,
                            validation_data=(x_test, y_test),
                            steps_per_epoch=steps_per_epochs,
                            epochs=10)
```

## 3. 자동으로 하이퍼파라미터 조정[GridSearchCV]

[실습] iris_Keras_GridSearchCV**

아이리스 데이터를 바탕으로 GridSearchCV을 활용해서 최적의 하이퍼파라미터를 찾기

[Wrappers for the Scikit-Learn API](https://keras.io/scikit-learn-api/)

기존의 사이킷런의 라이브러리를 keras에서 사용할 수 있도록 wrapping함



```python
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

```

```python
#함수화로 변경

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def iris_model(activation = 'relu', optimizer = 'adam', out_dim = '100') :
    i = Input(4)
    x = Dense(out_dim, activation=activation)(i)
    x = Dense(out_dim, activation=activation)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(i, x)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

```



## 4. 타이타닉 데이터분석

[실습]

innat/Kaggle-Play



타이타닉 분석 참고 페이지

[https://github.com/innat/Kaggle-Play/blob/gh-pages/Titanic%20Competition/README.md](https://github.com/innat/Kaggle-Play/blob/gh-pages/Titanic Competition/README.md)



실습 데이터

과제: Keras로 분석하기 

