---

title: 머신러닝 수업 5강
excerpt: 머신러닝 수업 4월 13일 멀티캠퍼스 강의
toc: true
toc_sticky: true

categories:
  - multicampus_class

tags:
- machine learning
- 
- 
- 
use_math: true
last_modified_at: 2020-04-13
---



오늘은 5번째  수업입니다.

최종 프로젝트에 대한 안내가 있었습니다. 5명씩 3조로 구성될거 같습니다. 

 에자일 방식 개발



## 1. Keras 방식

**가.Keras개요**

Model을 이용하여 데이터 구조를 레이어로 구성 

파이썬으로 구현된 딥러닝 라이브러리 

- 가장 간단한 레이어 모델  -> Sequential 
- Sequence 모델로 원하는 레이어를 순차적으로 적재

1) 데이터 생성 

- 데이터로부터 훈련셋 검증셋, 시험셋을 생성, 포맷 변환 등 

2) 모델 구성 

- 시퀄스 모델 생성, 레이어 추가 - 케라스 함수 API 사용 가능 

3) 모델 학습과정 설정 

- 학습에 대한 설정 
- 손실 함수, 최적화 방법 정의 
- compile()

4) 모델 학습 - 훈련셋으로 모델 학습, fit() 5) 학습과정 확인 - 학습 시 훈련셋, 검증셋의 손실, 정확도 측정, 반복에 따른 추이 분석, 판단 6) 모델 평가 - 시험셋으로 모델 평가, evaluate() 7) 모델 사용 - 임의값에 대한 모델 출력, predict()





1.complie

2. fit
3. evaluate
4. predict

![모델설명](https://i.imgur.com/DeGyOiv.png)

Output Shape : Input 1개, Output 1개인 모델

param이 2개인 이유 w(가중치)와 b(편차) 

학습을 안하는 params이 존재하나? 이전에 학습했던 모델을 사용할 때는 학습이 이뤄지지 않음







**나. Keras실습(TF1_Keras_fisrt)**

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

X = np.array([1, 2, 3], dtype="float32")
Y = np.array([2, 2.5, 3.5], dtype="float32")

model = Sequential()
model.add(Dense(units=1, input_dim=1))

model.summary()

model.compile(loss='mean_squared_error',
             # optimizer='adam',
               optimizer=Adam(lr=0.1))
model.fit(X,Y,epochs=1000)

model.layers[0].get_weights()
# [array([[0.75000006]], dtype=float32), array([1.1666666], dtype=float32)]

w=model.layers[0].get_weights()[0][0]
#array([0.75000006], dtype=float32)


b=model.layers[0].get_weights()[1][0]
#1.1666666

print("X=10, Y=", w*10+b)
print("X=20, Y=", w*20+b)

my_predict = model.predict([10,20])
print(my_predict)
#[[ 8.666667][16.166668]]
```



**다. Keras실습(00.Keras_features)**

1) keras 토크나이즈 기능 

```python
from tensorflow.keras.preprocessing.text import Tokenizer
t  = Tokenizer()
fit_text = "The earth is an awesome place live"
t.fit_on_texts([fit_text])

test_text = "The earth is an great place live"
sequences = t.texts_to_sequences([test_text])[0]

print("sequences : ",sequences) # great는 단어 집합(vocabulary)에 없으므로 출력되지 않는다.
print("word_index : ",t.word_index) # 단어 집합(vocabulary) 출력

#sequences :  [1, 2, 3, 4, 6, 7]
#word_index :  {'the': 1, 'earth': 2, 'is': 3, 'an': 4, 'awesome': 5, 'place': 6, 'live': 7}
```



2) 데이터전처리

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]], maxlen=3, padding='pre')
# 전처리가 끝나서 각 단어에 대한 정수 인코딩이 끝났다고 가정하고, 3개의 데이터를 입력으로 합니다.
# pre --> 앞에 0, post --> 뒤에 0

#결과
#array([[1, 2, 3],
#       [4, 5, 6],
#       [0, 7, 8]])
```



![모델링](https://i.imgur.com/fGT3Y3d.png)

prams 1단계 : 4x8(w) + 8(b) 

prams 2단계 : 8(w) + 8(b)

![Imgur](https://i.imgur.com/KriVPse.png)

categorical_crossentropy : 결과값을 실수로 출력

sparse_categorical_crossentropy  : 결과값을 정수로 출력



훈련(Traingin) 

-  fit()

model.fit(X_train, y_train, epochs=10, batch_size=32) 

model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data(X_val, y_val)) 

model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2)) 



`verbose`는 학습 중 출력되는 문구를 설정하는 것으로, 주피터노트북(Jupyter Notebook)을 사용할 때는 `verbose=2`로 설정하여 진행 막대(progress bar)가 나오지 않도록 설정한다



평가(Evaluation), 예측(Prediction) 

- evaluate() 

  테스트 데이터에 대한 정확도 평가

- predict() 

   임의의 입력에 대해 예측

  

  모델 저장(Save), 불러오기(Load) 

- save() 
  인공 신경망 모델 저장 

- load_model() 
  저장된 모델 불러오기

  

**마. 실습(03.TF1_Keras_ANN_AND)**

Keras로 AND연산 작업하기 

```python
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X = np.array([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
], dtype="float32")

y = np.array([0, 0, 0, 1], dtype="float32")

X
y
#array([0., 0., 0., 1.], dtype=float32)

model = Sequential()
model.add(Dense(1, input_dim=2, activation="sigmoid"))
model.summary()
Model: "sequential_1"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dense_1 (Dense)              (None, 1)                 3         
#=================================================================
#Total params: 3
#Trainable params: 3
#Non-trainable params: 0
#_________________________________________________________________

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.1), metrics=['acc'])

model.fit(X, y, epochs=100)

pred = model.predict(X)
pred

#array([[3.77293077e-06],
#       [1.50083462e-02],
#       [1.50727555e-02],
#		[9.84077096e-01]], dtype=float32)

#numpy 조건문으로 작업
predict01 = np.where(pred > 0.5, 1, 0)
print("=" * 30)
print("predict01")
print(predict01)

#predict01
#[[0]
# [0]
# [0]
# [1]]


```

**바.실습(02.TF1_Keras_Logistic_Regression)**

당뇨병 관련 데이터로 로지스틱 회귀분석 수행

```python
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt('./datasets/diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]


# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

#model.summary()
#Model: "sequential_1"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dense_1 (Dense)              (None, 12)                108       
#_________________________________________________________________
#dense_2 (Dense)              (None, 8)                 104       
#_________________________________________________________________
#dense_3 (Dense)              (None, 1)                 9         
#=================================================================
#Total params: 221
#Trainable params: 221
#Non-trainable params: 0
#_________________________________________________________________
#param 개수 8x12(w)+12(b)=108 /   8x12(w)+8(b)=104



# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
#768/768 [==============================] - 0s 30us/step
#Accuracy: 77.86


# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
   
#[6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0] => 1 (expected 1)
#[1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0] => 0 (expected 0)
#[8.0, 183.0, 64.0, 0.0, 0.0, 23.3, 0.672, 32.0] => 1 (expected 1)
#[1.0, 89.0, 66.0, 23.0, 94.0, 28.1, 0.167, 21.0] => 0 (expected 0)
#[0.0, 137.0, 40.0, 35.0, 168.0, 43.1, 2.288, 33.0] => 1 (expected 1)

   


```

**사.실습(04.TF1_Keras_Breast_cancer)**

```python
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #평균0.0 표준편차 1.0
scaler.fit(X_train)

X_train=scaler.transform(X_train)

model = Sequential()
model.add(Dense(1, input_dim=30, activation="sigmoid"))
model.summary()
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=['acc'])
model.fit(X_train, y_train, epochs=1000)
X_test = scaler.transform(X_test)

predict01 = np.where(pred > 0.5, 1, 0)
print("=" * 30)
print("predict01")
print(predict01)

predict02 = predict01.flatten()
predict03 = (predict02 == y_test)

np.sum(predict03)
len(predict03)
acc=np.sum(predict03)/len(predict03)
acc
#0.5175438596491229
```



## 2. COLAB설명



##  3. CNN 구현 with Tensorflow



Prams=1 x 5 x 5 x 16(필터수, 가중치) + 16(바이어스) = 416

**나. 



![cnn2](https://i.imgur.com/oV9VJrA.png)



![cnn3](https://i.imgur.com/TW1xuM5.png)

**다. 실습(05.TF1_Keras_CNN_MNIST)**

```python
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train.shape
#(60000, 28, 28)

Y_test[0]
#7
plt.imshow(X_test[0], cmap='binary')
plt.show()

X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

X_train  =X_train/255.0 
X_test  =X_test/255.0 

#Create a model
model =Sequential()
model.add(Conv2D(filters=16,
                kernel_size=(5,5),
                 padding="same",
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36,
                kernel_size=(3,3),
                 padding="same",
                 activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()

#Model: "sequential_4"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d_7 (Conv2D)            (None, 28, 28, 16)        416       
#_________________________________________________________________
#max_pooling2d_6 (MaxPooling2 (None, 14, 14, 16)        0         
#_________________________________________________________________
#conv2d_8 (Conv2D)            (None, 14, 14, 36)        5220      
#_________________________________________________________________
#flatten_4 (Flatten)          (None, 7056)              0         
#_________________________________________________________________
#dense_4 (Dense)              (None, 128)               903296    
#_________________________________________________________________
#dense_5 (Dense)              (None, 10)                1290      
#=================================================================
#Total params: 910,222
#Trainable params: 910,222
#Non-trainable params: 0
#_________________________________________________________________

model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X_train, Y_train,
         batch_size=200, epochs=1,
         validation_split=0.2)

#Train on 48000 samples, validate on 12000 samples
#Epoch 1/1
#48000/48000 [==============================] - 19s 406us/step - loss: 0.2605 - accuracy: #0.9220 - val_loss: 0.0923 - val_accuracy: 0.9726

result =model.evaluate(X_test, Y_test)
print(result)
#[0.08135137035716325, 0.9743000268936157]

layer1 = model.get_layer('conv2d_7')
print(layer1.get_weights()[0].shape)
#(5, 5, 1, 16)
```

prams값(w,b)

conv2d_1(5X5X16+16=416개) 

conv2d_2(16X3X3X36+36=5220개)  이미지가 각 16개씩 들어남

dense_1(7056x128+128=903296개)

dense_1(128x10+10=1290개)



강사님이 시각화 함수가 포함된 작업파일(05.TF1_Keras_CNN_MNIST)을 주심

```python
def plot_weight(w):
    w_min = np.min(w)
    w_max = np.max(w)
    num_grid = math.ceil(math.sqrt(w.shape[3]))
    fix, aixs = plt.subplots(num_grid, num_grid)
    for i, ax in enumerate(aixs.flat):
        if i < w.shape[3]:
            img = w[:,:,0,i]
            ax.imshow(img, vmin=w_min, vmax=w_max)
            
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
    
l1 = model.get_layer('conv2d_1')
w1 = l1.get_weights()[0]
plot_weight(w1)

l2 = model.get_layer('conv2d_2')
w2 = l2.get_weights()[0]
plot_weight(w2)
l2.get_weights()[0].shape

def plot_output(w):
    num_grid = math.ceil(math.sqrt(w.shape[3]))
    fix, aixs = plt.subplots(num_grid, num_grid)
    
    for i, ax in enumerate(aixs.flat):
        if i < w.shape[3]:
            img = w[0,:,:,i]
            ax.imshow(img, cmap='binary')
            
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
    plot_output(output)

```

![Imgur](https://i.imgur.com/C06qVHw.png)

![Imgur](https://i.imgur.com/yGRIm4F.png)







![슬라이드](https://i.imgur.com/6gZDBXq.png)

**실습(07.TF1_Keras_MLP_CIFAR10)**

```python
#pip install opencv-python
#pip install keras
#pip install tensorflow
pip install pillow

from keras.datasets import cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline

plt.figure(figsize=(10, 10))

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
for i in range(0, 40):
    im = Image.fromarray(X_train[i])
    plt.subplot(5, 8, i + 1)
    plt.title(labels[Y_train[i][0]])
    plt.tick_params(labelbottom="off", bottom="off")
    plt.tick_params(labelleft="off", left="off")
    plt.imshow(im)

plt.show()

#X_train
X_train.shape

import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout

num_classes = 10
im_rows = 32
im_cols = 32
im_size = im_rows * im_cols * 3

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.reshape(-1, im_size).astype('float32') / 255
X_test = X_test.reshape(-1, im_size).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(im_size,)))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

hist = model.fit(X_train, y_train, batch_size=32, epochs=6,
                verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=1)
print('정답률=', score[1], 'loss=',score[0])

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save_weights('cifar10-mlp-weight.h5')

import cv2
import numpy as np

num_classes = 10
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
im_size = 32 * 32 * 3

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(im_size, )))
model.add(Dense(num_classes, activation='softmax'))

model.load_weights('cifar10-mlp-weight.h5')
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

im = cv2.imread('test-ship.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (32,32))
plt.imshow(im)
plt.show()

im = im.reshape(im_size).astype('float32') / 255

r = model.predict(np.array([im]), batch_size=32, verbose=1)
res = r[0]

for i, acc in enumerate(res):
    print(labels[i], "=", int(acc * 100))
print("===")
print("예측한 결과=", labels[res.argmax()])


```

**실습(08.TF1_Keras_MLP_CIFAR10)**

실습

```python
matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.datasets import cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_test.shape # (50000, 32, 32, 3)
#Y_train[0]

flg = plt.figure(figsize=(20,5))
for i in range(36):
#     ax = flg.add_subplot(6, 6, i+1, xticks=[], yticks=[])
    ax = flg.add_subplot(3, 12, i+1, xticks=[], yticks=[])
    ax.imshow(X_train[i])

#X_train.shape
X_train[0,0,0]

#array([6], dtype=uint8)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Model(input, output)
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=4, padding='same', strides=1, activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=4, padding='same', strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=4, padding='same', strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
)
model.evaluate(X_test, Y_test)
pred = model.predict(X_test)

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

flg = plt.figure(figsize=(20, 10))
for i, idx in enumerate(np.random.choice(X_test.shape[0], size=32)):
    ax = flg.add_subplot(4, 8, i+1, xticks=[], yticks=[])
    ax.imshow(X_test[idx])
    
    pred_idx = np.argmax(pred[idx])
    true_idx = np.argmax(Y_test[idx])
    
    ax.set_title("{}_{}".format(labels[pred_idx], labels[true_idx]), color='green' if pred_idx == true_idx else 'red')
```



**실습(09.TF1_Keras_MLP_CIFAR10)**





CNN(TF2)