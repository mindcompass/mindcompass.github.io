---

title: 멀티캠퍼스 머신러닝 수업 5(Keras,CNN)
excerpt: 머신러닝 수업 4월 13일 멀티캠퍼스 강의
toc: true
toc_sticky: true

categories:
  - multicampus_class

tags:
- machine learning
- Keras
- CNN
- 
use_math: true
last_modified_at: 2020-04-15
---





## 1. Keras 개요

**가.Keras개요**

Model을 이용하여 데이터 구조를 레이어로 구성 

파이썬으로 구현된 딥러닝 라이브러리 

- 가장 간단한 레이어 모델  -> Sequential 
- Sequence 모델로 원하는 레이어를 순차적으로 적재

1) 모델 생성

> from keras.models import Sequential 
>
> model = Sequential()



2) 레이어 구성

>from keras.layers import Dense 
>
>model.add(Dense(units=64, activation='relu' , input_dim=100)) 
>
>model.add(Dense(units=10, activation='softmax'))



3-1) 학습과정 조정

> model.compile(loss='categorical_crossentropy’ , 
>
> ​										optimizer= 'sgd’ ,
>
> ​										metrics=['accuracy'])



3-2) 학습과정 조정 (Optimizer 조정)

> model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))



4) 학습

> \# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API. 
>
> model.fit(x_train, y_train, epochs=5, batch_size=32)



4-1) 수동으로 배치 전달

> model.train_on_batch(x_batch, y_batch)



5) 모델 성능 평가

> loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128) 



6) 새로운 데이터 예측

> classes = model.predict(x_test, batch_size=128) 





**quiz1. 해당 KERAS 모형은 어떻게 생겼나요? param이 0인 layer는 무엇을 의미하는가?**

![모델설명](https://i.imgur.com/DeGyOiv.png)

**answer** 

Input 층 노드 1개, Output 층 노드 1개이며 히든층이 없는 퍼셉트론 모델

param이 2개인 이유 w(가중치)와 b(편차) 가 각각 1개씩 존재하기 때문에

 params이 0인 층은 해당 층에서 학습(w,b의 업데이트)이 이뤄지지 않는다는 것임 

이전에 학습했던 모델을 단순히 사용할 때는 학습이 이뤄지지 않음



**나. KERAS 전처리 함수** 

전처리(Preprocessing) 

- Tokenizer() 
  - 토큰화와 정수 인코딩(단어에 대한 인덱싱) 처리 
  - 텍스트 벡터화 
-  pad_sequence() 
  -  전체 훈련데이터에서 모든 샘플의 길이를 맞출 때 사용 (padding)

워드 임베딩(Word Embedding)

-  텍스트 내의 단어들을 밀집 벡터(Dense Vector)로 생성 
-  저차원을 갖는 실수 값들의 데이터 집합 
  - 256, 512, 1024차원의 단어 임베딩을 사용 
  -  원-핫 인코딩은 20000차원 또는 그 이상의 벡터 

- Embedding() 
  - 단어를 밀집 벡터로 만드는 역할, Embedding Layer로 만드는 역할



## 2. 실습1(01.TF1_Keras_fisrt)

간단한 1차 함수를 통해 Keras layer에서 발생하는 학습을 결과를 파악함

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

model.layers[0].get_weights()#0번째 층에 학습된 params에 접근함
# [array([[0.75000006]], dtype=float32), array([1.1666666], dtype=float32)]

w=model.layers[0].get_weights()[0][0] #기울기값
#array([0.75000006], dtype=float32)


b=model.layers[0].get_weights()[1][0] #바이어스값
#1.1666666

print("X=10, Y=", w*10+b)
print("X=20, Y=", w*20+b)

my_predict = model.predict([10,20]) #X값에 10과 20을 각각 넣었을 때 결과값은?
print(my_predict)
#[[ 8.666667][16.166668]]
```



## 3. 실습2(0.Keras_features)_Tokenizer,pad_sequences

**가. keras 토크나이즈 기능** 

Tokenizer의  fit_on_texts()로 일종의 단어 인덱스 사전을 만드는 작업

```python
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
t  = Tokenizer()
fit_text = "The earth is an awesome place live"
t.fit_on_texts([fit_text]) #fit_text을 기준으로 word_index 생성

test_text = "The earth is an great place live"
sequences = t.texts_to_sequences([test_text])[0]

print("sequences : ",sequences) # great는 단어 집합(vocabulary)에 없으므로 출력되지 않는다.
print("word_index : ",t.word_index) # 단어 집합(vocabulary) 출력

#sequences :  [1, 2, 3, 4, 6, 7]
#word_index :  {'the': 1, 'earth': 2, 'is': 3, 'an': 4, 'awesome': 5, 'place': 6, 'live': 7}
```



**나. Keras 토크나이즈 기능**2

```python
# 코드 6-3 케라스를 사용한 단어 수준의 원-핫 인코딩하기

from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 가장 빈도가 높은 1,000개의 단어만 선택하도록 Tokenizer 객체를 만듭니다.
tokenizer = Tokenizer(num_words=1000)
# 단어 인덱스를 구축합니다.
tokenizer.fit_on_texts(samples)

# 문자열을 정수 인덱스의 리스트로 변환합니다.
sequences = tokenizer.texts_to_sequences(samples)
print(sequences)
#[[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]

# 직접 원-핫 이진 벡터 표현을 얻을 수 있습니다.
# 원-핫 인코딩 외에 다른 벡터화 방법들도 제공합니다!
# one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
# print(one_hot_results)

# 계산된 단어 인덱스를 구합니다.
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#Found 9 unique tokens.
print(word_index)
#{'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5, 'dog': 6, 'ate': 7, 'my': 8, 'homework': 9}
```



**다.pad_sequences**

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]], maxlen=3, padding='pre')
# 전처리가 끝나서 각 단어에 대한 정수 인코딩이 끝났다고 가정하고, 3개의 데이터를 입력으로 합니다.
# pre --> 앞에 0, post --> 뒤에 0

# 결과
# array([[1, 2, 3],
#       [4, 5, 6],
#       [0, 7, 8]])

# padding='post'
# array([[1, 2, 3],
#       [4, 5, 6],
#       [7, 8, 0]], dtype=int32)

```



## 4. Keras의 기능

**가. 모델링**

![모델링](https://i.imgur.com/fGT3Y3d.png)

**quiz. 각 층의 prams 수는?**

prams 1단계 : 4x8(w) + 8(b) =40개

prams 2단계 : 8(w) + 8(b) =16개

<br>

**나. 컴파일**

compile() 

-  모델에 오차함수, 최적화방법, 메트릭 함수 등 지정

![Imgur](https://i.imgur.com/KriVPse.png)

- categorical_crossentropy : Target값이 원핫인코딩으로 되어 있는 상태
  - 예) [[0,0,1],[0,1,0]]
- sparse_categorical_crossentropy  : Target값이 정수로 코딩되어 있는 상태
  - 예) [1,2]

<br>

**다. 훈련(Training)** 

-  fit()

>  model.fit(X_train, y_train, epochs=10, batch_size=32) 

>  model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data(X_val, y_val)) 

>  model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2)) 



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



## 5. 실습2(03.TF1_Keras_ANN_AND)

Keras로 퍼셉트론을 모형으로 만든 뒤, AND연산 작업하기 

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

<br>

## 6. 실습3(02.TF1_Keras_Logistic_Regression)

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
#훈련시켰던 데이터를 다시 넣어서 정확도(Accuracy)를 측정함

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



<br>

##  7.실습4(04.TF1_Keras_Breast_cancer)



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
scaler.fit(X_train) #X_train의 표균과 표준편차 등의 인덱스 파악?

X_train.shape
#(455, 30)

X_train=scaler.transform(X_train)

model = Sequential()
model.add(Dense(1, input_dim=30, activation="sigmoid"))
model.summary()
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=['acc'])
model.fit(X_train, y_train, epochs=1000)

X_test = scaler.transform(X_test)

pred = model.predict(X_test)
pred
#array([[9.69512224e-01],
#       [1.00000000e+00],
#       [9.99942422e-01]...

predict01 = np.where(pred > 0.5, 1, 0)
print("=" * 30)
print("predict01")
print(predict01)

#==============================
#predict01
#[[1]
# [1] ...


predict02 = predict01.flatten()
#array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1,
#       1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0,
#       0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
#       1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1,
#       1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
#       1, 1, 1, 1])


predict03 = (predict02 == y_test)

np.sum(predict03)
len(predict03)
acc=np.sum(predict03)/len(predict03)
acc
#0.9736842105263158
```



## 8. COLAB 사용 안내

**가.URL을 통한 데이터셋 다운로드**

```python
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

import tensorflow as tf
tf.keras.utils.get_file('auto-mpg.data',url)

!head /root/.keras/datasets/auto-mpg.data

import pandas as pd

df =pd.read_csv('/root/.keras/datasets/auto-mpg.data', header=None, delim_whitespace=True
```

<br>

**나.구글 코랩에 csv파일 업로드**

```python
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

df=pd.read_csv('diabetes.csv', header=None)
```

<br>

**다.구글 코랩에 파이썬 파일 업로드**

![Imgur](https://i.imgur.com/ytzmdy0.png)

**라. 구글드라이브에 접속하기**

```python
from google.colab import drive
drive.mount('/content/gdrive/') #기본적으로 필요한 위치입니다. 

# 아이디를 등록하면 코드를 부여하는데 복사해서 아래 생기는 칸에 붙여넣고 확인을 넣어줍니다.

!pwd #기본적으로 현재 작업중인 디렉토리가 출력되며 기본적으로 /content입니다. 

%cd gdrive/'My Drive' #해당 코드를 입력하면 현재 내가 작업하는 위치를 변경할 수 있습니다.
```



##  9. CNN 구현 with Tensorflow

![cnn2](https://i.imgur.com/oV9VJrA.png)

![cnn3](https://i.imgur.com/TW1xuM5.png)

Prams1=1 x 5 x 5 x 16(필터수, 가중치) + 16(바이어스) = 416

Prams1=16개(이미지갯수) x5 x5 x36 +36 =14436



## 10.실습5(05.TF1_Keras_CNN_MNIST)

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
Y_train = to_categorical(Y_train) #one-hot인코딩
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


```

prams값(w,b)

conv2d_1(5X5X16+16=416개) 

conv2d_2(16X3X3X36+36=5220개)  이미지가 각 16개씩 들어남

dense_1(7056x128+128=903296개)

dense_1(128x10+10=1290개)

```python
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
#(5, 5, 1, 16) #필터크기,필터수
```

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
w1 = l1.get_weights()[0] #w1.shape=(5, 5, 1, 16)
plot_weight(w1)

l2 = model.get_layer('conv2d_2')
w2 = l2.get_weights()[0]
plot_weight(w2)
l2.get_weights()[0].shape


#temp_model = Model(inputs=model.get_layer('conv2d_1').input, outputs=model.get_layer('conv2d_1').output)
temp_model = Model(inputs=model.get_layer('conv2d_1').input, outputs=model.get_layer('conv2d_2').output)
output = temp_model.predict(X_test)

output.shape #(10000, 14, 14, 36)


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





## 11.실습6(06.TF1_Keras_CNN_loaddata,07.TF1_Keras_MLP_CIFAR10)

![슬라이드](https://i.imgur.com/6gZDBXq.png)

**가. 06.TF1_Keras_CNN_loaddata**

```python
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline 

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#5x5 이미지표시
plt.figure(figsize=(10,10))
for i in range(25):
    rand_num=np.random.randint(0,50000)
    cifar_img=plt.subplot(5,5,i+1)
    plt.imshow(x_train[rand_num])
    #x좌표 눈금 삭제
    plt.tick_params(labelbottom='off')
    #y좌표 눈금 삭제
    plt.tick_params(labelleft='off')
    #정답 레이블 표시
    plt.title(y_train[rand_num])


plt.show()
```

![image1](https://i.imgur.com/s4Wogkx.png)

<br>

**나. 07.TF1_Keras_MLP_CIFAR10**

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

<br>

## 12.실습7(08.TF1_Keras_MLP_CIFAR10)

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
# array([59, 62, 63], dtype=uint8)RGB칼라


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

model.fit(X_train, Y_train, batch_size=150, epochs=5, validation_split=0.2)
#Epoch 5/5
#40000/40000 [==============================] - 61s 2ms/step - loss: 0.9821 - accuracy: 0.6525 - val_loss: 1.2857 - val_accuracy: 0.5640


model.evaluate(X_test, Y_test,verbose=1)
#[1.2880641901016234, 0.5580000281333923]
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

![result](https://i.imgur.com/HfCPLAt.png)



<br>

## 13.실습8 (09.TF1_Keras_CNN_CIFAR10)_모델재사용

학습한 모형을 저장하여, 다시 불러와서 사용할 수 있도록 하는 코드를 정리합니다.

**가. 모델 저장하기**

```python
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

%matplotlib inline

num_classes = 10
im_rows = 32
im_cols = 32
in_shape = (im_rows, im_cols, 3)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Model(input, output)
model = Sequential()
model.add(Conv2D(32,(3,3), padding='same', strides=1, activation='relu', input_shape=in_shape))
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))    
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))      
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))     
model.add(Dense(num_classes))          
model.add(Activation('softmax'))

model.summary()
```

![model.summary](https://i.imgur.com/uxB67Jp.png)

 ```python
model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
)

hist = model.fit(X_train, y_train, 
    batch_size=32, epochs=4, #epochs=50
    verbose=1,
    validation_data=(X_test, y_test)
)

#Epoch 4/4
#50000/50000 [==============================] - 259s 5ms/step - loss: 0.8023 - accuracy: 0.7188 - val_loss: 0.7481 - val_accuracy: 0.7415

score = model.evaluate(X_test, y_test, verbose=1)
print('정답률=', score[1], 'loss=', score[0])
# 정답률= 0.7415000200271606 loss= 0.7481481936454772
model.save_weights('cifar10-cnn-weight.h5') #모델을 저장함

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
 ```

<br>

**나. 모델 재사용**

추후 다시 사용하실 때는 다음 코드를 사용해서 모델을 로드해서 사용하시면 됩니다.

```python
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img

num_classes = 10
im_rows = 32
im_cols = 32
in_shape = (im_rows, im_cols, 3)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#모형구성하기
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
im_size = 32 * 32 * 3

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=in_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


#학습한 모델결과 가져오기
model.load_weights('cifar10-cnn-weight.h5')
model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
)

#model.summary()

score = model.evaluate(X_test, y_test, verbose=1)
print('정답률=', score[1], 'loss=', score[0])
#10000/10000 [==============================] - 11s 1ms/step
#정답률= 0.7415000200271606 loss= 0.7481481936454772


img_pred=model.predict_classes(X_test)

plt.figure(figsize=(10,10))
for i in range(1):
    rand_num=np.random.randint(0,10000)
    cifar_img=plt.subplot(5,5,i+1)
    plt.imshow(X_test[rand_num])
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    plt.title('pred:{0},ans:{1}'.format(labels[img_pred[rand_num]], labels[np.argmax(y_test[rand_num])]))
plt.tight_layout()
plt.show()

```

![result2](https://i.imgur.com/qCmuKh3.png)

```python
#내가 가진 이미지를 등록해서 분별이 잘 되는지 확인합니다.
temp_img=load_img("horse.jpg",target_size=(32,32))

#화상을 배열로 변환
temp_img_array=img_to_array(temp_img)
temp_img_array=temp_img_array.astype('float32')/255.0
temp_img_array=temp_img_array.reshape((1,32,32,3))

img_pred=model.predict_classes(temp_img_array)
print('\npredict_classes=', img_pred)

plt.imshow(temp_img)
plt.title('pred:{}'.format(labels[img_pred[0]]))
plt.show()

```

![horse](https://i.imgur.com/WLaIbJ0.png)