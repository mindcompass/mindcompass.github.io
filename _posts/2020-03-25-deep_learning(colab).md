---
title:  colab 활용하기
excerpt: colab으로 keras 수행하기
toc: true
toc_sticky: true

categories:
  - deep_learning
tags:
- deep_learning
- keras
- python
- colab
use_math: true
last_modified_at: 2020-03-25
---



## 1. 최신 KERAS를 사용하기 위한 작업

현재 기준으로 colab에 접속하면 tensorflow 1.15 버전이 설치되어 있고, keras의 경우 2.25 버전이 깔려있어서 tensorflow2.0으로 코딩 시에 작동이 안될 수 있습니다. 원활한 작업을 위해서 tensorflow와 keras를 지우고 다시 설치하시면 문제 없이 작업하실 수 있습니다. 

```python
!pip uninstall -y keras 
!pip uninstall -y tensorflow
!pip install tensorflow #현재 기준 2.1.0버전이 설치됩니다.
!pip install keras #현재 기준 2.3.1버전이 설치됩니다.
```



## 2. 구글드라이브 연결 및 활용

```python
from google.colab import drive
drive.mount('/content/gdrive/') #기본적으로 필요한 위치입니다. 

# 아이디를 등록하면 코드를 부여하는데 복사해서 아래 생기는 칸에 붙여넣고 확인을 넣어줍니다.

!pwd #기본적으로 현재 작업중인 디렉토리가 출력되며 기본적으로 /content입니다. 

%cd gdrive/'My Drive' #해당 코드를 입력하면 현재 내가 작업하는 위치를 변경할 수 있습니다.

#테스트를 위해서면 아래 코드를 넣으시면 됩니다. 
with open('/content/gdrive/My Drive/foo.txt', 'w') as f:
  f.write('Hello Google Drive!')
#구글드리이브 '내 드라이브'에 foo.txt파일이 생성되면 제대로 연결된 것입니다. 

```



위 내용을 활용하면 구글 '내 드라이브'에 데이터를 넣어두고 언제든지 불러올 수도 있고, 내가 colab으로 만든 모델을 언제든지 저장해서 보관할 수 있습니다. 

```python
# keras에서 모델 저장하고 불러오기

from keras.models import load_model
model.save('my_model.h5')
model =model.load('my_model.h5')

# keras에서 데이터 저장 및 불러오기
df=pd.read_csv('/content/gdrive/My Drive/my_data.csv', header=None)
df.to_csv('/content/gdrive/My Drive/my_data.csv', header=None)

```

