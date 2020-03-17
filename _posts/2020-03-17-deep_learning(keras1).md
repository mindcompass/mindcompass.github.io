---
title:  딥러닝 KERAS 수업 정리
excerpt: "03월 07일 KERAS 수업내용 정리"
toc: true
toc_sticky: true

categories:
  - deep_learning
tags:
- deep_learning
- keras
- python
- 
use_math: true
last_modified_at: 2020-03-17
---



## 1. 운영체계와 메모리

시스템프로그래머는 기본적으로 OS에 관심이 많음

1.범용운영체제(windows, linux) -32bit, 64bit---> 32bit는 4G만큼 사용 가능함

0x00000000 ~ 0xFFFFFFFF(32bit ==4G)

--------------------

JVM(알아서 메모리를 관리함) --> 다른 하드웨어에서 동일 하드웨어처럼 동작함

Python(C계열이지만, 자바보다 문법이 간결하고,   JVM의 속성을 갖고 있음 )

단점) 의존성--> Virtural Env, Anaconda(버전을 맞춰줘서 잘 기능할 수 있도록 해줌)



2. 클라우드 환경

요즘은 클라우드가 기본/ 궁극적으로 모든 대형회사가 클라우드 환경에서 작업하는 것을 목표로 하고 있음

CPU(고속) : Inter(Xeon) -->  X86 -> 빠른 

CPU(이동가능형, 저속) : ARM Server 



모바일에서 마이크로칩에 대해서 알고 싶다면,  ARM사이트에서 아키텍처된 기능을 미리 파악해서 기능을 미리 만들어서 파악할 수 있음

A시리즈 -> 고속의 비싼 CPU 

[Helio](https://namu.wiki/w/미디어텍 Helio) ->  중저가 CPU (Pelion은 Iot플랫폼)



Create를 눌러서 

![image-20200317103653930](.assets\image-20200317103653930.png)





```python

conda activate #  conda 가상환경 실행

conda activate [가상환경 이름] # 특정 가상환경 실행

conda install [모듈(프로그램)]  # 가상환경에 프로그램 설치

conda list #설치된 라이브러리를 확인함

conda list | find "[모듈(프로그램)]" #해당 모듈이 설치되어 있는지  확인함

conda uninstall [프로그램] #프로그램 지우기

```

모듈은 \student\Anaconda3\envs 안에 들어있음

ex) C:\Users\student\Anaconda3\envs\keras\Lib\site-packages 안에 keras를 위한 모듈이 다 들어있음. 안되면 이런걸 다 지워버리기





## 2. 인공지능 관련 사례

1) Amazon Polly 

Amazon Polly 인공지능을 활용해서 목소리를 흉내냄(서현 목소리)

목소리를 흉내내는 알고리즘은 공개되어 있기 때문에 서비스로 활용할 수 있음

![image-20200317105402553](D:/myblog/_posts/일반적으로 응용프로그래머.assets/image-20200317105402553.png)





2) Elasticsearch

엘라스틱서치 -> 키바나라는 것을 붙여서 사용함(실시간으로 로그 데이터 등을 대시보드 형태로 분석해줌)

![img](http://buzzclasswp.iwinv.net/wp/wp-content/uploads/2018/07/thumb-1025790345_AzBVofMr_9987fd6c13d4b3952dd96c2d3e6c5a0fccb2846b_900x472.png)



3) Splunk

스플렁크 -> 엘라스틱서치가 유사한 프로그램

간단한 쇼핑몰 같은 작은 사이트, 프로그램을 정말 필요한 기능만 가진 작은 툴을 이용해서 쉽게 제작하는 것

이걸 이용하면 사이트 개발과 운영이 분리됨, 서버 운영 몰라도 개발

그래서 운영법을 몰라도 하기 좋음

![In the Enterprise screenshot](https://www.splunk.com/content/dam/splunk2/images/sliders/homepage/september-2016/hpproducts-screenshot-enterprise.png)



4) [Yolo ](https://pjreddie.com/darknet/yolo/)

실시간으로 사물을 인식해서 알려주는 시스템

높은 사양이 필요함



5) 국내 주요한 AI 관련 회사(NLP가 주요 사업임)

[와이즈넛]('http://www.wisenut.com/')

[솔트룩스](http://www.saltlux.com/index.do)



##  3. 교육생 취업을 위한 팁 

- 현재 대부분의 인공지능의 과제는 1)분류할 수 있는가? 2)회귀를 그을 수 있는가? (예측) 임

- 대규모 자료 300MB이상의 자료를 한번에 다뤄야 한다면 판다스로 하기 어렵고, 하둡과 SPEARK를 활용해야 함

- 현재는 채권회사, 증권회사에서 주로 하둡과 스파크 기술을 많이 필요함

- 교육생 입장에서는 적당한 자료 크기를 활용해서 마이크로서비스를 웹으로 만드는 것이 유리함

- 아파치 로그를 분석하는 것만으로도 많은 분석을 할 수 있음

- 인공지능 분야는 매우 넓기 때문에 나중에 취업을 한 분야를 염두해두고 공부해야 함





## 4. ANACONDA 실습

```python
conda activate #  conda 가상환경 실행

conda activate [가상환경 이름] # 특정 가상환경 실행

conda install [모듈(프로그램)]  # 가상환경에 프로그램 설치

conda list #설치된 라이브러리를 확인함

conda list | find "[모듈(프로그램)]" #해당 모듈이 설치되어 있는지  확인함

conda uninstall [프로그램] #프로그램 지우기

```

모듈은 \student\Anaconda3\envs 안에 들어있음

ex) C:\Users\student\Anaconda3\envs\keras\Lib\site-packages 안에 keras를 위한 모듈이 다 들어있음. 안되면 이런걸 다 지워버리기





## 5. COLAB 실습

[자료 링크](https://github.com/HaSense/Keras)



https://www.tensorflow.org/

![image-20200317141725445](.assets/image-20200317141725445.png)



![image-20200317141329775](.assets/image-20200317141329775.png)

시작하기 -> 초보자용 -> 지금코드 실행



![image-20200317141417177](.assets/image-20200317141417177.png)



코랩 리서치(https://colab.research.google.com/)

github주소를 코랩 리서치 주소로 붙여줌

[예시링크](https://colab.research.google.com/github/HaSense/Keras/blob/master/keras001.ipynb)

![image-20200317143903354](.assets/image-20200317143903354.png)

드라이브로 복사

![image-20200317144203318](.assets/image-20200317144203318.png)

더블클릭 -> 밑에 있는 google colaboratory 클릭



ANACONDA 

![image-20200317152708328](.assets/image-20200317152708328.png)



계정/Code/Tensorflow/Keras 폴더를 만들고 

![image-20200317152909916](.assets/image-20200317152909916.png)

python3 (Notebook)을 사용하고 파일명을 exam001로 만듦



C:\Users\student\Anaconda3\envs

C:\Users\student\Anaconda3\envs\keras\Lib\site-packages



파이썬 가상환경에서 데이터를 받으면 데이터는 아래 폴더에 있음

C:\Users\student\.keras\datasets (계정\가상환경\datasets)



기타 툴 설치 없이 ipynb ( jupyter notebook ) 코드를 crome에서 바로 실행 & 구글 드라이브에 저장하는 방법

https://colab.research.google.com/

위 코드에 github 코드 주소 중 [https://와](https://xn--ol5b/) .com을 지우고 붙임

https://colab.research.google.com/github/HaSense/Keras/blob/master/keras001.ipynb