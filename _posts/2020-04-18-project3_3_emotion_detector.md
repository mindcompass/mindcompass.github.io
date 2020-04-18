---
title: open-cv와 딥러닝을 통해 인간의 얼굴표정 인식하기
excerpt: web캠을 통해 실시간으로 감정을 분석할 수 있는 프로그램 안내
toc: true
toc_sticky: true

categories:
  - multicampus

tags:
- emotion
- project 
- 
- 
use_math: true
last_modified_at: 2020-04-18
---

<br>

안녕하세요.

제가 평소에 궁금하게 생각했던 기술 중 하나가 인간의 얼굴표정을 통한 감정파악 대한 딥러닝 기술입니다. 



요즘 딥러닝을 공부하면서 대략 어떤 원리에서 이런 결과가 도출되는지 감을 잡을 수 있었습니다. 



자료를 찾다보니 해당 기술을 구현해 놓은 코드도 있어서 직접 수행해볼 수 있었습니다. 



저희 조 분들도 한번 체험해보시면 좋을 것 같아 내용 정리해서 공유드립니다. 



## 1. 딥러닝 기술을 활용한 인간의 얼굴 표정 인식하기 



**[프로그램 1 링크](https://github.com/petercunha/Emotion)**

링크에서 Clone or download에서 자료 받으시면 됩니다. 

단, tensorflow 패키지는 1.15로, scipy 패키지를 1.2.0으로 다운그레이드를 하셔야 합니다. 

코드를 살펴보니 scipy 최신버전에서는 몇가지 함수기능이 지원되지 않아서 오류가 발생합니다.  

<br>

프로그램이 있는 폴더로 이동한 다음 아래 명령어를 치면 실행됩니다. 

```python
python emotions.py
```



감점은, 중립(neutral), 놀람(surprise), 분노(angry), 행복(happy), 슬픔(sad) 5가지로 표현됩니다.

웹캠이 설치되어 있어야 활용 가능합니다. 

![Demo](https://github.com/petercunha/Emotion/blob/master/demo/demo.gif?raw=true)

<br>

아래와 같은 패키지가 설치되어 있어야 합니다. 

- tensorflow
- numpy
- scipy
- opencv-python
- pillow
- pandas
- matplotlib
- h5py
- keras



신경망 모델이 아래와 같이 설계되어 있다고 합니다.

 신경망이 2갈래로 나눠져 있는게 특이하네요.

![Model](https://i.imgur.com/vr9yDaF.png?1)







**[프로그램 2 링크](https://github.com/omar178/Emotion-recognition)**

angry, disgust, scared, happy, sad, surprised, neutral 라는 7가지 감정으로 구분됩니다. 



제 컴퓨터가 사양이 좋지 않아서 그런지 해당 프로그램을 실행하면 조금 버벅되는 경향이 있습니다. 

그리고 제 표정도 첫번째 프로그램 더 잘 잡아내는 것 같네요.

<br>

![이미지](https://github.com/omar178/Emotion-recognition/blob/master/emotions/Happy.PNG?raw=true)



프로그램이 있는 폴더로 이동한 다음 아래 명령어를 치면 실행됩니다. 

```python
python real_time_video.py
```



<br>

## 2. 얼굴 표정 인식에 대한 학습 과정에 대한 자료

 얼굴 표정에 대한 학습과정을 잘 설명한 자료가 있어서 첨부합니다. 



[Training a TensorFlow model to recognize emotions](https://medium.com/@jsflo.dev/training-a-tensorflow-model-to-recognize-emotions-a20c3bcd6468)



위 자료랑 코드를 보면서 유사한 프로그램을 만들어도 실력 향상에 많은 도움이 될 것 같습니다. 