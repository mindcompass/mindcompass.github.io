---
title: 멀티캠퍼스_머신러닝7
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
last_modified_at: 2020-04-16
---



## 1. Keras를 이용한 사진분류

- Python + Google Custom Search API 

- Python + OpenCV 

- Python + Keras 

- Python + Django



1) 이미지를 다운받음(연예인, 유명인)

2) 얼굴을 추출함(얼굴을 인식할 수 있는 모듈이 필요함, 대표적으로 OpenCV)

3) KERAS로 모델을 구분함

4)Django로 모델의 웨이트가 있는 모형을 넣어서 웹프로그램으로 만듦



연예인의 얼굴을 구분하는 웹프로그램을 만들자



api키AIzaSyBK2DwITOZCmeQi6osVaRhydW_0gS9rolk



2가지 키값이 필요함

api 및 서비스 > 사용자 인증정보

라이브러리 >Custom Search API > 사용설정 

사용자 인증정보(필요없음)



cumstom search engine 접속 엔진 추가

전지현은 00, 송혜교01로 만들것  keyword_count = 1을 0으로 만들 것



**클립보드로 복사**

• 설정 파일 추가 à settings.env CASCADE_FILE_PATH=”{작업경로}\haarcascades\haarcascade_frontalface_default.xml"

https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml



https://github.com/opencv/opencv/tree/master/data/haarcascades(다양한 표정)



사진에 해당 오브젝트(여기서는 얼굴)를 추출해주는 알고리즘

[open_cv링크]https://docs.opencv.org/4.1.0/dc/d88/tutorial_traincascade.html



![Imgur](https://i.imgur.com/DvoJ5Fw.png)





이미지 검색 사용함 on 변경 

전체 웹 검색 on 변경





pip install python-dotenv 정보를 숨겨주는 패키지

setting auto save



git에 올려주신 코드된 파일  

01.img_down_gcs.py



100장을 받지만 모두 신뢰할 수 없음

setting_1을 setting으로 변경



env파일 추가

CASCADE_FILE_PATH=C:\Users\student\Desktop\multicompass\machine_learning\project\haarcascades\haarcascade_frontalface_default.xml



setting파일 추가

CASCADE_FILE_PATH=os.environ.get("CASCADE_FILE_PATH")



2. img_face 파일 작성 내용 삭제

   강사님이 주신 02.img_face_detect.py파일로 대체함



```PYTHON





for name_image in name_images:
        file_path = os.path.join(OUTPUT_IMAGE_DIR, f"{name_image[0]}")
        image = name_image[1]

        cascade_filepath = settings.CASCADE_FILE_PATH

        detect_image_face(file_path, image,cascade_filepath)  
```



기본적으로 흑백으로 변경해서 처리함

```python
scaleFactor=1.1, #원본이미지를 1.1로 나눠야 하는데 점점 작아지면서 특징을 잡아야 되기 떄문에 1보다 커야 함
```





pip install graphviz 한다고 바로 설치된 것은 아님

![graphviz주소](https://i.imgur.com/hNWBhjC.png)

[사이트주소](https://graphviz.gitlab.io/_pages/Download/Download_windows.html)

여기 눌러서 다운로드 [graphviz-2.38.msi](https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.msi)



패스잡기

C:\Program Files (x86)\Graphviz2.38\bin 설정

잡답 : 페이스북 이노베이션 랩에서 무료교육이 있음