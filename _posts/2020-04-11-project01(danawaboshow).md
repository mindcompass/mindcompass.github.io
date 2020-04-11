---

title: 멀티캠퍼스 프로젝트1 <다나와보쇼>
excerpt: 멀티캠퍼스에서 수행했던 첫번째 프로젝트 결과물
toc: true
toc_sticky: true

categories:
  - multicampus_class

tags:
- java
- springboot
- project
- 
use_math: true
last_modified_at: 2020-04-11
---



<br>

<br>

올해 초 멀티캠퍼스에서 자바 프로그램을 배운 뒤 수행했던 첫 번째 프로젝트 내용을 공유드립니다. 

해당 페이지는 아래 링크를 클릭하시면 됩니다. 

[다나와보쇼 사이트 바로가기](http://selfcompass.synology.me:8080/) 

Email과 password는 Sign up에서 등록하시면 됩니다. 

그냥 들어가고 싶은 분은 Email과 password에 각각 1234를 입력해주세요. 

![프로젝트1_다나와보쇼](https://i.imgur.com/Yq7LYeS.jpg)





<br>

## 1. 다나와보쇼(프로젝트명) 개요

**가. 다나와보쇼는?**

우리가 사용하는 검색 포털(네이버, 다음, 유튜브)에서 최근 인기있는 주제의 콘텐츠를 확인하고, 스크랩하는 기능을 가진 어플리케이션입니다. 



내가 조사하고 싶은 키워드를 입력해 두면 각 사이트에서 제공하는 API기능을 활용하여 각 검색 엔진에서 자동으로 키워드에 해당하는 자료를 가져와서 화면에 출력해주는 프로그램입니다. 



더 자세히 알고 싶은 콘텐츠는 해당 사이트에 직접 들어가서 확인할 수 있고, 즐겨찾기에 스크랩하여 추후 다시 볼 수 있습니다. 





<br>

**나. 어떤 기술이 쓰였나요?**

개발하는데 아래 기술들이 사용되었습니다. 

JAVA Spring boot2.0

HTML5/CSS3

JAVASCRIPT(jQuery)

JPA

MySQL

Bootstrap4.0

 



<br>

**다. API를 사용한 사이트**

아래 총 6가지 사이트의 API를 사용했습니다. 

Youtube,  Kakao Video, Kakao Blog, Naver News, Naver Blog, Naver Cafe





<br>

## 2. 다나와보쇼 기능 안내





<br>

**가. Signin(로그인)/메인페이지**

주소를 치고 들어가면 가장 먼저 있는 항목입니다. 아이디, 비번은 각각 1234로 하시면 들어갈 수 있습니다. 

![signin페이지](https://i.imgur.com/mP1Fbxe.jpg)





<br>

**나. Signup 페이지/ 새롭게 아이디 등록**

이메일주소(아이디), 비번, 이름을 등록하는 페이지 입니다. 

![Signup 페이지](https://i.imgur.com/n7NmRz6.jpg)





<br>

**다. Main화면**

내가 선택한 주제에 대한 정보가 배열된 화면입니다. 새롭게 아이디를 등록한 분은 키워드를 설정하지 않아서 아무런 내용도 없을 것입니다. 



PC화면에는 3열씩 배열되어 있고, 스마트폰화면은 1열씩 콘텐츠가 배열되어 있습니다. 썸네일(사진) 부분이나 Read more 버튼을 클릭하면 실제 사이트 화면으로 이동합니다.



콘텐츠 상담에 빨간색으로 '+'으로 되어 있는 곳을 누르시면 즐겨찾기에 저장됩니다.

![Main화면](https://i.imgur.com/SgVzzhS.jpg)





<br>

**라. 즐겨찾기 페이지/스크랩 페이지**

메인화면에서 '+'로 스크랩한 콘텐츠를 한번에 보실 수 있습니다. 

즐겨찾기에는 '+' 기호가 '-'로 변경되어 있습니다. 

'-'누르면 스크랩한 자료가 삭제됩니다. 

![즐겨찾기페이지](https://i.imgur.com/Cy1ZiIn.jpg)





<br>

**마.Setting_keyword 페이지**

검색하고자 하는 주제(Keyword)와 어떤 검색엔진으로 정보를 찾을 지를 설정하는 화면입니다.

Keyword는 최대 5가지 설정할 수 있고, 검색엔진은 총 6가지 사용할 수 있습니다.  



save를 누르신 다음 **HOME 버튼**을 눌러 Main화면으로 이동하시면 됩니다.

![Setting_keyword ](https://i.imgur.com/fXmxj8r.jpg)





<br>

**바. Setting_user 페이지**

처음 등록한 user정보를 변경할 수 있는 페이지입니다.  

![Setting_user 페이지](https://i.imgur.com/HtwY9sf.jpg)





<br>

**사. About us 페이지**

함께 개발한 다른 팀원들의 화면이 나와있습니다.

![About us](https://i.imgur.com/0X0hr8h.jpg)