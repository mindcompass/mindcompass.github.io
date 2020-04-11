---

title: Sprint boot 프로젝트 WAR로 배포하기_02
excerpt: 스프링부트로 개발한 프로젝트를 synology Nas docker을 이용해 배포하기
toc: true
toc_sticky: true

categories:
  - web

tags:
- springboot
- tomcat
- synology
- docker
use_math: true
last_modified_at: 2020-04-07
---



이번에는 스프링부트로 개발한 프로젝트를 실제 톰캣서버에 배포하는 방법에 대해서 알아보겠습니다. 



지난번에는 Spring Tool Suite4를 이용해서 만들었던 프로젝트 파일을 war 파일로 생성했습니다. 이번에는 synology nas를 이용해서 배포하는 방법을 말씀드리겠습니다.



시놀로지의 기본 패키지는 tomcat7까지만 지원합니다. 스프링부트로 개발한 프로젝트는 최소 tomcat8이상에서만 작동합니다. tomcat7에서 실행시키는 몇가지 편법이 있지만, 일부 기능이 제대로 되지 않을 가능성이 있기 때문에 추천드리지 않습니다.   



## 1.Synology에서 도커 세팅하기

패키지센터에서 Docker 버튼을 누르면 그냥 설치됩니다. 

**Docker – 레지스트리 – “tomcat” 검색 후 “tomcat” 선택**해서 레지스트리를 생성합니다. 

스프링부트2.0 프로젝트를 tomcat7.0에서 수행했을 때, 일부 기능이 제대로 수행하지 않았습니다. 

그래서 저는 컨테이너에서 tomcat 8.5.5-jre8를 선택했습니다. 

**Docker –> 이미지 – tomcat 선택 – 실행 – 마법사로 실행**

마법사로 실행하면 아래 화면이 나타납니다. 해당 옵션은 컨테이너로 만든 다음에도 수정 가능합니다. 

![이미지에서 컨테이너생성](https://i.imgur.com/0pB90Un.png)



위에서 고급설정을 수행하면 볼륨이라는 항목이 있습니다. 

이는 docker의 파일이나 폴더와 시놀로지 파일을 파일이나 폴더를 동기화 시킬 수 있는 통로 역할을 해야 합니다. 

해당 기능을 잘 사용하면 밑에서 설명 들이는 복잡한 SSH통신이나 리눅스 명령어, 도커 관련 명령어를 사용하지 않고도 파일을 쉽게 전달 하고 받을 수 있습니다.  

![볼륨사용법](https://i.imgur.com/61cS8tW.png)

왼쪽 파일/폴더에 시놀로지 특정 폴더를 새롭게 만듭니다. 이 폴더는 앞으로 Docker의 tomcat서버의 특정 폴더와 연결시키거나 특정 경로에 새로운 폴더를 만드는데 사용할 것입니다. 



tomcat은 기본적으로 webapps에서 있는 ROOT 폴더에 있는 프로젝트 파일을 가장 먼저 읽습니다. 

그리고 거기에 파일 없으면 ROOT의 부모파일일 webaspps에 프로젝트 파일이 있는지 찾은 다음에 이를 읽어들입니다. 

따라서 ROOT에 우리가 작성한 파일을 넣기만 하면 사실 게임이 끝납니다. 

마운트 경로는 사람마다 차이가 있을 수 있지만 기본적 설정을 바꾸지 않은 한 **/usr/local/tomcat/webapps/ROOT** 로 해주시면 됩니다. 

이렇게 하면 Docker서버에 있는 webapps/ROOT에 있는 기존 파일들은 무시하고 내가 설정한 시놀로지에 있는 파일 경로에 있는 파일(프로젝트 war파일)을 읽어드립니다. 

저는 기존 web폴더에 docker라는 폴더를 생성하고 WAR파일의 압축을 풀어서 넣어두었습니다. 

 ![시놀로지 내 폴더 생성](https://i.imgur.com/bPNYjPq.png)



그리고 컨테이너 항목에 왼쪽 끝에 달린 토글버튼은 on해주시면 tomcat1 내부의 카탈리나 서버가 자동으로 수행되면서 프로젝트를 수행하게 됩니다.

![](https://i.imgur.com/cEXSYti.png)

제가 만든 프로젝트가 제대로 출력되는 것을 볼 수 있습니다. 

![](https://i.imgur.com/PXjEL9N.png)



만약 ROOT에 내가 만든 폴더를 설정하지 않으셨으면 아래처럼 톰캣 기본 설정 파일이 수행됩니다. 

![](https://lh3.googleusercontent.com/proxy/oInAxPhV3gG_EoG040FqSdDbaFTp6ugWUeAEwNpF4ZL7ZmAIpOjXrunYEJgCd5bdWVbpN2V7G0H62r85UT25Xo_FbVzpDL07d1LfZI-0GYc_hWqEl4lVxc2NS_rROdH4ppjCLtkEtsgGSdRsAxpHJQQ-3rcLmJLHTCc2Wa3Hs7Ge2SwzaJzSmV8A0sRlddfW0C-S4EBx8E7fOwSnY_SRMJSWcnODOYETF4UQxc_uFD4rDO-3eST7MM469IXlbFPucUqtL0ShENBwYykkSVg5vP8P0RJ8DdM8zw)

이것은 ROOT에 있는 기본적인 톰캣서버의 자료를 Load한 것입니다. 

그런데 만약 접속이 잘 안된다면 그것은 port와 관련해서 제대로 설정되지 않은 것입니다. 

위에서 tomcat서버에 on버튼을 수행하는 것만으로 서버는 제대로 구동합니다. 

시놀로지에서 port 관련 설정은 아래 정리가 잘 되어 있어서 참고하시면 좋을 것 같습니다. 



참고 링크1 : https://devks.tistory.com/16?category=686547 

참고 링크2 : https://isulnara.com/wp/archives/721





## 2.SSH를 이용하여 컨테이너에 터미널로 접속하기

저처럼 리눅스나 도커 명령어에 익숙하지 않으신 분들이 있을 수 있어 해당 내용도 정리했습니다. 



**가. Putty로 docker에 접속하기**

PuTTY를 설치하고 Synology ip를 넣고 open을 실행합니다. 

![](https://i.imgur.com/rGfoCza.png)



root 사용자로 접속하기 위해 synology 관리자 모드로 들어갑니다. 

docker ps로 설치된 컨테이너를 확인하고, docker exec -it [컨테이너ID] /bin/bash를 통해 컨테이너에 접속합니다.

![](https://i.imgur.com/stGeCYh.png)

ls를 입력하여 컨테이너에 있는 파일과 폴더를 확인합니다. 

파일과 폴더는 아래 내용들이 있네요.

```java
LICENSE  RELEASE-NOTES  bin   include  logs            temp     work
NOTICE   RUNNING.txt    conf  lib      native-jni-lib  webapps 
```



bin 폴더에는 아래와 같은 파일들이 있네요.

```java
bootstrap.jar                 configtest.sh    startup.sh
catalina-tasks.xml            daemon.sh        tomcat-juli.jar
catalina.sh                   digest.sh        tool-wrapper.sh
commons-daemon-native.tar.gz  setclasspath.sh  version.sh
commons-daemon.jar            shutdown.sh
```

카탈리나서버를 껐다가 키는 동작을 하는 startup.sh 와 shutdown.sh 가 눈에 보이네요.



**나. 알면 유익한 리눅스 명령어(삭제/ 복사)** 

리눅스명령어가 익숙하지 않으면 아래를 참고하세요.

1 ) 파일이나 폴더 삭제하기

rm 명령어를 사용하는 삭제 방법
rm을 사용할 경우 아래와 같이 옵션을 설정할 수 있습니다.

r : 파일 디렉토리 함께 삭제하기
f : 파일 유무와 상관없이 삭제하기
v : 어떻게 완료되었는지 설명하기

예를들어 아래와 같이 rm 명령어를 사용할 수 있습니다. 파일 및 디렉토리 모두 삭제하는 것이 가능합니다.

```java
rm -r -f [folderName]
rm -rf [folderName]
```



**2) 파일/ 폴더 복사하기**

기본적으로 파일을 복사하기 위해서는 cp명령어가 사용됩니다. 

먼저 해당 복사할 파일이 있는 디렉토리로 이동하고 아래처럼 명령어를 작성합니다.

cp [복사할 파일명]   [복사될 파일의 경로] [복사될 파일명]

예) cp a.txt   /usr/local/tomcat/webapps/ROOT/WEB-INF/a.txt

a.txt파일을 해당 폴더의 a.txt로 파일로 복사하는 명령어입니다. 



**디렉토리 전체를 복사**하기 위해서는 **-r 옵션**을 사용합니다. (recursive)

cp -r [복사할 폴더명]  [복사될 파일의 경로]

예) cp -r WEB-INF   /usr/local/tomcat/webapps/ROOT/WEB-INF

WEB-INF폴더를 해당 경로의 WEB-INF폴더로 복사하라는 명령어입니다. 
