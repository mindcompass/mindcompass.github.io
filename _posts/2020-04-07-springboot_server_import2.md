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







지난번에는 Spring Tool Suite4를 이용해서 war로 로컬 pc에서 만든 프로젝트를 배포해보겠습니다.

시놀로지의 기본 패키지는 tomcat7까지만 지원합니다. 스프링부트로 개발한 프로젝트는 최소 tomcat8이상에서만 작동합니다. tomcat7에서 실행시키는 몇가지 편법이 있지만, 일부 기능이 수행되지 않아 추천드리지 않습니다.   





## 1.Synology에서 도커 세팅하기

패키지센터에서 Docker 버튼을 누르면 그냥 설치됩니다. 

Docker – 레지스트리 – “tomcat” 검색 후 “tomcat” 선택해서 레지스트리를 생성합니다. 

 컨테이너에서 tomcat 8.5.5-jre8를 선택합니다. 

Docker – 이미지 – tomcat 선택 – 실행 – 마법사로 실행



![](https://i.imgur.com/cEXSYti.png)

기본적으로 우리가 만든 tomcat서버에 지난번 만든 war파일을 집어넣으면 끝납니다.

그런데... 이게 생각보다 간단하지 않더라구요. SSH를 이용해서 작업합니다. 



## 2.SSH를 이용하여 컨테이너에 터미널로 접속하기

PuTTY를 설치하고 Synology ip를 넣고 open을 실행합니다. 

![](https://i.imgur.com/rGfoCza.png)



root 사용자로 접속합니다.







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

그런데 docker에서 shutdown.sh을 수행하는 순간 tomcat서버만 죽는게 아니라 docker컨테이너 자체가 죽습니다. 따라서 미리 세팅을 다 해놓고 서버를 키면서 동작하게 하는게 최선인거 같습니다.

저는 컨테이너에 기존에 삭제해야 할 파일이 있습니다. 

리눅스명령어가 익숙하지 않으면 아래를 참고하세요.

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















1)기존 파일

```java
package com.kyuhyun.BoShow; #프로젝트 명

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class BoShowApplication {

	public static void main(String[] args) {
		SpringApplication.run(BoShowApplication.class, args);
	}

}

```

2)변경파일 

```java
package com.kyuhyun.BoShow;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.boot.web.servlet.support.SpringBootServletInitializer;


@SpringBootApplication
public class BoShowApplication extends SpringBootServletInitializer {

	@Override 
	protected SpringApplicationBuilder configure(SpringApplicationBuilder application) { return application.sources(BoShowApplication.class); }

	

	public static void main(String[] args) {
		SpringApplication.run(BoShowApplication.class, args);
	}

}
```



## 2.build.gradle내용 수정하기

build.gradle파일에는 여러가지 프로젝트 수행에 필요한 설정내용이 저장되어 있습니다. 

로컬 pc에서 돌릴 때는 h2를 사용해서 프로젝트를 수행했는데, 서버에 올릴 때는 mysql서버를 사용하여 서비스하기 위해 몇가지 사항을 수정합니다. 



1)기존 파일

```java
plugins {
	id 'org.springframework.boot' version '2.2.2.RELEASE'
	id 'io.spring.dependency-management' version '1.0.8.RELEASE'
	id 'java'
}

group = 'com.kyuhyun'
version = '0.0.1-SNAPSHOT'
sourceCompatibility = '1.8'

configurations {
	developmentOnly
	runtimeClasspath {
		extendsFrom developmentOnly
	}
	compileOnly {
		extendsFrom annotationProcessor
	}
}

repositories {
	mavenCentral()
}

dependencies {
	implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
	implementation 'org.springframework.boot:spring-boot-starter-thymeleaf'
	implementation 'org.springframework.boot:spring-boot-starter-web'
	compileOnly 'org.projectlombok:lombok'
	annotationProcessor 'org.projectlombok:lombok'
    developmentOnly 'org.springframework.boot:spring-boot-devtools'
	
    runtimeOnly 'com.h2database:h2'
	
	testImplementation('org.springframework.boot:spring-boot-starter-test') {
		exclude group: 'org.junit.vintage', module: 'junit-vintage-engine'
	}
}

test {
	useJUnitPlatform()
}
```



2)변경파일 

```java
plugins {
	id 'org.springframework.boot' version '2.2.2.RELEASE'
	id 'io.spring.dependency-management' version '1.0.8.RELEASE'
	id 'java'
	id 'war'  #war를 추가해줍니다. 
}

group = 'com.kyuhyun'
version = '0.0.1-SNAPSHOT'
sourceCompatibility = '1.8'

configurations {
	developmentOnly
	runtimeClasspath {
		extendsFrom developmentOnly
	}
	compileOnly {
		extendsFrom annotationProcessor
	}
}

repositories {
	mavenCentral()
}

dependencies {
	implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
	implementation 'org.springframework.boot:spring-boot-starter-thymeleaf'
	implementation 'org.springframework.boot:spring-boot-starter-web'
	compileOnly 'org.projectlombok:lombok'
	annotationProcessor 'org.projectlombok:lombok'
	developmentOnly 'org.springframework.boot:spring-boot-devtools'
	
	runtimeOnly 'mysql:mysql-connector-java' # h2서버를 mysql로 변경합니다. 

	providedCompile 'org.springframework.boot:spring-boot-starter-tomcat' #톰캣서버로 연결
	compile group: 'javax.el', name: 'javax.el-api', version: '3.0.0'
	providedCompile group: 'javax.servlet', name: 'javax.servlet-api', version: '3.1.0'
	
	testImplementation('org.springframework.boot:spring-boot-starter-test') {
		exclude group: 'org.junit.vintage', module: 'junit-vintage-engine'
	}
}

test {
	useJUnitPlatform()
}
```



## 3.application.properties 내용 수정

src/main/resources안에 있는 application.properties 내용을 수정합니다. 

앱에 사용할 데이터베이스 연결 정보를 수정합니다.  



1)기존파일

```java
# log level
#logging.level.com.kyuhyun.basic=trace

# datasource
#spring.datasource.url=jdbc:h2:~/test
spring.datasource.url=jdbc:h2:tcp://localhost/~/BoShow
spring.datasource.driverClassName=org.h2.Driver
spring.datasource.username=sa
spring.datasource.password=

# jpa
spring.jpa.hibernate.ddl-auto=update
spring.jpa.database-platform=org.hibernate.dialect.H2Dialect
spring.jpa.show-sql=true

# auto increment
spring.jpa.hibernate.use-new-id-generator-mappings=false
```



2)변경파일

```java
# log level
#logging.level.com.yonghyun.basic=trace

spring.jpa.database=mysql

# datasource

spring.datasource.url=jdbc:mysql://selfcompass.synology.me:3307/compsylab?serverTimezone=UTC&characterEncoding=UTF-8

spring.datasource.driverClassName=com.mysql.cj.jdbc.Driver
spring.datasource.username= mydata_base #Mysql에 등록한 db이름을 작성합니다. 
spring.datasource.password= 123456 # 등록한 db의 패스워드를 입력합니다. 

# jpa
spring.jpa.hibernate.ddl-auto=update
spring.jpa.database-platform=org.hibernate.dialect.MySQL5InnoDBDialect
spring.jpa.show-sql=true

# auto increment
spring.jpa.hibernate.use-new-id-generator-mappings=false
```



## 4.Gradle Executions으로 bootwar를 수행하기

창 상단에 window의 텝에서 Show View선택하고 Gradle에서 Gradle Tasks수행하기

![Gradle Executions](https://i.imgur.com/YtKIvks.png)

그럼  view에 Gradle Tasks가 새롭게 생성된 것을 보실 수 있습니다. 

그 다음 build에 들어가서 bootwar를 실행시킵니다. 

![](https://i.imgur.com/1xVIqIF.png)

그럼 프로젝트 폴더 build폴더 / libs폴더 안에 생성된 war파일을 확인할 수 있습니다.

