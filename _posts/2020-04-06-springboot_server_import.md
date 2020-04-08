---

title: Sprint boot 프로젝트 WAR로 배포하기_01
excerpt: 스프링부트로 개발한 프로젝트를 tomcatsever에 올리기 전 작업 내용을 정리함
toc: true
toc_sticky: true

categories:
  - web

tags:
- springboot
- tomcat
- WAR
- 
use_math: true
last_modified_at: 2020-04-06
---





이번에는 스프링부트로 개발한 프로젝트를 실제 톰캣서버에 배포하는 방법에 대해서 알아보겠습니다. 

IDE는 Spring Tool Suite4를 이용했고, 빌드도구로 Gradle을 사용했습니다. 





## 1.SpringBootServletInitializer 상속받는 것으로 변경함

프로젝트 메인 실행파일(src/main/java 안에 존재함)이 있는 파일의 내용을 수정합니다. 



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

