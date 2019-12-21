---
title:  "Java와 Python 기본 문법 비교"
excerpt: "자주 쓰는 java와 Python 문법 확인"
toc: true
toc_sticky: true

categories:
  - programming
tags:
- Java
- Python
- 비교
- 문법
last_modified_at: 2019-12-21
---


- 프로그램은 언어에 따라서 사용하는 문법이 약간 다릅니다. Java와 Python의 기본 문법의 차이를 확인해봅시다

## 1. IF - ELSE 구문

간단하게 숫자를 입력하면 숫자가 양수, 음수, 0 인지를 확인하는 조건문입니다.

python의 경우 중괄호와 소괄호가 포함되어 있지 않은 것이 큰 특징입니다.

그리고 python이 훨씬 간결한 문법이라는 것을 확인할 수 있습니다. 

### Java의 경우

```java
package java_project;

import java.util.Scanner;

public class if_else {

	public static void main(String[] args) {
		System.out.println("숫자를 입력하세요.");
		Scanner input = new Scanner(System.in);
		int number = input.nextInt();
		
	if (number > 0) {
			System.out.println("양수입니다.");
	} else if (number < 0) {
			System.out.println("음수입니다.");
	} else {
			System.out.println("0입니다.");
	}
  }
}
```

### Python의 경우

```python
number = input("숫자를 입력하세요. : ")
number = int (number)

if number > 0 :
    print("양수입니다.")
elif number < 0 :
    print("음수입니다.")
else :
	print("0입니다.")
```

<br>    

## 2. for 반목문 (배열 혹은 리스트 활용)

java에서는 array(배열)로 부르고, Python에서는 list라고 부릅니다. 

2가지 모두 한줄에 각각 1, 3, 5, 7, 9가 차례대로 출력됩니다. 

### Java의 경우

```java
package java_project;

public class for_배열 {

	public static void main(String[] args) {
		int[] arrays= {1,3,5,7,9};
		
        for (int i=0; i < arrays.length; i++) {
				System.out.println(arrays[i]);
            
			}
	}

}
```



### Python의 경우

```python
array =[1,3,5,7,9]

for element in array:
    print(element)
```

<br>  

## 3. While 반복문

i가 한개씩 증가하여 0~9까지 1줄씩 출력되는 반복문입니다. 

python의 경우 자바에서 사용하는 ++ 기호가 통하지 않습니다. 



### Java의 경우

```java
package java_project;

public class while_반복 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int i =0;
		while (i < 10) {
			System.out.printf("%d번째 반복입니다.\n",i);
			i++;
		}
	}
}
```

참고로 자바의 printf의 경우 **%d는 정수, %f는 실수, %c는 문자, %s는 문자열**을 나타냅니다.

### Python의 경우

```python
i =0
while i <10:
    print("{}번째 반복입니다.".format(i))
    i += 1
```

파이썬은 따로 형식을 정해줄 필요가 없어서 편한거 같습니다. 
fortmat 형식은 `print(f"{i}번째 반복입니다.")`으로 바꿀 수 있습니다. 

