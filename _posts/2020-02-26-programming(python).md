---
title:  "문자열을 python 실행코드로 변환해주는 eval(), exec()"
excerpt: "파이썬 내장함수인 eval(), exec()활용한 코드 작성"
toc: true
toc_sticky: true

categories:
  - programming
tags:
- Python
- function
use_math: true
last_modified_at: 2020-02-26
---

eval()과 exec()함수는 모두 python의 내장함수입니다. 

eval()은 표현된 python 식을 받아 파이썬 컴파일 코드로 변환합니다. 

반면 exec()은 문자열로 표현된 문을 받아서 파이썬 컴파일 코드로 변환합니다. 

```python
a = 2
a = eval('a+2') #단순한 연산식
print(a)
2

exec('b=a+2') # 변수에 수식의 값을 넣는 실행문
print(b)
2

만약 eval에 실행문을 넣으면,
a = 2
eval('b=a+2') 
print(b) syntax에러 발생

b = exec('a+2')
print(b) #리턴된 내용이 없으니 b값에 내용이 없음
```



한가지 문제를 풀면서 설명드리겠습니다. 

[hackerrank문제 바로가기](https://www.hackerrank.com/challenges/input/problem)



문제는 아래와 같습니다. 

입력은 1줄씩 2번에 나눠서 2번 진행됩니다. 

결과는 1을 아래 식에 대한 결과가 4이고, 그 값이 첫번째줄에 입력한 값과 같아서 True를 반환합니다. 

**Sample Input**

```python
1 4
x**3 + x**2 + x + 1
```

**Sample Output**

```python
True
```




## 1. eval()함수 활용

```python
input1, input2 =map(int, input().split()) #input으로 받는 데이터를 정수로 변환하여 각각 저장합니다. 

input3 =input() #다항식(x**3 + x**2 + x + 1)을 스트링으로 전달 받습니다. 

x=input1 #input으로 전달받은 숫자 1을 x변수에 집어 넣습니다. 

value=eval(input3) 
# 단순 문자열이었던 input3의 내용이 실제 python 상의 코드처럼 실행되어 이 값이 value에 저장됩니다.(value에 저장되는 값은 4입니다. ) 

print(value==input2) #value값과 input값이 동일하기 때문에 True가 출력됩니다. 

```

### 2.exec()함수 활용

```python
input1, input2 =map(int, input().split())
input3 =input() 
#여기 까지는 똑같습니다. 

s1='''
x=input1
value=eval(input3) 
print(value==input2)
'''
exec(s1)
# 여러 실행문을 하나의 스트링으로 저장해서 한번에 실행할 수 있습니다. 
```

