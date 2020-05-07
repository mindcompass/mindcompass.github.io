---
title: 멀티캠퍼스 강화학습_1
excerpt: 멀티캠퍼스 강화학습 수업 정리내용입니다
toc: true
toc_sticky: true

categories:
  - multicampus

tags:
- 멀티캠퍼스
- 강화학습
- 
- 
use_math: true
last_modified_at: 2020-05-07
---

<br>



오늘부터 새로운 권혜윤 강사님이 강의해주십니다.



## 1. 강화학습 이론

![Imgur](https://i.imgur.com/TmnYi80.png)![Imgur](https://i.imgur.com/ctB40On.png)

![Imgur](https://i.imgur.com/QPvqAhN.png)



<br>

## 2. 미로찾기 실습

**가. 랜덤하게 미로 찾기**

```python
# 미로 찾기에 필요한 라이브러리 임포트
# 배열을 사용하기 위해서 numpy 임포트
import numpy as np
#파이썬에서 데이타를 차트로 그려주는 라이브러리 패키지 matplotlib.pyplot 임포트
import matplotlib.pyplot as plt
#notebook을 실행한 브라우저에서 차트를 볼 수 있도록 설정
%matplotlib inline

#plt.figure(figsize=(가로 크기,세로 크기)) : 그림의 가로 세로 설정 (단위 인치)
#그림의 가로 5인치 세로 5인치 설정
fig = plt.figure(figsize=(5, 5))

# plt.gca().set_xlim(0,3) : x좌표를 0~3으로 설정
plt.gca().set_xlim(0, 3)
#plt.gca().set_ylim(0, 3) : y좌표를 0~3으로 설정
plt.gca().set_ylim(0, 3)


#plt.text(): 그래프에 글씨를 출력하는 함수
#plt.text(글씨 x좌표, 글씨 y좌표, 출력할 글씨,
#      size="글씨사이즈 (단위 pt)", ha="정렬방법 left, right,center")
plt.text(0.5, 2.5, 'S0', size=14, ha='center')
plt.text(1.5, 2.5, 'S1', size=14, ha='center')
plt.text(2.5, 2.5, 'S2', size=14, ha='center')
plt.text(0.5, 1.5, 'S3', size=14, ha='center')
plt.text(1.5, 1.5, 'S4', size=14, ha='center')
plt.text(2.5, 1.5, 'S5', size=14, ha='center')
plt.text(0.5, 0.5, 'S6', size=14, ha='center')
plt.text(1.5, 0.5, 'S7', size=14, ha='center')
plt.text(2.5, 0.5, 'S8', size=14, ha='center')
plt.text(0.5, 2.3, 'START', ha='center')
plt.text(2.5, 0.3, 'GOAL', ha='center')

# plt.gca().plot () : 그래프에 그림을 그림
#plt.gca().plot([그림의 x 좌표],[그림 y좌표], 
#              marker="그림모양 o 는 원 s는 사각형 d는 다이아몬드", color="그림색 g는 그린")
circle = plt.gca().plot(0.5, 2.5, marker="o", color='g', markersize=60)


#plt.tick_params () 눈금표시 숨기기
#labelbottom=False :아래쪽 label  숨기기
#labelleft=False : 왼쪽 label 숨기기
#bottom=False : 아래쪽 눈금 숨기기
#left=False : 왼쪽 눈금 숨기기
plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

import random as pr
#pr.choice([0,1,2,3]) : 0,1,2,3 중에서 하나의 수를 랜덤하게 선택해서 리턴
action=pr.choice([0,1,2,3])
print("action:",action)

# 목표 지점에 이를 때까지 에이전트를 계속 이동시키는 함수

# 시작 지점 s0 부터 시작
position = 0  
#이동 경로를 저장하는 리스트
#S0부터 시작하기 때문에 초기값 0이 저장되 있음
position_history = [0]  
#다음 이동 위치를 저장하는 변수
position_next=0

step=1
while (True):  
    print("="*100)
    print("step:",step)
    print("position:",position)
    
    

    #이동 방향의 숫자 값을 랜덤하게 선택해서 리턴
    #이동 방향은 0 -> 위로 이동
    #            1 -> 오른쪽 이동
    #            2 -> 아래로 이동
    #            3 -> 왼쪽 이동
    action=pr.choice([0,1,2,3])
    print("action:",action)

    #이동 방향을 저장하는 리스트
    #이동 방향은 0 -> 위로 이동
    #            1 -> 오른쪽 이동
    #            2 -> 아래로 이동
    #            3 -> 왼쪽 이동
    direction = ["up", "right", "down", "left"] # 0,1,2,3을 주석으로 만듦
    
    #이동방향을 next_direction 에 저장
    next_direction = direction[action]
    print("next_direction:",next_direction)

    #이동 방향에 따른 다음 이동 위치를 position_next에 대입
    if next_direction == "up":
        position_next = position - 3  # 위로 이동하면 상태값이 3 줄어든다
    elif next_direction == "right":
        position_next = position + 1  # 오른쪽으로 이동하면 상태값이 1 늘어난다
    elif next_direction == "down":
        position_next = position + 3  # 아래로 이동하면 상태값이 3 늘어난다
    elif next_direction == "left":
        position_next = position - 1  # 왼쪽으로 이동하면 상태값이 1 줄어든다

    print("position_next:", position_next)
    
    #position_next가 0 미만이거나 position_next가 8 초과 이면
    if (position_next<0) or (position_next>8):
        #continue 아래 부분을 실행하지 않고 반복문 처음으로 이동
        continue #제한된 범위 밖에 나가면 실행하지 말아라

    
    # 경로 리스트에 다음 상태(위치)를 추가
    position_history.append(position_next)  
    
    if position_next == 8:  # 목표 지점에 이르면 종료
        print("Goal: Game End!!")
        break

    #다음 이동 위치 position_next를 position에 대입
    position=position_next
    
    step=step+1
    print("="*100)

    
#이동 위치 출력
position_history

circle
#[<matplotlib.lines.Line2D at 0x7f6808691be0>]
circle[0]
#<matplotlib.lines.Line2D at 0x7eff084255f8>

#총 이동 경로 개수
len(position_history)
#9


from matplotlib import animation
from IPython.display import HTML

def animate(i):
    '''프레임 단위로 이미지 생성'''
    state = position_history[i]  # 현재 위치
    x = (state % 3) + 0.5  # 상태의 x좌표 : 3으로 나눈 나머지 + 0.5
    y = 2.5 - int(state / 3)  # y좌표 : 2.5에서 3으로 나눈 몫을 뺌
    circle[0].set_data(x, y)
    return (circle[0])

#애니메이션을 실행 할 객체
#matplotlib.animation.FuncAnimation(애니메이션을 실행할 객체 fig, 
#              애니메이션의 각 장면을 리턴할 함수 animate , 
#             frames=애니매이션에서 그릴 이미지 개수  , 
#             interval = 애니메이션 각 장면을 리턴할 함수를 호출하는 시간 (단위 1/1000 초) 200
#             repeat=애니메이션 반복 여부 False (반복 안함))

anim = animation.FuncAnimation(fig, animate,  
                               frames=len(position_history), interval=200, repeat=False) #0.2초마다 / position_history 위치내용

#ani.to_jshtml() : 애니메니션을 실행하고 각 결과를 html태그로 변환해서 리턴
#HTML() : 애니메이션을 화면에 출력
HTML(anim.to_jshtml())



```



**나. Q함수 미로 찾기**

```PYTHON
#동일한 내용 생략

import random as pr
#pr.choice([0,1,2,3]) : 0,1,2,3 중에서 하나의 수를 랜덤하게 선택해서 리턴
action=pr.choice([0,1,2,3])
print("action:",action)

np.amax([0, -1, 100, 30, 40]) #최대값을 리턴함
#100
np.argmax([0, -1, 100, 30, 40]) #최대값의 인덱스를 리턴함
#2
Q = np.zeros([8,4]) #0~7까지 8가지 배열을 생성함
#array([[0., 0., 0., 0.],
#       [0., 0., 0., 0.],
#       [0., 0., 0., 0.],
#       [0., 0., 0., 0.],
#       [0., 0., 0., 0.],
#       [0., 0., 0., 0.],
#       [0., 0., 0., 0.],
#       [0., 0., 0., 0.]])

for index in range(0,20):
  position = 0
  position_next =0

  step =1
  while(True):
    print("step: ", step)
    print("position: ", position)
    m= np.amax(Q[position,:])
    print("m:",m)
    if m==0:
      action = pr.choice([0,1,2,3])
    else:
      action=np.argmax(Q[position,:]) #m= 1이면 
    print("action:",action)
    direction = ["up","right","down","left"]
    next_direction = direction[action]
    print("next_direction:",next_direction)

    if next_direction =="up":
      position_next = position -3
    elif next_direction =="right":
      position_next = position +1
    elif next_direction =="down":
      position_next = position +3
    elif next_direction =="left":
      position_next = position -1

    print('position_next:', position_next)


    if (position_next <0) or (position_next >8):
      continue
    if position_next==8 :
      print('Goal: Game End!!')
      Q[position, action]=1
      print("="*100)
      print('Q')
      print(Q)
      print("="*100)
      break

    else :
      Q[position,action]= np.amax(Q[position_next, :])

    position= position_next

    step = step +1
    print("="*100)
    
    
    
#최적 경로 확인 코드 
position = 0
position_next =0
position_history=[0]

step =1
while(True):
  print("step: ", step)
  print("position: ", position)
  # m= np.amax(Q[position,:]) 이미 Q함수의 값이 정해져 있어서 필요없음
  # print("m:",m)
  # if m==0:
  #   action = pr.choice([0,1,2,3])
  # else:
  action=np.argmax(Q[position,:]) #m= 1이면 

  print("action:",action)
  direction = ["up","right","down","left"]
  next_direction = direction[action]
  print("next_direction:",next_direction)

  if next_direction =="up":
    position_next = position -3
  elif next_direction =="right":
    position_next = position +1
  elif next_direction =="down":
    position_next = position +3
  elif next_direction =="left":
    position_next = position -1

  print('position_next:', position_next)


  if (position_next <0) or (position_next >8):
    continue

  position_history.append(position_next)
  if position_next==8 :
    print('Goal: Game End!!')
    # Q[position, action]=1
    # print("="*100)
    # print('Q')
    # print(Q)
    # print("="*100)
    break

  # else :
  #   Q[position,action]= np.amax(Q[position_next, :])

  position= position_next

  step = step +1
  print("="*100)


position_history

# (중복내용 생략)

```



**다. 미로찾기 최적경로 **

```python
#(중복생략)

import random as pr
action=pr.choice([0,1,2,3])
print("action:",action)


Q = np.zeros([8,4])

##dis를 활용해서 최적경로 찾기
dis=0.99
for index in range(0,300):
  position = 0
  position_next =0

  step =1
  while(True):
    print("step: ", step)
    print("position: ", position)

    e= 1.0/((index//100)+1) # index가 100까지는 1,이후에는 점차 작아짐
    
    if np.random.rand(1) < e: #처음에는 랜덤으로 하다가 점차 랜덤이 아닌 경로를 찾게됨
      action = pr.choice([0,1,2,3])
    else :
      action = np.argmax(Q[position, :])

    print("action:",action)
    direction = ["up","right","down","left"]
    next_direction = direction[action]
    print("next_direction:",next_direction)

    if next_direction =="up":
      position_next = position -3
    elif next_direction =="right":
      position_next = position +1
    elif next_direction =="down":
      position_next = position +3
    elif next_direction =="left":
      position_next = position -1

    print('position_next:', position_next)


    if (position_next <0) or (position_next >8):
      continue
    if position_next==8 :
      print('Goal: Game End!!')
      Q[position, action]=1
      print("="*100)
      print('Q')
      print(Q)
      print("="*100)
      break

    else :
      Q[position,action]= dis*np.amax(Q[position_next, :])

    position= position_next

    step = step +1
    print("="*100)

#Q 같은 수치면 앞의 index가 반환됨

#[[0.         0.970299   0.970299   0.        ]
# [0.         0.9801     0.9801     0.96059601]
# [0.         0.970299   0.99       0.970299  ]
# [0.96059601 0.9801     0.9801     0.9801    ]
# [0.970299   0.99       0.99       0.970299  ]
# [0.9801     0.9801     1.         0.9801    ]
# [0.970299   0.99       0.         0.99      ]
# [0.9801     1.         0.         0.9801    ]]


#최적 경로 확인 코드 
position = 0
position_next =0
position_history=[0]

step =1
while(True):
  print("step: ", step)
  print("position: ", position)

  action=np.argmax(Q[position,:]) #m= 1이면 

  print("action:",action)
  direction = ["up","right","down","left"]
  next_direction = direction[action]
  print("next_direction:",next_direction)

  if next_direction =="up":
    position_next = position -3
  elif next_direction =="right":
    position_next = position +1
  elif next_direction =="down":
    position_next = position +3
  elif next_direction =="left":
    position_next = position -1

  print('position_next:', position_next)


  if (position_next <0) or (position_next >8):
    continue

  position_history.append(position_next)
  if position_next==8 :
    print('Goal: Game End!!')
    # Q[position, action]=1
    # print("="*100)
    # print('Q')
    # print(Q)
    # print("="*100)
    break

  # else :
  #   Q[position,action]= np.amax(Q[position_next, :])

  position= position_next

  step = step +1
  print("="*100)

#(중복 코드 생략)
```



<br>

## 3. Frozen Lake 실습

![Imgur](https://i.imgur.com/UThd6Bh.png)



**가. 실제 값을 넣는 Frozen Lake**

```python
import gym
import numpy as np
import random as pr
from gym.envs.registration import register

#강화 학습 환경을 설정
register(
    id="FrozenLake-v1",  #강화 학습 환경의 아이디를 설정
    #gym.envs.toy_text:FrozenLakeEnv : FrozenLake 환경을 설정해서 리턴 하도록 설정
    entry_point="gym.envs.toy_text:FrozenLakeEnv", 
    kwargs={
            "map_name": "4x4",  #FrozenLake 의 줄 칸을 설정 4줄 X4 칸
            "is_slippery":True #미끄러짐 효과 설정 안함
            }
)

#gym.make("FrozenLake-v1"): FrozenLake-v1 환경을 설정해서 env에 리턴
env=gym.make("FrozenLake-v1")

#env.render(): Frozen Lake 게임을 출력
env.render()

#env.reset() : 현재 위지를 s 시장 위치로 설정하고 현재 위치를 리턴
#              시작 위치는 첫번째줄 첫번째 칸이므로 0 이 리턴
position=env.reset()

#done : 게임 종료 여부를 저장 할 변수 
#       True : 게임 종료
#       False : 게임 계속 진행
done=False

#done이 False일동안 반복
while done==False:
    #env.render(): Frozen Lake 게임을 출력
    env.render()
    #input() : 입력을 받아서 입력한 값을 리턴 입력한 값은 문자열로 리턴
    #입력한 값이 str_action에 저장됨
    str_action=input("Left:0, Down:1, Right:2, Up:3 : ")
    #int(str_action) : str_action을 정수로 변환해서 리턴
    #str_action 이 정수로 변환 되서 action에 저장
    action=int(str_action)
    #env.step(action) : Frozen Lake 게임에서 action 방향으로 이동하고
    #새로운 위치, 보상(Goal 에 도착하면 1), 게임 종료 여부(Goal이거나 Hole에 빠지면 종료), 게임정보를 리턴
    #새로운 위치 - position에 저장
    #보상 -reward에 저장
    #게임 종료 여부 - done에 저장
    #게임 정보 - info에 저장
    position,reward,done,info=env.step(action)
    print("position:",position,":reward:",reward,":done:",done,":info:",info) 
```

**나. Q 함수를 이용한 Frozen Lake**

```python
#(중복코드 생략)
position=env.reset()

Q = np.array([
              [0,0,1,0],
              [0,0,0,0]
              ])

#Q[0, :] : Q배열의 0번째줄 모든 칸
#np.amax(Q[0, :]) :Q배열의 0번째줄 모든 칸에서 최대값 리턴 1
m=np.amax(Q[0, :])
print("m:",m)
#np.argmax(Q[0, :]): Q배열의 0번째줄 모든 칸에서 최대값이 저장된 인덱스 리턴 2
action=np.argmax(Q[0, :])
print("action:",action)
#m: 1
#action: 2

#pr.choice([0,1,2,3]) : 0,1,2,3 중에서 하나의 수를 랜덤하게 선택해서 리턴
action=pr.choice([0,1,2,3])
print("action:",action)

#np.zeros(): 0으로 초기화된 배열을 생성
#np.zeros([16(줄),4(칸)]) : 0으로 초기화된 16줄 4칸의 배열을 생성
Q=np.zeros([16,4])


or index in range(300):
    print("="*100)
    print("index:",index)
    position = env.reset()
    
    while True:
        #env.render(): Frozen Lake 게임을 출력
        env.render()
        print()
        #Q[position, :] : Q배열의 position번째줄 모든 칸
        #np.amax(Q[position, :]) :Q배열의 position 번째줄 모든 칸에서 최대값 리턴
        m=np.amax(Q[position, :])
        #최대값이 1일때
        if m==1:
            #np.argmax(Q[position, :]): Q배열의 position 번째줄 모든 칸에서 최대값이 저장된 인덱스 리턴 
            action=np.argmax(Q[position, :])
        else:#최대값이 1이 아닐때 , 최대값이 0
            #pr.choice([0,1,2,3]) : 0,1,2,3 중에서 하나의 수를 랜덤하게 선택해서 리턴
            action=pr.choice([0,1,2,3])
        #env.step(action): action을 실행
        new_position, reward,done,info = env.step(action)
        print("new_position:",new_position,":reward:",reward,":done:",done,":info:",info)
        #reward+np.max(Q[new_position, :]): 현재 reward+Q배열의 position 번째줄 모든 칸에서 최대값 (np.max(Q[new_position, :]) )
        #을 Q[position,action] 에 대입
        Q[position,action]=reward+np.max(Q[new_position, :])
        #new_position을 position에 대입
        position=new_position
        #done : 게임 종료 여부를 저장 할 변수 
        #       True : 게임 종료
        #       False : 게임 계속 진행
        if done==True:
            break;

    print("Q:",Q)
    print("="*100)
    
Q    #1이 있는 행은 무조건 1이 있던 방향으로 찾기 때문에 최적화된 길은 아님

#학습이 끝났으므로 길을 잘 찾아감

position = env.reset()

while True:
    #env.render(): Frozen Lake 게임을 출력
    env.render()
    print()
    #Q[position, :] : Q배열의 position번째줄 모든 칸
    #np.amax(Q[position, :]) :Q배열의 position 번째줄 모든 칸에서 최대값 리턴
    m=np.amax(Q[position, :])
    #최대값이 1일때
    if m==1:
        #np.argmax(Q[position, :]): Q배열의 position 번째줄 모든 칸에서 최대값이 저장된 인덱스 리턴 
        action=np.argmax(Q[position, :])
    else:#최대값이 1이 아닐때 , 최대값이 0
        #pr.choice([0,1,2,3]) : 0,1,2,3 중에서 하나의 수를 랜덤하게 선택해서 리턴
        action=pr.choice([0,1,2,3])
    #env.step(action): action을 실행
    new_position, reward,done,info = env.step(action)
    #reward+np.max(Q[new_position, :]): 현재 reward+Q배열의 position 번째줄 모든 칸에서 최대값 (np.max(Q[new_position, :]) )
    #을 Q[position,action] 에 대입
    Q[position,action]=reward+np.max(Q[new_position, :]) #인공지능에서는 조건문을 형식을 사용하는 스타일로 변경 
    #new_position을 position에 대입
    position=new_position
    #done : 게임 종료 여부를 저장 할 변수 
    #       True : 게임 종료
    #       False : 게임 계속 진행
    if done==True:
        break;
```



**다. 최적화 경로를 위한 Q 함수를 이용한 Frozen Lake**

```python
#(중복코드 생략)
#300번 동안 Q배열의 값을 계속적으로 갱신하면서 최적의 경로를 찾음
dis=0.99
for index in range(300):
    print("="*100)
    print("index:",index)
    position = env.reset()
    
    while True:
        #env.render(): Frozen Lake 게임을 출력
        env.render()
        print()
        #Q[position, :] : Q배열의 position번째줄 모든 칸
        #np.amax(Q[position, :]) :Q배열의 position 번째줄 모든 칸에서 최대값 리턴
        e= 1.0/((index//100)+1)

        # m=np.amax(Q[position, :])
        #난수가 e보다 작으면
        if np.random.rand(1) < e:
            #np.argmax(Q[position, :]): Q배열의 position 번째줄 모든 칸에서 최대값이 저장된 인덱스 리턴 
            action=pr.choice([0,1,2,3])
        else:#최대값이 1이 아닐때 , 최대값이 0
            #pr.choice([0,1,2,3]) : 0,1,2,3 중에서 하나의 수를 랜덤하게 선택해서 리턴
            action=np.argmax(Q[position, :])
        print("action:", action)

        #env.step(action): action을 실행
        new_position, reward,done,info = env.step(action)
        print("new_position:",new_position,":reward:",reward,":done:",done,":info:",info)
        #reward+np.max(Q[new_position, :]): 현재 reward+Q배열의 position 번째줄 모든 칸에서 최대값 (np.max(Q[new_position, :]) )
        #을 Q[position,action] 에 대입
        Q[position,action]=reward+dis*np.max(Q[new_position, :])
        print("="*100)
        print(Q[position,action])
        #new_position을 position에 대입
        position=new_position
        #done : 게임 종료 여부를 저장 할 변수 
        #       True : 게임 종료
        #       False : 게임 계속 진행
        if done==True:
            break;

    print("Q:",Q)
    print("="*100)
    
#Q: [[0.94148015 0.95099005 0.93206535 0.94148015]
# [0.94148015 0.         0.         0.93206535]
# [0.93206535 0.         0.         0.        ]
# [0.         0.         0.         0.        ]
# [0.95099005 0.96059601 0.         0.94148015]
# [0.         0.         0.         0.        ]
# [0.         0.         0.         0.        ]
# [0.         0.         0.         0.        ]
# [0.96059601 0.         0.970299   0.95099005]
# [0.96059601 0.         0.9801     0.        ]
# [0.970299   0.99       0.         0.        ]
# [0.         0.         0.         0.        ]
# [0.         0.         0.         0.        ]
# [0.         0.         0.         0.        ]
# [0.         0.99       1.         0.9801    ]
# [0.         0.         0.         0.        ]]    
    
    
#(중복 코드 생략)
```

![Imgur](https://i.imgur.com/tnBJbKC.png)

**라. 슬립기능이 있는 상태에서  Frozen Lake**

```python
#(중복코드 생략)
dis=0.99
#Goal에 도달 여부를 저장할 리스트
successList=[]

for index in range(2000):
    print("="*100)
    print("index:",index)
    position = env.reset()

    totalReward=0 #마지막에 결국 리워드를 받았는지 확인하기 위한 변수/ 중간에 스립에 걸려 초기화되는 경우는 0, 마지막에 리워드에 도착하면 1이 됨
    while True:
        #env.render(): Frozen Lake 게임을 출력
        env.render()
        print()
        #Q[position, :] : Q배열의 position번째줄 모든 칸
        #np.amax(Q[position, :]) :Q배열의 position 번째줄 모든 칸에서 최대값 리턴
        e= 1.0/((index//100)+1)

        # m=np.amax(Q[position, :])
        #난수가 e보다 작으면
        if np.random.rand(1) < e:
            #np.argmax(Q[position, :]): Q배열의 position 번째줄 모든 칸에서 최대값이 저장된 인덱스 리턴 
            action=pr.choice([0,1,2,3])
        else:#최대값이 1이 아닐때 , 최대값이 0
            #pr.choice([0,1,2,3]) : 0,1,2,3 중에서 하나의 수를 랜덤하게 선택해서 리턴
            action=np.argmax(Q[position, :])
        print("action:", action)

        #env.step(action): action을 실행
        new_position, reward, done, info = env.step(action)
        totalReward += reward
        print("totalReward:",totalReward, "new_position:",new_position,":reward:",reward,":done:",done,":info:",info)
        #reward+np.max(Q[new_position, :]): 현재 reward+Q배열의 position 번째줄 모든 칸에서 최대값 (np.max(Q[new_position, :]) )
        #을 Q[position,action] 에 대입
        Q[position,action]=reward+dis*np.max(Q[new_position, :])
        #new_position을 position에 대입
        position=new_position
        #done : 게임 종료 여부를 저장 할 변수 
        #       True : 게임 종료
        #       False : 게임 계속 진행
        if done==True:
            break;
    successList.append([index,totalReward])
    print("Q:",Q)
    print("="*100)
   

successList 
#[[0, 0.0],
# [1, 0.0],
# [2, 0.0],
# [3, 0.0],(생략) 2000개 리스트가 있는 2차원 리스트

resultList=list(zip(*successList))
#2개 리스트가 있는 2차원 리스트 (인덱스와 리워드를 각각 리스트에 넣음)


sum(resultList[1])
#61.0 2000번 중 총 61번 성공함
```



**마. 부록 ZIP함수**

```python
a =[1,2,3,4,5]
b = ['사과','배','바나나','딸기','수박']
c=zip(a, b)
print("list(c):",list(c))


d= list(zip(a,b))
print("d:",d)
#d: [(1, '사과'), (2, '배'), (3, '바나나'), (4, '딸기'), (5, '수박')]
```

