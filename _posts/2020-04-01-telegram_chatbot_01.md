---

title:  챗봇만들기_특강1(기초)
excerpt: python에서 가상환경만들기,가상환경실습, gitignore사용,FLASK 연습
toc: true
toc_sticky: true

categories:
  - programming
tags:
- FLASK
- python
- python_venv
- gitignore
use_math: true
last_modified_at: 2020-04-01
---



올해 초 멀티캠퍼스 특강에서 배웠던 chatbot만들기 수업을 정리하려고 합니다. 





## 1. Python 가상환경

- 파이썬에서는 한 라이브러리에 대해 하나의 버전만 설치가 가능합니다.

- 여러개의 프로젝트를 진행하게 되면 이는 문제가 됩니다. 파이썬의 경우 각각의 라이브러리들 간의 버전이 출동하면 작동하지 쉽게 에러가 나기 때문에 작업을 바꿀때마다 다른 버전의 라이브러리를 설치해야합니다.

- 이를 방지하기 위한 격리된 독립적인 가상환경을 제공합니다.

- 일반적으로 프로젝트마다 다른 하나의 가상환경을 생성한 후 작업을 시작하게 됩니다.

- 가상환경의 대표적인 모듈은 3가지가 있습니다

  

  - **venv : Python 3.3 버전 이후 부터 기본모듈에 포함됨**
  - virtualenv : Python 2 버전부터 사용해오던 가상환경 라이브러리, Python 3에서도 사용가능
  - conda : Anaconda Python을 설치했을 시 사용할 수있는 모듈
  - pyenv : pyenv의 경우 Python Version Manager임과 동시에 가상환경 기능을 플러그인 형태로 제공



venv는 python3에서 기본으로 제공하는 가상환경(virtualenv)을 만들 수 있는 라이브러리여서 따로 설치할 필요가 없음.



본격적으로 작업하기 위해 VSCODE를 사용하고 아래 내용은 터미널을 띄워서 작업합니다.



1) 가상환경 생성/ 활성화

```python
# window 
# python -m venv [venv 이름] 
$ python -m venv venv # venv를 해당위치에 venv라는 가상환경을 만들 겁니다.

# ls 명령어로 venv가 설치되었는지 확인해보세요.

# 내가 만든 가상환경을 활성화(사용) 할 겁니다.
$ source venv/Scripts/activate
 #cmd창에서 하려면 venv\Scripts\activate로 입력하세요.


deactivate #가상환경이 활성화된 상태에서 비활성화로 변경합니다.

pip list # 내 가상환경에 설치된 라이브러리와 그 버전을 확인합니다.

$ rm -rf venv/ #가상환경을 삭제합니다.
```



| Tip.

`pip` : 파이썬 패키지 관리 시스템

pip freeze > requirements.txt

pip install -r requirements.txt



## 2. 가상환경에서 크롤링 연습하기

venv) 가상환경 내에서

requests 설치

```
$ pip install requests
```

beautifulsoup4 설치

```
$ pip install beautifulsoup4
```



1. KOSPI 정보 가져오기

```python
import requests
from bs4 import BeautifulSoup

url = "http://finance.naver.com/sise"

# req = requests.get(url)
# print(req)
req = requests.get(url).text
# print(req)

soup = BeautifulSoup(req, 'html.parser')
kospi = soup.select_one("#KOSPI_now")

print(kospi.text)
```



2.환율 정보 가져오기

```python
import requests
from bs4 import BeautifulSoup

url = "https://finance.naver.com/marketindex/"

req = requests.get(url).text

soup = BeautifulSoup(req, 'html.parser')
exchange = soup.select_one("#exchangeList > li.on > a.head.usd > div > span.value")

print(exchange.text)
```



3. 네이버 헤드라인 뉴스 가지고 오기

```python
import requests
from bs4 import BeautifulSoup

url = "https://news.naver.com/"

req = requests.get(url).text

soup = BeautifulSoup(req, 'html.parser')

search = soup.select("#today_main_news > div.hdline_news > ul >li > div.hdline_article_tit > a")

for item in search :
    print(item.text)
```

Tip.

영화 정보 API

https://www.themoviedb.org/?language=ko-KR

https://developers.themoviedb.org/3/getting-started/introduction

텔레그램 API

https://core.telegram.org/api



## 3. gitignore 활용하기

원격저장소에 프로젝트 및 수정 파일을 올릴 때, 올리고 싶지 않은 파일 및 폴더를 설정하여 제외시키는 목적으로 사용



https://www.gitignore.io/

파일을 생성한 뒤에 해당 파일을 git파일 위치에 넣어두면, git에 파일을 push할 때  설정했던 파일은 자동으로 push되는 파일에서 제외됩니다.

![gitignore](https://i.imgur.com/SQDfM2Z.png)

##  4. Flask 연습하기



Flask는 Python으로 구동할 수 있는 서버 프레임워크입니다.

### 1. flask 설치

```
$ pip install flask==1.0.0
```

> hello.py

```python
from flask import Flask, escape, request

app = Flask(__name__)

@app.route('/')
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'

```

```python
$ env FLASK_APP=hello.py flask run
```



### 2. 예제

#### 1. 기본 구조

app.py 가 아닌경우 아래와 같이 작업 후 실행 가능

```python
from flask import Flask, escape, request, render_template
import random

app = Flask(__name__)

@app.route('/')
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'

... 여기에 추가 작업 ...


if __name__ == "__main__":
    app.run(debug=True)
```
#### 2. hello.py

##### hello.py 추가 작업 내용 1

```python
@app.route('/hi')
def hi():
    # return "반갑습니다."
    # return "<h1>Hello</h1>"
    name = "mindcompass_mune"
    return render_template('hi.html', html_name = name)
```
hello.py 같은 폴더 내에 templates이라는 폴더를 만들고 거기에 hi.html파일을 생성한다.

hi.html

![](https://i.imgur.com/gZbtPt1.png)

##### hello.py 추가 작업 내용 2

```python
@app.route('/greeting/<string:name>')
def greeting(name):
    def_name = name
    return render_template('greeting.html', html_name = def_name)
```

greeting.html

![greeting.html](https://i.imgur.com/MzceAqd.png)

##### hello.py 추가 작업 내용 3

```python
@app.route('/cube/<int:num>')
def cube(num):
    cube_num = num**3
    return render_template('cube.html', cube_num = cube_num, num = num)
```
cube.html

![cube.html](https://i.imgur.com/CaRGhsR.png)



##### hello.py 추가 작업 내용 4

```python
@app.route('/dinner')
def dinner():
    menu = ['삼각김밥', '컵라면', '스테이크', '마라탕', '훠궈']
    menu_img = {'삼각김밥' : 'http://recipe1.ezmember.co.kr/cache/recipe/2018/08/06/087c110e0149b1ce06b22fcc765d5694.jpg',
                '컵라면' : 'https://cdn.crowdpic.net/detail-thumb/thumb_d_378BA60B966894DC61DCEC443E424FA3.jpg',
                '스테이크' : 'http://recipe1.ezmember.co.kr/cache/recipe/2017/07/09/6741acc7f6bf0f7d04245851fb365c311.jpg',
                '마라탕' : 'https://t1.daumcdn.net/cfile/tistory/9990F0395BBD84EA01',
                '훠궈' : 'https://funshop.akamaized.net/products/0000062075/vs_image800.jpg'
                }

    dinner = random.choice(menu)
    return render_template('dinner.html', html_dinner = dinner, html_url = menu_img[dinner])

```

dinner.html

![dinner.html](https://i.imgur.com/8XLrB4n.png)



##### hello.py 추가 작업 내용 5

```python
@app.route('/movies')
def movies():
    movies = ['조커', '겨울왕국2', '터미네이터', '어벤져스']
    return render_template('movies.html', html_movies = movies)
```

movies.html

![movies.html](https://i.imgur.com/f68JBLi.png)



#### 3. ping_pong.py

##### ping_pong.py 추가 작업 내용 1 and 2

```python
@app.route('/ping')
def ping():
    return render_template('ping.html')

@app.route('/pong')
def pong():
    # data = request.form.get('keyword')    # post 방식은 form.get()
    data = request.args.get('keyword')  # get 방식은 args.get()
    return render_template('pong.html', html_data = data)
```

ping.html

```html
<body>
<h1>Here is Ping!!</h1>
<form action="/pong">
<input type="text" name="keyword">
<input type="submit">
</form>
</body>
```

 pong.html

![pong.html](https://i.imgur.com/VZez54L.png)

##### ping_pong.py 추가 작업 내용 3 and 4

 ```python
@app.route('/naver')
def naver():
    data = request.args.get('query')
    return render_template('naver.html', html_data = data)

@app.route('/google')
def google():
    data = request.args.get('q')
    return render_template('google.html', html_data = data)
 ```

naver.html

 ```html
 <body>
  <h1>Naver</h1>
  <form action="https://search.naver.com/search.naver">
      <input type="text" name="query">
      <input type="submit">
  </form>
 </body>
 ```

 google.html

 ```html
 <body>
  <h1>Google</h1>
  <form action="https://www.google.com/search">
      <input type="text" name="q">
     <input type="submit">
  </form>
</body>
 ```



#### 4.ping_pong.py(연습문제)

```python
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)

@app.route('/search')
def search():
    return render_template('search_opgg.html')

@app.route('/opgg')
def opgg():
    userName = request.args.get('userName')
    url = f"http://www.op.gg/summoner/userName={userName}"
    req = requests.get(url).text
    soup = BeautifulSoup(req, 'html.parser')
    
    tier = soup.select_one('#SummonerLayoutContent > div.tabItem.Content.SummonerLayoutContent.summonerLayout-summary > div.SideContent > div.TierBox.Box > div > div.TierRankInfo > div.TierRank')

    win = soup.select_one('#SummonerLayoutContent > div.tabItem.Content.SummonerLayoutContent.summonerLayout-summary > div.SideContent > div.TierBox.Box > div > div.TierRankInfo > div.TierInfo > span.WinLose > span.wins')

    return render_template('opgg.html', name = userName, tier = tier.text, win = win.text[: - 1], url = url)

if __name__ == ("__main__"):
    app.run(debug=True)
```



search_opgg.html

```html
<body>
 <h1>롤 아이디를 입력하세요 </h1>
 <form action="/opgg">
     <input type="text" name="userName">
     <input type="submit" value="검색">
 </form>
</body>
```



![opgg.html](https://i.imgur.com/jj6J1Ob.png)

위 코드 결과는 아래와 같습니다.

![](https://i.imgur.com/ANEFEtv.png)



![](https://i.imgur.com/seByn0W.png)