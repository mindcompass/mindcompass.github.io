---
title: 챗봇만들기_특강2(챗봇)
excerpt: "FLASK 연습"
toc: true
toc_sticky: true

categories:
  - chatbot
tags:
- Teleram
- python
- kakao번역
- 
use_math: true
last_modified_at: 2020-04-01
---



이번에는 본격적으로 텔레그램을 통해 챗봇을 만들어 보겠습니다.

kakao 번역 api를 통해서 자동으로 번역해주는 챗봇을 만들어보겠습니다.



## 1. 인증키 받기

- 텔레그램 PC 설치

  `@botfather` 검색

  `/newbot` 입력

  `[만들봇이름]_bot` 또는 `[만들봇이름]bot` 입력

  `API Key` 확인 후 따로 저장 (숫자:영문 형태로 구성)



## 2. API 기본 사용법

token = API Key

```
https://api.telegram.org/bot[봇토큰]/METHOD_NAME
```

나의 계정 ID 확인

봇으로 메시지 입력 후

```
https://api.telegram.org/bot[봇토큰]/getUpdates
```

텔레그램에서 개설한 챗봇에 대화를 하고 위 명령어를 다시 입력하면 대화 내용이 새롭게 업데이트 됨

여기서 from 다음에 내 텔레그램 id가 나타남

from > id 확인 후 따로 저장



## 3. sendMessage 실습

html 에서 입력받은 메시지를 텔레그램 봇으로 보내기

sendMessage 기본 사용법

```python
https://api.telegram.org/bot[봇토큰]/sendmessage?chat_id=[챗아이디]&text=[보낼메시지]
```

예를 들어" https://api.telegram.org/bot1132805451:AAG72kfrecgQqWGG7pDwEZxhe3jrRlyy-aY/sendmessage?chat_id=861147265&text=휠리리" 라고 전송하면 챗봇이 나에게 휠리리라는 메시지는 전송함


인증 키 보안 part1

1. python-decouple 설치

   ```
   $ pip install python-decouple
   ```

2. .env 파일 생성

   ```python
   CHAT_ID="000000000"
   TELEGRAM_BOT_TOKEN="111111111:gkjsdhgoihrgolwihflskhfsgkh"
   KAKAO_API='7e4wbaf4ae41d1259b2ab492fc432123' # kakao api 키는 필요한 경우 사용하세요.
   ```

3. python source 에서 위 데이터 사용

   app.py 에 아래 .env파일에 있는 키 자료를 사용하기 위해 아래 내용을 입력함

   ```
   token = config('TELEGRAM_BOR_TOKEN')
   chat_id = config('CHAT_ID')
   ```



a. html 에 form 작성

send.html 작성

```html
<form action="/send">
    <input type="text" name="text">
    <input type="submit" value="메세지 보내기">
</form>
```



wriete.html 작성

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>텔레그램 챗봇</title>
    <h1>텔레그램 챗봇에 입력할 내용을 넣어주세요.</h1>
    <form action="/send">
        <input type="text" name="text"> 
        <input type="submit">
    </form>
</head>
<body>
    
</body>
</html>
```



b. app.py 작성

```python
from flask import Flask, escape, request, render_template
import random

app = Flask(__name__)

token =config('TELEGRAM_BOT_TOKEN')
chat_id = config('CHAT_ID')
kakao_api=config('KAKAO_API') #kakao api 기능을 사용할 경우 .enve파일에 kakao키를 넣어두세요.
url = "https://api.telegram.org/bot"


@app.route('/')
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'

@app.route('/write')
def write():
    return render_template('write.html')

@app.route('/send')
def send():
    text = request.args.get('text')
    requests.get(f'{url}{token}/sendMessage?chat_id={chat_id}&text={text}')
    return render_template('send.html')

if __name__ == "__main__":
    app.run(debug=True)
```

인증키 보안 part2

git push 할때 일부 파일들을 제외할 수 있도록 gitignore 사용

gitignore.io 에서 venv, Flask, Python, Windows, VisualStudioCode 입력 후 생성버튼을 누르면 텍스트 파일이 화면에 출력

![gitignore](https://i.imgur.com/KRrMiqZ.png)

텍스트를 복사 후 .gitignore 파일 생성 후 붙여넣기

.gitignore 를 github에 push



## 4. webhook 실습

> ngrok 설치

https://ngrok.com/

download ngrok

> 실행

cmd 창을 열고

```
> ngrok.exe http 5000
```

![ngrok](https://i.imgur.com/59BaaNH.png)

아래와 같은 성공 화면이 뜨고 https://6ff3e494.ngrok.io 로 접속하면 나의 localhost:5000 로 포워딩 해준다는 메세지가 보인다.

텔레그램 setWebhook API 사용

> webhook.py 작성

```python
from decouple import config
import requests

token = config("TELEGRAM_BOT_TOKEN")
url = "https://api.telegram.org/bot"
ngrok_url = "https://6ff3e494.ngrok.io"

data = requests.get(f'{url}{token}/setWebhook?url={ngrok_url}/{token}')
print(data)
```

> webhook.py 실행

```
$ python webhook.py
<Response [200]>
```

200 (OK) 응답 오면 완료

webhook은 특정 토큰(내가 만든 챗봇)에 메시지가 전달되면  위에 있는 url에 접근하여 처리한다는 의미입니다.

즉 어떤 메시지가 내 챗봇에서 접속되면 그 정보는 바로 flask서버로 접속해서 처리하도록 하는 것입니다. 



## 5. KAKAO API를 통해 번역하기 

> app.py 내용에 추가

```python
@app.route(f'/{token}', methods=["POST"]) #post방식일때 무조건 실행. 아무나 못들어오게 토큰입력하고, post방식으로 들어오도록 함
def webhook():
    data=request.get_json() #플라스크 서버가 post 방식으로 받은 정보가 JSON 형식이므로 get_json을 통해 정보를 받음
    id=data['message']['chat']['id']
    message_text = data['message']['text']

    
    h={"Authorization": f"KakaoAK {kakao_api}"}
  
    response=requests.get(f'https://kapi.kakao.com/v1/translation/translate?src_lang=kr&target_lang=en&query={message_text}',headers=h)
    

    translated=response.json()["translated_text"][0][0]
  

    requests.get(f'{url}{token}/sendMessage?chat_id={id}&text={translated}')

    return "ok", 200
```



## 6. pythonanywhere

> webhook을 사용하여 로컬 작업하다가 외부에서도 사용할 수 있도록 pythonanywhere 사용해 봅니다.
>
> https://www.pythonanywhere.com/

가입 > web 탭 > add > flask > 3.7

Files 탭 > mysite 폴더

작업중인 app.py 붙여넣기

web 탭 > reload

console 탭 > bash > $ pip3 install python-decouple --user

아래처럼 제대로 설치되었는지 확인바랍니다. 

![](https://i.imgur.com/LOafHkp.png)



files 탭 > .env 생성 > 작업중인 내용 붙여넣기

> webhook.py 수정 및 재실행

```
paw_url = "https://kyuhyun85.pythonanywhere.com"
$ python webhook.py 
<Response [200]>
```



