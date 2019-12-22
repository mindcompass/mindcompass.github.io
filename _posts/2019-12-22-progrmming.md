---
title:  "venv을 활용하여 Python 가상환경 사용하기"
excerpt: "가상 환경을 사용하는 목적과 venv를 사용하는 방법 안내"
toc: true
toc_sticky: true

categories:
  - programming
tags:
- Python
- venv
- 가상화
- 라이브러리
last_modified_at: 2019-12-22
---

모든 프로그램 개발도구가 그렇지만 Python을 여러가지로 활용하기 위해서는 **다양한 라이브러리**가 필요합니다.



이런 다양한 라이브러리 기능이 상호작용을 통해 복잡한 프로그램이 돌아가게 됩니다.



그런데 문제는 이런 다양한 라이브러리들이 파이썬 버전과 충돌하거나 각 라이브러리가 충돌하면서 제대로 동작하지 않는 일이 빈번하게 일어납니다.



간단하게 혼자 작업하는 목적이라면 내 PC에서 잘 돌아같다면 큰 문제는 없겠지만.. 여럿이서 작업을 하는 상황이거나 상용화 프로그램을 만드는 과정이라면 이는 분명 큰 문제가 아닐 수 없습니다.



파이썬에서는 **하나의 pc에 한 라이브러리만 설치가 가능**하기 때문에 내 동료의 파이썬을 테스트하기 위해선 내 기존의 라이브러리 환경을 수정하고 새롭게 라이브러리를 설치해야 합니다.



내 컴퓨터에서든 내 동료 컴퓨터에서는 서버에서든 모두 동일한 구동 환경이 갖춰져 있다면, 다시 말해서 동일한 라이브러리가 설치되어 있다면 이런 문제가 발생하지 않겠죠?



이런 목적으로 우리는 가상환경을 설치하고 관리합니다.



python의 가상환경은 venv, virtulenv, conda, pyenv 등을 활용할 수 있습니다.



각각의 특징은 아래와 같습니다.

- venv : Python 3.3 버전 이후 부터 기본모듈에 포함됨
- virtualenv : Python 2 버전부터 사용해오던 가상환경 라이브러리, Python 3에서도 사용가능
- conda : Anaconda Python을 설치했을 시 사용할 수있는 모듈
- pyenv : pyenv의 경우 Python Version Manger임과 동시에 가상환경 기능을 플러그인 형태로 제공



이번에는 간단하게 venv를 활용한 가상화를 구현해보겠습니다.



특정 폴더를 가상환경으로 활용할 것이라면 해당 폴더로 이동하셔서 아래와 같은 명령문을 작성하세요.

작성한 폴더에 가시면 venv라는 폴더가 생성된 것을 보실 수 있습니다.

```python
python -m venv venv
```

여기서 첫번째 venv는 우리가 사용할 가상환경모듈을 의미하며 뒤에 venv는 이 가상환경을 지칭할 이름입니다.

보통 그냥 venv를 사용해서 동일하게 작성했습니다.

```python
source venv/Scripts/activate #가상환경을 on합니다.
deactivate # 가상환경을 off합니다.
```

가상환경을 활성화 하거나 끄는 명령문입니다.

아래는 venv 가상환경을 제거하는 명령문입니다.

```python
rm -rf venv # 가상환경 제거
```



현재 작업 중인 파이썬 라이브러리 환경을 별도의 txt파일에 저장하면 내가 어떤 환경에서 프로그램을 작동했는지  requirments.txt파일에 저장되게 됩니다.

```python
pip freeze > requirements.txt #pip freeze 수행한 결과를 requirements.txt에 저장
```



만약 내 동료가 자신이 작업한 python 작업물과 requirement.txt 파일을 함께 주게 되면, 나는 해당 폴더에 venv가상화를 실시하고 아래 명령어를 통해 라이브러리를 한번에 자동으로 인스톨하여 동료pc와 동일한 가상환경을 구축할 수 있게 됩니다.

가상화하려는 폴더 안에 requirments.txt파일을 복사하고 아래 명령어를 실행합니다.

```python
pip install -r requirements.txt #requirements.txt에 저장된 라이브러리 목록을 인스톨 한다.
```

설치를 하면 venv 폴더 안에 Lib 폴더 안에 site-packages 폴더 안에 해당 라이브러리가 설치된 것을 확인하실 수있습니다.
