---
layout: post
title:  FLASK를 활용하여 간단한 어플리케이션 만들기
date:   2020-12-06
image:  twitter.jpg
tags:   Data
---
## FLASK란?

**FLASK**는 파이썬 웹 어플리케이션을 만드는 프레임워크이다. 다양한 파이썬 웹 프레임워크들 중 FLASK는 `Micro Web Framework`라고 불린다. 간단하게 말하면, 최소한의 구성 요소와 요구사항을 제공하기 때문에 사용하기 쉽고, 필요에 따라 유연하게 사용할 수 있는 프레임워크이다. 이러한 장점이 있는 반면, 제공하는 기능이 상대적으로 덜하고, 복잡한 어플리케이션 제작 시, 처리해야 할 것들이 많다는 단점이 있다. 따라서, 모든 도구들이 그러하듯 필요에 따라 프레임워크를 선택한다면, 효율적으로 웹 어플리케이션을 만들 수 있을 것이다.<BR/><BR/><BR/><BR/>

## 어플리케이션 소개

FLASK를 통해 만들고자 하는 어플리케이션은 다음과 같다. Twitter API를 통해 유저이름, 팔로워 수, 트윗 등을 데이터베이스에 저장하여, 웹 페이지에서 해당 유저에 대한 정보와 트윗들을 보여주고 삭제하는 기능을 가진다. 또한, 트윗과 같은 Text 데이터를 벡터화하여 LogisticRegression을 통해 학습시키고, 임의의 Text와 비교할 두 유저를 입력하면 어떤 유저의 트윗일지를 예측하는 기능을 가진다.<BR/><BR/>

#### 간단한 기능들

먼저, 웹 어플리케이션의 홈 화면은 다음과 같다. 다른 탭에 있다가 `Home` 버튼을 누르면 다시 해당 화면으로 돌아올 수 있다.
<img width="1440" alt="스크린샷 2020-12-07 오전 3 18 47" src="https://user-images.githubusercontent.com/70478154/101288653-fd11e880-383a-11eb-805f-49f5b3153d4d.png">

`Add`는 Twitter API를 통해 원하는 유저의 정보를 불러올 수 있는 기능이다. 불러오기를 원하는 유저의 username을 입력하고 Add 버튼을 누르면 complete이라는 문구가 뜨면서 해당 유저의 정보가 추가된다.
<img width="1440" alt="스크린샷 2020-12-07 오전 3 23 31" src="https://user-images.githubusercontent.com/70478154/101288764-9d680d00-383b-11eb-9277-45734b9ee22e.png">

앞서 `Add`를 통해 추가한 유저의 정보는 `User`에서 확인할 수 있다. 간단하게 4개의 유저를 불러왔고, 다음과 같이 유저의 정보들을 확인할 수 있다.
<img width="1440" alt="스크린샷 2020-12-07 오전 3 20 23" src="https://user-images.githubusercontent.com/70478154/101288800-c8526100-383b-11eb-9690-39fafc7dba5d.png">

또한, `get`을 통해, 불러온 유저의 트윗들을 확인할 수 있다. 원하는 유저의 트윗을 보기 위해서는 먼저 Add를 통해 유저 정보를 불러와야 한다.
<img width="1440" alt="스크린샷 2020-12-07 오전 3 26 18" src="https://user-images.githubusercontent.com/70478154/101288864-3565f680-383c-11eb-82ab-a3c7160dc6cd.png">

`delete`는 불러온 유저의 정보를 삭제해주는 기능을 한다. `update`는 username을 원하는대로 변경할 수 있다. username에 변경하길 원하는 username을 입력하고, new_name에 새로운 username을 입력하면 입력한 값으로 바뀌며, User 탭에서 바뀐 결과를 확인할 수 있다.
<img width="1440" alt="스크린샷 2020-12-07 오전 3 32 00" src="https://user-images.githubusercontent.com/70478154/101288983-cb9a1c80-383c-11eb-8c9c-e2cf457a1bc8.png">

`compare`는 선택한 두 유저 중, 입력한 Text가 어느 유저의 트윗일지 예측해준다. 다음과 같이 두 유저에 adidas와 한국은행을 입력하고, 예측에 사용할 Text를 dollar로 입력했다. 예측 결과, dollar라는 Text는 한국은행의 트윗일 것이란 예측 결과를 확인할 수 있다.
<img width="1440" alt="스크린샷 2020-12-07 오전 3 36 35" src="https://user-images.githubusercontent.com/70478154/101289065-6eeb3180-383d-11eb-99c2-1fb177cb7e90.png"><BR/><BR/><BR/><BR/>

## 간단한 적용 과정

먼저, 다음과 같이 Flask를 설정하고, 데이터베이스를 연결한다. 기존에 데이터베이스가 없을 경우, 빈 데이터베이스로 생성되어 연결된다.

{% highlight ruby %}
# init
from flask import Flask

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///twitter_app.sqlite3"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
{% endhighlight %}

다음, 연결한 데이터베이스에 테이블과 칼럼을 설정한다. 테이블의 기본키와 외래키 역시 설정해준다. 데이터베이스의 스키마는 아래와 같다.

{% highlight ruby %}
# database
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

class User(db.Model):
    __tablename__ = "User"
    id = db.Column(db.BigInteger, primary_key = True)
    username = db.Column(db.String)
    full_name = db.Column(db.String)
    followers = db.Column(db.Integer)

    def __repr__(self):
        return "< User {} {} >".format(self.id, self.username)

class Tweet(db.Model):
    __tablename__ = "Tweet"
    id = db.Column(db.BigInteger, primary_key = True)
    text = db.Column(db.String)
    embedding = db.Column(db.PickleType)
    user_id = db.Column(db.BigInteger, db.ForeignKey("User.id"))

    user = db.relationship("User", backref = db.backref("Tweet", lazy = True))

    def __repr__(self):
        return "< Tweet {} >".format(self.id)
{% endhighlight %}
<img width="559" alt="스크린샷 2020-12-07 오전 3 48 08" src="https://user-images.githubusercontent.com/70478154/101289324-1321a800-383f-11eb-9003-c106008b53f3.png">

그 후, 위에서 언급한 `Add` `get` `compare` 등 원하는 기능을 수행하도록 route를 구성한다.<BR/><BR/><BR/><BR/>

## 결론

이러한 웹 어플리케이션을 만들면서 맞닥뜨렸던 가장 큰 어려움은 `오류`였다. 열심히 작성한 코드를 실행했을 때, 도무지 알아들을 수 없는 오류가 발생하면서 하얀 웹 페이지에 `ERROR`라는 글자가 나타나는 것이 가장 힘들었다. 코드 한 줄의 결과가 바로 눈 앞에 보이는 것이 아니기 때문에 어떤 부분에서 오류가 발생했는지 바로 파악하기 어려웠다. 이러한 문제는 코드가 길어질수록 더 빈번하게 발생했다. 당연한 이야기일 수 있겠지만, 이를 보완하기 위해 본격적으로 코드를 작성하기 전, 함수의 형태가 아닌 간단한 예시의 개별 코드를 작성하여 결과를 먼저 확인했다. 또한, `breakpoint`를 애용했다. breakpoint의 위치를 옮겨가면서 입력한 코드의 값들을 확인하고, 어떤 부분에서 오류가 발생했는지 파악했다. 비록 시간이 오래걸리기는 했지만, 이러한 과정을 통해서 오류를 해결하며 천천히 진행할 수 있었다.

간단한 웹 어플리케이션을 만들면서, 평소에 자주 사용하던 웹 사이트의 버튼들이 어떤 식으로 작동하는지 조금이나마 이해할 수 있는 시간이었다. 기회가 된다면, 더 많은 기능을 구현할 수 있는, 완성도 높은 어플리케이션을 만들어보고 싶다.
