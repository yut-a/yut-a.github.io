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

FLASK를 통해 만들고자 하는 어플리케이션은 다음과 같다. Twitter API를 통해 유저이름, 팔로워 수, 트윗 등을 데이터베이스에 저장하여, 웹 페이지에서 해당 유저에 대한 정보와 트윗들을 보여주고 삭제하는 기능을 가진다. 또한, 트윗과 같은 Text 데이터를 벡터화하여 LogisticRegression을 통해 학습시키고, 임의의 Text와 비교할 두 유저를 입력하면 어떤 유저의 트윗일지를 예측하는 기능을 가진다.

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

## 적용 과정

