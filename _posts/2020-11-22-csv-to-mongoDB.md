---
layout: post
title:  CSV 파일을 MongoDB에 옮기기
date:   2020-11-22
image:  database.jpg
tags:   Data
---
## NoSQL이란?

**SQL**은 Structured Query Language의 약자로, 구조화되어 있는 쿼리 언어라는 의미이다. 데이터들은 table에 record로 저장되며, 각 테이블에는 명확한 구조가 존재한다. 반면, **NoSQL**은 이름에서도 알 수 있듯이 구조화되지 않았음을 의미한다. NoSQL은 SQL과는 다르게, 구조가 다른 데이터들을 collection(SQL의 table)에 저장할 수 있다. 이러한 차이로, NoSQL의 장단점이 존재한다. 다음과 같은 특징으로, NoSQL은 정확한 데이터 구조를 알 수 없으며, 자주 변경, 확장되는 경우 사용하는 것이 바람직하다.

#### NoSQL 장점

* 스키마와 같은 명확한 구조가 존재하지 않기 때문에 유연하다.
* 데이터를 읽어오는 속도가 빠르다.
* 수평적 확장이 쉽다.

#### NoSQL 단점

* 데이터의 일관성이 보장되지 않는다.
* 복잡한 join이 어렵다.
* 데이터가 collection들에 중복되어 있는 경우, 데이터 업데이트 시 모든 collection의 데이터를 업데이트 해야 한다.

#### MongoDB란?

**MongoDB**는 이러한 NoSQL의 특징을 잘 반영한 범용 데이터베이스이다. NoSQL에 대한 잘 알려진 데이터베이스들 중 하나이다.<BR/><BR/><BR/><BR/>

## CSV 파일을 MongoDB에 옮기기

csv 형태의 파일을 데이터베이스로 저장하기 위한 방법을 알아보고자 한다. 사용한 csv 파일은 코스피 시장에 상장되어 있는 종목들의 리스트와 관련 거래 정보의 데이터들이다. 칼럼 명이 제대로 나와있지 않은데, 좌측부터 `종목코드` `종목명` `현재가` `전일 대비 등락` `등락률` `거래대금` `시가총액`을 나타낸다.
<img width="565" alt="스크린샷 2020-11-23 오전 2 13 00" src="https://user-images.githubusercontent.com/70478154/99910459-800e4b80-2d31-11eb-9116-f9edeb36d9de.png">

먼저, 다음과 같은 파이썬 라이브러리를 불러온다. `csv`와 `os`는 csv 파일을 불러오고, `.env`에 저장된 개인 정보를 읽어오는데 사용된다. `load_dotenv`는 `.env`의 내용을 load하는데 사용된다. `MongoClient`는 데이터를 옮기기 위해 MongoDB와 연결하기 위한 라이브러리이다.

{% highlight ruby %}
import csv
import os
from dotenv import load_dotenv
from pymongo import MongoClient
{% endhighlight %}











