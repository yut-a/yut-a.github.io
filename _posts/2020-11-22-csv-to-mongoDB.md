---
layout: post
title:  csv 파일을 MongoDB에 옮기기
date:   2020-11-22
image:  database.jpg
tags:   Data
---
## NoSQL이란?

**SQL**은 Structured Query Language의 약자로, 구조화되어 있는 쿼리 언어라는 의미이다. 데이터들은 table에 record로 저장되며, 각 테이블에는 명확한 구조가 존재한다. 반면, **NoSQL**은 이름에서도 알 수 있듯이 구조화되지 않았음을 의미한다. NoSQL은 SQL과는 다르게, 구조가 다른 데이터들을 collection(SQL의 table)에 저장할 수 있다. 이러한 차이로, NoSQL의 장단점이 존재한다. 다음과 같은 특징으로, NoSQL은 정확한 데이터 구조를 알 수 없으며, 자주 변경, 확장되는 경우 사용하는 것이 바람직하다.<BR/><BR/><BR/><BR/>

#### NoSQL 장점

* 스키마와 같은 명확한 구조가 존재하지 않기 때문에 유연하다.
* 데이터를 읽어오는 속도가 빠르다.
* 수평적 확장이 쉽다.

#### NoSQL 단점

* 데이터의 일관성이 보장되지 않는다.
* 복잡한 join이 어렵다.
* 데이터가 collection들에 중복되어 있는 경우, 데이터 업데이트 시 모든 collection의 데이터를 업데이트 해야 한다.

#### MongoDB란?

**MongoDB**는 이러한 NoSQL의 특징을 잘 반영한 범용 데이터베이스이다. NoSQL에 대한 잘 알려진 데이터베이스들 중 하나이다.

