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

작업을 하기 위해 pymongo라는 패키지가 필요하며, 추가적으로 dnspython이라는 패키지 역시 필요하다.

{% highlight ruby %}
# 사용할 라이브러리
import csv
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
{% endhighlight %}

csv 파일을 읽어와 칼럼과 데이터를 저장하는 함수를 만들었다. `csv.reader`는 데이터를 row 단위로 읽어온다. 첫 번째 행은 칼럼 명이기 때문에 제외하고 두 번째 행부터 데이터들을 저장했다. 칼럼 명은 따로 저장했다. 

{% highlight ruby %}
# csv 파일 읽고, 칼럼과 데이터 저장
def retrieve_data(filename):

    columns = ['종목코드', '종목명', '현재가', '전일대비등락', '등락률', '거래대금', '시가총액']
    data = []

    with open(filename, "r") as f:
        csv_reader = csv.reader(f, delimiter = ",")

        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass

            else:
                data.append(row)
            
            line_count += 1
    
    return columns, data

columns, data = retrieve_data(filename)

print(columns)
print(data[:10])
{% endhighlight %}
<img width="1080" alt="스크린샷 2020-11-23 오전 3 10 57" src="https://user-images.githubusercontent.com/70478154/99911738-c2d42180-2d39-11eb-9ad6-89444d8a14a9.png">

columns와 data가 잘 저장되었음을 확인할 수 있다.

다음은, columns와 data를 documents로 저장했다. MongoDB에서는 데이터를 {'key' : 'value'}의 형태로 저장하기 때문에 columns과 data를 이런 형태로 만들어주었다.

{% highlight ruby %}
# documents로 저장
def create_documents(columns, data):
    doc_list = []

    for row in data:
        doc = {}

        for i in range(0, len(columns)):
            doc[columns[i]] = row[i]

        doc_list.append(doc)

    return doc_list

columns, data = retrieve_data(filename)
doc_list = create_documents(columns, data)
print(doc_list[:10])
{% endhighlight %}
<img width="1075" alt="스크린샷 2020-11-23 오전 3 33 19" src="https://user-images.githubusercontent.com/70478154/99913763-c4531900-2d3c-11eb-96f4-5d9fdd08330c.png">

[사이트](https://account.mongodb.com/account/login?signedOut=true)에서 `create database`를 통해 데이터베이스 이름과 collection 이름을 설정한다. 그 후, `.env`에 저장했던 개인 정보를 getenv를 통해 불러와 클러스터 URL을 저장했다. 앞서 설정한 데이터베이스와 collection에 연결한 후, 저장해두었던 documents를 추가하면 MongoDB에 다음과 같이 데이터가 저장되는 것을 확인할 수 있다.

{% highlight ruby %}
# 실행
if __name__ == "__main__":

    columns, data = retrieve_data(filename)
    doc_list = create_documents(columns, data)

    # Atlas에서 제공하는 클러스터 URL
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_USER = os.getenv("DB_USER")
    DB_SERVER = os.getenv("DB_SERVER")
    DB_NAME = os.getenv("DB_NAME")
    
    mongo_URL = f"""mongodb+srv://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}?retryWrites=true&w=majority"""
    
    # client 객체 생성
    client = MongoClient(mongo_URL)
    
    # 데이터베이스 선택
    db = client[DB_NAME]
    
    # collection 연결
    collection = db["kospi_stock"]

    # documents 추가
    collection.insert_many(doc_list)
{% endhighlight %}
<img width="876" alt="스크린샷 2020-11-23 오전 4 19 36" src="https://user-images.githubusercontent.com/70478154/99914790-2dd62600-2d43-11eb-94d9-cd3a1812ced9.png">



