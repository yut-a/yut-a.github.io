---
layout: post
title:  K리그 축구 경기 데이터 Crawling with Selenium
date:   2020-10-18
image:  access.jpg
tags:   Data
---
## 크롤링(Crawling)이란?

웹사이트에 존재하는 각종 정보를 자동적으로 수집해오는 것을 말한다. 다양한 분석 도구를 적용하기 위해 필요한 데이터들이 항상 준비되어 있지 않다. 데이터 수집을 위해 다양한 매체에서 데이터를 수집해와야 하는데, 크롤링은 그 중 유용한 수집 도구 중 하나라고 할 수 있다.<BR/><BR/><BR/><BR/>

## 도구 소개

BeautifulSoup에 이어 이번에는 `Selenium`을 통해 크롤링을 해보고자 한다. Selenium은 웹 어플리케이션을 테스트할 때 사용하는 프레임워크로, webdriver라는 api를 통해 Chrome 브라우저를 제어할 수 있다. 일부 사이트들은 사용자가 직접 동작시키지 않으면 정보가 공개되지 않도록 설정되어 있기 때문에, 이러한 사이트에서는 태그를 기반으로 정보를 가져오는 것이 어렵다. 따라서, 동적 사이트에 대한 크롤링을 위해서는, 정적 사이트를 크롤링하는데 유용한 BeautifulSoup보다는 Selenium이 적절하다.

Selenium을 사용하기 위한 과정은 다음과 같다.

* [링크](https://sites.google.com/a/chromium.org/chromedriver/downloads)를 통해 자신의 Chrome 버전에 맞는 chromedriver를 설치한다. `설정 -> Chrome 정보`에서 Chrome 버전을 확인할 수 있다. 다른 방법은, 주소창에 `chrome://version/`라고 치면 확인할 수 있다.
<img width="1440" alt="스크린샷 2020-10-22 오전 1 18 33" src="https://user-images.githubusercontent.com/70478154/96748899-27cbed00-1405-11eb-8a72-428a3f954308.png">

* 다음의 코드를 활용하여 Selenium을 설치한다.

{% highlight ruby %}
pip install selenium
{% endhighlight %}

이러한 과정을 거치면, Selenium을 사용할 준비가 완료된 것이다.<BR/><BR/><BR/><BR/>

## 적용 과정

먼저, 기본적인 Selenium 사용 방법에 대해 알아보고자 한다.

Selenium의 webdriver를 불러와 Chrome을 제어하기 위한 `driver`를 만든다. Chrome의 괄호 안에 들어간 코드는 본인 PC의 chromedriver 위치를 의미한다. `implicitly_wait`는 암묵적으로 웹 자원들이 모두 로드될 때까지 기다리게 한다. 

{% highlight ruby %}
# Selenium의 webdriver 불러오기
from selenium import webdriver

driver = webdriver.Chrome("/Users/Downloads/chromedriver")
driver.implicitly_wait(3)       # 3초 대기
{% endhighlight %}

`get`을 이용하여 Chrome 작동을 제어할 수 있다.

{% highlight ruby %}
# google 접속
driver.get("https://google.com")

# 네이버증권 접속
driver.get("https://finance.naver.com/")
{% endhighlight %}

또, 다음과 같은 방법으로 로그인을 작동시킬 수 있다. `send_keys`의 큰 따옴표 안에 본인 아이디와 비밀번호를 입력하면 된다. `find_element_by_xpath`의 큰 따옴표 안에는 로그인 버튼의 `Copy full XPath`값을 입력하면 된다.

{% highlight ruby %}
# 네이버 로그인
driver.get("https://nid.naver.com/nidlogin.login")

# 아이디 / 비밀번호 입력
driver.find_element_by_name("id").send_keys("")
driver.find_element_by_name("pw").send_keys("")

# 로그인 버튼 클릭
driver.find_element_by_xpath("").click()
{% endhighlight %}

이상으로, 간단한 Selenium 사용법에 대해 알아보았다.

이제 본격적으로 Selenium을 통해 K리그 축구 경기 데이터를 크롤링 하고자 한다. 아래의 사이트에서 각 팀 별 `경기 스코어` `경기 기록` `포메이션` 정보를 불러올 것이다.
<img width="1440" alt="스크린샷 2020-10-22 오후 5 39 32" src="https://user-images.githubusercontent.com/70478154/96881533-55726e00-14b9-11eb-98be-f9957df61849.png">

<img width="1440" alt="스크린샷 2020-10-22 오후 5 40 01" src="https://user-images.githubusercontent.com/70478154/96881560-5c997c00-14b9-11eb-82c6-b5b5cb73064d.png">

<img width="1440" alt="스크린샷 2020-10-22 오후 5 40 29" src="https://user-images.githubusercontent.com/70478154/96881572-60c59980-14b9-11eb-9a88-c8b24e8cf680.png">

사이트 주소를 확인한 결과, 경기에 따라 주소 중간에 경기 코드가 바뀌는 것을 확인할 수 있다. 2020년 10월 17일의 성남 vs 서울 경기는 `80042793`로 되어 있다. 이 코드를 바꿔줌으로써 매 경기의 데이터를 불러올 수 있다.

먼저, 필요한 라이브러리들을 불러온 후, Chrome을 제어하기 위한 `driver`를 만든 후, 해당 사이트로 이동했다.

{% highlight ruby %}
# Selenium의 webdriver
from selenium import webdriver
import bs4

driver = webdriver.Chrome("/Users/Downloads/chromedriver")
driver.implicitly_wait(3)

# 2020.10.17 성남 vs 서울
driver.get("https://sports.daum.net/gamecenter/80042793/highlight")
{% endhighlight %}

다음은, Selenium과 BeaufifulSoup를 이용해 page source를 불러왔다.

{% highlight ruby %}
# page source
html = driver.page_source
soup = bs4.BeautifulSoup(html, "html.parser")
{% endhighlight %}

이전에 주식 종목 재무비율을 크롤링 할 때와는 다르게 데이터가 table 형태로 구성되어 있지 않다. 따라서, 필요한 데이터의 위치와 순서를 찾아 가져와야 한다. 아래와 같이 `검사`를 통해서 `팀 이름` 데이터의 구조를 파악했다.
<img width="1440" alt="스크린샷 2020-10-22 오후 11 10 59" src="https://user-images.githubusercontent.com/70478154/96884127-227da980-14bc-11eb-8981-da3e8df5a870.png">

확인한 결과, `성남`은 span 태그에 txt_team이라는 class인 것을 알 수 있다. 마찬가지의 방법으로 `서울`도 살펴보면 같은 구조를 가지는 것을 알 수 있다.
<img width="1440" alt="스크린샷 2020-10-22 오후 11 11 46" src="https://user-images.githubusercontent.com/70478154/96884143-24476d00-14bc-11eb-8e96-e03a16553c91.png">

`find_all`을 통해 데이터를 불러오면 첫 번째에는 `성남`, 두 번째에는 `서울` 데이터가 있는 것을 알 수 있다.

{% highlight ruby %}
# 팀 이름
team_1 = soup.find_all("span", class_="txt_team")[0].text
print(team_1)

team_2 = soup.find_all("span", class_="txt_team")[1].text
print(team_2)
{% endhighlight %}
<img width="36" alt="스크린샷 2020-10-22 오후 11 18 00" src="https://user-images.githubusercontent.com/70478154/96884843-f0207c00-14bc-11eb-85f9-c32e67c90a0e.png">

위와 같은 방법으로 `Score`의 위치를 찾아 다음과 같이 가져올 수 있다.

{% highlight ruby %}
# 스코어
score_1 = soup.find_all("span", class_="screen_out")[2].text
print(score_1)

score_2 = soup.find_all("span", class_="screen_out")[4].text
print(score_2)
{% endhighlight %}
<img width="37" alt="스크린샷 2020-10-22 오후 11 21 56" src="https://user-images.githubusercontent.com/70478154/96885380-7dfc6700-14bd-11eb-94e4-500700d5b6f3.png">

`경기 기록` 역시 같은 방법으로 위치를 찾아 불러올 수 있다. `성남`은 0과 짝수에, `서울`은 홀수에 위치해 있기 때문에 이를 구분하여 불러와 저장하면 된다.

{% highlight ruby %}
# 경기 기록
team_1_data = []
team_2_data = []

for i in range(0, len(soup.find_all("span", class_="vs_graph"))):
    graph = soup.find_all("span", class_="vs_graph")[i]
    data = graph.find("span", class_="num_g").text
    
    if i == 0:
        team_1_data.append(data)
        
    elif (i % 2) == 0:
        team_1_data.append(data)
    
    else:
        team_2_data.append(data)

print(team_1_data)
print(team_2_data)
{% endhighlight %}
<img width="408" alt="스크린샷 2020-10-22 오후 11 33 24" src="https://user-images.githubusercontent.com/70478154/96886926-04fe0f00-14bf-11eb-8108-4e8ed8b2d6c1.png">

마지막으로 포메이션까지 불러와, 팀 별로 데이터들을 list에 저장한다.

{% highlight ruby %}
# 포메이션
form_1 = soup.find_all("span", class_="team_info")[0].text
form_1 = form_1.split("\n")[4]
form_1

form_2 = soup.find_all("span", class_="team_info")[1].text
form_2 = form_2.split("\n")[4]
form_2
{% endhighlight %}
<img width="60" alt="스크린샷 2020-10-22 오후 11 34 58" src="https://user-images.githubusercontent.com/70478154/96887129-3c6cbb80-14bf-11eb-9fdc-a8ff42f86a2d.png">










