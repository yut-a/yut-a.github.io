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

BeautifulSoup에 이어 이번에는 `Selenium`을 통해 크롤링을 해보고자 한다. selenium은 웹 어플리케이션을 테스트할 때 사용하는 프레임워크로, webdriver라는 api를 통해 Chrome 브라우저를 제어할 수 있다. 일부 사이트들은 사용자가 직접 동작시키지 않으면 정보가 공개되지 않도록 설정되어 있기 때문에, 이러한 사이트에서는 태그를 기반으로 정보를 가져오는 것이 어렵다. 따라서, 동적 사이트에 대한 크롤링을 위해서는, 정적 사이트를 크롤링하는데 유용한 BeautifulSoup보다는 Selenium이 적절하다.

Selenium을 사용하기 위한 과정은 다음과 같다.

* [링크](https://sites.google.com/a/chromium.org/chromedriver/downloads)를 통해 자신의 Chrome 버전에 맞는 webdriver를 설치한다. `설정 -> Chrome 정보`에서 Chrome 버전을 확인할 수 있다. 다른 방법은, 주소창에 `chrome://version/`라고 치면 확인할 수 있다.
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

driver = webdriver.Chrome("/Users/yut_a_/Downloads/chromedriver")
driver.implicitly_wait(3)       # 3초 기다림
{% endhighlight %}




