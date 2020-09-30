---
layout: post
title:  주식 종목 재무비율 Crawling with BeautifulSoup
date:   2020-09-30
image:  information.jpg
tags:   Data
---
## 크롤링(Crawling)이란?

웹사이트에 존재하는 각종 정보를 자동적으로 수집해오는 것을 말한다. 다양한 분석 도구를 적용하기 위해 필요한 데이터들이 항상 준비되어 있지 않다. 데이터 수집을 위해 다양한 매체에서 데이터를 수집해와야 하는데, 크롤링은 그 중 유용한 수집 도구 중 하나라고 할 수 있다.<BR/><BR/><BR/><BR/>

## 도구 소개

데이터 수집을 위한 다양한 크롤링 방법이 존재하겠지만, 여기에서는 `BeautifulSoup` 라이브러리를 활용하고자 한다. 특히, 데이터프레임으로 데이터를 수집하기 위해서 `html_table_parser`도 함께 사용한다.

필요한 라이브러리는 설치 후 사용할 수 있다. 하지만, 설치 후 라이브러리를 import 할 경우 다음과 같은 오류가 나타날 수 있다.<BR/>
`module 'html5lib.treebuilders' has no attribute '_base'`

이를 해결하기 위해 다음과 같은 과정이 필요하다.

{% highlight ruby %}
!pip install html_table_parser
{% endhighlight %}

{% highlight ruby %}
pip install --upgrade html5lib
{% endhighlight %}

{% highlight ruby %}
pip install --upgrade beautifulsoup4
{% endhighlight %}

이러한 과정을 거친다면, 무리없이 필요한 라이브러리들을 불러와 사용할 수 있을 것이다.<BR/><BR/><BR/><BR/>

## 적용 과정

크롤링을 통해, 다음의 재무 비율 데이터를 수집하고자 한다.

<img width="994" alt="스크린샷 2020-10-01 오전 1 30 01" src="https://user-images.githubusercontent.com/70478154/94713429-bbba1400-0385-11eb-906d-7c974c7ee19a.png">

크롤링을 하기 위해서 `url`을 확인했다. 확인한 결과, 사이트 주소 중간의 종목 코드, 예를 들어 삼성전자는 `A005930`만 종목에 따라 바뀌는 것을 확인할 수 있다.

<img width="1440" alt="스크린샷 2020-10-01 오전 1 10 06" src="https://user-images.githubusercontent.com/70478154/94711810-94fade00-0383-11eb-9718-dd5e6aa06125.png">

따라서, 다음과 같이 url을 설정해주었다.

{% highlight ruby %}
# 삼성전자 코드
stock_code = "A005930"
url = "http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=" + stock_code + "&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701"
{% endhighlight %}

필요한 라이브러리들을 불러온 후, `urlopen`을 이용하여 url의 소스코드를 긁어온다. 긁어온 소스코드를 보기 좋게 해석하기 위해, `BeautifulSoup`를 이용하여 해석한다.

{% highlight ruby %}
from urllib.request import urlopen
from html_table_parser import parser_functions as parser
import pandas as pd
import bs4
import numpy as np

source = urlopen(url).read()                  # url의 소스코드 긁어오기
source = bs4.BeautifulSoup(source, "lxml")    # 소스코드 해석
{% endhighlight %}

