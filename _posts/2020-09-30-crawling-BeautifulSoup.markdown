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
