---
layout: post
title:  타이타닉 데이터 분석 - 생존과 사망에 가장 영향을 미치는 변수는 무엇일까
date:   2020-09-04
image:  05.jpg
tags:   Data
---
### 주제 선정 동기와 탐구 목적

* 앞서 동기분들께서 다양한 시각화 자료와 분석으로, 어떤 종류의 승객이 생존 가능성이 높았는지, 예상과 달랐던 흥미로운 결론 등 이미 여러 종류의 blogging을 해주셨다.
* 이에 따라, 다른 방식의 분석을 시도해보고자 했다.
* 어떤 종류의 승객이 생존 가능성이 더 높은 지에서 더 나아가, 변수들 중 어떤 변수가 가장 생존율/사망률에 영향을 미치는지 탐구해보고자 했다.


### 탐구 도구 - Logistic Regression이란?

일반적으로 종속 변수와 독립 변수의 관계를 분석할 때, 선형회귀분석 방법을 사용한다. 하지만, 이변량 종속 변수(1 또는 0)인 경우, 이러한 선형회귀분석의 적용은 적합하지 않다. 이변량 종속 변수라는 특성을 가지고 있음에도 이를 효과적으로 분석할 수 있는 도구가 로지스틱 회귀 분석(Logistic Regression)이다.

예를들어, 타이타닉 데이터와 같이 종속 변수(Survived)가 생존 or 사망인 경우에도 사용이 가능하며, 합격 혹은 불합격, 부도 혹은 생존 등의 데이터를 분석하는데도 용이하다.


### 분석 과정

먼저, 필요한 라이브러리와 데이터를 불러 온다.

{% highlight ruby %}
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression             # 로지스틱 회귀분석을 할 수 있는 라이브러리
from sklearn.model_selection import train_test_split              # 모델 평가를 위한 라이브러리 
df= pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
{% endhighlight %}

데이터를 불러왔으면, 어떤 변수를 종속 변수로 두고, 어떤 변수들을 독립 변수로 둘 지 결정한다.

종속변수는 ```Survived```

독립변수는 Age Pclass Fare Siblings/Spouses Aboard Parents/Children Aboard

로 설정했다.
