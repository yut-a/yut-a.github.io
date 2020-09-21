---
layout: post
title:  타이타닉 데이터 분석 - 생존과 사망에 가장 영향을 미치는 변수는 무엇일까
date:   2020-09-04
image:  05.jpg
tags:   Data
---
## 주제 선정 동기와 탐구 목적

* 앞서 동기분들께서 다양한 시각화 자료와 분석으로, 어떤 종류의 승객이 생존 가능성이 높았는지, 예상과 달랐던 흥미로운 결론 등 이미 여러 종류의 blogging을 해주셨다.
* 이에 따라, 다른 방식의 분석을 시도해보고자 했다.
* 어떤 종류의 승객이 생존 가능성이 더 높은 지에서 더 나아가, 변수들 중 어떤 변수가 가장 생존율/사망률에 영향을 미치는지 탐구해보고자 했다.<BR/><BR/><BR/>

## 탐구 도구 - Logistic Regression이란?

일반적으로 종속 변수와 독립 변수의 관계를 분석할 때, 선형회귀분석 방법을 사용한다. 하지만, 이변량 종속 변수(1 또는 0)인 경우, 이러한 선형회귀분석의 적용은 적합하지 않다. 이변량 종속 변수라는 특성을 가지고 있음에도 이를 효과적으로 분석할 수 있는 도구가 로지스틱 회귀 분석(Logistic Regression)이다.

예를들어, 타이타닉 데이터와 같이 종속 변수(Survived)가 생존 or 사망인 경우에도 사용이 가능하며, 합격 혹은 불합격, 부도 혹은 생존 등의 데이터를 분석하는데도 용이하다.<BR/><BR/><BR/><BR/>

## 분석 과정

먼저, 필요한 라이브러리와 데이터를 불러 온다.

{% highlight ruby %}
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression             # 로지스틱 회귀분석을 할 수 있는 라이브러리
from sklearn.model_selection import train_test_split              # 모델 평가를 위한 라이브러리 
df= pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
{% endhighlight %}

데이터를 불러왔으면, 어떤 변수를 종속 변수로 두고, 어떤 변수들을 독립 변수로 둘 지 결정한다.

종속변수는 ```Survived```<BR/>
독립변수는 `Age` `Pclass` `Fare` `Siblings/Spouses Aboard` `Parents/Children Aboard`<BR/>
로 설정했다.

**_중요한 변수 중 하나인 `Sex` 변수는 male과 femaie을 1과 0의 형태로 나타내어 함께 분석하고자 했으나, 1과 0으로 나타내는 걸 계속 실패해서 함께 분석하지 못했다... (더 공부한 후 보완하도록 하겠습니다!)_**

{% highlight ruby %}
# 독립변수 설정
x = df[["Age", "Pclass", "Fare", "Siblings/Spouses Aboard", "Parents/Children Aboard"]]
# 종속변수 설정
y = df[["Survived"]]
{% endhighlight %}

모델의 정확도를 체크하기 위해 test set과 training set으로 구분하고, 정확도를 측정한다.

{% highlight ruby %}
# training data를 로지스틱 회귀모델에 적용
log = LogisticRegression()
log.fit(x_train, y_train)

# 정확도 측정 / %.2f : 소수점 두자리까지 %로 표현
print('학습용 데이터셋 정확도 : %.2f' % log.score(x_train, y_train))
print('검증용 데이터셋 정확도 : %.2f' % log.score(x_test, y_test))
{% endhighlight %}
<img width="216" alt="스크린샷 2020-09-21 오후 9 53 04" src="https://user-images.githubusercontent.com/70478154/93769550-e9041500-fc55-11ea-8623-9d94985a8da5.png">

{% highlight ruby %}
# 모델의 정밀도, 재현율 등을 보여줌
from sklearn.metrics import classification_report
y_pred = log.predict(x_test)
print(classification_report(y_test, y_pred))
{% endhighlight %}
<img width="464" alt="스크린샷 2020-09-21 오후 10 04 02" src="https://user-images.githubusercontent.com/70478154/93769882-6cbe0180-fc56-11ea-8bfb-9570b2172405.png">

위의 결과에 따르면, 정확도와 정밀도, 재현율 모두 0.7정도인 것을 확인할 수 있다. 정확도가 많이 높지는 않지만, 정확성이 어느정도 있다고 판단을 하고, 독립 변수들이 종속 변수에 어떻게 영향을 미치는지 알아본다.

{% highlight ruby %}
import statsmodels.api as sm
logit = sm.Logit(df["Survived"], x)
result = logit.fit()
result.summary2()
{% endhighlight %}
<img width="472" alt="스크린샷 2020-09-21 오후 10 06 57" src="https://user-images.githubusercontent.com/70478154/93770185-d63e1000-fc56-11ea-825b-92e20edb2eef.png">

이 결과를 살펴보면, `Parents/Children Abroad` 변수를 제외한 모든 변수들의 P>\|z\| 가 0.05보다 작은 것으로 보아 통계적으로 유의함을 알 수 있다. `Parents/Children Abroad` 역시 P>\|z\|가 0.05보다 작지는 않지만 0.06으로 어느정도 통계적 유의성을 가지고 있다고 판단할 수 있다.

Coef는 각 변수가 종속변수에 영향을 미치는 민감도로, 회귀 모델에서 계수이다. 즉, `Age`의 경우 종속변수와 음의 상관관계가 있으며, `Age` 변수의 1단위 변화에 따라 -0.0132만큼 종속변수에 영향을 미치는 것이다.
<BR/>
<BR/>
<BR/>
**이를 종합해 분석을 하면,**

`Age` 변수는 종속변수와 음의 상관관계를 가지며, 나이가 어릴수록 생존율 상승에 기인한다.<BR/>
`Pclass` 변수는 종속변수와 음의 상관관계를 가지며, 등급이 높은 방일수록 생존율 상승에 기인한다.<BR/>
`Fare` 변수는 종속변수와 양의 상관관계를 가지며, 지불 금액이 클수록 생존률 상승에 기인한다.<BR/>
`Siblinggs/Spouses Abroad` 변수는 종속변수와 음의 상관관계를 가지며, 형제/자매 수가 작을수록,<BR/>
`Parents/Children Abroad` 변수는 종속변수와 양의 상관관계를 가지며, 부모/자식이 많을수록 생존율 상승에 기인한다.

마지막으로, 처음 알아보고자 했던 각 변수들 중 종속변수에 가장 큰 영향을 미치는 변수를 알아본다.

{% highlight ruby %}
np.exp(result.params)
{% endhighlight %}
<img width="313" alt="스크린샷 2020-09-21 오후 10 20 46" src="https://user-images.githubusercontent.com/70478154/93771635-bdcef500-fc58-11ea-939d-72f9d0ba8f19.png">

종속변수에 가장 큰 영향을 미치는 변수를 알아보는 방법은 `오즈비`를 구하는 것이다.

**오즈비**란, 독립변수의 회귀계수를 지수 변환하여 계산한 exp(coef)이며, 영향력을 파악하는 지표이다.<BR/>
오즈비가 1인 경우, 독립변수는 종속변수와 아무런 관계가 없음을 뜻한다.<BR/>
오즈비가 1로부터 멀어지는 경우, 멀리 떨어질수록 종속변수와의 관계가 강하다는 뜻이다.<BR/>
(coef가 음수면 오즈비는 1보다 작고, coef가 양수면 오즈비는 1보다 크다.)

즉, 위의 오즈비 결과에 따르면, `Pclass` 변수가 가장 종속변수에 영향을 미치는 변수임을 알 수 있다.
<BR/>
<BR/>
<BR/>
## 결론

* `Age` 변수는 종속변수와 음의 상관관계를 가지며, 나이가 어릴수록 생존율 상승에 기인한다.
* `Pclass` 변수는 종속변수와 음의 상관관계를 가지며, 등급이 높은 방일수록 생존율 상승에 기인한다.
* `Fare` 변수는 종속변수와 양의 상관관계를 가지며, 지불 금액이 클수록 생존률 상승에 기인한다.
* `Siblinggs/Spouses Abroad` 변수는 종속변수와 음의 상관관계를 가지며, 형제/자매 수가 작을수록 생존율 상승에 기인한다.
* `Parents/Children Abroad` 변수는 종속변수와 양의 상관관계를 가지며, 부모/자식이 많을수록 생존율 상승에 기인한다.
* `Pclass` 변수가 가장 종속변수에 영향을 미치는 변수이다.
<BR/>
<BR/>
## 추가 보완 사항
* 세부적인 로지스틱 회귀분석에 대한 내용은 공부가 더 필요!
* 수리, 통계적인 이해가 더 필요!
* 이를 바탕으로, 나와 독자에게 친절한 blog가 될 수 있는 친절한 설명 필수!
* `Sex` 변수에 대한 가공을 거쳐 추가적으로 변수와 모델을 검증할 것!
