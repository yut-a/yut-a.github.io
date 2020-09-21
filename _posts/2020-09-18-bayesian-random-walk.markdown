---
layout: post
title:  Bayesian을 통해 알아본 주식 시장의 Random Walk
date:   2020-09-18
image:  risk.jpg
tags:   Data finance
---
## Bayesian 이란?

`Bayesian`은 확률을 현상에 대한 관찰자의 주관적 믿음의 체계로 본다. 즉, `Bayesian`은 과거의 사건을 바탕으로 현재의 사건을 판단한다. `Bayesian`은 데이터를 얻기 전에 분포에 대한 믿음인 `사전확률분포`를 가지고 있다. 그리고 데이터를 얻게 되면, 과거의 자신의 견해를 갱신하여 새로운 믿음인 `사후확률분포`를 만들고 이러한 과정에 따라 참된 분포에 접근시킨다.

`사전확률`은 사건이 발생하기 전, 자신의 경험이나 데이터 또는 외부로 부터 받은 자료들을 근거로 해당 사건이 발생할 것이라는 기대를 확률적으로 나타낸 것이다.<BR/>
`사후확률`은 사건이 발생한 상황에서, 그 사건이 어떤 원인으로 부터 발생한 것인지를 역으로 추정하여 확률적으로 나타낸 것이다.<BR/><BR/><BR/><BR/>

## 분석 목적

주식 시장에서 미래의 가격을 예측하는 것은 매우 중요하다. 가격이 상승할 것인지, 하락할 것인지의 예측을 통해 큰 수익을 창출할 수 있기 때문이다. 미래의 주식 가격 예측에는 다양한 방법들이 존재한다. 재무제표를 활용한 Valuation을 할 수도 있고, 기업의 기술 발전이나 성장 가능성을 파악해 볼 수도 있다. 외부 경제 상황이 미치는 요인 역시 파악해 볼 수 있다.

이러한 여러 방법들은 일단 제쳐두고, 미래의 가격을 가장 쉽고 직관적으로 판단할 수 있는 것이 있다. 바로, 과거의 가격이다. 주식 투자를 해 본 사람이라면, 과거의 가격 추이가 미래의 가격 예측 판단에 영향을 준다는 것에 큰 공감을 할 것이다. 과거의 떨어진 가격을 보고, 혹은 오른 가격을 보고 해당 종목을 사야 하나, 말아야 하나 고민한 경험이 있을 것이다.

그렇다면, 어제 상승 혹은 하락한 가격이 오늘의 주가 상승에 어떤 영향을 미칠 수 있을까?<BR/>
이러한 궁금증에 대해 Bayesian 통계 분석을 통해, 어제 가격이 상승한 종목을 사야 하는지, 하락한 종목을 사야 하는지 혹은 어떤 종목이든 어제의 가격과는 별개인지에 대해 알아보고자 한다.<BR/><BR/><BR/><BR/>

## 금융 데이터 소개

우리나라의 경제를 가장 잘 대변할 수 있는 `Kospi 200 ETF` 데이터를 활용했다.

**ETF란?**<BR/>
Exchange Traded Fund의 약자로, 특정 주가지수를 주식처럼 거래가 가능하며, 특정 주가지수의 움직임에 따라 수익률이 결정되는 금융 상품이다.<BR/>
예를 들어, Kospi200 지수가 3% 올랐다고 가정하면, Kospi200 지수 자체를 주식처럼 거래할 수 없기 때문에 Kospi200 지수처럼 움직이는 상품을 거래할 수 있도록 만들어 놓은 것이 Kospi200 ETF이다.<BR/>
_참고 : `kodex200`은 `kospi200 ETF`의 상품 이름 중 하나이다._<BR/><BR/><BR/><BR/>

## 적용 과정

2012년 1월 2일 ~ 2018년 12월 28일의 kodex200 데이터를 불러왔다. header를 정리하고 필요 없는 column들을 제거했다.

{% highlight ruby %}
# 로컬 머신에서 데이터 셋 불러오기
from google.colab import files
uploaded = files.upload()

# 데이터 불러오기 및 칼럼 설정
import pandas as pd

kodex200 = pd.read_csv("kodex200_data.csv", skiprows = 4, engine = "python")
kodex200.columns = ["date", "open", "high", "low", "close", "daily_rate", "", "", "", "", "", "", ""]
kodex200 = kodex200.drop([""], axis = 1)
kodex200["date"] = pd.to_datetime(kodex200["date"], infer_datetime_format = True)
kodex200.head()
{% endhighlight %}
<img width="445" alt="스크린샷 2020-09-22 오전 1 04 04" src="https://user-images.githubusercontent.com/70478154/93791729-8f104900-fc6f-11ea-9ebd-c6dcd5a94ae4.png">

**간단하게 column들을 소개하자면,**

`date` 는 주식시장이 열렸던 날짜로, 주말이나 공휴일에는 개장을 하지 않기 때문에 영업일들로 구성되어 있다.<BR/>
`open` 은 시가로, 주식시장이 개장했을 시점의 가격을 말한다.<BR/>
`high` 는 고가로, 하루 동안의 가격 추이 중 가장 높았던 가격을 말한다.<BR/>
`low` 는 저가로, 하루 동안의 가격 추이 중 가장 낮았던 가격을 말한다.<BR/>
`close` 는 종가로, 주식시장이 마감했을 시점의 가격을 말한다.<BR/>
`daily_rate` 는 전일 대비 수익률을 말한다.

먼저, 일별 수익률이 0%인 갯수를 확인했다. 전일 종가 대비 가격 변동이 없는 데이터로, 데이터의 양이 매우 적기 때문에 분석에서 제외하고자 한다.

{% highlight ruby %}
# 일별 수익률의 0% 갯수 확인
cond = kodex200["daily_rate"] == 0
kodex200[cond].describe()
{% endhighlight %}
<img width="149" alt="스크린샷 2020-09-22 오전 1 06 36" src="https://user-images.githubusercontent.com/70478154/93791997-e7474b00-fc6f-11ea-9370-32d106fdb708.png">

다음은, 분석의 편의를 위해 일별 수익률이 각각 +, 0, -인지를 나타내는 `+/-` 칼럼을 추가했다. 또, 전일 수익률에 대한 조건이 필요하기 때문에 전일 수익률에 대한 +, 0, - 역시 `before` 칼럼을 추가하고 여기에 데이터를 입력했다. 이 때, 전일 수익률 데이터이기 때문에 일별 수익률 데이터에 비해 수가 1개 적다. 따라서 일별 수익률 데이터에서 마지막 행을 삭제하고 전일 수익률 데이터와 병합했다.

{% highlight ruby %}
# 일별 수익률이 +, 0, -인지를 나타내는 칼럼 생성
def change(x):
  if x > 0:
    return "+"
  elif x == 0:
    return 0
  else:
    return "-"

kodex200["+/-"] = kodex200["daily_rate"].apply(change)

# 일별 수익률과 전일 수익률 데이터 병합을 위한 수 맞추기를 위해 마지막 행 제거
kodex200_adjust = kodex200.drop(kodex200.index[1718])

# 전일 수익률이 +, 0, -인지를 나타내는 칼럼 생성
kodex200_before = kodex200["+/-"]
kodex200_before = kodex200_before.iloc[1:]
kodex200_before = kodex200_before.reset_index()
kodex200_before = kodex200_before.drop(["index"], axis = 1)
kodex200_before.columns = ["before"]

# 두 데이터 프레임 합치기
kodex200_df = pd.concat([kodex200_adjust, kodex200_before], axis = 1)
kodex200_df.head()
{% endhighlight %}
<img width="540" alt="스크린샷 2020-09-22 오전 1 07 41" src="https://user-images.githubusercontent.com/70478154/93792134-15c52600-fc70-11ea-9dd3-2ee060b6cc72.png">

분석을 통해 알고자 하는 것은 **어제 주가가 올랐을 때, 오늘 주가가 오를 확률**이다.<BR/>
주가가 오를 확률을 `P(A)`라고 하자.<BR/>
오늘 주가가 오를 때, 전날 주가도 올랐을 확률을 `P(B|A)`라고 하자.<BR/>
또, 오늘 주가가 떨어질 때, 전날 주가는 올랐을 확률을 `P(B|A-)`라고 하자.

이 확률을 구하기 위한 갯수를 파악하고자 한다.

병합한 데이터프레임을 바탕으로, 일별 수익률에서 +, 0, - 데이터 각각의 갯수를 산출했다.<BR/>
이를 통해 0을 제외한 전체 데이터의 갯수는 `1693개`, 일별 수익률이 +와 -인 갯수는 각각 `887개`, `806개`임을 알 수 있다.

{% highlight ruby %}
# 일별 주가 상승률 +/- 각각의 갯수
kodex200_df["+/-"].value_counts()
{% endhighlight %}
<img width="217" alt="스크린샷 2020-09-22 오전 1 10 25" src="https://user-images.githubusercontent.com/70478154/93792407-75233600-fc70-11ea-9c9f-b75314edfcb8.png">

또, 일별 수익률이 +일 때, 전날 주가 상승률이 +인 갯수를 산출했다.<BR/>
일별 수익률이 +일 때, 전날 주가 상승률이 +인 갯수는 `459개`, -인 갯수는 `414개`이다.

{% highlight ruby %}
# 일별 주가 상승률이 + 일 때, 전날 주가 상승률이 +인 갯수
cond_3 = kodex200_df["+/-"] == "+"
kodex200_df[cond_3].before.value_counts()
{% endhighlight %}
<img width="230" alt="스크린샷 2020-09-22 오전 1 11 37" src="https://user-images.githubusercontent.com/70478154/93792532-9c7a0300-fc70-11ea-942e-112e8c29bed2.png">

위에서 구한 데이터를 바탕으로, Bayesian 통계 분석을 실시하고자 한다.<BR/>
`prior probability`와 `true positive`, `false positive`를 계산한 후, Bayesian 확률을 계산할 수 있는 함수를 만들어 적용했다.

{% highlight ruby %}
# Bayesian을 적용하기 위한 데이터
p_up = 887 / 1693       # 주가가 오를 확률 - prior prob : P(A)
p_up_up = 459 / 873     # 오늘 주가가 오를 때, 전날 주가도 올랐을 확률 - true positive : P(B|A)
p_up_down = 412 / 795   # 오늘 주가가 떨어질 때, 전날 주가는 올랐을 확률 - false positive : P(B|A-)
p_down = 1 - p_up       # 주가가 떨어질 확률 : 1 - P(A)

# Bayesian 확률 계산 함수
def prob_up_given_positive(prob_up_prior, false_positive_rate, true_positive_rate):
  prob_down_prior = 1 - prob_up_prior

  numerator = true_positive_rate * prob_up_prior
  denominator = (true_positive_rate * prob_up_prior) + (false_positive_rate * prob_down_prior)

  posterior_prob = numerator / denominator
  return posterior_prob

# 어제 주가가 올랐을 때, 오늘 주가가 오를 확률
prob_up_given_positive(p_up, p_up_down, p_up_up)
{% endhighlight %}
<img width="173" alt="스크린샷 2020-09-22 오전 1 13 06" src="https://user-images.githubusercontent.com/70478154/93792707-d5b27300-fc70-11ea-8ae8-4ae3e4a69f78.png">

결과에 따르면, 사전확률은 약 52.39%이고, Bayesian 분석을 활용하여 확률을 구한 값이 52.75%임을 알 수 있다. 즉, 어제 주가가 올랐을 때, 오늘 주가가 오를 확률은 거의 **반반**이라고 할 수 있다.

그렇다면, 어제 주가가 떨어졌다면, 오늘 주가가 오를 확률은 얼마일까?

분석을 통해 알고자 하는 것은 **어제 주가가 떨어졌을 때, 오늘 주가가 오를 확률**이다.<BR/>
주가가 오를 확률을 `P(A)`라고 하자.<BR/>
오늘 주가가 오를 때, 전날 주가가 떨어졌을 확률을 `P(B-|A)`라고 하자.<BR/>
또, 오늘 주가가 떨어질 때, 전날 주가도 떨어졌을 확률을 `P(B-|A-)`라고 하자.

마찬가지로 Bayesian 통계 분석을 적용하기 위한 prior probability와 true positive, false positive를 설정했다.

{% highlight ruby %}
# Bayesian을 적용하기 위한 데이터
p_up = 887 / 1693       # 주가가 오를 확률 - prior prob : P(A)
p_down_up = 414 / 873   # 오늘 주가가 오를 때, 전날 주가는 떨어졌을 확률 - true positive : P(B-|A)
p_down_down = 383 / 795 # 오늘 주가가 떨어질 때, 전날 주가도 떨어졌을 확률 - false positive : P(B-|A-)
p_down = 1 - p_up       # 주가가 떨어질 확률 : 1 - P(A)

# 어제 주가가 떨어졌을 때, 오늘 주가가 오를 확률
prob_up_given_positive(p_up, p_down_down, p_down_up)
{% endhighlight %}
<img width="170" alt="스크린샷 2020-09-22 오전 1 15 05" src="https://user-images.githubusercontent.com/70478154/93792927-1d38ff00-fc71-11ea-844a-f90a1726d7d3.png">

결과에 따르면, 어제 주가가 떨어졌을 때, 오늘 주가가 오를 확률은 대략 52%로, 역시 오늘 주가가 오를 확률은 거의 **반반**이라고 할 수 있다.

정리하자면, 어제 주가가 오르든 떨어지든, 오늘 주가가 오를 확률은 약 52%로, 거의 **절반**임을 알 수 있다. 즉, 과거의 가격을 바탕으로 현재의 가격을 판단하는 것은, 동전 던지기 게임의 확률과 비슷하다는 의미이다. 이러한 결과를 종합해 보았을 때, '어제의 주가'처럼 과거의 가격을 통해 현재의 가격을 예측할 수 없다.

이러한 결론을 도출한 후, 추가적인 궁금증이 생겼다.<BR/>
이틀 연속 과거의 가격이 올랐다면, 오늘 가격 상승의 가능성을 더 높게 예측해 볼 수 있지 않을까.<BR/>
만약 2일 연속 주가가 올랐다면, 오늘 주가는 상승할까? 추가적으로 이러한 궁금증을 해결해보고자 한다.<BR/><BR/><BR/><BR/>

## 추가 분석

분석을 통해 알고자 하는 것은 **2일 연속 주가가 올랐을 때, 오늘 주가가 오를 확률**이다.<BR/>
주가가 오를 확률을 `P(A)`라고 하자.<BR/>
오늘 주가가 오를 때, 2일 연속 주가도 올랐을 확률을 `P(C|A)`라고 하자.<BR/>
또, 오늘 주가가 떨어질 때, 2일 연속 주가가 올랐을 확률을 `P(C|A-)`라고 하자.

위의 전일 수익률 데이터 병합과 마찬가지로, 분석을 위해 전전일 수익률 데이터를 `2_before` 칼럼에 추가하고, 데이터프레임을 병합했다.

{% highlight ruby %}
# 전전일 데이터와의 병합을 위한 수 맞추기를 위해 마지막 행 제거
kodex200_df.tail()
kodex200_df_adjust = kodex200_df.drop(kodex200_df.index[1717])

# 전전일 수익률이 +, 0, -인지를 나타내는 칼럼 생성
kodex200_df_before = kodex200_df["before"]
kodex200_df_before = kodex200_df_before.iloc[1:]
kodex200_df_before = kodex200_df_before.reset_index()
kodex200_df_before = kodex200_df_before.drop(["index"], axis = 1)
kodex200_df_before.columns = ["2_before"]

# 두 데이터 프레임 합치기
kodex200_last = pd.concat([kodex200_df_adjust, kodex200_df_before], axis = 1)
kodex200_last.head()
{% endhighlight %}
<img width="630" alt="스크린샷 2020-09-22 오전 1 17 53" src="https://user-images.githubusercontent.com/70478154/93793218-7e60d280-fc71-11ea-8239-c029b418660c.png">

0 데이터를 분석에서 제외하기 위해 일별 수익률이 +일 때, `before`과 `2_before` 열의 0 갯수를 파악했다. 0의 갯수는 총 29개이다.

{% highlight ruby %}
# +/-열이 +일 때, before과 2_before 열의 0 갯수 파악
cond_5 = kodex200_last[kodex200_last["+/-"] == "+"]
cond_5[cond_5["before"] == 0].before.value_counts()           # 14개
cond_5[cond_5["2_before"] == 0].before.value_counts()         # 15개
cond_5[(cond_5["before"] == 0) & (cond_5["2_before"] == 0)]   # 0개
{% endhighlight %}

다음은, 오늘 주가가 오를 때의 갯수와 오늘 주가가 오를 때, 2일 연속 주가가 올랐던 갯수를 산출했다. 0을 제외한 오늘 주가 상승의 갯수는 `857개`(886-29)이며, 오늘 주가가 오를 때, 2일 연속 주가가 올랐던 갯수는 `232개`이다.

{% highlight ruby %}
# 오늘 주가가 오를 때의 경우의 수 => +/-, before, 2_before에서 0이 있는 행을 제외하기 위해
print(cond_5["+/-"].value_counts())

# 오늘 주가가 오를 때, 2일 연속 주가가 올랐던 갯수
cond_5_1 = cond_5[(cond_5["before"] == "+")]
cond_5_2 = cond_5_1[(cond_5_1["2_before"] == "+")]
print(cond_5_2["2_before"].value_counts())
{% endhighlight %}
<img width="244" alt="스크린샷 2020-09-22 오전 1 19 34" src="https://user-images.githubusercontent.com/70478154/93793394-bbc56000-fc71-11ea-9b22-c9066458f1d2.png">

마찬가지로, 오늘 주가가 떨어질 때의 데이터들을 산출했다.

{% highlight ruby %}
# +/-열이 -일 때, before과 2_before 열의 0 갯수 파악
cond_6 = kodex200_last[kodex200_last["+/-"] == "-"]
cond_6[cond_6["before"] == 0].before.value_counts()           # 11개
cond_6[cond_6["2_before"] == 0].before.value_counts()         # 10개
cond_6[(cond_6["before"] == 0) & (cond_6["2_before"] == 0)]   # 0개
{% endhighlight %}

{% highlight ruby %}
# 오늘 주가가 떨어질 때의 경우의 수 => +/-, before, 2_before에서 0이 있는 행을 제외하기 위해
print(cond_6["+/-"].value_counts())

# 오늘 주가가 떨어질 때, 2일 연속 주가가 올랐던 갯수
cond_6_1 = cond_6[(cond_6["before"] == "+")]
cond_6_2 = cond_6_1[(cond_6_1["2_before"] == "+")]
print(cond_6_2["2_before"].value_counts())
{% endhighlight %}
<img width="248" alt="스크린샷 2020-09-22 오전 1 20 46" src="https://user-images.githubusercontent.com/70478154/93793572-edd6c200-fc71-11ea-999d-b5844c3c13ba.png">

0의 갯수는 21개이며, 0을 제외한 오늘 주가 하락의 갯수는 `785개`이다. 오늘 주가가 떨어질 때, 2일 연속 주가가 올랐던 갯수는 `214개`이다.

산출한 데이터를 바탕으로 Bayesian 통계 분석을 적용하기 위한 prior probability와 true positive, false positive를 설정한 후 분석을 실시했다.

{% highlight ruby %}
# Bayesian을 적용하기 위한 데이터
p_up = 887 / 1693       # 주가가 오를 확률 - prior prob : P(A)
p_2up_up = 232 / 857    # 오늘 주가가 오를 때, 2일 연속 주가가 올랐던 확률 - true positive : P(C|A)
p_2up_down = 214 / 785  # 오늘 주가가 떨어질 때, 2일 연속 주가가 올랐던 확률 - false positive : P(C|A-)

# 2일 연속 주가가 올랐을 때, 오늘 주가가 오를 확률
prob_up_given_positive(p_up, p_2up_down, p_2up_up)
{% endhighlight %}
<img width="162" alt="스크린샷 2020-09-22 오전 1 22 08" src="https://user-images.githubusercontent.com/70478154/93793669-152d8f00-fc72-11ea-8a21-ce77b72d4a7e.png">

2일 연속 주가가 상승하면, 오늘 주가 상승 확률에 큰 영향을 주지 않을까 라는 예상과는 다르게, 약 52.21%의 결과가 나왔다. 즉, 2일 연속 주가가 올랐을 때에도 오늘 주가가 오를 확률은 거의 **절반**이라고 해석할 수 있다.<BR/><BR/><BR/><BR/>

## 결론

두 종류의 Bayesian 통계 분석을 종합해 볼 때, 과거 하루 혹은 이틀의 데이터를 바탕으로 현재 주가 상승을 판단하는 것은 무리가 있음을 알 수 있다. 이러한 결과를 `주가의 Random Walk 가설`에 적용해 볼 수 있다. Random Walk란, 미래 가격 변동을 예측하는데 과거의 가격은 아무 쓸모가 없다는 가설이다. 미래 가격은 과거의 가격에 대한 일정한 규칙으로 움직이는 것이 아니라, 무작위적으로 움직인다는 의미이다. 이처럼 Bayesian 통계 분석의 결과를 비추어 봤을 때, Random Walk 가설이 어느정도 미래 가격 예측에 대한 현실을 일부 설명할 수 있음을 알 수 있다.<BR/><BR/><BR/><BR/>

## 한계

* 국내 증시의 전반적인 흐름을 나타내는 Kospi200 ETF를 활용했기 때문에, 이를 개별 종목, 해외 자산 등에 적용하기에는 무리가 있다.
* 활용한 데이터의 양이 충분하다고 할 수 없으며, 서브프라임 모기지, 코로나 19 등 심각한 경제 위기 시점에 적용하기에는 무리가 있다.
* 하루 혹은 이틀의 과거 데이터를 중심으로 분석했기 때문에 더 많은 과거 데이터에 따른 가격 예측을 설명할 수 없다.
