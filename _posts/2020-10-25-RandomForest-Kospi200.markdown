---
layout: post
title:  RandomForest를 활용한 Kospi200 주가 방향 예측
date:   2020-10-25
image:  arrows.jpg
tags:   Data Finance
---
## RandomForest란?

먼저, **의사결정트리(Decision Tree)**에 대한 이해가 필요하다. 의사결정트리는 아래의 그림과 같이 `기린`이라는 정답을 찾기 위해 질문을 이어가며 구분을 하는 방식이다. 즉, 각 데이터들이 가진 속성들로부터 패턴을 찾아낸 후 분류 혹은 회귀 문제를 풀 수 있도록 하는 머신러닝 모델이다. 의사결정나무는 복잡한 데이터에 대해 높은 예측력을 낼 수 있는 모델이라고 할 수 있다. 그러나 train set에는 매우 높은 예측도를 보이지만, test set에는 낮은 예측도를 보이는 `Overfitting` 문제가 발생할 수 있다.

<img width="505" alt="스크린샷 2020-10-25 오후 5 07 56" src="https://user-images.githubusercontent.com/70478154/97102035-64d5ff00-16e5-11eb-8d71-cbe69bc85467.png">

이를 보완해줄 수 있는 것이 **랜덤포레스트(RandomForest)**이다. 나무들이 모여 숲을 이루는 것처럼 이름에서도 알 수 있듯, 이 모델도 역시 여러 의사결정트리들을 만들고 그 트리들로부터 결과를 도출해낸다. 즉, 전체 train set에서 중복을 허용하여 샘플링한 데이터들을 개별 의사결정트리에 적용하고 이를 종합한 결론을 도출하는 방법으로, 다수의 학습 알고리즘을 사용하는 **앙상블(Ensemble) 기법** 중 하나이다. 이러한 모델은 Overfitting 문제를 줄여 일반화된 모델을 만들 수 있다는 장점을 가지고 있다.<BR/><BR/><BR/><BR/>

## 분석 목적

해외 증시, 주요 경제, 금융 지표들을 바탕으로 Kospi200 주가의 방향을 얼마나 효과적으로 예측할 수 있는지 알아보고자 한다. 또한, 주가 방향 예측에 가장 영향을 미치는 요소들을 파악하고자 한다. 더 나아가, 이러한 지표들이 주가 방향 예측에 가장 크게 영향을 줄 수 있는 시기를 `일주일` `한달` `3개월` `6개월`로 나누어 알아보고자 한다.<BR/><BR/><BR/><BR/>

## 데이터 소개

target 데이터인 Kospi200은 Kodex200 ETF의 주가를 활용했다. 일자 별 데이터들을 바탕으로 예측을 하는 것이기 때문에 다음 날 종가와 `일주일` `한달` `3개월` `6개월` 뒤의 종가와 비교하여 수익률을 산출했고, 이를 `1, 0, -1`로 분류했다. 기간에 따라 누적되는 수익률 크기에 차이가 있기 때문에 분류 기준점을 다르게 적용했다.

* `일주일` 기준 target : **수익률 >= 1%**면 `1`, **1% > 수익률 > 0%**면 `0`, **수익률 <= 0%**면 `-1`
* `한달` 기준 target : **수익률 >= 2%**면 `1`, **2% > 수익률 > 0%**면 `0`, **수익률 <= 0%**면 `-1`
* `3개월` 기준 target : **수익률 >= 4%**면 `1`, **4% > 수익률 > 0%**면 `0`, **수익률 <= 0%**면 `-1`
* `6개월` 기준 target : **수익률 >= 6%**면 `1`, **4% > 수익률 > 0%**면 `0`, **수익률 <= 0%**면 `-1`

예측을 위한 feature는 날짜를 제외하고 총 18개를 사용했다.

* `경기선행지수` : 3~6개월 후의 경기 흐름을 가늠하는 지표이다.
* `수출증가율` : 수출금액지수를 바탕으로 증가율을 산출했다.
* `콜금리` : 금융기관 간 거래에서의 단기 금리이다.
* `CD 금리` : CD(양도성예금증서)가 발행되어 유통시장에서 거래될 때 적용되는 금리이다.
* `DAX 선물` : 독일의 종합주가지수인 DAX 선물 가격이다.
* `DOW 선물` : 미국의 종합주가지수인 DOW 선물 가격이다.
* `FTSE 선물` : 영국의 FTSE 사가 발표하는 세계 주가지수에 대한 선물 가격이다.
* `Nikkei225 선물` : 일본의 종합주가지수인 Nikkei 선물 가격이다.
* `금 선물` `WTI 선물` : 각각 금, 원유에 대한 선물 가격이다.
* `국채 3년` : 우리나라 3년 만기 국채 가격이다.
* `미국 국채 10년` `미국 국채 3년`
* `유로/달러 환율` `원/달러 환율`
* `달러 선물 인덱스` : 달러인덱스는 유로, 엔, 파운드, 캐나다 달러, 스웨덴 크로네, 스위스 프랑 등 경제 규모가 크거나 통화가치가 안정적인 6개국 통화를 기준으로 산정한 미 달러화 가치를 지수화한 것으로, 이에 대한 선물이다.
* `VIX` : S&P500 지수옵션의 향후 30일 간의 변동성에 대한 시장의 기대를 나타낸 지수이다.
* `VKOSPI`: KOSPI200 지수옵션의 미래변동성을 측정한 지수이다.

target 기준에 따라 대략 2009년 4월부터 각각 2020년 1월, 6월, 9월, 10월까지의 데이터를 활용했다.<BR/><BR/><BR/><BR/>

## 적용 과정

{% highlight ruby %}
# 데이터 불러오기
import pandas as pd

CLI = pd.read_csv("CLI.csv", encoding = "cp949")        # 경기선행지수
EX = pd.read_csv("Export.csv", encoding = "cp949")      # 수출금액지수
IR = pd.read_csv("IR.csv", encoding = "cp949")          # 콜금리, CD 금리
DAX = pd.read_csv("DAX_futures.csv")                    # 독일 DAX 선물
DOW = pd.read_csv("dow_futures.csv")                    # 미국 DOW 선물
EUR_USD = pd.read_csv("EUR_USD.csv")                    # 유로/달러 환율
FTSE = pd.read_csv("FTSE_futures.csv")                  # 영국 FTSE 선물
Gold = pd.read_csv("gold_futures.csv")                  # 금 선물
Korea_3Y_bond = pd.read_csv("Korea_3Y_bond.csv")        # 국채 3년
Nikkei = pd.read_csv("Nikkei225_futures.csv")           # 일본 Nikkei225 선물
US_3Y_bond = pd.read_csv("US_3Y_bond.csv")              # 미국 국채 3년
US_10Y_bond = pd.read_csv("US_10Y_bond.csv")            # 미국 국채 10년
USD_index = pd.read_csv("USD_futures.csv")              # 달러 선물 인덱스
USD_KRW = pd.read_csv("USD_KRW.csv")                    # 원/달러 환율
VIX = pd.read_csv("VIX.csv")                            # VIX
VKOSPI = pd.read_csv("VKOSPI.csv")                      # KOSPI Volatility
WTI = pd.read_csv("WTI_futures.csv")                    # WTI 선물
kodex200 = pd.read_csv("kodex200_price.csv",            # Kodex200 ETF
                      skiprows = 4, engine = "python")
{% endhighlight %}

먼저, 경기선행지수, 수출증가율, 콜금리, CD 금리 데이터를 정리했다. 수출금액지수를 바탕으로 수출증가율을 산출했다. 경기선행지수와 수출증가율은 월 단위 데이터이기 때문에 해당 달의 데이터로 나머지 일자를 채웠다. 또한, 2020년 9월과 10월 데이터는 아직 발표되지 않았기 때문에 8월 데이터로 채웠다.

{% highlight ruby %}
# CLI, IR, EX 데이터 전처리
def change(df):
    
    if len(df.index) == 1:
        df = df.T.reset_index()
        df = df.drop([0, 1, 2, 3])
    else:
        df = df.drop([0, 1, 2])
        
    df = df.reset_index(drop = True)
    
    df = df.rename(columns = {df.columns[0] : "date"})
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format = True)
    
    return df

CLI = change(CLI)
IR = change(IR)
EX = change(EX)

CLI = CLI.rename(columns = {CLI.columns[1] : "CLI"})
IR = IR.rename(columns = {IR.columns[1] : "CD_rate", IR.columns[2] : "call_rate"})
EX = EX.rename(columns = {EX.columns[1] : "EX"})

# 수출증가율 산출
EX["EX(%)"] = round(EX["EX"].pct_change() * 100, 2)
EX = EX.drop(["EX"], axis = 1)
EX = EX.dropna(axis = 0)
{% endhighlight %}

{% highlight ruby %}
# 병합하고 결측치 채우기
from_bok = pd.merge(IR, CLI, on = "date", how = "outer")
from_bok = pd.merge(from_bok, EX, on = "date", how = "outer")
from_bok = from_bok.sort_values(by = "date", ascending = True)
from_bok = from_bok.reset_index(drop = True)

from_bok["CLI"] = from_bok["CLI"].fillna(method = "ffill")
from_bok["EX(%)"] = from_bok["EX(%)"].fillna(method = "ffill")

# 날짜 기준으로 내림차순
from_bok = from_bok.sort_values(by = "date", ascending = False).reset_index(drop = True)
from_bok
{% endhighlight %}
<img width="328" alt="스크린샷 2020-10-25 오후 7 04 26" src="https://user-images.githubusercontent.com/70478154/97104147-f13bee00-16f4-11eb-91af-27233670b0d8.png">

target 데이터를 제외한 나머지 데이터들을 정리했다.

{% highlight ruby %}
# 나머지 데이터 전처리
VKOSPI = VKOSPI.sort_values(by = "일자", ascending = False).reset_index(drop = True)

name_list = ["DAX", "DOW", "FTSE", "Gold", "Nikkei", "USD_index", "VIX", "VKOSPI", "WTI",
            "EUR_USD", "Korea_3Y_bond", "US_3Y_bond", "US_10Y_bond", "USD_KRW"]

for i in name_list:
    
    # 일부 칼럼만 사용 및 칼럼명 변경
    globals()["{}".format(i)] = globals()["{}".format(i)].iloc[:,:2]
    globals()["{}".format(i)] = globals()["{}".format(i)].rename(columns = {globals()["{}".format(i)].columns[0] : "date",
                                                                            globals()["{}".format(i)].columns[1] : i})
    
    # datetime 형식으로 바꾸기 위한 작업
    globals()["{}".format(i)]["date"] = globals()["{}".format(i)]["date"].str.replace("년", "-")
    globals()["{}".format(i)]["date"] = globals()["{}".format(i)]["date"].str.replace(" ", "")
    globals()["{}".format(i)]["date"] = globals()["{}".format(i)]["date"].str.replace("월", "-")
    globals()["{}".format(i)]["date"] = globals()["{}".format(i)]["date"].str.replace("일", "")
    
    # datetime 형식으로 변경
    globals()["{}".format(i)]["date"] = pd.to_datetime(globals()["{}".format(i)]["date"], infer_datetime_format = True)
{% endhighlight %}

{% highlight ruby %}
# 데이터 병합
all_data = from_bok.copy()

from_invest = [DAX, DOW, FTSE, Gold, Nikkei, USD_index, VIX, VKOSPI, WTI,
               EUR_USD, Korea_3Y_bond, US_3Y_bond, US_10Y_bond, USD_KRW]

for i in from_invest:
    all_data = pd.merge(all_data, i, on = "date", how = "outer")

all_data
{% endhighlight %}
<img width="984" alt="스크린샷 2020-10-25 오후 7 07 00" src="https://user-images.githubusercontent.com/70478154/97104203-542d8500-16f5-11eb-8a27-c02d6b400d03.png">

다음은, target 데이터를 정리하고 모든 데이터들을 병합했다. 그 후, 결측치들을 모두 제거했다. 해외와 우리나라의 공휴일 차이로 발생하는 결측치와 월 단위 데이터로 인해 주말, 공휴일에 데이터들이 속해있어 발생하는 결측치들을 제거했다. 또, 일부 데이터가 2009년부터 시작되기 때문에 그 이전 데이터들을 모두 제거했다.

{% highlight ruby %}
#target 데이터 전처리
kodex200 = kodex200.drop(kodex200.iloc[:,1:4], axis = 1)
kodex200 = kodex200.drop(kodex200.iloc[:,2:], axis = 1)

kodex200.columns = ["date", "price"]
kodex200 = kodex200.sort_values(by = "date", ascending = False).reset_index(drop = True)
kodex200["date"] = pd.to_datetime(kodex200["date"], infer_datetime_format = True)
kodex200.head()
{% endhighlight %}
<img width="149" alt="스크린샷 2020-10-25 오후 7 14 52" src="https://user-images.githubusercontent.com/70478154/97104330-678d2000-16f6-11eb-90b9-7b300a2d4a98.png">

{% highlight ruby %}
# 모든 데이터 통합
finance = pd.merge(all_data, kodex200, on = "date", how = "outer")

# 결측치 제거
finance = finance.dropna(axis = 0).reset_index(drop = True)
finance
{% endhighlight %}
<img width="985" alt="스크린샷 2020-10-25 오후 7 15 53" src="https://user-images.githubusercontent.com/70478154/97104345-8f7c8380-16f6-11eb-8c74-dfc7a112dc86.png">

마지막으로, 분석을 용이하게 하기 위해 숫자형으로 데이터 타입을 변경했다.

{% highlight ruby %}
# 숫자형으로 데이터 타입 변경
for i in range(0, len(finance.columns)):
    if finance.dtypes[i] == "object":
        finance.iloc[:,i] = finance.iloc[:,i].str.replace(",", "")

cols = list(finance.select_dtypes(include = "object").columns)[:8]

for col in cols:
    finance = finance.astype({col : "float"})

finance = finance.astype({"price" : "int"})

finance.dtypes
{% endhighlight %}
<img width="274" alt="스크린샷 2020-10-25 오후 7 17 20" src="https://user-images.githubusercontent.com/70478154/97104373-c05cb880-16f6-11eb-9187-129e1590bbb4.png">







