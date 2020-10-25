---
layout: post
title:  RandomForest를 활용한 Kospi200 주가 방향 예측
date:   2020-10-25
image:  arrows.jpg
tags:   Data Finance
---
## RandomForest란?

RandomForest에 대해 이해하기 위해서는 먼저, **의사결정트리(Decision Tree)**에 대한 이해가 필요하다. 의사결정트리는 아래의 그림과 같이 `기린`이라는 정답을 찾기 위해 질문을 이어가며 구분을 하는 방식이다. 즉, 각 데이터들이 가진 속성들로부터 패턴을 찾아낸 후 분류 혹은 회귀 문제를 풀 수 있도록 하는 머신러닝 모델이다. 의사결정나무는 복잡한 데이터에 대해 높은 예측력을 낼 수 있는 모델이라고 할 수 있다. 그러나 train set에는 매우 높은 예측도를 보이지만, test set에는 낮은 예측도를 보이는 `Overfitting` 문제가 발생할 수 있다.

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

분석을 용이하게 하기 위해 숫자형으로 데이터 타입을 변경했다.

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

이제 target을 시기에 맞게 `일주일` `한달` `3개월` `6개월` 4가지로 변경했다. 시기가 길어질수록 데이터의 크기가 조금씩 줄어드는 것을 알 수 있다. 이는 시기가 길어질수록 더 많은 최근 데이터가 사라지기 때문이다.

{% highlight ruby %}
 # target 변경 (target = 일주일 후)
finance_7 = finance.copy()

finance_7["lag_1"] = finance_7["price"].shift()
finance_7["lag_8"] = finance_7["lag_1"].shift(7)
finance_7["price_pred(%)"] = round((finance_7["lag_8"] - finance_7["lag_1"]) / finance_7["lag_1"] * 100, 2)

def rate_7(x):
    if x >= 1:
        return 1
    
    elif x > 0:
        return 0
    
    elif x <= 0:
        return -1
    
finance_7["predict"] = finance_7["price_pred(%)"].apply(rate_7)
finance_7 = finance_7.dropna(axis = 0).reset_index(drop = True)
finance_7 = finance_7.astype({"predict" : "int"})

finance_7 = finance_7.drop(["price", "lag_1", "lag_8", "price_pred(%)"], axis = 1)

finance_7
{% endhighlight %}
<img width="985" alt="스크린샷 2020-10-25 오후 7 26 49" src="https://user-images.githubusercontent.com/70478154/97104552-154cfe80-16f8-11eb-9ecf-1b0febb8f057.png">

{% highlight ruby %}
# target 변경 (target = 한 달 후)
finance_30 = finance.copy()

finance_30["lag_1"] = finance_30["price"].shift()
finance_30["lag_31"] = finance_30["lag_1"].shift(30)
finance_30["price_pred(%)"] = round((finance_30["lag_31"] - finance_30["lag_1"]) / finance_30["lag_1"] * 100, 2)

def rate_30(x):
    if x >= 2:
        return 1
    
    elif x > 0:
        return 0
    
    elif x <= 0:
        return -1
    
finance_30["predict"] = finance_30["price_pred(%)"].apply(rate_30)
finance_30 = finance_30.dropna(axis = 0).reset_index(drop = True)
finance_30 = finance_30.astype({"predict" : "int"})

finance_30 = finance_30.drop(["price", "lag_1", "lag_31", "price_pred(%)"], axis = 1)

finance_30
{% endhighlight %}
<img width="982" alt="스크린샷 2020-10-25 오후 7 27 55" src="https://user-images.githubusercontent.com/70478154/97104568-3a417180-16f8-11eb-9b69-0a6b9b29bcdd.png">

{% highlight ruby %}
# target 변경 (target = 3개월 후)
finance_90 = finance.copy()

finance_90["lag_1"] = finance_90["price"].shift()
finance_90["lag_91"] = finance_90["lag_1"].shift(90)
finance_90["price_pred(%)"] = round((finance_90["lag_91"] - finance_90["lag_1"]) / finance_90["lag_1"] * 100, 2)

def rate_90(x):
    if x >= 4:
        return 1
    
    elif x > 0:
        return 0
    
    elif x <= 0:
        return -1
    
finance_90["predict"] = finance_90["price_pred(%)"].apply(rate_90)
finance_90 = finance_90.dropna(axis = 0).reset_index(drop = True)
finance_90 = finance_90.astype({"predict" : "int"})

finance_90 = finance_90.drop(["price", "lag_1", "lag_91", "price_pred(%)"], axis = 1)

finance_90
{% endhighlight %}
<img width="981" alt="스크린샷 2020-10-25 오후 7 28 45" src="https://user-images.githubusercontent.com/70478154/97104586-59d89a00-16f8-11eb-9707-b7b4e4d7a3cf.png">

{% highlight ruby %}
# target 변경 (target = 6개월 후)
finance_180 = finance.copy()

finance_180["lag_1"] = finance_180["price"].shift()
finance_180["lag_181"] = finance_180["lag_1"].shift(180)
finance_180["price_pred(%)"] = round((finance_180["lag_181"] - finance_180["lag_1"]) / finance_180["lag_1"] * 100, 2)

def rate_180(x):
    if x >= 6:
        return 1
    
    elif x > 0:
        return 0
    
    elif x <= 0:
        return -1
    
finance_180["predict"] = finance_180["price_pred(%)"].apply(rate_180)
finance_180 = finance_180.dropna(axis = 0).reset_index(drop = True)
finance_180 = finance_180.astype({"predict" : "int"})

finance_180 = finance_180.drop(["price", "lag_1", "lag_181", "price_pred(%)"], axis = 1)

finance_180
{% endhighlight %}
<img width="982" alt="스크린샷 2020-10-25 오후 7 30 08" src="https://user-images.githubusercontent.com/70478154/97104607-8ab8cf00-16f8-11eb-9e69-b564cb03cd32.png">

시기에 따라 target 데이터가 `1, 0, -1`로 정리된 것을 확인할 수 있다. 이제 본격적으로 RandomForest model을 적용해보고자 한다.<BR/><BR/>

#### 일주일 뒤의 주가 방향 예측

train과 test set을 8:2로 분리했다.

{% highlight ruby %}
# train, test set 분리 / date 칼럼 삭제 (일주일)
train_7 = finance_7[finance_7["date"] < "2018.07.01"].drop("date", axis = 1)
test_7 = finance_7[finance_7["date"] >= "2018.07.01"].drop("date", axis = 1)

train_7.shape, test_7.shape
{% endhighlight %}
<img width="204" alt="스크린샷 2020-10-25 오후 7 41 01" src="https://user-images.githubusercontent.com/70478154/97104816-197a1b80-16fa-11eb-9b7f-e584b2a604dc.png">

target 데이터의 종류 별 비중을 확인한 결과, `-1`의 빈도가 가장 높았고, baseline을 44.5%로 설정했다. 그 다음 feature와 target을 분리했다.

{% highlight ruby %}
# baseline
train_7["predict"].value_counts(normalize = True)
{% endhighlight %}
<img width="257" alt="스크린샷 2020-10-25 오후 7 43 26" src="https://user-images.githubusercontent.com/70478154/97104862-63fb9800-16fa-11eb-95de-276f60e709e5.png">

{% highlight ruby %}
# features, target 분리
target_7 = "predict"
features_7 = train_7.drop(columns = [target_7]).columns

X_train_7 = train_7[features_7]
y_train_7 = train_7[target_7]

X_test_7 = test_7[features_7]
y_test_7 = test_7[target_7]
{% endhighlight %}

StandardScaler를 통해 정규화를 한 후, RandomForestClassifier를 진행하는 pipe를 만들었다. 또, 하이퍼 파라미터를 최적화한 모델을 만들기 위해 RandomizedSearchCV를 적용했고, cv는 15로 설정했다.

{% highlight ruby %}
# 하이퍼 파라미터를 최적화한 RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

pipe_7 = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(random_state = 999)
)

dists = {
    "randomforestclassifier__n_estimators" : randint(50, 1000),
    "randomforestclassifier__max_depth" : [5, 10, 15, 20, 25, None],
    "randomforestclassifier__max_leaf_nodes" : [10, 20, 30, 40],
    "randomforestclassifier__max_features" : randint(1, 10),
    "randomforestclassifier__min_samples_leaf" : randint(1, 10)
}

clf_7 = RandomizedSearchCV(
    pipe_7,
    param_distributions = dists,
    n_iter = 100,
    cv = 15,
    scoring = "accuracy",
    verbose = 1,
    n_jobs = -1,
    random_state = 999
)

clf_7.fit(X_train_7, y_train_7);
{% endhighlight %}

최적 하이퍼 파라미터와 Cross Validation 평균 정확도를 산출한 후, 최적 하이퍼 파라미터를 적용한 모델을 train set에 재학습시키고 train과 test set의 정확도를 산출했다.

{% highlight ruby %}
# 최적 하이퍼 파라미터 / CV score
print("최적 하이퍼파라미터: ", clf_7.best_params_, "\n")
print("CV accuracy score: ", clf_7.best_score_)

pipe_7 = clf_7.best_estimator_
{% endhighlight %}
<img width="977" alt="스크린샷 2020-10-25 오후 9 04 14" src="https://user-images.githubusercontent.com/70478154/97106555-c7d78e00-1705-11eb-9b7f-136a5dbfcbf6.png">

{% highlight ruby %}
# 최적 하이퍼 파라미터를 적용한 모델의 train, test set 정확도
fi_pipe_7 = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators = 454, max_depth = 5, max_features = 1, max_leaf_nodes = 10,
                               min_samples_leaf = 8, n_jobs = -1, random_state = 999)
)

fi_pipe_7.fit(X_train_7, y_train_7)
print("Train set accuracy score: ", fi_pipe_7.score(X_train_7, y_train_7))

print("Test set accuracy score: ", fi_pipe_7.score(X_test_7, y_test_7))
{% endhighlight %}
<img width="392" alt="스크린샷 2020-10-25 오후 9 08 59" src="https://user-images.githubusercontent.com/70478154/97106639-577d3c80-1706-11eb-9d3d-35965d5488ec.png">

결과에 따르면, CV의 정확도는 약 29%로, train set에 overfitting이 발생했다는 것을 알 수 있다. 하이퍼 파라미터와 반복 수를 조정하는 등 다양한 방법들을 시도했으나 overfitting 문제가 해결되지 않았다. 또, test set의 정확도는 약 50%로, baseline 보다는 높지만 낮은 예측력을 보인다는 것을 알 수 있다.<BR/><BR/>

#### 한 달 뒤의 주가 방향 예측

train과 test set을 분리하고 baseline을 확인했다. `1`의 빈도가 가장 높았고, baseline을 42.7%로 설정했다. 그 후 feature와 target을 분리했다.

{% highlight ruby %}
# train, test set 분리 / date 칼럼 삭제 (한 달)
train_30 = finance_30[finance_30["date"] < "2018.05.01"].drop("date", axis = 1)
test_30 = finance_30[finance_30["date"] >= "2018.05.01"].drop("date", axis = 1)

train_30.shape, test_30.shape
{% endhighlight %}
<img width="202" alt="스크린샷 2020-10-25 오후 9 16 49" src="https://user-images.githubusercontent.com/70478154/97106872-75976c80-1707-11eb-82a8-4ec12287274d.png">

{% highlight ruby %}
# baseline
train_30["predict"].value_counts(normalize = True)
{% endhighlight %}
<img width="260" alt="스크린샷 2020-10-25 오후 9 17 42" src="https://user-images.githubusercontent.com/70478154/97106886-8f38b400-1707-11eb-8086-fc03693ba24a.png">

{% highlight ruby %}
# features, target 분리
target_30 = "predict"
features_30 = train_30.drop(columns = [target_30]).columns

X_train_30 = train_30[features_30]
y_train_30 = train_30[target_30]

X_test_30 = test_30[features_30]
y_test_30 = test_30[target_30]
{% endhighlight %}

마찬가지의 방식으로, 최적 하이퍼 파라미터를 찾기 위해 RandomizedSearchCV를 적용했다.

{% highlight ruby %}
# 하이퍼 파라미터를 최적화한 RandomForestClassifier
pipe_30 = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(random_state = 999)
)

dists = {
    "randomforestclassifier__n_estimators" : randint(50, 1000),
    "randomforestclassifier__max_depth" : [5, 10, 15, 20, 25, None],
    "randomforestclassifier__max_leaf_nodes" : [10, 20, 30, 40],
    "randomforestclassifier__max_features" : randint(1, 10),
    "randomforestclassifier__min_samples_leaf" : randint(1, 10)
}

clf_30 = RandomizedSearchCV(
    pipe_30,
    param_distributions = dists,
    n_iter = 100,
    cv = 15,
    scoring = "accuracy",
    verbose = 1,
    n_jobs = -1,
    random_state = 999
)

clf_30.fit(X_train_30, y_train_30);
{% endhighlight %}

최적 하이퍼 파라미터와 Cross Validation 평균 정확도를 산출한 후, 최적 하이퍼 파라미터를 적용한 모델을 train set에 재학습시키고 train과 test set의 정확도를 산출했다.

{% highlight ruby %}
# 최적 하이퍼 파라미터 / CV score
print("최적 하이퍼파라미터: ", clf_30.best_params_, "\n")
print("CV accuracy score: ", clf_30.best_score_)

pipe_30 = clf_30.best_estimator_
{% endhighlight %}
<img width="977" alt="스크린샷 2020-10-25 오후 9 23 22" src="https://user-images.githubusercontent.com/70478154/97106988-5947ff80-1708-11eb-8f8c-0f3f55750685.png">

{% highlight ruby %}
# 최적 하이퍼 파라미터를 적용한 모델의 train, test set 정확도
fi_pipe_30 = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators = 224, max_depth = 20, max_features = 8, max_leaf_nodes = 10,
                               min_samples_leaf = 1, n_jobs = -1, random_state = 999)
)

fi_pipe_30.fit(X_train_30, y_train_30)
print("Train set accuracy score: ", fi_pipe_30.score(X_train_30, y_train_30))

print("Test set accuracy score: ", fi_pipe_30.score(X_test_30, y_test_30))
{% endhighlight %}
<img width="394" alt="스크린샷 2020-10-25 오후 9 25 06" src="https://user-images.githubusercontent.com/70478154/97107021-98765080-1708-11eb-8ec9-df7a94748544.png">

결과에 따르면, 여전히 Overfitting 문제가 존재하지만, 시기를 더 길게 조정했을 때, test set의 정확도가 올라간 것을 확인할 수 있다.<BR/><BR/>

#### 3개월 뒤의 주가 방향 예측

train과 test set을 분리하고 baseline을 확인했다. `1`의 빈도가 가장 높았고, baseline을 41.6%로 설정했다. 그 후 feature와 target을 분리했다.

{% highlight ruby %}
# train, test set 분리 / date 칼럼 삭제 (3개월)
train_90 = finance_90[finance_90["date"] < "2018.04.01"].drop("date", axis = 1)
test_90 = finance_90[finance_90["date"] >= "2018.04.01"].drop("date", axis = 1)

train_90.shape, test_90.shape
{% endhighlight %}
<img width="198" alt="스크린샷 2020-10-25 오후 9 28 09" src="https://user-images.githubusercontent.com/70478154/97107093-0589e600-1709-11eb-8184-8dade91ab64b.png">

{% highlight ruby %}
# baseline
train_90["predict"].value_counts(normalize = True)
{% endhighlight %}
<img width="262" alt="스크린샷 2020-10-25 오후 9 28 47" src="https://user-images.githubusercontent.com/70478154/97107110-1cc8d380-1709-11eb-8333-57bf646a4f3e.png">

{% highlight ruby %}
# features, target 분리
target_90 = "predict"
features_90 = train_90.drop(columns = [target_90]).columns

X_train_90 = train_90[features_90]
y_train_90 = train_90[target_90]

X_test_90 = test_90[features_90]
y_test_90 = test_90[target_90]
{% endhighlight %}

역시 최적 하이퍼 파라미터를 찾기 위해 RandomizedSearchCV를 적용했다. 최적 하이퍼 파라미터와 Cross Validation 평균 정확도를 산출한 후, 최적 하이퍼 파라미터를 적용한 모델을 train set에 재학습시키고 train과 test set의 정확도를 산출했다.

{% highlight ruby %}
# 하이퍼 파라미터를 최적화한 RandomForestClassifier
pipe_90 = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(random_state = 999)
)

dists = {
    "randomforestclassifier__n_estimators" : randint(50, 1000),
    "randomforestclassifier__max_depth" : [5, 10, 15, 20, 25, None],
    "randomforestclassifier__max_leaf_nodes" : [10, 20, 30, 40],
    "randomforestclassifier__max_features" : randint(1, 10),
    "randomforestclassifier__min_samples_leaf" : randint(1, 10)
}

clf_90 = RandomizedSearchCV(
    pipe_90,
    param_distributions = dists,
    n_iter = 100,
    cv = 15,
    scoring = "accuracy",
    verbose = 1,
    n_jobs = -1,
    random_state = 999
)

clf_90.fit(X_train_90, y_train_90);
{% endhighlight %}

{% highlight ruby %}
# 최적 하이퍼 파라미터 / CV score
print("최적 하이퍼파라미터: ", clf_90.best_params_)
print("accuracy score: ", clf_90.best_score_)

pipe_90 = clf_90.best_estimator_
{% endhighlight %}
<img width="979" alt="스크린샷 2020-10-25 오후 9 32 46" src="https://user-images.githubusercontent.com/70478154/97107202-ab3d5500-1709-11eb-87ad-cf739a924126.png">

{% highlight ruby %}
# 최적 하이퍼 파라미터를 적용한 모델의 train, test set 정확도
fi_pipe_90 = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators = 89, max_depth = 5, max_features = 1, max_leaf_nodes = 20,
                               min_samples_leaf = 5, n_jobs = -1, random_state = 999)
)

fi_pipe_90.fit(X_train_90, y_train_90)
print("Train set accuracy score: ", fi_pipe_90.score(X_train_90, y_train_90))

print("Test set accuracy score: ", fi_pipe_90.score(X_test_90, y_test_90))
{% endhighlight %}
<img width="385" alt="스크린샷 2020-10-25 오후 9 33 34" src="https://user-images.githubusercontent.com/70478154/97107227-c5773300-1709-11eb-8109-d098f31e4a9c.png">

결과에 따르면, 위의 결과보다는 Overfitting이 약간 줄었고, test set의 정확도가 증가했음을 알 수 있다.<BR/><BR/>

#### 6개월 뒤의 주가 방향 예측

train과 test set을 분리하고 baseline을 확인했다. `-1`의 빈도가 가장 높았고, baseline을 40.2%로 설정했다. 그 후 feature와 target을 분리했다.

{% highlight ruby %}
# train, test set 분리 / date 칼럼 삭제 (6개월)
train_180 = finance_180[finance_180["date"] < "2017.11.01"].drop("date", axis = 1)
test_180 = finance_180[finance_180["date"] >= "2017.11.01"].drop("date", axis = 1)

train_180.shape, test_180.shape
{% endhighlight %}
<img width="199" alt="스크린샷 2020-10-25 오후 9 37 06" src="https://user-images.githubusercontent.com/70478154/97107318-43d3d500-170a-11eb-8e1f-c541b62bc97a.png">

{% highlight ruby %}
# baseline
train_180["predict"].value_counts(normalize = True)
{% endhighlight %}
<img width="255" alt="스크린샷 2020-10-25 오후 9 38 29" src="https://user-images.githubusercontent.com/70478154/97107364-754ca080-170a-11eb-8803-88972f286ad8.png">

{% highlight ruby %}
# features, target 분리
target_180 = "predict"
features_180 = train_180.drop(columns = [target_180]).columns

X_train_180 = train_180[features_180]
y_train_180 = train_180[target_180]

X_test_180 = test_180[features_180]
y_test_180 = test_180[target_180]
{% endhighlight %}

마지막으로, 최적 하이퍼 파라미터를 찾기 위해 RandomizedSearchCV를 적용했다. 최적 하이퍼 파라미터와 Cross Validation 평균 정확도를 산출한 후, 최적 하이퍼 파라미터를 적용한 모델을 train set에 재학습시키고 train과 test set의 정확도를 산출했다.

{% highlight ruby %}
# 하이퍼 파라미터를 최적화한 RandomForestClassifier
pipe_180 = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(random_state = 999)
)

dists = {
    "randomforestclassifier__n_estimators" : randint(50, 1000),
    "randomforestclassifier__max_depth" : [5, 10, 15, 20, 25, None],
    "randomforestclassifier__max_leaf_nodes" : [10, 20, 30, 40],
    "randomforestclassifier__max_features" : randint(1, 10),
    "randomforestclassifier__min_samples_leaf" : randint(1, 10)
}

clf_180 = RandomizedSearchCV(
    pipe_180,
    param_distributions = dists,
    n_iter = 100,
    cv = 15,
    scoring = "accuracy",
    verbose = 1,
    n_jobs = -1,
    random_state = 999
)

clf_180.fit(X_train_180, y_train_180);
{% endhighlight %}

{% highlight ruby %}
# 최적 하이퍼 파라미터 / CV score
print("최적 하이퍼파라미터: ", clf_180.best_params_, "\n")
print("CV accuracy score: ", clf_180.best_score_)

pipe_180 = clf_180.best_estimator_
{% endhighlight %}
<img width="981" alt="스크린샷 2020-10-25 오후 9 41 21" src="https://user-images.githubusercontent.com/70478154/97107433-dd02eb80-170a-11eb-9e0d-3ba6bf2d6286.png">

{% highlight ruby %}
# 최적 하이퍼 파라미터를 적용한 모델의 train, test set 정확도
fi_pipe_180 = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators = 561, max_depth = 7, max_features = 7, max_leaf_nodes = 10,
                               min_samples_leaf = 6, n_jobs = -1, random_state = 999)
)

fi_pipe_180.fit(X_train_180, y_train_180)
print("Train set accuracy score: ", fi_pipe_180.score(X_train_180, y_train_180))

print("Test set accuracy score: ", fi_pipe_180.score(X_test_180, y_test_180))
{% endhighlight %}
<img width="393" alt="스크린샷 2020-10-25 오후 9 43 00" src="https://user-images.githubusercontent.com/70478154/97107467-19364c00-170b-11eb-8510-3c738f328bef.png">

결과에 따르면, 3개월 뒤의 주가 방향 예측 결과보다 test set의 정확도가 살짝 하락했지만, 이전의 결과들에 비해 Overfitting 문제가 가장 많이 줄어들었음을 알 수 있다.<BR/><BR/><BR/><BR/>

## 결론

결과를 종합하면, 상대적으로 낮은 Overfitting 문제와 높은 예측력을 보인 **6개월 뒤의 주가 방향 예측 모델**이 가장 효과적이라고 할 수 있다. 이 모델에 따르면 `64%`의 정확도를 가지고 예측한다는 것을 알 수 있다. 비록 이 모델에 비해 **3개월 뒤의 주가 방향 예측 모델**은 상대적으로 Overfitting 문제가 더 발생하지만, 거의 비슷한 예측력을 보이기 때문에 상황에 맞게 활용할 수 있다. 이처럼 시기가 길어질수록 모델의 성능이 더 좋아지는 이유는, 활용한 지표들의 변화가 시간을 두고 경제 전반에, 개별 기업 주가에 영향을 주기 때문이라고 할 수 있다.<BR/><BR/><BR/><BR/>

## 한계

* 일반화하기에 충분한 양의 데이터가 필요하다.
* 모델의 성능을 더 높일 수 있는 feature들을 찾아 적용할 필요가 있다.
* Overfitting 문제에 대한 해결이 필요하다.
