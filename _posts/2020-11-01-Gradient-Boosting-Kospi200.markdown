---
layout: post
title:  Gradient Boosting을 활용한 Kospi200 주가 방향 예측
date:   2020-11-01
image:  computer.png
tags:   Data Finance
---
## Gradient Boosting이란?

**Gradient Boosting** 모델 역시 트리 앙상블 모델 중 하나지만, Random Forest와는 차이가 있다. `Boosting` 기법은 weak learner를 결합하여 strong learner를 만드는 방식이다. 정확도가 낮더라도 일단 모델을 만든 후, 다음 모델들이 약점을 보완하는 방식으로 예측이 이루어진다. 그 중, Gradient Boosting은 잔차가 큰 데이터를 더 학습하도록 함으로써, 손실 함수를 최소화한다.

여기서 사용할 `XGBoost`는 Extreme Gradient Boosting이라고 부르며, Gradient Boosting 알고리즘 중 하나이다. 이 모델의 장점은 학습과 예측이 빠르고, 과적합을 규제할 수 있다. 하지만, 하이퍼 파라미터에 굉장히 민감하다는 단점이 존재한다.<BR/><BR/><BR/><BR/>

## 분석 목적

해외 증시, 주요 경제, 금융 지표들을 바탕으로 Kospi200 **주가의 방향을 얼마나 효과적으로 예측**할 수 있는지 알아보고자 한다. 또한, 주가 방향 예측에 **가장 영향을 미치는 요소**들을 파악하고자 한다. 더 나아가, [이전 포스트](https://yut-a.github.io/2020/10/25/RandomForest-Kospi200/)의 RandomForest 모델에서 가장 높은 정확도를 보였던 `6개월 뒤의 주가 방향 예측` 결과를 비교해보고자 한다.<BR/><BR/><BR/><BR/>

## 데이터 소개

target 데이터인 Kospi200은 `Kodex200 ETF`의 주가를 활용했다. 일자 별 데이터들을 바탕으로 예측을 하는 것이기 때문에 다음 날 종가와 `6개월` 뒤의 종가와 비교하여 수익률을 산출했고, 이를 **수익률 > 0**이면 `1`, **수익률 <= 0**이면 `0`으로 분류했다.

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
* `달러 선물 인덱스` : 달러인덱스는 유로, 엔, 파운드, 캐나다 달러, 스웨덴 크로네, 스위스 프랑 등 경제 규모가 크거나 통화가치가 안정적인 6개국 통화를 기준으로 산정한 미 달러화 가치를 지수화한 것으로, 이에 대한 선물지수이다.
* `VIX` : S&P500 지수옵션의 향후 30일 간의 변동성에 대한 시장의 기대를 나타낸 지수이다.
* `VKOSPI`: KOSPI200 지수옵션의 미래변동성을 측정한 지수이다.

target 기준에 따라 대략 2009년 4월부터 2020년 1월까지의 데이터를 활용했다.<BR/><BR/><BR/><BR/>

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

먼저, `경기선행지수`, `수출증가율`, `콜금리`, `CD 금리` 데이터를 정리했다. 수출금액지수를 바탕으로 수출증가율을 산출했다. 경기선행지수와 수출증가율은 월 단위 데이터이기 때문에 해당 달의 데이터로 나머지 일자를 채웠다. 또한, 2020년 9월과 10월 데이터는 아직 발표되지 않았기 때문에 8월 데이터로 채웠다.

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

다음날 종가 대비 6개월 뒤 종가의 수익률을 산출하고 target을 기준에 따라 `1, 0`으로 변경했다.

{% highlight ruby %}
# target 변경 (target = 6개월 후)
finance_180 = finance.copy()

finance_180["lag_1"] = finance_180["price"].shift()
finance_180["lag_181"] = finance_180["lag_1"].shift(180)
finance_180["price_pred(%)"] = round((finance_180["lag_181"] - finance_180["lag_1"]) / finance_180["lag_1"] * 100, 2)

def rate_180(x):
    if x > 0:
        return 1
    
    elif x <= 0:
        return 0
    
finance_180["predict"] = finance_180["price_pred(%)"].apply(rate_180)
finance_180 = finance_180.dropna(axis = 0).reset_index(drop = True)
finance_180 = finance_180.astype({"predict" : "int"})

finance_180 = finance_180.drop(["price", "lag_1", "lag_181", "price_pred(%)"], axis = 1)

finance_180
{% endhighlight %}
<img width="986" alt="스크린샷 2020-11-02 오후 4 26 06" src="https://user-images.githubusercontent.com/70478154/97840900-34ffaa80-1d28-11eb-8cd4-2a7b134c80d5.png">

#### Gradient Boosting의 6개월 뒤 주가 방향 예측

train과 test set을 분리하고 baseline을 확인했다. `1`의 빈도가 가장 높았고, baseline을 `59.7%`로 설정했다. 그 후, feature와 target을 분리했다.

{% highlight ruby %}
# train, test set 분리 / date 칼럼 삭제 (6개월)
train_180 = finance_180[finance_180["date"] < "2017.11.01"].drop("date", axis = 1)
test_180 = finance_180[finance_180["date"] >= "2017.11.01"].drop("date", axis = 1)

train_180.shape, test_180.shape
{% endhighlight %}
<img width="200" alt="스크린샷 2020-11-02 오후 4 29 12" src="https://user-images.githubusercontent.com/70478154/97841148-9758ab00-1d28-11eb-8b11-25d7652efd51.png">

{% highlight ruby %}
# baseline
train_180["predict"].value_counts(normalize = True)
{% endhighlight %}
<img width="256" alt="스크린샷 2020-11-02 오후 4 30 16" src="https://user-images.githubusercontent.com/70478154/97841224-bbb48780-1d28-11eb-94f7-ef0acfa70e8a.png">

{% highlight ruby %}
# features, target 분리
target_180 = "predict"
features_180 = train_180.drop(columns = [target_180]).columns

X_train_180 = train_180[features_180]
y_train_180 = train_180[target_180]

X_test_180 = test_180[features_180]
y_test_180 = test_180[target_180]
{% endhighlight %}

XGBClassifier 모델의 최적 하이퍼 파라미터를 찾기 위해 RandomizedSearchCV를 적용했다. 최적 하이퍼 파라미터와 Cross Validation 평균 정확도를 산출한 후, 최적 하이퍼 파라미터를 적용한 모델을 train set에 재학습시키고 train과 test set의 정확도를 산출했다.

{% highlight ruby %}
# 하이퍼 파라미터를 최적화한 XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

standard = StandardScaler()
X_train_180_s = standard.fit_transform(X_train_180)

model = XGBClassifier(random_state = 12)

params = {
    "max_depth" : [5, 10, 15, 20],
    "min_child_weight" : [1, 5, 10, 15, 20],
    "learning_rate" : [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample" : [0.1, 0.2, 0.3, 0.4, 0.5],
    "n_estimators" : randint(100, 1000),
    "gamma" : [0, 0.25, 0.5, 0.7, 1.0]
}

clf = RandomizedSearchCV(
    model,
    params,
    n_iter = 100,
    cv = 15,
    n_jobs = -1,
    scoring = "accuracy",
    verbose = 1,
    random_state = 12
)

clf.fit(X_train_180_s, y_train_180);
{% endhighlight %}

{% highlight ruby %}
# 최적 하이퍼 파라미터 / CV score
print("최적 하이퍼파라미터: ", clf.best_params_, "\n")
print("CV accuracy score: ", clf.best_score_)
{% endhighlight %}
<img width="980" alt="스크린샷 2020-11-02 오후 4 34 12" src="https://user-images.githubusercontent.com/70478154/97841503-4ac19f80-1d29-11eb-8afe-fa47ced13e12.png">

{% highlight ruby %}
# 최적 하이퍼 파라미터를 적용한 모델의 train, test set 정확도
model_GB = XGBClassifier(
    n_estimators = 323,
    max_depth = 10,
    min_child_weight = 20,
    subsample = 0.5,
    learning_rate = 0.01,
    gamma = 1.0,
    random_state = 12,
    n_jobs = -1,
    )
    
X_test_180_s = standard.fit_transform(X_test_180)
    
model_GB.fit(X_train_180_s, y_train_180)

print("Train set accuracy score: ", model_GB.score(X_train_180_s, y_train_180))
print("Test set accuracy score: ", model_GB.score(X_test_180_s, y_test_180))
{% endhighlight %}
<img width="400" alt="스크린샷 2020-11-02 오후 4 37 23" src="https://user-images.githubusercontent.com/70478154/97841710-be63ac80-1d29-11eb-84b3-9155f088a79e.png">

결과에 따르면, Overfitting 문제가 존재하기는 하지만, Test set의 정확도는 `77%`로 높은 정확도를 보임을 알 수 있다.

다음은, confusion matrix와 ROC-AUC score를 산출했다.

{% highlight ruby %}
# confusion matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

flg, ax = plt.subplots()
pcm = plot_confusion_matrix(model_GB, X_test_180_s, y_test_180,
                           cmap = plt.cm.Blues,
                           ax = ax);

plt.title(f"Confusion matrix, n = {len(y_test_180)}", fontsize = 15)
plt.show()
{% endhighlight %}
<img width="320" alt="스크린샷 2020-11-02 오후 4 48 25" src="https://user-images.githubusercontent.com/70478154/97842642-48f8db80-1d2b-11eb-8128-b488f509c694.png">

{% highlight ruby %}
# ROC-AUC score
from sklearn.metrics import roc_auc_score

GB_pred_proba = model_GB.predict_proba(X_test_180_s)[:,1]
print("ROC-AUC score: ", roc_auc_score(y_test_180, GB_pred_proba))
{% endhighlight %}
<img width="306" alt="스크린샷 2020-11-02 오후 4 50 00" src="https://user-images.githubusercontent.com/70478154/97842774-82314b80-1d2b-11eb-8789-4f01fdb1f5c3.png"><BR/><BR/>

#### RandomForest의 6개월 뒤 주가 방향 예측

이번에는 RandomForest 모델의 최적 하이퍼 파라미터를 찾기 위해 RandomizedSearchCV를 적용했다. 최적 하이퍼 파라미터와 Cross Validation 평균 정확도를 산출한 후, 최적 하이퍼 파라미터를 적용한 모델을 train set에 재학습시키고 train과 test set의 정확도를 산출했다.

{% highlight ruby %}
# 하이퍼 파라미터를 최적화한 RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

pipe_180 = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(random_state = 12)
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
    random_state = 12
)

clf_180.fit(X_train_180, y_train_180);
{% endhighlight %}

{% highlight ruby %}
# 최적 하이퍼 파라미터 / CV score
print("최적 하이퍼파라미터: ", clf_180.best_params_, "\n")
print("CV accuracy score: ", clf_180.best_score_)
{% endhighlight %}
<img width="977" alt="스크린샷 2020-11-02 오후 5 01 48" src="https://user-images.githubusercontent.com/70478154/97843729-25cf2b80-1d2d-11eb-8481-c103a7a24d04.png">

{% highlight ruby %}
# 최적 하이퍼 파라미터를 적용한 모델의 train, test set 정확도
fi_pipe_180 = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators = 695, max_depth = 25, max_features = 1, max_leaf_nodes = 10,
                               min_samples_leaf = 4, n_jobs = -1, random_state = 12)
)

fi_pipe_180.fit(X_train_180, y_train_180)
print("Train set accuracy score: ", fi_pipe_180.score(X_train_180, y_train_180))

print("Test set accuracy score: ", fi_pipe_180.score(X_test_180, y_test_180))
{% endhighlight %}
<img width="400" alt="스크린샷 2020-11-02 오후 5 02 47" src="https://user-images.githubusercontent.com/70478154/97843830-48f9db00-1d2d-11eb-8b92-aa32313071a3.png">

결과에 따르면, XGBoost에 비해 Overfitting 문제가 소폭 줄었고, test set의 정확도가 `78.9%` 소폭 증가했다.

마찬가지로, confusion matrix와 ROC-AUC score를 산출했다.

{% highlight ruby %}
# confusion matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

flg, ax = plt.subplots()
pcm = plot_confusion_matrix(fi_pipe_180, X_test_180, y_test_180,
                           cmap = plt.cm.Blues,
                           ax = ax);

plt.title(f"Confusion matrix, n = {len(y_test_180)}", fontsize = 15)
plt.show()
{% endhighlight %}
<img width="322" alt="스크린샷 2020-11-02 오후 5 06 19" src="https://user-images.githubusercontent.com/70478154/97844105-c58cb980-1d2d-11eb-86d6-d0e4a18a3309.png">

{% highlight ruby %}
# ROC-AUC score
from sklearn.metrics import roc_auc_score

RF_pred_proba = fi_pipe_180.predict_proba(X_test_180)[:,1]
print("ROC-AUC score: ", roc_auc_score(y_test_180, RF_pred_proba))
{% endhighlight %}
<img width="300" alt="스크린샷 2020-11-02 오후 5 07 22" src="https://user-images.githubusercontent.com/70478154/97844190-ee14b380-1d2d-11eb-8d95-8616a9558301.png">

XGBoost와 RandomForest 모델의 confusion matrix를 비교해보면, RandomForest 모델이 상대적으로 이상적인 예측 구성을 보여줌을 알 수 있으며, ROC-AUC score 역시 소폭 높았다.<BR/><BR/><BR/><BR/>

## 결론

결과를 종합하면, `XGBoost`와 `RandomForest` 모델 모두 비슷한 예측 정확도를 보이며, Overfitting 정도와 ROC-AUC score 역시 큰 차이가 없었다. 다만 RandomForest 모델이 전반적인 모형의 성능 측면에서 매우 미세하게 앞서고 있음을 확인했다.

분석을 진행하면서, XGBoost 모델의 경우 하이퍼 파라미터의 변화에 따라 모형의 성능이 매우 민감하게 반응한다는 것을 체감했다. 따라서, 더 세부적으로 하이퍼 파라미터에 변화를 준다면, 성능의 개선을 기대해 볼 수 있을 것이라 생각한다.<BR/><BR/><BR/><BR/>

## 한계

* 일반화하기에 충분한 양의 데이터가 필요하다.
* 모델의 성능을 더 높일 수 있는 feature들을 찾아 적용할 필요가 있다.
* Overfitting 문제에 대한 해결이 필요하다.
* 두 모델 모두 `class 1`의 precision에 대한 개선이 필요하다.
