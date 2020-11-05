---
layout: post
title:  주식 종목 추천 모델
date:   2020-11-05
image:  choice.jpg
tags:   Data Finance
---
## 분석 목적

투자를 위한 분석에는 다양한 종류가 있다. 가치투자를 위해 저평가된 주식 종목들을 발굴하는 분석도 있고, 차트의 흐름을 보는 기술적 분석도 있다. 애널리스트처럼 기업 탐방을 통해 비정형화된 정보를, 재무, 회계 지표와 같은 정형화된 정보와 종합하여 투자 판단을 내리기도 한다.

그 외에 `퀀트 투자`라는 것이 있다. 주관적 판단에 의해 비합리적인 선택을 할 수 있기 때문에 감정을 배제한 객관적 투자를 지향한다. 이 때문에 Data(Number) Driven Investment 라고도 한다. 가장 보편적으로 알려진 전략이 `저PER` `저PBR`전략이다. PER(주가수익비율)과 PBR(주가순자산비율)은 각각 `주가/주당순이익` `주가/주당순자산`으로 산출되는 지표이다. 즉, 저PER, 저PBR이라는 것은 주당순이익, 주당순자산 대비 현재 주가가 낮다는 것으로, 현재 해당 종목이 주식 시장에서 저평가되어 있고, 앞으로 오를 가능성이 높다는 것을 의미하기도 한다. 이러한 퀀트 전략을, Historical Data를 기반으로 백테스팅을 통해 유의미한 성과가 있음을 증명한다.

그러나, 백테스팅을 통한 증명 및 전략 수정은 과적합으로부터 자유롭지 못하다. 증명 및 수정의 과정에서 모델이 왜곡될 가능성이 크기 때문이다. Out-of-sample을 활용한 검증 과정이 필수적이지만, 이를 활용한 결과가 마찬가지로 좋은 성과를 만들어내기는 쉽지 않다. 

따라서, 이러한 퀀트 모델의 한계점을 보완하기 위해 머신러닝 모델을 이용하여 **어떤 종목에 투자하면 좋을지** 알아보고자 한다. 또한, 퀀트 전략에서 주로 사용하는 데이터인 **PER, PBR 등을 활용하여 예측을 하는 것이 효과적인지** 알아보고자 한다. 더 나아가, 이러한 **지표들이 모델에서 얼만큼의 영향력을 가지는지, 결과와 어떤 관계를 가지는지** 알아보고자 한다.<BR/><BR/><BR/><BR/>

## 데이터 소개

**코스피, 코스닥 시장에 상장되어 있는 종목 중 약 1500개의 종목**을 활용했다. 사용한 feature 특성 상 금융 관련 종목들은 제외했고, SPAC(기업인수목적회사), 리츠 등 특수한 형태의 종목들과 관리종목, 거래정지종목들을 제외했다.

2016년부터 2019년까지 각 종목별 사업보고서 공시 데이터를 사용했다. 예측을 위해 사용한 feature는 유동비율, 부채비율, ROA 등과 같은 `재무비율` 데이터 21개와 PER, PBR, EPS 등과 같은 `투자지표` 데이터 18개로, 총 39개를 사용했다.

* 영업이익증가율, EBITDA증가율, EPS증가율의 `흑자전환` `적자전환` `적자지속`과 같은 categorical data는 새로운 feature인 영업이익증가율_cat, EBITDA증가율_cat, EPS증가율_cat을 만들어 `1, -1, -2`로 변환했다.
* 배당성향의 결측치는 모두 `0`으로 채웠다.
* 나머지 데이터들의 결측치는 각 feature 별 `평균`으로 채웠다.
* 사용한 feature들 중 `완전잠식`이 있는 경우 해당 데이터를 제거했고, 분석의 일관성을 위해 매년 12년 결산을 제외한 데이터들은 제거했다.

target은 사업보고서 공시 이후 3개월 뒤의 수익률을 산출하여 `상승`이면 `1`, `하락`이면 `0`으로 설정했다. 일반적으로, 12월 결산 사업보고서는 다음 해 3월 말 혹은 4월 초에 공시가 되기 때문에, 다음 해 4월 10일(공휴일이면 그 다음주 월요일) 가격 대비 3개월 뒤 가격의 수익률을 산출했다.<BR/><BR/><BR/><BR/>

## 적용 과정

데이터 수집과 전처리는 `다음`과 같은 과정을 거쳤다.

정리하여 저장한 데이터를 불러왔다.

{% highlight ruby %}
# 데이터 불러오기
import pandas as pd

all_stock_info_3M = pd.read_csv("/Users/yut_a_/Desktop/all_stock_info_3M.csv")
all_stock_info_3M
{% endhighlight %}
<img width="985" alt="스크린샷 2020-11-05 오후 10 57 01" src="https://user-images.githubusercontent.com/70478154/98249998-44962200-1fba-11eb-97ee-76d0a6973042.png">

배당성향의 결측치을 0으로 채웠다.

{% highlight ruby %}
# 배당성향 결측치 0으로 채우기
all_stock_info_3M["배당성향"] = all_stock_info_3M["배당성향"].fillna(0)
all_stock_info_3M.isnull().sum()
{% endhighlight %}
<img width="227" alt="스크린샷 2020-11-05 오후 10 57 45" src="https://user-images.githubusercontent.com/70478154/98250166-78714780-1fba-11eb-9e70-feb46d247049.png">

다음은, 사업보고서 공시 직후 가격 대비 3개월 뒤의 수익률을 산출했다.

{% highlight ruby %}
# 가격 변화율
all_stock_info_3M["change(%)"] = round(((all_stock_info_3M["after_3M"] - all_stock_info_3M["price"]) / all_stock_info_3M["price"]) * 100, 2)
all_stock_info_3M
{% endhighlight %}
<img width="993" alt="스크린샷 2020-11-05 오후 11 00 10" src="https://user-images.githubusercontent.com/70478154/98250389-b53d3e80-1fba-11eb-9b25-222c3cfd7700.png">

산출된 수익률을 바탕으로, 상승이면 1, 하락이면 0으로 target을 만들었다. 그 후, train과 test set을 분리했고, 분석을 위해 필요없는 칼럼을 제거했다.

{% highlight ruby %}
# target 생성
def rate_3M(x):
    if x > 0:
        return 1
    
    elif x <= 0:
        return 0
    
all_stock_info_3M["grade"] = all_stock_info_3M["change(%)"].apply(rate_3M)
{% endhighlight %}

{% highlight ruby %}
# train, test set 분리
from sklearn.model_selection import train_test_split
train_3M_all, test_3M_all = train_test_split(all_stock_info_3M, train_size = 0.8, stratify = all_stock_info_3M["grade"], random_state = 999)

train_3M_all.shape, test_3M_all.shape
{% endhighlight %}
<img width="211" alt="스크린샷 2020-11-05 오후 11 02 32" src="https://user-images.githubusercontent.com/70478154/98250642-0a795000-1fbb-11eb-8b3c-e01fddb56018.png">

{% highlight ruby %}
# 분석을 위해 필요없는 칼럼 제거
train_3M = train_3M_all.drop(["time", "stock", "code", "price", "after_3M", "change(%)"], axis = 1)
test_3M = test_3M_all.drop(["time", "stock", "code", "price", "after_3M", "change(%)"], axis = 1)
test_3M
{% endhighlight %}
<img width="980" alt="스크린샷 2020-11-05 오후 11 03 47" src="https://user-images.githubusercontent.com/70478154/98250755-34327700-1fbb-11eb-896f-daab40c0fa07.png">

baseline을 확인한 결과, `0`의 빈도가 가장 높았고, baseline을 `52.67%`로 설정했다.

{% highlight ruby %}
# baseline
train_3M["grade"].value_counts(normalize = True)
{% endhighlight %}
<img width="246" alt="스크린샷 2020-11-05 오후 11 05 48" src="https://user-images.githubusercontent.com/70478154/98250976-7b206c80-1fbb-11eb-96cf-1d26691c0f80.png">

feature와 target을 분리한 후, 하이퍼 파라미터를 최적화한 `XGBClassifier` 모델을 구축하기 위해 RandomizedSearchCV를 진행했다. 최적 하이퍼 파라미터와 Cross Validation 평균 정확도를 산출한 후, 이를 기반으로 하이퍼 파라미터를 조정하면서 성능을 개선시켰다. 그 후, train set에 재학습시키고 train과 test set의 정확도를 산출했다.

{% highlight ruby %}
# 하이퍼 파라미터를 최적화한 XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

pipe_3M = make_pipeline(
    SimpleImputer(strategy = "mean"),
    MinMaxScaler(),
    XGBClassifier(scale_pos_weight = 1.1, random_state = 99)
)

params = {
    "xgbclassifier__max_depth" : [5, 10, 15, 20, 25],
    "xgbclassifier__min_child_weight" : [10, 15, 20, 25, 30],
    "xgbclassifier__learning_rate" : [0.001, 0.005, 0.01, 0.05],
    "xgbclassifier__subsample" : [0.3, 0.4, 0.5],
    "xgbclassifier__n_estimators" : randint(100, 1000),
    "xgbclassifier__gamma" : [0, 0.25, 0.5, 0.7, 1.0]
}

clf_3M = RandomizedSearchCV(
    pipe_3M,
    params,
    n_iter = 100,
    cv = 5,
    scoring = "f1_weighted",
    verbose = 1,
    n_jobs = -1,
    random_state = 99
)

clf_3M.fit(X_train_3M, y_train_3M)
{% endhighlight %}

{% highlight ruby %}
# 최적 하이퍼 파라미터 / CV score
print("최적 하이퍼파라미터: ", clf_3M.best_params_, "\n")
print("CV accuracy score: ", clf_3M.best_score_)
{% endhighlight %}
<img width="980" alt="스크린샷 2020-11-05 오후 11 21 37" src="https://user-images.githubusercontent.com/70478154/98252829-b459dc00-1fbd-11eb-9ddd-a4c14392c81a.png">

{% highlight ruby %}
# 최적 하이퍼 파라미터를 적용한 모델의 train, test set 정확도
from sklearn.model_selection import cross_val_score

fi_pipe_3M = make_pipeline(
    SimpleImputer(strategy = "mean"),
    MinMaxScaler(),
    XGBClassifier(n_estimators = 488, max_depth = 30, min_child_weight = 20, subsample = 0.3,
                  learning_rate = 0.003, gamma = 0.7, scale_pos_weight = 1.1, random_state = 99, n_jobs = -1)
)
    
fi_pipe_3M.fit(X_train_3M, y_train_3M)

print("Train set accuracy score: ", fi_pipe_3M.score(X_train_3M, y_train_3M))
print("CV score: ", cross_val_score(fi_pipe_3M, X_train_3M, y_train_3M, cv = 5).mean())
print("Test set accuracy score: ", fi_pipe_3M.score(X_test_3M, y_test_3M))
{% endhighlight %}
<img width="392" alt="스크린샷 2020-11-05 오후 11 12 10" src="https://user-images.githubusercontent.com/70478154/98251698-61335980-1fbc-11eb-935a-a4c7eecca2c0.png">

결과에 따르면, 약간의 Overfitting 문제가 존재하며, 예측 정확도가 약 `55.65%`로 많이 높지는 않지만, baseline을 상회하는 정확도를 보임을 알 수 있다.

{% highlight ruby %}
# 3개월 뒤 주가 방향 예측 모델의 confusion matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

flg, ax = plt.subplots()
pcm = plot_confusion_matrix(fi_pipe_3M, X_test_3M, y_test_3M,
                           cmap = plt.cm.Blues,
                           ax = ax);

plt.title(f"Confusion matrix, n = {len(y_test_3M)}", fontsize = 15)
plt.show()
{% endhighlight %}
<img width="319" alt="스크린샷 2020-11-05 오후 11 16 11" src="https://user-images.githubusercontent.com/70478154/98252139-efa7db00-1fbc-11eb-9a4a-4f4b4e0a48c0.png">

{% highlight ruby %}
# f1-score
from sklearn.metrics import classification_report

print(classification_report(y_test_3M, fi_pipe_3M.predict(X_test_3M)))
{% endhighlight %}
<img width="480" alt="스크린샷 2020-11-05 오후 11 16 31" src="https://user-images.githubusercontent.com/70478154/98252196-fcc4ca00-1fbc-11eb-8d3c-993472f98129.png">

Confusion matrix와 f1-score를 확인한 결과, 한 쪽으로 쏠리지 않고, 각 class가 균일하게 예측이 되었음을 확인할 수 있다. 비록, 예측 정확도가 매우 높지는 않지만, 상승 target의 비중이 약 47.3%이며 이를 상회하는 f1-score를 보이고 있기 때문에 이 모델을 사용하지 않았을 때보다 사용했을 때 더 긍정적인 결과를 기대할 수 있을 것이라 생각했다.

모델을 통해 `1`로 예측한 데이터들의 종목명과 수익률, 실제 결과를 다음과 같이 정리했다.

{% highlight ruby %}
# test set의 종목 별 수익률과 예측 및 실제 결과
result_3M = pd.DataFrame(fi_pipe_3M.predict_proba(X_test_3M))
result_3M["predict"] = fi_pipe_3M.predict(X_test_3M)
result_3M["true"] = y_test_3M.tolist()
result_3M["change(%)"] = test_3M_all["change(%)"].tolist()
result_3M["stock"] = test_3M_all["stock"].tolist()

result_3M.sort_values(by = [1], ascending = False)[:30]
{% endhighlight %}
<img width="428" alt="스크린샷 2020-11-06 오전 12 03 52" src="https://user-images.githubusercontent.com/70478154/98257932-9a22fc80-1fc3-11eb-91a4-86d434f9a554.png">

이를 바탕으로 상승을 예측한 종목들에 투자했을 때의 결과를, 전 종목 혹은 하락을 예측한 종목들에 투자했을 때의 결과와 비교했다.

{% highlight ruby %}
# 전 종목, 상승 예측, 하락 예측 투자 결과 백테스팅
import numpy as np

both = float(round(np.mean(result_3M["change(%)"].tolist()), 2))
up = float(round(np.mean(result_3M[result_3M["predict"] == 1]["change(%)"].tolist()), 2))
down = float(round(np.mean(result_3M[result_3M["predict"] == 0]["change(%)"].tolist()), 2))

rate_data = [both, up, down]
label = ["both", "up", "down"]

plt.figure(figsize = (3, 3))
plt.bar(label, rate_data, width = 0.5, color = ["gray", "red", "blue"], alpha = 0.6)

plt.text(-0.2, 4.5, "4.36%", fontsize = 8)
plt.text(0.8, 7.35, "7.19%", fontsize = 8)
plt.text(1.8, 2, "1.79%", fontsize = 8)
plt.ylim(0, 10)
plt.show()

print("전 종목 투자 결과: ", both, "%")
print("상승을 예측한 종목 투자 결과: ", up, "%")
print("하락을 예측한 종목 투자 결과: ", down, "%")
{% endhighlight %}
<img width="423" alt="스크린샷 2020-11-06 오전 12 06 43" src="https://user-images.githubusercontent.com/70478154/98258254-fede5700-1fc3-11eb-95d8-f299ded19753.png">

결과에 따르면, 상승을 예측한 종목들에 같은 비중으로 투자한 결과 `7.19%`의 수익률을 얻을 수 있음을 알 수 있다. 전 종목에 투자했을 때의 결과인 4.36%, 하락을 예측한 종목들에 투자했을 때의 결과인 1.79%와 비교했을 때 더 좋은 결과를 만들어냈다.

그렇다면, 이 예측 모델에서 결과에 가장 영향을 미치는 feature는 무엇인지 알아보고자 한다.

{% highlight ruby %}
# Permutation Importances
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import eli5
from eli5.sklearn import PermutationImportance

permuter = PermutationImportance(
    fi_pipe_3M.named_steps["xgbclassifier"],
    scoring = "accuracy",
    n_iter = 10,
    random_state = 99
)

X_test_imputed = fi_pipe_3M.named_steps["simpleimputer"].transform(X_test_3M)
X_test_scaler = fi_pipe_3M.named_steps["minmaxscaler"].transform(X_test_imputed)

permuter.fit(X_test_scaler, y_test_3M);

feature_names = X_test_3M.columns.tolist()

eli5.show_weights(
    permuter,
    top = None,
    feature_names = feature_names
)
{% endhighlight %}
<img width="225" alt="스크린샷 2020-11-06 오전 12 13 30" src="https://user-images.githubusercontent.com/70478154/98259053-f5a1ba00-1fc4-11eb-809e-145b15829ea5.png">

특성 중요도를 확인한 결과, 예측에 가장 많은 영향을 미치는 feature는 `EBITDA증가율` `유보율` `판매비와관리비증가율`임을 알 수 있다. 처음에 살펴본 PER, PBR 역시 중요도 TOP 10 안에 속한다는 것을 알 수 있다.

이 지표들에 대해 간단하게 설명을 하면 다음과 같다.

* `EBITDA증가율` : EBITDA는 이자비용, 세금, 감가상각비용 등을 빼기 전 순이익으로, 기업이 영업 활동으로 벌어들인 현금 창출 능력을 나타내는 지표이다.
* `유보율` : 기업이 영업활동을 통해 벌어들인 이익금을 사외로 유출시키지 않고 얼마나 사내에 축적해두고 있는지를 나타내는 지표이다.
* `판매비와관리비증가율` : 판관비(판매비와 관리비)는 기업의 판매와 관리, 유지에서 발생하는 비용을 통틀어 칭하는 용어로 여기에는 급여와 복리후생비, 임차료와 접대비 등이 포함된다.

마지막으로, 가장 높은 특성 중요도를 보인 `EBITDA증가율` `유보율` `판매비와관리비증가율`과 처음에 언급했던 `PER` `PBR`이 target과 어떤 관계를 가지고 있는지 알아보고자 한다.

먼저, 그래프에 따르면, `EBITDA증가율`은 대체로 target과 양의 상관관계를 가진다는 것을 알 수 있다. 즉, EBITDA증가율이 증가할수록 `상승`으로 예측할 가능성이 커진다는 의미이다. EBITDA증가율이 증가하면, 기업의 이익으로 직결되기 때문에 상승으로의 예측과 양의 상관관계가 있다는 것을 쉽게 이해할 수 있다.

{% highlight ruby %}
# PDP - EBITDA증가율
plt.rcParams['figure.dpi'] = 144

from pdpbox.pdp import pdp_isolate, pdp_plot

X_test_df = pd.DataFrame(X_test_scaler, columns = X_test_3M.columns.tolist())

feature_1 = "EBITDA증가율"

isolated = pdp_isolate(
    model = fi_pipe_3M.named_steps["xgbclassifier"],
    dataset = X_test_df,
    model_features = X_test_df.columns,
    feature = feature_1,
)

pdp_plot(isolated, feature_name = feature_1);
{% endhighlight %}
<img width="984" alt="스크린샷 2020-11-06 오전 12 48 18" src="https://user-images.githubusercontent.com/70478154/98263142-cd688a00-1fc9-11eb-8254-fe66477ef9f2.png">

두 번째로, `유보율`은 극초반에 target과 양의 상관관계를 보이다가 대체로 음의 상관관계를 가진다는 것을 확인할 수 있다. 유보율이 높으면 예상치 못한 위기 상황에 대응할 수 있는 현금이 있기 때문에 안정적인 재무 구조를 유지할 수 있지만, 유보율이 과도하게 높아진다면 기업이 마땅한 투자처를 찾지 못해 수익을 창출하지 못하고 현금만 쌓아두는 것일 수 있다. 따라서 이러한 유보율의 특징이 반영되었음을 예상해 볼 수 있다.

{% highlight ruby %}
# PDP - 유보율
plt.rcParams['figure.dpi'] = 144

from pdpbox.pdp import pdp_isolate, pdp_plot

X_test_df = pd.DataFrame(X_test_scaler, columns = X_test_3M.columns.tolist())

feature_2 = "유보율"

isolated = pdp_isolate(
    model = fi_pipe_3M.named_steps["xgbclassifier"],
    dataset = X_test_df,
    model_features = X_test_df.columns,
    feature = feature_2,
)

pdp_plot(isolated, feature_name = feature_2);
{% endhighlight %}
<img width="993" alt="스크린샷 2020-11-06 오전 12 49 12" src="https://user-images.githubusercontent.com/70478154/98263243-f0933980-1fc9-11eb-967c-093039b2af26.png">

세 번째로, `판매비와관리비증가율`은 전반적으로 `상승`으로의 예측에 양의 영향을 주고 있지만, 판매비와관리비증가율이 커질수록 그 영향의 크기는 조금씩 떨어짐을 알 수 있다. 판관비는 비용이기 때문에 일반적으로, 기업 이익에 긍정적인 영향을 줄 수만은 없다. 그러나 판관비에 속하는 비용에는 광고비, 복리후생비, 교육훈련비 등을 포함하고 있기 때문에 추후 기업의 이익으로 되돌아올 수 있는 비용이라고 할 수 있다. 이러한 점 때문에 전반적으로 `상승`으로의 예측에 긍정적인 영향을 주고 있지만, 이 비용이 과도할 경우 오히려 기업 이익에 부정적 효과를 야기할 수 있다고 해석할 수 있다.

{% highlight ruby %}
# PDP - 판매비와관리비증가율
plt.rcParams['figure.dpi'] = 144

from pdpbox.pdp import pdp_isolate, pdp_plot

X_test_df = pd.DataFrame(X_test_scaler, columns = X_test_3M.columns.tolist())

feature_3 = "판매비와관리비증가율"

isolated = pdp_isolate(
    model = fi_pipe_3M.named_steps["xgbclassifier"],
    dataset = X_test_df,
    model_features = X_test_df.columns,
    feature = feature_3,
)

pdp_plot(isolated, feature_name = feature_3);
{% endhighlight %}
<img width="987" alt="스크린샷 2020-11-06 오전 12 56 21" src="https://user-images.githubusercontent.com/70478154/98264122-ed4c7d80-1fca-11eb-9c1e-998046701b20.png">

네 번째로, `PBR`은 대체로 target과 음의 상관관계를 보임을 알 수 있다. 즉, PBR이 높을수록 `상승`으로 예측할 가능성이 낮아진다는 의미이다. 이를 통해, 처음에 언급했던 것처럼 주당순자산 대비 주가가 저평가되어 있는 종목들의 상승 기대가 높은 저PBR 전략이 이 모델에서 작동하고 있다고 해석할 수 있다.

{% highlight ruby %}
# PDP - PBR
plt.rcParams['figure.dpi'] = 144

from pdpbox.pdp import pdp_isolate, pdp_plot

X_test_df = pd.DataFrame(X_test_scaler, columns = X_test_3M.columns.tolist())

feature_4 = "PBR"

isolated = pdp_isolate(
    model = fi_pipe_3M.named_steps["xgbclassifier"],
    dataset = X_test_df,
    model_features = X_test_df.columns,
    feature = feature_4,
)

pdp_plot(isolated, feature_name = feature_4);
{% endhighlight %}
<img width="980" alt="스크린샷 2020-11-06 오전 1 08 09" src="https://user-images.githubusercontent.com/70478154/98265571-947de480-1fcc-11eb-8742-3283a34ceb8f.png">

다섯 번째로, `PER`은 target과 양의 상관관계를 보임을 알 수 있다. 즉, PER이 높을수록 `상승`으로 예측할 가능성이 커진다는 의미이다. PBR과는 반대로 처음에 언급했던 저PER 전략이 이 모델에서는 작동하지 않는다고 해석할 수 있다.

{% highlight ruby %}
# PDP - PER
plt.rcParams['figure.dpi'] = 144

from pdpbox.pdp import pdp_isolate, pdp_plot

X_test_df = pd.DataFrame(X_test_scaler, columns = X_test_3M.columns.tolist())

feature_5 = "PER"

isolated = pdp_isolate(
    model = fi_pipe_3M.named_steps["xgbclassifier"],
    dataset = X_test_df,
    model_features = X_test_df.columns,
    feature = feature_5,
)

pdp_plot(isolated, feature_name = feature_5);
{% endhighlight %}
<img width="972" alt="스크린샷 2020-11-06 오전 1 15 45" src="https://user-images.githubusercontent.com/70478154/98266424-a2803500-1fcd-11eb-8754-92e198452669.png"><BR/><BR/><BR/><BR/>

## 결론

모델의 test set 예측도는 `55.65%`로, 성능이 매우 좋지는 않지만, 모델을 사용하지 않았을 때보다는 상대적으로 좋은 투자 결과를 만들어낸다는 것을 알 수 있다. 상승을 예측한 종목들에 분산투자를 한다면, 시장 평균을 상회하는 결과를 기대할 수 있을 것이라 생각한다.

* test set에 대한 투자 백테스팅 결과, 모델을 사용하지 않았을 때의 기대수익률보다 모델을 사용했을 때의 성과가 더 좋다.
* 동일 비중으로의 투자를 가정한다면, 모델을 사용했을 때 더 적은 최소 투자 금액이 필요하므로 거래비용 감소 효과가 있다.
* 투자를 위한 개별 종목 탐색 시간을 절약할 수 있다.<BR/><BR/><BR/><BR/>

## 한계

* 예측도가 높지 않기 때문에 이 모델을 바탕으로 개별 종목에 대한 매수 추천을 할 수 없다.
* 만약 상승으로 예측한 종목들 중 폭락한 종목이 있을 경우, 모델을 통한 투자 결과에 악영향을 줄 수 있기 때문에 변동성을 고려한 target과 예측 모델을 적용할 필요가 있다.
* 1년 주기의 데이터를 바탕으로 모델을 만들었지만, target은 3개월 뒤만을 예측하므로 투자의 연속성을 유지할 수 없다.
