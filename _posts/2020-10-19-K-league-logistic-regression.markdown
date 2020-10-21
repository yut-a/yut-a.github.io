---
layout: post
title:  Logistic Regression을 통한 K리그 경기 결과 예측
date:   2020-10-19
image:  soccer.jpg
tags:   Data
---
## Logistic Regression이란?

**로지스틱 회귀(Logistic Regression)**은 데이터가 어떤 범주에 속할 확률을 0에서 1 사이의 값으로 예측한 후, 그 확률에 따라 더 가능성이 높은 범주에 속하도록 분류해주는 지도 학습 알고리즘이다. 이번 분석과 같은 승, 패 예측이나 부도 예측 등 target 변수가 1 또는 0인 이진형 변수에서 쓰이는 방법이다.<BR/><BR/><BR/><BR/>

## 분석 목적

학교 수업 시간에 다중 회귀 분석으로 축구 국가대표팀 승, 패 요인 분석이라는 주제를 가지고 과제를 수행한 적이 있다. 그 당시에는 데이터도 충분하지 않았고, 로지스틱 회귀에 대한 지식이 없었기 때문에 그 때의 분석 경험을 보충해보고자 한다. 승, 패에 영향을 미칠 수 있는 주요 데이터를 바탕으로, **어떤 요인이 경기 결과에 어떻게 영향을 줄 수 있는 지**를 파악하고자 한다. 또한, 이를 바탕으로 **경기 결과를 예측**해 보고자 한다.<BR/><BR/><BR/><BR/>

## 데이터 소개

2019년과 2020년 10월 17일까지의 `K리그` 경기 데이터를 활용했으며, 2019년의 일부 경기 데이터는 포메이션 정보 누락으로 제외했다. 이러한 데이터들은 크롤링하여 사용했다. 경기 결과 중 분석 편의를 위해 **무승부** 데이터는 제외했다. 경기 결과를 제외한 사용 데이터는 총 11개로, 다음과 같다.

`볼점유율` `슈팅` `유효슈팅` `코너킥` `파울` `오프사이드` `경고` `퇴장` `교체` `수비수` `공격수`<BR/><BR/><BR/><BR/>

## 적용 과정

먼저, 크롤링하여 K리그 경기 데이터를 불러오기 위해 함수를 만들었다. 코드에 대한 자세한 내용은 [이전 포스트](https://yut-a.github.io/2020/10/18/crawling-Selenium/)를 참고하기 바란다.

{% highlight ruby %}
# 경기 데이터 추출 함수
def football_game(code = ""):
    
    from selenium import webdriver
    import bs4
    import numpy as np
    import pandas as pd
    
    driver = webdriver.Chrome("/Users/Downloads/chromedriver")
    driver.implicitly_wait(3)
    driver.get("https://sports.daum.net/gamecenter/" + str(code) + "/highlight")
    
    html = driver.page_source
    soup = bs4.BeautifulSoup(html, "html.parser")
    
    # 팀 이름
    team_1 = soup.find_all("span", class_="txt_team")[0].text
    team_2 = soup.find_all("span", class_="txt_team")[1].text
    
    # 경기 기록
    team_1_data = []
    team_2_data = []

    for i in range(0, len(soup.find_all("span", class_="vs_graph"))):
        graph = soup.find_all("span", class_="vs_graph")[i]
        data = graph.find("span", class_="num_g").text
    
        if i == 0:
            team_1_data.append(data)
        
        elif (i % 2) == 0:
            team_1_data.append(data)
    
        else:
            team_2_data.append(data)
            
    # 스코어
    score_1 = soup.find_all("span", class_="screen_out")[2].text
    score_2 = soup.find_all("span", class_="screen_out")[4].text
    
    # 포메이션
    form_1 = soup.find_all("span", class_="team_info")[0].text
    form_1 = form_1.split("\n")[4]
    
    form_2 = soup.find_all("span", class_="team_info")[1].text
    form_2 = form_2.split("\n")[4]
    
    # 데이터 리스트화
    team_1_data.insert(0, team_1)
    team_1_data.insert(10, form_1)
    
    team_2_data.insert(0, team_2)
    team_2_data.insert(10, form_2)
    
    # 승, 패 설정
    if score_1 > score_2:
        team_1_data.insert(11, 1)
        team_2_data.insert(11, 0)
        
    elif score_1 < score_2:
        team_1_data.insert(11, 0)
        team_2_data.insert(11, 1)
    
    else:
        team_1_data.insert(11, "-")
        team_2_data.insert(11, "-")
    
    # 데이터프레임
    foot_ball = pd.DataFrame(columns = ["팀", "볼점유율", "슈팅", "유효슈팅", "코너킥", "파울", "오프사이드", "경고", "퇴장",
                                   "교체", "포메이션", "결과"])
    foot_ball.loc[0] = team_1_data
    foot_ball.loc[1] = team_2_data
    
    # 창 닫기
    driver.close()
    
    return foot_ball
{% endhighlight %}

이를 통해 정리한 데이터를 불러와 병합했다.

{% highlight ruby %}
# 데이터 불러오기
import pandas as pd

football_2020 = pd.read_csv("football_data_2020.csv")
football_2019 = pd.read_csv("football_data_2019.csv")

# 데이터 병합
football = pd.concat([football_2020, football_2019])
football.head()
{% endhighlight %}
<img width="617" alt="스크린샷 2020-10-20 오전 8 12 20" src="https://user-images.githubusercontent.com/70478154/96521357-0efbf400-12ac-11eb-8662-adfabb97b0c2.png">

필요없는 칼럼을 제거하고, 경기 결과 중 무승부 데이터들을 제거했다.

{% highlight ruby %}
# 볼점유율 숫자형으로 변경
import numpy as np

football["볼점유율"] = football["볼점유율"].str.replace("%", "")
football["볼점유율"] = football["볼점유율"].astype(np.int)

# 팀 칼럼 제거
football_data = football.drop("팀", axis = 1)

# 무승부 결과 제거
football_data = football_data[(football_data["결과"] == "1") | (football_data["결과"] == "0")]
football_data = football_data.reset_index(drop = True)
football_data.head()
{% endhighlight %}
<img width="575" alt="스크린샷 2020-10-20 오전 8 14 33" src="https://user-images.githubusercontent.com/70478154/96521474-5aae9d80-12ac-11eb-8256-97d6b0ca2767.png">

다음은, 포메이션 데이터를 바탕으로 수비수와 공격수 특성을 만들었다. 포메이션 데이터를 `OneHotEncoder`를 통해 분석에서 활용하고자 했으나 매우 큰 다중공선성이 의심되어 이와 같은 방식을 활용했다.

{% highlight ruby %}
# 포메이션으로 수비수, 공격수 수 추가
football_data_change = football_data.copy()
Df = []
Fw = []

for i in range(0, len(football_data)):
  
  num = len(football_data["포메이션"][i].split("-")) - 1

  df_data = football_data["포메이션"][i].split("-")[0]          # 수비수
  fw_data = football_data["포메이션"][i].split("-")[num]        # 공격수

  Df.append(df_data)
  Fw.append(fw_data)

football_data_change["수비수"] = Df
football_data_change["공격수"] = Fw

football_data_change["수비수"] = football_data_change["수비수"].astype(np.int)
football_data_change["공격수"] = football_data_change["공격수"].astype(np.int)

# 포메이션 제거
football_data_final = football_data_change.copy()
football_data_final = football_data_final.drop("포메이션", axis = 1)
football_data_final.head()
{% endhighlight %}
<img width="616" alt="스크린샷 2020-10-20 오전 8 18 59" src="https://user-images.githubusercontent.com/70478154/96521775-f6400e00-12ac-11eb-9dea-8d931729e4a6.png">

이로써 데이터 전처리를 완료했다. 본격적인 분석 전, 기준 모델의 정확도를 파악하고자 한다.

{% highlight ruby %}
# baseline
football_data_final["결과"].value_counts(normalize = True)
{% endhighlight %}
<img width="226" alt="스크린샷 2020-10-20 오전 8 35 30" src="https://user-images.githubusercontent.com/70478154/96522832-428c4d80-12af-11eb-907a-31c51fe54e12.png">

결과에 따르면, 승, 패 모두 50%이기 때문에 기준모델의 정확도를 50%로 설정하고 분석을 진행했다.

로지스틱 회귀 분석을 위해 train과 test set으로 나누고, feature와 target으로 데이터를 구분했다. validation set을 사용하지 않은 이유는 데이터의 양이 많지 않기 때문이며, 이를 위해 `LogisticRegressionCV`를 활용하고자 한다.

{% highlight ruby %}
# train, test set
from sklearn.model_selection import train_test_split

train, test = train_test_split(football_data_final, train_size = 0.8, stratify = football_data_final["결과"], random_state = 7777)

# feature, target
target = "결과"
features = train.drop(columns = [target]).columns

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

X_train.shape, X_test.shape
{% endhighlight %}
<img width="190" alt="스크린샷 2020-10-20 오전 8 24 44" src="https://user-images.githubusercontent.com/70478154/96522145-bf1e2c80-12ad-11eb-99cb-29d4af1a312d.png">

train과 test set으로 잘 구분이 된 것을 확인할 수 있다. 이를 바탕으로 Logistic Regression을 학습하고 예측 score를 확인해 보고자 한다.

{% highlight ruby %}
# Logistic Regression 학습
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    StandardScaler(), 
    LogisticRegressionCV(cv = 5, random_state = 0)
)
pipe.fit(X_train, y_train)

# Logistic Regression score
print("훈련 정확도: ", pipe.score(X_train, y_train))
print("테스트 정확도: ", pipe.score(X_test, y_test))
{% endhighlight %}
<img width="270" alt="스크린샷 2020-10-20 오전 8 29 27" src="https://user-images.githubusercontent.com/70478154/96522484-6bf8a980-12ae-11eb-842c-2a4809f1f1b2.png">

정확도를 보면, 테스트 정확도는 **66.97%**로 많이 높은 수치는 아니지만, baseline 보다 높은 정확도를 만들어냈고, 과적합 없이 잘 학습되었음을 알 수 있다.

위에서 만든 모델의 feature 간 다중공선성이 존재하는지 파악하고자 한다. `다중공선성`이란, 일부 feature가 다른 feature들과 높은 상관관계를 가지는 것으로, 다중공선성이 존재하지 않는 것이 선형회귀모형의 가정 중 하나이다. 일반적으로 `VIF > 10`인 경우 다중공선성이 있다고 이야기하며, 다중공선성이 존재하면 모델의 성과에 악영향을 끼칠 수 있다.

{% highlight ruby %}
# 다중공선성
from statsmodels.stats.outliers_influence import variance_inflation_factor

col = X_train.columns
pipe_m = pipe.named_steps["standardscaler"]
X_train_pipe = pipe_m.fit_transform(X_train)

X = pd.DataFrame(X_train_pipe, columns = col)

vif_data = pd.DataFrame() 
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(col))]

vif_data
{% endhighlight %}
<img width="182" alt="스크린샷 2020-10-20 오전 8 48 01" src="https://user-images.githubusercontent.com/70478154/96523577-035efc00-12b1-11eb-9fae-79dad26fd647.png">

결과에 따르면, 모든 feature의 VIF는 10보다 작으며, 모델에서 다중공선성 문제가 발생하지 않는다는 것을 알 수 있다.

마지막으로, 모델의 회귀계수를 통해 어떤 요인들이 결과에 얼마나 영향을 주는지 알아보고자 한다. 회귀계수 산출 결과와 시각화는 다음과 같다.

{% highlight ruby %}
# 회귀계수
logit = LogisticRegressionCV(cv = 5, random_state = 0)
logit.fit(X_train_pipe, y_train)

coef_logit = pd.Series(logit.coef_[0], X_train.columns)
coef_logit
{% endhighlight %}
<img width="174" alt="스크린샷 2020-10-20 오전 8 56 11" src="https://user-images.githubusercontent.com/70478154/96524044-28a03a00-12b2-11eb-8a32-e15035c0af27.png">

{% highlight ruby %}
# 한글 및 마이너스
import matplotlib as mpl

mpl.rc("font", family='AppleGothic')
mpl.rc('axes', unicode_minus=False)

# matplotlib 화질
%config InlineBackend.figure_format='retina'

# 회귀계수 시각화
import matplotlib.pyplot as plt

plt.figure(figsize = (10, 5))

color = []
for i in range(0, len(coef_logit.sort_values())):
    if coef_logit.sort_values()[i] >= 0:
        col = "salmon"
        color.append(col)
    else:
        col = "skyblue"
        color.append(col)

coef_logit.sort_values().plot.barh(color = color);
{% endhighlight %}
<img width="623" alt="스크린샷 2020-10-20 오후 8 58 30" src="https://user-images.githubusercontent.com/70478154/96583040-199fa300-1317-11eb-8cea-6a78b52e4e5a.png">

회귀계수에 따르면, `볼점유율` `코너킥` `퇴장` `공격수`가 증가하면 **승리에 부정적 영향**을 미치며, `슈팅` `유효슈팅` `오프사이드` `경고` `교체` `수비수`가 증가하면 **승리에 긍정적 영향**을 미친다는 것을 알 수 있다. 일반적으로, `볼점유율` `코너킥`이 증가하면 공격 포인트를 잡을 수 있는 가능성이 커지기 때문에 승리에 긍정적인 영향을 줄 것이라 예상할 수 있다. 또한, `오프사이드` `경고`가 증가하면 공격의 기회를 빼앗기는 것이기 때문에 승리에 부정적인 영향을 줄 것이라 예상할 수 있다. 그러나, 예상과는 반대의 결과가 나왔고 그 이유를 다음과 같이 생각해볼 수 있다.

* `볼점유율`의 경우, 예상과 반대되는 결과가 나온 이유를 K리그에서 **역습 전략**을 주로 사용하기 때문이라고 추측해볼 수 있다. 역습 전략은 상대가 공격에 실패했을 때 혹은 상대의 공격을 차단한 후, 빠른 공수 전환을 통해 골을 성공시키는 것이다. 따라서, 높은 볼점유율을 필요로 하지 않는 이 전략으로 인해 낮은 볼점유율이 승리에 더 긍정적 영향을 미친 것이라 예상해볼 수 있다.

* `코너킥`의 경우, 예상과 반대되는 결과가 나온 이유를 코너킥의 **골 성공률이 낮기** 때문이라고 예상해볼 수 있다. 한국프로축구연맹 자료에 따르면, 1983년 출범 이래 K리그에서 코너킥 73,335번 중 득점으로 연결된 경우는 1,063번으로, 약 **1.45%** 확률로 코너킥을 통해 점수가 만들어진다. 즉, 코너킥을 통해 골로 연결되는 경우가 적고, 오히려 코너킥은 슈팅 실패의 결과물로 볼 수 있기 때문에 승리에 부정적 영향을 미친 것이라 예상해볼 수 있다.

* `오프사이드`와 `경고`의 경우, 예상과 반대되는 결과가 나온 이유를 **적극적인 공격 시도와 상대 공격 차단**의 요인으로 작용했기 때문이라 추측해볼 수 있다.

추가적으로, `공격수`가 `수비수`보다 승리에 긍정적 영향을 줄 것이라는 기존의 생각과 다르게, 이 분석을 통해 K리그에서 `수비수`가 오히려 `공격수`보다 승리에 긍정적 영향을 준다는 것을 확인할 수 있었다.

Odds를 통해 조금 더 직관적으로 살펴보고자 한다.

{% highlight ruby %}
# Odds
import numpy as np
coef_odds = pd.Series(np.exp(logit.coef_)[0], X_train.columns)
coef_odds
{% endhighlight %}
<img width="173" alt="스크린샷 2020-10-20 오전 9 02 31" src="https://user-images.githubusercontent.com/70478154/96524372-0824af80-12b3-11eb-90c1-3efc1b57895c.png">

**Odds**는 각 feature가 한 단위 증가할 때, 패배 대비 승리에 몇 배의 영향을 주는지 나타내는 지표이다. `볼점유율`의 Odds를 확인해보면 0.88이며, 이는 패배 대비 승리에 0.88배 영향을 준다는 의미이다. 즉, 승리보다는 패배에 더 큰 영향을 준다고 해석할 수 있다. 위의 결과를 통해, 승리에 가장 큰 영향을 주는 요인은 `유효슈팅`임을 알 수 있다. 유효슈팅의 Odds는 1.22로, 패배 대비 승리에 1.22배 영향을 줄 수 있음을 의미한다.<BR/><BR/><BR/><BR/>

## 결론

* 로지스틱 회귀 모델을 통해 test set의 K리그 경기 결과를 **66.97%** 예측한다.
* 승리에 긍정적 영향을 주는 요인은 `슈팅` `유효슈팅` `오프사이드` `경고` `교체` `수비수`이며, 가장 큰 영향을 주는 요인은 `유효슈팅`이므로 승리를 위해 유효슈팅을 늘리기 위해 노력해야 한다.
* 승리에 부정적 영향을 주는 요인은 `볼점유율` `코너킥` `퇴장` `공격수`이다.<BR/><BR/><BR/><BR/>

## 한계

* 활용한 데이터가 2019년과 2020년 뿐이기 때문에 일반화하여 적용하기 어렵다.
* test set의 정확도가 매우 높지는 않기 때문에 유의미한 feature들을 추가할 필요가 있다.
