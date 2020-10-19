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

학교 수업 시간에 다중 회귀 분석으로 축구 국가대표팀 승, 패 요인 분석이라는 주제를 가지고 과제를 수행한 적이 있다. 그 당시에는 데이터도 충분하지 않았고, 로지스틱 회귀에 대한 지식이 없었기 때문에 그 때의 분석 경험을 보충해보고자 한다. 승, 패에 영향을 미칠 수 있는 주요 데이터를 바탕으로, 어떤 요인이 경기 결과에 어떻게 영향을 줄 수 있는 지를 파악하고자 한다. 또한, 이를 바탕으로 경기 결과를 예측해 보고자 한다.<BR/><BR/><BR/><BR/>

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
    
    driver = webdriver.Chrome("/Users/yut_a_/Downloads/chromedriver")
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

데이터 전처리를 완료한 후, 로지스틱 회귀 분석을 위해 train과 test set으로 나누고, feature와 target으로 데이터를 구분했다. validation set을 사용하지 않은 이유는 데이터의 양이 많지 않기 때문이며, 이를 위해 `LogisticRegressionCV`를 활용하고자 한다.

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

train과 test set으로 잘 구분이 된 것을 확인할 수 있다. 이를 바탕으로 Logistic Regression을 학습하고 score를 확인해 보고자 한다.

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



정확ㄷㄹ


