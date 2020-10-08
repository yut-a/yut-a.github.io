---
layout: post
title:  코로나 상황에서 재무비율 상태에 따른 Kospi100 종목들의 주가 충격과 회복
date:   2020-10-07
image:  corona_shock.jpg
tags:   Data Finance
---
## 분석 목적

코로나 19로 인해, 올해 초 2200선이였던 코스피 지수가 1400선까지 무려 35% 이상 급락하는 것을 보았다. 코스피 지수는 코스피 시장에 상장되어 있는 종목들을 대표로 하는 지수이기 때문에, 이는 평균적으로 모든 코스피 종목들의 가격이 35%씩 하락했다는 의미로 쉽게 생각해 볼 수 있다. 이 시기에 대부분의 종목들은 다음과 같이 푹 꺼지는 모양의 그래프를 보여주었다.<BR/>
<img width="320" alt="스크린샷 2020-10-07 오후 8 44 57" src="https://user-images.githubusercontent.com/70478154/95326874-3632e880-08de-11eb-80ec-d0345854eb13.png"><BR/>
이처럼 거의 모든 종목들이 급락을 면치 못하는 상황에서도, 하락률의 차이가 있을 것이라 생각했고, 이에 대한 차이를 나타내는 요인이 기업의 `부채 비중`, `자기자본 비중` 등 재무 상태가 아닐까 하는 궁금증이 생겼다. 따라서, 코로나와 같은 경제 위기에서 기업의 재무 상태를 나타내는 주요 재무 비율의 상태에 따라 **주가 하락 충격이 어떻게 달라지는지**를 알아보고자 한다. 또한, 주요 재무 비율에 따라 주가 급락 이후 현재까지 **회복 기조가 어떻게 나타나는지** 역시 알아보고자 한다.<BR/><BR/><BR/><BR/>

## 데이터 소개

`Kopsi100 구성 종목 중 78개의 데이터`를 활용했다. Kospi100의 구성 종목은 100개이나 78개의 데이터를 사용한 이유는 두 가지이다.
* 은행, 증권, 보험 등 금융 관련 종목들을 제외했다. 산업 특성 상, `부채 비율`과 같은 재무 비율이 타 종목들에 비해 매우 높게 나타날 수 있기 때문이다. 예를 들어, 은행의 경우 고객의 예금까지도 모두 부채에 포함된다.
* 데이터가 없거나 `완전잠식` `흑자전환` `적자전환` 등 숫자로 표현된 데이터가 아닌 경우, 해당 종목들을 제외했다.

78개의 종목에서 활용한 데이터는 `주요 재무비율` `78개 종목명과 종목코드` `2020-02-03 ~ 2020-02-28 종가` `2020-03-02 ~ 2020-04-17 종가` `2020-10-06 종가`이다.

**78개 종목명과 종목코드**는 KRX에서 데이터를 얻었고, **종가**들은 열람 가능한 사이트에서 데이터를 활용했다.<BR/>
**주요 재무비율**은 크롤링하여 데이터를 사용했으며, 활용한 비율은 8개로, 다음과 같다.

* `유동비율` : 유동자산 / 유동부채로 산출하며, 기업의 지불능력을 판단한다. 높을수록 지불 능력이 커진다.
* `당좌비율` : 당좌자산 / 유동부채로 산출하며, 유동비율에 비해 현금, 예금 등 단기간에 환금할 수 있는 지불능력을 판단한다. 높을수록 지불 능력이 커진다.
* `부채비율` : 총부채 / 총자본으로 산출하며, 대출과 같은 타인자본 의존도를 나타낸다. 낮을수록 자산건전성이 높다.
* `자기자본비율` : 총자본 / 총자산으로 산출하며, 기업의 건전성을 판단한다. 높을수록 자산건전성이 높다.
* `매출액증가율` : 기준연도 대비 비교연도 매출액이 얼마나 증가했는지 보여준다.
* `영업이익증가율` : 기준연도 대비 비교연도 영업이익이 얼마나 증가했는지 보여준다.
* `EBITDA증가율` : 기준연도 대비 비교연도 EBITDA가 얼마나 증가했는지 보여준다. EBITDA는 이자비용, 세금 등을 빼기 전 순이익으로, 영업활동을 통해 벌어들이는 현금창출 능력을 보여준다.
* `ROA` : (당기순이익 / 총자산) * 100으로 산출하며, 총자산순이익률이라고 한다. 총자산에서 이익을 얼마나 올렸는지 측정하는 지표로, 자산을 얼마나 효율적으로 운용했는지를 의미하기도 한다.
* `ROE` : 일반적으로 (당기순이익 / 총자본) * 100으로 산출하여, 자기자본이익률이라고 한다. 총자본에서 이익을 얼마나 올렸는지 측정하는 지표로, 자본을 얼마나 효율적으로 운용했는지를 의미하기도 한다. 이 데이터에서는 (지배주주순이익 / 지배주주지분) * 100으로 산출했다. 간단하게 말하면, 모회사가 가진 자회사 지분만큼 당기순이익에 반영한 수치이다.

일반적으로, `유동비율` `당좌비율` `부채비율` `자기자본비율`을 **안정성 지표**, `매출액증가율` `영업이익증가율` `EBITDA증가율`을 **성장성 지표**, `ROA` `ROE`를 **수익성 지표**로 구분한다.<BR/><BR/><BR/><BR/>

## 적용 과정

**주요 재무비율과 종목명, 종목코드 데이터 전처리**

먼저, 크롤링하여 주요 재무비율 데이터를 불러오기 위해 함수를 만들었다. 코드에 대한 자세한 내용은 [이전 포스트](https://yut-a.github.io/2020/09/30/crawling-BeautifulSoup/)를 참고하기 바란다.

{% highlight ruby %}
!pip install html_table_parser

pip install --upgrade html5lib

pip install --upgrade beautifulsoup4
{% endhighlight %}

{% highlight ruby %}
# 재무 비율 데이터 추출 함수
def stock_info(stock_code = ""):
    
    from urllib.request import urlopen
    from html_table_parser import parser_functions as parser
    import pandas as pd
    import bs4
    import numpy as np
    
    # url 소스코드 긁어와 table 태그 데이터를 list 형식으로 변환
    url =  "http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=" + stock_code + "&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701"
    source = urlopen(url).read()
    source = bs4.BeautifulSoup(source, "lxml")
    table = source.find("table")
    p = parser.make2d(table)
    
    # DataFrame 변환 / 필요없는 행 제거
    df = pd.DataFrame(p, columns = p[0])
    num = df.index[df.iloc[:,0].isin(["안정성비율", "성장성비율", "수익성비율", "활동성비율"])]
    df = df.drop(num, axis = 0)
    df = df.drop(0, axis = 0)
    df = df.reset_index(drop = True)
    
    # 재무 비율 데이터만 추출
    data = df.iloc[0:1]
    
    for i in range(2, (len(df) + 1)):
        a = (3 * i) - 3
        b = a + 1
        data = pd.concat([data, df.iloc[a:b]])
        
    data = data.reset_index(drop = True)
    
    # index name 정리 및 적용
    list_a = []
    
    for j in range(0, len(data)):
        spl = data.iloc[:,0][j].split("(")
        list_a.append(spl[0])
        
    if data.columns[0] == "IFRS(연결)":
      data = data.drop(["IFRS(연결)"], axis = 1)
      data.insert(0, "IFRS(연결)", list_a)

    else:
      data = data.drop(["IFRS(별도)"], axis = 1)
      data.insert(0, "IFRS(별도)", list_a)
    
    return data
{% endhighlight %}

Kospi100 구성 종목명과 종목코드가 무엇인지 알아보기 위해 KRX에서 Kospi100 구성 종목에 대한 데이터를 받아서 불러왔다.

{% highlight ruby %}
from google.colab import files
uploaded = files.upload()

# Kospi 구성종목 데이터셋 불러오기
import pandas as pd
name = pd.read_csv("kospi100_name.csv")
name.head()
{% endhighlight %}
<img width="592" alt="스크린샷 2020-10-07 오후 11 06 01" src="https://user-images.githubusercontent.com/70478154/95341860-b6af1480-08f1-11eb-890c-2f99522130a7.png">

종목코드를 처리하기 편하도록 `A000000` 형식으로 바꿔주었다.

{% highlight ruby %}
# 종목코드를 A000000 형식으로 변환
name = name.astype({"종목코드" : str})
code = name["종목코드"].tolist()
code_change = []

for i in range(0, len(code)):
  if len(code[i]) == 6:
    data = "A" + code[i]

  elif len(code[i]) == 5:
    data = "A0" + code[i]

  elif len(code[i]) == 4:
    data = "A00" + code[i]

  elif len(code[i]) == 3:
    data = "A000" + code[i]

  code_change.append(data)

name = name.drop(["종목코드"], axis = 1)
name.insert(0, "종목코드", code_change)
name.head()
{% endhighlight %}
<img width="592" alt="스크린샷 2020-10-07 오후 11 09 21" src="https://user-images.githubusercontent.com/70478154/95342260-35a44d00-08f2-11eb-8009-e0eb25665b47.png">

필요없는 열을 모두 제거했다. 또, `SK바이오팜`을 SK바이오팜 이전에 Kospi100 구성 종목이었던 `KCC`로 대체했다. 그 이유는, SK바이오팜이 2020년 7월 2일에 상장되어 데이터가 많지 않기 때문이다.

{% highlight ruby %}
# 종목코드, 종목명 열을 제외한 모든 열 제거
name = name.drop(["현재가", "대비", "등락률", "거래대금(원)", "상장시가총액(원)"], axis = 1)

# SK바이오팜을 KCC로 대체
name[name["종목명"] == "SK바이오팜"]    # 46행
name = name.drop([46], axis = 0)
name.loc[200] = ["A002380", "KCC"]
name = name.reset_index(drop = True)
name
{% endhighlight %}
<img width="200" alt="스크린샷 2020-10-07 오후 11 15 50" src="https://user-images.githubusercontent.com/70478154/95343093-178b1c80-08f3-11eb-82f3-70d7e8b88c6a.png">

데이터 소개에서 언급했던 것처럼, 금융 관련 종목들을 제외했다. 재무 비율 추출 함수를 통해 불러온 데이터프레임의 행 개수는 23개이지만, 금융 관련 종목들은 23개보다 적다. 따라서, 이를 이용해 제외할 리스트를 선별했다. 제외한 종목은 16개이다.

{% highlight ruby %}
# 은행, 증권, 보험, 카드 종목 제외 리스트
list_for_drop = name["종목코드"].tolist()
drop_list = []

for i in list_for_drop:
  stock = stock_info(i)
  if len(stock) != 23:
    datas = i
    drop_list.append(datas)

# 은행, 증권, 보험, 카드 종목 제외
number = name.index[name["종목코드"].isin(drop_list)]
name = name.drop(number, axis = 0)
name = name.reset_index(drop = True)
name
{% endhighlight %}
<img width="204" alt="스크린샷 2020-10-07 오후 11 26 53" src="https://user-images.githubusercontent.com/70478154/95344466-a5b3d280-08f4-11eb-8360-a2b24fa1c331.png">

다음은, 위의 종목코드 순서대로 재무 비율 추출 함수를 통해 필요한 재무 비율들을 종목 별로 불러와 데이터프레임으로 정리했다.

{% highlight ruby %}
# 종목 별 재무비율 데이터 통합
codes = name["종목코드"].tolist()
num = [0, 1, 2, 6, 7, 9, 10, 16, 17]      # 재무비율 데이터프레임에서 필요한 재무비율 위치
finance = pd.DataFrame(columns = ["유동비율", "당좌비율", "부채비율", "자기자본비율", "매출액증가율", "영업이익증가율", "EBITDA증가율", "ROA", "ROE"])
list_2019 = []
count = 0

for i in codes:
  stock = stock_info(i)
  for j in num:
    num_data = stock["2019/12"][j]
    list_2019.append(num_data)

  finance.loc[count] = list_2019
  list_2019 = []
  count = count + 1

finance.head()
{% endhighlight %}
<img width="673" alt="스크린샷 2020-10-07 오후 11 33 38" src="https://user-images.githubusercontent.com/70478154/95345424-aac55180-08f5-11eb-9427-6763a05364f4.png">

종목 별 재무비율 데이터에 종목 명 칼럼을 추가하고 결측치를 제거했다. 결측치가 존재하는 6개의 종목들을 제거했다.

{% highlight ruby %}
# 종목명 칼럼 추가
names = name["종목명"].tolist()
finance.insert(0, "종목명", names)

# 결측치 제거
import numpy as np
finance = finance.replace("N/A", np.nan)
finance = finance.replace("적전", np.nan)
finance = finance.replace("적지", np.nan)
finance = finance.replace("흑전", np.nan)
finance = finance.replace("흑지", np.nan)
finance = finance.replace("완전잠식", np.nan)
finance.isnull().sum()

finance = finance.dropna(axis = 0)
finance = finance.reset_index(drop = True)
finance
{% endhighlight %}
<img width="782" alt="스크린샷 2020-10-07 오후 11 39 09" src="https://user-images.githubusercontent.com/70478154/95346032-58d0fb80-08f6-11eb-9b27-2d6f0f56ac1f.png">

**종가 데이터 전처리**

78개 종목의 `2020-02-03 ~ 2020-04-17 종가` 데이터를 불러와 정리했다.

{% highlight ruby %}
uploaded = files.upload()

# 78개 종목 2020-02-03 ~ 2020-04-17 종가 데이터 불러오기
import pandas as pd

data_1 = pd.read_csv("kospi100_1.csv", skiprows = 3, engine = "python")
data_2 = pd.read_csv("kospi100_2.csv", skiprows = 3, engine = "python")
data_3 = pd.read_csv("kospi100_3.csv", skiprows = 3, engine = "python")
data_4 = pd.read_csv("kospi100_4.csv", skiprows = 3, engine = "python")
data_5 = pd.read_csv("kospi100_5.csv", skiprows = 3, engine = "python")
data_6 = pd.read_csv("kospi100_6.csv", skiprows = 3, engine = "python")

# 필요없는 칼럼 삭제
col_1 = data_1.columns[16:]
col_2 = data_2.columns[16:]
col_3 = data_3.columns[16:]
col_4 = data_4.columns[16:]
col_5 = data_5.columns[16:]
col_6 = data_6.columns[4:]

data_1 = data_1.drop(col_1, axis = 1)
data_2 = data_2.drop(col_2, axis = 1)
data_3 = data_3.drop(col_3, axis = 1)
data_4 = data_4.drop(col_4, axis = 1)
data_5 = data_5.drop(col_5, axis = 1)
data_6 = data_6.drop(col_6, axis = 1)

# 첫 번째 칼럼 명 재설정 및 데이터 병합
data_1.rename(columns = {"Unnamed: 0" : "date"}, inplace = True)
data_2.rename(columns = {"Unnamed: 0" : "date"}, inplace = True)
data_3.rename(columns = {"Unnamed: 0" : "date"}, inplace = True)
data_4.rename(columns = {"Unnamed: 0" : "date"}, inplace = True)
data_5.rename(columns = {"Unnamed: 0" : "date"}, inplace = True)
data_6.rename(columns = {"Unnamed: 0" : "date"}, inplace = True)

kospi100 = pd.merge(data_1, data_2, on = "date", how = "inner")
kospi100 = pd.merge(kospi100, data_3, on = "date", how = "inner")
kospi100 = pd.merge(kospi100, data_4, on = "date", how = "inner")
kospi100 = pd.merge(kospi100, data_5, on = "date", how = "inner")
kospi100 = pd.merge(kospi100, data_6, on = "date", how = "inner")

kospi100.head()
{% endhighlight %}
<img width="1124" alt="스크린샷 2020-10-07 오후 11 53 08" src="https://user-images.githubusercontent.com/70478154/95348096-a3ec0e00-08f8-11eb-9393-fd155348c1ca.png">

다음은, Kospi에 상장되어 있는 종목들의 `2020-10-06 종가` 데이터를 불러와 정리했다.

{% highlight ruby %}
uploaded = files.upload()

# 2020-10-06 종가 데이터 불러오기
today = pd.read_csv("today_price.csv", skiprows = 2, engine = "python")
today = today[["종목코드", "종목명", "종가(원)"]]

# 종목코드를 A000000 형식으로 변환
today_code = today["종목코드"].tolist()
today_change = []

for i in range(0, len(today_code)):
  if len(today_code[i]) == 6:
    today_data = "A" + today_code[i]
  
  today_change.append(today_data)

today = today.drop(["종목코드"], axis = 1)
today.insert(0, "종목코드", today_change)
today.head()
{% endhighlight %}
<img width="248" alt="스크린샷 2020-10-08 오전 12 34 37" src="https://user-images.githubusercontent.com/70478154/95353316-18757b80-08fe-11eb-9826-ce175318bd2b.png">

이 중에서 필요한 종목인 78개의 종목 데이터들만 선별했다.

{% highlight ruby %}
# 사용할 종목코드의 종가만 선별
extract_list = finance["종목명"].tolist()
today_list = today[today["종목명"].isin(extract_list)]
today_list = today_list.reset_index(drop = True)
today_list.head()

# 종가 숫자형 변환
today_list["종가(원)"] = today_list["종가(원)"].str.replace(",", "")
today_list["종가(원)"] = today_list["종가(원)"].astype(np.int)
today_list.head()
{% endhighlight %}
<img width="242" alt="스크린샷 2020-10-08 오전 12 38 31" src="https://user-images.githubusercontent.com/70478154/95353787-a2254900-08fe-11eb-8cc5-e15ffb3315d5.png">

**PCA와 K-Means**

필요한 데이터들을 불러와 전처리를 완료했다. 78개 종목 별 `주요 재무비율` `2020-02-03 ~ 2020-04-17 종가` `2020-10-06 종가` 데이터로 정리했다.

먼저, 주요 재무비율의 feature가 많기 때문에 PCA를 실시하여, 저차원으로 데이터를 활용하고자 한다.

{% highlight ruby %}
# PCA를 위한 label(종목명) 제거
finance_df = finance.drop(["종목명"], axis = 1)

# 숫자형 변환
for i in range(0, len(finance_df.columns)):
  finance_df.iloc[:,i] = finance_df.iloc[:,i].str.replace(",", "")

finance_df = finance_df.astype(np.float)
finance_df.dtypes
{% endhighlight %}
<img width="210" alt="스크린샷 2020-10-08 오전 12 52 31" src="https://user-images.githubusercontent.com/70478154/95355510-98044a00-0900-11eb-9ea8-91ede6ba5e68.png">

주요 재무비율 데이터에 대한 PCA를 진행했고, 전체 데이터에 대한 누적 기여율을 파악하기 위해 PCA Scree plot을 통해 시각화했다.

{% highlight ruby %}
# PCA(전체)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()                       # Standardized
standard = scaler.fit_transform(finance_df)

pca_samp = PCA()
pca_samp.fit(standard)

vals_samp = pca_samp.explained_variance_ratio_            # 분산 비율
eigenval_samp = pca_samp.explained_variance_              # Eigenvalues
cumvals_samp = np.cumsum(vals_samp)                       # 누적 분산 비율(누적 기여율)

print(cumvals_samp)
print(eigenval_samp)
{% endhighlight %}
<img width="567" alt="스크린샷 2020-10-08 오전 1 09 47" src="https://user-images.githubusercontent.com/70478154/95357589-00542b00-0903-11eb-96ea-6e578f19bbbb.png">

{% highlight ruby %}
# Scree plot을 통한 시각화 - 6개의 PC만
import matplotlib.pyplot as plt

num = 6
count = np.arange(num) + 1

# 분산 비율, 누적 분산 비율 그리기 
plt.figure(figsize = (6, 6))
ax = plt.subplot()
plt.bar(count, vals_samp[:6], color = "#a29bfe")
plt.plot(count, cumvals_samp[:6], color = "#fdcb6e", marker = "*")

# 분산 비율 그래프에 삽입
for i in range(num):
  ax.annotate(str(np.round(vals_samp[i] * 100, 2)) + "%",
              (count[i], vals_samp[i]),
              va = "bottom",
              ha = "center",
              fontsize = 10)

# 누적 분산 비율 그래프에 삽입
for i in range(1, 6):
  ax.annotate(str(np.round(cumvals_samp[i] * 100, 2)) + "%",
              (count[i], cumvals_samp[i]),
              va = "bottom",
              ha = "center",
              fontsize = "9",
              color = "gray")

ax.set_xlabel("PCs")
ax.set_ylabel("Variance rate");
{% endhighlight %}
<img width="387" alt="스크린샷 2020-10-08 오전 1 10 42" src="https://user-images.githubusercontent.com/70478154/95357709-2f6a9c80-0903-11eb-93d2-83d0375c31a9.png">

일반적으로, Eigenvalues > 1인 PC까지 선택하여 PCA를 진행한다. 위의 결과에 따르면, PC3까지 1보다 크다는 것을 알 수 있고, 3차원으로 차원 축소하는 경우 전체 데이터의 약 81%를 설명할 수 있음을 알 수 있다. 이에 따라, 3차원으로 PCA를 다시 진행했다.

{% highlight ruby %}
# PCA(n = 3)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()                       # Standardized
standard = scaler.fit_transform(finance_df)

pca = PCA(3)
pca.fit(standard)

vals = pca.explained_variance_ratio_            # 분산 비율
eigenval = pca.explained_variance_              # Eigenvalues
cumvals = np.cumsum(vals)                       # 누적 분산 비율(누적 기여율)

print(cumvals)
print(eigenval)
{% endhighlight %}
<img width="293" alt="스크린샷 2020-10-08 오전 1 16 39" src="https://user-images.githubusercontent.com/70478154/95358401-faab1500-0903-11eb-813a-d7bc499c0d9e.png">

PCA 결과를 바탕으로, Eigenvectors를 산출했다. 이는 각 PC에 대해 feature들이 얼마나 영향을 줄 수 있는지를 수치로 표현한 것이다. 이 데이터를 바탕으로 각 PC들의 특성을 확인해보고자 한다.

{% highlight ruby %}
# PC 특성 파악
eigen_vec = pca.components_       # Eigenvectors
eigen_vec = pd.DataFrame(eigen_vec, columns = finance_df.columns)
eigen_vec = eigen_vec.T
eigen_vec.columns = ["PC1", "PC2", "PC3"]
eigen_vec
{% endhighlight %}
<img width="347" alt="스크린샷 2020-10-08 오전 1 20 56" src="https://user-images.githubusercontent.com/70478154/95359232-e287c580-0904-11eb-9eaf-06088b4911c8.png">

위의 결과에 따르면, PC1에 가장 많은 영향을 주는 것은 `유동비율` `당좌비율` `부채비율` `자기자본비율`임을 알 수 있다. 즉, PC1은 **안정성 지표**와 관련이 크다고 할 수 있다. `유동비율` `당좌비율` `자기자본비율`이 PC1과 양의 상관관계를, `부채비율`이 음의 상관관계를 나타내는 것으로 보아, **PC1이 클수록 안정성 지표가 좋다**고 할 수 있다.

PC2에 가장 많은 영향을 주는 것은 `매출액증가율` `영업이익증가율` `EBITDA증가율`임을 알 수 있다. 즉, PC2는 **성장성 지표**와 관련이 크다고 할 수 있다. `매출액증가율` `영업이익증가율` `EBITDA증가율`이 PC2와 음의 상관관계를 나타내는 것으로 보아, **PC2가 작을수록 성장성 지표가 좋다**고 할 수 있다.

PC3에 가장 많은 영향을 주는 것은 `ROA` `ROE`임을 알 수 있다. 즉, PC3는 **수익성 지표**와 관련이 크다고 할 수 있다. `ROA` `ROE`가 PC3와 음의 상관관계를 나타내는 것으로 보아, **PC3이 작을수록 수익성 지표가 좋다**고 할 수 있다.

다음은, 각 PC에 대한 종목들의 Projected data를 산출했다. 즉, 각 종목에 대한 PC1, PC2, PC3로 이루어진 점이 원본 주요 재무비율을 설명할 수 있는 것이다.

{% highlight ruby %}
# Projected data
projec = pca.transform(standard)
projec = pd.DataFrame(projec, columns = ["PC1", "PC2", "PC3"])
finance_index = finance["종목명"].tolist()
projec.insert(0, "종목명", finance_index)
projec
{% endhighlight %}
<img width="383" alt="스크린샷 2020-10-08 오전 1 41 09" src="https://user-images.githubusercontent.com/70478154/95361195-62af2a80-0907-11eb-8412-8d699d2a5846.png">

PCA 결과를 바탕으로, 각 종목들을 안정성, 성장성, 수익성 등 특성에 따라 묶기 위해 K-Means clustering을 진행했다.

{% highlight ruby %}
# parameter 선택
sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(projec_data)
    sum_of_squared_distances.append(km.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
{% endhighlight %}
<img width="397" alt="스크린샷 2020-10-08 오전 1 58 56" src="https://user-images.githubusercontent.com/70478154/95363020-de11db80-0909-11eb-8101-2024dd5841be.png">

그래프에 따르면, k=3부터 하락폭이 완만하게 줄어들기 때문에 클러스터 수를 3으로 설정했다. 물론 보는 사람에 따라 다를 수 있기 때문에 스스로 알맞게 판단하는 것이 좋다. 

{% highlight ruby %}
# K-Means를 위한 label 제거
projec_data = projec.drop(["종목명"], axis = 1)

# K-Means
from sklearn.cluster import KMeans
kmeans_pca = KMeans(n_clusters = 3, random_state = 0)
kmeans_pca.fit(projec_data)
labels_pca = kmeans_pca.labels_

pca_data = pd.Series(labels_pca)
projec_data_cluster = projec_data.copy()
projec_data_cluster["clusters"] = pca_data.values

projec_data_cluster.head()
{% endhighlight %}
<img width="352" alt="스크린샷 2020-10-08 오전 2 01 49" src="https://user-images.githubusercontent.com/70478154/95363314-43fe6300-090a-11eb-987c-a5390fa6e2f2.png">

clustering 결과를 바탕으로, PC1, PC2, PC3를 두 개씩 묶어 scatter plot으로 시각화했다. clustering한 PC1과 PC2 그래프를 보면, 0, 1, 2로 잘 분류된 것을 확인할 수 있다. 해석하면, `0`은 **안정성과 성장성이 모두 낮은 종목**, `1`은 **안정성이 높은 종목**, `2`는 **성장성이 높은 종목**이라고 할 수 있다.

반면, clustering한 PC3과 PC1, PC3과 PC2 그래프를 보면, cluster에 상관 없이 모여있는 것을 확인할 수 있다.

{% highlight ruby %}
# cluster 시각화 (PC1 / PC2)
import seaborn as sns
sns.scatterplot(x = "PC1", y = "PC2", hue = "clusters", data = projec_data_cluster);
{% endhighlight %}
<img width="386" alt="스크린샷 2020-10-08 오전 2 21 01" src="https://user-images.githubusercontent.com/70478154/95365350-fb947480-090c-11eb-948b-47bddb4a9f53.png">

{% highlight ruby %}
# cluster 시각화 (PC3 / PC1)
import matplotlib.pyplot as plt
plt.subplot(211)
sns.scatterplot(x = "PC3", y = "PC1", hue = "clusters", data = projec_data_cluster)

# cluster 시각화 (PC3 / PC2)
plt.subplot(212)
sns.scatterplot(x = "PC3", y = "PC2", hue = "clusters", data = projec_data_cluster);
{% endhighlight %}
<img width="398" alt="스크린샷 2020-10-08 오전 2 22 05" src="https://user-images.githubusercontent.com/70478154/95365445-19fa7000-090d-11eb-9a60-715dc38f495f.png">

따라서, PC1과 PC2의 cluster를 중심으로 코로나 19에 따른 충격과 회복에 대한 분석을 실시하고자 한다.

**코로나 19로 인한 주가 급락과 회복 분석 및 Chi-squared test**

이제 코로나로 인한 타격으로 주가 급락이 어떻게 나타나는지를 재무 비율 특성에 따라 구분해보고자 한다.

먼저, 코로나 19로 인한 타격 이전에 비해 타격 후 주가 급락이 몇 퍼센트로 나타나는지 확인하고자 한다. 코로나 19가 시작된 것은 2월보다 이전이지만, 본격적으로 경제에 충격을 주기 시작한 것은 3월부터이다. 따라서, 코로나 19로 인한 경제 타격 직전인 `2020-02-03  ~ 2020-02-28 종가`와 타격 이후 급락하는 시점인 `2020-03-02 ~ 2020-04-17 종가`로 나눴다. 또, 코로나로 인한 타격의 하락률을 계산하기 위해 종목 별 `2020-02-03  ~ 2020-02-28 종가`의 max 가격과 `2020-03-02 ~ 2020-04-17 종가`의 min 가격을 구하고 새로운 데이터프레임으로 정리했다.

{% highlight ruby %}
# 2020-02-03 ~ 2020-02-28 종가와 2020-03-02 ~ 2020-04-17 종가로 분할

import numpy as np

# 2020-02-03 ~ 2020-02-28 종가 / 숫자형으로 변형
kospi100_max = kospi100[0:20]
kospi100_max = kospi100_max.reset_index(drop = True)

for i in range(0, len(kospi100_max.columns)):
  kospi100_max.iloc[:,i] = kospi100_max.iloc[:,i].str.replace(",", "")

kospi100_max = kospi100_max.iloc[:,1:].astype(np.int)

# 2020-03-02 ~ 2020-04-17 종가 / 숫자형으로 변형
kospi100_min = kospi100[20:54]
kospi100_min = kospi100_min.reset_index(drop = True)

for i in range(0, len(kospi100_min.columns)):
  kospi100_min.iloc[:,i] = kospi100_min.iloc[:,i].str.replace(",", "")

kospi100_min = kospi100_min.iloc[:,1:].astype(np.int)
{% endhighlight %}

{% highlight ruby %}
# 종목 별 max price
max_data = []

for i in range(0, len(kospi100_max.columns)):
  max_value = max(kospi100_max.iloc[:,i])
  max_data.append(max_value)

# 종목 별 min price
min_data = []

for i in range(0, len(kospi100_min.columns)):
  min_value = min(kospi100_min.iloc[:,i])
  min_data.append(min_value)

# 종목 별 max / min data 병합
max_min = pd.DataFrame({"종목명" : kospi100_max.columns,
                        "max_price" : max_data,
                        "min_price" : min_data})
max_min
{% endhighlight %}
<img width="311" alt="스크린샷 2020-10-08 오전 12 24 01" src="https://user-images.githubusercontent.com/70478154/95351933-abadb180-08fc-11eb-8d46-fa0a858ba4be.png">

max 가격과 min 가격을 바탕으로, 다음과 같이 하락률 산출하고 categorical data로 등급화했다.

{% highlight ruby %}
# 하락률 산출
down = []

for i in range(0, 78):
  down_value = (max_min["min_price"][i] - max_min["max_price"][i]) / max_min["max_price"][i]
  down_value = abs(round(down_value * 100, 2))
  down.append(down_value)

max_min["down(-%)"] = down

# 하락률 categorical data로 등급화
max_min["down(-%)_grade"] = pd.qcut(max_min["down(-%)"], 3, labels = ["low", "middle", "high"])
max_min
{% endhighlight %}
<img width="529" alt="스크린샷 2020-10-08 오전 2 34 03" src="https://user-images.githubusercontent.com/70478154/95366731-ca1ca880-090e-11eb-8c6a-0c992136bcce.png">

재무 비율의 특성에 따라 하락률 등급이 어떤 분포를 가지는지 파악하기 위해 crosstab을 통해 확인해 보았다. 그 결과, `안정성과 성장성 모두 나쁜 지표`를 가지는 종목일수록 하락률 등급이 high, middle, low 순으로 높았고, high 등급이 약 50%를 차지했다. 즉, **모두 나쁜 지표를 가지는 경우일수록 높은 하락률을 보일 가능성이 큼**을 알 수 있다. 또, `안정성이 높은 지표`를 가지는 종목일수록 하락률 등급이 low, middle, high 순으로 높았고, low 등급이 약 55%를 차지했다. 즉, **안정성이 높은 지표를 가지는 경우일수록 낮은 하락률을 보일 가능성이 큼**을 알 수 있다. 반면, `성장성이 높은 지표`의 경우 모든 하락률 등급에 고르게 분포되어 있어 특별한 관계를 파악할 수 없다.

{% highlight ruby %}
# crosstab_down(-%) => 0 - 둘 다 나쁨, 1 - 안정성, 2 - 성장성
crosstab = pd.crosstab(max_min["down(-%)_grade"], projec_data_cluster["clusters"])
crosstab
{% endhighlight %}
<img width="208" alt="스크린샷 2020-10-08 오전 2 50 34" src="https://user-images.githubusercontent.com/70478154/95368422-136df780-0911-11eb-8c36-700de7034323.png">

살펴본 두 변수 간의 관계가 통계적으로 유의미한지 알아보고자 한다.

귀무가설 : `down(-%)\_grade`와 `clusters` 변수는 서로 독립이다.<BR/>
대립가설 : `down(-%)\_grade`와 `clusters` 변수는 서로 연관이 있다.

`Chi-squared test` 결과, P-value는 약 0.0487로, 0.05보다 작다. 유의수준 5%에서 귀무가설을 기각한다. 즉, **두 변수는 서로 연관이 있음**을 알 수 있다.

{% highlight ruby %}
# chi-squared test_down(-%)
from scipy.stats import chi2_contingency
chi2_contingency(crosstab, correction = False)
{% endhighlight %}
<img width="399" alt="스크린샷 2020-10-08 오전 2 57 23" src="https://user-images.githubusercontent.com/70478154/95369156-0a315a80-0912-11eb-9c13-cb69a51f8652.png">

이번에는 코로나로 인한 급락 이후 현재 얼마나 회복했는지를 재무 비율 특성에 따라 구분해보고자 한다.

먼저, `2020-10-06 종가` 데이터를 max_min 데이터프레임에 병합하고, 몇 퍼센트 회복했는지를 나타냈다. 또, 회복률을 categorical data로 등급화했다.

{% highlight ruby %}
# 2020-10-06 종가 데이터 병합
max_min_today = pd.merge(max_min, today_list, on = "종목명", how = "inner")
max_min_today = max_min_today.drop("종목코드", axis = 1)

# 회복(%) 산출
recover = []

for i in range(0, 78):
  recover_value = (max_min_today["종가(원)"][i] / max_min_today["max_price"][i]) * 100
  recover_value = round(recover_value, 2)
  recover.append(recover_value)

max_min_today["recover(%)"] = recover

# 회복률 categorical data로 등급화
max_min_today["recover(%)_grade"] = pd.qcut(max_min_today["recover(%)"], 3, labels = ["low", "middle", "high"])
max_min_today
{% endhighlight %}
<img width="849" alt="스크린샷 2020-10-08 오전 3 14 55" src="https://user-images.githubusercontent.com/70478154/95370939-7ad97680-0914-11eb-96a4-a38364e1d46e.png">

재무 비율의 특성에 따라 회복률 등급이 어떤 분포를 가지는지 파악하기 위해 crosstab을 실시한 결과, 안정성과 성장성 모두 나쁜 지표를 가진 clusters=0의 경우 가장 많은 low 등급에 분포되어 있기는 하지만, 전체적으로 재무 비율 특성에 따른 큰 관계가 없음을 알 수 있다.

{% highlight ruby %}
# crosstab_recover(%) => 0 - 둘 다 나쁨, 1 - 안정성, 2 - 성장성
crosstab_recover = pd.crosstab(max_min_today["recover(%)_grade"], projec_data_cluster["clusters"])
crosstab_recover
{% endhighlight %}
<img width="235" alt="스크린샷 2020-10-08 오전 3 21 39" src="https://user-images.githubusercontent.com/70478154/95371623-6d70bc00-0915-11eb-83ca-27ccc13e709c.png">

마찬가지로, 두 변수 간의 관계가 통계적으로 유의미한지 알아보고자 한다.

귀무가설 : `recover(%)\_grade`와 `clusters` 변수는 서로 독립이다.<BR/>
대립가설 : `recover(%)\_grade`와 `clusters` 변수는 서로 연관이 있다.

`Chi-squared test` 결과, P-value는 약 0.58로, 0.05보다 매우 높다. 즉, 귀무가설을 기각할 수 없으며, **두 변수는 서로 연관이 없다**고 할 수 있다.

{% highlight ruby %}
# chi-squared test_recover
from scipy.stats import chi2_contingency
chi2_contingency(crosstab_recover, correction = False)
{% endhighlight %}
<img width="403" alt="스크린샷 2020-10-08 오전 3 24 38" src="https://user-images.githubusercontent.com/70478154/95371942-d8ba8e00-0915-11eb-987b-e27498229452.png"><BR/><BR/><BR/><BR/>

## 결론

