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
이처럼 거의 모든 종목들이 급락을 면치 못하는 상황에서도, 하락률의 차이가 있을 것이라 생각했고, 이에 대한 차이를 나타내는 요인이 기업의 `부채 비중`, `자기자본 비중` 등 재무 상태가 아닐까 하는 궁금증이 생겼다. 따라서, 코로나와 같은 경제 위기에서 기업의 재무 상태를 나타내는 주요 재무 비율에 따라 **주가 하락 충격을 얼마나 감수할 수 있는지**를 알아보고자 한다. 또한, 주요 재무 비율에 따라 주가 급락 이후 현재까지 **회복 기조가 어떻게 나타나는지** 역시 알아보고자 한다.<BR/><BR/><BR/><BR/>

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
















