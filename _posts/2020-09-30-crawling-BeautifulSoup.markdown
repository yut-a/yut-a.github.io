---
layout: post
title:  주식 종목 재무비율 Crawling with BeautifulSoup
date:   2020-09-30
image:  information.jpg
tags:   Data
---
## 크롤링(Crawling)이란?

웹사이트에 존재하는 각종 정보를 자동적으로 수집해오는 것을 말한다. 다양한 분석 도구를 적용하기 위해 필요한 데이터들이 항상 준비되어 있지 않다. 데이터 수집을 위해 다양한 매체에서 데이터를 수집해와야 하는데, 크롤링은 그 중 유용한 수집 도구 중 하나라고 할 수 있다.<BR/><BR/><BR/><BR/>

## 도구 소개

데이터 수집을 위한 다양한 크롤링 방법이 존재하겠지만, 여기에서는 `BeautifulSoup` 라이브러리를 활용하고자 한다. 특히, 데이터프레임으로 데이터를 수집하기 위해서 `html_table_parser`도 함께 사용한다.

필요한 라이브러리는 설치 후 사용할 수 있다. 하지만, 설치 후 라이브러리를 import 할 경우 다음과 같은 오류가 나타날 수 있다.<BR/>
`module 'html5lib.treebuilders' has no attribute '_base'`

이를 해결하기 위해 다음과 같은 과정이 필요하다.

{% highlight ruby %}
!pip install html_table_parser
{% endhighlight %}

{% highlight ruby %}
pip install --upgrade html5lib
{% endhighlight %}

{% highlight ruby %}
pip install --upgrade beautifulsoup4
{% endhighlight %}

이러한 과정을 거친다면, 무리없이 필요한 라이브러리들을 불러와 사용할 수 있을 것이다.<BR/><BR/><BR/><BR/>

## 적용 과정

크롤링을 통해, 다음의 재무 비율 데이터를 수집하고자 한다.

<img width="994" alt="스크린샷 2020-10-01 오전 1 30 01" src="https://user-images.githubusercontent.com/70478154/94713429-bbba1400-0385-11eb-906d-7c974c7ee19a.png">

크롤링을 하기 위해서 `url`을 확인했다. 확인한 결과, 사이트 주소 중간의 종목 코드, 예를 들어 삼성전자는 `A005930`만 종목에 따라 바뀌는 것을 확인할 수 있다.

<img width="1440" alt="스크린샷 2020-10-01 오전 1 10 06" src="https://user-images.githubusercontent.com/70478154/94711810-94fade00-0383-11eb-9718-dd5e6aa06125.png">

따라서, 다음과 같이 url을 설정해주었다.

{% highlight ruby %}
# 삼성전자 코드
stock_code = "A005930"
url = "http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gicode=" + stock_code + "&cID=&MenuYn=Y&ReportGB=&NewMenuID=104&stkGb=701"
{% endhighlight %}

필요한 라이브러리들을 불러온 후, `urlopen`을 이용하여 url의 소스코드를 긁어온다. 긁어온 소스코드를 보기 좋게 해석하기 위해, `BeautifulSoup`를 이용한다.

{% highlight ruby %}
from urllib.request import urlopen
from html_table_parser import parser_functions as parser
import pandas as pd
import bs4
import numpy as np

source = urlopen(url).read()                  # url의 소스코드 긁어오기
source = bs4.BeautifulSoup(source, "lxml")    # 소스코드 해석
{% endhighlight %}

웹사이트에서 크롤링하기를 원하는 부분의 구조를 파악하기 위해 `검사`를 클릭한다.
<img width="1438" alt="스크린샷 2020-10-01 오후 12 10 12" src="https://user-images.githubusercontent.com/70478154/94761473-184a1d00-03e0-11eb-8680-831b29d6c1aa.png">

확인한 결과, 다음과 같은 class의 table임을 알 수 있다. 해당 부분에 마우스 오른쪽 버튼으로 `copy Xpath`를 하면 더 구체적으로 위치를 파악할 수 있다. 페이지 소스를 통해 2개의 table 중 첫 번째 table임을 확인했다. 해당 table을 찾기 위해 사용할 수 있는 것은 `find`와 `find_all`이 있는데, 전자는 첫 번째 태그만을 가지고 오는 것이고, 후자는 해당되는 모든 태그를 가지고 오는 것이다. 여기서는 처음의 table만 필요하기 때문에 `find`를 사용했다.
<img width="1440" alt="스크린샷 2020-10-01 오후 12 18 47" src="https://user-images.githubusercontent.com/70478154/94763673-95c45c00-03e5-11eb-9d2f-c6e0a22ed90f.png">

{% highlight ruby %}
table = source.find("table")            # 첫 번째 table 태그 찾기
p = parser.make2d(table)                # table 태그의 데이터들을 list형식으로 정리
df = pd.DataFrame(p, columns = p[0])    # DataFrame으로 변환
df
{% endhighlight %}
<img width="778" alt="스크린샷 2020-10-01 오후 1 08 46" src="https://user-images.githubusercontent.com/70478154/94767445-50a12980-03e7-11eb-9e47-898a379f7499.png">

데이터들을 데이터프레임 형식으로 잘 불러왔다. 이제 index를 정리하고 필요한 데이터들만 따로 추출하고자 한다. 삼성전자가 아닌 다른 종목을 검색해보면, IFRS(연결) 혹은 IFRS(별도), 이렇게 두 종류의 데이터프레임이 존재하기 때문에 불러온 데이터프레임의 첫 행은 `isin`을 사용하지 않고, 따로 제거한다.

{% highlight ruby %}
# 필요없는 행 제거
num = df.index[df.iloc[:,0].isin(["안정성비율", "성장성비율", "수익성비율", "활동성비율"])]
df = df.drop(num, axis = 0)
df = df.drop(0, axis = 0)
df = df.reset_index(drop = True)
df
{% endhighlight %}
<img width="773" alt="스크린샷 2020-10-01 오후 1 14 05" src="https://user-images.githubusercontent.com/70478154/94767667-0ec4b300-03e8-11eb-82c1-a8a20e34698c.png">

불러온 데이터는 재무 비율뿐 아니라, 유동자산, 유동부채 등 재무 비율을 계산하기 위한 데이터들까지 함께 불러왔다. 따라서, 재무 비율 데이터들만 따로 뽑아 데이터프레임으로 구성하고자 한다.

{% highlight ruby %}
# 재무 비율 데이터만 추출
data = df.iloc[0:1]

for i in range(2, (int(len(df) / 3) + 1)):
    a = 3*i - 3
    b = a + 1
    data = pd.concat([data, df.iloc[a:b]])
    
data = data.reset_index(drop = True)
data.head()
{% endhighlight %}
<img width="719" alt="스크린샷 2020-10-01 오후 1 19 48" src="https://user-images.githubusercontent.com/70478154/94767916-db365880-03e8-11eb-8261-d2de51fcf62e.png">

다음은, index 이름을 깔끔하게 정리했다. 은행, 증권, 카드 등 금융 관련 종목들은 그 외의 종목들과는 다르게 11개의 재무비율 데이터가 포함되어 있다. 따라서, 데이터의 크기에 따라 index 이름을 다르게 적용했다.

{% highlight ruby %}
# index name 정리
list_finance_1 = ['유동비율', '당좌비율', '부채비율', '유보율', '순차입금비율', '이자보상배율', '자기자본비율', '매출액증가율',
                  '판매비와관리비증가율', '영업이익증가율', 'EBITDA증가율', 'EPS증가율', '매출총이익율', '세전계속사업이익률',
                  '영업이익률', 'EBITDA마진율', 'ROA', 'ROE', 'ROIC', '총자산회전율', '총부채회전율', '총자본회전율',
                  '순운전자본회전율']
list_finance_2 = ['예대율', '유가증권보유율', '부채비율', '유보율', '순영업손익증가율', '영업이익증가율', '지배주주순이익증가율',
                  'EPS증가율', '영업이익률', 'ROA', 'ROE']

list_a = []
    
if len(data) == 23:
    for i in range(0, len(data)):
        if (list_finance_1[i] in data.iloc[:,0][i]) == True:
            list_a.append(list_finance_1[i])
                
else:
    for j in range(0, len(data)):
        if (list_finance_2[j] in data.iloc[:,0][j]) == True:
            list_a.append(list_finance_2[j])
    
# 정리한 index name 적용
if data.columns[0] == "IFRS(연결)":
    data = data.drop(["IFRS(연결)"], axis = 1)
    data.insert(0, "IFRS(연결)", list_a)
        
elif data.columns[0] == "GAAP(개별)":
    data = data.drop(["GAAP(개별)"], axis = 1)
    data.insert(0, "GAAP(개별)", list_a)
        
elif data.columns[0] == "GAAP(연결)":
    data = data.drop(["GAAP(연결)"], axis = 1)
    data.insert(0, "GAAP(연결)", list_a)

else:
    data = data.drop(["IFRS(별도)"], axis = 1)
    data.insert(0, "IFRS(별도)", list_a)

data.head()
{% endhighlight %}
<img width="391" alt="스크린샷 2020-10-01 오후 1 25 45" src="https://user-images.githubusercontent.com/70478154/94768194-aecf0c00-03e9-11eb-8e53-6b2baaf9876e.png">

이렇게 `삼성전자`의 재무 비율 데이터를 크롤링하여 데이터프레임으로 보기 좋게 정리했다. 이를 바탕으로, 종목 코드만 입력하면 해당 종목의 재무 비율 데이터를 불러오도록 함수로 정리했다.

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
    
    for i in range(2, (int(len(df) / 3) + 1)):
        a = (3 * i) - 3
        b = a + 1
        data = pd.concat([data, df.iloc[a:b]])
        
    data = data.reset_index(drop = True)
    
    # index name 정리 및 적용
    list_finance_1 = ['유동비율', '당좌비율', '부채비율', '유보율', '순차입금비율', '이자보상배율', '자기자본비율', '매출액증가율',
                  '판매비와관리비증가율', '영업이익증가율', 'EBITDA증가율', 'EPS증가율', '매출총이익율', '세전계속사업이익률',
                  '영업이익률', 'EBITDA마진율', 'ROA', 'ROE', 'ROIC', '총자산회전율', '총부채회전율', '총자본회전율',
                  '순운전자본회전율']
    list_finance_2 = ['예대율', '유가증권보유율', '부채비율', '유보율', '순영업손익증가율', '영업이익증가율', '지배주주순이익증가율',
                  'EPS증가율', '영업이익률', 'ROA', 'ROE']

    list_a = []
    
    if len(data) == 23:
        for i in range(0, len(data)):
            if (list_finance_1[i] in data.iloc[:,0][i]) == True:
                list_a.append(list_finance_1[i])
                
    else:
        for j in range(0, len(data)):
            if (list_finance_2[j] in data.iloc[:,0][j]) == True:
                list_a.append(list_finance_2[j])

    if data.columns[0] == "IFRS(연결)":
        data = data.drop(["IFRS(연결)"], axis = 1)
        data.insert(0, "IFRS(연결)", list_a)
        
    elif data.columns[0] == "GAAP(개별)":
        data = data.drop(["GAAP(개별)"], axis = 1)
        data.insert(0, "GAAP(개별)", list_a)
        
    elif data.columns[0] == "GAAP(연결)":
        data = data.drop(["GAAP(연결)"], axis = 1)
        data.insert(0, "GAAP(연결)", list_a)

    else:
        data = data.drop(["IFRS(별도)"], axis = 1)
        data.insert(0, "IFRS(별도)", list_a)
    
    return data
{% endhighlight %}

`삼성전자`의 종목 코드를 통해 재무 비율을 잘 불러오는 것을 확인할 수 있다.

{% highlight ruby %}
# 삼성전자 재무 비율
stock_info("A005930").head()
{% endhighlight %}
<img width="396" alt="스크린샷 2020-10-01 오후 1 38 34" src="https://user-images.githubusercontent.com/70478154/94768841-80523080-03eb-11eb-8998-962558447883.png">

이번엔 `삼성증권`의 종목 코드를 입력해보자. 다른 종목의 재무 비율 역시 잘 불러오는 것을 확인할 수 있다.

{% highlight ruby %}
# 삼성증권 재무 비율
stock_info("A016360")
{% endhighlight %}
<img width="426" alt="스크린샷 2020-10-01 오후 1 38 50" src="https://user-images.githubusercontent.com/70478154/94768864-906a1000-03eb-11eb-9501-9a5cdd0fb6d7.png"><BR/><BR/><BR/><BR/>

## 정리

`BeautifulSoup`를 활용하여 웹 사이트의 원하는 데이터를 가져와 보았다. 이 라이브러리 외에도 활용 가능한 라이브러리가 많기 때문에 적용하기 편리한 방식을 사용하는 것이 좋다. 사실 이 코드는 전혀 만능이 아니다. 다른 웹 사이트에는 적용할 수 없는 일회용 코드일 뿐이기 때문이다. 하지만, 필요한 데이터가 `csv`나 `xlsx` 등으로 정리되어 있지 않고, 지속적으로 데이터를 추출할 필요가 있는 경우에 코드를 작성한다면 매우 유용할 것이라 생각한다.
