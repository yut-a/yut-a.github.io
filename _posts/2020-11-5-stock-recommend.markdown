---
layout: post
title:  주식 종목 추천 모델
date:   2020-11-05
image:  choice.jpg
tags:   Data Finance
---
## 분석 목적

투자를 위한 분석에는 다양한 종류가 있다. 가치투자를 위해 저평가된 주식 종목들을 발굴하는 분석도 있고, 차트의 흐름을 보는 기술적 분석도 있다. 애널리스트처럼 기업 탐방을 통해 비정형화된 정보를, 재무, 회계 지표와 같은 정형화된 정보와 종합하여 투자 판단을 내리기도 한다.

그 외에 `퀀트 투자`라는 것이 있다. 주관적 판단에 의해 비합리적인 선택을 할 수 있기 때문에 감정을 배제한 객관적 투자를 지향한다. 이 때문에 **Data(Number) Driven Investment** 라고도 한다. 가장 보편적으로 알려진 전략이 `저PER` `저PBR`전략이다. PER(주가수익비율)과 PBR(주가순자산비율)은 각각 `주가/주당순이익` `주가/주당순자산`으로 산출되는 지표이다. 즉, 저PER, 저PBR이라는 것은 주당순이익, 주당순자산 대비 현재 주가가 낮다는 것으로, 현재 해당 종목이 주식 시장에서 저평가되어 있고, 앞으로 오를 가능성이 높다는 것을 의미하기도 한다. 이러한 퀀트 전략을, Historical Data를 기반으로 백테스팅을 통해 유의미한 성과가 있음을 증명한다.

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
<BR/>
<details>
<summary>데이터 수집</summary>
<div markdown="1">
<BR/><BR/>
#### 투자지표

먼저, PER, PBR과 같은 종목 별 투자 지표를 추출할 수 있는 함수를 만들었다.

{% highlight ruby %}
# 투자 지표 데이터 추출 함수
def invest_info(stock_code = ""):
    from urllib.request import urlopen
    from html_table_parser import parser_functions as parser
    import pandas as pd
    import bs4
    import numpy as np
    
    # url 소스코드 긁어와 table 태그 데이터를 list 형식으로 변환
    url = "http://comp.fnguide.com/SVO2/ASP/SVD_Invest.asp?pGB=1&gicode=" + stock_code + "&cID=&MenuYn=Y&ReportGB=&NewMenuID=103&stkGb=701"
    source = urlopen(url).read()
    source = bs4.BeautifulSoup(source, "lxml")
    table = source.find_all("table")[1]
    p = parser.make2d(table)
    
    # DataFrame 변환 / 필요없는 행 제거
    df = pd.DataFrame(p, columns = p[0])
    
    if len(df) == 52:
        num = df.index[df.iloc[:,0].isin(["Per\xa0Share", "Dividends", "Multiples", "FCF"])]
        df = df.drop(num, axis = 0)
        df = df.drop(0, axis = 0)
        df = df.reset_index(drop = True)
    
    else:
        num = df.index[df.iloc[:,0].isin(["Per\xa0Share", "Dividends", "Multiples"])]
        df = df.drop(num, axis = 0)
        df = df.drop(0, axis = 0)
        df = df.reset_index(drop = True)
    
    # 투자 지표 비율 데이터만 추출
    data = df.iloc[0:1]

    if (len(df) % 3) == 0:
        for i in range(2, int(len(df) / 3) + 1):
            a = 3*i - 3
            b = a + 1
            data = pd.concat([data, df.iloc[a:b]])
        
    else:
        for i in range(2, (int((len(df) - 2) / 3) + 1)):
            a = 3*i - 3
            b = a + 1
            data = pd.concat([data, df.iloc[a:b]])
        data = pd.concat([data, df.iloc[(len(df) - 2):len(df)]])
    
    data = data.reset_index(drop = True)
    
    # index name 정리 및 적용
    list_index_1 = ["EPS", "EBITDAPS", "CFPS", "SPS", "BPS", "DPS(보통주,현금)", "DPS(1우선주,현금)", "배당성향",
                    "PER", "PCR", "PSR", "PBR", "EV/Sales", "EV/EBITDA", "총현금흐름", "총투자", "FCFF"]
    list_index_2 = ["EPS", "BPS", "DPS(보통주,현금)", "DPS(1우선주,현금)", "배당성향(현금,%)", "PER", "PBR"]
    list_spl = []

    if len(data) == 17:                 
        for i in range(0, len(data)):
            if (list_index_1[i] in data.iloc[:,0][i]) == True:
                list_spl.append(list_index_1[i])

    else:
        for j in range(0, len(data)):
            if (list_index_2[j] in data.iloc[:,0][j]) == True:
                list_spl.append(list_index_2[j])
                
    if data.columns[0] == "IFRS 연결":
        data = data.drop(["IFRS 연결"], axis = 1)
        data.insert(0, "IFRS(연결)", list_spl)
    
    elif data.columns[0] == "GAAP 개별":
        data = data.drop(["GAAP 개별"], axis = 1)
        data.insert(0, "GAAP(개별)", list_spl)
        
    elif data.columns[0] == "GAAP 연결":
        data = data.drop(["GAAP 연결"], axis = 1)
        data.insert(0, "GAAP(연결)", list_spl)

    else:
        data = data.drop(["IFRS 개별"], axis = 1)
        data.insert(0, "IFRS(별도)", list_spl)
        
    return data
{% endhighlight %}

다음은, 사용할 코스피, 코스닥 종목들을 필터링하기 위해 각 시장에 상장되어 있는 종목들의 코드와 종목명 데이터를 불러왔다. 코스피 종목들부터 진행했다.

{% highlight ruby %}
# 코스피 종목 코드 데이터
import pandas as pd

kospi_stock = pd.read_csv("kospi_stock.csv")
kospi_stock.rename(columns = {"null" : "code", "null.1" : "stock"}, inplace = True)
kospi_stock = kospi_stock.drop(kospi_stock.columns[2:], axis = 1)
kospi_stock
{% endhighlight %}
<img width="171" alt="스크린샷 2020-11-08 오후 5 13 27" src="https://user-images.githubusercontent.com/70478154/98460233-c6b65e80-21e5-11eb-8e80-e628d4a44fc9.png">

코스피 종목들 중 펀드 형태의 종목, 관리종목, 거래정지종목들을 제거했다. 또한, 코드를 `A000000` 형태로 변환했다.

{% highlight ruby %}
# 펀드 종목 제외 (메리츠는 금융 관련 주라 미리 제외)
cond_fund = kospi_stock["stock"].str.contains("리츠") | kospi_stock["stock"].str.contains("하이골드") | kospi_stock["stock"].str.contains("바다로")
kospi_stock = kospi_stock.drop(kospi_stock[cond_fund].index, axis = 0)

# 코스피 관리종목 리스트
kospi_admin = pd.read_csv("kospi_admin.csv")
kospi_admin = kospi_admin["종목명"].tolist()
kospi_admin = pd.DataFrame(kospi_admin)

# 코스피 거래정지종목 리스트
kospi_stop = pd.read_csv("kospi_stop.csv")
kospi_stop = kospi_stop["종목명"].tolist()
kospi_stop = pd.DataFrame(kospi_stop)

# 관리종목 / 거래정지종목 중복값 처리
kospi_admin_stop = pd.concat([kospi_admin, kospi_stop])
kospi_admin_stop = kospi_admin_stop.drop_duplicates([0], keep = "first")
kospi_admin_stop = kospi_admin_stop[0].tolist()

# 관리종목 / 거래정지종목 중 상장폐지 예정 종목 필터링
kospi_del_stocks = []

for i in range(0, len(kospi_admin_stop)):
    if (kospi_admin_stop[i] in list(kospi_stock["stock"])) == False:
        kospi_del_stock = kospi_admin_stop[i]
        kospi_del_stocks.append(kospi_del_stock)

for i in kospi_del_stocks:
    kospi_admin_stop.remove(i)

# 관리종목 / 거래정지종목 제거  
num_kospi = []

for i in kospi_admin_stop:
    index_num_kospi = kospi_stock[kospi_stock["stock"] == i].index[0]
    num_kospi.append(index_num_kospi)
    
kospi_stock = kospi_stock.drop(num_kospi, axis = 0).reset_index(drop = True)

# A000000 형태로 종목코드 변경
kospi_stock = kospi_stock.astype({"code" : str})
code_list = kospi_stock["code"].tolist()

code_change = []

for i in range(0, len(code_list)):
    if len(code_list[i]) == 6:
        code_data = "A" + code_list[i]
    
    elif len(code_list[i]) == 5:
        code_data = "A0" + code_list[i]
        
    elif len(code_list[i]) == 4:
        code_data = "A00" + code_list[i]
        
    elif len(code_list[i]) == 3:
        code_data = "A000" + code_list[i]
    
    elif len(code_list[i]) == 2:
        code_data = "A0000" + code_list[i]
        
    elif len(code_list[i]) == 1:
        code_data = "A00000" + code_list[i]
        
    code_change.append(code_data)
    
kospi_stock = kospi_stock.drop(["code"], axis = 1)
kospi_stock.insert(0, "code", code_change)
kospi_stock
{% endhighlight %}
<img width="179" alt="스크린샷 2020-11-08 오후 5 17 51" src="https://user-images.githubusercontent.com/70478154/98460281-6247cf00-21e6-11eb-972b-8a9aef30e3b6.png">

그 후, 금융 관련 종목들의 리스트를 추출하고, 이 종목들 역시 제외했다.

{% highlight ruby %}
# 금융 관련 종목 제외 리스트
list_for_drop = kospi_stock["code"].tolist()
drop_list = []

for i in list_for_drop:
    stock = invest_info(i)
    if len(stock) != 17:
        fi_stock = i
        drop_list.append(fi_stock)
        
# 금융 관련 종목 제외
fi_num = []

for i in drop_list:
    index_fi_num = kospi_stock[kospi_stock["code"] == i].index[0]
    fi_num.append(index_fi_num)

kospi_stock = kospi_stock.drop(fi_num, axis = 0).reset_index(drop = True)
kospi_stock
{% endhighlight %}
<img width="168" alt="스크린샷 2020-11-08 오후 5 19 46" src="https://user-images.githubusercontent.com/70478154/98460302-a509a700-21e6-11eb-95c6-b840aff9258f.png">

사용할 코스피 종목들 추출을 완료한 후, 이 데이터를 바탕으로 위에서 만든 투자지표 추출 함수에 코드를 입력하여 코스피 종목 별 투자 지표 데이터를 추출하고, 데이터들을 병합했다. 또한, 칼럼명을 다듬고, 데이터별 종목명 칼럼을 추가했다.

{% highlight ruby %}
# kospi 종목별 투자 지표
list_index_1 = ["EPS", "EBITDAPS", "CFPS", "SPS", "BPS", "DPS(보통주,현금)", "DPS(1우선주,현금)", "배당성향",
                    "PER", "PCR", "PSR", "PBR", "EV/Sales", "EV/EBITDA", "총현금흐름", "총투자", "FCFF", "code"]
kospi_invest = pd.DataFrame(columns = list_index_1)

all_kospi_code = kospi_stock["code"].tolist()

for i in all_kospi_code:
    stock_invest = invest_info(i).iloc[:,1:5].T
    stock_invest.columns = invest_info(i).iloc[:,0].tolist()
    stock_invest["code"] = i
    
    kospi_invest = pd.concat([kospi_invest, stock_invest])
    
kospi_invest = kospi_invest.reset_index()

# 코드별 종목 병합
kospi_invest = kospi_invest.rename(columns = {"index" : "time"})
kospi_invest_all = pd.merge(kospi_invest, kospi_stock, on = "code", how = "outer")
kospi_invest_all
{% endhighlight %}
<img width="788" alt="스크린샷 2020-11-08 오후 5 24 32" src="https://user-images.githubusercontent.com/70478154/98460381-527cba80-21e7-11eb-9442-96d6cdc262ca.png">

위와 같은 방식으로, 사용할 코스닥 종목들을 필터링한 후, 코스닥 종목 별 투자지표들 역시 추출하고 병합했다.

{% highlight ruby %}
# 코스닥 종목 데이터
kosdaq_stock = pd.read_csv("kosdaq_stock.csv")
kosdaq_stock.rename(columns = {"null" : "code", "null.1" : "stock"}, inplace = True)
kosdaq_stock = kosdaq_stock.drop(kosdaq_stock.columns[2:], axis = 1)

# 코스닥 관리종목 리스트
kosdaq_admin = pd.read_csv("kosdaq_admin.csv")
kosdaq_admin = kosdaq_admin["종목명"].tolist()
kosdaq_admin = pd.DataFrame(kosdaq_admin)

# 코스닥 거래정지종목 리스트
kosdaq_stop = pd.read_csv("kosdaq_stop.csv")
kosdaq_stop = kosdaq_stop["종목명"].tolist()
kosdaq_stop = pd.DataFrame(kosdaq_stop)

# 관리종목 / 거래정지종목 중복값 처리
kosdaq_admin_stop = pd.concat([kosdaq_admin, kosdaq_stop])
kosdaq_admin_stop = kosdaq_admin_stop.drop_duplicates([0], keep = "first")
kosdaq_admin_stop = kosdaq_admin_stop[0].tolist()

# 관리종목 / 거래정지종목 중 상장폐지 예정 종목 필터링
kosdaq_del_stocks = []

for i in range(0, len(kosdaq_admin_stop)):
    if (kosdaq_admin_stop[i] in list(kosdaq_stock["stock"])) == False:
        kosdaq_del_stock = kosdaq_admin_stop[i]
        kosdaq_del_stocks.append(kosdaq_del_stock)

for i in kosdaq_del_stocks:
    kosdaq_admin_stop.remove(i)

# 관리종목 / 거래정지종목 제거  
num_kosdaq = []

for i in kosdaq_admin_stop:
    index_num_kosdaq = kosdaq_stock[kosdaq_stock["stock"] == i].index[0]
    num_kosdaq.append(index_num_kosdaq)
    
kosdaq_stock = kosdaq_stock.drop(num_kosdaq, axis = 0).reset_index(drop = True)

# SPAC 종목 제외
cond_spac = kosdaq_stock["stock"].str.contains("스팩")
kosdaq_stock = kosdaq_stock.drop(kosdaq_stock[cond_spac].index, axis = 0).reset_index(drop = True)

# A000000 형태로 종목코드 변경
kosdaq_stock = kosdaq_stock.astype({"code" : str})
kosdaq_code_list = kosdaq_stock["code"].tolist()

kosdaq_code_change = []

for i in range(0, len(kosdaq_code_list)):
    if len(kosdaq_code_list[i]) == 6:
        kosdaq_code_data = "A" + kosdaq_code_list[i]
    
    elif len(kosdaq_code_list[i]) == 5:
        kosdaq_code_data = "A0" + kosdaq_code_list[i]
        
    elif len(kosdaq_code_list[i]) == 4:
        kosdaq_code_data = "A00" + kosdaq_code_list[i]
        
    elif len(kosdaq_code_list[i]) == 3:
        kosdaq_code_data = "A000" + kosdaq_code_list[i]
    
    elif len(kosdaq_code_list[i]) == 2:
        kosdaq_code_data = "A0000" + kosdaq_code_list[i]
        
    elif len(kosdaq_code_list[i]) == 1:
        kosdaq_code_data = "A00000" + kosdaq_code_list[i]
        
    kosdaq_code_change.append(kosdaq_code_data)
    
kosdaq_stock = kosdaq_stock.drop(["code"], axis = 1)
kosdaq_stock.insert(0, "code", kosdaq_code_change)
kosdaq_stock
{% endhighlight %}
<img width="182" alt="스크린샷 2020-11-08 오후 5 30 15" src="https://user-images.githubusercontent.com/70478154/98460477-1eee6000-21e8-11eb-9eac-c322cfc8f120.png">

한번에 너무 많은 양을 연속 추출하게 되면 오류가 발생하기 때문에 반으로 나눠 금융 관련 종목 리스트를 추출하고 제거했다. 또한, 코스닥 종목 별 투자지표들도 추출했다.

{% highlight ruby %}
# 반으로 분할 (오류 방지)
kosdaq_stock_1 = kosdaq_stock[:635]
kosdaq_stock_2 = kosdaq_stock[635:]

# 금융 관련 종목 제외 리스트_1
kosdaq_list_for_drop_1 = kosdaq_stock_1["code"].tolist()
kosdaq_drop_list_1 = []

for i in kosdaq_list_for_drop_1:
    stock = invest_info(i)
    if len(stock) != 17:
        fi_stock = i
        kosdaq_drop_list_1.append(fi_stock)
        
# 금융 관련 종목 제외 리스트_2
kosdaq_list_for_drop_2 = kosdaq_stock_2["code"].tolist()
kosdaq_drop_list_2 = []

for i in kosdaq_list_for_drop_2:
    stock = invest_info(i)
    if len(stock) != 17:
        fi_stock = i
        kosdaq_drop_list_2.append(fi_stock)
    
# 금융 관련 종목 제외_1
fi_num_kosdaq_1 = []

for i in kosdaq_drop_list_1:
    index_fi_num_kosdaq_1 = kosdaq_stock[kosdaq_stock["code"] == i].index[0]
    fi_num_kosdaq_1.append(index_fi_num_kosdaq_1)

kosdaq_stock = kosdaq_stock.drop(fi_num_kosdaq_1, axis = 0).reset_index(drop = True)

# 금융 관련 종목 제외_2
fi_num_kosdaq_2 = []

for i in kosdaq_drop_list_2:
    index_fi_num_kosdaq_2 = kosdaq_stock[kosdaq_stock["code"] == i].index[0]
    fi_num_kosdaq_2.append(index_fi_num_kosdaq_2)

kosdaq_stock = kosdaq_stock.drop(fi_num_kosdaq_2, axis = 0).reset_index(drop = True)
{% endhighlight %}

{% highlight ruby %}
# 반으로 분할 (오류 방지)
kosdaq_stock_1 = kosdaq_stock[:624]
kosdaq_stock_2 = kosdaq_stock[624:]

# kosdaq 종목별 투자 지표_1
list_index_1 = ["EPS", "EBITDAPS", "CFPS", "SPS", "BPS", "DPS(보통주,현금)", "DPS(1우선주,현금)", "배당성향",
                    "PER", "PCR", "PSR", "PBR", "EV/Sales", "EV/EBITDA", "총현금흐름", "총투자", "FCFF", "code"]
kosdaq_invest = pd.DataFrame(columns = list_index_1)

all_kosdaq_code_1 = kosdaq_stock_1["code"].tolist()

for i in all_kosdaq_code_1:
    kosdaq_stock_invest = invest_info(i).iloc[:,1:5].T
    kosdaq_stock_invest.columns = invest_info(i).iloc[:,0].tolist()
    kosdaq_stock_invest["code"] = i
    
    kosdaq_invest = pd.concat([kosdaq_invest, kosdaq_stock_invest])
    
# kosdaq 종목별 투자 지표_2
all_kosdaq_code_2 = kosdaq_stock_2["code"].tolist()

for i in all_kosdaq_code_2:
    kosdaq_stock_invest = invest_info(i).iloc[:,1:5].T
    kosdaq_stock_invest.columns = invest_info(i).iloc[:,0].tolist()
    kosdaq_stock_invest["code"] = i
    
    kosdaq_invest = pd.concat([kosdaq_invest, kosdaq_stock_invest])
    
kosdaq_invest = kosdaq_invest.reset_index()

# 코드별 종목 병합
kosdaq_invest = kosdaq_invest.rename(columns = {"index" : "time"})
kosdaq_invest_all = pd.merge(kosdaq_invest, kosdaq_stock, on = "code", how = "outer")
kosdaq_invest_all
{% endhighlight %}
<img width="781" alt="스크린샷 2020-11-08 오후 5 34 56" src="https://user-images.githubusercontent.com/70478154/98460546-c5d2fc00-21e8-11eb-8610-74aab94b7e1d.png">

추출한 코스피, 코스닥의 종목 별 투자지표들을 병합하고 저장했다.

{% highlight ruby %}
# 코스피, 코스닥 투자 지표 병합
invest_all = pd.concat([kospi_invest_all, kosdaq_invest_all])
invest_all
{% endhighlight %}
<img width="785" alt="스크린샷 2020-11-08 오후 5 36 28" src="https://user-images.githubusercontent.com/70478154/98460577-fb77e500-21e8-11eb-99a2-85444b00eace.png">

{% highlight ruby %}
# 투자 지표 데이터 저장
invest_all.to_csv("invest_all.csv", header = True, index = False)
{% endhighlight %}<BR/><BR/>

#### 재무비율

이번에는 코스피, 코스닥 종목 별 재무비율 데이터를 추출했다. 위에서 필터링한, 사용할 코스피, 코스닥 종목 리스트를 활용하여 재무비율 추출 함수와 함께 코스피, 코스닥 종목 별 재무비율 데이터를 추출하고 병합했다. 모든 과정과 방식은 위와 동일하다.

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
    num = df.index[df.iloc[:,0].isin(["안정성비율", "성장성비율", "수익성비율", "활동성비율"])]   # IFRS(연결), IFRS(별도) 두 종류이기 때문
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

{% highlight ruby %}
# kospi 재무비율
list_finance_1 = ['유동비율', '당좌비율', '부채비율', '유보율', '순차입금비율', '이자보상배율', '자기자본비율', '매출액증가율',
                  '판매비와관리비증가율', '영업이익증가율', 'EBITDA증가율', 'EPS증가율', '매출총이익율', '세전계속사업이익률',
                  '영업이익률', 'EBITDA마진율', 'ROA', 'ROE', 'ROIC', '총자산회전율', '총부채회전율', '총자본회전율',
                  '순운전자본회전율']
kospi_finance = pd.DataFrame(columns = list_finance_1)

all_kospi_code = kospi_stock["code"].tolist()

for i in all_kospi_code:
    stock_fi_kospi = stock_info(i).iloc[:,1:5].T
    stock_fi_kospi.columns = stock_info(i).iloc[:,0].tolist()
    stock_fi_kospi["code"] = i
    
    kospi_finance = pd.concat([kospi_finance, stock_fi_kospi])

kospi_finance = kospi_finance.reset_index()
kospi_finance_all = kospi_finance.rename(columns = {"index" : "time"})
{% endhighlight %}

{% highlight ruby %}
# kosdaq 재무비율_1
kosdaq_finance = pd.DataFrame(columns = list_finance_1)

all_kosdaq_code_1 = kosdaq_stock_1["code"].tolist()

for i in all_kosdaq_code_1:
    stock_fi_kosdaq = stock_info(i).iloc[:,1:5].T
    stock_fi_kosdaq.columns = stock_info(i).iloc[:,0].tolist()
    stock_fi_kosdaq["code"] = i
    
    kosdaq_finance = pd.concat([kosdaq_finance, stock_fi_kosdaq])
    
# kosdaq 재무비율_2
all_kosdaq_code_2 = kosdaq_stock_2["code"].tolist()

for i in all_kosdaq_code_2:
    stock_fi_kosdaq = stock_info(i).iloc[:,1:5].T
    stock_fi_kosdaq.columns = stock_info(i).iloc[:,0].tolist()
    stock_fi_kosdaq["code"] = i
    
    kosdaq_finance = pd.concat([kosdaq_finance, stock_fi_kosdaq])
    
kosdaq_finance = kosdaq_finance.reset_index()
kosdaq_finance_all = kosdaq_finance.rename(columns = {"index" : "time"})
{% endhighlight %}

{% highlight ruby %}
# 코스피, 코스닥 재무비율 병합
finance_all = pd.concat([kospi_finance_all, kosdaq_finance_all])
finance_all
{% endhighlight %}
<img width="777" alt="스크린샷 2020-11-08 오후 5 44 20" src="https://user-images.githubusercontent.com/70478154/98460714-18f97e80-21ea-11eb-9480-b9323763d76a.png">

마찬가지로, 재무비율 데이터를 저장했다.

{% highlight ruby %}
# 재무비율 데이터 저장
finance_all.to_csv("finance_all.csv", header = True, index = False)
{% endhighlight %}

</div>
</details>
<BR/>
<details>
<summary>데이터 전처리</summary>
<div markdown="1">
<BR/><BR/>
수집한 재무비율, 투자지표 데이터들을 불러와 병합했다.

{% highlight ruby %}
# 데이터 불러오기
import pandas as pd

finance_all = pd.read_csv("finance_all.csv")        # 재무비율 데이터
invest_all = pd.read_csv("invest_all.csv")          # 투자지표 데이터

# 데이터 병합
all_stock = pd.merge(finance_all, invest_all, on = ["time", "code"], how = "inner")
all_stock
{% endhighlight %}
<img width="779" alt="스크린샷 2020-11-08 오후 6 07 44" src="https://user-images.githubusercontent.com/70478154/98461172-59a6c700-21ed-11eb-9841-5a372b1a47cc.png">

먼저, target 데이터가 사업보고서 공시 직후부터 3개월 뒤의 주가 상승 혹은 하락이기 때문에 외부 경제 상황, 이슈 등을 동일하게 하여 분석의 일관성을 유지하기 위해 12월 결산을 제외한 데이터들을 삭제했다. 즉, time 칼럼에서 `0000/12`가 아닌 데이터들을 제거했다.

{% highlight ruby %}
# 12월 결산을 제외한 데이터 삭제
all_stock_data = all_stock.copy()
time_list = all_stock_data["time"].unique().tolist()

for i in ["2016/12", "2017/12", "2018/12", "2019/12"]:
    time_list.remove(i)

num_list = []

for i in time_list:
    num = all_stock[all_stock["time"] == i].index
    num_list.append(num)

for i in range(0, 20):
    all_stock_data = all_stock_data.drop(num_list[i], axis = 0)
    
all_stock_data = all_stock_data.reset_index(drop = True)
{% endhighlight %}

다음은, 결측치가 문자열로 되어있는 데이터를 결측치의 형태로 바꿔준 후, 한 행에서 결측치가 9개 이상인 행을 삭제했다. 그 후, 분석에서 사용하지 않을 칼럼인 순차입금비율, 순운전자본회전율, DPS(보통주,현금), DPS(1우선주,현금)를 제외했다. 다른 데이터와 겹치는 특성을 가지고 있거나 결측치가 매우 많은 칼럼을 제거했다.

{% highlight ruby %}
# 결측치인데 문자열로 되어있는 데이터 처리
import numpy as np

for i in range(0, len(all_stock_data.columns)):
    if "N/A(IFRS)" in list(all_stock_data.iloc[:,i]):
        all_stock_data.iloc[:,i] = all_stock_data.iloc[:,i].replace("N/A(IFRS)", np.nan)
        
# 한 행의 결측치가 9개 이상인 행 삭제
index_nan = pd.DataFrame(all_stock_data.isnull().sum(1) >= 9)
index_nan_list = list(index_nan[index_nan[0] == True].index)

all_stock_data = all_stock_data.drop(index_nan_list, axis = 0).reset_index(drop = True)

# 필요없는 feature 제거
all_stock_data = all_stock_data.drop(["순차입금비율", "순운전자본회전율", "DPS(보통주,현금)", "DPS(1우선주,현금)"], axis = 1)
all_stock_data
{% endhighlight %}

또, 완전잠식 데이터가 있는 행을 제거했다.

{% highlight ruby %}
# 완전잠식 데이터가 있는 행 제거
import itertools

drop_list = []

for i in [3, 6, 17, 21]:
    drop_data = list(all_stock_data[all_stock_data.iloc[:,i] == "완전잠식"].index)
    drop_list.append(drop_data)
    
drop_list = list(set(list(itertools.chain.from_iterable(drop_list))))

all_stock_data = all_stock_data.drop(drop_list, axis = 0).reset_index(drop = True)
all_stock_data
{% endhighlight %}
<img width="781" alt="스크린샷 2020-11-08 오후 6 20 06" src="https://user-images.githubusercontent.com/70478154/98461396-19e0df00-21ef-11eb-91cf-5b6cf27c3b3b.png">

target을 생성하기 위해, 보고서 공시 직후의 종목 별 종가 데이터를 불러왔다. 대체로 12월 결산 보고서는 다음 해 3월 말, 4월 초에 공시가 되기 때문에 약간의 여유를 두고 4월 10일 종가를 기준으로 했다. 만약 그 날이 주말인 경우, 그 다음주 월요일 종가를 기준으로 했다.

{% highlight ruby %}
# 기준 가격 (공시 직후 가격)
kospi_17_4 = pd.read_csv("kospi_17_4.csv")
kospi_18_4 = pd.read_csv("kospi_18_4.csv")
kospi_19_4 = pd.read_csv("kospi_19_4.csv")
kospi_20_4 = pd.read_csv("kospi_20_4.csv")

kosdaq_17_4 = pd.read_csv("kosdaq_17_4.csv")
kosdaq_18_4 = pd.read_csv("kosdaq_18_4.csv")
kosdaq_19_4 = pd.read_csv("kosdaq_19_4.csv")
kosdaq_20_4 = pd.read_csv("kosdaq_20_4.csv")
{% endhighlight %}

{% highlight ruby %}
# time 칼럼 추가
kospi_17_4["time"] = "2016/12"
kospi_18_4["time"] = "2017/12"
kospi_19_4["time"] = "2018/12"
kospi_20_4["time"] = "2019/12"

kosdaq_17_4["time"] = "2016/12"
kosdaq_18_4["time"] = "2017/12"
kosdaq_19_4["time"] = "2018/12"
kosdaq_20_4["time"] = "2019/12"
{% endhighlight %}

{% highlight ruby %}
# 데이터 병합
all_price = kospi_17_4.copy()

for i in [kospi_18_4, kospi_19_4, kospi_20_4, kosdaq_17_4, kosdaq_18_4, kosdaq_19_4, kosdaq_20_4]:
    all_price = pd.concat([all_price, i])
    
all_price = all_price.rename(columns = {"null.2" : "price"}).reset_index(drop = True)
all_price
{% endhighlight %}

{% highlight ruby %}
# 전처리를 위한 함수
def change_price(df):
    
    # 칼럼 명 변경 및 필요없는 칼럼 삭제
    df = df.drop(["null.3", "null.4", "null.5", "null.6", "null.7"], axis = 1)
    df = df.rename(columns = {"null" : "code", "null.1" : "stock"})
    
    # A000000 형태로 종목코드 변경
    df = df.astype({"code" : str})
    code_list = df["code"].tolist()

    code_change = []

    for i in range(0, len(code_list)):
        if len(code_list[i]) == 6:
            code_data = "A" + code_list[i]
    
        elif len(code_list[i]) == 5:
            code_data = "A0" + code_list[i]
        
        elif len(code_list[i]) == 4:
            code_data = "A00" + code_list[i]
        
        elif len(code_list[i]) == 3:
            code_data = "A000" + code_list[i]
    
        elif len(code_list[i]) == 2:
            code_data = "A0000" + code_list[i]
        
        elif len(code_list[i]) == 1:
            code_data = "A00000" + code_list[i]
        
        code_change.append(code_data)
    
    df = df.drop(["code"], axis = 1)
    df.insert(0, "code", code_change)

    # - 제거
    index_empty = df[df.iloc[:,2] == "-"].index
    df = df.drop(index_empty).reset_index(drop = True)

    # price 데이터 숫자형으로 변경
    df.iloc[:,2] = df.iloc[:,2].str.replace(",", "")
    df = df.astype({df.columns[2] : int})
    
    return df
{% endhighlight %}

{% highlight ruby %}
# 함수 적용
all_price = change_price(all_price)
all_price
{% endhighlight %}
<img width="301" alt="스크린샷 2020-11-08 오후 6 29 12" src="https://user-images.githubusercontent.com/70478154/98461552-58c36480-21f0-11eb-9ba6-5e22de9c435f.png">

기준 가격에 대한 데이터를 불러와 정리한 후, 기준 가격의 시점을 기준으로 3개월 뒤의 종가를 불러와 마찬가지로 정리했다.

{% highlight ruby %}
# 3개월 후 가격
kospi_17_7 = pd.read_csv("kospi_17_7.csv")
kospi_18_7 = pd.read_csv("kospi_18_7.csv")
kospi_19_7 = pd.read_csv("kospi_19_7.csv")
kospi_20_7 = pd.read_csv("kospi_20_7.csv")

kosdaq_17_7 = pd.read_csv("kosdaq_17_7.csv")
kosdaq_18_7 = pd.read_csv("kosdaq_18_7.csv")
kosdaq_19_7 = pd.read_csv("kosdaq_19_7.csv")
kosdaq_20_7 = pd.read_csv("kosdaq_20_7.csv")
{% endhighlight %}

{% highlight ruby %}
# time 칼럼 추가
kospi_17_7["time"] = "2016/12"
kospi_18_7["time"] = "2017/12"
kospi_19_7["time"] = "2018/12"
kospi_20_7["time"] = "2019/12"

kosdaq_17_7["time"] = "2016/12"
kosdaq_18_7["time"] = "2017/12"
kosdaq_19_7["time"] = "2018/12"
kosdaq_20_7["time"] = "2019/12"
{% endhighlight %}

{% highlight ruby %}
# 데이터 병합
after_3M = kospi_17_7.copy()

for i in [kospi_18_7, kospi_19_7, kospi_20_7, kosdaq_17_7, kosdaq_18_7, kosdaq_19_7, kosdaq_20_7]:
    after_3M = pd.concat([after_3M, i])
    
after_3M = after_3M.rename(columns = {"null.2" : "after_3M"}).reset_index(drop = True)
after_3M
{% endhighlight %}

{% highlight ruby %}
# 함수 적용
after_3M = change_price(after_3M)
after_3M
{% endhighlight %}
<img width="309" alt="스크린샷 2020-11-08 오후 6 32 12" src="https://user-images.githubusercontent.com/70478154/98461592-c2dc0980-21f0-11eb-9151-2aa2f035e179.png">

종목 별 재무비율, 투자지표 데이터를 기준으로 정리한 기준 가격과 3개월 뒤의 종가를 병합했다.

{% highlight ruby %}
# 기준 가격 병합
all_stock_info = all_stock_data.copy()
all_stock_info = pd.merge(all_stock_info, all_price, on = ["code", "stock", "time"], how = "inner")

# 기준 가격에 3개월 후 가격 병합
all_stock_info_3M = pd.merge(all_stock_info, after_3M, on = ["code", "stock", "time"], how = "inner")
all_stock_info_3M
{% endhighlight %}
<img width="788" alt="스크린샷 2020-11-08 오후 6 37 00" src="https://user-images.githubusercontent.com/70478154/98461668-77762b00-21f1-11eb-90d3-f850c206c3e9.png">

마지막으로, 몇몇 칼럼들에 존재하는 `흑자전환` `적자전환` `적자지속`과 같은 범주형 데이터들을 처리했다. 이 범주형 데이터의 수가 매우 적은 경우는 0으로 변환하고, 수가 많은 경우 새로운 feature를 만들어 처리했다. 또한, 분석에 활용할 데이터들의 타입을 숫자형으로 변경한 후, 데이터를 저장했다.

{% highlight ruby %}
def replace_str(df):
    
    # 판매비와관리비증가율의 흑전, 적전, 적지를 0으로 변환 (데이터 수가 너무 적어서)
    for i in ["흑전", "적전", "적지"]:
        df["판매비와관리비증가율"] = df["판매비와관리비증가율"].replace(i, 0)
        
    # 영업이익증가율, EBITDA증가율, EPS증가율의 흑전, 적전, 적지를 0으로 변환
    df["영업이익증가율_cat"] = df["영업이익증가율"]
    df["EBITDA증가율_cat"] = df["EBITDA증가율"]
    df["EPS증가율_cat"] = df["EPS증가율"]
    
    for i in ["흑전", "적전", "적지"]:
        for j in [9, 10, 11]:
            df.iloc[:,j] = df.iloc[:,j].replace(i, 0)
        
    # 추가한 범주형 feature에 흑전, 적전, 적지를 제외한 데이터들 모두 0으로 변환
    for i in range(0, len(df["영업이익증가율_cat"])):
        data_1 = df["영업이익증가율_cat"][i]
    
        if data_1 != "흑전" and data_1 != "적전" and data_1 != "적지":
            df["영업이익증가율_cat"] = df["영업이익증가율_cat"].replace(data_1, 0)
            
    for i in range(0, len(df["EBITDA증가율_cat"])):
        data_2 = df["EBITDA증가율_cat"][i]
        
        if data_2 != "흑전" and data_2 != "적전" and data_2 != "적지":
            df["EBITDA증가율_cat"] = df["EBITDA증가율_cat"].replace(data_2, 0)
            
    for i in range(0, len(df["EPS증가율_cat"])):
        data_3 = df["EPS증가율_cat"][i]
        
        if data_3 != "흑전" and data_3 != "적전" and data_3 != "적지":
            df["EPS증가율_cat"] = df["EPS증가율_cat"].replace(data_3, 0)
            
    # 추가한 범주형 feature에 흑전, 적전, 적지를 각각 1, -1, -2로 변환
    for j in [41, 42, 43]:
        df.iloc[:,j] = df.iloc[:,j].replace("흑전", 1)
        df.iloc[:,j] = df.iloc[:,j].replace("적전", -1)
        df.iloc[:,j] = df.iloc[:,j].replace("적지", -2)
    
    return df
{% endhighlight %}

{% highlight ruby %}
# 함수 적용
all_stock_info_3M = replace_str(all_stock_info_3M)
{% endhighlight %}

{% highlight ruby %}
# 숫자형으로 데이터 타입 변경
def to_float(df):
    for i in range(0, len(df.columns)):
        if df.dtypes[i] == "object":
            df.iloc[:,i] = df.iloc[:,i].str.replace(",", "")
            
    col_df = df.drop(df.columns[40], axis = 1)
    cols = col_df.drop(["time", "code", "stock", "price", "영업이익증가율_cat", "EBITDA증가율_cat", "EPS증가율_cat"], axis = 1).columns
    
    for col in cols:
        df = df.astype({col : "float"})
        
    return df
{% endhighlight %}

{% highlight ruby %}
# 함수 적용
all_stock_info_3M = to_float(all_stock_info_3M)
all_stock_info_3M
{% endhighlight %}
<img width="781" alt="스크린샷 2020-11-08 오후 6 41 52" src="https://user-images.githubusercontent.com/70478154/98461745-1ef35d80-21f2-11eb-9251-cf92201cb059.png">

{% highlight ruby %}
# 데이터 저장
all_stock_info_3M.to_csv("all_stock_info_3M.csv", header = True, index = False)
{% endhighlight %}

</div>
</details>
<BR/><BR/>
정리하여 저장한 데이터를 불러왔다.

{% highlight ruby %}
# 데이터 불러오기
import pandas as pd

all_stock_info_3M = pd.read_csv("all_stock_info_3M.csv")
all_stock_info_3M
{% endhighlight %}
<img width="985" alt="스크린샷 2020-11-05 오후 10 57 01" src="https://user-images.githubusercontent.com/70478154/98249998-44962200-1fba-11eb-97ee-76d0a6973042.png">

배당성향의 결측치을 `0`으로 채웠다.

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

산출된 수익률을 바탕으로, `상승`이면 `1`, `하락`이면 `0`으로 target을 만들었다. 그 후, train과 test set을 분리했고, 분석을 위해 필요없는 칼럼을 제거했다.

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

Confusion matrix와 f1-score를 확인한 결과, 한 쪽으로 쏠리지 않고, 각 class가 균일하게 예측이 되었음을 확인할 수 있다.

이 외에도 Class 수, 적용 모델, target의 기간을 변화시키며 결과를 확인했다. `Class`를 **2개와 3개**로, `모델`을 **RandomForestClassifier와 XGBClassifier**로, `target 기간`을 **3개월 뒤와 6개월 뒤**로 변화시켜 적용했다.<BR/><BR/>

**2 Class인 경우**

<p float="left">
<img width="320" alt="스크린샷 2020-11-06 오전 10 54 32" src="https://user-images.githubusercontent.com/70478154/98316835-80fb6980-201e-11eb-8707-bd7115b48e74.png" width="30%"/>
<img width="319" alt="스크린샷 2020-11-06 오전 11 05 49" src="https://user-images.githubusercontent.com/70478154/98317565-14816a00-2020-11eb-8b89-6115a936c4f6.png" width="30%"/>
<img width="328" alt="스크린샷 2020-11-06 오전 10 55 19" src="https://user-images.githubusercontent.com/70478154/98316898-a1c3bf00-201e-11eb-8646-1adac795faac.png" width="30%"/>
</p>

위의 결과에 따르면, target 기간이 6개월인 경우, baseline에 못 미치는 성능을 보여주었고, RandomForestClassifier 모델인 경우 과적합 문제가 상대적으로 크게 나타났다. **target 기간이 3개월이면서 XGBClassifier 모델**을 사용한 경우에 가장 높은 성능을 보였다.

**3 Class인 경우**

<p float="left">
<img width="321" alt="스크린샷 2020-11-06 오전 10 57 39" src="https://user-images.githubusercontent.com/70478154/98317004-ee0eff00-201e-11eb-88ce-a7051f4b10aa.png" width="30%"/>
<img width="320" alt="스크린샷 2020-11-06 오전 10 58 00" src="https://user-images.githubusercontent.com/70478154/98317023-fc5d1b00-201e-11eb-9e1c-b449aef57a31.png" width="30%"/>
<img width="319" alt="스크린샷 2020-11-06 오전 10 58 26" src="https://user-images.githubusercontent.com/70478154/98317037-097a0a00-201f-11eb-9cc1-20e95d19efba.png" width="30%"/>
</p>

위의 결과에 따르면, target 기간과 모델에 상관없이 비슷한 성능을 보여주었지만, 상대적으로 **target 기간이 6개월이면서 XGBClassifier 모델**을 사용한 경우에 가장 높은 성능을 보여주었다.

두 결과를 종합해보았을 때, 3 Class인 경우 역시 baseline을 상회하는 성능을 보여주고 있지만, 낮은 예측도를 보이고 있기 때문에 활용하기에는 무리가 있다. 따라서, **2 Class의 target 기간이 3개월인 데이터에 XGBClassifier를 적용한 모델**이 가장 적합하다고 판단했다.<BR/><BR/>

비록 선택한 모델의 예측 정확도가 매우 높지는 않지만, 상승 target의 비중이 약 47.3%이며 이를 상회하는 f1-score를 보이고 있기 때문에 이 모델을 사용하지 않았을 때보다 사용했을 때 더 긍정적인 결과를 기대할 수 있을 것이라 생각했다.

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

plt.figure(figsize = (5, 5))
plt.bar(label, rate_data, width = 0.5, color = ["gray", "red", "blue"], alpha = 0.6)

plt.text(-0.17, 4.5, "4.36%", fontsize = 12)
plt.text(0.81, 7.35, "7.19%", fontsize = 12)
plt.text(1.82, 2, "1.79%", fontsize = 12)
plt.ylim(0, 10)
plt.show()

print("전 종목 투자 결과: ", both, "%")
print("상승을 예측한 종목 투자 결과: ", up, "%")
print("하락을 예측한 종목 투자 결과: ", down, "%")
{% endhighlight %}
<img width="324" alt="스크린샷 2020-11-06 오전 10 28 48" src="https://user-images.githubusercontent.com/70478154/98315179-fc5b1c00-201a-11eb-9c2a-abacf5adbf62.png">

결과에 따르면, **상승을 예측한 종목들에 같은 비중으로 투자**한 결과 `7.19%`의 수익률을 얻을 수 있음을 알 수 있다. **전 종목에 투자**했을 때의 결과인 `4.36%`, **하락을 예측한 종목들에 투자**했을 때의 결과인 `1.79%`와 비교했을 때 더 좋은 결과를 만들어냈다.

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

먼저, 그래프에 따르면, `EBITDA증가율`은 대체로 target과 **양의 상관관계**를 가진다는 것을 알 수 있다. 즉, EBITDA증가율이 증가할수록 `상승`으로 예측할 가능성이 커진다는 의미이다. EBITDA증가율이 증가하면, 기업의 이익으로 직결되기 때문에 상승으로의 예측과 양의 상관관계가 있다는 것을 쉽게 이해할 수 있다.

{% highlight ruby %}
# PDP - EBITDA증가율
import matplotlib as mpl
from pdpbox.pdp import pdp_isolate, pdp_plot

plt.rcParams['figure.dpi'] = 144
mpl.rc('axes', unicode_minus = False)

X_test_df = pd.DataFrame(X_test_scaler, columns = X_test_3M.columns.tolist())

feature_1 = "EBITDA증가율"

isolated = pdp_isolate(
    model = fi_pipe_3M.named_steps["xgbclassifier"],
    dataset = X_test_df,
    model_features = X_test_df.columns,
    feature = feature_1
)

pdp_plot(isolated, feature_name = feature_1, plot_params = {'font_family' : 'AppleGothic'});
{% endhighlight %}
<img width="994" alt="스크린샷 2020-11-06 오후 5 17 01" src="https://user-images.githubusercontent.com/70478154/98342849-54624480-2054-11eb-9726-80dde6705814.png">

두 번째로, `유보율`은 **극초반에 target과 양의 상관관계를 보이다가 대체로 음의 상관관계**를 가진다는 것을 확인할 수 있다. 유보율이 높으면 예상치 못한 위기 상황에 대응할 수 있는 현금이 있기 때문에 안정적인 재무 구조를 유지할 수 있지만, 유보율이 과도하게 높아진다면 기업이 마땅한 투자처를 찾지 못해 수익을 창출하지 못하고 현금만 쌓아두는 것일 수 있다. 따라서 이러한 유보율의 특징이 반영되었음을 예상해 볼 수 있다.

{% highlight ruby %}
# PDP - 유보율
import matplotlib as mpl
from pdpbox.pdp import pdp_isolate, pdp_plot

plt.rcParams['figure.dpi'] = 144
mpl.rc('axes', unicode_minus = False)

X_test_df = pd.DataFrame(X_test_scaler, columns = X_test_3M.columns.tolist())

feature_2 = "유보율"

isolated = pdp_isolate(
    model = fi_pipe_3M.named_steps["xgbclassifier"],
    dataset = X_test_df,
    model_features = X_test_df.columns,
    feature = feature_2,
)

pdp_plot(isolated, feature_name = feature_2, plot_params = {'font_family' : 'AppleGothic'});
{% endhighlight %}
<img width="982" alt="스크린샷 2020-11-06 오후 5 17 17" src="https://user-images.githubusercontent.com/70478154/98342913-6e038c00-2054-11eb-8aa3-e1b2fff1fa01.png">

세 번째로, `판매비와관리비증가율`은 전반적으로 **상승으로의 예측에 양의 영향을 주고 있지만, 판매비와관리비증가율이 커질수록 그 영향의 크기는 조금씩 떨어짐**을 알 수 있다. 판관비는 비용이기 때문에 일반적으로, 기업 이익에 긍정적인 영향을 줄 수만은 없다. 그러나 판관비에 속하는 비용에는 광고비, 복리후생비, 교육훈련비 등을 포함하고 있기 때문에 추후 기업의 이익으로 되돌아올 수 있는 비용이라고 할 수 있다. 이러한 점 때문에 전반적으로 `상승`으로의 예측에 긍정적인 영향을 주고 있지만, 이 비용이 과도할 경우 오히려 기업 이익에 부정적 효과를 야기할 수 있다고 해석할 수 있다.

{% highlight ruby %}
# PDP - 판매비와관리비증가율
import matplotlib as mpl
from pdpbox.pdp import pdp_isolate, pdp_plot

plt.rcParams['figure.dpi'] = 144
mpl.rc('axes', unicode_minus = False)

X_test_df = pd.DataFrame(X_test_scaler, columns = X_test_3M.columns.tolist())

feature_3 = "판매비와관리비증가율"

isolated = pdp_isolate(
    model = fi_pipe_3M.named_steps["xgbclassifier"],
    dataset = X_test_df,
    model_features = X_test_df.columns,
    feature = feature_3,
)

pdp_plot(isolated, feature_name = feature_3, plot_params = {'font_family' : 'AppleGothic'});
{% endhighlight %}
<img width="993" alt="스크린샷 2020-11-06 오후 5 17 36" src="https://user-images.githubusercontent.com/70478154/98342992-84a9e300-2054-11eb-8478-e546857d7440.png">

네 번째로, `PBR`은 대체로 target과 **음의 상관관계**를 보임을 알 수 있다. 즉, PBR이 높을수록 `상승`으로 예측할 가능성이 낮아진다는 의미이다. 이를 통해, 처음에 언급했던 것처럼 주당순자산 대비 주가가 저평가되어 있는 종목들의 상승 기대가 높은 저PBR 전략이 이 모델에서 작동하고 있다고 해석할 수 있다.

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

다섯 번째로, `PER`은 target과 **양의 상관관계**를 보임을 알 수 있다. 즉, PER이 높을수록 `상승`으로 예측할 가능성이 커진다는 의미이다. PBR과는 반대로 처음에 언급했던 저PER 전략이 이 모델에서는 작동하지 않는다고 해석할 수 있다.

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

* test set에 대한 투자 백테스팅 결과, 모델을 사용하지 않았을 때의 기대수익률보다 모델을 사용했을 때 성과가 더 좋다.
* 동일 비중으로의 투자를 가정하면, 모델을 사용했을 때 더 적은 최소 투자 금액이 필요하므로 거래비용 감소 효과가 있다.
* 투자를 위한 개별 종목 탐색 시간을 절약할 수 있다.<BR/><BR/><BR/><BR/>

## 한계

* 예측도가 높지 않기 때문에 이 모델을 바탕으로 개별 종목에 대한 매수 추천을 할 수 없다.
* 만약 상승으로 예측한 종목들 중 폭락한 종목이 있을 경우, 모델을 통한 투자 결과에 악영향을 줄 수 있기 때문에 변동성을 고려한 target과 예측 모델을 적용할 필요가 있다.
* 1년 주기의 데이터를 바탕으로 모델을 만들었지만, target은 3개월 뒤만을 예측하므로 투자의 연속성을 유지할 수 없다.
