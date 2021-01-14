---
layout: post
title:  LSTM을 활용한 Risk Parity 전략 구현
date:   2021-01-12
image:  weight.png
tags:   Data Finance
---
## Risk Parity 전략 소개

**Risk Parity 전략**은 자산 배분 전략 중 하나로, 각 자산의 포트폴리오에 대한 **리스크 기여도를 동일하게 맞춰줄 수 있는 비중**으로 자산 배분하는 전략이다. 예를 들어, 아래와 같이 주식과 채권에 자산 배분을 하고자 하며, 각 자산 별 기대 수익률, 변동성, sharp ratio 다음과 같다고 한다.

만약 `60/40 포트폴리오` 전략을 사용한다면 주식에 60%, 채권에 40%를 투자하게 될 것이다. 이 때, 포트폴리오의 변동성은 아래와 같이 11.2%이며 기대 수익률은 5.35%이다. 또한, 각 자산 별 리스크 기여도도 다음과 같이 산출할 수 있다. 채권에 비해 주식의 리스크 기여도와 변동성이 매우 높다는 것을 확인할 수 있다.

이번에는 `Risk Parity 포트폴리오` 전략을 사용한다면, 주식과 채권의 리스크 기여도를 50대 50으로 동일하게 맞춰줄 수 있는 각 자산의 투자 비중을 산출하고, 그 비중으로 투자하게 될 것이다. 이러한 방식으로 각 자산 배분 비중을 산출하면, 주식은 20.5%, 채권은 79.5%에 투자하게 될 것이다. 이 때의 포트폴리오 변동성은 5.5%, 기대 수익률은 4.46%로 계산될 수 있다.

이러한 결과만 놓고 봤을 때, 60/40 포트폴리오 전략의 기대 수익률은 `5.35%`로, Risk Parity 포트폴리오 전략의 기대 수익률인 `4.46%`보다 높기 때문에 과연 Risk Parity 전략을 사용하는 것이 맞는 것인가 하는 의문이 들 수 있다. 하지만, 변동성을 확인해보면, 60/40 포트폴리오의 변동성이 2배 높은 것을 확인할 수 있다. Risk Parity 전략은 이러한 점을 활용한다. 60/40 포트폴리오 전략과 동일한 변동성으로 맞춰주기 위해 2배 레버리지를 사용한다. 레버리지 비용이 3%라고 가정했을 때, Risk Parity의 최종 수익률은 `(4.46% X 2) - 3% = 5.92%`이다. 따라서, 60/40 포트폴리오 전략보다 더 높은 수익률을 얻을 수 있다는 것을 알 수 있다.
<img width="1150" alt="스크린샷 2021-01-14 오후 12 09 46" src="https://user-images.githubusercontent.com/70478154/104539757-8352f300-5661-11eb-88b9-7dd6a65af44e.png"><BR/><BR/><BR/><BR/>

## 분석 목적

첫 번째 분석 목적은, 기존의 Risk Parity 전략을 사용했을 때, **과거 변동성으로 인한 괴리**가 발생할 수 있기 때문에 이를 해결하고자 분석을 진행했다. Risk Parity 전략에서 리스크 기여도를 동일하게 맞춰줄 수 있는 각 자산의 투자 비중을 산출하는 과정에서 각 자산의 과거 변동성을 활용한다. 만약 다음 Rebalancing 시점까지의 변동성이 크게 상승한다면, 투자 기간 동안의 변동성을 과소 평가한 결과를 낳을 것이고, 반대로 변동성이 크게 하락한다면, 과대 평가한 결과를 낳을 수 있다. 이러한 점으로 인해, 포트폴리오의 수익률을 극대화할 수 없게 된다. 따라서, LSTM 모델을 활용하여 예측된 변동성을 활용한다면, 이러한 괴리를 줄일 수 있지 않을까 기대한다.
<img width="1144" alt="스크린샷 2021-01-14 오후 12 20 32" src="https://user-images.githubusercontent.com/70478154/104540506-f1e48080-5662-11eb-9427-cb925917dfd0.png">

두 번째 분석 목적은, 적절한 자산 배분을 통해 **취약한 Risk 대응 능력**을 보완할 수 있기 때문에 분석을 진행했다. 최근 코로나로 인해, 주가가 급락하는 상황이 벌어졌다. 만약, 적절한 자산 배분을 하지 못했다면 이러한 Risk를 온전히 떠안을 지 모른다. 따라서, 적절한 자산 배분을 통해 안정적으로 수익률의 우상향 곡선을 그릴 수 있지 않을까 기대한다.
<img width="1144" alt="스크린샷 2021-01-14 오후 12 28 58" src="https://user-images.githubusercontent.com/70478154/104541037-1e4ccc80-5664-11eb-98e0-437024c1e59d.png"><BR/><BR/><BR/><BR/>

## 데이터 소개

활용한 데이터는 다음과 같다. 2011년 12월 23일부터 2020년 12월 1일까지의 S&P500 ETF와 국채 ETF 데이터를 활용했다. INPUT 데이터는 기술 지표들로 구성되어 있으며, OUTPUT 데이터는 다음 20일 간의 변동성으로 설정했다.
<img width="1142" alt="스크린샷 2021-01-14 오후 12 34 14" src="https://user-images.githubusercontent.com/70478154/104541362-d8443880-5664-11eb-8538-0f5848329c18.png"><BR/><BR/><BR/><BR/>

## 적용 과정

#### LSTM 모델 구조

LSTM 모델은 총 5개의 층으로 구성했으며, 이를 바탕으로 OUTPUT을 산출한다.
<img width="1140" alt="스크린샷 2021-01-14 오후 12 41 25" src="https://user-images.githubusercontent.com/70478154/104541896-d8910380-5665-11eb-8910-9d9fbe25cf6d.png">

{% highlight ruby %}
# LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow as tf

def lstm():
    model = Sequential()
    model.add(LSTM(128, input_shape = (20, 15), return_sequences = True, recurrent_regularizer = regularizers.l2(0.01), activation = "relu"))
    model.add(LSTM(128, return_sequences = True, recurrent_regularizer = regularizers.l2(0.01), activation = "relu"))
    model.add(LSTM(128, return_sequences = True, recurrent_regularizer = regularizers.l2(0.01), activation = "relu"))
    model.add(LSTM(128, return_sequences = True, recurrent_regularizer = regularizers.l2(0.01), activation = "relu"))
    model.add(LSTM(128, return_sequences = False, recurrent_regularizer = regularizers.l2(0.01), activation = "relu"))
    model.add(Dense(1))

    optimizer = Adam(0.0005)
    model.compile(loss = "mse", optimizer = optimizer, metrics = ["mse"])

    model.summary()
    
    return model
{% endhighlight %}
<BR/><BR/>

#### 학습 방법: Moving Window

학습은 Moving Window 방식을 사용했다. 특성 15개의 20영업일 데이터를 한 데이터로 보았고, 이 데이터의 50 묶음인 1000 영업일을 한 세트로 만들어 각 세트 별로 학습과 예측을 진행했다. 한 세트는 980 영업일의 Train set과 20 영업일의 Test set으로 구성되어 있다. 총 61개의 세트를 학습하고 예측했다. 시계열 길이가 짧기 때문에 더 긴 시계열의 Test 결과를 만들기 위해 이러한 방식을 사용했다.
<img width="1145" alt="스크린샷 2021-01-14 오후 12 44 15" src="https://user-images.githubusercontent.com/70478154/104542475-eeeb8f00-5666-11eb-8b0e-0dee65ecbb5e.png">

{% highlight ruby %}
# window sliding (train - (61, 50, 20, 15) / test - (61, 50))
def window_sliding(data, feature_list, s_step, step, n):
    
    # train data 1000개씩
    all_data = np.array(data[feature_list[1:n]])
    set_step = np.arange(((len(all_data) - s_step) / step) + 1, dtype = "int")
    
    x_per_set = []
    
    for i in set_step:
        each_set = all_data[20*i:(20*i) + s_step]
        x_per_set.append(each_set)
        
    # 20개씩 학습할 데이터 묶기
    all_set = []
    
    for j in range(0, len(x_per_set)):
        each_data = x_per_set[j]
        steps = np.arange(len(each_data) / step, dtype = "int")
        
        samps = []
        
        for k in steps:
            samp = each_data[20*k:(20*k)+step]
            samps.append(samp)
            
        all_set.append(samps)
    
    # target 설정
    all_target = np.array(data[feature_list[n]])
    target_step = np.arange(len(all_target) / step, dtype = "int")
    
    target_per_20 = []
    
    for l in target_step:
        t = all_target[(l*20) + 19]
        
        target_per_20.append(t)
        
    all_target_set = []
    
    for m in range(0, len(x_per_set)):
        each_target_set = target_per_20[m:m+50]
        
        all_target_set.append(each_target_set)
        
    # array 형태로 변경
    all_set = np.array(all_set)
    all_target_set = np.array(all_target_set)
    
    return all_set, all_target_set
{% endhighlight %}

{% highlight ruby %}
# 반복 학습 및 예측
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def train_test_result(x, y, model, data_name = ""):
    
    all_test_pred = []
    
    for routine in range(0, len(x)):
        
        # train, val, test 분리
        x_train, x_val = train_test_split(x[routine], test_size = 0.2, shuffle = False, random_state = 99)
        x_val, x_test = train_test_split(x_val, test_size = 0.1, shuffle = False, random_state = 99)
        
        y_train, y_val = train_test_split(y[routine], test_size = 0.2, shuffle = False, random_state = 99)
        y_val, y_test = train_test_split(y_val, test_size = 0.1, shuffle = False, random_state = 99)
        
        # early stoppping
        early_stopping = EarlyStopping(monitor = "val_loss", patience = 10)

        file = "{}_{}.ckpt".format(data_name, routine)
        folder = "{}".format(data_name)
        filename = os.path.join(folder, file)
        checkpoint = ModelCheckpoint(filename
                                    , save_weights_only = True
                                    , save_best_only = True
                                    , monitor = "val_loss"
                                    , verbose = 1)
        
        # 학습
        model.fit(x_train, y_train,
                  validation_data = (x_val, y_val),
                  batch_size = 20,
                  epochs = 400,
                  callbacks = [checkpoint, early_stopping],
                  verbose = 1)
            
        # 결과
        model.load_weights(filename)
        pred = model.predict(x_test)
        all_test_pred.append(pred[0][0])
        
    # actual data
    actual = []

    for n in range(0, len(y)):
        actual_data = y[n][-1]
        actual.append(actual_data)
        
    return all_test_pred, actual
{% endhighlight %}
<BR/><BR/>

#### 학습 결과

LSTM을 활용하여 변동성을 예측한 결과는 다음과 같다. 실제의 변동성 추이와 비교했을 때 두 자산 모두 좋은 예측 결과를 보인 것은 아니다. 하지만, 상대적으로 S&P500 ETF는 어느정도 비슷한 추이를 보였다. 더 적합한 LSTM 구조를 쌓고, 하이퍼 파라미터 튜닝을 더 진행한다면 지금보다 좋은 결과를 만들어 낼 수 있을 것이라 생각한다.
<img width="1144" alt="스크린샷 2021-01-14 오후 12 59 54" src="https://user-images.githubusercontent.com/70478154/104543228-6d94fc00-5668-11eb-84eb-aec3ca96b730.png">
<BR/><BR/>

#### 전략 구현

이렇게 예측된 두 자산의 변동성을 활용하여 LSTM Risk Parity 전략을 구현했다. RP_weight_1과 RP_weight_2는 각각 S&P500 ETF과 국채 ETF의 각 Rebalancing 별 비중이며, RP_pf_vol은 해당 비중으로 투자한 포트폴리오의 변동성이다. 자산 배분 비중 추이 그래프를 보면, 52번 째 전후의 Rebalancing 기간에 안정 자산인 국채 ETF의 비중이 높았고, 그 이후 S&P500 ETF의 비중이 잠깐 확 늘어나는 모습을 볼 수 있다. 각 기간은 코로나로 인해 주가가 급락하는 시기와 서서히 회복하는 시기이다. 이러한 시기적 특징을 LSTM Risk Parity 전략이 일부는 반영했다고 볼 수 있다.
<img width="1146" alt="스크린샷 2021-01-14 오후 1 02 51" src="https://user-images.githubusercontent.com/70478154/104543472-da0ffb00-5668-11eb-9313-4e03c1f3e960.png">

{% highlight ruby %}
# 두 자산 covariance
def cov_per_20(data_1, data_2):
    
    covariance = []
    
    for i in range(0, len(data_1) - 19):
        return_1 = data_1["return"][i:i+20]
        return_2 = data_2["return"][i:i+20]
        
        cov = np.cov(return_1, return_2)[0, 1]
        
        covariance.append(cov)

    cov_per_20_days = []
    
    for j in range(0, 110):
        cov_20 = covariance[j*20]
        
        cov_per_20_days.append(cov_20)
        
    # 첫 window sliding set 1000개에서 test set은 50번 째이기 때문에 61개 추출
    cov_per_20_days = cov_per_20_days[49:]
    
    return cov_per_20_days
{% endhighlight %}

{% highlight ruby %}
# risk parity p/f weights and p/f risk
def pf_weights_risk(vol_1, vol_2, cov, n):
    
    # weights
    w1 = ((vol_1 + vol_2) / vol_1) / (((vol_1 + vol_2) / vol_1) + ((vol_1 + vol_2) / vol_2))
    w2 = ((vol_1 + vol_2) / vol_2) / (((vol_1 + vol_2) / vol_1) + ((vol_1 + vol_2) / vol_2))
    
    # p/f risk
    corr = cov / (vol_1 * vol_2)
    mrc = 1 / n
    step_1 = (((w1*w1)*(vol_1*vol_1)) + (corr*w1*w2*vol_1*vol_2)) / mrc
    
    # 음수면 0으로 처리
    if step_1 < 0:
        pf_risk = 0
        
    else:
        pf_risk = step_1**0.5
    
    return w1, w2, pf_risk
{% endhighlight %}

{% highlight ruby %}
# LSTM risk parity 포트폴리오
def risk_parity_pf(data_1, data_2, data_1_pred, data_2_pred):
    df = pd.DataFrame()
    
    df_copy_1 = data_1.copy()
    df_copy_2 = data_2.copy()
    
    numbers = []
    
    for i in range(0, 61):
        num = 2199-(20*i)
        numbers.append(num)
    
    numbers.reverse()
    
    # date, historical_vol_1, historical_vol_2, covariance
    df["date"] = df_copy_1.iloc[numbers]["date"]
    df["pred_vol_1"] = data_1_pred
    df["pred_vol_2"] = data_2_pred
    df["cov"] = cov_per_20(data_1, data_2)
    df = df.reset_index(drop = True)
    
    # return_1, return_2 (monthly)
    df_copy_1 = df_copy_1.iloc[numbers].reset_index(drop = True)
    df_copy_1["close-1"] = df_copy_1["close"].shift(1)
    df_copy_1["monthly_return"] = (df_copy_1["close"] - df_copy_1["close-1"]) / df_copy_1["close-1"]
    df["monthly_return_1"] = df_copy_1["monthly_return"]
    
    df_copy_2 = df_copy_2.iloc[numbers].reset_index(drop = True)
    df_copy_2["close-1"] = df_copy_2["close"].shift(1)
    df_copy_2["monthly_return"] = (df_copy_2["close"] - df_copy_2["close-1"]) / df_copy_2["close-1"]
    df["monthly_return_2"] = df_copy_2["monthly_return"]
    
    # weight_1, weight_2, p/f_vol
    weight_1 = []
    weight_2 = []
    pf_vol = []
    
    for j in range(0, len(df)):
        w_1, w_2, pf_risk = pf_weights_risk(data_1_pred[j], data_2_pred[j], cov_per_20(data_1, data_2)[j], 2)
        
        weight_1.append(w_1)
        weight_2.append(w_2)
        pf_vol.append(pf_risk)
    
    df["RP_weight_1"] = weight_1
    df["RP_weight_2"] = weight_2
    df["RP_pf_vol"] = pf_vol
    
    # nan을 0으로 대체
    df = df.replace(np.nan, 0)
    
    return df
{% endhighlight %}
<BR/><BR/>

#### Benchmark 전략과 비교

LSTM Risk Parirt 전략을 기존의 Risk Parity 전략, 동일 가중 전략과 비교했다. Risk Parity의 레버리지 비용은 월 기준 0.25%(= 3%/12)로 설정했다. Risk Parity 전략 대비 동일 가중 전략의 변동성이 1보다 큰 경우에만 레버리지를 적용했다. 또한, Risk Parity 전략의 변동성이 0으로 산출되는 경우에는 레버리지를 적용하지 않았다. 각 전략 별 누적 수익률 그래프를 보면, LSTM Risk Parity 전략의 누적 수익률이 가장 높다는 것을 확인할 수 있다.
<img width="1144" alt="스크린샷 2021-01-14 오후 1 13 12" src="https://user-images.githubusercontent.com/70478154/104544082-4e976980-566a-11eb-97dd-75f42c2f8d2c.png">

{% highlight ruby %}
# 누적수익률 - leverage 추가
def cum_lev_return(pf, other_pf):
    new_asset = 1
    monthly_cum_return = []
    asset_value = []
    
    for i in range(0, len(pf) - 1):
        m_return_1 = pf["monthly_return_1"][i+1]
        m_return_2 = pf["monthly_return_2"][i+1]
        w_1 = pf.iloc[:,6][i]
        w_2 = pf.iloc[:,7][i]
            
        
        if pf.iloc[:,8][i] != 0:
        
            if other_pf.iloc[:,8][i] / pf.iloc[:,8][i] > 1:
                pf_return = (m_return_1 * w_1) + (m_return_2 * w_2)
                lev = other_pf.iloc[:,8][i] / pf.iloc[:,8][i]
                pf_lev_return = (pf_return * lev) - ((1 - lev) * 0.03 / 12)

            else:
                pf_lev_return = (m_return_1 * w_1) + (m_return_2 * w_2)
        
        else:
            pf_lev_return = (m_return_1 * w_1) + (m_return_2 * w_2)

        new_asset = new_asset * (1 + pf_lev_return)

        cum_return = (new_asset - 1) * 100
        monthly_cum_return.append(cum_return)
        asset_value.append(new_asset)
    
    return monthly_cum_return, asset_value
{% endhighlight %}
<BR/><BR/>

#### 성과 지표

각 포트폴리오 전략 별 성과 지표는 `누적 수익률` `CAGR(=연평균수익률)` `MDD(=최대낙폭)` `SR(=샤프비율)`로 산출하여 비교했다. LSTM Risk Parity 전략이 다른 전략들에 비해 모든 성과 지표에서 우수했다. 수익률 측면에서도 좋은 성과를 보였고, 최대 낙 폭이나 위험 대비 수익률 등 안정성 측면에서도 좋은 성과를 보였다.
<img width="1145" alt="스크린샷 2021-01-14 오후 1 22 45" src="https://user-images.githubusercontent.com/70478154/104544654-a08cbf00-566b-11eb-8dff-0b237503be26.png">

{% highlight ruby %}
# 투자 결과
def portfolio_report(df_pf, pf_value, pf_cum):
    pf_report = pd.DataFrame()
    
    # MDD
    arr_v = np.array(pf_value)
    peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
    peak_upper = np.argmax(arr_v[:peak_lower])
    mdd = (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper]
    
    # CAGR
    diff = df_pf["date"].iloc[-1] - df_pf["date"].iloc[0]
    N = diff.days / 365
    cum = (1 + (pf_cum[-1]*0.01))
    CAGR = (cum**(1/N)) - 1
    
    # Sharpe ratio (평균 수익률은 CAGR, 표준편차는 매년 표준편차의 평균치)
    m_data = []
    
    for i in range(0, len(pf_cum) - 1):
        m = ((1 + (pf_cum[i+1])*0.01) / (1 + (pf_cum[i])*0.01)) - 1
        m_data.append(m)
        
    rf = 0.03
    mean_vol = (np.std(m_data[:12]) + np.std(m_data[12:24]) + np.std(m_data[24:36]) + np.std(m_data[36:48]) + np.std(m_data[48:])) / 5
    SR = (CAGR - rf) / mean_vol
    
    pf_report["cum_return"] = [round(pf_cum[-1], 2)]
    pf_report["CAGR"] = round(CAGR * 100, 2)
    pf_report["MDD"] = round(mdd * 100, 2)
    pf_report["SR"] = round(SR, 2)
    
    return pf_report
{% endhighlight %}
<BR/><BR/><BR/><BR/>

## 결론

LSTM 모델을 활용하여 Risk Parity 전략을 구현한다면, 수익성과 안정성 등 다양한 성과 지표에서 강점을 가진 자산 배분이 가능할 것이라 생각한다. 다만, Overfitting과 모델의 성능 문제가 존재하며, 적은 데이터로 인한 문제 역시 존재한다. 따라서 Auto Encoder를 활용하여 노이즈를 제거하거나 세부적인 하이퍼 파라미터 최적화를 하는 등 모델의 성능 향상을 위한 노력이 수반되어야 할 것이다. 또한, Risk Parity는 레버리지를 사용하기 때문에 이를 위한 비용이 충분히 낮아야 한다. 이러한 점들이 종합적으로 충족된다면, 잘 구축된 LSTM Risk Parity는 좋은 자산 배분 전략이 될 수 있을 것이라 생각한다.
