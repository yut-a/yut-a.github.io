---
layout: post
title:  Gradient Boosting을 활용한 Kospi200 주가 방향 예측
date:   2020-11-01
image:  computer.png
tags:   Data Finance
---
## Gradient Boosting이란?

**Gradient Boosting** 모델 역시 트리 앙상블 모델 중 하나지만, Random Forest와는 차이가 있다. `Boosting` 기법은 weak learner를 결합하여 strong learner를 만드는 방식이다. 정확도가 낮더라도 일단 모델을 만든 후, 다음 모델들이 약점을 보완하는 방식으로 예측이 이루어진다. 그 중, Gradient Boosting은 잔차가 큰 데이터를 더 학습하도록 함으로써, 손실 함수를 최소화한다. 여기서 사용할 `XGBoost`는 Extreme Gradient Boosting이라고 부르며, Gradient Boosting 알고리즘 중 하나이다. 이 모델의 장점은 학습과 예측이 빠르고, 과적합을 규제할 수 있다. 하지만, 하이퍼 파라미터에 굉장히 민감하다는 단점이 존재한다.<BR/><BR/><BR/><BR/>

## 분석 목적

해외 증시, 주요 경제, 금융 지표들을 바탕으로 Kospi200 **주가의 방향을 얼마나 효과적으로 예측**할 수 있는지 알아보고자 한다. 또한, 주가 방향 예측에 **가장 영향을 미치는 요소**들을 파악하고자 한다. 더 나아가, 이러한 지표들이 주가 방향 예측에 **가장 크게 영향을 줄 수 있는 시기**를 `일주일` `한달` `3개월` `6개월`로 나누어 알아보고자 한다.

마지막으로, 이러한 결과들을 [이전 포스트](https://yut-a.github.io/2020/10/25/RandomForest-Kospi200/)의 RandomForest 모델과 비교해보고자 한다.<BR/><BR/><BR/><BR/>
