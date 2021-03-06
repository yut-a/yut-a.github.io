---
layout: post
title:  Deep Learning 기초
date:   2021-01-15
image:  ai.jpg
tags:   Data
---
![map-2](https://user-images.githubusercontent.com/70478154/104743438-0d4caa00-578f-11eb-841d-c80239adca72.jpeg)

## NN 기초
<BR/>

#### `퍼셉트론`
신경망을 이루는 가장 기본 단위로, 퍼셉트론은 다수의 신호(INPUT)를 입력받아서 하나의 신호(OUTPUT)를 출력한다.
<BR/><BR/>

#### `Weight`
INPUT으로 받은 값을 각 뉴런에 전달하게 되는데 그 과정에서 가중합을 활용한다. 즉, 가중치를 통해 활성화함수의 기울기를 변화시켜 결과 값에 대한 영향력의 크기를 결정한다.
<BR/><BR/>

#### `Bias`
가중합을 통해 각 뉴런으로 전달할 때 bias를 추가하여 연산하게 된다. bias를 통해 활성화함수를 왼쪽, 오른쪽으로 이동시켜 결과 값에 대한 영향력의 크기를 결정한다.
<BR/><BR/>

#### `활성화 함수`
* `sigmoid`<BR/>
sigmoid는 0과 1 사이의 범위를 가지고 있으며, 이진 분류 문제에 적절하다. 그림에서 보는 것처럼 INPUT 값이 매우 크거나 작을 경우 기울기가 0에 가까워진다. 이러한 경우, 층이 깊을수록 작은 기울기 값들이 곱해지면서 기울기가 점점 작아지는 gradient vanishing 문제가 발생할 수 있다. 또한, 학습을 지그재그 형태로 만들어 학습이 느려지는 문제가 발생한다.<BR/>
<img width="479" alt="스크린샷 2021-01-15 오후 8 17 11" src="https://user-images.githubusercontent.com/70478154/104718854-e5e5e500-576e-11eb-9b00-4c608aff154f.png">

* `tanh`<BR/>
sigmoid와 유사한 형태이지만 -1과 1 사이의 범위를 가지고 있다. 함수의 중심값을 0으로 옮겨 sigmoid의 학습 과정이 느려지는 문제를 해결했다. 하지만, 여전히 gradient vanishing 문제를 해결하지는 못했다.<BR/>
<img width="481" alt="스크린샷 2021-01-15 오후 8 17 28" src="https://user-images.githubusercontent.com/70478154/104718879-f39b6a80-576e-11eb-8715-678c353160e9.png">

* `ReLU`<BR/>
음수값은 0으로, 양수값은 그 값 그대로 출력하는 함수이다. 양수값이면 기울기가 항상 1이기 때문에 gradient vanishing 문제가 발생하지 않는다. 다만, 음수값이라면 기울기가 0이 되어 소멸한다는 문제가 발생한다.<BR/>
<img width="487" alt="스크린샷 2021-01-15 오후 8 17 42" src="https://user-images.githubusercontent.com/70478154/104718886-f6965b00-576e-11eb-86ce-ca0d9fa802fd.png">

* `Leaky ReLU`<BR/>
음수값을 가질 때 기울기가 0이 되어 소멸하는 ReLU의 문제점을 완화하기 위한 활성화 함수이다. 음수인 경우, 0.01이거나 다른 작은 값을 갖도록 한다.<BR/>
<img width="482" alt="스크린샷 2021-01-15 오후 8 17 56" src="https://user-images.githubusercontent.com/70478154/104718915-03b34a00-576f-11eb-89a3-7ac035ee737a.png">

* `softmax`<BR/>
softmax는 다중 분류에서 많이 사용되는 함수이다. 가장 높은 확률값을 가지는 클래스를 출력한다.
<BR/>

#### `Layer`
* `입력층 (Input layer)`<BR/>
데이터를 입력받는 층이다. 연산이 이루어지지 않으며, 신경망의 층 수에 포함되지 않는다.

* `은닉층 (Hidden layer)`<BR/>
입력층과 출력층 사이에 있는 층으로, 우리 눈에 직접적으로 보이지 않기 때문에 붙여진 이름이다.

* `출력층 (Output layer)`<BR/>
다른 층으로부터 받은 값을 활성화 함수를 통해 결과값으로 산출하는 층이다.<br><br><br><br>

## 신경망 학습
<BR/>

#### `손실함수`
실제 값과 출력 값의 차이를 최소화할 수 있는 가중치를 학습하게 되는데, 그 차이를 측정해주는 함수가 손실함수이다.
* `MSE (Mean Squared Error)`<BR/>
회귀 문제를 풀고자 할 때 사용한다. 예측값과 실제 값의 차이에 대한 제곱을 평균으로 산출한 값이다.<BR/>
![CodeCogsEqn](https://user-images.githubusercontent.com/70478154/104723844-2fd0ca00-5773-11eb-9684-bd070426422f.gif)

* `MAE (Mean Absolute Error)`<BR/>
MSE와 마찬가지로 회귀 문제를 풀고자 할 때 사용하며, MSE와는 약간 다르게 제곱을 하지 않고 절대값에 대한 평균으로 산출한다.<BR/>
![CodeCogsEqn (1)](https://user-images.githubusercontent.com/70478154/104724195-af5e9900-5773-11eb-89bf-092df3731db1.gif)

* `Binary Crossentropy`<BR/>
이진 분류 문제를 풀고자 할 때 사용한다.<BR/>
![CodeCogsEqn (2)](https://user-images.githubusercontent.com/70478154/104724429-07959b00-5774-11eb-8152-1d70cdf3b486.gif)

* `Categorial Crossentropy`<BR/>
3개 이상의 분류 문제를 풀고자 할 때 사용한다. 특히, 라벨이 [1, 0, 0], [0, 1, 0], [0, 0, 1]과 같이 one-hot 형태일 때 사용한다.<BR/>
![CodeCogsEqn (3)](https://user-images.githubusercontent.com/70478154/104724622-46c3ec00-5774-11eb-8166-e080bad2b377.gif)

* `Sparse Categorical Crossentropy`<BR/>
3개 이상의 분류 문제를 풀고자 할 때 사용하며, 라벨이 정수 형태일 때 사용한다.<BR/>
<BR/>

#### `Batch`
Batch는 전체 데이터 셋을 여러 작은 그룹으로 나누는 것을 말하며, 하나의 소그룹 안에 속하는 데이터의 크기를 batch size라고 한다. 것Batch size가 크면 메모리의 한계와 속도 저하 문제가 발생할 수 있기 때문에 적절한 batch size 할당이 필요하다.
<BR/><BR/>

#### `Epoch`
Epoch는 전체 데이터 셋을 몇 번 사용해서 학습을 할인가를 말한다. Epoch이 너무 작으면 Underfitting, 너무 크다면 Overfitting 문제가 발생할 수 있다.
<BR/><BR/>

#### `역전파`
역전파는 역방향으로 오차를 전파시키면서 각 층의 가중치를 업데이트하여 최적의 학습 결과를 찾아가는 과정이다. 순전파를 통해 출력층에서 계산된 오차에 따른 각 가중치의 미세 변화를 입력층 방향으로 역전파시키면서 가중치를 업데이트하고, 업데이트한 가중치를 활용하여 새로운 오차를 계산한 후, 이를 다시 역전파시켜 가중치를 업데이트하는 과정을 반복한다.
<BR/><BR/>

#### `학습 규제`
과적합을 막기 위한 방법
* `Early stopping`<BR/>
학습 과정에서 가장 적절한 지점의 가중치를 지나서 더 업데이트를 지속할 수도 있다. 따라서, 이를 방지하기 위해 early stopping을 사용함으로써 과하게 가중치가 업데이트 되는 것을 막는다.

* `Weight decay`<BR/>
weight decay는 가중치를 감소시키는 방법이다. 가중치가 과도하게 커지면 과적합이 발생할 수 있기 때문에 큰 가중치를 만들지 못하게 함으로써 과적합을 방지할 수 있다.

* `Dropout`<BR/>
dropout은 설정한 확률로 연결을 강제로 끊어주는 역할을 한다. 연결 없이 결과를 예측하도록 함으로써 과적합을 방지할 수 있다.

* `Contraint`<BR/>
가중치의 크기를 물리적으로 제한하는 방법이다. 설정한 값보다 가중치가 더 큰 경우 임의의 값으로 변경하는 방법이다.

* `Learning rate`<BR/>
학습이 진행될수록 학습률을 감소시키는 방법이다.
<BR/>

#### `Transfer learning`
기존 데이터로 학습된 네트워크를 재사용 가능하도록 하는 라이브러리이다. 학습 데이터를 적게 사용할 수 있고, 학습 속도가 빠르며, 더 잘 일반화된 모델을 만들 수 있다는 장점이 있다.<BR/><BR/><BR/><BR/>

## CNN
<BR/>

#### `Convolution`
* `Filter`<BR/>
가중치의 집합으로 이루어져 데이터의 특징을 잡아낸다.

* `Stride`<BR/>
filter를 얼만큼씩 움직일 지 결정한다. Stride가 1이면 한 칸씩 움직이고, 2이면 두 칸씩 움직인다.

* `Padding`<BR/>
데이터 셋의 외곽에 0 또는 다른 값을 배치하는 방법이다. Padding을 하지 않을 경우, 데이터 셋의 각 모서리 부분은 다른 부분에 비해 사용되는 횟수가 줄어든다. 따라서 골고루 데이터를 이용하고자 사용하는 방법이다.
<BR/>

#### `Pooling`
pooling은 convolution layer와 activation을 거쳐 나온 값을 중에서 일부를 추출하는 것이다. pooling에는 다양한 종류가 있지만, 대표적으로 가장 큰 값을 추출하는 max pooling, 평균값을 추출하는 average pooling이 있다.<BR/><BR/><BR/><BR/>

## RNN
<BR/>

#### `RNN`
RNN은 기본적인 신경망에 재귀 연결을 두면서 시간 축 정보를 반영할 수 있도록 만든 구조이다. 자기 자신의 입력으로 돌아가는 Edge가 존재하여 시퀀스 데이터를 다루기에 적절하다.
<BR/><BR/>

#### `LSTM`
LSTM은 RNN에 Gate를 추가한 모델이다. RNN의 gradient vanishing 문제를 해결하기 위해 고안되었다. 3가지 gate인 Forget gate, input gate, output gate를 통해 최근 이벤트에 더 많은 비중을 둘 수도 있으면서 동시에 오래된 정보를 완전히 잃지 않을 수 있다는 장점을 가지고 있다.
<BR/><BR/>

#### `GRU`
LSTM은 많은 gate를 가지고 있기 때문에 이를 간소화한 모델이 GRU이다.
<BR/><BR/>

#### `Attention`
Atttention은 RNN 기반의 모델이 가지는 주요 문제점인 정보 손실과 gradient vanishing을 해결하고자 등장했다. Attention은 문장 소스의 길이가 긴 문장의 기억을 돕기 위해 만들어졌다. 핵심 아이디어는 기계 번역 모델이 출력단어를 예측할 때 특정 단어를 집중해서 본다는 것이다.<BR/><BR/><BR/><BR/>

## AE (Auto Encoder)
<BR/>

#### `AE`
Auto encoder는 데이터의 중요한 특성을 추출하여 출력값을 입력값처럼 만들 수 있도록 훈련하는 모델이다. Auto encoder는 encoder와 decoder로 구성되어 있다. 입력값이 들어와 encoder 부분에서 레이어를 거치며 데이터를 저차원으로 표현한다. 이 과정을 거치면 데이터는 가장 중요한 속성만을 가지게 된다. 이렇게 저차원으로 표현된 데이터를 다시 decoder 부분에서 확장시켜 입력값의 구조로 재구성하고 출력값으로 반환한다.<BR/><BR/><BR/><BR/>

## GAN
<BR/>

#### `GAN`
GAN은 generator와 discriminator가 서로 대립하여 훈련하면서, 입력된 데이터를 진짜와 유사하게 생성하는 모델이다. GAN은 generator와 discriminator로 이루어져 있다. generator에서 가짜 이미지를 생성하고 이를 discriminator로 넘겨주면, discriminator는 이미지의 진위 여부를 판단하게 된다. 이 과정이 반복되면서 generator는 점점 더 진짜같은 이미지를 만들게 되고, discriminator는 진짜와 가짜를 더 잘 구별하게 된다. 이렇게 학습하는 과정에서 generator와 discriminator는 가중치를 공유하여 loss를 최소화하게 되고 결국은 입력된 데이터에 대해 50대 50으로 진위여부를 판단하게 되는 균형상태에 이르게 되며, 진짜인지 가짜인지 구별하기 힘든 정도의 이미지를 생성할 수 있게 된다.
<BR/><BR/>

#### `CycleGAN`
CycleGAN은 원래의 형상은 유지하면서 일부 특성만 변환하는 모델이다. 예를 들어, 사진을 그림으로 바꾸거나 얼룩말을 말로 바꿔줄 수 있다. 만약 CycleGAN을 통해 얼룩말을 말로 바꾼다면, 바꿔준 말 이미지를 실제 말 이미지와 비교하여 진짜인지 가짜인지 판단하면서 얼룩말 이미지를 말 이미지로 잘 바꿀 수 있도록 학습한다. 하지만, 이런 과정에서 처음 얼룩말 이미지의 속성인 말 두마리와 포즈 등은 유지한 채로 바꿔야 하기 때문에 바꾼 말 이미지를 다시 얼룩말 이미지로 바꿔, 처음의 얼룩말 이미지와 비교한다. 이를 통해 처음 얼룩말 이미지의 특성이 그대로 유지됐는지 판단을 하게 되고, 이러한 과정을 거치면서 본 특성은 유지한 채 얼룩말을 말로 잘 바꿀 수 있도록 한다.
