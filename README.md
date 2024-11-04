# dl-learn
numpy로 딥러닝 배우기

이 노트북은 책 &lt;Deep Learning from scratch&gt; by Saito Goki.를 참고하여 제작되었습니다\
모든 코드는 더 나은 이해를 위해 책의 예제에서 상당부분 수정하였으며 좋은 책을 선사해준 O'relly에게 감사드립니다.

<big>
<big>
<big>

[노트북 열기](./main.ipynb)

</big>
</big>
</big>

## Chapters.
* 퍼셉트론
* 신경망
  * 활성화 함수
  * 시그모이드 함수
  * ReLU 함수
  * 행렬의 점곱
  * 행렬의 점곱을 통한 퍼셉트론
  * 신경망 구현 정리
  * 소프트맥스 함수
  * MNIST 손글씨 데이터셋의 대한 신경망 구현
  * One-hot 인코딩
* 신경망 학습
  * 딥러닝 신경망이란?
  * 훈련 데이터와 시험 데이터
  * 손실 함수
    * 평균 제곱 오차
    * 교차 엔트로피 오차
    * 배칙밧을 위한 교차 엔트로피 오차
  * 미니배치
  * 정답률이 아니라 손실율을 쓰는 이유?
  * 함수의 대한 미분 (=특정 지점 기울기 구하기)
  * 경사하강법
  * 엑조디아
* 오차역전파법 (역전파)
  * 계산 그래프
  * 곱셈 계층의 역전파
  * 덧셈 계층의 역전파
  * 활성화 함수들의 역전파
  * 퍼셉트론 계층의 역전파
  * 소프트맥스-크로스 엔트로피 손실 함수 계층의 역전파
  * Better 엑조디아
* 최적화 알고리즘들 (Optimizers)
  * Momentum Optimizer
  * AdaGrad
  * Adam
* 가중치 초기값 설정
  * 정규분포를 활용한 초깃값
  * Xavier 초깃값
  * He 초기값
  * 배치 정규화 알고리즘
* 오버핏 방지
  * 가종치 감소
  * Dropout
* CNN (합성곱 신경망, Convolutional Neural Network)
  * 합성곱 레이어
  * Pooling 레이어

## Rights.
* 원서 &lt;Deep Learning from scratch&gt; by Saito Goki.의 여러 예제가 일부분 포함되어 있습니다.
* 이외 제가 작성한 부분은 &copy; Minhyeok Park. 2024. All rights reserved. 로 보호됩니다.
