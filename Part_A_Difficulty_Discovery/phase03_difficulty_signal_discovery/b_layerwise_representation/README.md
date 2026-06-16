# Phase03-B: Layer-wise Representation Change Analysis

(가설)   
Easy sample:
레이어를 거치며 representation이 빠르게 안정화된다.
→ layer-to-layer delta가 작거나 빠르게 감소한다.

Hard sample:
레이어를 거치며 representation이 계속 수정된다.
→ layer-to-layer delta가 더 크거나 후반까지 유지된다.

## 📁 Folder Structure
```
phase03_signal_discovery/
├── README.md
├── output_llama_500/
└── b_layerwise_representation/
    ├── compute_layerwise_delta.py  ✅ 1
    ├── analyze_delta_gap.py        ✅ 2
    ├── evaluate_delta_auc.py       ✅ 3 -- 어느정도 결과 확인
    ├── compare_probe_auc.py        ✅ 6
    ├── plot_delta_results.py       ✅ 4
    ├── length_control_analysis     ✅ 5
    └── README.md

✅
```



# 가설

Hard 샘플은 정답을 찾기 위해 더 많은 추론 과정을 필요로 한다.

따라서 Transformer 내부에서는

- Layer-to-Layer Representation Change가 더 크고
- 특히 Difficulty 정보가 형성되는 중간 레이어에서 차이가 크게 나타날 것이다.

---

# 데이터

모델

- Llama-3.1-8B-Instruct

데이터셋

- GSM8K

샘플 수

- 500

Difficulty Label

- Easy = 3/3 정답
- Hard = 나머지

분포

- Easy = 373
- Hard = 127

---

# 실험 설계

## Representation Change

각 레이어 l 에 대해

H_l
H_(l+1)

사이의 변화량을 계산한다.

---

### 1. L2 Distance

Δ_l = ||H_(l+1) - H_l||_2

---

### 2. Normalized L2 Distance

Δ_l = ||H_(l+1)-H_l||_2 / ||H_l||_2

---

### 3. Cosine Distance

Δ_l = 1 - cosine(H_l, H_(l+1))

---

Aggregation

- last-token representation
- mean-pooled representation

모두 실험

---

# 실험 절차

## Step 1

compute_layerwise_delta.py

각 샘플에 대해

- layer-to-layer delta 계산

산출물

sample_layer_delta.csv

---

## Step 2

analyze_delta_gap.py

Easy / Hard 평균 비교

산출물

layerwise_delta_gap.csv

---

## Step 3

evaluate_delta_auc.py

각 레이어의 단일 delta 값만 사용하여

Easy vs Hard 분류 ROC-AUC 계산

산출물

delta_signal_auc.csv

---

## Step 4

plot_delta_results.py

시각화

- Easy / Hard Delta Curve
- Delta Gap Curve
- Delta Signal ROC-AUC Curve

---

## Step 5

length_control_analysis.py

Token Length 영향 제거

검증 내용

- Length ↔ Delta 상관관계
- Delta Signal AUC
- Length 제거 후 Residual AUC

---

# 주요 결과

## Result 1

Hard 샘플은 중간 레이어에서 더 큰 Representation Change를 보임

특히

Layer 13~20

구간에서

Hard > Easy

패턴이 일관적으로 나타남.

---

## Result 2

Difficulty Signal로서의 성능

Best Delta Signal

(mean + normalized L2)

Layer 17

ROC-AUC ≈ 0.710

---

Cosine Distance도 유사

Layer 17

ROC-AUC ≈ 0.706

---

즉

Representation Change만으로도

약 0.71 수준의 Difficulty Prediction 가능.

---

## Result 3

Phase02와의 정렬

Phase02 결과

Difficulty Probe Peak

Layer 12~22

ROC-AUC ≈ 0.76~0.77

---

Phase03-B 결과

Representation Change Peak

Layer 14~20

ROC-AUC ≈ 0.69~0.71

---

두 결과의 Peak Layer가 거의 일치.

즉

Difficulty 정보가 가장 잘 형성되는 레이어와

Representation Update가 가장 활발한 레이어가 동일함.

---

## Result 4

Length 영향 존재

대표적으로

mean aggregation

Length ↔ Delta 상관계수

r ≈ 0.95~0.99

---

즉

Representation Change는 입력 길이의 영향을 강하게 받음.

---

## Result 5

Length 제거 후에도 신호 유지

Length Regression 제거 후

Residual AUC

최대

≈ 0.62

---

즉

Representation Change는

단순한 Length Signal이 아님.

Length로 설명되지 않는 Difficulty 관련 정보가 존재함.

---

# 해석

Hard 문제는

중간 레이어에서 더 많은 표현 수정 과정을 거친다.

이는

- 더 복잡한 추론 수행
- 더 많은 정보 통합
- 더 긴 reasoning trajectory

와 관련될 가능성이 있다.

---

또한

Difficulty Decodability Peak

(Phase02)

와

Representation Change Peak

(Phase03-B)

가 정렬된다는 사실은

Difficulty 정보 형성과 Representation Evolution 사이의 연결 가능성을 시사한다.

---

# 결론

## Phase03-B 결론

Layer-wise Representation Change는 Effective Rank보다 더 강력한 Difficulty Signal 후보이다.

주요 발견은 다음과 같다.

1. Hard 샘플은 중간 레이어에서 더 큰 Representation Update를 수행한다.

2. Representation Change Peak는 Phase02 Difficulty Peak와 일치한다.

3. Delta Signal만으로 ROC-AUC ≈ 0.71 수준의 Difficulty 예측이 가능하다.

4. 입력 길이의 영향을 강하게 받지만, Length 제거 후에도 Difficulty 정보가 남아있다.

5. 따라서 Representation Change는 Transformer 내부 Difficulty Signal 후보로서 연구 가치가 있다.

---

# 현재 Phase03 진행 상황

- Phase03-A Effective Rank
  - 완료
  - Length 의존성 매우 강함
  - Difficulty Signal 후보로는 약함

- Phase03-B Representation Change
  - 완료
  - 가장 유망한 Difficulty Signal 후보