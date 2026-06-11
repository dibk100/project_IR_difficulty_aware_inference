# phase03 : Difficulty Signal Discovery
*정의) signal : Difficulty를 반영하는 특정 측정값

질문 : 어떤 Signal이 Difficulty 정보를 가장 잘 설명하는가?

### Difficulty Signal 후보
- Entropy, Attention Entropy
- Logit Margin
- Hidden State Norm
- Effective Rank
- Intrinsic Dimension
- Layer-wise Variance

### 실험 설계 및 가정
- Branch A. Effective Rank
    - 어려운 문제일수록 representation이 더 많은 차원을 사용하는지?
    - 측정
    ```
Hidden State
→ SVD
→ Singular Value Spectrum
→ Effective Rank
    ```
    - 구현이 쉽고 , 논문이 많고, 해석 가능하지만, sample 수에 민갑하거나 token aggregation설계가 필요할 것이라고 함

- Branch B. Intrinsic Dimension
    - 어려운 문제일수록 representation manifold가 더 복잡한가?
    - TwoNN, MLE ID, FisherS
    - ettective rank보다 진짜 차원? 계산량 크고 노이즈에 민감하다고 함

- Branch C. Entropy
    - 모델이 어려운 입력에서 더 불확실한가?
    - 측정
    ```
Logits
→ Softmax
→ Entropy
    ```
    - 구현이 쉽고 , 논문이 많고, 해석 가능하지만, 새로움 부족

- Branch D. Margin
    - Top1과 Top2의 차이가 난이도를 반영하는가?
    - CALM의 confidence 계열과 연결, entropy와 비슷

- Branch E. Representation Complexity
    - Hidden Representation 자체의 복잡도가 난이도를 반영하는가?
    - 후보 :
    ```
Norm

Variance

Participation Ratio

Layer-wise Change

Trajectory Length

CKA Distance
    ```
    - 이게 맞나

### 스케치중
entropy/margin을 베이스라인(CAML)
effective rank, intrinsic demension, representation complexity


### 상세 문제
신호를 어떤 단위에서 계산할지..
- 마지막 토큰 hidden state
- 전체 토큰 평균
- 문제 토큰만?
- 레이어별 전체 hidden state


### 일단 스케치
1. 데이터셋 고정 (GSM8K)
2. Difficulty 라벨 고정 (Phase01 방식)
3. Layer 고정 (예: 마지막 레이어)
4. 마지막 토큰 기준으로 먼저 측정


```
입력:
Phase02에서 저장한 layerwise hidden states

후보 신호:
1. Hidden State Norm
2. Representation Variance
3. Effective Rank
4. Logit Entropy
5. Logit Margin

평가:
각 layer, 각 signal에 대해
Easy vs Hard 구분력 측정

출력:
signal, layer, aggregation, ROC-AUC

```