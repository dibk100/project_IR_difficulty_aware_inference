# phase03 : Difficulty Signal Discovery

질문 : 어떤 Signal이 Difficulty 정보를 가장 잘 설명하는가?

Hidden representation 속 difficulty information을 가장 잘 설명하는 signal은 무엇인가?

주의 : 
- Hard와 Easy는 각 signal을 평가하기 위한 기준(Ground Truth Label)
- 라벨 분류기 개발이 아닌 Signal을 발견하는 것이 목표

## 📁 Folder Structure
```
phase03_difficulty_signal_discovery/
├── README.md
├── output_llama_500/           ### 동일 모델/데이터셋 결과 저장
│
├── a_effective_rank/
│   ├── compute_effective_rank.py
│   ├── analyze_er_gap.py
│   ├── length_control_analysis.py
│   ├── compare_probe_auc.py
│   └── plot_er_results.py
│
├── norm/
│   ├── compute_norm.py
│   ├── analyze_norm_gap.py
│   └── plot_norm_results.py
│
├── token_variance/
│   ├── compute_token_variance.py
│   ├── analyze_variance_gap.py
│   └── plot_variance_results.py
│
└── intrinsic_dimension/
    ├── compute_intrinsic_dimension.py
    ├── analyze_id_gap.py
    └── plot_id_results.py
```

결과
```
output_llama_500/
├── effective_rank/
│   ├── sample_layer_effective_rank.csv
│   ├── layerwise_er_gap.csv
│   └── figures/
│
├── norm/
│   ├── sample_layer_norm.csv
│   ├── layerwise_norm_gap.csv
│   └── figures/
│
├── token_variance/
│   ├── sample_layer_token_variance.csv
│   ├── layerwise_variance_gap.csv
│   └── figures/
│
├── intrinsic_dimension/
│   ├── sample_layer_intrinsic_dimension.csv
│   ├── layerwise_id_gap.csv
│   └── figures/
│
└── summary/
    ├── signal_comparison.csv
    └── figures/
```


### 큰 스케치 

```
Hidden Representation
↓
Signal 추출
↓
Easy/Hard 비교
↓
Signal 유효성 검증
```

### Difficulty Signal 후보
- Entropy, Attention Entropy
- Logit Margin
- Hidden State Norm
- Effective Rank
- Intrinsic Dimension
- Layer-wise Variance

## 📝 실험 기록
- Phase03-A : Effective Rank(2026.06.11)
- Phase03-B : Representation Norm
- Phase03-C : Token-wise Variance
- Phase03-D : Intrinsic Dimension

### 세팅
- Model: Llama-3.1-8B-Instruct
- Dataset: GSM8K
    - Samples: 500개
- Difficulty label: Phase01의 Easy / Hard


### Phase03-A. Effective Rank as a Difficulty Signal

목표 : Hard/Easy 입력은 layer-wise hidden representation의 spectral structure에서 차이를 보이는지

가설1) Difficulty가 증가하면 representation structure가 변한다.

가설2) 그 변화는 Effective Rank로 측정 가능하다.

- 질문 : Transformer의 각 레이어 표현 공간(hidden representation)에서 Effective Rank가 입력 난이도(Easy/Hard)를 구분하는 유효한 Difficulty Signal이 될 수 있는가?

- 결과 : 조짐


### Phase03-C. Representation Norm
- 질문 : 각 레이어 hidden representation의 norm이 Easy/Hard를 구분하는 신호가 될 수 있는가?

- 결과 :

    - Easy/Hard 간 norm 차이는 존재
    - 중간 레이어에서 Easy > Hard 경향 관찰
    - 최고 AUC ≈ 0.586

- 결론 :

    - Representation Norm은 difficulty signal로서 일관된 경향은 보이나, 분류력은 부족하다.
    - Effective Rank보다 성능이 낮으며,controller 신호로 사용하기에는 근거가 약하다.

<!--
1. compute_norm_signal.py
2. analyze_norm_gap.py
3. evaluate_norm_auc.py
4. length_control_analysis.py
5. compare_probe_auc.py
6. plot_norm_results.py
```
phase03_signal_discovery/
└── c_representation_norm/
    ├── compute_norm_signal.py
    ├── analyze_norm_gap.py
    ├── evaluate_norm_auc.py
    ├── length_control_analysis.py
    ├── compare_probe_auc.py
    └── plot_norm_results.p

output_phi_1000/
└── representation_norm/
    ├── sample_layer_norm.csv
    ├── layerwise_norm_gap.csv
    ├── norm_signal_auc.csv
    ├── length_control_norm.csv
    ├── compare_probe/
    └── figures/
```
last_norm        = ||last-token hidden||
mean_norm        = ||mean-pooled hidden||
token_norm_mean  = mean_t ||h_t||


Hidden norm

∣∣h
l
	

∣∣

Layer-to-layer distance

∣∣h
l
	

−h
l−1
	

∣∣

Cosine similarity

cos(h
l
	

,h
l−1
	

)
Entropy
Logit margin
-->

### Phase03-D: Intrinsic Dimension 기반 Representation Geometry 분석

phase03_signal_discovery/
└── d_intrinsic_dimension/
    ├── compute_layerwise_id.py
    ├── analyze_id_profile.py
    ├── compare_probe_auc.py
    ├── evaluate_id_auc.py
    ├── length_control_analysis.py
    ├── plot_id_results.py
    └── README.md

output_phi_1000/
└── intrinsic_dimension/
    ├── layerwise_id_profile.csv
    ├── sample_layer_id.csv
    ├── layerwise_id_gap.csv
    ├── id_signal_auc.csv
    ├── length_control_id.csv
    ├── compare_probe/
    └── figures/

1. compute_layerwise_id.py
   - layer별 전체/easy/hard ID 계산

2. analyze_id_profile.py
   - ID profile의 peak/minimum/transition 확인

3. compare_probe_auc.py
   - Phase02 layerwise_probe_results.csv와 ID profile 비교

3-2. compute_samplewise_local_id.py

4. evaluate_id_auc.py
   - ID 자체가 signal로 쓰일 수 있는지 확인
   - 단, group-wise ID는 sample별 AUC가 아니라 layer-level 분석 중심

5. plot_id_results.py
   - ID profile, Probe AUC overlap plot



1. compute_samplewise_local_id.py
2. evaluate_id_auc.py
3. length/metadata sanity check
4. plot_id_results.py
<!--
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

-->