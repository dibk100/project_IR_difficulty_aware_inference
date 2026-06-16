# Phase03-A: Effective Rank 기반 Difficulty Signal 분석

## 연구 질문

> Transformer의 각 레이어 표현 공간(hidden representation)에서 Effective Rank가 입력 난이도(Easy/Hard)를 구분하는 유효한 Difficulty Signal이 될 수 있는가?

## 실험 설정

### 데이터셋

- GSM8K
- 500 samples
- Easy: 373
- Hard: 127

### 모델

- Llama-3.1-8B-Instruct

### Effective Rank 계산

각 샘플 \(x\), 레이어 \(l\)에 대해 다음 token hidden matrix를 구성하였다.

\[
H_l(x)\in\mathbb{R}^{T\times D}
\]

- \(T\): 입력 토큰 길이
- \(D\): hidden dimension

각 레이어의 token hidden matrix에 대해 다음 두 가지를 계산하였다.

- Raw Effective Rank
- Centered Effective Rank

본 실험에서는 Centered Effective Rank를 중심으로 분석하였다.

---

## 주요 결과

### Observation 1: Hard 샘플은 더 높은 Effective Rank를 가진다

모든 레이어에서 Hard 샘플이 Easy 샘플보다 더 높은 Effective Rank를 보였다.

\[
ER(Hard) > ER(Easy)
\]

가 거의 전 레이어에서 일관적으로 관찰되었다.

예시:

| Layer | Easy | Hard |
|---------|---------|---------|
| 1 | 71.41 | 81.75 |
| 16 | 26.79 | 33.10 |
| 32 | 90.85 | 102.07 |

---

### Observation 2: Hard-Easy Gap은 후반 레이어로 갈수록 증가한다

Hard-Easy Gap은 레이어가 깊어질수록 지속적으로 증가하였다.

| Layer | Hard-Easy Gap |
|---------|---------|
| 2 | 0.63 |
| 10 | 4.25 |
| 20 | 8.23 |
| 30 | 11.32 |
| 31 | 11.40 |

이는 후반 레이어에서 Hard 문제와 Easy 문제의 표현 구조 차이가 더욱 커짐을 의미한다.

---

### Observation 3: Effective Rank 단독으로도 난이도를 일정 수준 구분할 수 있다

Layer별 ROC-AUC 결과:

| Layer | ROC-AUC |
|---------|---------|
| 1 | 0.709 |
| 16 | 0.692 |
| 32 | 0.695 |

전체적으로

\[
AUC \approx 0.69 \sim 0.71
\]

수준을 유지하였다.

즉 Effective Rank만으로도 Easy/Hard를 일정 수준 구분할 수 있었다.

---

## Phase02 Difficulty Probe 결과와 비교

### 연구 질문

> Effective Rank Gap이 큰 레이어가 Difficulty Probe 성능도 높은가?

---

### Last-token Probe

- Pearson r = 0.223
- Spearman ρ = 0.094

### Mean-pooling Probe

- Pearson r = 0.305
- Spearman ρ = 0.080

---

### 결론

Effective Rank Gap과 Difficulty Probe ROC-AUC 사이의 상관관계는 매우 낮았다.

즉,

\[
Difficulty\ Decodability
\]

와

\[
Representation\ Complexity
\]

는 서로 다른 현상일 가능성이 높다.

---

## Length Control 분석

### Observation 4: Hard 샘플은 평균적으로 더 긴 입력을 가진다

| Metric | Easy | Hard |
|----------|----------|----------|
| Token Length | 106.4 | 120.6 |

차이:

\[
+14.1\ tokens
\]

효과 크기:

\[
Cohen's\ d = 0.705
\]

이는 상당히 큰 차이에 해당한다.

---

### Observation 5: 입력 길이만으로도 난이도를 구분할 수 있다

Token Length만 사용했을 때:

\[
ROC\text{-}AUC = 0.687
\]

이 나타났다.

즉 입력 길이 자체가 이미 강한 Difficulty Signal 역할을 하고 있었다.

---

### Observation 6: Effective Rank는 입력 길이와 매우 강하게 상관된다

Layer별 상관계수:

\[
corr(TokenLength, ER)
\]

\[
Pearson \approx 0.98 \sim 0.99
\]

즉 Effective Rank는 입력 길이와 거의 선형 관계 수준의 강한 상관관계를 보였다.

---

### Observation 7: Length 효과를 제거하면 Effective Rank 신호는 크게 약화된다

Length-Controlled Residual ER 분석 결과:

원래:

\[
AUC \approx 0.69 \sim 0.71
\]

↓

Length 제거 후:

\[
AUC \approx 0.50 \sim 0.61
\]

후반 레이어에서 약한 신호는 남아 있었지만, 원래의 분류력은 상당 부분 감소하였다.

---

## 종합 결론

### 확인된 사실

- Hard 문제는 Easy 문제보다 높은 Effective Rank를 가진다.
- Hard-Easy Gap은 후반 레이어로 갈수록 증가한다.
- Effective Rank만으로도 Easy/Hard를 일정 수준 구분할 수 있다.

### 한계

- Effective Rank는 입력 길이와 매우 강하게 결합되어 있다.
- 관찰된 Hard-Easy 차이의 상당 부분은 Token Length 효과로 설명된다.
- 따라서 Effective Rank를 순수한 Difficulty Signal로 해석하기 어렵다.

### 최종 판단

> Effective Rank는 Hard/Easy 간 표현 구조 차이를 보여주는 지표이다. 그러나 입력 길이에 대한 의존성이 매우 크며, 관찰된 신호의 상당 부분이 Length Effect에 의해 설명된다. 따라서 Effective Rank는 독립적인 Difficulty Signal이라기보다 Length-confounded Structural Signal로 해석하는 것이 적절하다.

---

## Phase03-A 요약

```text
Hard > Easy (전 레이어)

Hard-Easy Gap 증가 (후반 레이어)

ER 단독 ROC-AUC ≈ 0.69~0.71

Length 단독 ROC-AUC ≈ 0.687

corr(TokenLength, ER) ≈ 0.98~0.99

Length 제거 후 ROC-AUC ≈ 0.50~0.61

결론:
Effective Rank는 표현 구조 차이를 포착하지만,
독립적인 Difficulty Signal로 보기에는
입력 길이 의존성이 매우 크다.
```


## 📁 Folder Structure
```
phase03_signal_discovery/
├── README.md
├── output_llama_500/
└── effective_rank/
    ├── extract_token_hidden_matrices.py    ✅1
    ├── compute_effective_rank.py           ✅2
    ├── analyze_er_gap.py                   ✅3
    ├── length_control_analysis.py
    ├── evaluate_er_auc.py                  ✅6
    ├── compare_probe_auc.py                ✅5
    └── plot_er_results.py                  ✅4
    
```

> ompute_effective_rank.py

<!--
================================================================================
[Phase03-A] Compute Effective Rank
================================================================================
Input dir: output_llama_500/effective_rank/token_hidden_matrices
Output CSV: output_llama_500/effective_rank/sample_layer_effective_rank.csv
Number of sample files: 500
100%|█████████████████████████████████████████████████████████████████████████| 500/500 [03:55<00:00,  2.12it/s]
================================================================================
Done.
Saved: output_llama_500/effective_rank/sample_layer_effective_rank.csv
Shape: (16000, 9)
Columns: ['index', 'id', 'label', 'correct_count', 'layer', 'token_len', 'er_raw', 'er_centered', 'source_file']

   index id label  correct_count  layer  token_len     er_raw  er_centered        source_file
0      0  0  easy              3      1        115  78.839043    80.387253  sample_000000.npz
1      0  0  easy              3      2        115   4.734892     4.569815  sample_000000.npz
2      0  0  easy              3      3        115   7.050119     6.774529  sample_000000.npz
3      0  0  easy              3      4        115   9.741267     9.349012  sample_000000.npz
4      0  0  easy              3      5        115  12.454215    11.952125  sample_000000.npz
================================================================================
-->

> analyze_er_gap.py
- Layer별로 Easy와 Hard의 Effective Rank가 실제로 다른가?
- (input) sample_layer_effective_rank.csv
- (output) layerwise_er_gap.csv
- (해석01) Hard sample의 hidden representation은 Easy sample보다 더 높은 effective rank를 가진다.
- (해석02) Easy → 적은 방향 사용, Hard → 더 많은 방향 사용


<!--
Saved: output_llama_500/effective_rank/layerwise_er_gap.csv
Shape: (64, 14)

[Top positive gaps: er_centered]
    layer  easy_mean   hard_mean  gap_hard_minus_easy   cohen_d       welch_p
30     31  70.931366   82.329455            11.398089  0.730506  6.483711e-10
29     30  67.445747   78.764851            11.319104  0.730829  6.688533e-10
31     32  90.854858  102.065286            11.210428  0.730326  6.069224e-10
28     29  64.530328   75.691542            11.161214  0.731284  6.775254e-10
27     28  61.505825   72.442539            10.936715  0.730559  7.431696e-10
26     27  58.565966   69.288631            10.722664  0.731239  7.552536e-10
25     26  55.800067   66.310779            10.510711  0.730358  8.253109e-10
0       1  71.407838   81.753635            10.345796  0.773587  3.276018e-11
24     25  52.916175   63.140551            10.224376  0.728938  9.237989e-10
23     24  50.010568   59.930331             9.919762  0.727988  1.023729e-09

[Top negative gaps: er_centered]
    layer  easy_mean  hard_mean  gap_hard_minus_easy   cohen_d       welch_p
1       2   4.084497   4.715775             0.631278  0.737648  1.111490e-09
2       3   5.977826   7.086240             1.108415  0.731612  1.833068e-09
3       4   8.245452   9.924451             1.678999  0.726587  2.478913e-09
4       5  10.472608  12.717116             2.244508  0.727413  2.383653e-09
5       6  12.557841  15.285770             2.727930  0.716698  3.674623e-09
6       7  14.509011  17.691662             3.182651  0.710250  4.889866e-09
7       8  16.652845  20.338784             3.685939  0.706087  6.220709e-09
8       9  17.610687  21.564531             3.953845  0.706751  6.537988e-09
9      10  18.853966  23.106212             4.252247  0.706178  6.331360e-09
10     11  19.821248  24.355066             4.533818  0.707791  5.959798e-09
================================================================================
-->
