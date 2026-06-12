# phase01 : difficulty_verification

- 참고 논문 : The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations(EMNLP 2025)  

**Main Question**

> 입력 질문을 처리한 직후의 Hidden Representation은 토큰 생성 전에 모델의 문제 해결 난이도(model-perceived difficulty)를 예고할 수 있는가?

**Control Question**

> 해당 예고 신호는 단순히 입력 길이(token length, word length, character length)에 의해 설명되는가?

**Goal**

본 단계의 목표는 모델이 응답을 생성하기 전에 형성한 Hidden Representation 안에, 이후 생성 결과의 성공/실패 가능성과 관련된 정보가 존재하는지 검증하는 것이다. Easy/Hard 라벨 분류는 최종 목적이 아니라, model-perceived difficulty를 관찰하기 위한 operational proxy로 사용한다.


<!--
실행 순서
python run_rollouts.py -> python extract_hidden_states.py -> python visualize_embeddings.py -> linear_probe.py

bash run_all.sh

NUM_SAMPLES = 100으로 돌리고, 잘 되면 300~500으로 늘리기
-->


### 실험 세팅
- model : 
    - (base) meta-llama/Llama-3.1-8B-Instruct
    - (additional) Qwen/Qwen2.5-7B-Instruct
    - (additional) microsoft/Phi-3.5-mini-instruct
    - (additional) Qwen/Qwen2.5-14B-Instruct
    - (optional) mistralai/Mistral-7B-Instruct-v0.3
- dataset :
    - (base) GSM8K (main, test) → 수학 추론 난이도
    - (optional) MMLU-Pro → 일반/전문 지식 + 추론 난이도
- Difficulty Label :   
*해당 정의는 The LLM Already Knows 논문의 difficulty labeling 방식을 참고   
각 문제에 대해 3회의 독립적인 generation을 수행한다.
    - Easy : 3/3 정답
    - Hard : 그 외 모든 경우  

    

### Representation
- Last Layer Hidden Representation
- Last Token Representation
- Mean Pooling Representation


## Main Result

| Model                 | Hidden Dim | Easy | Hard | Representation |            Accuracy |             ROC-AUC |            Macro-F1 |
| --------------------- | ---------: | ---: | ---: | -------------- | ------------------: | ------------------: | ------------------: |
| Llama-3.1-8B-Instruct |       4096 |  373 |  127 | Last Token     |     0.7100 ± 0.0190 |     0.6889 ± 0.0275 |     0.6162 ± 0.0267 |
| Llama-3.1-8B-Instruct |       4096 |  373 |  127 | Mean Pooling   |     0.7400 ± 0.0346 | **0.7304 ± 0.0299** | **0.6418 ± 0.0627** |
| Qwen2.5-7B-Instruct   |       3584 |  410 |   90 | Last Token     | **0.8020 ± 0.0075** | **0.6974 ± 0.0618** | **0.6430 ± 0.0438** |
| Qwen2.5-7B-Instruct   |       3584 |  410 |   90 | Mean Pooling   |     0.7760 ± 0.0326 |     0.6774 ± 0.0803 |     0.5805 ± 0.0812 |
| Phi-3.5-mini-instruct |       3072 |  357 |  143 | Last Token     |     0.7100 ± 0.0363 | **0.7360 ± 0.0536** | **0.6413 ± 0.0402** |
| Phi-3.5-mini-instruct |       3072 |  357 |  143 | Mean Pooling   |     0.7180 ± 0.0392 |     0.6975 ± 0.0425 |     0.6403 ± 0.0407 |

## Conclusion

| Model                 | Best Representation | Best ROC-AUC |
| --------------------- | ------------------- | -----------: |
| Llama-3.1-8B-Instruct | Mean Pooling        |       0.7304 |
| Qwen2.5-7B-Instruct   | Last Token          |       0.6974 |
| Phi-3.5-mini-instruct | Last Token          |   **0.7360** |

1. PCA / t-SNE 시각화 : 명확한 Easy/Hard 분리 관찰되지 않음
2. Linear Probe : 모든 모델에서 ROC-AUC ≈ 0.69~0.74
3. 결론 :
- Hidden Representation 안에 Difficulty 관련 정보가 존재한다고 볼 수 있다.
- 해당 정보는 선형적으로 디코딩 가능하고, 특정 모델에만 나타나는 현상은 아니다.

<!--
Hidden Representation 안의 정보를 복잡한 모델 없이, 단순한 선형 함수만으로도 Easy/Hard 라벨로 어느 정도 읽어낼 수 있다.

hidden representation을 h라고 할 때 Linear Probe는 대략 : score=wTh+b
h = LLM hidden representation
w = linear probe가 학습한 가중치
b = bias
score = Easy/Hard를 구분하기 위한 점수

즉 Linear Probe는 hidden representation을 복잡하게 변형하지 않고, 4096차원 벡터의 각 방향에 가중치를 곱해서 더하는 방식
복잡한 MLP나 Transformer를 붙여야만 구분되는 게 아니라, 단순한 직선/초평면 하나로도 Easy와 Hard가 어느 정도 구분된다.
-->

## 📁 Folder Structure
```
phase01_difficulty_verification/
├── data/
├── dataEDA/
├── output_llama_500/
├── output_qwen_500/
├── output_phi_500/
├── run_rollouts.py
├── extract_hidden_states.py 
├── visualize_embeddings.py
├── linear_probe.py
├── length_probe.py
└── README.md
```

<!--
📁 파일 구조
```
output_llama_500/
├── gsm8k_main_rollouts.jsonl       (1.5MB) ← Step1: 라벨 생성 결과 (원본)
├── gsm8k_main_hidden_states.npz    (17.6MB) ← Step2: hidden state
├── pca_easy_hard_last_token.png        ┐
├── pca_easy_hard_mean_pooling.png      │ Step3: 시각화
├── tsne_easy_hard_last_token.png       │ (PCA/t-SNE × last/mean)
├── tsne_easy_hard_mean_pooling.png     ┘
├── run_all.log + 01~04 단계별 로그   (Step4 linear_probe 포함)
```

🧬 gsm8k_main_rollouts.jsonl — 라벨 생성 레코드 (500줄)
키	내용
id, question, gold, parsed_gold	문제·정답
correct_count	3회 중 정답 횟수
label	easy(3/3) / hard
rollouts	3개 생성 결과 (각: rollout_id, output, parsed_pred, parsed_gold, correct)

🧬 gsm8k_main_hidden_states.npz — 임베딩 (5개 배열)
키	shape	내용
embeddings_last	(500, 4096)	마지막 레이어, 마지막 토큰 representation
embeddings_mean	(500, 4096)	마지막 레이어, mean-pooling
labels	(500,)	easy/hard
ids	(500,)	문제 id
questions	(500,)	질문 텍스트
라벨 분포: easy 373 / hard 127

phase01 vs phase02 차이
phase01 (output_llama_500)	phase02 (output_llama_500)
레이어 범위	마지막 레이어만 (embeddings_last/mean)	32개 전 레이어 (layer01~32_last/mean)
npz 크기	17.6MB	525MB
rollouts 포함	✅ (라벨 생성 원본)	❌ (phase01 것을 입력으로 사용)
산출물	PCA/t-SNE 그림 + linear probe	레이어별 AUC 곡선

-->


> run_rollouts.py   
- 목적 : 문제를 실제로 풀게 해서 easy/hard 라벨을 만들게 함.   
- 방법 : 실험 세팅에 Difficulty Label에 내용 작성해둠.
- (model input) 프롬프트 템플릿 고정, gsm8k 질문
- (model output) 정답 text (후처리를 통해 답만 추출)
- (fin output 형태 jsonl) id, question(입력), 

> extract_hidden_states.py
- 목적 : 이미 만들어진 easy/hard라벨을 가져와서 각 질문의 hidden representation을 추출함.
- run_rollouts.py의 흐름처럼 모델에 질문을 넣지만, 답변을 생성하지 않게 함.
- text답변 대신에 "hidden_state"를 반환해서 마지막 레이어의 마지막 토큰 벡터를 추출하여 npz로 저장.
- 자세히 풀어서 설명하자면, 모델이 입력 prompt를 읽은 뒤, 다음 토큰을 예측할 수 있는 내부 상태까지 계산하고 실제로 다음 토큰을 생성하지 않게까지만 함.
    - 주어진 입력을 한 번 forward pass 해서 내부 hidden state만 반환하는 것
    - 시스템적으로 비효율적임. 하지만 실험구분을 위해 이렇게 진행하겠음

> visualize_embeddings.py   
- 목적 : hidden representation은 dim이 큼(4096-d). 그래서 시각적으로 빠르게 보기 위해 PCA와 t-SNE로 압축하여 보려고 함.
- PCA : 분산이 큰 방향을 찾아서 압축(분산 보존이 강함)
- t-SNE : 원래 공간에서 가까운 점들이 2차원 공간에서도 가깝게 보이도록 함.
- (결론) 가벼운 시각화로 구분되길 바랬지만, 명확하게 구분되지 않음. Logistic Regression으로 확인해야함

> linear_probe.py
- 목적 : PCA/t-SNE처럼 2차원 시각화에 의존하지 않고, 고차원 hidden representation 자체에 Easy/Hard를 구분할 수 있는 정보가 존재하는지 정량적으로 확인함.
- 방법 : hidden representation을 입력으로 하고 Easy/Hard label을 정답으로 하여 Logistic Regression 분류기를 학습함.
- 왜 Linear Probe인가? :
    - 복잡한 모델을 사용하면 분류 성능이 좋아져도 hidden representation 자체에 정보가 있는지, 분류기가 새롭게 복잡한 패턴을 학습한 것인지 구분하기 어려움.
    - 따라서 가장 단순한 선형 분류기를 사용하여, Easy/Hard 정보가 hidden representation 안에서 선형적으로 디코딩 가능한지 확인함.
- 평가 지표 :
    - Accuracy : 전체 문제 중 얼마나 맞췄는지
    - Macro-F1 : 클래스 불균형 보완 지표
    - ROC-AUC : 
- (결론) PCA/t-SNE에서는 명확한 시각적 분리가 관찰되지 않았지만, Linear Probe에서는 ROC-AUC 약 0.69~0.73 수준으로 Easy/Hard label을 예측할 수 있었음.
- (해석) Difficulty 관련 정보는 2차원 시각화에서는 뚜렷하게 드러나지 않지만, hidden representation 안에 선형적으로 디코딩 가능한 형태로 존재함.

<!--
ROC-AUC :
무작위로 Hard 하나와 Easy 하나를 뽑았을 때,Hard가 더 높은 점수를 받을 확률

예시 :
Hard 문제 하나, Easy 문제 하나를 랩덤으로 뽑았을 때, 73%확률로 hard가 더 높은 difficulty score를 받을 확률?
Hidden Representation으로부터 추출한 선형 score는 Easy 문제보다 Hard 문제를 약 73% 확률로 더 높게 평가

sample 1 data가 Logistic Regression 내부로 들어가서 (z=wtx+b)의 값을 sigmoid씌워서 확률을 얻으면 
문제 A → Hard 확률 0.96
문제 B → Hard 확률 0.86
문제 C → Hard 확률 0.38
이런식.

Accurancy는 hard확률 기준(0.5)로 hard와 easy를 나눔

ROC-AUC는 기준(0.5)가 아니라 더 높은 점수르 받는지 관점

ROC = threshold를 0~1까지 바꿔가며 분류 성능 변화를 그린 곡선
AUC = 그 ROC 곡선 아래 면적

ROC-AUC = 모든 threshold에서의 구분 능력을 하나의 숫자로 요약한 값

-->

> length_probe.py
- 목적 : Easy/Hard 라벨이 hidden representation이 아니라 단순히 입력 길이에 의해 구분되는지 확인하기 위한 sanity check
- 방법 : char_len, word_len, token_len만 사용하여 Logistic Regression으로 Easy/Hard를 예측
- 비교 : Length-only ROC-AUC와 Hidden Representation 기반 Linear Probe ROC-AUC를 비교
- 해석 :
    - Length-only 성능이 hidden representation과 비슷하면, difficulty 정보가 단순 길이 효과일 가능성이 있음
    - Length-only 성능이 낮고 hidden representation 성능이 더 높으면, hidden representation이 길이 이상의 difficulty 정보를 포함한다고 해석 가능


## 📝 실험 기록
### LLAMA_gsm8k
- model : meta-llama/Llama-3.1-8B-Instruct
- dataset : gsm8k_main/socratic_test
    - sample 100, 300, 500
- 결과 : archived_llama 폴더
    - easy/hard는 불균형 없이 라벨이 균형적
    - 시각적으로 잘 구분이 되지 않음.
- Solution :
    1. (Done)representation 위치 문제(Done)
    2. (Success)시각화 방법 문제
    3. (PASS)데이터셋의 문제(gsm8k)
    4. (Done)모델의 문제(llama가 아닌 다른 모델 실험)
    5. (Done)모델 크기 문제(7b가 아닌 더 큰/작은 모델로 실험)

### Solution01 : Representation
- model : Qwen/Qwen2.5-7B-Instruct
- dataset : gsm8k_main/socratic_test
    - sample 300

#### 01. hidden representation 추출 설정
- (원인 추정) llama instruct 모델에서는 chat template를 쓰면 마지막 토큰은 실제 질문 마직막 토큰이 아닌 assistant generation prompt 토큰일 가능성이 있다고 함.
- (해결 방법) extract_hidden_states.py의 add_generation_prompt=False로 바꿈(해당 스크립트에 이유 작성함)
- (결과) 그럼에도 불구하고 easy/hard가 구분되지 않음

#### 02. hidden representation 추출 방법
- (원인 추정) 현재는 **마지막 토큰 하나의 hidden state**를 뽑아서 진행함. 마지막 토큰이 입력문장 전체를 의미하지 못하는 것일 수 있어서 hidden state의 mean pooling을 시도해보기
- (이유) 입력된 토큰 시퀀스 전체(문장)의 hidden state를 보기 위해 평균
- (정리) 입력 문제를 대표하는 벡터를 마지막 토큰의 hidden vs 전체의 hidden으로 구분해서 easy/hard가 구분되는지 봐야함
- (결과)
    - last token: 시각적 분리 안 됨
    - mean pooling: 시각적 분리 안 됨

### Solution02 : 시각화
- model : Qwen/Qwen2.5-7B-Instruct
- dataset : gsm8k_main/socratic_test
    - sample 300
- (가설) 정보가 있지만 2D시각화에서 안보임. 즉 비선형적으로만 존재하고 현재 데이터셋과 모델로는 약함. 아니면 라벨 정의가 너무 거침

#### 01.Linear_probe
- (how) logistic regression으로 easy/hard 예측해서 ROC-AUC, Macro-F1 측정해보기
- (판단 기준) 
    - ROC-AUC ≈ 0.50 : 선형적으로 구분 가능한 정보 거의 없음
- (결과)
    ```
    [last_token]
    Embedding shape: (300, 4096)
    Easy: 122
    Hard: 178
    accuracy: 0.6300 ± 0.0756
    roc_auc: 0.6799 ± 0.0699
    macro_f1: 0.6217 ± 0.0765

    [mean_pooling]
    Embedding shape: (300, 4096)
    Easy: 122
    Hard: 178
    accuracy: 0.5967 ± 0.0306
    roc_auc: 0.6083 ± 0.0259
    macro_f1: 0.5880 ± 0.0310
    ```
- (해석) Llama-3.1-8B-Instruct의 last-layer hidden representation 안에 Easy/Hard를 구분하는 정보가 약하게 존재하는 것으로 보인다. 
다만 ROC-AUC ≈ 0.68 수준으로, 정보의 존재는 확인되었지만 강한 분리 신호라고 보기는 어렵다.

- (Next) 500개로 시도

### Solution03 : 코드 오류
- (배경) 실험 정리하다가 발견한 큰 이슈.
- (원인) run_rollouts.py에서 정답을 추출하는 re파싱(정규화, 문자열 비교) 때문에 조금이라도 틀리면 false로 함.
- (해결) 해결함. 하지만 pca, t-sen 시각화 변화는 없음. liner probing 수치는 높아짐.

---

### ISSUE : 
- (배경) phase03실험 중, Hard/Easy 라벨의 구분이 입력질문의 길이 및 토큰수에 의존적일 수 있다는 것을 발견함.
- (방법) inspect_labels_llama_500.ipynb, length_probe.py를 통해 실제로 의존적임을 확인함.
- (질문) Difficulty Information은 1) 단순 입력 길이 때문일까, 2) hidden representation은 길이를 넘는 정보를 담고 있는가
- (시도)
    1. Length-matched subset (Failed)
    2. Residualization (PASS) : Hidden representation에서 length 정보를 제거한 뒤 probe
    3. ohter dataset (TRY)

#### phase01-RE_A : Length-matched Difficulty Verification
> length_matched_probe.py

- (목적) Easy/Hard 분류 성능이 Hidden Representation 자체에서 오는 것인지, 아니면 입력 길이(token length) 효과 때문인지 확인.
- (방법) Easy와 Hard를 비슷한 token length 구간에서 매칭하고 비교
    - Hard 문제 1개 token_len=70, Easy 문제 중 token_len 65~75에서 1개 샘플링(Tolerance = ±5 tokens)
- (원하는 결과) Length-matched 상태에서 Hidden ROC-AUC > 0.60이어야 길이 이상의 정보를 담는다고 할 수 있음
- (세팅)
    - Llama-3.1-8B-Instruct
    - GSM8K test 500 samples
    - Matched subset 구성 : 평균 토큰 길이(119)
        - Easy 124 개
        - Hard 124 개
- (비교 실험)
    - Token Length Only
    - Last-token Hidden Representation
    - Mean-pooled Hidden Representation
- (결과)

    | Feature             | ROC-AUC |
    | ------------------- | ------: |
    | Token Length Only   |  0.4117 |
    | Last-token Hidden   |  0.5797 |
    | Mean Pooling Hidden |  0.5338 |

    - 기존 GSM8K 실험에서는 입력 길이가 Easy/Hard와 강하게 연관되어 있었음.
    - Length matching 이후 Token Length 기반 분류 성능은 무너짐.
    - Hidden Representation의 분류 성능도 크게 감소함. 다만 Last-token Hidden Representation은 ROC-AUC 0.58 수준으로 랜덤(0.5)보다 약간 높은 성능을 유지함.
    - 따라서 기존 Difficulty Signal의 상당 부분은 입력 길이와 관련되어 있었을 가능성이 높음.

- (결론) 입력 길이는 GSM8K Easy/Hard 라벨과 강하게 연관되어 있었으며, 기존 Phase01 결과의 상당 부분을 설명할 수 있었다.
Hidden Representation 내부에 Difficulty 관련 정보가 일부 존재할 가능성은 있으나 신호는 약함.
**다른 데이터셋**에서 동일 현상이 나타나는지 추가 검증 필요.

<details>
<summary>상세정보 : 실험 결과 수치</summary>

```
[Original length stats]
easy {'n': 373, 'mean': 106.42, 'median': 104.0, 'std': 18.99, 'min': 74, 'max': 184}
hard {'n': 127, 'mean': 120.56, 'median': 117.0, 'std': 22.76, 'min': 76, 'max': 188}

[Matched length stats]
easy {'n': 124, 'mean': 119.19, 'median': 117.0, 'std': 21.34, 'min': 75, 'max': 184}
hard {'n': 124, 'mean': 119.31, 'median': 117.0, 'std': 21.55, 'min': 76, 'max': 188}

[length_matched_token_len_only]
Feature shape: (248, 1)
Easy: 124
Hard: 124
accuracy: 0.4395 ± 0.0426
roc_auc: 0.4117 ± 0.0461
macro_f1: 0.4361 ± 0.0441

[length_matched_hidden_last_token]
Feature shape: (248, 4096)
Easy: 124
Hard: 124
accuracy: 0.5365 ± 0.0628
roc_auc: 0.5797 ± 0.0962
macro_f1: 0.5323 ± 0.0642

[length_matched_hidden_mean_pooling]
Feature shape: (248, 4096)
Easy: 124
Hard: 124
accuracy: 0.5164 ± 0.0547
roc_auc: 0.5338 ± 0.0431
macro_f1: 0.5113 ± 0.0520
```

</details>

### 3. 다른 데이터셋

1. StrategyQA
2. CommonsenseQA
3. MMLU bench - text