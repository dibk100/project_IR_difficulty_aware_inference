# phase01_difficulty_verification
## Goal
> 목표 : Llama-3-8B-Instruct에서 Easy/Hard 문제가 hidden representation 공간에서 분리되는지 확인하기   

본 단계의 목표는 LLM의 hidden representation 안에 difficulty 정보가 존재하는지 검증하는 것이다.

최근 연구 The LLM Already Knows는 입력 질문의 hidden representation만으로도 문제 난이도와 관련된 정보를 추정할 수 있음을 보여주었다.

본 실험에서는 해당 아이디어를 텍스트 기반 LLM 환경에서 검증하고, Easy/Hard 문제들이 hidden representation 공간에서 실제로 분리되는지 분석한다.

- 참고 논문 : The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations(EMNLP 2025)  

### 작업 요약
```
    Llama-3-8B 실행
    ↓
    hidden state 추출
    ↓
    Easy/Hard 생성
    ↓
    PCA / t-SNE
```

### 실험 세팅
- model : 
    - meta-llama/Llama-3.1-8B-Instruct
    - Qwen/Qwen2.5-7B-Instruct
    - microsoft/Phi-3.5-mini-instruct
    - mistralai/Mistral-7B-Instruct-v0.3
- dataset :
    - GSM8K
    - MATH-500 (optional)
- Difficulty Label :   
각 문제에 대해 3회의 독립적인 generation을 수행한다.
    - Easy : 3/3 정답
    - Hard : 그 외 모든 경우   
*해당 정의는 The LLM Already Knows 논문의 difficulty labeling 방식을 참고

### Method
Step 1. Difficulty Label Generation   
```
Question
    ↓
3 Independent Rollouts
    ↓
Correctness Evaluation
    ↓
Easy / Hard Label
```

Step 2. Hidden State Extraction

입력 질문만을 사용하여 hidden representation을 추출한다.

```
사용 정보:

Last Layer Hidden State
Last Input Token Representation
outputs = model(
    **inputs,
    output_hidden_states=True
)

last_hidden = outputs.hidden_states[-1]
representation = last_hidden[0, -1, :]
```

Step 3. Representation Visualization

Easy / Hard 문제의 hidden representation을 저차원 공간으로 투영하여 시각화한다.

Methods:
- PCA
- t-SNE

```
Hidden Representation
        ↓
PCA / t-SNE
        ↓
Easy vs Hard Visualization
```

### Success Criteria
- Minimum Success
    - Easy 문제와 Hard 문제가 representation space에서 부분적으로라도 분리되는 현상이 관찰된다.

- Expected Success
    - Difficulty label과 hidden representation 사이의 구조적 차이가 시각적으로 확인된다.

### Expected Output
- Visualization
    - PCA Plot
    - t-SNE Plot
- Dataset
    - Easy Question Set
    - Hard Question Set

### Representation
Hidden State Embeddings

### Relation to Future Phases

Phase 01은 hidden representation 안에 difficulty 정보가 존재하는지를 검증하는 단계이다.

만약 Easy/Hard 분리가 관찰된다면 다음 단계에서는 다음 질문을 탐구한다.

어떤 representation property가 이러한 difficulty 정보를 설명하는가?

이는 Phase 02 (Difficulty Signal Discovery)로 이어진다.

## 📁 Folder Structure
```
phase01_difficulty_verification/
├── README.md
├── requirements.txt
├── run_rollouts.py
├── extract_hidden_states.py linear_probe
├── visualize_embeddings.py
└── visualize_embeddings.py
```

> run_rollouts.py   
- 목적 : 문제를 실제로 풀게 해서 easy/hard 라벨을 만들게 함.   
- 방법 : 실험 세팅에 Difficulty Label에 내용 작성해둠.
- (model input) 프롬프트 템플릿 고정, gsm8k 질문
- (model output) 정답 text (후처리를 통해 답만 추출)
- (saved 형태) jsonl에 입력, 답변 생성, 정답 여부 판정, 라벨

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

<!--
실행 순서
python run_rollouts.py -> python extract_hidden_states.py -> python visualize_embeddings.py   

bash run_all.sh

NUM_SAMPLES = 100으로 돌리고, 잘 되면 300~500으로 늘리기
-->

## 📝 실험 기록
### LLAMA_gsm8k(Done)
- model : meta-llama/Llama-3.1-8B-Instruct
- dataset : gsm8k_main/socratic_test
    - sample 100, 300, 500
- 결과 : archived_llama 폴더
    - easy/hard는 불균형 없이 라벨이 균형적
    - 시각적으로 잘 구분이 되지 않음.
- Solution :
    1. representation 위치 문제
    2. 시각화 방법 문제
    3. 데이터셋의 문제(gsm8k)
    4. 모델의 문제(llama가 아닌 다른 모델 실험)
    5. 모델 크기 문제(7b가 아닌 더 큰/작은 모델로 실험)

### Test01 : Representation
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

### Test02 : 시각화
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
- (해석) Llama-3.1-8B-Instruct의 last-layer hidden representation 안에 Easy/Hard를 구분하는 정보가 약하게 존재한다라고 말할 수 있을거 같음. 근데 약함.
- (Next) 500개로 시도