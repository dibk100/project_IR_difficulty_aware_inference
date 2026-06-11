# phase01 : difficulty_verification

본 단계의 목표는 LLM의 hidden representation 안에 difficulty 정보가 존재하는지 검증하는 것이다.

최근 연구 The LLM Already Knows는 입력 질문의 hidden representation만으로도 문제 난이도와 관련된 정보를 추정할 수 있음을 보여주었다.

본 실험에서는 해당 아이디어를 텍스트 기반 LLM 환경에서 검증하고, Easy/Hard 문제들이 hidden representation 공간에서 실제로 분리되는지 분석한다.

- 참고 논문 : The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations(EMNLP 2025)  

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
    - (optional) mistralai/Mistral-7B-Instruct-v0.3
- dataset :
    - (base) GSM8K (main, test) → 수학 추론 난이도
    - (optional) MATH-500 
    - (additional) MMLU-Pro → 일반/전문 지식 + 추론 난이도
- Difficulty Label :   
각 문제에 대해 3회의 독립적인 generation을 수행한다.
    - Easy : 3/3 정답
    - Hard : 그 외 모든 경우  

    *해당 정의는 The LLM Already Knows 논문의 difficulty labeling 방식을 참고

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
├── README.md
├── data
├── output_llama_500
├── output_qwen_500
├── run_rollouts.py
├── extract_hidden_states.py 
├── visualize_embeddings.py
└── linear_probe.py
```

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
    4. (PASS)모델의 문제(llama가 아닌 다른 모델 실험)
    5. (PASS)모델 크기 문제(7b가 아닌 더 큰/작은 모델로 실험)

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


