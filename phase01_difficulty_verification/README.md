# phase01_difficulty_verification
### Goal
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
- model : meta-llama/Llama-3.1-8B-Instruct
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

## Relation to Future Phases

Phase 01은 hidden representation 안에 difficulty 정보가 존재하는지를 검증하는 단계이다.

만약 Easy/Hard 분리가 관찰된다면 다음 단계에서는 다음 질문을 탐구한다.

어떤 representation property가 이러한 difficulty 정보를 설명하는가?

이는 Phase 02 (Difficulty Signal Discovery)로 이어진다.

# 📁 Folder Structure
```
phase01_difficulty_verification/
├── README.md
├── requirements.txt
├── run_rollouts.py
├── extract_hidden_states.py
└── visualize_embeddings.py
```

실행 순서
python run_rollouts.py -> python extract_hidden_states.py -> python visualize_embeddings.py   
NUM_SAMPLES = 100으로 돌리고, 잘 되면 300~500으로 늘리기