# project_difficulty_aware_inference 🚀
본 레포지토리는 개인 연구 프로젝트 **project_difficulty_aware_inference**를 정리한 공간.   
**Type** : 개인 연구 프로젝트 (Independent Research)   
**Area** :
* Dynamic Inference
* Adaptive Computation
* Early Exit
* Representation Analysis
* LLM Inference Optimization

## Main Research Questions
(가설) 쉬운 질문, 입력은 적은 레이어만으로 충분히 표현되고, 어려운 입력에 대해 더 깊은 레이어 및 계산이 필요하다.

## Overview 📝
기존 Dynamic Inference 연구들은 주로 Entropy, Confidence 등의 출력 기반 신호를 사용하여 계산량을 조절한다.

최근 연구(The LLM Already Knows)는 hidden representation 안에 입력 난이도와 관련된 정보가 존재함을 보여주었다.

본 연구는 이러한 관찰을 바탕으로, hidden representation 내부의 구조적 특성(Representation Properties)이 난이도 정보를 설명할 수 있는지 분석한다.

구체적으로는 Effective Rank, Intrinsic Dimension, Representation Variance 등의 표현 공간 특성을 분석하고, 이러한 신호가 실제 계산 필요도(computational difficulty)를 설명할 수 있는지 검증한다.

최종적으로는 입력 난이도에 따라 계산량을 동적으로 조절하는 Difficulty-Aware Dynamic Inference Framework를 목표로 한다.

## 🧪 Research Roadmap(update.2026-06-10) 
### Phase 01 : Difficulty Information Verification(ing/Completed)

Research Question : Hidden Representation 안에 입력 난이도와 관련된 정보가 존재하는가?

#### Experiments

* Model : Llama-3-8B-Instruct(base), Qwen/Qwen2.5-7B-Instruct(additional)
* Dataset : GSM8K
    - main, test
    - sample 500
* Easy / Hard Difficulty Labeling
* Hidden State Extraction
* (Visualization) PCA ,t-SNE
* (Analysis) Linear Probe 

#### Findings
- PCA, t-SNE에서는 명확한 시각적 분리가 관찰되지 않음
- Linear Probe를 통해 Easy/Hard Difficulty Label과 관련된 정보가 Hidden Representation에 존재함을 확인

### Phase 02 : Layer-wise Difficulty Decodability
질문 : Difficulty 정보는 어느 레이어에서 형성되는가?

가설 : 초기 레이어에서 입력 난이도를 판단할 수 있다?   
- layerskip논문은 초기 레이어에서 다음 토큰 예측이 부정확하고 후반 레이어가 예측이 수렴한다고 함(비교 잘못됨. 이 논문은 생성쪽 초점)
- 내 연구는 다음 토큰 예측이 아닌 문제 난이도 예측(근거를 phae01에서 존재 자체를 확인함)
- The Bottom-up Evolution of Representations in the Transformer 논문 참고함

방법 : 각 레이어 hidden state에 대해 linear probe를 수행해서 측정   

### Phase 03 : Difficulty Signal Discovery
질문 : Difficulty 정보는 어떤 신호로 표현되는가?

목표 : Difficulty를 설명하는 Representation-level Signal 발견

후보 리스트 :
Effective Rank   
Intrinsic Dimension   
Entropy   
Margin   
Representation Complexity   

### Phase 04 : Difficulty Signal Discovery
질문 : 발견된 Difficulty Signal을 이용하여 레이어 사용량을 동적으로 조절할 수 있는가?


<!--
project_difficulty_aware_inference/

├── README.md

├── configs/
│
├── datasets/
│
├── signals/
│   ├── entropy.py
│   ├── margin.py
│   ├── effective_rank.py
│   └── intrinsic_dimension.py
│
├── utils/
│
├── outputs/
│
├── docs/
│   ├── roadmap.md
│   ├── stage01_notes.md
│   ├── stage02_notes.md
│   └── stage03_notes.md
│
├── stage01_signal_discovery/
│   ├── calm/
│   ├── layerskip/
│   ├── collect_signals.py
│   ├── analyze_signals.py
│   └── visualize_signals.py
│
├── stage02_signal_controller/
│   ├── entropy_controller.py
│   ├── rank_controller.py
│   ├── id_controller.py
│   └── evaluate_controller.py
│
└── stage03_generalization/
    ├── llama/
    ├── qwen/
    ├── mistral/
    └── cross_model_analysis.py
-->