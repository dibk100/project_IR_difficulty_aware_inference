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
* 입력/토큰 난이도를 반영하는 내부 신호가 존재하는가?
* hidden representation 안에는 difficulty 정보가 존재하는가?
* 어떤 representation property가 난이도를 설명하는가?
* 그 신호를 이용해 계산량을 동적으로 할당할 수 있는가?
* 이러한 신호는 모델이 바뀌어도 유지되는가?

## Overview 📝
기존 Dynamic Inference 연구들은 주로 Entropy, Confidence 등의 출력 기반 신호를 사용하여 계산량을 조절한다.

최근 연구(The LLM Already Knows)는 hidden representation 안에 입력 난이도와 관련된 정보가 존재함을 보여주었다.

본 연구는 이러한 관찰을 바탕으로, hidden representation 내부의 구조적 특성(Representation Properties)이 난이도 정보를 설명할 수 있는지 분석한다.

구체적으로는 Effective Rank, Intrinsic Dimension, Representation Variance 등의 표현 공간 특성을 분석하고, 이러한 신호가 실제 계산 필요도(computational difficulty)를 설명할 수 있는지 검증한다.

최종적으로는 입력 난이도에 따라 계산량을 동적으로 조절하는 Difficulty-Aware Dynamic Inference Framework를 목표로 한다.

## 🧪 Research Roadmap(update.2026-06-10) 
### Phase 01 : Difficulty Information Verification(ing/Completed)

Research Question : Hidden Representation 안에 입력 난이도와 관련된 정보가 존재하는가?

### Experiments

* Model : Llama-3-8B-Instruct(base), Qwen/Qwen2.5-7B-Instruct(additional)
* Dataset : GSM8K
    - main, test
    - sample 500
* Easy / Hard Difficulty Labeling
* Hidden State Extraction
* (Visualization) PCA ,t-SNE
* (Analysis) Linear Probe 

### Findings
- PCA, t-SNE에서는 명확한 시각적 분리가 관찰되지 않음
- Linear Probe를 통해 Easy/Hard Difficulty Label과 관련된 정보가 Hidden Representation에 존재함을 확인

### Phase 02 : Difficulty Signal Discovery
#### A. Difficulty Signal Localization
질문 : Difficulty 정보는 어느 레이어에서 형성되는가?

가설 : 초기 레이어에서 입력 난이도를 판단할 수 있다?   
- layerskip논문은 초기 레이어에서 다음 토큰 예측이 부정확하고 후반 레이어가 예측이 수렴한다고 함
- 내 연구는 다음 토큰 예측이 아닌 문제 난이도 예측(근거를 phae01에서 존재 자체를 확인함)

방법 : 각 레이어 hidden state에 대해 linear probe를 수행해서 측정   

# 📁 Folder Structure

```text
project_difficulty_aware_inference/

├── phase01_difficulty_verification/
│   ├── data/
│   ├── extraction/
│   ├── visualization/
│   └── analysis/
│
├── phase02_signal_discovery/
│   ├── entropy/
│   ├── margin/
│   ├── norm/
│   ├── effective_rank/
│   └── intrinsic_dimension/
│
├── phase03_generalization/
│   ├── llama/
│   ├── qwen/
│   └── comparison/
│
├── phase04_dynamic_inference/
│   ├── early_exit/
│   ├── layer_skipping/
│   └── adaptive_controller/
│
└── papers/
```

---

## Current Status

### Active Phase

**Phase 01 : Difficulty Information Verification**

### Current Goal

> Llama-3-8B-Instruct에서 Easy/Hard 문제가 hidden representation 공간에서 분리되는지 확인한다.

## 📁 Folder Structure

* stage01_signal_discovery/
* stage02_signal_controller/
* stage03_generalization/



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