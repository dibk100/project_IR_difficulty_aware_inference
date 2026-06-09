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

## 🧪 Research Roadmap(update.2026-06-09) 
### Phase 01 : Difficulty Information Verification

목표 : Hidden representation 안에 difficulty 정보가 실제로 존재하는지 검증

### Experiments

* Llama-3-8B-Instruct
* GSM8K
* MATH-500
* Easy / Hard Difficulty Labeling
* Hidden State Extraction
* PCA Visualization
* t-SNE Visualization

### Research Question

> Easy 문제와 Hard 문제는 hidden representation 공간에서 분리되는가?


### Phase 02 : Difficulty Signal Discovery

목표 : 어떤 representation property가 difficulty를 설명하는지 탐색

### Experiments

* Entropy
* Margin
* Hidden State Norm
* Representation Variance
* Effective Rank
* Intrinsic Dimension

### Research Question

> 어떤 representation property가 difficulty와 가장 강한 상관관계를 가지는가?

### Phase 03 : Signal Generalization

목표 : 발견된 difficulty signal의 일반화 가능성 검증

### Experiments

* Llama-3-8B-Instruct
* Qwen2.5-7B-Instruct
* Additional Open LLMs

### Research Question

> 발견된 difficulty signal은 모델이 바뀌어도 유지되는가?

### Phase 04 : Difficulty-Aware Dynamic Inference

목표 : Difficulty Signal 기반 동적 계산 제어

### Experiments

* Early Exit
* Layer Skipping
* Adaptive Depth
* Adaptive Compute Allocation
* CALM-style Baseline
* LayerSkip-style Baseline

### Research Question

> Difficulty Signal을 이용하여 추론 효율을 향상시킬 수 있는가?


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