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
* 그 신호는 왜 난이도를 반영하는가?
* 그 신호를 이용해 계산량을 동적으로 할당할 수 있는가?
* 이러한 신호는 모델이 바뀌어도 유지되는가?

## Overview 📝

기존 Dynamic Inference 연구들은 주로 Entropy, Confidence 등의 출력 기반 신호를 사용하여 계산량을 조절한다.

본 연구는 LLM 내부 표현(hidden representation)에 존재하는 난이도 신호를 탐색하고, 해당 신호가 계산 필요도(computational difficulty)를 설명할 수 있는지를 분석한다.

최종적으로는 이러한 신호를 활용하여 입력 난이도에 따라 계산량을 동적으로 할당하는 Difficulty-Aware Inference를 목표로 한다.

## 🧪 Research Roadmap(update.2026-06-09) 
### Stage 01 : Signal Discovery

* CALM 재현
* LayerSkip 재현
* Entropy 분석
* Margin 분석
* Effective Rank 분석
* Intrinsic Dimension 분석

Q. 연구 질문
> 어떤 내부 신호가 early-exit difficulty를 가장 잘 설명하는가?

### Stage 02 : Signal-based Controller

* 신호 기반 계산 제어
* 기존 Entropy 기반 방식과 비교(기존 방식 entropy말고 추가)

Q. 연구 질문
> Entropy 대신 Effective Rank를 사용하면 어떠한가?

### Stage 03 : Generalization

* 모델 간 일반화 분석
* 신호의 일관성 검증

Q. 연구 질문
> 이 신호는 모델이 바뀌어도 유지되는가?


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