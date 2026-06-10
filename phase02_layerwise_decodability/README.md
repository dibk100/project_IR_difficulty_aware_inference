# phase02 : Layer-wise Difficulty Decodability
- (phase02 배경) 신호 탐색 전에 difficulty 정보가 어디서(layer) 나타나는지 관찰이 필요함.
- (가설) 쉬운 입력은 중간 레이어에서 충분히 정보(e.g. 빠르게 패턴이 생긴다)를 얻을 수 있고, 어려운 입력은 끝까지 가야 알 수 있다.
- 참고 논문 : The Bottom-up Evolution of Representations in the Transformer(2019)

## 📁 Folder Structure
```
phase02_layerwise_decodability/
├── extract_layerwise_hidden_states.py
├── layerwise_probe.py
├── plot_layer_auc.py
└── README.md
```
*phase01은 마지막 레이어에서 difficulty 존재 여부를 봤고
*phase02는 모든 레이어에서 Difficulty 형성 위치 확인하기

### 큰 스케치
```
Layer 1
Layer 2
...
Layer 32

각 레이어 hidden state
↓
Difficulty Probe
↓
ROC-AUC 곡선
```

### expexted output
```
x축: Layer index
y축: ROC-AUC
곡선 1: Last-token hidden state
곡선 2: Mean-pooled hidden state
```

### 실험설계
- Model: Llama-3.1-8B-Instruct
- Dataset: GSM8K
    - Samples: 500개
- Difficulty label: Phase01의 Easy / Hard