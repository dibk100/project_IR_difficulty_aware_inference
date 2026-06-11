# phase02 : Layer-wise Difficulty Decodability
- (phase02 배경) difficulty 정보가 어디서(layer) 나타나는지 관찰이 필요함.
- (가설01) 쉬운 입력은 중간 레이어에서 충분히 정보(e.g. 빠르게 패턴이 생긴다)를 얻을 수 있고, 어려운 입력은 끝까지 가야 알 수 있다.
- (가설02) 초반 레이어는 문제 구조 파악을하고, 중간 레이어는 난이도/추론 경로/문제 복잡성을 다루고 마지막 레이어가 next token예측에 최적화되어있다.
- (방법) Layer별 hidden state → Difficulty Probe
- 참고 논문 : The Bottom-up Evolution of Representations in the Transformer(2019)

### Conclusion

1. Difficulty information emerges in early layers

세 모델 모두 초기 레이어에서 이미 난이도 정보가 선형적으로 디코딩 가능.
즉, 입력을 처리하는 초기 단계부터 표현 내부에 난이도 정보가 반영됨을 시사함.

| Model        | Early Layer    | ROC-AUC |
| ------------ | -------------- | ------- |
| Llama-3.1-8B | Layer 1 (Mean) | 0.709   |
| Qwen2.5-7B   | Layer 4 (Mean) | 0.697   |
| Phi-3.5-mini | Layer 4 (Last) | 0.672   |


2. intermediate layers

세 모델 모두 최종 레이어가 아닌 중간 또는 중후반 레이어에서 최대 ROC-AUC를 기록함.

| Model        | Best Layer | Representation | ROC-AUC |
| ------------ | ---------- | -------------- | ------- |
| Llama-3.1-8B | Layer 20   | Mean           | 0.768   |
| Qwen2.5-7B   | Layer 19   | Mean           | 0.766   |
| Phi-3.5-mini | Layer 29   | Last           | 0.772   |

3. 최종 Layer에서는 일부 약화됨

최종 레이어는 항상 최고의 난이도 분리 성능을 보이지 않음

| Model | Best ROC-AUC | Final Layer ROC-AUC | Gap    |
| ----- | ------------ | ------------------- | ------ |
| Llama | 0.768        | 0.689               | -0.079 |
| Qwen  | 0.766        | 0.697               | -0.069 |
| Phi   | 0.772        | 0.736               | -0.036 |

4. 모델마다 표현 형태가 다름.

| Model | Best Representation                 |
| ----- | ----------------------------------- |
| Llama | Mean Pooling                        |
| Qwen  | Mean Pooling (중반) / Last Token (후반) |
| Phi   | Last Token (후반)                     |

Llama는 전반적으로 Mean Pooling이 우세한 반면, Phi는 후반 레이어에서 Last Token 표현이 더 강한 분리력을 보임.

이는 난이도 정보가 모델마다 서로 다른 방식으로 분산 또는 집중되어 표현될 가능성을 시사함


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
