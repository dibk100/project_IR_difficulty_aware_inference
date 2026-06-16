# phase04 : 

(가정) Difficulty 정보는 scalar geometry signal로는 약하지만, hidden representation의 특정 linear subspace에는 존재한다.

Layer l hidden representation h_l
↓
Linear probe 학습
↓
weight vector w_l 추출
↓
difficulty score = w_l^T h_l
↓
Easy/Hard AUC 확인


### 큰 스케치

```
phase04_difficulty_subspace/
├── README.md
├── output_phi_1000/
└── a_probe_direction/
    ├── train_probe_extract_direction.py
    ├── project_difficulty_score.py             # 모델 내부 분석
    ├── compare_models_projection.py            # Phi vs Llama 비교용 코드
    └── README.md
```

> compare_models_projection.py

```
두 모델의 projection_score_analysis.csv를 읽고 아래들을 표/그림으로 정리하는 역할
best layer
best AUC
best aggregation
AUC curve 비교
```

### 모델별 최고 성능

| Model | Aggregation | Best Layer | Projection AUC |
| ----- | ----------- | ---------- | -------------- |
| Phi   | last        | 27         | **0.712**      |
| Phi   | mean        | 18         | **0.709**      |
| Llama | mean        | 17         | **0.674**      |
| Llama | last        | 31         | **0.647**      |

| Model      | Aggregation | Cohen's d |
| ---------- | ----------- | --------- |
| Phi-last   | 0.786       |           |
| Phi-mean   | 0.739       |           |
| Llama-mean | 0.592       |           |
| Llama-last | 0.458       |           |

### 결론

난이도는 1개의 scalar로 표현되는 게 아니라(phase03 실험들) representation 내부의 특정 방향(direction)에 인코딩되어 있다.
