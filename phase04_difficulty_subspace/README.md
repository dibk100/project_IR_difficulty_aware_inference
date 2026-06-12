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
    ├── project_difficulty_score.py
    ├── analyze_direction_stability.py
    ├── compare_with_phase03_signals.py
    └── plot_subspace_results.py
```
