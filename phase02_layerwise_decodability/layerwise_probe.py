# python layerwise_probe.py

import csv
import re
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


INPUT_PATH = "gsm8k_layerwise_hidden_states.npz"
OUTPUT_CSV = "layerwise_probe_results.csv"

RANDOM_STATE = 42
N_SPLITS = 5


def labels_to_binary(labels):
    return np.array([1 if label == "easy" else 0 for label in labels])


def evaluate_probe(embeddings, y):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "macro_f1": "f1_macro",
    }

    results = cross_validate(
        clf,
        embeddings,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    summary = {}
    for metric in scoring.keys():
        scores = results[f"test_{metric}"]
        summary[f"{metric}_mean"] = scores.mean()
        summary[f"{metric}_std"] = scores.std()

    return summary


def get_layer_indices(data_keys):
    layer_indices = []

    for key in data_keys:
        match = re.match(r"layer(\d+)_last", key)
        if match:
            layer_indices.append(int(match.group(1)))

    return sorted(layer_indices)


def main():
    data = np.load(INPUT_PATH, allow_pickle=True)

    labels = data["labels"]
    y = labels_to_binary(labels)

    print("Loaded:", INPUT_PATH)
    print("Labels shape:", labels.shape)
    print("Easy:", np.sum(y == 1))
    print("Hard:", np.sum(y == 0))

    layer_indices = get_layer_indices(data.files)

    print("Detected layers:", layer_indices)

    rows = []

    for layer_idx in layer_indices:
        for aggregation in ["last", "mean"]:
            key = f"layer{layer_idx:02d}_{aggregation}"

            if key not in data:
                print(f"Skip missing key: {key}")
                continue

            embeddings = data[key]

            print(f"\nLayer {layer_idx:02d} | {aggregation}")
            print("Embedding shape:", embeddings.shape)

            result = evaluate_probe(embeddings, y)

            row = {
                "layer": layer_idx,
                "aggregation": aggregation,
                **result,
            }

            rows.append(row)

            print(
                f"accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f} | "
                f"roc_auc: {result['roc_auc_mean']:.4f} ± {result['roc_auc_std']:.4f} | "
                f"macro_f1: {result['macro_f1_mean']:.4f} ± {result['macro_f1_std']:.4f}"
            )

    fieldnames = [
        "layer",
        "aggregation",
        "accuracy_mean",
        "accuracy_std",
        "roc_auc_mean",
        "roc_auc_std",
        "macro_f1_mean",
        "macro_f1_std",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
    
    
### csv가 생김 -> plot_laer_auc.py로 넘기기~