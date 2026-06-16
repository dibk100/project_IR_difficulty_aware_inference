import os
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


"""
Phase03-C: Evaluate Representation Norm Signal

Input:
    sample_layer_norm.csv

Output:
    norm_signal_auc.csv

For each layer and aggregation:
    - ROC-AUC
    - Direction-invariant ROC-AUC
    
python c_representation_norm/evaluate_norm_auc.py \
  --input_csv output_phi_1000/representation_norm/sample_layer_norm.csv \
  --output_csv output_phi_1000/representation_norm/norm_signal_auc.csv
"""


def compute_auc(labels_binary, scores):
    try:
        return roc_auc_score(labels_binary, scores)
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    print("=" * 80)
    print("[Phase03-C] Evaluate Representation Norm AUC")
    print("=" * 80)

    df = pd.read_csv(args.input_csv)

    df["label_binary"] = (
        df["label"]
        .map({"easy": 0, "hard": 1})
        .astype(int)
    )

    rows = []

    for aggregation in sorted(df["aggregation"].unique()):

        df_agg = df[df["aggregation"] == aggregation]

        for layer in sorted(df_agg["layer"].unique()):

            sub = df_agg[df_agg["layer"] == layer]

            y = sub["label_binary"].values
            scores = sub["norm_l2"].values

            auc_hard_positive = compute_auc(y, scores)

            auc_direction_invariant = max(
                auc_hard_positive,
                1.0 - auc_hard_positive,
            )

            easy_mean = (
                sub[sub["label_binary"] == 0]
                ["norm_l2"]
                .mean()
            )

            hard_mean = (
                sub[sub["label_binary"] == 1]
                ["norm_l2"]
                .mean()
            )

            rows.append(
                {
                    "aggregation": aggregation,
                    "layer": int(layer),
                    "auc_hard_positive": auc_hard_positive,
                    "auc_direction_invariant": auc_direction_invariant,
                    "easy_mean": easy_mean,
                    "hard_mean": hard_mean,
                    "gap_hard_minus_easy":
                        hard_mean - easy_mean,
                    "n_easy":
                        int((y == 0).sum()),
                    "n_hard":
                        int((y == 1).sum()),
                }
            )

    out_df = pd.DataFrame(rows)

    os.makedirs(
        os.path.dirname(args.output_csv),
        exist_ok=True,
    )

    out_df.to_csv(
        args.output_csv,
        index=False,
    )

    print()
    print("Saved:", args.output_csv)
    print("Shape:", out_df.shape)

    print()
    print("[Top AUC layers]")
    print(
        out_df.sort_values(
            "auc_direction_invariant",
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )

    print("=" * 80)


if __name__ == "__main__":
    main()