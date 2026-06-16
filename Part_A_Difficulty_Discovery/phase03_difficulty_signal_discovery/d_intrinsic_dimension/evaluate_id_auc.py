import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


"""
Phase03-D: Evaluate Sample-wise Local ID as Difficulty Signal

Input:
    sample_layer_local_id.csv

Output:
    local_id_signal_auc.csv

For each aggregation and layer:
    - local_id hard-positive ROC-AUC
    - direction-invariant ROC-AUC
    - Easy/Hard mean gap

python d_intrinsic_dimension/evaluate_id_auc.py \
  --input_csv output_phi_1000/intrinsic_dimension/sample_layer_local_id.csv \
  --output_csv output_phi_1000/intrinsic_dimension/local_id_signal_auc.csv \
  --overwrite
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to sample_layer_local_id.csv.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save local_id_signal_auc.csv.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def safe_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return np.nan

    mask = np.isfinite(scores)

    y_true = y_true[mask]
    scores = scores[mask]

    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, scores)


def main():
    args = parse_args()

    if os.path.exists(args.output_csv) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_csv}\n"
            f"Use --overwrite to overwrite."
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    df = pd.read_csv(args.input_csv)

    required_cols = {
        "index",
        "id",
        "label",
        "aggregation",
        "layer",
        "local_id",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=" * 80)
    print("[Phase03-D] Evaluate Local ID AUC")
    print("=" * 80)
    print("Input:", args.input_csv)
    print("Output:", args.output_csv)
    print("Shape:", df.shape)
    print()
    print("Label counts:")
    print(df[["index", "label"]].drop_duplicates()["label"].value_counts())
    print()
    print("Aggregations:", sorted(df["aggregation"].unique()))
    print("Layers:", int(df["layer"].min()), "to", int(df["layer"].max()))

    rows = []

    for aggregation in sorted(df["aggregation"].unique()):
        df_agg = df[df["aggregation"] == aggregation].copy()

        for layer in sorted(df_agg["layer"].unique()):
            sub = df_agg[df_agg["layer"] == layer].copy()

            y = (sub["label"] == "hard").astype(int).values
            scores = sub["local_id"].values.astype(float)

            auc_hard_positive = safe_auc(y, scores)

            if np.isnan(auc_hard_positive):
                auc_direction_invariant = np.nan
            else:
                auc_direction_invariant = max(
                    auc_hard_positive,
                    1.0 - auc_hard_positive,
                )

            easy_scores = sub[sub["label"] == "easy"]["local_id"].dropna().values
            hard_scores = sub[sub["label"] == "hard"]["local_id"].dropna().values

            easy_mean = float(np.mean(easy_scores)) if len(easy_scores) else np.nan
            hard_mean = float(np.mean(hard_scores)) if len(hard_scores) else np.nan

            rows.append(
                {
                    "aggregation": aggregation,
                    "layer": int(layer),
                    "auc_hard_positive": float(auc_hard_positive),
                    "auc_direction_invariant": float(auc_direction_invariant),
                    "easy_mean": easy_mean,
                    "hard_mean": hard_mean,
                    "gap_hard_minus_easy": hard_mean - easy_mean,
                    "n_easy": int(len(easy_scores)),
                    "n_hard": int(len(hard_scores)),
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["aggregation", "layer"]).reset_index(drop=True)
    out.to_csv(args.output_csv, index=False)

    print()
    print("Saved:", args.output_csv)
    print("Shape:", out.shape)

    print()
    print("=" * 80)
    print("[Top hard-positive AUC]")
    print("=" * 80)
    print(
        out.sort_values("auc_hard_positive", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print()
    print("=" * 80)
    print("[Top direction-invariant AUC]")
    print("=" * 80)
    print(
        out.sort_values("auc_direction_invariant", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("=" * 80)


if __name__ == "__main__":
    main()