import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


"""
Phase03-B: Evaluate Layer-wise Representation Delta as a Difficulty Signal

Input:
    sample_layer_delta.csv

Output:
    delta_signal_auc.csv
    delta_signal_auc_curve.png

Goal:
    For each aggregation, metric, and layer,
    compute how well delta alone separates Easy vs Hard.

Positive class:
    hard
    
python b_layerwise_representation/evaluate_delta_auc.py \
  --input_csv output_llama_500/layerwise_representation_change/sample_layer_delta.csv \
  --output_csv output_llama_500/layerwise_representation_change/delta_signal_auc.csv \
  --figure_dir output_llama_500/layerwise_representation_change/figures \
  --overwrite
"""


DELTA_METRICS = [
    "delta_l2",
    "delta_l2_normed",
    "delta_cosine",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to sample_layer_delta.csv.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save delta_signal_auc.csv.",
    )

    parser.add_argument(
        "--figure_dir",
        type=str,
        required=True,
        help="Directory to save AUC curve figures.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output CSV.",
    )

    return parser.parse_args()


def safe_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return np.nan

    auc = roc_auc_score(y_true, scores)
    auc_direction_invariant = max(auc, 1.0 - auc)

    return auc, auc_direction_invariant


def main():
    args = parse_args()

    if os.path.exists(args.output_csv) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_csv}\n"
            f"Use --overwrite to overwrite."
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    required_cols = {
        "index",
        "label",
        "aggregation",
        "layer",
        *DELTA_METRICS,
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=" * 80)
    print("[Phase03-B] Evaluate Delta Signal AUC")
    print("=" * 80)
    print("Input:", args.input_csv)
    print("Output CSV:", args.output_csv)
    print("Figure dir:", args.figure_dir)
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

        for metric in DELTA_METRICS:
            for layer in sorted(df_agg["layer"].unique()):
                sub = df_agg[df_agg["layer"] == layer].copy()

                y_true = (sub["label"] == "hard").astype(int).values
                scores = sub[metric].values

                auc, auc_abs = safe_auc(y_true, scores)

                easy_scores = sub[sub["label"] == "easy"][metric].values
                hard_scores = sub[sub["label"] == "hard"][metric].values

                rows.append(
                    {
                        "aggregation": aggregation,
                        "metric": metric,
                        "layer": int(layer),
                        "auc_hard_positive": float(auc),
                        "auc_direction_invariant": float(auc_abs),
                        "easy_mean": float(np.mean(easy_scores)),
                        "hard_mean": float(np.mean(hard_scores)),
                        "gap_hard_minus_easy": float(
                            np.mean(hard_scores) - np.mean(easy_scores)
                        ),
                        "n_easy": int(len(easy_scores)),
                        "n_hard": int(len(hard_scores)),
                    }
                )

    out = pd.DataFrame(rows)
    out = out.sort_values(["aggregation", "metric", "layer"]).reset_index(drop=True)
    out.to_csv(args.output_csv, index=False)

    print()
    print("Saved:", args.output_csv)
    print("Shape:", out.shape)

    for aggregation in sorted(out["aggregation"].unique()):
        for metric in DELTA_METRICS:
            sub = out[
                (out["aggregation"] == aggregation)
                & (out["metric"] == metric)
            ].copy()

            best_pos = sub.loc[sub["auc_hard_positive"].idxmax()]
            best_abs = sub.loc[sub["auc_direction_invariant"].idxmax()]

            print()
            print("=" * 80)
            print(f"[aggregation={aggregation}, metric={metric}]")
            print("=" * 80)
            print(
                "Best hard-positive AUC:",
                f"layer={int(best_pos['layer'])},",
                f"AUC={best_pos['auc_hard_positive']:.4f},",
                f"gap={best_pos['gap_hard_minus_easy']:.6f}",
            )
            print(
                "Best direction-invariant AUC:",
                f"layer={int(best_abs['layer'])},",
                f"AUC={best_abs['auc_direction_invariant']:.4f},",
                f"hard-positive AUC={best_abs['auc_hard_positive']:.4f},",
                f"gap={best_abs['gap_hard_minus_easy']:.6f}",
            )

            print()
            print(
                sub[
                    [
                        "layer",
                        "auc_hard_positive",
                        "auc_direction_invariant",
                        "gap_hard_minus_easy",
                        "easy_mean",
                        "hard_mean",
                    ]
                ].to_string(index=False)
            )

            # Plot curve
            plt.figure(figsize=(10, 6))
            plt.plot(
                sub["layer"],
                sub["auc_hard_positive"],
                marker="o",
                label="Hard-positive AUC",
            )
            plt.plot(
                sub["layer"],
                sub["auc_direction_invariant"],
                marker="s",
                label="Direction-invariant AUC",
            )
            plt.axhline(0.5, linestyle="--", linewidth=1)

            plt.xlabel("Layer")
            plt.ylabel("ROC-AUC")
            plt.title(f"Delta Signal AUC | {aggregation} | {metric}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            fig_path = os.path.join(
                args.figure_dir,
                f"delta_auc_{aggregation}_{metric}.png",
            )
            plt.savefig(fig_path, dpi=300)
            plt.close()

            print("Saved:", fig_path)

    print("=" * 80)


if __name__ == "__main__":
    main()