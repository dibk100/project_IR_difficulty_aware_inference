import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


"""
Phase03-A: Evaluate Effective Rank as a Difficulty Signal

Input:
    sample_layer_effective_rank.csv

Output:
    er_signal_auc.csv
    er_signal_auc_curve.png

Goal:
    For each layer, compute how well ER alone separates Easy vs Hard.

Note:
    label: easy / hard
    positive class: hard
    
python a_effective_rank/evaluate_er_auc.py \
  --input_csv output_llama_500/effective_rank/sample_layer_effective_rank.csv \
  --output_dir output_llama_500/effective_rank/er_signal_auc \
  --overwrite
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to sample_layer_effective_rank.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save outputs",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )

    return parser.parse_args()


def compute_auc(y_true, scores):
    """
    y_true:
        hard = 1
        easy = 0

    scores:
        ER value
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    auc = roc_auc_score(y_true, scores)

    # If AUC < 0.5, the signal is reversed.
    # Keep both original and direction-invariant AUC.
    auc_abs = max(auc, 1.0 - auc)

    return auc, auc_abs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    output_csv = os.path.join(args.output_dir, "er_signal_auc.csv")
    output_fig = os.path.join(args.output_dir, "er_signal_auc_curve.png")

    if os.path.exists(output_csv) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_csv}")

    df = pd.read_csv(args.input_csv)

    required_cols = {"label", "layer", "er_raw", "er_centered"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=" * 80)
    print("[Phase03-A] Evaluate ER Signal AUC")
    print("=" * 80)
    print("Input:", args.input_csv)
    print("Output dir:", args.output_dir)
    print("Shape:", df.shape)
    print("Label counts:")
    print(df[["index", "label"]].drop_duplicates()["label"].value_counts())

    rows = []

    for er_type in ["er_raw", "er_centered"]:
        for layer in sorted(df["layer"].unique()):
            sub = df[df["layer"] == layer].copy()

            y_true = (sub["label"] == "hard").astype(int).values
            scores = sub[er_type].values

            auc, auc_abs = compute_auc(y_true, scores)

            easy_scores = sub[sub["label"] == "easy"][er_type].values
            hard_scores = sub[sub["label"] == "hard"][er_type].values

            rows.append(
                {
                    "er_type": er_type,
                    "layer": int(layer),
                    "auc_hard_positive": auc,
                    "auc_direction_invariant": auc_abs,
                    "easy_mean": float(np.mean(easy_scores)),
                    "hard_mean": float(np.mean(hard_scores)),
                    "gap_hard_minus_easy": float(np.mean(hard_scores) - np.mean(easy_scores)),
                    "n_easy": int(len(easy_scores)),
                    "n_hard": int(len(hard_scores)),
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["er_type", "layer"]).reset_index(drop=True)
    out.to_csv(output_csv, index=False)

    print()
    print("Saved:", output_csv)

    for er_type in ["er_raw", "er_centered"]:
        sub = out[out["er_type"] == er_type].copy()
        best = sub.loc[sub["auc_hard_positive"].idxmax()]

        print()
        print("=" * 80)
        print(f"[{er_type}]")
        print("Best layer:", int(best["layer"]))
        print("Best AUC:", round(float(best["auc_hard_positive"]), 4))
        print("Best gap:", round(float(best["gap_hard_minus_easy"]), 4))
        print()
        print(
            sub[
                [
                    "layer",
                    "auc_hard_positive",
                    "gap_hard_minus_easy",
                    "easy_mean",
                    "hard_mean",
                ]
            ].to_string(index=False)
        )

    # Plot
    plt.figure(figsize=(10, 6))

    for er_type in ["er_raw", "er_centered"]:
        sub = out[out["er_type"] == er_type].sort_values("layer")
        plt.plot(
            sub["layer"],
            sub["auc_hard_positive"],
            marker="o",
            label=er_type,
        )

    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("ROC-AUC using ER only")
    plt.title("Effective Rank Signal ROC-AUC by Layer")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_fig, dpi=300)
    plt.close()

    print()
    print("Saved:", output_fig)
    print("=" * 80)


if __name__ == "__main__":
    main()