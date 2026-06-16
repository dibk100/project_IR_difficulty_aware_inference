import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score


"""
Phase04-B: Analyze Difficulty Projection Scores

Input:
    difficulty_projection_scores.csv

Output:
    projection_score_analysis.csv
    figures:
        projection_auc_curve_<aggregation>.png
        projection_gap_curve_<aggregation>.png
        projection_hist_<aggregation>_layerXX.png
        projection_box_<aggregation>_layerXX.png
        
python a_probe_direction/analyze_projection_scores.py \
  --input_csv output_llama_1000/probe_direction/difficulty_projection_scores.csv \
  --output_csv output_llama_1000/probe_direction/projection_score_analysis.csv \
  --figure_dir output_llama_1000/probe_direction/figures \
  --plot_top_k 3 \
  --overwrite
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to difficulty_projection_scores.csv.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save projection_score_analysis.csv.",
    )

    parser.add_argument(
        "--figure_dir",
        type=str,
        required=True,
        help="Directory to save figures.",
    )

    parser.add_argument(
        "--plot_top_k",
        type=int,
        default=3,
        help="Number of top layers to plot hist/box figures for each aggregation.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def safe_auc(y, scores):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, scores)


def cohen_d(easy, hard):
    easy = np.asarray(easy, dtype=np.float64)
    hard = np.asarray(hard, dtype=np.float64)

    n_easy = len(easy)
    n_hard = len(hard)

    if n_easy < 2 or n_hard < 2:
        return np.nan

    var_easy = np.var(easy, ddof=1)
    var_hard = np.var(hard, ddof=1)

    pooled_var = (
        ((n_easy - 1) * var_easy + (n_hard - 1) * var_hard)
        / (n_easy + n_hard - 2)
    )

    if pooled_var <= 0:
        return np.nan

    return float((np.mean(hard) - np.mean(easy)) / np.sqrt(pooled_var))


def plot_auc_curve(df, aggregation, figure_dir):
    sub = df[df["aggregation"] == aggregation].sort_values("layer")

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
    plt.title(f"Difficulty Projection Score AUC | {aggregation}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        figure_dir,
        f"projection_auc_curve_{aggregation}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def plot_gap_curve(df, aggregation, figure_dir):
    sub = df[df["aggregation"] == aggregation].sort_values("layer")

    plt.figure(figsize=(10, 6))
    plt.plot(
        sub["layer"],
        sub["gap_hard_minus_easy"],
        marker="o",
        label="Hard - Easy score",
    )
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Layer")
    plt.ylabel("Projection Score Gap")
    plt.title(f"Difficulty Projection Score Gap | {aggregation}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        figure_dir,
        f"projection_gap_curve_{aggregation}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def plot_hist(scores_df, aggregation, layer, figure_dir):
    sub = scores_df[
        (scores_df["aggregation"] == aggregation)
        & (scores_df["layer"] == layer)
    ].copy()

    easy = sub[sub["label"] == "easy"]["cv_difficulty_score"].values
    hard = sub[sub["label"] == "hard"]["cv_difficulty_score"].values

    plt.figure(figsize=(10, 6))
    plt.hist(easy, bins=40, alpha=0.6, label="Easy", density=True)
    plt.hist(hard, bins=40, alpha=0.6, label="Hard", density=True)

    plt.xlabel("Difficulty Projection Score")
    plt.ylabel("Density")
    plt.title(f"Projection Score Distribution | {aggregation} | Layer {layer}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        figure_dir,
        f"projection_hist_{aggregation}_layer{layer:02d}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def plot_box(scores_df, aggregation, layer, figure_dir):
    sub = scores_df[
        (scores_df["aggregation"] == aggregation)
        & (scores_df["layer"] == layer)
    ].copy()

    easy = sub[sub["label"] == "easy"]["cv_difficulty_score"].values
    hard = sub[sub["label"] == "hard"]["cv_difficulty_score"].values

    plt.figure(figsize=(7, 6))
    plt.boxplot(
        [easy, hard],
        labels=["Easy", "Hard"],
        showfliers=False,
    )

    plt.ylabel("Difficulty Projection Score")
    plt.title(f"Projection Score Boxplot | {aggregation} | Layer {layer}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        figure_dir,
        f"projection_box_{aggregation}_layer{layer:02d}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


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
        "id",
        "label",
        "aggregation",
        "layer",
        "cv_difficulty_score",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=" * 80)
    print("[Phase04-B] Analyze Difficulty Projection Scores")
    print("=" * 80)
    print("Input:", args.input_csv)
    print("Output:", args.output_csv)
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

        for layer in sorted(df_agg["layer"].unique()):
            sub = df_agg[df_agg["layer"] == layer].copy()

            y = (sub["label"] == "hard").astype(int).values
            scores = sub["cv_difficulty_score"].values.astype(float)

            auc = safe_auc(y, scores)
            auc_inv = max(auc, 1.0 - auc)

            easy_scores = sub[sub["label"] == "easy"]["cv_difficulty_score"].values
            hard_scores = sub[sub["label"] == "hard"]["cv_difficulty_score"].values

            easy_mean = float(np.mean(easy_scores))
            hard_mean = float(np.mean(hard_scores))
            easy_std = float(np.std(easy_scores, ddof=1))
            hard_std = float(np.std(hard_scores, ddof=1))

            gap = hard_mean - easy_mean
            d = cohen_d(easy_scores, hard_scores)

            welch_t, welch_p = stats.ttest_ind(
                easy_scores,
                hard_scores,
                equal_var=False,
            )

            try:
                mann_u, mann_p = stats.mannwhitneyu(
                    easy_scores,
                    hard_scores,
                    alternative="two-sided",
                )
            except ValueError:
                mann_u, mann_p = np.nan, np.nan

            rows.append(
                {
                    "aggregation": aggregation,
                    "layer": int(layer),
                    "auc_hard_positive": float(auc),
                    "auc_direction_invariant": float(auc_inv),
                    "n_easy": int(len(easy_scores)),
                    "n_hard": int(len(hard_scores)),
                    "easy_mean": easy_mean,
                    "hard_mean": hard_mean,
                    "easy_std": easy_std,
                    "hard_std": hard_std,
                    "gap_hard_minus_easy": float(gap),
                    "cohen_d": float(d),
                    "welch_t": float(welch_t),
                    "welch_p": float(welch_p),
                    "mannwhitney_u": float(mann_u),
                    "mannwhitney_p": float(mann_p),
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
    print("[Top Projection AUC]")
    print("=" * 80)
    print(
        out.sort_values("auc_hard_positive", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    for aggregation in sorted(out["aggregation"].unique()):
        plot_auc_curve(out, aggregation, args.figure_dir)
        plot_gap_curve(out, aggregation, args.figure_dir)

        top_layers = (
            out[out["aggregation"] == aggregation]
            .sort_values("auc_hard_positive", ascending=False)
            .head(args.plot_top_k)["layer"]
            .astype(int)
            .tolist()
        )

        for layer in top_layers:
            plot_hist(df, aggregation, layer, args.figure_dir)
            plot_box(df, aggregation, layer, args.figure_dir)

    print()
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()