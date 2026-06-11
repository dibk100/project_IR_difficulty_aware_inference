import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats


"""
Phase03-A: Analyze Layer-wise Effective Rank Gap

Input:
    sample_layer_effective_rank.csv

Output:
    layerwise_er_gap.csv

For each layer and each ER type:
    - easy_mean / hard_mean
    - gap = hard_mean - easy_mean
    - Cohen's d
    - Welch t-test p-value
    - Mann-Whitney U p-value
    
예시
python a_effective_rank/analyze_er_gap.py \
  --input_csv output_llama_500/effective_rank/sample_layer_effective_rank.csv \
  --output_csv output_llama_500/effective_rank/layerwise_er_gap.csv \
  --overwrite
"""


def cohen_d(x, y):
    """
    Cohen's d for two independent groups.
    d = (mean_y - mean_x) / pooled_std

    Here:
        x = easy
        y = hard
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    nx = len(x)
    ny = len(y)

    if nx < 2 or ny < 2:
        return np.nan

    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)

    pooled = ((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2)

    if pooled <= 0:
        return np.nan

    return float((np.mean(y) - np.mean(x)) / np.sqrt(pooled))


def summarize_layer(df_layer, value_col):
    easy = df_layer[df_layer["label"] == "easy"][value_col].dropna().values
    hard = df_layer[df_layer["label"] == "hard"][value_col].dropna().values

    easy_mean = np.mean(easy)
    hard_mean = np.mean(hard)

    easy_std = np.std(easy, ddof=1)
    hard_std = np.std(hard, ddof=1)

    gap = hard_mean - easy_mean
    d = cohen_d(easy, hard)

    # Welch's t-test
    t_stat, t_p = stats.ttest_ind(easy, hard, equal_var=False)

    # Mann-Whitney U test
    try:
        u_stat, u_p = stats.mannwhitneyu(easy, hard, alternative="two-sided")
    except ValueError:
        u_stat, u_p = np.nan, np.nan

    return {
        "n_easy": len(easy),
        "n_hard": len(hard),
        "easy_mean": easy_mean,
        "hard_mean": hard_mean,
        "easy_std": easy_std,
        "hard_std": hard_std,
        "gap_hard_minus_easy": gap,
        "cohen_d": d,
        "welch_t": t_stat,
        "welch_p": t_p,
        "mannwhitney_u": u_stat,
        "mannwhitney_p": u_p,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to sample_layer_effective_rank.csv.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save layerwise_er_gap.csv.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it already exists.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.output_csv) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_csv}\n"
            f"Use --overwrite to overwrite."
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    df = pd.read_csv(args.input_csv)

    required_cols = {"label", "layer", "er_raw", "er_centered"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=" * 80)
    print("[Phase03-A] Analyze ER Gap")
    print("=" * 80)
    print("Input:", args.input_csv)
    print("Output:", args.output_csv)
    print("Shape:", df.shape)
    print("Label counts:")
    print(df[["index", "label"]].drop_duplicates()["label"].value_counts())

    rows = []

    for er_type in ["er_raw", "er_centered"]:
        for layer in sorted(df["layer"].unique()):
            df_layer = df[df["layer"] == layer]

            summary = summarize_layer(df_layer, er_type)

            rows.append(
                {
                    "er_type": er_type,
                    "layer": int(layer),
                    **summary,
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["er_type", "layer"]).reset_index(drop=True)
    out.to_csv(args.output_csv, index=False)

    print()
    print("Saved:", args.output_csv)
    print("Shape:", out.shape)

    print()
    print("[Top positive gaps: er_centered]")
    print(
        out[out["er_type"] == "er_centered"]
        .sort_values("gap_hard_minus_easy", ascending=False)
        .head(10)[
            [
                "layer",
                "easy_mean",
                "hard_mean",
                "gap_hard_minus_easy",
                "cohen_d",
                "welch_p",
            ]
        ]
    )

    print()
    print("[Top negative gaps: er_centered]")
    print(
        out[out["er_type"] == "er_centered"]
        .sort_values("gap_hard_minus_easy", ascending=True)
        .head(10)[
            [
                "layer",
                "easy_mean",
                "hard_mean",
                "gap_hard_minus_easy",
                "cohen_d",
                "welch_p",
            ]
        ]
    )

    print("=" * 80)


if __name__ == "__main__":
    main()