import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats


"""
Phase03-B: Analyze Layer-wise Representation Delta Gap

Input:
    sample_layer_delta.csv

Output:
    layerwise_delta_gap.csv

For each aggregation, layer, and delta metric:
    - easy_mean / hard_mean
    - gap = hard_mean - easy_mean
    - Cohen's d
    - Welch t-test p-value
    - Mann-Whitney U p-value
    
python b_layerwise_representation/analyze_delta_gap.py \
  --input_csv output_llama_500/layerwise_representation_change/sample_layer_delta.csv \
  --output_csv output_llama_500/layerwise_representation_change/layerwise_delta_gap.csv \
  --overwrite
"""


DELTA_METRICS = [
    "delta_l2",
    "delta_l2_normed",
    "delta_cosine",
]


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


def summarize_group(df_layer, metric):
    easy = df_layer[df_layer["label"] == "easy"][metric].dropna().values
    hard = df_layer[df_layer["label"] == "hard"][metric].dropna().values

    easy_mean = np.mean(easy)
    hard_mean = np.mean(hard)

    easy_std = np.std(easy, ddof=1)
    hard_std = np.std(hard, ddof=1)

    gap = hard_mean - easy_mean
    d = cohen_d(easy, hard)

    welch_t, welch_p = stats.ttest_ind(easy, hard, equal_var=False)

    try:
        mann_u, mann_p = stats.mannwhitneyu(easy, hard, alternative="two-sided")
    except ValueError:
        mann_u, mann_p = np.nan, np.nan

    return {
        "n_easy": len(easy),
        "n_hard": len(hard),
        "easy_mean": float(easy_mean),
        "hard_mean": float(hard_mean),
        "easy_std": float(easy_std),
        "hard_std": float(hard_std),
        "gap_hard_minus_easy": float(gap),
        "cohen_d": float(d),
        "welch_t": float(welch_t),
        "welch_p": float(welch_p),
        "mannwhitney_u": float(mann_u),
        "mannwhitney_p": float(mann_p),
    }


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
        help="Path to save layerwise_delta_gap.csv.",
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
    print("[Phase03-B] Analyze Delta Gap")
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
        df_agg = df[df["aggregation"] == aggregation]

        for metric in DELTA_METRICS:
            for layer in sorted(df_agg["layer"].unique()):
                df_layer = df_agg[df_agg["layer"] == layer]

                summary = summarize_group(df_layer, metric)

                rows.append(
                    {
                        "aggregation": aggregation,
                        "metric": metric,
                        "layer": int(layer),
                        **summary,
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

            print()
            print("=" * 80)
            print(f"[Top positive gaps] aggregation={aggregation}, metric={metric}")
            print("=" * 80)
            print(
                sub.sort_values("gap_hard_minus_easy", ascending=False)
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
                .to_string(index=False)
            )

            print()
            print(f"[Top negative gaps] aggregation={aggregation}, metric={metric}")
            print(
                sub.sort_values("gap_hard_minus_easy", ascending=True)
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
                .to_string(index=False)
            )

    print("=" * 80)


if __name__ == "__main__":
    main()