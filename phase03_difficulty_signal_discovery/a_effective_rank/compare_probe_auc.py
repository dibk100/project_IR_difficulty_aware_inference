import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


"""
Compare:

Phase02:
    Layer-wise Probe ROC-AUC

vs

Phase03-A:
    Layer-wise Effective Rank Gap
    
python a_effective_rank/compare_probe_auc.py \
  --probe_csv ../phase02_layerwise_decodability/output_llama_500/layerwise_probe_results.csv \
  --er_csv output_llama_500/effective_rank/layerwise_er_gap.csv \
  --output_dir output_llama_500/effective_rank/compare_probe \
  --aggregation last
  
python a_effective_rank/compare_probe_auc.py \
  --probe_csv ../phase02_layerwise_decodability/output_llama_500/layerwise_probe_results.csv \
  --er_csv output_llama_500/effective_rank/layerwise_er_gap.csv \
  --output_dir output_llama_500/effective_rank/compare_probe \
  --aggregation mean
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--probe_csv",
        type=str,
        required=True,
        help="Path to layerwise_probe_results.csv",
    )

    parser.add_argument(
        "--er_csv",
        type=str,
        required=True,
        help="Path to layerwise_er_gap.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save outputs",
    )
    
    parser.add_argument(
    "--aggregation",
    type=str,
    default="last_token",
    help="Aggregation type to use from probe results. Example: last_token or mean_pooling.",
    )

    return parser.parse_args()


def find_auc_column(df):
    candidates = [
        "roc_auc_mean",
        "roc_auc",
        "auc",
        "mean_roc_auc",
        "avg_roc_auc",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find ROC-AUC column.\nAvailable columns:\n{df.columns.tolist()}"
    )


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    probe_df = pd.read_csv(args.probe_csv)
    if "aggregation" in probe_df.columns:
        print("Available aggregations:", probe_df["aggregation"].unique())

        probe_df = probe_df[probe_df["aggregation"] == args.aggregation].copy()

        if len(probe_df) == 0:
            raise ValueError(f"No rows found for aggregation={args.aggregation}")
    
    er_df = pd.read_csv(args.er_csv)

    auc_col = find_auc_column(probe_df)

    print("=" * 80)
    print("Probe CSV:", args.probe_csv)
    print("ER CSV:", args.er_csv)
    print("AUC column:", auc_col)

    # ER는 centered 사용
    er_df = er_df[er_df["er_type"] == "er_centered"].copy()

    merged = pd.merge(
        probe_df,
        er_df[["layer", "gap_hard_minus_easy"]],
        on="layer",
        how="inner",
    )

    merged = merged.sort_values("layer")

    # 상관계수
    pearson_r, pearson_p = pearsonr(
        merged[auc_col],
        merged["gap_hard_minus_easy"],
    )

    spearman_rho, spearman_p = spearmanr(
        merged[auc_col],
        merged["gap_hard_minus_easy"],
    )

    print()
    print("Pearson r:", round(pearson_r, 4))
    print("Pearson p:", pearson_p)

    print()
    print("Spearman rho:", round(spearman_rho, 4))
    print("Spearman p:", spearman_p)

    print()
    print(merged.head())

    # 저장
    merged.to_csv(
        os.path.join(
            args.output_dir,
            "probe_auc_vs_er_gap.csv",
        ),
        index=False,
    )

    # -------------------------------------------------
    # Figure 1
    # -------------------------------------------------

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        merged["layer"],
        merged[auc_col],
        marker="o",
        label="Probe ROC-AUC",
    )

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Probe ROC-AUC")

    ax2 = ax1.twinx()

    ax2.plot(
        merged["layer"],
        merged["gap_hard_minus_easy"],
        marker="s",
        label="ER Gap",
    )

    ax2.set_ylabel("ER Gap (Hard - Easy)")

    plt.title(
        f"Probe ROC-AUC vs Effective Rank Gap\n"
        f"Pearson r={pearson_r:.3f}, Spearman rho={spearman_rho:.3f}"
    )

    fig.tight_layout()

    out_path = os.path.join(
        args.output_dir,
        "probe_auc_vs_er_gap.png",
    )

    plt.savefig(out_path, dpi=300)
    plt.close()

    print()
    print("Saved:", out_path)

    # -------------------------------------------------
    # Figure 2
    # -------------------------------------------------

    plt.figure(figsize=(7, 6))

    plt.scatter(
        merged["gap_hard_minus_easy"],
        merged[auc_col],
    )

    for _, row in merged.iterrows():
        plt.annotate(
            int(row["layer"]),
            (
                row["gap_hard_minus_easy"],
                row[auc_col],
            ),
            fontsize=8,
        )

    plt.xlabel("Effective Rank Gap")
    plt.ylabel("Probe ROC-AUC")

    plt.title(
        f"Layer-wise Probe AUC vs ER Gap\n"
        f"Pearson r={pearson_r:.3f}"
    )

    plt.tight_layout()

    out_path = os.path.join(
        args.output_dir,
        "probe_auc_vs_er_gap_scatter.png",
    )

    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)

    print("=" * 80)


if __name__ == "__main__":
    main()