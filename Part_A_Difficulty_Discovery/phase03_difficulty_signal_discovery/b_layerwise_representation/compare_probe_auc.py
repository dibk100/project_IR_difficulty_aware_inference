import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


"""
Phase03-B: Compare Phase02 Probe AUC with Delta Signal

Compare:
    Phase02 layer-wise Difficulty Probe ROC-AUC
vs
    Phase03-B layer-wise representation delta gap / delta AUC

Inputs:
    layerwise_probe_results.csv
    layerwise_delta_gap.csv
    delta_signal_auc.csv

Outputs:
    probe_auc_vs_delta.csv
    probe_auc_vs_delta_gap.png
    probe_auc_vs_delta_auc.png
    probe_auc_vs_delta_scatter.png
    
python b_layerwise_representation/compare_probe_auc.py \
  --probe_csv ../phase02_layerwise_decodability/output_llama_500/layerwise_probe_results.csv \
  --gap_csv output_llama_500/layerwise_representation_change/layerwise_delta_gap.csv \
  --auc_csv output_llama_500/layerwise_representation_change/delta_signal_auc.csv \
  --output_dir output_llama_500/layerwise_representation_change/compare_probe \
  --aggregation mean \
  --metric delta_l2_normed \
  --overwrite

python b_layerwise_representation/compare_probe_auc.py \
  --probe_csv ../phase02_layerwise_decodability/output_llama_500/layerwise_probe_results.csv \
  --gap_csv output_llama_500/layerwise_representation_change/layerwise_delta_gap.csv \
  --auc_csv output_llama_500/layerwise_representation_change/delta_signal_auc.csv \
  --output_dir output_llama_500/layerwise_representation_change/compare_probe \
  --aggregation mean \
  --metric delta_cosine \
  --overwrite
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--probe_csv",
        type=str,
        required=True,
        help="Path to Phase02 layerwise_probe_results.csv.",
    )

    parser.add_argument(
        "--gap_csv",
        type=str,
        required=True,
        help="Path to Phase03-B layerwise_delta_gap.csv.",
    )

    parser.add_argument(
        "--auc_csv",
        type=str,
        required=True,
        help="Path to Phase03-B delta_signal_auc.csv.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save comparison outputs.",
    )

    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["last", "mean"],
        help="Aggregation to compare: last or mean.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="delta_l2_normed",
        choices=["delta_l2", "delta_l2_normed", "delta_cosine"],
        help="Delta metric to compare.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
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
        f"Could not find ROC-AUC column. Available columns: {df.columns.tolist()}"
    )


def corr_pair(x, y):
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    output_csv = os.path.join(
        args.output_dir,
        f"probe_auc_vs_delta_{args.aggregation}_{args.metric}.csv",
    )

    if os.path.exists(output_csv) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_csv}\nUse --overwrite to overwrite."
        )

    probe_df = pd.read_csv(args.probe_csv)
    gap_df = pd.read_csv(args.gap_csv)
    auc_df = pd.read_csv(args.auc_csv)

    auc_col = find_auc_column(probe_df)

    print("=" * 80)
    print("[Phase03-B] Compare Probe AUC with Delta Signal")
    print("=" * 80)
    print("Probe CSV:", args.probe_csv)
    print("Gap CSV:", args.gap_csv)
    print("AUC CSV:", args.auc_csv)
    print("Aggregation:", args.aggregation)
    print("Metric:", args.metric)
    print("Probe AUC column:", auc_col)

    print()
    print("Available probe aggregations:", probe_df["aggregation"].unique())

    probe_sub = probe_df[probe_df["aggregation"] == args.aggregation].copy()
    gap_sub = gap_df[
        (gap_df["aggregation"] == args.aggregation)
        & (gap_df["metric"] == args.metric)
    ].copy()
    auc_sub = auc_df[
        (auc_df["aggregation"] == args.aggregation)
        & (auc_df["metric"] == args.metric)
    ].copy()

    if len(probe_sub) == 0:
        raise ValueError(f"No probe rows for aggregation={args.aggregation}")

    if len(gap_sub) == 0:
        raise ValueError(
            f"No gap rows for aggregation={args.aggregation}, metric={args.metric}"
        )

    if len(auc_sub) == 0:
        raise ValueError(
            f"No AUC rows for aggregation={args.aggregation}, metric={args.metric}"
        )

    merged = probe_sub[
        [
            "layer",
            "aggregation",
            auc_col,
            "accuracy_mean",
            "macro_f1_mean",
        ]
    ].merge(
        gap_sub[
            [
                "layer",
                "metric",
                "gap_hard_minus_easy",
                "cohen_d",
                "welch_p",
            ]
        ],
        on="layer",
        how="inner",
    ).merge(
        auc_sub[
            [
                "layer",
                "auc_hard_positive",
                "auc_direction_invariant",
            ]
        ],
        on="layer",
        how="inner",
    )

    merged = merged.sort_values("layer").reset_index(drop=True)

    corr_gap = corr_pair(merged[auc_col], merged["gap_hard_minus_easy"])
    corr_auc = corr_pair(merged[auc_col], merged["auc_hard_positive"])
    corr_auc_abs = corr_pair(merged[auc_col], merged["auc_direction_invariant"])

    merged.to_csv(output_csv, index=False)

    print()
    print("Saved:", output_csv)
    print("Shape:", merged.shape)

    print()
    print("[Correlation: Probe AUC vs Delta Gap]")
    print(f"Pearson r: {corr_gap['pearson_r']:.4f}")
    print(f"Pearson p: {corr_gap['pearson_p']:.6g}")
    print(f"Spearman rho: {corr_gap['spearman_rho']:.4f}")
    print(f"Spearman p: {corr_gap['spearman_p']:.6g}")

    print()
    print("[Correlation: Probe AUC vs Delta AUC]")
    print(f"Pearson r: {corr_auc['pearson_r']:.4f}")
    print(f"Pearson p: {corr_auc['pearson_p']:.6g}")
    print(f"Spearman rho: {corr_auc['spearman_rho']:.4f}")
    print(f"Spearman p: {corr_auc['spearman_p']:.6g}")

    print()
    print("[Correlation: Probe AUC vs Direction-invariant Delta AUC]")
    print(f"Pearson r: {corr_auc_abs['pearson_r']:.4f}")
    print(f"Pearson p: {corr_auc_abs['pearson_p']:.6g}")
    print(f"Spearman rho: {corr_auc_abs['spearman_rho']:.4f}")
    print(f"Spearman p: {corr_auc_abs['spearman_p']:.6g}")

    print()
    print(merged.head())

    # Figure 1: Probe AUC vs Delta Gap
    plt.figure(figsize=(10, 6))
    plt.plot(
        merged["layer"],
        merged[auc_col],
        marker="o",
        label="Probe ROC-AUC",
    )
    plt.plot(
        merged["layer"],
        merged["gap_hard_minus_easy"],
        marker="s",
        label="Delta Gap",
    )
    plt.xlabel("Layer")
    plt.title(
        f"Probe AUC vs Delta Gap | {args.aggregation} | {args.metric}\n"
        f"Pearson r={corr_gap['pearson_r']:.3f}, "
        f"Spearman rho={corr_gap['spearman_rho']:.3f}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(
        args.output_dir,
        f"probe_auc_vs_delta_gap_{args.aggregation}_{args.metric}.png",
    )
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)

    # Figure 2: Probe AUC vs Delta AUC
    plt.figure(figsize=(10, 6))
    plt.plot(
        merged["layer"],
        merged[auc_col],
        marker="o",
        label="Probe ROC-AUC",
    )
    plt.plot(
        merged["layer"],
        merged["auc_hard_positive"],
        marker="s",
        label="Delta ROC-AUC",
    )
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("ROC-AUC")
    plt.title(
        f"Probe AUC vs Delta AUC | {args.aggregation} | {args.metric}\n"
        f"Pearson r={corr_auc['pearson_r']:.3f}, "
        f"Spearman rho={corr_auc['spearman_rho']:.3f}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(
        args.output_dir,
        f"probe_auc_vs_delta_auc_{args.aggregation}_{args.metric}.png",
    )
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)

    # Figure 3: Scatter
    plt.figure(figsize=(7, 6))
    plt.scatter(
        merged["auc_hard_positive"],
        merged[auc_col],
    )

    for _, row in merged.iterrows():
        plt.annotate(
            int(row["layer"]),
            (
                row["auc_hard_positive"],
                row[auc_col],
            ),
            fontsize=8,
        )

    plt.xlabel("Delta ROC-AUC")
    plt.ylabel("Probe ROC-AUC")
    plt.title(
        f"Probe AUC vs Delta AUC Scatter | {args.aggregation} | {args.metric}\n"
        f"Pearson r={corr_auc['pearson_r']:.3f}"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(
        args.output_dir,
        f"probe_auc_vs_delta_auc_scatter_{args.aggregation}_{args.metric}.png",
    )
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)

    print("=" * 80)


if __name__ == "__main__":
    main()