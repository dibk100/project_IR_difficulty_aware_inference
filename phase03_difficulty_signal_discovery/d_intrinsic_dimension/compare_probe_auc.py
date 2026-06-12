import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


"""
Phase03-D: Compare Intrinsic Dimension with Phase02 Difficulty Probe

Inputs:
    1. Phase02 layerwise_probe_results.csv
    2. Phase03-D layerwise_id_gap.csv
    3. Phase03-D layerwise_id_profile.csv

Outputs:
    compare_probe/probe_auc_vs_id_<aggregation>.csv
    figures:
        probe_auc_vs_id_gap_<aggregation>.png
        probe_auc_vs_id_profile_<aggregation>.png
        probe_auc_vs_id_gap_scatter_<aggregation>.png

Questions:
    - Does ID gap align with difficulty decodability?
    - Does ID profile minimum/maximum align with probe AUC peak?
    
python d_intrinsic_dimension/compare_probe_auc.py \
  --probe_csv ../phase02_layerwise_decodability/output_phi_1000/layerwise_probe_results.csv \
  --id_gap_csv output_phi_1000/intrinsic_dimension/layerwise_id_gap.csv \
  --id_profile_csv output_phi_1000/intrinsic_dimension/layerwise_id_profile.csv \
  --output_dir output_phi_1000/intrinsic_dimension/compare_probe \
  --aggregation mean \
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
        "--id_gap_csv",
        type=str,
        required=True,
        help="Path to Phase03-D layerwise_id_gap.csv.",
    )

    parser.add_argument(
        "--id_profile_csv",
        type=str,
        required=True,
        help="Path to Phase03-D layerwise_id_profile.csv.",
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
        help="Aggregation to compare.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
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


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def print_corr(name, corr):
    print()
    print(f"[Correlation: Probe AUC vs {name}]")
    print(f"Pearson r: {corr['pearson_r']:.4f}")
    print(f"Pearson p: {corr['pearson_p']:.6g}")
    print(f"Spearman rho: {corr['spearman_rho']:.4f}")
    print(f"Spearman p: {corr['spearman_p']:.6g}")


def plot_two_curves(
    merged,
    x_col,
    y1_col,
    y2_col,
    y1_label,
    y2_label,
    title,
    out_path,
):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        merged[x_col],
        merged[y1_col],
        marker="o",
        label=y1_label,
    )
    ax1.set_xlabel("Layer")
    ax1.set_ylabel(y1_label)

    ax2 = ax1.twinx()
    ax2.plot(
        merged[x_col],
        merged[y2_col],
        marker="s",
        label=y2_label,
    )
    ax2.set_ylabel(y2_label)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best",
    )

    ax1.grid(True, alpha=0.3)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def plot_scatter(
    merged,
    x_col,
    y_col,
    title,
    xlabel,
    ylabel,
    out_path,
):
    plt.figure(figsize=(7, 6))

    plt.scatter(
        merged[x_col],
        merged[y_col],
    )

    for _, row in merged.iterrows():
        plt.annotate(
            int(row["layer"]),
            (row[x_col], row[y_col]),
            fontsize=8,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    output_csv = os.path.join(
        args.output_dir,
        f"probe_auc_vs_id_{args.aggregation}.csv",
    )

    if os.path.exists(output_csv) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_csv}\n"
            f"Use --overwrite to overwrite."
        )

    probe_df = pd.read_csv(args.probe_csv)
    gap_df = pd.read_csv(args.id_gap_csv)
    profile_df = pd.read_csv(args.id_profile_csv)

    auc_col = find_auc_column(probe_df)

    print("=" * 80)
    print("[Phase03-D] Compare Probe AUC with Intrinsic Dimension")
    print("=" * 80)
    print("Probe CSV:", args.probe_csv)
    print("ID gap CSV:", args.id_gap_csv)
    print("ID profile CSV:", args.id_profile_csv)
    print("Aggregation:", args.aggregation)
    print("Probe AUC column:", auc_col)

    print()
    print("Available probe aggregations:", sorted(probe_df["aggregation"].unique()))
    print("Available ID aggregations:", sorted(gap_df["aggregation"].unique()))

    probe_sub = probe_df[probe_df["aggregation"] == args.aggregation].copy()
    gap_sub = gap_df[gap_df["aggregation"] == args.aggregation].copy()

    if len(probe_sub) == 0:
        raise ValueError(f"No probe rows for aggregation={args.aggregation}")

    if len(gap_sub) == 0:
        raise ValueError(f"No ID gap rows for aggregation={args.aggregation}")

    # Pivot ID profile into all/easy/hard columns.
    profile_sub = profile_df[profile_df["aggregation"] == args.aggregation].copy()

    profile_pivot = profile_sub.pivot_table(
        index="layer",
        columns="group",
        values="id_twonn",
        aggfunc="first",
    ).reset_index()

    rename_map = {}
    for col in profile_pivot.columns:
        if col != "layer":
            rename_map[col] = f"id_{col}"

    profile_pivot = profile_pivot.rename(columns=rename_map)

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
                "easy_id",
                "hard_id",
                "gap_hard_minus_easy",
                "abs_gap",
            ]
        ],
        on="layer",
        how="inner",
    ).merge(
        profile_pivot,
        on="layer",
        how="inner",
    )

    merged = merged.sort_values("layer").reset_index(drop=True)

    merged.to_csv(output_csv, index=False)

    print()
    print("Saved:", output_csv)
    print("Shape:", merged.shape)
    print()
    print(merged.head())

    corr_gap = safe_corr(merged[auc_col], merged["gap_hard_minus_easy"])
    corr_abs_gap = safe_corr(merged[auc_col], merged["abs_gap"])

    print_corr("ID gap (Hard - Easy)", corr_gap)
    print_corr("absolute ID gap", corr_abs_gap)

    if "id_all" in merged.columns:
        corr_all = safe_corr(merged[auc_col], merged["id_all"])
        print_corr("ID all", corr_all)

    if "id_easy" in merged.columns:
        corr_easy = safe_corr(merged[auc_col], merged["id_easy"])
        print_corr("ID easy", corr_easy)

    if "id_hard" in merged.columns:
        corr_hard = safe_corr(merged[auc_col], merged["id_hard"])
        print_corr("ID hard", corr_hard)

    # Best layers
    best_probe = merged.loc[merged[auc_col].idxmax()]
    best_gap = merged.loc[merged["gap_hard_minus_easy"].idxmax()]
    best_abs_gap = merged.loc[merged["abs_gap"].idxmax()]

    print()
    print("=" * 80)
    print("[Best layers]")
    print("=" * 80)
    print(
        f"Best Probe AUC: layer={int(best_probe['layer'])}, "
        f"AUC={best_probe[auc_col]:.4f}"
    )
    print(
        f"Best ID gap: layer={int(best_gap['layer'])}, "
        f"gap={best_gap['gap_hard_minus_easy']:.4f}"
    )
    print(
        f"Best abs ID gap: layer={int(best_abs_gap['layer'])}, "
        f"abs_gap={best_abs_gap['abs_gap']:.4f}"
    )

    if "id_all" in merged.columns:
        min_all = merged.loc[merged["id_all"].idxmin()]
        max_all = merged.loc[merged["id_all"].idxmax()]

        print(
            f"Min ID all: layer={int(min_all['layer'])}, "
            f"ID={min_all['id_all']:.4f}"
        )
        print(
            f"Max ID all: layer={int(max_all['layer'])}, "
            f"ID={max_all['id_all']:.4f}"
        )

    # Plots
    fig_gap = os.path.join(
        args.output_dir,
        f"probe_auc_vs_id_gap_{args.aggregation}.png",
    )
    plot_two_curves(
        merged=merged,
        x_col="layer",
        y1_col=auc_col,
        y2_col="gap_hard_minus_easy",
        y1_label="Probe ROC-AUC",
        y2_label="ID Gap (Hard - Easy)",
        title=f"Probe AUC vs ID Gap | {args.aggregation}",
        out_path=fig_gap,
    )

    if "id_all" in merged.columns:
        fig_profile = os.path.join(
            args.output_dir,
            f"probe_auc_vs_id_profile_{args.aggregation}.png",
        )
        plot_two_curves(
            merged=merged,
            x_col="layer",
            y1_col=auc_col,
            y2_col="id_all",
            y1_label="Probe ROC-AUC",
            y2_label="ID all",
            title=f"Probe AUC vs ID Profile | {args.aggregation}",
            out_path=fig_profile,
        )

    fig_scatter_gap = os.path.join(
        args.output_dir,
        f"probe_auc_vs_id_gap_scatter_{args.aggregation}.png",
    )
    plot_scatter(
        merged=merged,
        x_col="gap_hard_minus_easy",
        y_col=auc_col,
        title=f"Probe AUC vs ID Gap Scatter | {args.aggregation}",
        xlabel="ID Gap (Hard - Easy)",
        ylabel="Probe ROC-AUC",
        out_path=fig_scatter_gap,
    )

    if "id_all" in merged.columns:
        fig_scatter_all = os.path.join(
            args.output_dir,
            f"probe_auc_vs_id_all_scatter_{args.aggregation}.png",
        )
        plot_scatter(
            merged=merged,
            x_col="id_all",
            y_col=auc_col,
            title=f"Probe AUC vs ID all Scatter | {args.aggregation}",
            xlabel="ID all",
            ylabel="Probe ROC-AUC",
            out_path=fig_scatter_all,
        )

    print()
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()