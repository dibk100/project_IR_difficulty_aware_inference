import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


"""
Phase03-B: Plot Layer-wise Representation Delta Results

Inputs:
    sample_layer_delta.csv
    layerwise_delta_gap.csv
    delta_signal_auc.csv

Outputs:
    Easy/Hard delta curve
    Hard-Easy gap curve
    AUC curve

For each aggregation and metric:
    - last / mean
    - delta_l2 / delta_l2_normed / delta_cosine
    
python b_layerwise_representation/plot_delta_results.py \
  --delta_csv output_phi_1000/layerwise_representation_change/sample_layer_delta.csv \
  --gap_csv output_phi_1000/layerwise_representation_change/layerwise_delta_gap.csv \
  --auc_csv output_phi_1000/layerwise_representation_change/delta_signal_auc.csv \
  --output_dir output_phi_1000/layerwise_representation_change/figures
  
delta_easy_hard_mean_delta_cosine.png
delta_gap_mean_delta_cosine.png
delta_auc_mean_delta_l2_normed.png
"""


DELTA_METRICS = [
    "delta_l2",
    "delta_l2_normed",
    "delta_cosine",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--delta_csv",
        type=str,
        required=True,
        help="Path to sample_layer_delta.csv.",
    )

    parser.add_argument(
        "--gap_csv",
        type=str,
        required=True,
        help="Path to layerwise_delta_gap.csv.",
    )

    parser.add_argument(
        "--auc_csv",
        type=str,
        required=True,
        help="Path to delta_signal_auc.csv.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save figures.",
    )

    return parser.parse_args()


def plot_easy_hard_curve(gap_df, aggregation, metric, output_dir):
    sub = gap_df[
        (gap_df["aggregation"] == aggregation)
        & (gap_df["metric"] == metric)
    ].sort_values("layer")

    plt.figure(figsize=(10, 6))

    plt.plot(
        sub["layer"],
        sub["easy_mean"],
        marker="o",
        label="Easy",
    )

    plt.plot(
        sub["layer"],
        sub["hard_mean"],
        marker="o",
        label="Hard",
    )

    plt.xlabel("Layer")
    plt.ylabel(metric)
    plt.title(f"Layer-wise Delta Curve | {aggregation} | {metric}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        output_dir,
        f"delta_easy_hard_{aggregation}_{metric}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def plot_gap_curve(gap_df, aggregation, metric, output_dir):
    sub = gap_df[
        (gap_df["aggregation"] == aggregation)
        & (gap_df["metric"] == metric)
    ].sort_values("layer")

    plt.figure(figsize=(10, 6))

    plt.plot(
        sub["layer"],
        sub["gap_hard_minus_easy"],
        marker="o",
        label="Hard - Easy",
    )

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Layer")
    plt.ylabel("Gap")
    plt.title(f"Layer-wise Delta Gap | {aggregation} | {metric}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        output_dir,
        f"delta_gap_{aggregation}_{metric}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def plot_auc_curve(auc_df, aggregation, metric, output_dir):
    sub = auc_df[
        (auc_df["aggregation"] == aggregation)
        & (auc_df["metric"] == metric)
    ].sort_values("layer")

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

    out_path = os.path.join(
        output_dir,
        f"delta_auc_{aggregation}_{metric}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def print_summary(gap_df, auc_df):
    print()
    print("=" * 80)
    print("[Summary: best AUC by aggregation/metric]")
    print("=" * 80)

    for aggregation in sorted(auc_df["aggregation"].unique()):
        for metric in DELTA_METRICS:
            sub_auc = auc_df[
                (auc_df["aggregation"] == aggregation)
                & (auc_df["metric"] == metric)
            ].copy()

            sub_gap = gap_df[
                (gap_df["aggregation"] == aggregation)
                & (gap_df["metric"] == metric)
            ].copy()

            if len(sub_auc) == 0:
                continue

            best_auc = sub_auc.loc[sub_auc["auc_hard_positive"].idxmax()]
            best_abs = sub_auc.loc[sub_auc["auc_direction_invariant"].idxmax()]
            best_gap = sub_gap.loc[sub_gap["gap_hard_minus_easy"].idxmax()]

            print()
            print(f"aggregation={aggregation}, metric={metric}")
            print(
                f"  Best hard-positive AUC: "
                f"layer={int(best_auc['layer'])}, "
                f"AUC={best_auc['auc_hard_positive']:.4f}, "
                f"gap={best_auc['gap_hard_minus_easy']:.6f}"
            )
            print(
                f"  Best direction-invariant AUC: "
                f"layer={int(best_abs['layer'])}, "
                f"AUC={best_abs['auc_direction_invariant']:.4f}, "
                f"hard-positive AUC={best_abs['auc_hard_positive']:.4f}"
            )
            print(
                f"  Best positive gap: "
                f"layer={int(best_gap['layer'])}, "
                f"gap={best_gap['gap_hard_minus_easy']:.6f}, "
                f"cohen_d={best_gap['cohen_d']:.4f}"
            )


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    delta_df = pd.read_csv(args.delta_csv)
    gap_df = pd.read_csv(args.gap_csv)
    auc_df = pd.read_csv(args.auc_csv)

    print("=" * 80)
    print("[Phase03-B] Plot Delta Results")
    print("=" * 80)
    print("Delta CSV:", args.delta_csv)
    print("Gap CSV:", args.gap_csv)
    print("AUC CSV:", args.auc_csv)
    print("Output dir:", args.output_dir)

    print()
    print("Delta shape:", delta_df.shape)
    print("Gap shape:", gap_df.shape)
    print("AUC shape:", auc_df.shape)

    print_summary(gap_df, auc_df)

    for aggregation in sorted(gap_df["aggregation"].unique()):
        for metric in DELTA_METRICS:
            plot_easy_hard_curve(gap_df, aggregation, metric, args.output_dir)
            plot_gap_curve(gap_df, aggregation, metric, args.output_dir)
            plot_auc_curve(auc_df, aggregation, metric, args.output_dir)

    print()
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()