import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
Phase03-D: Analyze Layer-wise Intrinsic Dimension Profile

Input:
    layerwise_id_profile.csv

Output:
    layerwise_id_gap.csv
    id_profile_summary.csv
    figures/
        id_profile_<aggregation>.png
        id_gap_<aggregation>.png

Purpose:
    1. Plot ID profile for all/easy/hard groups.
    2. Compute Hard - Easy ID gap.
    3. Identify ID peak/minimum layers.
    
python d_intrinsic_dimension/analyze_id_profile.py \
  --input_csv output_phi_1000/intrinsic_dimension/layerwise_id_profile.csv \
  --output_gap_csv output_phi_1000/intrinsic_dimension/layerwise_id_gap.csv \
  --output_summary_csv output_phi_1000/intrinsic_dimension/id_profile_summary.csv \
  --figure_dir output_phi_1000/intrinsic_dimension/figures \
  --overwrite

"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to layerwise_id_profile.csv.",
    )

    parser.add_argument(
        "--output_gap_csv",
        type=str,
        required=True,
        help="Path to save layerwise_id_gap.csv.",
    )

    parser.add_argument(
        "--output_summary_csv",
        type=str,
        required=True,
        help="Path to save id_profile_summary.csv.",
    )

    parser.add_argument(
        "--figure_dir",
        type=str,
        required=True,
        help="Directory to save figures.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def check_overwrite(path, overwrite):
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}\n"
            f"Use --overwrite to overwrite."
        )


def local_minima_layers(layers, values):
    minima = []

    layers = np.asarray(layers)
    values = np.asarray(values)

    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            minima.append((int(layers[i]), float(values[i])))

    return minima


def local_maxima_layers(layers, values):
    maxima = []

    layers = np.asarray(layers)
    values = np.asarray(values)

    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            maxima.append((int(layers[i]), float(values[i])))

    return maxima


def plot_id_profile(df, aggregation, figure_dir):
    sub = df[df["aggregation"] == aggregation].copy()

    plt.figure(figsize=(10, 6))

    for group in ["all", "easy", "hard"]:
        g = sub[sub["group"] == group].sort_values("layer")
        if len(g) == 0:
            continue

        plt.plot(
            g["layer"],
            g["id_twonn"],
            marker="o",
            label=group,
        )

    plt.xlabel("Layer")
    plt.ylabel("TwoNN Intrinsic Dimension")
    plt.title(f"Layer-wise Intrinsic Dimension Profile | {aggregation}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        figure_dir,
        f"id_profile_{aggregation}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def plot_id_gap(gap_df, aggregation, figure_dir):
    sub = gap_df[gap_df["aggregation"] == aggregation].copy()
    sub = sub.sort_values("layer")

    plt.figure(figsize=(10, 6))

    plt.plot(
        sub["layer"],
        sub["gap_hard_minus_easy"],
        marker="o",
        label="Hard - Easy ID",
    )

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Layer")
    plt.ylabel("ID Gap")
    plt.title(f"Hard - Easy Intrinsic Dimension Gap | {aggregation}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        figure_dir,
        f"id_gap_{aggregation}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def main():
    args = parse_args()

    check_overwrite(args.output_gap_csv, args.overwrite)
    check_overwrite(args.output_summary_csv, args.overwrite)

    os.makedirs(os.path.dirname(args.output_gap_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_summary_csv), exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    required_cols = {
        "aggregation",
        "group",
        "layer",
        "n_samples",
        "hidden_dim",
        "id_twonn",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=" * 80)
    print("[Phase03-D] Analyze Intrinsic Dimension Profile")
    print("=" * 80)
    print("Input:", args.input_csv)
    print("Output gap:", args.output_gap_csv)
    print("Output summary:", args.output_summary_csv)
    print("Figure dir:", args.figure_dir)
    print("Shape:", df.shape)
    print()
    print("Aggregations:", sorted(df["aggregation"].unique()))
    print("Groups:", sorted(df["group"].unique()))
    print("Layers:", int(df["layer"].min()), "to", int(df["layer"].max()))

    gap_rows = []
    summary_rows = []

    for aggregation in sorted(df["aggregation"].unique()):
        sub = df[df["aggregation"] == aggregation].copy()

        # Summary for each group
        for group in sorted(sub["group"].unique()):
            g = sub[sub["group"] == group].sort_values("layer").copy()

            layers = g["layer"].values
            ids = g["id_twonn"].values

            global_min_idx = int(np.nanargmin(ids))
            global_max_idx = int(np.nanargmax(ids))

            minima = local_minima_layers(layers, ids)
            maxima = local_maxima_layers(layers, ids)

            summary_rows.append(
                {
                    "aggregation": aggregation,
                    "group": group,
                    "global_min_layer": int(layers[global_min_idx]),
                    "global_min_id": float(ids[global_min_idx]),
                    "global_max_layer": int(layers[global_max_idx]),
                    "global_max_id": float(ids[global_max_idx]),
                    "first_local_min_layer": (
                        minima[0][0] if len(minima) > 0 else np.nan
                    ),
                    "first_local_min_id": (
                        minima[0][1] if len(minima) > 0 else np.nan
                    ),
                    "first_local_max_layer": (
                        maxima[0][0] if len(maxima) > 0 else np.nan
                    ),
                    "first_local_max_id": (
                        maxima[0][1] if len(maxima) > 0 else np.nan
                    ),
                    "num_local_minima": len(minima),
                    "num_local_maxima": len(maxima),
                }
            )

        # Easy/Hard gap
        easy = sub[sub["group"] == "easy"][
            ["layer", "id_twonn", "n_samples"]
        ].rename(
            columns={
                "id_twonn": "easy_id",
                "n_samples": "n_easy",
            }
        )

        hard = sub[sub["group"] == "hard"][
            ["layer", "id_twonn", "n_samples"]
        ].rename(
            columns={
                "id_twonn": "hard_id",
                "n_samples": "n_hard",
            }
        )

        merged = easy.merge(hard, on="layer", how="inner")
        merged["aggregation"] = aggregation
        merged["gap_hard_minus_easy"] = (
            merged["hard_id"] - merged["easy_id"]
        )
        merged["abs_gap"] = merged["gap_hard_minus_easy"].abs()

        for _, row in merged.iterrows():
            gap_rows.append(
                {
                    "aggregation": aggregation,
                    "layer": int(row["layer"]),
                    "n_easy": int(row["n_easy"]),
                    "n_hard": int(row["n_hard"]),
                    "easy_id": float(row["easy_id"]),
                    "hard_id": float(row["hard_id"]),
                    "gap_hard_minus_easy": float(
                        row["gap_hard_minus_easy"]
                    ),
                    "abs_gap": float(row["abs_gap"]),
                }
            )

    gap_df = pd.DataFrame(gap_rows)
    gap_df = gap_df.sort_values(["aggregation", "layer"]).reset_index(drop=True)
    gap_df.to_csv(args.output_gap_csv, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["aggregation", "group"]
    ).reset_index(drop=True)
    summary_df.to_csv(args.output_summary_csv, index=False)

    print()
    print("Saved:", args.output_gap_csv)
    print("Gap shape:", gap_df.shape)
    print("Saved:", args.output_summary_csv)
    print("Summary shape:", summary_df.shape)

    print()
    print("=" * 80)
    print("[Profile Summary]")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print()
    print("=" * 80)
    print("[Top absolute Easy/Hard ID gaps]")
    print("=" * 80)
    print(
        gap_df.sort_values("abs_gap", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    for aggregation in sorted(df["aggregation"].unique()):
        plot_id_profile(df, aggregation, args.figure_dir)
        plot_id_gap(gap_df, aggregation, args.figure_dir)

    print()
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()