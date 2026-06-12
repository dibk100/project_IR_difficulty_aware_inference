import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

"""
python a_effective_rank/plot_er_results.py \
  --input_csv output_phi_1000/effective_rank/layerwise_er_gap.csv \
  --output_dir output_phi_1000/effective_rank/figures

"""
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to layerwise_er_gap.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save figures",
    )

    return parser.parse_args()


def plot_er_curve(df, er_type, output_dir):
    sub = df[df["er_type"] == er_type].sort_values("layer")

    plt.figure(figsize=(10, 6))
    plt.plot(sub["layer"], sub["easy_mean"], marker="o", label="Easy")
    plt.plot(sub["layer"], sub["hard_mean"], marker="o", label="Hard")

    plt.xlabel("Layer")
    plt.ylabel("Effective Rank")
    plt.title(f"Layer-wise Effective Rank ({er_type})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{er_type}_easy_hard_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def plot_gap_curve(df, er_type, output_dir):
    sub = df[df["er_type"] == er_type].sort_values("layer")

    plt.figure(figsize=(10, 6))
    plt.plot(
        sub["layer"],
        sub["gap_hard_minus_easy"],
        marker="o",
        label="Hard - Easy",
    )
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Layer")
    plt.ylabel("ER Gap (Hard - Easy)")
    plt.title(f"Layer-wise Effective Rank Gap ({er_type})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{er_type}_gap_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def print_layer_table(df, er_type):
    sub = df[df["er_type"] == er_type].sort_values("layer")

    cols = [
        "layer",
        "easy_mean",
        "hard_mean",
        "gap_hard_minus_easy",
        "cohen_d",
        "welch_p",
    ]

    print()
    print("=" * 80)
    print(f"[{er_type}] Layer-wise values")
    print("=" * 80)
    print(sub[cols].to_string(index=False))

    print()
    print(f"[{er_type}] Gap summary")
    print("min gap:", sub["gap_hard_minus_easy"].min())
    print("max gap:", sub["gap_hard_minus_easy"].max())
    print("mean gap:", sub["gap_hard_minus_easy"].mean())
    print("best gap layer:", sub.loc[sub["gap_hard_minus_easy"].idxmax(), "layer"])
    print("best Cohen's d layer:", sub.loc[sub["cohen_d"].idxmax(), "layer"])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    print("Input:", args.input_csv)
    print("Shape:", df.shape)
    print("ER types:", df["er_type"].unique())

    for er_type in ["er_raw", "er_centered"]:
        print_layer_table(df, er_type)
        plot_er_curve(df, er_type, args.output_dir)
        plot_gap_curve(df, er_type, args.output_dir)

    print()
    print("Done.")


if __name__ == "__main__":
    main()