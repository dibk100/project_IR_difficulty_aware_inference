# python plot_layer_auc.py

import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "layerwise_probe_results.csv"
OUTPUT_FIG = "layerwise_roc_auc.png"


def main():
    df = pd.read_csv(INPUT_CSV)

    required_cols = {"layer", "aggregation", "roc_auc_mean", "roc_auc_std"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.sort_values(["aggregation", "layer"])

    plt.figure(figsize=(10, 6))

    for aggregation in ["last", "mean"]:
        sub = df[df["aggregation"] == aggregation]

        if sub.empty:
            continue

        plt.plot(
            sub["layer"],
            sub["roc_auc_mean"],
            marker="o",
            label=aggregation,
        )

        plt.fill_between(
            sub["layer"],
            sub["roc_auc_mean"] - sub["roc_auc_std"],
            sub["roc_auc_mean"] + sub["roc_auc_std"],
            alpha=0.15,
        )

    plt.axhline(
        y=0.5,
        linestyle="--",
        linewidth=1,
        label="random baseline",
    )

    plt.xlabel("Layer")
    plt.ylabel("ROC-AUC")
    plt.title("Layer-wise Difficulty Decodability")
    plt.ylim(0.45, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_FIG, dpi=300)
    print("Saved:", OUTPUT_FIG)


if __name__ == "__main__":
    main()