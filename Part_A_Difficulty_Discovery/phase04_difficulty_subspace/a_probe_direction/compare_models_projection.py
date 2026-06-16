
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


"""
Phase04: Compare Difficulty Projection Scores Across Models

Inputs:
    projection_score_analysis.csv for multiple models

Outputs:
    model_projection_summary.csv
    model_projection_auc_curves_<aggregation>.png
    model_projection_gap_curves_<aggregation>.png
    
python a_probe_direction/compare_models_projection.py \
  --phi_csv output_phi_1000/probe_direction/projection_score_analysis.csv \
  --llama_csv output_llama_1000/probe_direction/projection_score_analysis.csv \
  --output_dir output_compare/probe_direction \
  --overwrite
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phi_csv",
        type=str,
        required=True,
        help="Path to Phi projection_score_analysis.csv",
    )

    parser.add_argument(
        "--llama_csv",
        type=str,
        required=True,
        help="Path to Llama projection_score_analysis.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save comparison outputs",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_model_csv(path, model_name):
    df = pd.read_csv(path)
    df["model"] = model_name
    return df


def summarize_best_layers(df):
    rows = []

    for model in sorted(df["model"].unique()):
        for aggregation in sorted(df["aggregation"].unique()):
            sub = df[
                (df["model"] == model)
                & (df["aggregation"] == aggregation)
            ].copy()

            if len(sub) == 0:
                continue

            best = sub.loc[sub["auc_hard_positive"].idxmax()]
            best_inv = sub.loc[sub["auc_direction_invariant"].idxmax()]

            rows.append(
                {
                    "model": model,
                    "aggregation": aggregation,
                    "best_layer_hard_positive": int(best["layer"]),
                    "best_auc_hard_positive": float(best["auc_hard_positive"]),
                    "best_layer_direction_invariant": int(best_inv["layer"]),
                    "best_auc_direction_invariant": float(
                        best_inv["auc_direction_invariant"]
                    ),
                    "best_gap_hard_minus_easy": float(best["gap_hard_minus_easy"]),
                    "best_cohen_d": float(best["cohen_d"]),
                    "best_macro_note": "higher score = harder if hard-positive AUC is used",
                }
            )

    return pd.DataFrame(rows)


def plot_auc_curves(df, aggregation, output_dir):
    sub = df[df["aggregation"] == aggregation].copy()

    plt.figure(figsize=(10, 6))

    for model in sorted(sub["model"].unique()):
        m = sub[sub["model"] == model].sort_values("layer")

        plt.plot(
            m["layer"],
            m["auc_hard_positive"],
            marker="o",
            label=f"{model} hard-positive AUC",
        )

    plt.axhline(0.5, linestyle="--", linewidth=1)

    plt.xlabel("Layer")
    plt.ylabel("ROC-AUC")
    plt.title(f"Difficulty Projection AUC Comparison | {aggregation}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        output_dir,
        f"model_projection_auc_curves_{aggregation}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def plot_gap_curves(df, aggregation, output_dir):
    sub = df[df["aggregation"] == aggregation].copy()

    plt.figure(figsize=(10, 6))

    for model in sorted(sub["model"].unique()):
        m = sub[sub["model"] == model].sort_values("layer")

        plt.plot(
            m["layer"],
            m["gap_hard_minus_easy"],
            marker="o",
            label=f"{model} Hard - Easy",
        )

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Layer")
    plt.ylabel("Projection Score Gap")
    plt.title(f"Difficulty Projection Gap Comparison | {aggregation}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        output_dir,
        f"model_projection_gap_curves_{aggregation}.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary_csv = os.path.join(args.output_dir, "model_projection_summary.csv")

    if os.path.exists(summary_csv) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {summary_csv}\n"
            f"Use --overwrite to overwrite."
        )

    phi = load_model_csv(args.phi_csv, "Phi-3.5-mini")
    llama = load_model_csv(args.llama_csv, "Llama-3.1-8B")

    df = pd.concat([phi, llama], ignore_index=True)

    required_cols = {
        "model",
        "aggregation",
        "layer",
        "auc_hard_positive",
        "auc_direction_invariant",
        "gap_hard_minus_easy",
        "cohen_d",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=" * 80)
    print("[Phase04] Compare Projection Scores Across Models")
    print("=" * 80)
    print("Phi CSV:", args.phi_csv)
    print("Llama CSV:", args.llama_csv)
    print("Output dir:", args.output_dir)
    print()
    print("Combined shape:", df.shape)

    summary = summarize_best_layers(df)
    summary.to_csv(summary_csv, index=False)

    print()
    print("Saved:", summary_csv)
    print()
    print("[Best layers by model/aggregation]")
    print(summary.to_string(index=False))

    for aggregation in sorted(df["aggregation"].unique()):
        plot_auc_curves(df, aggregation, args.output_dir)
        plot_gap_curves(df, aggregation, args.output_dir)

    full_csv = os.path.join(args.output_dir, "model_projection_all_layers.csv")
    df.to_csv(full_csv, index=False)
    print("Saved:", full_csv)

    print()
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()