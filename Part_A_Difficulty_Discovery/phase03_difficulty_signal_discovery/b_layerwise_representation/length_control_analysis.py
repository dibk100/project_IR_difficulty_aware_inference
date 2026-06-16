import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


"""
Phase03-B: Length Control Analysis for Layer-wise Representation Delta

Goal:
    Check whether delta signals are merely caused by token length.

Input:
    sample_layer_delta.csv

Output:
    length_control_delta.csv
    length_control_delta_auc_curve_*.png

Analyses:
    1. corr(token_len, delta)
    2. length-only AUC
    3. delta-only AUC
    4. length + delta logistic AUC
    5. delta residual after removing token_len
    
    
선행작업
python - <<'PY'
import pandas as pd

delta_path = "output_llama_500/layerwise_representation_change/sample_layer_delta.csv"
er_path = "output_llama_500/effective_rank/sample_layer_effective_rank.csv"
out_path = "output_llama_500/layerwise_representation_change/sample_layer_delta_with_len.csv"

delta = pd.read_csv(delta_path)
er = pd.read_csv(er_path)

length_map = (
    er[["index", "token_len"]]
    .drop_duplicates("index")
)

merged = delta.merge(length_map, on="index", how="left")

if merged["token_len"].isna().any():
    raise ValueError("Some rows have missing token_len after merge.")

merged.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Shape:", merged.shape)
print(merged.head())
PY

python b_layerwise_representation/length_control_analysis.py \
  --input_csv output_llama_500/layerwise_representation_change/sample_layer_delta_with_len.csv \
  --output_csv output_llama_500/layerwise_representation_change/length_control_delta.csv \
  --figure_dir output_llama_500/layerwise_representation_change/figures \
  --overwrite
"""


DELTA_METRICS = [
    "delta_l2",
    "delta_l2_normed",
    "delta_cosine",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to sample_layer_delta.csv.",
    )

    parser.add_argument(
        "--rollout_jsonl",
        type=str,
        required=False,
        default=None,
        help=(
            "Optional path to gsm8k_main_rollouts.jsonl. "
            "If sample_layer_delta.csv does not contain token_len, "
            "token length must be added beforehand or provided separately."
        ),
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save length_control_delta.csv.",
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
        help="Overwrite existing output CSV.",
    )

    return parser.parse_args()


def safe_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, scores)


def logistic_auc(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_scaled, y)

    scores = clf.predict_proba(X_scaled)[:, 1]
    return roc_auc_score(y, scores)


def add_token_len_if_missing(df):
    """
    sample_layer_delta.csv from the current compute script does not include token_len.
    For now, token_len can be approximated only if already present.

    Recommended:
        modify compute_layerwise_delta.py later to include token_len,
        or merge token_len from effective-rank sample_layer_effective_rank.csv.

    This function intentionally fails clearly if token_len is absent.
    """
    if "token_len" not in df.columns:
        raise ValueError(
            "Column 'token_len' is missing from sample_layer_delta.csv.\n\n"
            "Recommended fix:\n"
            "1) Merge token_len from Phase03-A effective_rank/sample_layer_effective_rank.csv, or\n"
            "2) Modify compute_layerwise_delta.py to include token_len.\n\n"
            "Expected columns include: token_len, label, aggregation, layer, delta metrics."
        )

    return df


def main():
    args = parse_args()

    if os.path.exists(args.output_csv) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_csv}\n"
            f"Use --overwrite to overwrite."
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = add_token_len_if_missing(df)

    required_cols = {
        "index",
        "label",
        "aggregation",
        "layer",
        "token_len",
        *DELTA_METRICS,
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=" * 80)
    print("[Phase03-B] Length Control Analysis for Delta")
    print("=" * 80)
    print("Input:", args.input_csv)
    print("Output:", args.output_csv)
    print("Figure dir:", args.figure_dir)
    print("Shape:", df.shape)

    print()
    print("Label counts:")
    print(df[["index", "label"]].drop_duplicates()["label"].value_counts())

    sample_df = df[["index", "label", "token_len"]].drop_duplicates("index")
    y_sample = (sample_df["label"] == "hard").astype(int).values
    length_sample = sample_df["token_len"].values.astype(float)
    length_only_auc = safe_auc(y_sample, length_sample)

    print()
    print("Length-only AUC:", length_only_auc)

    rows = []
    residual_rows = []

    for aggregation in sorted(df["aggregation"].unique()):
        df_agg = df[df["aggregation"] == aggregation].copy()

        for metric in DELTA_METRICS:
            for layer in sorted(df_agg["layer"].unique()):
                sub = df_agg[df_agg["layer"] == layer].copy()

                y = (sub["label"] == "hard").astype(int).values
                length = sub["token_len"].values.astype(float)
                delta = sub[metric].values.astype(float)

                pearson_r, pearson_p = pearsonr(length, delta)
                spearman_rho, spearman_p = spearmanr(length, delta)

                auc_length_only = safe_auc(y, length)
                auc_delta_only = safe_auc(y, delta)

                X_len = length.reshape(-1, 1)
                X_delta = delta.reshape(-1, 1)
                X_both = np.column_stack([length, delta])

                logit_auc_length_only = logistic_auc(X_len, y)
                logit_auc_delta_only = logistic_auc(X_delta, y)
                logit_auc_length_plus_delta = logistic_auc(X_both, y)

                # residual delta after removing token length
                linreg = LinearRegression()
                linreg.fit(X_len, delta)
                delta_pred = linreg.predict(X_len)
                delta_resid = delta - delta_pred

                auc_delta_residual = safe_auc(y, delta_resid)

                easy_resid = delta_resid[sub["label"].values == "easy"]
                hard_resid = delta_resid[sub["label"].values == "hard"]

                residual_gap = float(np.mean(hard_resid) - np.mean(easy_resid))

                rows.append(
                    {
                        "aggregation": aggregation,
                        "metric": metric,
                        "layer": int(layer),
                        "pearson_len_delta": float(pearson_r),
                        "pearson_p": float(pearson_p),
                        "spearman_len_delta": float(spearman_rho),
                        "spearman_p": float(spearman_p),
                        "auc_length_only": float(auc_length_only),
                        "auc_delta_only": float(auc_delta_only),
                        "logit_auc_length_only": float(logit_auc_length_only),
                        "logit_auc_delta_only": float(logit_auc_delta_only),
                        "logit_auc_length_plus_delta": float(logit_auc_length_plus_delta),
                        "auc_delta_residual_after_length": float(auc_delta_residual),
                        "residual_gap_hard_minus_easy": residual_gap,
                        "length_coef_for_delta": float(linreg.coef_[0]),
                        "length_intercept_for_delta": float(linreg.intercept_),
                    }
                )

                for i, idx in enumerate(sub["index"].values):
                    residual_rows.append(
                        {
                            "index": int(idx),
                            "label": sub.iloc[i]["label"],
                            "aggregation": aggregation,
                            "metric": metric,
                            "layer": int(layer),
                            "token_len": float(length[i]),
                            "delta": float(delta[i]),
                            "delta_residual_after_length": float(delta_resid[i]),
                        }
                    )

    out = pd.DataFrame(rows)
    out = out.sort_values(["aggregation", "metric", "layer"]).reset_index(drop=True)
    out.to_csv(args.output_csv, index=False)

    residual_path = args.output_csv.replace(".csv", "_residual_samples.csv")
    pd.DataFrame(residual_rows).to_csv(residual_path, index=False)

    print()
    print("Saved:", args.output_csv)
    print("Saved:", residual_path)
    print("Shape:", out.shape)

    print()
    print("[Best residual AUC]")
    print(
        out.sort_values("auc_delta_residual_after_length", ascending=False)
        .head(20)[
            [
                "aggregation",
                "metric",
                "layer",
                "pearson_len_delta",
                "auc_delta_only",
                "auc_delta_residual_after_length",
                "logit_auc_length_only",
                "logit_auc_length_plus_delta",
                "residual_gap_hard_minus_easy",
            ]
        ]
        .to_string(index=False)
    )

    # Plot per aggregation/metric
    for aggregation in sorted(out["aggregation"].unique()):
        for metric in DELTA_METRICS:
            sub = out[
                (out["aggregation"] == aggregation)
                & (out["metric"] == metric)
            ].sort_values("layer")

            plt.figure(figsize=(10, 6))
            plt.plot(
                sub["layer"],
                sub["auc_delta_only"],
                marker="o",
                label="Delta only",
            )
            plt.plot(
                sub["layer"],
                sub["auc_delta_residual_after_length"],
                marker="o",
                label="Delta residual after length",
            )
            plt.plot(
                sub["layer"],
                sub["auc_length_only"],
                marker="o",
                label="Length only",
            )
            plt.axhline(0.5, linestyle="--", linewidth=1)
            plt.xlabel("Layer")
            plt.ylabel("ROC-AUC")
            plt.title(f"Length-controlled Delta AUC | {aggregation} | {metric}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            fig_path = os.path.join(
                args.figure_dir,
                f"length_control_auc_{aggregation}_{metric}.png",
            )
            plt.savefig(fig_path, dpi=300)
            plt.close()
            print("Saved:", fig_path)

            plt.figure(figsize=(10, 6))
            plt.plot(
                sub["layer"],
                sub["pearson_len_delta"],
                marker="o",
                label="Pearson(token_len, delta)",
            )
            plt.axhline(0, linestyle="--", linewidth=1)
            plt.xlabel("Layer")
            plt.ylabel("Correlation")
            plt.title(f"Token Length vs Delta | {aggregation} | {metric}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            fig_path = os.path.join(
                args.figure_dir,
                f"length_delta_corr_{aggregation}_{metric}.png",
            )
            plt.savefig(fig_path, dpi=300)
            plt.close()
            print("Saved:", fig_path)

    print("=" * 80)


if __name__ == "__main__":
    main()