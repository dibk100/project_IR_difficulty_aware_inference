import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


"""
Phase03-A: Length Control Analysis

Goal:
    Check whether Effective Rank differences between Easy/Hard
    are merely caused by token length.

Analyses:
    1. Token length difference between Easy and Hard
    2. Correlation between token_len and ER
    3. Logistic regression:
        length only vs ER only vs length + ER
    4. Residual ER analysis:
        ER residual after regressing out token_len
    5. Layer-wise residual ER AUC
    
python a_effective_rank/length_control_analysis.py \
  --input_csv output_phi_1000/effective_rank/sample_layer_effective_rank.csv \
  --output_dir output_phi_1000/effective_rank/length_control \
  --er_type er_centered \
  --overwrite
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to sample_layer_effective_rank.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save length-control outputs",
    )

    parser.add_argument(
        "--er_type",
        type=str,
        default="er_centered",
        choices=["er_raw", "er_centered"],
        help="Which ER value to analyze",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )

    return parser.parse_args()


def safe_auc(y, score):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, score)


def fit_logistic_auc(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_scaled, y)

    score = clf.predict_proba(X_scaled)[:, 1]
    return roc_auc_score(y, score)


def cohen_d(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan

    return float((np.mean(y) - np.mean(x)) / np.sqrt(pooled))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    required_cols = {"index", "label", "layer", "token_len", args.er_type}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=" * 80)
    print("[Phase03-A] Length Control Analysis")
    print("=" * 80)
    print("Input:", args.input_csv)
    print("Output dir:", args.output_dir)
    print("ER type:", args.er_type)
    print("Shape:", df.shape)

    # sample-level length check
    sample_df = df[["index", "label", "token_len"]].drop_duplicates("index").copy()
    sample_df["y"] = (sample_df["label"] == "hard").astype(int)

    easy_len = sample_df[sample_df["label"] == "easy"]["token_len"].values
    hard_len = sample_df[sample_df["label"] == "hard"]["token_len"].values

    t_stat, p_value = ttest_ind(easy_len, hard_len, equal_var=False)

    length_summary = {
        "n_easy": len(easy_len),
        "n_hard": len(hard_len),
        "easy_token_len_mean": float(np.mean(easy_len)),
        "hard_token_len_mean": float(np.mean(hard_len)),
        "gap_hard_minus_easy": float(np.mean(hard_len) - np.mean(easy_len)),
        "easy_token_len_std": float(np.std(easy_len, ddof=1)),
        "hard_token_len_std": float(np.std(hard_len, ddof=1)),
        "cohen_d": cohen_d(easy_len, hard_len),
        "welch_t": float(t_stat),
        "welch_p": float(p_value),
        "length_only_auc": safe_auc(sample_df["y"].values, sample_df["token_len"].values),
    }

    pd.DataFrame([length_summary]).to_csv(
        os.path.join(args.output_dir, "token_length_summary.csv"),
        index=False,
    )

    print()
    print("[Token length summary]")
    for k, v in length_summary.items():
        print(f"{k}: {v}")

    rows = []
    residual_rows = []

    for layer in sorted(df["layer"].unique()):
        sub = df[df["layer"] == layer].copy()

        y = (sub["label"] == "hard").astype(int).values
        er = sub[args.er_type].values.astype(float)
        length = sub["token_len"].values.astype(float)

        # correlations
        pearson_r, pearson_p = pearsonr(length, er)
        spearman_rho, spearman_p = spearmanr(length, er)

        # AUCs
        auc_length_only = safe_auc(y, length)
        auc_er_only = safe_auc(y, er)

        X_len = length.reshape(-1, 1)
        X_er = er.reshape(-1, 1)
        X_both = np.column_stack([length, er])

        logit_auc_length = fit_logistic_auc(X_len, y)
        logit_auc_er = fit_logistic_auc(X_er, y)
        logit_auc_length_er = fit_logistic_auc(X_both, y)

        # residual ER after removing token_len effect
        linreg = LinearRegression()
        linreg.fit(X_len, er)
        er_pred = linreg.predict(X_len)
        er_resid = er - er_pred

        auc_resid = safe_auc(y, er_resid)

        easy_resid = er_resid[sub["label"].values == "easy"]
        hard_resid = er_resid[sub["label"].values == "hard"]
        resid_gap = float(np.mean(hard_resid) - np.mean(easy_resid))
        resid_d = cohen_d(easy_resid, hard_resid)
        resid_t, resid_p = ttest_ind(easy_resid, hard_resid, equal_var=False)

        rows.append(
            {
                "layer": int(layer),
                "pearson_len_er": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_len_er": float(spearman_rho),
                "spearman_p": float(spearman_p),
                "auc_length_only": float(auc_length_only),
                "auc_er_only": float(auc_er_only),
                "logit_auc_length_only": float(logit_auc_length),
                "logit_auc_er_only": float(logit_auc_er),
                "logit_auc_length_plus_er": float(logit_auc_length_er),
                "auc_er_residual_after_length": float(auc_resid),
                "residual_gap_hard_minus_easy": resid_gap,
                "residual_cohen_d": resid_d,
                "residual_welch_t": float(resid_t),
                "residual_welch_p": float(resid_p),
                "length_coef_for_er": float(linreg.coef_[0]),
                "length_intercept_for_er": float(linreg.intercept_),
            }
        )

        for i, idx in enumerate(sub["index"].values):
            residual_rows.append(
                {
                    "index": int(idx),
                    "label": sub.iloc[i]["label"],
                    "layer": int(layer),
                    "token_len": float(length[i]),
                    args.er_type: float(er[i]),
                    "er_residual_after_length": float(er_resid[i]),
                }
            )

    result_df = pd.DataFrame(rows)
    result_path = os.path.join(args.output_dir, "length_control_layerwise.csv")
    result_df.to_csv(result_path, index=False)

    residual_df = pd.DataFrame(residual_rows)
    residual_path = os.path.join(args.output_dir, "sample_layer_er_residual_after_length.csv")
    residual_df.to_csv(residual_path, index=False)

    print()
    print("Saved:", result_path)
    print("Saved:", residual_path)

    print()
    print("[Layer-wise length control summary]")
    print(
        result_df[
            [
                "layer",
                "pearson_len_er",
                "auc_er_only",
                "auc_er_residual_after_length",
                "logit_auc_length_only",
                "logit_auc_er_only",
                "logit_auc_length_plus_er",
                "residual_gap_hard_minus_easy",
                "residual_cohen_d",
            ]
        ].to_string(index=False)
    )

    # Plot 1: AUC comparison
    plt.figure(figsize=(10, 6))
    plt.plot(result_df["layer"], result_df["auc_er_only"], marker="o", label="ER only")
    plt.plot(
        result_df["layer"],
        result_df["auc_er_residual_after_length"],
        marker="o",
        label="ER residual after length",
    )
    plt.plot(
        result_df["layer"],
        result_df["logit_auc_length_only"],
        marker="o",
        label="Length only",
    )
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("ROC-AUC")
    plt.title(f"Length-controlled ER Signal AUC ({args.er_type})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(args.output_dir, "length_control_auc_curve.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)

    # Plot 2: length-ER correlation
    plt.figure(figsize=(10, 6))
    plt.plot(
        result_df["layer"],
        result_df["pearson_len_er"],
        marker="o",
        label="Pearson(token_len, ER)",
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("Correlation")
    plt.title(f"Token Length vs Effective Rank Correlation ({args.er_type})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(args.output_dir, "token_length_er_correlation.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)

    print("=" * 80)


if __name__ == "__main__":
    main()