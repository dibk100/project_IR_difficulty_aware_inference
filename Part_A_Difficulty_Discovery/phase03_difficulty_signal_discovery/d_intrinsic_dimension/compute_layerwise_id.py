import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


"""
Phase03-D: Layer-wise Intrinsic Dimension Analysis

Estimator:
    TwoNN intrinsic dimension estimator

Input:
    Phase02 layerwise hidden states npz
    gsm8k_layerwise_hidden_states.npz

Required keys:
    labels
    ids
    layer01_last ... layer32_last
    layer01_mean ... layer32_mean

Output:
    layerwise_id_profile.csv

Rows:
    aggregation, group, layer, n_samples, id_twonn
    
python d_intrinsic_dimension/compute_layerwise_id.py \
  --input_npz ../phase02_layerwise_decodability/output_llama_1000/gsm8k_layerwise_hidden_states.npz \
  --output_csv output_llama_1000/intrinsic_dimension/layerwise_id_profile.csv \
  --overwrite
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_npz",
        type=str,
        required=True,
        help="Path to gsm8k_layerwise_hidden_states.npz",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save layerwise_id_profile.csv",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional max samples per group. Use all samples if None.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def infer_num_layers(npz_data, aggregation):
    layers = []

    for key in npz_data.files:
        if key.startswith("layer") and key.endswith(f"_{aggregation}"):
            layer_num = int(
                key.replace("layer", "").replace(f"_{aggregation}", "")
            )
            layers.append(layer_num)

    if len(layers) == 0:
        raise ValueError(f"No layer keys found for aggregation={aggregation}")

    return max(layers)


def standardize_rows(X, eps=1e-12):
    """
    Feature-wise standardization across samples.
    X: (N, D)
    """
    X = X.astype(np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / (std + eps)


def twonn_id(X, eps=1e-12):
    """
    TwoNN estimator.

    Reference idea:
        For each point, compute distances to 1st and 2nd nearest neighbors.
        mu_i = r2_i / r1_i
        ID = 1 / mean(log(mu_i))

    X: (N, D)
    """
    if X.shape[0] < 10:
        return np.nan

    # NearestNeighbors includes the point itself as the first neighbor.
    # So we need 3 neighbors: self, nearest, second nearest.
    nn = NearestNeighbors(
        n_neighbors=3,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )
    nn.fit(X)

    distances, _ = nn.kneighbors(X)

    r1 = distances[:, 1]
    r2 = distances[:, 2]

    valid = (r1 > eps) & (r2 > eps) & (r2 > r1)

    if valid.sum() < 10:
        return np.nan

    mu = r2[valid] / (r1[valid] + eps)
    log_mu = np.log(mu + eps)

    mean_log_mu = np.mean(log_mu)

    if mean_log_mu <= eps:
        return np.nan

    return float(1.0 / mean_log_mu)


def maybe_subsample(X, max_samples, rng):
    if max_samples is None:
        return X

    if X.shape[0] <= max_samples:
        return X

    idx = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[idx]


def compute_group_id(X, max_samples, rng, standardize=True):
    X_sub = maybe_subsample(X, max_samples, rng)

    if standardize:
        X_sub = standardize_rows(X_sub)

    return twonn_id(X_sub)


def main():
    args = parse_args()

    if os.path.exists(args.output_csv) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_csv}\n"
            f"Use --overwrite to overwrite."
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    rng = np.random.default_rng(args.random_seed)

    print("=" * 80)
    print("[Phase03-D] Compute Layer-wise Intrinsic Dimension")
    print("=" * 80)
    print("Input NPZ:", args.input_npz)
    print("Output CSV:", args.output_csv)
    print("Estimator: TwoNN")
    print("Max samples:", args.max_samples)
    print("Random seed:", args.random_seed)

    data = np.load(args.input_npz, allow_pickle=True)

    labels = data["labels"]
    ids = data["ids"]

    print()
    print("Number of samples:", len(labels))
    print("Label counts:")
    print(pd.Series(labels).value_counts())

    rows = []

    for aggregation in ["last", "mean"]:
        num_layers = infer_num_layers(data, aggregation)

        print()
        print("=" * 80)
        print(f"Aggregation: {aggregation}")
        print("Number of layers:", num_layers)
        print("=" * 80)

        for layer in tqdm(range(1, num_layers + 1)):
            key = f"layer{layer:02d}_{aggregation}"

            if key not in data.files:
                raise KeyError(f"Missing key: {key}")

            X_all = data[key].astype(np.float32)

            X_easy = X_all[labels == "easy"]
            X_hard = X_all[labels == "hard"]

            group_data = {
                "all": X_all,
                "easy": X_easy,
                "hard": X_hard,
            }

            for group_name, X_group in group_data.items():
                id_value = compute_group_id(
                    X_group,
                    max_samples=args.max_samples,
                    rng=rng,
                    standardize=True,
                )

                rows.append(
                    {
                        "aggregation": aggregation,
                        "group": group_name,
                        "layer": int(layer),
                        "n_samples": int(X_group.shape[0]),
                        "hidden_dim": int(X_group.shape[1]),
                        "id_twonn": id_value,
                        "estimator": "TwoNN",
                        "standardized": True,
                    }
                )

    out = pd.DataFrame(rows)
    out = out.sort_values(["aggregation", "group", "layer"]).reset_index(drop=True)
    out.to_csv(args.output_csv, index=False)

    print()
    print("=" * 80)
    print("Done.")
    print("Saved:", args.output_csv)
    print("Shape:", out.shape)
    print()
    print(out.head(12))
    print("=" * 80)


if __name__ == "__main__":
    main()