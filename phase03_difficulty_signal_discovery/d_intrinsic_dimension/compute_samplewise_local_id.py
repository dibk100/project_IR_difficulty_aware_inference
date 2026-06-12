import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


"""
Phase03-D: Sample-wise Local Intrinsic Dimension

Goal:
    Convert group-wise ID analysis into sample-wise difficulty signal.

Method:
    For each layer and aggregation:
        X_l = (N, D) hidden representations

    For each sample x_i:
        Find k nearest neighbors.
        Use local TwoNN-style estimator:

            mu_j = r2_j / r1_j

        around local neighborhood.

    Here we use a local neighborhood version:
        For each sample, collect distances to k nearest neighbors.
        Estimate local ID by slope approximation:

            ID_i = 1 / mean(log(r_m / r_1))

        where m = 2..k.

Output:
    sample_layer_local_id.csv

Columns:
    index
    id
    label
    aggregation
    layer
    local_id
    knn_k
    
python d_intrinsic_dimension/compute_samplewise_local_id.py \
  --input_npz ../phase02_layerwise_decodability/output_phi_1000/gsm8k_layerwise_hidden_states.npz \
  --output_csv output_phi_1000/intrinsic_dimension/sample_layer_local_id.csv \
  --k 20 \
  --standardize \
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
        help="Path to save sample_layer_local_id.csv",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of nearest neighbors for local ID estimation",
    )

    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Feature-wise standardization within each layer before kNN",
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


def standardize_features(X, eps=1e-12):
    X = X.astype(np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / (std + eps)


def compute_local_id(X, k=20, eps=1e-12):
    """
    Local intrinsic dimension estimate per sample.

    X: (N, D)

    Returns:
        local_id: (N,)

    NearestNeighbors returns:
        distance[:, 0] = self-distance = 0
        distance[:, 1:] = actual neighbors

    We compute:
        r1 = nearest neighbor distance
        ratios = r_m / r1 for m = 2..k
        local_id = 1 / mean(log(ratios))
    """
    n = X.shape[0]

    if k >= n:
        raise ValueError(
            f"k must be smaller than number of samples. k={k}, n={n}"
        )

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )
    nn.fit(X)

    distances, _ = nn.kneighbors(X)

    # Remove self-distance.
    d = distances[:, 1:]  # (N, k)

    r1 = d[:, [0]]  # (N, 1)
    ratios = d[:, 1:] / (r1 + eps)  # r2..rk / r1

    valid = ratios > 1.0 + eps

    log_ratios = np.where(valid, np.log(ratios + eps), np.nan)
    mean_log = np.nanmean(log_ratios, axis=1)

    local_id = 1.0 / (mean_log + eps)

    # Clean invalid values.
    local_id[~np.isfinite(local_id)] = np.nan

    return local_id


def main():
    args = parse_args()

    if os.path.exists(args.output_csv) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_csv}\n"
            f"Use --overwrite to overwrite."
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    print("=" * 80)
    print("[Phase03-D] Compute Sample-wise Local ID")
    print("=" * 80)
    print("Input NPZ:", args.input_npz)
    print("Output CSV:", args.output_csv)
    print("k:", args.k)
    print("Standardize:", args.standardize)

    data = np.load(args.input_npz, allow_pickle=True)

    labels = data["labels"]
    ids = data["ids"]

    n_samples = len(labels)

    print()
    print("Number of samples:", n_samples)
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

            X = data[key].astype(np.float64)

            if args.standardize:
                X = standardize_features(X)

            local_id = compute_local_id(
                X,
                k=args.k,
            )

            for i in range(n_samples):
                rows.append(
                    {
                        "index": int(i),
                        "id": ids[i],
                        "label": labels[i],
                        "aggregation": aggregation,
                        "layer": int(layer),
                        "local_id": float(local_id[i]),
                        "knn_k": int(args.k),
                        "standardized": bool(args.standardize),
                    }
                )

    out = pd.DataFrame(rows)
    out = out.sort_values(["aggregation", "index", "layer"]).reset_index(drop=True)
    out.to_csv(args.output_csv, index=False)

    print()
    print("=" * 80)
    print("Done.")
    print("Saved:", args.output_csv)
    print("Shape:", out.shape)
    print("Columns:", list(out.columns))
    print()
    print(out.head())
    print("=" * 80)


if __name__ == "__main__":
    main()