import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


"""
Phase03-A: Compute Effective Rank

Input:
    sample-wise token hidden matrix files
    sample_000000.npz
    sample_000001.npz
    ...

Each sample file contains:
    layer01_hidden: (T, D)
    ...
    layerN_hidden: (T, D)

Output:
    sample_layer_effective_rank.csv

Columns:
    index
    id
    label
    correct_count
    layer
    token_len
    er_raw
    er_centered
    
    
실행 예시
python a_effective_rank/compute_effective_rank.py \
  --input_dir output_llama_500/effective_rank/token_hidden_matrices \
  --output_csv output_llama_500/effective_rank/sample_layer_effective_rank.csv \
  --overwrite
  
python a_effective_rank/compute_effective_rank.py \
  --input_dir output_llama_500/effective_rank/token_hidden_matrices \
  --output_csv output_llama_500/effective_rank/sample_layer_effective_rank_debug.csv \
  --max_samples 3 \
  --overwrite
"""


def effective_rank(matrix: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute effective rank using entropy of normalized singular values.

    ER(X) = exp(H(p))
    p_i = sigma_i / sum_j sigma_j
    H(p) = - sum_i p_i log(p_i)
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape={matrix.shape}")

    X = matrix.astype(np.float32)

    # SVD singular values only
    singular_values = np.linalg.svd(X, full_matrices=False, compute_uv=False)

    total = singular_values.sum()
    if total <= eps:
        return 0.0

    p = singular_values / total
    p = p[p > eps]

    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def get_scalar(npz_file, key):
    value = npz_file[key]
    if value.shape == ():
        return value.item()
    return value.tolist()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing sample_*.npz token hidden matrix files.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save sample_layer_effective_rank.csv.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional maximum number of samples for debugging.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it already exists.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.output_csv) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_csv}\n"
            f"Use --overwrite to overwrite."
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    sample_files = sorted(
        [
            os.path.join(args.input_dir, fname)
            for fname in os.listdir(args.input_dir)
            if fname.startswith("sample_") and fname.endswith(".npz")
        ]
    )

    if args.max_samples is not None:
        sample_files = sample_files[: args.max_samples]

    if len(sample_files) == 0:
        raise FileNotFoundError(f"No sample_*.npz files found in {args.input_dir}")

    print("=" * 80)
    print("[Phase03-A] Compute Effective Rank")
    print("=" * 80)
    print("Input dir:", args.input_dir)
    print("Output CSV:", args.output_csv)
    print("Number of sample files:", len(sample_files))

    rows = []

    for file_path in tqdm(sample_files):
        data = np.load(file_path, allow_pickle=True)

        index = int(get_scalar(data, "index"))
        sample_id = str(get_scalar(data, "id"))
        label = str(get_scalar(data, "label"))
        correct_count = int(get_scalar(data, "correct_count"))
        token_len = int(get_scalar(data, "token_len"))

        layer_keys = sorted(
            [
                key for key in data.files
                if key.startswith("layer") and key.endswith("_hidden")
            ]
        )

        for layer_key in layer_keys:
            # layer01_hidden -> 1
            layer = int(layer_key.replace("layer", "").replace("_hidden", ""))

            H = data[layer_key].astype(np.float32)  # (T, D)

            er_raw = effective_rank(H)

            H_centered = H - H.mean(axis=0, keepdims=True)
            er_centered = effective_rank(H_centered)

            rows.append(
                {
                    "index": index,
                    "id": sample_id,
                    "label": label,
                    "correct_count": correct_count,
                    "layer": layer,
                    "token_len": token_len,
                    "er_raw": er_raw,
                    "er_centered": er_centered,
                    "source_file": os.path.basename(file_path),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["index", "layer"]).reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)

    print("=" * 80)
    print("Done.")
    print("Saved:", args.output_csv)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print()
    print(df.head())
    print("=" * 80)


if __name__ == "__main__":
    main()