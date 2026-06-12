import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


"""
Phase03-B: Layer-wise Representation Change Analysis

Goal:
    Compute layer-to-layer representation change.

Input:
    phase02_layerwise_decodability/output_*/gsm8k_layerwise_hidden_states.npz

Required keys:
    labels
    ids
    questions
    layer01_last, layer02_last, ...
    layer01_mean, layer02_mean, ...

Output:
    sample_layer_delta.csv

Columns:
    index
    id
    label
    aggregation     # last or mean
    layer           # current layer l, delta between l-1 and l
    delta_l2
    delta_l2_normed
    delta_cosine
    
python b_layerwise_representation/compute_layerwise_delta.py \
  --input_npz ../phase02_layerwise_decodability/output_llama_500/gsm8k_layerwise_hidden_states.npz \
  --output_csv output_llama_500/layerwise_representation_change/sample_layer_delta.csv \
  --overwrite
  
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_npz",
        type=str,
        required=True,
        help="Path to gsm8k_layerwise_hidden_states.npz from Phase02.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save sample_layer_delta.csv.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it already exists.",
    )

    return parser.parse_args()


def l2_norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=1)


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Cosine distance = 1 - cosine similarity.
    a, b: (N, D)
    """
    dot = np.sum(a * b, axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    cos_sim = dot / (a_norm * b_norm + eps)
    cos_dist = 1.0 - cos_sim

    return cos_dist


def infer_num_layers(npz_data, aggregation: str) -> int:
    layers = []

    for key in npz_data.files:
        if key.startswith("layer") and key.endswith(f"_{aggregation}"):
            layer_num = int(key.replace("layer", "").replace(f"_{aggregation}", ""))
            layers.append(layer_num)

    if not layers:
        raise ValueError(f"No layer keys found for aggregation={aggregation}")

    return max(layers)


def main():
    args = parse_args()

    if os.path.exists(args.output_csv) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_csv}\n"
            f"Use --overwrite to overwrite."
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    print("=" * 80)
    print("[Phase03-B] Compute Layer-wise Representation Delta")
    print("=" * 80)
    print("Input NPZ:", args.input_npz)
    print("Output CSV:", args.output_csv)

    data = np.load(args.input_npz, allow_pickle=True)

    labels = data["labels"]
    ids = data["ids"]

    n_samples = len(labels)

    print("Number of samples:", n_samples)
    print("Label counts:")
    print(pd.Series(labels).value_counts())

    rows = []

    for aggregation in ["last", "mean"]:
        num_layers = infer_num_layers(data, aggregation)

        print()
        print(f"[Aggregation: {aggregation}]")
        print("Number of layers:", num_layers)

        for layer in tqdm(range(2, num_layers + 1)):
            prev_key = f"layer{layer - 1:02d}_{aggregation}"
            curr_key = f"layer{layer:02d}_{aggregation}"

            if prev_key not in data.files:
                raise KeyError(f"Missing key: {prev_key}")
            if curr_key not in data.files:
                raise KeyError(f"Missing key: {curr_key}")

            h_prev = data[prev_key].astype(np.float32)  # (N, D)
            h_curr = data[curr_key].astype(np.float32)  # (N, D)

            diff = h_curr - h_prev

            delta_l2 = l2_norm(diff)

            # Normalize by previous representation norm.
            # This controls for magnitude scale differences across layers.
            prev_norm = l2_norm(h_prev)
            delta_l2_normed = delta_l2 / (prev_norm + 1e-12)

            delta_cosine = cosine_distance(h_prev, h_curr)

            for i in range(n_samples):
                rows.append(
                    {
                        "index": i,
                        "id": ids[i],
                        "label": labels[i],
                        "aggregation": aggregation,
                        "layer": layer,
                        "delta_l2": float(delta_l2[i]),
                        "delta_l2_normed": float(delta_l2_normed[i]),
                        "delta_cosine": float(delta_cosine[i]),
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sort_values(["aggregation", "index", "layer"]).reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)

    print()
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