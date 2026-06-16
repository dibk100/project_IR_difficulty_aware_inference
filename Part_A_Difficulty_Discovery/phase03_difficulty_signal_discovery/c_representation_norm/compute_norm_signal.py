import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

"""
python c_representation_norm/compute_norm_signal.py \
  --input_npz ../phase02_layerwise_decodability/output_phi_1000/gsm8k_layerwise_hidden_states.npz \
  --output_csv output_phi_1000/representation_norm/sample_layer_norm.csv

"""
def compute_l2_norm(x):
    return np.linalg.norm(x, ord=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_npz",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    print("=" * 80)
    print("[Phase03-C] Compute Representation Norm Signal")
    print("=" * 80)
    print("Input:", args.input_npz)
    print("Output:", args.output_csv)

    data = np.load(args.input_npz, allow_pickle=True)

    labels = data["labels"]
    ids = data["ids"]

    layer_keys = sorted(
        [k for k in data.files if k.startswith("layer")]
    )

    rows = []

    for idx in tqdm(range(len(labels))):

        sample_id = ids[idx]
        label = labels[idx]

        for key in layer_keys:

            layer_num = int(key.split("_")[0].replace("layer", ""))

            if key.endswith("_last"):
                aggregation = "last"
            elif key.endswith("_mean"):
                aggregation = "mean"
            else:
                continue

            h = data[key][idx]

            norm = compute_l2_norm(h)

            rows.append(
                {
                    "index": idx,
                    "id": sample_id,
                    "label": label,
                    "aggregation": aggregation,
                    "layer": layer_num,
                    "norm_l2": norm,
                }
            )

    df = pd.DataFrame(rows)

    os.makedirs(
        os.path.dirname(args.output_csv),
        exist_ok=True,
    )

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