import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

"""
python a_probe_direction/train_probe_extract_direction.py \
  --input_npz ../phase02_layerwise_decodability/output_llama_1000/gsm8k_layerwise_hidden_states.npz \
  --output_dir output_llama_1000/probe_direction \
  --n_splits 5 \
  --overwrite

"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def infer_num_layers(data, aggregation):
    layers = []
    for key in data.files:
        if key.startswith("layer") and key.endswith(f"_{aggregation}"):
            layer = int(key.replace("layer", "").replace(f"_{aggregation}", ""))
            layers.append(layer)

    if not layers:
        raise ValueError(f"No layer keys found for aggregation={aggregation}")

    return max(layers)


def train_one_probe(X, y, n_splits, random_seed):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )

    aucs = []
    accs = []
    f1s = []
    fold_weights = []
    fold_biases = []
    all_scores = np.zeros(len(y), dtype=np.float64)

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=random_seed,
        )

        clf.fit(X_train, y[train_idx])

        prob = clf.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y[test_idx], prob))
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))

        # Convert direction back to original hidden space:
        # standardized x = (x - mean) / scale
        # logit = w_std^T ((x - mean)/scale) + b
        #       = (w_std/scale)^T x + adjusted_b
        w_std = clf.coef_[0]
        w_orig = w_std / scaler.scale_
        b_orig = clf.intercept_[0] - np.sum(w_std * scaler.mean_ / scaler.scale_)

        fold_weights.append(w_orig)
        fold_biases.append(b_orig)
        all_scores[test_idx] = prob

    mean_weight = np.mean(np.stack(fold_weights), axis=0)
    mean_bias = float(np.mean(fold_biases))

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "direction": mean_weight,
        "bias": mean_bias,
        "cv_scores": all_scores,
    }


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    direction_dir = os.path.join(args.output_dir, "directions")
    os.makedirs(direction_dir, exist_ok=True)

    result_csv = os.path.join(args.output_dir, "layerwise_probe_direction_results.csv")
    score_csv = os.path.join(args.output_dir, "difficulty_projection_scores.csv")

    if os.path.exists(result_csv) and not args.overwrite:
        raise FileExistsError(f"Output exists: {result_csv}. Use --overwrite.")

    print("=" * 80)
    print("[Phase04] Train Probe and Extract Difficulty Direction")
    print("=" * 80)
    print("Input:", args.input_npz)
    print("Output dir:", args.output_dir)

    data = np.load(args.input_npz, allow_pickle=True)

    labels = data["labels"]
    ids = data["ids"]
    questions = data["questions"] if "questions" in data.files else np.array([""] * len(labels))

    y = (labels == "hard").astype(int)

    print("Samples:", len(y))
    print(pd.Series(labels).value_counts())

    result_rows = []
    score_rows = []

    for aggregation in ["last", "mean"]:
        num_layers = infer_num_layers(data, aggregation)

        for layer in range(1, num_layers + 1):
            key = f"layer{layer:02d}_{aggregation}"
            X = data[key].astype(np.float64)

            print(f"Training probe | aggregation={aggregation} | layer={layer:02d} | X={X.shape}")

            out = train_one_probe(
                X=X,
                y=y,
                n_splits=args.n_splits,
                random_seed=args.random_seed,
            )

            direction_path = os.path.join(
                direction_dir,
                f"layer{layer:02d}_{aggregation}_probe_direction.npy",
            )
            bias_path = os.path.join(
                direction_dir,
                f"layer{layer:02d}_{aggregation}_probe_bias.npy",
            )

            np.save(direction_path, out["direction"])
            np.save(bias_path, np.array([out["bias"]]))

            result_rows.append(
                {
                    "aggregation": aggregation,
                    "layer": layer,
                    "roc_auc_mean": out["auc_mean"],
                    "roc_auc_std": out["auc_std"],
                    "accuracy_mean": out["accuracy_mean"],
                    "accuracy_std": out["accuracy_std"],
                    "macro_f1_mean": out["macro_f1_mean"],
                    "macro_f1_std": out["macro_f1_std"],
                    "direction_norm": float(np.linalg.norm(out["direction"])),
                    "direction_path": direction_path,
                    "bias_path": bias_path,
                }
            )

            for i in range(len(y)):
                score_rows.append(
                    {
                        "index": i,
                        "id": ids[i],
                        "label": labels[i],
                        "aggregation": aggregation,
                        "layer": layer,
                        "cv_difficulty_score": float(out["cv_scores"][i]),
                        "question": questions[i],
                    }
                )

    result_df = pd.DataFrame(result_rows)
    score_df = pd.DataFrame(score_rows)

    result_df.to_csv(result_csv, index=False)
    score_df.to_csv(score_csv, index=False)

    print()
    print("Saved:", result_csv)
    print("Saved:", score_csv)
    print()
    print("[Top layers]")
    print(
        result_df.sort_values("roc_auc_mean", ascending=False)
        .head(20)
        .to_string(index=False)
    )
    print("=" * 80)


if __name__ == "__main__":
    main()