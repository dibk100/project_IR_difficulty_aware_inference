import json
import numpy as np

from transformers import AutoTokenizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

"""
python length_matched_probe.py

"""

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

ROLLOUT_PATH = "output_phi_500/gsm8k_main_rollouts.jsonl"
HIDDEN_PATH = "output_phi_500/gsm8k_main_hidden_states.npz"

RANDOM_STATE = 42
N_SPLITS = 5
TOKEN_TOLERANCE = 5


def build_prompt(question: str):
    return (
        "Solve the following math problem. "
        "Give the final answer after 'Final Answer:'.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compute_token_lengths(records, tokenizer):
    token_lens = []

    for record in records:
        prompt = build_prompt(record["question"])
        messages = [{"role": "user", "content": prompt}]

        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=False,
        )

        token_lens.append(input_ids.shape[-1])

    return np.array(token_lens)


def make_length_matched_indices(labels, token_lens, tolerance=5, random_state=42):
    rng = np.random.default_rng(random_state)

    labels = np.array(labels)
    easy_indices = np.where(labels == "easy")[0].tolist()
    hard_indices = np.where(labels == "hard")[0].tolist()

    used_easy = set()
    matched_easy = []
    matched_hard = []

    rng.shuffle(hard_indices)

    for h_idx in hard_indices:
        h_len = token_lens[h_idx]

        candidates = [
            e_idx for e_idx in easy_indices
            if e_idx not in used_easy
            and abs(token_lens[e_idx] - h_len) <= tolerance
        ]

        if len(candidates) == 0:
            continue

        # 가장 token_len 차이가 작은 easy를 선택
        min_diff = min(abs(token_lens[e_idx] - h_len) for e_idx in candidates)
        best_candidates = [
            e_idx for e_idx in candidates
            if abs(token_lens[e_idx] - h_len) == min_diff
        ]

        e_idx = rng.choice(best_candidates)

        matched_hard.append(h_idx)
        matched_easy.append(e_idx)
        used_easy.add(e_idx)

    matched_indices = np.array(matched_easy + matched_hard)

    return matched_indices, np.array(matched_easy), np.array(matched_hard)


def evaluate_probe(X, y, name):
    print(f"\n[{name}]")
    print("Feature shape:", X.shape)
    print("Easy:", np.sum(y == 0))
    print("Hard:", np.sum(y == 1))

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "macro_f1": "f1_macro",
    }

    results = cross_validate(
        clf,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    for metric in scoring.keys():
        scores = results[f"test_{metric}"]
        print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")


def print_length_stats(token_lens, labels, title):
    print(f"\n[{title}]")

    for label in ["easy", "hard"]:
        mask = labels == label
        values = token_lens[mask]

        print(
            label,
            {
                "n": len(values),
                "mean": round(float(values.mean()), 2),
                "median": round(float(np.median(values)), 2),
                "std": round(float(values.std()), 2),
                "min": int(values.min()),
                "max": int(values.max()),
            },
        )


def main():
    records = load_records(ROLLOUT_PATH)
    hidden = np.load(HIDDEN_PATH, allow_pickle=True)

    labels = np.array([record["label"] for record in records])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_lens = compute_token_lengths(records, tokenizer)

    embeddings_last = hidden["embeddings_last"]
    embeddings_mean = hidden["embeddings_mean"]

    print("Original dataset")
    print("Total:", len(records))
    print("Easy:", np.sum(labels == "easy"))
    print("Hard:", np.sum(labels == "hard"))

    print_length_stats(token_lens, labels, "Original length stats")

    matched_indices, matched_easy, matched_hard = make_length_matched_indices(
        labels=labels,
        token_lens=token_lens,
        tolerance=TOKEN_TOLERANCE,
        random_state=RANDOM_STATE,
    )

    matched_labels = labels[matched_indices]
    matched_token_lens = token_lens[matched_indices]

    print("\nMatched dataset")
    print("Tolerance:", TOKEN_TOLERANCE)
    print("Matched total:", len(matched_indices))
    print("Matched easy:", len(matched_easy))
    print("Matched hard:", len(matched_hard))

    print_length_stats(matched_token_lens, matched_labels, "Matched length stats")

    y = np.array([1 if label == "hard" else 0 for label in matched_labels])

    X_token_len = matched_token_lens.reshape(-1, 1)
    X_last = embeddings_last[matched_indices]
    X_mean = embeddings_mean[matched_indices]

    evaluate_probe(X_token_len, y, "length_matched_token_len_only")
    evaluate_probe(X_last, y, "length_matched_hidden_last_token")
    evaluate_probe(X_mean, y, "length_matched_hidden_mean_pooling")


if __name__ == "__main__":
    main()