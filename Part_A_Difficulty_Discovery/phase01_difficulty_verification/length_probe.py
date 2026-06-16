import json
import numpy as np

from transformers import AutoTokenizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
INPUT_PATH = "./output_llama_1000/gsm8k_main_rollouts.jsonl"

RANDOM_STATE = 42
N_SPLITS = 5


def load_records(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_prompt(question: str):
    return (
        "Solve the following math problem. "
        "Give the final answer after 'Final Answer:'.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def compute_length_features(records, tokenizer):
    features = []
    labels = []

    for record in records:
        question = record["question"]
        prompt = build_prompt(question)

        char_len = len(question)
        word_len = len(question.split())

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=False,
        )

        token_len = input_ids.shape[-1]

        features.append([char_len, word_len, token_len])
        labels.append(1 if record["label"] == "hard" else 0)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


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


def main():
    records = load_records(INPUT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    X_all, y = compute_length_features(records, tokenizer)

    char_len = X_all[:, [0]]
    word_len = X_all[:, [1]]
    token_len = X_all[:, [2]]

    evaluate_probe(char_len, y, "char_len_only")
    evaluate_probe(word_len, y, "word_len_only")
    evaluate_probe(token_len, y, "token_len_only")
    evaluate_probe(X_all, y, "char_word_token_len")


if __name__ == "__main__":
    main()
    