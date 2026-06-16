# python linear_probe.py
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


INPUT_PATH = "gsm8k_main_hidden_states.npz"
RANDOM_STATE = 42
N_SPLITS = 5


def evaluate_probe(embeddings, labels, name):
    y = np.array([1 if label == "easy" else 0 for label in labels])

    print(f"\n[{name}]")
    print("Embedding shape:", embeddings.shape)
    print("Easy:", np.sum(y == 1))
    print("Hard:", np.sum(y == 0))

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
        embeddings,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    for metric in scoring.keys():
        scores = results[f"test_{metric}"]
        print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")


def main():
    data = np.load(INPUT_PATH, allow_pickle=True)

    labels = data["labels"]

    embeddings_last = data["embeddings_last"]
    embeddings_mean = data["embeddings_mean"]

    evaluate_probe(
        embeddings=embeddings_last,
        labels=labels,
        name="last_token",
    )

    evaluate_probe(
        embeddings=embeddings_mean,
        labels=labels,
        name="mean_pooling",
    )


if __name__ == "__main__":
    main()