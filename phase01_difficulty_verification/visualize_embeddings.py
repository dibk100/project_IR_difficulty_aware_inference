import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

INPUT_PATH = "gsm8k_main_hidden_states.npz"


def plot_2d(points, labels, title, output_path):
    labels = np.array(labels)

    plt.figure(figsize=(7, 6))

    for label in ["easy", "hard"]:
        mask = labels == label
        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            label=label,
            alpha=0.7,
            s=30,
        )

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


def visualize_one(embeddings, labels, name):
    print(f"\n[{name}]")
    print("Embedding shape:", embeddings.shape)
    print("Easy:", np.sum(labels == "easy"))
    print("Hard:", np.sum(labels == "hard"))

    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(embeddings)

    plot_2d(
        pca_points,
        labels,
        f"PCA of {name} Hidden Representations",
        f"pca_easy_hard_{name}.png",
    )

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    tsne_points = tsne.fit_transform(embeddings)

    plot_2d(
        tsne_points,
        labels,
        f"t-SNE of {name} Hidden Representations",
        f"tsne_easy_hard_{name}.png",
    )


def main():
    data = np.load(INPUT_PATH, allow_pickle=True)

    labels = data["labels"]

    embeddings_last = data["embeddings_last"]
    embeddings_mean = data["embeddings_mean"]

    visualize_one(
        embeddings=embeddings_last,
        labels=labels,
        name="last_token",
    )

    visualize_one(
        embeddings=embeddings_mean,
        labels=labels,
        name="mean_pooling",
    )


if __name__ == "__main__":
    main()