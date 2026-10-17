import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import CIFAR10_CLASSES
from src.evaluate import append_row, caption_nn_accuracy, knn_accuracy, load_split

# Captions mentioning exactly one of these words get that word as a label for the text UMAP.
KEYWORDS = ["dog", "bike", "snow", "water", "guitar", "ball", "horse", "food"]


def umap_2d(x, seed=0):
    import umap
    return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(x)


def scatter(points, labels, names, title, path):
    plt.figure(figsize=(6, 5))
    for c, name in enumerate(names):
        mask = labels == c
        plt.scatter(points[mask, 0], points[mask, 1], s=3, label=name)
    plt.legend(markerscale=4, fontsize=7, loc="best")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def keyword_labels(captions):
    labels = np.full(len(captions), -1)
    for i, caption in enumerate(captions):
        words = set(caption.lower().replace(".", " ").replace(",", " ").split())
        hits = [k for k, word in enumerate(KEYWORDS) if word in words]
        if len(hits) == 1:
            labels[i] = hits[0]
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--out-dir", default="results/e1")
    parser.add_argument("--umap-points", type=int, default=3000)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    csv_path = os.path.join(args.out_dir, "knn.csv")

    train_labels = np.load(os.path.join(args.emb_dir, "cifar_train_labels.npy"))
    test_labels = np.load(os.path.join(args.emb_dir, "cifar_test_labels.npy"))
    for name in ["resnet50", "vit_b_16"]:
        bank = torch.from_numpy(np.load(os.path.join(args.emb_dir, f"cifar_train_img_{name}.npy"))).float().to(device)
        queries = torch.from_numpy(np.load(os.path.join(args.emb_dir, f"cifar_test_img_{name}.npy"))).float().to(device)
        acc = knn_accuracy(bank, train_labels, queries, test_labels, k=10)
        append_row(csv_path, {"encoder": name, "task": "cifar10_knn10", "accuracy": acc})
        print(name, acc)

        pick = rng.choice(len(queries), args.umap_points, replace=False)
        points = umap_2d(queries[pick].cpu().numpy())
        scatter(points, test_labels[pick], CIFAR10_CLASSES, f"{name} on CIFAR-10 test",
                os.path.join(args.out_dir, f"umap_{name}.png"))

    with open(os.path.join(args.emb_dir, "flickr_meta.json")) as f:
        meta = json.load(f)
    test_captions = [meta["captions"][j] for j, i in enumerate(meta["caption_image"]) if meta["split"][i] == "test"]
    labels = keyword_labels(test_captions)
    keep = labels >= 0

    for name in ["minilm", "mpnet"]:
        _, txt, caption_image = load_split(args.emb_dir, "resnet50", name, "test")
        acc = caption_nn_accuracy(txt.to(device), caption_image)
        append_row(csv_path, {"encoder": name, "task": "flickr_test_caption_1nn", "accuracy": acc})
        print(name, acc)

        points = umap_2d(txt[keep].numpy())
        scatter(points, labels[keep], KEYWORDS, f"{name} on Flickr30k test captions",
                os.path.join(args.out_dir, f"umap_{name}.png"))


if __name__ == "__main__":
    main()
